"""
Master feature store for Agent 29.

Combines per-ticker:
- Technical features from OHLCV bars (RSI, MACD, ATR, etc.)
- Sentiment aggregations from news (per ticker, multiple time windows)
- Macro context (VIX, yield curve, Crypto F&G — same for all tickers)

Output: data/features/master.parquet — one row per (ticker, timestamp).
This is the ML-ready feature matrix.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.features.technical import add_technical_features


RAW_DIR = Path("data/raw")
DB_PATH = Path("data/agent29.db")
FEATURES_DIR = Path("data/features")
MASTER_PATH = FEATURES_DIR / "master.parquet"


# ============================================================
# 1. Load OHLCV per ticker
# ============================================================
def load_ohlcv_for_ticker(ticker: str, prefer_long: bool = True) -> pd.DataFrame:
    """
    Load the most useful OHLCV file for a ticker.
    Prefers _365d_ files when available (longer history = more features populated).
    """
    ticker_lower = ticker.lower()

    if prefer_long:
        # Try in order: 1095d (3yr) → 365d (1yr) → anything
        for pattern in [f"{ticker_lower}_1095d*.parquet",
                        f"{ticker_lower}_365d*.parquet"]:
            candidates = sorted(RAW_DIR.glob(pattern), reverse=True)
            if candidates:
                return pd.read_parquet(candidates[0], engine="fastparquet")

    # Fallback to any _daily file
    candidates = sorted(RAW_DIR.glob(f"{ticker_lower}_*daily*.parquet"), reverse=True)
    if candidates:
        return pd.read_parquet(candidates[0], engine="fastparquet")

    return pd.DataFrame()


def discover_tickers() -> list[str]:
    """Find all tickers we have OHLCV files for."""
    # Prefer longer-history files. Skip _30d_ files (not enough warmup).
    files = (
        list(RAW_DIR.glob("*_1095d_daily.parquet"))
        + list(RAW_DIR.glob("*_365d_daily.parquet"))
    )

    tickers = sorted(set(f.stem.split("_")[0].upper() for f in files))
    return tickers


# ============================================================
# 2. Sentiment aggregation per ticker
# ============================================================
def load_sentiment_per_ticker() -> pd.DataFrame:
    """
    Pull from SQLite all scored news, aggregate by ticker over time windows.

    Returns columns: ticker, sent_avg_24h, sent_count_24h, sent_pos_pct_24h, ...
    """
    if not DB_PATH.exists():
        logger.warning(f"[feature_store] {DB_PATH} not found — skipping sentiment")
        return pd.DataFrame()

    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(
            """
            SELECT primary_ticker as ticker, published_at,
                   finbert_compound, finbert_label
            FROM news
            WHERE primary_ticker IS NOT NULL
              AND finbert_compound IS NOT NULL
            """,
            conn,
        )

    if df.empty:
        return pd.DataFrame()

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])

    now = datetime.now(timezone.utc)
    rows = []

    for ticker in df["ticker"].unique():
        td = df[df["ticker"] == ticker]
        for window_h in [24, 72, 168]:  # 1d, 3d, 7d
            cutoff = now - timedelta(hours=window_h)
            recent = td[td["published_at"] >= cutoff]
            if recent.empty:
                continue
            n_pos = (recent["finbert_label"] == "positive").sum()
            n_neg = (recent["finbert_label"] == "negative").sum()
            row = {
                "ticker": ticker,
                f"sent_avg_{window_h}h":      float(recent["finbert_compound"].mean()),
                f"sent_count_{window_h}h":    int(len(recent)),
                f"sent_pos_pct_{window_h}h":  float(n_pos / len(recent)),
                f"sent_neg_pct_{window_h}h":  float(n_neg / len(recent)),
                f"sent_net_{window_h}h":      int(n_pos - n_neg),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    # Each row is one (ticker, window) — pivot to wide format
    long_df = pd.DataFrame(rows)
    wide = long_df.groupby("ticker").first().reset_index()
    # Some windows might have NaN if no headlines; fill with 0 for counts and 0 for sentiment
    return wide


# ============================================================
# 3. Macro features (latest cycle)
# ============================================================
def load_macro_features() -> pd.DataFrame:
    """
    Load macro features from the most recent cycle. These are constant across
    tickers (same VIX, yield curve, F&G for everyone).

    Returns single-row DataFrame to be cross-joined with each ticker.
    """
    if not DB_PATH.exists():
        return pd.DataFrame()

    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(
            """
            SELECT feature_name, feature_val, feature_text
            FROM features
            WHERE cycle_id = (SELECT cycle_id FROM cycles ORDER BY started_at DESC LIMIT 1)
            """,
            conn,
        )

    if df.empty:
        return pd.DataFrame()

    # Wide single-row frame with each feature as a column
    wide: dict = {}
    for _, r in df.iterrows():
        name = r["feature_name"]
        val = r["feature_val"]
        text = r["feature_text"]
        wide[f"macro_{name}"] = val if val is not None else text

    return pd.DataFrame([wide])


# ============================================================
# 4. Build master feature frame
# ============================================================
def build_master_features(
    tickers: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    For each ticker, compute technical features and merge with sentiment + macro.

    Returns long-format DataFrame: one row per (ticker, timestamp).
    """
    if tickers is None:
        tickers = discover_tickers()
    logger.info(f"[feature_store] building features for {len(tickers)} tickers: {tickers}")

    sent_df = load_sentiment_per_ticker()
    macro_df = load_macro_features()
    logger.info(
        f"[feature_store] loaded sentiment for {len(sent_df)} tickers, "
        f"{len(macro_df.columns) if not macro_df.empty else 0} macro features"
    )

    all_rows = []

    for ticker in tickers:
        ohlcv = load_ohlcv_for_ticker(ticker)
        if ohlcv.empty:
            logger.warning(f"[feature_store] no OHLCV for {ticker}, skipping")
            continue

        # Reset multi-index if present
        if isinstance(ohlcv.index, pd.MultiIndex):
            ohlcv = ohlcv.reset_index().sort_values("timestamp").set_index("timestamp")
        elif "timestamp" in ohlcv.columns:
            ohlcv = ohlcv.set_index("timestamp")

        # Compute technical features
        feats = add_technical_features(ohlcv, drop_na=False)

        # Add ticker column
        feats["ticker"] = ticker

        # Merge sentiment for this ticker (broadcast to all rows — sentiment is "current state")
        if not sent_df.empty and ticker in sent_df["ticker"].values:
            srow = sent_df[sent_df["ticker"] == ticker].iloc[0]
            for col in sent_df.columns:
                if col == "ticker":
                    continue
                feats[col] = srow[col]
        else:
            # Set sentiment columns to NaN if no news for this ticker
            for col in sent_df.columns:
                if col != "ticker":
                    feats[col] = np.nan

        # Merge macro (broadcast — same for everyone). Build a wide row first
        # then concat — avoids DataFrame fragmentation warnings.
        if not macro_df.empty:
            macro_row = macro_df.iloc[0]
            macro_block = pd.DataFrame(
                [macro_row.values] * len(feats),
                columns=macro_row.index,
                index=feats.index,
            )
            feats = pd.concat([feats, macro_block], axis=1)
        # Make timestamp a column instead of index
        feats = feats.reset_index()
        all_rows.append(feats)

    if not all_rows:
        logger.error("[feature_store] no tickers produced features")
        return pd.DataFrame()

    master = pd.concat(all_rows, ignore_index=True)
    master = master.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    logger.success(
        f"[feature_store] master frame: {master.shape[0]} rows × {master.shape[1]} columns"
    )

    if save:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        # All-object columns can break parquet — coerce via dtypes
        master_clean = master.copy()
        for col in master_clean.columns:
            if master_clean[col].dtype == object:
                # Try numeric, fall back to string
                try:
                    master_clean[col] = pd.to_numeric(master_clean[col])
                except (ValueError, TypeError):
                    master_clean[col] = master_clean[col].astype(str)
        master_clean.to_parquet(MASTER_PATH, engine="fastparquet")
        logger.success(f"[feature_store] saved to {MASTER_PATH}")

    return master


# ============================================================
# CLI / smoke test
# ============================================================
if __name__ == "__main__":
    print("\n--- Discovered tickers ---")
    tickers = discover_tickers()
    print(f"  {tickers}")

    print("\n--- Sentiment per ticker (from DB) ---")
    sent = load_sentiment_per_ticker()
    if not sent.empty:
        print(f"  shape: {sent.shape}")
        print(sent.head(10).to_string(index=False))
    else:
        print("  (none — DB empty?)")

    print("\n--- Macro features (latest cycle) ---")
    macro = load_macro_features()
    if not macro.empty:
        print(f"  {macro.shape[1]} features")
        # Show 10 most relevant
        important = [c for c in macro.columns if any(k in c for k in
                     ["vix", "yield", "fng", "regime", "fed_funds"])]
        for col in sorted(important):
            val = macro.iloc[0][col]
            print(f"    {col:45s} = {val}")

    print("\n--- Building master feature frame ---")
    master = build_master_features(save=True)

    if not master.empty:
        print(f"\n  shape: {master.shape}")
        print(f"  unique tickers: {master['ticker'].nunique()}")
        print(f"  date range: {master['timestamp'].min()} → {master['timestamp'].max()}")

        # Show columns by category
        cols = master.columns.tolist()
        groups = {
            "OHLCV":      [c for c in cols if c in ["open", "high", "low", "close", "volume", "vwap", "trade_count", "symbol"]],
            "Identity":   [c for c in cols if c in ["ticker", "timestamp"]],
            "Technical":  [c for c in cols if any(c.startswith(p) for p in ["sma_", "ema_", "ret_", "log_ret", "rsi_", "macd", "stoch", "adx", "atr", "bb_", "kc_", "ttm", "obv", "mfi", "volume_z", "volume_sp", "realized_vol", "vol_reg", "donchian", "ma_cross", "price_vs", "overnight", "range_", "trending"])],
            "Sentiment":  [c for c in cols if c.startswith("sent_")],
            "Macro":      [c for c in cols if c.startswith("macro_")],
        }
        print("\n  Columns by category:")
        for group, gcols in groups.items():
            print(f"    {group:12s}: {len(gcols)} columns")

        # Show a snapshot of the last bar for one ticker
        if "AAPL" in master["ticker"].values:
            print("\n  Latest AAPL row (sample of features):")
            last = master[master["ticker"] == "AAPL"].iloc[-1]
            sample_cols = ["timestamp", "close", "ret_5", "rsi_14", "macd_bull",
                          "atr_pct", "realized_vol_20", "sent_avg_168h",
                          "sent_count_168h", "macro_fred_vix_latest",
                          "macro_fng_latest", "macro_regime_vix_normal"]
            for col in sample_cols:
                if col in last.index:
                    val = last[col]
                    if isinstance(val, float):
                        print(f"    {col:30s} = {val:>12.4f}")
                    else:
                        print(f"    {col:30s} = {val}")
