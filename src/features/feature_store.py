"""
Master feature store for Market Pulse.

Combines per-ticker:
- Technical features from OHLCV bars (RSI, MACD, ATR, etc.)
- Sentiment aggregations from news (per ticker, multiple time windows)
- Macro context — TIME-ALIGNED via macro_history table (v2 P0 fix)

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
    Prefers longer-history files (1095d > 365d > anything) so warmup-heavy
    indicators have enough bars.
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
    """Find all tickers we have OHLCV files for. Skips short-history files."""
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

    Returns one row per ticker with columns: ticker, sent_avg_24h, sent_count_24h,
    sent_pos_pct_24h, ... (×3 windows)

    NOTE: this is a CURRENT snapshot, not time-aligned. Will be replaced in
    P1 with proper time-aligned sentiment computed from raw news timestamps.
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

    long_df = pd.DataFrame(rows)
    wide = long_df.groupby("ticker").first().reset_index()
    return wide


# ============================================================
# 3a. Macro features — time-aligned (v2 P0 fix)
# ============================================================
def load_macro_history() -> pd.DataFrame:
    """
    Load full daily macro history from the macro_history table.

    Returns wide DataFrame indexed by date, one column per macro series.
    Forward-fills gaps (e.g., weekends for daily series, between monthly
    observations for unemployment/CPI) so every trading day has values.
    """
    if not DB_PATH.exists():
        logger.warning(f"[feature_store] {DB_PATH} not found — no macro history")
        return pd.DataFrame()

    with sqlite3.connect(str(DB_PATH)) as conn:
        df = pd.read_sql_query(
            "SELECT date, friendly, value FROM macro_history",
            conn,
        )

    if df.empty:
        logger.warning(
            "[feature_store] macro_history is empty. "
            "Run: python -m src.connectors.fred_history"
        )
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], utc=True)
    wide = df.pivot(index="date", columns="friendly", values="value")
    wide = wide.sort_index()

    # Forward-fill so weekends + monthly-series gaps get last known value.
    # At any given trading day, the "current" macro value is the most recent
    # FRED observation, not NaN.
    wide = wide.ffill()

    # Prefix with macro_ to match feature_store conventions
    wide.columns = [f"macro_{c}" for c in wide.columns]

    logger.info(
        f"[feature_store] loaded macro history: "
        f"{wide.shape[0]} dates × {wide.shape[1]} series"
    )
    return wide


# ============================================================
# 3b. Macro snapshot — for the dashboard's current-state display
# ============================================================
def load_macro_snapshot() -> pd.DataFrame:
    """
    Load latest-cycle macro snapshot (single row, latest values + regime flags).
    Used by the dashboard for current-state display, NOT for ML training.
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
    macro_history = load_macro_history()
    logger.info(
        f"[feature_store] loaded sentiment for {len(sent_df)} tickers, "
        f"{macro_history.shape[1] if not macro_history.empty else 0} macro series "
        f"× {macro_history.shape[0] if not macro_history.empty else 0} dates"
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
        feats["ticker"] = ticker

        # Merge sentiment for this ticker
        # (Still a snapshot — proper time-aligned sentiment comes in P1)
        if not sent_df.empty and ticker in sent_df["ticker"].values:
            srow = sent_df[sent_df["ticker"] == ticker].iloc[0]
            for col in sent_df.columns:
                if col == "ticker":
                    continue
                feats[col] = srow[col]
        else:
            for col in sent_df.columns:
                if col != "ticker":
                    feats[col] = np.nan

        # Merge macro by JOINING on date — the v2 fix.
        # Each (ticker, date) row gets the actual macro values for THAT date.
        if not macro_history.empty:
            feats_with_date = feats.copy()
            feats_with_date["_join_date"] = pd.to_datetime(
                feats_with_date.index, utc=True
            ).normalize()
            macro_for_join = macro_history.copy()
            macro_for_join.index = macro_for_join.index.normalize()
            feats = feats_with_date.merge(
                macro_for_join,
                how="left",
                left_on="_join_date",
                right_index=True,
            )
            feats = feats.drop(columns=["_join_date"])

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

    print("\n--- Sentiment per ticker (snapshot, not yet time-aligned) ---")
    sent = load_sentiment_per_ticker()
    if not sent.empty:
        print(f"  shape: {sent.shape}")
        print(sent.head(5).to_string(index=False))
    else:
        print("  (none — DB empty?)")

    print("\n--- Macro history (time-aligned, v2 P0 fix) ---")
    macro = load_macro_history()
    if not macro.empty:
        print(f"  shape: {macro.shape}  ({macro.shape[0]} dates × {macro.shape[1]} series)")
        print(f"  date range: {macro.index.min().date()} → {macro.index.max().date()}")
        print(f"  series: {list(macro.columns)}")
        print(f"\n  Sample (5 most recent dates):")
        print(macro.tail(5).to_string())
    else:
        print("  (empty — run: python -m src.connectors.fred_history)")

    print("\n--- Building master feature frame ---")
    master = build_master_features(save=True)

    if not master.empty:
        print(f"\n  shape: {master.shape}")
        print(f"  unique tickers: {master['ticker'].nunique()}")
        print(f"  date range: {master['timestamp'].min()} → {master['timestamp'].max()}")

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

        # Variance check on macro columns — proves the v2 fix
        print("\n  Macro variance check (std should be > 0 for all):")
        macro_cols = [c for c in master.columns if c.startswith("macro_")]
        for col in sorted(macro_cols):
            series = master[col].dropna()
            if len(series) == 0:
                print(f"    {col:35s} all NaN")
            else:
                marker = "✓" if series.std() > 0 else "✗"
                print(f"    {marker} {col:35s} std={series.std():>8.4f}  "
                      f"min={series.min():>8.2f}  max={series.max():>8.2f}  "
                      f"unique={series.nunique()}")

        # Sample latest AAPL row
        if "AAPL" in master["ticker"].values:
            print("\n  Latest AAPL row (sample features):")
            last = master[master["ticker"] == "AAPL"].iloc[-1]
            sample_cols = ["timestamp", "close", "ret_5", "rsi_14",
                          "sent_avg_168h", "macro_vix", "macro_yield_curve_10y2y",
                          "macro_fed_funds"]
            for col in sample_cols:
                if col in last.index:
                    val = last[col]
                    if isinstance(val, float):
                        print(f"    {col:30s} = {val:>12.4f}")
                    else:
                        print(f"    {col:30s} = {val}")
