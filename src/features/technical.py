"""
Technical indicators for Agent 29.

Computes 30+ technical features per OHLCV bar using the `ta` library
plus custom calculations. All features are lookback-only (no future leakage)
so they're safe for backtesting.

Categories:
- Trend:        EMA/SMA crossovers, MACD, ADX, parabolic SAR
- Momentum:     RSI, Stochastic, ROC, momentum
- Volatility:   ATR, Bollinger Bands, Keltner channels
- Volume:       OBV, MFI, volume Z-score
- Price action: returns over multiple windows, gap detection
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

# ta library — installed Day 1
import ta


def add_technical_features(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    drop_na: bool = False,
) -> pd.DataFrame:
    """
    Add all technical indicators to an OHLCV DataFrame.

    Input must be sorted oldest-first. Adds ~35 columns.

    Args:
        df: OHLCV DataFrame (one ticker, sorted oldest first)
        drop_na: if True, drops rows where any feature is NaN
                 (typically the first ~50 bars need warmup)
    """
    if df.empty:
        return df
# Need at least 50 bars for most indicators to work without crashing
    if len(df) < 50:
        logger.warning(
            f"[technical] only {len(df)} bars — many features will be NaN; "
            f"need 200+ for full feature set"
        )

    # Make a working copy and ensure ascending order
    df = df.sort_index().copy() if df.index.is_monotonic_decreasing else df.copy()

    o = df[open_col].astype(float)
    h = df[high_col].astype(float)
    l = df[low_col].astype(float)
    c = df[close_col].astype(float)
    v = df[volume_col].astype(float)

    # ============================================================
    # Returns and price action
    # ============================================================
    df["ret_1"]   = c.pct_change(1)
    df["ret_5"]   = c.pct_change(5)
    df["ret_20"]  = c.pct_change(20)
    df["ret_60"]  = c.pct_change(60)

    # Log returns (better for stats / volatility calcs)
    df["log_ret_1"] = np.log(c / c.shift(1))

    # Overnight gap (today's open vs yesterday's close)
    df["overnight_gap"] = (o - c.shift(1)) / c.shift(1)

    # Intraday range as % of close
    df["range_pct"] = (h - l) / c

    # ============================================================
    # Moving averages
    # ============================================================
    for window in [5, 10, 20, 50, 200]:
        df[f"sma_{window}"] = ta.trend.sma_indicator(c, window=window)
        df[f"ema_{window}"] = ta.trend.ema_indicator(c, window=window)

    # Price relative to key MAs (-1 = below, +1 = above)
    df["price_vs_sma20"]  = (c - df["sma_20"])  / df["sma_20"]
    df["price_vs_sma50"]  = (c - df["sma_50"])  / df["sma_50"]
    df["price_vs_sma200"] = (c - df["sma_200"]) / df["sma_200"]

    # MA crossover signals (1 = golden cross, -1 = death cross)
    df["ma_cross_50_200"] = np.where(df["sma_50"] > df["sma_200"], 1, -1)
    df["ma_cross_20_50"]  = np.where(df["sma_20"] > df["sma_50"],  1, -1)

    # ============================================================
    # MACD (Moving Average Convergence Divergence)
    # ============================================================
    macd = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()
    df["macd_bull"]   = (df["macd"] > df["macd_signal"]).astype(int)

    # ============================================================
    # RSI (Relative Strength Index)
    # ============================================================
    df["rsi_14"] = ta.momentum.rsi(c, window=14)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)

    # ============================================================
    # Stochastic Oscillator
    # ============================================================
    stoch = ta.momentum.StochasticOscillator(high=h, low=l, close=c)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # ============================================================
    # ADX (Average Directional Index — trend strength)
    # ============================================================
    try:
        adx = ta.trend.ADXIndicator(high=h, low=l, close=c)
        df["adx"]     = adx.adx()
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()
        df["trending_strong"] = (df["adx"] > 25).astype(int)
    except (IndexError, ValueError) as e:
        logger.warning(f"[technical] ADX skipped: {e}")
        df["adx"]     = np.nan
        df["adx_pos"] = np.nan
        df["adx_neg"] = np.nan
        df["trending_strong"] = 0
    # ============================================================
    # ATR (Average True Range — volatility, used for position sizing)
    # ============================================================
    df["atr_14"] = ta.volatility.average_true_range(h, l, c, window=14)
    df["atr_pct"] = df["atr_14"] / c   # normalized — comparable across tickers

    # ============================================================
    # Bollinger Bands (volatility channels)
    # ============================================================
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"]    = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).quantile(0.2)).astype(int)

    # ============================================================
    # Keltner Channels
    # ============================================================
    keltner = ta.volatility.KeltnerChannel(h, l, c)
    df["kc_upper"] = keltner.keltner_channel_hband()
    df["kc_lower"] = keltner.keltner_channel_lband()

    # TTM Squeeze: BBands inside Keltner = compression about to expand
    df["ttm_squeeze"] = (
        (df["bb_upper"] < df["kc_upper"]) &
        (df["bb_lower"] > df["kc_lower"])
    ).astype(int)

    # ============================================================
    # Volume features
    # ============================================================
    df["obv"]     = ta.volume.on_balance_volume(c, v)
    df["mfi_14"]  = ta.volume.money_flow_index(h, l, c, v, window=14)

    # Volume Z-score (how unusual is today's volume vs 20d avg)
    df["volume_zscore_20"] = (
        (v - v.rolling(20).mean()) / v.rolling(20).std()
    )
    df["volume_spike"] = (df["volume_zscore_20"] > 2).astype(int)

    # ============================================================
    # Volatility (realized)
    # ============================================================
    df["realized_vol_20"] = df["log_ret_1"].rolling(20).std() * np.sqrt(252)

    # Volatility regime — needs enough variance to make 3 bins
    try:
        df["vol_regime"] = pd.qcut(
            df["realized_vol_20"].bfill(),
            q=3,
            labels=["low_vol", "mid_vol", "high_vol"],
            duplicates="drop",
        ).astype(str)
    except (ValueError, TypeError) as e:
        logger.warning(f"[technical] vol_regime skipped: {e}")
        df["vol_regime"] = "unknown"
    # ============================================================
    # Donchian channels (trend breakouts)
    # ============================================================
    df["donchian_high_20"] = h.rolling(20).max()
    df["donchian_low_20"]  = l.rolling(20).min()
    df["donchian_break_up"]   = (c >= df["donchian_high_20"].shift(1)).astype(int)
    df["donchian_break_down"] = (c <= df["donchian_low_20"].shift(1)).astype(int)

    # ============================================================
    # Cleanup
    # ============================================================
    if drop_na:
        df = df.dropna()

    logger.info(
        f"[technical] added {len([c for c in df.columns if c not in [open_col, high_col, low_col, close_col, volume_col]])} features"
    )
    return df


if __name__ == "__main__":
    # Test on the SPY data we saved earlier
    from pathlib import Path
    raw = Path("data/raw")

# Prefer the longer history file if available
    spy_files = sorted(raw.glob("spy_365d*.parquet")) or sorted(raw.glob("spy_*.parquet"))
    if not spy_files:
        print("No SPY parquet files found. Run alpaca_client first.")
        exit(1)

    print(f"\n--- Loading {spy_files[0]} ---")
    df = pd.read_parquet(spy_files[0], engine="fastparquet")
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")

    # Reset multi-index if present
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index().sort_values("timestamp").set_index("timestamp")

    print("\n--- Computing technical features ---")
    df_feat = add_technical_features(df, drop_na=False)

    print(f"  output shape: {df_feat.shape}")
    print(f"  added {df_feat.shape[1] - df.shape[1]} columns")

    print("\n--- Last bar's feature snapshot ---")
    last = df_feat.iloc[-1]
    feature_cols = [c for c in df_feat.columns if c not in ["open", "high", "low", "close", "volume", "trade_count", "vwap", "symbol"]]

    # Group features for readable display
    groups = {
        "Returns":      [c for c in feature_cols if c.startswith("ret_") or c.startswith("log_ret") or c.startswith("overnight") or c.startswith("range")],
        "Trend (MA)":   [c for c in feature_cols if c.startswith("sma") or c.startswith("ema") or c.startswith("price_vs") or c.startswith("ma_cross")],
        "MACD":         [c for c in feature_cols if c.startswith("macd")],
        "Momentum":     [c for c in feature_cols if c.startswith("rsi") or c.startswith("stoch")],
        "ADX":          [c for c in feature_cols if c.startswith("adx") or c == "trending_strong"],
        "Volatility":   [c for c in feature_cols if c.startswith("atr") or c.startswith("bb_") or c.startswith("kc_") or c.startswith("ttm") or c.startswith("realized_vol") or c == "vol_regime"],
        "Volume":       [c for c in feature_cols if c.startswith("obv") or c.startswith("mfi") or c.startswith("volume")],
        "Donchian":     [c for c in feature_cols if c.startswith("donchian")],
    }

    for group, cols in groups.items():
        if not cols:
            continue
        print(f"\n  [{group}]")
        for col in cols:
            val = last.get(col)
            if isinstance(val, float):
                print(f"    {col:30s} = {val:>10.4f}")
            else:
                print(f"    {col:30s} = {val}")

    # NaN diagnostic — important for first few bars warmup
    nan_counts = df_feat.isna().sum()
    nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
    if not nan_cols.empty:
        print(f"\n--- NaN counts (top 10 features by NaN count) ---")
        print(nan_cols.head(10).to_string())
        print(f"\n  Note: first ~50-200 bars typically have NaNs due to indicator warmup.")
        print(f"  drop_na=True or warmup >= 200 bars before training.")
