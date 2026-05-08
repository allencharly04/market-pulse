"""
Dataset prep for Market Pulse ML models.

Handles:
1. Loading the master feature parquet
2. Building targets (next-day direction, etc.)
3. Time-aware train/test splits (NEVER random — that leaks the future)
4. Feature column selection (what's safe to feed the model)

The biggest mistakes in finance ML, all preventable here:
- Random train/test splits → information leakage from the future
- Using features that were unknown at the time → "look-ahead bias"
- Forgetting to drop the warmup period → models train on NaNs
- Joining sentiment that came AFTER the target's outcome → leakage
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


MASTER_PATH = Path("data/features/master.parquet")


# ============================================================
# Loading
# ============================================================
def load_master() -> pd.DataFrame:
    """Load the master feature frame."""
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_PATH} not found. Run: python -m src.features.feature_store"
        )
    df = pd.read_parquet(MASTER_PATH, engine="fastparquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return df


# ============================================================
# Target construction
# ============================================================
def build_targets(
    df: pd.DataFrame,
    horizon: int = 1,
    target_col: str = "target_direction",
    return_col: str = "target_return",
) -> pd.DataFrame:
    """
    Add target columns to the master frame.

    For each row (ticker, day t), the target is whether (close at t+horizon)
    is higher than (close at t).

    Created columns:
      target_return:    raw forward return = close_{t+h}/close_t - 1
      target_direction: 1 if return > 0, else 0

    The last `horizon` rows per ticker get NaN target — they have no future yet.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    # Forward return per ticker
    df[return_col] = df.groupby("ticker")["close"].transform(
        lambda s: s.shift(-horizon) / s - 1.0
    )
    df[target_col] = (df[return_col] > 0).astype("Int64")
    df.loc[df[return_col].isna(), target_col] = pd.NA

    n_complete = df[target_col].notna().sum()
    logger.info(
        f"[dataset] built target horizon={horizon}d: "
        f"{n_complete}/{len(df)} rows with target "
        f"(class balance: {df[target_col].dropna().mean():.2%} positive)"
    )
    return df


# ============================================================
# Feature column selection
# ============================================================
NON_FEATURE_COLS = {
    "ticker", "timestamp", "symbol",
    "open", "high", "low", "close", "volume", "vwap", "trade_count",
    "target_direction", "target_return",
}


def select_feature_columns(
    df: pd.DataFrame,
    exclude_macro: bool = False,
    exclude_sentiment: bool = True,
    exclude_categorical: bool = True,
) -> list[str]:
    """
    Return feature columns from a master frame.

    Defaults are chosen for the v2 baseline (macro time-aligned, sentiment broadcast):
    - exclude_macro=False: macro is now time-aligned (v2 P0 shipped). Include it.
    - exclude_sentiment=True: sentiment is still broadcast as a snapshot
      (not yet time-aligned). Once P1 ships and news history accumulates, set False.
    - exclude_categorical=True: 'vol_regime' is a string; LightGBM can handle
      it as categorical, but skip for the clean baseline.
    """
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if exclude_macro:
        cols = [c for c in cols if not c.startswith("macro_")]
    if exclude_sentiment:
        cols = [c for c in cols if not c.startswith("sent_")]
    if exclude_categorical:
        cols = [c for c in cols if df[c].dtype != "object"]
    return cols


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "target_direction",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return (X, y, feature_names) ready for the model."""
    feat_cols = select_feature_columns(df)
    X = df[feat_cols].copy()
    y = df[target_col].astype("float64")
    return X, y, feat_cols


# ============================================================
# Time-aware splits (legacy single-split, used by train.py)
# ============================================================
def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically: oldest 70% → train, next 15% → val, last 15% → test.
    Splits are made on time (not row index), so all tickers are split at the
    same date.

    NOTE: For more rigorous evaluation, use walk_forward.run_walkforward()
    instead. This single-split is preserved for the legacy train.py pipeline.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"].sort_values().unique()
    n = len(timestamps)
    train_end = timestamps[int(n * train_frac)]
    val_end = timestamps[int(n * (train_frac + val_frac))]

    train = df[df["timestamp"] <= train_end].copy()
    val   = df[(df["timestamp"] > train_end) & (df["timestamp"] <= val_end)].copy()
    test  = df[df["timestamp"] > val_end].copy()

    logger.info(
        f"[dataset] chronological split: "
        f"train {len(train)} ({train['timestamp'].min().date()}..{train['timestamp'].max().date()}), "
        f"val {len(val)} ({val['timestamp'].min().date()}..{val['timestamp'].max().date()}), "
        f"test {len(test)} ({test['timestamp'].min().date()}..{test['timestamp'].max().date()})"
    )
    return train, val, test


# ============================================================
# Cleanup helpers
# ============================================================
def drop_warmup_and_no_target(
    df: pd.DataFrame,
    min_required_features: list[str] | None = None,
    target_col: str = "target_direction",
) -> pd.DataFrame:
    """
    Drop:
    - Rows where target is NaN (last `horizon` rows per ticker, plus first `horizon` of any series)
    - Rows where critical warmup features are NaN (e.g. sma_200 for first 200 bars)

    `min_required_features` defaults to ['sma_200'] — the longest-warmup feature.
    """
    if min_required_features is None:
        min_required_features = ["sma_200"]

    initial = len(df)
    df = df.dropna(subset=[target_col])
    n_after_target = len(df)

    df = df.dropna(subset=[c for c in min_required_features if c in df.columns])
    n_after_features = len(df)

    logger.info(
        f"[dataset] dropped {initial - n_after_target} rows missing target, "
        f"{n_after_target - n_after_features} more missing required warmup features. "
        f"{n_after_features} rows remain."
    )
    return df.reset_index(drop=True)


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    print("\n--- Loading master frame ---")
    df = load_master()
    print(f"  shape: {df.shape}")
    print(f"  date range: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  tickers: {sorted(df['ticker'].unique())}")

    print("\n--- Building targets (horizon=5d) ---")
    df = build_targets(df, horizon=5)

    print("\n--- Target distribution per ticker ---")
    summary = df.groupby("ticker").agg(
        n_rows=("close", "count"),
        n_with_target=("target_direction", lambda s: s.notna().sum()),
        pct_positive=("target_direction", lambda s: s.dropna().mean()),
        avg_return=("target_return", "mean"),
    ).round(4)
    print(summary.to_string())

    print("\n--- Dropping warmup + no-target rows ---")
    df_clean = drop_warmup_and_no_target(df)
    print(f"  cleaned shape: {df_clean.shape}")

    print("\n--- Feature columns (default exclusions) ---")
    feat_cols = select_feature_columns(df_clean)
    print(f"  {len(feat_cols)} feature columns (macro included, sentiment excluded)")
