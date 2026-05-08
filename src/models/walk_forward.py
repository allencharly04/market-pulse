"""
Walk-forward cross-validation for Market Pulse.

Walk-forward CV is the correct evaluation methodology for time-series ML:
- Never train on data from the future
- Test on the period right after the training window
- Slide the window forward and repeat
- Average metrics across all folds

This gives a much more honest estimate of out-of-sample performance than
a single chronological train/test split.

Two scheme options:
- 'expanding': train window grows over time (mimics deploying once and
  retraining as new data arrives)
- 'rolling':   train window has fixed size (mimics deploying with a
  forgetting horizon)

We use 'expanding' as default — it's the more common choice for finance
when you have ~3 years of data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.dataset import (
    build_targets,
    drop_warmup_and_no_target,
    load_master,
    select_feature_columns,
)
from src.models.lightgbm_classifier import LGBMDirectionClassifier


# ============================================================
# Fold construction
# ============================================================
def make_walkforward_folds(
    df: pd.DataFrame,
    initial_train_months: int = 12,
    test_window_months: int = 1,
    step_months: int = 1,
    scheme: str = "expanding",
) -> list[dict]:
    """
    Build a list of fold dicts. Each dict has:
        train_mask, test_mask, fold_idx,
        train_start, train_end, test_start, test_end

    Args:
        initial_train_months: minimum training window before first fold
        test_window_months:   how long each test period is
        step_months:          how far to slide between folds
        scheme:               'expanding' or 'rolling'

    Returns:
        list of fold dicts, each with boolean masks over df rows.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")

    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    earliest = timestamps.min()
    latest = timestamps.max()

    folds = []
    fold_idx = 0

    # Anchor the rolling cursor at the first possible test_start
    test_start = earliest + pd.DateOffset(months=initial_train_months)

    while test_start + pd.DateOffset(months=test_window_months) <= latest + pd.Timedelta(days=1):
        test_end = test_start + pd.DateOffset(months=test_window_months)

        if scheme == "expanding":
            train_start = earliest
        elif scheme == "rolling":
            train_start = test_start - pd.DateOffset(months=initial_train_months)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

        train_end = test_start  # train period excludes test

        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        test_mask  = (timestamps >= test_start) & (timestamps < test_end)

        if train_mask.sum() < 100 or test_mask.sum() < 20:
            # Skip folds that are too tiny
            test_start = test_start + pd.DateOffset(months=step_months)
            continue

        folds.append({
            "fold_idx":    fold_idx,
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  test_start,
            "test_end":    test_end,
            "train_mask":  train_mask.values,
            "test_mask":   test_mask.values,
            "n_train":     int(train_mask.sum()),
            "n_test":      int(test_mask.sum()),
        })
        fold_idx += 1
        test_start = test_start + pd.DateOffset(months=step_months)

    logger.info(
        f"[walk_forward] built {len(folds)} {scheme} folds "
        f"(initial_train={initial_train_months}mo, test={test_window_months}mo, step={step_months}mo)"
    )
    return folds


# ============================================================
# Run all folds
# ============================================================
def run_walkforward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_direction",
    initial_train_months: int = 12,
    test_window_months: int = 1,
    step_months: int = 1,
    scheme: str = "expanding",
) -> pd.DataFrame:
    """
    Run walk-forward CV. Returns DataFrame with one row per fold.
    """
    folds = make_walkforward_folds(
        df,
        initial_train_months=initial_train_months,
        test_window_months=test_window_months,
        step_months=step_months,
        scheme=scheme,
    )

    if not folds:
        logger.error("[walk_forward] no folds produced — check date range and parameters")
        return pd.DataFrame()

    results = []

    for fold in folds:
        train_df = df[fold["train_mask"]].copy()
        test_df  = df[fold["test_mask"]].copy()

        # Within training, hold out the last 15% as val for early stopping
        train_split_idx = int(len(train_df) * 0.85)
        actual_train = train_df.iloc[:train_split_idx]
        val_df       = train_df.iloc[train_split_idx:]

        if len(val_df) < 20 or len(actual_train) < 100:
            logger.warning(
                f"[walk_forward fold {fold['fold_idx']}] insufficient data, skipping"
            )
            continue

        X_train = actual_train[feature_cols]
        y_train = actual_train[target_col]
        X_val   = val_df[feature_cols]
        y_val   = val_df[target_col]
        X_test  = test_df[feature_cols]
        y_test  = test_df[target_col]

        clf = LGBMDirectionClassifier()
        clf.train(X_train, y_train, X_val, y_val,
                 num_boost_round=500, early_stopping_rounds=30)

        metrics = clf.evaluate(X_test, y_test)

        result = {
            "fold":         fold["fold_idx"],
            "train_start":  fold["train_start"].date(),
            "train_end":    fold["train_end"].date(),
            "test_start":   fold["test_start"].date(),
            "test_end":     fold["test_end"].date(),
            "n_train":      fold["n_train"],
            "n_test":       fold["n_test"],
            "best_iter":    clf.model.best_iteration if clf.model else 0,
            "test_acc":     metrics["accuracy"],
            "baseline":     metrics["baseline"],
            "lift":         metrics["lift_over_baseline"],
            "auc":          metrics["auc"],
            "logloss":      metrics["logloss"],
        }
        results.append(result)

        logger.info(
            f"[fold {fold['fold_idx']}] "
            f"{fold['test_start'].date()}..{fold['test_end'].date()}  "
            f"acc={metrics['accuracy']:.3f}  "
            f"baseline={metrics['baseline']:.3f}  "
            f"lift={metrics['lift_over_baseline']:+.3f}  "
            f"auc={metrics['auc']:.3f}"
        )

    return pd.DataFrame(results)


# ============================================================
# Main entry
# ============================================================
def main(
    horizon: int = 1,
    initial_train_months: int = 12,
    test_window_months: int = 1,
    scheme: str = "expanding",
) -> None:
    print("=" * 78)
    print(f"Market Pulse — Walk-forward CV (horizon={horizon}d, scheme={scheme})")
    print("=" * 78)

    # Load + prep
    print("\n[1/4] Loading master frame...")
    df = load_master()
    df = build_targets(df, horizon=horizon)
    df = drop_warmup_and_no_target(df)

    feat_cols = select_feature_columns(
        df,
        exclude_macro=False,      # macro is now time-aligned, include it
        exclude_sentiment=True,   # sentiment is still broadcast, exclude it
        exclude_categorical=True,
    )
    print(f"     using {len(feat_cols)} features (macro included, sentiment excluded)")

    # Run folds
    print(f"\n[2/4] Running walk-forward folds (initial_train={initial_train_months}mo, test={test_window_months}mo)...")
    results = run_walkforward(
        df,
        feature_cols=feat_cols,
        initial_train_months=initial_train_months,
        test_window_months=test_window_months,
        scheme=scheme,
    )

    if results.empty:
        print("\nNo folds produced. Check parameters.")
        return

    # Per-fold table
    print("\n[3/4] Per-fold results:")
    print(results.round(4).to_string(index=False))

    # Aggregates
    print("\n[4/4] Aggregate metrics:")
    print(f"     mean acc       = {results['test_acc'].mean():.4f}")
    print(f"     mean baseline  = {results['baseline'].mean():.4f}")
    print(f"     mean lift      = {results['lift'].mean():+.4f}")
    print(f"     mean auc       = {results['auc'].mean():.4f}")
    print(f"     std lift       = {results['lift'].std():.4f}")
    print(f"     #folds w/ lift>0 = {(results['lift'] > 0).sum()} / {len(results)}")
    print(f"     #folds w/ auc>0.52 = {(results['auc'] > 0.52).sum()} / {len(results)}")

    # Summary line
    print("\n" + "=" * 78)
    print(f"WALKFORWARD SUMMARY  horizon={horizon}d  "
          f"folds={len(results)}  "
          f"mean_lift={results['lift'].mean():+.4f}  "
          f"mean_auc={results['auc'].mean():.4f}")
    print("=" * 78)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--initial-train-months", type=int, default=12)
    parser.add_argument("--test-window-months", type=int, default=1)
    parser.add_argument("--scheme", type=str, default="expanding",
                       choices=["expanding", "rolling"])
    args = parser.parse_args()
    main(
        horizon=args.horizon,
        initial_train_months=args.initial_train_months,
        test_window_months=args.test_window_months,
        scheme=args.scheme,
    )
