"""
Per-ticker walk-forward CV for Market Pulse.

Theory: pooling all tickers into one universal model dilutes signal across
heterogeneous instruments (AMD ≠ JNJ ≠ NVDA). Training a separate model
per ticker often outperforms — at the cost of less data per model.

This module:
1. For each ticker, runs walk-forward CV using only that ticker's rows
2. Aggregates results across tickers
3. Reports both per-ticker and overall metrics
4. Compares to the universal-model baseline (run walk_forward.py first)

Usage:
    python -m src.models.per_ticker --horizon 5
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
from src.models.walk_forward import make_walkforward_folds


# ============================================================
# Per-ticker walk-forward
# ============================================================
def run_per_ticker_walkforward(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_direction",
    initial_train_months: int = 12,
    test_window_months: int = 1,
    step_months: int = 1,
    scheme: str = "expanding",
    min_rows_per_ticker: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run walk-forward CV separately for each ticker.

    Returns:
        per_fold_df: one row per (ticker, fold)
        per_ticker_summary_df: one row per ticker (aggregated across folds)
    """
    all_fold_results = []
    tickers = sorted(df["ticker"].unique())

    logger.info(f"[per_ticker] running walk-forward for {len(tickers)} tickers")

    for ticker in tickers:
        ticker_df = df[df["ticker"] == ticker].copy().reset_index(drop=True)

        if len(ticker_df) < min_rows_per_ticker:
            logger.warning(
                f"[per_ticker:{ticker}] only {len(ticker_df)} rows, skipping"
            )
            continue

        folds = make_walkforward_folds(
            ticker_df,
            initial_train_months=initial_train_months,
            test_window_months=test_window_months,
            step_months=step_months,
            scheme=scheme,
        )

        if not folds:
            logger.warning(f"[per_ticker:{ticker}] no folds produced, skipping")
            continue

        for fold in folds:
            train_df = ticker_df[fold["train_mask"]].copy()
            test_df  = ticker_df[fold["test_mask"]].copy()

            # 85/15 train/val split inside the train window
            split_idx = int(len(train_df) * 0.85)
            actual_train = train_df.iloc[:split_idx]
            val_df       = train_df.iloc[split_idx:]

            if len(val_df) < 5 or len(actual_train) < 50:
                continue

            X_train = actual_train[feature_cols]
            y_train = actual_train[target_col]
            X_val   = val_df[feature_cols]
            y_val   = val_df[target_col]
            X_test  = test_df[feature_cols]
            y_test  = test_df[target_col]

            try:
                clf = LGBMDirectionClassifier()
                clf.train(X_train, y_train, X_val, y_val,
                         num_boost_round=300, early_stopping_rounds=20)
                metrics = clf.evaluate(X_test, y_test)

                all_fold_results.append({
                    "ticker":      ticker,
                    "fold":        fold["fold_idx"],
                    "test_start":  fold["test_start"].date(),
                    "test_end":    fold["test_end"].date(),
                    "n_train":     fold["n_train"],
                    "n_test":      fold["n_test"],
                    "best_iter":   clf.model.best_iteration if clf.model else 0,
                    "test_acc":    metrics["accuracy"],
                    "baseline":    metrics["baseline"],
                    "lift":        metrics["lift_over_baseline"],
                    "auc":         metrics["auc"],
                })
            except Exception as e:
                logger.warning(f"[per_ticker:{ticker} fold {fold['fold_idx']}] error: {e}")
                continue

        # Quick per-ticker progress logging
        n_folds_run = sum(1 for r in all_fold_results if r["ticker"] == ticker)
        if n_folds_run > 0:
            ticker_lifts = [r["lift"] for r in all_fold_results if r["ticker"] == ticker]
            logger.info(
                f"[per_ticker:{ticker}] {n_folds_run} folds  "
                f"mean_lift={np.mean(ticker_lifts):+.4f}"
            )

    if not all_fold_results:
        return pd.DataFrame(), pd.DataFrame()

    per_fold_df = pd.DataFrame(all_fold_results)

    # Per-ticker aggregates
    per_ticker_summary_df = (
        per_fold_df.groupby("ticker")
        .agg(
            n_folds=("fold", "count"),
            mean_acc=("test_acc", "mean"),
            mean_baseline=("baseline", "mean"),
            mean_lift=("lift", "mean"),
            std_lift=("lift", "std"),
            mean_auc=("auc", "mean"),
            folds_positive=("lift", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("mean_lift", ascending=False)
        .reset_index(drop=True)
    )
    per_ticker_summary_df["pct_folds_positive"] = (
        per_ticker_summary_df["folds_positive"] / per_ticker_summary_df["n_folds"]
    )

    return per_fold_df, per_ticker_summary_df


# ============================================================
# Main
# ============================================================
def main(
    horizon: int = 5,
    initial_train_months: int = 12,
    test_window_months: int = 1,
    scheme: str = "expanding",
) -> None:
    print("=" * 78)
    print(f"Market Pulse — Per-ticker Walk-forward CV (horizon={horizon}d, scheme={scheme})")
    print("=" * 78)

    # Load + prep
    print("\n[1/5] Loading master frame...")
    df = load_master()
    df = build_targets(df, horizon=horizon)
    df = drop_warmup_and_no_target(df)

    feat_cols = select_feature_columns(
        df,
        exclude_macro=False,
        exclude_sentiment=True,
        exclude_categorical=True,
    )
    print(f"     using {len(feat_cols)} features (macro included, sentiment excluded)")

    # Run
    print(f"\n[2/5] Running per-ticker walk-forward (this trains many models, takes a few minutes)...")
    per_fold, per_ticker = run_per_ticker_walkforward(
        df,
        feature_cols=feat_cols,
        initial_train_months=initial_train_months,
        test_window_months=test_window_months,
        scheme=scheme,
    )

    if per_fold.empty:
        print("\nNo results produced.")
        return

    # Per-ticker summary
    print("\n[3/5] Per-ticker summary (sorted by mean lift):")
    display = per_ticker.copy()
    for col in ["mean_acc", "mean_baseline", "mean_lift", "std_lift", "mean_auc", "pct_folds_positive"]:
        display[col] = display[col].round(4)
    print(display.to_string(index=False))

    # Overall
    print("\n[4/5] Overall (across all ticker × fold combinations):")
    print(f"     total fold-runs       = {len(per_fold)}")
    print(f"     mean accuracy         = {per_fold['test_acc'].mean():.4f}")
    print(f"     mean baseline         = {per_fold['baseline'].mean():.4f}")
    print(f"     mean lift             = {per_fold['lift'].mean():+.4f}")
    print(f"     mean auc              = {per_fold['auc'].mean():.4f}")
    print(f"     std lift              = {per_fold['lift'].std():.4f}")
    print(f"     fold-runs w/ lift>0   = {(per_fold['lift'] > 0).sum()} / {len(per_fold)}")
    print(f"     tickers w/ mean_lift>0 = {(per_ticker['mean_lift'] > 0).sum()} / {len(per_ticker)}")

    # Best and worst tickers
    print("\n[5/5] Best and worst tickers:")
    print(f"     Best  → {per_ticker.iloc[0]['ticker']}: mean_lift={per_ticker.iloc[0]['mean_lift']:+.4f}, mean_auc={per_ticker.iloc[0]['mean_auc']:.3f}")
    print(f"     Worst → {per_ticker.iloc[-1]['ticker']}: mean_lift={per_ticker.iloc[-1]['mean_lift']:+.4f}, mean_auc={per_ticker.iloc[-1]['mean_auc']:.3f}")

    # Final summary
    print("\n" + "=" * 78)
    print(f"PER-TICKER SUMMARY  horizon={horizon}d  "
          f"tickers={len(per_ticker)}  "
          f"total_folds={len(per_fold)}  "
          f"mean_lift={per_fold['lift'].mean():+.4f}  "
          f"mean_auc={per_fold['auc'].mean():.4f}")
    print("=" * 78)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=5)
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
