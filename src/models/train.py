"""
End-to-end training script for Agent 29.

Pipeline:
1. Load master features
2. Build next-day direction target
3. Drop warmup rows + rows without target
4. Chronological train/val/test split
5. Train LightGBM with early stopping
6. Evaluate on val + test (overall + per-ticker)
7. Show feature importance
8. Save the model

Run:
    python -m src.models.train
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.dataset import (
    build_targets,
    chronological_split,
    drop_warmup_and_no_target,
    load_master,
    select_feature_columns,
)
from src.models.lightgbm_classifier import LGBMDirectionClassifier, MODELS_DIR


def main(horizon: int = 1) -> None:
    print("=" * 70)
    print(f"Agent 29 — LightGBM training (horizon={horizon}d)")
    print("=" * 70)

    # 1. Load + target
    print("\n[1/7] Loading master frame and building targets...")
    df = load_master()
    df = build_targets(df, horizon=horizon)
    df = drop_warmup_and_no_target(df)

    # 2. Feature selection (technical + sentiment, no broken macro)
    feat_cols = select_feature_columns(
        df,
        exclude_macro=True,
        exclude_sentiment=False,
        exclude_categorical=True,
    )
    print(f"     using {len(feat_cols)} features")

    # 3. Chronological split (preserves time order — never random)
    print("\n[2/7] Chronological split...")
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)

    X_train, y_train = train[feat_cols], train["target_direction"]
    X_val,   y_val   = val[feat_cols],   val["target_direction"]
    X_test,  y_test  = test[feat_cols],  test["target_direction"]

    # Ticker meta for per-ticker eval
    meta_test = test[["ticker", "timestamp"]].reset_index(drop=True)
    meta_val  = val[["ticker", "timestamp"]].reset_index(drop=True)

    # 4. Train
    print("\n[3/7] Training LightGBM...")
    clf = LGBMDirectionClassifier()
    clf.train(X_train, y_train, X_val, y_val,
              num_boost_round=500, early_stopping_rounds=30)

    # 5. Evaluate
    print("\n[4/7] Evaluating on VALIDATION set...")
    val_metrics = clf.evaluate(X_val, y_val, meta=meta_val)
    _print_metrics(val_metrics, label="VAL")

    print("\n[5/7] Evaluating on TEST set (held out, never seen during training)...")
    test_metrics = clf.evaluate(X_test, y_test, meta=meta_test)
    _print_metrics(test_metrics, label="TEST")

    # 6. Feature importance
    print("\n[6/7] Top 20 features by gain...")
    imp = clf.feature_importance(top_n=20)
    print(imp.to_string(index=False))

    # 7. Save
    print("\n[7/7] Saving model...")
    path = clf.save()
    print(f"     model saved to {path}")

    # Summary line for easy log scanning
    print("\n" + "=" * 70)
    print(f"SUMMARY  test_acc={test_metrics['accuracy']:.4f}  "
          f"baseline={test_metrics['baseline']:.4f}  "
          f"lift={test_metrics['lift_over_baseline']:+.4f}  "
          f"auc={test_metrics['auc']:.4f}")
    print("=" * 70)


def _print_metrics(m: dict, label: str) -> None:
    print(f"  [{label}] n={m['n']}, "
          f"accuracy={m['accuracy']:.4f}, "
          f"baseline={m['baseline']:.4f}, "
          f"lift={m['lift_over_baseline']:+.4f}, "
          f"auc={m['auc']:.4f}, "
          f"logloss={m['logloss']:.4f}")
    print(f"  confusion matrix [tn fp; fn tp]: {m['confusion']}")
    if "per_ticker" in m:
        print(f"  per-ticker accuracy:")
        for r in sorted(m["per_ticker"], key=lambda x: -x["accuracy"]):
            print(f"    {r['ticker']:6s} n={r['n']:3d}  "
                  f"acc={r['accuracy']:.3f}  "
                  f"avg_proba={r['avg_proba']:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1,
                        help="Forward return horizon in days (1, 5, 20, etc.)")
    args = parser.parse_args()
    main(horizon=args.horizon)
