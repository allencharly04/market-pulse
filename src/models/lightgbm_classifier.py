"""
LightGBM binary classifier for next-day direction prediction.

Wraps LightGBM with sensible defaults for small-N tabular finance data:
- Conservative regularization (small dataset, lots of features → easy to overfit)
- Early stopping on validation set
- Built-in feature importance
- Per-ticker accuracy breakdown for diagnostics

Usage:
    from src.models.lightgbm_classifier import LGBMDirectionClassifier
    clf = LGBMDirectionClassifier()
    clf.train(X_train, y_train, X_val, y_val)
    preds = clf.predict(X_test)
    clf.evaluate(X_test, y_test, ticker_col=test_meta['ticker'])
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


MODELS_DIR = Path("data/models")


# ============================================================
# Default hyperparameters
# ============================================================
# Tuned for small-N tabular finance:
# - num_leaves low → less overfit
# - learning_rate moderate → fast convergence with early stopping
# - feature_fraction + bagging → de-correlate trees
# - min_data_in_leaf relatively high → less overfit on rare patterns
DEFAULT_PARAMS = {
    "objective":        "binary",
    "metric":           ["binary_logloss", "auc"],
    "boosting_type":    "gbdt",
    "num_leaves":       15,
    "max_depth":        4,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_data_in_leaf": 10,
    "lambda_l1":        0.1,
    "lambda_l2":        0.1,
    "verbose":          -1,
    "force_col_wise":   True,
}


class LGBMDirectionClassifier:
    """LightGBM binary classifier for next-day direction (up/down)."""

    def __init__(self, params: dict | None = None):
        self.params = params or DEFAULT_PARAMS.copy()
        self.model: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self.train_history: dict = {}

    # ============================================================
    # Training
    # ============================================================
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 30,
    ) -> "LGBMDirectionClassifier":
        """Train with early stopping on validation set."""
        self.feature_names = list(X_train.columns)

        # LightGBM needs y as plain int — Int64/float64 with NaN doesn't work
        y_train_clean = y_train.astype("float64").astype("int32")
        y_val_clean   = y_val.astype("float64").astype("int32")

        train_set = lgb.Dataset(X_train, label=y_train_clean)
        val_set   = lgb.Dataset(X_val,   label=y_val_clean, reference=train_set)

        eval_history: dict = {}
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=50),
                lgb.record_evaluation(eval_history),
            ],
        )
        self.train_history = eval_history

        best_iter = self.model.best_iteration
        best_val_loss = eval_history["val"]["binary_logloss"][best_iter - 1]
        best_val_auc  = eval_history["val"]["auc"][best_iter - 1]
        logger.success(
            f"[lgbm] training done. best_iter={best_iter}, "
            f"val_logloss={best_val_loss:.4f}, val_auc={best_val_auc:.4f}"
        )
        return self

    # ============================================================
    # Inference
    # ============================================================
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up) for each row."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return 0/1 predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ============================================================
    # Evaluation
    # ============================================================
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5,
        meta: pd.DataFrame | None = None,
    ) -> dict:
        """
        Compute metrics on a held-out set.
        If `meta` (with ticker, timestamp columns) is provided, also compute
        per-ticker accuracy.
        """
        y_clean = y.astype("float64").astype("int32")
        proba = self.predict_proba(X)
        preds = (proba >= threshold).astype(int)

        metrics = {
            "n":         len(y),
            "accuracy":  accuracy_score(y_clean, preds),
            "auc":       roc_auc_score(y_clean, proba) if len(np.unique(y_clean)) > 1 else float("nan"),
            "logloss":   log_loss(y_clean, np.clip(proba, 1e-7, 1 - 1e-7)),
            "baseline":  max(y_clean.mean(), 1 - y_clean.mean()),  # always-predict-majority
        }
        metrics["lift_over_baseline"] = metrics["accuracy"] - metrics["baseline"]
        metrics["confusion"] = confusion_matrix(y_clean, preds).tolist()

        if meta is not None and "ticker" in meta.columns:
            per_ticker = []
            for t in meta["ticker"].unique():
                mask = meta["ticker"] == t
                if mask.sum() < 3:
                    continue
                acc = accuracy_score(y_clean[mask.values], preds[mask.values])
                per_ticker.append({
                    "ticker":   t,
                    "n":        int(mask.sum()),
                    "accuracy": acc,
                    "avg_proba": float(proba[mask.values].mean()),
                })
            metrics["per_ticker"] = per_ticker

        return metrics

    # ============================================================
    # Feature importance
    # ============================================================
    def feature_importance(self, top_n: int | None = None) -> pd.DataFrame:
        """Return feature importance sorted descending."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        gain = self.model.feature_importance(importance_type="gain")
        split = self.model.feature_importance(importance_type="split")
        df = pd.DataFrame({
            "feature":         self.feature_names,
            "importance_gain": gain,
            "importance_split": split,
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True)
        if top_n:
            df = df.head(top_n)
        return df

    # ============================================================
    # Persistence
    # ============================================================
    def save(self, path: Path | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("Model not trained")
        path = path or (MODELS_DIR / f"lgbm_direction_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.success(f"[lgbm] saved to {path}")
        return path

    def load(self, path: Path) -> "LGBMDirectionClassifier":
        self.model = lgb.Booster(model_file=str(path))
        self.feature_names = self.model.feature_name()
        logger.info(f"[lgbm] loaded from {path}")
        return self
