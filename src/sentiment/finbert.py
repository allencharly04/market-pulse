"""
FinBERT sentiment scorer for Agent 29.

Uses ProsusAI/finbert from Hugging Face — a BERT model fine-tuned on
financial news (Reuters TRC2 dataset, ~10K labeled financial sentences).

Outputs softmax probabilities over [positive, negative, neutral].

Performance:
- First run: downloads ~440MB model (one-time, cached at ~/.cache/huggingface)
- GPU (RTX 2060): ~1000-2000 headlines/second in batches of 32
- CPU: ~50-100 headlines/second (still usable)

Model: https://huggingface.co/ProsusAI/finbert
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]  # FinBERT's label order
MAX_LENGTH = 128  # most headlines fit; longer get truncated


class FinBERTScorer:
    """GPU-accelerated financial sentiment scorer."""

    name = "finbert"

    def __init__(self, device: str | None = None, batch_size: int = 32):
        # Auto-pick device: prefer GPU if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.batch_size = batch_size

        logger.info(f"[{self.name}] loading {MODEL_NAME} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()  # inference mode, no grad
        logger.success(
            f"[{self.name}] loaded. "
            f"params={sum(p.numel() for p in self.model.parameters())/1e6:.0f}M"
        )

    @torch.no_grad()
    def score_batch(self, texts: list[str]) -> pd.DataFrame:
        """
        Score a batch of texts. Returns DataFrame with columns:
          positive, negative, neutral, label, compound
        compound = positive - negative (analogous to VADER)
        """
        if not texts:
            return pd.DataFrame(columns=["positive", "negative", "neutral", "label", "compound"])

        # Filter empty / non-string entries
        clean_texts = [(t if isinstance(t, str) and t.strip() else "neutral text") for t in texts]

        all_probs: list[list[float]] = []

        # Process in mini-batches to fit in GPU memory
        n_batches = math.ceil(len(clean_texts) / self.batch_size)
        for i in range(n_batches):
            batch = clean_texts[i * self.batch_size : (i + 1) * self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().tolist()
            all_probs.extend(probs)

        # FinBERT label order: positive, negative, neutral
        df = pd.DataFrame(all_probs, columns=LABELS)
        df["label"] = df[LABELS].idxmax(axis=1)
        df["compound"] = df["positive"] - df["negative"]
        return df

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "title",
        prefix: str = "finbert",
    ) -> pd.DataFrame:
        """
        Add FinBERT columns to a DataFrame in-place.

        Adds: {prefix}_pos, {prefix}_neg, {prefix}_neu,
              {prefix}_label, {prefix}_compound
        """
        if df.empty or text_col not in df.columns:
            return df

        scores = self.score_batch(df[text_col].fillna("").tolist())

        df[f"{prefix}_pos"]      = scores["positive"].values
        df[f"{prefix}_neg"]      = scores["negative"].values
        df[f"{prefix}_neu"]      = scores["neutral"].values
        df[f"{prefix}_label"]    = scores["label"].values
        df[f"{prefix}_compound"] = scores["compound"].values
        return df

    def aggregate_features(
        self,
        df: pd.DataFrame,
        prefix: str = "finbert",
    ) -> dict:
        """
        After scoring a batch, compute aggregate features:
          - avg compound score
          - net sentiment (count(pos) - count(neg))
          - dominant label
          - confidence (avg of dominant-class probability)
        """
        if df.empty:
            return {}

        compound_col = f"{prefix}_compound"
        label_col = f"{prefix}_label"
        if compound_col not in df.columns or label_col not in df.columns:
            return {}

        n_pos = (df[label_col] == "positive").sum()
        n_neg = (df[label_col] == "negative").sum()
        n_neu = (df[label_col] == "neutral").sum()
        total = len(df)

        return {
            f"{prefix}_avg_compound":   float(df[compound_col].mean()),
            f"{prefix}_net_sentiment":  int(n_pos - n_neg),
            f"{prefix}_pct_positive":   float(n_pos / total) if total else 0.0,
            f"{prefix}_pct_negative":   float(n_neg / total) if total else 0.0,
            f"{prefix}_pct_neutral":    float(n_neu / total) if total else 0.0,
            f"{prefix}_dominant_label": df[label_col].mode().iloc[0] if total else "neutral",
            f"{prefix}_n_scored":       total,
        }


if __name__ == "__main__":
    scorer = FinBERTScorer()

    # Same test set as VADER, plus the ones VADER got wrong
    test_headlines = [
        "Nvidia stock surges 12% on blowout Q3 earnings, AI chip demand exceeds expectations",
        "Tesla shares plummet as Q3 deliveries miss analyst estimates",
        "Federal Reserve holds rates steady, signals patience on future cuts",
        "Bitcoin breaks above $100,000 for first time in history",
        "Major bank announces $2 billion writedown amid commercial real estate fears",
        "Apple reports record iPhone sales, beats Wall Street forecasts",
        "SEC charges crypto exchange with fraud, freezes $500M in assets",
        "Markets close mixed after volatile trading session",
        "Hedge fund liquidates positions amid mounting losses",
        "Company posts strong revenue growth, raises full-year guidance",
    ]

    print("\n--- FinBERT scoring ---\n")
    df = pd.DataFrame({"title": test_headlines})
    df = scorer.score_dataframe(df)

    print(f"{'Compound':>10s}  {'Label':>10s}  Headline")
    print("-" * 110)
    for _, row in df.iterrows():
        print(f"{row['finbert_compound']:>10.3f}  {row['finbert_label']:>10s}  {row['title'][:85]}")

    print("\n--- Aggregate features ---")
    feats = scorer.aggregate_features(df)
    for k, v in feats.items():
        if isinstance(v, float):
            print(f"  {k:30s} = {v:.4f}")
        else:
            print(f"  {k:30s} = {v}")

    # Compare to VADER on the same set
    print("\n--- Speed test: 1000 headlines ---")
    bulk = test_headlines * 100
    import time
    t0 = time.time()
    bulk_df = scorer.score_batch(bulk)
    elapsed = time.time() - t0
    print(f"  Scored {len(bulk)} headlines in {elapsed:.2f}s")
    print(f"  = {len(bulk) / elapsed:.0f} headlines/second")
    print(f"  Device: {scorer.device}")
