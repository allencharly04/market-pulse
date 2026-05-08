"""
VADER sentiment scorer for Agent 29.

Fast rule-based sentiment scoring. Not financial-domain-specific,
but serves as:
1. A baseline to compare FinBERT against
2. A fallback when FinBERT is unavailable (no GPU)
3. A first-pass filter for very obvious headlines

Speed: ~100,000 headlines/second on a single CPU core.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# VADER returns 4 scores: neg, neu, pos, compound
# - compound: a single score from -1 (most negative) to +1 (most positive)
# - We'll use compound as the primary score and add a 3-class label
COMPOUND_THRESHOLD = 0.05  # standard VADER convention


class VaderScorer:
    """Fast rule-based sentiment scorer."""

    name = "vader"

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        logger.info(f"[{self.name}] initialised (CPU, no model download)")

    def score(self, text: str) -> dict:
        """Score a single text. Returns dict with neg/neu/pos/compound/label."""
        if not text or not isinstance(text, str):
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0, "label": "neutral"}

        scores = self.analyzer.polarity_scores(text)
        compound = scores["compound"]

        if compound >= COMPOUND_THRESHOLD:
            label = "positive"
        elif compound <= -COMPOUND_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": compound,
            "label": label,
        }

    def score_batch(self, texts: list[str]) -> pd.DataFrame:
        """Score a batch of texts. Returns DataFrame indexed by position."""
        rows = [self.score(t) for t in texts]
        df = pd.DataFrame(rows)
        return df

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "title",
        prefix: str = "vader",
    ) -> pd.DataFrame:
        """
        Add VADER columns to a DataFrame in-place.

        Adds:  {prefix}_compound, {prefix}_label, {prefix}_pos, {prefix}_neg
        """
        if df.empty or text_col not in df.columns:
            return df

        scores = self.score_batch(df[text_col].fillna("").tolist())
        df[f"{prefix}_compound"] = scores["compound"].values
        df[f"{prefix}_label"]    = scores["label"].values
        df[f"{prefix}_pos"]      = scores["pos"].values
        df[f"{prefix}_neg"]      = scores["neg"].values
        return df


if __name__ == "__main__":
    scorer = VaderScorer()

    # Test on sample financial headlines
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

    print("\n--- VADER scoring test ---\n")
    print(f"{'Compound':>10s}  {'Label':>10s}  Headline")
    print("-" * 100)

    for headline in test_headlines:
        s = scorer.score(headline)
        print(f"{s['compound']:>10.3f}  {s['label']:>10s}  {headline[:80]}")

    print("\n--- Batch scoring ---")
    df = pd.DataFrame({"title": test_headlines})
    df = scorer.score_dataframe(df)
    print(df[["vader_label", "vader_compound", "title"]].to_string(index=False, max_colwidth=60))

    # Show distribution
    print("\n--- Label distribution ---")
    print(df["vader_label"].value_counts().to_string())
