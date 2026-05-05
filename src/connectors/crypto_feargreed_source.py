"""
Crypto Fear & Greed Index source for Agent 29.

Aggregates several crypto sentiment dimensions (volatility, market momentum,
social media, surveys, BTC dominance, trends) into a single 0-100 score:
- 0-24:   Extreme Fear
- 25-44:  Fear
- 45-55:  Neutral
- 56-74:  Greed
- 75-100: Extreme Greed

Famous contrarian signal: extreme fear historically marks bottoms,
extreme greed marks tops. Not always right but a strong regime feature.

API: https://api.alternative.me/fng/
Free, public, no auth, no rate limits worth worrying about.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from loguru import logger

from src.connectors.base import DataSource, SourceHealth


class CryptoFearGreedSource(DataSource):
    name = "crypto_feargreed"
    category = "crypto"
    requires_auth = False
    rate_limit_per_min = 60

    BASE_URL = "https://api.alternative.me/fng/"

    def __init__(self, enabled: bool = True, weight: float = 1.0):
        super().__init__(enabled=enabled, weight=weight)

    def fetch(self, days: int = 90, **kwargs) -> pd.DataFrame:
        """
        Fetch the past `days` days of Fear & Greed scores.

        Returns DataFrame with: date, value (0-100), classification.
        """
        params = {"limit": days, "format": "json"}
        logger.info(f"[{self.name}] fetching {days}d of fear & greed scores")
        r = requests.get(self.BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        items = data.get("data", [])
        if not items:
            return pd.DataFrame()

        rows = []
        for item in items:
            rows.append({
                "datetime":       pd.to_datetime(int(item["timestamp"]), unit="s", utc=True),
                "value":          int(item["value"]),
                "classification": item["value_classification"],
            })
        df = pd.DataFrame(rows).sort_values("datetime", ascending=False).reset_index(drop=True)
        return df

    def to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        latest = raw.iloc[0]
        values = raw["value"].values  # newest first

        feat: dict = {
            "fng_latest":              int(latest["value"]),
            "fng_classification":      latest["classification"],
            "fng_fetched_at":          datetime.now(timezone.utc),
        }

        # Changes over windows (note: index 0 is newest)
        if len(values) >= 7:
            feat["fng_chg_7d"]  = int(values[0] - values[6])
        if len(values) >= 30:
            feat["fng_chg_30d"] = int(values[0] - values[29])

        # Z-score over the window
        if len(values) >= 20:
            mean = values.mean()
            std = values.std()
            feat["fng_zscore"] = float((values[0] - mean) / std) if std > 0 else 0.0

        # Regime flags (the contrarian-trade signal)
        v = latest["value"]
        feat["fng_extreme_fear"]  = int(v <= 24)
        feat["fng_fear"]          = int(25 <= v <= 44)
        feat["fng_neutral"]       = int(45 <= v <= 55)
        feat["fng_greed"]         = int(56 <= v <= 74)
        feat["fng_extreme_greed"] = int(v >= 75)

        # Volatility of sentiment (high = unstable regime)
        if len(values) >= 14:
            feat["fng_volatility_14d"] = float(values[:14].std())

        return pd.DataFrame([feat])

    def health_check(self) -> SourceHealth:
        t0 = datetime.now(timezone.utc)
        try:
            r = requests.get(self.BASE_URL, params={"limit": 1}, timeout=10)
            healthy = r.status_code == 200 and "data" in r.json()
            elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
            return SourceHealth(
                name=self.name, healthy=healthy,
                last_check=datetime.now(timezone.utc),
                latency_ms=elapsed_ms,
                error=None if healthy else f"HTTP {r.status_code}",
            )
        except Exception as e:
            return SourceHealth(
                name=self.name, healthy=False,
                last_check=datetime.now(timezone.utc),
                latency_ms=-1, error=f"{type(e).__name__}: {e}",
            )


if __name__ == "__main__":
    src = CryptoFearGreedSource()

    print("\n--- Health check ---")
    h = src.health_check()
    print(f"  healthy: {h.healthy}   latency: {h.latency_ms:.0f} ms")

    print("\n--- Fetch: 90 days ---")
    df = src.fetch(days=90)
    print(f"  rows: {len(df)}")

    if not df.empty:
        print(f"\n  Most recent 10 days:")
        for _, row in df.head(10).iterrows():
            print(f"    {row['datetime'].date()}  "
                  f"value={row['value']:>3}  "
                  f"({row['classification']})")

        print(f"\n  Distribution over last 90d:")
        print(df["classification"].value_counts().to_string())

    print("\n--- Features ---")
    feat = src.to_features(df)
    if not feat.empty:
        for col in feat.columns:
            print(f"  {col:30s} = {feat[col].iloc[0]}")
