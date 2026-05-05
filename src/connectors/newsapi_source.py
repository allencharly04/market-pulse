"""
NewsAPI source for Agent 29.

Pulls news headlines for a given ticker/keyword. Features:
- headline_count_1h, headline_count_24h
- avg_headline_length
- source_diversity (number of unique outlets)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

from src.connectors.base import DataSource, SourceHealth

load_dotenv()


class NewsAPISource(DataSource):
    name = "newsapi"
    category = "news"
    requires_auth = True
    rate_limit_per_min = 1  # free tier: 100/day

    BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, enabled: bool = True, weight: float = 1.0):
        super().__init__(enabled=enabled, weight=weight)
        self.api_key = os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise RuntimeError("NEWSAPI_KEY missing in .env")

    def fetch(self, query: str = "stock market", days: int = 7, page_size: int = 100) -> pd.DataFrame:
        """
        Fetch articles matching `query` from the past `days` days.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        params = {
            "q": query,
            "from": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self.api_key,
        }

        logger.info(f"[{self.name}] fetching '{query}' from past {days}d")
        r = requests.get(self.BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "ok":
            raise RuntimeError(f"NewsAPI error: {data.get('message')}")

        articles = data.get("articles", [])
        if not articles:
            return pd.DataFrame()

        rows = []
        for a in articles:
            rows.append({
                "published_at": pd.to_datetime(a.get("publishedAt"), utc=True),
                "source": (a.get("source") or {}).get("name"),
                "author": a.get("author"),
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "query": query,
            })
        df = pd.DataFrame(rows).sort_values("published_at", ascending=False)
        return df

    def to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw articles into time-windowed features.
        Returns one row per query with summary stats.
        """
        if raw.empty:
            return pd.DataFrame()

        now = datetime.now(timezone.utc)
        h1_cutoff = now - timedelta(hours=1)
        h24_cutoff = now - timedelta(hours=24)

        recent_1h = raw[raw["published_at"] >= h1_cutoff]
        recent_24h = raw[raw["published_at"] >= h24_cutoff]

        feat = pd.DataFrame([{
            "newsapi_count_1h":      len(recent_1h),
            "newsapi_count_24h":     len(recent_24h),
            "newsapi_source_div_24h": recent_24h["source"].nunique() if not recent_24h.empty else 0,
            "newsapi_avg_title_len_24h": recent_24h["title"].fillna("").str.len().mean() if not recent_24h.empty else 0,
            "newsapi_query":         raw["query"].iloc[0] if not raw.empty else None,
            "newsapi_fetched_at":    now,
        }])
        return feat

    def health_check(self) -> SourceHealth:
        t0 = datetime.now(timezone.utc)
        try:
            r = requests.get(
                self.BASE_URL,
                params={"q": "test", "pageSize": 1, "apiKey": self.api_key},
                timeout=10,
            )
            healthy = r.status_code == 200
            elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
            error = None if healthy else f"HTTP {r.status_code}"
            return SourceHealth(
                name=self.name, healthy=healthy,
                last_check=datetime.now(timezone.utc),
                latency_ms=elapsed_ms, error=error,
            )
        except Exception as e:
            return SourceHealth(
                name=self.name, healthy=False,
                last_check=datetime.now(timezone.utc),
                latency_ms=-1, error=f"{type(e).__name__}: {e}",
            )


if __name__ == "__main__":
    src = NewsAPISource()

    print("\n--- Health check ---")
    health = src.health_check()
    print(f"  healthy: {health.healthy}")
    print(f"  latency: {health.latency_ms:.0f} ms")
    if health.error:
        print(f"  error: {health.error}")

    print("\n--- Fetch: 'NVIDIA stock' last 1 day ---")
    df = src.fetch(query="NVIDIA stock", days=7)
    print(f"  rows: {len(df)}")
    if not df.empty:
        print(df[["published_at", "source", "title"]].head(5).to_string(index=False))

    print("\n--- Features ---")
    feat = src.to_features(df)
    print(feat.T)
