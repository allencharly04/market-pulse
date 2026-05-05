"""
Finnhub source for Agent 29.

Free tier: 60 calls/minute, real-time (no delay), ticker-tagged news.
This is the PRIMARY real-time news source — use this over NewsAPI for
live signals.

Endpoints used:
- /company-news       : ticker-specific news
- /news               : general market/crypto news
- /news-sentiment     : Finnhub's pre-computed sentiment scores
- /calendar/earnings  : upcoming earnings (used for risk-off filtering)
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


class FinnhubSource(DataSource):
    name = "finnhub"
    category = "news"
    requires_auth = True
    rate_limit_per_min = 60

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, enabled: bool = True, weight: float = 1.2):
        super().__init__(enabled=enabled, weight=weight)
        self.api_key = os.getenv("FINNHUB_KEY")
        if not self.api_key:
            raise RuntimeError("FINNHUB_KEY missing in .env")

    # ---------- Endpoints ----------
    def _get(self, endpoint: str, params: dict) -> dict | list:
        params = {**params, "token": self.api_key}
        url = f"{self.BASE_URL}/{endpoint}"
        r = requests.get(url, params=params, timeout=15)
        if not r.ok:
            safe_url = f"{url}?" + "&".join(
                f"{k}=***" if k == "token" else f"{k}={v}"
                for k, v in params.items()
            )
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} for {safe_url}", response=r
            )
        return r.json()

    def fetch_company_news(self, ticker: str, days: int = 7) -> pd.DataFrame:
        """
        Real-time company news for a specific ticker.

        Returns DataFrame with: datetime, headline, summary, source, url, ticker
        """
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days)

        logger.info(f"[{self.name}] fetching company news for {ticker} ({days}d)")
        articles = self._get(
            "company-news",
            {"symbol": ticker, "from": str(start), "to": str(end)},
        )

        if not articles:
            return pd.DataFrame()

        rows = []
        for a in articles:
            rows.append({
                "datetime":  pd.to_datetime(a.get("datetime"), unit="s", utc=True),
                "ticker":    ticker,
                "category":  a.get("category"),
                "headline":  a.get("headline"),
                "summary":   a.get("summary"),
                "source":    a.get("source"),
                "url":       a.get("url"),
                "image":     a.get("image"),
                "id":        a.get("id"),
            })
        df = pd.DataFrame(rows).sort_values("datetime", ascending=False)
        return df

    def fetch_market_news(self, category: str = "general") -> pd.DataFrame:
        """
        General market news (no ticker filter).

        category: 'general' | 'forex' | 'crypto' | 'merger'
        """
        logger.info(f"[{self.name}] fetching {category} market news")
        articles = self._get("news", {"category": category})

        if not articles:
            return pd.DataFrame()

        rows = []
        for a in articles:
            rows.append({
                "datetime":  pd.to_datetime(a.get("datetime"), unit="s", utc=True),
                "category":  a.get("category"),
                "headline":  a.get("headline"),
                "summary":   a.get("summary"),
                "source":    a.get("source"),
                "url":       a.get("url"),
                "id":        a.get("id"),
            })
        df = pd.DataFrame(rows).sort_values("datetime", ascending=False)
        return df

    def fetch_sentiment(self, ticker: str) -> pd.DataFrame:
        """
        Finnhub's pre-computed sentiment scores for a ticker.

        Returns hourly sentiment buckets with bullish/bearish percentages.
        """
        logger.info(f"[{self.name}] fetching sentiment for {ticker}")
        try:
            data = self._get("news-sentiment", {"symbol": ticker})
        except requests.HTTPError as e:
            if "403" in str(e):
                logger.warning(
                    f"[{self.name}] news-sentiment is paid-tier; "
                    "skipping (FinBERT will handle this on Day 6)"
                )
                return pd.DataFrame()
            raise

        if not data or "buzz" not in data:
            return pd.DataFrame()

        # Flatten the response
        row = {
            "ticker":              ticker,
            "company":             data.get("companyNewsScore"),
            "sector_avg_news":     data.get("sectorAverageNewsScore"),
            "sector_avg_bull":     data.get("sectorAverageBullishPercent"),
            "buzz_articles_week":  (data.get("buzz") or {}).get("articlesInLastWeek"),
            "buzz_weekly_avg":     (data.get("buzz") or {}).get("weeklyAverage"),
            "buzz_buzz_score":     (data.get("buzz") or {}).get("buzz"),
            "sentiment_bull_pct":  (data.get("sentiment") or {}).get("bullishPercent"),
            "sentiment_bear_pct":  (data.get("sentiment") or {}).get("bearishPercent"),
            "fetched_at":          datetime.now(timezone.utc),
        }
        return pd.DataFrame([row])

    def fetch_earnings_calendar(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Upcoming earnings — used for risk-off filtering (don't trade into earnings).
        """
        start = datetime.now(timezone.utc).date()
        end = start + timedelta(days=days_ahead)

        logger.info(f"[{self.name}] fetching earnings calendar (next {days_ahead}d)")
        data = self._get(
            "calendar/earnings",
            {"from": str(start), "to": str(end)},
        )

        cal = (data or {}).get("earningsCalendar", [])
        if not cal:
            return pd.DataFrame()

        return pd.DataFrame(cal)

    # ---------- DataSource interface ----------
    def fetch(self, ticker: str = "AAPL", days: int = 7, **kwargs) -> pd.DataFrame:
        """Default fetch: company news for a ticker."""
        return self.fetch_company_news(ticker, days)

    def to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news into time-windowed features.
        Returns one row per ticker.
        """
        if raw.empty:
            return pd.DataFrame()

        ticker = raw["ticker"].iloc[0]
        now = datetime.now(timezone.utc)

        h1 = raw[raw["datetime"] >= now - timedelta(hours=1)]
        h4 = raw[raw["datetime"] >= now - timedelta(hours=4)]
        d1 = raw[raw["datetime"] >= now - timedelta(hours=24)]
        d7 = raw[raw["datetime"] >= now - timedelta(days=7)]

        feat = pd.DataFrame([{
            "finnhub_count_1h":          len(h1),
            "finnhub_count_4h":          len(h4),
            "finnhub_count_24h":         len(d1),
            "finnhub_count_7d":          len(d7),
            "finnhub_source_div_24h":    d1["source"].nunique() if not d1.empty else 0,
            "finnhub_velocity_4h_vs_24h": (
                len(h4) / max(len(d1) / 6, 1)  # vs hourly avg over 24h
            ),
            "finnhub_ticker":            ticker,
            "finnhub_fetched_at":        now,
        }])
        return feat

    def health_check(self) -> SourceHealth:
        t0 = datetime.now(timezone.utc)
        try:
            # Quick endpoint that costs almost nothing
            r = requests.get(
                f"{self.BASE_URL}/quote",
                params={"symbol": "AAPL", "token": self.api_key},
                timeout=10,
            )
            healthy = r.status_code == 200 and "c" in r.json()
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
    src = FinnhubSource()

    print("\n--- Health check ---")
    h = src.health_check()
    print(f"  healthy: {h.healthy}   latency: {h.latency_ms:.0f} ms")

    print("\n--- Company news: NVDA, last 2 days ---")
    df = src.fetch_company_news("NVDA", days=2)
    print(f"  rows: {len(df)}")
    if not df.empty:
        print(f"  newest: {df['datetime'].max()}")
        print(f"  oldest: {df['datetime'].min()}")
        print(df[["datetime", "source", "headline"]].head(5).to_string(index=False))

    print("\n--- Market news: general ---")
    mdf = src.fetch_market_news("general")
    print(f"  rows: {len(mdf)}")
    if not mdf.empty:
        print(f"  newest: {mdf['datetime'].max()}")
        print(mdf[["datetime", "source", "headline"]].head(3).to_string(index=False))

    print("\n--- Sentiment: NVDA ---")
    sdf = src.fetch_sentiment("NVDA")
    if not sdf.empty:
        print(sdf.T)

    print("\n--- Earnings calendar: next 7 days ---")
    edf = src.fetch_earnings_calendar(days_ahead=7)
    print(f"  rows: {len(edf)}")
    if not edf.empty:
        print(edf.head(5).to_string(index=False))

    print("\n--- Features (from NVDA news) ---")
    feat = src.to_features(df)
    print(feat.T)
