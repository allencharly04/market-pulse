"""
RSS aggregator for Agent 29.

Polls 15+ financial news RSS feeds — Reuters, CNBC, Yahoo Finance,
MarketWatch, Seeking Alpha, Investing.com, Benzinga, plus crypto-specific
sources like CoinDesk, The Block, Decrypt, Cointelegraph.

Pros over API-based news:
- True real-time (seconds of latency)
- No rate limits, no API keys
- Free forever
- Works even when APIs are down

Cons:
- Headlines + summaries only (no full text)
- Some feeds break or change URLs occasionally (graceful failure handles this)

Strategy: poll every 60s, dedupe by URL+title, keep last 24h in memory.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import pandas as pd
from loguru import logger

from src.connectors.base import DataSource, SourceHealth


# Curated list of high-quality financial RSS feeds
RSS_FEEDS = {
    # === General financial news (stocks + macro) ===
    "yahoo_finance":   "https://finance.yahoo.com/news/rssindex",
    "marketwatch_top": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "marketwatch_mkt": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "cnbc_top":        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "cnbc_business":   "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10001147",
    "cnbc_finance":    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "investing_news":  "https://www.investing.com/rss/news.rss",
    "investing_stock": "https://www.investing.com/rss/news_25.rss",
    "seekingalpha":    "https://seekingalpha.com/market_currents.xml",
    "benzinga_news":   "https://www.benzinga.com/feed",

    # === Crypto-specific ===
    "coindesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph":   "https://cointelegraph.com/rss",
    "decrypt":         "https://decrypt.co/feed",
    "theblock":        "https://www.theblock.co/rss.xml",
    "bitcoinmagazine": "https://bitcoinmagazine.com/feed",

    # === Macro / Fed / regulatory ===
    "fed_press":       "https://www.federalreserve.gov/feeds/press_all.xml",
    "sec_press":       "https://www.sec.gov/news/pressreleases.rss",
}


class RSSSource(DataSource):
    name = "rss"
    category = "news"
    requires_auth = False
    rate_limit_per_min = 9999  # no real limit

    def __init__(self, enabled: bool = True, weight: float = 1.0,
                 feeds: dict[str, str] | None = None):
        super().__init__(enabled=enabled, weight=weight)
        self.feeds = feeds or RSS_FEEDS
        # In-memory dedup cache: {hash: timestamp}
        self._seen: dict[str, datetime] = {}

    @staticmethod
    def _entry_hash(entry: Any) -> str:
        """Stable hash from URL + title for deduplication."""
        key = f"{getattr(entry, 'link', '')}|{getattr(entry, 'title', '')}"
        return hashlib.md5(key.encode()).hexdigest()

    @staticmethod
    def _parse_published(entry: Any) -> datetime | None:
        """Best-effort timestamp extraction from RSS entry."""
        for attr in ("published_parsed", "updated_parsed"):
            t = getattr(entry, attr, None)
            if t:
                try:
                    return datetime(*t[:6], tzinfo=timezone.utc)
                except Exception:
                    continue
        return None

    def _fetch_one(self, feed_name: str, url: str) -> list[dict]:
        """Fetch and parse a single RSS feed. Returns list of article dicts."""
        try:
            d = feedparser.parse(url)
            if d.bozo and not d.entries:
                logger.warning(
                    f"[rss:{feed_name}] feed parse failed: "
                    f"{d.bozo_exception if hasattr(d, 'bozo_exception') else 'unknown'}"
                )
                return []

            articles = []
            for entry in d.entries:
                h = self._entry_hash(entry)
                if h in self._seen:
                    continue
                published = self._parse_published(entry)
                articles.append({
                    "feed":       feed_name,
                    "datetime":   published or datetime.now(timezone.utc),
                    "title":      getattr(entry, "title", ""),
                    "summary":    getattr(entry, "summary", "")[:500],  # truncate
                    "link":       getattr(entry, "link", ""),
                    "author":     getattr(entry, "author", ""),
                    "hash":       h,
                })
                self._seen[h] = datetime.now(timezone.utc)
            return articles
        except Exception as e:
            logger.warning(f"[rss:{feed_name}] error: {type(e).__name__}: {e}")
            return []

    def fetch(self, max_age_hours: int = 24, **kwargs) -> pd.DataFrame:
        """
        Poll all configured RSS feeds, return new articles from past `max_age_hours`.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        all_articles = []

        # Prune dedup cache (keep only last 48h to avoid memory growth)
        prune_cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        self._seen = {h: t for h, t in self._seen.items() if t > prune_cutoff}

        logger.info(f"[{self.name}] polling {len(self.feeds)} feeds")
        for feed_name, url in self.feeds.items():
            articles = self._fetch_one(feed_name, url)
            # Filter by recency
            articles = [a for a in articles if a["datetime"] >= cutoff]
            all_articles.extend(articles)
            logger.debug(f"[rss:{feed_name}] {len(articles)} new articles")

        if not all_articles:
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        df = df.sort_values("datetime", ascending=False).reset_index(drop=True)
        return df

    def to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        now = datetime.now(timezone.utc)
        h1 = raw[raw["datetime"] >= now - timedelta(hours=1)]
        h4 = raw[raw["datetime"] >= now - timedelta(hours=4)]
        d1 = raw[raw["datetime"] >= now - timedelta(hours=24)]

        feat = pd.DataFrame([{
            "rss_count_1h":         len(h1),
            "rss_count_4h":         len(h4),
            "rss_count_24h":        len(d1),
            "rss_feed_div_24h":     d1["feed"].nunique() if not d1.empty else 0,
            "rss_velocity_1h_avg":  len(h1) / 1 if len(h1) else 0,
            "rss_velocity_4h_avg":  len(h4) / 4 if len(h4) else 0,
            "rss_fetched_at":       now,
        }])
        return feat

    def health_check(self) -> SourceHealth:
        """Check that at least 50% of feeds are reachable."""
        t0 = datetime.now(timezone.utc)
        results = []
        for name, url in list(self.feeds.items())[:5]:  # sample 5 feeds
            try:
                d = feedparser.parse(url)
                results.append(bool(d.entries))
            except Exception:
                results.append(False)
        elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
        success_rate = sum(results) / len(results) if results else 0
        healthy = success_rate >= 0.5
        return SourceHealth(
            name=self.name, healthy=healthy,
            last_check=datetime.now(timezone.utc),
            latency_ms=elapsed_ms,
            error=None if healthy else f"only {success_rate:.0%} feeds reachable",
            metadata={"sampled_feeds": len(results), "success_rate": success_rate},
        )


if __name__ == "__main__":
    src = RSSSource()

    print("\n--- Health check (samples 5 feeds) ---")
    h = src.health_check()
    print(f"  healthy: {h.healthy}   latency: {h.latency_ms:.0f} ms")
    print(f"  metadata: {h.metadata}")

    print(f"\n--- Polling all {len(src.feeds)} feeds (last 24h) ---")
    df = src.fetch(max_age_hours=24)
    print(f"  total articles: {len(df)}")

    if not df.empty:
        print(f"  newest: {df['datetime'].max()}")
        print(f"  oldest: {df['datetime'].min()}")
        print(f"\n  Articles per feed:")
        print(df["feed"].value_counts().to_string())

        print(f"\n  Most recent 10 headlines:")
        print(df[["datetime", "feed", "title"]].head(10).to_string(index=False))

    print("\n--- Features ---")
    feat = src.to_features(df)
    print(feat.T)
