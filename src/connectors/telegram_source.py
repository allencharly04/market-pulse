"""
Telegram channel scraper for Agent 29.

Reads public Telegram channels for breaking financial/crypto news.
Telegram channels are often the FASTEST source — sometimes seconds before
mainstream news, especially for crypto and geopolitical events.

Setup (done once):
1. Get api_id + api_hash from https://my.telegram.org
2. Add to .env: TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE
3. First run prompts for SMS code via Telegram app

Subsequent runs use a session file (telegram_session.session) so no auth needed.

Channels are configured below in DEFAULT_CHANNELS — curated for trading signals.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import nest_asyncio
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.connectors.base import DataSource, SourceHealth

load_dotenv()

# Allow nested event loops (Jupyter-friendly + safe for Streamlit later)
nest_asyncio.apply()


# Curated public Telegram channels for financial signals.
# These are PUBLIC channels — anyone can read them, no joining required for read access.
# Categorized for clarity, not for filtering yet.
DEFAULT_CHANNELS = [
    # Verified working - all confirmed via Telegram app
    "WatcherGuru",        # 800K+ subs, breaking macro/crypto news
    "bloomberg",          # 160K+ subs, official Bloomberg
    "bloombergcrypto",    # 25K+ subs, Bloomberg's crypto coverage
    "cointelegraph",      # 375K+ subs, top crypto news outlet
]

class TelegramSource(DataSource):
    name = "telegram"
    category = "news"
    requires_auth = True
    rate_limit_per_min = 100  # generous, real limit is much higher

    def __init__(self, enabled: bool = True, weight: float = 1.1,
                 channels: list[str] | None = None):
        super().__init__(enabled=enabled, weight=weight)
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.phone = os.getenv("TELEGRAM_PHONE")

        if not all([self.api_id, self.api_hash, self.phone]):
            raise RuntimeError(
                "TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_PHONE "
                "must all be set in .env"
            )

        self.api_id = int(self.api_id)
        self.channels = channels or DEFAULT_CHANNELS
        # Session file in data/ so it's gitignored
        self.session_path = Path("data") / "telegram_session"

    async def _fetch_channel_async(self, client, channel: str,
                                    max_age_hours: int) -> list[dict]:
        """Fetch recent messages from one channel."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        messages = []
        try:
            async for msg in client.iter_messages(channel, limit=100):
                if msg.date < cutoff:
                    break
                if msg.text:  # skip media-only messages
                    messages.append({
                        "channel":  channel,
                        "datetime": msg.date,
                        "text":     msg.text[:500],  # truncate long posts
                        "msg_id":   msg.id,
                        "views":    getattr(msg, "views", 0) or 0,
                        "forwards": getattr(msg, "forwards", 0) or 0,
                    })
        except Exception as e:
            logger.warning(f"[telegram:{channel}] error: {type(e).__name__}: {e}")
        return messages

    async def _fetch_async(self, max_age_hours: int) -> pd.DataFrame:
        from telethon import TelegramClient

        async with TelegramClient(
            str(self.session_path), self.api_id, self.api_hash
        ) as client:
            # Ensure we're authorized (first run will prompt)
            if not await client.is_user_authorized():
                await client.send_code_request(self.phone)
                code = input("Enter the Telegram code you received: ")
                await client.sign_in(self.phone, code)

            logger.info(f"[{self.name}] polling {len(self.channels)} channels")
            tasks = [
                self._fetch_channel_async(client, ch, max_age_hours)
                for ch in self.channels
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        all_messages = []
        for ch, msgs in zip(self.channels, results):
            logger.debug(f"[telegram:{ch}] {len(msgs)} messages")
            all_messages.extend(msgs)

        if not all_messages:
            return pd.DataFrame()

        df = pd.DataFrame(all_messages)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime", ascending=False).reset_index(drop=True)
        return df

    def fetch(self, max_age_hours: int = 24, **kwargs) -> pd.DataFrame:
        """Synchronous wrapper around async fetch."""
        return asyncio.run(self._fetch_async(max_age_hours))

    def to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        now = datetime.now(timezone.utc)
        h1 = raw[raw["datetime"] >= now - timedelta(hours=1)]
        h4 = raw[raw["datetime"] >= now - timedelta(hours=4)]
        d1 = raw[raw["datetime"] >= now - timedelta(hours=24)]

        feat = pd.DataFrame([{
            "tg_count_1h":          len(h1),
            "tg_count_4h":          len(h4),
            "tg_count_24h":         len(d1),
            "tg_channel_div_24h":   d1["channel"].nunique() if not d1.empty else 0,
            "tg_avg_views_1h":      h1["views"].mean() if not h1.empty else 0,
            "tg_max_views_1h":      h1["views"].max() if not h1.empty else 0,
            "tg_total_forwards_4h": h4["forwards"].sum() if not h4.empty else 0,
            "tg_velocity_1h":       len(h1) / 1,
            "tg_velocity_4h":       len(h4) / 4,
            "tg_fetched_at":        now,
        }])
        return feat

    def health_check(self) -> SourceHealth:
        """Try to connect; first run will require interactive auth."""
        from telethon.sync import TelegramClient
        t0 = datetime.now(timezone.utc)
        try:
            with TelegramClient(
                str(self.session_path), self.api_id, self.api_hash
            ) as client:
                authorized = client.is_user_authorized()
            elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
            return SourceHealth(
                name=self.name, healthy=authorized,
                last_check=datetime.now(timezone.utc),
                latency_ms=elapsed_ms,
                error=None if authorized else "not authorized — run fetch() once interactively",
            )
        except Exception as e:
            return SourceHealth(
                name=self.name, healthy=False,
                last_check=datetime.now(timezone.utc),
                latency_ms=-1, error=f"{type(e).__name__}: {e}",
            )


if __name__ == "__main__":
    src = TelegramSource()

    print("\n--- Health check ---")
    h = src.health_check()
    print(f"  healthy: {h.healthy}   latency: {h.latency_ms:.0f} ms")
    if h.error:
        print(f"  error: {h.error}")

    print(f"\n--- Polling {len(src.channels)} channels (last 24h) ---")
    print("  (FIRST RUN: Telegram will send a code to your app — enter it when prompted)")
    df = src.fetch(max_age_hours=24)
    print(f"\n  total messages: {len(df)}")

    if not df.empty:
        print(f"  newest: {df['datetime'].max()}")
        print(f"  oldest: {df['datetime'].min()}")

        print("\n  Messages per channel:")
        print(df["channel"].value_counts().to_string())

        print("\n  Most recent 10 messages:")
        for _, row in df.head(10).iterrows():
            txt = row["text"][:120].replace("\n", " ")
            print(f"  [{row['datetime']}] @{row['channel']}: {txt}")

    print("\n--- Features ---")
    feat = src.to_features(df)
    print(feat.T)
