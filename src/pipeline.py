"""
Agent 29 — Pipeline Orchestrator

Coordinates all data sources and sentiment scorers into a single fetch cycle.

A "cycle" is one full pass:
1. Fetch from all enabled sources (parallel where possible)
2. Save raw data to data/raw/
3. Score news headlines with FinBERT (and VADER baseline)
4. Persist features + scored news to SQLite (data/agent29.db)
5. Return a summary of what happened

Run modes:
- One-shot:     `python -m src.pipeline --once`
- Scheduled:    `python -m src.pipeline --interval 300`  (every 5 min)
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

from src.connectors.alpaca_client import AlpacaConnector
from src.connectors.binance_client import BinanceConnector
from src.connectors.crypto_feargreed_source import CryptoFearGreedSource
from src.connectors.finnhub_source import FinnhubSource
from src.connectors.fred_source import FREDSource
from src.connectors.newsapi_source import NewsAPISource
from src.connectors.rss_source import RSSSource
from src.connectors.telegram_source import TelegramSource
from src.sentiment.finbert import FinBERTScorer
from src.sentiment.vader import VaderScorer

load_dotenv()


CONFIG_PATH = Path("config/sources.yaml")
DB_PATH = Path("data/agent29.db")
RAW_DIR = Path("data/raw")


# ============================================================
# DB schema setup
# ============================================================
def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create tables if they don't exist. Returns an open connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    # News table — every scored headline
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            source        TEXT NOT NULL,
            published_at  TEXT NOT NULL,
            ticker        TEXT,
            title         TEXT NOT NULL,
            url           TEXT,
            origin        TEXT,
            -- sentiment scores
            finbert_pos       REAL,
            finbert_neg       REAL,
            finbert_neu       REAL,
            finbert_label     TEXT,
            finbert_compound  REAL,
            vader_compound    REAL,
            vader_label       TEXT,
            -- bookkeeping
            content_hash      TEXT UNIQUE,
            scored_at         TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker    ON news(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_source    ON news(source)")

    # Features table — one row per source per cycle
    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            cycle_id     TEXT NOT NULL,
            source       TEXT NOT NULL,
            feature_name TEXT NOT NULL,
            feature_val  REAL,
            feature_text TEXT,
            recorded_at  TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_cycle ON features(cycle_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_name  ON features(feature_name)")

    # Cycles table — one row per orchestrator run
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cycles (
            cycle_id        TEXT PRIMARY KEY,
            started_at      TEXT NOT NULL,
            finished_at     TEXT,
            duration_sec    REAL,
            sources_ok      INTEGER,
            sources_failed  INTEGER,
            news_scored     INTEGER,
            notes           TEXT
        )
    """)

    conn.commit()
    return conn


# ============================================================
# Source registry
# ============================================================
def load_source_config(path: Path = CONFIG_PATH) -> dict:
    """Load sources.yaml — tells us which sources are enabled."""
    if not path.exists():
        logger.warning(f"{path} missing — all sources will run with defaults")
        return {"sources": {}, "fetch_settings": {"max_concurrent": 5}}
    with open(path) as f:
        return yaml.safe_load(f)


def build_news_sources(config: dict) -> list:
    """Instantiate enabled news/sentiment-relevant sources."""
    enabled = config.get("sources", {})
    sources = []

    if enabled.get("finnhub", {}).get("enabled", True):
        sources.append(FinnhubSource())
    if enabled.get("rss", {}).get("enabled", True):
        sources.append(RSSSource())
    if enabled.get("telegram", {}).get("enabled", True):
        sources.append(TelegramSource())
    if enabled.get("newsapi", {}).get("enabled", True):
        sources.append(NewsAPISource())

    return sources


def build_macro_sources(config: dict) -> list:
    """Instantiate enabled macro/regime/crypto-context sources."""
    enabled = config.get("sources", {})
    sources = []

    if enabled.get("fred", {}).get("enabled", True):
        sources.append(FREDSource())
    if enabled.get("crypto_feargreed", {}).get("enabled", True):
        sources.append(CryptoFearGreedSource())

    return sources


# ============================================================
# Fetching
# ============================================================
def fetch_all_news(sources: list, max_workers: int = 5) -> dict[str, pd.DataFrame]:
    """
    Fetch from all news sources in parallel using thread pool.
    Each source's fetch_safe() returns empty DataFrame on failure.
    """
    results: dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_source = {
            ex.submit(s.fetch_safe): s for s in sources
        }
        for fut in as_completed(future_to_source):
            src = future_to_source[fut]
            try:
                df = fut.result(timeout=120)
                results[src.name] = df
                logger.info(f"[orchestrator] {src.name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"[orchestrator] {src.name} error: {e}")
                results[src.name] = pd.DataFrame()

    return results


def fetch_macro(sources: list) -> dict[str, pd.DataFrame]:
    """Fetch macro/context sources (sequential, they're fast)."""
    results: dict[str, pd.DataFrame] = {}
    for src in sources:
        df = src.fetch_safe()
        results[src.name] = df
        logger.info(f"[orchestrator] {src.name}: {len(df)} rows")
    return results


# ============================================================
# News normalization
# ============================================================
def normalize_news(news_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine headlines from all news sources into a unified schema:
    columns: source, published_at, ticker, title, url, origin
    """
    rows = []

    # Finnhub: 'datetime', 'ticker', 'headline', 'url', 'source'
    df = news_results.get("finnhub")
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            rows.append({
                "source":       "finnhub",
                "published_at": r.get("datetime"),
                "ticker":       r.get("ticker"),
                "title":        r.get("headline", ""),
                "url":          r.get("url"),
                "origin":       r.get("source"),
            })

    # RSS: 'datetime', 'feed', 'title', 'link'
    df = news_results.get("rss")
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            rows.append({
                "source":       "rss",
                "published_at": r.get("datetime"),
                "ticker":       None,
                "title":        r.get("title", ""),
                "url":          r.get("link"),
                "origin":       r.get("feed"),
            })

    # Telegram: 'datetime', 'channel', 'text'
    df = news_results.get("telegram")
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            rows.append({
                "source":       "telegram",
                "published_at": r.get("datetime"),
                "ticker":       None,
                "title":        (r.get("text") or "")[:300],
                "url":          None,
                "origin":       r.get("channel"),
            })

    # NewsAPI: 'published_at', 'source', 'title', 'url'
    df = news_results.get("newsapi")
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            rows.append({
                "source":       "newsapi",
                "published_at": r.get("published_at"),
                "ticker":       None,
                "title":        r.get("title", "") or "",
                "url":          r.get("url"),
                "origin":       r.get("source"),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["title"].str.len() > 0]
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df


def add_content_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Stable hash for dedup against DB."""
    import hashlib
    if df.empty:
        return df
    def _h(row):
        key = f"{row['source']}|{row.get('url','')}|{row['title'][:200]}"
        return hashlib.md5(key.encode()).hexdigest()
    df["content_hash"] = df.apply(_h, axis=1)
    return df


# ============================================================
# DB persistence
# ============================================================
def save_news(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert scored news, skipping rows with existing content_hash."""
    if df.empty:
        return 0

    # Get existing hashes to skip
    existing = pd.read_sql_query(
        "SELECT content_hash FROM news WHERE content_hash IN ({})".format(
            ",".join("?" * len(df))
        ),
        conn,
        params=df["content_hash"].tolist(),
    )
    existing_set = set(existing["content_hash"]) if not existing.empty else set()

    new_rows = df[~df["content_hash"].isin(existing_set)].copy()
    if new_rows.empty:
        return 0

    new_rows["scored_at"] = datetime.now(timezone.utc).isoformat()
    new_rows["published_at"] = new_rows["published_at"].astype(str)

    cols = [
        "source", "published_at", "ticker", "title", "url", "origin",
        "finbert_pos", "finbert_neg", "finbert_neu", "finbert_label", "finbert_compound",
        "vader_compound", "vader_label",
        "content_hash", "scored_at",
    ]
    # ensure all cols exist
    for c in cols:
        if c not in new_rows.columns:
            new_rows[c] = None

    new_rows[cols].to_sql("news", conn, if_exists="append", index=False)
    return len(new_rows)


def save_macro_features(
    conn: sqlite3.Connection,
    cycle_id: str,
    macro_results: dict[str, pd.DataFrame],
    macro_sources: list,
) -> int:
    """Run to_features() on each macro source's data and save to features table."""
    rows_saved = 0
    now = datetime.now(timezone.utc).isoformat()

    for src in macro_sources:
        raw = macro_results.get(src.name, pd.DataFrame())
        if raw.empty:
            continue
        try:
            feat_df = src.to_features(raw)
        except Exception as e:
            logger.error(f"[features] {src.name} to_features error: {e}")
            continue
        if feat_df.empty:
            continue

        # feat_df is a single-row DataFrame; expand to long format
        row = feat_df.iloc[0]
        for col, val in row.items():
            import numpy as np
            try:
                num_val = float(val) if val is not None and not (isinstance(val, str)) else None
                is_num = num_val is not None and not pd.isna(num_val)
            except (ValueError, TypeError):
                is_num = False
            conn.execute(
                "INSERT INTO features (cycle_id, source, feature_name, feature_val, feature_text, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (cycle_id, src.name, str(col),
                 float(val) if is_num else None,
                 str(val) if not is_num else None,
                 now),
            )
            rows_saved += 1

    return rows_saved


# ============================================================
# Main cycle
# ============================================================
def run_cycle(
    finbert: FinBERTScorer | None = None,
    vader: VaderScorer | None = None,
) -> dict:
    """Run one full orchestrator cycle. Returns a summary dict."""
    cycle_start = time.time()
    cycle_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== CYCLE {cycle_id} START ===")

    config = load_source_config()
    max_workers = config.get("fetch_settings", {}).get("max_concurrent", 5)

    # 1. Build sources
    news_sources = build_news_sources(config)
    macro_sources = build_macro_sources(config)
    logger.info(
        f"[orchestrator] {len(news_sources)} news + {len(macro_sources)} macro sources"
    )

    # 2. Fetch
    news_results = fetch_all_news(news_sources, max_workers=max_workers)
    macro_results = fetch_macro(macro_sources)

    # 3. Normalize news headlines into one DataFrame
    headlines = normalize_news(news_results)
    logger.info(f"[orchestrator] {len(headlines)} headlines normalized")

    # 4. Score sentiment
    if not headlines.empty and finbert is not None:
        headlines = finbert.score_dataframe(headlines, text_col="title")
        logger.success(f"[orchestrator] FinBERT scored {len(headlines)} headlines")
    if not headlines.empty and vader is not None:
        headlines = vader.score_dataframe(headlines, text_col="title")

    # 5. Add content hashes for dedup
    headlines = add_content_hash(headlines)

    # 6. Persist
    conn = init_db()
    news_inserted = save_news(conn, headlines) if not headlines.empty else 0
    macro_features_inserted = save_macro_features(
        conn, cycle_id, macro_results, macro_sources
    )

    # 7. Cycle row
    duration = time.time() - cycle_start
    sources_ok = sum(1 for v in {**news_results, **macro_results}.values() if not v.empty)
    sources_failed = sum(1 for v in {**news_results, **macro_results}.values() if v.empty)
    conn.execute(
        "INSERT INTO cycles (cycle_id, started_at, finished_at, duration_sec, "
        "sources_ok, sources_failed, news_scored, notes) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (cycle_id,
         datetime.fromtimestamp(cycle_start, timezone.utc).isoformat(),
         datetime.now(timezone.utc).isoformat(),
         duration, sources_ok, sources_failed, news_inserted, ""),
    )
    conn.commit()
    conn.close()

    summary = {
        "cycle_id":           cycle_id,
        "duration_sec":       round(duration, 2),
        "sources_ok":         sources_ok,
        "sources_failed":     sources_failed,
        "headlines_total":    len(headlines),
        "news_inserted_new":  news_inserted,
        "macro_features":     macro_features_inserted,
    }
    logger.info(f"=== CYCLE {cycle_id} DONE: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between cycles when not --once")
    parser.add_argument("--no-finbert", action="store_true",
                        help="Skip FinBERT (use VADER only — faster, less accurate)")
    args = parser.parse_args()

    # Initialize scorers once (model loading is expensive)
    finbert = None if args.no_finbert else FinBERTScorer()
    vader = VaderScorer()

    if args.once:
        summary = run_cycle(finbert, vader)
        print("\n", summary)
        return

    while True:
        try:
            run_cycle(finbert, vader)
        except KeyboardInterrupt:
            logger.info("Interrupted, exiting")
            break
        except Exception as e:
            logger.error(f"Cycle failed: {e}")
        logger.info(f"Sleeping {args.interval}s...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
