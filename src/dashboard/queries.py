"""
SQL query layer for the Streamlit dashboard.

Centralizes all DB access. Each function returns a clean DataFrame
ready to render. Cached by Streamlit to avoid hammering SQLite.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/agent29.db")


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


# ============================================================
# System health
# ============================================================
def latest_cycle() -> dict | None:
    """Most recent orchestrator cycle (system health summary)."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM cycles ORDER BY started_at DESC LIMIT 1", conn
        )
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def cycles_recent(n: int = 50) -> pd.DataFrame:
    """Last N cycles for the trend chart."""
    with _connect() as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM cycles ORDER BY started_at DESC LIMIT {n}", conn
        )
    if not df.empty:
        df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    return df.sort_values("started_at")


# ============================================================
# Headlines + sentiment
# ============================================================
def total_headlines() -> int:
    with _connect() as conn:
        r = conn.execute("SELECT COUNT(*) FROM news").fetchone()
    return r[0] if r else 0


def headlines_by_source() -> pd.DataFrame:
    with _connect() as conn:
        return pd.read_sql_query(
            "SELECT source, COUNT(*) as n FROM news GROUP BY source ORDER BY n DESC",
            conn,
        )


def sentiment_distribution(hours: int = 24) -> pd.DataFrame:
    """Distribution of finbert_label for headlines in the last N hours."""
    with _connect() as conn:
        return pd.read_sql_query(
            f"""
            SELECT finbert_label, COUNT(*) as n
            FROM news
            WHERE published_at >= datetime('now', '-{hours} hours')
              AND finbert_label IS NOT NULL
            GROUP BY finbert_label
            """,
            conn,
        )


def recent_headlines(
    limit: int = 50,
    ticker: str | None = None,
    source: str | None = None,
    min_compound: float | None = None,
    max_compound: float | None = None,
) -> pd.DataFrame:
    """Most recent headlines with optional filters."""
    where = ["1=1"]
    params: list = []

    if ticker:
        where.append("primary_ticker = ?")
        params.append(ticker)
    if source:
        where.append("source = ?")
        params.append(source)
    if min_compound is not None:
        where.append("finbert_compound >= ?")
        params.append(min_compound)
    if max_compound is not None:
        where.append("finbert_compound <= ?")
        params.append(max_compound)

    sql = f"""
        SELECT published_at, source, origin, primary_ticker, tickers,
               finbert_label, finbert_compound, title, url
        FROM news
        WHERE {' AND '.join(where)}
        ORDER BY published_at DESC
        LIMIT {limit}
    """
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    return df


# ============================================================
# Per-ticker aggregations
# ============================================================
def ticker_leaderboard(min_headlines: int = 2, hours: int = 168) -> pd.DataFrame:
    """
    Per-ticker sentiment summary over the last `hours`.

    Default 168h = 7 days (matches Finnhub's lookback).
    """
    with _connect() as conn:
        return pd.read_sql_query(
            f"""
            SELECT primary_ticker as ticker,
                   COUNT(*)        as headlines,
                   AVG(finbert_compound) as avg_sentiment,
                   SUM(CASE WHEN finbert_label='positive' THEN 1 ELSE 0 END) as n_pos,
                   SUM(CASE WHEN finbert_label='negative' THEN 1 ELSE 0 END) as n_neg,
                   SUM(CASE WHEN finbert_label='neutral'  THEN 1 ELSE 0 END) as n_neu
            FROM news
            WHERE primary_ticker IS NOT NULL
              AND published_at >= datetime('now', '-{hours} hours')
            GROUP BY primary_ticker
            HAVING headlines >= {min_headlines}
            ORDER BY headlines DESC
            """,
            conn,
        )


def all_tickers_seen() -> list[str]:
    """List of all tickers that have appeared in news (for filter dropdown)."""
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT primary_ticker FROM news "
            "WHERE primary_ticker IS NOT NULL ORDER BY primary_ticker",
            conn,
        )
    return df["primary_ticker"].tolist()


def all_sources_seen() -> list[str]:
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT DISTINCT source FROM news ORDER BY source", conn
        )
    return df["source"].tolist()


# ============================================================
# Macro regime
# ============================================================
def latest_macro_features() -> pd.DataFrame:
    """Macro features from the most recent cycle."""
    with _connect() as conn:
        df = pd.read_sql_query(
            """
            SELECT source, feature_name, feature_val, feature_text
            FROM features
            WHERE cycle_id = (SELECT cycle_id FROM cycles ORDER BY started_at DESC LIMIT 1)
            ORDER BY source, feature_name
            """,
            conn,
        )
    return df


def macro_value(feature_name: str) -> float | str | None:
    """Get a single feature value from the most recent cycle."""
    with _connect() as conn:
        r = conn.execute(
            """
            SELECT feature_val, feature_text FROM features
            WHERE feature_name = ?
              AND cycle_id = (SELECT cycle_id FROM cycles ORDER BY started_at DESC LIMIT 1)
            LIMIT 1
            """,
            (feature_name,),
        ).fetchone()
    if r is None:
        return None
    val, text = r
    return val if val is not None else text


# ============================================================
# Headline volume timeline
# ============================================================
def headline_volume_by_hour(hours: int = 48) -> pd.DataFrame:
    """Headlines per hour for the timeline chart."""
    with _connect() as conn:
        df = pd.read_sql_query(
            f"""
            SELECT substr(published_at, 1, 13) || ':00:00' as hour,
                   COUNT(*) as n,
                   AVG(finbert_compound) as avg_sentiment,
                   SUM(CASE WHEN finbert_label='positive' THEN 1 ELSE 0 END) as n_pos,
                   SUM(CASE WHEN finbert_label='negative' THEN 1 ELSE 0 END) as n_neg
            FROM news
            WHERE published_at >= datetime('now', '-{hours} hours')
            GROUP BY hour
            ORDER BY hour
            """,
            conn,
        )
    if not df.empty:
        df["hour"] = pd.to_datetime(df["hour"], utc=True, errors="coerce")
    return df
