"""
FRED historical time series fetcher for Market Pulse v2.

The existing fred_source.py fetches latest values for the live dashboard.
This module fetches FULL DAILY HISTORY for ML training — the data layer
fix that solves the "macro features broadcast as constants" bug.

Stores to a dedicated `macro_history` SQLite table keyed on (date, series_id).
Each ML training run can join its (ticker, date) rows against this table
and get the actual macro values that existed at that historical moment.

Usage:
    from src.connectors.fred_history import fetch_and_store_macro_history
    fetch_and_store_macro_history(days=1095)   # 3 years
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from loguru import logger

load_dotenv()


DB_PATH = Path("data/agent29.db")

# Same series as fred_source.py — keep them in sync.
# (series_id, friendly_name)
SERIES = [
    ("VIXCLS",   "vix"),
    ("DGS10",    "dgs10"),
    ("DGS2",     "dgs2"),
    ("T10Y2Y",   "yield_curve_10y2y"),
    ("T10Y3M",   "yield_curve_10y3m"),
    ("DTWEXBGS", "dxy_broad"),
    ("DFF",      "fed_funds"),
    ("UNRATE",   "unemployment"),
    ("CPIAUCSL", "cpi"),
]


# ============================================================
# DB schema
# ============================================================
def init_macro_history_table(conn: sqlite3.Connection) -> None:
    """Create the macro_history table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro_history (
            date         TEXT NOT NULL,
            series_id    TEXT NOT NULL,
            friendly     TEXT NOT NULL,
            value        REAL,
            inserted_at  TEXT NOT NULL,
            PRIMARY KEY (date, series_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_history_date ON macro_history(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_history_series ON macro_history(series_id)")
    conn.commit()


# ============================================================
# Fetch with retry (FRED occasionally 500s)
# ============================================================
def fetch_series_with_retry(
    fred: Fred,
    series_id: str,
    start: datetime,
    end: datetime,
    max_retries: int = 3,
    backoff_sec: float = 1.0,
) -> pd.Series:
    """Fetch one FRED series with simple exponential backoff on 5xx errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            s = fred.get_series(series_id,
                               observation_start=start,
                               observation_end=end)
            return s.dropna()
        except Exception as e:
            last_exc = e
            err_msg = str(e).lower()
            # Retry only on transient errors
            if "internal server error" in err_msg or "503" in err_msg or "504" in err_msg:
                wait = backoff_sec * (2 ** attempt)
                logger.warning(
                    f"[fred_history:{series_id}] attempt {attempt+1}/{max_retries} "
                    f"failed ({type(e).__name__}). Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                continue
            # Non-transient: re-raise immediately
            raise
    # Out of retries
    logger.error(f"[fred_history:{series_id}] failed after {max_retries} retries: {last_exc}")
    return pd.Series(dtype=float)


# ============================================================
# Main fetcher
# ============================================================
def fetch_and_store_macro_history(
    days: int = 1095,
    db_path: Path = DB_PATH,
) -> dict[str, int]:
    """
    Fetch full daily history for all FRED series and upsert into macro_history.

    Returns: {friendly_name: rows_inserted_or_updated}
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in .env")

    fred = Fred(api_key=api_key)
    end = datetime.now()
    start = end - timedelta(days=days)

    logger.info(
        f"[fred_history] fetching {len(SERIES)} series "
        f"from {start.date()} to {end.date()}"
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    init_macro_history_table(conn)

    now_iso = datetime.now(timezone.utc).isoformat()
    summary: dict[str, int] = {}

    for series_id, friendly in SERIES:
        s = fetch_series_with_retry(fred, series_id, start, end)
        if s.empty:
            summary[friendly] = 0
            continue

        # Upsert each (date, series_id) row
        rows = [
            (str(date.date()), series_id, friendly, float(value), now_iso)
            for date, value in s.items()
            if pd.notna(value)
        ]

        # SQLite ON CONFLICT REPLACE upsert
        conn.executemany(
            """
            INSERT INTO macro_history (date, series_id, friendly, value, inserted_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date, series_id) DO UPDATE SET
                value = excluded.value,
                inserted_at = excluded.inserted_at
            """,
            rows,
        )
        conn.commit()
        summary[friendly] = len(rows)
        logger.success(
            f"[fred_history:{friendly}] upserted {len(rows)} rows "
            f"({s.index.min().date()} → {s.index.max().date()}, latest={s.iloc[-1]:.4f})"
        )

    conn.close()
    return summary


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    print("\n--- Fetching 3 years of FRED history ---\n")
    summary = fetch_and_store_macro_history(days=1095)

    print("\n--- Summary ---")
    total = sum(summary.values())
    for friendly, n in summary.items():
        marker = "✓" if n > 0 else "✗"
        print(f"  {marker} {friendly:25s} {n:5d} rows")
    print(f"\n  TOTAL: {total} rows across {sum(1 for n in summary.values() if n > 0)}/{len(summary)} series")

    # Verify what landed
    import pandas as pd
    conn = sqlite3.connect(str(DB_PATH))

    print("\n--- Verification: rows per series in DB ---")
    df = pd.read_sql_query(
        "SELECT friendly, COUNT(*) as n, MIN(date) as oldest, MAX(date) as newest "
        "FROM macro_history GROUP BY friendly ORDER BY n DESC",
        conn,
    )
    print(df.to_string(index=False))

    print("\n--- Sample: 5 most recent rows for VIX ---")
    df = pd.read_sql_query(
        "SELECT date, value FROM macro_history "
        "WHERE friendly = 'vix' ORDER BY date DESC LIMIT 5",
        conn,
    )
    print(df.to_string(index=False))

    conn.close()
