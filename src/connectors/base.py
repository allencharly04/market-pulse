"""
Base class for all Agent 29 data sources.

Design principles:
- Every source implements the same interface
- Sources are config-driven (enable/disable in sources.yaml)
- Failures in one source don't break the pipeline
- All fetched data goes through a standard schema
- Importance tracking is automatic
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


@dataclass
class SourceHealth:
    """Reports a source's health status to the orchestrator."""
    name: str
    healthy: bool
    last_check: datetime
    latency_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """
    Abstract base for any data source plugged into Agent 29.

    Concrete implementations only need to define:
        name           : str   - unique identifier
        category       : str   - 'news' | 'social' | 'macro' | 'crypto' | 'filings' | 'price'
        requires_auth  : bool  - whether it needs API keys
        fetch()        : pulls raw data
        to_features()  : converts raw data to feature DataFrame
        health_check() : verifies the source is reachable

    Each source has an `enabled` flag (set by config).
    """

    name: str = "unnamed"
    category: str = "unknown"
    requires_auth: bool = False
    rate_limit_per_min: int = 60  # default conservative limit

    def __init__(self, enabled: bool = True, weight: float = 1.0):
        self.enabled = enabled
        self.weight = weight  # used by source scorer in later phases
        self._last_fetch: datetime | None = None
        self._last_error: str | None = None

    # ---------- Required overrides ----------
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """Pull raw data from the source. Returns a DataFrame."""
        ...

    @abstractmethod
    def to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into feature columns suitable for ML."""
        ...

    @abstractmethod
    def health_check(self) -> SourceHealth:
        """Check if the source is reachable and credentials are valid."""
        ...

    # ---------- Common implementations ----------
    def fetch_safe(self, **kwargs) -> pd.DataFrame:
        """
        Fetch with error handling. Returns empty DataFrame on failure
        instead of raising — this is what the orchestrator calls.
        """
        if not self.enabled:
            logger.debug(f"[{self.name}] disabled, skipping fetch")
            return pd.DataFrame()

        try:
            t0 = datetime.now(timezone.utc)
            df = self.fetch(**kwargs)
            self._last_fetch = datetime.now(timezone.utc)
            self._last_error = None
            elapsed = (self._last_fetch - t0).total_seconds()
            logger.success(
                f"[{self.name}] fetched {len(df)} rows in {elapsed:.2f}s"
            )
            return df
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            logger.error(f"[{self.name}] fetch failed: {self._last_error}")
            return pd.DataFrame()

    def save_raw(self, df: pd.DataFrame, suffix: str = "") -> Path:
        """Save raw data to data/raw/<source>_<suffix>.parquet."""
        if df.empty:
            return Path()
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix_part = f"_{suffix}" if suffix else ""
        out_path = out_dir / f"{self.name}{suffix_part}_{ts}.parquet"
        df.to_parquet(out_path, engine="fastparquet")
        return out_path

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"<{self.__class__.__name__}({self.name}, {status})>"
