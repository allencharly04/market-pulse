"""
Alpaca connector for Agent 29.

Handles:
- Authenticated client setup (paper/live based on MODE env var)
- Historical bar fetching with pagination
- Latest quote/trade lookups
- Saving raw bars to parquet (fastparquet engine)

Usage:
    from src.connectors.alpaca import AlpacaConnector

    conn = AlpacaConnector()
    df = conn.get_bars("SPY", days=30)
    print(df.head())
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

# Load .env once at import time
load_dotenv()


class AlpacaConnector:
    """Thin wrapper around alpaca-py for Agent 29."""

    def __init__(self, paper: bool | None = None):
        # If paper not explicitly set, infer from MODE env
        if paper is None:
            paper = os.getenv("MODE", "paper").lower() == "paper"
        self.paper = paper

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env"
            )

        # Trading client (orders, positions, account)
        self.trading = TradingClient(api_key, secret_key, paper=paper)

        # Data client (historical bars, quotes) - same regardless of paper/live
        self.data = StockHistoricalDataClient(api_key, secret_key)

        mode = "paper" if paper else "LIVE"
        logger.info(f"AlpacaConnector initialised in {mode} mode")

    # ---------- Account ----------
    def get_account(self) -> dict:
        """Returns key account info (cash, equity, buying_power)."""
        acct = self.trading.get_account()
        return {
            "cash": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "currency": acct.currency,
            "status": acct.status.value if hasattr(acct.status, "value") else str(acct.status),
        }

    # ---------- Historical bars ----------
    def get_bars(
        self,
        symbol: str | list[str],
        days: int = 30,
        timeframe: TimeFrame = TimeFrame.Day,
    ) -> pd.DataFrame:
        """
        Fetch historical bars for one or more symbols.

        Returns a DataFrame indexed by (symbol, timestamp) with OHLCV columns.
        """
        symbols = [symbol] if isinstance(symbol, str) else list(symbol)

        # Alpaca free tier: data has 15-minute delay and goes back to 2016+
        # Always pull a buffer to handle weekends/holidays
        end = datetime.now(timezone.utc) - timedelta(minutes=20)
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        logger.info(
            f"Fetching {timeframe} bars for {symbols} from {start.date()} to {end.date()}"
        )
        bars = self.data.get_stock_bars(request)
        df = bars.df  # multi-index DataFrame: (symbol, timestamp)

        if df.empty:
            logger.warning(f"No bars returned for {symbols}")
            return df

        logger.success(f"Fetched {len(df)} bars across {len(symbols)} symbols")
        return df

    # ---------- Saving ----------
    def save_bars(self, df: pd.DataFrame, name: str) -> Path:
        """Save bars DataFrame to data/raw/<name>.parquet using fastparquet."""
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.parquet"

        # fastparquet, NOT pyarrow (Agent 20 lesson on tz handling)
        df.to_parquet(out_path, engine="fastparquet")
        logger.success(f"Saved {len(df)} rows to {out_path}")
        return out_path

def fetch_long_history(symbols: list[str] = None, days: int = 365) -> dict:
    """
    Fetch and save a long history of daily bars for multiple tickers.
    Used for feature engineering warmup (some indicators need 200+ bars).
    """
    symbols = symbols or ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "GOOGL", "TSLA", "META", "AMZN", "JPM"]
    conn = AlpacaConnector()
    saved = {}
    for sym in symbols:
        df = conn.get_bars(sym, days=days)
        if df.empty:
            logger.warning(f"No data for {sym}")
            continue
        path = conn.save_bars(df, f"{sym.lower()}_{days}d_daily")
        saved[sym] = path
    return saved

if __name__ == "__main__":
    # Smoke test when run directly
    conn = AlpacaConnector()

    print("\n--- Account ---")
    acct = conn.get_account()
    for k, v in acct.items():
        print(f"  {k:18s} {v}")

    print("\n--- SPY 30 daily bars ---")
    df = conn.get_bars("SPY", days=30)
    print(df.tail())

    print("\n--- Saving to parquet ---")
    conn.save_bars(df, "spy_30d_daily")
