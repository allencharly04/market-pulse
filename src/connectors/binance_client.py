"""
Binance connector for Agent 29 (testnet by default).

Handles:
- Authenticated client setup (testnet/live based on BINANCE_TESTNET env var)
- Historical klines (candles) fetching with pagination
- Account info (balances)
- Saving bars to parquet (fastparquet)

Usage:
    from src.connectors.binance_client import BinanceConnector

    conn = BinanceConnector()
    df = conn.get_bars("BTCUSDT", days=30)
    print(df.head())
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from binance.client import Client

load_dotenv()


# Map human-friendly intervals to python-binance constants
INTERVAL_MAP = {
    "1m":  Client.KLINE_INTERVAL_1MINUTE,
    "5m":  Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h":  Client.KLINE_INTERVAL_1HOUR,
    "4h":  Client.KLINE_INTERVAL_4HOUR,
    "1d":  Client.KLINE_INTERVAL_1DAY,
}


class BinanceConnector:
    """Thin wrapper around python-binance for Agent 29."""

    def __init__(self, testnet: bool | None = None):
        # If not explicitly set, infer from env (default to testnet=True for safety)
        if testnet is None:
            env_val = os.getenv("BINANCE_TESTNET", "True").lower()
            testnet = env_val in ("true", "1", "yes")
        self.testnet = testnet

        if testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            secret = os.getenv("BINANCE_TESTNET_SECRET")
            mode = "testnet"
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            secret = os.getenv("BINANCE_SECRET")
            mode = "LIVE"

        if not api_key or not secret:
            raise RuntimeError(
                f"Binance {mode} keys missing in .env "
                f"(need {'BINANCE_TESTNET_*' if testnet else 'BINANCE_*'})"
            )

        self.client = Client(api_key, secret, testnet=testnet)
        logger.info(f"BinanceConnector initialised in {mode} mode")

    # ---------- Account ----------
    def get_account(self) -> dict:
        """Returns simplified balance info: only assets with non-zero balance."""
        info = self.client.get_account()
        non_zero = [
            {
                "asset": b["asset"],
                "free": float(b["free"]),
                "locked": float(b["locked"]),
            }
            for b in info["balances"]
            if float(b["free"]) > 0 or float(b["locked"]) > 0
        ]
        return {
            "balances": non_zero,
            "can_trade": info.get("canTrade", False),
            "account_type": info.get("accountType", "unknown"),
        }

    # ---------- Historical bars ----------
    def get_bars(
        self,
        symbol: str,
        days: int = 30,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical klines.

        Returns a DataFrame with timestamp index and OHLCV columns
        (consistent with Alpaca's output for downstream code).
        """
        if interval not in INTERVAL_MAP:
            raise ValueError(
                f"interval must be one of {list(INTERVAL_MAP.keys())}, got {interval!r}"
            )

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        # python-binance accepts string dates like "30 days ago UTC"
        # but explicit timestamps are more robust
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        logger.info(
            f"Fetching {interval} klines for {symbol} from {start.date()} to {end.date()}"
        )
        klines = self.client.get_historical_klines(
            symbol=symbol,
            interval=INTERVAL_MAP[interval],
            start_str=str(start_ms),
            end_str=str(end_ms),
        )

        if not klines:
            logger.warning(f"No klines returned for {symbol}")
            return pd.DataFrame()

        # Binance kline columns:
        # [open_time, open, high, low, close, volume,
        #  close_time, quote_volume, trades, taker_base, taker_quote, ignore]
        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count",
            "taker_buy_base", "taker_buy_quote", "_ignore",
        ]
        df = pd.DataFrame(klines, columns=cols)

        # Convert types
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume", "quote_volume",
                    "taker_buy_base", "taker_buy_quote"]:
            df[col] = df[col].astype(float)
        df["trade_count"] = df["trade_count"].astype(int)

        # Add symbol col for consistency with Alpaca
        df["symbol"] = symbol

        # Set index, drop noise columns
        df = df.set_index(["symbol", "timestamp"])
        df = df[["open", "high", "low", "close", "volume", "quote_volume", "trade_count"]]

        logger.success(f"Fetched {len(df)} klines for {symbol}")
        return df

    # ---------- Saving ----------
    def save_bars(self, df: pd.DataFrame, name: str) -> Path:
        """Save klines DataFrame to data/raw/<name>.parquet using fastparquet."""
        out_dir = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.parquet"
        df.to_parquet(out_path, engine="fastparquet")
        logger.success(f"Saved {len(df)} rows to {out_path}")
        return out_path


if __name__ == "__main__":
    conn = BinanceConnector()

    print("\n--- Account ---")
    acct = conn.get_account()
    print(f"  account_type: {acct['account_type']}")
    print(f"  can_trade:    {acct['can_trade']}")
    print(f"  balances ({len(acct['balances'])} non-zero):")
    for b in acct["balances"][:10]:
        print(f"    {b['asset']:6s}  free={b['free']:>15.4f}  locked={b['locked']:>10.4f}")

    print("\n--- BTCUSDT 30 daily klines ---")
    df = conn.get_bars("BTCUSDT", days=30, interval="1d")
    print(df.tail())

    print("\n--- Saving to parquet ---")
    conn.save_bars(df, "btcusdt_30d_daily")
