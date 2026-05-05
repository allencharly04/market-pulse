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

# ---------- Funding rates (perpetual futures) ----------
    def get_funding_rate(
        self,
        symbol: str = "BTCUSDT",
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates for a perpetual futures pair.

        Funding rate = small periodic payment between longs and shorts on
        perpetual contracts. Strongly positive = longs over-leveraged
        (bearish contrarian). Strongly negative = shorts over-leveraged
        (bullish contrarian).

        Notes:
        - Endpoint is on Binance Futures (fapi), not Spot
        - Free, no auth needed for public endpoint
        - Rates settle every 8 hours (3 per day)
        """
        # Funding endpoint is on the futures API, not the spot API the client
        # is configured for. We hit it directly via requests.
        import requests
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - days * 24 * 3600 * 1000

        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }

        logger.info(f"[binance:funding] fetching {symbol} funding rates ({days}d)")
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df["symbol"] = symbol
        df = df[["symbol", "datetime", "funding_rate"]].sort_values("datetime", ascending=False)
        return df.reset_index(drop=True)

    # ---------- Open interest ----------
    def get_open_interest(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        days: int = 7,
    ) -> pd.DataFrame:
        """
        Fetch historical open interest stats for a perpetual futures pair.

        Open Interest = total notional value of open futures contracts.
        Rising OI + rising price = healthy trend (new money entering).
        Rising OI + falling price = bearish leverage building up.
        Falling OI = positions closing (trend exhaustion or capitulation).

        period: "5m" | "15m" | "30m" | "1h" | "2h" | "4h" | "6h" | "12h" | "1d"
        """
        import requests
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - days * 24 * 3600 * 1000

        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 500,
        }

        logger.info(f"[binance:oi] fetching {symbol} open interest ({days}d, {period})")
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["datetime"]   = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["oi_units"]   = df["sumOpenInterest"].astype(float)        # contracts
        df["oi_usd"]     = df["sumOpenInterestValue"].astype(float)   # notional USD
        df["symbol"]     = symbol
        df = df[["symbol", "datetime", "oi_units", "oi_usd"]].sort_values("datetime", ascending=False)
        return df.reset_index(drop=True)

    # ---------- Funding/OI features ----------
    def funding_oi_features(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Build a single-row feature DataFrame combining latest funding + OI signals.
        """
        funding = self.get_funding_rate(symbol, days=30)
        oi = self.get_open_interest(symbol, period="1h", days=7)

        feat: dict = {
            "fund_oi_symbol":     symbol,
            "fund_oi_fetched_at": datetime.now(timezone.utc),
        }

        if not funding.empty:
            f_latest = funding.iloc[0]["funding_rate"]
            f_mean_30d = funding["funding_rate"].mean()
            f_std_30d = funding["funding_rate"].std()
            feat[f"funding_latest"]      = float(f_latest)
            feat[f"funding_mean_30d"]    = float(f_mean_30d)
            feat[f"funding_zscore_30d"]  = float((f_latest - f_mean_30d) / f_std_30d) if f_std_30d > 0 else 0.0
            feat[f"funding_positive"]    = int(f_latest > 0)
            feat[f"funding_extreme_pos"] = int(f_latest > 0.01 / 100 * 3)   # > 3x typical 0.01%
            feat[f"funding_extreme_neg"] = int(f_latest < -0.01 / 100 * 3)

        if not oi.empty and len(oi) >= 24:
            oi_latest = oi.iloc[0]["oi_usd"]
            oi_24h_ago = oi.iloc[24]["oi_usd"] if len(oi) > 24 else oi.iloc[-1]["oi_usd"]
            oi_chg_24h_pct = (oi_latest - oi_24h_ago) / oi_24h_ago * 100 if oi_24h_ago > 0 else 0
            feat[f"oi_usd_latest"]       = float(oi_latest)
            feat[f"oi_chg_pct_24h"]      = float(oi_chg_24h_pct)
            feat[f"oi_rising_24h"]       = int(oi_chg_24h_pct > 0)
            feat[f"oi_extreme_rise_24h"] = int(oi_chg_24h_pct > 10)
            feat[f"oi_extreme_drop_24h"] = int(oi_chg_24h_pct < -10)

        return pd.DataFrame([feat])

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

    print("\n--- Funding rate (BTCUSDT, 7d) ---")
    fdf = conn.get_funding_rate("BTCUSDT", days=7)
    print(f"  rows: {len(fdf)}")
    if not fdf.empty:
        print(fdf.head(5).to_string(index=False))
        print(f"  Latest funding: {fdf.iloc[0]['funding_rate']*100:.4f}% per 8h")
        annualized = fdf.iloc[0]['funding_rate'] * 3 * 365 * 100
        print(f"  Annualized:     {annualized:.2f}%")

    print("\n--- Open interest (BTCUSDT, 7d, 1h) ---")
    oidf = conn.get_open_interest("BTCUSDT", period="1h", days=7)
    print(f"  rows: {len(oidf)}")
    if not oidf.empty:
        print(oidf.head(5).to_string(index=False))

    print("\n--- Combined funding+OI features ---")
    feat = conn.funding_oi_features("BTCUSDT")
    if not feat.empty:
        for col in feat.columns:
            print(f"  {col:30s} = {feat[col].iloc[0]}")
