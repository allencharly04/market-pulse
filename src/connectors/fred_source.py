"""
FRED (Federal Reserve Economic Data) source for Agent 29.

Pulls macro context features that determine which strategies should
be active in current market conditions:
- VIX: equity volatility regime
- Yield curve (10Y-2Y): recession/growth signal
- DXY: dollar strength (affects multinationals + crypto inversely)
- Fed funds: monetary policy stance
- CBOE put/call ratio: smart money positioning

These don't predict individual stocks. They tell us:
- "Trend strategies should run hard right now" (low VIX, normal curve)
- "Risk-off mode" (VIX spike, inverted curve, strong DXY)
- "Crisis mode" (VIX > 30, kill all but mean-reversion)

Free tier: 120 calls/min, no daily cap. Plenty.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from loguru import logger

from src.connectors.base import DataSource, SourceHealth

load_dotenv()


# Curated FRED series IDs and what they mean.
# Series can be added/removed via the dict — to_features() iterates over what's present.
FRED_SERIES = {
    # Volatility regime
    "VIXCLS":      "vix",                # CBOE Volatility Index (S&P implied vol)

    # Yield curve / recession signals
    "DGS10":       "treasury_10y",       # 10-Year Treasury yield
    "DGS2":        "treasury_2y",        # 2-Year Treasury yield
    "T10Y2Y":      "yield_curve_10y2y",  # 10Y - 2Y spread (recession indicator if negative)
    "T10Y3M":      "yield_curve_10y3m",  # 10Y - 3M (more sensitive recession indicator)

    # Dollar strength
    "DTWEXBGS":    "dxy_broad",          # Trade-weighted USD index (broad)

    # Fed policy
    "DFF":         "fed_funds",          # Effective Federal Funds Rate

    # Smart money positioning
    # NOTE: CBOE put/call data on FRED has limited history; we'll add
    # a fallback to a direct CBOE scraper later if this proves unreliable.

    # Macro health
    "UNRATE":      "unemployment",       # Unemployment rate (monthly)
    "CPIAUCSL":    "cpi",                # CPI (monthly, lagging)
}


class FREDSource(DataSource):
    name = "fred"
    category = "macro"
    requires_auth = True
    rate_limit_per_min = 120

    def __init__(self, enabled: bool = True, weight: float = 1.5,
                 series: dict[str, str] | None = None):
        super().__init__(enabled=enabled, weight=weight)
        self.api_key = os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise RuntimeError("FRED_API_KEY missing in .env")
        self.fred = Fred(api_key=self.api_key)
        self.series = series or FRED_SERIES

    def fetch(self, days: int = 90, **kwargs) -> pd.DataFrame:
        """
        Fetch all configured FRED series for the past `days` days.

        Returns long-format DataFrame: columns = [series_id, name, date, value]
        """
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=days)

        logger.info(f"[{self.name}] fetching {len(self.series)} series for {days}d")
        rows = []
        for series_id, friendly_name in self.series.items():
            try:
                s = self.fred.get_series(
                    series_id,
                    observation_start=str(start),
                    observation_end=str(end),
                )
                if s.empty:
                    logger.warning(f"[fred:{series_id}] empty series")
                    continue
                for date, value in s.items():
                    if pd.notna(value):
                        rows.append({
                            "series_id":  series_id,
                            "name":       friendly_name,
                            "date":       pd.to_datetime(date).tz_localize("UTC"),
                            "value":      float(value),
                        })
            except Exception as e:
                logger.warning(f"[fred:{series_id}] error: {type(e).__name__}: {e}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values(["series_id", "date"], ascending=[True, False])
        return df

    def to_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Compute regime features:
        - Latest value of each series
        - 5d / 30d change for trending indicators
        - Z-score over 90d for VIX (regime detection)
        """
        if raw.empty:
            return pd.DataFrame()

        feat: dict = {"fred_fetched_at": datetime.now(timezone.utc)}

        for series_id, friendly_name in self.series.items():
            s = raw[raw["series_id"] == series_id].sort_values("date")
            if s.empty:
                continue

            values = s["value"].values
            # Latest value
            feat[f"fred_{friendly_name}_latest"] = values[-1]

            # 5d change
            if len(values) >= 5:
                feat[f"fred_{friendly_name}_chg_5d"] = values[-1] - values[-5]

            # 30d change (skip if not enough data)
            if len(values) >= 30:
                feat[f"fred_{friendly_name}_chg_30d"] = values[-1] - values[-30]

            # Z-score over full window (regime detection)
            if len(values) >= 20:
                mean = values.mean()
                std = values.std()
                feat[f"fred_{friendly_name}_zscore"] = (
                    (values[-1] - mean) / std if std > 0 else 0
                )

        # Derived regime flags (the actual "is the market behaving normally?" check)
        if "fred_vix_latest" in feat:
            vix = feat["fred_vix_latest"]
            feat["regime_vix_calm"]     = int(vix < 15)
            feat["regime_vix_normal"]   = int(15 <= vix < 25)
            feat["regime_vix_elevated"] = int(25 <= vix < 35)
            feat["regime_vix_crisis"]   = int(vix >= 35)

        if "fred_yield_curve_10y2y_latest" in feat:
            curve = feat["fred_yield_curve_10y2y_latest"]
            feat["regime_curve_inverted"] = int(curve < 0)
            feat["regime_curve_steep"]    = int(curve > 1.5)

        return pd.DataFrame([feat])

    def health_check(self) -> SourceHealth:
        t0 = datetime.now(timezone.utc)
        try:
            # Cheap call — just get latest VIX
            s = self.fred.get_series_latest_release("VIXCLS")
            healthy = s is not None and len(s) > 0
            elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
            return SourceHealth(
                name=self.name, healthy=healthy,
                last_check=datetime.now(timezone.utc),
                latency_ms=elapsed_ms,
                error=None if healthy else "empty response",
            )
        except Exception as e:
            return SourceHealth(
                name=self.name, healthy=False,
                last_check=datetime.now(timezone.utc),
                latency_ms=-1, error=f"{type(e).__name__}: {e}",
            )


if __name__ == "__main__":
    src = FREDSource()

    print("\n--- Health check ---")
    h = src.health_check()
    print(f"  healthy: {h.healthy}   latency: {h.latency_ms:.0f} ms")
    if h.error:
        print(f"  error: {h.error}")

    print(f"\n--- Fetching {len(src.series)} series, last 90 days ---")
    df = src.fetch(days=90)
    print(f"  total rows: {len(df)}")

    if not df.empty:
        print(f"\n  Latest value of each series:")
        latest = df.sort_values("date").groupby("series_id").tail(1)
        for _, row in latest.iterrows():
            print(f"    {row['series_id']:10s} ({row['name']:25s}) "
                  f"= {row['value']:>10.4f}  on {row['date'].date()}")

    print("\n--- Features (regime context) ---")
    feat = src.to_features(df)
    if not feat.empty:
        for col in feat.columns:
            print(f"  {col:40s} = {feat[col].iloc[0]}")
