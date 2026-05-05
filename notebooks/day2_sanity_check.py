"""
Day 2 sanity check: load the parquet files we just saved,
verify OHLC integrity, and plot price + volume for both markets.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Paths
RAW = Path("data/raw")
SPY_FILE = RAW / "spy_30d_daily.parquet"
BTC_FILE = RAW / "btcusdt_30d_daily.parquet"

for f in (SPY_FILE, BTC_FILE):
    if not f.exists():
        sys.exit(f"Missing: {f} — run the connector smoke tests first.")

# Load
spy = pd.read_parquet(SPY_FILE, engine="fastparquet")
btc = pd.read_parquet(BTC_FILE, engine="fastparquet")

print("=" * 60)
print("DATA INTEGRITY CHECKS")
print("=" * 60)

def check(df: pd.DataFrame, name: str):
    print(f"\n[{name}]")
    print(f"  Rows:                {len(df)}")
    print(f"  Columns:             {list(df.columns)}")
    print(f"  Date range:          {df.index.get_level_values('timestamp').min()} -> {df.index.get_level_values('timestamp').max()}")
    print(f"  Any NaNs:            {df.isna().any().any()}")
    # OHLC integrity: high >= low, high >= open/close, low <= open/close
    bad_hl = (df["high"] < df["low"]).sum()
    bad_ho = (df["high"] < df["open"]).sum()
    bad_hc = (df["high"] < df["close"]).sum()
    bad_lo = (df["low"] > df["open"]).sum()
    bad_lc = (df["low"] > df["close"]).sum()
    bad_total = bad_hl + bad_ho + bad_hc + bad_lo + bad_lc
    print(f"  OHLC violations:     {bad_total} (must be 0)")
    print(f"  Last close:          {df['close'].iloc[-1]:.2f}")
    print(f"  30d return:          {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")
    return bad_total == 0

ok_spy = check(spy, "SPY")
ok_btc = check(btc, "BTCUSDT")

print("\n" + "=" * 60)
print(f"OVERALL: {'PASSED' if (ok_spy and ok_btc) else 'FAILED'}")
print("=" * 60)

# Build a 2x2 plot: price + volume for each market
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "SPY — Daily Close",  "SPY — Daily Volume",
        "BTCUSDT — Daily Close (testnet)", "BTCUSDT — Daily Volume (testnet)",
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.1,
)

# Reset index so timestamp is a column for plotting
spy_p = spy.reset_index()
btc_p = btc.reset_index()

fig.add_trace(go.Scatter(x=spy_p["timestamp"], y=spy_p["close"], name="SPY close",
                         line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Bar(x=spy_p["timestamp"], y=spy_p["volume"], name="SPY vol",
                     marker_color="#1f77b4"), row=1, col=2)
fig.add_trace(go.Scatter(x=btc_p["timestamp"], y=btc_p["close"], name="BTC close",
                         line=dict(color="#f7931a")), row=2, col=1)
fig.add_trace(go.Bar(x=btc_p["timestamp"], y=btc_p["volume"], name="BTC vol",
                     marker_color="#f7931a"), row=2, col=2)

fig.update_layout(
    title="Agent 29 Day 2 — Data Sanity Check",
    height=700, width=1200, showlegend=False,
    template="plotly_dark",
)

out = Path("notebooks/day2_sanity_check.html")
fig.write_html(out)
print(f"\nPlot saved to: {out.resolve()}")
print("Open it from File Explorer (D:\\allen\\Agents\\29...\\notebooks\\day2_sanity_check.html) in any browser.")
