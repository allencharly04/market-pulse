"""
Agent 29 — Live Dashboard

Real-time visualization of the multi-source sentiment pipeline.
Reads from SQLite (data/agent29.db), auto-refreshes every 30 seconds.

Run:
    streamlit run src/dashboard/app.py
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import time
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard import queries as q

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Agent 29 | Multi-Market Trading Signals",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a darker, more "trading terminal" feel
st.markdown(
    """
    <style>
    .metric-positive { color: #22c55e; font-weight: 600; }
    .metric-negative { color: #ef4444; font-weight: 600; }
    .metric-neutral  { color: #94a3b8; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .small-caption { color: #64748b; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("⚡ Agent 29")
    st.caption("Multi-Market Trading Signal Platform")

    st.divider()

    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if st.button("🔄 Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.subheader("Filters")
    available_tickers = ["(all)"] + q.all_tickers_seen()
    selected_ticker = st.selectbox("Ticker", available_tickers, index=0)
    available_sources = ["(all)"] + q.all_sources_seen()
    selected_source = st.selectbox("Source", available_sources, index=0)

    sentiment_filter = st.select_slider(
        "Sentiment range",
        options=["very negative", "negative", "any", "positive", "very positive"],
        value="any",
    )
    sentiment_map = {
        "very negative": (-1.0, -0.5),
        "negative":      (-1.0, -0.05),
        "any":           (-1.0, 1.0),
        "positive":      (0.05, 1.0),
        "very positive": (0.5, 1.0),
    }
    min_c, max_c = sentiment_map[sentiment_filter]

    st.divider()
    st.caption("v0.1 • Day 4 build")


# ============================================================
# Header — system health
# ============================================================
st.title("📊 Agent 29 — Live Signals")

cycle = q.latest_cycle()
if cycle is None:
    st.error("No pipeline cycles in the database yet. Run: `python -m src.pipeline --once`")
    st.stop()

started = pd.to_datetime(cycle["started_at"], utc=True)
age_seconds = (datetime.now(timezone.utc) - started).total_seconds()
age_str = (
    f"{int(age_seconds)}s ago" if age_seconds < 60
    else f"{int(age_seconds / 60)}m ago" if age_seconds < 3600
    else f"{int(age_seconds / 3600)}h ago"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last cycle", age_str)
c2.metric("Sources OK",
          f"{int(cycle['sources_ok'])}/{int(cycle['sources_ok'] + cycle['sources_failed'])}")
c3.metric("Cycle duration", f"{cycle['duration_sec']:.1f}s")
c4.metric("Headlines (this cycle)", int(cycle["news_scored"]))
c5.metric("Total headlines (DB)", q.total_headlines())


# ============================================================
# Macro regime panel
# ============================================================
st.subheader("🌍 Macro Regime")

vix = q.macro_value("fred_vix_latest")
yield_curve = q.macro_value("fred_yield_curve_10y2y_latest")
fed_funds = q.macro_value("fred_fed_funds_latest")
dxy = q.macro_value("fred_dxy_broad_latest")
fng_value = q.macro_value("fng_latest")
fng_class = q.macro_value("fng_classification")

m1, m2, m3, m4, m5 = st.columns(5)


def _fmt(val, fmt: str = ".2f") -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


m1.metric("VIX", _fmt(vix), help="<15 calm, 15-25 normal, 25-35 elevated, >35 crisis")
m2.metric("10Y-2Y curve", _fmt(yield_curve), help="Negative = recession signal")
m3.metric("Fed Funds", _fmt(fed_funds, ".2f") + "%" if fed_funds is not None else "—")
m4.metric("USD Index", _fmt(dxy, ".1f"))
m5.metric(
    "Crypto F&G",
    f"{_fmt(fng_value, '.0f')} ({fng_class or '—'})",
    help="0=Extreme Fear, 100=Extreme Greed"
)

# Regime flag chips
regime_flags = []
for flag_name, label in [
    ("regime_vix_calm", "VIX Calm"),
    ("regime_vix_normal", "VIX Normal"),
    ("regime_vix_elevated", "VIX Elevated"),
    ("regime_vix_crisis", "VIX Crisis"),
    ("regime_curve_inverted", "Curve Inverted"),
    ("regime_curve_steep", "Curve Steep"),
    ("fng_extreme_fear", "Crypto Extreme Fear"),
    ("fng_fear", "Crypto Fear"),
    ("fng_greed", "Crypto Greed"),
    ("fng_extreme_greed", "Crypto Extreme Greed"),
]:
    val = q.macro_value(flag_name)
    if val == 1.0 or val == 1:
        regime_flags.append(label)

if regime_flags:
    st.markdown(
        "**Active regimes:** " + " · ".join(f"`{r}`" for r in regime_flags)
    )


st.divider()


# ============================================================
# Sentiment overview
# ============================================================
st.subheader("📰 News Sentiment (last 24h)")

col_a, col_b = st.columns([1, 2])

with col_a:
    sd = q.sentiment_distribution(hours=24)
    if not sd.empty:
        order_map = {"positive": 1, "neutral": 2, "negative": 3}
        sd["order"] = sd["finbert_label"].map(order_map)
        sd = sd.sort_values("order")

        colors = {"positive": "#22c55e", "neutral": "#94a3b8", "negative": "#ef4444"}
        fig = go.Figure(
            data=[go.Pie(
                labels=sd["finbert_label"],
                values=sd["n"],
                hole=0.5,
                marker=dict(colors=[colors.get(l, "#94a3b8") for l in sd["finbert_label"]]),
                textinfo="label+percent",
            )]
        )
        fig.update_layout(
            showlegend=False,
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No headlines in last 24h yet")

with col_b:
    vol = q.headline_volume_by_hour(hours=48)
    if not vol.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vol["hour"], y=vol["n_pos"], name="Positive", marker_color="#22c55e",
        ))
        fig.add_trace(go.Bar(
            x=vol["hour"], y=vol["n_neg"], name="Negative", marker_color="#ef4444",
        ))
        fig.add_trace(go.Bar(
            x=vol["hour"], y=vol["n"] - vol["n_pos"] - vol["n_neg"],
            name="Neutral", marker_color="#94a3b8",
        ))
        fig.update_layout(
            barmode="stack",
            height=260,
            margin=dict(t=10, b=10, l=10, r=10),
            title="Headlines per hour (48h)",
            yaxis_title="count",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for hourly chart yet")


# ============================================================
# Per-ticker leaderboard
# ============================================================
st.subheader("🎯 Ticker Leaderboard (last 7 days)")

lb = q.ticker_leaderboard(min_headlines=2, hours=168)
if not lb.empty:
    lb_display = lb.copy()
    lb_display["avg_sentiment"] = lb_display["avg_sentiment"].round(3)
    lb_display = lb_display.rename(columns={
        "ticker":         "Ticker",
        "headlines":      "Headlines",
        "avg_sentiment":  "Avg sentiment",
        "n_pos":          "Pos",
        "n_neg":          "Neg",
        "n_neu":          "Neu",
    })

# Build the leaderboard table manually for proper sentiment bars
    rows_html = []
    for _, row in lb_display.iterrows():
        val = float(row["Avg sentiment"])
        pct = abs(val) * 50  # 0..1 → 0..50% of bar width

        if val >= 0:
            bar = (
                f'<div style="display:flex;align-items:center;width:100%;">'
                f'<div style="flex:1;"></div>'
                f'<div style="width:50%;display:flex;align-items:center;">'
                f'<div style="width:{pct}%;background:#22c55e;height:10px;border-radius:2px;"></div>'
                f'<span style="margin-left:8px;color:#22c55e;font-size:0.9rem;">{val:+.3f}</span>'
                f'</div>'
                f'</div>'
            )
        else:
            bar = (
                f'<div style="display:flex;align-items:center;width:100%;">'
                f'<div style="width:50%;display:flex;align-items:center;justify-content:flex-end;">'
                f'<span style="margin-right:8px;color:#ef4444;font-size:0.9rem;">{val:+.3f}</span>'
                f'<div style="width:{pct}%;background:#ef4444;height:10px;border-radius:2px;"></div>'
                f'</div>'
                f'<div style="flex:1;"></div>'
            )

        rows_html.append(
            f"<tr>"
            f"<td style='padding:8px 12px;font-weight:600;'>{row['Ticker']}</td>"
            f"<td style='padding:8px 12px;'>{int(row['Headlines'])}</td>"
            f"<td style='padding:8px 12px;width:40%;'>{bar}</td>"
            f"<td style='padding:8px 12px;color:#22c55e;'>{int(row['Pos'])}</td>"
            f"<td style='padding:8px 12px;color:#ef4444;'>{int(row['Neg'])}</td>"
            f"<td style='padding:8px 12px;color:#94a3b8;'>{int(row['Neu'])}</td>"
            f"</tr>"
        )

    table_html = f"""
    <style>
    .ticker-leaderboard {{
        width: 100%;
        border-collapse: collapse;
        font-family: inherit;
    }}
    .ticker-leaderboard th {{
        text-align: left;
        padding: 10px 12px;
        border-bottom: 2px solid #334155;
        color: #94a3b8;
        font-weight: 500;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .ticker-leaderboard td {{
        border-bottom: 1px solid #1e293b;
    }}
    .ticker-leaderboard tr:hover td {{
        background: #1e293b;
    }}
    </style>
    <table class="ticker-leaderboard">
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Headlines</th>
                <th style="text-align:center;">Sentiment</th>
                <th>Pos</th>
                <th>Neg</th>
                <th>Neu</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)

else:
    st.info("No ticker-tagged headlines yet")


# ============================================================
# Headline feed
# ============================================================
# Build a clear filter-aware section with visible match count
heads = q.recent_headlines(
    limit=100,
    ticker=None if selected_ticker == "(all)" else selected_ticker,
    source=None if selected_source == "(all)" else selected_source,
    min_compound=min_c if min_c > -1.0 else None,
    max_compound=max_c if max_c < 1.0 else None,
)

filter_chips = []
if selected_ticker != "(all)":
    filter_chips.append(f"<span style='background:#3b82f6;color:white;padding:2px 8px;border-radius:4px;font-size:0.8rem;'>📌 {selected_ticker}</span>")
if selected_source != "(all)":
    filter_chips.append(f"<span style='background:#6366f1;color:white;padding:2px 8px;border-radius:4px;font-size:0.8rem;'>🌐 {selected_source}</span>")
if sentiment_filter != "any":
    filter_chips.append(f"<span style='background:#a855f7;color:white;padding:2px 8px;border-radius:4px;font-size:0.8rem;'>🎯 {sentiment_filter}</span>")

st.markdown(
    f"<h3>📡 Recent Headlines &nbsp; "
    f"<span style='color:#22c55e;'>{len(heads)} matches</span></h3>"
    f"<div style='margin-bottom:12px;'>{' '.join(filter_chips) if filter_chips else '<small style=color:#64748b>No filters active — showing latest from all tickers/sources</small>'}</div>",
    unsafe_allow_html=True,
)


# Filter result count — visual confirmation
if heads.empty:
    st.info("No headlines match the current filters")
else:
    for _, row in heads.head(50).iterrows():
        compound = row["finbert_compound"]
        if compound is None:
            color = "neutral"
            indicator = "•"
        elif compound >= 0.3:
            color = "positive"
            indicator = "▲"
        elif compound <= -0.3:
            color = "negative"
            indicator = "▼"
        else:
            color = "neutral"
            indicator = "•"

        ticker_chip = f"`{row['primary_ticker']}`" if row["primary_ticker"] else ""
        source_chip = f"_{row['source']}_"
        origin_chip = f"_{row['origin']}_" if row["origin"] else ""
        ts = row["published_at"].strftime("%H:%M") if pd.notna(row["published_at"]) else "—"

        line = (
            f"<span class='metric-{color}'>{indicator} "
            f"{compound:+.2f}</span> "
            f"<span class='small-caption'>{ts} · {source_chip} · {origin_chip} {ticker_chip}</span><br>"
            f"{row['title']}"
        )
        st.markdown(line, unsafe_allow_html=True)
        st.divider()


# ============================================================
# Source health
# ============================================================
st.subheader("🔧 Source Health")

src_counts = q.headlines_by_source()
if not src_counts.empty:
    fig = px.bar(
        src_counts, x="source", y="n",
        title="Headlines collected per source (all time)",
        color="n", color_continuous_scale="viridis",
    )
    fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Auto-refresh
# ============================================================
if auto_refresh:
    time.sleep(30)
    st.cache_data.clear()
    st.rerun()
