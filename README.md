# Agent 29 — Multi-Market Trading Signal Platform

Production-grade signal engine that scans US stocks and crypto, generates
ML-enhanced signals across multiple strategies with regime-aware switching,
paper-trades them via Alpaca and Binance testnet, and alerts via Telegram.
Live-trading rails behind a config flag.

## Status

- **Day 1**: ✅ Environment, dependencies, smoke test passed
- **Day 2**: 🔜 Broker connectors (Alpaca, Binance) + first data pull
- **Day 3+**: See build plan below

## Markets

- **US stocks** (primary) — Alpaca paper → live, ~50 liquid tickers
- **Crypto spot** (secondary) — Binance testnet → live, top 20 by volume
  *(no leverage, ever)*

Forex was evaluated and dropped: 80%+ retail loss rates, retail accounts
for only 2.5% of FX volume, weakest signal sources.

## Architecture
data sources         feature layer        signal layer         risk + execution
────────────         ─────────────        ────────────         ────────────────
Alpaca   ─┐                              ┌─ Trend / momentum
Binance  ─┼─→  raw bars + book  ─→  features  ─┼─ Mean reversion   ─→  regime  ─→  risk  ─→  Alpaca / Binance
News     ─┤                              ├─ Volatility            detector     manager      (paper or live)
Reddit   ─┤                              ├─ Microstructure                                       │
FRED     ─┘                              ├─ Sentiment (FinBERT)                                  ▼
└─ ML predictor (LightGBM)                          Telegram +
Streamlit

## Tech stack

- **Python 3.11** in WSL2 Ubuntu 24.04
- **Data:** pandas 2.2, numpy 1.26, fastparquet
- **ML:** scikit-learn, lightgbm, xgboost, torch (CUDA on RTX 2060)
- **Sentiment:** transformers + FinBERT
- **Brokers:** alpaca-py, python-binance
- **Dashboard:** Streamlit + Plotly
- **Alerts:** python-telegram-bot

## Project structure
.
├── config/         # YAML configs: markets, strategies, risk
├── data/
│   ├── raw/        # raw bars, news, sentiment (gitignored)
│   └── processed/  # feature matrices (gitignored)
├── src/
│   ├── connectors/ # Alpaca, Binance, news, Reddit, FRED
│   ├── features/   # feature engineering
│   ├── strategies/ # signal generators
│   ├── regime/     # market regime detector
│   ├── risk/       # position sizing, circuit breakers
│   ├── execution/  # OrderManager (paper / live)
│   ├── alerts/     # Telegram bot
│   └── dashboard/  # Streamlit app
├── notebooks/      # exploratory analysis
├── tests/
└── logs/           # runtime logs (gitignored)

## Setup

```bash
conda create -n agent29 python=3.11 -y
conda activate agent29
pip install -r requirements.txt
python src/smoke_test.py
```

## Build plan (5 weeks)

| Week | Days | Deliverable |
|------|------|-------------|
| 1 | 1–3 | WSL env, brokers wired, data ingestion |
| 2 | 4–8 | Historical pipeline, features, backtester |
| 3 | 9–13 | All 6 signal layers + unit tests |
| 4 | 14–18 | Regime detector, ensembling, risk manager |
| 5 | 19–23 | Live paper loop, Telegram, dashboard |

## License

Personal project — not for redistribution.
