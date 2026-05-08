# Market Pulse

A multi-source AI trading signal platform. Built in 4 days as a personal project to explore production-grade ML engineering for finance: real-time data ingestion across 9 sources, GPU-accelerated sentiment analysis, technical feature engineering, and walk-forward model validation.

> **Status:** v1 baseline complete. Pipeline is end-to-end functional; ML modeling shows the honest "lift ≈ 0" result expected on liquid daily data without time-aligned macro/sentiment features. See [`docs/V2_PLAN.md`](docs/V2_PLAN.md) for the structural fixes scheduled for v2.

---

## What it does

![Dashboard screenshot](docs/screenshots/dashboard.png)

- **Ingests** real-time market data from 9 sources: Alpaca (stocks), Binance (crypto), Finnhub (news), 17 RSS feeds, 4 Telegram channels, NewsAPI, FRED (macro), Crypto Fear & Greed, plus Binance funding rates and open interest
- **Scores** every news headline with FinBERT (financial sentiment, GPU-accelerated, ~700 headlines/sec on RTX 2060)
- **Tags** headlines with affected tickers using cashtag + alias + symbol matching (78 stocks + 23 crypto, 14/14 unit tests passing)
- **Computes** 56 technical indicators per ticker per bar (RSI, MACD, ATR, Bollinger, ADX, momentum, volume, regime detection)
- **Aggregates** sentiment per ticker across multiple time windows (24h / 72h / 168h)
- **Persists** to SQLite + Parquet for fast querying and ML training
- **Visualizes** via live Streamlit dashboard with macro regime panel, ticker leaderboard, and filter-aware headline feed
- **Trains** a LightGBM binary classifier with chronological train/val/test split, early stopping, per-ticker accuracy breakdown, and feature importance

## Architecture
src/
├── connectors/        # 9 data source plugins (parallel fetch, ~14s/cycle)
├── sentiment/         # FinBERT + VADER scorers, ticker extraction
├── features/          # 56 technical indicators + master feature store
├── models/            # LightGBM trainer + dataset prep
├── dashboard/         # Streamlit live dashboard
└── pipeline.py        # Orchestrator

## Honest results (v1 baseline)

Tested 3 horizons on 20 tickers, 7,720 train / 1,640 test rows, chronological split:

| Horizon  | Test Accuracy | Baseline | Lift   | AUC  |
|----------|---------------|----------|--------|------|
| 1 day    | 52.6%         | 52.7%    | -0.001 | 0.51 |
| 5 days   | 50.3%         | 50.3%    |  0.000 | 0.45 |
| 20 days  | 45.4%         | 54.6%    | -0.091 | 0.44 |

This is the **expected** result on liquid daily data when macro and sentiment features are broadcast as constants instead of time-aligned. Pipeline is structurally sound; data layer needs upgrade before the model layer can show signal. **Most AI trading content shows 70%+ accuracy through leakage; this is what honest validation looks like.**

## Tech stack

**Data:** Alpaca, Binance, Finnhub, RSS (feedparser), Telegram (Telethon), NewsAPI, FRED, alternative.me

**ML:** PyTorch, transformers (FinBERT — ProsusAI), LightGBM, scikit-learn, ta (technical analysis library)

**Storage:** SQLite, Parquet (fastparquet)

**Frontend:** Streamlit, Plotly

**Infra:** Python 3.11, conda env, WSL2, RTX 2060 (CUDA 12.1)

## Getting started

```bash
git clone https://github.com/allencharly04/market-pulse.git
cd market-pulse

conda create -n market-pulse python=3.11 -y
conda activate market-pulse
pip install -r requirements.txt

# Add API keys to .env (see .env.example)
cp .env.example .env  # then edit with your own keys

# Run one orchestrator cycle (fetches all sources, scores sentiment, persists)
python -m src.pipeline --once

# Launch the dashboard
streamlit run src/dashboard/app.py

# Build the master feature matrix and train a baseline model
python -m src.features.feature_store
python -m src.models.train --horizon 5
```

## Roadmap

See [`docs/V2_PLAN.md`](docs/V2_PLAN.md) for detailed v2 priorities.

- **P0:** Time-aligned macro features (FRED daily history, joined on date instead of broadcast)
- **P1:** Time-aligned sentiment features (rolling windows over `published_at`)
- **P2:** Walk-forward CV instead of single chronological split
- **P3:** Per-ticker models (universal model dilutes signal across heterogeneous tickers)
- **P4:** Better targets (volatility forecasting, regime classification, conviction-weighted direction)

## Notes

This was built in 4 days as a learning project. Code is structured for clarity over performance; the pipeline could be faster, the dashboard nicer, the tests more comprehensive. Where shortcuts were taken, they're documented in the v2 plan.

---

Built by [@allencharly04](https://github.com/allencharly04) — currently doing M.Sc. Digital Engineering and Management at RWTH Aachen. Find me on Twitter / TikTok as `@charnelally`
