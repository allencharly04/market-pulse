# Agent 29 — V2 Plan

> Status: Day 4 complete. End-to-end pipeline working (data → features → ML).
> ML baseline shows lift ≈ 0, which is structurally explained, not a model bug.

## Honest assessment of v1 (Day 1-4)

**What works:**
- 9 data sources fetched in parallel (~14 sec/cycle for 558 headlines)
- FinBERT GPU sentiment scoring (~700 headlines/sec on RTX 2060)
- Ticker extraction (78 stocks + 23 crypto, 14/14 unit tests passing)
- Streamlit dashboard with macro panel + ticker leaderboard + filter-aware feed
- Feature engineering layer (56 technical indicators per ticker per bar)
- Master feature parquet (15,020 rows × 118 cols)
- LightGBM training with chronological CV, early stopping, per-ticker eval, save/load

**What doesn't work yet (and why):**
- ML lift ≈ 0% across 1d/5d/20d horizons
- Reason: macro and sentiment features are broadcast as constants (today's value on all historical rows)
- This is a data-layer time-alignment bug, not a model bug
- Without proper time alignment, model is being asked to predict from technical features alone, which is genuinely close to random at daily horizons

## V2 priorities (in order)

### P0 — Time-aligned macro features (~3 hr)
Currently `feature_store.py` broadcasts the latest cycle's macro values to every historical row. Need to:
- Modify `fred_source.py` to fetch + store full daily history per series (not just current value)
- Build a `macro_history` table in SQLite keyed on (date, feature_name)
- In `feature_store.build_master_features()`, merge macro features by joining on date
- Verify variance: each macro column should have non-zero std across rows

### P1 — Time-aligned sentiment features (~2 hr)
Currently `feature_store.load_sentiment_per_ticker()` returns the *current* sentiment for each ticker, broadcast across history.
Need to: for each (ticker, date) row, compute sentiment using only news with `published_at <= that_date`. Rolling window aggregation.

### P2 — Walk-forward CV (~1.5 hr)
Replace single chronological split with rolling windows:
- Train on months 1-12, test on month 13
- Train on months 1-13, test on month 14
- ... etc.
- Average metrics across all folds
- Better generalization estimate, more honest reporting

### P3 — Per-ticker models (~2 hr)
Universal model dilutes signal across heterogeneous tickers (AMD ≠ CRM). Train one model per ticker; aggregate results.

### P4 — Better targets (~1 hr)
Try volatility forecasting, regime classification, conviction-weighted direction (only predict when confidence high).

## Out of scope for v2

- Live paper trading (needs all of P0-P3 first)
- Crypto modeling (need longer Binance history first)
- Backtester with costs/slippage (build after model layer is honest)

## Known small bugs

- `model.safetensors` partial-download warning on first FinBERT run (harmless)
- One FRED series occasionally 500s — graceful failure works
- BitcoinMagazine RSS feed has malformed XML — graceful failure works
- `XLM` ticker false-positive rate ~1/258 (low priority)
