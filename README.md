# STOCK

A personal project exploring short-horizon crypto price direction forecasting.
Each pipeline pulls OHLCV candles for a set of coins straight from Binance's
public REST API, engineers a handful of technical-indicator features, trains
an XGBoost classifier to predict whether price will rise more than a set
threshold within a fixed horizon, and writes out a forecast CSV plus a chart
of the current signal.

This is not a trading bot — it doesn't place orders or connect to a broker.
It's a self-contained research pipeline: fetch data, build features, train a
model, score the latest bars, and plot the result.

## Key features

- **No API key required** — market data comes from Binance's public `klines`
  and `ticker/price` endpoints, which don't need authentication for read-only
  access.
- **Two independent pipelines**, each a single self-driving script that
  fetches, engineers features, (re)trains, and forecasts in one run:
  - `Stock_Predict/Hourly Model/crypto_pipeline_hourly.py` — hourly bars
    across five coins (BTC, ETH, SOL, BNB, DOGE); predicts the probability
    that price rises more than 1.5% within the next 10 hours.
  - `Stock_Predict/crypto_pipeline.py` — 30-minute bars for ETH only, same
    1.5% / 10-hour target, just expressed in finer-grained bars.
- **Leakage-aware training** — chronological train/validation/test splits
  with an embargo gap around the prediction horizon, plus walk-forward
  cross-validation (`TimeSeriesSplit`) for a sanity-checked AUC before the
  final fit.
- **Incremental & idempotent** — each run only re-fetches, re-engineers, or
  retrains the pieces that are stale (configurable freshness/cooldown
  windows), so re-running the script often is cheap.
- **Human-readable output** — a forecast CSV per pipeline plus a matplotlib
  chart of the rolling bullish-probability signal against a threshold line.

## Tech stack

- Python 3
- [XGBoost](https://xgboost.readthedocs.io/) for the classifier
- [scikit-learn](https://scikit-learn.org/) for time-series cross-validation
  and evaluation metrics
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) for data
  wrangling and feature engineering
- [Matplotlib](https://matplotlib.org/) for the forecast chart
- [Requests](https://requests.readthedocs.io/) against Binance's public
  market-data REST API (no SDK, no key)

## Project layout

```
Stock_Predict/
├── crypto_pipeline.py              # ETH, 30-minute bars
├── data_eth_30m/                   # raw/training/forecast CSVs + chart for the 30m model
├── models_eth_30m/                 # saved XGBoost model (30m)
├── data/, models/                  # output from an earlier (yfinance-based) version of the pipeline; no longer written to
└── Hourly Model/
    ├── crypto_pipeline_hourly.py   # BTC/ETH/SOL/BNB/DOGE, hourly bars
    ├── data_hourly/                # raw/training/forecast CSVs + chart for the hourly model
    └── models_hourly/              # saved XGBoost model (hourly)
```

## Setup

```bash
pip install requests numpy pandas xgboost matplotlib scikit-learn
```

No environment variables or credentials are needed — both pipelines hit
Binance's public, unauthenticated market-data endpoints.

## Run

Each pipeline is a standalone script that runs the full fetch → feature
engineer → train → forecast cycle:

```bash
# Hourly, multi-coin model
python "Stock_Predict/Hourly Model/crypto_pipeline_hourly.py"

# 30-minute ETH model
python Stock_Predict/crypto_pipeline.py
```

Useful flags on both:

- `--force` — redo every stage from scratch (full re-fetch, re-engineer,
  retrain), ignoring the freshness/cooldown checks.
- `--force-train` — retrain immediately, ignoring the retrain cooldown, but
  skip re-fetching/re-engineering if the data is already fresh.

Output lands in that pipeline's `data*/` (forecast CSV + PNG chart) and
`models*/` (saved XGBoost model) directories.
