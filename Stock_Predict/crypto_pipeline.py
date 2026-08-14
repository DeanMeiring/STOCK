import os
import time
import argparse
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# ==========================================================================
# Config
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_eth_30m")
MODEL_DIR = os.path.join(BASE_DIR, "models_eth_30m")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "eth_raw_30m.csv")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "eth_training_30m.csv")
FORECAST_CSV_PATH = os.path.join(DATA_DIR, "eth_forecast_30m.csv")
FORECAST_CHART_PATH = os.path.join(DATA_DIR, "eth_forecast_30m.png")
MODEL_PATH = os.path.join(MODEL_DIR, "eth_xgboost_model_30m.json")

# Binance public REST API -- no key needed for market data (klines)
BINANCE_SYMBOL = "ETHUSDT"
BINANCE_INTERVAL = "30m"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_MAX_LIMIT = 1000          # Binance's per-request candle cap
INTERVAL_MINUTES = 30
FETCH_START_DATE = "2019-01-01"   # ETHUSDT has traded on Binance since 2017;
                                   # 2019+ keeps the history clean and plenty long

HORIZON_BARS = 20                 # 20 * 30min = 10 hours ahead (matches the hourly model's horizon)
EMBARGO_BARS = HORIZON_BARS
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
N_CV_SPLITS = 5
RANDOM_STATE = 42

# Same 1.5% / 10h target used in the hourly model -- horizon length is
# unchanged (still 10h), just expressed in more, smaller bars now.
TARGET_THRESHOLD = 0.015

BULLISH_PROB_THRESHOLD = 0.45

# Windows doubled vs the hourly model (14->28, 50->100, 30->60) so each
# indicator still looks back over roughly the same real-world time span,
# since each bar is now half as long (30min vs 1h).
FEATURE_COLUMNS = [
    'sma_28', 'sma_100', 'macd', 'macd_signal', 'rsi_28',
    'bar_return', 'volatility_28b', 'volatility_60b', 'volume_ratio'
]

FETCH_FRESHNESS_MINUTES = 30      # refetch if raw data is older than this
MIN_MINUTES_BETWEEN_RETRAIN = 360 # don't retrain more often than every 6h


# ==========================================================================
# Freshness checks
# ==========================================================================
def _mtime_dt(path):
    return datetime.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else None


def needs_fetch():
    mt = _mtime_dt(RAW_DATA_PATH)
    if mt is None:
        return True
    return (datetime.now() - mt) > timedelta(minutes=FETCH_FRESHNESS_MINUTES)


def needs_feature_engineering():
    if not os.path.exists(TRAINING_DATA_PATH):
        return True
    return os.path.getmtime(TRAINING_DATA_PATH) < os.path.getmtime(RAW_DATA_PATH)


def needs_train():
    model_mt = _mtime_dt(MODEL_PATH)
    if model_mt is None:
        return True
    data_is_newer = os.path.getmtime(MODEL_PATH) < os.path.getmtime(TRAINING_DATA_PATH)
    if not data_is_newer:
        return False
    return (datetime.now() - model_mt) >= timedelta(minutes=MIN_MINUTES_BETWEEN_RETRAIN)


# ==========================================================================
# Stage 1: Fetch (Binance klines, paginated -- no 60-day cap like yfinance)
# ==========================================================================
def fetch_binance_klines(symbol, interval, start_time_ms, end_time_ms=None):
    """Pull ALL klines from start_time_ms to now (or end_time_ms), paginating
    forward in chunks of up to 1000 candles at a time."""
    all_rows = []
    cursor = start_time_ms
    session = requests.Session()

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "limit": BINANCE_MAX_LIMIT,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        resp = session.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        all_rows.extend(batch)

        last_open_time = batch[-1][0]
        next_cursor = last_open_time + (INTERVAL_MINUTES * 60 * 1000)

        if len(batch) < BINANCE_MAX_LIMIT:
            break
        if next_cursor <= cursor:
            break
        cursor = next_cursor

        time.sleep(0.25)  # be polite to Binance's rate limits

    return all_rows


def fetch_data(force=False):
    if not force and not needs_fetch():
        print(f"[1/4] Fetch: raw 30m data is under {FETCH_FRESHNESS_MINUTES}min old, skipping.")
        return

    start_dt = datetime.strptime(FETCH_START_DATE, "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)

    print(f"[1/4] Fetch: downloading {BINANCE_SYMBOL} {BINANCE_INTERVAL} klines from {FETCH_START_DATE} to now...")
    raw = fetch_binance_klines(BINANCE_SYMBOL, BINANCE_INTERVAL, start_ms)

    if not raw:
        raise RuntimeError("Fetch failed -- no data returned from Binance. Check your internet connection.")

    df = pd.DataFrame(raw, columns=[
        "OpenTime", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"
    ])
    df["Date"] = pd.to_datetime(df["OpenTime"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    df["Ticker"] = "ETH-USD"  # kept as ETH-USD for consistency with your other pipelines

    df = df[["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]]
    df = df.drop_duplicates(subset="Date").sort_values("Date").reset_index(drop=True)

    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"  Saved {len(df)} rows ({df['Date'].min()} -> {df['Date'].max()}) to '{RAW_DATA_PATH}'")


# ==========================================================================
# Stage 2: Feature engineering
# ==========================================================================
def engineer_features(force=False):
    if not force and not needs_feature_engineering():
        print("[2/4] Features: already up to date with the latest raw data, skipping.")
        return

    print("[2/4] Features: computing indicators and labels...")
    df = pd.read_csv(RAW_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['sma_28'] = df['Close'].rolling(window=28).mean()
    df['sma_100'] = df['Close'].rolling(window=100).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=28).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=28).mean()
    rs = gain / (loss + 1e-9)
    df['rsi_28'] = 100 - (100 / (1 + rs))

    df['bar_return'] = df['Close'].pct_change()
    df['volatility_28b'] = df['bar_return'].rolling(window=28).std()
    df['volatility_60b'] = df['bar_return'].rolling(window=60).std()

    df['volume_sma_28'] = df['Volume'].rolling(window=28).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_sma_28'] + 1e-9)

    df['future_return'] = (df['Close'].shift(-HORIZON_BARS) - df['Close']) / df['Close']
    df['target'] = (df['future_return'] > TARGET_THRESHOLD).astype(float)

    df['Ticker'] = 'ETH-USD'
    df = df.dropna(subset=FEATURE_COLUMNS)
    keep_cols = ['Date', 'Ticker', 'Close', 'Volume'] + FEATURE_COLUMNS + ['target']
    final_df = df[keep_cols]
    final_df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"  Saved {len(final_df)} labeled 30m rows to '{TRAINING_DATA_PATH}'")


# ==========================================================================
# Stage 3: Train (chronological split + embargo, same leak-fix as before)
# ==========================================================================
def train_model(force=False):
    if not force and not needs_train():
        print(f"[3/4] Train: model retrained within the last {MIN_MINUTES_BETWEEN_RETRAIN}min "
              f"and/or data hasn't changed, skipping.")
        return

    print("[3/4] Train: fitting XGBoost on a chronological split...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"  {len(df)} rows, positive rate {df['target'].mean():.3%}")

    timestamps = np.sort(df['Date'].unique())
    n_ts = len(timestamps)
    test_start_idx = int(n_ts * (1 - TEST_FRACTION))
    val_start_idx = int(test_start_idx * (1 - VAL_FRACTION))
    test_start_ts = timestamps[test_start_idx]
    val_start_ts = timestamps[val_start_idx]
    embargo = pd.Timedelta(minutes=EMBARGO_BARS * INTERVAL_MINUTES)

    train_df = df[df['Date'] < (val_start_ts - embargo)]
    val_df = df[(df['Date'] >= val_start_ts) & (df['Date'] < (test_start_ts - embargo))]
    test_df = df[df['Date'] >= test_start_ts].copy()

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df['target']
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df['target']
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df['target']

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    n_pos = max(y_train.sum(), 1)
    n_neg = len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / n_pos

    model_params = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='auc',
        early_stopping_rounds=30,
    )

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_aucs = []
    for tr_idx, va_idx in tscv.split(X_train):
        fold_model = xgb.XGBClassifier(**{k: v for k, v in model_params.items() if k != 'early_stopping_rounds'})
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_y = y_train.iloc[va_idx]
        if fold_y.nunique() > 1:
            proba = fold_model.predict_proba(X_train.iloc[va_idx])[:, 1]
            cv_aucs.append(roc_auc_score(fold_y, proba))
    if cv_aucs:
        print(f"  Walk-forward CV AUC: {np.mean(cv_aucs):.4f} (+/- {np.std(cv_aucs):.4f})")

    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"  Holdout accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Holdout precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Holdout recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Holdout F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
    if y_test.nunique() > 1:
        print(f"  Holdout ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    model.save_model(MODEL_PATH)
    print(f"  Saved model to '{MODEL_PATH}'")


# ==========================================================================
# Stage 4: Predict + report (always runs)
# ==========================================================================
def get_current_live_price(fallback_price):
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": BINANCE_SYMBOL}, timeout=10
        )
        resp.raise_for_status()
        price = float(resp.json()["price"])
        if not np.isnan(price):
            return round(price, 2)
    except Exception:
        pass
    return round(float(fallback_price), 2)


def categorize_signal(prob):
    if prob >= BULLISH_PROB_THRESHOLD:
        return f"BULLISH (P >= {BULLISH_PROB_THRESHOLD})"
    elif prob >= (BULLISH_PROB_THRESHOLD - 0.08):
        return "NEUTRAL / WATCHLIST"
    else:
        return "BEARISH / WEAK"


def predict_and_report():
    print("[4/4] Forecast: scoring latest 30m data for ETH...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').tail(20)  # last 20 bars = last ~10h of signal history

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    latest_close = float(df['Close'].iloc[-1])
    live_price = get_current_live_price(latest_close)

    forecast_results = []
    for _, row in df.iterrows():
        input_data = pd.DataFrame([row[FEATURE_COLUMNS]])
        prob = float(model.predict_proba(input_data)[:, 1][0])
        obs_time = row['Date']
        target_time = obs_time + pd.Timedelta(minutes=HORIZON_BARS * INTERVAL_MINUTES)

        forecast_results.append({
            'Ticker': 'ETH-USD',
            'Current_Price': live_price,
            'Observation_Time': obs_time.strftime('%Y-%m-%d %H:%M'),
            'Target_Forecast_Time': target_time.strftime('%Y-%m-%d %H:%M'),
            '10H_Bullish_Probability': round(prob, 4),
            'Signal': categorize_signal(prob)
        })

    df_forecast = pd.DataFrame(forecast_results)
    df_forecast.to_csv(FORECAST_CSV_PATH, index=False)

    print("\n================ LATEST ETH 30m SIGNAL ================")
    last_row = df_forecast.iloc[-1]
    print(f"  ETH-USD    ${last_row['Current_Price']:>10,.2f}  "
          f"P(bullish, 10h)={last_row['10H_Bullish_Probability']:.3f}  -> {last_row['Signal']}  "
          f"(target {last_row['Target_Forecast_Time']})")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_forecast['Target_Forecast_Time'], df_forecast['10H_Bullish_Probability'],
            marker='o', linewidth=2, label=f"ETH-USD (${live_price:,.2f})")
    ax.axhline(y=BULLISH_PROB_THRESHOLD, color='g', linestyle='--', label=f'Bullish Threshold ({BULLISH_PROB_THRESHOLD})')
    ax.set_title('ETH 30-Min Bars: 10-Hour Advance Forecast', fontsize=13, fontweight='bold')
    ax.set_xlabel('Target Forecast Time (+10h)')
    ax.set_ylabel('10-Hour Bullish Probability')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(FORECAST_CHART_PATH, dpi=200)
    plt.close()

    print(f"\nSaved forecast CSV to: '{FORECAST_CSV_PATH}'")
    print(f"Saved chart to:        '{FORECAST_CHART_PATH}'")


# ==========================================================================
# Entry point
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="ETH 30-minute signal pipeline (Binance data, 10-hour horizon)")
    parser.add_argument("--force", action="store_true", help="Redo every step from scratch")
    parser.add_argument("--force-train", action="store_true", help="Retrain now, ignoring the retrain cooldown")
    args = parser.parse_args()

    fetch_data(force=args.force)
    engineer_features(force=args.force)
    train_model(force=args.force or args.force_train)
    predict_and_report()


if __name__ == "__main__":
    main()