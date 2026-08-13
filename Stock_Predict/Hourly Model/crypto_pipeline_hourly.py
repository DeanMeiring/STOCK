import os
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
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
DATA_DIR = os.path.join(BASE_DIR, "data_hourly")
MODEL_DIR = os.path.join(BASE_DIR, "models_hourly")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "crypto_raw_hourly.csv")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "crypto_training_hourly.csv")
FORECAST_CSV_PATH = os.path.join(DATA_DIR, "crypto_forecast_hourly.csv")
FORECAST_CHART_PATH = os.path.join(DATA_DIR, "crypto_forecast_hourly.png")
MODEL_PATH = os.path.join(MODEL_DIR, "crypto_xgboost_model_hourly.json")

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"]

INTERVAL = "1h"
FETCH_PERIOD = "730d"     # Yahoo's hard cap for 1h-interval data

HORIZON_HOURS = 10        # predict: will price rise X% within the next 10 hours?
EMBARGO_HOURS = HORIZON_HOURS
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
N_CV_SPLITS = 5
RANDOM_STATE = 42

# A 3% move in 2 days (the daily model's target) is a much bigger ask than a
# 3% move in 10 hours. Volatility scales roughly with sqrt(time), and
# 10h / 48h ~= 0.21, sqrt(0.21) ~= 0.46, so 3% * 0.46 ~= 1.4%. Rounded to a
# clean 1.5% starting point -- watch the printed "positive rate" after a run
# and adjust up/down per coin if it looks too rare or too common.
TARGET_THRESHOLDS = {t: 0.015 for t in CRYPTO_TICKERS}

BULLISH_PROB_THRESHOLD = 0.45

FEATURE_COLUMNS = [
    'sma_14', 'sma_50', 'macd', 'macd_signal', 'rsi_14',
    'daily_return', 'volatility_14d', 'volatility_30d', 'volume_ratio'
]
# Note: these still use 14/50/30-PERIOD windows -- now that's 14/50/30 HOURS
# of lookback instead of days, which fits a 10-hour-ahead target much better
# than a multi-week window would. Column names kept as-is so the rest of the
# code (and your daily script) stay easy to compare side by side.

FETCH_FRESHNESS_HOURS = 1        # refetch if raw data is older than this
MIN_HOURS_BETWEEN_RETRAIN = 6    # don't retrain more often than this


# ==========================================================================
# Freshness checks
# ==========================================================================
def _mtime_dt(path):
    return datetime.fromtimestamp(os.path.getmtime(path)) if os.path.exists(path) else None


def needs_fetch():
    mt = _mtime_dt(RAW_DATA_PATH)
    if mt is None:
        return True
    return (datetime.now() - mt) > timedelta(hours=FETCH_FRESHNESS_HOURS)


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
    return (datetime.now() - model_mt) >= timedelta(hours=MIN_HOURS_BETWEEN_RETRAIN)


# ==========================================================================
# Stage 1: Fetch
# ==========================================================================
def fetch_data(force=False):
    if not force and not needs_fetch():
        print(f"[1/4] Fetch: raw hourly data is under {FETCH_FRESHNESS_HOURS}h old, skipping.")
        return

    print(f"[1/4] Fetch: downloading {FETCH_PERIOD} of {INTERVAL} bars for {len(CRYPTO_TICKERS)} coins...")
    all_data = []
    for ticker in CRYPTO_TICKERS:
        print(f"  Downloading {ticker} ({INTERVAL}, {FETCH_PERIOD})...")
        df = yf.download(ticker, period=FETCH_PERIOD, interval=INTERVAL, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: no data returned for {ticker}, skipping it.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        # Intraday intervals come back with an index named 'Datetime', not 'Date'
        if 'Datetime' in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df['Ticker'] = ticker
        all_data.append(df)

    if not all_data:
        raise RuntimeError("Fetch failed for every ticker -- check your internet connection.")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"  Saved {len(combined_df)} hourly rows to '{RAW_DATA_PATH}'")


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
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    processed_groups = []
    for ticker, group in df.groupby('Ticker'):
        group = group.copy()

        group['sma_14'] = group['Close'].rolling(window=14).mean()
        group['sma_50'] = group['Close'].rolling(window=50).mean()
        group['ema_12'] = group['Close'].ewm(span=12, adjust=False).mean()
        group['ema_26'] = group['Close'].ewm(span=26, adjust=False).mean()

        group['macd'] = group['ema_12'] - group['ema_26']
        group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()

        delta = group['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        group['rsi_14'] = 100 - (100 / (1 + rs))

        group['daily_return'] = group['Close'].pct_change()  # hourly return, name kept for consistency
        group['volatility_14d'] = group['daily_return'].rolling(window=14).std()
        group['volatility_30d'] = group['daily_return'].rolling(window=30).std()

        group['volume_sma_14'] = group['Volume'].rolling(window=14).mean()
        group['volume_ratio'] = group['Volume'] / (group['volume_sma_14'] + 1e-9)

        target_return = TARGET_THRESHOLDS.get(ticker, 0.015)
        group['future_return'] = (group['Close'].shift(-HORIZON_HOURS) - group['Close']) / group['Close']
        group['target'] = (group['future_return'] > target_return).astype(float)

        processed_groups.append(group.dropna(subset=FEATURE_COLUMNS))

    final_df = pd.concat(processed_groups, ignore_index=True)
    keep_cols = ['Date', 'Ticker', 'Close', 'Volume'] + FEATURE_COLUMNS + ['target']
    final_df = final_df[keep_cols]
    final_df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"  Saved {len(final_df)} labeled hourly rows to '{TRAINING_DATA_PATH}'")


# ==========================================================================
# Stage 3: Train (chronological split + embargo, same leak-fix as the daily
# script -- just measured in hours instead of days now)
# ==========================================================================
def train_model(force=False):
    if not force and not needs_train():
        print(f"[3/4] Train: model retrained within the last {MIN_HOURS_BETWEEN_RETRAIN}h "
              f"and/or data hasn't changed, skipping.")
        return

    print("[3/4] Train: fitting XGBoost on a chronological split...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Ticker'].isin(CRYPTO_TICKERS)].copy()
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
    embargo = pd.Timedelta(hours=EMBARGO_HOURS)

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
def get_current_live_price(ticker, fallback_price):
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.get('lastPrice')
        if price is not None and not np.isnan(price):
            return round(float(price), 2)
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
    print("[4/4] Forecast: scoring latest hourly data for each coin...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    forecast_results = []
    latest_prices = {}

    for ticker in CRYPTO_TICKERS:
        t_df = df[df['Ticker'] == ticker].sort_values('Date').tail(10)
        if t_df.empty:
            continue

        latest_close = float(t_df['Close'].iloc[-1])
        live_price = get_current_live_price(ticker, latest_close)
        latest_prices[ticker] = live_price

        for _, row in t_df.iterrows():
            input_data = pd.DataFrame([row[FEATURE_COLUMNS]])
            prob = float(model.predict_proba(input_data)[:, 1][0])
            obs_time = row['Date']
            target_time = obs_time + pd.Timedelta(hours=HORIZON_HOURS)

            forecast_results.append({
                'Ticker': ticker,
                'Current_Price': live_price,
                'Observation_Time': obs_time.strftime('%Y-%m-%d %H:%M'),
                'Target_Forecast_Time': target_time.strftime('%Y-%m-%d %H:%M'),
                f'{HORIZON_HOURS}H_Bullish_Probability': round(prob, 4),
                'Signal': categorize_signal(prob)
            })

    df_forecast = pd.DataFrame(forecast_results)
    df_forecast.to_csv(FORECAST_CSV_PATH, index=False)

    prob_col = f'{HORIZON_HOURS}H_Bullish_Probability'

    print(f"\n================ LATEST {HORIZON_HOURS}-HOUR SIGNAL PER COIN ================")
    latest_rows = df_forecast.sort_values('Observation_Time').groupby('Ticker').tail(1)
    for _, row in latest_rows.iterrows():
        print(f"  {row['Ticker']:<10} ${row['Current_Price']:>10,.2f}  "
              f"P(bullish, {HORIZON_HOURS}h)={row[prob_col]:.3f}  -> {row['Signal']}  "
              f"(target {row['Target_Forecast_Time']})")

    fig, ax = plt.subplots(figsize=(11, 6))
    for ticker in CRYPTO_TICKERS:
        t_data = df_forecast[df_forecast['Ticker'] == ticker]
        if t_data.empty:
            continue
        lbl = f"{ticker} (${latest_prices.get(ticker, 0):,.2f})"
        ax.plot(t_data['Target_Forecast_Time'], t_data[prob_col], marker='o', linewidth=2, label=lbl)

    ax.axhline(y=BULLISH_PROB_THRESHOLD, color='g', linestyle='--', label=f'Bullish Threshold ({BULLISH_PROB_THRESHOLD})')
    ax.set_title(f'Crypto {HORIZON_HOURS}-Hour Advance Forecast', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Target Forecast Time (+{HORIZON_HOURS}h)')
    ax.set_ylabel(f'{HORIZON_HOURS}-Hour Bullish Probability')
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
    parser = argparse.ArgumentParser(description="Hourly crypto signal pipeline (10-hour horizon)")
    parser.add_argument("--force", action="store_true", help="Redo every step from scratch")
    parser.add_argument("--force-train", action="store_true", help="Retrain now, ignoring the retrain cooldown")
    args = parser.parse_args()

    fetch_data(force=args.force)
    engineer_features(force=args.force)
    train_model(force=args.force or args.force_train)
    predict_and_report()


if __name__ == "__main__":
    main()