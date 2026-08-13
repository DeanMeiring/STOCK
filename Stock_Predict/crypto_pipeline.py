import os
import argparse
from datetime import date, datetime

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
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "crypto_raw.csv")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "crypto_training.csv")
FORECAST_CSV_PATH = os.path.join(DATA_DIR, "crypto_forecast.csv")
FORECAST_CHART_PATH = os.path.join(DATA_DIR, "crypto_forecast.png")
MODEL_PATH = os.path.join(MODEL_DIR, "crypto_xgboost_model.json")

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "DOGE-USD"]

# 2-day-ahead bullish threshold per coin -- tune individually if you want,
# defaults to 3% for everything.
TARGET_THRESHOLDS = {t: 0.03 for t in CRYPTO_TICKERS}

# Signal thresholds used only for the human-readable forecast label
BULLISH_PROB_THRESHOLD = 0.45

FEATURE_COLUMNS = [
    'sma_14', 'sma_50', 'macd', 'macd_signal', 'rsi_14',
    'daily_return', 'volatility_14d', 'volatility_30d', 'volume_ratio'
]

HORIZON_DAYS = 2
EMBARGO_DAYS = HORIZON_DAYS
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
N_CV_SPLITS = 5
RANDOM_STATE = 42


# ==========================================================================
# Freshness checks (this is the "smart, skip what doesn't need to rerun" bit)
# ==========================================================================
def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else None


def needs_fetch():
    if not os.path.exists(RAW_DATA_PATH):
        return True
    fetched_on = datetime.fromtimestamp(_mtime(RAW_DATA_PATH)).date()
    return fetched_on != date.today()


def needs_feature_engineering():
    if not os.path.exists(TRAINING_DATA_PATH):
        return True
    return _mtime(TRAINING_DATA_PATH) < _mtime(RAW_DATA_PATH)


def needs_train():
    if not os.path.exists(MODEL_PATH):
        return True
    return _mtime(MODEL_PATH) < _mtime(TRAINING_DATA_PATH)


# ==========================================================================
# Stage 1: Fetch
# ==========================================================================
def fetch_data(force=False):
    if not force and not needs_fetch():
        print("[1/4] Fetch: raw data already downloaded today, skipping.")
        return

    print(f"[1/4] Fetch: downloading max available history for {len(CRYPTO_TICKERS)} coins...")
    all_data = []
    for ticker in CRYPTO_TICKERS:
        print(f"  Downloading {ticker} (period=max)...")
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: no data returned for {ticker}, skipping it.")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df['Ticker'] = ticker
        all_data.append(df)

    if not all_data:
        raise RuntimeError("Fetch failed for every ticker -- check your internet connection.")

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"  Saved {len(combined_df)} rows to '{RAW_DATA_PATH}'")


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

        group['daily_return'] = group['Close'].pct_change()
        group['volatility_14d'] = group['daily_return'].rolling(window=14).std()
        group['volatility_30d'] = group['daily_return'].rolling(window=30).std()

        group['volume_sma_14'] = group['Volume'].rolling(window=14).mean()
        group['volume_ratio'] = group['Volume'] / (group['volume_sma_14'] + 1e-9)

        target_return = TARGET_THRESHOLDS.get(ticker, 0.03)
        group['future_2d_return'] = (group['Close'].shift(-HORIZON_DAYS) - group['Close']) / group['Close']
        group['target'] = (group['future_2d_return'] > target_return).astype(float)

        processed_groups.append(group.dropna(subset=FEATURE_COLUMNS))

    final_df = pd.concat(processed_groups, ignore_index=True)
    keep_cols = ['Date', 'Ticker', 'Close', 'Volume'] + FEATURE_COLUMNS + ['target']
    final_df = final_df[keep_cols]
    final_df.to_csv(TRAINING_DATA_PATH, index=False)
    print(f"  Saved {len(final_df)} labeled rows to '{TRAINING_DATA_PATH}'")


# ==========================================================================
# Stage 3: Train (chronological split + embargo -- see earlier explanation:
# random train_test_split on time series leaks future info into training)
# ==========================================================================
def train_model(force=False):
    if not force and not needs_train():
        print("[3/4] Train: model already up to date with the latest features, skipping.")
        return

    print("[3/4] Train: fitting XGBoost on a chronological split...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Ticker'].isin(CRYPTO_TICKERS)].copy()
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"  {len(df)} rows, positive rate {df['target'].mean():.3%}")

    dates = np.sort(df['Date'].unique())
    n_dates = len(dates)
    test_start_idx = int(n_dates * (1 - TEST_FRACTION))
    val_start_idx = int(test_start_idx * (1 - VAL_FRACTION))
    test_start_date = dates[test_start_idx]
    val_start_date = dates[val_start_idx]
    embargo = pd.Timedelta(days=EMBARGO_DAYS)

    train_df = df[df['Date'] < (val_start_date - embargo)]
    val_df = df[(df['Date'] >= val_start_date) & (df['Date'] < (test_start_date - embargo))]
    test_df = df[df['Date'] >= test_start_date].copy()

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

    # Walk-forward CV sanity check on the training portion
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

    # Final fit
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {model.best_iteration}")

    # Holdout evaluation
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
    print("[4/4] Forecast: scoring latest data for each coin...")
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
            obs_date = row['Date']
            target_date = (obs_date + pd.Timedelta(days=HORIZON_DAYS)).strftime('%Y-%m-%d')

            forecast_results.append({
                'Ticker': ticker,
                'Current_Price': live_price,
                'Observation_Date': obs_date.strftime('%Y-%m-%d'),
                'Target_Forecast_Date': target_date,
                '2D_Bullish_Probability': round(prob, 4),
                'Signal': categorize_signal(prob)
            })

    df_forecast = pd.DataFrame(forecast_results)
    df_forecast.to_csv(FORECAST_CSV_PATH, index=False)

    # Console summary: latest signal per coin
    print("\n================ LATEST SIGNAL PER COIN ================")
    latest_rows = df_forecast.sort_values('Observation_Date').groupby('Ticker').tail(1)
    for _, row in latest_rows.iterrows():
        print(f"  {row['Ticker']:<10} ${row['Current_Price']:>10,.2f}  "
              f"P(bullish, 2d)={row['2D_Bullish_Probability']:.3f}  -> {row['Signal']}  "
              f"(target {row['Target_Forecast_Date']})")

    # Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for ticker in CRYPTO_TICKERS:
        t_data = df_forecast[df_forecast['Ticker'] == ticker]
        if t_data.empty:
            continue
        lbl = f"{ticker} (${latest_prices.get(ticker, 0):,.2f})"
        ax.plot(t_data['Target_Forecast_Date'], t_data['2D_Bullish_Probability'], marker='o', linewidth=2, label=lbl)

    ax.axhline(y=BULLISH_PROB_THRESHOLD, color='g', linestyle='--', label=f'Bullish Threshold ({BULLISH_PROB_THRESHOLD})')
    ax.set_title('Crypto 2-Day Advance Forecast', fontsize=13, fontweight='bold')
    ax.set_xlabel('Target Forecast Date (+2 days)')
    ax.set_ylabel('2-Day Bullish Probability')
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
    parser = argparse.ArgumentParser(description="One-shot crypto signal pipeline")
    parser.add_argument("--force", action="store_true", help="Redo every step from scratch")
    parser.add_argument("--force-train", action="store_true", help="Retrain even if features haven't changed")
    args = parser.parse_args()

    fetch_data(force=args.force)
    engineer_features(force=args.force)
    train_model(force=args.force or args.force_train)
    predict_and_report()


if __name__ == "__main__":
    main()