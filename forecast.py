import argparse
import gc
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser(description="Forecast call volume and tickets received using Chronos via GluonTS.")
    parser.add_argument(
        "--target_date",
        type=str,
        required=True,
        help="Target date for future forecasting (e.g., 2026-04-30). Must be greater than the last date in data.csv."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="amazon/chronos-t5-mini",
        help="Hugging Face model id or local model directory path."
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default=None,
        help="Optional Hugging Face endpoint, e.g. https://hf-mirror.com"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model from local cache/path only, without network access."
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=None,
        help="Optional fixed context length. If omitted, the script will search the best context length via rolling backtests."
    )
    parser.add_argument(
        "--backtest_horizon",
        type=int,
        default=14,
        help="Forecast horizon used during backtesting. Increase this to better match long-range forecasting targets."
    )
    parser.add_argument(
        "--rolling_windows",
        type=int,
        default=4,
        help="Number of rolling backtest windows used to evaluate each series."
    )
    parser.add_argument(
        "--context_candidates",
        type=str,
        default="90,180,365,512,730",
        help="Comma-separated context lengths to evaluate when --context_length is not provided."
    )
    return parser.parse_args()

def load_and_preprocess_data(filepath="data.csv"):
    df = pd.read_csv(filepath)
    # Remove accidental empty columns exported by spreadsheets (e.g., "Unnamed: 4").
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]

    required_columns = ['date', 'call_volume', 'tickets_received']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {filepath}: {missing_columns}. "
            "Expected columns: ['date', 'call_volume', 'tickets_received']."
        )

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Ensure regular frequency 'D' without interpolating 0s
    df = df.set_index('date').asfreq('D').reset_index()

    # Only check/fill forecast-relevant columns.
    forecast_columns = [col for col in ['call_volume', 'tickets_received'] if col in df.columns]
    if forecast_columns and df[forecast_columns].isna().any().any():
        nan_counts = df[forecast_columns].isna().sum().to_dict()
        print(f"Warning: Missing values detected in forecast columns {nan_counts}. Filling with 0.")
        df[forecast_columns] = df[forecast_columns].fillna(0)

    return df

def calculate_metrics(y_true, y_pred):
    # Using Symmetric Mean Absolute Percentage Error (sMAPE)
    # This natively handles true 0s better than standard MAPE which goes to inf.
    # sMAPE formula: 100/n * sum( |y_true - y_pred| / ((|y_true| + |y_pred|)/2) )
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0

    # Avoid division by zero where both true and pred are exactly 0
    diff = np.abs(y_true - y_pred)
    smape_array = np.zeros_like(diff, dtype=float)
    nonzero_mask = denominator > 0
    smape_array[nonzero_mask] = diff[nonzero_mask] / denominator[nonzero_mask]

    smape = np.mean(smape_array)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return smape, rmse


def parse_context_candidates(raw_candidates):
    candidates = []
    for item in raw_candidates.split(','):
        value = item.strip()
        if not value:
            continue
        candidate = int(value)
        if candidate <= 0:
            raise ValueError("Context candidates must be positive integers.")
        candidates.append(candidate)

    if not candidates:
        raise ValueError("At least one context candidate is required.")

    return sorted(set(candidates))


def load_pipeline(model_id, device, local_files_only=False):
    pipeline = BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    return pipeline


def get_model_context_limit(pipeline):
    if isinstance(pipeline, Chronos2Pipeline):
        config = pipeline.model.config.chronos_config
        return int(config["context_length"])
    return int(pipeline.model.config.context_length)


def normalize_context_length(requested_context_length, history_length, model_context_limit):
    return max(1, min(requested_context_length, history_length, model_context_limit))


def predict_quantiles(pipeline, context_values, prediction_length, device, quantiles, num_samples):
    if isinstance(pipeline, Chronos2Pipeline):
        context_tensor = torch.tensor(context_values, dtype=torch.float32).reshape(1, 1, -1).to(device)
        quantile_forecast, _ = pipeline.predict_quantiles(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=quantiles,
        )
    else:
        context_tensor = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0).to(device)
        quantile_forecast, _ = pipeline.predict_quantiles(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=quantiles,
            num_samples=num_samples,
        )

    if isinstance(quantile_forecast, list):
        quantile_np = quantile_forecast[0].cpu().numpy()[0]
    else:
        quantile_np = quantile_forecast.cpu().numpy()[0]

    return [quantile_np[:, index] for index, _ in enumerate(quantiles)]


def build_supervised_features(df, target_col, reference_col=None):
    features = pd.DataFrame(index=df.index)

    date_series = pd.to_datetime(df['date'])
    day_of_week = date_series.dt.dayofweek
    day_of_year = date_series.dt.dayofyear

    # Temporal features
    features['day_of_week'] = day_of_week
    features['is_weekend'] = (day_of_week >= 5).astype(int)
    features['day_of_month'] = date_series.dt.day
    features['month'] = date_series.dt.month
    features['week_of_year'] = date_series.dt.isocalendar().week.astype(int)
    features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    features['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    features['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

    target = df[target_col].astype(float)

    # Lag features
    for lag in range(1, 15):
        features[f'{target_col}_lag_{lag}'] = target.shift(lag)

    # Rolling features
    for window in [3, 7, 14, 30]:
        features[f'{target_col}_roll_mean_{window}'] = target.shift(1).rolling(window).mean()
        features[f'{target_col}_roll_std_{window}'] = target.shift(1).rolling(window).std()

    # Difference features
    features[f'{target_col}_diff_1'] = target.shift(1) - target.shift(2)
    features[f'{target_col}_diff_7'] = target.shift(1) - target.shift(8)

    if reference_col and reference_col in df.columns:
        ref = df[reference_col].astype(float)
        features[f'{reference_col}_lag_1'] = ref.shift(1)
        features[f'{reference_col}_lag_7'] = ref.shift(7)
        features[f'{reference_col}_lag_14'] = ref.shift(14)

        for window in [3, 7, 14, 30]:
            features[f'{reference_col}_roll_mean_{window}'] = ref.shift(1).rolling(window).mean()
            features[f'{reference_col}_roll_std_{window}'] = ref.shift(1).rolling(window).std()

        ratio_raw = ref / np.clip(target, 1e-6, None)
        features['call_to_ticket_ratio_lag1'] = ratio_raw.shift(1)
        features['call_to_ticket_ratio_lag7'] = ratio_raw.shift(7)

    if target_col == 'tickets_received' and 'tickets_resolved' in df.columns:
        resolved = df['tickets_resolved'].astype(float)
        backlog = target - resolved
        features['estimated_backlog_lag1'] = backlog.shift(1)
        features['estimated_backlog_roll_mean_7'] = backlog.shift(1).rolling(7).mean()

    return features


def train_feature_model(train_df, target_col, reference_col=None):
    features = build_supervised_features(train_df, target_col, reference_col=reference_col)
    target = train_df[target_col].astype(float)

    supervised = features.copy()
    supervised[target_col] = target.values
    supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna()

    if len(supervised) < 120:
        raise ValueError(
            f"Not enough samples to train feature model for {target_col}. "
            "Need at least 120 rows after feature construction."
        )

    feature_cols = [col for col in supervised.columns if col != target_col]
    X = supervised[feature_cols]
    y = supervised[target_col]

    selector_scaler = StandardScaler()
    X_scaled = selector_scaler.fit_transform(X)
    selector_estimator = Lasso(alpha=0.001, max_iter=20000)
    selector_estimator.fit(X_scaled, y)

    selected_features = [
        name for name, coef in zip(feature_cols, np.abs(selector_estimator.coef_))
        if coef > 1e-6
    ]

    if len(selected_features) < 8:
        corr_scores = X.corrwith(y).abs().fillna(0.0)
        selected_features = corr_scores.sort_values(ascending=False).head(min(20, len(corr_scores))).index.tolist()

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0)),
    ])
    model.fit(X[selected_features], y)

    return {
        'model': model,
        'selected_features': selected_features,
    }


def recursive_feature_forecast(train_df, target_col, horizon, reference_col=None, reference_future=None):
    if reference_col and reference_future is None:
        raise ValueError(f"reference_future is required when reference_col={reference_col}.")
    if reference_future is not None and len(reference_future) < horizon:
        raise ValueError("reference_future length is shorter than horizon.")

    bundle = train_feature_model(train_df, target_col, reference_col=reference_col)
    model = bundle['model']
    selected_features = bundle['selected_features']

    work_df = train_df.copy().reset_index(drop=True)
    predictions = []

    for step in range(horizon):
        next_date = pd.to_datetime(work_df['date'].iloc[-1]) + pd.Timedelta(days=1)
        next_row = {column: np.nan for column in work_df.columns}
        next_row['date'] = next_date

        if reference_col and reference_col in work_df.columns:
            next_row[reference_col] = float(reference_future[step])

        work_df = pd.concat([work_df, pd.DataFrame([next_row])], ignore_index=True)

        feature_table = build_supervised_features(work_df, target_col, reference_col=reference_col)
        latest_features = feature_table.iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_input = pd.DataFrame([latest_features[selected_features].to_dict()])
        prediction = max(0.0, float(model.predict(x_input)[0]))
        work_df.at[work_df.index[-1], target_col] = prediction
        predictions.append(prediction)

    return np.array(predictions), selected_features


def evaluate_and_forecast_tickets_with_features(
    df,
    target_date_dt,
    backtest_horizon,
    rolling_windows,
    call_future_p50,
):
    print("\n--- Processing tickets_received (feature model with call_volume reference) ---")

    last_date = df['date'].max()
    if target_date_dt <= last_date:
        print(f"Error: Target date {target_date_dt.date()} must be after the last historical date {last_date.date()}.")
        sys.exit(1)

    future_horizon = (target_date_dt - last_date).days
    print(f"Last historical date: {last_date.date()}")
    print(f"Target forecasting date: {target_date_dt.date()}")
    print(f"Future prediction length: {future_horizon} days")

    base_columns = ['date', 'tickets_received', 'call_volume']
    if 'tickets_resolved' in df.columns:
        base_columns.append('tickets_resolved')
    tickets_df = df[base_columns].copy().reset_index(drop=True)

    print(
        f"Starting rolling backtesting for tickets_received with engineered features "
        f"({rolling_windows} windows x {backtest_horizon} days)..."
    )

    window_results = []
    all_residuals = []
    selected_feature_history = []

    for window_index in range(rolling_windows):
        test_end = len(tickets_df) - (window_index * backtest_horizon)
        test_start = test_end - backtest_horizon
        if test_start <= 120:
            break

        train_df = tickets_df.iloc[:test_start].copy()
        test_df = tickets_df.iloc[test_start:test_end].copy()

        # Backtest uses known call_volume in the evaluation window as reference signal.
        reference_future = test_df['call_volume'].to_numpy()
        predicted, selected_features = recursive_feature_forecast(
            train_df,
            target_col='tickets_received',
            horizon=backtest_horizon,
            reference_col='call_volume',
            reference_future=reference_future,
        )

        y_true = test_df['tickets_received'].to_numpy()
        smape, rmse = calculate_metrics(y_true, predicted)
        residuals = y_true - predicted

        window_results.append({
            'window_index': window_index + 1,
            'dates': test_df['date'].to_numpy(),
            'actual': y_true,
            'predicted': predicted,
            'smape': smape,
            'rmse': rmse,
        })
        all_residuals.extend(residuals.tolist())
        selected_feature_history.append(selected_features)

    if not window_results:
        raise ValueError(
            "Not enough history to run tickets feature-model backtests. "
            "Reduce --backtest_horizon or --rolling_windows."
        )

    avg_smape = float(np.mean([row['smape'] for row in window_results]))
    avg_rmse = float(np.mean([row['rmse'] for row in window_results]))
    latest_window = window_results[0]
    latest_window['avg_smape'] = avg_smape
    latest_window['avg_rmse'] = avg_rmse
    latest_window['windows_used'] = len(window_results)
    latest_window['horizon'] = backtest_horizon

    print("Backtesting Metrics for tickets_received:")
    print(f"  Windows used: {latest_window['windows_used']}")
    print(f"  Avg sMAPE: {avg_smape:.2%}")
    print(f"  Avg RMSE: {avg_rmse:.2f}")

    final_train_df = tickets_df.copy()
    p50_future, selected_features = recursive_feature_forecast(
        final_train_df,
        target_col='tickets_received',
        horizon=future_horizon,
        reference_col='call_volume',
        reference_future=call_future_p50,
    )

    residual_std = float(np.std(all_residuals)) if all_residuals else max(1.0, avg_rmse)
    band = 1.2816 * residual_std
    p10_future = np.maximum(0.0, p50_future - band)
    p90_future = np.maximum(p50_future, p50_future + band)

    print(f"  Selected feature count (final training): {len(selected_features)}")

    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_horizon + 1)]
    last_90_df = tickets_df.iloc[-90:]
    future_results = {
        'hist_dates': last_90_df['date'].values,
        'hist_actual': last_90_df['tickets_received'].values,
        'future_dates': future_dates,
        'p10': p10_future,
        'p50': p50_future,
        'p90': p90_future,
        'context_length': None,
    }

    return latest_window, future_results


def run_rolling_backtest(series_data, series_name, pipeline, device, context_length, backtest_horizon, rolling_windows):
    window_results = []

    for window_index in range(rolling_windows):
        test_end = len(series_data) - (window_index * backtest_horizon)
        test_start = test_end - backtest_horizon
        if test_start <= 0:
            break

        train_data = series_data.iloc[:test_start]
        test_data = series_data.iloc[test_start:test_end]
        effective_context_length = min(context_length, len(train_data))
        if effective_context_length <= 0:
            break

        context_values = train_data.iloc[-effective_context_length:][series_name].values
        with torch.no_grad():
            p50_bt = predict_quantiles(
                pipeline,
                context_values,
                backtest_horizon,
                device,
                quantiles=[0.5],
                num_samples=20,
            )[0]

        y_true_bt = test_data[series_name].values
        smape, rmse = calculate_metrics(y_true_bt, p50_bt)
        window_results.append({
            'window_index': window_index + 1,
            'dates': test_data['date'].values,
            'actual': y_true_bt,
            'predicted': p50_bt,
            'smape': smape,
            'rmse': rmse,
            'context_length': effective_context_length,
        })

    if not window_results:
        raise ValueError(
            f"Not enough history to run rolling backtests for {series_name}. "
            f"Reduce --backtest_horizon or --rolling_windows."
        )

    avg_smape = float(np.mean([result['smape'] for result in window_results]))
    avg_rmse = float(np.mean([result['rmse'] for result in window_results]))
    latest_window = window_results[0]
    latest_window['avg_smape'] = avg_smape
    latest_window['avg_rmse'] = avg_rmse
    latest_window['windows_used'] = len(window_results)
    latest_window['horizon'] = backtest_horizon
    return latest_window


def select_best_context_length(series_data, series_name, pipeline, device, candidate_context_lengths, backtest_horizon, rolling_windows, model_context_limit):
    candidate_results = []
    for candidate in candidate_context_lengths:
        normalized_candidate = normalize_context_length(candidate, len(series_data) - backtest_horizon, model_context_limit)
        result = run_rolling_backtest(
            series_data,
            series_name,
            pipeline,
            device,
            normalized_candidate,
            backtest_horizon,
            rolling_windows,
        )
        candidate_results.append((normalized_candidate, result))
        print(
            f"  Context {normalized_candidate}: avg sMAPE={result['avg_smape']:.2%}, "
            f"avg RMSE={result['avg_rmse']:.2f}"
        )

    best_context_length, best_result = min(
        candidate_results,
        key=lambda item: (item[1]['avg_smape'], item[1]['avg_rmse'], item[0])
    )
    return best_context_length, best_result


def evaluate_and_forecast_series(
    df,
    series_name,
    pipeline,
    target_date_dt,
    device,
    backtest_horizon,
    rolling_windows,
    fixed_context_length=None,
    context_candidates=None,
):
    print(f"\n--- Processing {series_name} ---")

    # Calculate horizons
    last_date = df['date'].max()
    if target_date_dt <= last_date:
        print(f"Error: Target date {target_date_dt.date()} must be after the last historical date {last_date.date()}.")
        sys.exit(1)

    future_horizon = (target_date_dt - last_date).days
    model_context_limit = get_model_context_limit(pipeline)

    print(f"Last historical date: {last_date.date()}")
    print(f"Target forecasting date: {target_date_dt.date()}")
    print(f"Future prediction length: {future_horizon} days")
    print(f"Model context limit: {model_context_limit}")

    series_data = df[['date', series_name]].copy()

    # ==========================================
    # BACKTESTING
    # ==========================================
    if fixed_context_length is not None:
        selected_context_length = normalize_context_length(fixed_context_length, len(series_data), model_context_limit)
        print(
            f"Starting rolling backtesting for {series_name} with fixed context_length="
            f"{selected_context_length} ({rolling_windows} windows x {backtest_horizon} days)..."
        )
        bt_results = run_rolling_backtest(
            series_data,
            series_name,
            pipeline,
            device,
            selected_context_length,
            backtest_horizon,
            rolling_windows,
        )
    else:
        print(
            f"Searching best context_length for {series_name} via rolling backtesting "
            f"({rolling_windows} windows x {backtest_horizon} days)..."
        )
        selected_context_length, bt_results = select_best_context_length(
            series_data,
            series_name,
            pipeline,
            device,
            context_candidates,
            backtest_horizon,
            rolling_windows,
            model_context_limit,
        )

    smape = bt_results['avg_smape']
    rmse = bt_results['avg_rmse']

    print(f"Backtesting Metrics for {series_name}:")
    print(f"  Selected context_length: {selected_context_length}")
    print(f"  Windows used: {bt_results['windows_used']}")
    print(f"  Avg sMAPE: {smape:.2%}")
    print(f"  Avg RMSE: {rmse:.2f}")

    # ==========================================
    # FUTURE FORECASTING
    # ==========================================
    print(f"Starting future forecasting for {series_name} (Predicting {future_horizon} days)...")
    # We need up to context_length of the most recent data
    if len(series_data) > selected_context_length:
        future_train_df = series_data.iloc[-selected_context_length:].copy()
    else:
        future_train_df = series_data.copy()

    with torch.no_grad():
        p10_future, p50_future, p90_future = predict_quantiles(
            pipeline,
            future_train_df[series_name].values,
            future_horizon,
            device,
            quantiles=[0.1, 0.5, 0.9],
            num_samples=100,
        )

    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_horizon + 1)]

    # Get last 90 days for plotting context
    last_90_df = series_data.iloc[-90:]

    future_results = {
        'hist_dates': last_90_df['date'].values,
        'hist_actual': last_90_df[series_name].values,
        'future_dates': future_dates,
        'p10': p10_future,
        'p50': p50_future,
        'p90': p90_future,
        'context_length': selected_context_length,
    }

    # Clear memory explicitly
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return bt_results, future_results

def plot_evaluation(bt_results_call, bt_results_ticket):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Call Volume Subplot
    axes[0].plot(bt_results_call['dates'], bt_results_call['actual'], label='Actual', marker='o')
    axes[0].plot(bt_results_call['dates'], bt_results_call['predicted'], label='Predicted (p50)', marker='x')
    axes[0].set_title(
        f"Rolling Backtest ({bt_results_call['windows_used']} windows x {bt_results_call['horizon']} days) - Call Volume"
    )
    axes[0].set_ylabel('Volume')
    axes[0].grid(True)
    axes[0].legend()

    # Tickets Received Subplot
    axes[1].plot(bt_results_ticket['dates'], bt_results_ticket['actual'], label='Actual', marker='o', color='green')
    axes[1].plot(bt_results_ticket['dates'], bt_results_ticket['predicted'], label='Predicted (p50)', marker='x', color='orange')
    axes[1].set_title(
        f"Rolling Backtest ({bt_results_ticket['windows_used']} windows x {bt_results_ticket['horizon']} days) - Tickets Received"
    )
    axes[1].set_ylabel('Tickets')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("Saved evaluation_results.png")
    plt.close()

def plot_future(future_results_call, future_results_ticket):
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    for i, (res, name, color) in enumerate([
        (future_results_call, 'Call Volume', 'blue'),
        (future_results_ticket, 'Tickets Received', 'green')
    ]):
        ax = axes[i]
        # Plot last 90 days
        ax.plot(res['hist_dates'], res['hist_actual'], label='Past 90 Days Actual', color='black')

        # Plot future predictions
        ax.plot(res['future_dates'], res['p50'], label='Predicted (p50)', color=color)

        # Shade p10-p90
        ax.fill_between(
            res['future_dates'],
            res['p10'],
            res['p90'],
            color=color,
            alpha=0.2,
            label='80% Confidence Interval (p10-p90)\n*p90 is the business baseline for scheduling redundancy'
        )

        ax.set_title(f'Future Forecast - {name}')
        ax.set_ylabel('Count')
        ax.grid(True)
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('future_forecast.png')
    print("Saved future_forecast.png")
    plt.close()

def main():
    args = parse_args()

    if args.backtest_horizon <= 0 or args.rolling_windows <= 0:
        print("Error: --backtest_horizon and --rolling_windows must be positive integers.")
        sys.exit(1)

    if args.context_length is not None and args.context_length <= 0:
        print("Error: --context_length must be a positive integer when provided.")
        sys.exit(1)

    try:
        context_candidates = parse_context_candidates(args.context_candidates)
    except ValueError as e:
        print(f"Error: invalid --context_candidates. {e}")
        sys.exit(1)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"Using HF endpoint: {args.hf_endpoint}")

    # Safely parse the date, handling arbitrary formatting errors
    target_date_dt = pd.to_datetime(args.target_date, errors='coerce')
    if pd.isna(target_date_dt):
        print(f"Error: Invalid date format for --target_date '{args.target_date}'. Please use YYYY-MM-DD.")
        sys.exit(1)

    try:
        df = load_and_preprocess_data("data.csv")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Setup device strictly for CPU inference per requirements
    device = "cpu"
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_id} ...")
    try:
        pipeline = load_pipeline(
            args.model_id,
            device=device,
            local_files_only=args.local_files_only,
        )
    except Exception as e:
        print("Error: failed to load model from Hugging Face/local cache.")
        print(f"Detail: {e}")
        print("Hints:")
        print("  1) If your network is restricted, try --hf_endpoint https://hf-mirror.com")
        print("  2) If model already exists in cache/local folder, use --local_files_only")
        print("  3) You can also pass a local directory via --model_id")
        sys.exit(1)

    # 1. Process Call Volume
    bt_call, fut_call = evaluate_and_forecast_series(
        df,
        'call_volume',
        pipeline,
        target_date_dt,
        device,
        args.backtest_horizon,
        args.rolling_windows,
        fixed_context_length=args.context_length,
        context_candidates=context_candidates,
    )

    # 2. Process Tickets Received with engineered features
    bt_ticket, fut_ticket = evaluate_and_forecast_tickets_with_features(
        df,
        target_date_dt,
        args.backtest_horizon,
        args.rolling_windows,
        call_future_p50=fut_call['p50'],
    )

    # 3. Generate Visualizations
    plot_evaluation(bt_call, bt_ticket)
    plot_future(fut_call, fut_ticket)

    print("\nForecasting complete. Check evaluation_results.png and future_forecast.png.")

if __name__ == "__main__":
    main()
