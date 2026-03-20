import argparse
import gc
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from chronos import ChronosPipeline
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def parse_args():
    parser = argparse.ArgumentParser(description="Forecast call volume and tickets received using Chronos via GluonTS.")
    parser.add_argument(
        "--target_date",
        type=str,
        required=True,
        help="Target date for future forecasting (e.g., 2026-04-30). Must be greater than the last date in data.csv."
    )
    return parser.parse_args()

def load_and_preprocess_data(filepath="data.csv"):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Ensure regular frequency 'D' without interpolating 0s
    df = df.set_index('date').asfreq('D').reset_index()

    # Check for NaNs that might have been introduced by asfreq if there were missing dates
    # But as per the user, it is strictly continuous.
    # Just to be safe, if there are NaNs, fill with 0
    if df.isna().any().any():
        print("Warning: Missing dates found in historical data. Filling introduced NaNs with 0.")
        df = df.fillna(0)

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

def evaluate_and_forecast_series(df, series_name, pipeline, target_date_dt, device):
    print(f"\n--- Processing {series_name} ---")

    # Calculate horizons
    last_date = df['date'].max()
    if target_date_dt <= last_date:
        print(f"Error: Target date {target_date_dt.date()} must be after the last historical date {last_date.date()}.")
        sys.exit(1)

    future_horizon = (target_date_dt - last_date).days
    backtest_horizon = 14
    context_length = 512

    print(f"Last historical date: {last_date.date()}")
    print(f"Target forecasting date: {target_date_dt.date()}")
    print(f"Future prediction length: {future_horizon} days")

    series_data = df[['date', series_name]].copy()

    # ==========================================
    # BACKTESTING
    # ==========================================
    print(f"Starting backtesting for {series_name} (Last 14 days)...")
    # We need backtest_horizon for testing, and context_length for context
    total_backtest_slice = context_length + backtest_horizon
    if len(series_data) > total_backtest_slice:
        backtest_df = series_data.iloc[-total_backtest_slice:].copy()
    else:
        backtest_df = series_data.copy()

    train_data_bt = backtest_df.iloc[:-backtest_horizon]
    test_data_bt = backtest_df.iloc[-backtest_horizon:]

    # Prepare context for Chronos
    context_tensor_bt = torch.tensor(train_data_bt[series_name].values, dtype=torch.float32).unsqueeze(0).to(device)

    # Predict for backtesting
    # predict() returns shape (batch_size, num_samples, prediction_length)
    with torch.no_grad():
        forecast_bt = pipeline.predict(
            context_tensor_bt,
            prediction_length=backtest_horizon,
            num_samples=20 # adequate for p50
        )

    # Extract median (p50) as point forecast
    forecast_bt_np = forecast_bt.cpu().numpy()[0] # shape (num_samples, prediction_length)
    p50_bt = np.median(forecast_bt_np, axis=0)

    y_true_bt = test_data_bt[series_name].values
    smape, rmse = calculate_metrics(y_true_bt, p50_bt)

    print(f"Backtesting Metrics for {series_name}:")
    print(f"  sMAPE: {smape:.2%}")
    print(f"  RMSE: {rmse:.2f}")

    bt_results = {
        'dates': test_data_bt['date'].values,
        'actual': y_true_bt,
        'predicted': p50_bt
    }

    # ==========================================
    # FUTURE FORECASTING
    # ==========================================
    print(f"Starting future forecasting for {series_name} (Predicting {future_horizon} days)...")
    # We need up to context_length of the most recent data
    if len(series_data) > context_length:
        future_train_df = series_data.iloc[-context_length:].copy()
    else:
        future_train_df = series_data.copy()

    context_tensor_future = torch.tensor(future_train_df[series_name].values, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        forecast_future = pipeline.predict(
            context_tensor_future,
            prediction_length=future_horizon,
            num_samples=100 # higher samples for reliable quantiles
        )

    forecast_future_np = forecast_future.cpu().numpy()[0]

    p10_future = np.percentile(forecast_future_np, 10, axis=0)
    p50_future = np.median(forecast_future_np, axis=0)
    p90_future = np.percentile(forecast_future_np, 90, axis=0)

    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_horizon + 1)]

    # Get last 90 days for plotting context
    last_90_df = series_data.iloc[-90:]

    future_results = {
        'hist_dates': last_90_df['date'].values,
        'hist_actual': last_90_df[series_name].values,
        'future_dates': future_dates,
        'p10': p10_future,
        'p50': p50_future,
        'p90': p90_future
    }

    # Clear memory explicitly
    del context_tensor_bt, forecast_bt, context_tensor_future, forecast_future
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    return bt_results, future_results

def plot_evaluation(bt_results_call, bt_results_ticket):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Call Volume Subplot
    axes[0].plot(bt_results_call['dates'], bt_results_call['actual'], label='Actual', marker='o')
    axes[0].plot(bt_results_call['dates'], bt_results_call['predicted'], label='Predicted (p50)', marker='x')
    axes[0].set_title('Backtesting (Last 14 Days) - Call Volume')
    axes[0].set_ylabel('Volume')
    axes[0].grid(True)
    axes[0].legend()

    # Tickets Received Subplot
    axes[1].plot(bt_results_ticket['dates'], bt_results_ticket['actual'], label='Actual', marker='o', color='green')
    axes[1].plot(bt_results_ticket['dates'], bt_results_ticket['predicted'], label='Predicted (p50)', marker='x', color='orange')
    axes[1].set_title('Backtesting (Last 14 Days) - Tickets Received')
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

    # Safely parse the date, handling arbitrary formatting errors
    target_date_dt = pd.to_datetime(args.target_date, errors='coerce')
    if pd.isna(target_date_dt):
        print(f"Error: Invalid date format for --target_date '{args.target_date}'. Please use YYYY-MM-DD.")
        sys.exit(1)

    df = load_and_preprocess_data("data.csv")

    # Setup device strictly for CPU inference per requirements
    device = "cpu"
    print(f"Using device: {device}")

    print("Loading amazon/chronos-t5-mini model...")
    # Load model. It might take a moment.
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-mini",
        device_map=device,
        dtype=torch.float32,
    )

    # 1. Process Call Volume
    bt_call, fut_call = evaluate_and_forecast_series(df, 'call_volume', pipeline, target_date_dt, device)

    # 2. Process Tickets Received
    bt_ticket, fut_ticket = evaluate_and_forecast_series(df, 'tickets_received', pipeline, target_date_dt, device)

    # 3. Generate Visualizations
    plot_evaluation(bt_call, bt_ticket)
    plot_future(fut_call, fut_ticket)

    print("\nForecasting complete. Check evaluation_results.png and future_forecast.png.")

if __name__ == "__main__":
    main()
