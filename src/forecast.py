import argparse
import gc
import importlib
import os
import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from src.feature_engineering import forecast_direct_multistep
from src.output_manager import ensure_output_dirs, export_dataframe_csv, export_forecast_csv
from src.visualization import plot_evaluation, plot_forecast_export_csv, plot_future_from_csv

try:
    calendar_module_name = "chinese" + "_calendar"
    cn_calendar = importlib.import_module(calendar_module_name)
    HAS_CN_CALENDAR = True
except Exception:
    cn_calendar = None
    HAS_CN_CALENDAR = False

warnings.filterwarnings(
    "ignore",
    message=r"Glyph .* missing from font\(s\) DejaVu Sans",
    category=UserWarning,
)

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
        default=30,
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
        default="auto",
        help="Comma-separated context lengths or 'auto' to generate candidates automatically when --context_length is not provided."
    )
    parser.add_argument(
        "--context_search_points",
        type=int,
        default=50,
        help="Number of context candidates to search when --context_candidates=auto."
    )
    parser.add_argument(
        "--residual_weight_search_points",
        type=int,
        default=21,
        help="Number of weight candidates in [0,1] when searching the best residual correction weight."
    )
    parser.add_argument(
        "--direct_weight_search_points",
        type=int,
        default=11,
        help="Number of blend weights in [0,1] when searching the best Chronos/direct fusion weight for tickets_received."
    )
    parser.add_argument(
        "--context_ensemble_topk",
        type=int,
        default=3,
        help="Use top-K context candidates for weighted ensemble when searching context automatically. Set 1 to disable ensemble."
    )
    parser.add_argument(
        "--interval_coverage",
        type=float,
        default=0.80,
        help="Target empirical coverage for calibrated prediction interval, e.g. 0.8 for p10/p90 style interval."
    )
    parser.add_argument(
        "--disable_interval_calibration",
        action="store_true",
        help="Disable conformal interval post-calibration and keep raw p10/p90 outputs."
    )
    parser.add_argument(
        "--disable_leadwise_correction",
        action="store_true",
        help="Disable lead-wise bias correction on top of residual correction."
    )
    parser.add_argument(
        "--leadwise_weight_search_points",
        type=int,
        default=11,
        help="Number of candidate weights in [0,1] when searching lead-wise correction strength."
    )
    parser.add_argument(
        "--auto_backtest_horizon",
        dest="auto_backtest_horizon",
        action="store_true",
        help="Automatically align backtest horizon with requested forecast horizon."
    )
    parser.add_argument(
        "--no_auto_backtest_horizon",
        dest="auto_backtest_horizon",
        action="store_false",
        help="Disable automatic backtest horizon alignment and use --backtest_horizon directly."
    )
    parser.set_defaults(auto_backtest_horizon=True)
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


def generate_auto_context_candidates(max_context_length, points=50, min_context=30):
    if max_context_length <= 0:
        raise ValueError("max_context_length must be positive.")

    lower = max(1, min_context)
    upper = max(lower, int(max_context_length))

    if upper <= lower:
        return [upper]

    points = max(2, int(points))
    grid = np.linspace(lower, upper, num=points)
    candidates = sorted(set(int(round(value)) for value in grid if value >= 1))

    if upper not in candidates:
        candidates.append(upper)

    return sorted(set(candidates))


def resolve_adaptive_backtest_horizon(requested_horizon, future_horizon, history_length, rolling_windows):
    # Align backtest window with requested forecast horizon while respecting data limits.
    base_horizon = max(int(requested_horizon), int(future_horizon), 7)
    max_windows = max(int(rolling_windows), 1)
    max_allowed = max(1, int(history_length) // (max_windows + 1))
    return max(1, min(base_horizon, max_allowed))


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
    latest_window['all_windows'] = window_results
    return latest_window


def build_residual_adjuster(bt_results):
    residual_records = []
    windows = bt_results.get('all_windows', [])

    if windows:
        for window in windows:
            dates = pd.to_datetime(window['dates'])
            actual = np.asarray(window['actual'], dtype=float)
            predicted = np.asarray(window['predicted'], dtype=float)
            residuals = actual - predicted
            for dt, resid, pred in zip(dates, residuals, predicted):
                residual_records.append((dt, float(resid), float(pred)))
    else:
        dates = pd.to_datetime(bt_results['dates'])
        actual = np.asarray(bt_results['actual'], dtype=float)
        predicted = np.asarray(bt_results['predicted'], dtype=float)
        residuals = actual - predicted
        for dt, resid, pred in zip(dates, residuals, predicted):
            residual_records.append((dt, float(resid), float(pred)))

    if not residual_records:
        return {
            'global_bias': 0.0,
            'state_bias': {},
            'state_dow_bias': {},
            'state_month_phase_bias': {},
            'state_counts': {},
            'weight_state': 0.6,
            'weight_dow': 0.3,
            'weight_month': 0.1,
            'weight_state_model': 0.7,
            'state_models': {},
            'residual_std': 0.0,
        }

    residual_df = pd.DataFrame(residual_records, columns=['date', 'residual', 'predicted'])
    residual_df['dow'] = residual_df['date'].dt.dayofweek
    residual_df['day'] = residual_df['date'].dt.day
    residual_df['state'] = residual_df['date'].apply(get_operational_state)
    residual_df['is_holiday'] = residual_df['date'].apply(lambda d: int(is_china_holiday(d)))
    residual_df['is_makeup_workday'] = residual_df['date'].apply(lambda d: int(is_china_makeup_workday(d)))
    residual_df['is_month_end_settlement'] = (residual_df['day'] >= 26).astype(int)
    residual_df['month_phase'] = np.where(
        residual_df['day'] <= 5,
        'month_start',
        np.where(residual_df['day'] >= 26, 'month_end', 'month_mid')
    )
    residual_df['month_phase_code'] = residual_df['month_phase'].map({
        'month_start': 0,
        'month_mid': 1,
        'month_end': 2,
    }).astype(int)

    global_bias = float(residual_df['residual'].mean())
    residual_std = float(residual_df['residual'].std(ddof=0)) if len(residual_df) > 1 else 0.0

    state_bias = {}
    state_dow_bias = {}
    state_month_phase_bias = {}
    state_counts = {}
    state_models = {}

    def _coerce_int_key(key):
        try:
            return int(key)
        except (TypeError, ValueError):
            return int(float(key))

    for state_name, state_group in residual_df.groupby('state'):
        state_bias[state_name] = float(state_group['residual'].mean())
        state_counts[state_name] = int(len(state_group))
        state_dow_bias[state_name] = {
            _coerce_int_key(key): float(value)
            for key, value in state_group.groupby('dow')['residual'].mean().to_dict().items()
        }
        state_month_phase_bias[state_name] = {
            str(key): float(value) for key, value in state_group.groupby('month_phase')['residual'].mean().to_dict().items()
        }

        feature_cols = [
            'dow',
            'month_phase_code',
            'is_month_end_settlement',
            'is_holiday',
            'is_makeup_workday',
            'predicted',
        ]
        state_clean = state_group[feature_cols + ['residual']].replace([np.inf, -np.inf], np.nan).dropna()
        if len(state_clean) >= 2:
            model = Ridge(alpha=1.0)
            model.fit(state_clean[feature_cols], state_clean['residual'])
            state_models[state_name] = {
                'model': model,
                'feature_cols': feature_cols,
                'fallback_bias': float(state_bias[state_name]),
            }

    return {
        'global_bias': global_bias,
        'state_bias': state_bias,
        'state_dow_bias': state_dow_bias,
        'state_month_phase_bias': state_month_phase_bias,
        'state_counts': state_counts,
        'weight_state': 0.6,
        'weight_dow': 0.3,
        'weight_month': 0.1,
        'weight_state_model': 0.7,
        'state_models': state_models,
        'residual_std': residual_std,
    }


def is_china_holiday(dt):
    dt = pd.to_datetime(dt).date()
    if HAS_CN_CALENDAR and cn_calendar is not None:
        return bool(cn_calendar.is_holiday(dt))
    return dt.weekday() >= 5


def is_china_makeup_workday(dt):
    dt_obj = pd.to_datetime(dt)
    dt_date = dt_obj.date()
    if HAS_CN_CALENDAR and cn_calendar is not None:
        return bool(cn_calendar.is_workday(dt_date) and dt_date.weekday() >= 5)
    return False


def get_month_phase(dt):
    day = pd.to_datetime(dt).day
    if day <= 5:
        return 'month_start'
    if day >= 26:
        return 'month_end'
    return 'month_mid'


def get_operational_state(dt):
    dt = pd.to_datetime(dt)
    if is_china_holiday(dt) or is_china_makeup_workday(dt):
        return 'holiday_or_makeup'
    if dt.dayofweek >= 5:
        return 'weekend'
    return 'workday'


def classify_holiday_blocks(date_series):
    dates = pd.Series(pd.to_datetime(date_series)).reset_index(drop=True)
    holiday_mask = dates.apply(
        lambda dt: int(is_china_holiday(dt) and (not is_china_makeup_workday(dt)))
    ).to_numpy(dtype=int)
    makeup_mask = dates.apply(lambda dt: int(is_china_makeup_workday(dt))).to_numpy(dtype=int)
    weekday_mask = (dates.dt.dayofweek < 5).to_numpy(dtype=int)
    workday_mask = np.maximum(weekday_mask, makeup_mask)

    block_lengths = np.zeros(len(dates), dtype=int)
    block_positions = np.zeros(len(dates), dtype=int)
    pre_holiday_n_day = np.zeros(len(dates), dtype=int)
    post_holiday_workday_n = np.zeros(len(dates), dtype=int)

    start = 0
    while start < len(dates):
        if holiday_mask[start] != 1:
            start += 1
            continue

        end = start
        while end + 1 < len(dates) and holiday_mask[end + 1] == 1:
            end += 1

        block_len = end - start + 1
        block_lengths[start:end + 1] = block_len
        block_positions[start:end + 1] = np.arange(1, block_len + 1)

        for n in range(1, 3):
            idx = start - n
            if idx >= 0 and holiday_mask[idx] != 1:
                pre_holiday_n_day[idx] = n

        found = 0
        probe = end + 1
        while probe < len(dates) and found < 3:
            if holiday_mask[probe] != 1 and workday_mask[probe] == 1:
                found += 1
                post_holiday_workday_n[probe] = found
            probe += 1

        start = end + 1

    return {
        'is_holiday_non_makeup': holiday_mask,
        'block_length': block_lengths,
        'block_position': block_positions,
        'pre_holiday_n_day': pre_holiday_n_day,
        'post_holiday_workday_n': post_holiday_workday_n,
    }


def _build_anchor_stats(values, default_strength, strong_strength=None):
    if values.empty:
        return {
            'enabled': False,
            'median': 0.0,
            'p25': 0.0,
            'p75': 0.0,
            'zero_ratio': 0.0,
            'strength': 0.0,
            'count': 0,
        }

    values = values.astype(float)
    zero_ratio = float(np.mean(values <= 1.0))
    strength = strong_strength if (strong_strength is not None and zero_ratio >= 0.5) else default_strength
    return {
        'enabled': True,
        'median': float(max(0.0, np.quantile(values, 0.5))),
        'p25': float(max(0.0, np.quantile(values, 0.25))),
        'p75': float(max(0.0, np.quantile(values, 0.75))),
        'zero_ratio': zero_ratio,
        'strength': float(np.clip(strength, 0.0, 1.0)),
        'count': int(values.shape[0]),
    }


def estimate_holiday_zero_anchor(series_data, series_name):
    hist = series_data[['date', series_name]].copy()
    hist['date'] = pd.to_datetime(hist['date'])
    holiday_block_info = classify_holiday_blocks(hist['date'])
    hist['is_holiday_non_makeup'] = holiday_block_info['is_holiday_non_makeup']
    hist['holiday_block_length'] = holiday_block_info['block_length']
    hist['holiday_block_position'] = holiday_block_info['block_position']
    holiday_values = hist.loc[hist['is_holiday_non_makeup'] == 1, series_name].astype(float)

    if holiday_values.empty:
        return {
            'enabled': False,
            'median': 0.0,
            'p90': 0.0,
            'zero_ratio': 0.0,
            'strength': 0.0,
            'long_holiday_enabled': False,
            'long_holiday_median': 0.0,
            'long_holiday_p75': 0.0,
            'long_holiday_zero_ratio': 0.0,
            'long_holiday_strength': 0.0,
            'proximity_anchors': {},
        }

    median = float(np.quantile(holiday_values, 0.5))
    p90 = float(np.quantile(holiday_values, 0.9))
    zero_ratio = float(np.mean(holiday_values <= 1.0))
    strength = 0.85 if zero_ratio >= 0.5 else 0.55

    long_holiday_core_mask = (
        (hist['is_holiday_non_makeup'] == 1)
        & (hist['holiday_block_length'] >= 4)
        & (hist['holiday_block_position'] >= 2)
        & (hist['holiday_block_position'] <= 5)
    )
    long_holiday_values = hist.loc[long_holiday_core_mask, series_name].astype(float)
    long_holiday_enabled = not long_holiday_values.empty
    long_holiday_median = float(np.quantile(long_holiday_values, 0.5)) if long_holiday_enabled else median
    long_holiday_p75 = float(np.quantile(long_holiday_values, 0.75)) if long_holiday_enabled else median
    long_holiday_zero_ratio = float(np.mean(long_holiday_values <= 1.0)) if long_holiday_enabled else zero_ratio
    if long_holiday_enabled:
        long_holiday_strength = 0.98 if long_holiday_zero_ratio >= 0.5 else 0.80
    else:
        long_holiday_strength = 0.0

    proximity_anchors = {
        'pre_holiday_1d': _build_anchor_stats(
            hist.loc[holiday_block_info['pre_holiday_n_day'] == 1, series_name],
            default_strength=0.40,
            strong_strength=0.55,
        ),
        'pre_holiday_2d': _build_anchor_stats(
            hist.loc[holiday_block_info['pre_holiday_n_day'] == 2, series_name],
            default_strength=0.30,
            strong_strength=0.45,
        ),
        'post_holiday_workday_1': _build_anchor_stats(
            hist.loc[holiday_block_info['post_holiday_workday_n'] == 1, series_name],
            default_strength=0.45,
            strong_strength=0.60,
        ),
        'post_holiday_workday_2': _build_anchor_stats(
            hist.loc[holiday_block_info['post_holiday_workday_n'] == 2, series_name],
            default_strength=0.35,
            strong_strength=0.50,
        ),
        'post_holiday_workday_3': _build_anchor_stats(
            hist.loc[holiday_block_info['post_holiday_workday_n'] == 3, series_name],
            default_strength=0.25,
            strong_strength=0.40,
        ),
    }

    return {
        'enabled': True,
        'median': max(0.0, median),
        'p90': max(0.0, p90),
        'zero_ratio': zero_ratio,
        'strength': strength,
        'long_holiday_enabled': long_holiday_enabled,
        'long_holiday_median': max(0.0, long_holiday_median),
        'long_holiday_p75': max(0.0, long_holiday_p75),
        'long_holiday_zero_ratio': long_holiday_zero_ratio,
        'long_holiday_strength': long_holiday_strength,
        'proximity_anchors': proximity_anchors,
    }


def _apply_proximity_anchor_to_quantiles(p10_now, p50_now, p90_now, anchor_stats):
    if not bool(anchor_stats.get('enabled', False)):
        return p10_now, p50_now, p90_now

    strength = float(np.clip(anchor_stats.get('strength', 0.0), 0.0, 1.0))
    median = float(max(0.0, anchor_stats.get('median', p50_now)))
    p25 = float(max(0.0, anchor_stats.get('p25', min(p10_now, median))))
    p75 = float(max(median, anchor_stats.get('p75', max(p90_now, median))))

    p50_new = max(0.0, (1.0 - strength) * p50_now + strength * median)
    p10_target = min(p50_new, max(0.0, p25))
    p90_target = max(p50_new, p75)
    p10_new = max(0.0, min(p50_new, (1.0 - strength) * p10_now + strength * p10_target))
    p90_new = max(p50_new, (1.0 - strength) * p90_now + strength * p90_target)
    return p10_new, p50_new, p90_new


def format_proximity_anchor_summary(proximity_anchors):
    summary_parts = []
    for anchor_name in [
        'pre_holiday_1d',
        'pre_holiday_2d',
        'post_holiday_workday_1',
        'post_holiday_workday_2',
        'post_holiday_workday_3',
    ]:
        anchor_stats = proximity_anchors.get(anchor_name, {})
        if not bool(anchor_stats.get('enabled', False)):
            continue
        summary_parts.append(
            f"{anchor_name}:n={int(anchor_stats.get('count', 0))},"
            f"median={float(anchor_stats.get('median', 0.0)):.2f},"
            f"strength={float(anchor_stats.get('strength', 0.0)):.2f}"
        )
    return ' | '.join(summary_parts)


def apply_holiday_zero_adjustment(future_results, holiday_anchor):
    if not bool(holiday_anchor.get('enabled', False)):
        return future_results

    base_strength = float(np.clip(holiday_anchor.get('strength', 0.0), 0.0, 1.0))
    base_median = float(max(0.0, holiday_anchor.get('median', 0.0)))
    base_p90_cap = float(max(base_median, holiday_anchor.get('p90', base_median)))

    adjusted = future_results.copy()
    p10_adj = np.asarray(adjusted['p10'], dtype=float).copy()
    p50_adj = np.asarray(adjusted['p50'], dtype=float).copy()
    p90_adj = np.asarray(adjusted['p90'], dtype=float).copy()
    holiday_block_info = classify_holiday_blocks(adjusted['future_dates'])
    future_block_lengths = holiday_block_info['block_length']
    future_block_positions = holiday_block_info['block_position']
    future_pre_holiday = holiday_block_info['pre_holiday_n_day']
    future_post_holiday = holiday_block_info['post_holiday_workday_n']
    proximity_anchors = holiday_anchor.get('proximity_anchors', {})

    for idx, dt in enumerate(adjusted['future_dates']):
        p50_now = float(p50_adj[idx])
        p10_now = float(p10_adj[idx])
        p90_now = float(p90_adj[idx])

        if is_china_holiday(dt) and (not is_china_makeup_workday(dt)):
            use_long_holiday_anchor = bool(
                holiday_anchor.get('long_holiday_enabled', False)
                and future_block_lengths[idx] >= 4
                and 2 <= future_block_positions[idx] <= 5
            )
            if use_long_holiday_anchor:
                strength = float(np.clip(holiday_anchor.get('long_holiday_strength', base_strength), 0.0, 1.0))
                median = float(max(0.0, holiday_anchor.get('long_holiday_median', base_median)))
                p90_cap = float(max(median, holiday_anchor.get('long_holiday_p75', median)))
            else:
                strength = base_strength
                median = base_median
                p90_cap = base_p90_cap

            p50_target = (1.0 - strength) * p50_now + strength * median
            p50_new = max(0.0, min(p50_now, p50_target))
            p10_new = max(0.0, min(p50_new, (1.0 - strength) * p10_now + strength * (0.6 * median)))
            p90_new = max(p50_new, min(p90_now, (1.0 - strength) * p90_now + strength * p90_cap))
        elif future_pre_holiday[idx] == 1:
            p10_new, p50_new, p90_new = _apply_proximity_anchor_to_quantiles(
                p10_now,
                p50_now,
                p90_now,
                proximity_anchors.get('pre_holiday_1d', {}),
            )
        elif future_pre_holiday[idx] == 2:
            p10_new, p50_new, p90_new = _apply_proximity_anchor_to_quantiles(
                p10_now,
                p50_now,
                p90_now,
                proximity_anchors.get('pre_holiday_2d', {}),
            )
        elif future_post_holiday[idx] == 1:
            p10_new, p50_new, p90_new = _apply_proximity_anchor_to_quantiles(
                p10_now,
                p50_now,
                p90_now,
                proximity_anchors.get('post_holiday_workday_1', {}),
            )
        elif future_post_holiday[idx] == 2:
            p10_new, p50_new, p90_new = _apply_proximity_anchor_to_quantiles(
                p10_now,
                p50_now,
                p90_now,
                proximity_anchors.get('post_holiday_workday_2', {}),
            )
        elif future_post_holiday[idx] == 3:
            p10_new, p50_new, p90_new = _apply_proximity_anchor_to_quantiles(
                p10_now,
                p50_now,
                p90_now,
                proximity_anchors.get('post_holiday_workday_3', {}),
            )
        else:
            p10_new, p50_new, p90_new = p10_now, p50_now, p90_now

        p10_adj[idx] = p10_new
        p50_adj[idx] = p50_new
        p90_adj[idx] = p90_new

    adjusted['p10'] = p10_adj
    adjusted['p50'] = p50_adj
    adjusted['p90'] = p90_adj
    return adjusted


def apply_holiday_zero_adjustment_to_backtest(bt_results, holiday_anchor):
    if not bool(holiday_anchor.get('enabled', False)):
        return bt_results

    adjusted = dict(bt_results)
    dates = pd.to_datetime(adjusted['dates'])
    predicted = np.asarray(adjusted['predicted'], dtype=float).copy()
    holiday_block_info = classify_holiday_blocks(dates)

    base_strength = float(np.clip(holiday_anchor.get('strength', 0.0), 0.0, 1.0))
    base_median = float(max(0.0, holiday_anchor.get('median', 0.0)))
    proximity_anchors = holiday_anchor.get('proximity_anchors', {})

    for idx, dt in enumerate(dates):
        if holiday_block_info['is_holiday_non_makeup'][idx] == 1:
            use_long_holiday_anchor = bool(
                holiday_anchor.get('long_holiday_enabled', False)
                and holiday_block_info['block_length'][idx] >= 4
                and 2 <= holiday_block_info['block_position'][idx] <= 5
            )
            if use_long_holiday_anchor:
                strength = float(np.clip(holiday_anchor.get('long_holiday_strength', base_strength), 0.0, 1.0))
                median = float(max(0.0, holiday_anchor.get('long_holiday_median', base_median)))
            else:
                strength = base_strength
                median = base_median

            predicted[idx] = max(0.0, min(predicted[idx], (1.0 - strength) * predicted[idx] + strength * median))
        elif holiday_block_info['pre_holiday_n_day'][idx] == 1:
            anchor_stats = proximity_anchors.get('pre_holiday_1d', {})
            if bool(anchor_stats.get('enabled', False)):
                strength = float(np.clip(anchor_stats.get('strength', 0.0), 0.0, 1.0))
                median = float(max(0.0, anchor_stats.get('median', predicted[idx])))
                predicted[idx] = max(0.0, (1.0 - strength) * predicted[idx] + strength * median)
        elif holiday_block_info['pre_holiday_n_day'][idx] == 2:
            anchor_stats = proximity_anchors.get('pre_holiday_2d', {})
            if bool(anchor_stats.get('enabled', False)):
                strength = float(np.clip(anchor_stats.get('strength', 0.0), 0.0, 1.0))
                median = float(max(0.0, anchor_stats.get('median', predicted[idx])))
                predicted[idx] = max(0.0, (1.0 - strength) * predicted[idx] + strength * median)
        elif holiday_block_info['post_holiday_workday_n'][idx] == 1:
            anchor_stats = proximity_anchors.get('post_holiday_workday_1', {})
            if bool(anchor_stats.get('enabled', False)):
                strength = float(np.clip(anchor_stats.get('strength', 0.0), 0.0, 1.0))
                median = float(max(0.0, anchor_stats.get('median', predicted[idx])))
                predicted[idx] = max(0.0, (1.0 - strength) * predicted[idx] + strength * median)
        elif holiday_block_info['post_holiday_workday_n'][idx] == 2:
            anchor_stats = proximity_anchors.get('post_holiday_workday_2', {})
            if bool(anchor_stats.get('enabled', False)):
                strength = float(np.clip(anchor_stats.get('strength', 0.0), 0.0, 1.0))
                median = float(max(0.0, anchor_stats.get('median', predicted[idx])))
                predicted[idx] = max(0.0, (1.0 - strength) * predicted[idx] + strength * median)
        elif holiday_block_info['post_holiday_workday_n'][idx] == 3:
            anchor_stats = proximity_anchors.get('post_holiday_workday_3', {})
            if bool(anchor_stats.get('enabled', False)):
                strength = float(np.clip(anchor_stats.get('strength', 0.0), 0.0, 1.0))
                median = float(max(0.0, anchor_stats.get('median', predicted[idx])))
                predicted[idx] = max(0.0, (1.0 - strength) * predicted[idx] + strength * median)
        else:
            continue

    adjusted['predicted'] = predicted
    return adjusted


def predict_state_model_adjustment(dt, base_pred, residual_adjuster):
    state_models = residual_adjuster.get('state_models', {})
    state_name = get_operational_state(dt)
    state_bundle = state_models.get(state_name)
    if state_bundle is None:
        return None

    dt = pd.to_datetime(dt)
    month_phase = get_month_phase(dt)
    month_phase_code = {'month_start': 0, 'month_mid': 1, 'month_end': 2}.get(month_phase, 1)
    row = {
        'dow': int(dt.dayofweek),
        'month_phase_code': int(month_phase_code),
        'is_month_end_settlement': int(dt.day >= 26),
        'is_holiday': int(is_china_holiday(dt)),
        'is_makeup_workday': int(is_china_makeup_workday(dt)),
        'predicted': float(base_pred),
    }

    feature_cols = state_bundle.get('feature_cols', [])
    if not feature_cols:
        return None

    x = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
    model = state_bundle['model']
    return float(model.predict(x)[0])


def compute_residual_adjustment(dt, residual_adjuster, base_pred=None):
    dt = pd.to_datetime(dt)

    global_bias = float(residual_adjuster.get('global_bias', 0.0))
    state_bias_map = residual_adjuster.get('state_bias', {})
    state_dow_bias = residual_adjuster.get('state_dow_bias', {})
    state_month_phase_bias = residual_adjuster.get('state_month_phase_bias', {})

    weight_state = float(residual_adjuster.get('weight_state', 0.6))
    weight_dow = float(residual_adjuster.get('weight_dow', 0.3))
    weight_month = float(residual_adjuster.get('weight_month', 0.1))

    weights = np.array([weight_state, weight_dow, weight_month], dtype=float)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0:
        weights = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        weights = weights / float(weights.sum())

    state_name = get_operational_state(dt)
    month_phase = get_month_phase(dt)
    state_bias = float(state_bias_map.get(state_name, global_bias))
    dow_resid = float(state_dow_bias.get(state_name, {}).get(dt.dayofweek, state_bias))
    month_phase_resid = float(state_month_phase_bias.get(state_name, {}).get(month_phase, state_bias))

    bias_adjustment = float(
        weights[0] * state_bias +
        weights[1] * dow_resid +
        weights[2] * month_phase_resid
    )

    if base_pred is None:
        return bias_adjustment

    model_adjustment = predict_state_model_adjustment(dt, float(base_pred), residual_adjuster)
    if model_adjustment is None:
        return bias_adjustment

    model_weight = float(np.clip(residual_adjuster.get('weight_state_model', 0.7), 0.0, 1.0))
    mixed_adjustment = model_weight * model_adjustment + (1.0 - model_weight) * bias_adjustment
    clip_std = float(residual_adjuster.get('residual_std', 0.0))
    if clip_std > 0:
        mixed_adjustment = float(np.clip(mixed_adjustment, -3.0 * clip_std, 3.0 * clip_std))
    return mixed_adjustment


def apply_residual_adjustment(future_results, residual_adjuster):
    p10_adj = []
    p50_adj = []
    p90_adj = []

    for idx, dt in enumerate(future_results['future_dates']):
        adjustment = compute_residual_adjustment(
            dt,
            residual_adjuster,
            base_pred=float(future_results['p50'][idx]),
        )

        p10_val = max(0.0, float(future_results['p10'][idx]) + adjustment)
        p50_val = max(0.0, float(future_results['p50'][idx]) + adjustment)
        p90_val = max(p50_val, float(future_results['p90'][idx]) + adjustment)

        p10_adj.append(p10_val)
        p50_adj.append(p50_val)
        p90_adj.append(p90_val)

    adjusted = future_results.copy()
    adjusted['p10'] = np.array(p10_adj)
    adjusted['p50'] = np.array(p50_adj)
    adjusted['p90'] = np.array(p90_adj)
    return adjusted


def build_leadwise_adjuster(bt_results, residual_adjuster):
    windows = bt_results.get('all_windows', [])
    if not windows:
        windows = [bt_results]

    lead_residuals = {}
    all_residuals = []

    for window in windows:
        dates = pd.to_datetime(window['dates'])
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        for idx, (dt, y_true, y_pred) in enumerate(zip(dates, actual, predicted), start=1):
            residual_after_base = float(
                y_true - max(0.0, y_pred + compute_residual_adjustment(dt, residual_adjuster, base_pred=float(y_pred)))
            )
            lead_residuals.setdefault(idx, []).append(residual_after_base)
            all_residuals.append(residual_after_base)

    lead_bias = {int(k): float(np.mean(v)) for k, v in lead_residuals.items() if len(v) > 0}
    global_lead_bias = float(np.mean(all_residuals)) if all_residuals else 0.0

    return {
        'lead_bias': lead_bias,
        'global_lead_bias': global_lead_bias,
        'leadwise_weight': 1.0,
    }


def apply_leadwise_adjustment(future_results, leadwise_adjuster):
    lead_bias = leadwise_adjuster.get('lead_bias', {})
    global_lead_bias = float(leadwise_adjuster.get('global_lead_bias', 0.0))
    leadwise_weight = float(np.clip(leadwise_adjuster.get('leadwise_weight', 1.0), 0.0, 1.0))

    p10_adj = []
    p50_adj = []
    p90_adj = []

    for idx in range(len(future_results['future_dates'])):
        lead = idx + 1
        lead_adjust = float(lead_bias.get(lead, global_lead_bias))
        adjustment = leadwise_weight * lead_adjust

        p10_val = max(0.0, float(future_results['p10'][idx]) + adjustment)
        p50_val = max(0.0, float(future_results['p50'][idx]) + adjustment)
        p90_val = max(p50_val, float(future_results['p90'][idx]) + adjustment)

        p10_adj.append(p10_val)
        p50_adj.append(p50_val)
        p90_adj.append(p90_val)

    adjusted = future_results.copy()
    adjusted['p10'] = np.array(p10_adj)
    adjusted['p50'] = np.array(p50_adj)
    adjusted['p90'] = np.array(p90_adj)
    return adjusted


def evaluate_leadwise_adjustment_effect(bt_results, residual_adjuster, leadwise_adjuster, custom_weight=None):
    windows = bt_results.get('all_windows', [])
    if not windows:
        windows = [bt_results]

    lead_bias = leadwise_adjuster.get('lead_bias', {})
    global_lead_bias = float(leadwise_adjuster.get('global_lead_bias', 0.0))
    leadwise_weight = float(leadwise_adjuster.get('leadwise_weight', 1.0)) if custom_weight is None else float(custom_weight)
    leadwise_weight = float(np.clip(leadwise_weight, 0.0, 1.0))

    smape_before_list = []
    rmse_before_list = []
    smape_after_list = []
    rmse_after_list = []

    for window in windows:
        dates = pd.to_datetime(window['dates'])
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        corrected_pred = []
        for idx, (dt, pred) in enumerate(zip(dates, predicted), start=1):
            residual_adj = compute_residual_adjustment(dt, residual_adjuster, base_pred=float(pred))
            lead_adj = float(lead_bias.get(idx, global_lead_bias))
            final_pred = max(0.0, float(pred) + residual_adj + leadwise_weight * lead_adj)
            corrected_pred.append(final_pred)

        corrected_pred = np.asarray(corrected_pred, dtype=float)
        base_pred = np.asarray(
            [
                max(0.0, float(pred) + compute_residual_adjustment(dt, residual_adjuster, base_pred=float(pred)))
                for dt, pred in zip(dates, predicted)
            ],
            dtype=float,
        )

        smape_before, rmse_before = calculate_metrics(actual, base_pred)
        smape_after, rmse_after = calculate_metrics(actual, corrected_pred)

        smape_before_list.append(smape_before)
        rmse_before_list.append(rmse_before)
        smape_after_list.append(smape_after)
        rmse_after_list.append(rmse_after)

    return {
        'avg_smape_before': float(np.mean(smape_before_list)),
        'avg_rmse_before': float(np.mean(rmse_before_list)),
        'avg_smape_after': float(np.mean(smape_after_list)),
        'avg_rmse_after': float(np.mean(rmse_after_list)),
    }


def search_best_leadwise_weight(bt_results, residual_adjuster, leadwise_adjuster, search_points=11):
    search_points = max(2, int(search_points))
    candidates = np.linspace(0.0, 1.0, num=search_points)

    best_weight = 1.0
    best_effect = None
    best_score = (float('inf'), float('inf'))
    for w in candidates:
        effect = evaluate_leadwise_adjustment_effect(
            bt_results,
            residual_adjuster,
            leadwise_adjuster,
            custom_weight=float(w),
        )
        score = (effect['avg_smape_after'], effect['avg_rmse_after'])
        if best_effect is None or score < best_score:
            best_effect = effect
            best_weight = float(w)
            best_score = score

    if best_effect is None:
        best_effect = evaluate_leadwise_adjustment_effect(
            bt_results,
            residual_adjuster,
            leadwise_adjuster,
            custom_weight=best_weight,
        )

    tuned = dict(leadwise_adjuster)
    tuned['leadwise_weight'] = best_weight
    return tuned, best_effect


def evaluate_residual_adjustment_effect(bt_results, residual_adjuster, custom_weights=None):
    windows = bt_results.get('all_windows', [])
    if not windows:
        windows = [bt_results]

    smape_before_list = []
    rmse_before_list = []
    smape_after_list = []
    rmse_after_list = []

    use_adjuster = dict(residual_adjuster)
    if custom_weights:
        use_adjuster.update(custom_weights)

    for window in windows:
        dates = pd.to_datetime(window['dates'])
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        adjusted_pred = []
        for dt, pred in zip(dates, predicted):
            adjustment = compute_residual_adjustment(dt, use_adjuster, base_pred=float(pred))
            adjusted_pred.append(max(0.0, float(pred) + adjustment))

        adjusted_pred = np.asarray(adjusted_pred, dtype=float)

        smape_before, rmse_before = calculate_metrics(actual, predicted)
        smape_after, rmse_after = calculate_metrics(actual, adjusted_pred)

        smape_before_list.append(smape_before)
        rmse_before_list.append(rmse_before)
        smape_after_list.append(smape_after)
        rmse_after_list.append(rmse_after)

    return {
        'avg_smape_before': float(np.mean(smape_before_list)),
        'avg_rmse_before': float(np.mean(rmse_before_list)),
        'avg_smape_after': float(np.mean(smape_after_list)),
        'avg_rmse_after': float(np.mean(rmse_after_list)),
    }


def collect_adjusted_backtest_residuals(bt_results, residual_adjuster, leadwise_adjuster=None):
    windows = bt_results.get('all_windows', [])
    if not windows:
        windows = [bt_results]

    residuals = []
    for window in windows:
        dates = pd.to_datetime(window['dates'])
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        adjusted_pred = []
        for idx, (dt, pred) in enumerate(zip(dates, predicted), start=1):
            adjustment = compute_residual_adjustment(dt, residual_adjuster, base_pred=float(pred))
            if leadwise_adjuster is not None:
                lead_bias = float(leadwise_adjuster.get('lead_bias', {}).get(idx, leadwise_adjuster.get('global_lead_bias', 0.0)))
                lead_weight = float(np.clip(leadwise_adjuster.get('leadwise_weight', 1.0), 0.0, 1.0))
                adjustment += lead_weight * lead_bias
            adjusted_pred.append(max(0.0, float(pred) + adjustment))

        adjusted_pred = np.asarray(adjusted_pred, dtype=float)
        residuals.extend((actual - adjusted_pred).tolist())

    return np.asarray(residuals, dtype=float)


def calibrate_prediction_intervals(future_results, residuals, coverage=0.80):
    coverage = float(np.clip(coverage, 0.5, 0.99))
    if residuals.size == 0:
        return future_results, {'radius': 0.0, 'empirical_coverage': 0.0}

    abs_residuals = np.abs(residuals)
    radius = float(np.quantile(abs_residuals, coverage))
    empirical_coverage = float(np.mean(abs_residuals <= radius))

    calibrated = future_results.copy()
    p50 = np.asarray(calibrated['p50'], dtype=float)
    p10 = np.maximum(0.0, p50 - radius)
    p90 = np.maximum(p50, p50 + radius)

    calibrated['p10'] = p10
    calibrated['p90'] = p90
    return calibrated, {
        'radius': radius,
        'empirical_coverage': empirical_coverage,
    }


def evaluate_interval_calibration_backtest(bt_results, residual_adjuster, coverage=0.80, leadwise_adjuster=None):
    windows = bt_results.get('all_windows', [])
    if not windows:
        windows = [bt_results]

    coverage = float(np.clip(coverage, 0.5, 0.99))
    window_coverages = []
    window_widths = []
    used_radii = []

    adjusted_windows = []
    for window in windows:
        dates = pd.to_datetime(window['dates'])
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)
        adjusted_pred = []
        for idx, (dt, pred) in enumerate(zip(dates, predicted), start=1):
            adjustment = compute_residual_adjustment(dt, residual_adjuster, base_pred=float(pred))
            if leadwise_adjuster is not None:
                lead_bias = float(leadwise_adjuster.get('lead_bias', {}).get(idx, leadwise_adjuster.get('global_lead_bias', 0.0)))
                lead_weight = float(np.clip(leadwise_adjuster.get('leadwise_weight', 1.0), 0.0, 1.0))
                adjustment += lead_weight * lead_bias
            adjusted_pred.append(max(0.0, float(pred) + adjustment))
        adjusted_pred = np.asarray(adjusted_pred, dtype=float)
        adjusted_windows.append({'actual': actual, 'pred': adjusted_pred})

    for idx, target_window in enumerate(adjusted_windows):
        calib_residuals = []
        for j, other_window in enumerate(adjusted_windows):
            if j == idx:
                continue
            calib_residuals.extend((other_window['actual'] - other_window['pred']).tolist())

        if not calib_residuals:
            calib_residuals = (target_window['actual'] - target_window['pred']).tolist()

        calib_residuals = np.asarray(calib_residuals, dtype=float)
        radius = float(np.quantile(np.abs(calib_residuals), coverage)) if calib_residuals.size > 0 else 0.0

        lower = np.maximum(0.0, target_window['pred'] - radius)
        upper = np.maximum(target_window['pred'], target_window['pred'] + radius)
        within = (target_window['actual'] >= lower) & (target_window['actual'] <= upper)

        window_coverages.append(float(np.mean(within)))
        window_widths.append(float(np.mean(upper - lower)))
        used_radii.append(radius)

    return {
        'avg_coverage': float(np.mean(window_coverages)) if window_coverages else 0.0,
        'avg_width': float(np.mean(window_widths)) if window_widths else 0.0,
        'avg_radius': float(np.mean(used_radii)) if used_radii else 0.0,
    }


def search_best_residual_weight(bt_results, residual_adjuster, search_points=21):
    search_points = max(3, int(search_points))
    levels = min(10, max(4, int(round(search_points ** 0.5)) + 1))
    candidate_grid = np.linspace(0.0, 1.0, num=levels)

    best_weights = {
        'weight_state': 0.6,
        'weight_dow': 0.3,
        'weight_month': 0.1,
    }
    best_effect = None
    best_score = (float('inf'), float('inf'))

    for ws in candidate_grid:
        for wd in candidate_grid:
            wm = 1.0 - float(ws) - float(wd)
            if wm < 0:
                continue

            custom_weights = {
                'weight_state': float(ws),
                'weight_dow': float(wd),
                'weight_month': float(wm),
            }
            effect = evaluate_residual_adjustment_effect(
                bt_results,
                residual_adjuster,
                custom_weights=custom_weights,
            )

            score = (effect['avg_smape_after'], effect['avg_rmse_after'])
            if best_effect is None or score < best_score:
                best_effect = effect
                best_weights = custom_weights
                best_score = score

    if best_effect is None:
        best_effect = evaluate_residual_adjustment_effect(
            bt_results,
            residual_adjuster,
            custom_weights=best_weights,
        )

    tuned_adjuster = dict(residual_adjuster)
    tuned_adjuster.update(best_weights)
    return tuned_adjuster, best_effect


def run_direct_multistep_backtest(df, series_name, backtest_horizon, rolling_windows, reference_col=None):
    window_results = []
    print(
        f"[Direct-Backtest] {series_name}: 开始，窗口数={rolling_windows}，"
        f"每窗天数={backtest_horizon}，reference_col={reference_col if reference_col else 'None'}"
    )

    for window_index in range(rolling_windows):
        test_end = len(df) - (window_index * backtest_horizon)
        test_start = test_end - backtest_horizon
        if test_start <= 0:
            break

        train_df = df.iloc[:test_start].copy()
        test_df = df.iloc[test_start:test_end].copy()
        if train_df.empty or test_df.empty:
            break

        direct_result = forecast_direct_multistep(
            train_df,
            series_name,
            horizon=len(test_df),
            reference_col=reference_col,
            holiday_fn=is_china_holiday,
            makeup_fn=is_china_makeup_workday,
            return_feature_frame=False,
            progress_prefix=f"{series_name} 回测窗{window_index + 1}",
            log_every=7,
            enable_progress_log=True,
        )
        p50_bt = np.asarray(direct_result['predictions'], dtype=float)
        y_true_bt = test_df[series_name].to_numpy(dtype=float)
        smape, rmse = calculate_metrics(y_true_bt, p50_bt)
        window_results.append({
            'window_index': window_index + 1,
            'dates': test_df['date'].values,
            'actual': y_true_bt,
            'predicted': p50_bt,
            'smape': smape,
            'rmse': rmse,
            'context_length': 'direct_multistep',
        })
        print(
            f"[Direct-Backtest] {series_name}: 窗口 {window_index + 1} 完成，"
            f"sMAPE={smape:.2%}，RMSE={rmse:.2f}"
        )

    if not window_results:
        raise ValueError(
            f"Not enough history to run direct multi-step backtests for {series_name}. "
            f"Reduce --backtest_horizon or --rolling_windows."
        )

    avg_smape = float(np.mean([result['smape'] for result in window_results]))
    avg_rmse = float(np.mean([result['rmse'] for result in window_results]))
    latest_window = window_results[0]
    latest_window['avg_smape'] = avg_smape
    latest_window['avg_rmse'] = avg_rmse
    latest_window['windows_used'] = len(window_results)
    latest_window['horizon'] = backtest_horizon
    latest_window['all_windows'] = window_results
    print(
        f"[Direct-Backtest] {series_name}: 全部完成，平均 sMAPE={avg_smape:.2%}，平均 RMSE={avg_rmse:.2f}"
    )
    return latest_window


def evaluate_direct_fusion_effect(chronos_bt_results, direct_bt_results, custom_weight=None):
    fusion_weight = 0.5 if custom_weight is None else float(custom_weight)
    fusion_weight = float(np.clip(fusion_weight, 0.0, 1.0))
    windows_chronos = chronos_bt_results.get('all_windows', [chronos_bt_results])
    windows_direct = direct_bt_results.get('all_windows', [direct_bt_results])

    smape_chronos_list = []
    rmse_chronos_list = []
    smape_fused_list = []
    rmse_fused_list = []

    for chronos_window, direct_window in zip(windows_chronos, windows_direct):
        actual = np.asarray(chronos_window['actual'], dtype=float)
        chronos_pred = np.asarray(chronos_window['predicted'], dtype=float)
        direct_pred = np.asarray(direct_window['predicted'], dtype=float)
        fused_pred = fusion_weight * chronos_pred + (1.0 - fusion_weight) * direct_pred

        smape_chronos, rmse_chronos = calculate_metrics(actual, chronos_pred)
        smape_fused, rmse_fused = calculate_metrics(actual, fused_pred)
        smape_chronos_list.append(smape_chronos)
        rmse_chronos_list.append(rmse_chronos)
        smape_fused_list.append(smape_fused)
        rmse_fused_list.append(rmse_fused)

    return {
        'avg_smape_before': float(np.mean(smape_chronos_list)),
        'avg_rmse_before': float(np.mean(rmse_chronos_list)),
        'avg_smape_after': float(np.mean(smape_fused_list)),
        'avg_rmse_after': float(np.mean(rmse_fused_list)),
    }


def search_best_direct_fusion_weight(chronos_bt_results, direct_bt_results, search_points=11):
    search_points = max(2, int(search_points))
    candidates = np.linspace(0.0, 1.0, num=search_points)

    best_weight = 1.0
    best_effect = None
    best_score = (float('inf'), float('inf'))
    for weight in candidates:
        effect = evaluate_direct_fusion_effect(
            chronos_bt_results,
            direct_bt_results,
            custom_weight=float(weight),
        )
        score = (effect['avg_smape_after'], effect['avg_rmse_after'])
        if best_effect is None or score < best_score:
            best_weight = float(weight)
            best_effect = effect
            best_score = score

    if best_effect is None:
        best_effect = evaluate_direct_fusion_effect(
            chronos_bt_results,
            direct_bt_results,
            custom_weight=best_weight,
        )

    return best_weight, best_effect


def blend_backtest_results(chronos_bt_results, direct_bt_results, fusion_weight):
    fusion_weight = float(np.clip(fusion_weight, 0.0, 1.0))
    windows_chronos = chronos_bt_results.get('all_windows', [chronos_bt_results])
    windows_direct = direct_bt_results.get('all_windows', [direct_bt_results])
    fused_windows = []

    for index, (chronos_window, direct_window) in enumerate(zip(windows_chronos, windows_direct), start=1):
        actual = np.asarray(chronos_window['actual'], dtype=float)
        chronos_pred = np.asarray(chronos_window['predicted'], dtype=float)
        direct_pred = np.asarray(direct_window['predicted'], dtype=float)
        fused_pred = fusion_weight * chronos_pred + (1.0 - fusion_weight) * direct_pred
        smape, rmse = calculate_metrics(actual, fused_pred)
        fused_windows.append({
            'window_index': index,
            'dates': chronos_window['dates'],
            'actual': actual,
            'predicted': fused_pred,
            'smape': smape,
            'rmse': rmse,
            'context_length': 'chronos_direct_fusion',
        })

    avg_smape = float(np.mean([result['smape'] for result in fused_windows]))
    avg_rmse = float(np.mean([result['rmse'] for result in fused_windows]))
    latest_window = fused_windows[0]
    latest_window['avg_smape'] = avg_smape
    latest_window['avg_rmse'] = avg_rmse
    latest_window['windows_used'] = len(fused_windows)
    latest_window['horizon'] = chronos_bt_results['horizon']
    latest_window['all_windows'] = fused_windows
    return latest_window


def blend_future_results(chronos_future_results, direct_predictions, fusion_weight):
    fusion_weight = float(np.clip(fusion_weight, 0.0, 1.0))
    fused = chronos_future_results.copy()
    direct_predictions = np.asarray(direct_predictions, dtype=float)
    chronos_p10 = np.asarray(chronos_future_results['p10'], dtype=float)
    chronos_p50 = np.asarray(chronos_future_results['p50'], dtype=float)
    chronos_p90 = np.asarray(chronos_future_results['p90'], dtype=float)

    fused_p50 = fusion_weight * chronos_p50 + (1.0 - fusion_weight) * direct_predictions
    lower_radius = np.maximum(0.0, chronos_p50 - chronos_p10)
    upper_radius = np.maximum(0.0, chronos_p90 - chronos_p50)

    fused['p50'] = np.maximum(0.0, fused_p50)
    fused['p10'] = np.maximum(0.0, fused['p50'] - lower_radius)
    fused['p90'] = np.maximum(fused['p50'], fused['p50'] + upper_radius)
    return fused


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
            f"  候选 Context {normalized_candidate}: 平均 sMAPE={result['avg_smape']:.2%}, "
            f"平均 RMSE={result['avg_rmse']:.2f}"
        )

    best_context_length, best_result = min(
        candidate_results,
        key=lambda item: (item[1]['avg_smape'], item[1]['avg_rmse'], item[0])
    )
    return best_context_length, best_result, candidate_results


def select_context_ensemble(candidate_results, topk=3):
    ranked = sorted(
        candidate_results,
        key=lambda item: (item[1]['avg_smape'], item[1]['avg_rmse'], item[0])
    )
    topk = max(1, min(int(topk), len(ranked)))
    selected = ranked[:topk]

    raw_weights = []
    for _, result in selected:
        raw_weights.append(1.0 / max(1e-6, float(result['avg_smape'])))

    weight_sum = float(np.sum(raw_weights))
    if weight_sum <= 0:
        weights = [1.0 / topk for _ in range(topk)]
    else:
        weights = [float(w / weight_sum) for w in raw_weights]

    return selected, weights


def build_ensemble_backtest_result(selected_items, weights, backtest_horizon):
    base_windows = selected_items[0][1]['all_windows']
    combined_windows = []

    for window_index in range(len(base_windows)):
        dates = pd.to_datetime(base_windows[window_index]['dates'])
        actual = np.asarray(base_windows[window_index]['actual'], dtype=float)

        combined_pred = np.zeros_like(actual, dtype=float)
        for (context_len, result), weight in zip(selected_items, weights):
            pred = np.asarray(result['all_windows'][window_index]['predicted'], dtype=float)
            combined_pred += weight * pred

        smape, rmse = calculate_metrics(actual, combined_pred)
        combined_windows.append({
            'window_index': window_index + 1,
            'dates': dates.to_numpy(),
            'actual': actual,
            'predicted': combined_pred,
            'smape': smape,
            'rmse': rmse,
            'context_length': 'ensemble',
        })

    avg_smape = float(np.mean([result['smape'] for result in combined_windows]))
    avg_rmse = float(np.mean([result['rmse'] for result in combined_windows]))
    latest_window = combined_windows[0]
    latest_window['avg_smape'] = avg_smape
    latest_window['avg_rmse'] = avg_rmse
    latest_window['windows_used'] = len(combined_windows)
    latest_window['horizon'] = backtest_horizon
    latest_window['all_windows'] = combined_windows
    return latest_window


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
    context_search_points=50,
    context_ensemble_topk=3,
):
    print(f"\n--- 开始处理 {series_name} ---")

    # Calculate horizons
    last_date = df['date'].max()
    if target_date_dt <= last_date:
        print(f"错误：目标日期 {target_date_dt.date()} 必须晚于历史最后日期 {last_date.date()}。")
        sys.exit(1)

    future_horizon = (target_date_dt - last_date).days
    model_context_limit = get_model_context_limit(pipeline)

    print(f"历史最后日期：{last_date.date()}")
    print(f"目标预测日期：{target_date_dt.date()}")
    print(f"未来预测步长：{future_horizon} 天")
    print(f"模型上下文上限：{model_context_limit}")

    series_data = df[['date', series_name]].copy()
    effective_backtest_horizon = max(1, int(backtest_horizon))
    print(f"回测窗口：{effective_backtest_horizon} 天")

    # ==========================================
    # BACKTESTING
    # ==========================================
    if fixed_context_length is not None:
        selected_context_length = normalize_context_length(fixed_context_length, len(series_data), model_context_limit)
        print(
            f"使用固定 context_length={selected_context_length} 进行滚动回测："
            f"{rolling_windows} 个窗口 x {effective_backtest_horizon} 天"
        )
        bt_results = run_rolling_backtest(
            series_data,
            series_name,
            pipeline,
            device,
            selected_context_length,
            effective_backtest_horizon,
            rolling_windows,
        )
        selected_ensemble = None
        ensemble_weights = None
    else:
        if not context_candidates:
            max_search_context = max(1, min(len(series_data) - effective_backtest_horizon, model_context_limit))
            context_candidates = generate_auto_context_candidates(
                max_search_context,
                points=context_search_points,
                min_context=30,
            )
            print(f"自动生成 Context 候选数量：{len(context_candidates)}（最小=30，最大={max_search_context}）")

        print(
            f"开始搜索最优 context_length（{series_name}）："
            f"{rolling_windows} 个窗口 x {effective_backtest_horizon} 天"
        )
        selected_context_length, bt_results_single, candidate_results = select_best_context_length(
            series_data,
            series_name,
            pipeline,
            device,
            context_candidates,
            effective_backtest_horizon,
            rolling_windows,
            model_context_limit,
        )

        selected_ensemble = None
        ensemble_weights = None
        bt_results = bt_results_single

        if int(context_ensemble_topk) > 1:
            selected_ensemble, ensemble_weights = select_context_ensemble(
                candidate_results,
                topk=context_ensemble_topk,
            )
            bt_results = build_ensemble_backtest_result(
                selected_ensemble,
                ensemble_weights,
                effective_backtest_horizon,
            )
            ensemble_desc = ", ".join(
                f"{ctx}(w={w:.2f})" for (ctx, _), w in zip(selected_ensemble, ensemble_weights)
            )
            print(f"已启用 Context 融合（Top-{len(selected_ensemble)}）：{ensemble_desc}")

    smape = bt_results['avg_smape']
    rmse = bt_results['avg_rmse']

    print(f"{series_name} 回测结果：")
    if selected_ensemble:
        print(f"  选定 context_length：{selected_context_length}（并启用融合）")
    else:
        print(f"  选定 context_length：{selected_context_length}")
    print(f"  回测窗口数：{bt_results['windows_used']}")
    print(f"  平均 sMAPE：{smape:.2%}")
    print(f"  平均 RMSE：{rmse:.2f}")

    # ==========================================
    # FUTURE FORECASTING
    # ==========================================
    print(f"开始未来预测 {series_name}（共 {future_horizon} 天）...")
    if selected_ensemble:
        p10_future = np.zeros(future_horizon, dtype=float)
        p50_future = np.zeros(future_horizon, dtype=float)
        p90_future = np.zeros(future_horizon, dtype=float)
        active_weights = (
            ensemble_weights
            if ensemble_weights is not None
            else [1.0 / len(selected_ensemble)] * len(selected_ensemble)
        )

        with torch.no_grad():
            for (context_len, _), weight in zip(selected_ensemble, active_weights):
                effective_context = min(context_len, len(series_data))
                future_train_df = series_data.iloc[-effective_context:].copy()
                p10_i, p50_i, p90_i = predict_quantiles(
                    pipeline,
                    future_train_df[series_name].values,
                    future_horizon,
                    device,
                    quantiles=[0.1, 0.5, 0.9],
                    num_samples=100,
                )
                p10_future += weight * p10_i
                p50_future += weight * p50_i
                p90_future += weight * p90_i
    else:
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

def main():
    args = parse_args()

    if args.backtest_horizon <= 0 or args.rolling_windows <= 0:
        print("错误：--backtest_horizon 和 --rolling_windows 必须是正整数。")
        sys.exit(1)

    if args.context_length is not None and args.context_length <= 0:
        print("错误：--context_length 提供时必须是正整数。")
        sys.exit(1)

    if args.context_search_points <= 1:
        print("错误：--context_search_points 必须大于 1。")
        sys.exit(1)

    if args.residual_weight_search_points <= 1:
        print("错误：--residual_weight_search_points 必须大于 1。")
        sys.exit(1)

    if args.direct_weight_search_points <= 1:
        print("错误：--direct_weight_search_points 必须大于 1。")
        sys.exit(1)

    if args.context_ensemble_topk <= 0:
        print("错误：--context_ensemble_topk 必须是正整数。")
        sys.exit(1)

    if args.interval_coverage <= 0.0 or args.interval_coverage >= 1.0:
        print("错误：--interval_coverage 必须在 (0, 1) 之间。")
        sys.exit(1)

    if args.leadwise_weight_search_points <= 1:
        print("错误：--leadwise_weight_search_points 必须大于 1。")
        sys.exit(1)

    context_candidates = None
    if args.context_candidates.strip().lower() != 'auto':
        try:
            context_candidates = parse_context_candidates(args.context_candidates)
        except ValueError as e:
            print(f"错误：--context_candidates 非法。{e}")
            sys.exit(1)
    else:
        print(f"Context 候选采用自动模式，搜索点数：{args.context_search_points}")

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print(f"使用 HF 镜像端点：{args.hf_endpoint}")

    # Safely parse the date, handling arbitrary formatting errors
    target_date_dt = pd.to_datetime(args.target_date, errors='coerce')
    if pd.isna(target_date_dt):
        print(f"错误：--target_date '{args.target_date}' 日期格式无效，请使用 YYYY-MM-DD。")
        sys.exit(1)

    try:
        df = load_and_preprocess_data("data.csv")
    except ValueError as e:
        print(f"数据读取错误：{e}")
        sys.exit(1)

    last_date = df['date'].max()
    future_horizon_main = (target_date_dt - last_date).days
    if future_horizon_main <= 0:
        print(f"错误：目标日期 {target_date_dt.date()} 必须晚于历史最后日期 {last_date.date()}。")
        sys.exit(1)

    if args.auto_backtest_horizon:
        effective_backtest_horizon = resolve_adaptive_backtest_horizon(
            requested_horizon=args.backtest_horizon,
            future_horizon=future_horizon_main,
            history_length=len(df),
            rolling_windows=args.rolling_windows,
        )
        print(
            f"主流程回测窗口（自动对齐）：{args.backtest_horizon} -> {effective_backtest_horizon} 天 "
            f"（预测目标 {future_horizon_main} 天）"
        )
    else:
        effective_backtest_horizon = args.backtest_horizon
        print(f"主流程回测窗口（固定）：{effective_backtest_horizon} 天")

    if HAS_CN_CALENDAR:
        print("[Calendar] 中国节假日日历：已启用（法定节假日/补班日纳入纠偏）。")
    else:
        print("[Calendar] 中国节假日日历：未安装 chinese-calendar，当前使用周末近似。")

    # Setup device strictly for CPU inference per requirements
    device = "cpu"
    print(f"使用设备：{device}")

    print(f"加载 Chronos 模型：{args.model_id} ...")
    try:
        pipeline = load_pipeline(
            args.model_id,
            device,
            local_files_only=args.local_files_only,
        )
    except Exception as e:
        print(f"模型加载失败：{e}")
        sys.exit(1)

    csv_dir, png_dir = ensure_output_dirs()

    # 1. call_volume: Chronos 基线 + direct 多步监督融合
    bt_call_chronos, fut_call_chronos_base = evaluate_and_forecast_series(
        df,
        'call_volume',
        pipeline,
        target_date_dt,
        device,
        effective_backtest_horizon,
        args.rolling_windows,
        fixed_context_length=args.context_length,
        context_candidates=context_candidates,
        context_search_points=args.context_search_points,
        context_ensemble_topk=args.context_ensemble_topk,
    )

    direct_call_future = forecast_direct_multistep(
        df,
        'call_volume',
        horizon=future_horizon_main,
        reference_col='tickets_received',
        holiday_fn=is_china_holiday,
        makeup_fn=is_china_makeup_workday,
        return_feature_frame=True,
        progress_prefix='call_volume 未来预测',
        log_every=7,
        enable_progress_log=True,
    )
    bt_call_direct = run_direct_multistep_backtest(
        df,
        'call_volume',
        effective_backtest_horizon,
        args.rolling_windows,
        reference_col='tickets_received',
    )
    call_direct_fusion_weight, call_direct_fusion_effect = search_best_direct_fusion_weight(
        bt_call_chronos,
        bt_call_direct,
        search_points=args.direct_weight_search_points,
    )
    bt_call = blend_backtest_results(bt_call_chronos, bt_call_direct, call_direct_fusion_weight)
    fut_call_base = blend_future_results(
        fut_call_chronos_base,
        direct_call_future['predictions'],
        call_direct_fusion_weight,
    )

    residual_adjuster_call = build_residual_adjuster(bt_call)
    residual_adjuster_call, effect_call = search_best_residual_weight(
        bt_call,
        residual_adjuster_call,
        search_points=args.residual_weight_search_points,
    )
    fut_call = apply_residual_adjustment(fut_call_base, residual_adjuster_call)

    if not args.disable_leadwise_correction:
        leadwise_call = build_leadwise_adjuster(bt_call, residual_adjuster_call)
        leadwise_call, lead_effect_call = search_best_leadwise_weight(
            bt_call,
            residual_adjuster_call,
            leadwise_call,
            search_points=args.leadwise_weight_search_points,
        )
        fut_call = apply_leadwise_adjustment(fut_call, leadwise_call)
    else:
        leadwise_call = None
        lead_effect_call = None

    if not args.disable_interval_calibration:
        residuals_call = collect_adjusted_backtest_residuals(bt_call, residual_adjuster_call, leadwise_adjuster=leadwise_call)
        fut_call, calib_call = calibrate_prediction_intervals(
            fut_call,
            residuals_call,
            coverage=args.interval_coverage,
        )
        interval_bt_call = evaluate_interval_calibration_backtest(
            bt_call,
            residual_adjuster_call,
            coverage=args.interval_coverage,
            leadwise_adjuster=leadwise_call,
        )
    else:
        calib_call = {'radius': 0.0, 'empirical_coverage': 0.0}
        interval_bt_call = {'avg_coverage': 0.0, 'avg_width': 0.0, 'avg_radius': 0.0}

    holiday_anchor_call = estimate_holiday_zero_anchor(df[['date', 'call_volume']].copy(), 'call_volume')
    fut_call = apply_holiday_zero_adjustment(fut_call, holiday_anchor_call)
    print("\n[Residual纠偏] call_volume 已应用修正。")
    print(f"  全局残差均值：{residual_adjuster_call['global_bias']:.4f}")
    print(
        f"  状态小模型数（Ridge）：{len(residual_adjuster_call.get('state_models', {}))}，"
        f"模型融合权重={float(residual_adjuster_call.get('weight_state_model', 0.7)):.2f}"
    )
    print(
        "  状态分段残差权重："
        f"state={residual_adjuster_call['weight_state']:.2f}, "
        f"dow={residual_adjuster_call['weight_dow']:.2f}, "
        f"month={residual_adjuster_call['weight_month']:.2f}"
    )
    print(f"  状态样本数：{residual_adjuster_call['state_counts']}")
    if holiday_anchor_call.get('enabled', False):
        print(
            f"  节假日零值锚定：已启用，holiday_median={holiday_anchor_call['median']:.2f}, "
            f"holiday_p90={holiday_anchor_call['p90']:.2f}, 零值占比={holiday_anchor_call['zero_ratio']:.0%}, "
            f"强度={holiday_anchor_call['strength']:.2f}"
        )
        if holiday_anchor_call.get('long_holiday_enabled', False):
            print(
                f"  长假核心日锚定：median={holiday_anchor_call['long_holiday_median']:.2f}, "
                f"p75={holiday_anchor_call['long_holiday_p75']:.2f}, "
                f"零值占比={holiday_anchor_call['long_holiday_zero_ratio']:.0%}, "
                f"强度={holiday_anchor_call['long_holiday_strength']:.2f}"
            )
        proximity_summary_call = format_proximity_anchor_summary(holiday_anchor_call.get('proximity_anchors', {}))
        if proximity_summary_call:
            print(f"  节前后锚点：{proximity_summary_call}")
    print("  回测复评（纠偏前 -> 纠偏后）：")
    print(
        f"    平均 sMAPE：{effect_call['avg_smape_before']:.2%} -> {effect_call['avg_smape_after']:.2%} "
        f"(变化 {effect_call['avg_smape_after'] - effect_call['avg_smape_before']:+.2%})"
    )
    print(
        f"    平均 RMSE：{effect_call['avg_rmse_before']:.2f} -> {effect_call['avg_rmse_after']:.2f} "
        f"(变化 {effect_call['avg_rmse_after'] - effect_call['avg_rmse_before']:+.2f})"
    )
    print("  Chronos + Direct 融合复评（融合前 -> 融合后）：")
    print(
        f"    平均 sMAPE：{call_direct_fusion_effect['avg_smape_before']:.2%} -> {call_direct_fusion_effect['avg_smape_after']:.2%} "
        f"(变化 {call_direct_fusion_effect['avg_smape_after'] - call_direct_fusion_effect['avg_smape_before']:+.2%})"
    )
    print(
        f"    平均 RMSE：{call_direct_fusion_effect['avg_rmse_before']:.2f} -> {call_direct_fusion_effect['avg_rmse_after']:.2f} "
        f"(变化 {call_direct_fusion_effect['avg_rmse_after'] - call_direct_fusion_effect['avg_rmse_before']:+.2f})"
    )
    print(
        f"    最优 Chronos 权重：{call_direct_fusion_weight:.2f}，"
        f"Direct 权重：{1.0 - call_direct_fusion_weight:.2f}"
    )
    if lead_effect_call is not None and leadwise_call is not None:
        print("  Lead-wise 分层纠偏复评（Residual后 -> Lead-wise后）：")
        print(
            f"    平均 sMAPE：{lead_effect_call['avg_smape_before']:.2%} -> {lead_effect_call['avg_smape_after']:.2%} "
            f"(变化 {lead_effect_call['avg_smape_after'] - lead_effect_call['avg_smape_before']:+.2%})"
        )
        print(
            f"    平均 RMSE：{lead_effect_call['avg_rmse_before']:.2f} -> {lead_effect_call['avg_rmse_after']:.2f} "
            f"(变化 {lead_effect_call['avg_rmse_after'] - lead_effect_call['avg_rmse_before']:+.2f})"
        )
        print(f"    最优 Lead-wise 强度：{leadwise_call['leadwise_weight']:.2f}")
    if not args.disable_interval_calibration:
        print(
            f"  区间后校准：覆盖率目标={args.interval_coverage:.0%}, "
            f"经验覆盖率={calib_call['empirical_coverage']:.0%}, "
            f"校准半径={calib_call['radius']:.2f}"
        )
        print(
            f"  区间回测复评：平均覆盖率={interval_bt_call['avg_coverage']:.0%}, "
            f"平均区间宽度={interval_bt_call['avg_width']:.2f}, "
            f"平均校准半径={interval_bt_call['avg_radius']:.2f}"
        )

    # 2. 使用 Chronos 预测 tickets_received（基线）
    bt_ticket_chronos, fut_ticket_chronos_base = evaluate_and_forecast_series(
        df,
        'tickets_received',
        pipeline,
        target_date_dt,
        device,
        effective_backtest_horizon,
        args.rolling_windows,
        fixed_context_length=args.context_length,
        context_candidates=context_candidates,
        context_search_points=args.context_search_points,
        context_ensemble_topk=args.context_ensemble_topk,
    )

    direct_ticket_future = forecast_direct_multistep(
        df,
        'tickets_received',
        horizon=future_horizon_main,
        reference_col='call_volume',
        holiday_fn=is_china_holiday,
        makeup_fn=is_china_makeup_workday,
        return_feature_frame=True,
        progress_prefix='tickets_received 未来预测',
        log_every=7,
        enable_progress_log=True,
    )
    bt_ticket_direct = run_direct_multistep_backtest(
        df,
        'tickets_received',
        effective_backtest_horizon,
        args.rolling_windows,
        reference_col='call_volume',
    )
    direct_fusion_weight, direct_fusion_effect = search_best_direct_fusion_weight(
        bt_ticket_chronos,
        bt_ticket_direct,
        search_points=args.direct_weight_search_points,
    )
    bt_ticket = blend_backtest_results(bt_ticket_chronos, bt_ticket_direct, direct_fusion_weight)
    fut_ticket_base = blend_future_results(
        fut_ticket_chronos_base,
        direct_ticket_future['predictions'],
        direct_fusion_weight,
    )

    call_feature_df = direct_call_future['feature_frame'].copy()
    call_feature_df['target_name'] = 'call_volume'
    ticket_feature_df = direct_ticket_future['feature_frame'].copy()
    ticket_feature_df['target_name'] = 'tickets_received'
    merged_feature_df = pd.concat([call_feature_df, ticket_feature_df], ignore_index=True)
    feature_export_filename = os.path.join(csv_dir, 'feature_engineering_merged.csv')
    export_dataframe_csv(merged_feature_df, feature_export_filename)

    call_model_summary_df = direct_call_future['model_summaries'].copy()
    call_model_summary_df['target_name'] = 'call_volume'
    ticket_model_summary_df = direct_ticket_future['model_summaries'].copy()
    ticket_model_summary_df['target_name'] = 'tickets_received'
    merged_model_summary_df = pd.concat([call_model_summary_df, ticket_model_summary_df], ignore_index=True)
    direct_model_summary_filename = os.path.join(csv_dir, 'direct_model_summary_merged.csv')
    export_dataframe_csv(merged_model_summary_df, direct_model_summary_filename)

    # 3. Residual 纠偏：根据回测残差做全局+周几偏差修正
    residual_adjuster = build_residual_adjuster(bt_ticket)
    residual_adjuster, effect_ticket = search_best_residual_weight(
        bt_ticket,
        residual_adjuster,
        search_points=args.residual_weight_search_points,
    )
    fut_ticket = apply_residual_adjustment(fut_ticket_base, residual_adjuster)

    if not args.disable_leadwise_correction:
        leadwise_ticket = build_leadwise_adjuster(bt_ticket, residual_adjuster)
        leadwise_ticket, lead_effect_ticket = search_best_leadwise_weight(
            bt_ticket,
            residual_adjuster,
            leadwise_ticket,
            search_points=args.leadwise_weight_search_points,
        )
        fut_ticket = apply_leadwise_adjustment(fut_ticket, leadwise_ticket)
    else:
        leadwise_ticket = None
        lead_effect_ticket = None

    if not args.disable_interval_calibration:
        residuals_ticket = collect_adjusted_backtest_residuals(bt_ticket, residual_adjuster, leadwise_adjuster=leadwise_ticket)
        fut_ticket, calib_ticket = calibrate_prediction_intervals(
            fut_ticket,
            residuals_ticket,
            coverage=args.interval_coverage,
        )
        interval_bt_ticket = evaluate_interval_calibration_backtest(
            bt_ticket,
            residual_adjuster,
            coverage=args.interval_coverage,
            leadwise_adjuster=leadwise_ticket,
        )
    else:
        calib_ticket = {'radius': 0.0, 'empirical_coverage': 0.0}
        interval_bt_ticket = {'avg_coverage': 0.0, 'avg_width': 0.0, 'avg_radius': 0.0}

    holiday_anchor_ticket = estimate_holiday_zero_anchor(df[['date', 'tickets_received']].copy(), 'tickets_received')
    fut_ticket = apply_holiday_zero_adjustment(fut_ticket, holiday_anchor_ticket)
    print("\n[Residual纠偏] tickets_received 已应用修正。")
    print(f"  全局残差均值：{residual_adjuster['global_bias']:.4f}")
    print(
        f"  状态小模型数（Ridge）：{len(residual_adjuster.get('state_models', {}))}，"
        f"模型融合权重={float(residual_adjuster.get('weight_state_model', 0.7)):.2f}"
    )
    print(
        "  状态分段残差权重："
        f"state={residual_adjuster['weight_state']:.2f}, "
        f"dow={residual_adjuster['weight_dow']:.2f}, "
        f"month={residual_adjuster['weight_month']:.2f}"
    )
    print(f"  状态样本数：{residual_adjuster['state_counts']}")
    if holiday_anchor_ticket.get('enabled', False):
        print(
            f"  节假日零值锚定：已启用，holiday_median={holiday_anchor_ticket['median']:.2f}, "
            f"holiday_p90={holiday_anchor_ticket['p90']:.2f}, 零值占比={holiday_anchor_ticket['zero_ratio']:.0%}, "
            f"强度={holiday_anchor_ticket['strength']:.2f}"
        )
        if holiday_anchor_ticket.get('long_holiday_enabled', False):
            print(
                f"  长假核心日锚定：median={holiday_anchor_ticket['long_holiday_median']:.2f}, "
                f"p75={holiday_anchor_ticket['long_holiday_p75']:.2f}, "
                f"零值占比={holiday_anchor_ticket['long_holiday_zero_ratio']:.0%}, "
                f"强度={holiday_anchor_ticket['long_holiday_strength']:.2f}"
            )
        proximity_summary_ticket = format_proximity_anchor_summary(holiday_anchor_ticket.get('proximity_anchors', {}))
        if proximity_summary_ticket:
            print(f"  节前后锚点：{proximity_summary_ticket}")
    print("  Chronos + Direct 融合复评（融合前 -> 融合后）：")
    print(
        f"    平均 sMAPE：{direct_fusion_effect['avg_smape_before']:.2%} -> {direct_fusion_effect['avg_smape_after']:.2%} "
        f"(变化 {direct_fusion_effect['avg_smape_after'] - direct_fusion_effect['avg_smape_before']:+.2%})"
    )
    print(
        f"    平均 RMSE：{direct_fusion_effect['avg_rmse_before']:.2f} -> {direct_fusion_effect['avg_rmse_after']:.2f} "
        f"(变化 {direct_fusion_effect['avg_rmse_after'] - direct_fusion_effect['avg_rmse_before']:+.2f})"
    )
    print(f"    最优 Chronos 权重：{direct_fusion_weight:.2f}，Direct 权重：{1.0 - direct_fusion_weight:.2f}")
    print("  回测复评（纠偏前 -> 纠偏后）：")
    print(
        f"    平均 sMAPE：{effect_ticket['avg_smape_before']:.2%} -> {effect_ticket['avg_smape_after']:.2%} "
        f"(变化 {effect_ticket['avg_smape_after'] - effect_ticket['avg_smape_before']:+.2%})"
    )
    print(
        f"    平均 RMSE：{effect_ticket['avg_rmse_before']:.2f} -> {effect_ticket['avg_rmse_after']:.2f} "
        f"(变化 {effect_ticket['avg_rmse_after'] - effect_ticket['avg_rmse_before']:+.2f})"
    )
    if lead_effect_ticket is not None:
        print("  Lead-wise 分层纠偏复评（Residual后 -> Lead-wise后）：")
        print(
            f"    平均 sMAPE：{lead_effect_ticket['avg_smape_before']:.2%} -> {lead_effect_ticket['avg_smape_after']:.2%} "
            f"(变化 {lead_effect_ticket['avg_smape_after'] - lead_effect_ticket['avg_smape_before']:+.2%})"
        )
        print(
            f"    平均 RMSE：{lead_effect_ticket['avg_rmse_before']:.2f} -> {lead_effect_ticket['avg_rmse_after']:.2f} "
            f"(变化 {lead_effect_ticket['avg_rmse_after'] - lead_effect_ticket['avg_rmse_before']:+.2f})"
        )
        print(f"    最优 Lead-wise 强度：{leadwise_ticket['leadwise_weight']:.2f}")
    if not args.disable_interval_calibration:
        print(
            f"  区间后校准：覆盖率目标={args.interval_coverage:.0%}, "
            f"经验覆盖率={calib_ticket['empirical_coverage']:.0%}, "
            f"校准半径={calib_ticket['radius']:.2f}"
        )
        print(
            f"  区间回测复评：平均覆盖率={interval_bt_ticket['avg_coverage']:.0%}, "
            f"平均区间宽度={interval_bt_ticket['avg_width']:.2f}, "
            f"平均校准半径={interval_bt_ticket['avg_radius']:.2f}"
        )

    # 4. 导出标准化 CSV（作为 API 数据源）
    export_filename = os.path.join(csv_dir, 'forecast_export.csv')
    export_forecast_csv(fut_call, fut_ticket, export_filename)
    print(f"\n已导出标准预测数据：{export_filename}")
    print(f"已导出 call+tickets 特征工程明细：{feature_export_filename}")
    print(f"已导出 call+tickets direct 模型摘要：{direct_model_summary_filename}")

    # 5. 基于 CSV 进行绘图
    eval_png = os.path.join(png_dir, 'evaluation_results.png')
    future_png = os.path.join(png_dir, 'future_forecast.png')
    export_png = os.path.join(png_dir, 'forecast_export_plot.png')
    bt_call_plot = apply_holiday_zero_adjustment_to_backtest(bt_call, holiday_anchor_call)
    bt_ticket_plot = apply_holiday_zero_adjustment_to_backtest(bt_ticket, holiday_anchor_ticket)

    plot_evaluation(bt_call_plot, bt_ticket_plot, eval_png)
    plot_future_from_csv("data.csv", export_filename, future_png)
    plot_forecast_export_csv(export_filename, export_png)

    print(f"\n预测完成。请查看 {export_filename}、{eval_png}、{future_png}、{export_png}。")

if __name__ == "__main__":
    main()
