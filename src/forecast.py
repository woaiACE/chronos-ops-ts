import argparse
import gc
import importlib
import json
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
from src.output_manager import ensure_output_dirs, export_dataframe_csv, export_forecast_csv, export_markdown_report
from src.visualization import (
    plot_evaluation,
    plot_forecast_export_csv,
    plot_future_from_csv,
    plot_monitor_bucket_report,
    plot_monitor_bucket_sample_scope,
    plot_stage_transition,
    plot_tuning_report,
)

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

DEFAULT_SPRING_SERVICE_RULES = {
    'default': {
        'shutdown_positions': [2, 3, 4, 5],
        'duty_positions': [1, 6, 7, 8, 9],
    },
    'year_overrides': {},
    'date_overrides': {},
}
SPRING_SERVICE_RULES = {
    'default': dict(DEFAULT_SPRING_SERVICE_RULES['default']),
    'year_overrides': {},
    'date_overrides': {},
}

DEFAULT_SERIES_TUNING = {
    'call_volume': {
        'leadwise_weight_cap': 0.75,
        'holiday_segment_model_weight': 0.65,
        'post_holiday_anchor_strengths': {
            'day1_default': 0.40,
            'day1_strong': 0.55,
            'day2_default': 0.30,
            'day2_strong': 0.45,
            'day3_default': 0.20,
            'day3_strong': 0.35,
        },
        'holiday_layered_model': {
            'enabled': True,
            'min_samples': 6,
            'blend_weight': 0.55,
        },
        'bias_gate': {
            'enabled': True,
            'recent_days': 84,
            'min_samples': 14,
            'target_bucket': 'post_holiday_workday_1_3',
            'target_buckets': ['post_holiday_workday_1_3'],
            'bucket_scales': {},
            'apply_to_backtest': True,
            'only_positive_bias': True,
            'bias_trigger': 2.0,
            'adjustment_scale': 0.35,
            'max_adjustment_ratio': 0.10,
            'enforce_rmse_guard': False,
            'rmse_guard_scale': 0.75,
            'allow_scope_fallback': True,
            'fallback_scale': 0.60,
            'allow_low_sample_gate': False,
            'low_sample_floor': 0.35,
        },
        'dynamic_leadwise_cap': {
            'enabled': True,
            'recent_days': 84,
            'smape_high': 0.16,
            'smape_low': 0.12,
            'down_shift': 0.10,
            'up_shift': 0.03,
            'min_cap': 0.55,
            'max_cap': 0.85,
        },
        'bucket_fusion_secondary': {
            'enabled': True,
            'min_samples': 8,
            'enable_holiday_layers': True,
            'holiday_min_samples': 6,
        },
        'asymmetric_interval': {
            'enabled': True,
            'lower_coverage': 0.80,
            'upper_coverage': 0.80,
        },
    },
    'tickets_received': {
        'leadwise_weight_cap': 0.80,
        'holiday_segment_model_weight': 0.75,
        'post_holiday_anchor_strengths': {
            'day1_default': 0.45,
            'day1_strong': 0.60,
            'day2_default': 0.35,
            'day2_strong': 0.50,
            'day3_default': 0.25,
            'day3_strong': 0.40,
        },
        'holiday_layered_model': {
            'enabled': True,
            'min_samples': 6,
            'blend_weight': 0.55,
        },
        'bias_gate': {
            'enabled': True,
            'recent_days': 84,
            'min_samples': 14,
            'target_bucket': 'post_holiday_workday_1_3',
            'target_buckets': ['post_holiday_workday_1_3'],
            'bucket_scales': {},
            'apply_to_backtest': True,
            'only_positive_bias': False,
            'bias_trigger': 1.5,
            'adjustment_scale': 0.25,
            'max_adjustment_ratio': 0.08,
            'enforce_rmse_guard': False,
            'rmse_guard_scale': 0.75,
            'allow_scope_fallback': True,
            'fallback_scale': 0.60,
            'allow_low_sample_gate': False,
            'low_sample_floor': 0.35,
        },
        'dynamic_leadwise_cap': {
            'enabled': True,
            'recent_days': 84,
            'smape_high': 0.18,
            'smape_low': 0.12,
            'down_shift': 0.08,
            'up_shift': 0.03,
            'min_cap': 0.60,
            'max_cap': 0.90,
        },
        'bucket_fusion_secondary': {
            'enabled': True,
            'min_samples': 8,
            'enable_holiday_layers': True,
            'holiday_min_samples': 6,
        },
        'asymmetric_interval': {
            'enabled': True,
            'lower_coverage': 0.80,
            'upper_coverage': 0.80,
        },
    },
}
SERIES_TUNING = {
    'call_volume': dict(DEFAULT_SERIES_TUNING['call_volume']),
    'tickets_received': dict(DEFAULT_SERIES_TUNING['tickets_received']),
}


def _normalize_position_list(values):
    if values is None:
        return []
    normalized = []
    for value in values:
        try:
            pos = int(value)
        except Exception:
            continue
        if pos >= 1:
            normalized.append(pos)
    return sorted(set(normalized))


def _normalize_state_label(value):
    label = str(value).strip().lower()
    if label in {'shutdown', 'duty', 'none'}:
        return label
    return None


def _deep_copy_series_tuning_defaults():
    payload = {}
    for series_name, tuning in DEFAULT_SERIES_TUNING.items():
        payload[series_name] = {
            'leadwise_weight_cap': float(tuning['leadwise_weight_cap']),
            'holiday_segment_model_weight': float(tuning['holiday_segment_model_weight']),
            'post_holiday_anchor_strengths': dict(tuning['post_holiday_anchor_strengths']),
            'holiday_layered_model': dict(tuning.get('holiday_layered_model', {})),
            'bias_gate': dict(tuning.get('bias_gate', {})),
            'dynamic_leadwise_cap': dict(tuning.get('dynamic_leadwise_cap', {})),
            'bucket_fusion_secondary': dict(tuning.get('bucket_fusion_secondary', {})),
            'asymmetric_interval': dict(tuning.get('asymmetric_interval', {})),
        }
    return payload


def _normalize_float_in_range(value, minimum=0.0, maximum=1.0):
    try:
        num = float(value)
    except Exception:
        return None
    return float(np.clip(num, minimum, maximum))


def configure_series_tuning(raw_json):
    global SERIES_TUNING
    configured = _deep_copy_series_tuning_defaults()

    if raw_json is None:
        SERIES_TUNING = configured
        return

    if isinstance(raw_json, str):
        payload_text = raw_json.strip()
        if not payload_text:
            SERIES_TUNING = configured
            return
        try:
            payload = json.loads(payload_text)
        except Exception as exc:
            raise ValueError(f"Invalid --series_tuning_json: {exc}") from exc
    elif isinstance(raw_json, dict):
        payload = raw_json
    else:
        raise ValueError("series_tuning must be a JSON object.")

    if not isinstance(payload, dict):
        raise ValueError("series_tuning must be a JSON object.")

    for series_name in ['call_volume', 'tickets_received']:
        series_payload = payload.get(series_name)
        if not isinstance(series_payload, dict):
            continue

        leadwise_cap = _normalize_float_in_range(series_payload.get('leadwise_weight_cap'))
        if leadwise_cap is not None:
            configured[series_name]['leadwise_weight_cap'] = leadwise_cap

        holiday_weight = _normalize_float_in_range(series_payload.get('holiday_segment_model_weight'))
        if holiday_weight is not None:
            configured[series_name]['holiday_segment_model_weight'] = holiday_weight

        strength_payload = series_payload.get('post_holiday_anchor_strengths')
        if isinstance(strength_payload, dict):
            for key in [
                'day1_default',
                'day1_strong',
                'day2_default',
                'day2_strong',
                'day3_default',
                'day3_strong',
            ]:
                strength_value = _normalize_float_in_range(strength_payload.get(key))
                if strength_value is not None:
                    configured[series_name]['post_holiday_anchor_strengths'][key] = strength_value

        layered_payload = series_payload.get('holiday_layered_model')
        if isinstance(layered_payload, dict):
            if 'enabled' in layered_payload:
                configured[series_name]['holiday_layered_model']['enabled'] = bool(layered_payload['enabled'])
            if 'min_samples' in layered_payload:
                try:
                    configured[series_name]['holiday_layered_model']['min_samples'] = max(1, int(layered_payload['min_samples']))
                except Exception:
                    pass
            blend_weight = _normalize_float_in_range(layered_payload.get('blend_weight'))
            if blend_weight is not None:
                configured[series_name]['holiday_layered_model']['blend_weight'] = float(blend_weight)

        bias_gate_payload = series_payload.get('bias_gate')
        if isinstance(bias_gate_payload, dict):
            for key in [
                'recent_days',
                'min_samples',
            ]:
                if key in bias_gate_payload:
                    try:
                        configured[series_name]['bias_gate'][key] = max(1, int(bias_gate_payload[key]))
                    except Exception:
                        pass
            for key in [
                'enabled',
                'apply_to_backtest',
                'only_positive_bias',
                'enforce_rmse_guard',
                'allow_scope_fallback',
                'allow_low_sample_gate',
            ]:
                if key in bias_gate_payload:
                    configured[series_name]['bias_gate'][key] = bool(bias_gate_payload[key])
            for key in [
                'bias_trigger',
                'adjustment_scale',
                'max_adjustment_ratio',
                'fallback_scale',
                'low_sample_floor',
            ]:
                value = _normalize_float_in_range(bias_gate_payload.get(key), minimum=0.0, maximum=10.0)
                if value is not None:
                    configured[series_name]['bias_gate'][key] = float(value)
            if 'bucket_scales' in bias_gate_payload and isinstance(bias_gate_payload.get('bucket_scales'), dict):
                bucket_scales = {}
                for raw_bucket, raw_scale in bias_gate_payload.get('bucket_scales', {}).items():
                    bucket_name = _normalize_bucket_name(raw_bucket)
                    if not bucket_name:
                        continue
                    bucket_scale = _normalize_float_in_range(raw_scale, minimum=0.0, maximum=10.0)
                    if bucket_scale is None:
                        continue
                    bucket_scales[bucket_name] = float(bucket_scale)
                if bucket_scales:
                    configured[series_name]['bias_gate']['bucket_scales'] = bucket_scales
            if 'target_bucket' in bias_gate_payload:
                target_bucket = str(bias_gate_payload['target_bucket']).strip() or 'post_holiday_workday_1_3'
                configured[series_name]['bias_gate']['target_bucket'] = target_bucket
                configured[series_name]['bias_gate']['target_buckets'] = [target_bucket]
            if 'target_buckets' in bias_gate_payload:
                raw_targets = bias_gate_payload.get('target_buckets')
                if isinstance(raw_targets, (list, tuple)):
                    target_buckets = []
                    for item in raw_targets:
                        bucket_name = _normalize_bucket_name(item)
                        if bucket_name and bucket_name not in target_buckets:
                            target_buckets.append(bucket_name)
                    if target_buckets:
                        configured[series_name]['bias_gate']['target_buckets'] = target_buckets
                        configured[series_name]['bias_gate']['target_bucket'] = target_buckets[0]
                elif isinstance(raw_targets, str):
                    target_buckets = []
                    for part in raw_targets.split(','):
                        bucket_name = _normalize_bucket_name(part)
                        if bucket_name and bucket_name not in target_buckets:
                            target_buckets.append(bucket_name)
                    if target_buckets:
                        configured[series_name]['bias_gate']['target_buckets'] = target_buckets
                        configured[series_name]['bias_gate']['target_bucket'] = target_buckets[0]

        dynamic_cap_payload = series_payload.get('dynamic_leadwise_cap')
        if isinstance(dynamic_cap_payload, dict):
            if 'enabled' in dynamic_cap_payload:
                configured[series_name]['dynamic_leadwise_cap']['enabled'] = bool(dynamic_cap_payload['enabled'])
            for key in ['recent_days']:
                if key in dynamic_cap_payload:
                    try:
                        configured[series_name]['dynamic_leadwise_cap'][key] = max(1, int(dynamic_cap_payload[key]))
                    except Exception:
                        pass
            for key in ['smape_high', 'smape_low', 'down_shift', 'up_shift', 'min_cap', 'max_cap']:
                value = _normalize_float_in_range(dynamic_cap_payload.get(key), minimum=0.0, maximum=1.0)
                if value is not None:
                    configured[series_name]['dynamic_leadwise_cap'][key] = float(value)

        bucket_fusion_payload = series_payload.get('bucket_fusion_secondary')
        if isinstance(bucket_fusion_payload, dict):
            if 'enabled' in bucket_fusion_payload:
                configured[series_name]['bucket_fusion_secondary']['enabled'] = bool(bucket_fusion_payload['enabled'])
            if 'min_samples' in bucket_fusion_payload:
                try:
                    configured[series_name]['bucket_fusion_secondary']['min_samples'] = max(1, int(bucket_fusion_payload['min_samples']))
                except Exception:
                    pass
            if 'enable_holiday_layers' in bucket_fusion_payload:
                configured[series_name]['bucket_fusion_secondary']['enable_holiday_layers'] = bool(bucket_fusion_payload['enable_holiday_layers'])
            if 'holiday_min_samples' in bucket_fusion_payload:
                try:
                    configured[series_name]['bucket_fusion_secondary']['holiday_min_samples'] = max(1, int(bucket_fusion_payload['holiday_min_samples']))
                except Exception:
                    pass

        asymmetric_payload = series_payload.get('asymmetric_interval')
        if isinstance(asymmetric_payload, dict):
            if 'enabled' in asymmetric_payload:
                configured[series_name]['asymmetric_interval']['enabled'] = bool(asymmetric_payload['enabled'])
            for key in ['lower_coverage', 'upper_coverage']:
                value = _normalize_float_in_range(asymmetric_payload.get(key), minimum=0.5, maximum=0.99)
                if value is not None:
                    configured[series_name]['asymmetric_interval'][key] = float(value)

    SERIES_TUNING = configured


def get_series_tuning(series_name):
    series_key = str(series_name).strip().lower()
    defaults = _deep_copy_series_tuning_defaults()
    base = defaults.get(series_key, {})
    current = SERIES_TUNING.get(series_key, {})

    merged = {
        'leadwise_weight_cap': float(current.get('leadwise_weight_cap', base.get('leadwise_weight_cap', 1.0))),
        'holiday_segment_model_weight': float(current.get('holiday_segment_model_weight', base.get('holiday_segment_model_weight', 0.75))),
        'post_holiday_anchor_strengths': dict(base.get('post_holiday_anchor_strengths', {})),
        'holiday_layered_model': dict(base.get('holiday_layered_model', {})),
        'bias_gate': dict(base.get('bias_gate', {})),
        'dynamic_leadwise_cap': dict(base.get('dynamic_leadwise_cap', {})),
        'bucket_fusion_secondary': dict(base.get('bucket_fusion_secondary', {})),
        'asymmetric_interval': dict(base.get('asymmetric_interval', {})),
    }
    merged['post_holiday_anchor_strengths'].update(current.get('post_holiday_anchor_strengths', {}))
    merged['holiday_layered_model'].update(current.get('holiday_layered_model', {}))
    merged['bias_gate'].update(current.get('bias_gate', {}))
    merged['dynamic_leadwise_cap'].update(current.get('dynamic_leadwise_cap', {}))
    merged['bucket_fusion_secondary'].update(current.get('bucket_fusion_secondary', {}))
    merged['asymmetric_interval'].update(current.get('asymmetric_interval', {}))
    return merged


def configure_spring_service_rules(raw_json):
    global SPRING_SERVICE_RULES
    configured = {
        'default': dict(DEFAULT_SPRING_SERVICE_RULES['default']),
        'year_overrides': {},
        'date_overrides': {},
    }

    if raw_json is None:
        SPRING_SERVICE_RULES = configured
        return

    if isinstance(raw_json, str):
        payload_text = raw_json.strip()
        if not payload_text:
            SPRING_SERVICE_RULES = configured
            return
        try:
            payload = json.loads(payload_text)
        except Exception as exc:
            raise ValueError(f"Invalid --spring_service_rules_json: {exc}") from exc
    elif isinstance(raw_json, dict):
        payload = raw_json
    else:
        raise ValueError("spring_service_rules must be a JSON object.")

    if not isinstance(payload, dict):
        raise ValueError("spring_service_rules must be a JSON object.")

    default_rule = payload.get('default', {})
    if isinstance(default_rule, dict):
        shutdown_positions = _normalize_position_list(
            default_rule.get('shutdown_positions', configured['default']['shutdown_positions'])
        )
        duty_positions = _normalize_position_list(
            default_rule.get('duty_positions', configured['default']['duty_positions'])
        )
        if shutdown_positions:
            configured['default']['shutdown_positions'] = shutdown_positions
        if duty_positions:
            configured['default']['duty_positions'] = duty_positions

    year_overrides = payload.get('year_overrides', {})
    if isinstance(year_overrides, dict):
        for year_key, rule in year_overrides.items():
            try:
                year = int(year_key)
            except Exception:
                continue
            if not isinstance(rule, dict):
                continue
            shutdown_positions = _normalize_position_list(
                rule.get('shutdown_positions', configured['default']['shutdown_positions'])
            )
            duty_positions = _normalize_position_list(
                rule.get('duty_positions', configured['default']['duty_positions'])
            )
            configured['year_overrides'][year] = {
                'shutdown_positions': shutdown_positions,
                'duty_positions': duty_positions,
            }

    date_overrides = payload.get('date_overrides', {})
    if isinstance(date_overrides, dict):
        for date_key, state in date_overrides.items():
            label = _normalize_state_label(state)
            if label is None:
                continue
            try:
                normalized_date = pd.to_datetime(date_key).date().isoformat()
            except Exception:
                continue
            configured['date_overrides'][normalized_date] = label

    SPRING_SERVICE_RULES = configured

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
        "--leadwise_weight_cap",
        type=float,
        default=1.0,
        help="Upper bound for lead-wise correction strength search. Use <1.0 for safer online behavior."
    )
    parser.add_argument(
        "--monitor_recent_days",
        type=int,
        default=84,
        help="Recent-day window used for bucket monitoring report (default 84)."
    )
    parser.add_argument(
        "--monitor_low_sample_threshold",
        type=int,
        default=12,
        help="Low-confidence threshold for bucket backtest sample size in report text."
    )
    parser.add_argument(
        "--disable_feature_export",
        action="store_true",
        help="Disable exporting large feature engineering detail CSV."
    )
    parser.add_argument(
        "--feature_export_rows_limit",
        type=int,
        default=0,
        help="Optional max rows for feature engineering export. 0 means export all rows."
    )
    parser.add_argument(
        "--spring_service_rules_json",
        type=str,
        default=None,
        help=(
            "Optional JSON for Spring Festival service rules. "
            "Supports keys: default.shutdown_positions/duty_positions, "
            "year_overrides, date_overrides."
        ),
    )
    parser.add_argument(
        "--series_tuning_json",
        type=str,
        default=None,
        help=(
            "Optional JSON for per-series tuning. "
            "Supports call_volume/tickets_received: leadwise_weight_cap, "
            "holiday_segment_model_weight, post_holiday_anchor_strengths."
        ),
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


def build_residual_adjuster(bt_results, holiday_segment_model_weight=0.75, holiday_layered_config=None):
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
            'weight_holiday_segment_model': float(np.clip(holiday_segment_model_weight, 0.0, 1.0)),
            'weight_holiday_layered_model': float(np.clip(float((holiday_layered_config or {}).get('blend_weight', 0.55)), 0.0, 1.0)),
            'state_models': {},
            'holiday_layered_models': {},
            'holiday_layer_bias': {},
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

    holiday_ctx = classify_holiday_blocks(residual_df['date'])
    residual_df['holiday_block_length'] = holiday_ctx['block_length']
    residual_df['holiday_block_position'] = holiday_ctx['block_position']
    residual_df['pre_holiday_n_day'] = holiday_ctx['pre_holiday_n_day']
    residual_df['post_holiday_workday_n'] = holiday_ctx['post_holiday_workday_n']
    spring_ctx = classify_spring_festival_service_phases(residual_df['date'])
    residual_df['is_spring_shutdown'] = spring_ctx['is_spring_shutdown']
    residual_df['is_spring_duty'] = spring_ctx['is_spring_duty']

    def _classify_holiday_layer_from_row(row):
        if int(row['is_holiday']) == 1:
            if int(row.get('is_spring_shutdown', 0)) == 1:
                return 'holiday_spring_shutdown'
            if int(row.get('is_spring_duty', 0)) == 1:
                return 'holiday_spring_duty'
            if int(row.get('holiday_block_length', 0)) >= 4 and 2 <= int(row.get('holiday_block_position', 0)) <= 5:
                return 'holiday_long_core'
            return 'holiday_general'
        pre_n = int(row.get('pre_holiday_n_day', 0))
        if pre_n == 1:
            return 'pre_holiday_1d'
        if pre_n == 2:
            return 'pre_holiday_2d'
        post_n = int(row.get('post_holiday_workday_n', 0))
        if post_n == 1:
            return 'post_holiday_workday_1'
        if post_n == 2:
            return 'post_holiday_workday_2'
        if post_n == 3:
            return 'post_holiday_workday_3'
        return 'non_holiday'

    residual_df['holiday_layer'] = residual_df.apply(_classify_holiday_layer_from_row, axis=1)

    global_bias = float(residual_df['residual'].mean())
    residual_std = float(residual_df['residual'].std(ddof=0)) if len(residual_df) > 1 else 0.0

    state_bias = {}
    state_dow_bias = {}
    state_month_phase_bias = {}
    state_counts = {}
    state_models = {}
    holiday_segment_model = None
    holiday_layered_models = {}
    holiday_layer_bias = {}

    layered_cfg = holiday_layered_config or {}
    layered_enabled = bool(layered_cfg.get('enabled', True))
    layered_min_samples = max(1, int(layered_cfg.get('min_samples', 6)))
    layered_blend_weight = float(np.clip(float(layered_cfg.get('blend_weight', 0.55)), 0.0, 1.0))

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

    holiday_segment_frame = residual_df[
        (residual_df['state'] == 'holiday_or_makeup') |
        (residual_df['pre_holiday_n_day'] > 0) |
        (residual_df['post_holiday_workday_n'] > 0)
    ].copy()
    holiday_feature_cols = [
        'dow',
        'month_phase_code',
        'is_month_end_settlement',
        'is_holiday',
        'is_makeup_workday',
        'predicted',
        'holiday_block_length',
        'holiday_block_position',
        'pre_holiday_n_day',
        'post_holiday_workday_n',
    ]
    holiday_clean = holiday_segment_frame[holiday_feature_cols + ['residual']].replace([np.inf, -np.inf], np.nan).dropna()
    if len(holiday_clean) >= 6:
        holiday_model = Ridge(alpha=0.8)
        holiday_model.fit(holiday_clean[holiday_feature_cols], holiday_clean['residual'])
        holiday_segment_model = {
            'model': holiday_model,
            'feature_cols': holiday_feature_cols,
            'fallback_bias': float(holiday_clean['residual'].mean()),
        }

    if layered_enabled:
        for layer_name, layer_group in holiday_segment_frame.groupby('holiday_layer'):
            layer_key = str(layer_name)
            if layer_key == 'non_holiday':
                continue
            holiday_layer_bias[layer_key] = float(layer_group['residual'].mean())
            layer_clean = layer_group[holiday_feature_cols + ['residual']].replace([np.inf, -np.inf], np.nan).dropna()
            if len(layer_clean) < layered_min_samples:
                continue
            layer_model = Ridge(alpha=0.8)
            layer_model.fit(layer_clean[holiday_feature_cols], layer_clean['residual'])
            holiday_layered_models[layer_key] = {
                'model': layer_model,
                'feature_cols': holiday_feature_cols,
                'fallback_bias': float(layer_clean['residual'].mean()),
                'sample_size': int(len(layer_clean)),
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
        'weight_holiday_segment_model': float(np.clip(holiday_segment_model_weight, 0.0, 1.0)),
        'weight_holiday_layered_model': layered_blend_weight,
        'state_models': state_models,
        'holiday_segment_model': holiday_segment_model,
        'holiday_layered_models': holiday_layered_models,
        'holiday_layer_bias': holiday_layer_bias,
        'residual_std': residual_std,
    }


def build_holiday_context_rows(date_series):
    dates = pd.Series(pd.to_datetime(date_series)).reset_index(drop=True)
    info = classify_holiday_blocks(dates)
    spring_info = classify_spring_festival_service_phases(dates)
    rows = []

    def _classify_holiday_layer(idx):
        if int(info['is_holiday_non_makeup'][idx]) == 1:
            if int(spring_info['is_spring_shutdown'][idx]) == 1:
                return 'holiday_spring_shutdown'
            if int(spring_info['is_spring_duty'][idx]) == 1:
                return 'holiday_spring_duty'
            if int(info['block_length'][idx]) >= 4 and 2 <= int(info['block_position'][idx]) <= 5:
                return 'holiday_long_core'
            return 'holiday_general'
        pre_n = int(info['pre_holiday_n_day'][idx])
        if pre_n == 1:
            return 'pre_holiday_1d'
        if pre_n == 2:
            return 'pre_holiday_2d'
        post_n = int(info['post_holiday_workday_n'][idx])
        if post_n == 1:
            return 'post_holiday_workday_1'
        if post_n == 2:
            return 'post_holiday_workday_2'
        if post_n == 3:
            return 'post_holiday_workday_3'
        return 'non_holiday'

    for idx in range(len(dates)):
        rows.append({
            'holiday_block_length': int(info['block_length'][idx]),
            'holiday_block_position': int(info['block_position'][idx]),
            'pre_holiday_n_day': int(info['pre_holiday_n_day'][idx]),
            'post_holiday_workday_n': int(info['post_holiday_workday_n'][idx]),
            'is_holiday_non_makeup': int(info['is_holiday_non_makeup'][idx]),
            'is_spring_shutdown': int(spring_info['is_spring_shutdown'][idx]),
            'is_spring_duty': int(spring_info['is_spring_duty'][idx]),
            'holiday_layer': _classify_holiday_layer(idx),
        })
    return rows


def is_china_holiday(dt):
    dt = pd.to_datetime(dt).date()
    if HAS_CN_CALENDAR and cn_calendar is not None:
        # chinese_calendar.is_holiday includes regular weekends.
        # For business buckets we only treat statutory holidays as holiday.
        if hasattr(cn_calendar, 'get_holiday_detail'):
            try:
                is_holiday, holiday_label = cn_calendar.get_holiday_detail(dt)
                return bool(is_holiday and holiday_label is not None)
            except Exception:
                pass
        try:
            if bool(cn_calendar.is_workday(dt)):
                return False
        except Exception:
            return False
        return dt.weekday() < 5
    return False


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


def get_holiday_detail_label(dt):
    dt = pd.to_datetime(dt).date()
    if HAS_CN_CALENDAR and cn_calendar is not None and hasattr(cn_calendar, 'get_holiday_detail'):
        try:
            _, holiday_label = cn_calendar.get_holiday_detail(dt)
            return str(holiday_label) if holiday_label is not None else None
        except Exception:
            return None
    return None


def classify_spring_festival_service_phases(date_series):
    dates = pd.Series(pd.to_datetime(date_series)).reset_index(drop=True)
    n_rows = len(dates)
    is_spring_festival = np.zeros(n_rows, dtype=int)
    is_spring_holiday_non_makeup = np.zeros(n_rows, dtype=int)
    shutdown_phase = np.zeros(n_rows, dtype=int)
    duty_phase = np.zeros(n_rows, dtype=int)

    labels = [get_holiday_detail_label(dt) for dt in dates]
    for idx, label in enumerate(labels):
        if label == 'Spring Festival':
            is_spring_festival[idx] = 1
            if is_china_holiday(dates.iloc[idx]) and (not is_china_makeup_workday(dates.iloc[idx])):
                is_spring_holiday_non_makeup[idx] = 1

    start = 0
    while start < n_rows:
        if is_spring_holiday_non_makeup[start] != 1:
            start += 1
            continue

        end = start
        while end + 1 < n_rows and is_spring_holiday_non_makeup[end + 1] == 1:
            end += 1

        block_positions = np.arange(1, end - start + 2)

        rule_year = int(dates.iloc[start].year)
        rule = SPRING_SERVICE_RULES.get('year_overrides', {}).get(
            rule_year,
            SPRING_SERVICE_RULES.get('default', DEFAULT_SPRING_SERVICE_RULES['default']),
        )
        shutdown_set = set(_normalize_position_list(rule.get('shutdown_positions', [])))
        duty_set = set(_normalize_position_list(rule.get('duty_positions', [])))

        shutdown_mask = np.array([int(pos in shutdown_set) for pos in block_positions], dtype=int)
        duty_mask = np.array([int(pos in duty_set) for pos in block_positions], dtype=int)
        shutdown_phase[start:end + 1] = shutdown_mask.astype(int)
        duty_phase[start:end + 1] = duty_mask.astype(int)

        start = end + 1

    date_overrides = SPRING_SERVICE_RULES.get('date_overrides', {})
    if date_overrides:
        for idx, dt in enumerate(dates):
            key = dt.date().isoformat()
            state = date_overrides.get(key)
            if state == 'shutdown':
                shutdown_phase[idx] = 1
                duty_phase[idx] = 0
            elif state == 'duty':
                shutdown_phase[idx] = 0
                duty_phase[idx] = 1
            elif state == 'none':
                shutdown_phase[idx] = 0
                duty_phase[idx] = 0

    return {
        'is_spring_festival': is_spring_festival,
        'is_spring_holiday_non_makeup': is_spring_holiday_non_makeup,
        'is_spring_shutdown': shutdown_phase,
        'is_spring_duty': duty_phase,
    }


def estimate_spring_festival_service_anchor(series_data, series_name):
    hist = series_data[['date', series_name]].copy()
    hist['date'] = pd.to_datetime(hist['date'])
    phase = classify_spring_festival_service_phases(hist['date'])
    hist['is_spring_shutdown'] = phase['is_spring_shutdown']
    hist['is_spring_duty'] = phase['is_spring_duty']

    shutdown_values = hist.loc[hist['is_spring_shutdown'] == 1, series_name].astype(float)
    duty_values = hist.loc[hist['is_spring_duty'] == 1, series_name].astype(float)

    shutdown_enabled = not shutdown_values.empty
    duty_enabled = not duty_values.empty

    return {
        'enabled': bool(shutdown_enabled or duty_enabled),
        'shutdown_enabled': shutdown_enabled,
        'shutdown_median': float(np.quantile(shutdown_values, 0.5)) if shutdown_enabled else 0.0,
        'shutdown_zero_ratio': float(np.mean(shutdown_values <= 1.0)) if shutdown_enabled else 0.0,
        'shutdown_strength': 0.95 if shutdown_enabled else 0.0,
        'duty_enabled': duty_enabled,
        'duty_median': float(np.quantile(duty_values, 0.5)) if duty_enabled else 0.0,
        'duty_p75': float(np.quantile(duty_values, 0.75)) if duty_enabled else 0.0,
        'duty_strength': 0.65 if duty_enabled else 0.0,
    }


def apply_spring_festival_service_adjustment(future_results, spring_anchor):
    if not bool(spring_anchor.get('enabled', False)):
        return future_results

    adjusted = future_results.copy()
    p10_adj = np.asarray(adjusted['p10'], dtype=float).copy()
    p50_adj = np.asarray(adjusted['p50'], dtype=float).copy()
    p90_adj = np.asarray(adjusted['p90'], dtype=float).copy()

    phase = classify_spring_festival_service_phases(adjusted['future_dates'])

    for idx in range(len(adjusted['future_dates'])):
        p10_now = float(p10_adj[idx])
        p50_now = float(p50_adj[idx])
        p90_now = float(p90_adj[idx])

        if phase['is_spring_shutdown'][idx] == 1 and bool(spring_anchor.get('shutdown_enabled', False)):
            strength = float(np.clip(spring_anchor.get('shutdown_strength', 0.0), 0.0, 1.0))
            median = float(max(0.0, spring_anchor.get('shutdown_median', 0.0)))
            p50_new = max(0.0, (1.0 - strength) * p50_now + strength * median)
            p10_new = max(0.0, min(p50_new, (1.0 - strength) * p10_now + strength * (0.5 * median)))
            p90_new = max(p50_new, (1.0 - strength) * p90_now + strength * max(median, 1.2 * median))
        elif phase['is_spring_duty'][idx] == 1 and bool(spring_anchor.get('duty_enabled', False)):
            strength = float(np.clip(spring_anchor.get('duty_strength', 0.0), 0.0, 1.0))
            median = float(max(0.0, spring_anchor.get('duty_median', p50_now)))
            p75 = float(max(median, spring_anchor.get('duty_p75', median)))
            p50_new = max(0.0, (1.0 - strength) * p50_now + strength * median)
            p10_new = max(0.0, min(p50_new, (1.0 - strength) * p10_now + strength * (0.75 * median)))
            p90_new = max(p50_new, (1.0 - strength) * p90_now + strength * p75)
        else:
            p10_new, p50_new, p90_new = p10_now, p50_now, p90_now

        p10_adj[idx] = p10_new
        p50_adj[idx] = p50_new
        p90_adj[idx] = p90_new

    adjusted['p10'] = p10_adj
    adjusted['p50'] = p50_adj
    adjusted['p90'] = p90_adj
    return adjusted


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

    series_tuning = get_series_tuning(series_name)
    post_holiday_strengths = series_tuning.get('post_holiday_anchor_strengths', {})

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
            default_strength=float(post_holiday_strengths.get('day1_default', 0.45)),
            strong_strength=float(post_holiday_strengths.get('day1_strong', 0.60)),
        ),
        'post_holiday_workday_2': _build_anchor_stats(
            hist.loc[holiday_block_info['post_holiday_workday_n'] == 2, series_name],
            default_strength=float(post_holiday_strengths.get('day2_default', 0.35)),
            strong_strength=float(post_holiday_strengths.get('day2_strong', 0.50)),
        ),
        'post_holiday_workday_3': _build_anchor_stats(
            hist.loc[holiday_block_info['post_holiday_workday_n'] == 3, series_name],
            default_strength=float(post_holiday_strengths.get('day3_default', 0.25)),
            strong_strength=float(post_holiday_strengths.get('day3_strong', 0.40)),
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


def apply_spring_festival_service_adjustment_to_backtest(bt_results, spring_anchor):
    if not bool(spring_anchor.get('enabled', False)):
        return bt_results

    adjusted = dict(bt_results)
    dates = pd.to_datetime(adjusted['dates'])
    predicted = np.asarray(adjusted['predicted'], dtype=float).copy()
    phase = classify_spring_festival_service_phases(dates)

    for idx in range(len(dates)):
        pred_now = float(predicted[idx])
        if phase['is_spring_shutdown'][idx] == 1 and bool(spring_anchor.get('shutdown_enabled', False)):
            strength = float(np.clip(spring_anchor.get('shutdown_strength', 0.0), 0.0, 1.0))
            median = float(max(0.0, spring_anchor.get('shutdown_median', 0.0)))
            pred_new = max(0.0, (1.0 - strength) * pred_now + strength * median)
        elif phase['is_spring_duty'][idx] == 1 and bool(spring_anchor.get('duty_enabled', False)):
            strength = float(np.clip(spring_anchor.get('duty_strength', 0.0), 0.0, 1.0))
            median = float(max(0.0, spring_anchor.get('duty_median', pred_now)))
            pred_new = max(0.0, (1.0 - strength) * pred_now + strength * median)
        else:
            pred_new = pred_now
        predicted[idx] = pred_new

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


def predict_holiday_segment_adjustment(dt, base_pred, residual_adjuster, holiday_context=None):
    segment_bundle = residual_adjuster.get('holiday_segment_model')
    if segment_bundle is None:
        return None

    dt = pd.to_datetime(dt)
    month_phase = get_month_phase(dt)
    month_phase_code = {'month_start': 0, 'month_mid': 1, 'month_end': 2}.get(month_phase, 1)

    holiday_context = holiday_context or {}
    row = {
        'dow': int(dt.dayofweek),
        'month_phase_code': int(month_phase_code),
        'is_month_end_settlement': int(dt.day >= 26),
        'is_holiday': int(is_china_holiday(dt)),
        'is_makeup_workday': int(is_china_makeup_workday(dt)),
        'predicted': float(base_pred),
        'holiday_block_length': int(holiday_context.get('holiday_block_length', 0)),
        'holiday_block_position': int(holiday_context.get('holiday_block_position', 0)),
        'pre_holiday_n_day': int(holiday_context.get('pre_holiday_n_day', 0)),
        'post_holiday_workday_n': int(holiday_context.get('post_holiday_workday_n', 0)),
    }

    feature_cols = segment_bundle.get('feature_cols', [])
    if not feature_cols:
        return None

    x = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
    model = segment_bundle['model']
    return float(model.predict(x)[0])


def predict_holiday_layered_adjustment(dt, base_pred, residual_adjuster, holiday_context=None):
    layered_models = residual_adjuster.get('holiday_layered_models', {})
    layer_bias_map = residual_adjuster.get('holiday_layer_bias', {})
    holiday_context = holiday_context or {}
    layer_name = str(holiday_context.get('holiday_layer', 'non_holiday'))
    if layer_name == 'non_holiday':
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
        'holiday_block_length': int(holiday_context.get('holiday_block_length', 0)),
        'holiday_block_position': int(holiday_context.get('holiday_block_position', 0)),
        'pre_holiday_n_day': int(holiday_context.get('pre_holiday_n_day', 0)),
        'post_holiday_workday_n': int(holiday_context.get('post_holiday_workday_n', 0)),
    }

    layer_bundle = layered_models.get(layer_name)
    if layer_bundle is None:
        if layer_name in layer_bias_map:
            return float(layer_bias_map[layer_name])
        return None

    feature_cols = layer_bundle.get('feature_cols', [])
    if not feature_cols:
        return float(layer_bundle.get('fallback_bias', 0.0))

    x = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])
    model = layer_bundle['model']
    return float(model.predict(x)[0])


def compute_residual_adjustment(dt, residual_adjuster, base_pred=None, holiday_context=None):
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
        mixed_adjustment = bias_adjustment
    else:
        model_weight = float(np.clip(residual_adjuster.get('weight_state_model', 0.7), 0.0, 1.0))
        mixed_adjustment = model_weight * model_adjustment + (1.0 - model_weight) * bias_adjustment

    holiday_model_adjustment = predict_holiday_segment_adjustment(
        dt,
        float(base_pred),
        residual_adjuster,
        holiday_context=holiday_context,
    )
    if holiday_model_adjustment is not None:
        should_apply_holiday_model = (
            state_name == 'holiday_or_makeup' or
            int((holiday_context or {}).get('pre_holiday_n_day', 0)) > 0 or
            int((holiday_context or {}).get('post_holiday_workday_n', 0)) > 0
        )
        if should_apply_holiday_model:
            holiday_weight = float(np.clip(residual_adjuster.get('weight_holiday_segment_model', 0.75), 0.0, 1.0))
            mixed_adjustment = holiday_weight * holiday_model_adjustment + (1.0 - holiday_weight) * mixed_adjustment

    layered_adjustment = predict_holiday_layered_adjustment(
        dt,
        float(base_pred),
        residual_adjuster,
        holiday_context=holiday_context,
    )
    if layered_adjustment is not None:
        layered_weight = float(np.clip(residual_adjuster.get('weight_holiday_layered_model', 0.55), 0.0, 1.0))
        mixed_adjustment = layered_weight * layered_adjustment + (1.0 - layered_weight) * mixed_adjustment

    clip_std = float(residual_adjuster.get('residual_std', 0.0))
    if clip_std > 0:
        mixed_adjustment = float(np.clip(mixed_adjustment, -3.0 * clip_std, 3.0 * clip_std))
    return mixed_adjustment


def apply_residual_adjustment(future_results, residual_adjuster):
    p10_adj = []
    p50_adj = []
    p90_adj = []

    holiday_context_rows = build_holiday_context_rows(future_results['future_dates'])
    for idx, dt in enumerate(future_results['future_dates']):
        adjustment = compute_residual_adjustment(
            dt,
            residual_adjuster,
            base_pred=float(future_results['p50'][idx]),
            holiday_context=holiday_context_rows[idx],
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
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        for idx, (dt, y_true, y_pred) in enumerate(zip(dates, actual, predicted), start=1):
            residual_after_base = float(
                y_true - max(
                    0.0,
                    y_pred + compute_residual_adjustment(
                        dt,
                        residual_adjuster,
                        base_pred=float(y_pred),
                        holiday_context=holiday_context_rows[idx - 1],
                    ),
                )
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
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        corrected_pred = []
        for idx, (dt, pred) in enumerate(zip(dates, predicted), start=1):
            residual_adj = compute_residual_adjustment(
                dt,
                residual_adjuster,
                base_pred=float(pred),
                holiday_context=holiday_context_rows[idx - 1],
            )
            lead_adj = float(lead_bias.get(idx, global_lead_bias))
            final_pred = max(0.0, float(pred) + residual_adj + leadwise_weight * lead_adj)
            corrected_pred.append(final_pred)

        corrected_pred = np.asarray(corrected_pred, dtype=float)
        base_pred = np.asarray(
            [
                max(
                    0.0,
                    float(predicted[base_idx]) + compute_residual_adjustment(
                        dates[base_idx],
                        residual_adjuster,
                        base_pred=float(predicted[base_idx]),
                        holiday_context=holiday_context_rows[base_idx],
                    ),
                )
                for base_idx in range(len(predicted))
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
    leadwise_weight_cap = float(np.clip(leadwise_adjuster.get('leadwise_weight_cap', 1.0), 0.0, 1.0))
    candidates = np.linspace(0.0, leadwise_weight_cap, num=search_points)

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


def _classify_monitor_bucket(dt, holiday_context):
    dt = pd.to_datetime(dt)
    holiday_context = holiday_context or {}

    if is_china_holiday(dt) and (not is_china_makeup_workday(dt)):
        return 'holiday'

    post_n = int(holiday_context.get('post_holiday_workday_n', 0))
    if 1 <= post_n <= 3:
        return 'post_holiday_workday_1_3'

    if is_china_makeup_workday(dt):
        return 'makeup_workday'

    if get_operational_state(dt) == 'workday':
        return 'workday_normal'
    if get_operational_state(dt) == 'weekend':
        return 'weekend'
    return 'makeup_workday'


def _classify_fusion_bucket(dt, holiday_context=None, use_holiday_layers=False):
    base_bucket = _classify_monitor_bucket(dt, holiday_context)
    if (not use_holiday_layers) or base_bucket != 'holiday':
        return base_bucket

    holiday_layer = str((holiday_context or {}).get('holiday_layer', 'holiday_general'))
    if holiday_layer in {'holiday_spring_shutdown', 'holiday_spring_duty', 'holiday_long_core', 'holiday_general'}:
        return holiday_layer
    return 'holiday'


def _build_bucket_bias_snapshot_from_detail(detail_df, recent_days=84):
    if detail_df is None or detail_df.empty:
        return {}

    max_date = pd.to_datetime(detail_df['date']).max()
    cutoff = max_date - pd.Timedelta(days=max(1, int(recent_days)) - 1)
    scoped_frames = [
        ('all', detail_df),
        ('recent', detail_df.loc[detail_df['date'] >= cutoff].copy()),
    ]

    snapshot = {}
    for scope_name, scope_df in scoped_frames:
        if scope_df.empty:
            continue
        bucket_map = {}
        for bucket_name, group in scope_df.groupby('bucket'):
            y_true = group['actual'].to_numpy(dtype=float)
            y_pred = group['predicted'].to_numpy(dtype=float)
            _, bucket_rmse = calculate_metrics(y_true, y_pred)
            bucket_map[str(bucket_name)] = {
                'sample_size': int(len(group)),
                'bias': float(np.mean(y_pred - y_true)),
                'rmse': float(bucket_rmse),
            }
        snapshot[str(scope_name)] = bucket_map
    return snapshot


def _apply_bucket_bias_gate_to_backtest_detail(detail_df, series_tuning, recent_days=84):
    if detail_df is None or detail_df.empty:
        return detail_df
    gate_cfg = (series_tuning or {}).get('bias_gate', {})
    if not bool(gate_cfg.get('enabled', False)):
        return detail_df
    if not bool(gate_cfg.get('apply_to_backtest', True)):
        return detail_df

    snapshot = _build_bucket_bias_snapshot_from_detail(detail_df, recent_days=recent_days)
    if not snapshot:
        return detail_df

    proxy_future = {
        'future_dates': pd.to_datetime(detail_df['date']).to_numpy(),
        'p10': detail_df['predicted'].to_numpy(dtype=float),
        'p50': detail_df['predicted'].to_numpy(dtype=float),
        'p90': detail_df['predicted'].to_numpy(dtype=float),
    }
    adjusted_proxy, gate_info = apply_bucket_bias_gate(proxy_future, snapshot, series_tuning)
    if int(gate_info.get('applied_count', 0)) <= 0:
        return detail_df

    adjusted_df = detail_df.copy()
    adjusted_df['predicted'] = np.asarray(adjusted_proxy['p50'], dtype=float)
    return adjusted_df


def build_monitor_bucket_report(bt_results, residual_adjuster, leadwise_adjuster=None, series_name=None, recent_days=84, series_tuning=None):
    windows = bt_results.get('all_windows', [])
    if not windows:
        windows = [bt_results]

    records = []
    for window in windows:
        dates = pd.to_datetime(window['dates'])
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        for idx, (dt, y_true, pred) in enumerate(zip(dates, actual, predicted), start=1):
            residual_adj = compute_residual_adjustment(
                dt,
                residual_adjuster,
                base_pred=float(pred),
                holiday_context=holiday_context_rows[idx - 1],
            )
            lead_adj = 0.0
            if leadwise_adjuster is not None:
                lead_adj = float(leadwise_adjuster.get('lead_bias', {}).get(idx, leadwise_adjuster.get('global_lead_bias', 0.0)))
                lead_adj *= float(np.clip(leadwise_adjuster.get('leadwise_weight', 1.0), 0.0, 1.0))

            y_pred = max(0.0, float(pred) + residual_adj + lead_adj)
            records.append({
                'date': pd.to_datetime(dt),
                'actual': float(y_true),
                'predicted': float(y_pred),
                'bucket': _classify_monitor_bucket(dt, holiday_context_rows[idx - 1]),
            })

    if not records:
        return pd.DataFrame()

    detail_df = pd.DataFrame(records)
    if series_tuning is not None:
        detail_df = _apply_bucket_bias_gate_to_backtest_detail(
            detail_df,
            series_tuning=series_tuning,
            recent_days=recent_days,
        )
    max_date = pd.to_datetime(detail_df['date']).max()
    cutoff = max_date - pd.Timedelta(days=max(1, int(recent_days)) - 1)

    scoped_frames = [
        ('all', detail_df),
        ('recent', detail_df.loc[detail_df['date'] >= cutoff].copy()),
    ]

    rows = []
    for scope_name, scope_df in scoped_frames:
        if scope_df.empty:
            continue
        for bucket_name, group in scope_df.groupby('bucket'):
            y_true = group['actual'].to_numpy(dtype=float)
            y_pred = group['predicted'].to_numpy(dtype=float)
            smape, rmse = calculate_metrics(y_true, y_pred)
            rows.append({
                'series_name': str(series_name) if series_name is not None else '',
                'scope': str(scope_name),
                'bucket': str(bucket_name),
                'sample_size': int(len(group)),
                'smape': float(smape),
                'rmse': float(rmse),
                'mae': float(np.mean(np.abs(y_true - y_pred))),
                'bias': float(np.mean(y_pred - y_true)),
                'mean_actual': float(np.mean(y_true)),
                'mean_predicted': float(np.mean(y_pred)),
            })

    return pd.DataFrame(rows)


def build_history_bucket_profile(history_df, target_col, series_name=None, recent_days=84):
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    if 'date' not in history_df.columns or target_col not in history_df.columns:
        return pd.DataFrame()

    profile_df = history_df[['date', target_col]].copy()
    profile_df = profile_df.dropna(subset=[target_col]).reset_index(drop=True)
    if profile_df.empty:
        return pd.DataFrame()

    profile_df['date'] = pd.to_datetime(profile_df['date'])
    holiday_context_rows = build_holiday_context_rows(profile_df['date'])
    profile_df['bucket'] = [
        _classify_monitor_bucket(dt, holiday_context_rows[idx])
        for idx, dt in enumerate(profile_df['date'])
    ]

    max_date = pd.to_datetime(profile_df['date']).max()
    cutoff = max_date - pd.Timedelta(days=max(1, int(recent_days)) - 1)
    scoped_frames = [
        ('history_all', profile_df),
        ('history_recent', profile_df.loc[profile_df['date'] >= cutoff].copy()),
    ]

    rows = []
    for scope_name, scope_df in scoped_frames:
        if scope_df.empty:
            continue
        for bucket_name, group in scope_df.groupby('bucket'):
            values = group[target_col].to_numpy(dtype=float)
            rows.append({
                'series_name': str(series_name) if series_name is not None else str(target_col),
                'scope': str(scope_name),
                'bucket': str(bucket_name),
                'hist_sample_size': int(len(group)),
                'hist_mean': float(np.mean(values)),
                'hist_std': float(np.std(values, ddof=0)),
                'hist_zero_ratio': float(np.mean(values <= 0.0)),
                'hist_min': float(np.min(values)),
                'hist_max': float(np.max(values)),
            })

    return pd.DataFrame(rows)


def build_bucket_bias_snapshot(bt_results, residual_adjuster, leadwise_adjuster=None, recent_days=84):
    report_df = build_monitor_bucket_report(
        bt_results,
        residual_adjuster,
        leadwise_adjuster=leadwise_adjuster,
        series_name=None,
        recent_days=recent_days,
    )
    if report_df.empty:
        return {}

    snapshot = {}
    for _, row in report_df.iterrows():
        scope = str(row.get('scope', 'all'))
        bucket = _normalize_bucket_name(row.get('bucket', 'makeup_workday'))
        bucket_map = snapshot.setdefault(scope, {})
        bucket_map[bucket] = {
            'bias': float(row.get('bias', 0.0)),
            'sample_size': int(row.get('sample_size', 0)),
            'smape': float(row.get('smape', 0.0)),
            'rmse': float(row.get('rmse', 0.0)),
        }
    return snapshot


def build_stage_transition_frame(
    chronos_bt_results,
    direct_bt_results,
    fusion_policy,
    residual_adjuster,
    leadwise_adjuster=None,
    series_name=None,
    latest_only=True,
):
    windows_chronos = chronos_bt_results.get('all_windows', [chronos_bt_results])
    windows_direct = direct_bt_results.get('all_windows', [direct_bt_results])
    if len(windows_chronos) != len(windows_direct):
        return pd.DataFrame()

    paired_windows = list(zip(windows_chronos, windows_direct))
    if latest_only and paired_windows:
        paired_windows = paired_windows[:1]

    rows = []
    for window_idx, (chronos_window, direct_window) in enumerate(paired_windows, start=1):
        dates = pd.to_datetime(chronos_window['dates'])
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(chronos_window['actual'], dtype=float)
        chronos_pred = np.asarray(chronos_window['predicted'], dtype=float)
        direct_pred = np.asarray(direct_window['predicted'], dtype=float)

        for idx, dt in enumerate(dates):
            fusion_weight = resolve_fusion_weight(dt, fusion_policy, holiday_context=holiday_context_rows[idx])
            fused_pred = float(fusion_weight) * float(chronos_pred[idx]) + (1.0 - float(fusion_weight)) * float(direct_pred[idx])

            residual_adj = compute_residual_adjustment(
                dt,
                residual_adjuster,
                base_pred=float(fused_pred),
                holiday_context=holiday_context_rows[idx],
            )
            residual_pred = max(0.0, float(fused_pred) + float(residual_adj))

            lead_adjust = 0.0
            if leadwise_adjuster is not None:
                lead_bias = float(leadwise_adjuster.get('lead_bias', {}).get(idx + 1, leadwise_adjuster.get('global_lead_bias', 0.0)))
                lead_weight = float(np.clip(leadwise_adjuster.get('leadwise_weight', 1.0), 0.0, 1.0))
                lead_adjust = lead_weight * lead_bias
            final_pred = max(0.0, float(residual_pred) + float(lead_adjust))

            rows.append({
                'series_name': str(series_name) if series_name is not None else '',
                'window_index': int(window_idx),
                'lead': int(idx + 1),
                'date': pd.to_datetime(dt),
                'actual': float(actual[idx]),
                'chronos_pred': float(chronos_pred[idx]),
                'direct_pred': float(direct_pred[idx]),
                'fused_pred': float(fused_pred),
                'residual_pred': float(residual_pred),
                'final_pred': float(final_pred),
            })

    return pd.DataFrame(rows)


def build_stage_transition_summary(stage_df):
    if stage_df is None or stage_df.empty:
        return pd.DataFrame()

    stage_specs = [
        ('chronos_pred', 'Chronos基线'),
        ('direct_pred', 'Direct基线'),
        ('fused_pred', '融合后'),
        ('residual_pred', 'Residual后'),
        ('final_pred', 'Lead-wise后(最终)'),
    ]

    rows = []
    for series_name, group in stage_df.groupby('series_name'):
        previous = None
        y_true = group['actual'].to_numpy(dtype=float)
        for column_name, stage_name in stage_specs:
            if column_name not in group.columns:
                continue
            y_pred = group[column_name].to_numpy(dtype=float)
            smape, rmse = calculate_metrics(y_true, y_pred)
            mae = float(np.mean(np.abs(y_true - y_pred)))
            bias = float(np.mean(y_pred - y_true))

            row = {
                'series_name': str(series_name),
                'stage': str(stage_name),
                'smape': float(smape),
                'rmse': float(rmse),
                'mae': float(mae),
                'bias': float(bias),
                'mean_predicted': float(np.mean(y_pred)),
                'sample_size': int(len(group)),
                'smape_delta_vs_prev': None,
                'rmse_delta_vs_prev': None,
                'mae_delta_vs_prev': None,
            }
            if previous is not None:
                row['smape_delta_vs_prev'] = float(row['smape'] - previous['smape'])
                row['rmse_delta_vs_prev'] = float(row['rmse'] - previous['rmse'])
                row['mae_delta_vs_prev'] = float(row['mae'] - previous['mae'])
            rows.append(row)
            previous = row

    return pd.DataFrame(rows)


BUCKET_DISPLAY_NAMES = {
    'workday_normal': '普通工作日',
    'weekend': '周末',
    'holiday': '法定节假日',
    'post_holiday_workday_1_3': '节后1-3个工作日',
    'makeup_workday': '补班日',
    'other': '补班日',
}

SERIES_DISPLAY_NAMES = {
    'call_volume': '电话量',
    'tickets_received': '工单量',
}

STAGE_DISPLAY_NAMES = {
    'direct_fusion': '基线融合',
    'residual': '偏差修正',
    'leadwise': '预测步长修正',
}

TERM_GLOSSARY = {
    'sMAPE': '对误差比例指标，越低越好；适合包含低值或零值场景。',
    'RMSE': '均方根误差，对大误差更敏感；越低越好。',
    'MAE': '平均绝对误差，表示平均每个点偏离多少；越低越好。',
    'Bias': '平均偏差，正值常表示高估、负值常表示低估；越接近0越稳。',
    'P50': '中位预测，通常可理解为“最可能值”。',
    'P10/P90': '预测区间下沿/上沿，用于表示不确定性范围。',
    'recent': '最近窗口口径（本报告默认最近84天）',
    'all': '全历史口径（当前数据表全部历史区间）',
}

PIPELINE_STAGE_EXPLANATIONS = {
    '基线融合': '将 Chronos 基线预测与 Direct 基线预测按场景权重融合，得到更稳的基础预测。',
    '偏差修正': '基于回测残差做系统性偏差校正，重点修正长期高估/低估问题。',
    '预测步长修正': '按预测步长(lead)做分层修正，处理“离预测日越远误差越大”的现象。',
    'Chronos基线': '仅使用 Chronos 模型输出的初始预测结果。',
    'Direct基线': '仅使用监督学习 Direct 分支输出的初始预测结果。',
    '融合后': 'Chronos 与 Direct 按策略融合后的结果。',
    'Residual后': '在融合结果上完成残差偏差修正后的结果。',
    'Lead-wise后(最终)': '在 Residual 后继续做按步长修正后的最终结果。',
}

PIPELINE_STAGE_FLOW_DETAILS = {
    'Chronos基线': (
        '输入=历史时序上下文；流程=直接由 Chronos 生成分位数预测（p10/p50/p90）；'
        '产出=大模型原始基线，通常覆盖面广但在节假日极端场景可能偏保守或偏离。'
    ),
    'Direct基线': (
        '输入=监督学习特征（日期、状态、节假日邻近特征、滞后项等）；流程=多步监督分支逐 lead 训练/推断；'
        '产出=结构化基线，对近期模式与业务特征更敏感。'
    ),
    '基线融合': (
        '输入=Chronos基线 + Direct基线；流程=按全局/分桶策略搜索并应用融合权重；'
        '产出=更稳的基础预测，降低单一路径偏差。'
    ),
    '融合后': (
        '输入=双基线预测；流程=执行权重融合并形成统一预测序列；'
        '产出=后续偏差修正和步长修正的统一输入。'
    ),
    '偏差修正': (
        '输入=回测残差明细（actual-predicted）与节假日上下文；流程=学习全局/状态/周几/月相位偏差 + '
        '状态小模型(Ridge) + 节假日分段模型 + 分层节假日模型，并按权重融合、再做3σ截断；'
        '产出=Residual后结果，主要消除系统性高估/低估。'
    ),
    'Residual后': (
        '输入=融合后预测 + 残差学习器；流程=逐天应用残差校正，同时叠加节假日零值锚定、春节停工/值班锚定、'
        '节前/节后恢复锚点；产出=完成场景偏差治理后的中间结果。'
    ),
    '预测步长修正': (
        '输入=Residual后预测 + 各 lead 的历史偏差；流程=按预测步长(lead)学习分层偏差并搜索最佳修正强度；'
        '产出=修正远期步长误差扩散，降低“越往后越偏”的问题。'
    ),
    'Lead-wise后(最终)': (
        '输入=Residual后结果 + lead-wise 校正器；流程=逐步长施加校正并受动态cap/护栏约束；'
        '产出=最终线上口径预测结果。'
    ),
}


def _format_percent(value):
    return f"{float(value):.2%}"


def _format_number(value):
    return f"{float(value):.2f}"


def _normalize_bucket_name(bucket_name):
    bucket_name = str(bucket_name)
    if bucket_name == 'other':
        return 'makeup_workday'
    return bucket_name


def _classify_smape_level(smape):
    smape = float(smape)
    if smape <= 0.12:
        return '表现较稳，可作为上线基线'
    if smape <= 0.18:
        return '表现中等偏稳，建议持续观察'
    if smape <= 0.25:
        return '波动偏大，需要重点监控'
    return '风险较高，不建议无保护直接放量'


def _classify_bias_direction(bias, mean_actual):
    denominator = max(abs(float(mean_actual)), 1.0)
    ratio = float(bias) / denominator
    if ratio >= 0.05:
        return '整体偏高估'
    if ratio <= -0.05:
        return '整体偏低估'
    return '整体偏差可控'


def _bucket_sort_key(bucket_name):
    bucket_name = _normalize_bucket_name(bucket_name)
    bucket_order = {
        'workday_normal': 0,
        'weekend': 1,
        'holiday': 2,
        'post_holiday_workday_1_3': 3,
        'makeup_workday': 4,
        'other': 4,
    }
    return bucket_order.get(str(bucket_name), 99)


def _build_markdown_table(df, columns, rename_map=None, formatters=None):
    if df is None or df.empty:
        return ['无可用数据。']

    rename_map = rename_map or {}
    formatters = formatters or {}
    table_df = df.loc[:, columns].copy()
    table_df = table_df.rename(columns=rename_map)

    for column_name, formatter in formatters.items():
        target_name = rename_map.get(column_name, column_name)
        if target_name in table_df.columns:
            table_df[target_name] = table_df[target_name].map(
                lambda value: formatter(value) if pd.notna(value) else ''
            )

    table_df = table_df.fillna('')
    headers = list(table_df.columns)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for _, row in table_df.iterrows():
        values = [str(row[header]).replace('\n', ' ') for header in headers]
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def _append_markdown_table(lines, df, columns, rename_map=None, formatters=None):
    lines.append('')
    lines.extend(
        _build_markdown_table(
            df,
            columns=columns,
            rename_map=rename_map,
            formatters=formatters,
        )
    )
    lines.append('')


def build_forecast_interpretation_report(
    tuning_report_df,
    monitor_report_df,
    monitor_history_profile_df,
    forecast_export_df,
    stage_transition_summary_df,
    image_links,
    monitor_low_sample_threshold=12,
    monitor_recent_days=84,
    generated_at=None,
):
    generated_at = generated_at or datetime.now()
    lines = [
        '# 业务预测解读报告',
        '',
        '## 先看这页',
        '',
        f"- 生成时间：{generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
        '- 本报告面向业务查看：先看结论与风险，再看样本口径，最后再看分场景和未来预测。',
        '- 口径定义：回测样本数用于看模型近期准不准；历史样本数用于看该场景是否常见、结论是否稳。',
        f'- 窗口范围：recent 表示最近 {int(monitor_recent_days)} 天；all 表示当前数据表中的全历史。',
        '',
        '### 专业名词速查',
        '',
        f"- sMAPE：{TERM_GLOSSARY['sMAPE']}",
        f"- RMSE：{TERM_GLOSSARY['RMSE']}",
        f"- MAE：{TERM_GLOSSARY['MAE']}",
        f"- Bias：{TERM_GLOSSARY['Bias']}",
        f"- P50：{TERM_GLOSSARY['P50']}",
        f"- P10/P90：{TERM_GLOSSARY['P10/P90']}",
        f"- recent/all：{TERM_GLOSSARY['recent']}；{TERM_GLOSSARY['all']}",
        '',
        '### 调整环节说明',
        '',
        f"- Chronos基线：{PIPELINE_STAGE_FLOW_DETAILS['Chronos基线']}",
        f"- Direct基线：{PIPELINE_STAGE_FLOW_DETAILS['Direct基线']}",
        f"- 基线融合：{PIPELINE_STAGE_FLOW_DETAILS['基线融合']}",
        f"- 融合后：{PIPELINE_STAGE_FLOW_DETAILS['融合后']}",
        f"- 偏差修正：{PIPELINE_STAGE_FLOW_DETAILS['偏差修正']}",
        f"- Residual后：{PIPELINE_STAGE_FLOW_DETAILS['Residual后']}",
        f"- 预测步长修正：{PIPELINE_STAGE_FLOW_DETAILS['预测步长修正']}",
        f"- Lead-wise后(最终)：{PIPELINE_STAGE_FLOW_DETAILS['Lead-wise后(最终)']}",
        '',
        '## 一页结论',
        '',
    ]

    monitor_low_sample_threshold = max(1, int(monitor_low_sample_threshold))

    recent_monitor_df = pd.DataFrame()
    if monitor_report_df is not None and not monitor_report_df.empty:
        recent_monitor_df = monitor_report_df.loc[monitor_report_df['scope'] == 'recent'].copy()
        if recent_monitor_df.empty:
            recent_monitor_df = monitor_report_df.copy()
    if not recent_monitor_df.empty and 'bucket' in recent_monitor_df.columns:
        recent_monitor_df['bucket'] = recent_monitor_df['bucket'].map(_normalize_bucket_name)

    if recent_monitor_df.empty:
        lines.append('- 当前缺少分桶监控结果，本次仅能输出基础预测文件，无法对工作日、周末、节假日进行稳定性评价。')
    else:
        for series_name in ['call_volume', 'tickets_received']:
            series_df = recent_monitor_df.loc[recent_monitor_df['series_name'] == series_name].copy()
            if series_df.empty:
                continue
            total_sample = int(series_df['sample_size'].sum())
            weighted_smape = float(np.average(series_df['smape'], weights=series_df['sample_size']))
            weighted_bias = float(np.average(series_df['bias'], weights=series_df['sample_size']))
            mean_actual = float(np.average(series_df['mean_actual'], weights=series_df['sample_size']))
            worst_row = series_df.sort_values(['smape', 'sample_size'], ascending=[False, False]).iloc[0]
            lines.append(f"### {SERIES_DISPLAY_NAMES.get(series_name, series_name)}")
            lines.append('')
            lines.append(
                '- '
                f"近期整体表现：最近{int(monitor_recent_days)}天分场景加权 sMAPE 为 {_format_percent(weighted_smape)}，可用回测样本 {total_sample}，"
                f"{_classify_smape_level(weighted_smape)}。"
            )
            lines.append(
                '- '
                f"整体偏差方向：{_classify_bias_direction(weighted_bias, mean_actual)}，"
                f"加权 Bias {_format_number(weighted_bias)}。"
            )
            lines.append(
                '- '
                f"当前最需要关注的场景：{BUCKET_DISPLAY_NAMES.get(str(worst_row['bucket']), str(worst_row['bucket']))}，"
                f"该场景 sMAPE {_format_percent(worst_row['smape'])}，回测样本 {int(worst_row['sample_size'])}。"
            )
            lines.append('')

    lines.append('## 风险提示')
    lines.append('')
    if recent_monitor_df.empty:
        lines.append('- 当前缺少可用于业务解读的风险数据。')
    else:
        risk_found = False
        for series_name in ['call_volume', 'tickets_received']:
            series_df = recent_monitor_df.loc[recent_monitor_df['series_name'] == series_name].copy()
            if series_df.empty:
                continue
            for _, row in series_df.iterrows():
                bucket_name = BUCKET_DISPLAY_NAMES.get(str(row['bucket']), str(row['bucket']))
                if int(row['sample_size']) < monitor_low_sample_threshold:
                    risk_found = True
                    lines.append(
                        '- '
                        f"{SERIES_DISPLAY_NAMES.get(series_name, series_name)} - {bucket_name}：回测样本数仅 {int(row['sample_size'])}，"
                        '属于低置信度场景，适合观察，不适合直接下强结论。'
                    )
                elif float(row['smape']) > 0.25:
                    risk_found = True
                    lines.append(
                        '- '
                        f"{SERIES_DISPLAY_NAMES.get(series_name, series_name)} - {bucket_name}：近期误差偏高，"
                        f"sMAPE 为 {_format_percent(row['smape'])}，建议优先关注。"
                    )
        if not risk_found:
            lines.append('- 当前未发现明显的高风险场景，主要场景可继续按现有节奏跟踪。')

    if tuning_report_df is not None and not tuning_report_df.empty:
        lines.append('')
        lines.append('## 调整效果')
        lines.append('')
        lines.append('- 基线融合：先把 Chronos 与 Direct 两条基线按搜索到的最优权重做合并，得到统一基线。')
        lines.append('- 偏差修正：在统一基线上，用回测残差学习器（含状态/节假日分层）修正系统性偏差。')
        lines.append('- 预测步长修正：对不同预测步长分别做偏差校正，处理远期步长误差放大。')
        lines.append('')
        for series_name in ['call_volume', 'tickets_received']:
            series_df = tuning_report_df.loc[tuning_report_df['series_name'] == series_name].copy()
            if series_df.empty:
                continue
            if 'smape_delta' not in series_df.columns and {'smape_before', 'smape_after'}.issubset(series_df.columns):
                series_df['smape_delta'] = series_df['smape_after'] - series_df['smape_before']
            if 'rmse_delta' not in series_df.columns and {'rmse_before', 'rmse_after'}.issubset(series_df.columns):
                series_df['rmse_delta'] = series_df['rmse_after'] - series_df['rmse_before']

            lines.append(f"### {SERIES_DISPLAY_NAMES.get(series_name, series_name)}")
            lines.append('')

            if 'smape_delta' in series_df.columns:
                best_row = series_df.sort_values('smape_delta').iloc[0]
                lines.append(
                    f"- 本次改善最大的调整环节是{STAGE_DISPLAY_NAMES.get(str(best_row['stage']), str(best_row['stage']))}，"
                    f"sMAPE 变化 {_format_percent(best_row['smape_delta'])}，RMSE 变化 {_format_number(best_row.get('rmse_delta', 0.0))}。"
                )

            stage_table_df = series_df.copy()
            stage_table_df['stage'] = stage_table_df['stage'].map(lambda value: STAGE_DISPLAY_NAMES.get(str(value), str(value)))
            table_columns = ['stage', 'smape_before', 'smape_after', 'smape_delta', 'rmse_before', 'rmse_after', 'rmse_delta']
            if not set(table_columns).issubset(stage_table_df.columns):
                table_columns = [col for col in table_columns if col in stage_table_df.columns]

            if table_columns:
                _append_markdown_table(
                    lines,
                    stage_table_df,
                    columns=table_columns,
                    rename_map={
                        'stage': '调整环节',
                        'smape_before': '调整前 sMAPE',
                        'smape_after': '调整后 sMAPE',
                        'smape_delta': 'sMAPE 变化',
                        'rmse_before': '调整前 RMSE',
                        'rmse_after': '调整后 RMSE',
                        'rmse_delta': 'RMSE 变化',
                    },
                    formatters={
                        'smape_before': _format_percent,
                        'smape_after': _format_percent,
                        'smape_delta': _format_percent,
                        'rmse_before': _format_number,
                        'rmse_after': _format_number,
                        'rmse_delta': _format_number,
                    },
                )

    lines.append('## 分阶段变化（技术附录）')
    lines.append('')
    lines.append('- 阶段释义（流程版）：')
    lines.append(f"- Chronos基线：{PIPELINE_STAGE_FLOW_DETAILS['Chronos基线']}")
    lines.append(f"- Direct基线：{PIPELINE_STAGE_FLOW_DETAILS['Direct基线']}")
    lines.append(f"- 融合后：{PIPELINE_STAGE_FLOW_DETAILS['融合后']}")
    lines.append(f"- Residual后：{PIPELINE_STAGE_FLOW_DETAILS['Residual后']}")
    lines.append(f"- Lead-wise后(最终)：{PIPELINE_STAGE_FLOW_DETAILS['Lead-wise后(最终)']}")
    lines.append('')
    if stage_transition_summary_df is None or stage_transition_summary_df.empty:
        lines.append('- 无分阶段变化数据。')
    else:
        for series_name in ['call_volume', 'tickets_received']:
            series_df = stage_transition_summary_df.loc[stage_transition_summary_df['series_name'] == series_name].copy()
            if series_df.empty:
                continue
            lines.append(f"### {SERIES_DISPLAY_NAMES.get(series_name, series_name)}")
            lines.append('')
            _append_markdown_table(
                lines,
                series_df,
                columns=['stage', 'smape', 'smape_delta_vs_prev', 'rmse', 'rmse_delta_vs_prev', 'mae', 'mae_delta_vs_prev', 'bias'],
                rename_map={
                    'stage': '调整环节',
                    'smape': 'sMAPE',
                    'smape_delta_vs_prev': '相对上阶段 sMAPE 变化',
                    'rmse': 'RMSE',
                    'rmse_delta_vs_prev': '相对上阶段 RMSE 变化',
                    'mae': 'MAE',
                    'mae_delta_vs_prev': '相对上阶段 MAE 变化',
                    'bias': 'Bias',
                },
                formatters={
                    'smape': _format_percent,
                    'smape_delta_vs_prev': _format_percent,
                    'rmse': _format_number,
                    'rmse_delta_vs_prev': _format_number,
                    'mae': _format_number,
                    'mae_delta_vs_prev': _format_number,
                    'bias': _format_number,
                },
            )
            improving_df = series_df.loc[series_df['smape_delta_vs_prev'].notna()].copy()
            if not improving_df.empty:
                best_row = improving_df.sort_values('smape_delta_vs_prev').iloc[0]
                lines.append(
                    '- '
                    f"分阶段中改善最大的是{best_row['stage']}，相对上一环节 sMAPE 变化 {_format_percent(best_row['smape_delta_vs_prev'])}。"
                )
            lines.append('')

    lines.append('## 样本口径说明')
    lines.append('')
    lines.append('- 回测样本数：用于判断模型在最近一段时间内是否稳定，反映“近期准不准”。')
    lines.append('- 历史样本(recent)：用于判断最近一段时间该场景是否常见。')
    lines.append('- 历史样本(all)：用于判断放到全历史看，这个场景是否本来就常见。')
    lines.append('- 常见误解：如果看到“普通工作日 回测22、历史20”，意思是最近窗口里普通工作日样本接近，而不是全历史只有20天。')
    lines.append('')
    if monitor_history_profile_df is None or monitor_history_profile_df.empty or recent_monitor_df.empty:
        lines.append('- 无历史分桶画像数据，暂无法展示回测与历史样本口径对照。')
    else:
        history_recent_df = monitor_history_profile_df.loc[
            monitor_history_profile_df['scope'] == 'history_recent'
        ].copy()
        history_all_df = monitor_history_profile_df.loc[
            monitor_history_profile_df['scope'] == 'history_all'
        ].copy()
        history_recent_df['bucket'] = history_recent_df['bucket'].map(_normalize_bucket_name)
        history_all_df['bucket'] = history_all_df['bucket'].map(_normalize_bucket_name)

        for series_name in ['call_volume', 'tickets_received']:
            bt_series_df = recent_monitor_df.loc[
                recent_monitor_df['series_name'] == series_name
            ].copy()
            hist_recent_series_df = history_recent_df.loc[
                history_recent_df['series_name'] == series_name
            ].copy()
            hist_all_series_df = history_all_df.loc[
                history_all_df['series_name'] == series_name
            ].copy()
            if bt_series_df.empty and hist_recent_series_df.empty and hist_all_series_df.empty:
                continue

            bt_series_df['bucket'] = bt_series_df['bucket'].map(_normalize_bucket_name)
            merged_scope_df = pd.merge(
                bt_series_df,
                hist_recent_series_df[['series_name', 'bucket', 'hist_sample_size', 'hist_zero_ratio']].rename(
                    columns={
                        'hist_sample_size': 'hist_sample_size_recent',
                        'hist_zero_ratio': 'hist_zero_ratio_recent',
                    }
                ),
                on=['series_name', 'bucket'],
                how='outer',
            )
            merged_scope_df = pd.merge(
                merged_scope_df,
                hist_all_series_df[['series_name', 'bucket', 'hist_sample_size']].rename(
                    columns={'hist_sample_size': 'hist_sample_size_all'}
                ),
                on=['series_name', 'bucket'],
                how='outer',
            )
            if merged_scope_df.empty:
                continue

            merged_scope_df['series_name'] = merged_scope_df['series_name'].fillna(series_name)
            merged_scope_df['sample_size'] = merged_scope_df['sample_size'].fillna(0).astype(int)
            merged_scope_df['hist_sample_size_recent'] = merged_scope_df['hist_sample_size_recent'].fillna(0).astype(int)
            merged_scope_df['hist_sample_size_all'] = merged_scope_df['hist_sample_size_all'].fillna(0).astype(int)
            merged_scope_df['hist_zero_ratio_recent'] = merged_scope_df['hist_zero_ratio_recent'].fillna(0.0)
            merged_scope_df['smape'] = merged_scope_df['smape'].fillna(0.0)
            merged_scope_df['bucket'] = merged_scope_df['bucket'].map(
                lambda value: BUCKET_DISPLAY_NAMES.get(str(value), str(value))
            )
            merged_scope_df = merged_scope_df.sort_values(
                by='bucket',
                key=lambda col: col.map(
                    lambda value: _bucket_sort_key(
                        next((k for k, v in BUCKET_DISPLAY_NAMES.items() if v == value), value)
                    )
                )
            )

            lines.append(f"### {SERIES_DISPLAY_NAMES.get(series_name, series_name)}")
            lines.append('')
            _append_markdown_table(
                lines,
                merged_scope_df,
                columns=['bucket', 'sample_size', 'hist_sample_size_recent', 'hist_sample_size_all', 'hist_zero_ratio_recent', 'smape'],
                rename_map={
                    'bucket': '场景',
                    'sample_size': f'回测样本数(最近{int(monitor_recent_days)}天)',
                    'hist_sample_size_recent': f'历史样本数(最近{int(monitor_recent_days)}天)',
                    'hist_sample_size_all': '历史样本数(全历史)',
                    'hist_zero_ratio_recent': '历史零值占比(最近窗口)',
                    'smape': '回测 sMAPE',
                },
                formatters={
                    'hist_zero_ratio_recent': _format_percent,
                    'smape': _format_percent,
                },
            )

            lines.append('#### 三句话速读')
            lines.append('')
            lines.append('- 第一句：回测样本数看“模型最近有没有被充分验证”。')
            lines.append('- 第二句：最近历史样本数看“这个场景最近是否常出现”。')
            lines.append('- 第三句：全历史样本数看“这个场景长期是否常见”，避免把近期少量误解成长期稀有。')
            low_conf_df = merged_scope_df.loc[
                merged_scope_df['sample_size'] < monitor_low_sample_threshold
            ].copy()
            if not low_conf_df.empty:
                for _, low_row in low_conf_df.iterrows():
                    lines.append(
                        '- '
                        f"{low_row['bucket']}：回测样本数 {int(low_row['sample_size'])} 低于阈值 {monitor_low_sample_threshold}，"
                        '这类结果更适合观察趋势，不建议直接作为强业务结论。'
                    )
            lines.append('')

    lines.append('## 分场景表现')
    lines.append('')
    if recent_monitor_df.empty:
        lines.append('- 无分场景复评数据。')
    else:
        for series_name in ['call_volume', 'tickets_received']:
            series_df = recent_monitor_df.loc[recent_monitor_df['series_name'] == series_name].copy()
            if series_df.empty:
                continue
            lines.append(f"### {SERIES_DISPLAY_NAMES.get(series_name, series_name)}")
            lines.append('')
            series_df['bucket'] = series_df['bucket'].map(_normalize_bucket_name)
            series_df['bucket'] = series_df['bucket'].map(lambda value: BUCKET_DISPLAY_NAMES.get(str(value), str(value)))
            series_df = series_df.sort_values(by='bucket', key=lambda col: col.map(lambda value: _bucket_sort_key(next((key for key, label in BUCKET_DISPLAY_NAMES.items() if label == value), value))))
            _append_markdown_table(
                lines,
                series_df,
                columns=['bucket', 'sample_size', 'smape', 'rmse', 'mae', 'bias', 'mean_actual', 'mean_predicted'],
                rename_map={
                    'bucket': '场景',
                    'sample_size': f'回测样本数(最近{int(monitor_recent_days)}天)',
                    'smape': 'sMAPE',
                    'rmse': 'RMSE',
                    'mae': 'MAE',
                    'bias': 'Bias',
                    'mean_actual': '平均实际值',
                    'mean_predicted': '平均预测值',
                },
                formatters={
                    'smape': _format_percent,
                    'rmse': _format_number,
                    'mae': _format_number,
                    'bias': _format_number,
                    'mean_actual': _format_number,
                    'mean_predicted': _format_number,
                },
            )
            lines.append('#### 重点解读')
            lines.append('')
            for focus_bucket in ['workday_normal', 'weekend', 'holiday', 'post_holiday_workday_1_3', 'makeup_workday']:
                bucket_df = series_df.loc[series_df['bucket'] == BUCKET_DISPLAY_NAMES.get(focus_bucket, focus_bucket)]
                if bucket_df.empty:
                    lines.append(f"- {BUCKET_DISPLAY_NAMES.get(focus_bucket, focus_bucket)}：近期样本不足，当前不建议单独下业务结论。")
                    continue
                row = bucket_df.iloc[0]
                if int(row['sample_size']) < monitor_low_sample_threshold:
                    lines.append(
                        '- '
                        f"{row['bucket']}：近期回测样本数 {int(row['sample_size'])}，低于阈值 {monitor_low_sample_threshold}，当前判定为低置信度场景。"
                    )
                    continue
                lines.append(
                    '- '
                    f"{row['bucket']}：近期 sMAPE {_format_percent(row['smape'])}，Bias {_format_number(row['bias'])}，"
                    f"{_classify_bias_direction(row['bias'], row['mean_actual'])}。"
                )
            lines.append('')

    lines.append('## 未来预测摘要')
    lines.append('')
    if forecast_export_df is None or forecast_export_df.empty:
        lines.append('- 无未来预测结果。')
    else:
        future_df = forecast_export_df.copy()
        future_df['date'] = pd.to_datetime(future_df['date'])
        future_df['interval_width'] = future_df['p90'] - future_df['p10']
        for series_name in ['call_volume', 'tickets_received']:
            series_df = future_df.loc[future_df['target_name'] == series_name].sort_values('date').copy()
            if series_df.empty:
                continue
            peak_row = series_df.loc[series_df['p50'].idxmax()]
            trough_row = series_df.loc[series_df['p50'].idxmin()]
            lines.append(f"### {SERIES_DISPLAY_NAMES.get(series_name, series_name)}")
            lines.append('')
            lines.append(
                '- '
                f"未来 {len(series_df)} 天 P50 平均值为 {_format_number(series_df['p50'].mean())}，"
                f"最高点出现在 {peak_row['date'].strftime('%Y-%m-%d')}，P50 为 {_format_number(peak_row['p50'])}；"
                f"最低点出现在 {trough_row['date'].strftime('%Y-%m-%d')}，P50 为 {_format_number(trough_row['p50'])}。"
            )
            lines.append(
                '- '
                f"平均区间宽度为 {_format_number(series_df['interval_width'].mean())}，"
                f"P90 相比 P50 的平均上浮为 {_format_number((series_df['p90'] - series_df['p50']).mean())}。"
            )
            lines.append('')

    lines.append('## 配图')
    lines.append('')
    image_title_map = {
        'evaluation': '回测总览图',
        'future': '未来预测图',
        'export': '标准导出图',
        'tuning': '调参效果图',
        'monitor': '分桶监控图',
        'monitor_sample_scope': '样本口径对照图',
        'stage_transition': '分阶段预测变化图',
    }
    for image_key in ['evaluation', 'future', 'export', 'tuning', 'monitor', 'monitor_sample_scope', 'stage_transition']:
        image_link = image_links.get(image_key)
        if not image_link:
            continue
        lines.append(f"### {image_title_map.get(image_key, image_key)}")
        lines.append('')
        lines.append(f"![{image_title_map.get(image_key, image_key)}]({image_link})")
        lines.append('')

    lines.append('## 结论与建议')
    lines.append('')
    if recent_monitor_df.empty:
        lines.append('- 当前建议先补齐分桶回测样本，再对工作日、周末、节假日形成稳定业务口径。')
    else:
        for series_name in ['call_volume', 'tickets_received']:
            series_df = recent_monitor_df.loc[recent_monitor_df['series_name'] == series_name].copy()
            if series_df.empty:
                continue
            weighted_smape = float(np.average(series_df['smape'], weights=series_df['sample_size']))
            worst_row = series_df.sort_values(['smape', 'sample_size'], ascending=[False, False]).iloc[0]
            lines.append(
                '- '
                f"{SERIES_DISPLAY_NAMES.get(series_name, series_name)}当前最近分桶综合 sMAPE 为 {_format_percent(weighted_smape)}；"
                f"如需继续优化，优先处理{BUCKET_DISPLAY_NAMES.get(str(worst_row['bucket']), str(worst_row['bucket']))}场景。"
            )
            if str(worst_row['bucket']) == 'post_holiday_workday_1_3':
                lines.append('- 节后恢复段仍是核心风险点，建议继续重点观察节后前3个工作日的高估/低估偏差。')
            if str(worst_row['bucket']) == 'holiday':
                lines.append('- 节假日桶是当前主要误差来源，建议复核节假日残差模型和春节值班规则。')
            if str(worst_row['bucket']) == 'weekend':
                lines.append('- 周末桶波动偏大，建议复核周末单独权重或增加周末样本约束。')
            if str(worst_row['bucket']) in {'makeup_workday', 'other'}:
                lines.append('- 补班日样本很少但波动通常明显，建议单独跟踪补班日规则并避免直接套用普通工作日口径。')

    return '\n'.join(lines).strip() + '\n'


def compute_dynamic_leadwise_cap(bt_results, residual_adjuster, base_cap, series_tuning):
    base_cap = float(np.clip(base_cap, 0.0, 1.0))
    dynamic_cfg = series_tuning.get('dynamic_leadwise_cap', {})
    if not bool(dynamic_cfg.get('enabled', False)):
        return base_cap, {'mode': 'disabled', 'recent_smape': None}

    recent_days = max(1, int(dynamic_cfg.get('recent_days', 84)))
    report_df = build_monitor_bucket_report(
        bt_results,
        residual_adjuster,
        leadwise_adjuster=None,
        series_name=None,
        recent_days=recent_days,
    )
    if report_df.empty:
        return base_cap, {'mode': 'empty-report', 'recent_smape': None}

    recent_df = report_df.loc[report_df['scope'] == 'recent'].copy()
    if recent_df.empty or int(recent_df['sample_size'].sum()) <= 0:
        return base_cap, {'mode': 'empty-recent', 'recent_smape': None}

    weighted_smape = float(np.average(recent_df['smape'], weights=recent_df['sample_size']))
    smape_high = float(np.clip(dynamic_cfg.get('smape_high', 0.18), 0.0, 1.0))
    smape_low = float(np.clip(dynamic_cfg.get('smape_low', 0.12), 0.0, 1.0))
    down_shift = float(np.clip(dynamic_cfg.get('down_shift', 0.08), 0.0, 1.0))
    up_shift = float(np.clip(dynamic_cfg.get('up_shift', 0.03), 0.0, 1.0))
    min_cap = float(np.clip(dynamic_cfg.get('min_cap', 0.5), 0.0, 1.0))
    max_cap = float(np.clip(dynamic_cfg.get('max_cap', 0.95), 0.0, 1.0))
    if min_cap > max_cap:
        min_cap, max_cap = max_cap, min_cap

    dynamic_cap = base_cap
    mode = 'keep'
    if weighted_smape >= smape_high:
        dynamic_cap = base_cap - down_shift
        mode = 'down'
    elif weighted_smape <= smape_low:
        dynamic_cap = base_cap + up_shift
        mode = 'up'

    dynamic_cap = float(np.clip(dynamic_cap, min_cap, max_cap))
    return dynamic_cap, {
        'mode': mode,
        'recent_smape': weighted_smape,
        'smape_high': smape_high,
        'smape_low': smape_low,
    }


def apply_bucket_bias_gate(future_results, bias_snapshot, series_tuning):
    gate_cfg = series_tuning.get('bias_gate', {})
    if not bool(gate_cfg.get('enabled', False)):
        return future_results, {'enabled': False, 'applied_count': 0}

    raw_targets = gate_cfg.get('target_buckets')
    if isinstance(raw_targets, (list, tuple)):
        target_buckets = []
        for item in raw_targets:
            bucket_name = _normalize_bucket_name(item)
            if bucket_name and bucket_name not in target_buckets:
                target_buckets.append(bucket_name)
    else:
        target_bucket = str(gate_cfg.get('target_bucket', 'post_holiday_workday_1_3')).strip() or 'post_holiday_workday_1_3'
        target_buckets = [_normalize_bucket_name(target_bucket)]
    if not target_buckets:
        return future_results, {'enabled': True, 'applied_count': 0, 'reason': 'empty-targets'}

    min_samples = max(1, int(gate_cfg.get('min_samples', 14)))
    bias_trigger = float(np.clip(gate_cfg.get('bias_trigger', 0.0), 0.0, 10.0))
    only_positive = bool(gate_cfg.get('only_positive_bias', True))
    allow_scope_fallback = bool(gate_cfg.get('allow_scope_fallback', True))
    fallback_scale = float(np.clip(gate_cfg.get('fallback_scale', 0.60), 0.0, 1.0))
    allow_low_sample_gate = bool(gate_cfg.get('allow_low_sample_gate', False))
    low_sample_floor = float(np.clip(gate_cfg.get('low_sample_floor', 0.35), 0.0, 1.0))
    enforce_rmse_guard = bool(gate_cfg.get('enforce_rmse_guard', False))
    rmse_guard_scale = float(np.clip(gate_cfg.get('rmse_guard_scale', 0.75), 0.0, 10.0))
    bucket_scales_raw = gate_cfg.get('bucket_scales', {})
    bucket_scales = {}
    if isinstance(bucket_scales_raw, dict):
        for raw_bucket, raw_scale in bucket_scales_raw.items():
            bucket_name = _normalize_bucket_name(raw_bucket)
            if not bucket_name:
                continue
            try:
                bucket_scales[bucket_name] = float(np.clip(float(raw_scale), 0.0, 10.0))
            except Exception:
                continue

    gate_details = []
    effective_gate_by_bucket = {}
    recent_snapshot = (bias_snapshot.get('recent', {}) or {})
    all_snapshot = (bias_snapshot.get('all', {}) or {})

    for target_bucket in target_buckets:
        bucket_entry_recent = recent_snapshot.get(target_bucket, {})
        recent_sample_size = int(bucket_entry_recent.get('sample_size', 0))
        bucket_entry = bucket_entry_recent
        scope_key = 'recent'
        scope_factor = 1.0

        if recent_sample_size < min_samples and allow_scope_fallback:
            bucket_entry_all = all_snapshot.get(target_bucket, {})
            all_sample_size = int(bucket_entry_all.get('sample_size', 0))
            if all_sample_size >= min_samples:
                bucket_entry = bucket_entry_all
                scope_key = 'all'
                scope_factor = fallback_scale

        sample_size = int(bucket_entry.get('sample_size', 0))
        bias_value = float(bucket_entry.get('bias', 0.0))
        rmse_value = float(bucket_entry.get('rmse', 0.0))
        bucket_scale = float(bucket_scales.get(target_bucket, 1.0))

        detail = {
            'bucket': target_bucket,
            'scope': scope_key,
            'sample_size': sample_size,
            'bias': bias_value,
            'rmse': rmse_value,
            'bucket_scale': bucket_scale,
        }

        if sample_size < min_samples:
            if allow_low_sample_gate and sample_size > 0:
                sample_ratio = float(sample_size) / float(min_samples)
                scope_factor *= float(np.clip(sample_ratio, low_sample_floor, 1.0))
                detail['reason'] = 'low-sample-soft'
            else:
                detail['reason'] = 'insufficient-samples'
                gate_details.append(detail)
                continue

        if abs(bias_value) < bias_trigger:
            detail['reason'] = 'below-trigger'
            gate_details.append(detail)
            continue

        if only_positive and bias_value <= 0:
            detail['reason'] = 'not-positive-bias'
            gate_details.append(detail)
            continue

        detail['reason'] = 'ready'
        gate_details.append(detail)
        effective_gate_by_bucket[target_bucket] = {
            'bias': bias_value,
            'rmse': rmse_value,
            'bucket_scale': bucket_scale,
            'sample_size': sample_size,
            'scope': scope_key,
            'scope_factor': scope_factor,
        }

    if not effective_gate_by_bucket:
        return future_results, {
            'enabled': True,
            'applied_count': 0,
            'targets': target_buckets,
            'details': gate_details,
            'reason': 'no-effective-target',
        }

    scale = float(np.clip(gate_cfg.get('adjustment_scale', 0.25), 0.0, 2.0))
    max_ratio = float(np.clip(gate_cfg.get('max_adjustment_ratio', 0.10), 0.0, 1.0))

    adjusted = future_results.copy()
    p10 = np.asarray(adjusted['p10'], dtype=float).copy()
    p50 = np.asarray(adjusted['p50'], dtype=float).copy()
    p90 = np.asarray(adjusted['p90'], dtype=float).copy()

    future_dates = pd.to_datetime(adjusted.get('future_dates', []))
    holiday_context_rows = build_holiday_context_rows(future_dates)
    applied_count = 0
    applied_by_bucket = {}
    for idx, dt in enumerate(future_dates):
        bucket = _classify_monitor_bucket(dt, holiday_context_rows[idx])
        if bucket not in effective_gate_by_bucket:
            continue
        bucket_gate = effective_gate_by_bucket[bucket]

        p50_now = float(p50[idx])
        raw_adj = (
            scale
            * float(bucket_gate.get('bucket_scale', 1.0))
            * float(bucket_gate['bias'])
            * float(bucket_gate['scope_factor'])
        )
        ratio_bound = max_ratio * max(1.0, p50_now)
        rmse_bound = np.inf
        if enforce_rmse_guard:
            rmse_bound = rmse_guard_scale * max(0.0, float(bucket_gate.get('rmse', 0.0)))
        bounded_limit = float(min(ratio_bound, rmse_bound))
        bounded_adj = float(np.clip(raw_adj, -bounded_limit, bounded_limit))

        p50_new = max(0.0, p50_now - bounded_adj)
        p10_new = max(0.0, float(p10[idx]) - bounded_adj)
        p90_new = max(p50_new, float(p90[idx]) - bounded_adj)
        p10[idx] = min(p10_new, p50_new)
        p50[idx] = p50_new
        p90[idx] = p90_new
        applied_count += 1
        applied_by_bucket[bucket] = int(applied_by_bucket.get(bucket, 0)) + 1

    adjusted['p10'] = p10
    adjusted['p50'] = p50
    adjusted['p90'] = p90
    return adjusted, {
        'enabled': True,
        'applied_count': int(applied_count),
        'targets': target_buckets,
        'details': gate_details,
        'applied_by_bucket': applied_by_bucket,
        'adjustment_scale': scale,
        'max_adjustment_ratio': max_ratio,
        'bucket_scales': bucket_scales,
        'enforce_rmse_guard': enforce_rmse_guard,
        'rmse_guard_scale': rmse_guard_scale,
    }


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
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        adjusted_pred = []
        for idx, (dt, pred) in enumerate(zip(dates, predicted)):
            adjustment = compute_residual_adjustment(
                dt,
                use_adjuster,
                base_pred=float(pred),
                holiday_context=holiday_context_rows[idx],
            )
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
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)

        adjusted_pred = []
        for idx, (dt, pred) in enumerate(zip(dates, predicted), start=1):
            adjustment = compute_residual_adjustment(
                dt,
                residual_adjuster,
                base_pred=float(pred),
                holiday_context=holiday_context_rows[idx - 1],
            )
            if leadwise_adjuster is not None:
                lead_bias = float(leadwise_adjuster.get('lead_bias', {}).get(idx, leadwise_adjuster.get('global_lead_bias', 0.0)))
                lead_weight = float(np.clip(leadwise_adjuster.get('leadwise_weight', 1.0), 0.0, 1.0))
                adjustment += lead_weight * lead_bias
            adjusted_pred.append(max(0.0, float(pred) + adjustment))

        adjusted_pred = np.asarray(adjusted_pred, dtype=float)
        residuals.extend((actual - adjusted_pred).tolist())

    return np.asarray(residuals, dtype=float)


def calibrate_prediction_intervals(
    future_results,
    residuals,
    coverage=0.80,
    asymmetric=False,
    lower_coverage=None,
    upper_coverage=None,
):
    coverage = float(np.clip(coverage, 0.5, 0.99))
    lower_coverage = coverage if lower_coverage is None else float(np.clip(lower_coverage, 0.5, 0.99))
    upper_coverage = coverage if upper_coverage is None else float(np.clip(upper_coverage, 0.5, 0.99))
    if residuals.size == 0:
        return future_results, {
            'radius': 0.0,
            'lower_radius': 0.0,
            'upper_radius': 0.0,
            'empirical_coverage': 0.0,
            'asymmetric': bool(asymmetric),
        }

    residuals = np.asarray(residuals, dtype=float)

    if bool(asymmetric):
        lower_errors = np.maximum(0.0, -residuals)
        upper_errors = np.maximum(0.0, residuals)
        lower_radius = float(np.quantile(lower_errors, lower_coverage)) if lower_errors.size > 0 else 0.0
        upper_radius = float(np.quantile(upper_errors, upper_coverage)) if upper_errors.size > 0 else 0.0
    else:
        abs_residuals = np.abs(residuals)
        radius = float(np.quantile(abs_residuals, coverage))
        lower_radius = radius
        upper_radius = radius

    empirical_coverage = float(np.mean((residuals >= -lower_radius) & (residuals <= upper_radius)))

    calibrated = future_results.copy()
    p50 = np.asarray(calibrated['p50'], dtype=float)
    p10 = np.maximum(0.0, p50 - lower_radius)
    p90 = np.maximum(p50, p50 + upper_radius)

    calibrated['p10'] = p10
    calibrated['p90'] = p90
    return calibrated, {
        'radius': float((lower_radius + upper_radius) / 2.0),
        'lower_radius': float(lower_radius),
        'upper_radius': float(upper_radius),
        'empirical_coverage': empirical_coverage,
        'asymmetric': bool(asymmetric),
    }


def evaluate_interval_calibration_backtest(
    bt_results,
    residual_adjuster,
    coverage=0.80,
    leadwise_adjuster=None,
    asymmetric=False,
    lower_coverage=None,
    upper_coverage=None,
):
    windows = bt_results.get('all_windows', [])
    if not windows:
        windows = [bt_results]

    coverage = float(np.clip(coverage, 0.5, 0.99))
    lower_coverage = coverage if lower_coverage is None else float(np.clip(lower_coverage, 0.5, 0.99))
    upper_coverage = coverage if upper_coverage is None else float(np.clip(upper_coverage, 0.5, 0.99))
    window_coverages = []
    window_widths = []
    used_radii = []
    used_lower_radii = []
    used_upper_radii = []

    adjusted_windows = []
    for window in windows:
        dates = pd.to_datetime(window['dates'])
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(window['actual'], dtype=float)
        predicted = np.asarray(window['predicted'], dtype=float)
        adjusted_pred = []
        for idx, (dt, pred) in enumerate(zip(dates, predicted), start=1):
            adjustment = compute_residual_adjustment(
                dt,
                residual_adjuster,
                base_pred=float(pred),
                holiday_context=holiday_context_rows[idx - 1],
            )
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
        if bool(asymmetric):
            lower_errors = np.maximum(0.0, -calib_residuals)
            upper_errors = np.maximum(0.0, calib_residuals)
            lower_radius = float(np.quantile(lower_errors, lower_coverage)) if calib_residuals.size > 0 else 0.0
            upper_radius = float(np.quantile(upper_errors, upper_coverage)) if calib_residuals.size > 0 else 0.0
        else:
            radius = float(np.quantile(np.abs(calib_residuals), coverage)) if calib_residuals.size > 0 else 0.0
            lower_radius = radius
            upper_radius = radius

        lower = np.maximum(0.0, target_window['pred'] - lower_radius)
        upper = np.maximum(target_window['pred'], target_window['pred'] + upper_radius)
        within = (target_window['actual'] >= lower) & (target_window['actual'] <= upper)

        window_coverages.append(float(np.mean(within)))
        window_widths.append(float(np.mean(upper - lower)))
        used_radii.append(float((lower_radius + upper_radius) / 2.0))
        used_lower_radii.append(float(lower_radius))
        used_upper_radii.append(float(upper_radius))

    return {
        'avg_coverage': float(np.mean(window_coverages)) if window_coverages else 0.0,
        'avg_width': float(np.mean(window_widths)) if window_widths else 0.0,
        'avg_radius': float(np.mean(used_radii)) if used_radii else 0.0,
        'avg_lower_radius': float(np.mean(used_lower_radii)) if used_lower_radii else 0.0,
        'avg_upper_radius': float(np.mean(used_upper_radii)) if used_upper_radii else 0.0,
        'asymmetric': bool(asymmetric),
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
    windows_chronos = chronos_bt_results.get('all_windows', [chronos_bt_results])
    windows_direct = direct_bt_results.get('all_windows', [direct_bt_results])

    if len(windows_chronos) != len(windows_direct):
        raise ValueError(
            "Chronos and direct backtest windows mismatch: "
            f"{len(windows_chronos)} vs {len(windows_direct)}."
        )

    smape_chronos_list = []
    rmse_chronos_list = []
    smape_fused_list = []
    rmse_fused_list = []

    for chronos_window, direct_window in zip(windows_chronos, windows_direct):
        dates = pd.to_datetime(chronos_window['dates'])
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(chronos_window['actual'], dtype=float)
        chronos_pred = np.asarray(chronos_window['predicted'], dtype=float)
        direct_pred = np.asarray(direct_window['predicted'], dtype=float)
        fused_pred = np.zeros_like(chronos_pred, dtype=float)
        for idx, dt in enumerate(dates):
            weight = resolve_fusion_weight(dt, custom_weight, holiday_context=holiday_context_rows[idx])
            fused_pred[idx] = weight * chronos_pred[idx] + (1.0 - weight) * direct_pred[idx]

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


def resolve_fusion_weight(dt, fusion_policy, holiday_context=None):
    if fusion_policy is None:
        return 0.5
    if isinstance(fusion_policy, (float, int)):
        return float(np.clip(float(fusion_policy), 0.0, 1.0))

    if isinstance(fusion_policy, dict):
        global_weight = float(np.clip(float(fusion_policy.get('global_weight', 0.5)), 0.0, 1.0))
        bucket_weights = fusion_policy.get('bucket_weights', {})
        if isinstance(bucket_weights, dict) and bucket_weights:
            use_holiday_layers = bool(fusion_policy.get('use_holiday_layers', False))
            bucket_name = _classify_fusion_bucket(dt, holiday_context=holiday_context, use_holiday_layers=use_holiday_layers)
            if bucket_name in bucket_weights:
                return float(np.clip(float(bucket_weights[bucket_name]), 0.0, 1.0))
            if use_holiday_layers and bucket_name.startswith('holiday_') and ('holiday' in bucket_weights):
                return float(np.clip(float(bucket_weights['holiday']), 0.0, 1.0))
        state_weights = fusion_policy.get('state_weights', {})
        if isinstance(state_weights, dict):
            state_name = get_operational_state(dt)
            if state_name in state_weights:
                return float(np.clip(float(state_weights[state_name]), 0.0, 1.0))
        return global_weight

    return 0.5


def format_fusion_policy_summary(fusion_policy):
    if isinstance(fusion_policy, (float, int)):
        w = float(np.clip(float(fusion_policy), 0.0, 1.0))
        return f"global={w:.2f}"

    if not isinstance(fusion_policy, dict):
        return "global=0.50"

    global_weight = float(np.clip(float(fusion_policy.get('global_weight', 0.5)), 0.0, 1.0))
    bucket_weights = fusion_policy.get('bucket_weights', {})
    state_weights = fusion_policy.get('state_weights', {})
    if isinstance(bucket_weights, dict) and bucket_weights:
        parts = [f"global={global_weight:.2f}"]
        for key in [
            'holiday_spring_shutdown',
            'holiday_spring_duty',
            'holiday_long_core',
            'holiday_general',
            'holiday',
            'post_holiday_workday_1_3',
            'workday_normal',
            'weekend',
            'makeup_workday',
            'other',
        ]:
            if key in bucket_weights:
                parts.append(f"{key}={float(np.clip(float(bucket_weights[key]), 0.0, 1.0)):.2f}")
        return ", ".join(parts)
    if isinstance(state_weights, dict) and state_weights:
        parts = [f"global={global_weight:.2f}"]
        for key in ['workday', 'weekend', 'holiday_or_makeup']:
            if key in state_weights:
                parts.append(f"{key}={float(np.clip(float(state_weights[key]), 0.0, 1.0)):.2f}")
        return ", ".join(parts)
    return f"global={global_weight:.2f}"


def get_global_fusion_weight(fusion_policy):
    if isinstance(fusion_policy, (float, int)):
        return float(np.clip(float(fusion_policy), 0.0, 1.0))
    if isinstance(fusion_policy, dict):
        return float(np.clip(float(fusion_policy.get('global_weight', 0.5)), 0.0, 1.0))
    return 0.5


def search_best_direct_fusion_weight(chronos_bt_results, direct_bt_results, search_points=11, series_tuning=None):
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

    windows_chronos = chronos_bt_results.get('all_windows', [chronos_bt_results])
    windows_direct = direct_bt_results.get('all_windows', [direct_bt_results])
    if len(windows_chronos) != len(windows_direct):
        return best_weight, best_effect

    state_records = {'workday': [], 'weekend': [], 'holiday_or_makeup': []}
    for chronos_window, direct_window in zip(windows_chronos, windows_direct):
        dates = pd.to_datetime(chronos_window['dates'])
        actual = np.asarray(chronos_window['actual'], dtype=float)
        chronos_pred = np.asarray(chronos_window['predicted'], dtype=float)
        direct_pred = np.asarray(direct_window['predicted'], dtype=float)
        for dt, y_true, c_pred, d_pred in zip(dates, actual, chronos_pred, direct_pred):
            state_name = get_operational_state(dt)
            state_records.setdefault(state_name, []).append((float(y_true), float(c_pred), float(d_pred)))

    state_weights = {}
    for state_name in ['workday', 'weekend', 'holiday_or_makeup']:
        records = state_records.get(state_name, [])
        if len(records) < 4:
            state_weights[state_name] = float(best_weight)
            continue

        y_true = np.asarray([r[0] for r in records], dtype=float)
        c_pred = np.asarray([r[1] for r in records], dtype=float)
        d_pred = np.asarray([r[2] for r in records], dtype=float)

        best_state_weight = float(best_weight)
        best_state_score = (float('inf'), float('inf'))
        for candidate in candidates:
            fused = float(candidate) * c_pred + (1.0 - float(candidate)) * d_pred
            s, r = calculate_metrics(y_true, fused)
            score = (float(s), float(r))
            if score < best_state_score:
                best_state_score = score
                best_state_weight = float(candidate)
        state_weights[state_name] = best_state_weight

    fusion_policy = {
        'global_weight': float(best_weight),
        'state_weights': state_weights,
    }

    bucket_cfg = (series_tuning or {}).get('bucket_fusion_secondary', {})
    if bool(bucket_cfg.get('enabled', False)):
        min_samples = max(1, int(bucket_cfg.get('min_samples', 8)))
        use_holiday_layers = bool(bucket_cfg.get('enable_holiday_layers', False))
        holiday_min_samples = max(1, int(bucket_cfg.get('holiday_min_samples', min_samples)))
        bucket_order = [
            'holiday_spring_shutdown',
            'holiday_spring_duty',
            'holiday_long_core',
            'holiday_general',
            'post_holiday_workday_1_3',
            'workday_normal',
            'weekend',
            'makeup_workday',
        ] if use_holiday_layers else [
            'holiday',
            'post_holiday_workday_1_3',
            'workday_normal',
            'weekend',
            'makeup_workday',
        ]
        bucket_records = {key: [] for key in bucket_order}

        for chronos_window, direct_window in zip(windows_chronos, windows_direct):
            dates = pd.to_datetime(chronos_window['dates'])
            holiday_context_rows = build_holiday_context_rows(dates)
            actual = np.asarray(chronos_window['actual'], dtype=float)
            chronos_pred = np.asarray(chronos_window['predicted'], dtype=float)
            direct_pred = np.asarray(direct_window['predicted'], dtype=float)

            for idx, (dt, y_true, c_pred, d_pred) in enumerate(zip(dates, actual, chronos_pred, direct_pred)):
                bucket_name = _classify_fusion_bucket(
                    dt,
                    holiday_context=holiday_context_rows[idx],
                    use_holiday_layers=use_holiday_layers,
                )
                bucket_records.setdefault(bucket_name, []).append((float(y_true), float(c_pred), float(d_pred)))

        bucket_weights = {}
        for bucket_name in bucket_order:
            records = bucket_records.get(bucket_name, [])
            local_min_samples = holiday_min_samples if bucket_name.startswith('holiday_') else min_samples
            if len(records) < local_min_samples:
                continue

            y_true = np.asarray([r[0] for r in records], dtype=float)
            c_pred = np.asarray([r[1] for r in records], dtype=float)
            d_pred = np.asarray([r[2] for r in records], dtype=float)

            best_bucket_weight = float(fusion_policy.get('global_weight', best_weight))
            best_bucket_score = (float('inf'), float('inf'))
            for candidate in candidates:
                fused = float(candidate) * c_pred + (1.0 - float(candidate)) * d_pred
                s, r = calculate_metrics(y_true, fused)
                score = (float(s), float(r))
                if score < best_bucket_score:
                    best_bucket_score = score
                    best_bucket_weight = float(candidate)
            bucket_weights[bucket_name] = best_bucket_weight

        if bucket_weights:
            fusion_policy['bucket_weights'] = bucket_weights
            fusion_policy['use_holiday_layers'] = bool(use_holiday_layers)

    policy_effect = evaluate_direct_fusion_effect(
        chronos_bt_results,
        direct_bt_results,
        custom_weight=fusion_policy,
    )
    if (policy_effect['avg_smape_after'], policy_effect['avg_rmse_after']) <= (best_effect['avg_smape_after'], best_effect['avg_rmse_after']):
        return fusion_policy, policy_effect
    return float(best_weight), best_effect


def blend_backtest_results(chronos_bt_results, direct_bt_results, fusion_weight):
    windows_chronos = chronos_bt_results.get('all_windows', [chronos_bt_results])
    windows_direct = direct_bt_results.get('all_windows', [direct_bt_results])

    if len(windows_chronos) != len(windows_direct):
        raise ValueError(
            "Chronos and direct backtest windows mismatch: "
            f"{len(windows_chronos)} vs {len(windows_direct)}."
        )

    fused_windows = []

    for index, (chronos_window, direct_window) in enumerate(zip(windows_chronos, windows_direct), start=1):
        dates = pd.to_datetime(chronos_window['dates'])
        holiday_context_rows = build_holiday_context_rows(dates)
        actual = np.asarray(chronos_window['actual'], dtype=float)
        chronos_pred = np.asarray(chronos_window['predicted'], dtype=float)
        direct_pred = np.asarray(direct_window['predicted'], dtype=float)
        fused_pred = np.zeros_like(chronos_pred, dtype=float)
        for row_idx, dt in enumerate(dates):
            weight = resolve_fusion_weight(dt, fusion_weight, holiday_context=holiday_context_rows[row_idx])
            fused_pred[row_idx] = weight * chronos_pred[row_idx] + (1.0 - weight) * direct_pred[row_idx]
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

    if not fused_windows:
        raise ValueError("No fused backtest windows were produced.")

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
    fused = chronos_future_results.copy()
    direct_predictions = np.asarray(direct_predictions, dtype=float)
    chronos_p10 = np.asarray(chronos_future_results['p10'], dtype=float)
    chronos_p50 = np.asarray(chronos_future_results['p50'], dtype=float)
    chronos_p90 = np.asarray(chronos_future_results['p90'], dtype=float)
    future_dates = pd.to_datetime(chronos_future_results.get('future_dates', []))
    holiday_context_rows = build_holiday_context_rows(future_dates)

    fused_p50 = np.zeros_like(chronos_p50, dtype=float)
    if len(future_dates) == len(chronos_p50):
        for idx, dt in enumerate(future_dates):
            weight = resolve_fusion_weight(dt, fusion_weight, holiday_context=holiday_context_rows[idx])
            fused_p50[idx] = weight * chronos_p50[idx] + (1.0 - weight) * direct_predictions[idx]
    else:
        global_weight = resolve_fusion_weight(pd.Timestamp(datetime.now()), fusion_weight, holiday_context=None)
        fused_p50 = global_weight * chronos_p50 + (1.0 - global_weight) * direct_predictions
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
    configure_spring_service_rules(args.spring_service_rules_json)
    configure_series_tuning(args.series_tuning_json)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_csv_path = os.path.join(project_root, "data.csv")

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

    if args.leadwise_weight_cap < 0.0 or args.leadwise_weight_cap > 1.0:
        print("错误：--leadwise_weight_cap 必须在 [0, 1] 之间。")
        sys.exit(1)

    if args.monitor_recent_days <= 0:
        print("错误：--monitor_recent_days 必须是正整数。")
        sys.exit(1)

    if args.monitor_low_sample_threshold <= 0:
        print("错误：--monitor_low_sample_threshold 必须是正整数。")
        sys.exit(1)

    if args.feature_export_rows_limit < 0:
        print("错误：--feature_export_rows_limit 不能为负数。")
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
        df = load_and_preprocess_data(data_csv_path)
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
        print("[Calendar] 中国节假日日历：未安装 chinese-calendar，仅区分工作日/周末，不识别法定节假日与补班日。")

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

    csv_dir, png_dir, report_dir = ensure_output_dirs()
    feature_export_enabled = not args.disable_feature_export
    call_tuning = get_series_tuning('call_volume')
    ticket_tuning = get_series_tuning('tickets_received')
    print(
        "系列调参："
        f"call(cap={call_tuning['leadwise_weight_cap']:.2f}, holiday_weight={call_tuning['holiday_segment_model_weight']:.2f})，"
        f"tickets(cap={ticket_tuning['leadwise_weight_cap']:.2f}, holiday_weight={ticket_tuning['holiday_segment_model_weight']:.2f})"
    )

    tuning_report_rows = []

    def add_tuning_row(series_name, stage_name, effect_dict):
        tuning_report_rows.append({
            'series_name': str(series_name),
            'stage': str(stage_name),
            'smape_before': float(effect_dict.get('avg_smape_before', 0.0)),
            'smape_after': float(effect_dict.get('avg_smape_after', 0.0)),
            'rmse_before': float(effect_dict.get('avg_rmse_before', 0.0)),
            'rmse_after': float(effect_dict.get('avg_rmse_after', 0.0)),
            'smape_delta': float(effect_dict.get('avg_smape_after', 0.0) - effect_dict.get('avg_smape_before', 0.0)),
            'rmse_delta': float(effect_dict.get('avg_rmse_after', 0.0) - effect_dict.get('avg_rmse_before', 0.0)),
        })

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
        return_feature_frame=feature_export_enabled,
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
        series_tuning=call_tuning,
    )
    bt_call = blend_backtest_results(bt_call_chronos, bt_call_direct, call_direct_fusion_weight)
    fut_call_base = blend_future_results(
        fut_call_chronos_base,
        direct_call_future['predictions'],
        call_direct_fusion_weight,
    )

    residual_adjuster_call = build_residual_adjuster(
        bt_call,
        holiday_segment_model_weight=float(call_tuning.get('holiday_segment_model_weight', 0.75)),
        holiday_layered_config=call_tuning.get('holiday_layered_model', {}),
    )
    residual_adjuster_call, effect_call = search_best_residual_weight(
        bt_call,
        residual_adjuster_call,
        search_points=args.residual_weight_search_points,
    )
    fut_call = apply_residual_adjustment(fut_call_base, residual_adjuster_call)

    if not args.disable_leadwise_correction:
        leadwise_call = build_leadwise_adjuster(bt_call, residual_adjuster_call)
        base_call_cap = float(np.clip(call_tuning.get('leadwise_weight_cap', args.leadwise_weight_cap), 0.0, 1.0))
        dynamic_call_cap, dynamic_call_cap_info = compute_dynamic_leadwise_cap(
            bt_call,
            residual_adjuster_call,
            base_call_cap,
            call_tuning,
        )
        leadwise_call['leadwise_weight_cap'] = float(dynamic_call_cap)
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
        dynamic_call_cap_info = {'mode': 'disabled', 'recent_smape': None}

    if not args.disable_interval_calibration:
        call_interval_cfg = call_tuning.get('asymmetric_interval', {})
        residuals_call = collect_adjusted_backtest_residuals(bt_call, residual_adjuster_call, leadwise_adjuster=leadwise_call)
        fut_call, calib_call = calibrate_prediction_intervals(
            fut_call,
            residuals_call,
            coverage=args.interval_coverage,
            asymmetric=bool(call_interval_cfg.get('enabled', False)),
            lower_coverage=call_interval_cfg.get('lower_coverage', args.interval_coverage),
            upper_coverage=call_interval_cfg.get('upper_coverage', args.interval_coverage),
        )
        interval_bt_call = evaluate_interval_calibration_backtest(
            bt_call,
            residual_adjuster_call,
            coverage=args.interval_coverage,
            leadwise_adjuster=leadwise_call,
            asymmetric=bool(call_interval_cfg.get('enabled', False)),
            lower_coverage=call_interval_cfg.get('lower_coverage', args.interval_coverage),
            upper_coverage=call_interval_cfg.get('upper_coverage', args.interval_coverage),
        )
    else:
        calib_call = {'radius': 0.0, 'lower_radius': 0.0, 'upper_radius': 0.0, 'empirical_coverage': 0.0, 'asymmetric': False}
        interval_bt_call = {
            'avg_coverage': 0.0,
            'avg_width': 0.0,
            'avg_radius': 0.0,
            'avg_lower_radius': 0.0,
            'avg_upper_radius': 0.0,
            'asymmetric': False,
        }

    holiday_anchor_call = estimate_holiday_zero_anchor(df[['date', 'call_volume']].copy(), 'call_volume')
    fut_call = apply_holiday_zero_adjustment(fut_call, holiday_anchor_call)
    spring_anchor_call = estimate_spring_festival_service_anchor(df[['date', 'call_volume']].copy(), 'call_volume')
    fut_call = apply_spring_festival_service_adjustment(fut_call, spring_anchor_call)
    call_bias_snapshot = build_bucket_bias_snapshot(
        bt_call,
        residual_adjuster_call,
        leadwise_adjuster=leadwise_call,
        recent_days=max(1, int(call_tuning.get('bias_gate', {}).get('recent_days', args.monitor_recent_days))),
    )
    fut_call, call_bias_gate_info = apply_bucket_bias_gate(fut_call, call_bias_snapshot, call_tuning)
    print("\n[Residual纠偏] call_volume 已应用修正。")
    print(f"  全局残差均值：{residual_adjuster_call['global_bias']:.4f}")
    print(
        f"  状态小模型数（Ridge）：{len(residual_adjuster_call.get('state_models', {}))}，"
        f"模型融合权重={float(residual_adjuster_call.get('weight_state_model', 0.7)):.2f}"
    )
    layered_call_models = residual_adjuster_call.get('holiday_layered_models', {})
    print(
        f"  节假日分层模型数（Ridge）：{len(layered_call_models)}，"
        f"分层融合权重={float(residual_adjuster_call.get('weight_holiday_layered_model', 0.55)):.2f}"
    )
    if layered_call_models:
        layered_call_samples = {
            key: int(value.get('sample_size', 0))
            for key, value in layered_call_models.items()
        }
        print(f"  节假日分层样本：{layered_call_samples}")
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
    if spring_anchor_call.get('enabled', False):
        print(
            f"  春节停工/值班锚定：停工启用={spring_anchor_call['shutdown_enabled']}"
            f"(median={spring_anchor_call['shutdown_median']:.2f}, strength={spring_anchor_call['shutdown_strength']:.2f})，"
            f"值班启用={spring_anchor_call['duty_enabled']}"
            f"(median={spring_anchor_call['duty_median']:.2f}, p75={spring_anchor_call['duty_p75']:.2f}, strength={spring_anchor_call['duty_strength']:.2f})"
        )
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
        f"    最优融合策略：{format_fusion_policy_summary(call_direct_fusion_weight)}，"
        f"global Direct 权重：{1.0 - get_global_fusion_weight(call_direct_fusion_weight):.2f}"
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
        print(
            "    动态 Lead-wise 上限："
            f"base={base_call_cap:.2f} -> used={leadwise_call.get('leadwise_weight_cap', base_call_cap):.2f} "
            f"(mode={dynamic_call_cap_info.get('mode')}, recent_sMAPE={dynamic_call_cap_info.get('recent_smape') if dynamic_call_cap_info.get('recent_smape') is not None else 'NA'})"
        )
    if bool(call_bias_gate_info.get('enabled', False)):
        details = call_bias_gate_info.get('details', []) or []
        detail_parts = []
        for item in details:
            detail_parts.append(
                f"{item.get('bucket')}[{item.get('scope')}]:sample={int(item.get('sample_size', 0))},bias={float(item.get('bias', 0.0)):.2f},reason={item.get('reason', 'NA')}"
            )
        print(
            "  分桶偏差闸门："
            f"targets={','.join(call_bias_gate_info.get('targets', [])) or 'NA'}，"
            f"applied={int(call_bias_gate_info.get('applied_count', 0))}，"
            f"detail={' | '.join(detail_parts) if detail_parts else 'NA'}"
        )
    if not args.disable_interval_calibration:
        mode_text = "非对称" if bool(calib_call.get('asymmetric', False)) else "对称"
        print(
            f"  区间后校准（{mode_text}）：覆盖率目标={args.interval_coverage:.0%}, "
            f"经验覆盖率={calib_call['empirical_coverage']:.0%}, "
            f"下半径={calib_call.get('lower_radius', calib_call['radius']):.2f}, "
            f"上半径={calib_call.get('upper_radius', calib_call['radius']):.2f}"
        )
        print(
            f"  区间回测复评：平均覆盖率={interval_bt_call['avg_coverage']:.0%}, "
            f"平均区间宽度={interval_bt_call['avg_width']:.2f}, "
            f"平均下半径={interval_bt_call.get('avg_lower_radius', interval_bt_call['avg_radius']):.2f}, "
            f"平均上半径={interval_bt_call.get('avg_upper_radius', interval_bt_call['avg_radius']):.2f}"
        )

    add_tuning_row('call_volume', 'direct_fusion', call_direct_fusion_effect)
    add_tuning_row('call_volume', 'residual', effect_call)
    if lead_effect_call is not None:
        add_tuning_row('call_volume', 'leadwise', lead_effect_call)

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
        return_feature_frame=feature_export_enabled,
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
        series_tuning=ticket_tuning,
    )
    bt_ticket = blend_backtest_results(bt_ticket_chronos, bt_ticket_direct, direct_fusion_weight)
    fut_ticket_base = blend_future_results(
        fut_ticket_chronos_base,
        direct_ticket_future['predictions'],
        direct_fusion_weight,
    )

    feature_export_filename = None
    if feature_export_enabled:
        call_feature_df = direct_call_future['feature_frame'].copy()
        call_feature_df['target_name'] = 'call_volume'
        ticket_feature_df = direct_ticket_future['feature_frame'].copy()
        ticket_feature_df['target_name'] = 'tickets_received'
        merged_feature_df = pd.concat([call_feature_df, ticket_feature_df], ignore_index=True)

        if args.feature_export_rows_limit > 0 and len(merged_feature_df) > args.feature_export_rows_limit:
            merged_feature_df = merged_feature_df.tail(args.feature_export_rows_limit).reset_index(drop=True)
            print(f"[Export] feature_engineering_merged.csv 已按最近样本限流到 {len(merged_feature_df)} 行。")

        feature_export_filename = os.path.join(csv_dir, 'feature_engineering_merged.csv')
        export_dataframe_csv(merged_feature_df, feature_export_filename)
    else:
        print("[Export] 已关闭 feature_engineering_merged.csv 导出。")

    call_model_summary_df = direct_call_future['model_summaries'].copy()
    call_model_summary_df['target_name'] = 'call_volume'
    ticket_model_summary_df = direct_ticket_future['model_summaries'].copy()
    ticket_model_summary_df['target_name'] = 'tickets_received'
    merged_model_summary_df = pd.concat([call_model_summary_df, ticket_model_summary_df], ignore_index=True)
    direct_model_summary_filename = os.path.join(csv_dir, 'direct_model_summary_merged.csv')
    export_dataframe_csv(merged_model_summary_df, direct_model_summary_filename)

    # 3. Residual 纠偏：根据回测残差做全局+周几偏差修正
    residual_adjuster = build_residual_adjuster(
        bt_ticket,
        holiday_segment_model_weight=float(ticket_tuning.get('holiday_segment_model_weight', 0.75)),
        holiday_layered_config=ticket_tuning.get('holiday_layered_model', {}),
    )
    residual_adjuster, effect_ticket = search_best_residual_weight(
        bt_ticket,
        residual_adjuster,
        search_points=args.residual_weight_search_points,
    )
    fut_ticket = apply_residual_adjustment(fut_ticket_base, residual_adjuster)

    if not args.disable_leadwise_correction:
        leadwise_ticket = build_leadwise_adjuster(bt_ticket, residual_adjuster)
        base_ticket_cap = float(np.clip(ticket_tuning.get('leadwise_weight_cap', args.leadwise_weight_cap), 0.0, 1.0))
        dynamic_ticket_cap, dynamic_ticket_cap_info = compute_dynamic_leadwise_cap(
            bt_ticket,
            residual_adjuster,
            base_ticket_cap,
            ticket_tuning,
        )
        leadwise_ticket['leadwise_weight_cap'] = float(dynamic_ticket_cap)
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
        dynamic_ticket_cap_info = {'mode': 'disabled', 'recent_smape': None}

    if not args.disable_interval_calibration:
        ticket_interval_cfg = ticket_tuning.get('asymmetric_interval', {})
        residuals_ticket = collect_adjusted_backtest_residuals(bt_ticket, residual_adjuster, leadwise_adjuster=leadwise_ticket)
        fut_ticket, calib_ticket = calibrate_prediction_intervals(
            fut_ticket,
            residuals_ticket,
            coverage=args.interval_coverage,
            asymmetric=bool(ticket_interval_cfg.get('enabled', False)),
            lower_coverage=ticket_interval_cfg.get('lower_coverage', args.interval_coverage),
            upper_coverage=ticket_interval_cfg.get('upper_coverage', args.interval_coverage),
        )
        interval_bt_ticket = evaluate_interval_calibration_backtest(
            bt_ticket,
            residual_adjuster,
            coverage=args.interval_coverage,
            leadwise_adjuster=leadwise_ticket,
            asymmetric=bool(ticket_interval_cfg.get('enabled', False)),
            lower_coverage=ticket_interval_cfg.get('lower_coverage', args.interval_coverage),
            upper_coverage=ticket_interval_cfg.get('upper_coverage', args.interval_coverage),
        )
    else:
        calib_ticket = {'radius': 0.0, 'lower_radius': 0.0, 'upper_radius': 0.0, 'empirical_coverage': 0.0, 'asymmetric': False}
        interval_bt_ticket = {
            'avg_coverage': 0.0,
            'avg_width': 0.0,
            'avg_radius': 0.0,
            'avg_lower_radius': 0.0,
            'avg_upper_radius': 0.0,
            'asymmetric': False,
        }

    holiday_anchor_ticket = estimate_holiday_zero_anchor(df[['date', 'tickets_received']].copy(), 'tickets_received')
    fut_ticket = apply_holiday_zero_adjustment(fut_ticket, holiday_anchor_ticket)
    spring_anchor_ticket = estimate_spring_festival_service_anchor(df[['date', 'tickets_received']].copy(), 'tickets_received')
    fut_ticket = apply_spring_festival_service_adjustment(fut_ticket, spring_anchor_ticket)
    ticket_bias_snapshot = build_bucket_bias_snapshot(
        bt_ticket,
        residual_adjuster,
        leadwise_adjuster=leadwise_ticket,
        recent_days=max(1, int(ticket_tuning.get('bias_gate', {}).get('recent_days', args.monitor_recent_days))),
    )
    fut_ticket, ticket_bias_gate_info = apply_bucket_bias_gate(fut_ticket, ticket_bias_snapshot, ticket_tuning)
    print("\n[Residual纠偏] tickets_received 已应用修正。")
    print(f"  全局残差均值：{residual_adjuster['global_bias']:.4f}")
    print(
        f"  状态小模型数（Ridge）：{len(residual_adjuster.get('state_models', {}))}，"
        f"模型融合权重={float(residual_adjuster.get('weight_state_model', 0.7)):.2f}"
    )
    layered_ticket_models = residual_adjuster.get('holiday_layered_models', {})
    print(
        f"  节假日分层模型数（Ridge）：{len(layered_ticket_models)}，"
        f"分层融合权重={float(residual_adjuster.get('weight_holiday_layered_model', 0.55)):.2f}"
    )
    if layered_ticket_models:
        layered_ticket_samples = {
            key: int(value.get('sample_size', 0))
            for key, value in layered_ticket_models.items()
        }
        print(f"  节假日分层样本：{layered_ticket_samples}")
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
    if spring_anchor_ticket.get('enabled', False):
        print(
            f"  春节停工/值班锚定：停工启用={spring_anchor_ticket['shutdown_enabled']}"
            f"(median={spring_anchor_ticket['shutdown_median']:.2f}, strength={spring_anchor_ticket['shutdown_strength']:.2f})，"
            f"值班启用={spring_anchor_ticket['duty_enabled']}"
            f"(median={spring_anchor_ticket['duty_median']:.2f}, p75={spring_anchor_ticket['duty_p75']:.2f}, strength={spring_anchor_ticket['duty_strength']:.2f})"
        )
    print("  Chronos + Direct 融合复评（融合前 -> 融合后）：")
    print(
        f"    平均 sMAPE：{direct_fusion_effect['avg_smape_before']:.2%} -> {direct_fusion_effect['avg_smape_after']:.2%} "
        f"(变化 {direct_fusion_effect['avg_smape_after'] - direct_fusion_effect['avg_smape_before']:+.2%})"
    )
    print(
        f"    平均 RMSE：{direct_fusion_effect['avg_rmse_before']:.2f} -> {direct_fusion_effect['avg_rmse_after']:.2f} "
        f"(变化 {direct_fusion_effect['avg_rmse_after'] - direct_fusion_effect['avg_rmse_before']:+.2f})"
    )
    print(
        f"    最优融合策略：{format_fusion_policy_summary(direct_fusion_weight)}，"
        f"global Direct 权重：{1.0 - get_global_fusion_weight(direct_fusion_weight):.2f}"
    )
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
        print(
            "    动态 Lead-wise 上限："
            f"base={base_ticket_cap:.2f} -> used={leadwise_ticket.get('leadwise_weight_cap', base_ticket_cap):.2f} "
            f"(mode={dynamic_ticket_cap_info.get('mode')}, recent_sMAPE={dynamic_ticket_cap_info.get('recent_smape') if dynamic_ticket_cap_info.get('recent_smape') is not None else 'NA'})"
        )
    if bool(ticket_bias_gate_info.get('enabled', False)):
        details = ticket_bias_gate_info.get('details', []) or []
        detail_parts = []
        for item in details:
            detail_parts.append(
                f"{item.get('bucket')}[{item.get('scope')}]:sample={int(item.get('sample_size', 0))},bias={float(item.get('bias', 0.0)):.2f},reason={item.get('reason', 'NA')}"
            )
        print(
            "  分桶偏差闸门："
            f"targets={','.join(ticket_bias_gate_info.get('targets', [])) or 'NA'}，"
            f"applied={int(ticket_bias_gate_info.get('applied_count', 0))}，"
            f"detail={' | '.join(detail_parts) if detail_parts else 'NA'}"
        )
    if not args.disable_interval_calibration:
        mode_text_ticket = "非对称" if bool(calib_ticket.get('asymmetric', False)) else "对称"
        print(
            f"  区间后校准（{mode_text_ticket}）：覆盖率目标={args.interval_coverage:.0%}, "
            f"经验覆盖率={calib_ticket['empirical_coverage']:.0%}, "
            f"下半径={calib_ticket.get('lower_radius', calib_ticket['radius']):.2f}, "
            f"上半径={calib_ticket.get('upper_radius', calib_ticket['radius']):.2f}"
        )
        print(
            f"  区间回测复评：平均覆盖率={interval_bt_ticket['avg_coverage']:.0%}, "
            f"平均区间宽度={interval_bt_ticket['avg_width']:.2f}, "
            f"平均下半径={interval_bt_ticket.get('avg_lower_radius', interval_bt_ticket['avg_radius']):.2f}, "
            f"平均上半径={interval_bt_ticket.get('avg_upper_radius', interval_bt_ticket['avg_radius']):.2f}"
        )

    add_tuning_row('tickets_received', 'direct_fusion', direct_fusion_effect)
    add_tuning_row('tickets_received', 'residual', effect_ticket)
    if lead_effect_ticket is not None:
        add_tuning_row('tickets_received', 'leadwise', lead_effect_ticket)

    tuning_report_filename = os.path.join(csv_dir, 'tuning_report.csv')
    tuning_report_png = os.path.join(png_dir, 'tuning_report.png')
    if tuning_report_rows:
        tuning_report_df = pd.DataFrame(tuning_report_rows)
        export_dataframe_csv(tuning_report_df, tuning_report_filename)
        plot_tuning_report(tuning_report_df, tuning_report_png)
        print(f"已导出调参效果报告：{tuning_report_filename}")
    else:
        tuning_report_df = pd.DataFrame()
        print("调参效果报告：无可用复评数据，跳过导出。")

    # 4. 导出标准化 CSV（作为 API 数据源）
    export_filename = os.path.join(csv_dir, 'forecast_export.csv')
    export_forecast_csv(fut_call, fut_ticket, export_filename)

    monitor_call_df = build_monitor_bucket_report(
        bt_call,
        residual_adjuster_call,
        leadwise_adjuster=leadwise_call,
        series_name='call_volume',
        recent_days=args.monitor_recent_days,
        series_tuning=call_tuning,
    )
    monitor_ticket_df = build_monitor_bucket_report(
        bt_ticket,
        residual_adjuster,
        leadwise_adjuster=leadwise_ticket,
        series_name='tickets_received',
        recent_days=args.monitor_recent_days,
        series_tuning=ticket_tuning,
    )
    monitor_report_df = pd.concat([monitor_call_df, monitor_ticket_df], ignore_index=True)
    monitor_history_call_df = build_history_bucket_profile(
        df[['date', 'call_volume']].copy(),
        'call_volume',
        series_name='call_volume',
        recent_days=args.monitor_recent_days,
    )
    monitor_history_ticket_df = build_history_bucket_profile(
        df[['date', 'tickets_received']].copy(),
        'tickets_received',
        series_name='tickets_received',
        recent_days=args.monitor_recent_days,
    )
    monitor_history_profile_df = pd.concat(
        [monitor_history_call_df, monitor_history_ticket_df],
        ignore_index=True,
    )
    monitor_report_filename = os.path.join(csv_dir, 'monitor_bucket_report.csv')
    monitor_history_profile_filename = os.path.join(csv_dir, 'monitor_bucket_history_profile.csv')
    if not monitor_report_df.empty:
        export_dataframe_csv(monitor_report_df, monitor_report_filename)
        print(f"已导出分桶监控报告：{monitor_report_filename}")
    if not monitor_history_profile_df.empty:
        export_dataframe_csv(monitor_history_profile_df, monitor_history_profile_filename)
        print(f"已导出历史分桶画像：{monitor_history_profile_filename}")

    print(f"\n已导出标准预测数据：{export_filename}")
    if feature_export_filename is not None:
        print(f"已导出 call+tickets 特征工程明细：{feature_export_filename}")
    else:
        print("已跳过 call+tickets 特征工程明细导出。")
    print(f"已导出 call+tickets direct 模型摘要：{direct_model_summary_filename}")

    # 5. 基于 CSV 进行绘图
    eval_png = os.path.join(png_dir, 'evaluation_results.png')
    future_png = os.path.join(png_dir, 'future_forecast.png')
    export_png = os.path.join(png_dir, 'forecast_export_plot.png')
    monitor_png = os.path.join(png_dir, 'monitor_bucket_report.png')
    monitor_sample_scope_png = os.path.join(png_dir, 'monitor_bucket_sample_scope.png')
    stage_transition_png = os.path.join(png_dir, 'stage_transition_backtest.png')
    bt_call_plot = apply_holiday_zero_adjustment_to_backtest(bt_call, holiday_anchor_call)
    bt_ticket_plot = apply_holiday_zero_adjustment_to_backtest(bt_ticket, holiday_anchor_ticket)
    bt_call_plot = apply_spring_festival_service_adjustment_to_backtest(bt_call_plot, spring_anchor_call)
    bt_ticket_plot = apply_spring_festival_service_adjustment_to_backtest(bt_ticket_plot, spring_anchor_ticket)

    plot_evaluation(bt_call_plot, bt_ticket_plot, eval_png)
    plot_future_from_csv(data_csv_path, export_filename, future_png)
    plot_forecast_export_csv(export_filename, export_png)
    if not monitor_report_df.empty:
        plot_monitor_bucket_report(monitor_report_df, monitor_png)
    if (not monitor_report_df.empty) and (not monitor_history_profile_df.empty):
        plot_monitor_bucket_sample_scope(monitor_report_df, monitor_history_profile_df, monitor_sample_scope_png)

    stage_transition_call_df = build_stage_transition_frame(
        bt_call_chronos,
        bt_call_direct,
        call_direct_fusion_weight,
        residual_adjuster_call,
        leadwise_adjuster=leadwise_call,
        series_name='call_volume',
        latest_only=True,
    )
    stage_transition_ticket_df = build_stage_transition_frame(
        bt_ticket_chronos,
        bt_ticket_direct,
        direct_fusion_weight,
        residual_adjuster,
        leadwise_adjuster=leadwise_ticket,
        series_name='tickets_received',
        latest_only=True,
    )
    stage_transition_df = pd.concat([stage_transition_call_df, stage_transition_ticket_df], ignore_index=True)
    stage_transition_filename = os.path.join(csv_dir, 'stage_transition_latest_window.csv')
    stage_transition_summary_filename = os.path.join(csv_dir, 'stage_transition_summary.csv')
    if not stage_transition_df.empty:
        export_dataframe_csv(stage_transition_df, stage_transition_filename)
        stage_transition_summary_df = build_stage_transition_summary(stage_transition_df)
        if not stage_transition_summary_df.empty:
            export_dataframe_csv(stage_transition_summary_df, stage_transition_summary_filename)
        plot_stage_transition(stage_transition_df, stage_transition_png)
        print(f"已导出分阶段预测变化明细：{stage_transition_filename}")
        if not stage_transition_summary_df.empty:
            print(f"已导出分阶段指标汇总：{stage_transition_summary_filename}")
    else:
        stage_transition_summary_df = pd.DataFrame()

    expected_pngs = [eval_png, future_png, export_png]
    if tuning_report_rows:
        expected_pngs.append(tuning_report_png)
    if not monitor_report_df.empty:
        expected_pngs.append(monitor_png)
    if (not monitor_report_df.empty) and (not monitor_history_profile_df.empty):
        expected_pngs.append(monitor_sample_scope_png)
    if not stage_transition_df.empty:
        expected_pngs.append(stage_transition_png)
    failed_pngs = [
        path for path in expected_pngs
        if (not os.path.exists(path)) or os.path.getsize(path) == 0
    ]
    if failed_pngs:
        print("\n错误：绘图导出失败，以下 PNG 不存在或为空文件：")
        for path in failed_pngs:
            print(f"  - {os.path.abspath(path)}")
        sys.exit(1)

    report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = os.path.join(report_dir, f'forecast_interpretation_report_{report_timestamp}.md')
    forecast_export_df = pd.read_csv(export_filename)
    report_text = build_forecast_interpretation_report(
        tuning_report_df,
        monitor_report_df,
        monitor_history_profile_df,
        forecast_export_df,
        stage_transition_summary_df,
        image_links={
            'evaluation': os.path.relpath(eval_png, report_dir).replace('\\', '/'),
            'future': os.path.relpath(future_png, report_dir).replace('\\', '/'),
            'export': os.path.relpath(export_png, report_dir).replace('\\', '/'),
            'tuning': os.path.relpath(tuning_report_png, report_dir).replace('\\', '/') if tuning_report_rows else None,
            'monitor': os.path.relpath(monitor_png, report_dir).replace('\\', '/') if not monitor_report_df.empty else None,
            'monitor_sample_scope': os.path.relpath(monitor_sample_scope_png, report_dir).replace('\\', '/') if ((not monitor_report_df.empty) and (not monitor_history_profile_df.empty)) else None,
            'stage_transition': os.path.relpath(stage_transition_png, report_dir).replace('\\', '/') if not stage_transition_df.empty else None,
        },
        monitor_low_sample_threshold=args.monitor_low_sample_threshold,
        monitor_recent_days=args.monitor_recent_days,
    )
    export_markdown_report(report_text, report_filename)

    print(
        "\n预测完成。请查看 "
        f"{os.path.abspath(export_filename)}、"
        f"{os.path.abspath(eval_png)}、"
        f"{os.path.abspath(future_png)}、"
        f"{os.path.abspath(export_png)}。"
    )
    if tuning_report_rows:
        print(f"调参对比图：{os.path.abspath(tuning_report_png)}")
    if not monitor_report_df.empty:
        print(f"分桶监控图：{os.path.abspath(monitor_png)}")
    if (not monitor_report_df.empty) and (not monitor_history_profile_df.empty):
        print(f"样本口径对照图：{os.path.abspath(monitor_sample_scope_png)}")
    if not stage_transition_df.empty:
        print(f"分阶段预测变化图：{os.path.abspath(stage_transition_png)}")
    print(f"预测解读报告：{os.path.abspath(report_filename)}")

if __name__ == "__main__":
    main()
