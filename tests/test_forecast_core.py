import numpy as np
import pandas as pd
import pytest

from src.forecast import (
    apply_bucket_bias_gate,
    blend_backtest_results,
    build_forecast_interpretation_report,
    build_history_bucket_profile,
    build_monitor_bucket_report,
    calibrate_prediction_intervals,
    classify_holiday_blocks,
    generate_auto_context_candidates,
    is_china_holiday,
    parse_context_candidates,
)
import src.forecast as forecast_module
from src.visualization import plot_monitor_bucket_sample_scope
from src.output_manager import export_forecast_csv
from src.visualization import plot_evaluation


def test_parse_context_candidates_sorted_deduplicated():
    values = parse_context_candidates("120,30,120,60")
    assert values == [30, 60, 120]


def test_generate_auto_context_candidates_keeps_upper_bound():
    values = generate_auto_context_candidates(97, points=5, min_context=30)
    assert values[-1] == 97


def test_blend_backtest_results_mismatch_raises():
    chronos_bt = {
        "horizon": 3,
        "all_windows": [{"actual": np.array([1.0]), "predicted": np.array([1.0]), "dates": np.array(["2026-01-01"])}],
    }
    direct_bt = {
        "horizon": 3,
        "all_windows": [
            {"actual": np.array([1.0]), "predicted": np.array([1.0]), "dates": np.array(["2026-01-01"])},
            {"actual": np.array([1.0]), "predicted": np.array([1.0]), "dates": np.array(["2026-01-02"])},
        ],
    }

    with pytest.raises(ValueError, match="windows mismatch"):
        blend_backtest_results(chronos_bt, direct_bt, fusion_weight=0.5)


def test_calibrate_prediction_intervals_empty_residuals():
    future = {
        "p10": np.array([1.0, 2.0]),
        "p50": np.array([2.0, 3.0]),
        "p90": np.array([3.0, 4.0]),
    }
    calibrated, info = calibrate_prediction_intervals(future, np.array([]), coverage=0.8)

    assert np.array_equal(calibrated["p10"], future["p10"])
    assert np.array_equal(calibrated["p90"], future["p90"])
    assert info["radius"] == 0.0


def test_build_history_bucket_profile_has_expected_scopes_and_columns():
    history_df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=14, freq="D"),
            "call_volume": [10, 11, 0, 12, 14, 0, 13, 12, 11, 10, 9, 8, 0, 7],
        }
    )

    profile_df = build_history_bucket_profile(
        history_df,
        target_col="call_volume",
        series_name="call_volume",
        recent_days=7,
    )

    assert not profile_df.empty
    assert set(["history_all", "history_recent"]).issubset(set(profile_df["scope"].unique()))
    assert {"series_name", "scope", "bucket", "hist_sample_size", "hist_zero_ratio"}.issubset(profile_df.columns)


def test_build_forecast_interpretation_report_supports_history_scope_section():
    tuning_report_df = pd.DataFrame(
        {
            "series_name": ["call_volume"],
            "stage": ["residual"],
            "smape_before": [0.2],
            "smape_after": [0.18],
            "smape_delta": [-0.02],
            "rmse_before": [20.0],
            "rmse_after": [18.0],
            "rmse_delta": [-2.0],
        }
    )
    monitor_report_df = pd.DataFrame(
        {
            "series_name": ["call_volume"],
            "scope": ["recent"],
            "bucket": ["workday_normal"],
            "sample_size": [6],
            "smape": [0.15],
            "rmse": [10.0],
            "mae": [8.0],
            "bias": [1.0],
            "mean_actual": [50.0],
            "mean_predicted": [51.0],
        }
    )
    history_profile_df = pd.DataFrame(
        {
            "series_name": ["call_volume", "call_volume"],
            "scope": ["history_recent", "history_all"],
            "bucket": ["workday_normal", "workday_normal"],
            "hist_sample_size": [28, 302],
            "hist_mean": [49.0, 56.0],
            "hist_std": [8.5, 9.5],
            "hist_zero_ratio": [0.05, 0.04],
            "hist_min": [0.0, 0.0],
            "hist_max": [80.0, 95.0],
        }
    )
    forecast_export_df = pd.DataFrame(
        {
            "date": ["2026-04-01"],
            "target_name": ["call_volume"],
            "p10": [30.0],
            "p50": [40.0],
            "p90": [50.0],
        }
    )

    report_text = build_forecast_interpretation_report(
        tuning_report_df,
        monitor_report_df,
        history_profile_df,
        forecast_export_df,
        pd.DataFrame(),
        image_links={},
        monitor_low_sample_threshold=12,
        monitor_recent_days=84,
    )

    assert "样本口径说明" in report_text
    assert "低置信度" in report_text
    assert "历史样本数(全历史)" in report_text
    assert "常见误解" in report_text


def test_plot_monitor_bucket_sample_scope_includes_all_history_series(tmp_path):
    monitor_report_df = pd.DataFrame(
        {
            "series_name": ["call_volume", "call_volume"],
            "scope": ["recent", "recent"],
            "bucket": ["workday_normal", "holiday"],
            "sample_size": [22, 29],
        }
    )
    history_profile_df = pd.DataFrame(
        {
            "series_name": ["call_volume", "call_volume", "call_volume", "call_volume"],
            "scope": ["history_recent", "history_recent", "history_all", "history_all"],
            "bucket": ["workday_normal", "holiday", "workday_normal", "holiday"],
            "hist_sample_size": [20, 29, 302, 377],
            "hist_zero_ratio": [0.0, 0.1, 0.0, 0.1],
        }
    )
    output_path = tmp_path / "sample_scope.png"

    plot_monitor_bucket_sample_scope(monitor_report_df, history_profile_df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_is_china_holiday_excludes_plain_weekend():
    # Regular weekend should not be treated as statutory holiday.
    assert is_china_holiday(pd.Timestamp('2026-03-21')) is False


def test_classify_holiday_blocks_no_weekend_in_holiday_mask():
    # A month without statutory holidays should not produce holiday/post-holiday tags from weekends.
    dates = pd.date_range('2026-03-01', '2026-03-31', freq='D')
    info = classify_holiday_blocks(dates)
    assert int(np.sum(info['is_holiday_non_makeup'])) == 0
    assert int(np.sum(info['post_holiday_workday_n'])) == 0


def test_apply_bucket_bias_gate_supports_multi_targets_with_scope_fallback(monkeypatch):
    monkeypatch.setattr(forecast_module, 'build_holiday_context_rows', lambda dates: [{} for _ in dates])

    def _mock_bucket(dt, _ctx):
        day = pd.to_datetime(dt).day
        if day == 1:
            return 'holiday'
        if day == 2:
            return 'post_holiday_workday_1_3'
        return 'workday_normal'

    monkeypatch.setattr(forecast_module, '_classify_monitor_bucket', _mock_bucket)

    future_results = {
        'future_dates': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03']),
        'p10': np.array([9.0, 9.0, 9.0]),
        'p50': np.array([10.0, 10.0, 10.0]),
        'p90': np.array([11.0, 11.0, 11.0]),
    }
    bias_snapshot = {
        'recent': {
            'holiday': {'sample_size': 2, 'bias': 3.0},
            'post_holiday_workday_1_3': {'sample_size': 12, 'bias': 2.0},
        },
        'all': {
            'holiday': {'sample_size': 20, 'bias': 2.0},
            'post_holiday_workday_1_3': {'sample_size': 30, 'bias': 1.5},
        },
    }
    series_tuning = {
        'bias_gate': {
            'enabled': True,
            'target_buckets': ['holiday', 'post_holiday_workday_1_3'],
            'min_samples': 10,
            'allow_scope_fallback': True,
            'fallback_scale': 0.5,
            'bias_trigger': 0.0,
            'adjustment_scale': 1.0,
            'max_adjustment_ratio': 1.0,
            'only_positive_bias': False,
        }
    }

    adjusted, info = apply_bucket_bias_gate(future_results, bias_snapshot, series_tuning)

    # holiday: recent sample不足，回退all并乘fallback_scale，10 - (2 * 0.5) = 9
    assert adjusted['p50'][0] == pytest.approx(9.0)
    # post_holiday: recent样本足够，按recent偏差修正，10 - 2 = 8
    assert adjusted['p50'][1] == pytest.approx(8.0)
    # non-target bucket unchanged
    assert adjusted['p50'][2] == pytest.approx(10.0)
    assert int(info.get('applied_count', 0)) == 2
    assert set(info.get('targets', [])) == {'holiday', 'post_holiday_workday_1_3'}


def test_build_monitor_bucket_report_applies_bias_gate_to_backtest(monkeypatch):
    monkeypatch.setattr(forecast_module, 'build_holiday_context_rows', lambda dates: [{} for _ in dates])
    monkeypatch.setattr(forecast_module, '_classify_monitor_bucket', lambda dt, _ctx: 'post_holiday_workday_1_3')

    bt_results = {
        'dates': pd.to_datetime(['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-04']),
        'actual': np.array([10.0, 10.0, 10.0, 10.0]),
        'predicted': np.array([20.0, 20.0, 20.0, 20.0]),
    }
    residual_adjuster = {
        'global_bias': 0.0,
        'state_bias': {},
        'state_dow_bias': {},
        'state_month_phase_bias': {},
        'weight_state': 1.0,
        'weight_dow': 0.0,
        'weight_month': 0.0,
        'weight_state_model': 0.0,
        'residual_std': 0.0,
    }
    series_tuning = {
        'bias_gate': {
            'enabled': True,
            'apply_to_backtest': True,
            'target_buckets': ['post_holiday_workday_1_3'],
            'min_samples': 1,
            'allow_scope_fallback': False,
            'allow_low_sample_gate': False,
            'only_positive_bias': True,
            'bias_trigger': 0.0,
            'adjustment_scale': 1.0,
            'max_adjustment_ratio': 1.0,
        }
    }

    report_df = build_monitor_bucket_report(
        bt_results,
        residual_adjuster,
        leadwise_adjuster=None,
        series_name='call_volume',
        recent_days=84,
        series_tuning=series_tuning,
    )

    row = report_df.loc[(report_df['scope'] == 'recent') & (report_df['bucket'] == 'post_holiday_workday_1_3')].iloc[0]
    assert float(row['bias']) < 10.0


def test_apply_bucket_bias_gate_rmse_guard_limits_adjustment(monkeypatch):
    monkeypatch.setattr(forecast_module, 'build_holiday_context_rows', lambda dates: [{} for _ in dates])
    monkeypatch.setattr(forecast_module, '_classify_monitor_bucket', lambda dt, _ctx: 'holiday')

    future_results = {
        'future_dates': pd.to_datetime(['2026-01-01']),
        'p10': np.array([90.0]),
        'p50': np.array([100.0]),
        'p90': np.array([110.0]),
    }
    bias_snapshot = {
        'recent': {
            'holiday': {
                'sample_size': 20,
                'bias': 50.0,
                'rmse': 5.0,
            },
        },
    }
    series_tuning = {
        'bias_gate': {
            'enabled': True,
            'target_buckets': ['holiday'],
            'min_samples': 1,
            'allow_scope_fallback': False,
            'only_positive_bias': True,
            'bias_trigger': 0.0,
            'adjustment_scale': 1.0,
            'max_adjustment_ratio': 1.0,
            'enforce_rmse_guard': True,
            'rmse_guard_scale': 0.2,
        }
    }

    adjusted, info = apply_bucket_bias_gate(future_results, bias_snapshot, series_tuning)

    # raw_adj=50, 但rmse护栏上限=0.2*5=1，因此仅调整1
    assert adjusted['p50'][0] == pytest.approx(99.0)
    assert bool(info.get('enforce_rmse_guard', False)) is True


def test_export_forecast_csv_raises_on_length_mismatch(tmp_path):
    fut_call = {
        'future_dates': pd.to_datetime(['2026-01-01', '2026-01-02']),
        'p10': np.array([1.0, 2.0]),
        'p50': np.array([2.0, 3.0]),
        'p90': np.array([3.0]),
    }
    fut_ticket = {
        'future_dates': pd.to_datetime(['2026-01-01']),
        'p10': np.array([1.0]),
        'p50': np.array([2.0]),
        'p90': np.array([3.0]),
    }

    with pytest.raises(ValueError):
        export_forecast_csv(fut_call, fut_ticket, tmp_path / 'forecast.csv')


def test_plot_evaluation_uses_all_windows_payload(tmp_path):
    import matplotlib

    matplotlib.use('Agg')

    bt_call = {
        'windows_used': 2,
        'horizon': 2,
        'dates': np.array(['2026-01-03', '2026-01-04']),
        'actual': np.array([3.0, 4.0]),
        'predicted': np.array([3.1, 4.1]),
        'all_windows': [
            {
                'window_index': 1,
                'dates': np.array(['2026-01-03', '2026-01-04']),
                'actual': np.array([3.0, 4.0]),
                'predicted': np.array([3.1, 4.1]),
            },
            {
                'window_index': 2,
                'dates': np.array(['2026-01-01', '2026-01-02']),
                'actual': np.array([1.0, 2.0]),
                'predicted': np.array([1.1, 2.1]),
            },
        ],
    }
    bt_ticket = {
        'windows_used': 2,
        'horizon': 2,
        'dates': np.array(['2026-01-03', '2026-01-04']),
        'actual': np.array([6.0, 8.0]),
        'predicted': np.array([5.9, 7.9]),
        'all_windows': [
            {
                'window_index': 1,
                'dates': np.array(['2026-01-03', '2026-01-04']),
                'actual': np.array([6.0, 8.0]),
                'predicted': np.array([5.9, 7.9]),
            },
            {
                'window_index': 2,
                'dates': np.array(['2026-01-01', '2026-01-02']),
                'actual': np.array([2.0, 4.0]),
                'predicted': np.array([2.1, 4.1]),
            },
        ],
    }

    output_path = tmp_path / 'evaluation_results.png'
    plot_evaluation(bt_call, bt_ticket, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
