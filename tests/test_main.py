from main import build_forecast_argv


def test_build_forecast_argv_includes_feature_export_controls():
    cfg = {
        "target_date": "2026-04-30",
        "disable_feature_export": True,
        "feature_export_rows_limit": 123,
        "monitor_low_sample_threshold": 15,
        "auto_backtest_horizon": True,
    }

    argv = build_forecast_argv(cfg)

    assert "--disable_feature_export" in argv
    assert "--feature_export_rows_limit" in argv
    assert "123" in argv
    assert "--monitor_low_sample_threshold" in argv
    assert "15" in argv
