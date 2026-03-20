import argparse
import os
import sys

import yaml


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a key-value mapping.")
    return data


def build_forecast_argv(cfg: dict) -> list[str]:
    args = []

    flag_map = {
        "target_date": "--target_date",
        "model_id": "--model_id",
        "hf_endpoint": "--hf_endpoint",
        "context_length": "--context_length",
        "backtest_horizon": "--backtest_horizon",
        "rolling_windows": "--rolling_windows",
        "context_candidates": "--context_candidates",
        "context_search_points": "--context_search_points",
        "residual_weight_search_points": "--residual_weight_search_points",
        "direct_weight_search_points": "--direct_weight_search_points",
        "context_ensemble_topk": "--context_ensemble_topk",
        "interval_coverage": "--interval_coverage",
        "leadwise_weight_search_points": "--leadwise_weight_search_points",
    }

    for key, flag in flag_map.items():
        value = cfg.get(key)
        if value is None:
            continue
        args.extend([flag, str(value)])

    if bool(cfg.get("local_files_only", False)):
        args.append("--local_files_only")

    if bool(cfg.get("auto_backtest_horizon", True)):
        args.append("--auto_backtest_horizon")
    else:
        args.append("--no_auto_backtest_horizon")

    if bool(cfg.get("disable_interval_calibration", False)):
        args.append("--disable_interval_calibration")

    if bool(cfg.get("disable_leadwise_correction", False)):
        args.append("--disable_leadwise_correction")

    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Project entrypoint using config.yaml + CLI overrides")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to global config file")
    parser.add_argument("--target_date", type=str, default=None, help="Override target_date in config")
    args, passthrough = parser.parse_known_args()

    cfg = load_config(args.config)
    if args.target_date:
        cfg["target_date"] = args.target_date

    if not cfg.get("target_date"):
        raise ValueError("target_date is required. Set it in config.yaml or pass --target_date.")

    forecast_argv = build_forecast_argv(cfg)
    sys.argv = ["forecast.py", *forecast_argv, *passthrough]

    from src.forecast import main as forecast_main

    forecast_main()


if __name__ == "__main__":
    main()
