import argparse
import json
import os
import sys
from copy import deepcopy

import yaml


QUICK_PROFILE_OVERRIDES = {
    "rolling_windows": 2,
    "backtest_horizon": 21,
    "context_search_points": 20,
    "context_ensemble_topk": 2,
    "residual_weight_search_points": 11,
    "direct_weight_search_points": 7,
    "leadwise_weight_search_points": 7,
    "leadwise_weight_cap": 0.8,
    "disable_interval_calibration": True,
}


def choose_runtime_profile(profile_arg: str | None) -> str:
    if profile_arg in {"quick", "formal"}:
        return profile_arg

    if not sys.stdin.isatty():
        print("[profile] Non-interactive session detected, defaulting to formal profile.")
        return "formal"

    while True:
        raw = input("Select run profile (quick/formal): ").strip().lower()
        if raw in {"quick", "formal"}:
            return raw
        print("Invalid profile. Please enter 'quick' or 'formal'.")


def apply_profile_overrides(cfg: dict, profile: str) -> dict:
    effective = deepcopy(cfg)
    if profile == "quick":
        effective.update(QUICK_PROFILE_OVERRIDES)
    return effective


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
        "leadwise_weight_cap": "--leadwise_weight_cap",
        "monitor_recent_days": "--monitor_recent_days",
        "monitor_low_sample_threshold": "--monitor_low_sample_threshold",
        "feature_export_rows_limit": "--feature_export_rows_limit",
        "spring_service_rules": "--spring_service_rules_json",
        "series_tuning": "--series_tuning_json",
    }

    for key, flag in flag_map.items():
        value = cfg.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            args.extend([flag, json.dumps(value, ensure_ascii=False)])
        else:
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

    if bool(cfg.get("disable_feature_export", False)):
        args.append("--disable_feature_export")

    return args


def main() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Project entrypoint using config.yaml + CLI overrides")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(project_root, "config.yaml"),
        help="Path to global config file",
    )
    parser.add_argument("--target_date", type=str, default=None, help="Override target_date in config")
    parser.add_argument(
        "--profile",
        type=str,
        choices=["quick", "formal"],
        default=None,
        help="Run profile. If omitted, you will be prompted each run.",
    )
    args, passthrough = parser.parse_known_args()

    cfg = load_config(args.config)
    if args.target_date:
        cfg["target_date"] = args.target_date

    selected_profile = choose_runtime_profile(args.profile)
    cfg = apply_profile_overrides(cfg, selected_profile)
    print(f"[profile] Using '{selected_profile}' profile")

    if not cfg.get("target_date"):
        raise ValueError("target_date is required. Set it in config.yaml or pass --target_date.")

    forecast_argv = build_forecast_argv(cfg)
    sys.argv = ["forecast.py", *forecast_argv, *passthrough]

    from src.forecast import main as forecast_main

    forecast_main()


if __name__ == "__main__":
    main()
