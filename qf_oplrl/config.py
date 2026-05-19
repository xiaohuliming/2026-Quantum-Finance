from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "price_column_preference": ["adj_close", "close"],
        "date_column_candidates": ["date", "datetime"],
        "ticker_column_candidates": ["ticker", "tic", "symbol"],
        "keep_all_tickers": True,
        "max_missing_ratio": 0.2,
    },
    "split": {
        "train_ratio": 0.6,
        "val_ratio": 0.2,
        "test_ratio": 0.2,
    },
    "backtest": {
        "initial_capital": 1.0,
        "transaction_cost_rate": 0.001,
        "annualization_factor": 252,
    },
    "classical": {
        "min_variance": {
            "lookback_window": 60,
        },
        "mean_variance": {
            "lookback_window": 60,
            "risk_aversion": 10.0,
        },
    },
    "opl": {
        "ons": {
            "beta": 1.0,
            "delta": 0.125,
            "eta": 0.01,
        },
        "pamr": {
            "epsilon": 0.5,
            "C": 500.0,
            "variant": 0,
        },
        "olmar": {
            "window": 5,
            "epsilon": 10.0,
        },
    },
    "plain_rl": {
        "algorithm": "PPO",
        "lookback_window": 20,
        "total_timesteps": 5000,
        "transaction_cost_rate": 0.001,
        "seed": 42,
    },
    "output": {
        "result_dir": "results",
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = deep_merge(DEFAULT_CONFIG, loaded)
    config["_config_path"] = str(config_path)
    return config


def result_dir(config: dict[str, Any]) -> Path:
    return resolve_project_path(config.get("output", {}).get("result_dir", "results"))
