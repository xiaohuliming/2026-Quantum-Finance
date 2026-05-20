from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12
DEFAULT_FEATURE_NAMES = [
    "ma_ratio_5",
    "ma_ratio_20",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "volatility_20",
]


def moving_average_ratio(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    moving_average = prices.rolling(int(window), min_periods=int(window)).mean()
    return prices.div(moving_average.replace(0.0, np.nan)) - 1.0


def rolling_volatility(
    returns: pd.DataFrame,
    window: int = 20,
    annualize: bool = False,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    volatility = returns.rolling(int(window), min_periods=int(window)).std()
    if annualize:
        volatility = volatility * np.sqrt(int(annualization_factor))
    return volatility


def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    changes = prices.diff()
    gains = changes.clip(lower=0.0)
    losses = -changes.clip(upper=0.0)
    average_gain = gains.ewm(alpha=1.0 / int(window), adjust=False, min_periods=int(window)).mean()
    average_loss = losses.ewm(alpha=1.0 / int(window), adjust=False, min_periods=int(window)).mean()
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    raw_rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    raw_rsi = raw_rsi.fillna(50.0)
    return raw_rsi / 50.0 - 1.0


def macd(
    prices: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.DataFrame]:
    fast_ema = prices.ewm(span=int(fast), adjust=False, min_periods=int(fast)).mean()
    slow_ema = prices.ewm(span=int(slow), adjust=False, min_periods=int(slow)).mean()
    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=int(signal), adjust=False, min_periods=int(signal)).mean()
    macd_hist = macd_line - macd_signal
    scale = prices.replace(0.0, np.nan)
    return {
        "macd_line": macd_line.div(scale),
        "macd_signal": macd_signal.div(scale),
        "macd_hist": macd_hist.div(scale),
    }


def build_technical_feature_package(
    prices: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or {}
    feature_config = config.get("features", {})
    clip_value = config.get("clip_value", 10.0)
    prices = prices.sort_index().astype(float)
    returns = prices.pct_change(fill_method=None)

    package: dict[str, pd.DataFrame] = {}
    ma_windows = feature_config.get("ma_ratio", {}).get("windows", [5, 20])
    for window in ma_windows:
        name = f"ma_ratio_{int(window)}"
        package[name] = moving_average_ratio(prices, int(window))

    rsi_window = int(feature_config.get("rsi", {}).get("window", 14))
    package[f"rsi_{rsi_window}"] = rsi(prices, rsi_window)

    macd_config = feature_config.get("macd", {})
    package.update(
        macd(
            prices,
            fast=int(macd_config.get("fast", 12)),
            slow=int(macd_config.get("slow", 26)),
            signal=int(macd_config.get("signal", 9)),
        )
    )

    volatility_config = feature_config.get("volatility", {})
    volatility_window = int(volatility_config.get("window", 20))
    package[f"volatility_{volatility_window}"] = rolling_volatility(
        returns,
        window=volatility_window,
        annualize=bool(volatility_config.get("annualize", False)),
    )

    if clip_value is not None:
        bound = float(clip_value)
        package = {name: frame.clip(lower=-bound, upper=bound) for name, frame in package.items()}
    return package


def select_technical_features(
    technical_features: dict[str, pd.DataFrame],
    feature_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if feature_names is None:
        names = [name for name in DEFAULT_FEATURE_NAMES if name in technical_features]
        names.extend(name for name in technical_features if name not in names)
    else:
        names = feature_names
    missing = [name for name in names if name not in technical_features]
    if missing:
        raise KeyError(f"Missing technical features: {missing}")
    return {name: technical_features[name] for name in names}


def lag_technical_package_for_returns(
    technical_features: dict[str, pd.DataFrame],
    returns_index: pd.Index,
    shift_features: bool = True,
) -> dict[str, pd.DataFrame]:
    shift = 1 if shift_features else 0
    return {
        name: frame.shift(shift).reindex(returns_index)
        for name, frame in technical_features.items()
    }


def build_lagged_technical_features(
    prices: pd.DataFrame,
    returns_index: pd.Index,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or {}
    package = build_technical_feature_package(prices, config)
    return lag_technical_package_for_returns(
        package,
        returns_index=returns_index,
        shift_features=bool(config.get("shift_features", True)),
    )

