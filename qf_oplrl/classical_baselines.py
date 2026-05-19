from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    n = values.size
    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, n + 1)
    condition = sorted_values - cssv / indices > 0
    if not condition.any():
        return np.full(n, 1.0 / n)
    theta = cssv[condition][-1] / indices[condition][-1]
    projected = np.maximum(values - theta, 0.0)
    total = projected.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(n, 1.0 / n)
    return projected / total


def equal_weight(returns: pd.DataFrame) -> pd.DataFrame:
    n_assets = returns.shape[1]
    return pd.DataFrame(
        1.0 / n_assets,
        index=returns.index,
        columns=returns.columns,
    )


def buy_and_hold(price_relatives: pd.DataFrame) -> pd.DataFrame:
    n_assets = price_relatives.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    weights = []

    for _, relatives in price_relatives.iterrows():
        weights.append(current_weight.copy())
        post_move = current_weight * relatives.to_numpy(dtype=float)
        total = post_move.sum()
        if total > 0 and np.isfinite(total):
            current_weight = post_move / total
        else:
            current_weight = np.full(n_assets, 1.0 / n_assets)

    return pd.DataFrame(weights, index=price_relatives.index, columns=price_relatives.columns)


def solve_long_only_portfolio(
    covariance: np.ndarray,
    expected_return: np.ndarray | None = None,
    risk_aversion: float = 10.0,
) -> np.ndarray | None:
    n_assets = covariance.shape[0]
    covariance = covariance + np.eye(n_assets) * 1e-8
    ones = np.ones(n_assets)
    try:
        inverse = np.linalg.pinv(covariance)
        if expected_return is None:
            raw_weights = inverse @ ones
            denominator = ones @ raw_weights
            if abs(denominator) <= EPS:
                return None
            raw_weights = raw_weights / denominator
        else:
            risk_aversion = max(float(risk_aversion), EPS)
            a = inverse @ expected_return / (2.0 * risk_aversion)
            b = inverse @ ones
            denominator = ones @ b
            if abs(denominator) <= EPS:
                return None
            lagrange = (ones @ a - 1.0) / denominator
            raw_weights = a - lagrange * b
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(raw_weights)):
        return None
    return project_to_simplex(raw_weights)


def minimum_variance(returns: pd.DataFrame, lookback_window: int = 60) -> pd.DataFrame:
    n_assets = returns.shape[1]
    fallback = np.full(n_assets, 1.0 / n_assets)
    weights = []

    for i in range(len(returns)):
        if i < lookback_window:
            weights.append(fallback.copy())
            continue
        window = returns.iloc[i - lookback_window : i]
        covariance = window.cov().to_numpy(dtype=float)
        solved = solve_long_only_portfolio(covariance)
        weights.append(fallback.copy() if solved is None else solved)

    return pd.DataFrame(weights, index=returns.index, columns=returns.columns)


def mean_variance(
    returns: pd.DataFrame,
    lookback_window: int = 60,
    risk_aversion: float = 10.0,
) -> pd.DataFrame:
    n_assets = returns.shape[1]
    fallback = np.full(n_assets, 1.0 / n_assets)
    weights = []

    for i in range(len(returns)):
        if i < lookback_window:
            weights.append(fallback.copy())
            continue
        window = returns.iloc[i - lookback_window : i]
        covariance = window.cov().to_numpy(dtype=float)
        expected_return = window.mean().to_numpy(dtype=float)
        solved = solve_long_only_portfolio(covariance, expected_return, risk_aversion)
        weights.append(fallback.copy() if solved is None else solved)

    return pd.DataFrame(weights, index=returns.index, columns=returns.columns)


def market_proxy(returns: pd.DataFrame) -> pd.DataFrame:
    return equal_weight(returns)


def generate_classical_weights(
    returns: pd.DataFrame,
    price_relatives: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    min_var_config = config.get("classical", {}).get("min_variance", {})
    mean_var_config = config.get("classical", {}).get("mean_variance", {})
    return {
        "Equal Weight": equal_weight(returns),
        "Buy and Hold": buy_and_hold(price_relatives),
        "Minimum Variance": minimum_variance(
            returns,
            lookback_window=int(min_var_config.get("lookback_window", 60)),
        ),
        "Mean-Variance": mean_variance(
            returns,
            lookback_window=int(mean_var_config.get("lookback_window", 60)),
            risk_aversion=float(mean_var_config.get("risk_aversion", 10.0)),
        ),
        "Market Proxy": market_proxy(returns),
    }
