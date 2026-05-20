from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


EPS = 1e-12


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1:
        raise ValueError("Simplex projection expects a one-dimensional vector")
    n = values.size
    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values) - 1
    indices = np.arange(1, n + 1)
    condition = sorted_values - cssv / indices > 0
    if not condition.any():
        return np.full(n, 1.0 / n)
    rho = indices[condition][-1]
    theta = cssv[condition][-1] / rho
    projected = np.maximum(values - theta, 0.0)
    total = projected.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(n, 1.0 / n)
    return projected / total


def bcrp(price_relatives: pd.DataFrame) -> pd.DataFrame:
    matrix = price_relatives.to_numpy(dtype=float)
    n_assets = matrix.shape[1]
    initial = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: np.ndarray) -> float:
        growth = matrix @ weights
        return -float(np.log(np.maximum(growth, EPS)).sum())

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10, "disp": False},
    )
    weights = initial if not result.success else project_to_simplex(result.x)
    return pd.DataFrame(
        np.tile(weights, (len(price_relatives), 1)),
        index=price_relatives.index,
        columns=price_relatives.columns,
    )


def pamr(
    price_relatives: pd.DataFrame,
    epsilon: float = 0.5,
    c_value: float = 500.0,
    variant: int = 0,
) -> pd.DataFrame:
    if variant not in {0, 1, 2}:
        raise ValueError("PAMR variant must be one of 0, 1, 2")
    n_assets = price_relatives.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    weights = []

    for _, row in price_relatives.iterrows():
        x = row.to_numpy(dtype=float)
        weights.append(current_weight.copy())
        x_mean = x.mean()
        centered = x - x_mean
        loss = max(0.0, float(current_weight @ x - epsilon))
        denominator = float(centered @ centered)
        if denominator <= EPS:
            tau = 0.0
        elif variant == 0:
            tau = loss / denominator
        elif variant == 1:
            tau = min(c_value, loss / denominator)
        else:
            tau = loss / (denominator + 0.5 / max(c_value, EPS))
        tau = min(100000.0, tau)
        current_weight = project_to_simplex(current_weight - tau * centered)

    return pd.DataFrame(weights, index=price_relatives.index, columns=price_relatives.columns)


def olmar(price_relatives: pd.DataFrame, window: int = 5, epsilon: float = 10.0) -> pd.DataFrame:
    prices = price_relatives.cumprod()
    n_assets = price_relatives.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    weights = []

    for i, (_, row) in enumerate(price_relatives.iterrows()):
        weights.append(current_weight.copy())
        if i + 1 < window:
            continue
        current_prices = prices.iloc[i].to_numpy(dtype=float)
        moving_average = prices.iloc[i + 1 - window : i + 1].mean().to_numpy(dtype=float)
        prediction = moving_average / np.maximum(current_prices, EPS)
        prediction_mean = prediction.mean()
        centered = prediction - prediction_mean
        denominator = float(centered @ centered)
        expected_return = float(current_weight @ prediction)
        step = 0.0 if denominator <= EPS else max(0.0, (epsilon - expected_return) / denominator)
        current_weight = project_to_simplex(current_weight + step * centered)

    return pd.DataFrame(weights, index=price_relatives.index, columns=price_relatives.columns)


def ons_diagonal(
    price_relatives: pd.DataFrame,
    beta: float = 1.0,
    delta: float = 0.125,
    eta: float = 0.01,
) -> pd.DataFrame:
    n_assets = price_relatives.shape[1]
    current_weight = np.full(n_assets, 1.0 / n_assets)
    gradient_scale = np.full(n_assets, delta)
    weights = []

    for _, row in price_relatives.iterrows():
        x = row.to_numpy(dtype=float)
        weights.append(current_weight.copy())
        growth = max(float(current_weight @ x), EPS)
        gradient = x / growth
        gradient_scale += gradient * gradient
        step = eta * beta * gradient / np.sqrt(gradient_scale)
        current_weight = project_to_simplex(current_weight + step)

    return pd.DataFrame(weights, index=price_relatives.index, columns=price_relatives.columns)


def generate_opl_weights(
    price_relatives: pd.DataFrame,
    config: dict,
) -> dict[str, pd.DataFrame]:
    opl_config = config.get("opl", {})
    pamr_config = opl_config.get("pamr", {})
    olmar_config = opl_config.get("olmar", {})
    ons_config = opl_config.get("ons", {})

    return {
        "PAMR": pamr(
            price_relatives,
            epsilon=float(pamr_config.get("epsilon", 0.5)),
            c_value=float(pamr_config.get("C", 500.0)),
            variant=int(pamr_config.get("variant", 0)),
        ),
        "OLMAR": olmar(
            price_relatives,
            window=int(olmar_config.get("window", 5)),
            epsilon=float(olmar_config.get("epsilon", 10.0)),
        ),
        "ONS Diagonal": ons_diagonal(
            price_relatives,
            beta=float(ons_config.get("beta", 1.0)),
            delta=float(ons_config.get("delta", 0.125)),
            eta=float(ons_config.get("eta", 0.01)),
        ),
    }
