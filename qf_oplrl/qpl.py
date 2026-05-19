from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


def compute_momentum(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute trailing close-to-close momentum."""
    return prices.pct_change(periods=int(window), fill_method=None)


def compute_rolling_qpl(
    prices: pd.DataFrame,
    lookback_window: int = 252,
    n_levels: int = 1,
    n_bins: int = 80,
    use_open_anchor: bool = True,
) -> dict[str, pd.DataFrame]:
    """Approximate rolling QPL support/resistance levels using only past data.

    The available prepared datasets have daily close prices. For date t, the
    level anchor is therefore the previous close P_{t-1}, and volatility is
    estimated from returns available before t.
    """
    del n_bins  # Kept in the public config for compatibility with the plan.

    prices = prices.sort_index().astype(float)
    lookback = int(lookback_window)
    level = max(1, int(n_levels))
    min_periods = max(20, min(lookback, 60))

    returns = prices.pct_change(fill_method=None)
    rolling_sigma = returns.rolling(lookback, min_periods=min_periods).std().shift(1)
    anchor = prices.shift(1) if use_open_anchor else prices.shift(1)
    nqpr = np.exp(level * rolling_sigma.clip(lower=0.0))

    qpl_plus = anchor * nqpr
    qpl_minus = anchor / nqpr.replace(0.0, np.nan)
    qpl_plus.columns = prices.columns
    qpl_minus.columns = prices.columns
    return {
        "qpl_plus_1": qpl_plus,
        "qpl_minus_1": qpl_minus,
    }


def compute_qpl_features(
    prices: pd.DataFrame,
    qpl_plus: pd.DataFrame,
    qpl_minus: pd.DataFrame,
    epsilon_touch: float = 0.01,
    momentum_window: int = 5,
) -> dict[str, pd.DataFrame]:
    prices, qpl_plus = prices.align(qpl_plus, join="inner", axis=0)
    prices, qpl_minus = prices.align(qpl_minus, join="inner", axis=0)
    qpl_plus = qpl_plus.reindex(columns=prices.columns)
    qpl_minus = qpl_minus.reindex(columns=prices.columns)

    safe_prices = prices.replace(0.0, np.nan)
    d_plus = (qpl_plus - prices) / safe_prices
    d_minus = (prices - qpl_minus) / safe_prices

    z_qpl = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)
    z_qpl = z_qpl.mask(prices > qpl_plus, 1)
    z_qpl = z_qpl.mask(prices < qpl_minus, -1)

    touch = float(epsilon_touch)
    near_support = (prices.sub(qpl_minus).abs().div(safe_prices) <= touch) | (prices < qpl_minus)
    near_resistance = (prices.sub(qpl_plus).abs().div(safe_prices) <= touch) | (prices > qpl_plus)
    breakdown = prices < qpl_minus
    momentum = compute_momentum(prices, window=int(momentum_window))

    qpl_signal = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)
    qpl_signal = qpl_signal.mask(near_support & (momentum > 0), 1)
    qpl_signal = qpl_signal.mask(near_resistance & (momentum < 0), -1)
    qpl_signal = qpl_signal.mask(breakdown & (momentum < 0), -2)
    qpl_signal = qpl_signal.fillna(0).astype(int)

    return {
        "qpl_d_plus": d_plus,
        "qpl_d_minus": d_minus,
        "qpl_z": z_qpl,
        "near_support": near_support.astype(int),
        "near_resistance": near_resistance.astype(int),
        "breakdown": breakdown.astype(int),
        "qpl_momentum": momentum,
        "qpl_signal": qpl_signal,
    }


def build_qpl_package(prices: pd.DataFrame, qpl_config: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    qpl_config = qpl_config or {}
    levels = compute_rolling_qpl(
        prices,
        lookback_window=int(qpl_config.get("lookback_window", 252)),
        n_levels=int(qpl_config.get("n_levels", 1)),
        n_bins=int(qpl_config.get("n_bins", 80)),
        use_open_anchor=bool(qpl_config.get("use_open_anchor", True)),
    )
    features = compute_qpl_features(
        prices,
        levels["qpl_plus_1"],
        levels["qpl_minus_1"],
        epsilon_touch=float(qpl_config.get("epsilon_touch", 0.01)),
        momentum_window=int(qpl_config.get("momentum_window", 5)),
    )
    return {**levels, **features}


def lag_qpl_package_for_returns(
    qpl_package: dict[str, pd.DataFrame],
    returns_index: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Lag QPL features by one date before aligning them with same-date returns."""
    return {
        name: frame.shift(1).reindex(returns_index)
        for name, frame in qpl_package.items()
    }

