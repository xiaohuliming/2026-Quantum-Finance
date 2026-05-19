from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qf_oplrl.backtest import normalize_weights


def _cap_vector(vector: np.ndarray, max_single_weight: float | None) -> np.ndarray:
    weights = np.asarray(vector, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    n_assets = weights.size
    total = weights.sum()
    if total <= 0:
        weights = np.full(n_assets, 1.0 / n_assets)
    else:
        weights = weights / total

    if max_single_weight is None:
        return weights

    cap = float(max_single_weight)
    if cap <= 0:
        return np.full(n_assets, 1.0 / n_assets)
    cap = max(cap, 1.0 / n_assets)

    capped = weights.copy()
    free = np.ones(n_assets, dtype=bool)
    for _ in range(n_assets):
        over = free & (capped > cap)
        if not over.any():
            break
        capped[over] = cap
        free[over] = False
        remaining = 1.0 - capped[~free].sum()
        if remaining <= 0 or not free.any():
            break
        free_weights = weights[free]
        free_total = free_weights.sum()
        if free_total <= 0:
            capped[free] = remaining / free.sum()
        else:
            capped[free] = remaining * free_weights / free_total

    total = capped.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(n_assets, 1.0 / n_assets)
    return capped / total


def apply_max_weight_cap(weights: pd.DataFrame, max_single_weight: float | None) -> pd.DataFrame:
    if max_single_weight is None:
        return normalize_weights(weights)
    capped = np.vstack([_cap_vector(row, max_single_weight) for row in weights.to_numpy(dtype=float)])
    return pd.DataFrame(capped, index=weights.index, columns=weights.columns)


def signal_multipliers(
    qpl_signal: pd.DataFrame,
    qpl_config: dict[str, Any] | None = None,
    strategy_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    qpl_config = qpl_config or {}
    strategy_config = strategy_config or {}
    multipliers = pd.DataFrame(
        float(strategy_config.get("neutral_multiplier", qpl_config.get("neutral_multiplier", 1.0))),
        index=qpl_signal.index,
        columns=qpl_signal.columns,
    )
    multipliers = multipliers.mask(
        qpl_signal == 1,
        float(strategy_config.get("support_boost", qpl_config.get("support_boost", 1.10))),
    )
    multipliers = multipliers.mask(
        qpl_signal == -1,
        float(strategy_config.get("resistance_cut", qpl_config.get("resistance_cut", 0.70))),
    )
    multipliers = multipliers.mask(
        qpl_signal == -2,
        float(strategy_config.get("breakdown_cut", qpl_config.get("breakdown_cut", 0.50))),
    )
    min_multiplier = qpl_config.get("min_multiplier")
    max_multiplier = qpl_config.get("max_multiplier")
    if min_multiplier is not None or max_multiplier is not None:
        multipliers = multipliers.clip(
            lower=None if min_multiplier is None else float(min_multiplier),
            upper=None if max_multiplier is None else float(max_multiplier),
        )
    return multipliers


def qpl_rule_weights(
    returns: pd.DataFrame,
    qpl_signal: pd.DataFrame,
    qpl_config: dict[str, Any] | None = None,
    strategy_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build no-lookahead QPL rule weights aligned to return dates."""
    qpl_config = qpl_config or {}
    strategy_config = strategy_config or {}
    if str(strategy_config.get("base_weight", "equal")).lower() != "equal":
        raise ValueError("Only equal base_weight is currently supported")

    lagged_signal = qpl_signal.shift(1).reindex(index=returns.index, columns=returns.columns)
    lagged_signal = lagged_signal.fillna(0).astype(int)
    multipliers = signal_multipliers(lagged_signal, qpl_config, strategy_config)

    base_weight = 1.0 / returns.shape[1]
    weights = pd.DataFrame(base_weight, index=returns.index, columns=returns.columns) * multipliers
    weights = normalize_weights(weights)
    return apply_max_weight_cap(weights, strategy_config.get("max_single_weight"))

