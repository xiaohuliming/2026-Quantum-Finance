from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12


def _as_array(value, n_assets: int, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(n_assets, default, dtype=float)
    if isinstance(value, pd.Series):
        array = value.to_numpy(dtype=float)
    else:
        array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(n_assets, float(array), dtype=float)
    if array.size != n_assets:
        resized = np.full(n_assets, default, dtype=float)
        resized[: min(n_assets, array.size)] = array[: min(n_assets, array.size)]
        array = resized
    return np.nan_to_num(array, nan=default, posinf=default, neginf=default)


def _row_value(row, names: list[str], n_assets: int, default: float = 0.0) -> np.ndarray:
    if row is None:
        return np.full(n_assets, default, dtype=float)
    if isinstance(row, dict):
        for name in names:
            if name in row:
                return _as_array(row[name], n_assets, default)
        return np.full(n_assets, default, dtype=float)
    if isinstance(row, pd.Series):
        for name in names:
            if name in row.index:
                return _as_array(row[name], n_assets, default)
    return _as_array(row, n_assets, default)


def _score_near(distance: np.ndarray, band: float) -> np.ndarray:
    band = max(float(band), EPS)
    return np.clip(1.0 - np.abs(distance) / band, 0.0, 1.0)


def _momentum_scores(momentum: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    scale = max(float(scale), EPS)
    positive = np.clip(momentum / scale, 0.0, 1.0)
    weak = np.clip(-momentum / scale, 0.0, 1.0)
    return positive, weak


def _volatility_score(tech_feature_row, n_assets: int, config: dict[str, Any]) -> np.ndarray:
    if tech_feature_row is None:
        return np.zeros(n_assets, dtype=float)
    window = int(config.get("volatility_window", 20))
    volatility = _row_value(
        tech_feature_row,
        [f"volatility_{window}", "volatility", "volatility_20"],
        n_assets,
        default=0.0,
    )
    median = float(np.nanmedian(volatility[volatility > 0])) if np.any(volatility > 0) else 0.0
    if median <= 0 or not np.isfinite(median):
        return np.zeros(n_assets, dtype=float)
    ratio = volatility / max(median, EPS)
    cap = max(float(config.get("volatility_ratio_cap", 3.0)), 1.01)
    return np.clip((ratio - 1.0) / (cap - 1.0), 0.0, 1.0)


def compute_qpl_gate_scores(
    raw_weights,
    previous_weights,
    qpl_feature_row,
    tech_feature_row=None,
    portfolio_drawdown: float = 0.0,
    qpl_config: dict[str, Any] | None = None,
) -> np.ndarray:
    config = qpl_config or {}
    raw = _as_array(raw_weights, len(raw_weights), default=0.0).clip(min=0.0)
    n_assets = raw.size
    previous = _as_array(previous_weights, n_assets, default=1.0 / n_assets).clip(min=0.0)

    add_intent = np.maximum(raw - previous, 0.0)
    reduce_intent = np.maximum(previous - raw, 0.0)
    add_intent_score = np.clip(add_intent * n_assets, 0.0, 1.0)
    reduce_intent_score = np.clip(reduce_intent * n_assets, 0.0, 1.0)

    d_minus = _row_value(qpl_feature_row, ["qpl_d_minus", "d_minus"], n_assets, default=1.0)
    d_plus = _row_value(qpl_feature_row, ["qpl_d_plus", "d_plus"], n_assets, default=1.0)
    signal = _row_value(qpl_feature_row, ["qpl_signal", "signal"], n_assets, default=0.0)
    momentum = _row_value(qpl_feature_row, ["qpl_momentum", "momentum"], n_assets, default=0.0)

    support_score = _score_near(d_minus, float(config.get("support_band", 0.01)))
    resistance_score = _score_near(d_plus, float(config.get("resistance_band", 0.01)))
    breakdown_score = np.where((signal <= -2) | (d_minus < 0.0), 1.0, 0.0)
    positive_momentum_score, weak_momentum_score = _momentum_scores(
        momentum,
        float(config.get("momentum_scale", 0.05)),
    )
    high_volatility_score = _volatility_score(tech_feature_row, n_assets, config)
    drawdown_scale = max(float(config.get("drawdown_scale", 0.20)), EPS)
    portfolio_drawdown_score = np.clip(float(portfolio_drawdown) / drawdown_scale, 0.0, 1.0)

    alpha_support = float(config.get("alpha_support", 0.15))
    beta_resistance = float(config.get("beta_resistance", 0.35))
    gamma_breakdown = float(config.get("gamma_breakdown", 0.50))
    delta_volatility = float(config.get("delta_volatility", 0.10))
    eta_drawdown = float(config.get("eta_drawdown", 0.20))

    attraction = support_score * positive_momentum_score * add_intent_score * alpha_support
    resistance_penalty = resistance_score * weak_momentum_score * add_intent_score * beta_resistance
    breakdown_penalty = breakdown_score * (0.5 + 0.5 * weak_momentum_score) * gamma_breakdown
    volatility_penalty = high_volatility_score * (0.5 + 0.5 * add_intent_score) * delta_volatility
    risk_score = np.maximum.reduce([resistance_score, breakdown_score, high_volatility_score])
    drawdown_penalty = risk_score * portfolio_drawdown_score * eta_drawdown

    release = 1.0 - 0.5 * reduce_intent_score
    total_penalty = (resistance_penalty + breakdown_penalty + volatility_penalty + drawdown_penalty) * release
    multiplier = 1.0 + attraction - total_penalty
    return np.clip(
        multiplier,
        float(config.get("g_min", 0.30)),
        float(config.get("g_max", 1.20)),
    )


def apply_qpl_gate_v2_to_weight_vector(
    raw_weights,
    previous_weights,
    qpl_feature_row,
    tech_feature_row=None,
    portfolio_drawdown: float = 0.0,
    qpl_config: dict[str, Any] | None = None,
) -> np.ndarray:
    raw = _as_array(raw_weights, len(raw_weights), default=0.0).clip(min=0.0)
    n_assets = raw.size
    total = raw.sum()
    if total <= 0 or not np.isfinite(total):
        raw = np.full(n_assets, 1.0 / n_assets, dtype=float)
    else:
        raw = raw / total
    multipliers = compute_qpl_gate_scores(
        raw,
        previous_weights,
        qpl_feature_row,
        tech_feature_row=tech_feature_row,
        portfolio_drawdown=portfolio_drawdown,
        qpl_config=qpl_config,
    )
    gated = raw * multipliers
    gated_total = gated.sum()
    if gated_total <= 0 or not np.isfinite(gated_total):
        return np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
    return (gated / gated_total).astype(np.float32)


def apply_qpl_gate_v2_to_weights(
    raw_weights_df: pd.DataFrame,
    previous_weights_df: pd.DataFrame,
    qpl_feature_rows: dict[str, pd.DataFrame],
    tech_feature_rows: dict[str, pd.DataFrame] | None = None,
    qpl_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    common_index = raw_weights_df.index.intersection(previous_weights_df.index)
    common_columns = raw_weights_df.columns.intersection(previous_weights_df.columns)
    weights = []
    for index in common_index:
        qpl_row = {
            name: frame.loc[index, common_columns]
            for name, frame in qpl_feature_rows.items()
            if index in frame.index
        }
        tech_row = None
        if tech_feature_rows is not None:
            tech_row = {
                name: frame.loc[index, common_columns]
                for name, frame in tech_feature_rows.items()
                if index in frame.index
            }
        weights.append(
            apply_qpl_gate_v2_to_weight_vector(
                raw_weights_df.loc[index, common_columns].to_numpy(dtype=float),
                previous_weights_df.loc[index, common_columns].to_numpy(dtype=float),
                qpl_row,
                tech_feature_row=tech_row,
                qpl_config=qpl_config,
            )
        )
    return pd.DataFrame(weights, index=common_index, columns=common_columns)

