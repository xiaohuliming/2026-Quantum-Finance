"""QPL Gate V3: Gate V2 with Lee-Oscillator-encoded chaotic momentum.

Gate V2 uses a simple ``compute_momentum`` (window-N percentage change) as the
``momentum`` input to ``compute_qpl_gate_scores``. The empirical result is that
Gate V2 already lifts Sharpe / lowers MDD substantially, but its momentum
estimate is linear and reacts the same way in low-volatility "noisy" regimes
and in clean trending regimes.

Gate V3 replaces only that single signal: we pass each asset's trailing
``lookback`` daily returns through a Lee Oscillator (Lee 2004, our
:mod:`qf_oplrl.lee_oscillator`) and use ``L - 0.5`` as the momentum value.

The downstream Gate V2 logic (action-aware penalties, support/resistance
scoring, volatility/drawdown penalties, multiplier clipping) is unchanged. This
keeps the ablation clean: any difference between Gate V2 and Gate V3 is
attributable to the chaotic momentum encoding.

Two public entry points:

* :func:`compute_lee_momentum_panel` - pre-compute a ``(T x N)`` DataFrame of
  Lee-Oscillator momentum that can be plugged into the existing pipeline next
  to ``qpl_momentum``.
* :func:`apply_qpl_gate_v3` / :func:`apply_qpl_gate_v3_to_weight_vector` /
  :func:`apply_qpl_gate_v3_to_weights` - drop-in replacements for the Gate V2
  apply functions; they swap ``qpl_momentum`` for ``lee_momentum`` then defer
  to Gate V2.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qf_oplrl.lee_oscillator import LeeOscillator, LeeOscillatorConfig
from qf_oplrl.qpl_gate_v2 import (
    apply_qpl_gate_v2,
    apply_qpl_gate_v2_to_weight_vector,
    apply_qpl_gate_v2_to_weights,
    compute_qpl_gate_scores,
)


def _build_oscillator(config: dict[str, Any] | None) -> LeeOscillator:
    cfg = config or {}
    return LeeOscillator(
        LeeOscillatorConfig(
            a1=float(cfg.get("lee_a1", 5.0)),
            a2=float(cfg.get("lee_a2", 5.0)),
            a3=float(cfg.get("lee_a3", 5.0)),
            b1=float(cfg.get("lee_b1", 5.0)),
            b2=float(cfg.get("lee_b2", 5.0)),
            b3=float(cfg.get("lee_b3", 5.0)),
            c1=float(cfg.get("lee_c1", 5.0)),
            theta_e=float(cfg.get("lee_theta_e", 0.0)),
            theta_i=float(cfg.get("lee_theta_i", 0.0)),
            k=float(cfg.get("lee_k", 50.0)),
            e0=float(cfg.get("lee_e0", 0.2)),
            i0=float(cfg.get("lee_i0", 0.2)),
            transient_steps=int(cfg.get("lee_transient_steps", 100)),
        )
    )


def compute_lee_momentum_panel(
    returns: pd.DataFrame,
    lookback: int = 20,
    *,
    input_scale: float = 50.0,
    input_clip: float = 1.0,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Encode a trailing window of daily returns into a Lee-Oscillator score.

    For every ``(date t, asset i)`` we feed the ``lookback`` most recent
    log-style returns into the oscillator (rescaled by ``input_scale`` so daily
    returns of ~1% land near the centre of the chaotic band) and store the
    final ``L - 0.5`` value.

    The signed output is the "chaotic momentum": positive in clean uptrends,
    negative in clean downtrends, near zero (but noisy) in low-signal regimes.
    """

    oscillator = _build_oscillator(config)
    cfg = oscillator.config
    arr = returns.to_numpy(dtype=float, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr * float(input_scale), -float(input_clip), float(input_clip))

    n_rows, n_cols = arr.shape
    out = np.zeros_like(arr, dtype=float)
    lookback = max(1, int(lookback))

    for t in range(n_rows):
        start = max(0, t - lookback + 1)
        window = arr[start : t + 1]
        if window.shape[0] == 0:
            continue
        e = np.full(n_cols, cfg.e0, dtype=float)
        i = np.full(n_cols, cfg.i0, dtype=float)
        l_value = np.zeros(n_cols, dtype=float)
        for s_row in window:
            e, i, _omega, l_value = oscillator.step(s_row, e, i)
        out[t] = l_value - 0.5

    return pd.DataFrame(out, index=returns.index, columns=returns.columns)


def _swap_momentum_in_row(
    qpl_feature_row,
    lee_momentum_row,
) -> Any:
    """Return a copy of ``qpl_feature_row`` with ``qpl_momentum`` replaced.

    Accepts the same shapes that Gate V2 accepts (``dict`` of arrays, ``pd.Series``,
    or raw array). When ``lee_momentum_row`` is ``None`` we return the input
    unchanged - the caller should also fall back to Gate V2 behavior.
    """

    if lee_momentum_row is None:
        return qpl_feature_row
    if isinstance(qpl_feature_row, dict):
        patched = dict(qpl_feature_row)
        patched["qpl_momentum"] = lee_momentum_row
        patched.setdefault("lee_momentum", lee_momentum_row)
        return patched
    if isinstance(qpl_feature_row, pd.Series):
        patched = qpl_feature_row.copy()
        if "qpl_momentum" in patched.index:
            patched.loc["qpl_momentum"] = lee_momentum_row
        return patched
    return qpl_feature_row


def apply_qpl_gate_v3(
    raw_weights,
    previous_weights,
    qpl_features: dict[str, np.ndarray] | dict[str, pd.Series],
    technical_features: dict[str, np.ndarray] | dict[str, pd.Series] | None = None,
    portfolio_drawdown: float = 0.0,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Gate V2 with the momentum input swapped for Lee Oscillator encoding."""

    features = dict(qpl_features) if isinstance(qpl_features, dict) else qpl_features
    if isinstance(features, dict) and "lee_momentum" in features:
        features = _swap_momentum_in_row(features, features["lee_momentum"])
    weights, diagnostics = apply_qpl_gate_v2(
        raw_weights,
        previous_weights,
        features,
        technical_features=technical_features,
        portfolio_drawdown=portfolio_drawdown,
        config=config,
    )
    diagnostics["gate_version"] = "v3"
    return weights, diagnostics


def apply_qpl_gate_v3_to_weight_vector(
    raw_weights,
    previous_weights,
    qpl_feature_row,
    tech_feature_row=None,
    portfolio_drawdown: float = 0.0,
    qpl_config: dict[str, Any] | None = None,
) -> np.ndarray:
    if isinstance(qpl_feature_row, dict) and "lee_momentum" in qpl_feature_row:
        qpl_feature_row = _swap_momentum_in_row(qpl_feature_row, qpl_feature_row["lee_momentum"])
    elif isinstance(qpl_feature_row, pd.Series) and "lee_momentum" in qpl_feature_row.index:
        qpl_feature_row = _swap_momentum_in_row(qpl_feature_row, qpl_feature_row.loc["lee_momentum"])
    return apply_qpl_gate_v2_to_weight_vector(
        raw_weights,
        previous_weights,
        qpl_feature_row,
        tech_feature_row=tech_feature_row,
        portfolio_drawdown=portfolio_drawdown,
        qpl_config=qpl_config,
    )


def apply_qpl_gate_v3_to_weights(
    raw_weights_df: pd.DataFrame,
    previous_weights_df: pd.DataFrame,
    qpl_feature_rows: dict[str, pd.DataFrame],
    tech_feature_rows: dict[str, pd.DataFrame] | None = None,
    qpl_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Time-indexed Gate V3 - mirrors :func:`apply_qpl_gate_v2_to_weights`.

    If the caller has provided ``lee_momentum`` in ``qpl_feature_rows`` it is
    swapped into the ``qpl_momentum`` slot before delegating to Gate V2.
    """

    feature_rows = dict(qpl_feature_rows)
    if "lee_momentum" in feature_rows:
        feature_rows["qpl_momentum"] = feature_rows["lee_momentum"]
    return apply_qpl_gate_v2_to_weights(
        raw_weights_df,
        previous_weights_df,
        feature_rows,
        tech_feature_rows=tech_feature_rows,
        qpl_config=qpl_config,
    )


def compute_qpl_gate_v3_scores(
    raw_weights,
    previous_weights,
    qpl_feature_row,
    tech_feature_row=None,
    portfolio_drawdown: float = 0.0,
    qpl_config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Score-only version - for diagnostics and ablation plotting.

    Same as :func:`qf_oplrl.qpl_gate_v2.compute_qpl_gate_scores` but performs
    the Lee-momentum hot-swap first.
    """

    if isinstance(qpl_feature_row, dict) and "lee_momentum" in qpl_feature_row:
        qpl_feature_row = _swap_momentum_in_row(qpl_feature_row, qpl_feature_row["lee_momentum"])
    elif isinstance(qpl_feature_row, pd.Series) and "lee_momentum" in qpl_feature_row.index:
        qpl_feature_row = _swap_momentum_in_row(qpl_feature_row, qpl_feature_row.loc["lee_momentum"])
    return compute_qpl_gate_scores(
        raw_weights,
        previous_weights,
        qpl_feature_row,
        tech_feature_row=tech_feature_row,
        portfolio_drawdown=portfolio_drawdown,
        qpl_config=qpl_config,
    )
