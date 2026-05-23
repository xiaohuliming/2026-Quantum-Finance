"""Standalone Lee Oscillator predictor used as a baseline.

This is the "Plan C" comparison strategy referenced in the project plan: it uses
*only* the Lee Oscillator (no QPL, no RL) to produce a portfolio. The point of
having it in the leaderboard is to isolate the contribution of the chaotic
neural component from the contribution of QPL.

Strategy:

1. For every ``(date t, asset i)`` encode the trailing ``lookback`` daily
   returns through a Lee Oscillator and store the signed score ``L - 0.5``.
2. Shift the score by one trading day so the trade on day ``t`` uses only
   information available at the close of day ``t-1`` (no lookahead).
3. Convert per-asset scores into long-only weights using one of two modes:

   * ``soft`` (default) - softmax over the scores (with a temperature) so
     bullish-encoded assets get larger but still smooth weights.
   * ``long_only`` - keep only assets with positive score and equal-weight
     them; fall back to equal-weight across all assets on days with no
     positive scores.
4. Normalise and (optionally) cap single-name weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qf_oplrl.backtest import normalize_weights
from qf_oplrl.qpl_gate_v3 import _build_oscillator, compute_lee_momentum_panel
from qf_oplrl.qpl_strategy import apply_max_weight_cap


def _softmax_row(scores: np.ndarray, temperature: float) -> np.ndarray:
    finite = np.where(np.isfinite(scores), scores, 0.0)
    temperature = max(float(temperature), 1e-6)
    shifted = finite / temperature
    shifted = shifted - np.max(shifted)
    exp = np.exp(shifted)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(scores.size, 1.0 / scores.size)
    return exp / total


def _long_only_row(scores: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(scores), scores, 0.0)
    mask = finite > 0
    n = scores.size
    if not mask.any():
        return np.full(n, 1.0 / n)
    weights = mask.astype(float) / float(mask.sum())
    return weights


def lee_predictor_weights(
    returns: pd.DataFrame,
    *,
    lookback: int = 20,
    input_scale: float = 50.0,
    input_clip: float = 1.0,
    mode: str = "soft",
    temperature: float = 0.2,
    max_single_weight: float | None = 0.25,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build Lee-Oscillator-only portfolio weights aligned to return dates.

    Parameters mirror :func:`qf_oplrl.qpl_strategy.qpl_rule_weights` so this
    function can be dropped into the baseline runner with minimal changes.
    """

    if returns.empty:
        return returns.copy()

    score_panel = compute_lee_momentum_panel(
        returns,
        lookback=lookback,
        input_scale=input_scale,
        input_clip=input_clip,
        config=config,
    )

    lagged = score_panel.shift(1).reindex(index=returns.index, columns=returns.columns)
    lagged = lagged.fillna(0.0)

    mode_normalised = (mode or "soft").lower()
    rows = []
    for values in lagged.to_numpy(dtype=float):
        if mode_normalised == "long_only":
            rows.append(_long_only_row(values))
        else:
            rows.append(_softmax_row(values, temperature))
    weights = pd.DataFrame(rows, index=returns.index, columns=returns.columns)
    weights = normalize_weights(weights)
    return apply_max_weight_cap(weights, max_single_weight)


def lee_predictor_scores(
    returns: pd.DataFrame,
    *,
    lookback: int = 20,
    input_scale: float = 50.0,
    input_clip: float = 1.0,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Expose the raw Lee scores - useful for diagnostic plots."""

    return compute_lee_momentum_panel(
        returns,
        lookback=lookback,
        input_scale=input_scale,
        input_clip=input_clip,
        config=config,
    )


def attach_lee_momentum_to_qpl_package(
    package: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    *,
    lookback: int = 20,
    input_scale: float = 50.0,
    input_clip: float = 1.0,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Convenience: add ``lee_momentum`` to an existing QPL feature package.

    Returns a new dict (does not mutate the caller's package). Useful inside
    pipelines that already build the QPL feature bundle and need Gate V3's
    extra input.
    """

    enriched = dict(package)
    enriched["lee_momentum"] = compute_lee_momentum_panel(
        returns,
        lookback=lookback,
        input_scale=input_scale,
        input_clip=input_clip,
        config=config,
    )
    return enriched


# Smoke-test entry point - "python -m qf_oplrl.lee_predictor" prints a small
# table that confirms the module imports correctly without needing pytest.
if __name__ == "__main__":  # pragma: no cover - manual sanity check
    rng = np.random.default_rng(42)
    fake_returns = pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.01, size=(60, 4)),
        columns=["A", "B", "C", "D"],
    )
    fake_returns.iloc[30:, 0] += 0.003  # inject regime change in column A
    weights = lee_predictor_weights(fake_returns, lookback=10, mode="soft")
    print("last 5 weight rows:")
    print(weights.tail().round(4).to_string())
