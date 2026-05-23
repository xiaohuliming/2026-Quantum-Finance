"""Lee Oscillator.

Implementation of the classical Lee Oscillator from:
    Lee, R. S. T. (2004). "A transient-chaotic autoassociative network (TCAN)
    based on Lee oscillators." IEEE Transactions on Neural Networks, 15(5).

The oscillator is a 4-node discrete-time recurrent unit (E, I, Omega, L) with a
transient-chaotic envelope ``exp(-k * S^2)`` that produces sigmoid-like behavior
for ``|S| >> 0`` and a chaotic region around ``|S| ~ 0``. This is the activation
that Dr. Lee uses inside CRNN / TSCNON for financial prediction.

Equations (LeeOscillator 2004 / Wong-Lee 2017 CRNN form):

    E(t+1) = sigmoid(a1 * E(t) - a2 * I(t) + a3 * S - theta_E)
    I(t+1) = sigmoid(b1 * E(t) - b2 * I(t) - b3 * S - theta_I)
    Omega(t+1) = sigmoid(c1 * S)                       # input projection
    L(t+1)     = (E(t+1) - I(t+1)) * exp(-k * S**2) + Omega(t+1)

The I neuron uses self-inhibition (-b2 * I) and the opposite-sign input gain
(-b3 * S) so the E-I dynamics are symmetric at S = 0 and drive E > I for S > 0
(bullish) and E < I for S < 0 (bearish).

Reasonable defaults (a1=a2=b1=b2=5, k=50) give the characteristic Lee-bifurcation
where ``L(S)`` shows transient chaos for ``|S| <~ 0.2`` and looks like ``tanh``
outside that band.

The module exposes:

* :class:`LeeOscillator` - stateful oscillator (vectorised over assets).
* :func:`encode_series` - apply the oscillator iteratively to a 1d series,
  return the last-step ``L`` value (this is what Gate V3 uses to replace the
  linear momentum feature).
* :func:`bifurcation_curve` - sweep ``S`` over a range and return the resulting
  ``L`` values; used for the visualisation script and for unit-style sanity
  checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LeeOscillatorConfig:
    """Parameter bundle for :class:`LeeOscillator`.

    Defaults follow Lee 2004 / CRNN-2021 with k=50 giving a chaotic band of
    roughly ``|S| < 0.2``. ``e0`` / ``i0`` are the initial neuron states.
    """

    a1: float = 5.0
    a2: float = 5.0
    a3: float = 5.0
    b1: float = 5.0
    b2: float = 5.0
    b3: float = 5.0
    c1: float = 5.0
    theta_e: float = 0.0
    theta_i: float = 0.0
    k: float = 50.0
    e0: float = 0.2
    i0: float = 0.2
    transient_steps: int = 100


class LeeOscillator:
    """Vectorised Lee Oscillator.

    Accepts a scalar or an array of inputs ``S`` and iterates the (E, I, Omega, L)
    recurrence ``num_steps`` times. The shape of the internal state matches the
    shape of ``S`` so the same oscillator can be applied across many assets at
    once.
    """

    def __init__(self, config: LeeOscillatorConfig | None = None):
        self.config = config or LeeOscillatorConfig()

    def step(
        self,
        s: np.ndarray,
        e_prev: np.ndarray,
        i_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        e_next = _sigmoid(cfg.a1 * e_prev - cfg.a2 * i_prev + cfg.a3 * s - cfg.theta_e)
        i_next = _sigmoid(cfg.b1 * e_prev - cfg.b2 * i_prev - cfg.b3 * s - cfg.theta_i)
        omega = _sigmoid(cfg.c1 * s)
        envelope = np.exp(-cfg.k * np.square(s))
        l_next = (e_next - i_next) * envelope + omega
        return e_next, i_next, omega, l_next

    def run(self, s: np.ndarray, num_steps: int | None = None) -> np.ndarray:
        """Iterate the oscillator and return the final ``L`` value(s).

        ``s`` is treated as a constant external input across the unrolled
        ``num_steps`` iterations. Returns an array with the same shape as ``s``.
        """

        cfg = self.config
        steps = int(num_steps if num_steps is not None else cfg.transient_steps)
        s_arr = np.asarray(s, dtype=float)
        e = np.full_like(s_arr, cfg.e0, dtype=float)
        i = np.full_like(s_arr, cfg.i0, dtype=float)
        l_value = np.zeros_like(s_arr, dtype=float)
        for _ in range(steps):
            e, i, _omega, l_value = self.step(s_arr, e, i)
        return l_value

    def run_trajectory(self, s: float, num_steps: int | None = None) -> dict[str, np.ndarray]:
        """Single-scalar run that records the (E, I, Omega, L) trajectory.

        Useful for diagnostics and for plotting how the neurons evolve under a
        fixed input ``s``.
        """

        cfg = self.config
        steps = int(num_steps if num_steps is not None else cfg.transient_steps)
        s_scalar = float(s)
        e = np.array([cfg.e0], dtype=float)
        i = np.array([cfg.i0], dtype=float)
        history = {
            "E": np.empty(steps, dtype=float),
            "I": np.empty(steps, dtype=float),
            "Omega": np.empty(steps, dtype=float),
            "L": np.empty(steps, dtype=float),
        }
        for t in range(steps):
            e, i, omega, l_value = self.step(np.array([s_scalar]), e, i)
            history["E"][t] = float(e[0])
            history["I"][t] = float(i[0])
            history["Omega"][t] = float(omega[0])
            history["L"][t] = float(l_value[0])
        return history


def encode_series(
    series: Iterable[float],
    oscillator: LeeOscillator | None = None,
    *,
    scale: float = 1.0,
    clip: float = 1.0,
) -> float:
    """Encode a 1d numeric series into a single Lee-Oscillator score.

    Each value of ``series`` is fed as the external input ``S`` for one update
    step, sharing state ``(E, I)`` across the sequence. The returned scalar is
    the final ``L`` minus ``0.5`` so that the output is roughly centred on zero
    (the sigmoid baseline) and can be used as a signed "chaotic momentum".

    ``scale`` rescales the input before feeding it into the oscillator (Lee
    Oscillator parameters expect ``|S|`` in roughly the [-1, 1] range; raw daily
    log-returns are tiny, so we usually pre-multiply them).
    """

    if oscillator is None:
        oscillator = LeeOscillator()
    cfg = oscillator.config
    values = np.asarray(list(series), dtype=float)
    if values.size == 0:
        return 0.0
    values = np.clip(values * float(scale), -float(clip), float(clip))
    e = np.array([cfg.e0], dtype=float)
    i = np.array([cfg.i0], dtype=float)
    l_value = np.zeros(1, dtype=float)
    for s in values:
        e, i, _omega, l_value = oscillator.step(np.array([float(s)]), e, i)
    return float(l_value[0]) - 0.5


def encode_panel(
    panel: np.ndarray,
    *,
    oscillator: LeeOscillator | None = None,
    scale: float = 1.0,
    clip: float = 1.0,
) -> np.ndarray:
    """Apply :func:`encode_series` to each column of a 2d ``(T, N)`` panel.

    Returns a length-``N`` 1d array of Lee scores. NaNs are treated as zeros.
    """

    if oscillator is None:
        oscillator = LeeOscillator()
    arr = np.asarray(panel, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"encode_panel expects a 2d (T, N) array, got shape {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr * float(scale), -float(clip), float(clip))
    n_assets = arr.shape[1]
    cfg = oscillator.config
    e = np.full(n_assets, cfg.e0, dtype=float)
    i = np.full(n_assets, cfg.i0, dtype=float)
    l_value = np.zeros(n_assets, dtype=float)
    for s_row in arr:
        e, i, _omega, l_value = oscillator.step(s_row, e, i)
    return l_value - 0.5


def bifurcation_curve(
    s_grid: np.ndarray,
    oscillator: LeeOscillator | None = None,
    num_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Run the oscillator at each ``S`` in ``s_grid`` and collect the result.

    Returns a dict with the steady-state ``E``, ``I``, ``Omega``, and ``L``
    values at each ``S``. Used by the visualisation script.
    """

    if oscillator is None:
        oscillator = LeeOscillator()
    grid = np.asarray(s_grid, dtype=float)
    cfg = oscillator.config
    steps = int(num_steps if num_steps is not None else cfg.transient_steps)
    e = np.full_like(grid, cfg.e0, dtype=float)
    i = np.full_like(grid, cfg.i0, dtype=float)
    omega = np.zeros_like(grid, dtype=float)
    l_value = np.zeros_like(grid, dtype=float)
    for _ in range(steps):
        e, i, omega, l_value = oscillator.step(grid, e, i)
    return {"E": e, "I": i, "Omega": omega, "L": l_value, "S": grid}
