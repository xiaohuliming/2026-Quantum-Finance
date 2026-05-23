"""Plot the Lee Oscillator bifurcation curve and a sample trajectory.

Outputs three PNGs to ``results/figures/``:

* ``lee_oscillator_bifurcation.png`` - L(S), E(S), I(S), Omega(S) vs S
* ``lee_oscillator_trajectory_chaotic.png`` - (E,I,L) trajectory at S=0
* ``lee_oscillator_trajectory_stable.png`` - same at S=0.5 (outside chaotic band)

These are sanity checks: if your implementation matches Lee 2004, the
bifurcation plot should show a noisy chaotic band for |S| < ~0.2 and a clean
sigmoid-like curve outside.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qf_oplrl.lee_oscillator import LeeOscillator, LeeOscillatorConfig, bifurcation_curve


def plot_bifurcation(out_path: Path) -> None:
    oscillator = LeeOscillator(LeeOscillatorConfig(transient_steps=200))
    s_grid = np.linspace(-1.0, 1.0, 2001)
    curves = bifurcation_curve(s_grid, oscillator)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_l, ax_eio = axes

    ax_l.plot(curves["S"], curves["L"], lw=0.7, color="tab:blue")
    ax_l.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax_l.axvspan(-0.2, 0.2, alpha=0.08, color="red", label="approx. chaotic band")
    ax_l.set_ylabel("L(S)")
    ax_l.set_title("Lee Oscillator bifurcation (steady-state L vs input S)")
    ax_l.legend(loc="best")
    ax_l.grid(alpha=0.3)

    ax_eio.plot(curves["S"], curves["E"], lw=0.8, label="E(S)", color="tab:orange")
    ax_eio.plot(curves["S"], curves["I"], lw=0.8, label="I(S)", color="tab:green")
    ax_eio.plot(curves["S"], curves["Omega"], lw=0.8, label="Ω(S)", color="tab:purple")
    ax_eio.set_xlabel("S")
    ax_eio.set_ylabel("neuron output")
    ax_eio.legend(loc="best")
    ax_eio.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_trajectory(s_value: float, out_path: Path) -> None:
    oscillator = LeeOscillator(LeeOscillatorConfig(transient_steps=300))
    history = oscillator.run_trajectory(s_value, num_steps=300)
    steps = np.arange(history["L"].size)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, history["E"], lw=0.8, label="E(t)")
    ax.plot(steps, history["I"], lw=0.8, label="I(t)")
    ax.plot(steps, history["L"], lw=0.9, label="L(t)", color="tab:blue")
    ax.set_xlabel("iteration t")
    ax.set_ylabel("neuron output")
    ax.set_title(f"Lee Oscillator trajectory at S = {s_value}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    bifurcation_path = out_dir / "lee_oscillator_bifurcation.png"
    chaotic_path = out_dir / "lee_oscillator_trajectory_chaotic.png"
    stable_path = out_dir / "lee_oscillator_trajectory_stable.png"

    plot_bifurcation(bifurcation_path)
    plot_trajectory(0.0, chaotic_path)
    plot_trajectory(0.5, stable_path)

    for path in (bifurcation_path, chaotic_path, stable_path):
        rel = os.path.relpath(path, ROOT)
        print(f"wrote {rel}")


if __name__ == "__main__":
    main()
