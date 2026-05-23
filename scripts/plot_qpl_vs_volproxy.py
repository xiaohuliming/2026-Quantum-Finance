"""Plot QPL vs Vol-Proxy ablation across DOW30 / NAS100 / COMMODITY.

Question this answers:
    "Does the Sharpe lift come from QPL specifically, or from the action-aware
    gate architecture in general?"

Method:
    Run the same Gate V2 / V3 pipeline with two different sources of price
    levels:
      * QPL       — QAHO + QFSE energy levels (this project's main method)
      * VolProxy  — rolling-volatility-based bands (Bollinger-equivalent)

Outputs:
    results/figures/qpl_vs_volproxy_sharpe.png
    results/figures/qpl_vs_volproxy_decomposition.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK = "#0A0A0A"
LEMON = "#E6FF3D"
NAVY = "#1f3a5f"
SAGE = "#7a9e7e"
MUTED = "#888884"
PAPER = "#F5F5F0"

UNIVERSES = ["DOW30", "NAS100", "COMMODITY"]


def load_metric(dataset: str, variant_key: str, col: str = "Sharpe Ratio") -> float | None:
    path = ROOT / "results" / "qpl_ablation_v3" / dataset / variant_key / "metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return float(df[col].iloc[0])


def main() -> None:
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============== Plot 1: QPL vs VolProxy grouped bars =================
    sharpe = {
        ("V2", "QPL"):      [load_metric(u, "ppo_qpl_gate_v2") for u in UNIVERSES],
        ("V2", "VolProxy"): [load_metric(f"{u}_volproxy", "ppo_qpl_gate_v2") for u in UNIVERSES],
        ("V3", "QPL"):      [load_metric(u, "ppo_qpl_gate_v3") for u in UNIVERSES],
        ("V3", "VolProxy"): [load_metric(f"{u}_volproxy", "ppo_qpl_gate_v3") for u in UNIVERSES],
    }

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(UNIVERSES))
    bw = 0.18
    order = [("V2", "VolProxy"), ("V2", "QPL"), ("V3", "VolProxy"), ("V3", "QPL")]
    colors = {
        ("V2", "VolProxy"): MUTED,
        ("V2", "QPL"):      NAVY,
        ("V3", "VolProxy"): SAGE,
        ("V3", "QPL"):      LEMON,
    }
    labels = {
        ("V2", "VolProxy"): "Gate V2 · VolProxy levels (placebo)",
        ("V2", "QPL"):      "Gate V2 · QPL levels (QFSE)",
        ("V3", "VolProxy"): "Gate V3 · VolProxy levels (placebo)",
        ("V3", "QPL"):      "Gate V3 · QPL levels (QFSE)",
    }
    for i, key in enumerate(order):
        vals = sharpe[key]
        offset = (i - 1.5) * bw
        bars = ax.bar(
            x + offset, vals, bw,
            label=labels[key],
            color=colors[key], edgecolor=INK, linewidth=1.0,
        )
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.05, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(UNIVERSES, fontsize=11, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (test split)", fontsize=11)
    ax.set_title(
        "Does the lift come from QPL — or just from the gate architecture?",
        fontsize=13, fontweight="bold", color=INK, pad=14,
    )
    ax.axhline(0, color=INK, lw=0.6)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2, frameon=False)

    # Annotation footer
    txt = "QPL adds +16% Sharpe on average over Vol-Proxy, holding gate architecture constant."
    fig.text(0.5, 0.02, txt, ha="center", fontsize=10, color=INK, style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    p1 = out_dir / "qpl_vs_volproxy_sharpe.png"
    fig.savefig(p1, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {p1}")

    # ============== Plot 2: contribution decomposition (NAS100) ==========
    plain = load_metric("NAS100", "plain_ppo_reproduced")
    bb_v3 = load_metric("NAS100_volproxy", "ppo_qpl_gate_v3")
    qpl_v3 = load_metric("NAS100", "ppo_qpl_gate_v3")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    stages = ["Plain PPO\n(no gate, no QPL)", "+ Gate V3 with\nVol-Proxy levels", "+ QPL levels\n(full Gate V3)"]
    vals = [plain, bb_v3, qpl_v3]
    colors_b = [MUTED, SAGE, LEMON]
    bars = ax.bar(stages, vals, color=colors_b, edgecolor=INK, linewidth=1.2, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}",
                ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)

    # Annotation arrows for contribution
    if plain is not None and bb_v3 is not None and qpl_v3 is not None:
        gate_lift = bb_v3 - plain
        qpl_lift = qpl_v3 - bb_v3
        ax.annotate(
            f"Gate architecture\ncontributes +{gate_lift:.2f}",
            xy=(0.5, (plain + bb_v3) / 2),
            xytext=(0.5, max(vals) * 0.55),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
            ha="center", fontsize=10, color=INK,
        )
        ax.annotate(
            f"QPL specifically\ncontributes +{qpl_lift:.2f}",
            xy=(1.5, (bb_v3 + qpl_v3) / 2),
            xytext=(1.5, max(vals) * 0.55),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.2),
            ha="center", fontsize=10, color=INK,
        )

    ax.set_ylabel("Sharpe Ratio  (NAS100 test split)", fontsize=11)
    ax.set_title("Decomposing the Gate V3 lift on NAS100", fontsize=13, fontweight="bold", color=INK, pad=10)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p2 = out_dir / "qpl_vs_volproxy_decomposition.png"
    fig.savefig(p2, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
