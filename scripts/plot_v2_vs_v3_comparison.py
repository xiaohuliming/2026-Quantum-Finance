"""Plot V2 vs V3 cumulative portfolio value, drawdown, and Sharpe bars.

Reads ``results/qpl_ablation_v3/<dataset>/<variant>/test_portfolio_value.csv``
files and produces three comparison figures per dataset:

* ``<dataset>_v2_vs_v3_cumulative.png``  - net asset value over the test set
* ``<dataset>_v2_vs_v3_drawdown.png``    - drawdown curve over the test set
* ``v2_vs_v3_sharpe_bars.png``           - cross-dataset Sharpe / MDD bar chart

Variants compared (matching the keys produced by ``run_gate_v3_experiments.py``):
``plain_ppo_reproduced``, ``ppo_qpl_gate_v2``, ``ppo_qpl_gate_v3``,
``ppo_qpl_state_gate_v3``, ``full_qf_oplrl_v3``.
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

DATASETS = ["DOW30", "NAS100", "OLPS_djia"]

VARIANTS = [
    ("plain_ppo_reproduced", "Plain PPO", "#888888", "--"),
    ("ppo_qpl_gate_v2", "Gate V2", "#1f77b4", "-"),
    ("ppo_qpl_gate_v3", "Gate V3 (Lee)", "#d62728", "-"),
    ("ppo_qpl_state_gate_v3", "State + Gate V3", "#ff7f0e", "-"),
    ("full_qf_oplrl_v3", "Full QF-OPLRL V3", "#2ca02c", "-"),
]


def _load_value(dataset: str, variant_key: str) -> pd.Series | None:
    path = ROOT / "results" / "qpl_ablation_v3" / dataset / variant_key / "test_portfolio_value.csv"
    if not path.exists():
        return None
    series = pd.read_csv(path, index_col=0).iloc[:, 0]
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _drawdown(value: pd.Series) -> pd.Series:
    peak = value.cummax()
    return value / peak - 1.0


def plot_dataset(dataset: str, out_dir: Path) -> None:
    fig_cum, ax_cum = plt.subplots(figsize=(10, 5))
    fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
    for key, label, color, ls in VARIANTS:
        series = _load_value(dataset, key)
        if series is None or series.empty:
            continue
        ax_cum.plot(series.index, series.values, color=color, ls=ls, lw=1.3, label=label)
        ax_dd.plot(series.index, _drawdown(series).values * 100.0, color=color, ls=ls, lw=1.3, label=label)

    ax_cum.set_title(f"{dataset} — Cumulative Portfolio Value (test set)")
    ax_cum.set_ylabel("Portfolio Value (start = 1.0)")
    ax_cum.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax_cum.legend(loc="best", fontsize=9)
    ax_cum.grid(alpha=0.3)
    fig_cum.autofmt_xdate()
    fig_cum.tight_layout()
    fig_cum.savefig(out_dir / f"{dataset}_v2_vs_v3_cumulative.png", dpi=140)
    plt.close(fig_cum)

    ax_dd.set_title(f"{dataset} — Drawdown (test set)")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.axhline(0.0, color="grey", lw=0.5)
    ax_dd.legend(loc="best", fontsize=9)
    ax_dd.grid(alpha=0.3)
    fig_dd.autofmt_xdate()
    fig_dd.tight_layout()
    fig_dd.savefig(out_dir / f"{dataset}_v2_vs_v3_drawdown.png", dpi=140)
    plt.close(fig_dd)


def plot_sharpe_bars(out_dir: Path) -> None:
    metrics_path = ROOT / "results" / "qpl_ablation_v3" / "gate_v3_metrics.csv"
    if not metrics_path.exists():
        print(f"missing {metrics_path}; skip sharpe bars")
        return
    df = pd.read_csv(metrics_path)
    method_label = {
        "ppo_qpl_gate_v2": "Gate V2",
        "ppo_qpl_gate_v3": "Gate V3 (Lee)",
        "ppo_qpl_state_gate_v3": "State + Gate V3",
        "full_qf_oplrl_v3": "Full QF-OPLRL V3",
    }
    df = df[df["Variant Key"].isin(method_label)].copy()
    df["Label"] = df["Variant Key"].map(method_label)
    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(13, 4.5))
    pivot_s = df.pivot(index="Dataset", columns="Label", values="Sharpe Ratio").reindex(DATASETS)
    pivot_d = df.pivot(index="Dataset", columns="Label", values="Maximum Drawdown").reindex(DATASETS)
    label_order = ["Gate V2", "Gate V3 (Lee)", "State + Gate V3", "Full QF-OPLRL V3"]
    pivot_s = pivot_s[label_order]
    pivot_d = pivot_d[label_order]
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    x = np.arange(len(DATASETS))
    width = 0.18
    for i, label in enumerate(label_order):
        ax_s.bar(x + (i - 1.5) * width, pivot_s[label].values, width, label=label, color=colors[i])
        ax_d.bar(x + (i - 1.5) * width, np.abs(pivot_d[label].values) * 100, width, label=label, color=colors[i])

    ax_s.set_xticks(x)
    ax_s.set_xticklabels(DATASETS)
    ax_s.set_title("Sharpe Ratio (higher is better)")
    ax_s.set_ylabel("Sharpe Ratio")
    ax_s.legend(fontsize=8, ncol=2)
    ax_s.grid(axis="y", alpha=0.3)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(DATASETS)
    ax_d.set_title("Maximum Drawdown (lower is better)")
    ax_d.set_ylabel("|MDD| (%)")
    ax_d.legend(fontsize=8, ncol=2)
    ax_d.grid(axis="y", alpha=0.3)

    fig.suptitle("Gate V2 vs Gate V3 (Lee Oscillator-augmented) across 3 datasets", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "v2_vs_v3_sharpe_bars.png", dpi=140)
    plt.close(fig)


def main() -> None:
    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        plot_dataset(ds, out_dir)
        print(f"wrote {ds} v2-vs-v3 plots")
    plot_sharpe_bars(out_dir)
    print("wrote v2_vs_v3_sharpe_bars.png")


if __name__ == "__main__":
    main()
