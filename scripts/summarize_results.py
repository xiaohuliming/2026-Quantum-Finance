from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf_oplrl.plots import plot_cumulative_values, plot_drawdowns, plot_weight_heatmap


QPL_VARIANT_LABELS = {
    "plain_ppo_reproduced": "Plain PPO Reproduced",
    "ppo_qpl_state": "PPO + QPL State",
    "ppo_qpl_gate": "PPO + QPL Gate",
    "ppo_qpl_state_gate": "PPO + QPL State + Gate",
    "full_qf_oplrl": "Full QF-OPLRL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine baseline metrics and generate figures.")
    parser.add_argument("--results-dir", default="results", help="Base results directory.")
    return parser.parse_args()


def read_values(
    dataset_name: str,
    baselines_dir: Path,
    plain_rl_dir: Path,
    qpl_baselines_dir: Path,
    qpl_ablation_dir: Path,
) -> pd.DataFrame:
    frames = []
    dataset_dir = baselines_dir / dataset_name
    for prefix, file_name in [("Classical", "classical_values.csv"), ("OPL", "opl_values.csv")]:
        path = dataset_dir / file_name
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        frame = frame.add_prefix(f"{prefix} - ")
        frames.append(frame)
    rl_value_path = plain_rl_dir / dataset_name / "test_portfolio_value.csv"
    if rl_value_path.exists():
        frame = pd.read_csv(rl_value_path, index_col=0, parse_dates=True)
        frame.columns = ["RL - Plain PPO"]
        frames.append(frame)
    qpl_value_path = qpl_baselines_dir / dataset_name / "qpl_rule_values.csv"
    if qpl_value_path.exists():
        frame = pd.read_csv(qpl_value_path, index_col=0, parse_dates=True)
        frame.columns = ["QPL - QPL Rule"]
        frames.append(frame)
    qpl_dataset_dir = qpl_ablation_dir / dataset_name
    if qpl_dataset_dir.exists():
        for value_path in sorted(qpl_dataset_dir.glob("*/test_portfolio_value.csv")):
            frame = pd.read_csv(value_path, index_col=0, parse_dates=True)
            label = QPL_VARIANT_LABELS.get(value_path.parent.name, value_path.parent.name)
            frame.columns = [f"QPL RL - {label}"]
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def normalize_values(values: pd.DataFrame) -> pd.DataFrame:
    values = values.replace([float("inf"), float("-inf")], pd.NA).dropna(how="any")
    if values.empty:
        return values
    base = values.iloc[0].replace(0.0, pd.NA)
    return values.div(base).dropna(how="any")


def choose_weight_file(dataset_dir: Path) -> Path | None:
    preferred = [
        dataset_dir / "opl_weights" / "ons_diagonal_weights.csv",
        dataset_dir / "classical_weights" / "equal_weight_weights.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    for folder in ["opl_weights", "classical_weights"]:
        weight_dir = dataset_dir / folder
        if weight_dir.exists():
            files = sorted(weight_dir.glob("*weights.csv"))
            if files:
                return files[0]
    return None


def main() -> None:
    args = parse_args()
    results_dir = PROJECT_ROOT / args.results_dir
    baselines_dir = results_dir / "baselines"
    plain_rl_dir = results_dir / "plain_rl"
    qpl_baselines_dir = results_dir / "qpl_baselines"
    qpl_ablation_dir = results_dir / "qpl_ablation"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    basic_metric_frames = []
    if baselines_dir.exists():
        for metrics_path in sorted(baselines_dir.glob("*/*_metrics.csv")):
            basic_metric_frames.append(pd.read_csv(metrics_path))
    if plain_rl_dir.exists():
        for metrics_path in sorted(plain_rl_dir.glob("*/metrics.csv")):
            basic_metric_frames.append(pd.read_csv(metrics_path))

    final_metric_frames = list(basic_metric_frames)
    if qpl_baselines_dir.exists():
        for metrics_path in sorted(qpl_baselines_dir.glob("*/qpl_rule_metrics.csv")):
            final_metric_frames.append(pd.read_csv(metrics_path))
    if qpl_ablation_dir.exists():
        for metrics_path in sorted(qpl_ablation_dir.glob("*/qpl_ablation_metrics.csv")):
            final_metric_frames.append(pd.read_csv(metrics_path))

    if basic_metric_frames:
        metrics = pd.concat(basic_metric_frames, ignore_index=True)
        metrics.to_csv(tables_dir / "basic_experiment_metrics.csv", index=False)
        print(f"Wrote {tables_dir / 'basic_experiment_metrics.csv'}")
    if final_metric_frames:
        metrics = pd.concat(final_metric_frames, ignore_index=True)
        metrics.to_csv(tables_dir / "final_experiment_metrics.csv", index=False)
        print(f"Wrote {tables_dir / 'final_experiment_metrics.csv'}")

    dataset_names = set()
    for parent in [baselines_dir, plain_rl_dir, qpl_baselines_dir, qpl_ablation_dir]:
        if parent.exists():
            dataset_names.update(path.name for path in parent.iterdir() if path.is_dir())

    for dataset_name in sorted(dataset_names):
        values = read_values(dataset_name, baselines_dir, plain_rl_dir, qpl_baselines_dir, qpl_ablation_dir)
        if values.empty:
            continue
        values = normalize_values(values)
        if values.empty:
            continue
        plot_cumulative_values(
            values,
            f"{dataset_name} normalized cumulative portfolio value",
            figures_dir / f"{dataset_name}_cumulative_value.png",
        )
        plot_drawdowns(
            values,
            f"{dataset_name} drawdown",
            figures_dir / f"{dataset_name}_drawdown.png",
        )

        dataset_dir = baselines_dir / dataset_name
        weight_file = choose_weight_file(dataset_dir)
        if weight_file is not None:
            weights = pd.read_csv(weight_file, index_col=0, parse_dates=True)
            plot_weight_heatmap(
                weights,
                f"{dataset_name} weights: {weight_file.stem}",
                figures_dir / f"{dataset_name}_weight_heatmap.png",
            )
        plain_weight_path = plain_rl_dir / dataset_name / "test_weights.csv"
        if plain_weight_path.exists():
            plot_weight_heatmap(
                pd.read_csv(plain_weight_path, index_col=0, parse_dates=True),
                f"{dataset_name} weights: Plain PPO",
                figures_dir / f"{dataset_name}_plain_ppo_weight_heatmap.png",
            )
        full_qpl_weight_path = qpl_ablation_dir / dataset_name / "full_qf_oplrl" / "test_weights.csv"
        if full_qpl_weight_path.exists():
            plot_weight_heatmap(
                pd.read_csv(full_qpl_weight_path, index_col=0, parse_dates=True),
                f"{dataset_name} weights: Full QF-OPLRL",
                figures_dir / f"{dataset_name}_full_qf_oplrl_weight_heatmap.png",
            )
        print(f"Wrote figures for {dataset_name}")


if __name__ == "__main__":
    main()
