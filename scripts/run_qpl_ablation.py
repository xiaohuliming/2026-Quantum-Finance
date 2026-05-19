from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf_oplrl.config import load_config, result_dir
from qf_oplrl.data_loader import load_datasets
from qf_oplrl.qpl import build_qpl_package
from qf_oplrl.qpl_rl import QPL_VARIANTS, run_qpl_ablation_for_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QPL RL ablation experiments.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override PPO timesteps.")
    parser.add_argument("--first-only", action="store_true", help="Only process the first dataset source.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=[variant["key"] for variant in QPL_VARIANTS],
        help="Run only one variant key. Can be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.timesteps is not None:
        config.setdefault("qpl_rl", {})["total_timesteps"] = args.timesteps

    datasets = load_datasets(config)
    if args.first_only:
        datasets = datasets[:1]

    output_dir = result_dir(config) / "qpl_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frames = []
    for data in datasets:
        print(f"Training QPL ablation variants for {data.dataset_name}")
        qpl_package = build_qpl_package(data.prices, config.get("qpl", {}))
        metrics_frames.append(
            run_qpl_ablation_for_dataset(data, qpl_package, config, output_dir, variants=args.variant)
        )
        print(f"Wrote QPL ablation results for {data.dataset_name} to {output_dir / data.dataset_name}")

    combined = pd.concat(metrics_frames, ignore_index=True)
    combined.to_csv(output_dir / "qpl_ablation_metrics.csv", index=False)
    print(f"Wrote {output_dir / 'qpl_ablation_metrics.csv'}")


if __name__ == "__main__":
    main()

