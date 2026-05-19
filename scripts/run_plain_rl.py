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
from qf_oplrl.plain_rl import run_plain_rl_for_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a plain PPO portfolio baseline.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total PPO timesteps.")
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Only run the first dataset source in a multi-file config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.timesteps is not None:
        config.setdefault("plain_rl", {})["total_timesteps"] = args.timesteps
    datasets = load_datasets(config)
    if args.first_only:
        datasets = datasets[:1]

    output_dir = result_dir(config) / "plain_rl"
    metrics_frames = []
    for data in datasets:
        print(f"Training Plain PPO for {data.dataset_name}")
        metrics_frames.append(run_plain_rl_for_dataset(data, config, output_dir))
        print(f"Wrote Plain PPO results for {data.dataset_name} to {output_dir / data.dataset_name}")

    combined = pd.concat(metrics_frames, ignore_index=True)
    combined.to_csv(output_dir / "plain_rl_metrics.csv", index=False)
    print(f"Wrote {output_dir / 'plain_rl_metrics.csv'}")


if __name__ == "__main__":
    main()
