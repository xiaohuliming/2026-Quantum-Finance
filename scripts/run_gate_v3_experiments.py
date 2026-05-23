"""Run Gate V3 (Lee-Oscillator-augmented) experiments and Lee-only baseline.

This script is the parallel of ``run_qpl_ablation.py`` but with two additions:

1. Before training, it computes a Lee-Oscillator-encoded "chaotic momentum"
   panel and attaches it to the QPL feature package as ``lee_momentum``.
   The new Gate V3 variants in :data:`qf_oplrl.qpl_rl.QPL_VARIANTS` pick this
   up and use it in place of the linear ``qpl_momentum``.

2. It also evaluates a standalone Lee Oscillator portfolio (no QPL, no RL)
   via :func:`qf_oplrl.lee_predictor.lee_predictor_weights`. This shows up in
   the comparison table as ``Lee Oscillator Only``.

Outputs are written to ``results/qpl_ablation_v3/<dataset>/`` so they don't
overwrite the teammate's existing Gate V2 results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf_oplrl.backtest import run_backtest
from qf_oplrl.config import load_config, result_dir
from qf_oplrl.data_loader import load_datasets
from qf_oplrl.lee_predictor import (
    attach_lee_momentum_to_qpl_package,
    lee_predictor_weights,
)
from qf_oplrl.metrics import compute_metrics
from qf_oplrl.qpl import build_qpl_package
from qf_oplrl.qpl_rl import QPL_VARIANTS, run_qpl_ablation_for_dataset
from qf_oplrl.splits import split_by_time


GATE_V3_VARIANT_KEYS = {
    "plain_ppo_reproduced",
    "ppo_qpl_gate_v2",
    "ppo_qpl_gate_v3",
    "ppo_qpl_state_gate_v3",
    "full_qf_oplrl_v3",
    # V4 (Q1+Q3 optimised) variants
    "ppo_qpl_state_agg_gate_v3",
    "ppo_qpl_gate_v3_lee_reward",
    "full_qf_oplrl_v4",
    # V5 (velocity + whipsaw) variants
    "ppo_qpl_velocity_gate_v3",
    "ppo_qpl_gate_v3_whipsaw",
    "full_qf_oplrl_v5",
    # DDPG algorithm-comparability baselines
    "plain_ddpg",
    "ddpg_qpl_gate_v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gate V3 + Lee predictor experiments.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override PPO timesteps.")
    parser.add_argument("--first-only", action="store_true", help="Only process the first dataset source.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(GATE_V3_VARIANT_KEYS),
        help="Restrict to a subset of variants. Can be repeated.",
    )
    parser.add_argument(
        "--skip-lee-only",
        action="store_true",
        help="Skip the standalone Lee Oscillator baseline.",
    )
    return parser.parse_args()


def _lee_only_baseline(
    data,
    config: dict,
    output_dir: Path,
) -> dict | None:
    lee_cfg = config.get("lee_oscillator", {})
    predictor_cfg = config.get("lee_predictor", {})

    weights = lee_predictor_weights(
        data.returns,
        lookback=int(predictor_cfg.get("lookback", 20)),
        input_scale=float(predictor_cfg.get("input_scale", 50.0)),
        input_clip=float(predictor_cfg.get("input_clip", 1.0)),
        mode=str(predictor_cfg.get("mode", "soft")),
        temperature=float(predictor_cfg.get("temperature", 0.2)),
        max_single_weight=predictor_cfg.get("max_single_weight", 0.25),
        config=lee_cfg,
    )
    split = split_by_time(data.returns, **config.get("split", {}))
    test_weights = weights.loc[weights.index.isin(split.test.index)]
    if test_weights.empty:
        return None

    backtest_config = config.get("backtest", {})
    bt = run_backtest(
        data.returns.loc[test_weights.index],
        test_weights,
        initial_capital=float(backtest_config.get("initial_capital", 1.0)),
        transaction_cost_rate=float(backtest_config.get("transaction_cost_rate", 0.001)),
    )

    dataset_dir = output_dir / data.dataset_name / "lee_only"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    test_weights.to_csv(dataset_dir / "test_weights.csv")
    bt["portfolio_value"].to_csv(dataset_dir / "test_portfolio_value.csv")

    metrics = compute_metrics(
        bt,
        initial_capital=float(backtest_config.get("initial_capital", 1.0)),
        annualization_factor=int(backtest_config.get("annualization_factor", 252)),
    )
    row = {
        "Dataset": data.dataset_name,
        "Method": "Lee Oscillator Only",
        "Method Type": "Lee Baseline",
        "Variant Key": "lee_only",
        "Use Technical State": False,
        "Use QPL State": False,
        "Use QPL Gate": False,
        "Use QPL Gate V2": False,
        "Use QPL Gate V3": False,
        "Use QPL Reward": False,
        **metrics,
    }
    pd.DataFrame([row]).to_csv(dataset_dir / "metrics.csv", index=False)
    return row


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.timesteps is not None:
        config.setdefault("qpl_rl", {})["total_timesteps"] = args.timesteps

    datasets = load_datasets(config)
    if args.first_only:
        datasets = datasets[:1]

    output_dir = result_dir(config) / "qpl_ablation_v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    lee_cfg = config.get("lee_oscillator", {})
    lookback = int(config.get("lee_momentum", {}).get("lookback", 20))
    input_scale = float(config.get("lee_momentum", {}).get("input_scale", 50.0))
    input_clip = float(config.get("lee_momentum", {}).get("input_clip", 1.0))

    selected_keys = set(args.variant) if args.variant else GATE_V3_VARIANT_KEYS
    selected_keys = selected_keys & {variant["key"] for variant in QPL_VARIANTS}
    if not selected_keys:
        raise SystemExit("No matching Gate V3 variants found")
    variant_keys_list = sorted(selected_keys)

    metrics_frames: list[pd.DataFrame] = []
    extra_rows: list[dict] = []

    for data in datasets:
        print(f"=== {data.dataset_name} ===")
        qpl_package = build_qpl_package(data.ohlcv, config.get("qpl", {}))
        returns_for_lee = data.returns
        package_with_lee = attach_lee_momentum_to_qpl_package(
            qpl_package,
            returns_for_lee,
            lookback=lookback,
            input_scale=input_scale,
            input_clip=input_clip,
            config=lee_cfg,
        )

        if not args.skip_lee_only:
            row = _lee_only_baseline(data, config, output_dir)
            if row is not None:
                extra_rows.append(row)
                print(
                    f"  Lee Oscillator Only  Sharpe={row.get('Sharpe Ratio'):.3f}  "
                    f"MDD={row.get('Maximum Drawdown'):.3f}"
                )

        metrics_frames.append(
            run_qpl_ablation_for_dataset(
                data,
                package_with_lee,
                config,
                output_dir,
                variants=variant_keys_list,
            )
        )

    combined = pd.concat(metrics_frames, ignore_index=True)
    if extra_rows:
        combined = pd.concat([pd.DataFrame(extra_rows), combined], ignore_index=True)
    combined.to_csv(output_dir / "gate_v3_metrics.csv", index=False)
    print(f"wrote {output_dir / 'gate_v3_metrics.csv'}")


if __name__ == "__main__":
    main()
