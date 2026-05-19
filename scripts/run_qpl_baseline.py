from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf_oplrl.backtest import run_backtest
from qf_oplrl.config import load_config, result_dir
from qf_oplrl.data_loader import load_datasets
from qf_oplrl.metrics import compute_metrics
from qf_oplrl.qpl import build_qpl_package
from qf_oplrl.qpl_strategy import qpl_rule_weights
from qf_oplrl.splits import split_by_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QPL rule-based portfolio baseline.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--first-only", action="store_true", help="Only process the first dataset source.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = load_datasets(config)
    if args.first_only:
        datasets = datasets[:1]

    base_output_dir = result_dir(config) / "qpl_baselines"
    backtest_config = config.get("backtest", {})
    all_metrics = []
    for data in datasets:
        output_dir = base_output_dir / data.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        qpl_package = build_qpl_package(data.prices, config.get("qpl", {}))
        weights = qpl_rule_weights(
            data.returns,
            qpl_package["qpl_signal"],
            config.get("qpl", {}),
            config.get("qpl_strategy", {}),
        )
        split = split_by_time(data.returns, **config.get("split", {}))
        test_returns = split.test
        test_weights = weights.loc[test_returns.index]

        result = run_backtest(
            test_returns,
            test_weights,
            initial_capital=float(backtest_config.get("initial_capital", 1.0)),
            transaction_cost_rate=float(backtest_config.get("transaction_cost_rate", 0.001)),
        )
        metrics = compute_metrics(
            result,
            initial_capital=float(backtest_config.get("initial_capital", 1.0)),
            annualization_factor=int(backtest_config.get("annualization_factor", 252)),
        )
        row = {
            "Dataset": data.dataset_name,
            "Method": "QPL Rule",
            "Method Type": "QPL Rule",
            **metrics,
        }
        pd.DataFrame([row]).to_csv(output_dir / "qpl_rule_metrics.csv", index=False)
        result["portfolio_value"].rename("QPL Rule").to_csv(output_dir / "qpl_rule_values.csv")
        result["weights"].to_csv(output_dir / "qpl_rule_weights.csv")
        all_metrics.append(row)
        print(f"Wrote QPL rule baseline for {data.dataset_name} to {output_dir}")

    combined = pd.DataFrame(all_metrics)
    combined.to_csv(base_output_dir / "qpl_rule_metrics.csv", index=False)
    print(f"Wrote {base_output_dir / 'qpl_rule_metrics.csv'}")


if __name__ == "__main__":
    main()

