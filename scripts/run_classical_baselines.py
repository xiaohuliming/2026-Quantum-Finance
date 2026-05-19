from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf_oplrl.backtest import run_backtest
from qf_oplrl.classical_baselines import generate_classical_weights
from qf_oplrl.config import load_config, result_dir
from qf_oplrl.data_loader import load_datasets
from qf_oplrl.metrics import compute_metrics
from qf_oplrl.splits import split_by_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classical portfolio baselines.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    return parser.parse_args()


def save_series_frame(path: Path, series_by_method: dict[str, pd.Series]) -> None:
    frame = pd.DataFrame(series_by_method)
    frame.index.name = "date"
    frame.to_csv(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = load_datasets(config)
    base_output_dir = result_dir(config) / "baselines"
    backtest_config = config.get("backtest", {})

    for data in datasets:
        output_dir = base_output_dir / data.dataset_name
        weights_dir = output_dir / "classical_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        split = split_by_time(data.returns, **config.get("split", {}))
        test_returns = split.test
        all_weights = generate_classical_weights(data.returns, data.price_relatives, config)

        metrics_rows = []
        value_series = {}
        for method, weights in all_weights.items():
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
            metrics_rows.append(
                {
                    "Dataset": data.dataset_name,
                    "Method": method,
                    "Method Type": "Classical",
                    **metrics,
                }
            )
            value_series[method] = result["portfolio_value"]
            test_weights.to_csv(weights_dir / f"{method.lower().replace(' ', '_')}_weights.csv")

        pd.DataFrame(metrics_rows).to_csv(output_dir / "classical_metrics.csv", index=False)
        save_series_frame(output_dir / "classical_values.csv", value_series)
        print(f"Wrote classical baseline results for {data.dataset_name} to {output_dir}")


if __name__ == "__main__":
    main()

