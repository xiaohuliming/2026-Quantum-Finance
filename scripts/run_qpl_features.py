from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf_oplrl.config import load_config, result_dir
from qf_oplrl.data_loader import load_datasets
from qf_oplrl.plots import plot_qpl_trigger_example
from qf_oplrl.qpl import build_qpl_package


OUTPUT_FILES = {
    "qpl_plus_1": "qpl_plus_1.csv",
    "qpl_minus_1": "qpl_minus_1.csv",
    "qpl_d_plus": "qpl_d_plus.csv",
    "qpl_d_minus": "qpl_d_minus.csv",
    "qpl_z": "qpl_z.csv",
    "qpl_signal": "qpl_signal.csv",
    "qpl_momentum": "qpl_momentum.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling QPL features.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    parser.add_argument("--first-only", action="store_true", help="Only process the first dataset source.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = load_datasets(config)
    if args.first_only:
        datasets = datasets[:1]

    base_output_dir = result_dir(config) / "qpl_features"
    figures_dir = result_dir(config) / "figures"
    for data in datasets:
        package = build_qpl_package(data.prices, config.get("qpl", {}))
        output_dir = base_output_dir / data.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, file_name in OUTPUT_FILES.items():
            frame = package[key].copy()
            frame.index.name = "date"
            frame.to_csv(output_dir / file_name)

        signal_values = sorted(package["qpl_signal"].stack().dropna().unique().tolist())
        ticker = data.prices.columns[0]
        plot_qpl_trigger_example(
            data.prices[ticker],
            package["qpl_plus_1"][ticker],
            package["qpl_minus_1"][ticker],
            package["qpl_signal"][ticker],
            f"{data.dataset_name} QPL trigger example: {ticker}",
            figures_dir / f"{data.dataset_name}_qpl_trigger_example.png",
        )
        print(
            f"Wrote QPL features for {data.dataset_name} to {output_dir}. "
            f"rows={len(data.prices)}, tickers={data.prices.shape[1]}, signals={signal_values}"
        )


if __name__ == "__main__":
    main()

