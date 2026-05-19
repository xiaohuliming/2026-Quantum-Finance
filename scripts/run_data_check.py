from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qf_oplrl.config import load_config, result_dir
from qf_oplrl.data_loader import MarketData, load_datasets
from qf_oplrl.splits import split_by_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data loading and alignment checks.")
    parser.add_argument("--config", required=True, help="Path to a dataset YAML config.")
    return parser.parse_args()


def format_tickers(tickers: list[str]) -> str:
    return "|".join(map(str, tickers))


def build_summary(data: MarketData, split_config: dict) -> dict:
    splits = split_by_time(
        data.returns,
        train_ratio=float(split_config.get("train_ratio", 0.6)),
        val_ratio=float(split_config.get("val_ratio", 0.2)),
        test_ratio=float(split_config.get("test_ratio", 0.2)),
    )
    return {
        "dataset": data.dataset_name,
        "source_file": str(data.source_path),
        "raw_rows": len(data.raw_frame),
        "raw_tickers": data.raw_prices.shape[1],
        "retained_tickers": data.prices.shape[1],
        "dropped_ticker_count": len(data.dropped_tickers),
        "dropped_tickers": format_tickers(data.dropped_tickers),
        "all_tickers_retained": len(data.dropped_tickers) == 0,
        "price_start_date": data.prices.index.min().date().isoformat(),
        "price_end_date": data.prices.index.max().date().isoformat(),
        "price_trading_days": len(data.prices),
        "return_start_date": data.returns.index.min().date().isoformat(),
        "return_end_date": data.returns.index.max().date().isoformat(),
        "return_trading_days": len(data.returns),
        "mean_missing_ratio_before_alignment": float(data.missing_before.mean()),
        "mean_missing_ratio_after_alignment": float(data.missing_after.mean()),
        "train_days": len(splits.train),
        "validation_days": len(splits.validation),
        "test_days": len(splits.test),
        "retained_tickers_list": format_tickers(list(data.prices.columns)),
    }


def build_missing_report(data: MarketData) -> pd.DataFrame:
    first_valid = data.raw_prices.apply(lambda column: column.first_valid_index())
    last_valid = data.raw_prices.apply(lambda column: column.last_valid_index())
    non_missing_count = data.raw_prices.notna().sum()
    report = pd.DataFrame(
        {
            "dataset": data.dataset_name,
            "ticker": data.raw_prices.columns,
            "missing_ratio_before_alignment": data.missing_before.reindex(data.raw_prices.columns).values,
            "missing_ratio_after_alignment": data.missing_after.reindex(data.raw_prices.columns).values,
            "first_valid_date": [
                value.date().isoformat() if pd.notna(value) else ""
                for value in first_valid.reindex(data.raw_prices.columns)
            ],
            "last_valid_date": [
                value.date().isoformat() if pd.notna(value) else ""
                for value in last_valid.reindex(data.raw_prices.columns)
            ],
            "raw_non_missing_count": non_missing_count.reindex(data.raw_prices.columns).values,
            "retained": [ticker in data.prices.columns for ticker in data.raw_prices.columns],
        }
    )
    return report


def write_date_range(path: Path, summaries: list[dict]) -> None:
    lines = []
    for summary in summaries:
        lines.append(f"dataset: {summary['dataset']}")
        lines.append(f"source_file: {summary['source_file']}")
        lines.append(f"price_range: {summary['price_start_date']} -> {summary['price_end_date']}")
        lines.append(f"return_range: {summary['return_start_date']} -> {summary['return_end_date']}")
        lines.append(
            "split_days: "
            f"train={summary['train_days']}, "
            f"validation={summary['validation_days']}, "
            f"test={summary['test_days']}"
        )
        lines.append(f"retained_tickers: {summary['retained_tickers']}")
        lines.append(f"dropped_tickers: {summary['dropped_ticker_count']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    datasets = load_datasets(config)

    output_dir = result_dir(config) / "data_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = str(config["dataset"]["name"]).lower()
    summaries = [build_summary(data, config.get("split", {})) for data in datasets]
    missing_reports = [build_missing_report(data) for data in datasets]

    summary_path = output_dir / f"{base_name}_summary.csv"
    missing_path = output_dir / f"{base_name}_missing_values.csv"
    date_range_path = output_dir / f"{base_name}_date_range.txt"

    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.concat(missing_reports, ignore_index=True).to_csv(missing_path, index=False)
    write_date_range(date_range_path, summaries)

    print(f"Wrote {summary_path}")
    print(f"Wrote {missing_path}")
    print(f"Wrote {date_range_path}")
    for summary in summaries:
        print(
            f"{summary['dataset']}: {summary['retained_tickers']} tickers, "
            f"{summary['return_trading_days']} return days, "
            f"{summary['return_start_date']} -> {summary['return_end_date']}"
        )


if __name__ == "__main__":
    main()
