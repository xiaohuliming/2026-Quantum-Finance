from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qf_oplrl.config import resolve_project_path


@dataclass
class MarketData:
    dataset_name: str
    source_path: Path
    raw_frame: pd.DataFrame
    raw_prices: pd.DataFrame
    prices: pd.DataFrame
    returns: pd.DataFrame
    price_relatives: pd.DataFrame
    missing_before: pd.Series
    missing_after: pd.Series
    dropped_tickers: list[str]


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: normalize_column_name(column) for column in df.columns}
    return df.rename(columns=renamed)


def first_present(columns: pd.Index, candidates: list[str], role: str) -> str:
    normalized_candidates = [normalize_column_name(candidate) for candidate in candidates]
    for candidate in normalized_candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"No {role} column found. Tried: {', '.join(candidates)}")


def choose_price_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized_candidates = [normalize_column_name(candidate) for candidate in candidates]
    for candidate in normalized_candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No usable price column found. Tried: {', '.join(candidates)}")


def read_price_source(
    source_path: Path,
    dataset_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(source_path)
    df = normalize_columns(df)

    date_column = first_present(
        df.columns,
        dataset_config.get("date_column_candidates", ["date", "datetime"]),
        "date",
    )

    ticker_candidates = dataset_config.get("ticker_column_candidates", ["ticker", "tic", "symbol"])
    has_ticker_column = any(normalize_column_name(candidate) in df.columns for candidate in ticker_candidates)

    if has_ticker_column:
        ticker_column = first_present(df.columns, ticker_candidates, "ticker")
        price_column = choose_price_column(
            df,
            dataset_config.get("price_column_preference", ["adj_close", "close"]),
        )
        work = df[[date_column, ticker_column, price_column]].copy()
        work.columns = ["date", "ticker", "price"]
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work["ticker"] = work["ticker"].astype(str).str.strip()
        work["price"] = pd.to_numeric(work["price"], errors="coerce")
        work = work.dropna(subset=["date", "ticker"])
        work = work.sort_values(["date", "ticker"])
        work = work.drop_duplicates(["date", "ticker"], keep="last")
        prices = work.pivot(index="date", columns="ticker", values="price").sort_index()
        return df, prices

    date_column = first_present(df.columns, [date_column], "date")
    work = df.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column])
    work = work.set_index(date_column).sort_index()
    prices = work.apply(pd.to_numeric, errors="coerce")
    prices.columns = prices.columns.astype(str)
    return df, prices


def clean_price_matrix(
    prices: pd.DataFrame,
    keep_all_tickers: bool = True,
    max_missing_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    if prices.empty:
        raise ValueError("Price matrix is empty")

    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.where(prices > 0)

    missing_before = prices.isna().mean().sort_index()
    original_columns = list(prices.columns)

    if keep_all_tickers:
        kept_prices = prices.copy()
        dropped_tickers: list[str] = []
    else:
        kept_columns = missing_before[missing_before <= max_missing_ratio].index.tolist()
        kept_prices = prices[kept_columns].copy()
        dropped_tickers = [ticker for ticker in original_columns if ticker not in kept_columns]

    kept_prices = kept_prices.ffill()
    first_valid = kept_prices.apply(lambda column: column.first_valid_index())
    last_valid = kept_prices.apply(lambda column: column.last_valid_index())

    if first_valid.isna().any() or last_valid.isna().any():
        empty_tickers = first_valid[first_valid.isna()].index.union(last_valid[last_valid.isna()].index)
        if keep_all_tickers:
            raise ValueError(f"Cannot keep tickers with no valid price data: {list(empty_tickers)}")
        kept_prices = kept_prices.drop(columns=list(empty_tickers))
        dropped_tickers.extend([ticker for ticker in empty_tickers if ticker not in dropped_tickers])
        first_valid = kept_prices.apply(lambda column: column.first_valid_index())
        last_valid = kept_prices.apply(lambda column: column.last_valid_index())

    common_start = max(first_valid)
    common_end = min(last_valid)
    cleaned = kept_prices.loc[common_start:common_end].dropna(how="any")

    if cleaned.empty:
        raise ValueError("No common date range remains after cleaning and alignment")

    missing_after = cleaned.isna().mean().sort_index()
    cleaned = cleaned.sort_index()
    cleaned.columns.name = "ticker"
    return cleaned, missing_before, missing_after, dropped_tickers


def load_market_data_from_file(
    source_path: str | Path,
    dataset_name: str,
    dataset_config: dict[str, Any],
) -> MarketData:
    source = resolve_project_path(source_path)
    raw_frame, raw_prices = read_price_source(source, dataset_config)
    prices, missing_before, missing_after, dropped_tickers = clean_price_matrix(
        raw_prices,
        keep_all_tickers=bool(dataset_config.get("keep_all_tickers", True)),
        max_missing_ratio=float(dataset_config.get("max_missing_ratio", 0.2)),
    )
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    price_relatives = prices.div(prices.shift(1)).dropna(how="any")
    returns = returns.loc[price_relatives.index]

    return MarketData(
        dataset_name=dataset_name,
        source_path=source,
        raw_frame=raw_frame,
        raw_prices=raw_prices,
        prices=prices,
        returns=returns,
        price_relatives=price_relatives,
        missing_before=missing_before,
        missing_after=missing_after,
        dropped_tickers=dropped_tickers,
    )


def discover_dataset_sources(config: dict[str, Any]) -> list[tuple[str, Path]]:
    dataset_config = config["dataset"]
    dataset_name = str(dataset_config["name"])
    path = resolve_project_path(dataset_config["path"])

    if path.is_file():
        return [(dataset_name, path)]

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    files = dataset_config.get("files")
    if files:
        return [(f"{dataset_name}_{Path(file).stem}", path / file) for file in files]

    pattern = dataset_config.get("file_pattern", "*.csv")
    sources = sorted(path.glob(pattern))
    if not sources:
        raise FileNotFoundError(f"No dataset files matched {pattern} under {path}")
    return [(f"{dataset_name}_{source.stem}", source) for source in sources]


def load_datasets(config: dict[str, Any]) -> list[MarketData]:
    dataset_config = config["dataset"]
    return [
        load_market_data_from_file(source_path, dataset_name, dataset_config)
        for dataset_name, source_path in discover_dataset_sources(config)
    ]

