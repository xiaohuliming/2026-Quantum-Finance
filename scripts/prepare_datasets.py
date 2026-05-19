from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "reference"
DATA_DIR = PROJECT_ROOT / "data"
TICKER_CONFIG = REFERENCE_DIR / "FinRL" / "finrl" / "config_tickers.py"
OLPS_SOURCE_DIR = REFERENCE_DIR / "OLPS" / "Data"

OUTPUT_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]

DEFAULT_MARKET_END_DATE = dt.date.today().isoformat()

DEFAULT_MARKET_RANGES = {
    "DOW30": ("2009-01-01", DEFAULT_MARKET_END_DATE),
    "NAS100": ("2009-01-01", DEFAULT_MARKET_END_DATE),
}

TICKER_LIST_NAMES = {
    "DOW30": "DOW_30_TICKER",
    "NAS100": "NAS_100_TICKER",
}

OLPS_PERIOD_STARTS = {
    "djia": "2001-01-14",
    "msci": "2006-04-01",
    "nyse-n": "1985-01-01",
    "nyse-o": "1962-07-03",
    "sp500": "1998-01-02",
    "tse": "1994-01-04",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare DOW30, NAS100, and OLPS datasets in "
            "date,ticker,open,high,low,close,volume format."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["DOW30", "NAS100", "OLPS"],
        default=["DOW30", "NAS100", "OLPS"],
        help="Datasets to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Base output directory. Defaults to the project data folder.",
    )
    parser.add_argument(
        "--dow30-start",
        default=DEFAULT_MARKET_RANGES["DOW30"][0],
        help="Start date for DOW30 Yahoo Finance download.",
    )
    parser.add_argument(
        "--dow30-end",
        default=DEFAULT_MARKET_RANGES["DOW30"][1],
        help="End date for DOW30 Yahoo Finance download.",
    )
    parser.add_argument(
        "--nas100-start",
        default=DEFAULT_MARKET_RANGES["NAS100"][0],
        help="Start date for NAS100 Yahoo Finance download.",
    )
    parser.add_argument(
        "--nas100-end",
        default=DEFAULT_MARKET_RANGES["NAS100"][1],
        help="End date for NAS100 Yahoo Finance download.",
    )
    parser.add_argument(
        "--raw-prices",
        action="store_true",
        help=(
            "Keep Yahoo raw open/high/low/close. By default, OHLC prices are "
            "adjusted using Adj Close when available, following FinRL's style."
        ),
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between market data requests.",
    )
    parser.add_argument(
        "--olps-initial-price",
        type=float,
        default=1.0,
        help="Initial synthetic price used when converting OLPS price relatives.",
    )
    return parser.parse_args()


def ensure_output_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_ticker_list(list_name: str) -> list[str]:
    if not TICKER_CONFIG.exists():
        raise FileNotFoundError(f"Ticker config not found: {TICKER_CONFIG}")

    tree = ast.parse(TICKER_CONFIG.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == list_name:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list):
                        raise ValueError(f"{list_name} is not a list in {TICKER_CONFIG}")
                    return [str(ticker).strip().upper() for ticker in value]

    raise ValueError(f"{list_name} not found in {TICKER_CONFIG}")


def require_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "The yfinance package is required for DOW30/NAS100 downloads. "
            "Install it with: python -m pip install yfinance"
        ) from exc
    return yf


def download_one_ticker(
    yf,
    ticker: str,
    start_date: str,
    end_date: str,
    adjust_prices: bool,
) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No rows returned for ticker {ticker}")

    if df.columns.nlevels != 1:
        df.columns = df.columns.droplevel(1)

    df = df.reset_index()
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    if adjust_prices and "adj_close" in df.columns:
        adjustment = df["adj_close"] / df["close"]
        for column in ["open", "high", "low", "close"]:
            df[column] = df[column] * adjustment

    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[OUTPUT_COLUMNS]
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close"])
    return df


def yahoo_period_timestamp(date_text: str) -> int:
    date_value = dt.datetime.strptime(date_text, "%Y-%m-%d")
    date_value = date_value.replace(tzinfo=dt.timezone.utc)
    return int(date_value.timestamp())


def download_one_ticker_chart_api(
    ticker: str,
    start_date: str,
    end_date: str,
    adjust_prices: bool,
) -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "period1": yahoo_period_timestamp(start_date),
            "period2": yahoo_period_timestamp(end_date),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
        },
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])

    results = chart.get("result") or []
    if not results:
        raise ValueError(f"No chart result returned for ticker {ticker}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quotes = indicators.get("quote") or []
    if not timestamps or not quotes:
        raise ValueError(f"No chart rows returned for ticker {ticker}")

    quote = quotes[0]
    row_count = len(timestamps)

    def field(name: str) -> list:
        values = quote.get(name) or [None] * row_count
        if len(values) != row_count:
            raise ValueError(f"Unexpected {name} length for ticker {ticker}")
        return values

    df = pd.DataFrame(
        {
            "date": [
                dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%Y-%m-%d")
                for ts in timestamps
            ],
            "open": field("open"),
            "high": field("high"),
            "low": field("low"),
            "close": field("close"),
            "volume": field("volume"),
        }
    )

    adjclose_values = indicators.get("adjclose") or []
    if adjclose_values:
        adjusted = adjclose_values[0].get("adjclose") or [None] * row_count
        if len(adjusted) == row_count:
            df["adj_close"] = adjusted

    if adjust_prices and "adj_close" in df.columns:
        adjustment = df["adj_close"] / df["close"]
        adjustment = adjustment.replace([np.inf, -np.inf], np.nan)
        for column in ["open", "high", "low", "close"]:
            df[column] = df[column] * adjustment

    df["ticker"] = ticker
    df = df[OUTPUT_COLUMNS]
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close"])
    if df.empty:
        raise ValueError(f"No usable chart rows returned for ticker {ticker}")
    return df


def download_one_ticker_with_fallback(
    yf,
    ticker: str,
    start_date: str,
    end_date: str,
    adjust_prices: bool,
) -> pd.DataFrame:
    try:
        return download_one_ticker(yf, ticker, start_date, end_date, adjust_prices)
    except Exception as primary_exc:
        print(f"[market] yfinance failed for {ticker}; trying Yahoo chart API")
        try:
            return download_one_ticker_chart_api(ticker, start_date, end_date, adjust_prices)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"yfinance failed: {primary_exc}; "
                f"Yahoo chart API failed: {fallback_exc}"
            ) from fallback_exc


def prepare_market_dataset(
    dataset_name: str,
    output_dir: Path,
    start_date: str,
    end_date: str,
    adjust_prices: bool,
    request_sleep: float = 0.25,
) -> Path:
    yf = require_yfinance()
    ticker_list = load_ticker_list(TICKER_LIST_NAMES[dataset_name])
    output_folder = output_dir / dataset_name
    ensure_output_folder(output_folder)

    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    for index, ticker in enumerate(ticker_list, start=1):
        print(f"[{dataset_name}] Downloading {ticker} ({index}/{len(ticker_list)})")
        try:
            frames.append(
                download_one_ticker_with_fallback(
                    yf,
                    ticker,
                    start_date,
                    end_date,
                    adjust_prices,
                )
            )
        except Exception as exc:  # Keep going so one stale ticker does not stop the dataset.
            failures.append((ticker, str(exc)))
            print(f"[{dataset_name}] Skipped {ticker}: {exc}")
        if request_sleep > 0:
            time.sleep(request_sleep)

    if not frames:
        raise RuntimeError(f"No market data was downloaded for {dataset_name}")

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    output_path = output_folder / "processed.csv"
    result.to_csv(output_path, index=False)

    failure_path = output_folder / "download_failures.csv"
    if failures:
        pd.DataFrame(failures, columns=["ticker", "error"]).to_csv(failure_path, index=False)
        print(f"[{dataset_name}] Wrote download failures to {failure_path}")
    else:
        pd.DataFrame(columns=["ticker", "error"]).to_csv(failure_path, index=False)

    print(f"[{dataset_name}] Wrote {len(result):,} rows to {output_path}")
    return output_path


def load_olps_mat(path: Path) -> np.ndarray:
    try:
        import scipy.io as scipy_io

        mat = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
        if "data" in mat:
            return np.asarray(mat["data"], dtype=float)
    except NotImplementedError:
        pass

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for MATLAB v7.3 .mat files. "
            "Install it with: python -m pip install h5py"
        ) from exc

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise ValueError(f"'data' variable not found in {path}")
        return np.asarray(handle["data"], dtype=float)


def orient_olps_data(data: np.ndarray) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D OLPS matrix, got shape {data.shape}")

    # Most OLPS files are days x assets. MATLAB v7.3 files may be read as assets x days.
    if data.shape[0] < data.shape[1] and data.shape[0] <= 200 and data.shape[1] > 365:
        data = data.T

    return data


def synthetic_olps_dates(dataset_name: str, n_days: int) -> pd.Index:
    start = OLPS_PERIOD_STARTS.get(dataset_name, "2000-01-01")
    return pd.bdate_range(start=start, periods=n_days).strftime("%Y-%m-%d")


def olps_to_ohlcv(
    dataset_name: str,
    relatives: np.ndarray,
    initial_price: float,
) -> pd.DataFrame:
    relatives = orient_olps_data(relatives)
    if np.any(relatives <= 0):
        raise ValueError(f"{dataset_name} contains non-positive price relatives")

    close = initial_price * np.cumprod(relatives, axis=0)
    open_ = np.vstack([np.full((1, close.shape[1]), initial_price), close[:-1]])
    high = np.maximum(open_, close)
    low = np.minimum(open_, close)
    volume = np.zeros_like(close)

    dates = synthetic_olps_dates(dataset_name, close.shape[0])
    tickers = [f"{dataset_name.upper().replace('-', '_')}_{i + 1:03d}" for i in range(close.shape[1])]

    frames = []
    for asset_index, ticker in enumerate(tickers):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_[:, asset_index],
                    "high": high[:, asset_index],
                    "low": low[:, asset_index],
                    "close": close[:, asset_index],
                    "volume": volume[:, asset_index],
                }
            )
        )

    return pd.concat(frames, ignore_index=True)[OUTPUT_COLUMNS]


def prepare_olps_dataset(output_dir: Path, initial_price: float) -> list[Path]:
    if not OLPS_SOURCE_DIR.exists():
        raise FileNotFoundError(f"OLPS source folder not found: {OLPS_SOURCE_DIR}")

    output_folder = output_dir / "OLPS"
    ensure_output_folder(output_folder)

    output_paths: list[Path] = []
    for source_path in sorted(OLPS_SOURCE_DIR.glob("*.mat")):
        dataset_name = source_path.stem
        print(f"[OLPS] Converting {source_path.name}")
        relatives = load_olps_mat(source_path)
        result = olps_to_ohlcv(dataset_name, relatives, initial_price)

        output_path = output_folder / f"{dataset_name}.csv"
        result.to_csv(output_path, index=False)
        output_paths.append(output_path)
        print(f"[OLPS] Wrote {len(result):,} rows to {output_path}")

    if not output_paths:
        raise RuntimeError(f"No .mat files found in {OLPS_SOURCE_DIR}")

    return output_paths


def selected(values: Iterable[str], dataset_name: str) -> bool:
    return dataset_name in set(values)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_folder(output_dir)

    adjust_prices = not args.raw_prices

    if selected(args.datasets, "DOW30"):
        prepare_market_dataset(
            dataset_name="DOW30",
            output_dir=output_dir,
            start_date=args.dow30_start,
            end_date=args.dow30_end,
            adjust_prices=adjust_prices,
            request_sleep=args.request_sleep,
        )

    if selected(args.datasets, "NAS100"):
        prepare_market_dataset(
            dataset_name="NAS100",
            output_dir=output_dir,
            start_date=args.nas100_start,
            end_date=args.nas100_end,
            adjust_prices=adjust_prices,
            request_sleep=args.request_sleep,
        )

    if selected(args.datasets, "OLPS"):
        prepare_olps_dataset(output_dir=output_dir, initial_price=args.olps_initial_price)

    print(f"Done at {dt.datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
