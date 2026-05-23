"""Fetch daily OHLCV for major cryptocurrencies via yfinance.

Output: ``data/CRYPTO/processed.csv`` in the long-format schema expected by
:mod:`qf_oplrl.data_loader`:

    date,ticker,open,high,low,close,volume

Used to switch the project's experiments from US equities (DOW30 / NAS100 /
OLPS_djia) to the QFFC-aligned asset universes required by the project spec.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TICKERS = [
    "BTC-USD",
    "ETH-USD",
    "SOL-USD",
    "BNB-USD",
    "XRP-USD",
    "ADA-USD",
]

START = "2020-01-01"
END = "2026-05-23"

OUTPUT_DIR = PROJECT_ROOT / "data" / "CRYPTO"
OUTPUT_PATH = OUTPUT_DIR / "processed.csv"


def fetch_one(ticker: str) -> pd.DataFrame:
    print(f"  {ticker}: downloading...")
    df = yf.download(
        ticker,
        start=START,
        end=END,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty for {ticker}")
    # yfinance returns MultiIndex columns when threads are off + single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index().rename(columns=str.lower)
    needed = ["date", "open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            raise RuntimeError(f"missing column {col} for {ticker}; got {list(df.columns)}")
    df["ticker"] = ticker.replace("-USD", "USD")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in TICKERS:
        try:
            frames.append(fetch_one(t))
        except Exception as exc:
            print(f"  {t}: SKIP ({exc})", file=sys.stderr)
    if not frames:
        raise SystemExit("no crypto frames fetched")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)
    n_tickers = out["ticker"].nunique()
    n_rows = len(out)
    print(f"wrote {OUTPUT_PATH}  ({n_rows} rows · {n_tickers} tickers · {out['date'].min()} → {out['date'].max()})")


if __name__ == "__main__":
    main()
