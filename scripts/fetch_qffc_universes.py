"""Fetch daily OHLCV for Forex, Commodities, and Indices universes (yfinance).

Outputs three long-format CSVs:

  data/FOREX/processed.csv
  data/COMMODITY/processed.csv
  data/INDEX/processed.csv

Same schema as the existing DOW30/NAS100/OLPS files:
  date,ticker,open,high,low,close,volume

These cover the QFFC asset classes (Forex / Crypto / Commodity / Index)
required by the project specification. CRYPTO is already produced by
``fetch_crypto_data.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIVERSES = {
    "FOREX": [
        "EURUSD=X",
        "GBPUSD=X",
        "USDJPY=X",
        "AUDUSD=X",
        "USDCHF=X",
        "USDCAD=X",
        "NZDUSD=X",
        "EURJPY=X",
    ],
    "COMMODITY": [
        "GC=F",   # Gold
        "SI=F",   # Silver
        "CL=F",   # WTI Crude
        "NG=F",   # Natural Gas
        "HG=F",   # Copper
    ],
    "INDEX": [
        "^GSPC",  # S&P 500
        "^DJI",   # Dow Jones
        "^IXIC",  # NASDAQ Composite
        "^GDAXI", # DAX 40
        "^N225",  # Nikkei 225
        "^FTSE",  # FTSE 100
    ],
}

START = "2018-01-01"
END = "2026-05-23"


def clean_ticker_for_storage(raw: str) -> str:
    return (
        raw.replace("=X", "")
           .replace("=F", "")
           .replace("^", "")
    )


def fetch_one(ticker_yf: str) -> pd.DataFrame:
    df = yf.download(ticker_yf, start=START, end=END, progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty for {ticker_yf}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index().rename(columns=str.lower)
    needed = ["date", "open", "high", "low", "close"]
    for col in needed:
        if col not in df.columns:
            raise RuntimeError(f"missing column {col} for {ticker_yf}; got {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0
    df["ticker"] = clean_ticker_for_storage(ticker_yf)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def build_universe(name: str, tickers: list[str]) -> None:
    out_dir = PROJECT_ROOT / "data" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "processed.csv"
    frames = []
    print(f"=== {name} ({len(tickers)} tickers) ===")
    for t in tickers:
        try:
            f = fetch_one(t)
            frames.append(f)
            print(f"  {t:<12} → {len(f):>4} rows  ({f['date'].min()} → {f['date'].max()})")
        except Exception as exc:
            print(f"  {t:<12} SKIP ({exc})", file=sys.stderr)
    if not frames:
        print(f"  no data for {name}, skipping", file=sys.stderr)
        return
    out = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    out.to_csv(out_path, index=False)
    print(f"  wrote {out_path}  ({len(out)} rows · {out['ticker'].nunique()} tickers)")


def main() -> None:
    for name, tickers in UNIVERSES.items():
        build_universe(name, tickers)


if __name__ == "__main__":
    main()
