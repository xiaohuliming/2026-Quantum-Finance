from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qf_oplrl.data_loader import load_market_data_from_file


class OHLCVLoaderTests(unittest.TestCase):
    def test_long_ohlcv_loader_and_close_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prices.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
                    "ticker": ["A", "A", "B", "B"],
                    "close": [100.0, 101.0, 50.0, 51.0],
                    "volume": [10, 11, 12, 13],
                }
            ).to_csv(path, index=False)
            data = load_market_data_from_file(path, "toy", {"keep_all_tickers": True})
            self.assertEqual(data.ohlcv.close.shape, (2, 2))
            pd.testing.assert_frame_equal(data.ohlcv.open, data.ohlcv.close)
            pd.testing.assert_frame_equal(data.ohlcv.high, data.ohlcv.close)
            pd.testing.assert_frame_equal(data.ohlcv.low, data.ohlcv.close)
            self.assertEqual(data.ohlcv.fallback_fields["open"], "close")
            self.assertIsNotNone(data.ohlcv.volume)


if __name__ == "__main__":
    unittest.main()

