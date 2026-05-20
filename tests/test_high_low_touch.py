from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qf_oplrl.data_loader import MarketOHLCV
from qf_oplrl.qpl import build_qpl_package


class HighLowTouchTests(unittest.TestCase):
    def test_touch_and_intraday_flags(self) -> None:
        dates = pd.date_range("2024-01-01", periods=25)
        open_prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=dates)
        close = pd.DataFrame({"A": 106.0, "B": 94.0}, index=dates)
        high = pd.DataFrame({"A": 106.0, "B": 100.0}, index=dates)
        low = pd.DataFrame({"A": 100.0, "B": 94.0}, index=dates)
        market = MarketOHLCV(open=open_prices, high=high, low=low, close=close)
        package = build_qpl_package(
            market,
            {
                "method": "rolling_vol_proxy",
                "window": 5,
                "num_levels": 1,
                "nqpr_clip_min": 1.0001,
                "nqpr_clip_max": 1.01,
                "use_high_low_touch": True,
            },
        )
        last = dates[-1]
        self.assertEqual(int(package["touch_plus_by_high"].loc[last, "A"]), 1)
        self.assertEqual(int(package["touch_minus_by_low"].loc[last, "B"]), 1)
        self.assertEqual(int(package["intraday_breakout"].loc[last, "A"]), 1)
        self.assertEqual(int(package["intraday_breakdown"].loc[last, "B"]), 1)


if __name__ == "__main__":
    unittest.main()

