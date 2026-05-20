from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qf_oplrl.qpl_rl_env import QPLPortfolioEnv


class RewardFormulaTests(unittest.TestCase):
    def test_reward_matches_report_formula(self) -> None:
        returns = pd.DataFrame(
            {
                "A": [0.01, 0.02, -0.01, 0.03],
                "B": [0.00, 0.01, 0.02, -0.01],
            },
            index=pd.date_range("2024-01-01", periods=4),
        )
        env = QPLPortfolioEnv(
            returns,
            lookback_window=1,
            transaction_cost_rate=0.01,
            reward_config={"lambda_tc": 2.0, "lambda_dd": 0.5, "lambda_qpl": 0.0},
        )
        observation, _ = env.reset()
        _, reward, _, _, info = env.step(np.array([2.0, -2.0], dtype=np.float32))
        expected = (
            np.log(max(1.0 + info["gross_return"], 1e-12))
            - 2.0 * info["transaction_cost"]
            - 0.5 * info["drawdown"]
        )
        self.assertAlmostEqual(float(reward), float(expected), places=8)
        self.assertAlmostEqual(
            float(info["portfolio_value"]),
            float(1.0 + info["daily_return"]),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()

