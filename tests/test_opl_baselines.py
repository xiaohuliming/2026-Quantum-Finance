from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qf_oplrl.opl_baselines import bcrp, generate_opl_weights, project_to_simplex


class OPLBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.price_relatives = pd.DataFrame(
            [
                [1.00, 1.00, 1.00],
                [1.02, 0.99, 1.01],
                [0.98, 1.03, 1.00],
                [1.01, 1.02, 0.99],
                [1.04, 0.98, 1.01],
                [0.99, 1.01, 1.03],
            ],
            columns=["A", "B", "C"],
        )
        self.config = {
            "opl": {
                "pamr": {"epsilon": 0.5, "C": 500.0, "variant": 0},
                "olmar": {"window": 3, "epsilon": 2.0},
                "ons": {"beta": 1.0, "delta": 0.125, "eta": 0.01},
            }
        }

    def assert_valid_weights(self, weights: pd.DataFrame) -> None:
        self.assertEqual(weights.shape, self.price_relatives.shape)
        self.assertTrue(np.isfinite(weights.to_numpy(dtype=float)).all())
        self.assertGreaterEqual(float(weights.min().min()), -1e-10)
        np.testing.assert_allclose(weights.sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-8)

    def test_project_to_simplex(self) -> None:
        weights = project_to_simplex(np.array([0.3, -0.2, 2.0]))
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreaterEqual(float(weights.min()), -1e-12)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)

    def test_opl_methods_produce_valid_weights(self) -> None:
        weights_by_method = {"BCRP": bcrp(self.price_relatives)}
        weights_by_method.update(generate_opl_weights(self.price_relatives, self.config))
        for weights in weights_by_method.values():
            self.assert_valid_weights(weights)

    def test_bcrp_not_much_worse_than_equal_weight_crp(self) -> None:
        bcrp_weights = bcrp(self.price_relatives)
        bcrp_wealth = float((self.price_relatives * bcrp_weights).sum(axis=1).prod())
        equal_weights = pd.DataFrame(1.0 / 3.0, index=self.price_relatives.index, columns=self.price_relatives.columns)
        equal_wealth = float((self.price_relatives * equal_weights).sum(axis=1).prod())
        self.assertGreaterEqual(bcrp_wealth + 1e-8, equal_wealth * 0.999)


if __name__ == "__main__":
    unittest.main()

