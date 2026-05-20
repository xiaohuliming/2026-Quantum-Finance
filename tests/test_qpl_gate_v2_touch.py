from __future__ import annotations

import unittest

import numpy as np

from qf_oplrl.qpl_gate_v2 import apply_qpl_gate_v2


class QPLGateV2TouchTests(unittest.TestCase):
    def test_resistance_add_intent_reduces_weight(self) -> None:
        raw = np.array([0.8, 0.2])
        previous = np.array([0.5, 0.5])
        adjusted, diagnostics = apply_qpl_gate_v2(
            raw,
            previous,
            {
                "qpl_d_plus": np.array([0.0, 1.0]),
                "qpl_d_minus": np.array([1.0, 1.0]),
                "qpl_momentum": np.array([-0.05, 0.0]),
                "touch_plus_by_high": np.array([1, 0]),
            },
            config={"resistance_penalty": 0.5, "g_min": 0.3, "g_max": 1.2},
        )
        self.assertGreaterEqual(float(adjusted.min()), 0.0)
        self.assertAlmostEqual(float(adjusted.sum()), 1.0, places=7)
        self.assertLess(float(adjusted[0]), float(raw[0]))
        self.assertIn("touch_plus_by_high", diagnostics["reasons"][0])

    def test_support_rebound_can_increase_weight(self) -> None:
        raw = np.array([0.8, 0.2])
        previous = np.array([0.5, 0.5])
        adjusted, _ = apply_qpl_gate_v2(
            raw,
            previous,
            {
                "qpl_d_plus": np.array([1.0, 1.0]),
                "qpl_d_minus": np.array([0.0, 1.0]),
                "qpl_momentum": np.array([0.05, 0.0]),
                "touch_minus_by_low": np.array([1, 0]),
            },
            config={"support_rebound_bonus": 0.2, "g_min": 0.3, "g_max": 1.2},
        )
        self.assertGreater(float(adjusted[0]), float(raw[0]))

    def test_intraday_breakdown_reduces_weight(self) -> None:
        raw = np.array([0.8, 0.2])
        previous = np.array([0.5, 0.5])
        adjusted, diagnostics = apply_qpl_gate_v2(
            raw,
            previous,
            {
                "qpl_d_plus": np.array([1.0, 1.0]),
                "qpl_d_minus": np.array([-0.1, 1.0]),
                "qpl_momentum": np.array([-0.05, 0.0]),
                "intraday_breakdown": np.array([1, 0]),
            },
            config={"breakdown_penalty": 0.6, "g_min": 0.3, "g_max": 1.2},
        )
        self.assertLess(float(adjusted[0]), float(raw[0]))
        self.assertIn("gate_multipliers", diagnostics)


if __name__ == "__main__":
    unittest.main()

