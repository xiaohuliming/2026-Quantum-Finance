from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qf_oplrl.data_loader import MarketOHLCV
from qf_oplrl.qpl import (
    build_qfse_hamiltonian,
    build_qpl_package,
    energy_levels_to_nqpr,
    estimate_return_density,
    qaho_potential,
    solve_energy_levels,
    wavefunction_from_density,
)


class QPLQAHOQFSETests(unittest.TestCase):
    def test_density_wavefunction_hamiltonian_and_nqpr(self) -> None:
        returns = np.array([0.01, -0.02, 0.005, 0.012, -0.004, 0.0, 0.006])
        grid, rho = estimate_return_density(returns, num_bins=21, smoothing=1.0)
        dx = float(np.median(np.diff(grid)))
        self.assertTrue((rho >= 0).all())
        self.assertAlmostEqual(float(np.sum(rho) * dx), 1.0, places=6)

        psi = wavefunction_from_density(rho, dx)
        self.assertAlmostEqual(float(np.sum(np.abs(psi) ** 2) * dx), 1.0, places=6)

        potential = qaho_potential(grid)
        hamiltonian = build_qfse_hamiltonian(grid, potential)
        np.testing.assert_allclose(hamiltonian, hamiltonian.T, atol=1e-10)

        energies = solve_energy_levels(hamiltonian, num_levels=3)
        self.assertTrue(np.isfinite(energies).all())
        self.assertTrue(np.all(np.diff(energies) >= 0))

        nqpr = energy_levels_to_nqpr(energies, returns_std=float(np.std(returns, ddof=1)))
        self.assertTrue((nqpr > 1.0).all())

    def test_qpl_bands_and_fallback(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8)
        close = pd.DataFrame({"A": np.linspace(100.0, 103.0, len(dates))}, index=dates)
        market = MarketOHLCV(open=close.copy(), high=close.copy(), low=close.copy(), close=close)
        package = build_qpl_package(
            market,
            {
                "method": "qaho_qfse",
                "window": 20,
                "num_levels": 2,
                "num_bins": 21,
                "min_observations": 20,
            },
        )
        self.assertTrue((package["qpl_minus_1"]["A"] < market.open["A"]).all())
        self.assertTrue((package["qpl_plus_1"]["A"] > market.open["A"]).all())
        width_1 = package["qpl_plus_1"]["A"] - package["qpl_minus_1"]["A"]
        width_2 = package["qpl_plus_2"]["A"] - package["qpl_minus_2"]["A"]
        self.assertTrue((width_2 >= width_1).all())
        self.assertIn("rolling_vol_proxy", set(package["method_used"]["A"]))

    def test_no_future_leakage(self) -> None:
        dates = pd.date_range("2024-01-01", periods=40)
        close = pd.DataFrame({"A": 100.0 + np.arange(40) * 0.2}, index=dates)
        market = MarketOHLCV(open=close.copy(), high=close + 0.5, low=close - 0.5, close=close)
        config = {"method": "qaho_qfse", "window": 10, "num_bins": 21, "num_levels": 1, "min_observations": 5}
        original = build_qpl_package(market, config)

        changed_close = close.copy()
        changed_close.iloc[30:] = changed_close.iloc[30:] * 3.0
        changed = MarketOHLCV(
            open=close.copy(),
            high=changed_close + 1.0,
            low=changed_close - 1.0,
            close=changed_close,
        )
        modified = build_qpl_package(changed, config)
        pd.testing.assert_series_equal(
            original["qpl_plus_1"].iloc[:25, 0],
            modified["qpl_plus_1"].iloc[:25, 0],
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()

