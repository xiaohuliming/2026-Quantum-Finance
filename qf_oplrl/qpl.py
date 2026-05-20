from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qf_oplrl.data_loader import MarketOHLCV


EPS = 1e-12


def compute_momentum(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute trailing close-to-close momentum."""
    return prices.pct_change(periods=int(window), fill_method=None)


def estimate_return_density(
    returns_window: np.ndarray,
    num_bins: int = 101,
    smoothing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate rho(r) on a fixed return grid from a rolling return window."""
    values = np.asarray(returns_window, dtype=float)
    values = values[np.isfinite(values)]
    bins = max(11, int(num_bins))
    smoothing = max(float(smoothing), 0.0)

    if values.size == 0:
        values = np.array([0.0], dtype=float)

    center = float(np.nanmean(values))
    std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.01
    if not np.isfinite(std) or std <= EPS:
        std = max(abs(center) * 0.25, 0.01)

    if values.size >= 10:
        lower_q, upper_q = np.nanquantile(values, [0.01, 0.99])
    else:
        lower_q, upper_q = float(values.min()), float(values.max())
    span = max(float(upper_q - lower_q), 6.0 * std, 0.02)
    lower = center - span / 2.0
    upper = center + span / 2.0
    if lower >= upper:
        lower, upper = center - 0.01, center + 0.01

    edges = np.linspace(lower, upper, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    counts = counts.astype(float) + smoothing
    grid_r = (edges[:-1] + edges[1:]) / 2.0
    dx = float(np.median(np.diff(grid_r))) if len(grid_r) > 1 else 1.0
    total_mass = float(counts.sum() * dx)
    if total_mass <= 0 or not np.isfinite(total_mass):
        rho = np.full_like(grid_r, 1.0 / (len(grid_r) * dx), dtype=float)
    else:
        rho = counts / total_mass
    return grid_r, rho


def wavefunction_from_density(
    rho: np.ndarray,
    dx: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build psi(r) from rho(r)=|psi(r)|^2 and normalize psi."""
    density = np.maximum(np.asarray(rho, dtype=float), float(eps))
    psi = np.sqrt(density)
    norm = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx))
    if norm <= 0 or not np.isfinite(norm):
        return np.full_like(psi, 1.0 / np.sqrt(len(psi) * dx), dtype=float)
    return psi / norm


def qaho_potential(
    grid_r: np.ndarray,
    c_gamma_d: float = 1.0,
    c_gamma_v: float = 0.0,
) -> np.ndarray:
    """Compute a QAHO-style potential V(r) for the course-project QPL pipeline.

    This is a computable approximation of the report idea:
    V(r) = c_gamma_d^2 * r^2 - (c_gamma_v / 4) * r^4. The default disables the
    quartic term to keep the Hamiltonian stable on short financial windows.
    """
    r = np.asarray(grid_r, dtype=float)
    return float(c_gamma_d) ** 2 * r**2 - (float(c_gamma_v) / 4.0) * r**4


def build_qfse_hamiltonian(
    grid_r: np.ndarray,
    potential: np.ndarray,
    hbar: float = 1.0,
    mass: float = 1.0,
) -> np.ndarray:
    """Build a finite-difference Hamiltonian for the time-independent QFSE."""
    grid = np.asarray(grid_r, dtype=float)
    values = np.asarray(potential, dtype=float)
    if grid.ndim != 1 or values.ndim != 1 or grid.size != values.size:
        raise ValueError("grid_r and potential must be one-dimensional arrays with the same length")
    if grid.size < 3:
        raise ValueError("Need at least three grid points to build a Hamiltonian")

    dx = float(np.median(np.diff(grid)))
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError("grid_r must be strictly increasing")

    n = grid.size
    second_derivative = np.diag(np.full(n, -2.0))
    second_derivative += np.diag(np.ones(n - 1), k=1)
    second_derivative += np.diag(np.ones(n - 1), k=-1)
    kinetic = -(float(hbar) ** 2 / (2.0 * max(float(mass), EPS))) * second_derivative / (dx**2)
    hamiltonian = kinetic + np.diag(values)
    return (hamiltonian + hamiltonian.T) / 2.0


def solve_energy_levels(
    hamiltonian: np.ndarray,
    num_levels: int = 2,
) -> np.ndarray:
    """Solve the lowest stable energy levels from the QFSE Hamiltonian."""
    matrix = np.asarray(hamiltonian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Hamiltonian contains NaN or infinite values")
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    eigenvalues = np.sort(eigenvalues[np.isfinite(eigenvalues)])
    if eigenvalues.size == 0:
        raise ValueError("No finite energy levels were found")
    levels = eigenvalues[: max(1, int(num_levels))]
    min_level = float(levels.min())
    if min_level < 0:
        levels = levels - min_level + EPS
    return levels


def energy_levels_to_nqpr(
    energies: np.ndarray,
    returns_std: float,
    scale: float = 1.0,
    eps: float = 1e-8,
    clip_min: float = 1.0001,
    clip_max: float = 1.5,
) -> np.ndarray:
    """Convert QFSE energy levels to normalized quantum price relatives.

    The mapping is an experiment-facing bridge from energy spacing to tradable
    price relatives: larger low-order energy gaps widen QPL bands, scaled by
    trailing return volatility.
    """
    values = np.sort(np.asarray(energies, dtype=float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([float(clip_min)], dtype=float)
    base = float(values[0])
    gaps = values[1:] - base if values.size > 1 else values - base + eps
    gaps = np.maximum(gaps, float(eps))
    relative = np.sqrt(gaps)
    mean_relative = float(np.mean(relative))
    if mean_relative <= 0 or not np.isfinite(mean_relative):
        relative = np.ones_like(relative)
        mean_relative = 1.0
    std = max(float(returns_std), float(eps))
    shifts = float(scale) * std * relative / mean_relative
    nqpr = np.exp(shifts)
    return np.clip(nqpr, float(clip_min), float(clip_max))


def _coerce_ohlcv(data: MarketOHLCV | dict[str, pd.DataFrame] | pd.DataFrame) -> MarketOHLCV:
    if isinstance(data, MarketOHLCV):
        return data
    if isinstance(data, dict):
        close = data.get("close")
        if close is None:
            close = data.get("prices")
        if close is None:
            raise ValueError("OHLCV dict must include a close or prices matrix")
        fallback_fields: dict[str, str] = {}
        open_prices = data.get("open")
        high_prices = data.get("high")
        low_prices = data.get("low")
        volume = data.get("volume")
        if open_prices is None:
            open_prices = close.copy()
            fallback_fields["open"] = "close"
        if high_prices is None:
            high_prices = close.copy()
            fallback_fields["high"] = "close"
        if low_prices is None:
            low_prices = close.copy()
            fallback_fields["low"] = "close"
        if volume is None:
            fallback_fields["volume"] = "missing"
        return MarketOHLCV(open_prices, high_prices, low_prices, close, volume, fallback_fields)
    close = data.sort_index().astype(float)
    return MarketOHLCV(
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=None,
        fallback_fields={"open": "close", "high": "close", "low": "close", "volume": "missing"},
    )


def _qpl_config_value(config: dict[str, Any], new_key: str, old_key: str, default: Any) -> Any:
    if new_key in config:
        return config[new_key]
    if old_key in config:
        return config[old_key]
    return default


def _proxy_nqpr(levels: int, returns_std: float, config: dict[str, Any]) -> np.ndarray:
    scale = float(config.get("nqpr_scale", 1.0))
    clip_min = float(config.get("nqpr_clip_min", 1.0001))
    clip_max = float(config.get("nqpr_clip_max", 1.5))
    std = max(float(returns_std), 1e-4)
    shifts = scale * std * np.arange(1, levels + 1, dtype=float)
    return np.clip(np.exp(shifts), clip_min, clip_max)


def _qaho_nqpr_for_window(window_returns: np.ndarray, levels: int, config: dict[str, Any]) -> tuple[np.ndarray, str]:
    finite = np.asarray(window_returns, dtype=float)
    finite = finite[np.isfinite(finite)]
    returns_std = float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.0
    min_observations = int(config.get("min_observations", max(20, min(60, int(config.get("window", 252))))))
    fallback_method = str(config.get("fallback_method", config.get("qpl_fallback_method", "rolling_vol_proxy")))
    if finite.size < min_observations or returns_std <= EPS:
        return _proxy_nqpr(levels, max(returns_std, 0.01), config), fallback_method

    try:
        grid_r, rho = estimate_return_density(
            finite,
            num_bins=int(config.get("num_bins", config.get("n_bins", 101))),
            smoothing=float(config.get("density_smoothing", 1.0)),
        )
        dx = float(np.median(np.diff(grid_r)))
        _ = wavefunction_from_density(rho, dx, eps=float(config.get("qpl_eps", 1e-8)))
        potential = qaho_potential(
            grid_r,
            c_gamma_d=float(config.get("c_gamma_d", 1.0)),
            c_gamma_v=float(config.get("c_gamma_v", 0.0)),
        )
        hamiltonian = build_qfse_hamiltonian(
            grid_r,
            potential,
            hbar=float(config.get("hbar", 1.0)),
            mass=float(config.get("mass", 1.0)),
        )
        energies = solve_energy_levels(hamiltonian, num_levels=levels + 1)
        nqpr = energy_levels_to_nqpr(
            energies,
            returns_std=returns_std,
            scale=float(config.get("nqpr_scale", 1.0)),
            eps=float(config.get("qpl_eps", 1e-8)),
            clip_min=float(config.get("nqpr_clip_min", 1.0001)),
            clip_max=float(config.get("nqpr_clip_max", 1.5)),
        )
        if nqpr.size < levels:
            nqpr = np.pad(nqpr, (0, levels - nqpr.size), mode="edge")
        return nqpr[:levels], "qaho_qfse"
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _proxy_nqpr(levels, max(returns_std, 0.01), config), fallback_method


def compute_rolling_qpl(
    prices: pd.DataFrame,
    lookback_window: int = 252,
    n_levels: int = 1,
    n_bins: int = 80,
    use_open_anchor: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute QPL levels using the configured QAHO/QFSE pipeline where possible."""
    config = {
        "method": "qaho_qfse",
        "fallback_method": "rolling_vol_proxy",
        "window": lookback_window,
        "num_levels": n_levels,
        "num_bins": n_bins,
        "anchor": "open" if use_open_anchor else "close",
    }
    package = compute_qpl_package(prices, config)
    return {
        key: value
        for key, value in package.items()
        if key.startswith("qpl_plus_") or key.startswith("qpl_minus_")
    }


def compute_qpl_features(
    prices: pd.DataFrame,
    qpl_plus: pd.DataFrame,
    qpl_minus: pd.DataFrame,
    epsilon_touch: float = 0.01,
    momentum_window: int = 5,
) -> dict[str, pd.DataFrame]:
    prices, qpl_plus = prices.align(qpl_plus, join="inner", axis=0)
    prices, qpl_minus = prices.align(qpl_minus, join="inner", axis=0)
    qpl_plus = qpl_plus.reindex(columns=prices.columns)
    qpl_minus = qpl_minus.reindex(columns=prices.columns)

    safe_prices = prices.replace(0.0, np.nan)
    d_plus = (qpl_plus - prices) / safe_prices
    d_minus = (prices - qpl_minus) / safe_prices
    z_qpl = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)
    z_qpl = z_qpl.mask(prices > qpl_plus, 1)
    z_qpl = z_qpl.mask(prices < qpl_minus, -1)

    touch = float(epsilon_touch)
    near_support = (prices.sub(qpl_minus).abs().div(safe_prices) <= touch) | (prices < qpl_minus)
    near_resistance = (prices.sub(qpl_plus).abs().div(safe_prices) <= touch) | (prices > qpl_plus)
    breakdown = prices < qpl_minus
    momentum = compute_momentum(prices, window=int(momentum_window))
    qpl_signal = pd.DataFrame(0, index=prices.index, columns=prices.columns, dtype=int)
    qpl_signal = qpl_signal.mask(near_support & (momentum > 0), 1)
    qpl_signal = qpl_signal.mask(near_resistance & (momentum < 0), -1)
    qpl_signal = qpl_signal.mask(breakdown & (momentum < 0), -2)
    return {
        "qpl_d_plus": d_plus,
        "qpl_d_minus": d_minus,
        "qpl_z": z_qpl,
        "near_support": near_support.astype(int),
        "near_resistance": near_resistance.astype(int),
        "breakdown": breakdown.astype(int),
        "qpl_momentum": momentum,
        "qpl_signal": qpl_signal.fillna(0).astype(int),
    }


def compute_qpl_package(
    ohlcv: MarketOHLCV | dict[str, pd.DataFrame] | pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute QPL+ and QPL- with qaho_qfse as the main method and proxy fallback."""
    config = config or {}
    market = _coerce_ohlcv(ohlcv)
    close = market.close.sort_index().astype(float)
    open_prices = market.open.reindex(index=close.index, columns=close.columns).fillna(close)
    high = market.high.reindex(index=close.index, columns=close.columns).fillna(close)
    low = market.low.reindex(index=close.index, columns=close.columns).fillna(close)

    method = str(config.get("method", config.get("qpl_method", "qaho_qfse")))
    window = int(_qpl_config_value(config, "window", "lookback_window", 252))
    levels = int(_qpl_config_value(config, "num_levels", "n_levels", 1))
    levels = max(1, levels)
    anchor_name = str(config.get("anchor", "open" if config.get("use_open_anchor", True) else "close")).lower()
    anchor = open_prices if anchor_name == "open" else close
    momentum = compute_momentum(close, window=int(config.get("momentum_window", 5)))
    returns = close.pct_change(fill_method=None)

    package: dict[str, pd.DataFrame] = {
        "qpl_momentum": momentum,
        "method_used": pd.DataFrame("", index=close.index, columns=close.columns),
    }
    for level in range(1, levels + 1):
        package[f"nqpr_{level}"] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        package[f"qpl_plus_{level}"] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        package[f"qpl_minus_{level}"] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)

    for ticker in close.columns:
        ticker_returns = returns[ticker].to_numpy(dtype=float)
        for row_pos, date in enumerate(close.index):
            start = max(0, row_pos - window)
            window_returns = ticker_returns[start:row_pos]
            if method == "rolling_vol_proxy":
                finite = window_returns[np.isfinite(window_returns)]
                returns_std = float(np.nanstd(finite, ddof=1)) if finite.size > 1 else 0.01
                nqpr_values = _proxy_nqpr(levels, returns_std, config)
                method_used = "rolling_vol_proxy"
            else:
                nqpr_values, method_used = _qaho_nqpr_for_window(window_returns, levels, config)
            anchor_value = float(anchor.at[date, ticker])
            if not np.isfinite(anchor_value) or anchor_value <= 0:
                continue
            package["method_used"].at[date, ticker] = method_used
            for idx, nqpr in enumerate(nqpr_values, start=1):
                package[f"nqpr_{idx}"].at[date, ticker] = float(nqpr)
                package[f"qpl_plus_{idx}"].at[date, ticker] = anchor_value * float(nqpr)
                package[f"qpl_minus_{idx}"].at[date, ticker] = anchor_value / float(nqpr)

    safe_close = close.replace(0.0, np.nan)
    use_high_low = bool(config.get("use_high_low_touch", True))
    fallback_to_close_touch = bool(config.get("fallback_to_close_touch", True))
    fallback_fields = market.fallback_fields or {}
    touch_high = high if use_high_low and not (fallback_fields.get("high") == "close" and fallback_to_close_touch) else close
    touch_low = low if use_high_low and not (fallback_fields.get("low") == "close" and fallback_to_close_touch) else close

    for level in range(1, levels + 1):
        plus = package[f"qpl_plus_{level}"]
        minus = package[f"qpl_minus_{level}"]
        d_plus = (plus - close) / safe_close
        d_minus = (close - minus) / safe_close
        z_qpl = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=int)
        z_qpl = z_qpl.mask(close > plus, 1)
        z_qpl = z_qpl.mask(close < minus, -1)
        touch_plus = touch_high >= plus
        touch_minus = touch_low <= minus
        intraday_breakout = touch_plus & (close > plus)
        intraday_breakdown = touch_minus & (close < minus)
        near_support = touch_minus | (d_minus.abs() <= float(config.get("epsilon_touch", 0.01)))
        near_resistance = touch_plus | (d_plus.abs() <= float(config.get("epsilon_touch", 0.01)))
        qpl_signal = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=int)
        qpl_signal = qpl_signal.mask(near_support & (momentum > 0), 1)
        qpl_signal = qpl_signal.mask(near_resistance & (momentum < 0), -1)
        qpl_signal = qpl_signal.mask(intraday_breakdown & (momentum < 0), -2)

        package[f"qpl_d_plus_{level}"] = d_plus
        package[f"qpl_d_minus_{level}"] = d_minus
        package[f"qpl_z_{level}"] = z_qpl
        package[f"qpl_signal_{level}"] = qpl_signal.fillna(0).astype(int)
        package[f"touch_plus_by_high_{level}"] = touch_plus.fillna(False).astype(int)
        package[f"touch_minus_by_low_{level}"] = touch_minus.fillna(False).astype(int)
        package[f"intraday_breakout_{level}"] = intraday_breakout.fillna(False).astype(int)
        package[f"intraday_breakdown_{level}"] = intraday_breakdown.fillna(False).astype(int)

    package["qpl_d_plus"] = package["qpl_d_plus_1"]
    package["qpl_d_minus"] = package["qpl_d_minus_1"]
    package["qpl_z"] = package["qpl_z_1"]
    package["qpl_signal"] = package["qpl_signal_1"]
    package["touch_plus_by_high"] = package["touch_plus_by_high_1"]
    package["touch_minus_by_low"] = package["touch_minus_by_low_1"]
    package["intraday_breakout"] = package["intraday_breakout_1"]
    package["intraday_breakdown"] = package["intraday_breakdown_1"]
    return package


def build_qpl_package(
    prices: MarketOHLCV | dict[str, pd.DataFrame] | pd.DataFrame,
    qpl_config: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    return compute_qpl_package(prices, qpl_config or {})


def lag_qpl_package_for_returns(
    qpl_package: dict[str, pd.DataFrame],
    returns_index: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Lag QPL features by one date before aligning them with same-date returns."""
    return {
        name: frame.shift(1).reindex(returns_index)
        for name, frame in qpl_package.items()
        if isinstance(frame, pd.DataFrame)
    }
