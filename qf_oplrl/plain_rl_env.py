from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("gymnasium is required for PlainPortfolioEnv") from exc


EPS = 1e-12


def softmax(action: np.ndarray) -> np.ndarray:
    shifted = action - np.max(action)
    exp_values = np.exp(shifted)
    total = exp_values.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full_like(action, 1.0 / len(action), dtype=np.float32)
    return (exp_values / total).astype(np.float32)


class PlainPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 20,
        transaction_cost_rate: float = 0.001,
    ):
        super().__init__()
        if len(returns) <= lookback_window + 1:
            raise ValueError("Not enough return rows for the requested lookback window")
        self.returns = returns.astype(np.float32)
        self.lookback_window = int(lookback_window)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.n_assets = self.returns.shape[1]
        self.tickers = list(self.returns.columns)

        obs_dim = self.lookback_window * self.n_assets + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )
        self.current_step = self.lookback_window
        self.previous_weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0

    def _get_observation(self) -> np.ndarray:
        end = min(self.current_step, len(self.returns))
        start = end - self.lookback_window
        window = self.returns.iloc[start:end].to_numpy(dtype=np.float32).reshape(-1)
        return np.concatenate([window, self.previous_weights]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.previous_weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0
        return self._get_observation(), {}

    def step(self, action):
        weights = softmax(np.asarray(action, dtype=np.float32))
        returns_row = self.returns.iloc[self.current_step].to_numpy(dtype=np.float32)
        gross_return = float(weights @ returns_row)
        turnover = float(np.abs(weights - self.previous_weights).sum())
        transaction_cost = self.transaction_cost_rate * turnover
        net_return = gross_return - transaction_cost
        reward = float(np.log(max(1.0 + net_return, EPS)))
        self.portfolio_value *= 1.0 + net_return

        date = self.returns.index[self.current_step]
        self.previous_weights = weights
        self.current_step += 1
        terminated = self.current_step >= len(self.returns)
        truncated = False
        observation = self._get_observation()
        info = {
            "date": date,
            "weights": weights.copy(),
            "gross_return": gross_return,
            "daily_return": net_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "portfolio_value": self.portfolio_value,
        }
        return observation, reward, terminated, truncated, info

