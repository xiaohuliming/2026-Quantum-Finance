from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("gymnasium is required for QPLPortfolioEnv") from exc

from qf_oplrl.qpl_gate import apply_qpl_gate_to_weight_vector
from qf_oplrl.plain_rl_env import softmax


EPS = 1e-12


def _feature_from_package(qpl_features: dict[str, pd.DataFrame], *names: str) -> pd.DataFrame:
    for name in names:
        if name in qpl_features:
            return qpl_features[name]
    raise KeyError(f"Missing QPL feature. Tried: {', '.join(names)}")


class QPLPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        qpl_features: dict[str, pd.DataFrame] | None = None,
        lookback_window: int = 20,
        transaction_cost_rate: float = 0.001,
        use_qpl_state: bool = False,
        use_qpl_gate: bool = False,
        use_qpl_reward: bool = False,
        qpl_config: dict[str, Any] | None = None,
        reward_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.use_qpl_state = bool(use_qpl_state)
        self.use_qpl_gate = bool(use_qpl_gate)
        self.use_qpl_reward = bool(use_qpl_reward)
        self.qpl_config = qpl_config or {}
        self.reward_config = reward_config or {}
        self.lookback_window = int(lookback_window)
        self.transaction_cost_rate = float(transaction_cost_rate)

        needs_qpl = self.use_qpl_state or self.use_qpl_gate or self.use_qpl_reward
        if needs_qpl and qpl_features is None:
            raise ValueError("QPL features are required for the selected QPL RL mode")

        aligned = returns.sort_index().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        self.qpl_d_plus = None
        self.qpl_d_minus = None
        self.qpl_z = None
        self.qpl_signal = None

        if needs_qpl:
            assert qpl_features is not None
            required_frames: list[pd.DataFrame] = []
            if self.use_qpl_state:
                self.qpl_d_plus = _feature_from_package(qpl_features, "qpl_d_plus", "d_plus")
                self.qpl_d_minus = _feature_from_package(qpl_features, "qpl_d_minus", "d_minus")
                self.qpl_z = _feature_from_package(qpl_features, "qpl_z", "z_qpl")
                required_frames.extend([self.qpl_d_plus, self.qpl_d_minus, self.qpl_z])
            if self.use_qpl_gate or self.use_qpl_reward:
                self.qpl_signal = _feature_from_package(qpl_features, "qpl_signal", "signal")
                required_frames.append(self.qpl_signal)

            common_index = aligned.index
            common_columns = aligned.columns
            for frame in required_frames:
                common_index = common_index.intersection(frame.index)
                common_columns = common_columns.intersection(frame.columns)
            aligned = aligned.loc[common_index, common_columns].sort_index()

            finite_mask = pd.Series(True, index=aligned.index)
            aligned_frames = []
            for frame in required_frames:
                feature = frame.loc[aligned.index, aligned.columns].replace([np.inf, -np.inf], np.nan)
                aligned_frames.append(feature)
                finite_mask &= feature.notna().all(axis=1)
            finite_mask &= aligned.notna().all(axis=1)
            aligned = aligned.loc[finite_mask]
            aligned_frames = [frame.loc[finite_mask] for frame in aligned_frames]

            cursor = 0
            if self.use_qpl_state:
                self.qpl_d_plus = aligned_frames[cursor].astype(np.float32)
                self.qpl_d_minus = aligned_frames[cursor + 1].astype(np.float32)
                self.qpl_z = aligned_frames[cursor + 2].astype(np.float32)
                cursor += 3
            if self.use_qpl_gate or self.use_qpl_reward:
                self.qpl_signal = aligned_frames[cursor].fillna(0).astype(int)

        if len(aligned) <= self.lookback_window + 1:
            raise ValueError("Not enough aligned return rows for the requested lookback window")

        self.returns = aligned.astype(np.float32)
        self.n_assets = self.returns.shape[1]
        self.tickers = list(self.returns.columns)
        qpl_state_dim = 3 * self.n_assets if self.use_qpl_state else 0
        obs_dim = self.lookback_window * self.n_assets + self.n_assets + qpl_state_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.n_assets,), dtype=np.float32)

        self.current_step = self.lookback_window
        self.previous_weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0
        self.peak_value = 1.0

    def _get_observation(self) -> np.ndarray:
        end = min(self.current_step, len(self.returns))
        start = end - self.lookback_window
        window = self.returns.iloc[start:end].to_numpy(dtype=np.float32).reshape(-1)
        parts = [window, self.previous_weights]
        if self.use_qpl_state:
            assert self.qpl_d_plus is not None
            assert self.qpl_d_minus is not None
            assert self.qpl_z is not None
            feature_step = min(self.current_step, len(self.returns) - 1)
            parts.extend(
                [
                    self.qpl_d_plus.iloc[feature_step].to_numpy(dtype=np.float32),
                    self.qpl_d_minus.iloc[feature_step].to_numpy(dtype=np.float32),
                    self.qpl_z.iloc[feature_step].to_numpy(dtype=np.float32),
                ]
            )
        observation = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.previous_weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._get_observation(), {}

    def step(self, action):
        raw_weights = softmax(np.asarray(action, dtype=np.float32))
        if self.use_qpl_gate:
            assert self.qpl_signal is not None
            signal_row = self.qpl_signal.iloc[self.current_step]
            weights = apply_qpl_gate_to_weight_vector(raw_weights, signal_row, self.qpl_config)
        else:
            weights = raw_weights.copy()

        returns_row = self.returns.iloc[self.current_step].to_numpy(dtype=np.float32)
        gross_return = float(weights @ returns_row)
        turnover = float(np.abs(weights - self.previous_weights).sum())
        transaction_cost = self.transaction_cost_rate * turnover
        net_return = gross_return - transaction_cost
        base_reward = float(np.log(max(1.0 + net_return, EPS)))

        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = max(0.0, 1.0 - self.portfolio_value / max(self.peak_value, EPS))

        qpl_bonus = 0.0
        if self.use_qpl_reward:
            assert self.qpl_signal is not None
            signal = self.qpl_signal.iloc[self.current_step].to_numpy(dtype=float)
            direction = np.where(signal == 1, 1.0, np.where(signal < 0, -1.0, 0.0))
            qpl_bonus = float(np.sum(weights * direction * returns_row))

        reward = (
            base_reward
            + float(self.reward_config.get("lambda_qpl", 0.05)) * qpl_bonus
            - float(self.reward_config.get("lambda_dd", 0.1)) * current_drawdown
        )

        date = self.returns.index[self.current_step]
        self.previous_weights = weights.astype(np.float32)
        self.current_step += 1
        terminated = self.current_step >= len(self.returns)
        truncated = False
        observation = self._get_observation()
        info = {
            "date": date,
            "raw_weights": raw_weights.copy(),
            "weights": weights.copy(),
            "gross_return": gross_return,
            "daily_return": net_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "portfolio_value": self.portfolio_value,
            "drawdown": current_drawdown,
            "qpl_bonus": qpl_bonus,
        }
        return observation, float(reward), terminated, truncated, info
