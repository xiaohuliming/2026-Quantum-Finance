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


def align_technical_features(
    returns: pd.DataFrame,
    technical_features: dict[str, pd.DataFrame] | None,
    technical_feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not technical_features:
        return returns, {}

    names = technical_feature_names or list(technical_features.keys())
    missing = [name for name in names if name not in technical_features]
    if missing:
        raise KeyError(f"Missing technical features: {missing}")

    aligned = returns.sort_index().replace([np.inf, -np.inf], np.nan)
    common_index = aligned.index
    common_columns = aligned.columns
    for name in names:
        frame = technical_features[name]
        common_index = common_index.intersection(frame.index)
        common_columns = common_columns.intersection(frame.columns)

    aligned = aligned.loc[common_index, common_columns].sort_index()
    finite_mask = aligned.notna().all(axis=1)
    selected = {}
    for name in names:
        feature = technical_features[name].loc[aligned.index, aligned.columns].replace([np.inf, -np.inf], np.nan)
        finite_mask &= feature.notna().all(axis=1)
        selected[name] = feature

    aligned = aligned.loc[finite_mask]
    selected = {name: frame.loc[finite_mask].astype(np.float32) for name, frame in selected.items()}
    return aligned, selected


class PlainPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        returns: pd.DataFrame,
        lookback_window: int = 20,
        transaction_cost_rate: float = 0.001,
        use_technical_state: bool = False,
        technical_features: dict[str, pd.DataFrame] | None = None,
        technical_feature_names: list[str] | None = None,
    ):
        super().__init__()
        self.use_technical_state = bool(use_technical_state)
        if self.use_technical_state and not technical_features:
            raise ValueError("Technical features are required when use_technical_state=True")

        aligned_returns = returns.sort_index().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        self.technical_feature_names: list[str] = []
        self.technical_features: dict[str, pd.DataFrame] = {}
        if self.use_technical_state:
            aligned_returns, self.technical_features = align_technical_features(
                aligned_returns,
                technical_features,
                technical_feature_names,
            )
            self.technical_feature_names = list(self.technical_features.keys())

        if len(aligned_returns) <= lookback_window + 1:
            raise ValueError("Not enough aligned return rows for the requested lookback window")
        self.returns = aligned_returns.astype(np.float32)
        self.lookback_window = int(lookback_window)
        self.transaction_cost_rate = float(transaction_cost_rate)
        self.n_assets = self.returns.shape[1]
        self.tickers = list(self.returns.columns)

        tech_dim = len(self.technical_feature_names) * self.n_assets if self.use_technical_state else 0
        obs_dim = self.lookback_window * self.n_assets + self.n_assets + tech_dim
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
        parts = [window, self.previous_weights]
        if self.use_technical_state:
            feature_step = min(self.current_step, len(self.returns) - 1)
            for name in self.technical_feature_names:
                parts.append(self.technical_features[name].iloc[feature_step].to_numpy(dtype=np.float32))
        observation = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

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
