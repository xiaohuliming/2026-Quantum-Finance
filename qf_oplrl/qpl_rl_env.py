from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("gymnasium is required for QPLPortfolioEnv") from exc

from qf_oplrl.qpl_gate import apply_qpl_gate_to_weight_vector, signal_to_multiplier
from qf_oplrl.qpl_gate_v2 import apply_qpl_gate_v2_to_weight_vector, compute_qpl_gate_scores
from qf_oplrl.plain_rl_env import align_technical_features, softmax


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
        use_qpl_state_agg: bool = False,
        use_qpl_state_velocity: bool = False,
        use_qpl_gate: bool = False,
        use_qpl_gate_v2: bool = False,
        use_qpl_gate_v3: bool = False,
        use_qpl_reward: bool = False,
        use_lee_reward: bool = False,
        use_whipsaw_penalty: bool = False,
        velocity_lookback: int = 5,
        use_technical_state: bool = False,
        technical_features: dict[str, pd.DataFrame] | None = None,
        technical_feature_names: list[str] | None = None,
        qpl_execution_features: dict[str, pd.DataFrame] | None = None,
        qpl_config: dict[str, Any] | None = None,
        qpl_gate_v2_config: dict[str, Any] | None = None,
        reward_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.use_qpl_state = bool(use_qpl_state)
        self.use_qpl_state_agg = bool(use_qpl_state_agg)
        self.use_qpl_state_velocity = bool(use_qpl_state_velocity)
        self.use_qpl_gate = bool(use_qpl_gate)
        self.use_qpl_gate_v2 = bool(use_qpl_gate_v2)
        self.use_qpl_gate_v3 = bool(use_qpl_gate_v3)
        self.use_qpl_reward = bool(use_qpl_reward)
        self.use_lee_reward = bool(use_lee_reward)
        self.use_whipsaw_penalty = bool(use_whipsaw_penalty)
        self.velocity_lookback = max(1, int(velocity_lookback))
        if self.use_qpl_state and self.use_qpl_state_agg:
            raise ValueError("Use either raw QPL state or aggregated QPL state, not both")
        self.use_technical_state = bool(use_technical_state)
        self.qpl_config = qpl_config or {}
        self.qpl_gate_v2_config = qpl_gate_v2_config or {}
        self.reward_config = reward_config or {}
        self.lookback_window = int(lookback_window)
        self.transaction_cost_rate = float(transaction_cost_rate)

        active_gates = [self.use_qpl_gate, self.use_qpl_gate_v2, self.use_qpl_gate_v3]
        if sum(bool(flag) for flag in active_gates) > 1:
            raise ValueError("Use only one of QPL Gate V1 / V2 / V3 at a time")

        needs_qpl = (
            self.use_qpl_state
            or self.use_qpl_state_agg
            or self.use_qpl_state_velocity
            or self.use_qpl_gate
            or self.use_qpl_gate_v2
            or self.use_qpl_gate_v3
            or self.use_qpl_reward
            or self.use_lee_reward
        )
        if needs_qpl and qpl_features is None:
            raise ValueError("QPL features are required for the selected QPL RL mode")
        if self.use_technical_state and not technical_features:
            raise ValueError("Technical features are required when use_technical_state=True")

        aligned = returns.sort_index().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        self.qpl_d_plus = None
        self.qpl_d_minus = None
        self.qpl_z = None
        self.qpl_momentum = None
        self.lee_momentum = None
        self.qpl_signal = None
        self.qpl_execution_features: dict[str, pd.DataFrame] = {}
        self.technical_feature_names: list[str] = []
        self.technical_features: dict[str, pd.DataFrame] = {}

        if needs_qpl:
            assert qpl_features is not None
            required_frames: list[pd.DataFrame] = []
            if (
                self.use_qpl_state
                or self.use_qpl_state_agg
                or self.use_qpl_state_velocity
                or self.use_qpl_gate_v2
                or self.use_qpl_gate_v3
            ):
                self.qpl_d_plus = _feature_from_package(qpl_features, "qpl_d_plus", "d_plus")
                self.qpl_d_minus = _feature_from_package(qpl_features, "qpl_d_minus", "d_minus")
                required_frames.extend([self.qpl_d_plus, self.qpl_d_minus])
            if self.use_qpl_state or self.use_qpl_state_agg:
                self.qpl_z = _feature_from_package(qpl_features, "qpl_z", "z_qpl")
                required_frames.append(self.qpl_z)
            if self.use_qpl_gate or self.use_qpl_gate_v2 or self.use_qpl_gate_v3 or self.use_qpl_reward:
                self.qpl_signal = _feature_from_package(qpl_features, "qpl_signal", "signal")
                required_frames.append(self.qpl_signal)
            if self.use_qpl_gate_v2 or self.use_qpl_gate_v3:
                self.qpl_momentum = _feature_from_package(qpl_features, "qpl_momentum", "momentum")
                required_frames.append(self.qpl_momentum)
            if self.use_qpl_gate_v3 or self.use_lee_reward:
                self.lee_momentum = _feature_from_package(qpl_features, "lee_momentum")
                required_frames.append(self.lee_momentum)

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
            if (
                self.use_qpl_state
                or self.use_qpl_state_agg
                or self.use_qpl_state_velocity
                or self.use_qpl_gate_v2
                or self.use_qpl_gate_v3
            ):
                self.qpl_d_plus = aligned_frames[cursor].astype(np.float32)
                self.qpl_d_minus = aligned_frames[cursor + 1].astype(np.float32)
                cursor += 2
            if self.use_qpl_state or self.use_qpl_state_agg:
                self.qpl_z = aligned_frames[cursor].astype(np.float32)
                cursor += 1
            if self.use_qpl_gate or self.use_qpl_gate_v2 or self.use_qpl_gate_v3 or self.use_qpl_reward:
                self.qpl_signal = aligned_frames[cursor].fillna(0).astype(int)
                cursor += 1
            if self.use_qpl_gate_v2 or self.use_qpl_gate_v3:
                self.qpl_momentum = aligned_frames[cursor].astype(np.float32)
                cursor += 1
            if self.use_qpl_gate_v3 or self.use_lee_reward:
                self.lee_momentum = aligned_frames[cursor].astype(np.float32)
                cursor += 1

        if self.use_technical_state:
            aligned, self.technical_features = align_technical_features(
                aligned,
                technical_features,
                technical_feature_names,
            )
            self.technical_feature_names = list(self.technical_features.keys())
            if self.qpl_d_plus is not None:
                self.qpl_d_plus = self.qpl_d_plus.loc[aligned.index, aligned.columns]
            if self.qpl_d_minus is not None:
                self.qpl_d_minus = self.qpl_d_minus.loc[aligned.index, aligned.columns]
            if self.qpl_z is not None:
                self.qpl_z = self.qpl_z.loc[aligned.index, aligned.columns]
            if self.qpl_momentum is not None:
                self.qpl_momentum = self.qpl_momentum.loc[aligned.index, aligned.columns]
            if self.lee_momentum is not None:
                self.lee_momentum = self.lee_momentum.loc[aligned.index, aligned.columns]
            if self.qpl_signal is not None:
                self.qpl_signal = self.qpl_signal.loc[aligned.index, aligned.columns]
            if self.qpl_d_plus is not None and self.qpl_d_minus is not None and self.qpl_z is None and self.use_qpl_state_agg:
                pass  # qpl_z already loaded above when use_qpl_state_agg is on

        if qpl_execution_features is not None:
            for name, frame in qpl_execution_features.items():
                if isinstance(frame, pd.DataFrame):
                    self.qpl_execution_features[name] = frame.reindex(index=aligned.index, columns=aligned.columns)

        if qpl_execution_features is not None:
            for name, frame in qpl_execution_features.items():
                if isinstance(frame, pd.DataFrame):
                    self.qpl_execution_features[name] = frame.reindex(index=aligned.index, columns=aligned.columns)

        if len(aligned) <= self.lookback_window + 1:
            raise ValueError("Not enough aligned return rows for the requested lookback window")

        self.returns = aligned.astype(np.float32)
        self.n_assets = self.returns.shape[1]
        self.tickers = list(self.returns.columns)
        tech_state_dim = len(self.technical_feature_names) * self.n_assets if self.use_technical_state else 0
        qpl_state_dim = 3 * self.n_assets if self.use_qpl_state else 0
        qpl_state_agg_dim = 8 if self.use_qpl_state_agg else 0
        qpl_state_velocity_dim = 4 if self.use_qpl_state_velocity else 0
        obs_dim = (
            self.lookback_window * self.n_assets
            + self.n_assets
            + tech_state_dim
            + qpl_state_dim
            + qpl_state_agg_dim
            + qpl_state_velocity_dim
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.n_assets,), dtype=np.float32)

        self.current_step = self.lookback_window
        self.previous_weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self.last_weight_diff = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0
        self.peak_value = 1.0

    def _execution_row(self, name: str, step: int, fallback: pd.Series | None = None) -> pd.Series:
        frame = self.qpl_execution_features.get(name)
        if frame is not None:
            row = frame.iloc[step]
            if row.notna().any():
                return row
        if fallback is not None:
            return fallback
        return pd.Series(0.0, index=self.tickers)

    def _get_observation(self) -> np.ndarray:
        end = min(self.current_step, len(self.returns))
        start = end - self.lookback_window
        window = self.returns.iloc[start:end].to_numpy(dtype=np.float32).reshape(-1)
        parts = [window, self.previous_weights]
        if self.use_technical_state:
            feature_step = min(self.current_step, len(self.returns) - 1)
            for name in self.technical_feature_names:
                parts.append(self.technical_features[name].iloc[feature_step].to_numpy(dtype=np.float32))
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
        if self.use_qpl_state_agg:
            assert self.qpl_d_plus is not None
            assert self.qpl_d_minus is not None
            assert self.qpl_z is not None
            feature_step = min(self.current_step, len(self.returns) - 1)
            d_plus = self.qpl_d_plus.iloc[feature_step].to_numpy(dtype=np.float32)
            d_minus = self.qpl_d_minus.iloc[feature_step].to_numpy(dtype=np.float32)
            z_row = self.qpl_z.iloc[feature_step].to_numpy(dtype=np.float32)
            n = max(1, d_plus.size)
            finite_dp = d_plus[np.isfinite(d_plus)]
            finite_dm = d_minus[np.isfinite(d_minus)]
            min_dp = float(finite_dp.min()) if finite_dp.size else 0.0
            min_dm = float(finite_dm.min()) if finite_dm.size else 0.0
            aggregates = np.array(
                [
                    float(np.nanmean(d_plus)) if np.isfinite(d_plus).any() else 0.0,
                    float(np.nanmean(d_minus)) if np.isfinite(d_minus).any() else 0.0,
                    min_dp,
                    min_dm,
                    float(np.sum(z_row == 1)) / n,
                    float(np.sum(z_row == -1)) / n,
                    float(np.nanmean(np.tanh(d_plus * 50.0))) if np.isfinite(d_plus).any() else 0.0,
                    float(np.nanmean(np.tanh(d_minus * 50.0))) if np.isfinite(d_minus).any() else 0.0,
                ],
                dtype=np.float32,
            )
            parts.append(np.nan_to_num(aggregates, nan=0.0, posinf=0.0, neginf=0.0))
        if self.use_qpl_state_velocity:
            assert self.qpl_d_plus is not None
            assert self.qpl_d_minus is not None
            feature_step = min(self.current_step, len(self.returns) - 1)
            prev_step = max(0, feature_step - self.velocity_lookback)
            dp_now = self.qpl_d_plus.iloc[feature_step].to_numpy(dtype=np.float32)
            dp_prev = self.qpl_d_plus.iloc[prev_step].to_numpy(dtype=np.float32)
            dm_now = self.qpl_d_minus.iloc[feature_step].to_numpy(dtype=np.float32)
            dm_prev = self.qpl_d_minus.iloc[prev_step].to_numpy(dtype=np.float32)
            dp_velocity = dp_now - dp_prev  # negative means approaching resistance
            dm_velocity = dm_now - dm_prev  # negative means approaching support
            finite_dpv = dp_velocity[np.isfinite(dp_velocity)]
            finite_dmv = dm_velocity[np.isfinite(dm_velocity)]
            min_dpv = float(finite_dpv.min()) if finite_dpv.size else 0.0
            min_dmv = float(finite_dmv.min()) if finite_dmv.size else 0.0
            velocity_aggregates = np.array(
                [
                    float(np.nanmean(dp_velocity)) if np.isfinite(dp_velocity).any() else 0.0,
                    float(np.nanmean(dm_velocity)) if np.isfinite(dm_velocity).any() else 0.0,
                    min_dpv,
                    min_dmv,
                ],
                dtype=np.float32,
            )
            parts.append(np.nan_to_num(velocity_aggregates, nan=0.0, posinf=0.0, neginf=0.0))
        observation = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = self.lookback_window
        self.previous_weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float32)
        self.last_weight_diff = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._get_observation(), {}

    def step(self, action):
        raw_weights = softmax(np.asarray(action, dtype=np.float32))
        current_drawdown_before_trade = max(0.0, 1.0 - self.portfolio_value / max(self.peak_value, EPS))
        gate_multipliers = np.ones(self.n_assets, dtype=np.float32)
        if self.use_qpl_gate:
            assert self.qpl_signal is not None
            signal_row = self.qpl_signal.iloc[self.current_step]
            gate_multipliers = signal_to_multiplier(signal_row, self.qpl_config).astype(np.float32)
            weights = apply_qpl_gate_to_weight_vector(raw_weights, signal_row, self.qpl_config)
        elif self.use_qpl_gate_v2 or self.use_qpl_gate_v3:
            assert self.qpl_d_plus is not None
            assert self.qpl_d_minus is not None
            assert self.qpl_signal is not None
            assert self.qpl_momentum is not None
            # Gate V2/V3 share the same downstream logic; V3 just feeds a
            # Lee-Oscillator-encoded chaotic momentum into the gate instead of
            # the linear pct_change momentum.
            momentum_row = self.qpl_momentum.iloc[self.current_step]
            momentum_fallback = momentum_row
            momentum_lookup = "qpl_momentum"
            if self.use_qpl_gate_v3:
                assert self.lee_momentum is not None
                momentum_row = self.lee_momentum.iloc[self.current_step]
                momentum_fallback = momentum_row
                momentum_lookup = "lee_momentum"
            qpl_row = {
                "qpl_d_plus": self._execution_row("qpl_d_plus", self.current_step, self.qpl_d_plus.iloc[self.current_step]),
                "qpl_d_minus": self._execution_row(
                    "qpl_d_minus",
                    self.current_step,
                    self.qpl_d_minus.iloc[self.current_step],
                ),
                "qpl_signal": self._execution_row("qpl_signal", self.current_step, self.qpl_signal.iloc[self.current_step]),
                "qpl_momentum": self._execution_row(momentum_lookup, self.current_step, momentum_fallback),
                "touch_plus_by_high": self._execution_row("touch_plus_by_high", self.current_step),
                "touch_minus_by_low": self._execution_row("touch_minus_by_low", self.current_step),
                "intraday_breakout": self._execution_row("intraday_breakout", self.current_step),
                "intraday_breakdown": self._execution_row("intraday_breakdown", self.current_step),
            }
            tech_row = None
            if self.use_technical_state:
                tech_row = {
                    name: frame.iloc[self.current_step]
                    for name, frame in self.technical_features.items()
                }
            gate_multipliers = compute_qpl_gate_scores(
                raw_weights,
                self.previous_weights,
                qpl_row,
                tech_feature_row=tech_row,
                portfolio_drawdown=current_drawdown_before_trade,
                qpl_config=self.qpl_gate_v2_config,
            ).astype(np.float32)
            weights = apply_qpl_gate_v2_to_weight_vector(
                raw_weights,
                self.previous_weights,
                qpl_row,
                tech_feature_row=tech_row,
                portfolio_drawdown=current_drawdown_before_trade,
                qpl_config=self.qpl_gate_v2_config,
            )
        else:
            weights = raw_weights.copy()

        returns_row = self.returns.iloc[self.current_step].to_numpy(dtype=np.float32)
        gross_return = float(weights @ returns_row)
        turnover = float(np.abs(weights - self.previous_weights).sum())
        transaction_cost = self.transaction_cost_rate * turnover
        net_return = gross_return - transaction_cost
        base_reward = float(np.log(max(1.0 + gross_return, EPS)))

        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = max(0.0, 1.0 - self.portfolio_value / max(self.peak_value, EPS))

        qpl_bonus = 0.0
        if self.use_qpl_reward:
            assert self.qpl_signal is not None
            signal = self.qpl_signal.iloc[self.current_step].to_numpy(dtype=float)
            direction = np.where(signal == 1, 1.0, np.where(signal < 0, -1.0, 0.0))
            qpl_bonus = float(np.sum(weights * direction * returns_row))

        lee_bonus = 0.0
        if self.use_lee_reward:
            assert self.lee_momentum is not None
            lee_score = self.lee_momentum.iloc[self.current_step].to_numpy(dtype=float)
            # Lee score is centred at 0 with magnitude up to ~0.5; normalise to roughly
            # +-1 so the reward sees a saturated directional signal.
            lee_direction = np.clip(lee_score / 0.5, -1.0, 1.0)
            lee_direction = np.nan_to_num(lee_direction, nan=0.0, posinf=0.0, neginf=0.0)
            lee_bonus = float(np.sum(weights * lee_direction * returns_row))

        whipsaw_penalty = 0.0
        current_weight_diff = weights.astype(np.float32) - self.previous_weights
        if self.use_whipsaw_penalty:
            # Penalise direction reversals: per asset, if current Δw and last Δw have
            # opposite signs, accumulate the magnitude product. Stationary periods
            # (one side ~0) contribute almost nothing.
            product = current_weight_diff * self.last_weight_diff
            whipsaw_penalty = float(np.sum(np.maximum(-product, 0.0)))

        reward = (
            base_reward
            - float(self.reward_config.get("lambda_tc", 1.0)) * transaction_cost
            + float(self.reward_config.get("lambda_qpl", 0.05)) * qpl_bonus
            + float(self.reward_config.get("lambda_lee", 0.30)) * lee_bonus
            - float(self.reward_config.get("lambda_whipsaw", 0.50)) * whipsaw_penalty
            - float(self.reward_config.get("lambda_dd", 0.1)) * current_drawdown
        )

        date = self.returns.index[self.current_step]
        self.last_weight_diff = current_weight_diff
        self.previous_weights = weights.astype(np.float32)
        self.current_step += 1
        terminated = self.current_step >= len(self.returns)
        truncated = False
        observation = self._get_observation()
        info = {
            "date": date,
            "raw_weights": raw_weights.copy(),
            "weights": weights.copy(),
            "gate_multipliers": gate_multipliers.copy(),
            "gross_return": gross_return,
            "daily_return": net_return,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "portfolio_value": self.portfolio_value,
            "drawdown": current_drawdown,
            "qpl_bonus": qpl_bonus,
            "lee_bonus": lee_bonus,
            "whipsaw_penalty": whipsaw_penalty,
            "base_reward": base_reward,
        }
        return observation, float(reward), terminated, truncated, info
