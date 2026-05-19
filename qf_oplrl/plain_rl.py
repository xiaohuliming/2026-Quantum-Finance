from __future__ import annotations

from pathlib import Path

import pandas as pd

from qf_oplrl.metrics import compute_metrics
from qf_oplrl.plain_rl_env import PlainPortfolioEnv
from qf_oplrl.splits import split_by_time


def train_ppo(train_returns: pd.DataFrame, rl_config: dict):
    from stable_baselines3 import PPO

    env = PlainPortfolioEnv(
        train_returns,
        lookback_window=int(rl_config.get("lookback_window", 20)),
        transaction_cost_rate=float(rl_config.get("transaction_cost_rate", 0.001)),
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=int(rl_config.get("seed", 42)),
        verbose=0,
        n_steps=min(128, max(16, len(train_returns) // 2)),
        batch_size=64,
    )
    model.learn(total_timesteps=int(rl_config.get("total_timesteps", 5000)))
    return model


def evaluate_model(model, test_returns: pd.DataFrame, rl_config: dict) -> dict:
    env = PlainPortfolioEnv(
        test_returns,
        lookback_window=int(rl_config.get("lookback_window", 20)),
        transaction_cost_rate=float(rl_config.get("transaction_cost_rate", 0.001)),
    )
    observation, _ = env.reset()
    records = []
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        record = dict(info)
        record["reward"] = reward
        records.append(record)
        done = terminated or truncated

    index = pd.to_datetime([record["date"] for record in records])
    weights = pd.DataFrame(
        [record["weights"] for record in records],
        index=index,
        columns=test_returns.columns,
    )
    portfolio_value = pd.Series(
        [record["portfolio_value"] for record in records],
        index=index,
        name="Plain PPO",
    )
    daily_return = pd.Series([record["daily_return"] for record in records], index=index)
    turnover = pd.Series([record["turnover"] for record in records], index=index)
    transaction_cost = pd.Series([record["transaction_cost"] for record in records], index=index)
    return {
        "portfolio_value": portfolio_value,
        "daily_return": daily_return,
        "weights": weights,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
    }


def run_plain_rl_for_dataset(data, config: dict, output_dir: Path) -> pd.DataFrame:
    rl_config = config.get("plain_rl", {})
    split = split_by_time(data.returns, **config.get("split", {}))
    train_returns = split.train
    test_returns = split.test

    model = train_ppo(train_returns, rl_config)
    result = evaluate_model(model, test_returns, rl_config)

    dataset_dir = output_dir / data.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model.save(dataset_dir / "model.zip")
    result["weights"].to_csv(dataset_dir / "test_weights.csv")
    result["portfolio_value"].to_csv(dataset_dir / "test_portfolio_value.csv")

    metrics = compute_metrics(
        result,
        initial_capital=float(config.get("backtest", {}).get("initial_capital", 1.0)),
        annualization_factor=int(config.get("backtest", {}).get("annualization_factor", 252)),
    )
    metrics_row = {
        "Dataset": data.dataset_name,
        "Method": "Plain PPO",
        "Method Type": "RL",
        **metrics,
    }
    metrics_frame = pd.DataFrame([metrics_row])
    metrics_frame.to_csv(dataset_dir / "metrics.csv", index=False)
    return metrics_frame

