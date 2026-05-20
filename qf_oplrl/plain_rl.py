from __future__ import annotations

from pathlib import Path

import pandas as pd

from qf_oplrl.metrics import compute_metrics
from qf_oplrl.plain_rl_env import PlainPortfolioEnv
from qf_oplrl.splits import split_by_time
from qf_oplrl.technical_indicators import build_lagged_technical_features


def train_ppo(
    train_returns: pd.DataFrame,
    rl_config: dict,
    technical_features: dict[str, pd.DataFrame] | None = None,
    use_technical_state: bool = False,
    technical_feature_names: list[str] | None = None,
):
    from stable_baselines3 import PPO

    env = PlainPortfolioEnv(
        train_returns,
        lookback_window=int(rl_config.get("lookback_window", 20)),
        transaction_cost_rate=float(rl_config.get("transaction_cost_rate", 0.001)),
        use_technical_state=use_technical_state,
        technical_features=technical_features,
        technical_feature_names=technical_feature_names,
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


def evaluate_model(
    model,
    test_returns: pd.DataFrame,
    rl_config: dict,
    technical_features: dict[str, pd.DataFrame] | None = None,
    use_technical_state: bool = False,
    technical_feature_names: list[str] | None = None,
    method_name: str = "Plain PPO",
) -> dict:
    env = PlainPortfolioEnv(
        test_returns,
        lookback_window=int(rl_config.get("lookback_window", 20)),
        transaction_cost_rate=float(rl_config.get("transaction_cost_rate", 0.001)),
        use_technical_state=use_technical_state,
        technical_features=technical_features,
        technical_feature_names=technical_feature_names,
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
        columns=env.tickers,
    )
    portfolio_value = pd.Series(
        [record["portfolio_value"] for record in records],
        index=index,
        name=method_name,
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


def _save_plain_result(dataset_dir: Path, model, result: dict, metrics_row: dict, subdir: str | None = None) -> None:
    target_dir = dataset_dir if subdir is None else dataset_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save(target_dir / "model.zip")
    result["weights"].to_csv(target_dir / "test_weights.csv")
    result["portfolio_value"].to_csv(target_dir / "test_portfolio_value.csv")
    metrics_name = "metrics.csv" if subdir is None else f"{subdir}_metrics.csv"
    pd.DataFrame([metrics_row]).to_csv(dataset_dir / metrics_name, index=False)


def run_plain_rl_for_dataset(data, config: dict, output_dir: Path) -> pd.DataFrame:
    rl_config = config.get("plain_rl", {})
    technical_config = config.get("technical_indicators", {})
    technical_enabled = bool(technical_config.get("enabled", False))
    split = split_by_time(data.returns, **config.get("split", {}))
    train_returns = split.train
    test_returns = split.test
    technical_features = None
    technical_feature_names = technical_config.get("feature_names")
    if technical_enabled:
        technical_features = build_lagged_technical_features(
            data.prices,
            data.returns.index,
            technical_config,
        )

    dataset_dir = output_dir / data.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for method_name, use_tech, subdir in [
        ("Plain PPO", False, None),
        ("Plain PPO + Tech State", True, "tech_state"),
    ]:
        if use_tech and not technical_enabled:
            continue
        model = train_ppo(
            train_returns,
            rl_config,
            technical_features=technical_features,
            use_technical_state=use_tech,
            technical_feature_names=technical_feature_names,
        )
        result = evaluate_model(
            model,
            test_returns,
            rl_config,
            technical_features=technical_features,
            use_technical_state=use_tech,
            technical_feature_names=technical_feature_names,
            method_name=method_name,
        )
        metrics = compute_metrics(
            result,
            initial_capital=float(config.get("backtest", {}).get("initial_capital", 1.0)),
            annualization_factor=int(config.get("backtest", {}).get("annualization_factor", 252)),
        )
        metrics_row = {
            "Dataset": data.dataset_name,
            "Method": method_name,
            "Method Type": "RL",
            "Use Technical State": use_tech,
            **metrics,
        }
        _save_plain_result(dataset_dir, model, result, metrics_row, subdir=subdir)
        rows.append(metrics_row)

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(dataset_dir / "all_metrics.csv", index=False)
    return metrics_frame
