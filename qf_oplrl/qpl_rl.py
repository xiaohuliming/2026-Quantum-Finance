from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from qf_oplrl.metrics import compute_metrics
from qf_oplrl.qpl import lag_qpl_package_for_returns
from qf_oplrl.qpl_rl_env import QPLPortfolioEnv
from qf_oplrl.splits import split_by_time
from qf_oplrl.technical_indicators import build_lagged_technical_features


QPL_VARIANTS = [
    {
        "key": "plain_ppo_reproduced",
        "method": "Plain PPO Reproduced",
        "use_technical_state": False,
        "use_qpl_state": False,
        "use_qpl_gate": False,
        "use_qpl_gate_v2": False,
        "use_qpl_reward": False,
    },
    {
        "key": "plain_ppo_tech_state",
        "method": "Plain PPO + Tech State",
        "use_technical_state": True,
        "use_qpl_state": False,
        "use_qpl_gate": False,
        "use_qpl_gate_v2": False,
        "use_qpl_reward": False,
    },
    {
        "key": "ppo_qpl_state",
        "method": "PPO + QPL State",
        "use_technical_state": False,
        "use_qpl_state": True,
        "use_qpl_gate": False,
        "use_qpl_gate_v2": False,
        "use_qpl_reward": False,
    },
    {
        "key": "ppo_qpl_gate",
        "method": "PPO + QPL Gate V1",
        "use_technical_state": False,
        "use_qpl_state": False,
        "use_qpl_gate": True,
        "use_qpl_gate_v2": False,
        "use_qpl_reward": False,
    },
    {
        "key": "ppo_qpl_gate_v2",
        "method": "PPO + QPL Gate V2",
        "use_technical_state": False,
        "use_qpl_state": False,
        "use_qpl_gate": False,
        "use_qpl_gate_v2": True,
        "use_qpl_reward": False,
    },
    {
        "key": "ppo_qpl_state_gate",
        "method": "PPO + QPL State + Gate V1",
        "use_technical_state": False,
        "use_qpl_state": True,
        "use_qpl_gate": True,
        "use_qpl_gate_v2": False,
        "use_qpl_reward": False,
    },
    {
        "key": "ppo_qpl_state_gate_v2",
        "method": "PPO + QPL State + Gate V2",
        "use_technical_state": True,
        "use_qpl_state": True,
        "use_qpl_gate": False,
        "use_qpl_gate_v2": True,
        "use_qpl_reward": False,
    },
    {
        "key": "full_qf_oplrl",
        "method": "Full QF-OPLRL V1",
        "use_technical_state": False,
        "use_qpl_state": True,
        "use_qpl_gate": True,
        "use_qpl_gate_v2": False,
        "use_qpl_reward": True,
    },
    {
        "key": "full_qf_oplrl_v2",
        "method": "Full QF-OPLRL V2",
        "use_technical_state": True,
        "use_qpl_state": True,
        "use_qpl_gate": False,
        "use_qpl_gate_v2": True,
        "use_qpl_reward": True,
    },
]


def _build_env(
    returns: pd.DataFrame,
    qpl_features: dict[str, pd.DataFrame],
    qpl_config: dict[str, Any],
    qpl_gate_v2_config: dict[str, Any],
    rl_config: dict[str, Any],
    variant: dict[str, Any],
    technical_features: dict[str, pd.DataFrame] | None = None,
    technical_feature_names: list[str] | None = None,
    qpl_execution_features: dict[str, pd.DataFrame] | None = None,
) -> QPLPortfolioEnv:
    return QPLPortfolioEnv(
        returns,
        qpl_features=qpl_features,
        lookback_window=int(rl_config.get("lookback_window", 20)),
        transaction_cost_rate=float(rl_config.get("transaction_cost_rate", 0.001)),
        use_technical_state=bool(variant.get("use_technical_state", False)),
        use_qpl_state=bool(variant["use_qpl_state"]),
        use_qpl_gate=bool(variant["use_qpl_gate"]),
        use_qpl_gate_v2=bool(variant.get("use_qpl_gate_v2", False)),
        use_qpl_reward=bool(variant["use_qpl_reward"]),
        technical_features=technical_features,
        technical_feature_names=technical_feature_names,
        qpl_execution_features=qpl_execution_features,
        qpl_config=qpl_config,
        qpl_gate_v2_config=qpl_gate_v2_config,
        reward_config=rl_config,
    )


def train_qpl_ppo(
    train_returns: pd.DataFrame,
    qpl_features: dict[str, pd.DataFrame],
    qpl_config: dict[str, Any],
    qpl_gate_v2_config: dict[str, Any],
    rl_config: dict[str, Any],
    variant: dict[str, Any],
    technical_features: dict[str, pd.DataFrame] | None = None,
    technical_feature_names: list[str] | None = None,
    qpl_execution_features: dict[str, pd.DataFrame] | None = None,
):
    if str(rl_config.get("algorithm", "PPO")).upper() != "PPO":
        raise ValueError("Only PPO is currently supported for QPL RL")
    from stable_baselines3 import PPO

    env = _build_env(
        train_returns,
        qpl_features,
        qpl_config,
        qpl_gate_v2_config,
        rl_config,
        variant,
        technical_features=technical_features,
        technical_feature_names=technical_feature_names,
        qpl_execution_features=qpl_execution_features,
    )
    model = PPO(
        "MlpPolicy",
        env,
        seed=int(rl_config.get("seed", 42)),
        verbose=0,
        n_steps=min(128, max(16, len(env.returns) // 2)),
        batch_size=64,
    )
    model.learn(total_timesteps=int(rl_config.get("total_timesteps", 5000)))
    return model


def evaluate_qpl_model(
    model,
    test_returns: pd.DataFrame,
    qpl_features: dict[str, pd.DataFrame],
    qpl_config: dict[str, Any],
    qpl_gate_v2_config: dict[str, Any],
    rl_config: dict[str, Any],
    variant: dict[str, Any],
    technical_features: dict[str, pd.DataFrame] | None = None,
    technical_feature_names: list[str] | None = None,
    qpl_execution_features: dict[str, pd.DataFrame] | None = None,
) -> dict:
    env = _build_env(
        test_returns,
        qpl_features,
        qpl_config,
        qpl_gate_v2_config,
        rl_config,
        variant,
        technical_features=technical_features,
        technical_feature_names=technical_feature_names,
        qpl_execution_features=qpl_execution_features,
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
    weights = pd.DataFrame([record["weights"] for record in records], index=index, columns=env.tickers)
    raw_weights = pd.DataFrame([record["raw_weights"] for record in records], index=index, columns=env.tickers)
    gate_multipliers = pd.DataFrame(
        [record["gate_multipliers"] for record in records],
        index=index,
        columns=env.tickers,
    )
    portfolio_value = pd.Series([record["portfolio_value"] for record in records], index=index, name=variant["method"])
    daily_return = pd.Series([record["daily_return"] for record in records], index=index)
    turnover = pd.Series([record["turnover"] for record in records], index=index)
    transaction_cost = pd.Series([record["transaction_cost"] for record in records], index=index)
    return {
        "portfolio_value": portfolio_value,
        "daily_return": daily_return,
        "weights": weights,
        "raw_weights": raw_weights,
        "gate_multipliers": gate_multipliers,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
    }


def run_qpl_ablation_for_dataset(
    data,
    qpl_package: dict[str, pd.DataFrame],
    config: dict[str, Any],
    output_dir: Path,
    variants: list[str] | None = None,
) -> pd.DataFrame:
    qpl_config = config.get("qpl", {})
    qpl_gate_v2_config = {**config.get("qpl_gate_v2", {}), **config.get("gate", {})}
    rl_config = {**config.get("qpl_rl", {}), **config.get("reward", {})}
    technical_config = config.get("technical_indicators", {})
    backtest_config = config.get("backtest", {})
    split = split_by_time(data.returns, **config.get("split", {}))
    train_returns = split.train
    test_returns = split.test
    lagged_features = lag_qpl_package_for_returns(qpl_package, data.returns.index)
    execution_features = {
        name: frame.reindex(data.returns.index)
        for name, frame in qpl_package.items()
        if isinstance(frame, pd.DataFrame)
    }
    technical_features = None
    technical_feature_names = technical_config.get("feature_names")
    if bool(technical_config.get("enabled", False)):
        technical_features = build_lagged_technical_features(
            data.prices,
            data.returns.index,
            technical_config,
        )

    selected = QPL_VARIANTS
    if variants:
        wanted = set(variants)
        selected = [variant for variant in QPL_VARIANTS if variant["key"] in wanted or variant["method"] in wanted]
        if not selected:
            raise ValueError(f"No QPL RL variants matched: {variants}")

    dataset_dir = output_dir / data.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in selected:
        variant_dir = dataset_dir / variant["key"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        model = train_qpl_ppo(
            train_returns,
            lagged_features,
            qpl_config,
            qpl_gate_v2_config,
            rl_config,
            variant,
            technical_features=technical_features,
            technical_feature_names=technical_feature_names,
            qpl_execution_features=execution_features,
        )
        result = evaluate_qpl_model(
            model,
            test_returns,
            lagged_features,
            qpl_config,
            qpl_gate_v2_config,
            rl_config,
            variant,
            technical_features=technical_features,
            technical_feature_names=technical_feature_names,
            qpl_execution_features=execution_features,
        )

        model.save(variant_dir / "model.zip")
        result["weights"].to_csv(variant_dir / "test_weights.csv")
        result["raw_weights"].to_csv(variant_dir / "test_raw_weights.csv")
        result["gate_multipliers"].to_csv(variant_dir / "test_gate_multipliers.csv")
        result["portfolio_value"].to_csv(variant_dir / "test_portfolio_value.csv")

        metrics = compute_metrics(
            result,
            initial_capital=float(backtest_config.get("initial_capital", 1.0)),
            annualization_factor=int(backtest_config.get("annualization_factor", 252)),
        )
        row = {
            "Dataset": data.dataset_name,
            "Method": variant["method"],
            "Method Type": "QPL RL",
            "Variant Key": variant["key"],
            "Use Technical State": variant.get("use_technical_state", False),
            "Use QPL State": variant["use_qpl_state"],
            "Use QPL Gate": variant["use_qpl_gate"],
            "Use QPL Gate V2": variant.get("use_qpl_gate_v2", False),
            "Use QPL Reward": variant["use_qpl_reward"],
            **metrics,
        }
        pd.DataFrame([row]).to_csv(variant_dir / "metrics.csv", index=False)
        rows.append(row)

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(dataset_dir / "qpl_ablation_metrics.csv", index=False)
    return metrics_frame
