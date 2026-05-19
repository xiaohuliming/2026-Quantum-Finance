from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(weights: pd.DataFrame) -> pd.DataFrame:
    clean = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    row_sums = clean.sum(axis=1)
    fallback = row_sums <= 0
    if fallback.any():
        clean.loc[fallback, :] = 1.0 / clean.shape[1]
        row_sums = clean.sum(axis=1)
    return clean.div(row_sums, axis=0)


def run_backtest(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    initial_capital: float = 1.0,
    transaction_cost_rate: float = 0.001,
) -> dict[str, pd.Series | pd.DataFrame]:
    common_index = returns.index.intersection(weights.index)
    common_columns = returns.columns.intersection(weights.columns)
    if len(common_index) == 0 or len(common_columns) == 0:
        raise ValueError("Returns and weights do not share dates and tickers")

    aligned_returns = returns.loc[common_index, common_columns].sort_index()
    aligned_weights = normalize_weights(weights.loc[common_index, common_columns].sort_index())

    portfolio_return = (aligned_weights * aligned_returns).sum(axis=1)
    previous_weights = aligned_weights.shift(1).fillna(0.0)
    turnover = (aligned_weights - previous_weights).abs().sum(axis=1)
    transaction_cost = transaction_cost_rate * turnover
    daily_return = portfolio_return - transaction_cost
    portfolio_value = initial_capital * (1.0 + daily_return).cumprod()

    return {
        "portfolio_value": portfolio_value,
        "daily_return": daily_return,
        "gross_return": portfolio_return,
        "weights": aligned_weights,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
    }

