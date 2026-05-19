from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def max_drawdown(portfolio_value: pd.Series) -> float:
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1.0
    return float(drawdown.min())


def drawdown_series(portfolio_value: pd.Series) -> pd.Series:
    return portfolio_value / portfolio_value.cummax() - 1.0


def compute_metrics(
    backtest_result: dict[str, Any],
    initial_capital: float = 1.0,
    annualization_factor: int = TRADING_DAYS,
) -> dict[str, float]:
    value = pd.Series(backtest_result["portfolio_value"]).dropna()
    daily_return = pd.Series(backtest_result["daily_return"]).dropna()
    turnover = pd.Series(backtest_result["turnover"]).dropna()
    transaction_cost = pd.Series(backtest_result["transaction_cost"]).dropna()

    if value.empty or daily_return.empty:
        raise ValueError("Backtest result is empty")

    final_value = float(value.iloc[-1])
    periods = len(daily_return)
    cumulative_return = final_value / initial_capital - 1.0
    annualized_return = (final_value / initial_capital) ** (annualization_factor / periods) - 1.0
    annualized_volatility = float(daily_return.std(ddof=1) * math.sqrt(annualization_factor))
    sharpe = safe_divide(float(daily_return.mean() * annualization_factor), annualized_volatility)

    downside = daily_return[daily_return < 0]
    downside_vol = float(downside.std(ddof=1) * math.sqrt(annualization_factor)) if len(downside) > 1 else np.nan
    sortino = safe_divide(float(daily_return.mean() * annualization_factor), downside_vol)

    mdd = max_drawdown(value)
    calmar = safe_divide(annualized_return, abs(mdd))
    win_rate = float((daily_return > 0).mean())

    return {
        "Final Portfolio Value": final_value,
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": mdd,
        "Calmar Ratio": calmar,
        "Average Turnover": float(turnover.mean()),
        "Total Transaction Cost": float(transaction_cost.sum()),
        "Win Rate": win_rate,
    }

