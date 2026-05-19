from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from qf_oplrl.metrics import drawdown_series


def plot_cumulative_values(values: pd.DataFrame, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = values.dropna(how="any")
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    values.plot(ax=ax, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_drawdowns(values: pd.DataFrame, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = values.dropna(how="any")
    if values.empty:
        return
    drawdowns = values.apply(lambda column: drawdown_series(column.dropna()))
    fig, ax = plt.subplots(figsize=(11, 6))
    drawdowns.plot(ax=ax, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_weight_heatmap(weights: pd.DataFrame, title: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = weights.iloc[:: max(1, len(weights) // 300)]
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(sample.T, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Ticker")
    ax.set_yticks(range(len(sample.columns)))
    ax.set_yticklabels(sample.columns, fontsize=6)
    fig.colorbar(image, ax=ax, label="Weight")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_qpl_trigger_example(
    prices: pd.Series,
    qpl_plus: pd.Series,
    qpl_minus: pd.Series,
    signal: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [
            prices.rename("price"),
            qpl_plus.rename("qpl_plus"),
            qpl_minus.rename("qpl_minus"),
            signal.rename("signal"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=["price", "qpl_plus", "qpl_minus"])
    if frame.empty:
        return

    triggered = frame[frame["signal"].fillna(0) != 0]
    if not triggered.empty:
        center = triggered.index[len(triggered) // 2]
        position = frame.index.get_indexer([center], method="nearest")[0]
        start = max(0, position - 90)
        end = min(len(frame), position + 91)
        frame = frame.iloc[start:end]
        triggered = frame[frame["signal"].fillna(0) != 0]
    else:
        frame = frame.tail(min(180, len(frame)))
        triggered = frame.iloc[0:0]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(frame.index, frame["price"], label="Price", linewidth=1.6)
    ax.plot(frame.index, frame["qpl_plus"], label="QPL+", linewidth=1.1, linestyle="--")
    ax.plot(frame.index, frame["qpl_minus"], label="QPL-", linewidth=1.1, linestyle="--")
    if not triggered.empty:
        positive = triggered[triggered["signal"] == 1]
        negative = triggered[triggered["signal"] < 0]
        if not positive.empty:
            ax.scatter(positive.index, positive["price"], label="Support signal", s=28, marker="^")
        if not negative.empty:
            ax.scatter(negative.index, negative["price"], label="Risk signal", s=28, marker="v")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
