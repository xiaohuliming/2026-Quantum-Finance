from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_by_time(
    frame: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> TimeSplits:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    n_rows = len(frame)
    if n_rows < 3:
        raise ValueError("Need at least three rows for train/validation/test split")

    train_end = max(1, int(n_rows * train_ratio))
    val_end = max(train_end + 1, int(n_rows * (train_ratio + val_ratio)))
    val_end = min(val_end, n_rows - 1)

    return TimeSplits(
        train=frame.iloc[:train_end].copy(),
        validation=frame.iloc[train_end:val_end].copy(),
        test=frame.iloc[val_end:].copy(),
    )

