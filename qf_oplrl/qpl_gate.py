from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qf_oplrl.qpl_strategy import signal_multipliers


def signal_to_multiplier(qpl_signal_row: pd.Series | np.ndarray, qpl_config: dict[str, Any] | None = None) -> np.ndarray:
    if isinstance(qpl_signal_row, pd.Series):
        signal_frame = pd.DataFrame([qpl_signal_row.to_numpy()], columns=qpl_signal_row.index)
    else:
        values = np.asarray(qpl_signal_row)
        signal_frame = pd.DataFrame([values], columns=range(values.size))
    return signal_multipliers(signal_frame.fillna(0).astype(int), qpl_config, {}).iloc[0].to_numpy(dtype=float)


def apply_qpl_gate_to_weight_vector(
    raw_weights: np.ndarray,
    qpl_signal_row: pd.Series | np.ndarray,
    qpl_config: dict[str, Any] | None = None,
) -> np.ndarray:
    raw = np.asarray(raw_weights, dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    multipliers = signal_to_multiplier(qpl_signal_row, qpl_config)
    gated = raw * multipliers
    total = gated.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(raw.size, 1.0 / raw.size, dtype=np.float32)
    return (gated / total).astype(np.float32)


def apply_qpl_gate_to_weights(
    raw_weights_df: pd.DataFrame,
    qpl_signal_df: pd.DataFrame,
    qpl_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    common_index = raw_weights_df.index.intersection(qpl_signal_df.index)
    common_columns = raw_weights_df.columns.intersection(qpl_signal_df.columns)
    raw = raw_weights_df.loc[common_index, common_columns].sort_index()
    signals = qpl_signal_df.loc[common_index, common_columns].sort_index().fillna(0).astype(int)
    gated = [
        apply_qpl_gate_to_weight_vector(raw.loc[index].to_numpy(dtype=float), signals.loc[index], qpl_config)
        for index in raw.index
    ]
    return pd.DataFrame(gated, index=raw.index, columns=raw.columns)

