"""Time-based train / validation / test split utilities.

All splits are performed on a **timestamp** column to respect temporal
ordering and avoid data leakage.  Random shuffling is never applied.

Split strategy
--------------
Given a DataFrame with a timestamp column:

* ``train``: records with ``timestamp < val_start``
* ``val``:   records with ``val_start <= timestamp < test_start``
* ``test``:  records with ``timestamp >= test_start``

The split boundaries can be supplied either as absolute timestamps or as
fractional positions within the dataset's time range.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    timestamp_col: str = "timestamp",
    val_start: Optional[float] = None,
    test_start: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into train / val / test using time ordering.

    Parameters
    ----------
    df:
        Input DataFrame.  Must contain *timestamp_col*.
    val_frac:
        Fraction of the time range to reserve for the validation split.
        Ignored when *val_start* is provided.
    test_frac:
        Fraction of the time range to reserve for the test split.
        Ignored when *test_start* is provided.
    timestamp_col:
        Name of the timestamp column (Unix seconds or any numeric ordering).
    val_start:
        Explicit start timestamp for the validation split.  If ``None``,
        computed from *val_frac*.
    test_start:
        Explicit start timestamp for the test split.  If ``None``,
        computed from *test_frac*.

    Returns
    -------
    train_df, val_df, test_df:
        Three DataFrames with non-overlapping, temporally ordered records.

    Raises
    ------
    ValueError
        If *timestamp_col* is not in *df*, or if the split boundaries
        result in an empty training set.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    df = df.sort_values(timestamp_col).reset_index(drop=True)
    t_min = float(df[timestamp_col].min())
    t_max = float(df[timestamp_col].max())
    t_range = t_max - t_min

    if t_range <= 0:
        raise ValueError("All timestamps are identical; cannot perform a time-based split.")

    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0.")

    # Compute boundaries
    if val_start is None:
        val_start = t_min + (1.0 - val_frac - test_frac) * t_range
    if test_start is None:
        test_start = t_min + (1.0 - test_frac) * t_range

    if not (t_min <= val_start < test_start <= t_max):
        raise ValueError(
            f"Invalid split boundaries: t_min={t_min}, val_start={val_start}, "
            f"test_start={test_start}, t_max={t_max}. Ensure t_min < val_start < test_start <= t_max."
        )

    ts = df[timestamp_col]
    train_df = df[ts < val_start].reset_index(drop=True)
    val_df = df[(ts >= val_start) & (ts < test_start)].reset_index(drop=True)
    test_df = df[ts >= test_start].reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError("Training split is empty after time-based split.")

    logger.info(
        "Time-based split: train=%d val=%d test=%d  (t_min=%.2f val_start=%.2f test_start=%.2f t_max=%.2f)",
        len(train_df),
        len(val_df),
        len(test_df),
        t_min,
        val_start,
        test_start,
        t_max,
    )

    return train_df, val_df, test_df
