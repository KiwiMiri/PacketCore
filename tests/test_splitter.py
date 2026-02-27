"""Tests for time-based splitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packetcore.data.splitter import time_based_split


@pytest.fixture()
def df():
    n = 1000
    t0 = 1_700_000_000.0
    return pd.DataFrame(
        {
            "timestamp": np.linspace(t0, t0 + 3600, n),
            "value": np.arange(n, dtype=np.float32),
        }
    )


def test_split_sizes_sum_to_total(df):
    train, val, test = time_based_split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_no_overlap(df):
    train, val, test = time_based_split(df)
    train_ts = set(train["timestamp"].tolist())
    val_ts = set(val["timestamp"].tolist())
    test_ts = set(test["timestamp"].tolist())
    assert train_ts.isdisjoint(val_ts)
    assert train_ts.isdisjoint(test_ts)
    assert val_ts.isdisjoint(test_ts)


def test_temporal_ordering(df):
    train, val, test = time_based_split(df)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_missing_timestamp_raises(df):
    with pytest.raises(ValueError):
        time_based_split(df, timestamp_col="nonexistent")


def test_all_same_timestamps_raises():
    df = pd.DataFrame({"timestamp": [1.0] * 100, "v": range(100)})
    with pytest.raises(ValueError):
        time_based_split(df)


def test_invalid_fracs_raise(df):
    with pytest.raises(ValueError):
        time_based_split(df, val_frac=0.5, test_frac=0.6)


def test_explicit_boundaries(df):
    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    t_range = t_max - t_min
    val_start = t_min + 0.7 * t_range
    test_start = t_min + 0.85 * t_range
    train, val, test = time_based_split(df, val_start=val_start, test_start=test_start)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0


def test_fraction_fracs_respected(df):
    train, val, test = time_based_split(df, val_frac=0.15, test_frac=0.15)
    total = len(df)
    # Train should be roughly 70% of the data
    assert 0.6 * total < len(train) < 0.8 * total
