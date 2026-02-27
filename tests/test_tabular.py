"""Tests for tabular feature extractor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packetcore.features.schema import DEFAULT_SCHEMA
from packetcore.features.tabular import TabularFeatureExtractor


def test_fit_transform_shape(sample_df):
    ext = TabularFeatureExtractor()
    X = ext.fit_transform(sample_df)
    assert X.ndim == 2
    assert X.shape[0] == len(sample_df)
    assert X.shape[1] == len(DEFAULT_SCHEMA.fields)


def test_transform_without_fit_raises(sample_df):
    ext = TabularFeatureExtractor()
    with pytest.raises(RuntimeError):
        ext.transform(sample_df)


def test_payload_gating_zeros_when_invisible(sample_df):
    """Payload-derived columns must be 0 where is_payload_visible=0."""
    ext = TabularFeatureExtractor()
    # Force all rows to invisible
    df = sample_df.copy()
    df["is_payload_visible"] = 0
    X = ext.fit_transform(df)

    gated_names = DEFAULT_SCHEMA.payload_gated_names()
    for name in gated_names:
        col_idx = DEFAULT_SCHEMA.index_of(name)
        # All values should be 0 before scaling shifts them; after scaling with
        # all-zero input the mean will be 0 so scaled values are also 0.
        assert np.allclose(X[:, col_idx], 0.0), f"Column '{name}' should be 0 when payload invisible"


def test_payload_gating_nonzero_when_visible(sample_df):
    """Payload-derived columns must carry signal when is_payload_visible=1."""
    ext = TabularFeatureExtractor()
    df = sample_df.copy()
    df["is_payload_visible"] = 1
    X = ext.fit_transform(df)
    gated_names = DEFAULT_SCHEMA.payload_gated_names()
    # At least one gated column should be non-zero
    gated_idxs = [DEFAULT_SCHEMA.index_of(n) for n in gated_names]
    assert not np.allclose(X[:, gated_idxs], 0.0)


def test_missing_column_fills_zero(sample_df):
    ext = TabularFeatureExtractor()
    df = sample_df.drop(columns=["pkt_len"])
    X = ext.fit_transform(df)
    assert X.shape[1] == len(DEFAULT_SCHEMA.fields)


def test_feature_names_length():
    ext = TabularFeatureExtractor()
    assert len(ext.feature_names()) == ext.n_features()


def test_fit_then_transform_consistent(sample_df):
    ext = TabularFeatureExtractor()
    X1 = ext.fit_transform(sample_df)
    X2 = ext.transform(sample_df)
    np.testing.assert_array_almost_equal(X1, X2)
