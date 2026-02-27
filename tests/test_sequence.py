"""Tests for SequenceBuilder."""

from __future__ import annotations

import numpy as np
import pytest

from packetcore.features.sequence import SequenceBuilder


@pytest.fixture()
def sample_X():
    rng = np.random.default_rng(1)
    return rng.random((100, 10)).astype(np.float32)


def test_output_shape_full_windows(sample_X):
    T, F = 8, sample_X.shape[1]
    builder = SequenceBuilder(window_size=T, stride=1, pad_partial=False)
    windows = builder.build(sample_X)
    expected_w = len(sample_X) - T + 1
    assert windows.shape == (expected_w, T, F)


def test_output_shape_padded(sample_X):
    T, F = 8, sample_X.shape[1]
    builder = SequenceBuilder(window_size=T, stride=1, pad_partial=True)
    windows = builder.build(sample_X)
    # With padding we get N windows
    assert windows.shape == (len(sample_X), T, F)


def test_stride_reduces_windows(sample_X):
    builder_s1 = SequenceBuilder(window_size=4, stride=1, pad_partial=False)
    builder_s4 = SequenceBuilder(window_size=4, stride=4, pad_partial=False)
    w1 = builder_s1.build(sample_X)
    w4 = builder_s4.build(sample_X)
    assert len(w4) < len(w1)


def test_empty_when_too_short():
    X = np.zeros((3, 5), dtype=np.float32)
    builder = SequenceBuilder(window_size=10, stride=1, pad_partial=False)
    windows = builder.build(X)
    assert windows.shape[0] == 0


def test_build_single_shape(sample_X):
    T = 8
    builder = SequenceBuilder(window_size=T, stride=1)
    window = builder.build_single(sample_X, idx=50)
    assert window.shape == (T, sample_X.shape[1])


def test_build_single_first_row_padded(sample_X):
    T = 8
    builder = SequenceBuilder(window_size=T, stride=1, pad_value=0.0)
    window = builder.build_single(sample_X, idx=0)
    # First T-1 rows should be padding (zeros)
    assert np.allclose(window[: T - 1], 0.0)
    assert np.allclose(window[T - 1], sample_X[0])


def test_invalid_window_size():
    with pytest.raises(ValueError):
        SequenceBuilder(window_size=0)


def test_invalid_stride():
    with pytest.raises(ValueError):
        SequenceBuilder(stride=0)
