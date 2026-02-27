"""Tests for CDFCalibrator."""

from __future__ import annotations

import numpy as np
import pytest

from packetcore.ensemble.calibrator import CDFCalibrator


@pytest.fixture()
def val_scores():
    rng = np.random.default_rng(7)
    return rng.exponential(scale=1.0, size=500).astype(np.float32)


def test_fit_transform_range(val_scores):
    cal = CDFCalibrator()
    cal.fit(val_scores)
    calibrated = cal.transform(val_scores)
    assert calibrated.min() > 0
    assert calibrated.max() < 1


def test_fit_transform_shape(val_scores):
    cal = CDFCalibrator()
    calibrated = cal.fit_transform(val_scores)
    assert calibrated.shape == val_scores.shape


def test_high_score_maps_near_one(val_scores):
    cal = CDFCalibrator()
    cal.fit(val_scores)
    high_score = np.array([val_scores.max() * 10])
    assert cal.transform(high_score)[0] > 0.9


def test_low_score_maps_near_zero(val_scores):
    cal = CDFCalibrator()
    cal.fit(val_scores)
    low_score = np.array([val_scores.min() - 1.0])
    assert cal.transform(low_score)[0] < 0.1


def test_threshold_at_percentile(val_scores):
    cal = CDFCalibrator()
    cal.fit(val_scores)
    thr = cal.threshold_at_percentile(99.0)
    assert isinstance(thr, float)
    # ~99% of val_scores should be below this threshold
    below = (val_scores <= thr).mean()
    assert below >= 0.98


def test_unfitted_transform_raises():
    cal = CDFCalibrator()
    with pytest.raises(RuntimeError):
        cal.transform(np.array([1.0, 2.0]))


def test_unfitted_threshold_raises():
    cal = CDFCalibrator()
    with pytest.raises(RuntimeError):
        cal.threshold_at_percentile(95.0)


def test_empty_scores_raises():
    cal = CDFCalibrator()
    with pytest.raises(ValueError):
        cal.fit(np.array([]))
