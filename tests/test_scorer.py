"""Tests for ProtocolConditionalScorer."""

from __future__ import annotations

import numpy as np
import pytest

from packetcore.ensemble.scorer import (
    ProtocolConditionalScorer,
    compute_protocol_thresholds,
)


@pytest.fixture()
def val_data():
    rng = np.random.default_rng(8)
    n = 300
    scores = rng.uniform(0, 1, n).astype(np.float32)
    protocol_ids = rng.choice([6, 17, 502], n).astype(np.int32)
    return scores, protocol_ids


def test_compute_thresholds_keys(val_data):
    scores, protocol_ids = val_data
    thresholds = compute_protocol_thresholds(scores, protocol_ids, target_fpr=0.05)
    expected_protocols = set(np.unique(protocol_ids).tolist())
    assert set(thresholds.keys()) == expected_protocols


def test_compute_thresholds_values_in_range(val_data):
    scores, protocol_ids = val_data
    thresholds = compute_protocol_thresholds(scores, protocol_ids, target_fpr=0.05)
    for v in thresholds.values():
        assert 0.0 <= v <= 1.0


def test_scorer_predict_shape(val_data):
    scores, protocol_ids = val_data
    scorer = ProtocolConditionalScorer()
    scorer.fit(scores, protocol_ids, target_fpr=0.05)
    labels = scorer.predict(scores, protocol_ids)
    assert labels.shape == (len(scores),)
    assert set(np.unique(labels)).issubset({0, 1})


def test_scorer_default_threshold_used_for_unknown_protocol(val_data):
    scores, _ = val_data
    scorer = ProtocolConditionalScorer(thresholds={}, default_threshold=0.5)
    unknown_ids = np.full(len(scores), 999, dtype=np.int32)
    labels = scorer.predict(scores, unknown_ids)
    expected = (scores > 0.5).astype(np.int32)
    np.testing.assert_array_equal(labels, expected)


def test_threshold_for_known_protocol(val_data):
    scores, protocol_ids = val_data
    scorer = ProtocolConditionalScorer()
    scorer.fit(scores, protocol_ids, target_fpr=0.01)
    thr = scorer.threshold_for(6)
    assert isinstance(thr, float)


def test_mismatched_lengths_raises():
    scorer = ProtocolConditionalScorer()
    with pytest.raises(ValueError):
        compute_protocol_thresholds(
            np.array([0.1, 0.2, 0.3]),
            np.array([6, 17]),
        )
