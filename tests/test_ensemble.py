"""Tests for WeightedEnsemble."""

from __future__ import annotations

import numpy as np
import pytest

from packetcore.ensemble.ensemble import WeightedEnsemble


MODELS = ["ae", "lstm_ae", "iforest", "ocsvm"]


@pytest.fixture()
def score_dict():
    rng = np.random.default_rng(9)
    n = 100
    return {name: rng.uniform(0, 1, n).astype(np.float32) for name in MODELS}


def test_uniform_weights_sum_to_one():
    ens = WeightedEnsemble(model_names=MODELS)
    assert abs(ens.weights.sum() - 1.0) < 1e-6


def test_custom_weights_are_normalised():
    ens = WeightedEnsemble(model_names=MODELS, weights=[1.0, 2.0, 1.0, 0.0])
    assert abs(ens.weights.sum() - 1.0) < 1e-6


def test_combine_shape(score_dict):
    ens = WeightedEnsemble(model_names=MODELS)
    combined = ens.combine(score_dict)
    assert combined.shape == (100,)


def test_combine_range(score_dict):
    ens = WeightedEnsemble(model_names=MODELS)
    combined = ens.combine(score_dict)
    assert combined.min() >= 0.0
    assert combined.max() <= 1.0


def test_combine_missing_model_logs_warning(score_dict, caplog):
    import logging
    ens = WeightedEnsemble(model_names=MODELS)
    partial = {k: v for k, v in score_dict.items() if k != "ocsvm"}
    with caplog.at_level(logging.WARNING, logger="packetcore.ensemble.ensemble"):
        combined = ens.combine(partial)
    assert combined.shape == (100,)
    assert any("ocsvm" in r.message for r in caplog.records)


def test_combine_all_missing_raises():
    ens = WeightedEnsemble(model_names=MODELS)
    with pytest.raises(ValueError):
        ens.combine({"unknown_model": np.zeros(10)})


def test_empty_model_names_raises():
    with pytest.raises(ValueError):
        WeightedEnsemble(model_names=[])


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        WeightedEnsemble(model_names=["a", "b"], weights=[-1.0, 2.0])


def test_all_zero_weights_raise():
    with pytest.raises(ValueError):
        WeightedEnsemble(model_names=["a", "b"], weights=[0.0, 0.0])


def test_fit_weights_from_auroc(score_dict):
    rng = np.random.default_rng(10)
    labels = rng.integers(0, 2, 100).astype(np.int32)
    ens = WeightedEnsemble(model_names=MODELS)
    ens.fit_weights_from_auroc(score_dict, labels)
    assert abs(ens.weights.sum() - 1.0) < 1e-6


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        WeightedEnsemble(model_names=MODELS, weights=[1.0, 1.0])
