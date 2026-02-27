"""Tests for IsolationForest and OneClassSVM wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from packetcore.models.isolation_forest import IsolationForestModel
from packetcore.models.ocsvm import OneClassSVMModel


@pytest.fixture()
def X_train():
    rng = np.random.default_rng(5)
    return rng.random((100, 8)).astype(np.float32)


@pytest.fixture()
def X_test():
    rng = np.random.default_rng(6)
    return rng.random((20, 8)).astype(np.float32)


class TestIsolationForest:
    def test_fit_and_score(self, X_train, X_test):
        model = IsolationForestModel(n_estimators=10)
        model.fit(X_train)
        scores = model.anomaly_score(X_test)
        assert scores.shape == (20,)

    def test_scores_are_finite(self, X_train, X_test):
        model = IsolationForestModel(n_estimators=10)
        model.fit(X_train)
        scores = model.anomaly_score(X_test)
        assert np.all(np.isfinite(scores))

    def test_predict_returns_plus_minus_one(self, X_train, X_test):
        model = IsolationForestModel(n_estimators=10)
        model.fit(X_train)
        preds = model.predict(X_test)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_unfitted_raises(self, X_test):
        model = IsolationForestModel()
        with pytest.raises(RuntimeError):
            model.anomaly_score(X_test)


class TestOneClassSVM:
    def test_fit_and_score(self, X_train, X_test):
        model = OneClassSVMModel()
        model.fit(X_train)
        scores = model.anomaly_score(X_test)
        assert scores.shape == (20,)

    def test_scores_are_finite(self, X_train, X_test):
        model = OneClassSVMModel()
        model.fit(X_train)
        scores = model.anomaly_score(X_test)
        assert np.all(np.isfinite(scores))

    def test_predict_returns_plus_minus_one(self, X_train, X_test):
        model = OneClassSVMModel()
        model.fit(X_train)
        preds = model.predict(X_test)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_unfitted_raises(self, X_test):
        model = OneClassSVMModel()
        with pytest.raises(RuntimeError):
            model.anomaly_score(X_test)
