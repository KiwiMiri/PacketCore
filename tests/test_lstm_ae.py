"""Tests for LSTMAutoencoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from packetcore.models.lstm_ae import LSTMAutoencoder


@pytest.fixture()
def model():
    return LSTMAutoencoder(input_dim=10, hidden_dim=32, latent_dim=16, num_layers=1)


@pytest.fixture()
def X_windows():
    rng = np.random.default_rng(4)
    return rng.random((16, 8, 10)).astype(np.float32)  # (B=16, T=8, F=10)


def test_forward_shapes(model, X_windows):
    x = torch.tensor(X_windows)
    x_hat, z = model(x)
    assert x_hat.shape == x.shape
    assert z.shape == (16, 16)


def test_encode_shape(model, X_windows):
    x = torch.tensor(X_windows)
    z = model.encode(x)
    assert z.shape == (16, 16)


def test_reconstruction_error_shape(model, X_windows):
    x = torch.tensor(X_windows)
    err = model.reconstruction_error(x)
    assert err.shape == (16,)
    assert (err >= 0).all()


def test_anomaly_score_numpy(model, X_windows):
    scores = model.anomaly_score(X_windows)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (16,)
    assert np.all(scores >= 0)


def test_decode_shape(model):
    z = torch.zeros(4, 16)
    x_hat = model.decode(z, seq_len=8)
    assert x_hat.shape == (4, 8, 10)
