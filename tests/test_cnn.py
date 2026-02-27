"""Tests for CNN encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from packetcore.models.cnn import CNNEncoder


@pytest.fixture()
def windows():
    rng = np.random.default_rng(11)
    return rng.random((8, 16, 10)).astype(np.float32)  # (B=8, T=16, F=10)


def test_forward_shape(windows):
    model = CNNEncoder(input_dim=10, num_filters=[16, 32], latent_dim=8)
    x = torch.tensor(windows)
    z = model(x)
    assert z.shape == (8, 8)


def test_anomaly_score_shape(windows):
    model = CNNEncoder(input_dim=10, latent_dim=8)
    rng = np.random.default_rng(12)
    reference = rng.random(8).astype(np.float32)
    scores = model.anomaly_score(windows, reference)
    assert scores.shape == (8,)
    assert np.all(scores >= 0)
