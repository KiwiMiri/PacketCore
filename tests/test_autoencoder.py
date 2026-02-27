"""Tests for Autoencoder and VariationalAutoencoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from packetcore.models.autoencoder import Autoencoder, VariationalAutoencoder


@pytest.fixture()
def ae():
    return Autoencoder(input_dim=16, hidden_dims=[32, 16], latent_dim=8)


@pytest.fixture()
def vae():
    return VariationalAutoencoder(input_dim=16, hidden_dims=[32, 16], latent_dim=8, beta=1.0)


@pytest.fixture()
def X_np():
    rng = np.random.default_rng(3)
    return rng.random((32, 16)).astype(np.float32)


class TestAutoencoder:
    def test_forward_shapes(self, ae, X_np):
        x = torch.tensor(X_np)
        x_hat, z = ae(x)
        assert x_hat.shape == x.shape
        assert z.shape == (32, 8)

    def test_encode_shape(self, ae, X_np):
        x = torch.tensor(X_np)
        z = ae.encode(x)
        assert z.shape == (32, 8)

    def test_reconstruction_error_shape(self, ae, X_np):
        x = torch.tensor(X_np)
        err = ae.reconstruction_error(x)
        assert err.shape == (32,)

    def test_anomaly_score_numpy(self, ae, X_np):
        scores = ae.anomaly_score(X_np)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (32,)
        assert np.all(scores >= 0)

    def test_reconstruction_error_nonnegative(self, ae, X_np):
        x = torch.tensor(X_np)
        err = ae.reconstruction_error(x)
        assert (err >= 0).all()


class TestVariationalAutoencoder:
    def test_forward_shapes(self, vae, X_np):
        x = torch.tensor(X_np)
        x_hat, z, mu, log_var = vae(x)
        assert x_hat.shape == x.shape
        assert z.shape == (32, 8)
        assert mu.shape == (32, 8)
        assert log_var.shape == (32, 8)

    def test_loss_tuple(self, vae, X_np):
        vae.train()
        x = torch.tensor(X_np)
        total, recon, kl = vae.loss(x)
        assert total.item() > 0

    def test_anomaly_score_numpy(self, vae, X_np):
        scores = vae.anomaly_score(X_np)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (32,)

    def test_eval_mode_uses_mean(self, vae, X_np):
        """In eval mode reparameterise should return mu (no noise)."""
        vae.eval()
        x = torch.tensor(X_np)
        mu1, log_var1 = vae.encode(x)
        z1 = vae.reparameterise(mu1, log_var1)
        z2 = vae.reparameterise(mu1, log_var1)
        torch.testing.assert_close(z1, z2)
