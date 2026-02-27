"""Autoencoder (AE) and Variational Autoencoder (VAE) for tabular features.

Both models are implemented as plain ``torch.nn.Module`` subclasses and share
a common ``reconstruction_error`` method so that they can be used
interchangeably as anomaly scorers.

Anomaly score
-------------
The anomaly score is the mean squared reconstruction error per sample:

    score(x) = mean((x - x_hat)^2)

For VAE an additional KL-divergence term can optionally be included.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _build_mlp(layer_dims: List[int], activation: nn.Module = nn.ReLU()) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(activation)
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Vanilla Autoencoder for tabular anomaly detection.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dims:
        List of hidden layer widths for the encoder.  The decoder mirrors
        this in reverse.
    latent_dim:
        Dimensionality of the bottleneck representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 16,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder
        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        self.encoder = _build_mlp(enc_dims)

        # Decoder
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.decoder = _build_mlp(dec_dims)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_hat, z)``."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample MSE reconstruction error, shape ``(N,)``."""
        x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Numpy wrapper returning per-sample anomaly scores."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            return self.reconstruction_error(x).cpu().numpy()


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE) for tabular anomaly detection.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dims:
        List of hidden layer widths for the encoder/decoder.
    latent_dim:
        Dimensionality of the latent Gaussian.
    beta:
        Weight for the KL-divergence term (``beta-VAE``).  ``beta=1``
        recovers the standard ELBO.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 16,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder body
        enc_dims = [input_dim] + hidden_dims
        self.encoder_body = _build_mlp(enc_dims + [hidden_dims[-1]], activation=nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.decoder = _build_mlp(dec_dims)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(x_hat, z, mu, log_var)``."""
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, z, mu, log_var

    def loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss.

        Returns
        -------
        total_loss, recon_loss, kl_loss
        """
        x_hat, _, mu, log_var = self.forward(x)
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + self.beta * kl, recon, kl

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error, shape ``(N,)``."""
        x_hat, _, _, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Numpy wrapper returning per-sample anomaly scores."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            return self.reconstruction_error(x).cpu().numpy()
