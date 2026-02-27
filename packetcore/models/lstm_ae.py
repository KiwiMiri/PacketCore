"""LSTM Autoencoder for sequential anomaly detection.

Operates on window tensors of shape ``(batch, T, F)`` produced by
:class:`~packetcore.features.sequence.SequenceBuilder`.

Architecture
------------
* **Encoder**: multi-layer LSTM that compresses ``(T, F)`` → latent vector
* **Decoder**: multi-layer LSTM that reconstructs the input sequence from
  the latent vector, conditioned on the encoder's final hidden state

Anomaly score
-------------
Mean squared reconstruction error per window: ``mean((x - x_hat)^2)``
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LSTMAutoencoder(nn.Module):
    """LSTM-based sequence autoencoder.

    Parameters
    ----------
    input_dim:
        Number of features per time-step (``F``).
    hidden_dim:
        LSTM hidden state size.
    latent_dim:
        Size of the compressed representation produced by a linear
        bottleneck on top of the encoder's final hidden state.
    num_layers:
        Number of stacked LSTM layers in both encoder and decoder.
    dropout:
        Dropout probability between LSTM layers (applied when
        ``num_layers > 1``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Linear bottleneck
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder: project latent back to hidden_dim, then LSTM
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Output projection
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, F)`` → latent ``(B, latent_dim)``."""
        _, (h_n, _) = self.encoder_lstm(x)
        # h_n: (num_layers, B, hidden_dim) – take the last layer
        h_last = h_n[-1]  # (B, hidden_dim)
        return self.encoder_fc(h_last)  # (B, latent_dim)

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent ``(B, latent_dim)`` → reconstructed ``(B, T, F)``."""
        # Expand latent to (B, T, hidden_dim) as repeated input to decoder LSTM
        h0 = self.decoder_fc(z)  # (B, hidden_dim)
        decoder_input = h0.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, hidden_dim)
        out, _ = self.decoder_lstm(decoder_input)  # (B, T, hidden_dim)
        return self.output_fc(out)  # (B, T, F)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x_hat, z)`` where shapes are ``(B, T, F)`` and ``(B, latent_dim)``."""
        z = self.encode(x)
        x_hat = self.decode(z, seq_len=x.size(1))
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE averaged over time and features, shape ``(B,)``."""
        x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Numpy wrapper: ``X`` shape ``(B, T, F)`` → scores ``(B,)``."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            return self.reconstruction_error(x).cpu().numpy()
