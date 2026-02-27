"""Optional 1-D CNN encoder for sequence inputs.

Processes window tensors of shape ``(B, T, F)`` and produces a latent
vector of shape ``(B, latent_dim)``.  Can be used as a drop-in replacement
for (or in combination with) :class:`~packetcore.models.lstm_ae.LSTMAutoencoder`
for the sequence view.

The module is self-contained and only imported when it is explicitly
requested; its availability does not affect the rest of the pipeline.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Lightweight 1-D convolutional encoder for sequence windows.

    Parameters
    ----------
    input_dim:
        Number of features per time-step ``F``.
    num_filters:
        List of output channel counts for successive conv layers.
    kernel_size:
        Convolutional kernel width (applied to the time axis).
    latent_dim:
        Output dimensionality of the final linear projection.
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: Optional[List[int]] = None,
        kernel_size: int = 3,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        if num_filters is None:
            num_filters = [32, 64]

        # Build conv layers operating on (B, F, T) tensors
        conv_layers: List[nn.Module] = []
        in_ch = input_dim
        for out_ch in num_filters:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch

        self.conv_net = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global average pooling over time
        self.fc = nn.Linear(in_ch, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, F)`` → ``(B, latent_dim)``."""
        # Conv1d expects (B, C, L) → transpose
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.conv_net(x)  # (B, num_filters[-1], T)
        x = self.pool(x).squeeze(-1)  # (B, num_filters[-1])
        return self.fc(x)  # (B, latent_dim)

    def anomaly_score(self, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance of encoded *X* from *reference* centroid.

        Parameters
        ----------
        X:
            Window tensor ``(B, T, F)``.
        reference:
            Centroid latent vector ``(latent_dim,)`` computed on training data.

        Returns
        -------
        scores:
            Array ``(B,)`` with higher values indicating more anomalous.
        """
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            z = self.forward(t).cpu().numpy()
        return np.linalg.norm(z - reference, axis=1)
