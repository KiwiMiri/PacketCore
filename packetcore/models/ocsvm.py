"""One-Class SVM wrapper operating on latent vectors.

The intended usage is to:

1. Train an autoencoder (AE or VAE) or LSTM-AE.
2. Extract latent vectors from the encoder for all training samples.
3. Fit :class:`OneClassSVMModel` on those latent vectors.
4. At inference time, compute the latent vector and pass it to
   :meth:`~OneClassSVMModel.anomaly_score`.

This separates the deep feature learning from the decision boundary,
keeping each component simple and independently re-trainable.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


class OneClassSVMModel:
    """One-Class SVM applied to latent representations.

    Parameters
    ----------
    kernel:
        SVM kernel.  ``"rbf"`` is a sensible default for continuous
        latent spaces.
    nu:
        An upper bound on the fraction of training errors and a lower
        bound of the fraction of support vectors.  Acts similarly to
        ``contamination`` in Isolation Forest.
    gamma:
        Kernel coefficient.  ``"scale"`` uses ``1 / (n_features * X.var())``.
    **kwargs:
        Additional keyword arguments forwarded to
        :class:`~sklearn.svm.OneClassSVM`.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.01,
        gamma: object = "scale",
        **kwargs,
    ) -> None:
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma, **kwargs)
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, Z: np.ndarray) -> "OneClassSVMModel":
        """Fit the OCSVM on latent vectors *Z* of shape ``(N, latent_dim)``."""
        self.model.fit(Z)
        self._fitted = True
        logger.info("OneClassSVM fitted on %d latent vectors (dim=%d).", *Z.shape)
        return self

    def anomaly_score(self, Z: np.ndarray) -> np.ndarray:
        """Return anomaly scores for latent vectors *Z*.

        Scores are non-negative (higher = more anomalous):
        ``-decision_function(Z)``.

        Parameters
        ----------
        Z:
            Latent matrix of shape ``(N, latent_dim)``.

        Returns
        -------
        scores:
            Array of shape ``(N,)``.
        """
        self._check_fitted()
        return -self.model.decision_function(Z)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Return binary labels: ``1`` inlier, ``-1`` outlier."""
        self._check_fitted()
        return self.model.predict(Z)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before calling anomaly_score() or predict().")
