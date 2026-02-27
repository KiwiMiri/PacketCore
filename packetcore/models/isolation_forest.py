"""Isolation Forest wrapper for tabular anomaly detection.

Wraps ``sklearn.ensemble.IsolationForest`` to provide a consistent
``anomaly_score`` interface that returns non-negative scores (higher = more
anomalous) by inverting the sklearn ``decision_function`` output.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """Thin wrapper around :class:`sklearn.ensemble.IsolationForest`.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_samples:
        Number of samples to draw for each tree.  ``"auto"`` uses
        ``min(256, n_samples)``.
    contamination:
        Expected fraction of outliers.  Used only for the ``predict``
        threshold (not for ``anomaly_score``).
    random_state:
        Random seed for reproducibility.
    **kwargs:
        Additional keyword arguments forwarded to
        :class:`~sklearn.ensemble.IsolationForest`.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: object = "auto",
        contamination: float = 0.01,
        random_state: Optional[int] = 42,
        **kwargs,
    ) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            **kwargs,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        """Fit the isolation forest on *X* (training data)."""
        self.model.fit(X)
        self._fitted = True
        logger.info("IsolationForest fitted on %d samples, %d features.", *X.shape)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores for *X*.

        Scores are non-negative (higher = more anomalous).  They are
        computed as ``-decision_function(X)`` so that inliers have scores
        near 0 and outliers have positive scores.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(N, F)``.

        Returns
        -------
        scores:
            Array of shape ``(N,)``.
        """
        self._check_fitted()
        # decision_function returns positive for inliers, negative for outliers
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions: ``1`` for inlier, ``-1`` for outlier."""
        self._check_fitted()
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before calling anomaly_score() or predict().")
