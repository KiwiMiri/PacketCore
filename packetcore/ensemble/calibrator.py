"""CDF-based score calibrator.

Each base model produces raw anomaly scores on an unconstrained scale.
Before combining scores in the weighted ensemble we calibrate them so
that every model's output lies in ``[0, 1]`` with the same semantics:
0 = perfectly normal, 1 = maximally anomalous.

Calibration procedure
---------------------
1. Run the model on the **validation** split (never on training data to
   avoid leakage, and never on test data to avoid peeking at the future).
2. Fit an empirical CDF on those validation scores.
3. At inference time, map a raw score ``s`` to its percentile rank in the
   validation CDF: ``calibrated = CDF(s)``.

This makes scores directly comparable across models that would otherwise
return values in completely different ranges.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CDFCalibrator:
    """Empirical CDF calibrator for anomaly scores.

    Parameters
    ----------
    eps:
        Small value added to avoid perfectly 0 or 1 calibrated scores,
        which can cause numerical issues in downstream log transforms.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self._sorted_scores: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, scores: np.ndarray) -> "CDFCalibrator":
        """Fit the calibrator on validation-set *scores*.

        Parameters
        ----------
        scores:
            1-D array of raw anomaly scores from the validation split.
        """
        scores = np.asarray(scores, dtype=np.float64).ravel()
        if len(scores) == 0:
            raise ValueError("Cannot fit CDFCalibrator on empty scores array.")
        self._sorted_scores = np.sort(scores)
        self._fitted = True
        logger.info("CDFCalibrator fitted on %d validation scores.", len(scores))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw *scores* to calibrated ``[0, 1]`` values.

        Parameters
        ----------
        scores:
            1-D (or broadcastable) array of raw anomaly scores.

        Returns
        -------
        calibrated:
            Array of same shape with values in ``(eps, 1-eps)``.
        """
        self._check_fitted()
        scores = np.asarray(scores, dtype=np.float64)
        original_shape = scores.shape
        scores_flat = scores.ravel()

        # Percentile rank via searchsorted
        n = len(self._sorted_scores)
        ranks = np.searchsorted(self._sorted_scores, scores_flat, side="right")
        calibrated = ranks.astype(np.float64) / n

        # Clip to avoid 0 and 1
        calibrated = np.clip(calibrated, self.eps, 1.0 - self.eps)
        return calibrated.reshape(original_shape).astype(np.float32)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        """Fit then transform."""
        self.fit(scores)
        return self.transform(scores)

    # ------------------------------------------------------------------
    # Threshold helper
    # ------------------------------------------------------------------

    def threshold_at_percentile(self, percentile: float) -> float:
        """Return the raw score corresponding to a given *percentile*.

        For example, ``threshold_at_percentile(99.0)`` gives the raw score
        that separates the top 1 % of validation scores from the rest.
        This is the recommended way to set per-model thresholds.

        Parameters
        ----------
        percentile:
            Value in ``[0, 100]``.
        """
        self._check_fitted()
        return float(np.percentile(self._sorted_scores, percentile))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before calling transform() or threshold_at_percentile().")
