"""Weighted ensemble of calibrated anomaly scores.

Combines per-model calibrated scores into a single ensemble score using a
fixed weight vector.  Weights can be derived automatically from validation
AUROC values or set manually.

Design choices
--------------
* Scores are expected to be **already calibrated** (i.e. passed through
  :class:`~packetcore.ensemble.calibrator.CDFCalibrator`) so that every
  model contributes on the same ``[0, 1]`` scale.
* Weights are normalised internally so that they sum to 1.
* Each model is identified by a string name for interpretability.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """Weighted combination of calibrated anomaly scores.

    Parameters
    ----------
    model_names:
        Ordered list of model identifier strings.
    weights:
        Weight for each model.  If ``None``, uniform weights are used.
        Weights are normalised to sum to 1 automatically.
    """

    def __init__(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
    ) -> None:
        if not model_names:
            raise ValueError("model_names must not be empty.")
        self.model_names = list(model_names)
        n = len(model_names)

        if weights is None:
            self._weights = np.ones(n, dtype=np.float64) / n
        else:
            if len(weights) != n:
                raise ValueError(
                    f"Length of weights ({len(weights)}) must match "
                    f"length of model_names ({n})."
                )
            w = np.array(weights, dtype=np.float64)
            if np.any(w < 0):
                raise ValueError("All weights must be non-negative.")
            total = w.sum()
            if total == 0:
                raise ValueError("Weights must not all be zero.")
            self._weights = w / total

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        """Normalised weight vector (read-only copy)."""
        return self._weights.copy()

    # ------------------------------------------------------------------
    # Score combination
    # ------------------------------------------------------------------

    def combine(self, score_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted ensemble score from per-model score arrays.

        Parameters
        ----------
        score_dict:
            Mapping of model name → calibrated score array ``(N,)``.
            Models in ``self.model_names`` not present in *score_dict*
            contribute 0 to the ensemble (logged as a warning).

        Returns
        -------
        ensemble_score:
            Weighted sum of calibrated scores, shape ``(N,)``.
        """
        n_samples: Optional[int] = None
        for name in self.model_names:
            if name in score_dict:
                n_samples = len(score_dict[name])
                break

        if n_samples is None:
            raise ValueError("score_dict contains none of the expected model names.")

        combined = np.zeros(n_samples, dtype=np.float64)
        for name, w in zip(self.model_names, self._weights):
            if name not in score_dict:
                logger.warning("Model '%s' missing from score_dict; contributing 0.", name)
                continue
            scores = np.asarray(score_dict[name], dtype=np.float64).ravel()
            if len(scores) != n_samples:
                raise ValueError(
                    f"Score array for '{name}' has length {len(scores)}, "
                    f"expected {n_samples}."
                )
            combined += w * scores

        return combined.astype(np.float32)

    # ------------------------------------------------------------------
    # Weight fitting from AUROC
    # ------------------------------------------------------------------

    def fit_weights_from_auroc(
        self,
        score_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> "WeightedEnsemble":
        """Set weights proportional to per-model AUROC on the validation set.

        Parameters
        ----------
        score_dict:
            Per-model calibrated scores on the validation split.
        labels:
            Binary ground-truth labels ``(1 = anomaly, 0 = normal)``.

        Returns
        -------
        self (for method chaining).
        """
        from sklearn.metrics import roc_auc_score

        labels = np.asarray(labels, dtype=np.int32).ravel()
        new_weights = np.zeros(len(self.model_names), dtype=np.float64)

        for i, name in enumerate(self.model_names):
            if name not in score_dict:
                logger.warning("Model '%s' missing; AUROC weight set to 0.", name)
                continue
            scores = np.asarray(score_dict[name], dtype=np.float64).ravel()
            try:
                auc = roc_auc_score(labels, scores)
            except ValueError as exc:
                logger.warning("AUROC computation failed for '%s': %s; using 0.", name, exc)
                auc = 0.0
            # Clip below 0.5: models worse than random should not count
            new_weights[i] = max(0.0, auc - 0.5)
            logger.info("Model '%s' AUROC=%.4f  weight (before norm)=%.4f", name, auc, new_weights[i])

        total = new_weights.sum()
        if total == 0:
            logger.warning("All AUROC weights are 0; falling back to uniform weights.")
            self._weights = np.ones(len(self.model_names), dtype=np.float64) / len(self.model_names)
        else:
            self._weights = new_weights / total

        return self
