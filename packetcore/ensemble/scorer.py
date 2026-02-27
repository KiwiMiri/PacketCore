"""Protocol-conditional scorer and thresholder.

Different protocols have different normal behaviour, so a single global
anomaly threshold leads to high false-positive rates on high-volume
benign protocols and high false-negative rates on low-volume attack protocols.

This module provides:

* :class:`ProtocolConditionalScorer` – applies per-protocol thresholds to
  calibrated ensemble scores.
* :func:`compute_protocol_thresholds` – utility to derive per-protocol
  thresholds from validation-set scores at a specified false-positive rate.

Protocol IDs used throughout the pipeline come from the ``protocol`` column
in the feature schema (IP protocol number, e.g. 6=TCP, 17=UDP) plus an
optional application-layer override stored in ``app_protocol_id``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold derivation
# ---------------------------------------------------------------------------


def compute_protocol_thresholds(
    scores: np.ndarray,
    protocol_ids: np.ndarray,
    target_fpr: float = 0.01,
) -> Dict[int, float]:
    """Derive per-protocol anomaly thresholds targeting *target_fpr*.

    Parameters
    ----------
    scores:
        Calibrated ensemble scores from the validation split, shape ``(N,)``.
    protocol_ids:
        Integer protocol identifiers for each sample, shape ``(N,)``.
    target_fpr:
        Desired false-positive rate on the validation set per protocol.
        Threshold is set at the ``(1 - target_fpr)`` quantile of validation
        scores for each protocol.

    Returns
    -------
    thresholds:
        Mapping of ``protocol_id → threshold``.
    """
    scores = np.asarray(scores, dtype=np.float32).ravel()
    protocol_ids = np.asarray(protocol_ids, dtype=np.int32).ravel()

    if len(scores) != len(protocol_ids):
        raise ValueError("scores and protocol_ids must have the same length.")

    percentile = (1.0 - target_fpr) * 100.0
    thresholds: Dict[int, float] = {}
    for proto in np.unique(protocol_ids):
        mask = protocol_ids == proto
        proto_scores = scores[mask]
        if len(proto_scores) == 0:
            continue
        thresholds[int(proto)] = float(np.percentile(proto_scores, percentile))
        logger.debug(
            "Protocol %d: threshold=%.4f (FPR target=%.2f%%, n=%d)",
            proto,
            thresholds[int(proto)],
            target_fpr * 100,
            mask.sum(),
        )

    return thresholds


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class ProtocolConditionalScorer:
    """Apply protocol-specific thresholds to produce binary anomaly labels.

    Parameters
    ----------
    thresholds:
        Mapping of ``protocol_id → threshold`` (calibrated score).
    default_threshold:
        Threshold used for protocols not present in *thresholds*.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[int, float]] = None,
        default_threshold: float = 0.95,
    ) -> None:
        self.thresholds: Dict[int, float] = thresholds if thresholds is not None else {}
        self.default_threshold = default_threshold

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        scores: np.ndarray,
        protocol_ids: np.ndarray,
        target_fpr: float = 0.01,
    ) -> "ProtocolConditionalScorer":
        """Derive per-protocol thresholds from validation-set data.

        Parameters
        ----------
        scores:
            Calibrated ensemble scores from the **validation** split.
        protocol_ids:
            Protocol identifier per sample.
        target_fpr:
            Target false-positive rate.
        """
        self.thresholds = compute_protocol_thresholds(scores, protocol_ids, target_fpr)
        logger.info("ProtocolConditionalScorer fitted with %d protocol thresholds.", len(self.thresholds))
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def predict(self, scores: np.ndarray, protocol_ids: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels ``(1 = anomaly, 0 = normal)``.

        Parameters
        ----------
        scores:
            Calibrated ensemble scores, shape ``(N,)``.
        protocol_ids:
            Protocol identifier per sample, shape ``(N,)``.
        """
        scores = np.asarray(scores, dtype=np.float32).ravel()
        protocol_ids = np.asarray(protocol_ids, dtype=np.int32).ravel()

        thresholds_per_sample = np.array(
            [self.thresholds.get(int(p), self.default_threshold) for p in protocol_ids],
            dtype=np.float32,
        )
        return (scores > thresholds_per_sample).astype(np.int32)

    def anomaly_flag(self, score: float, protocol_id: int) -> bool:
        """Return ``True`` if *score* exceeds the threshold for *protocol_id*."""
        thr = self.thresholds.get(protocol_id, self.default_threshold)
        return bool(score > thr)

    def threshold_for(self, protocol_id: int) -> float:
        """Return the threshold for *protocol_id* (or the default)."""
        return self.thresholds.get(protocol_id, self.default_threshold)
