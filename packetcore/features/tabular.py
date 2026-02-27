"""Tabular feature extractor.

Converts raw packet/flow records (``pandas.DataFrame``) into a numeric
feature matrix aligned with :data:`~packetcore.features.schema.DEFAULT_SCHEMA`.

Payload-visibility gating
-------------------------
Any field whose :attr:`~packetcore.features.schema.FieldSpec.requires_payload`
flag is ``True`` is zeroed out for rows where ``is_payload_visible == 0``.
This ensures that encrypted-protocol records (e.g. OPC UA over TLS) never leak
DPI signal into the model.

Scaling
-------
A :class:`sklearn.preprocessing.StandardScaler` is fit on training data and
applied at inference time.  The fitted scaler is saved as an artefact so that
inference is reproducible.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .schema import DEFAULT_SCHEMA, FeatureSchema

logger = logging.getLogger(__name__)


class TabularFeatureExtractor:
    """Extract and scale tabular features from a packet/flow DataFrame.

    Parameters
    ----------
    schema:
        Feature schema to use.  Defaults to :data:`DEFAULT_SCHEMA`.
    scaler:
        Pre-fitted scaler.  If ``None`` a new :class:`StandardScaler` is
        created; call :meth:`fit` before :meth:`transform`.
    """

    def __init__(
        self,
        schema: FeatureSchema = DEFAULT_SCHEMA,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.schema = schema
        self.scaler: StandardScaler = scaler if scaler is not None else StandardScaler()
        self._fitted: bool = scaler is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TabularFeatureExtractor":
        """Fit the scaler on *df* (training split only)."""
        X = self._extract_raw(df)
        self.scaler.fit(X)
        self._fitted = True
        logger.info("TabularFeatureExtractor fitted on %d samples.", len(X))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Return scaled feature matrix of shape ``(N, F)``."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = self._extract_raw(df)
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit on *df* then return scaled feature matrix."""
        self.fit(df)
        return self.transform(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Build raw (unscaled) feature matrix with payload gating applied."""
        feature_names = self.schema.names()
        payload_gated = set(self.schema.payload_gated_names())

        # Build matrix column by column
        cols: List[np.ndarray] = []
        for name in feature_names:
            if name not in df.columns:
                logger.debug("Column '%s' missing – filling with zeros.", name)
                col = np.zeros(len(df), dtype=np.float64)
            else:
                col = df[name].to_numpy(dtype=np.float64, na_value=0.0)

            # Gate payload-derived features
            if name in payload_gated:
                if "is_payload_visible" in df.columns:
                    visible_mask = df["is_payload_visible"].to_numpy(dtype=bool)
                    col = np.where(visible_mask, col, 0.0)
                else:
                    # Conservative: no visibility flag → treat all as invisible
                    col = np.zeros_like(col)

            cols.append(col)

        return np.column_stack(cols).astype(np.float64)

    def feature_names(self) -> List[str]:
        """Return the ordered list of feature names produced by this extractor."""
        return self.schema.names()

    def n_features(self) -> int:
        """Return the number of features."""
        return len(self.schema.fields)
