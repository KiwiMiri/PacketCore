"""Communication graph builder.

Aggregates packet/flow records within a time window into a graph where
nodes are host endpoints and edges carry flow-level statistics.

The output is a dictionary of dense matrices suitable for downstream GNN
or simpler graph-statistic features:

* ``node_features``:  ``(V, Nf)`` – per-node aggregated statistics
* ``edge_index``:     ``(2, E)`` – COO format edge connectivity
* ``edge_features``:  ``(E, Ef)`` – per-edge aggregated statistics

Dependencies on heavy graph libraries (e.g. PyTorch Geometric) are kept
optional: if they are not installed the builder still returns plain numpy
arrays so that the rest of the pipeline can run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build a per-window communication graph from flow/packet records.

    Parameters
    ----------
    window_seconds:
        Duration of each time window in seconds.  Records within the same
        window are aggregated together.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.window_seconds = window_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Build a graph from *df* (a single time window).

        Parameters
        ----------
        df:
            DataFrame of flow/packet records with at minimum the columns
            ``src_ip_int``, ``dst_ip_int``, ``pkt_len``, ``flow_byte_count``.

        Returns
        -------
        graph:
            Dictionary with keys:
            ``node_ids``, ``node_features``, ``edge_index``, ``edge_features``.
        """
        if df.empty:
            return self._empty_graph()

        src_col = "src_ip_int"
        dst_col = "dst_ip_int"

        for col in (src_col, dst_col):
            if col not in df.columns:
                logger.warning("Column '%s' not found; using zeros.", col)
                df = df.copy()
                df[col] = 0

        # --- Build node index ---
        all_nodes = pd.unique(np.concatenate([df[src_col].values, df[dst_col].values]))
        node_to_idx: Dict[int, int] = {int(n): i for i, n in enumerate(all_nodes)}
        n_nodes = len(all_nodes)

        # Node features: out-degree, in-degree, total bytes sent, total bytes recv
        out_degree = np.zeros(n_nodes, dtype=np.float32)
        in_degree = np.zeros(n_nodes, dtype=np.float32)
        bytes_sent = np.zeros(n_nodes, dtype=np.float32)
        bytes_recv = np.zeros(n_nodes, dtype=np.float32)

        byte_col = "flow_byte_count" if "flow_byte_count" in df.columns else "pkt_len"

        for _, row in df.iterrows():
            s = node_to_idx[int(row[src_col])]
            d = node_to_idx[int(row[dst_col])]
            b = float(row.get(byte_col, 0))
            out_degree[s] += 1
            in_degree[d] += 1
            bytes_sent[s] += b
            bytes_recv[d] += b

        node_features = np.column_stack([out_degree, in_degree, bytes_sent, bytes_recv])

        # --- Build edge list ---
        edge_groups = (
            df.groupby([src_col, dst_col])[byte_col]
            .agg(["count", "sum", "mean", "std"])
            .reset_index()
        )
        edge_groups.columns = [src_col, dst_col, "pkt_count", "byte_sum", "byte_mean", "byte_std"]
        edge_groups["byte_std"] = edge_groups["byte_std"].fillna(0.0)

        src_idx = edge_groups[src_col].map(node_to_idx).values
        dst_idx = edge_groups[dst_col].map(node_to_idx).values
        edge_index = np.stack([src_idx, dst_idx], axis=0).astype(np.int64)

        edge_features = edge_groups[["pkt_count", "byte_sum", "byte_mean", "byte_std"]].values.astype(np.float32)

        return {
            "node_ids": all_nodes.astype(np.int64),
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

    def build_windows(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> list:
        """Build one graph per time window for the entire DataFrame.

        Parameters
        ----------
        df:
            Full dataset sorted by *timestamp_col*.
        timestamp_col:
            Name of the Unix-epoch timestamp column.

        Returns
        -------
        List of graph dictionaries, one per window.
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

        df = df.sort_values(timestamp_col).reset_index(drop=True)
        t_min = df[timestamp_col].min()
        t_max = df[timestamp_col].max()

        graphs = []
        t_start = t_min
        while t_start <= t_max:
            t_end = t_start + self.window_seconds
            mask = (df[timestamp_col] >= t_start) & (df[timestamp_col] < t_end)
            window_df = df[mask]
            graphs.append(self.build(window_df))
            t_start = t_end

        return graphs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_graph() -> Dict[str, np.ndarray]:
        return {
            "node_ids": np.empty(0, dtype=np.int64),
            "node_features": np.empty((0, 4), dtype=np.float32),
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_features": np.empty((0, 4), dtype=np.float32),
        }
