"""Optional Graph Neural Network encoder.

Processes the communication graph produced by
:class:`~packetcore.features.graph.GraphBuilder` and returns per-graph or
per-node anomaly representations.

Heavy dependencies (torch_geometric) are kept **optional**: the module
defines a pure-numpy fallback :class:`GraphStatEncoder` that computes
simple graph statistics (node degree moments, edge weight moments) as
features for downstream models when PyTorch Geometric is not available.

Usage
-----
When ``torch_geometric`` is available::

    from packetcore.models.gnn import GCNEncoder
    model = GCNEncoder(node_feat_dim=4, hidden_dim=32, latent_dim=16)

Always available::

    from packetcore.models.gnn import GraphStatEncoder
    enc = GraphStatEncoder()
    feat = enc.encode(graph_dict)
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Always-available fallback: graph statistics encoder
# ---------------------------------------------------------------------------


class GraphStatEncoder:
    """Encode a graph dictionary as a fixed-length statistical feature vector.

    The feature vector contains moments (mean, std, min, max) of the node
    features and edge features, giving a compact summary regardless of the
    number of nodes/edges.

    Parameters
    ----------
    node_feat_dim:
        Expected number of node feature columns (default 4 from GraphBuilder).
    edge_feat_dim:
        Expected number of edge feature columns (default 4 from GraphBuilder).
    """

    def __init__(self, node_feat_dim: int = 4, edge_feat_dim: int = 4) -> None:
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

    def encode(self, graph: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a 1-D feature vector summarising *graph*.

        Parameters
        ----------
        graph:
            Dictionary as returned by :meth:`~packetcore.features.graph.GraphBuilder.build`.

        Returns
        -------
        features:
            1-D array of shape ``(4 * node_feat_dim + 4 * edge_feat_dim + 2,)``
            containing moments of node/edge features plus node/edge counts.
        """
        nf = graph.get("node_features", np.empty((0, self.node_feat_dim)))
        ef = graph.get("edge_features", np.empty((0, self.edge_feat_dim)))

        def _moments(arr: np.ndarray, dim: int) -> np.ndarray:
            if arr.shape[0] == 0:
                return np.zeros(4 * dim, dtype=np.float32)
            return np.concatenate([
                arr.mean(axis=0),
                arr.std(axis=0),
                arr.min(axis=0),
                arr.max(axis=0),
            ]).astype(np.float32)

        node_moments = _moments(nf, self.node_feat_dim)
        edge_moments = _moments(ef, self.edge_feat_dim)
        counts = np.array([len(nf), graph["edge_index"].shape[1] if graph["edge_index"].ndim == 2 else 0], dtype=np.float32)

        return np.concatenate([node_moments, edge_moments, counts])

    def encode_batch(self, graphs: list) -> np.ndarray:
        """Encode a list of graph dicts → 2-D array ``(W, feat_dim)``."""
        return np.stack([self.encode(g) for g in graphs], axis=0)


# ---------------------------------------------------------------------------
# Optional PyTorch Geometric GCN encoder
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, global_mean_pool

    class GCNEncoder(nn.Module):
        """Two-layer GCN encoder producing a graph-level latent vector.

        Requires ``torch_geometric`` to be installed.

        Parameters
        ----------
        node_feat_dim:
            Number of input node features.
        hidden_dim:
            Hidden GCN layer width.
        latent_dim:
            Output latent dimensionality (after global mean pooling + linear).
        """

        def __init__(
            self,
            node_feat_dim: int = 4,
            hidden_dim: int = 32,
            latent_dim: int = 16,
        ) -> None:
            super().__init__()
            self.conv1 = GCNConv(node_feat_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, latent_dim)

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor", batch: "torch.Tensor") -> "torch.Tensor":
            """Return graph-level embedding ``(B, latent_dim)``."""
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = global_mean_pool(x, batch)
            return self.fc(x)

    logger.debug("torch_geometric found; GCNEncoder is available.")

except ImportError:
    logger.debug("torch_geometric not found; GCNEncoder is not available.")

    class GCNEncoder:  # type: ignore[no-redef]
        """Stub raised when torch_geometric is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GCNEncoder requires torch_geometric. "
                "Install it with: pip install torch_geometric"
            )
