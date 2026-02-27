"""Tests for GraphStatEncoder (GNN fallback, no torch_geometric required)."""

from __future__ import annotations

import numpy as np
import pytest

from packetcore.models.gnn import GraphStatEncoder


@pytest.fixture()
def graph():
    rng = np.random.default_rng(13)
    n_nodes = 5
    n_edges = 8
    return {
        "node_ids": np.arange(n_nodes, dtype=np.int64),
        "node_features": rng.random((n_nodes, 4)).astype(np.float32),
        "edge_index": rng.integers(0, n_nodes, (2, n_edges)).astype(np.int64),
        "edge_features": rng.random((n_edges, 4)).astype(np.float32),
    }


def test_encode_shape(graph):
    enc = GraphStatEncoder()
    feat = enc.encode(graph)
    # 4*4 node moments + 4*4 edge moments + 2 counts = 34
    assert feat.ndim == 1
    assert feat.shape[0] == 4 * 4 + 4 * 4 + 2


def test_encode_batch_shape(graph):
    enc = GraphStatEncoder()
    graphs = [graph, graph, graph]
    batch = enc.encode_batch(graphs)
    assert batch.shape == (3, enc.encode(graph).shape[0])


def test_encode_empty_graph():
    enc = GraphStatEncoder()
    empty = {
        "node_ids": np.empty(0, dtype=np.int64),
        "node_features": np.empty((0, 4), dtype=np.float32),
        "edge_index": np.empty((2, 0), dtype=np.int64),
        "edge_features": np.empty((0, 4), dtype=np.float32),
    }
    feat = enc.encode(empty)
    assert feat.shape[0] == 4 * 4 + 4 * 4 + 2
    # All moments should be zero for empty graph
    assert np.allclose(feat[:-2], 0.0)
