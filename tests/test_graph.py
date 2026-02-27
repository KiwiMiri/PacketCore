"""Tests for GraphBuilder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from packetcore.features.graph import GraphBuilder


@pytest.fixture()
def flow_df():
    rng = np.random.default_rng(2)
    n = 50
    t0 = 1_700_000_000.0
    return pd.DataFrame(
        {
            "timestamp": np.linspace(t0, t0 + 300, n),
            "src_ip_int": rng.choice([0xC0A80001, 0xC0A80002, 0xC0A80003], n),
            "dst_ip_int": rng.choice([0xC0A80004, 0xC0A80005], n),
            "pkt_len": rng.uniform(60, 1500, n),
            "flow_byte_count": rng.uniform(60, 150_000, n),
        }
    )


def test_build_returns_expected_keys(flow_df):
    builder = GraphBuilder(window_seconds=60.0)
    graph = builder.build(flow_df)
    assert "node_ids" in graph
    assert "node_features" in graph
    assert "edge_index" in graph
    assert "edge_features" in graph


def test_node_count(flow_df):
    builder = GraphBuilder()
    graph = builder.build(flow_df)
    n_nodes = len(graph["node_ids"])
    assert n_nodes >= 2  # at least src and dst


def test_edge_index_shape(flow_df):
    builder = GraphBuilder()
    graph = builder.build(flow_df)
    assert graph["edge_index"].shape[0] == 2


def test_empty_df_returns_empty_graph():
    builder = GraphBuilder()
    empty = pd.DataFrame(columns=["timestamp", "src_ip_int", "dst_ip_int", "pkt_len", "flow_byte_count"])
    graph = builder.build(empty)
    assert len(graph["node_ids"]) == 0
    assert graph["edge_index"].shape == (2, 0)


def test_build_windows_count(flow_df):
    builder = GraphBuilder(window_seconds=60.0)
    graphs = builder.build_windows(flow_df, timestamp_col="timestamp")
    # linspace(t0, t0+300, 50): t_max == t0+300, so the while loop fires for
    # t_start in {0, 60, 120, 180, 240, 300} → 6 windows (the last contains
    # only the single endpoint).  Assert at least 5 non-empty windows.
    assert len(graphs) >= 5


def test_build_windows_missing_timestamp_raises(flow_df):
    builder = GraphBuilder()
    with pytest.raises(ValueError):
        builder.build_windows(flow_df, timestamp_col="nonexistent")


def test_node_features_shape(flow_df):
    builder = GraphBuilder()
    graph = builder.build(flow_df)
    nf = graph["node_features"]
    assert nf.ndim == 2
    assert nf.shape[1] == 4  # out_degree, in_degree, bytes_sent, bytes_recv


def test_invalid_window_seconds():
    with pytest.raises(ValueError):
        GraphBuilder(window_seconds=0)
