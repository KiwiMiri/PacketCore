"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Return a tiny synthetic packet/flow DataFrame for testing."""
    rng = np.random.default_rng(0)
    n = 200
    t0 = 1_700_000_000.0
    df = pd.DataFrame(
        {
            "timestamp": np.linspace(t0, t0 + 3600, n),
            "src_ip_int": rng.integers(0xC0A80001, 0xC0A800FF, n),
            "dst_ip_int": rng.integers(0xC0A80001, 0xC0A800FF, n),
            "src_port": rng.integers(1024, 65535, n),
            "dst_port": rng.integers(80, 8080, n),
            "protocol": rng.choice([6, 17], n),
            "pkt_len": rng.uniform(60, 1500, n).astype(np.float32),
            "ip_ttl": rng.uniform(32, 128, n).astype(np.float32),
            "tcp_flags": rng.integers(0, 63, n),
            "flow_duration": rng.uniform(0, 60, n).astype(np.float32),
            "flow_pkt_count": rng.uniform(1, 100, n).astype(np.float32),
            "flow_byte_count": rng.uniform(60, 150_000, n).astype(np.float32),
            "flow_pkt_rate": rng.uniform(1, 1000, n).astype(np.float32),
            "flow_byte_rate": rng.uniform(100, 1_000_000, n).astype(np.float32),
            "inter_arrival_mean": rng.uniform(0, 1, n).astype(np.float32),
            "inter_arrival_std": rng.uniform(0, 0.5, n).astype(np.float32),
            "pkt_len_mean": rng.uniform(60, 1500, n).astype(np.float32),
            "pkt_len_std": rng.uniform(0, 200, n).astype(np.float32),
            "pkt_len_min": rng.uniform(40, 100, n).astype(np.float32),
            "pkt_len_max": rng.uniform(500, 1500, n).astype(np.float32),
            "is_payload_visible": rng.integers(0, 2, n).astype(np.int8),
            "opcua_msg_type": np.zeros(n, dtype=np.int32),
            "opcua_security_mode": np.zeros(n, dtype=np.int32),
            "payload_entropy": rng.uniform(0, 8, n).astype(np.float32),
            "payload_len": rng.uniform(0, 1000, n).astype(np.float32),
            "dpi_app_proto": rng.integers(0, 10, n).astype(np.int32),
        }
    )
    return df
