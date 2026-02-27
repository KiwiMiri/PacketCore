"""End-to-end integration test for the full pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from packetcore.data.splitter import time_based_split
from packetcore.ensemble.calibrator import CDFCalibrator
from packetcore.ensemble.ensemble import WeightedEnsemble
from packetcore.ensemble.scorer import ProtocolConditionalScorer
from packetcore.features.schema import DEFAULT_SCHEMA
from packetcore.features.sequence import SequenceBuilder
from packetcore.features.tabular import TabularFeatureExtractor
from packetcore.models.autoencoder import Autoencoder
from packetcore.models.isolation_forest import IsolationForestModel
from packetcore.models.lstm_ae import LSTMAutoencoder
from packetcore.models.ocsvm import OneClassSVMModel
from packetcore.utils.artifacts import load_artifact, save_artifact

MODEL_NAMES = ["ae", "lstm_ae", "iforest", "ocsvm"]


@pytest.fixture()
def full_df():
    """Synthetic dataset with all required columns."""
    rng = np.random.default_rng(42)
    n = 300
    t0 = 1_700_000_000.0
    return pd.DataFrame(
        {
            "timestamp": np.linspace(t0, t0 + 3600, n),
            "src_ip_int": rng.integers(0xC0A80001, 0xC0A800FF, n),
            "dst_ip_int": rng.integers(0xC0A80001, 0xC0A800FF, n),
            "src_port": rng.integers(1024, 65535, n),
            "dst_port": rng.integers(80, 8080, n),
            "protocol": rng.choice([6, 17], n).astype(np.int32),
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


def test_full_pipeline(full_df, tmp_path):
    """Train → calibrate → infer pipeline produces valid outputs."""

    # 1. Time-based split
    train_df, val_df, test_df = time_based_split(full_df, val_frac=0.15, test_frac=0.15)
    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

    # 2. Feature extraction
    extractor = TabularFeatureExtractor(schema=DEFAULT_SCHEMA)
    X_train = extractor.fit_transform(train_df)
    X_val = extractor.transform(val_df)
    X_test = extractor.transform(test_df)

    # 3. Train AE
    ae = Autoencoder(input_dim=X_train.shape[1], hidden_dims=[32], latent_dim=8)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for _ in range(2):  # minimal training for speed
        x = torch.tensor(X_train)
        x_hat, _ = ae(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 4. Train LSTM-AE
    seq_builder = SequenceBuilder(window_size=8, stride=1, pad_partial=True)
    X_train_w = seq_builder.build(X_train)
    X_val_w = seq_builder.build(X_val)
    X_test_w = seq_builder.build(X_test)

    lstm = LSTMAutoencoder(input_dim=X_train.shape[1], hidden_dim=16, latent_dim=8)
    opt2 = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    for _ in range(2):
        x = torch.tensor(X_train_w)
        x_hat, _ = lstm(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        opt2.zero_grad()
        loss.backward()
        opt2.step()

    # 5. Isolation Forest
    iforest = IsolationForestModel(n_estimators=10)
    iforest.fit(X_train)

    # 6. OCSVM on latent vectors
    ae.eval()
    with torch.no_grad():
        Z_train = ae.encode(torch.tensor(X_train)).numpy()
        Z_val = ae.encode(torch.tensor(X_val)).numpy()
        Z_test = ae.encode(torch.tensor(X_test)).numpy()
    ocsvm = OneClassSVMModel()
    ocsvm.fit(Z_train)

    # 7. Raw scores on val
    raw_scores_val = {
        "ae": ae.anomaly_score(X_val),
        "lstm_ae": lstm.anomaly_score(X_val_w),
        "iforest": iforest.anomaly_score(X_val),
        "ocsvm": ocsvm.anomaly_score(Z_val),
    }

    # 8. Calibrate
    calibrators = {}
    for name, scores in raw_scores_val.items():
        cal = CDFCalibrator()
        cal.fit(scores)
        calibrators[name] = cal

    cal_scores_val = {name: calibrators[name].transform(raw) for name, raw in raw_scores_val.items()}

    # 9. Ensemble (uniform weights)
    ensemble = WeightedEnsemble(model_names=MODEL_NAMES)
    ensemble_scores_val = ensemble.combine(cal_scores_val)

    # 10. Protocol-conditional thresholds
    protocol_ids_val = val_df["protocol"].to_numpy(dtype=np.int32)
    scorer = ProtocolConditionalScorer()
    scorer.fit(ensemble_scores_val, protocol_ids_val, target_fpr=0.05)

    # 11. Inference on test set
    raw_scores_test = {
        "ae": ae.anomaly_score(X_test),
        "lstm_ae": lstm.anomaly_score(X_test_w),
        "iforest": iforest.anomaly_score(X_test),
        "ocsvm": ocsvm.anomaly_score(Z_test),
    }
    cal_scores_test = {name: calibrators[name].transform(raw) for name, raw in raw_scores_test.items()}
    ensemble_scores_test = ensemble.combine(cal_scores_test)
    protocol_ids_test = test_df["protocol"].to_numpy(dtype=np.int32)
    labels = scorer.predict(ensemble_scores_test, protocol_ids_test)

    assert labels.shape == (len(test_df),)
    assert set(np.unique(labels)).issubset({0, 1})

    # 12. Artefact round-trip
    save_artifact(extractor.scaler, tmp_path / "scaler.pkl")
    scaler_loaded = load_artifact(tmp_path / "scaler.pkl")
    X_test_reloaded = extractor.__class__(schema=DEFAULT_SCHEMA, scaler=scaler_loaded).transform(test_df)
    np.testing.assert_array_almost_equal(X_test, X_test_reloaded)
