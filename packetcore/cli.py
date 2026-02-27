"""PacketCore CLI – OT/IT anomaly detection pipeline.

Sub-commands
------------
train
    Load data, extract features, train all models, save artefacts.
calibrate
    Load trained models, run them on the validation split, fit CDF
    calibrators and per-protocol thresholds, save artefacts.
infer
    Load all artefacts, run inference on new data, output anomaly labels
    and ensemble scores.

Usage examples
--------------
::

    # Train on a CSV of packet/flow records
    python -m packetcore.cli train \\
        --data data/train.csv \\
        --artifact-dir artefacts/ \\
        --val-frac 0.15 --test-frac 0.15

    # Calibrate using the validation set produced during training
    python -m packetcore.cli calibrate \\
        --val-data artefacts/val.csv \\
        --artifact-dir artefacts/

    # Run inference on new data
    python -m packetcore.cli infer \\
        --data data/new_traffic.csv \\
        --artifact-dir artefacts/ \\
        --output results/scores.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

logger = logging.getLogger("packetcore.cli")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_NAMES = ["ae", "lstm_ae", "iforest", "ocsvm"]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_csv(path: str) -> pd.DataFrame:
    logger.info("Loading data from '%s' …", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))
    return df


def _train_ae(
    X_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    device: str,
) -> Autoencoder:
    model = Autoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = nn.functional.mse_loss(x_hat, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch)
        if epoch % max(1, epochs // 5) == 0:
            logger.info("AE epoch %d/%d  loss=%.6f", epoch, epochs, total_loss / len(X_train))

    return model


def _train_lstm_ae(
    X_windows: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    device: str,
) -> LSTMAutoencoder:
    _, T, F = X_windows.shape
    model = LSTMAutoencoder(input_dim=F, latent_dim=latent_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(X_windows, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, _ = model(batch)
            loss = nn.functional.mse_loss(x_hat, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * len(batch)
        if epoch % max(1, epochs // 5) == 0:
            logger.info("LSTM-AE epoch %d/%d  loss=%.6f", epoch, epochs, total_loss / len(X_windows))

    return model


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> None:
    _setup_logging(args.verbose)
    artifact_dir = Path(args.artifact_dir)

    # 1. Load and split data
    df = _load_csv(args.data)
    train_df, val_df, test_df = time_based_split(
        df,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        timestamp_col=args.timestamp_col,
    )

    # Save splits for later calibration/inspection
    val_df.to_csv(artifact_dir / "val.csv", index=False)
    test_df.to_csv(artifact_dir / "test.csv", index=False)

    # 2. Tabular features
    extractor = TabularFeatureExtractor(schema=DEFAULT_SCHEMA)
    X_train = extractor.fit_transform(train_df)
    save_artifact(extractor.scaler, artifact_dir / "tabular_scaler.pkl")
    logger.info("Tabular feature matrix: %s", X_train.shape)

    device = args.device

    # 3. Train AE
    logger.info("Training Autoencoder …")
    ae_model = _train_ae(
        X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        device=device,
    )
    save_artifact(ae_model.state_dict(), artifact_dir / "ae_weights.pt")
    save_artifact({"input_dim": X_train.shape[1], "latent_dim": args.latent_dim}, artifact_dir / "ae_config.json")

    # 4. Sequence windows for LSTM-AE
    seq_builder = SequenceBuilder(window_size=args.window_size, stride=1, pad_partial=True)
    X_windows = seq_builder.build(X_train)
    logger.info("Window tensor shape: %s", X_windows.shape)

    logger.info("Training LSTM-AE …")
    lstm_model = _train_lstm_ae(
        X_windows,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        device=device,
    )
    save_artifact(lstm_model.state_dict(), artifact_dir / "lstm_ae_weights.pt")
    save_artifact(
        {"input_dim": X_train.shape[1], "latent_dim": args.latent_dim, "window_size": args.window_size},
        artifact_dir / "lstm_ae_config.json",
    )

    # 5. Isolation Forest on tabular features
    logger.info("Training Isolation Forest …")
    iforest = IsolationForestModel(n_estimators=args.n_estimators, random_state=42)
    iforest.fit(X_train)
    save_artifact(iforest.model, artifact_dir / "iforest.pkl")

    # 6. OCSVM on AE latent vectors
    logger.info("Extracting AE latent vectors for OCSVM …")
    ae_model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        Z_train = ae_model.encode(x_t).cpu().numpy()
    ocsvm = OneClassSVMModel()
    ocsvm.fit(Z_train)
    save_artifact(ocsvm.model, artifact_dir / "ocsvm.pkl")

    # 7. Save schema
    save_artifact(DEFAULT_SCHEMA.to_dict(), artifact_dir / "feature_schema.json")

    logger.info("Training complete.  Artefacts saved to '%s'.", artifact_dir)


# ---------------------------------------------------------------------------
# Sub-command: calibrate
# ---------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> None:
    _setup_logging(args.verbose)
    artifact_dir = Path(args.artifact_dir)

    # Load validation data
    val_csv = args.val_data if args.val_data else str(artifact_dir / "val.csv")
    val_df = _load_csv(val_csv)

    # Load artefacts
    from sklearn.preprocessing import StandardScaler
    scaler: StandardScaler = load_artifact(artifact_dir / "tabular_scaler.pkl")
    extractor = TabularFeatureExtractor(schema=DEFAULT_SCHEMA, scaler=scaler)
    X_val = extractor.transform(val_df)

    ae_config = load_artifact(artifact_dir / "ae_config.json")
    ae_model = Autoencoder(input_dim=ae_config["input_dim"], latent_dim=ae_config["latent_dim"])
    ae_model.load_state_dict(load_artifact(artifact_dir / "ae_weights.pt", map_location="cpu"))
    ae_model.eval()

    lstm_config = load_artifact(artifact_dir / "lstm_ae_config.json")
    lstm_model = LSTMAutoencoder(input_dim=lstm_config["input_dim"], latent_dim=lstm_config["latent_dim"])
    lstm_model.load_state_dict(load_artifact(artifact_dir / "lstm_ae_weights.pt", map_location="cpu"))
    lstm_model.eval()

    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    iforest_sk = load_artifact(artifact_dir / "iforest.pkl")
    iforest = IsolationForestModel()
    iforest.model = iforest_sk
    iforest._fitted = True

    ocsvm_sk = load_artifact(artifact_dir / "ocsvm.pkl")
    ocsvm = OneClassSVMModel()
    ocsvm.model = ocsvm_sk
    ocsvm._fitted = True

    # Build window tensor for LSTM-AE
    seq_builder = SequenceBuilder(window_size=lstm_config["window_size"], stride=1, pad_partial=True)
    X_val_windows = seq_builder.build(X_val)

    # Raw scores on validation
    scores: Dict[str, np.ndarray] = {
        "ae": ae_model.anomaly_score(X_val),
        "lstm_ae": lstm_model.anomaly_score(X_val_windows),
        "iforest": iforest.anomaly_score(X_val),
        "ocsvm": ocsvm.anomaly_score(_ae_latents(ae_model, X_val)),
    }

    # Fit and save calibrators
    calibrators: Dict[str, CDFCalibrator] = {}
    for name, raw_scores in scores.items():
        cal = CDFCalibrator()
        cal.fit(raw_scores)
        calibrators[name] = cal
        save_artifact(cal, artifact_dir / f"calibrator_{name}.pkl")
        logger.info("Calibrator for '%s' saved.", name)

    # Calibrated scores
    cal_scores: Dict[str, np.ndarray] = {
        name: calibrators[name].transform(raw) for name, raw in scores.items()
    }

    # Ensemble with uniform weights initially
    ensemble = WeightedEnsemble(model_names=MODEL_NAMES)
    if args.labels_col and args.labels_col in val_df.columns:
        labels = val_df[args.labels_col].to_numpy(dtype=np.int32)
        ensemble.fit_weights_from_auroc(cal_scores, labels)
        logger.info("Ensemble weights fitted from AUROC: %s", dict(zip(MODEL_NAMES, ensemble.weights.tolist())))
    save_artifact(ensemble.weights.tolist(), artifact_dir / "ensemble_weights.json")

    # Ensemble scores on validation
    ensemble_scores = ensemble.combine(cal_scores)

    # Per-protocol thresholds
    protocol_col = args.protocol_col
    if protocol_col in val_df.columns:
        protocol_ids = val_df[protocol_col].to_numpy(dtype=np.int32)
    else:
        logger.warning("Protocol column '%s' not found; using protocol=0 for all rows.", protocol_col)
        protocol_ids = np.zeros(len(val_df), dtype=np.int32)

    scorer = ProtocolConditionalScorer()
    scorer.fit(ensemble_scores, protocol_ids, target_fpr=args.target_fpr)
    save_artifact(scorer.thresholds, artifact_dir / "protocol_thresholds.json")
    logger.info("Per-protocol thresholds saved.")

    logger.info("Calibration complete.")


def _ae_latents(ae_model: Autoencoder, X: np.ndarray) -> np.ndarray:
    ae_model.eval()
    with torch.no_grad():
        return ae_model.encode(torch.tensor(X, dtype=torch.float32)).cpu().numpy()


# ---------------------------------------------------------------------------
# Sub-command: infer
# ---------------------------------------------------------------------------


def cmd_infer(args: argparse.Namespace) -> None:
    _setup_logging(args.verbose)
    artifact_dir = Path(args.artifact_dir)

    df = _load_csv(args.data)

    # Load artefacts
    from sklearn.preprocessing import StandardScaler
    scaler: StandardScaler = load_artifact(artifact_dir / "tabular_scaler.pkl")
    extractor = TabularFeatureExtractor(schema=DEFAULT_SCHEMA, scaler=scaler)
    X = extractor.transform(df)

    ae_config = load_artifact(artifact_dir / "ae_config.json")
    ae_model = Autoencoder(input_dim=ae_config["input_dim"], latent_dim=ae_config["latent_dim"])
    ae_model.load_state_dict(load_artifact(artifact_dir / "ae_weights.pt", map_location="cpu"))
    ae_model.eval()

    lstm_config = load_artifact(artifact_dir / "lstm_ae_config.json")
    lstm_model = LSTMAutoencoder(input_dim=lstm_config["input_dim"], latent_dim=lstm_config["latent_dim"])
    lstm_model.load_state_dict(load_artifact(artifact_dir / "lstm_ae_weights.pt", map_location="cpu"))
    lstm_model.eval()

    iforest_sk = load_artifact(artifact_dir / "iforest.pkl")
    iforest = IsolationForestModel()
    iforest.model = iforest_sk
    iforest._fitted = True

    ocsvm_sk = load_artifact(artifact_dir / "ocsvm.pkl")
    ocsvm = OneClassSVMModel()
    ocsvm.model = ocsvm_sk
    ocsvm._fitted = True

    # Load calibrators
    calibrators = {
        name: load_artifact(artifact_dir / f"calibrator_{name}.pkl")
        for name in MODEL_NAMES
    }

    weights_list = load_artifact(artifact_dir / "ensemble_weights.json")
    ensemble = WeightedEnsemble(model_names=MODEL_NAMES, weights=weights_list)

    thresholds_raw = load_artifact(artifact_dir / "protocol_thresholds.json")
    thresholds = {int(k): float(v) for k, v in thresholds_raw.items()}
    scorer = ProtocolConditionalScorer(thresholds=thresholds)

    # Sequence windows
    seq_builder = SequenceBuilder(window_size=lstm_config["window_size"], stride=1, pad_partial=True)
    X_windows = seq_builder.build(X)

    # Raw scores
    raw_scores: Dict[str, np.ndarray] = {
        "ae": ae_model.anomaly_score(X),
        "lstm_ae": lstm_model.anomaly_score(X_windows),
        "iforest": iforest.anomaly_score(X),
        "ocsvm": ocsvm.anomaly_score(_ae_latents(ae_model, X)),
    }

    # Calibrate
    cal_scores: Dict[str, np.ndarray] = {
        name: calibrators[name].transform(raw) for name, raw in raw_scores.items()
    }

    # Ensemble
    ensemble_scores = ensemble.combine(cal_scores)

    # Protocol-conditional labelling
    protocol_col = args.protocol_col
    if protocol_col in df.columns:
        protocol_ids = df[protocol_col].to_numpy(dtype=np.int32)
    else:
        logger.warning("Protocol column '%s' not found; using protocol=0 for all rows.", protocol_col)
        protocol_ids = np.zeros(len(df), dtype=np.int32)

    labels = scorer.predict(ensemble_scores, protocol_ids)

    # Build output DataFrame
    result = df.copy()
    result["ensemble_score"] = ensemble_scores
    result["is_anomaly"] = labels
    for name in MODEL_NAMES:
        result[f"score_{name}"] = raw_scores[name]
        result[f"cal_score_{name}"] = cal_scores[name]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Inference results written to '%s'.", output_path)
    logger.info("Anomalies detected: %d / %d (%.2f%%)", labels.sum(), len(labels), 100 * labels.mean())


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="packetcore",
        description="PacketCore – OT/IT anomaly detection pipeline",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_p = subparsers.add_parser("train", help="Train all models and save artefacts")
    train_p.add_argument("--data", required=True, help="Path to training CSV file")
    train_p.add_argument("--artifact-dir", required=True, help="Directory to save artefacts")
    train_p.add_argument("--timestamp-col", default="timestamp", help="Timestamp column name")
    train_p.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction of time range")
    train_p.add_argument("--test-frac", type=float, default=0.15, help="Test fraction of time range")
    train_p.add_argument("--epochs", type=int, default=50, help="Training epochs for deep models")
    train_p.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    train_p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_p.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality")
    train_p.add_argument("--window-size", type=int, default=32, help="Sequence window size (T)")
    train_p.add_argument("--n-estimators", type=int, default=100, help="Isolation Forest n_estimators")
    train_p.add_argument("--device", default="cpu", help="PyTorch device string (cpu / cuda)")
    train_p.set_defaults(func=cmd_train)

    # --- calibrate ---
    cal_p = subparsers.add_parser("calibrate", help="Fit CDF calibrators and protocol thresholds")
    cal_p.add_argument("--artifact-dir", required=True, help="Artefact directory (from train)")
    cal_p.add_argument("--val-data", default=None, help="Path to validation CSV (default: artefact-dir/val.csv)")
    cal_p.add_argument("--labels-col", default=None, help="Binary label column for AUROC weight fitting")
    cal_p.add_argument("--protocol-col", default="protocol", help="Protocol ID column name")
    cal_p.add_argument("--target-fpr", type=float, default=0.01, help="Target false-positive rate for thresholds")
    cal_p.set_defaults(func=cmd_calibrate)

    # --- infer ---
    inf_p = subparsers.add_parser("infer", help="Run inference on new data")
    inf_p.add_argument("--data", required=True, help="Path to input CSV file")
    inf_p.add_argument("--artifact-dir", required=True, help="Artefact directory (from train + calibrate)")
    inf_p.add_argument("--output", required=True, help="Path to output CSV file")
    inf_p.add_argument("--protocol-col", default="protocol", help="Protocol ID column name")
    inf_p.set_defaults(func=cmd_infer)

    return parser


def main(argv: List[str] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
