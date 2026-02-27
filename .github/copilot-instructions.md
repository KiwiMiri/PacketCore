# Copilot Instructions

This repository implements OT/IT anomaly detection with a multi-view ensemble:
- Tabular view (packet/flow statistics)
- Sequence view (window tensor, `[T x F]`)
- Graph view (window-aggregated communication graph)

## Model Stack
- AE or VAE
- LSTM-AE
- Isolation Forest
- One-Class SVM on latent vectors
- Optional extensions: CNN, GNN

## Domain Constraints
- Use protocol-conditional scoring and thresholding.
- Use payload-visibility-aware gating (`is_payload_visible`).
- For encrypted OPC UA traffic, rely on metadata/statistical features rather than DPI payload fields.
- Keep strictly time-based data splits to prevent leakage.
- Calibrate model scores using validation CDF before weighted ensemble.

## Coding Guidance
- Keep preprocessing and model code modular.
- Expose CLI for train/infer/calibrate workflows.
- Save reproducible artifacts (weights, scalers, calibrators, thresholds).
- Use explicit feature schema/version fields in outputs.
- Avoid hidden assumptions across protocols; keep protocol-specific logic explicit.
