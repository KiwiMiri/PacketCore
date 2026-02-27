# PacketCore OT/IT Multi-View Anomaly Ensemble

This repository focuses on anomaly detection for mixed OT/IT network traffic under realistic constraints, including partially encrypted industrial protocols (for example, OPC UA secure channels where deep payload visibility is limited).

## Scope
- Convert packet exports (`json`) into stream-friendly `jsonl`
- Extract protocol-preserving `raw_packet.csv` datasets
- Build feature pipelines for anomaly detection
- Support multi-view ensemble design:
  - Tabular view (packet/flow/window statistics)
  - Sequence view (window tensor)
  - Graph view (window-aggregated communication graph)

## Protocol Coverage in Raw Extraction
- Modbus
- MQTT
- OPC UA
- S7comm
- PROFINET

## Design Principles
- Protocol-conditional anomaly scoring
- Payload-visibility-aware modeling (`is_payload_visible`)
- Time-based split only (`train < val < test`) to avoid leakage
- Calibration-first ensemble (CDF-based score alignment)

## Repository Layout
- `convert_json_array_to_jsonl.py`: robust large JSON array to JSONL conversion
- `extract_raw_packet_csv.py`: protocol-aware raw CSV extraction
- `modbus_preprocessing`, `mqtt_preprocessing`: window/stat feature builders
- `modbus_feature_builder.py`, `modbus_postprocess_sparse_split.py`: feature shaping/post-processing

## Quick Start
1. Convert JSON to JSONL

```bash
python3 convert_json_array_to_jsonl.py --in <protocol>.json --out <protocol>.jsonl --progress-state-file <protocol>.convert.progress.json
```

2. Extract raw packet CSV

```bash
python3 extract_raw_packet_csv.py --protocol <modbus|mqtt|opcua|s7comm|profinet> --in <protocol>.jsonl --out <protocol>.raw_packet.csv
```

## Data and Security Notes
- Large raw datasets and generated CSV/JSONL are intentionally ignored by `.gitignore`.
- Do not commit sensitive internal identifiers (asset labels, private IP mappings, operational metadata) without review.

## License
This repository uses Apache License 2.0. See `LICENSE`.
