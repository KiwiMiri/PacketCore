#!/usr/bin/env python3
import csv
import json
import os
import time
from collections import Counter

WINDOW_INPUT = "modbus_window_features.csv"
SESSION_INPUT = "modbus_session_features.csv"
WINDOW_OUTPUT = "modbus_window_features_flat_v1.csv"
SESSION_OUTPUT = "modbus_session_features_flat_v1.csv"
SCHEMA_OUTPUT = "modbus_feature_schema_v1.json"
QUALITY_REPORT = "modbus_feature_quality.json"

TOP_K = 16
READ_FC = {1, 2, 3, 4}
WRITE_FC = {5, 6, 15, 16, 22, 23}


def to_int(val, default=0):
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default


def parse_hist(hist_text):
    if not hist_text:
        return {}
    try:
        parsed = json.loads(hist_text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out = {}
    for k, v in parsed.items():
        out[str(k)] = to_int(v, 0)
    return out


def infer_key_mode(window_csv):
    if not os.path.exists(window_csv):
        return "unknown"
    with open(window_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        unit_seen = False
        for idx, row in enumerate(r, 1):
            unit = row.get("mb_unit_id", "-1")
            if str(unit) not in ("-1", "", "None"):
                unit_seen = True
                break
            if idx >= 10000:
                break
    return "src_dst_unit" if unit_seen else "src_dst"


def load_quality_key_mode(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    config = data.get("config", {})
    key_mode = config.get("key_mode")
    return str(key_mode) if key_mode else None


def collect_fc_topk(paths, top_k):
    fc_counter = Counter()
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                hist = parse_hist(row.get("fc_hist_json"))
                for fc, cnt in hist.items():
                    fc_counter[fc] += cnt
    top_fcs = [fc for fc, _ in fc_counter.most_common(top_k)]
    return top_fcs, fc_counter


def flatten_file(input_csv, output_csv, top_fcs):
    top_set = set(top_fcs)
    fc_fields = [f"FC_{fc}" for fc in top_fcs]
    extra_fc_fields = ["FC_OTHER", "FC_READ_TOTAL", "FC_WRITE_TOTAL", "FC_EXCEPTION_TOTAL"]

    with open(input_csv, "r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        base_fields = [c for c in reader.fieldnames if c != "fc_hist_json"]
        fieldnames = base_fields + fc_fields + extra_fc_fields

        with open(output_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                hist = parse_hist(row.get("fc_hist_json"))
                out = {k: row.get(k, "") for k in base_fields}

                for fc in top_fcs:
                    out[f"FC_{fc}"] = 0

                other = 0
                read_total = 0
                write_total = 0
                exception_total = 0

                for fc, cnt in hist.items():
                    fc_int = to_int(fc, -1)
                    if fc in top_set:
                        out[f"FC_{fc}"] += cnt
                    else:
                        other += cnt

                    if fc_int in READ_FC:
                        read_total += cnt
                    elif fc_int in WRITE_FC:
                        write_total += cnt
                    if fc_int >= 128:
                        exception_total += cnt

                out["FC_OTHER"] = other
                out["FC_READ_TOTAL"] = read_total
                out["FC_WRITE_TOTAL"] = write_total
                out["FC_EXCEPTION_TOTAL"] = exception_total
                writer.writerow(out)

    return fieldnames


def main():
    sources = [p for p in (WINDOW_INPUT, SESSION_INPUT) if os.path.exists(p)]
    if not sources:
        raise RuntimeError("입력 파일이 없습니다. modbus_feature_builder.py를 먼저 실행하세요.")

    top_fcs, fc_counter = collect_fc_topk(sources, TOP_K)

    window_fields = None
    if os.path.exists(WINDOW_INPUT):
        window_fields = flatten_file(WINDOW_INPUT, WINDOW_OUTPUT, top_fcs)
    session_fields = None
    if os.path.exists(SESSION_INPUT):
        session_fields = flatten_file(SESSION_INPUT, SESSION_OUTPUT, top_fcs)

    key_mode = load_quality_key_mode(QUALITY_REPORT)
    if key_mode is None:
        key_mode = infer_key_mode(WINDOW_INPUT)

    schema = {
        "schema_version": "modbus_feature_v1",
        "generated_at_epoch": time.time(),
        "key_mode": key_mode,
        "top_k": TOP_K,
        "top_fc_list": top_fcs,
        "fc_counter_top50": dict(fc_counter.most_common(50)),
        "inputs": {
            "window": WINDOW_INPUT if os.path.exists(WINDOW_INPUT) else None,
            "session": SESSION_INPUT if os.path.exists(SESSION_INPUT) else None,
            "quality_report": QUALITY_REPORT if os.path.exists(QUALITY_REPORT) else None,
        },
        "outputs": {
            "window_flat": WINDOW_OUTPUT if window_fields else None,
            "session_flat": SESSION_OUTPUT if session_fields else None,
        },
        "fieldnames": {
            "window_flat": window_fields,
            "session_flat": session_fields,
        },
    }

    with open(SCHEMA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print("done")
    print(f"key_mode={key_mode} top_k={TOP_K} top_fcs={top_fcs}")
    if window_fields:
        print(f"saved: {WINDOW_OUTPUT}")
    if session_fields:
        print(f"saved: {SESSION_OUTPUT}")
    print(f"saved: {SCHEMA_OUTPUT}")


if __name__ == "__main__":
    main()
