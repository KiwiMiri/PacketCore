import json
import math
import os
from collections import Counter

PREFIX = "modbus"
WINDOWS = (1, 5, 30)
TOPK_BY_WINDOW = {
    1: 100_000,
    5: 50_000,
    30: 20_000,
}
SPLIT_RATIOS = (0.7, 0.1, 0.2)
OTHER_KEY = {"src_ip": "__other__", "dst_ip": "__other__", "unit_id": -1}
SCHEMA_VERSION = "v1"
DROP_FC_FOR_OTHER = True
WRITE_SPLIT_FILES = False


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_to_str(key_obj):
    if not isinstance(key_obj, dict):
        return json.dumps({"src_ip": "__bad__", "dst_ip": "__bad__", "unit_id": -1}, sort_keys=True)
    return json.dumps(
        {
            "src_ip": str(key_obj.get("src_ip", "0.0.0.0")),
            "dst_ip": str(key_obj.get("dst_ip", "0.0.0.0")),
            "unit_id": int(key_obj.get("unit_id", -1)),
        },
        sort_keys=True,
    )


def safe_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return default


def safe_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def input_paths(window_sec):
    return {
        "key": f"{PREFIX}.win{window_sec}.key.jsonl",
        "sparse": f"{PREFIX}.win{window_sec}.sparse.jsonl",
        "global": f"{PREFIX}.win{window_sec}.global.jsonl",
    }


def output_paths(window_sec):
    return {
        "key_kept": f"{PREFIX}.win{window_sec}.flow_kept.jsonl",
        "other": f"{PREFIX}.win{window_sec}.other.jsonl",
        "global": f"{PREFIX}.win{window_sec}.global_post.jsonl",
        "all_flow": f"{PREFIX}.win{window_sec}.flow_all.jsonl",
        "keep_keys": f"{PREFIX}.win{window_sec}.keep_keys.train.jsonl",
    }


def collect_flow_rows(window_sec):
    rows = []
    paths = input_paths(window_sec)
    for kind in ("key", "sparse"):
        p = paths[kind]
        if not os.path.exists(p):
            continue
        for rec in iter_jsonl(p):
            if rec.get("key_type") == "flow":
                rows.append(rec)
    rows.sort(key=lambda r: safe_float(r.get("t0", 0.0), 0.0))
    return rows


def pick_keep_keys(window_sec, train_end_t0):
    counts = Counter()
    rows = collect_flow_rows(window_sec)
    for rec in rows:
        t0 = safe_float(rec.get("t0", 0.0), 0.0)
        if t0 >= train_end_t0:
            continue
        key_s = key_to_str(rec.get("key"))
        counts[key_s] += safe_int(rec.get("pkt_cnt", 0), 0)

    # Fallback: if strict "< train_end_t0" yields nothing, include boundary window.
    if not counts and train_end_t0 is not None:
        for rec in rows:
            t0 = safe_float(rec.get("t0", 0.0), 0.0)
            if t0 > train_end_t0:
                continue
            key_s = key_to_str(rec.get("key"))
            counts[key_s] += safe_int(rec.get("pkt_cnt", 0), 0)

    topk = TOPK_BY_WINDOW[window_sec]
    keep = {k for k, _ in counts.most_common(topk)}
    return keep, counts


def init_other_bucket(window_sec, t0):
    return {
        "schema_version": SCHEMA_VERSION,
        "t0": float(t0),
        "window_sec": int(window_sec),
        "key_type": "flow",
        "key": dict(OTHER_KEY),
        "pkt_cnt": 0,
        "byte_cnt": 0,
        "req_cnt": 0,
        "resp_cnt": 0,
        "exc_cnt": 0,
        "inter_cnt_pos": 0,
        "neg_delta_cnt": 0,
        "delta_src_frame_cnt": 0,
        "delta_src_tcp_cnt": 0,
        "delta_src_none_cnt": 0,
    }


def merge_into_other(bucket, rec):
    bucket["pkt_cnt"] += safe_int(rec.get("pkt_cnt", 0), 0)
    bucket["byte_cnt"] += safe_int(rec.get("byte_cnt", 0), 0)
    bucket["req_cnt"] += safe_int(rec.get("req_cnt", 0), 0)
    bucket["resp_cnt"] += safe_int(rec.get("resp_cnt", 0), 0)
    bucket["exc_cnt"] += safe_int(rec.get("exc_cnt", 0), 0)
    bucket["inter_cnt_pos"] += safe_int(rec.get("inter_cnt_pos", 0), 0)
    bucket["neg_delta_cnt"] += safe_int(rec.get("neg_delta_cnt", 0), 0)
    bucket["delta_src_frame_cnt"] += safe_int(rec.get("delta_src_frame_cnt", 0), 0)
    bucket["delta_src_tcp_cnt"] += safe_int(rec.get("delta_src_tcp_cnt", 0), 0)
    bucket["delta_src_none_cnt"] += safe_int(rec.get("delta_src_none_cnt", 0), 0)


def finalize_other_bucket(bucket):
    w = safe_float(bucket["window_sec"], 1.0)
    bucket["pkt_rate"] = bucket["pkt_cnt"] / max(w, 1.0)
    bucket["byte_rate"] = bucket["byte_cnt"] / max(w, 1.0)
    bucket["req_resp_ratio"] = bucket["req_cnt"] / max(bucket["resp_cnt"], 1)
    bucket["exc_ratio"] = bucket["exc_cnt"] / max(bucket["pkt_cnt"], 1)
    if DROP_FC_FOR_OTHER:
        bucket["fc_unique_cnt"] = None
        bucket["fc_entropy"] = None
    else:
        bucket["fc_unique_cnt"] = 0
        bucket["fc_entropy"] = 0.0
    bucket["inter_mean"] = None
    bucket["inter_std"] = None
    bucket["inter_p50"] = None
    bucket["inter_p95"] = None
    bucket["burstiness"] = None
    return bucket


def rewrite_window(window_sec, keep):
    in_paths = input_paths(window_sec)
    out_paths = output_paths(window_sec)

    kept_rows = []
    other_buckets = {}
    global_rows = []

    for kind in ("key", "sparse"):
        p = in_paths[kind]
        if not os.path.exists(p):
            continue
        for rec in iter_jsonl(p):
            if rec.get("key_type") != "flow":
                continue
            key_s = key_to_str(rec.get("key"))
            if key_s in keep:
                kept_rows.append(rec)
            else:
                t0 = safe_float(rec.get("t0", 0.0), 0.0)
                bk = other_buckets.get(t0)
                if bk is None:
                    bk = init_other_bucket(window_sec, t0)
                    other_buckets[t0] = bk
                merge_into_other(bk, rec)

    if os.path.exists(in_paths["global"]):
        for rec in iter_jsonl(in_paths["global"]):
            global_rows.append(rec)

    kept_rows.sort(key=lambda r: safe_float(r.get("t0", 0.0), 0.0))
    other_rows = [finalize_other_bucket(other_buckets[t0]) for t0 in sorted(other_buckets)]
    global_rows.sort(key=lambda r: safe_float(r.get("t0", 0.0), 0.0))
    all_flow_rows = sorted([*kept_rows, *other_rows], key=lambda r: safe_float(r.get("t0", 0.0), 0.0))

    write_jsonl(out_paths["key_kept"], kept_rows)
    write_jsonl(out_paths["other"], other_rows)
    write_jsonl(out_paths["global"], global_rows)
    write_jsonl(out_paths["all_flow"], all_flow_rows)
    write_jsonl(
        out_paths["keep_keys"],
        [{"key": json.loads(k), "source_split": "train"} for k in sorted(keep)],
    )

    return {
        "window_sec": window_sec,
        "kept_keys": len(keep),
        "rows_kept": len(kept_rows),
        "rows_other": len(other_rows),
        "rows_global": len(global_rows),
        "rows_all_flow": len(all_flow_rows),
        "outputs": out_paths,
    }


def split_boundaries(rows):
    if not rows:
        return None, None
    ts = sorted(safe_float(r.get("t0", 0.0), 0.0) for r in rows)
    n = len(ts)
    i_train = max(0, min(n - 1, int(math.floor(n * SPLIT_RATIOS[0])) - 1))
    i_val = max(i_train, min(n - 1, int(math.floor(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))) - 1))
    return ts[i_train], ts[i_val]


def split_rows(rows, train_end_t0, val_end_t0):
    train, val, test = [], [], []
    for r in rows:
        t0 = safe_float(r.get("t0", 0.0), 0.0)
        if t0 <= train_end_t0:
            train.append(r)
        elif t0 <= val_end_t0:
            val.append(r)
        else:
            test.append(r)
    return train, val, test


def write_split_files(base_path, rows, train_end_t0, val_end_t0):
    train, val, test = split_rows(rows, train_end_t0, val_end_t0)
    if WRITE_SPLIT_FILES:
        write_jsonl(base_path.replace(".jsonl", ".train.jsonl"), train)
        write_jsonl(base_path.replace(".jsonl", ".val.jsonl"), val)
        write_jsonl(base_path.replace(".jsonl", ".test.jsonl"), test)
    return {"train": len(train), "val": len(val), "test": len(test)}


def main():
    summary = {
        "schema_version": SCHEMA_VERSION,
        "keep_policy": {
            "type": "topk_by_pkt_cnt",
            "source_split": "train",
            "weight_field": "pkt_cnt",
            "topk_by_window": TOPK_BY_WINDOW,
        },
        "split_policy": {
            "type": "chronological",
            "ratios": SPLIT_RATIOS,
            "write_split_files": WRITE_SPLIT_FILES,
        },
        "windows": [],
    }

    for w in WINDOWS:
        raw_flow_rows = collect_flow_rows(w)
        train_end_t0, val_end_t0 = split_boundaries(raw_flow_rows)
        if train_end_t0 is None:
            summary["windows"].append(
                {
                    "window_sec": w,
                    "split": {"train_end_t0": None, "val_end_t0": None, "counts": {}},
                }
            )
            continue

        keep, counts = pick_keep_keys(w, train_end_t0)
        win_info = rewrite_window(w, keep)

        split_counts = {}
        for name in ("key_kept", "other", "global", "all_flow"):
            p = win_info["outputs"][name]
            rows = list(iter_jsonl(p))
            split_counts[name] = write_split_files(p, rows, train_end_t0, val_end_t0)

        win_info["split"] = {
            "train_end_t0": train_end_t0,
            "val_end_t0": val_end_t0,
            "counts": split_counts,
        }
        win_info["keep_policy"] = {
            "source_split": "train",
            "top_k": TOPK_BY_WINDOW[w],
            "weight_field": "pkt_cnt",
            "cutoff": {"train_end_t0": train_end_t0, "val_end_t0": val_end_t0},
            "train_unique_keys_seen": len(counts),
        }
        summary["windows"].append(win_info)

    out_meta = f"{PREFIX}.postprocess.meta.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"done: {out_meta}")


if __name__ == "__main__":
    main()
