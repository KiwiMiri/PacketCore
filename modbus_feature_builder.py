import csv
import gzip
import json
import math
import random
import time
from collections import Counter, deque

INPUT_FILE = "modbus_parsed.csv"
OUTPUT_WINDOW_FILE = "modbus_window_features.csv"
OUTPUT_SESSION_FILE = "modbus_session_features.csv"
OUTPUT_REPORT_FILE = "modbus_feature_quality.json"

KEY_MODE = "src_dst_unit"  # src_dst_unit | src_dst
WINDOW_SIZES_SEC = [1, 5]
WINDOW_ALLOWED_LATENESS_SEC = 2.0
FLUSH_EVERY_ROWS = 100000

SESSION_IDLE_TIMEOUT_SEC = 30.0
SESSION_NEGATIVE_RESET_SEC = 1.0

BURST_THRESHOLD_SEC = 0.01
IDLE_GAP_THRESHOLD_SEC = 1.0
SAMPLE_SIZE = 2000
RANDOM_SEED = 42  # None이면 매번 다른 샘플
ERROR_SAMPLE_LIMIT = 10

NEG_HIST_BINS_SEC = [0.001, 0.01, 0.1, 1.0]
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


def to_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = None
        self.max = None

    def add(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.min is None or x < self.min:
            self.min = x
        if self.max is None or x > self.max:
            self.max = x

    def std(self):
        if self.n < 2:
            return 0.0
        return math.sqrt(self.M2 / (self.n - 1))


def reservoir_add(sample, value, seen, k):
    if k <= 0:
        return
    if len(sample) < k:
        sample.append(value)
        return
    j = random.randint(0, seen - 1)
    if j < k:
        sample[j] = value


def quantile(sample, p):
    if not sample:
        return None
    data = sorted(sample)
    idx = int(p * (len(data) - 1))
    return data[idx]


def add_negative_hist(counter, neg_delta):
    for bound in NEG_HIST_BINS_SEC:
        if neg_delta < bound:
            counter[f"<{bound}s"] += 1
            return
    counter[f">={NEG_HIST_BINS_SEC[-1]}s"] += 1


class FeatureAgg:
    def __init__(self):
        self.packet_count = 0
        self.req_count = 0
        self.resp_count = 0
        self.exception_count = 0
        self.read_count = 0
        self.write_count = 0
        self.other_fc_count = 0
        self.fc_counter = Counter()

        self.addr_min = None
        self.addr_max = None

        self.qty_stats = RunningStats()
        self.resp_time_stats = RunningStats()
        self.resp_time_seen = 0
        self.resp_time_sample = []

        self.iat_stats = RunningStats()
        self.iat_seen = 0
        self.iat_sample = []
        self.iat_pairs = 0
        self.burst_count = 0
        self.idle_gap_count = 0

        self.negative_delta_count = 0
        self.max_negative_delta_sec = 0.0
        self.negative_hist = Counter()
        self.negative_seen = 0
        self.negative_sample = []

    def add_packet(self, row, raw_iat):
        self.packet_count += 1

        is_response = to_int(row.get("is_response"), 0)
        if is_response == 1:
            self.resp_count += 1
        else:
            self.req_count += 1

        fc = to_int(row.get("mb_func_code"), -1)
        self.fc_counter[fc] += 1
        if fc >= 128:
            self.exception_count += 1
        if fc in READ_FC:
            self.read_count += 1
        elif fc in WRITE_FC:
            self.write_count += 1
        else:
            self.other_fc_count += 1

        addr = to_int(row.get("mb_reg_addr"), -1)
        if addr >= 0:
            if self.addr_min is None or addr < self.addr_min:
                self.addr_min = addr
            if self.addr_max is None or addr > self.addr_max:
                self.addr_max = addr

        qty = to_int(row.get("mb_word_cnt"), -1)
        if qty > 0:
            self.qty_stats.add(float(qty))

        resp_time = to_float(row.get("mb_response_time"), -1.0)
        if resp_time > 0:
            self.resp_time_stats.add(resp_time)
            self.resp_time_seen += 1
            reservoir_add(self.resp_time_sample, resp_time, self.resp_time_seen, SAMPLE_SIZE)

        if raw_iat is None:
            return
        self.iat_pairs += 1
        if raw_iat < 0:
            neg_delta = -raw_iat
            self.negative_delta_count += 1
            if neg_delta > self.max_negative_delta_sec:
                self.max_negative_delta_sec = neg_delta
            add_negative_hist(self.negative_hist, neg_delta)
            self.negative_seen += 1
            reservoir_add(self.negative_sample, neg_delta, self.negative_seen, SAMPLE_SIZE)
            return

        self.iat_stats.add(raw_iat)
        self.iat_seen += 1
        reservoir_add(self.iat_sample, raw_iat, self.iat_seen, SAMPLE_SIZE)
        if raw_iat < BURST_THRESHOLD_SEC:
            self.burst_count += 1
        if raw_iat > IDLE_GAP_THRESHOLD_SEC:
            self.idle_gap_count += 1

    def finalize(self, duration_sec):
        iat_p50 = quantile(self.iat_sample, 0.50)
        iat_p95 = quantile(self.iat_sample, 0.95)
        neg_p99 = quantile(self.negative_sample, 0.99)
        resp_time_p95 = quantile(self.resp_time_sample, 0.95)

        req_resp_ratio = self.req_count / self.resp_count if self.resp_count else float(self.req_count)
        exception_ratio = self.exception_count / self.packet_count if self.packet_count else 0.0

        if self.write_count > 0:
            read_write_ratio = self.read_count / self.write_count
        elif self.read_count > 0:
            read_write_ratio = float(self.read_count)
        else:
            read_write_ratio = 0.0

        if self.addr_min is not None and self.addr_max is not None:
            addr_span = self.addr_max - self.addr_min
        else:
            addr_span = -1

        out_of_order_rate = (
            self.negative_delta_count / self.iat_pairs if self.iat_pairs > 0 else 0.0
        )
        burst_ratio = self.burst_count / self.iat_seen if self.iat_seen > 0 else 0.0

        return {
            "packet_count": self.packet_count,
            "pps": self.packet_count / duration_sec if duration_sec > 0 else 0.0,
            "iat_pairs": self.iat_pairs,
            "iat_count": self.iat_stats.n,
            "iat_mean": self.iat_stats.mean if self.iat_stats.n else 0.0,
            "iat_std": self.iat_stats.std() if self.iat_stats.n else 0.0,
            "iat_p50": iat_p50 if iat_p50 is not None else 0.0,
            "iat_p95": iat_p95 if iat_p95 is not None else 0.0,
            "burst_ratio": burst_ratio,
            "idle_gap_count": self.idle_gap_count,
            "negative_delta_count": self.negative_delta_count,
            "out_of_order_rate": out_of_order_rate,
            "max_negative_delta_sec": self.max_negative_delta_sec,
            "p99_negative_delta_sec": neg_p99 if neg_p99 is not None else 0.0,
            "negative_delta_hist_json": json.dumps(dict(self.negative_hist), separators=(",", ":")),
            "req_count": self.req_count,
            "resp_count": self.resp_count,
            "req_resp_ratio": req_resp_ratio,
            "exception_count": self.exception_count,
            "exception_ratio": exception_ratio,
            "read_count": self.read_count,
            "write_count": self.write_count,
            "read_write_ratio": read_write_ratio,
            "other_fc_count": self.other_fc_count,
            "fc_hist_json": json.dumps(dict(self.fc_counter), separators=(",", ":")),
            "addr_min": self.addr_min if self.addr_min is not None else -1,
            "addr_max": self.addr_max if self.addr_max is not None else -1,
            "addr_span": addr_span,
            "qty_count": self.qty_stats.n,
            "qty_mean": self.qty_stats.mean if self.qty_stats.n else 0.0,
            "qty_std": self.qty_stats.std() if self.qty_stats.n else 0.0,
            "qty_max": self.qty_stats.max if self.qty_stats.max is not None else 0.0,
            "resp_time_count": self.resp_time_stats.n,
            "resp_time_mean": self.resp_time_stats.mean if self.resp_time_stats.n else 0.0,
            "resp_time_p95": resp_time_p95 if resp_time_p95 is not None else 0.0,
        }


class SessionState:
    def __init__(self, session_id, start_ts):
        self.session_id = session_id
        self.start_ts = start_ts
        self.last_ts = start_ts
        self.agg = FeatureAgg()


def make_flow_key(row):
    src = row.get("src_ip", "")
    dst = row.get("dst_ip", "")
    if KEY_MODE == "src_dst":
        unit = "-1"
    else:
        unit = str(to_int(row.get("mb_unit_id"), -1))
    return src, dst, unit


def open_csv_reader(path):
    if path.endswith(".gz"):
        fh = gzip.open(path, "rt", newline="", encoding="utf-8")
    else:
        fh = open(path, "r", newline="", encoding="utf-8")
    return fh, csv.DictReader(fh)


def quality_gate(out_of_order_rate, max_negative_delta_sec):
    if out_of_order_rate <= 0.0001:
        return "none"
    if out_of_order_rate <= 0.001 and max_negative_delta_sec <= 1.0:
        return "micro-sort(1~2s)"
    return "investigate-pipeline-first"


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    else:
        random.seed()

    started = time.time()
    row_count = 0
    parsed_count = 0
    parse_errors = 0
    error_samples = deque(maxlen=ERROR_SAMPLE_LIMIT)

    global_iat_pairs = 0
    global_negative_count = 0
    global_max_negative_sec = 0.0

    window_maps = {ws: {} for ws in WINDOW_SIZES_SEC}
    prev_ts_by_flow = {}
    max_ts_seen = None

    active_sessions = {}
    next_session_id = 1

    global_neg_seen = 0
    global_neg_sample = []
    global_neg_hist = Counter()

    window_fields = [
        "window_sec",
        "window_start_epoch",
        "window_end_epoch",
        "window_key",
        "src_ip",
        "dst_ip",
        "mb_unit_id",
        "packet_count",
        "pps",
        "iat_pairs",
        "iat_count",
        "iat_mean",
        "iat_std",
        "iat_p50",
        "iat_p95",
        "burst_ratio",
        "idle_gap_count",
        "negative_delta_count",
        "out_of_order_rate",
        "max_negative_delta_sec",
        "p99_negative_delta_sec",
        "negative_delta_hist_json",
        "req_count",
        "resp_count",
        "req_resp_ratio",
        "exception_count",
        "exception_ratio",
        "read_count",
        "write_count",
        "read_write_ratio",
        "other_fc_count",
        "fc_hist_json",
        "addr_min",
        "addr_max",
        "addr_span",
        "qty_count",
        "qty_mean",
        "qty_std",
        "qty_max",
        "resp_time_count",
        "resp_time_mean",
        "resp_time_p95",
    ]

    session_fields = [
        "session_id",
        "session_start_epoch",
        "session_end_epoch",
        "session_duration_sec",
        "session_key",
        "src_ip",
        "dst_ip",
        "mb_unit_id",
    ] + [
        x
        for x in window_fields
        if x
        not in {
            "window_sec",
            "window_start_epoch",
            "window_end_epoch",
            "window_key",
            "src_ip",
            "dst_ip",
            "mb_unit_id",
        }
    ]

    with open(OUTPUT_WINDOW_FILE, "w", newline="", encoding="utf-8") as wf, open(
        OUTPUT_SESSION_FILE, "w", newline="", encoding="utf-8"
    ) as sf:
        window_writer = csv.DictWriter(wf, fieldnames=window_fields)
        session_writer = csv.DictWriter(sf, fieldnames=session_fields)
        window_writer.writeheader()
        session_writer.writeheader()

        def flush_window_rows(cutoff_ts):
            for ws in WINDOW_SIZES_SEC:
                keys = []
                for key in window_maps[ws]:
                    wstart = key[0]
                    if (wstart + ws) <= cutoff_ts:
                        keys.append(key)
                for key in keys:
                    agg = window_maps[ws].pop(key)
                    wstart, src, dst, unit = key
                    row = {
                        "window_sec": ws,
                        "window_start_epoch": wstart,
                        "window_end_epoch": wstart + ws,
                        "window_key": f"{src}|{dst}|{unit}",
                        "src_ip": src,
                        "dst_ip": dst,
                        "mb_unit_id": unit,
                    }
                    row.update(agg.finalize(float(ws)))
                    window_writer.writerow(row)

        def flush_session(flow_key):
            state = active_sessions.pop(flow_key, None)
            if state is None:
                return
            src, dst, unit = flow_key
            duration = max(0.0, state.last_ts - state.start_ts)
            row = {
                "session_id": state.session_id,
                "session_start_epoch": state.start_ts,
                "session_end_epoch": state.last_ts,
                "session_duration_sec": duration,
                "session_key": f"{src}|{dst}|{unit}",
                "src_ip": src,
                "dst_ip": dst,
                "mb_unit_id": unit,
            }
            row.update(state.agg.finalize(duration))
            session_writer.writerow(row)

        in_fh, reader = open_csv_reader(INPUT_FILE)
        try:
            for row in reader:
                row_count += 1
                ts = to_float(row.get("timestamp"), None)
                if ts is None:
                    parse_errors += 1
                    error_samples.append(f"row#{row_count}: invalid timestamp")
                    continue

                flow_key = make_flow_key(row)
                prev_ts = prev_ts_by_flow.get(flow_key)
                raw_iat = None
                if prev_ts is not None:
                    raw_iat = ts - prev_ts
                    global_iat_pairs += 1
                    if raw_iat < 0:
                        neg_delta = -raw_iat
                        global_negative_count += 1
                        if neg_delta > global_max_negative_sec:
                            global_max_negative_sec = neg_delta
                        add_negative_hist(global_neg_hist, neg_delta)
                        global_neg_seen += 1
                        reservoir_add(global_neg_sample, neg_delta, global_neg_seen, SAMPLE_SIZE)
                prev_ts_by_flow[flow_key] = ts
                parsed_count += 1

                if max_ts_seen is None or ts > max_ts_seen:
                    max_ts_seen = ts

                for ws in WINDOW_SIZES_SEC:
                    wstart = math.floor(ts / ws) * ws
                    wkey = (wstart, flow_key[0], flow_key[1], flow_key[2])
                    agg = window_maps[ws].get(wkey)
                    if agg is None:
                        agg = FeatureAgg()
                        window_maps[ws][wkey] = agg
                    agg.add_packet(row, raw_iat)

                session = active_sessions.get(flow_key)
                if session is None:
                    session = SessionState(next_session_id, ts)
                    next_session_id += 1
                    active_sessions[flow_key] = session
                else:
                    gap = ts - session.last_ts
                    if gap > SESSION_IDLE_TIMEOUT_SEC or gap < -SESSION_NEGATIVE_RESET_SEC:
                        flush_session(flow_key)
                        session = SessionState(next_session_id, ts)
                        next_session_id += 1
                        active_sessions[flow_key] = session

                session.agg.add_packet(row, raw_iat)
                session.last_ts = ts

                if parsed_count and parsed_count % FLUSH_EVERY_ROWS == 0:
                    if max_ts_seen is not None:
                        cutoff = max_ts_seen - WINDOW_ALLOWED_LATENESS_SEC
                        flush_window_rows(cutoff)
                    wf.flush()
                    sf.flush()
                    print(
                        f"processed={parsed_count:,} parse_errors={parse_errors:,} "
                        f"active_windows={sum(len(v) for v in window_maps.values()):,} "
                        f"active_sessions={len(active_sessions):,}"
                    )
        finally:
            in_fh.close()

        if max_ts_seen is not None:
            flush_window_rows(max_ts_seen + max(WINDOW_SIZES_SEC) + WINDOW_ALLOWED_LATENESS_SEC + 1.0)
        for key in list(active_sessions.keys()):
            flush_session(key)

    out_of_order_rate = global_negative_count / global_iat_pairs if global_iat_pairs else 0.0
    p99_negative_sec = quantile(global_neg_sample, 0.99)
    decision = quality_gate(out_of_order_rate, global_max_negative_sec)

    report = {
        "generated_at_epoch": time.time(),
        "elapsed_sec": time.time() - started,
        "config": {
            "input_file": INPUT_FILE,
            "key_mode": KEY_MODE,
            "window_sizes_sec": WINDOW_SIZES_SEC,
            "window_allowed_lateness_sec": WINDOW_ALLOWED_LATENESS_SEC,
            "session_idle_timeout_sec": SESSION_IDLE_TIMEOUT_SEC,
            "session_negative_reset_sec": SESSION_NEGATIVE_RESET_SEC,
            "burst_threshold_sec": BURST_THRESHOLD_SEC,
            "idle_gap_threshold_sec": IDLE_GAP_THRESHOLD_SEC,
            "sample_size": SAMPLE_SIZE,
            "random_seed": RANDOM_SEED,
        },
        "counts": {
            "rows_total": row_count,
            "rows_parsed": parsed_count,
            "parse_errors": parse_errors,
        },
        "out_of_order_quality": {
            "iat_pairs": global_iat_pairs,
            "negative_delta_count": global_negative_count,
            "out_of_order_rate": out_of_order_rate,
            "max_negative_delta_sec": global_max_negative_sec,
            "p99_negative_delta_sec_sampled": p99_negative_sec,
            "negative_delta_histogram": dict(global_neg_hist),
            "recommended_action": decision,
        },
        "output": {
            "window_feature_file": OUTPUT_WINDOW_FILE,
            "session_feature_file": OUTPUT_SESSION_FILE,
            "quality_report_file": OUTPUT_REPORT_FILE,
        },
        "error_samples": list(error_samples),
    }

    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("done")
    print(
        f"rows_parsed={parsed_count:,} parse_errors={parse_errors:,} "
        f"out_of_order_rate={out_of_order_rate:.6f} decision={decision}"
    )
    print(f"saved: {OUTPUT_WINDOW_FILE}, {OUTPUT_SESSION_FILE}, {OUTPUT_REPORT_FILE}")


if __name__ == "__main__":
    main()
