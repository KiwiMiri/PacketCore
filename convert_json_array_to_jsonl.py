#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import deque
from contextlib import nullcontext

try:
    import ijson  # type: ignore
except Exception:
    ijson = None

try:
    import orjson  # type: ignore
except Exception:
    orjson = None


def iter_json_array_stdlib(path, chunk_size_mb=8, max_buffer_mb=64, stats=None):
    decoder = json.JSONDecoder()
    chunk_size = chunk_size_mb * 1024 * 1024
    max_buffer_chars = max_buffer_mb * 1024 * 1024
    buffer = ""
    eof = False
    in_array = False
    need_more_data = False

    with open(path, "rb", buffering=chunk_size) as fin:
        while True:
            if not eof and (need_more_data or len(buffer) < chunk_size):
                chunk = fin.read(chunk_size)
                if chunk:
                    if stats is not None:
                        stats["bytes_read"] = stats.get("bytes_read", 0) + len(chunk)
                    if b"\x00" in chunk:
                        chunk = chunk.replace(b"\x00", b"")
                    buffer += chunk.decode("utf-8", errors="ignore")
                    need_more_data = False
                else:
                    eof = True

            if not in_array:
                idx = buffer.find("[")
                if idx == -1:
                    if eof:
                        return
                    buffer = ""
                    continue
                buffer = buffer[idx + 1 :]
                in_array = True

            i = 0
            blen = len(buffer)
            while i < blen and buffer[i] in " \r\n\t,":
                i += 1
            if i:
                buffer = buffer[i:]

            if not buffer:
                if eof:
                    return
                continue

            if buffer[0] == "]":
                return

            try:
                obj, idx = decoder.raw_decode(buffer)
                yield obj
                buffer = buffer[idx:]
                need_more_data = False
            except json.JSONDecodeError:
                if eof:
                    return
                need_more_data = True
                if len(buffer) > max_buffer_chars:
                    read_mb = 0.0
                    if stats is not None:
                        read_mb = stats.get("bytes_read", 0) / (1024 * 1024)
                    raise ValueError(
                        f"buffer exceeded {max_buffer_mb}MB while waiting for a complete JSON object "
                        f"(bytes_read={read_mb:.1f}MB)"
                    )
                if len(buffer) > 50_000_000:
                    next_obj = buffer.find("{", 1)
                    if next_obj == -1:
                        buffer = ""
                    else:
                        buffer = buffer[next_obj:]


def build_line_encoder():
    if orjson is not None:
        return "orjson", lambda obj: orjson.dumps(obj) + b"\n"

    encoder = json.JSONEncoder(
        ensure_ascii=False,
        check_circular=False,
        separators=(",", ":"),
    ).encode
    return "json", lambda obj: (encoder(obj) + "\n").encode("utf-8")


class NullStrippingReader:
    """File-like wrapper that removes NUL bytes for tolerant streaming parsing."""

    def __init__(self, base):
        self.base = base

    def read(self, n=-1):
        while True:
            data = self.base.read(n)
            if not data:
                return data
            if b"\x00" in data:
                data = data.replace(b"\x00", b"")
            # Important: do not return b"" before real EOF.
            if data:
                return data


def is_tolerable_ijson_tail_error(err_repr, bytes_read, input_size):
    if input_size <= 0:
        return False
    # Treat parse errors very near EOF as tail corruption/padding noise.
    if bytes_read < (input_size * 0.98):
        return False
    low = str(err_repr).lower()
    if "incompletejsonerror" not in low:
        return False
    needles = (
        "premature eof",
        "trailing garbage",
        "invalid char in json text",
        "lexical error",
    )
    return any(n in low for n in needles)


def log_progress(
    parsed,
    skipped,
    written_this_run,
    resume_index,
    bad,
    t0,
    bytes_read,
    input_size,
    recent_window,
    bytes_baseline=0,
):
    dt = max(time.time() - t0, 1e-9)
    parse_delta = max(parsed, 0)
    write_delta = max(written_this_run, 0)
    bytes_delta = max(bytes_read - bytes_baseline, 0)
    parse_rate = parse_delta / dt
    write_rate = write_delta / dt
    mb_read = bytes_read / (1024 * 1024)
    mb_rate = (bytes_delta / (1024 * 1024)) / dt
    pct = 0.0
    if input_size > 0:
        pct = (bytes_read / input_size) * 100.0

    now = time.time()
    recent_window.append((now, parsed, written_this_run, bytes_read))
    while len(recent_window) >= 2 and (now - recent_window[0][0]) > 60.0:
        recent_window.popleft()

    recent_parse_rate = parse_rate
    recent_write_rate = write_rate
    recent_mb_rate = mb_rate
    if len(recent_window) >= 2:
        t0_recent, parsed0_recent, written0_recent, br0_recent = recent_window[0]
        dt_recent = max(now - t0_recent, 1e-9)
        recent_parse_rate = (parsed - parsed0_recent) / dt_recent
        recent_write_rate = (written_this_run - written0_recent) / dt_recent
        recent_mb_rate = ((bytes_read - br0_recent) / (1024 * 1024)) / dt_recent

    print(
        "[progress] "
        f"parsed={parsed:,} skipped={skipped:,} written+={written_this_run:,} resume_index={resume_index:,} bad={bad:,} "
        f"read={mb_read:,.1f}MB ({pct:.2f}%) "
        f"elapsed={dt:,.1f}s "
        f"parse={parse_rate:,.1f} rec/s write={write_rate:,.1f} rec/s io={mb_rate:,.2f} MB/s "
        f"1m_parse={recent_parse_rate:,.1f} 1m_write={recent_write_rate:,.1f} 1m_io={recent_mb_rate:,.2f}",
        file=sys.stderr,
        flush=True,
    )


def write_progress_snapshot(path, payload):
    if not path:
        return
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as ftmp:
        json.dump(payload, ftmp, ensure_ascii=False, indent=2)
        ftmp.write("\n")
    os.replace(tmp_path, path)


def detect_output_device(path):
    norm = os.path.normcase(str(path))
    base_upper = os.path.basename(norm).upper()
    if base_upper == "NUL":
        return "NUL", True
    if norm == os.path.normcase(os.devnull):
        return os.devnull, True
    return "file", False


def scan_jsonl_output(path, truncate_incomplete_last_line=True, chunk_size_mb=8):
    chunk_size = max(1, chunk_size_mb) * 1024 * 1024
    newline_count = 0
    last_newline_pos = -1
    offset = 0
    last_byte = b""

    with open(path, "rb", buffering=chunk_size) as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            newline_count += chunk.count(b"\n")
            rel = chunk.rfind(b"\n")
            if rel >= 0:
                last_newline_pos = offset + rel
            offset += len(chunk)
            last_byte = chunk[-1:]

    original_size = offset
    had_incomplete_tail = original_size > 0 and last_byte != b"\n"
    truncated = False
    final_size = original_size

    if had_incomplete_tail and truncate_incomplete_last_line:
        if last_newline_pos >= 0:
            final_size = last_newline_pos + 1
        else:
            final_size = 0
            newline_count = 0
        with open(path, "r+b") as f:
            f.truncate(final_size)
        truncated = True

    return {
        "line_count": int(newline_count),
        "output_bytes": int(final_size),
        "had_incomplete_tail": had_incomplete_tail,
        "truncated": truncated,
        "original_output_bytes": int(original_size),
    }


def load_resume_state(path, in_path, out_path):
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)

    snap_in = state.get("input")
    snap_out = state.get("output")
    if snap_in and os.path.abspath(str(snap_in)) != os.path.abspath(str(in_path)):
        raise ValueError(
            f"resume snapshot input mismatch: snapshot={snap_in} current={in_path}"
        )
    if snap_out and os.path.abspath(str(snap_out)) != os.path.abspath(str(out_path)):
        raise ValueError(
            f"resume snapshot output mismatch: snapshot={snap_out} current={out_path}"
        )

    def _ival(key, default=0):
        try:
            val = int(state.get(key, default) or default)
        except Exception:
            val = default
        return max(val, 0)

    return {
        "written_ok": _ival("written_ok", 0),
        "output_line_count": _ival("output_line_count", _ival("written_ok", 0)),
        "resume_index": _ival("resume_index", _ival("written_ok", 0)),
        "output_bytes": _ival("output_bytes", 0),
        "raw": state,
    }


def resolve_resume_plan(
    start_index,
    resume_state,
    out_path,
    output_is_sink,
    truncate_incomplete_last_line=True,
    scan_chunk_size_mb=8,
):
    explicit_start = start_index is not None
    requested_start = max(int(start_index or 0), 0)

    state_resume_index = None
    state_written = None
    state_lines = None
    state_bytes = None
    if resume_state is not None:
        state_resume_index = resume_state["resume_index"]
        state_written = resume_state["written_ok"]
        state_lines = resume_state["output_line_count"]
        state_bytes = resume_state["output_bytes"]

    anchors = []
    if explicit_start:
        anchors.append(requested_start)
    if state_resume_index is not None:
        anchors.append(state_resume_index)
    if state_written is not None:
        anchors.append(state_written)
    if state_lines is not None:
        anchors.append(state_lines)

    if not anchors:
        return {
            "resume_base_index": 0,
            "output_line_count": 0,
            "output_bytes": 0,
            "resumed": False,
        }

    requested = min(anchors)

    if output_is_sink:
        return {
            "resume_base_index": requested,
            "output_line_count": requested,
            "output_bytes": 0,
            "resumed": requested > 0,
        }

    if requested > 0 and not os.path.exists(out_path):
        raise ValueError(
            f"resume requested at index {requested:,}, but output file does not exist: {out_path}"
        )

    actual_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    need_scan = state_bytes is None or state_bytes != actual_bytes
    actual_lines = None
    truncated = False
    original_output_bytes = actual_bytes

    if need_scan:
        scan = scan_jsonl_output(
            out_path,
            truncate_incomplete_last_line=truncate_incomplete_last_line,
            chunk_size_mb=scan_chunk_size_mb,
        )
        actual_lines = scan["line_count"]
        actual_bytes = scan["output_bytes"]
        truncated = scan["truncated"]
        original_output_bytes = scan["original_output_bytes"]
    corrected = requested
    if state_written is not None:
        corrected = min(corrected, state_written)
    if state_lines is not None:
        corrected = min(corrected, state_lines)
    if actual_lines is not None:
        corrected = min(corrected, actual_lines)

    mismatch = False
    if resume_state is not None:
        mismatch = (
            state_bytes != actual_bytes
            or (state_written is not None and state_written != corrected)
            or (state_lines is not None and state_lines != corrected)
            or (state_resume_index is not None and state_resume_index != corrected)
        )
    if mismatch or truncated:
        print(
            "[resume-guard] mismatch: "
            f"state_bytes={state_bytes} actual_bytes={actual_bytes} "
            f"state_lines={state_lines} actual_lines={actual_lines} "
            f"state_written={state_written} state_resume={state_resume_index} "
            f"truncated={truncated} original_output_bytes={original_output_bytes} "
            f"-> resume_index={corrected} (auto-corrected)",
            file=sys.stderr,
            flush=True,
        )

    return {
        "resume_base_index": corrected,
        "output_line_count": corrected,
        "output_bytes": actual_bytes,
        "resumed": corrected > 0,
    }


def convert(
    in_path,
    out_path,
    flush_every=200000,
    progress_every=100000,
    progress_seconds=10,
    max_err_samples=50,
    force_stdlib=False,
    chunk_size_mb=8,
    max_buffer_mb=64,
    progress_state_file=None,
    start_index=None,
    resume_from=None,
    benchmark_mode="full",
):
    err_samples = deque(maxlen=max_err_samples)
    recent_window = deque()
    total = 0
    ok = 0
    bad = 0
    parser_backend = "ijson"
    serializer_backend, encode_line = build_line_encoder()
    interrupted = False
    bytes_read = 0
    input_size = os.path.getsize(in_path) if os.path.exists(in_path) else 0
    last_progress_ts = time.time()
    output_device, output_is_sink = detect_output_device(out_path)
    mode = str(benchmark_mode or "full").strip().lower()
    if mode not in {"full", "parse-encode", "parse-only"}:
        raise ValueError(f"unsupported benchmark_mode: {benchmark_mode}")

    resume_state = load_resume_state(resume_from, in_path, out_path) if resume_from else None
    resume_plan = resolve_resume_plan(
        start_index=start_index,
        resume_state=resume_state,
        out_path=out_path,
        output_is_sink=output_is_sink,
    )
    resume_base_index = resume_plan["resume_base_index"]
    output_line_count = resume_plan["output_line_count"]
    output_bytes = resume_plan["output_bytes"]
    resumed = resume_plan["resumed"]
    ok = output_line_count
    bytes_baseline = 0

    write_enabled = mode == "full" and not output_is_sink
    encode_enabled = mode in {"full", "parse-encode"} or (mode == "full" and output_is_sink)
    if mode == "full" and output_is_sink:
        print(
            "[benchmark] output sink detected; write disabled (effective mode=parse-encode)",
            file=sys.stderr,
            flush=True,
        )

    out_mode = "ab" if resumed else "wb"
    if resumed:
        print(
            f"[resume] start_index={resume_base_index:,} (append mode)",
            file=sys.stderr,
            flush=True,
        )
    recent_window.append((time.time(), 0, 0, bytes_read))
    last_record_seen = -1
    tail_error_treated_as_eof = False
    last_error_type = None
    last_error_bytes_read = 0
    last_error_parsed_this_run = 0

    t0 = time.time()

    def parsed_this_run():
        return total

    def skipped_this_run():
        return min(total, resume_base_index)

    def written_this_run():
        return max(0, ok - resume_base_index)

    def resume_index_value():
        return ok

    def compute_progress_fields(current_bytes_read, tail_done=False):
        if input_size > 0:
            byte_progress_pct = min(max((current_bytes_read / input_size) * 100.0, 0.0), 100.0)
            tail_bytes_unparsed = max(0, input_size - int(current_bytes_read))
            tail_unparsed_pct = max(0.0, 100.0 - byte_progress_pct)
        else:
            byte_progress_pct = 0.0
            tail_bytes_unparsed = 0
            tail_unparsed_pct = 0.0
        logical_input_progress_pct = 100.0 if tail_done else byte_progress_pct
        return {
            "byte_progress_pct": byte_progress_pct,
            "logical_input_progress_pct": logical_input_progress_pct,
            "tail_bytes_unparsed": tail_bytes_unparsed,
            "tail_unparsed_pct": tail_unparsed_pct,
            # Backward compatibility for existing dashboards/scripts.
            "input_progress_pct": byte_progress_pct,
        }

    # Prevent infinite full re-parse loops when the previous run already hit
    # the same tail corruption near EOF and produced no new rows.
    if resume_state is not None and start_index is None:
        prev = resume_state.get("raw", {})
        prev_tail = bool(prev.get("tail_error_treated_as_eof", False))
        prev_err_bytes = int(prev.get("last_error_bytes_read", 0) or 0)
        prev_written_this_run = int(prev.get("written_this_run", 0) or 0)
        prev_resume_index = int(prev.get("resume_index", 0) or 0)
        prev_input_size = int(prev.get("input_size", 0) or 0)
        if (
            prev_tail
            and prev_written_this_run == 0
            and prev_input_size == input_size
            and prev_resume_index == resume_base_index
            and prev_err_bytes >= int(input_size * 0.98)
        ):
            progress_fields = compute_progress_fields(prev_err_bytes, tail_done=True)
            err_samples.append(
                {
                    "type": "tail_repeat_short_circuit",
                    "err": "previous run ended with tail parse warning near EOF; skipping re-parse",
                }
            )
            print(
                "[resume-guard] prior tail warning at EOF detected with no new writes; skipping re-parse.",
                file=sys.stderr,
                flush=True,
            )
            summary = {
                "input": in_path,
                "output": out_path,
                "output_device": output_device,
                "benchmark_mode": mode,
                "write_enabled": write_enabled,
                "encode_enabled": encode_enabled,
                "parser_backend": parser_backend,
                "serializer_backend": serializer_backend,
                "ijson_available": ijson is not None,
                "orjson_available": orjson is not None,
                "interrupted": False,
                "total_seen": resume_index_value(),
                "parsed_this_run": 0,
                "skipped_this_run": 0,
                "written_this_run": 0,
                "written_ok": ok,
                "output_line_count": output_line_count,
                "output_bytes": output_bytes,
                "bad": 0,
                "bytes_read": prev_err_bytes,
                "input_size": input_size,
                **progress_fields,
                "resumed": resumed,
                "start_index": resume_base_index,
                "resume_index": resume_index_value(),
                "started_new_output": False,
                "tail_error_treated_as_eof": True,
                "last_error_type": "IncompleteJSONError",
                "last_error_bytes_read": prev_err_bytes,
                "last_error_parsed_this_run": 0,
                "sec": 0.0,
                "err_samples_recent": list(err_samples),
            }
            write_progress_snapshot(
                progress_state_file,
                {
                    "input": in_path,
                    "output": out_path,
                    "output_device": output_device,
                    "benchmark_mode": mode,
                    "write_enabled": write_enabled,
                    "encode_enabled": encode_enabled,
                    "parser_backend": parser_backend,
                    "serializer_backend": serializer_backend,
                    "total_seen": resume_index_value(),
                    "parsed_this_run": 0,
                    "skipped_this_run": 0,
                    "written_this_run": 0,
                    "written_ok": ok,
                    "output_line_count": output_line_count,
                    "output_bytes": output_bytes,
                    "bad": 0,
                    "bytes_read": prev_err_bytes,
                    "input_size": input_size,
                    **progress_fields,
                    "start_index": resume_base_index,
                    "resume_index": resume_index_value(),
                    "resumed": resumed,
                    "tail_error_treated_as_eof": True,
                    "last_error_type": "IncompleteJSONError",
                    "last_error_bytes_read": prev_err_bytes,
                    "last_error_parsed_this_run": 0,
                    "updated_at": time.time(),
                },
            )
            return summary

    out_ctx = open(out_path, out_mode) if write_enabled else nullcontext(None)
    with out_ctx as fout:
        def capture_output_bytes():
            nonlocal output_bytes
            if output_is_sink:
                output_bytes = 0
                return output_bytes
            if not write_enabled:
                output_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
                return output_bytes
            fout.flush()
            output_bytes = os.fstat(fout.fileno()).st_size
            return output_bytes

        def emit_progress(now, current_bytes_read):
            nonlocal last_progress_ts, last_record_seen
            parsed = parsed_this_run()
            hit_record_threshold = (
                progress_every > 0
                and parsed > 0
                and parsed % progress_every == 0
                and parsed != last_record_seen
            )
            hit_time_threshold = progress_seconds > 0 and (now - last_progress_ts) >= progress_seconds
            if not (hit_record_threshold or hit_time_threshold):
                return
            if hit_record_threshold:
                last_record_seen = parsed

            log_progress(
                parsed,
                skipped_this_run(),
                written_this_run(),
                resume_index_value(),
                bad,
                t0,
                current_bytes_read,
                input_size,
                recent_window,
                bytes_baseline=bytes_baseline,
            )
            capture_output_bytes()
            progress_fields = compute_progress_fields(
                current_bytes_read, tail_done=tail_error_treated_as_eof
            )
            write_progress_snapshot(
                progress_state_file,
                {
                    "input": in_path,
                    "output": out_path,
                    "output_device": output_device,
                    "benchmark_mode": mode,
                    "write_enabled": write_enabled,
                    "encode_enabled": encode_enabled,
                    "parser_backend": parser_backend,
                    "serializer_backend": serializer_backend,
                    "total_seen": resume_index_value(),
                    "parsed_this_run": parsed,
                    "skipped_this_run": skipped_this_run(),
                    "written_this_run": written_this_run(),
                    "written_ok": ok,
                    "output_line_count": output_line_count,
                    "output_bytes": output_bytes,
                    "bad": bad,
                    "bytes_read": current_bytes_read,
                    "input_size": input_size,
                    **progress_fields,
                    "start_index": resume_base_index,
                    "resume_index": resume_index_value(),
                    "resumed": resumed,
                    "tail_error_treated_as_eof": tail_error_treated_as_eof,
                    "last_error_type": last_error_type,
                    "last_error_bytes_read": last_error_bytes_read,
                    "last_error_parsed_this_run": last_error_parsed_this_run,
                    "updated_at": now,
                },
            )
            last_progress_ts = now

        try:
            if ijson is not None and not force_stdlib:
                with open(in_path, "rb") as fin:
                    iterator = ijson.items(NullStrippingReader(fin), "item")
                    for obj in iterator:
                        total += 1
                        if total <= resume_base_index:
                            bytes_read = fin.tell()
                            emit_progress(time.time(), bytes_read)
                            continue

                        try:
                            encoded = encode_line(obj) if encode_enabled else None
                            if write_enabled:
                                fout.write(encoded)
                                ok += 1
                                output_line_count += 1
                        except Exception as e:
                            bad += 1
                            err_samples.append({"type": "dump_write_error", "err": repr(e)})
                        if write_enabled and flush_every > 0 and total % flush_every == 0:
                            fout.flush()
                        bytes_read = fin.tell()
                        emit_progress(time.time(), bytes_read)
            else:
                parser_backend = "stdlib"
                parser_stats = {"bytes_read": 0}
                for obj in iter_json_array_stdlib(
                    in_path,
                    chunk_size_mb=chunk_size_mb,
                    max_buffer_mb=max_buffer_mb,
                    stats=parser_stats,
                ):
                    total += 1
                    if total <= resume_base_index:
                        bytes_read = parser_stats.get("bytes_read", 0)
                        emit_progress(time.time(), bytes_read)
                        continue

                    try:
                        encoded = encode_line(obj) if encode_enabled else None
                        if write_enabled:
                            fout.write(encoded)
                            ok += 1
                            output_line_count += 1
                    except Exception as e:
                        bad += 1
                        err_samples.append({"type": "dump_write_error", "err": repr(e)})
                    if write_enabled and flush_every > 0 and total % flush_every == 0:
                        fout.flush()
                    bytes_read = parser_stats.get("bytes_read", 0)
                    emit_progress(time.time(), bytes_read)
        except KeyboardInterrupt:
            interrupted = True
            err_samples.append({"type": "keyboard_interrupt", "err": "KeyboardInterrupt"})
        except Exception as e:
            err_repr = repr(e)
            if is_tolerable_ijson_tail_error(err_repr, bytes_read, input_size):
                tail_error_treated_as_eof = True
                last_error_type = "IncompleteJSONError"
                last_error_bytes_read = int(bytes_read)
                last_error_parsed_this_run = int(total)
                err_samples.append({"type": "array_parse_tail_warning", "err": err_repr})
                print(
                    "[warn] tail parse warning ignored (likely trailing NUL/partial tail): "
                    f"{err_repr}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                bad += 1
                last_error_type = type(e).__name__
                last_error_bytes_read = int(bytes_read)
                last_error_parsed_this_run = int(total)
                err_samples.append({"type": "array_parse_error", "err": err_repr})
        finally:
            capture_output_bytes()
            progress_fields = compute_progress_fields(
                bytes_read, tail_done=tail_error_treated_as_eof
            )
            write_progress_snapshot(
                progress_state_file,
                {
                    "input": in_path,
                    "output": out_path,
                    "output_device": output_device,
                    "benchmark_mode": mode,
                    "write_enabled": write_enabled,
                    "encode_enabled": encode_enabled,
                    "parser_backend": parser_backend,
                    "serializer_backend": serializer_backend,
                    "total_seen": resume_index_value(),
                    "parsed_this_run": parsed_this_run(),
                    "skipped_this_run": skipped_this_run(),
                    "written_this_run": written_this_run(),
                    "written_ok": ok,
                    "output_line_count": output_line_count,
                    "output_bytes": output_bytes,
                    "bad": bad,
                    "bytes_read": bytes_read,
                    "input_size": input_size,
                    **progress_fields,
                    "start_index": resume_base_index,
                    "resume_index": resume_index_value(),
                    "resumed": resumed,
                    "tail_error_treated_as_eof": tail_error_treated_as_eof,
                    "last_error_type": last_error_type,
                    "last_error_bytes_read": last_error_bytes_read,
                    "last_error_parsed_this_run": last_error_parsed_this_run,
                    "updated_at": time.time(),
                },
            )

    dt = time.time() - t0
    if bytes_read <= 0:
        bytes_read = input_size if ok > 0 and not interrupted else bytes_read
    progress_fields = compute_progress_fields(bytes_read, tail_done=tail_error_treated_as_eof)
    summary = {
        "input": in_path,
        "output": out_path,
        "output_device": output_device,
        "benchmark_mode": mode,
        "write_enabled": write_enabled,
        "encode_enabled": encode_enabled,
        "parser_backend": parser_backend,
        "serializer_backend": serializer_backend,
        "ijson_available": ijson is not None,
        "orjson_available": orjson is not None,
        "interrupted": interrupted,
        "total_seen": resume_index_value(),
        "parsed_this_run": parsed_this_run(),
        "skipped_this_run": skipped_this_run(),
        "written_this_run": written_this_run(),
        "written_ok": ok,
        "output_line_count": output_line_count,
        "output_bytes": output_bytes,
        "bad": bad,
        "bytes_read": bytes_read,
        "input_size": input_size,
        **progress_fields,
        "resumed": resumed,
        "start_index": resume_base_index,
        "resume_index": resume_index_value(),
        "started_new_output": write_enabled and out_mode == "wb",
        "tail_error_treated_as_eof": tail_error_treated_as_eof,
        "last_error_type": last_error_type,
        "last_error_bytes_read": last_error_bytes_read,
        "last_error_parsed_this_run": last_error_parsed_this_run,
        "sec": dt,
        "err_samples_recent": list(err_samples),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--flush-every", type=int, default=200000)
    ap.add_argument("--progress-every", type=int, default=100000)
    ap.add_argument("--progress-seconds", type=int, default=10)
    ap.add_argument("--chunk-size-mb", type=int, default=8)
    ap.add_argument("--max-buffer-mb", type=int, default=64)
    ap.add_argument("--progress-state-file", type=str, default=None)
    ap.add_argument("--resume-from", type=str, default=None, help="이전 progress snapshot 경로")
    ap.add_argument("--start-index", type=int, default=None, help="N번째 객체부터 이어쓰기(앞 객체는 skip)")
    ap.add_argument(
        "--benchmark-mode",
        type=str,
        default="full",
        choices=["full", "parse-encode", "parse-only"],
        help="full=파싱+직렬화+쓰기, parse-encode=파싱+직렬화(쓰기 없음), parse-only=파싱만",
    )
    ap.add_argument("--max-err-samples", type=int, default=50)
    ap.add_argument(
        "--force-stdlib",
        action="store_true",
        help="ijson이 있더라도 stdlib 스트리밍 파서를 사용",
    )
    args = ap.parse_args()

    summary = convert(
        args.inp,
        args.outp,
        flush_every=args.flush_every,
        progress_every=args.progress_every,
        progress_seconds=args.progress_seconds,
        max_err_samples=args.max_err_samples,
        force_stdlib=args.force_stdlib,
        chunk_size_mb=args.chunk_size_mb,
        max_buffer_mb=args.max_buffer_mb,
        progress_state_file=args.progress_state_file,
        start_index=args.start_index,
        resume_from=args.resume_from,
        benchmark_mode=args.benchmark_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
