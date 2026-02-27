#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CORES=$(nproc)
DURATION=70

echo "cores=$CORES"

cat > _bench_modbus_preprocess_runner.py <<'PY'
import runpy

ctx = runpy.run_path("modbus_preprocessing", run_name="bench_modbus_preprocessing")
ctx["INPUT_FILE"] = "modbus.jsonl"
ctx["OUTPUT_FILE"] = "NUL"
ctx["OUTPUT_GZIP"] = False
ctx["ENABLE_RESUME"] = False
ctx["SNAPSHOT_FILE"] = "modbus_progress_probe.json"
ctx["PRINT_EVERY"] = 20000
ctx["FLUSH_EVERY"] = 20000
ctx["main"]()
PY

monitor_run() {
  local label="$1"
  local duration="$2"
  shift 2

  local log="${label}.bench.log"
  local sample="${label}.bench.cpu"
  : > "$log"
  : > "$sample"

  local start_epoch now elapsed last_print
  start_epoch=$(date +%s)
  last_print=0

  "$@" > "$log" 2>&1 &
  local pid=$!
  echo "[$label] pid=$pid start=$start_epoch"

  while kill -0 "$pid" 2>/dev/null; do
    now=$(date +%s)
    elapsed=$((now - start_epoch))
    if (( elapsed >= duration )); then
      break
    fi

    local ps_line
    ps_line=$(LC_ALL=C ps -p "$pid" -o %cpu=,psr=,nlwp=,rss= | awk 'NR==1{print $1, $2, $3, $4}') || true
    if [[ -n "${ps_line:-}" ]]; then
      echo "$now $ps_line" >> "$sample"
      if (( elapsed - last_print >= 10 )); then
        echo "[$label] t=${elapsed}s cpu_psr_nlwp_rss=${ps_line}"
        last_print=$elapsed
      fi
    fi
    sleep 2
  done

  if kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid" 2>/dev/null || true
    sleep 1
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  wait "$pid" || true

  local end_epoch runtime
  end_epoch=$(date +%s)
  runtime=$((end_epoch - start_epoch))

  awk -v label="$label" -v runtime="$runtime" '
    {
      cpu += $2
      if ($2 > max_cpu) max_cpu = $2
      psr[$3] = 1
      nlwp += $4
      rss += $5
      n += 1
    }
    END {
      uniq = 0
      for (k in psr) uniq += 1
      if (n == 0) {
        printf("RESULT label=%s runtime_s=%d samples=0 avg_cpu=0 max_cpu=0 unique_psr=0 avg_nlwp=0 avg_rss_kb=0\n", label, runtime)
      } else {
        printf("RESULT label=%s runtime_s=%d samples=%d avg_cpu=%.2f max_cpu=%.2f unique_psr=%d avg_nlwp=%.2f avg_rss_kb=%.0f\n", label, runtime, n, cpu/n, max_cpu, uniq, nlwp/n, rss/n)
      }
    }
  ' "$sample"
}

monitor_run "convert_ijson_orjson" "$DURATION" \
  .venv/Scripts/python.exe convert_json_array_to_jsonl.py --in modbus.json --out NUL --progress-every 100000 --progress-seconds 10

if [[ -f convert_ijson_orjson.bench.log ]]; then
  echo "LAST_PROGRESS convert=$(rg '^\\[progress\\]' convert_ijson_orjson.bench.log | tail -n 1 || true)"
fi

monitor_run "feature_modbus_preprocessing" "$DURATION" \
  .venv/Scripts/python.exe _bench_modbus_preprocess_runner.py

if [[ -f feature_modbus_preprocessing.bench.log ]]; then
  echo "LAST_PROGRESS feature=$(rg '처리중' feature_modbus_preprocessing.bench.log | tail -n 1 || true)"
fi

python3 - <<'PY'
import json
from pathlib import Path

p = Path('modbus_progress_probe.json')
if not p.exists():
    print('SNAPSHOT feature=missing')
else:
    data = json.loads(p.read_text(encoding='utf-8'))
    counts = data.get('counts', {})
    print(
        'SNAPSHOT feature '
        f"decoded={counts.get('decoded_packets', 0)} "
        f"valid={counts.get('valid_packets', 0)} "
        f"updated_at_epoch={data.get('updated_at_epoch', 0)}"
    )
PY
