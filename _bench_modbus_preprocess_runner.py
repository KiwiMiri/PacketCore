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
