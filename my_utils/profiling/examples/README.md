# Unified Metrics Demo

Run:

```bash
python -m my_utils.profiling.examples.unified_metrics_demo --output-dir ./demo_metrics_output --steps 30
```

This demo shows:

1. Custom provider (`SyntheticTrainProvider`)
2. Generic external CSV ingestion (`TableCsvMetricsProvider`)
3. NCU CSV ingestion (`NcuCsvMetricsProvider`)
4. Unified collection + analysis + report export (`json/md/html`)

Output files:

- `metrics_events.jsonl`
- `report.json`
- `report.md`
- `report.html`
- `external_tool_metrics.csv`
- `ncu_metrics.csv`

You can replace the synthetic provider with your runtime provider (MyTimer / torch.profiler / ModuleProfiler).

## P0-P13 End-to-End Acceptance Demo

Run:

```bash
python -m my_utils.profiling.examples.p0_p13_end_to_end_demo --output-dir ./p0_p13_demo_output --steps 20
```

Optional NSYS schema probe:

```bash
python -m my_utils.profiling.examples.p0_p13_end_to_end_demo \
  --output-dir ./p0_p13_demo_output \
  --nsys-sqlite ./train_rank0.sqlite
```

This demo covers:

1. Multi-provider unified collection
2. Workload-aware analyzer (`pretrain` and `inference`)
3. Report export (`json/md/html`)
4. Run-to-run diff generation (`report_diff.json`, `report_diff.md`)
5. Optional NSYS schema introspection (`describe_schema`)

## Collector Config JSON Examples

Example config files in this directory:

- `collector_config_example.json`
  - Full offline configuration in one file (`table_csv`, `ncu_csv`, `nsys_sqlite`, `cprofile`, `perf_stat`).
- `collector_config_nsys_sqlite_full.json`
  - Full `nsys_sqlite` single-rank configuration for one SQLite file.
- `collector_config_nsys_multi_rank_full.json`
  - Full `nsys_sqlite` multi-rank configuration for two SQLite files.

Run with CLI:

```bash
# 1) single-rank sqlite analysis
myutils-profile ingest \
  --config ./my_utils/profiling/examples/collector_config_nsys_sqlite_full.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html

# 2) multi-rank sqlite analysis
myutils-profile ingest \
  --config ./my_utils/profiling/examples/collector_config_nsys_multi_rank_full.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html

# 3) export chrome trace from unified events
myutils-profile trace \
  --events ./nsys_metrics_out/metrics_events.jsonl \
  --output ./nsys_metrics_out/metrics_trace.json \
  --auto-align-ranks \
  --reference-rank 0
```

Notes:

- `nsys_sqlite` provider parameter is `sqlite_path` (not `db_path`).
- For `my_timer` / `torch_profiler` / `module_profiler`, use Python API with `provider_context` object injection instead of CLI-only config.
