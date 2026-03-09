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
  - Full `nsys_sqlite_glob` multi-rank configuration using one glob pattern.
  - Default pattern: `./logs/light_bagel_pretrain/xxxprefix_rank_*_.log`

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

# 4) list built-in nsys SQL skills
myutils-profile nsys-sql-skill \
  --sqlite ./logs/light_bagel_pretrain/train_rank_0.sqlite \
  --list-skills \
  --pretty

# 5) run one nsys SQL skill
myutils-profile nsys-sql-skill \
  --sqlite ./logs/light_bagel_pretrain/train_rank_0.sqlite \
  --skill top_kernels \
  --param device_id=0 \
  --param limit=20 \
  --pretty \
  --output ./nsys_metrics_out/top_kernels.json

# 6) nsys summarize/analyze
myutils-profile nsys-analyze \
  --sqlite ./logs/light_bagel_pretrain/train_rank_0.sqlite \
  --device-id 0 \
  --top-k 20 \
  --output ./nsys_metrics_out/nsys_analyze.json

# 7) nsys flat export
myutils-profile nsys-export \
  --sqlite ./logs/light_bagel_pretrain/train_rank_0.sqlite \
  --device-id 0 \
  --format csv \
  --output ./nsys_metrics_out/kernels_flat.csv

# 8) nsys before/after diff
myutils-profile nsys-diff \
  --before-sqlite ./logs/light_bagel_pretrain/run_a.sqlite \
  --after-sqlite ./logs/light_bagel_pretrain/run_b.sqlite \
  --device-id 0 \
  --output ./nsys_metrics_out/nsys_diff.json

# 9) static timeline html
myutils-profile nsys-timeline-html \
  --sqlite ./logs/light_bagel_pretrain/train_rank_0.sqlite \
  --device-id 0 \
  --output ./nsys_metrics_out/timeline.html
```

Notes:

- `nsys_sqlite` provider parameter is `sqlite_path` (not `db_path`).
- `nsys_sqlite_glob` provider parameter is `sqlite_glob`.
- `sqlite_glob` can match any extension (including `*.log`); file content must be SQLite export format.
- For `my_timer` / `torch_profiler` / `module_profiler`, use Python API with `provider_context` object injection instead of CLI-only config.
