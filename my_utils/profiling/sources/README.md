# sources

Offline Nsight Systems SQLite post-processing parsers and analysis utilities.

## Files

| File | Description |
|---|---|
| `nsys_schema_adapter.py` | SQLite schema auto-detection across nsys versions (`NsightSchema`, `detect_nsys_version`) |
| `nsys_sql_skills.py` | Built-in SQL skill engine (`NsysSqlSkillEngine`) and overlap utilities |
| `nsys_sqlite_provider.py` | Main provider (`NsysSqliteMetricsProvider`) that wraps analysis APIs |
| `nsys_iterations.py` | NVTX-marker-based iteration detection (`detect_iterations`) |
| `nsys_mfu.py` | MFU helpers (`compute_mfu_single`, `infer_peak_tflops`) |
| `nsys_flat_export.py` | Flat kernel timeline export to JSON/CSV (`export_kernels_flat`) |
| `nsys_analyze.py` | All-in-one analysis (`analyze_nsys_sqlite`, `analyze_to_markdown`) |
| `nsys_diff.py` | Before/after comparison (`diff_nsys_sqlite`, `diff_to_markdown`) |
| `nsys_timeline_html.py` | Static HTML timeline export (`export_timeline_html`) |

---

## nsys_schema_adapter.py

### NsightSchema

Auto-detects SQLite schema produced by `nsys export` and handles table/column variants across exporter versions.

Key attributes:

| Attribute | Description |
|---|---|
| `kernel_table` | `CUPTI_ACTIVITY_KIND_KERNEL` (or variant) |
| `runtime_table` | `CUPTI_ACTIVITY_KIND_RUNTIME` |
| `nvtx_table` | `NVTX_EVENTS` |
| `string_table` | `StringIds` / `STRINGIDS` for ID->string mapping |
| `memcpy_table` | `CUPTI_ACTIVITY_KIND_MEMCPY` |
| `memset_table` | `CUPTI_ACTIVITY_KIND_MEMSET` |
| `sync_table` | `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` |
| `metrics_table` | GPU sampling metrics table (`GPU_METRICS` / `CUPTI_ACTIVITY_KIND_GPU_METRIC` / variants) |
| `metrics_timestamp_col` | timestamp column alias (`timestamp` / `start` / `time`) |
| `metrics_id_col` | metric id alias (`metricId` / `nameId` / `eventId`) |
| `metrics_value_col` | metric value alias (`value` / `metricValue` / `val`) |
| `meta_table` | `META_DATA_EXPORT` / `EXPORT_META_DATA` |
| `version` | `NsysVersionInfo` (`exporter_version`, `adapter_family`) |

Version families:

| Family | Condition |
|---|---|
| `nsys_2024_plus` | year >= 2024 |
| `nsys_2023` | year == 2023 |
| `nsys_2022` | year == 2022 |
| `generic` | unknown / old |

Example:

```python
schema = NsightSchema(conn)
schema.table_exists("NVTX_EVENTS")
schema.columns("CUPTI_ACTIVITY_KIND_KERNEL")
schema.resolve_column("CUPTI_ACTIVITY_KIND_KERNEL", ["shortName", "demangledName"])
schema.summary()
```

`globalTid` decode:

```python
pid = (global_tid // 0x1000000) % 0x1000000
tid = global_tid % 0x1000000
```

---

## nsys_sql_skills.py

### NsysSqlSkillEngine

Executes built-in parameterized SQL queries against an nsys SQLite export. SQL is built from detected schema aliases.

```python
engine = NsysSqlSkillEngine(conn)
engine.list_skills()
engine.describe_skill("top_kernels")
engine.execute("top_kernels", device_id=0, limit=20)
```

### Built-in SQL skills

| # | Skill | Category | Required Tables | Key Output |
|---|---|---|---|---|
| 1 | `top_kernels` | kernels | KERNEL | kernel_name, total_ms, avg_ms, invocations |
| 2 | `aggregate_kernels` | kernels | KERNEL | kernel_name, total/avg/min/max ms, invocations |
| 3 | `nccl_breakdown` | communication | KERNEL | NCCL kernel aggregates |
| 4 | `kernel_map` | kernels | KERNEL | start_ns, end_ns, stream_id, correlation_id, kernel_name |
| 5 | `gpu_idle_gaps` | kernels | KERNEL | stream gap analysis |
| 6 | `kernel_launch_overhead` | kernels | KERNEL + RUNTIME | api_ms, kernel_ms, overhead_us |
| 7 | `aggregate_nvtx_ranges` | nvtx | NVTX_EVENTS | nvtx_name, range_count, total_ms, avg_ms |
| 8 | `nvtx_kernel_map` | nvtx | NVTX_EVENTS + RUNTIME + KERNEL | launch-attribution map |
| 9 | `memcpy_in_window` | memory | MEMCPY | copy_kind, count, total_ms |
| 10 | `thread_utilization` | system | COMPOSITE_EVENTS | global_tid, thread_name, cpu_pct |
| 11 | `schema_inspect` | utility | sqlite_master | table/column metadata |
| 12 | `gpu_metrics_aggregate` | metrics | GPU_METRICS + StringIds | metric_name, sample_count, avg/min/max value |
| 13 | `memcpy_bandwidth_analysis` | memory | MEMCPY | total_gb, total_ms, avg/min/max gbps |
| 14 | `sync_breakdown` | pipeline | SYNCHRONIZATION | sync_type, count, total/avg/max ms |
| 15 | `memset_breakdown` | memory | MEMSET | fill_value, total_gb, total_ms, avg_gbps |
| 16 | `kernel_occupancy_estimate` | compute | KERNEL | raw launch metrics (threads_per_block, registersPerThread, static_shared_bytes, dynamic_shared_bytes, total_shared_bytes); occupancy_pct_estimate uses sqlite theoretical occupancy when available |
| 17 | `stream_parallelism` | pipeline | KERNEL | bucket-based concurrent stream stats (cross-bucket kernel expansion) |
| 18 | `nvtx_memcpy_breakdown` | memory | NVTX_EVENTS + MEMCPY | nvtx_text + memcpy aggregates |
| 19 | `nvtx_kernel_sm_detail` | compute | NVTX_EVENTS + KERNEL | per-kernel launch config in NVTX range, including static/dynamic shared memory; occupancy_pct_estimate uses sqlite theoretical occupancy when available |
| 20 | `nvtx_ranges_hierarchy` | nvtx | NVTX_EVENTS | raw NVTX rows; hierarchy derived in Python O(N) |

Skills are schema-guarded and appear only when required tables/columns exist.

Common params:

| Param | Type | Default | Applies |
|---|---|---|---|
| `device_id` | int | -1 | most GPU skills |
| `limit` | int | varies | most skills |
| `start_ns` / `end_ns` | int | -1 | windowed skills |
| `min_gap_ns` | int | 1_000_000 | `gpu_idle_gaps` |
| `bucket_ns` | int | 1_000_000 | `stream_parallelism` |
| `metric_name_like` | str | `%` | `gpu_metrics_aggregate` |
| `nvtx_text` | str | `%` or required | NVTX text filter skills |
| `top_level_only` | bool | false | `nvtx_ranges_hierarchy` |

### Occupancy notes (H100)

For occupancy fields:
- `occupancy_pct_estimate` comes from sqlite theoretical occupancy columns when present.
- if sqlite does not provide that column, `occupancy_pct_estimate` is `NULL`.
- H100 helper execution (`execute_*_h100`) keeps `occupancy_pct_estimate` and additionally appends
  `occupancy_pct_h100_estimate`, so you can compare sqlite-reported theoretical occupancy vs our strict H100 calculation.

Use Python helper for H100(sm_90):

```python
from my_utils.profiling.sources.nsys_sql_skills import calculate_h100_occupancy
```

Convenience engine APIs:

```python
engine.execute_kernel_occupancy_estimate_h100(device_id=0, limit=50)
engine.execute_nvtx_kernel_sm_detail_h100(nvtx_text="%forward%", device_id=0)
```

These add `occupancy_pct_h100_estimate` per row.

### Stream parallelism note

`stream_parallelism` now expands each kernel over every covered bucket (`start_bucket..end_bucket`) instead of assigning only by start time. This avoids undercounting for long kernels.

### NVTX hierarchy note

`nvtx_ranges_hierarchy` no longer uses O(N^2) SQL self-joins. It fetches sorted raw rows and builds parent/child with an O(N) per-thread stack in Python.

### Engine methods

```python
engine.list_skills()
engine.describe_skills()
engine.execute("top_kernels", device_id=0, limit=20)
engine.execute_kernel_occupancy_estimate_h100(device_id=0, limit=50)
engine.execute_nvtx_kernel_sm_detail_h100(nvtx_text="%sample_0%", device_id=0)

engine.analyze_compute_comm_overlap(device_id=0)
engine.summarize_gpu_kernels(device_id=0, top_k=10)
engine.detect_iterations(marker="sample_0", device_id=0)
engine.analyze_per_iteration_overlap(marker="sample_0", device_id=0)
engine.detect_iteration_outliers(marker="sample_0", device_id=0, threshold_sigma=2.0)
```

---

## nsys_analyze.py

`analyze_nsys_sqlite()` provides an all-in-one summary:
- schema info
- gpu summary (span/busy/idle/utilization)
- compute/comm overlap
- top kernels, NCCL breakdown
- iteration stats
- sync and memcpy bandwidth breakdown
- optional MFU

MFU step-time priority:
1. median `iterations[*].duration_ms`
2. fallback `summary.timing.span_ms`

---

## nsys_diff.py

`diff_nsys_sqlite()` compares two SQLite profiles and reports delta for:
- utilization
- overlap
- top kernel deltas
- top NVTX deltas

---

## nsys_flat_export.py

`export_kernels_flat()` exports kernel timeline rows (JSON/CSV), optionally attaching iteration index.

---

## NsysSqliteMetricsProvider

High-level provider API:

```python
provider = NsysSqliteMetricsProvider("train_rank0.sqlite")
provider.describe_schema()
provider.list_sql_skills()
provider.describe_sql_skills()
provider.run_sql_skill("top_kernels", device_id=0, limit=20)
provider.summarize_gpu_kernels(device_id=0, top_k=20)
provider.analyze_compute_comm_overlap(device_id=0)
provider.detect_iterations(marker="sample_0", device_id=0)
provider.compute_mfu(model_flops_per_step=1e15, peak_tflops=989.0, precision="fp16")
```

It can also be registered in `MetricsCollector` as a standard provider.

---

## CLI quick map

| Subcommand | Purpose |
|---|---|
| `nsys-sql-skill` | run one SQL skill with params |
| `nsys-analyze` | all-in-one analysis |
| `nsys-iter-overlap` | per-iteration compute/comm/overlap |
| `nsys-iter-outliers` | iteration outlier detection |
| `nsys-export` | flat kernel export |
| `nsys-diff` | before/after diff |
| `nsys-timeline-html` | static timeline html |

Tip:
- use `nsys-analyze` first for overview
- use `nsys-sql-skill` for deep dive
- use `nsys-iter-overlap` when stream overlap matters
- for `kernel_occupancy_estimate` / `nvtx_kernel_sm_detail`, CLI supports `--occupancy-arch auto|h100|none` (default `auto`)
  to attach `occupancy_pct_h100_estimate` when GPU is H100.
- if a skill is unavailable for current sqlite schema (for example missing GPU metrics table),
  `nsys-sql-skill` now prints explicit reason/hint instead of failing silently.

---

All internal timestamps are nanoseconds. Most report values are milliseconds.
Compute vs comm classification uses kernel name containing `nccl` (case-insensitive).
