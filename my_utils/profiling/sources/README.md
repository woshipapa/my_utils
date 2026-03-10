# sources

Offline Nsight Systems SQLite post-processing parsers and analysis utilities.

## Files

| File | Description |
|---|---|
| `nsys_schema_adapter.py` | SQLite schema auto-detection across nsys versions (`NsightSchema`, `detect_nsys_version`) |
| `nsys_sql_skills.py` | Built-in SQL skill engine (`NsysSqlSkillEngine`) and compute/comm overlap analysis |
| `nsys_sqlite_provider.py` | Main provider (`NsysSqliteMetricsProvider`) 鈥?wraps all analysis APIs |
| `nsys_iterations.py` | NVTX-marker-based training iteration detection (`detect_iterations`) |
| `nsys_mfu.py` | MFU computation helpers (`compute_mfu_single`, `infer_peak_tflops`) |
| `nsys_flat_export.py` | Flat kernel timeline export to JSON/CSV (`export_kernels_flat`) |
| `nsys_analyze.py` | Top-level comprehensive analysis (`analyze_nsys_sqlite`, `analyze_to_markdown`) |
| `nsys_diff.py` | Before/after profile comparison (`diff_nsys_sqlite`, `diff_to_markdown`) |
| `nsys_timeline_html.py` | Static HTML timeline export (`export_timeline_html`) |

---

## nsys_schema_adapter.py

### NsightSchema

Auto-detects SQLite schema produced by `nsys export`. Handles differences across nsys versions (2022 / 2023 / 2024+).

**Key attributes:**

| Attribute | Description |
|---|---|
| `kernel_table` | `CUPTI_ACTIVITY_KIND_KERNEL` (or variant) |
| `runtime_table` | `CUPTI_ACTIVITY_KIND_RUNTIME` |
| `nvtx_table` | `NVTX_EVENTS` |
| `string_table` | `StringIds` or `STRINGIDS` 鈥?maps integer IDs to kernel/nvtx name strings |
| `memcpy_table` | `CUPTI_ACTIVITY_KIND_MEMCPY` |
| `memset_table` | `CUPTI_ACTIVITY_KIND_MEMSET` |
| `sync_table` | `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` |
| `meta_table` | `META_DATA_EXPORT` / `EXPORT_META_DATA` |
| `version` | `NsysVersionInfo` (exporter_version, adapter_family) |

**Version families** (resolved from nsys exporter version string):

| Family | Condition |
|---|---|
| `nsys_2024_plus` | year >= 2024 |
| `nsys_2023` | year == 2023 |
| `nsys_2022` | year == 2022 |
| `generic` | unknown / old |

**Key methods:**

```python
schema = NsightSchema(conn)
schema.table_exists("NVTX_EVENTS")          # bool
schema.columns("CUPTI_ACTIVITY_KIND_KERNEL") # List[str]
schema.resolve_column(table, ["shortName", "demangledName"])  # first match or None
schema.summary()   # dict with tables, meta, version, canonical_tables
```

**globalTid decode:**
```python
pid = (global_tid // 0x1000000) % 0x1000000
tid = global_tid % 0x1000000
```

---

## nsys_sql_skills.py

### NsysSqlSkillEngine

Executes built-in parameterized SQL queries against an nsys SQLite export.
SQL is dynamically constructed to match the detected schema (column names, string table joins, etc.).

```python
engine = NsysSqlSkillEngine(conn)
engine.list_skills()           # List[str]
engine.describe_skill("top_kernels")  # dict with params info
engine.execute("top_kernels", device_id=0, limit=20)  # List[Dict]
```

### Built-in SQL Skills

| # | Skill Name | Category | Required Tables | Key Output Columns |
|---|---|---|---|---|
| 1 | `top_kernels` | kernels | KERNEL | kernel_name, total_ms, avg_ms, invocations |
| 2 | `aggregate_kernels` | kernels | KERNEL | kernel_name, total_ms, avg_ms, min_ms, max_ms, invocations |
| 3 | `nccl_breakdown` | communication | KERNEL | kernel_name, total_ms, avg_ms, min_ms, max_ms, count |
| 4 | `kernel_map` | kernels | KERNEL | start_ns, end_ns, stream_id, correlation_id, kernel_name |
| 5 | `gpu_idle_gaps` | kernels | KERNEL | stream_id, gap_ms, before_kernel, after_kernel |
| 6 | `kernel_launch_overhead` | kernels | KERNEL + RUNTIME | api_ms, kernel_ms, overhead_us |
| 7 | `aggregate_nvtx_ranges` | nvtx | NVTX\_EVENTS | nvtx_name, range_count, total_ms, avg_ms |
| 8 | `nvtx_kernel_map` | nvtx | NVTX\_EVENTS + KERNEL | nvtx_text, kernel_name, start_ns, end_ns |
| 9 | `memcpy_in_window` | memory | MEMCPY | copy_kind, memcpy_count, total_ms |
| 10 | `thread_utilization` | system | COMPOSITE\_EVENTS | global_tid, thread_name, cpu_pct, cpu_cycles |
| 11 | `schema_inspect` | utility | sqlite_master | table_name, column_name, column_type |
| 12 | `memcpy_bandwidth_analysis` | memory | MEMCPY | copy_kind, total_gb, total_ms, avg_gbps, min_gbps, max_gbps |
| 13 | `sync_breakdown` | pipeline | SYNCHRONIZATION | sync_type, count, total_ms, avg_ms, max_ms |
| 14 | `memset_breakdown` | memory | MEMSET | fill_value, count, total_gb, total_ms, avg_gbps |
| 15 | `kernel_occupancy_estimate` | compute | KERNEL | kernel_name, threads_per_block, registersPerThread, total_shared_bytes, occupancy_pct_estimate |
| 16 | `stream_parallelism` | pipeline | KERNEL | max_concurrent_streams, avg_concurrent_streams, pct_time_multi_stream |
| 17 | `nvtx_memcpy_breakdown` | memory | NVTX\_EVENTS + MEMCPY | nvtx_text, copy_kind, total_gb, total_ms, avg_gbps |
| 18 | `nvtx_kernel_sm_detail` | compute | NVTX\_EVENTS + KERNEL | nvtx_text, kernel_name, kind, duration_ms, stream_id, threads_per_block, registersPerThread, total_shared_bytes, occupancy_pct_estimate |
| 19 | `nvtx_ranges_hierarchy` | nvtx | NVTX\_EVENTS | nvtx_text, start_ns, end_ns, depth, parent_nvtx_text, global_tid |

Skills 12-19 are **schema-guarded**: each skill silently absent from `list_skills()` if its required table does not exist in the SQLite export (e.g. older nsys versions without SYNCHRONIZATION or MEMSET tables).

**Common parameters:**

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `device_id` | int | -1 | Most skills; -1 = all devices |
| `limit` | int | 15鈥?0 | Most skills |
| `start_ns` | int | -1 | `kernel_map`, `memcpy_in_window`, `nvtx_kernel_map`; -1 = no filter |
| `end_ns` | int | -1 | Same as start_ns |
| `min_gap_ns` | int | 1\_000\_000 | `gpu_idle_gaps` |
| `bucket_ns` | int | 1\_000\_000 | `stream_parallelism` (time bucket width) |
| `nvtx_text` | str | *(required)* | `nvtx_kernel_sm_detail` 鈥?SQL LIKE pattern, e.g. `%forward%` |
| `top_level_only` | bool | false | `nvtx_ranges_hierarchy`; true means only root ranges |

**New skill details:**

`memcpy_bandwidth_analysis` 鈥?Groups by `copyKind` (1=H2D, 2=D2H, 8=D2D) and reports bandwidth as
`SUM(bytes) / SUM(duration_ns) * 1e9` (avg) and per-transfer MIN/MAX. Identifies PCIe vs NVLink saturation.

`sync_breakdown` 鈥?Aggregates `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` events by `syncType`.
Exposes hidden synchronization overhead such as `cudaDeviceSynchronize` stalls.

`memset_breakdown` 鈥?Groups by fill `value` (0 = zero-init, non-zero = custom fill).
Quantifies the cost of buffer initialization hidden inside training steps.

`kernel_occupancy_estimate` 鈥?Estimates theoretical SM occupancy using:
`occupancy_pct 鈮?MIN(max_warps_per_sm=64, warps_per_block 脳 4) / 64 脳 100`.
Flags kernels with small `threads_per_block` or high register/shared-mem pressure.

`stream_parallelism` 鈥?Divides timeline into `bucket_ns`-wide buckets and counts
`DISTINCT stream_id` per bucket. Reports `pct_time_multi_stream` (fraction of buckets with >1 active stream).

`nvtx_memcpy_breakdown` 鈥?Joins NVTX ranges with MEMCPY rows (`memcpy.start >= nvtx.start AND memcpy.end <= nvtx.end`).
Identifies which training phase (forward / backward / optimizer step) drives the most data movement.

`nvtx_kernel_sm_detail` 鈥?For each NVTX range matching `nvtx_text` (SQL LIKE), lists every kernel that ran **strictly inside** that range (`kernel.start >= nvtx.start AND kernel.end <= nvtx.end`) with full SM launch configuration:
- `kind`: `'compute'` or `'comm'` (name contains `nccl`)
- `threads_per_block`: `blockX 脳 blockY 脳 blockZ`
- `registersPerThread`: register pressure per thread
- `total_shared_bytes`: `staticSharedMemory + dynamicSharedMemory`
- `occupancy_pct_estimate`: `MIN(64, warps_per_block 脳 4) / 64 脳 100`
- `gridX/Y/Z`, `total_blocks`, `localMemoryPerThread`: included when present in the export

Use this skill to audit the launch configuration of every kernel inside a specific training phase, identify low-occupancy kernels, or confirm compute vs comm kernel mix within a range.

`nvtx_ranges_hierarchy` — Lists raw NVTX ranges and annotates nesting:
- depth=0 means root NVTX range.
- depth>0 means nested child range.
- parent_nvtx_text gives direct parent range text.

Use this skill to enumerate all NVTX names including nested parent/child ranges, and inspect hierarchy structure per thread.

### Engine Methods

```python
engine = NsysSqlSkillEngine(conn)

# 鈹€鈹€ SQL skill execution 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
engine.list_skills()                          # List[str] — 19 entries when all tables present
engine.describe_skill("sync_breakdown")       # dict with params metadata
engine.execute("sync_breakdown", device_id=0, limit=50)  # List[Dict]

# 鈹€鈹€ Compute/comm overlap (global window) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
engine.analyze_compute_comm_overlap(device_id=0, start_ns=-1, end_ns=-1)

# 鈹€鈹€ GPU kernel summary 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
engine.summarize_gpu_kernels(device_id=0, top_k=10)

# 鈹€鈹€ Iteration detection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
engine.detect_iterations(marker="sample_0", device_id=0, top_level_only=True)

# 鈹€鈹€ Per-iteration compute/comm breakdown (new) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
engine.analyze_per_iteration_overlap(
    marker="sample_0",
    device_id=0,
    top_level_only=True,
    limit=200,
)
# Returns List[Dict] 鈥?one entry per iteration, all detect_iterations() fields plus:
#   compute_ms, comm_ms, overlap_ms, comm_pct, kernel_count

# 鈹€鈹€ Iteration outlier detection (new) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
engine.detect_iteration_outliers(
    marker="sample_0",
    device_id=0,
    threshold_sigma=2.0,   # flag iterations where |duration - median| > 2蟽
    limit=2000,
)
# Returns:
#   {
#     "stats":    {count, mean_ms, median_ms, std_ms, p95_ms, p99_ms},
#     "outliers": [{iteration, duration_ms, deviation_sigma}, ...]
#   }
```

### Compute/Comm Overlap Analysis

```python
engine.analyze_compute_comm_overlap(device_id=0, start_ns=-1, end_ns=-1)
```

**Classify kernels:** name contains `nccl` (case-insensitive) 鈫?comm; otherwise 鈫?compute.

**Algorithm:** merge overlapping intervals per class, then intersect the two merged sets.

**Returns:**

```json
{
  "compute_only_ms": 1234.5,
  "comm_only_ms": 56.7,
  "overlap_ms": 89.0,
  "compute_total_ms": 1323.5,
  "comm_total_ms": 145.7,
  "overlap_pct_of_comm": 61.08,
  "overlap_pct_of_compute": 6.72
}
```

---

## nsys_iterations.py

### detect_iterations()

Detects training step boundaries using NVTX markers and computes per-iteration kernel statistics.

```python
from my_utils.profiling.sources.nsys_iterations import detect_iterations

iterations = detect_iterations(
    conn,
    schema=schema,          # optional NsightSchema; auto-created if None
    marker="sample_0",      # NVTX text LIKE pattern (auto-wrapped with %)
    device_id=-1,
    start_ns=-1,
    end_ns=-1,
    top_level_only=True,    # skip nested NVTX ranges with same marker
    limit=2000,
)
```

**How marker matching works:**
- If `marker` has no `%`, it is wrapped as `%marker%` (substring match).
- `top_level_only=True` keeps only non-overlapping top-level ranges (no nested duplicates).

**Step/rank extraction from marker text (regex):**
- `step[:=_-](\d+)` or `iter[:=_-](\d+)`
- `rank[:=_-](\d+)`

**Per-iteration output fields:**

| Field | Description |
|---|---|
| `iteration` | Sequential index (0-based) |
| `marker_text` | Full NVTX text of the marker |
| `start_ns` / `end_ns` | NVTX range boundaries (ns) |
| `duration_ms` | NVTX span duration |
| `kernel_count` | Total kernels dispatched in this iteration |
| `nccl_kernel_count` | NCCL kernels only |
| `compute_ms` | Sum of non-NCCL kernel durations |
| `comm_ms` | Sum of NCCL kernel durations |
| `gpu_start_ns` / `gpu_end_ns` | Actual GPU activity window |
| `gpu_duration_ms` | GPU activity span |
| `step` | Extracted step number (if found in marker text) |
| `rank` | Extracted rank number (if found in marker text) |

---

## nsys_mfu.py

### infer_peak_tflops()

```python
peak = infer_peak_tflops("NVIDIA H100 SXM5 80GB", precision="fp16")
# Returns 989.0
```

**GPU peak TFLOPS reference table:**

| GPU | fp16 / bf16 (TFLOPS) | fp32 (TFLOPS) |
|---|---|---|
| H100 / H200 | 989 | 67 |
| H800 | 989 | 67 |
| A100 / A800 | 312 | 19.5 |
| L40S | 362 | 91 |
| L40 | 181 | 91 |

GPU name matching is substring-based (case-insensitive). Returns `None` if unrecognized.

### compute_mfu_single()

```python
result = compute_mfu_single(
    step_time_s=1.2,
    model_flops_per_step=1e15,
    peak_tflops=989.0,
)
```

**Formula:**
```
achieved_tflops = (model_flops_per_step / step_time_s) / 1e12
mfu_pct         = 100.0 * achieved_tflops / peak_tflops
```

**Returns:**
```json
{
  "step_time_s": 1.2,
  "model_flops_per_step": 1e15,
  "peak_tflops": 989.0,
  "achieved_model_tflops": 833.3333,
  "mfu_pct": 84.2602
}
```

### compute_mfu_compare()

Compare MFU before and after an optimization:

```python
result = compute_mfu_compare(
    step_time_before_s=1.5,
    step_time_after_s=1.2,
    model_flops_per_step=1e15,
    peak_tflops=989.0,
)
# Returns: {"before": {...}, "after": {...}, "delta_mfu_pct": ..., "delta_achieved_tflops": ...}
```

---

## nsys_analyze.py

### analyze_nsys_sqlite()

Top-level comprehensive analysis. Calls all sub-analyses and returns a single result dict.

```python
from my_utils.profiling.sources.nsys_analyze import analyze_nsys_sqlite, analyze_to_markdown

result = analyze_nsys_sqlite(
    sqlite_path="train_rank0.sqlite",
    device_id=0,
    start_ns=-1,
    end_ns=-1,
    top_k=20,
    iteration_marker="sample_0",
    model_flops_per_step=1e15,   # optional; enables MFU
    peak_tflops=None,             # optional; auto-inferred from GPU name if None
    peak_precision="fp16",
    limit=500000,
)
```

**step_time resolution order for MFU:**
1. Median of `iterations[*].duration_ms` (preferred 鈥?excludes warm-up)
2. Fallback: `summary.timing.span_ms` (full profiled window)

**Result structure:**

```python
{
    "sqlite_path": str,
    "device_id": int,
    "window": {"start_ns": int, "end_ns": int},
    "schema": {...},          # NsightSchema.summary()
    "summary": {              # from summarize_gpu_kernels()
        "timing": {
            "span_ms": float,
            "busy_ms": float,
            "idle_ms": float,
            "utilization_pct": float,
            "sum_kernel_ms": float,
        },
        "top_kernels": [...],
        "stream_count": int,
        "kernel_rows": int,
    },
    "overlap": {              # from analyze_compute_comm_overlap()
        "compute_only_ms": float,
        "comm_only_ms": float,
        "overlap_ms": float,
        "compute_total_ms": float,
        "comm_total_ms": float,
        "overlap_pct_of_comm": float,
        "overlap_pct_of_compute": float,
    },
    "top_kernels": [...],        # from SQL skill "top_kernels"
    "nccl_breakdown": [...],     # from SQL skill "nccl_breakdown"
    "iterations": [...],         # from detect_iterations()
    "sync_breakdown": [...],     # from SQL skill "sync_breakdown" ([] if table absent)
    "memcpy_bandwidth": [...],   # from SQL skill "memcpy_bandwidth_analysis" ([] if table absent)
    "mfu": {                     # None if model_flops_per_step not provided
        "step_time_s": float,
        "achieved_model_tflops": float,
        "mfu_pct": float,
    },
    "warnings": [str, ...],
}
```

**Markdown report:**
```python
md = analyze_to_markdown(result)
```

---

## nsys_diff.py

### diff_nsys_sqlite()

Compares two nsys SQLite profiles (e.g., before/after an optimization).

```python
from my_utils.profiling.sources.nsys_diff import diff_nsys_sqlite, diff_to_markdown

diff = diff_nsys_sqlite(
    before_sqlite="run_a.sqlite",
    after_sqlite="run_b.sqlite",
    device_id=0,
    start_ns=-1,
    end_ns=-1,
    top_k=20,
)
```

**Result structure:**

```python
{
    "before_sqlite": str,
    "after_sqlite": str,
    "device_id": int,
    "window": {"start_ns": int, "end_ns": int},
    "summary": {
        "utilization_pct": {"before": float, "after": float, "delta": float},
        "overlap_ms":       {"before": float, "after": float, "delta": float},
    },
    "kernel_diff_top": [   # sorted by |delta total_ms| DESC
        {"name": str, "before": float, "after": float, "delta": float, "ratio": float},
        ...
    ],
    "nvtx_diff_top": [...],   # same structure, keyed by nvtx_name
}
```

**Markdown report:**
```python
md = diff_to_markdown(diff)
```

---

## nsys_flat_export.py

### export_kernels_flat()

Exports all kernel timeline rows as a flat JSON or CSV file. Optionally annotates each row with which training iteration it belongs to.

```python
from my_utils.profiling.sources.nsys_flat_export import export_kernels_flat

output_path = export_kernels_flat(
    "train_rank0.sqlite",
    output_path="./kernels_flat.csv",
    fmt="csv",              # "json" or "csv"
    device_id=0,
    start_ns=-1,
    end_ns=-1,
    limit=500000,
    attach_iteration=True,  # annotate with iteration index
    iteration_marker="sample_0",
)
```

**Per-row fields:**

| Field | Description |
|---|---|
| `device_id` | CUDA device ID |
| `stream_id` | CUDA stream |
| `correlation_id` | Correlates with runtime API call |
| `kernel_name` | Demangled kernel name |
| `start_ns` / `end_ns` | Kernel boundaries (ns) |
| `duration_us` | Duration in microseconds |
| `duration_ms` | Duration in milliseconds |
| `is_nccl` | True if kernel name contains `nccl` |
| `iteration` | Iteration index (-1 if `attach_iteration=False` or unmatched) |

---

## NsysSqliteMetricsProvider

The main high-level provider that wraps all of the above into a unified `BaseMetricsProvider` interface.

```python
from my_utils.profiling.sources.nsys_sqlite_provider import NsysSqliteMetricsProvider

provider = NsysSqliteMetricsProvider(
    sqlite_path="train_rank0.sqlite",
    include_osrt=False,     # include OS runtime API events
)
```

### Provider helper APIs

```python
provider.describe_schema()                        # schema version + canonical tables
provider.list_sql_skills()                        # List[str] of skill names
provider.describe_sql_skills()                    # List[Dict] with params metadata
provider.run_sql_skill("top_kernels", device_id=0, limit=20)  # List[Dict]
provider.summarize_gpu_kernels(device_id=0, top_k=20)
provider.analyze_compute_comm_overlap(device_id=0)
provider.detect_iterations(marker="sample_0", device_id=0)
provider.compute_mfu(
    model_flops_per_step=1e15,
    peak_tflops=989.0,
    peak_precision="fp16",
    iteration_marker="sample_0",
)
```

### Standard MetricsProvider interface

```python
# Register with MetricsCollector for unified pipeline
from my_utils.profiling.pipeline.metrics_collector import MetricsCollector

collector = MetricsCollector(output_dir="./metrics_out")
collector.register_provider(NsysSqliteMetricsProvider("train_rank0.sqlite"))
collector.collect(step=0)
report = collector.analyze()
```

---

## Quick Reference: Analysis Pipeline

```
nsys profile ... 鈫?.nsys-rep
nsys export --sqlite 鈫?.sqlite
        鈹?        鈹溾攢 NsightSchema          (schema + version detection)
        鈹溾攢 NsysSqlSkillEngine    (SQL queries: top_kernels, nccl_breakdown, ...)
        鈹溾攢 detect_iterations     (NVTX marker 鈫?per-step stats)
        鈹溾攢 compute_mfu_single    (MFU %)
        鈹溾攢 analyze_compute_comm_overlap  (overlap_ms, overlap_pct)
        鈹溾攢 summarize_gpu_kernels (busy_ms, idle_ms, utilization_pct)
        鈹?        鈹溾攢 analyze_nsys_sqlite   (all-in-one result dict)
        鈹溾攢 diff_nsys_sqlite      (before vs after comparison)
        鈹溾攢 export_kernels_flat   (flat JSON/CSV timeline)
        鈹斺攢 export_timeline_html  (static HTML timeline)
```

**All times are in nanoseconds internally; all reported values are in milliseconds.**
**Compute vs comm classification: kernel name contains `nccl` (case-insensitive) 鈫?comm.**

---

## CLI Subcommand Selection Guide

### All nsys subcommands at a glance

| Subcommand | One-line purpose | Input | Output |
|---|---|---|---|
| `nsys-sql-skill` | Run any one of 19 SQL skills with custom params | 1 sqlite | JSON rows |
| `nsys-analyze` | All-in-one report: kernel/NCCL/overlap/sync/bandwidth/MFU | 1 sqlite | JSON or Markdown |
| `nsys-iter-overlap` | Per-step compute / comm / overlap breakdown (interval-merge) | 1 sqlite | JSON list |
| `nsys-iter-outliers` | Step duration statistics + 蟽-based anomaly detection | 1 sqlite | JSON {stats, outliers} |
| `nsys-export` | Raw kernel timeline rows, optionally annotated with step index | 1 sqlite | JSON or CSV |
| `nsys-diff` | Before-vs-after delta on utilization, overlap, kernel times | 2 sqlite | JSON or Markdown |
| `nsys-timeline-html` | Interactive HTML timeline for visual inspection | 1 sqlite | HTML |

---

### Decision tree

```
Have a .sqlite file from nsys export?
鈹?鈹溾攢 Want a single comprehensive report?
鈹?  鈹斺攢鈻?nsys-analyze [--format markdown] [--model-flops-per-step N --peak-tflops N]
鈹?鈹溾攢 Want one specific metric with custom parameters?
鈹?  鈹斺攢鈻?nsys-sql-skill --skill <name> --param key=val [--param ...]
鈹?       鈹?鈹?       鈹溾攢 Compute hotspots      鈫?top_kernels / aggregate_kernels
鈹?       鈹溾攢 NCCL collectives      鈫?nccl_breakdown
鈹?       鈹溾攢 Sync stalls           鈫?sync_breakdown
鈹?       鈹溾攢 PCIe / NVLink BW      鈫?memcpy_bandwidth_analysis
鈹?       鈹溾攢 Buffer init cost      鈫?memset_breakdown
鈹?       鈹溾攢 SM occupancy          鈫?kernel_occupancy_estimate
鈹?       鈹溾攢 Multi-stream overlap  鈫?stream_parallelism
鈹?       鈹溾攢 Per-phase data moves  鈫?nvtx_memcpy_breakdown
鈹?       鈹溾攢 CPU thread cost       鈫?thread_utilization
鈹?       鈹溾攢 Pipeline bubbles      鈫?gpu_idle_gaps
鈹?       鈹溾攢 Launch latency        鈫?kernel_launch_overhead
鈹?       鈹斺攢 Schema debug          鈫?schema_inspect
鈹?鈹溾攢 Want per-step compute / comm balance (parallel-stream-correct)?
鈹?  鈹斺攢鈻?nsys-iter-overlap --iteration-marker sample_0
鈹?鈹溾攢 Want to find slow or anomalous training steps?
鈹?  鈹斺攢鈻?nsys-iter-outliers --sigma 2.0 --iteration-marker sample_0
鈹?鈹溾攢 Want raw kernel rows for custom post-processing?
鈹?  鈹斺攢鈻?nsys-export --format csv [--attach-iteration]
鈹?鈹溾攢 Want to visually inspect the kernel timeline?
鈹?  鈹斺攢鈻?nsys-timeline-html --output timeline.html
鈹?鈹斺攢 Have two sqlite files and want to measure an optimization?
    鈹斺攢鈻?nsys-diff --before-sqlite A.sqlite --after-sqlite B.sqlite
```

---

### Overlap and differences between subcommands

Several subcommands share underlying computations. Understanding where they differ
prevents misreading results.

#### `nsys-analyze` vs `nsys-sql-skill`

`nsys-analyze` internally runs six SQL skills with fixed parameters:

| Skill run inside nsys-analyze | Fixed params |
|---|---|
| `top_kernels` | `limit = top_k` (default 10) |
| `nccl_breakdown` | `limit = top_k` |
| `sync_breakdown` | `limit = 50` |
| `memcpy_bandwidth_analysis` | no limit |
| `summarize_gpu_kernels` (uses `top_kernels` internally) | `top_k` rows |
| `analyze_compute_comm_overlap` | full window |

Use `nsys-sql-skill` when you need a **different limit, a time window filter, or
any of the 11 skills not included in `nsys-analyze`** (e.g. `kernel_occupancy_estimate`,
`stream_parallelism`, `memset_breakdown`, `nvtx_memcpy_breakdown`, `gpu_idle_gaps`, etc.).

#### `nsys-analyze` iterations vs `nsys-iter-overlap` 鈥?same field name, different algorithm

Both expose `compute_ms` and `comm_ms` per iteration, but they are computed differently:

| | `nsys-analyze` 鈫?`iterations[].compute_ms` | `nsys-iter-overlap` 鈫?`[].compute_ms` |
|---|---|---|
| Source | `detect_iterations()` in `nsys_iterations.py` | `analyze_per_iteration_overlap()` |
| Formula | `SUM(kernel.duration)` for all non-NCCL kernels | `covered_ns(merge_intervals(non-NCCL))` |
| Multi-stream parallel kernels | **Double-counted** (each kernel's full duration added independently) | **Correct** 鈥?overlapping kernels counted once (wall-clock) |
| Has `overlap_ms` field? | No | Yes 鈥?`intersect(compute_intervals, comm_intervals)` |

Example with two concurrent streams:

```
stream 7: [gemm   0鈥?ms ]
stream 8: [nccl     4鈥?0ms]

detect_iterations:              analyze_per_iteration_overlap:
  compute_ms = 8               compute_ms = 8   (wall-clock, merged)
  comm_ms    = 6               comm_ms    = 6
  total      = 14  鈫?inflated  overlap_ms = 4   鈫?correct
```

**Rule of thumb:** if your training uses `--overlap-grad-reduce` or any asynchronous
NCCL, use `nsys-iter-overlap` for compute/comm numbers. `nsys-analyze` iterations are
sufficient for fully-sequential pipelines.

#### `nsys-analyze` global overlap vs `nsys-iter-overlap` per-step overlap

| | `nsys-analyze` 鈫?`overlap` key | `nsys-iter-overlap` |
|---|---|---|
| Granularity | Entire profiled window | Per training step |
| Use case | Overall efficiency score | Find which step has worst overlap |

They are complementary. Run `nsys-analyze` first; if `overlap_pct_of_comm` is low,
use `nsys-iter-overlap` to locate the specific steps causing the issue.

#### `nsys-iter-overlap` vs `nsys-iter-outliers`

Both call `detect_iterations()` internally. After that they diverge:

- `nsys-iter-overlap` 鈥?for each iteration, fetches all kernel intervals and computes
  compute/comm/overlap wall-clock durations. Expensive for many iterations.
- `nsys-iter-outliers` 鈥?uses only the `duration_ms` field already returned by
  `detect_iterations()`. No additional per-iteration DB query. Fast.

Run `nsys-iter-outliers` first to identify anomalous step indices, then use
`nsys-iter-overlap` on a narrowed window (`--start-ns` / `--end-ns`) to drill into those steps.

#### `nsys-diff` vs `nsys-analyze`

`nsys-diff` internally runs the equivalent of `nsys-analyze` twice (once per file) and
produces delta values. It does **not** include sync_breakdown, memcpy_bandwidth, or
per-iteration data. Use `nsys-analyze` separately on each file when you need those fields.

---

### When `compute_ms` from `nsys-analyze` 鈮?`nsys-iter-overlap`

If you see discrepancies between the two, the cause is always one of:

1. **Multi-stream parallelism** 鈥?kernels on different streams overlap in wall-clock
   time. `detect_iterations` sums raw durations; `analyze_per_iteration_overlap` merges intervals.
2. **NVTX window mismatch** 鈥?`nsys-analyze` uses the global `--start-ns`/`--end-ns` window;
   `nsys-iter-overlap` uses each NVTX range boundary per iteration.
3. **`top_level_only`** 鈥?`nsys-analyze` always sets `top_level_only=True`;
   `nsys-iter-overlap` exposes `--include-nested` to override this.

