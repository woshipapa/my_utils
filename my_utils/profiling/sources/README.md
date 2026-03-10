# sources

Offline Nsight Systems SQLite post-processing parsers and analysis utilities.

## Files

| File | Description |
|---|---|
| `nsys_schema_adapter.py` | SQLite schema auto-detection across nsys versions (`NsightSchema`, `detect_nsys_version`) |
| `nsys_sql_skills.py` | Built-in SQL skill engine (`NsysSqlSkillEngine`) and compute/comm overlap analysis |
| `nsys_sqlite_provider.py` | Main provider (`NsysSqliteMetricsProvider`) — wraps all analysis APIs |
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
| `string_table` | `StringIds` or `STRINGIDS` — maps integer IDs to kernel/nvtx name strings |
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

Skills 12–17 are **schema-guarded**: each skill silently absent from `list_skills()` if its required table does not exist in the SQLite export (e.g. older nsys versions without SYNCHRONIZATION or MEMSET tables).

**Common parameters:**

| Parameter | Type | Default | Applies to |
|---|---|---|---|
| `device_id` | int | -1 | Most skills; -1 = all devices |
| `limit` | int | 15–50 | Most skills |
| `start_ns` | int | -1 | `kernel_map`, `memcpy_in_window`, `nvtx_kernel_map`; -1 = no filter |
| `end_ns` | int | -1 | Same as start_ns |
| `min_gap_ns` | int | 1\_000\_000 | `gpu_idle_gaps` |
| `bucket_ns` | int | 1\_000\_000 | `stream_parallelism` (time bucket width) |

**New skill details:**

`memcpy_bandwidth_analysis` — Groups by `copyKind` (1=H2D, 2=D2H, 8=D2D) and reports bandwidth as
`SUM(bytes) / SUM(duration_ns) * 1e9` (avg) and per-transfer MIN/MAX. Identifies PCIe vs NVLink saturation.

`sync_breakdown` — Aggregates `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` events by `syncType`.
Exposes hidden synchronization overhead such as `cudaDeviceSynchronize` stalls.

`memset_breakdown` — Groups by fill `value` (0 = zero-init, non-zero = custom fill).
Quantifies the cost of buffer initialization hidden inside training steps.

`kernel_occupancy_estimate` — Estimates theoretical SM occupancy using:
`occupancy_pct ≈ MIN(max_warps_per_sm=64, warps_per_block × 4) / 64 × 100`.
Flags kernels with small `threads_per_block` or high register/shared-mem pressure.

`stream_parallelism` — Divides timeline into `bucket_ns`-wide buckets and counts
`DISTINCT stream_id` per bucket. Reports `pct_time_multi_stream` (fraction of buckets with >1 active stream).

`nvtx_memcpy_breakdown` — Joins NVTX ranges with MEMCPY rows (`memcpy.start >= nvtx.start AND memcpy.end <= nvtx.end`).
Identifies which training phase (forward / backward / optimizer step) drives the most data movement.

### Engine Methods

```python
engine = NsysSqlSkillEngine(conn)

# ── SQL skill execution ──────────────────────────────────────────────────────
engine.list_skills()                          # List[str] — 17 entries when all tables present
engine.describe_skill("sync_breakdown")       # dict with params metadata
engine.execute("sync_breakdown", device_id=0, limit=50)  # List[Dict]

# ── Compute/comm overlap (global window) ────────────────────────────────────
engine.analyze_compute_comm_overlap(device_id=0, start_ns=-1, end_ns=-1)

# ── GPU kernel summary ───────────────────────────────────────────────────────
engine.summarize_gpu_kernels(device_id=0, top_k=10)

# ── Iteration detection ──────────────────────────────────────────────────────
engine.detect_iterations(marker="sample_0", device_id=0, top_level_only=True)

# ── Per-iteration compute/comm breakdown (new) ───────────────────────────────
engine.analyze_per_iteration_overlap(
    marker="sample_0",
    device_id=0,
    top_level_only=True,
    limit=200,
)
# Returns List[Dict] — one entry per iteration, all detect_iterations() fields plus:
#   compute_ms, comm_ms, overlap_ms, comm_pct, kernel_count

# ── Iteration outlier detection (new) ────────────────────────────────────────
engine.detect_iteration_outliers(
    marker="sample_0",
    device_id=0,
    threshold_sigma=2.0,   # flag iterations where |duration - median| > 2σ
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

**Classify kernels:** name contains `nccl` (case-insensitive) → comm; otherwise → compute.

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
1. Median of `iterations[*].duration_ms` (preferred — excludes warm-up)
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
nsys profile ... → .nsys-rep
nsys export --sqlite → .sqlite
        │
        ├─ NsightSchema          (schema + version detection)
        ├─ NsysSqlSkillEngine    (SQL queries: top_kernels, nccl_breakdown, ...)
        ├─ detect_iterations     (NVTX marker → per-step stats)
        ├─ compute_mfu_single    (MFU %)
        ├─ analyze_compute_comm_overlap  (overlap_ms, overlap_pct)
        ├─ summarize_gpu_kernels (busy_ms, idle_ms, utilization_pct)
        │
        ├─ analyze_nsys_sqlite   (all-in-one result dict)
        ├─ diff_nsys_sqlite      (before vs after comparison)
        ├─ export_kernels_flat   (flat JSON/CSV timeline)
        └─ export_timeline_html  (static HTML timeline)
```

**All times are in nanoseconds internally; all reported values are in milliseconds.**
**Compute vs comm classification: kernel name contains `nccl` (case-insensitive) → comm.**
