# Nsight Systems SQLite Provider Complete Guide

This document describes how to use and understand:

- `my_utils.profiling.nsys_sqlite_provider.NsysSqliteMetricsProvider`

It covers:

1. End-to-end usage with `MetricsCollector`
2. Internal parsing flow
3. What can be analyzed from SQLite
4. Schema/version compatibility
5. SQL snippets and troubleshooting

## 1. Scope and Output Model

`NsysSqliteMetricsProvider` reads Nsight Systems SQLite export and emits normalized `MetricEvent` records:

- `name`: canonical metric name (`latency.*`, `memory.*`, `io.*`, `compute.*`, `calls.*`)
- `value`, `unit`
- `tags`: rich dimensions (`pid`, `tid`, `stream`, `kernel`, `runtime_api`, `step`, `rank`, and schema tags)

These records can then be consumed by:

- `MetricsCollector` (persisting events)
- `MetricsAnalyzer` (bottleneck/memory/variance/anomaly findings)
- `MetricsReportRenderer` (JSON/Markdown/HTML reports)

Coverage strategy in this implementation:

- Explicit parsing for core CUPTI/NVTX/memory/network/PMU/UM tables
- Dynamic parsing for `GENERIC_EVENTS` JSON numeric fields
- Automatic fallback parsing for unknown future tables that expose `start/end`

## 2. End-to-End Usage

### 2.1 Generate an `.nsys-rep`

Example profile command:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --output=./logs/nsys/train_rank_%q{RANK} \
  python train.py
```

### 2.2 Export SQLite

```bash
nsys export \
  --type sqlite \
  --output ./logs/nsys/train_rank_0.sqlite \
  ./logs/nsys/train_rank_0.nsys-rep
```

### 2.3 Read with provider

```python
from my_utils.profiling import MetricsCollector, NsysSqliteMetricsProvider

collector = MetricsCollector(output_dir="./metrics")
collector.register_provider(
    NsysSqliteMetricsProvider(
        sqlite_path="./logs/nsys/train_rank_0.sqlite",
        include_runtime=True,
        include_kernels=True,
        include_memcpy=True,
        include_memset=True,
        include_sync=True,
        include_nvtx=True,
        include_memory_usage=True,
        include_gpu_metrics=True,
        include_generic_events=True,
        include_network_metrics=True,
        include_pmu_metrics=True,
        include_cuda_um=True,
        include_auto_duration_tables=True,
        include_osrt=False,
        parse_step_from_nvtx=True,
    )
)

# For offline SQLite, one collect is usually enough.
collector.collect(step=100)
report = collector.analyze()
collector.export_report(fmt="json")
collector.export_report(fmt="markdown")
collector.export_report(fmt="html")
```

### 2.4 Direct provider usage (debug mode)

```python
from my_utils.profiling.nsys_sqlite_provider import NsysSqliteMetricsProvider

provider = NsysSqliteMetricsProvider("./logs/nsys/train_rank_0.sqlite")
events = provider.get_metrics()
print("events:", len(events))
print("example:", events[0].name, events[0].value, events[0].unit, events[0].tags)
```

### 2.5 Run built-in SQL skills (nsys-ai style)

This project now includes a built-in SQL skill engine inspired by `nsys-ai`:

- `top_kernels`
- `aggregate_kernels`
- `aggregate_nvtx_ranges`
- `memcpy_in_window`
- `kernel_map`
- `gpu_idle_gaps`
- `kernel_launch_overhead`

Python API:

```python
from my_utils.profiling import NsysSqliteMetricsProvider

p = NsysSqliteMetricsProvider("./logs/nsys/train_rank_0.sqlite")
print(p.list_sql_skills())
print(p.describe_sql_skills())  # includes params/defaults

rows = p.run_sql_skill("top_kernels", device_id=0, limit=20)
for row in rows[:3]:
    print(row)

# overlap summary (compute vs NCCL kernels)
print(p.analyze_compute_comm_overlap(device_id=0))

# GPU kernel timeline summary
print(p.summarize_gpu_kernels(device_id=0, top_k=10))
```

CLI:

```bash
# list skills and parameter schema
myutils-profile nsys-sql-skill \
  --sqlite ./logs/nsys/train_rank_0.sqlite \
  --list-skills \
  --pretty

# run one skill
myutils-profile nsys-sql-skill \
  --sqlite ./logs/nsys/train_rank_0.sqlite \
  --skill gpu_idle_gaps \
  --param device_id=0 \
  --param min_gap_ns=1000000 \
  --param limit=30 \
  --pretty \
  --output ./nsys_metrics_out/gpu_idle_gaps.json

# nsys-oriented summary report (json)
myutils-profile nsys-analyze \
  --sqlite ./logs/nsys/train_rank_0.sqlite \
  --device-id 0 \
  --top-k 20 \
  --output ./nsys_metrics_out/nsys_analyze.json

# kernel flat export
myutils-profile nsys-export \
  --sqlite ./logs/nsys/train_rank_0.sqlite \
  --device-id 0 \
  --format csv \
  --output ./nsys_metrics_out/kernels_flat.csv

# before/after sqlite diff
myutils-profile nsys-diff \
  --before-sqlite ./logs/nsys/run_a.sqlite \
  --after-sqlite ./logs/nsys/run_b.sqlite \
  --device-id 0 \
  --output ./nsys_metrics_out/nsys_diff.json

# static timeline html
myutils-profile nsys-timeline-html \
  --sqlite ./logs/nsys/train_rank_0.sqlite \
  --device-id 0 \
  --output ./nsys_metrics_out/timeline.html
```

## 3. Constructor Options

`NsysSqliteMetricsProvider(...)` options:

- `sqlite_path`: SQLite file path
- `enabled`: global switch
- `include_runtime`: parse `CUPTI_ACTIVITY_KIND_RUNTIME`
- `include_kernels`: parse `CUPTI_ACTIVITY_KIND_KERNEL`
- `include_memcpy`: parse `CUPTI_ACTIVITY_KIND_MEMCPY`
- `include_memset`: parse `CUPTI_ACTIVITY_KIND_MEMSET`
- `include_sync`: parse `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`
- `include_nvtx`: parse `NVTX_EVENTS`
- `include_memory_usage`: parse memory usage + memory pool tables
- `include_osrt`: parse `OSRT_API`
- `include_gpu_metrics`: parse `GPU_METRICS` + `TARGET_INFO_GPU_METRICS`
- `include_generic_events`: parse `GENERIC_EVENTS` JSON metrics + `GENERIC_EVENT_TYPES/SOURCES`
- `include_network_metrics`: parse `NET_NIC_METRIC` / `NET_IB_SWITCH_METRIC` + `TARGET_INFO_NETWORK_METRICS`
- `include_pmu_metrics`: parse `PMU_EVENTS` / `PMU_EVENT_COUNTERS` / `PMU_EVENT_REQUESTS`
- `include_cuda_um`: parse CUDA Unified Memory page-fault tables
- `include_auto_duration_tables`: auto-parse other `start/end` tables not explicitly handled
- `parse_step_from_nvtx`: extract `step`/`rank` from NVTX text

## 4. What SQLite Tables Are Parsed

Current provider parses these tables and emits canonical metrics:

| SQLite table | Main metrics | Typical questions answered |
| --- | --- | --- |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | `latency.cuda.runtime_api` | Which CUDA runtime APIs are expensive? |
| `CUPTI_ACTIVITY_KIND_KERNEL` | `latency.kernel.cuda`, `compute.kernel.registers_per_thread`, `compute.kernel.block_x/y/z`, `compute.kernel.grid_x/y/z`, `compute.kernel.threads_per_block`, `compute.kernel.blocks_per_grid`, `memory.kernel.shared_*`, `memory.kernel.local_total_bytes`, `memory.kernel.local_per_thread_bytes` | Which kernels dominate time? Is launch config/resource usage reasonable? |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | `latency.memcpy.cuda`, `io.memcpy.bytes`, `io.memcpy.bandwidth_gbps` | Is transfer time/throughput the bottleneck? |
| `CUPTI_ACTIVITY_KIND_MEMSET` | `latency.memset.cuda`, `io.memset.bytes` | Is memset overhead significant? |
| `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` | `latency.cuda.sync` | Is synchronization causing stalls? |
| `CUDA_GPU_MEMORY_USAGE_EVENTS` | `memory.cuda.alloc.bytes`, `memory.cuda.free.bytes`, `memory.cuda.usage.bytes` | Is memory growth suspicious? Allocate/free imbalance? |
| `CUDA_GPU_MEMORY_POOL_EVENTS` | `calls.cuda.memory_pool_event`, `memory.cuda.pool.*` | Are memory pools being created/destroyed/trimmed frequently? |
| `GPU_METRICS` + `TARGET_INFO_GPU_METRICS` | `compute.gpu.sm.*`, `memory.gpu.dram.*`, `io.gpu.pcie.*`, `io.gpu.nvlink.*`, `external.nsys.gpu_metric.*` | How are SM and bandwidth-related device metrics changing over time? |
| `GENERIC_EVENTS` + `GENERIC_EVENT_TYPES` + `GENERIC_EVENT_SOURCES` | `external.nsys.generic.*` + auto-mapped GPU metric keys | Parse version-dependent JSON metrics without hardcoding each field |
| `NET_NIC_METRIC` / `NET_IB_SWITCH_METRIC` + `TARGET_INFO_NETWORK_METRICS` | `comm.nic.*`, `comm.ib.switch.*` | What are NIC / IB switch telemetry trends during the run? |
| `PMU_EVENTS` / `PMU_EVENT_COUNTERS` / `PMU_EVENT_REQUESTS` | `perf.pmu.*`, `latency.pmu.sample_window` | What are CPU PMU counter samples around profiling windows? |
| `CUDA_UM_CPU_PAGE_FAULT_EVENTS` | `calls.cuda.um.cpu_page_fault` | Are UM CPU page faults frequent? |
| `CUDA_UM_GPU_PAGE_FAULT_EVENTS` | `calls.cuda.um.gpu_page_fault_event`, `memory.cuda.um.gpu_page_faults`, `latency.cuda.um.gpu_page_fault` | Are UM GPU page faults or migrations expensive? |
| `NVTX_EVENTS` | `latency.nvtx.range`, `calls.nvtx.mark` | Which NVTX stage/range is expensive? (Custom names are exposed in tags `nvtx_text` and `name`) |
| `OSRT_API` (optional) | `latency.osrt.api` | Which host OS runtime calls are expensive? |
| Other duration tables with `start/end` (auto mode) | `latency.nsys.<table>`, optional `io.nsys.<table>.bytes` | Collect duration-style metrics from newly introduced schema tables |
| `StringIds` | name resolution support table | Convert id fields to readable names |

Also used for enrichment:

- `ThreadNames` and `PROCESSES` for readable thread/process tags
- metadata table (`META_DATA_EXPORT` or `EXPORT_META_DATA`) for export schema tags

## 5. Tag Enrichment and Alignment Logic

Each emitted event may include:

- identity tags: `pid`, `tid`, `global_pid`, `global_tid`
- execution tags: `deviceid`, `contextid`, `greencontextid`, `streamid`
- correlation tags: `correlationid`, `runtime_api`
- semantic tags: `kernel`, `api`, `copy_kind`, `src_kind`, `dst_kind`, `sync_type`, `mem_kind`, `memory_pool_type`
- NVTX tags: `nvtx_event_type`, `nvtx_text`, `nvtx_domain_id`, `nvtx_category`, `nvtx_color`
- export tags: `export_schema_version`, `export_schema_major`, `export_schema_minor`, `export_schema_micro`, `export_meta_table`

Alignment mechanisms:

1. Runtime-to-GPU alignment:
   - Runtime API rows are cached by `(pid, correlationId)` (fallback to `correlationId` when pid is absent).
   - Kernel/memcpy/memset/sync/memory events attach `runtime_api` when the key matches.
2. Process/thread decode:
   - `globalTid` and `globalPid` are decoded using official formula.
3. Step/rank alignment:
   - Optional regex extraction from NVTX text:
     - `step=<n>` / `step:<n>` / `iteration=<n>`
     - `rank=<n>`

## 6. Parsing Flow (Implementation View)

Parsing pipeline in `get_metrics()`:

1. Open SQLite and set `row_factory=sqlite3.Row`
2. Load schema metadata (`META_DATA_EXPORT` or `EXPORT_META_DATA`)
3. Parse enabled tables in order:
   - runtime -> kernels -> memcpy -> memset -> sync -> memory -> gpu_metrics -> generic_events -> network -> pmu -> cuda_um -> NVTX -> OSRT -> auto duration tables
4. Convert durations from ns to us where applicable
5. Emit `MetricEvent` list

Robustness strategy:

- Table existence check via `sqlite_master`
- Column existence check via `PRAGMA table_info`
- Incremental row reading with `rowid > last_rowid`
- One-shot fallback if table has no `rowid` incremental behavior
- Numeric parsing guarded by `_safe_int`/`_safe_float`

## 7. What You Can Analyze from SQLite

From current provider output, you can do:

1. Latency structure:
   - Runtime API, kernel, memcpy, memset, sync, NVTX ranges, OSRT API
2. Kernel resource characterization:
   - registers/thread, shared memory, local memory, launch geometry
3. Data movement efficiency:
   - transfer bytes + bandwidth (`GB/s`)
4. Memory lifecycle:
   - alloc/free/usage events and memory pool operations
5. Device-level sampled telemetry:
   - SM active/throughput, DRAM throughput, PCIe/NVLink throughput (from `GPU_METRICS` or `GENERIC_EVENTS`)
6. Network telemetry:
   - NIC and IB switch metrics from `NET_*_METRIC`
7. CPU PMU counters:
   - event counters from `PMU_*` tables
8. Unified memory page-fault diagnostics:
   - CPU/GPU UM page-fault counts and GPU fault duration
9. Runtime call path context:
   - GPU activities tagged with triggering runtime API
10. Multi-rank/multi-step slicing:
   - by `rank`/`step` tags (from external tags or NVTX text)
11. Process/thread/stream slicing:
   - by `pid`, `tid`, `streamid`, `deviceid`

Important interpretation note:

- Activity tables (runtime/kernel/memcpy/...) are process/thread attributable.
- `gpu-metrics` sampled telemetry is typically device-level; per-process split requires time-window attribution rather than direct pid ownership.

Combined with `MetricsAnalyzer`, you additionally get:

- bottleneck detection (`latency.*` share)
- memory growth warnings across steps
- high variance groups (CV threshold)
- latency outlier detection (3-sigma style)

## 8. Version Compatibility

Nsight Systems SQLite schema evolves over releases. This provider targets compatibility by dynamic probing instead of hardcoded schema assumptions.

Known schema generations observed in official docs:

1. 2022.4 era (`nsys-exporter` docs)
2. 2023.2 era (`EXPORT_SCHEMA_VERSION` `3.1.7`, often `EXPORT_META_DATA`)
3. 2024.4 era (`EXPORT_SCHEMA_VERSION` `3.15.1`, often `META_DATA_EXPORT`)
4. 2025.4 era (`EXPORT_SCHEMA_VERSION` `3.20.7`)
5. 2025.5 era (`EXPORT_SCHEMA_VERSION` `3.23.2`, documented in 2025.6/2026.1 analysis guide)

Compatibility tactics implemented:

- support both metadata table names
- select only existing columns
- memory table alias handling (`memoryOperationType`/`operationType`, `memoryPoolType`/`poolType`, etc.)
- support both dedicated memory pool table and legacy shape fallback
- parse dynamic JSON metrics in `GENERIC_EVENTS` without fixed key list
- auto-parse unknown future duration tables (`start`/`end`) for forward compatibility

Practical expectation:

- Nsight Systems 2024.7.1 and 2026.2 exports are supported
- future major schema jumps may require additional adaptation

## 9. Quick SQL Inspection Snippets

Check exported schema version:

```sql
SELECT name, value FROM META_DATA_EXPORT WHERE name LIKE 'EXPORT_SCHEMA_VERSION%';
SELECT name, value FROM EXPORT_META_DATA WHERE name LIKE 'EXPORT_SCHEMA_VERSION%';
```

List key activity tables:

```sql
SELECT name
FROM sqlite_master
WHERE type='table' AND (
  name LIKE 'CUPTI_ACTIVITY_KIND_%'
  OR name LIKE 'CUDA_GPU_MEMORY_%'
  OR name IN ('NVTX_EVENTS', 'OSRT_API', 'StringIds', 'ThreadNames', 'PROCESSES')
)
ORDER BY name;
```

Top kernels by total duration:

```sql
SELECT
  k.demangledName,
  COUNT(*) AS launches,
  SUM(k.end - k.start) / 1000.0 AS total_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
GROUP BY k.demangledName
ORDER BY total_us DESC
LIMIT 20;
```

Memcpy throughput summary:

```sql
SELECT
  SUM(bytes) AS total_bytes,
  SUM(end - start) / 1000.0 AS total_us,
  SUM(bytes) / (SUM(end - start) / 1000.0 * 1000.0) AS approx_gbps
FROM CUPTI_ACTIVITY_KIND_MEMCPY
WHERE end > start;
```

Runtime API to kernel correlation (via `correlationId`):

```sql
SELECT
  r.correlationId,
  r.nameId AS runtime_name_id,
  k.demangledName AS kernel_name,
  (k.end - k.start) / 1000.0 AS kernel_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
  ON r.correlationId = k.correlationId
ORDER BY kernel_us DESC
LIMIT 50;
```

GPU metrics with readable names:

```sql
SELECT
  g.timestamp,
  g.typeId,
  g.metricId,
  info.metricName,
  g.value
FROM GPU_METRICS g
LEFT JOIN TARGET_INFO_GPU_METRICS info
  ON g.typeId = info.typeId
 AND g.metricId = info.metricId
ORDER BY g.timestamp
LIMIT 100;
```

Generic events (JSON payload) sample:

```sql
SELECT
  ge.timestamp,
  ge.typeId,
  gety.sourceId,
  gety.data AS type_json,
  ge.data AS event_json
FROM GENERIC_EVENTS ge
LEFT JOIN GENERIC_EVENT_TYPES gety
  ON ge.typeId = gety.typeId
ORDER BY ge.timestamp
LIMIT 50;
```

## 10. Recommended NVTX Naming Convention

To improve auto alignment:

- Include step/rank in NVTX text, for example:
  - `step=120 rank=3 stage=forward`
  - `iteration:120 rank:3`
- Keep stage names stable so report grouping is stable

## 11. Troubleshooting

No events parsed:

1. Verify SQLite path exists and is readable
2. Verify export actually contains CUPTI/NVTX tables
3. Ensure profiler trace options included required domains (`cuda`, `nvtx`, `osrt`)

Kernel names appear as ids:

1. Check `StringIds` table exists
2. Ensure `demangledName` or `shortName` column exists

`step`/`rank` missing:

1. Ensure NVTX text carries step/rank patterns
2. Keep `parse_step_from_nvtx=True`
3. Or pass `step` via `collector.collect(step=...)`

Large SQLite and high memory:

1. Disable unneeded sections (`include_*` flags)
2. Parse offline in per-rank/per-window batches

## 12. Multi-Process GPU and GPU-Metrics Clarification

### 12.1 One GPU with multiple processes

You can attribute most activity metrics to different processes on the same GPU:

- runtime/kernel/memcpy/memset/sync rows can be tagged with `pid`/`tid`/`global_pid`/`global_tid`
- provider emits those tags so you can group by process/rank

For device-sampled counters (`GPU_METRICS` / some `GENERIC_EVENTS`):

- those samples are usually device-level time-series
- they may not directly carry process ownership
- process attribution is done by time-window correlation with process activity

### 12.2 `--gpu-metrics-devices` support

If Nsight Systems capture enables GPU metrics, SQLite commonly contains:

- `GPU_METRICS`
- `TARGET_INFO_GPU_METRICS`
- and/or `GENERIC_EVENTS` + `GENERIC_EVENT_TYPES`

The provider parses them into canonical metrics such as:

- `compute.gpu.sm.active`
- `compute.gpu.sm.throughput`
- `memory.gpu.dram.throughput`
- `io.gpu.pcie.*`
- `io.gpu.nvlink.*`
- plus dynamic fallback `external.nsys.gpu_metric.*`

## 13. Schema Adapter and Probing APIs

This implementation adds a version adapter helper:

- module: `my_utils.profiling.nsys_schema_adapter`
- detects exporter version and maps to adapter family (`nsys_2022`, `nsys_2023`, `nsys_2024_plus`, `generic`)

Provider surface:

- `NsysSqliteMetricsProvider.describe_schema()`
  - returns table list, column map, schema metadata, detected version family
- `NsysSqliteMetricsProvider.list_sql_skills()`
  - returns available skill names for this SQLite schema
- `NsysSqliteMetricsProvider.describe_sql_skills()`
  - returns skill metadata and parameter definitions
- `NsysSqliteMetricsProvider.run_sql_skill(name, **params)`
  - executes one built-in SQL skill and returns result rows
- `NsysSqliteMetricsProvider.analyze_compute_comm_overlap(...)`
  - returns compute/comm overlap summary based on kernel intervals
- `NsysSqliteMetricsProvider.summarize_gpu_kernels(...)`
  - returns span/busy/idle/utilization and top kernels summary
- `NsysSqliteMetricsProvider.detect_iterations(marker=...)`
  - detects iteration windows by NVTX marker ranges
- `NsysSqliteMetricsProvider.compute_mfu(...)`
  - computes MFU from step time + model FLOPs + peak TFLOPS

Example:

```python
from my_utils.profiling import NsysSqliteMetricsProvider

p = NsysSqliteMetricsProvider("./train.sqlite")
schema = p.describe_schema()
print(schema["table_count"], schema["version_info"])
```

## 14. Source References

- Provider implementation:
  - `my_utils/profiling/sources/nsys_sqlite_provider.py`
  - `my_utils/profiling/sources/nsys_schema_adapter.py`
  - `my_utils/profiling/sources/nsys_sql_skills.py`
- Unified metrics flow:
  - `my_utils/profiling/pipeline/metrics_collector.py`
  - `my_utils/profiling/analyzers/metrics_analyzer.py`
- Official docs:
  - Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/
  - Nsight Systems docs archives: https://docs.nvidia.com/nsight-systems/archives/index.html
  - 2022.4 exported data: https://docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/exported_data.html
  - 2026.2 User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
  - 2024.7 User Guide (archive baseline): https://docs.nvidia.com/nsight-systems/2024.7/UserGuide/index.html
  - 2026.1 analysis guide exported data: https://docs.nvidia.com/nsight-systems/2026.1/AnalysisGuide/exported_data.html
