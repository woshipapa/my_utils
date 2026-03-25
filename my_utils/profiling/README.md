# profiling

The `profiling` package is now split by responsibility. The root keeps entry points (API/CLI), docs, and templates.

## Layout

```text
profiling/
  analyzers/   # rules, workload profiles, rank alignment
  metrics/     # metric schema/types/providers/store/registry/taxonomy
  output/      # report/diff/chrome-trace export
  runtime/     # capture runtime integration and ProfileManager
  sources/     # nsys sqlite/schema parsers
  pipeline/    # MetricsCollector orchestration
  docs/        # design/spec/roadmap docs
  adapters/
  visualization/
  templates/
  examples/
  cli.py
  __init__.py
  torch_compile_reference.yaml
  torch_compile_catalog.snapshot.yaml
  torch_compile_catalog_versions.yaml
  generate_torch_compile_catalog.py
```

## Flow

1. `metrics/*` providers convert tool outputs to `MetricEvent`.
2. `pipeline/MetricsCollector` does collect -> normalize/validate -> store.
3. `analyzers/*` runs rule-based analysis in `analyze()`.
4. `output/*` exports reports and chrome trace.
5. `runtime/*` integrates capture windows and runtime controls.

## CLI Quick Commands

All commands run via `python -m my_utils.profiling.cli <subcommand>` or the `myutils-profile` entry point.
For nsys workflows, direct short aliases are also installed (for example `nsys-sql-skill`, `nsys-analyze`, `nsys-diff`).

```bash
# list built-in providers
myutils-profile list-providers

# run built-in Nsight SQLite SQL skills
myutils-profile nsys-sql-skill --sqlite ./train_rank0.sqlite --list-skills --pretty
myutils-profile nsys-sql-skill --sqlite ./train_rank0.sqlite --skill top_kernels --param device_id=0 --param limit=20 --pretty

# nsys-oriented offline workflow
myutils-profile nsys-analyze --sqlite ./train_rank0.sqlite --device-id 0 --top-k 20 --output ./nsys_analyze.json
myutils-profile nsys-export --sqlite ./train_rank0.sqlite --device-id 0 --format csv --output ./kernels_flat.csv
myutils-profile nsys-diff --before-sqlite ./run_a.sqlite --after-sqlite ./run_b.sqlite --device-id 0 --output ./nsys_diff.json
myutils-profile nsys-timeline-html --sqlite ./train_rank0.sqlite --device-id 0 --output ./timeline.html
myutils-profile nsys-timeline-html --sqlite ./train_rank0.sqlite --output ./timeline_nvtx.html --nvtx-text "%sample_0%" --include-metrics --metric-name-like "%active%"

# direct aliases (same behavior as above)
nsys-sql-skill --sqlite ./train_rank0.sqlite --list-skills --pretty
nsys-analyze --sqlite ./train_rank0.sqlite --device-id 0 --top-k 20 --output ./nsys_analyze.json
nsys-timeline-html --sqlite ./train_rank0.sqlite --output ./timeline_nvtx.html --nvtx-text "%sample_0%" --include-metrics
```

## Nsight Systems Offline Analysis Reference

### nsys-analyze

Runs comprehensive analysis on a single nsys SQLite export:
summary (GPU utilization), compute/comm overlap, top kernels, NCCL breakdown,
iteration detection, and optional MFU calculation.

```bash
myutils-profile nsys-analyze \
  --sqlite        ./train_rank0.sqlite \
  --device-id     0          \   # -1 = all devices
  --start-ns      -1         \   # -1 = no window trim
  --end-ns        -1         \
  --top-k         20         \   # top N kernels
  --limit         500000     \   # max kernel rows loaded
  --iteration-marker sample_0 \  # NVTX marker for step boundaries
  --model-flops-per-step 1e15 \  # optional: enables MFU output
  --peak-tflops   989.0      \   # optional: auto-inferred from GPU name if omitted
  --peak-precision fp16      \   # fp16 | bf16 | fp32
  --format        json       \   # json | markdown
  --output        ./nsys_analyze.json \
  --pretty
```

**Key output fields:**
- `summary.timing`: `span_ms`, `busy_ms`, `idle_ms`, `utilization_pct`
- `overlap`: `compute_only_ms`, `comm_only_ms`, `overlap_ms`, `overlap_pct_of_comm`
- `top_kernels`: sorted by `total_ms`
- `nccl_breakdown`: NCCL collective breakdown
- `iterations`: per-step `duration_ms`, `compute_ms`, `comm_ms`, `kernel_count`
- `mfu`: `achieved_model_tflops`, `mfu_pct` (only when `--model-flops-per-step` is set)

### nsys-sql-skill

Run any of the 11 built-in SQL skills against a SQLite export.

```bash
# List available skills
myutils-profile nsys-sql-skill --sqlite ./train_rank0.sqlite --list-skills --pretty

# Run a skill
myutils-profile nsys-sql-skill \
  --sqlite ./train_rank0.sqlite \
  --skill  top_kernels          \   # see table below
  --param  device_id=0          \
  --param  limit=20             \
  --pretty                      \
  --output ./top_kernels.json
```

**Available skills:**

| Skill | Description |
|---|---|
| `top_kernels` | Top kernels by cumulative time |
| `aggregate_kernels` | All kernels grouped: count, total/avg/min/max ms |
| `nccl_breakdown` | NCCL kernels only |
| `kernel_map` | Raw timeline: start_ns, end_ns, stream, correlation_id |
| `gpu_idle_gaps` | Gaps between consecutive kernels per stream |
| `kernel_launch_overhead` | CPU launch latency to GPU start |
| `aggregate_nvtx_ranges` | NVTX ranges grouped by text |
| `nvtx_kernel_map` | NVTX range → kernels inside |
| `memcpy_in_window` | Memcpy by copyKind in a time window |
| `thread_utilization` | CPU thread utilization % |
| `schema_inspect` | Table/column schema viewer |

For `schema_inspect`, CLI also supports richer display modes:

```bash
myutils-profile nsys-sql-skill \
  --sqlite ./train_rank0.sqlite \
  --skill schema_inspect \
  --schema-view both \
  --pretty
```

`--schema-view`:
- `flat`: raw rows
- `grouped`: grouped columns by table
- `mermaid`: inferred table relations and Mermaid flowchart
- `both`: grouped + mermaid (default)

### nsys-export

Exports kernel timeline rows as a flat JSON or CSV file.
Optionally annotates each row with the training iteration index it falls into.

```bash
myutils-profile nsys-export \
  --sqlite           ./train_rank0.sqlite \
  --output           ./kernels_flat.csv   \
  --format           csv                  \   # json | csv
  --device-id        0                    \
  --limit            500000               \
  --attach-iteration                      \   # annotate with iteration index
  --iteration-marker sample_0
```

### nsys-diff

Compares two nsys SQLite profiles (e.g., before/after an optimization).
Reports utilization delta, overlap delta, and top kernel time changes.

```bash
myutils-profile nsys-diff \
  --before-sqlite ./run_a.sqlite \
  --after-sqlite  ./run_b.sqlite \
  --device-id     0              \
  --top-k         20             \
  --format        markdown       \
  --output        ./nsys_diff.md \
  --pretty
```

**Key output fields:**
- `summary.utilization_pct.delta`: GPU utilization change
- `summary.overlap_ms.delta`: compute/comm overlap change
- `kernel_diff_top`: kernels sorted by `|Δtotal_ms|`
- `nvtx_diff_top`: NVTX ranges sorted by `|Δtotal_ms|`

### nsys-timeline-html

Exports a static HTML timeline visualization of GPU kernel activity.

```bash
myutils-profile nsys-timeline-html \
  --sqlite    ./train_rank0.sqlite \
  --output    ./timeline.html      \
  --device-id 0                    \
  --limit     100000               \
  --width-px  1800
```

### One-shot full pipeline (shell template)

```bash
SQLITE_PATH=/abs/path/to/trace.sqlite \
OUT_DIR=./nsys_metrics_out            \
OUTPUT_PREFIX=run_a                   \
bash my_utils/profiling/templates/run_nsys_full_postprocess.sh
```

Generates: `*_analyze.json`, `*_kernels_flat.json`, `*_kernels_flat.csv`,
`*_timeline.html`, `*_skill_<name>.json`.

Environment variable controls: `ENABLE_ANALYZE`, `ENABLE_EXPORT`, `ENABLE_TIMELINE`,
`ENABLE_SQL_SKILL`, `ATTACH_ITERATION`, `ITERATION_MARKER`, `TOP_K`, `LIMIT`,
`DEVICE_ID`, `START_NS`, `END_NS`.

---

## Submodule Docs

- [docs/README.md](./docs/README.md)
- [analyzers/README.md](./analyzers/README.md)
- [metrics/README.md](./metrics/README.md)
- [output/README.md](./output/README.md)
- [runtime/README.md](./runtime/README.md)
- [sources/README.md](./sources/README.md)
- [pipeline/README.md](./pipeline/README.md)
- [templates/README.md](./templates/README.md)
- [examples/README.md](./examples/README.md)

## Compatibility

Legacy module paths are still supported via aliases in `profiling.__init__`, e.g.
- `my_utils.profiling.metrics_collector`
- `my_utils.profiling.metrics_analyzer`
- `my_utils.profiling.metrics_trace`

New code should use layered paths, e.g.
- `my_utils.profiling.pipeline.metrics_collector`
- `my_utils.profiling.analyzers.metrics_analyzer`
- `my_utils.profiling.output.metrics_trace`
