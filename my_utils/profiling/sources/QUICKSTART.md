# Nsight SQLite Profiling — Beginner's Guide

This guide helps you go from zero to useful analysis results in under 15 minutes.
No prior nsys experience required.

---

## What this system does

You have trained a GPU model and want to know **where time is being spent**.
This system reads the SQLite file that Nsight Systems exports from a profiling run and
answers questions like:

- Which GPU kernels consume the most time?
- Is my NCCL communication overlapping with compute, or blocking it?
- Are some training steps much slower than others?
- Is PCIe bandwidth a bottleneck?
- Are my kernels launching with low SM occupancy?

---

## Step 0 — Prerequisites

```bash
# Install myutils-profile CLI (if not already done)
pip install -e /path/to/Teletron-lite/third_party/my_utils

# Verify the CLI is available
myutils-profile --help
```

---

## Step 1 — Collect a profile

Run your training script under nsys:

```bash
nsys profile \
  --trace=cuda,nvtx,cudnn,cublas \
  --output=train_rank0 \
  python train.py
```

This produces `train_rank0.nsys-rep`.

> **Tip:** Use NVTX markers in your training loop to label each iteration.
> The system uses them to detect per-step boundaries automatically.
>
> ```python
> import torch.cuda.nvtx as nvtx
> for step in range(num_steps):
>     nvtx.range_push(f"sample_{step}")
>     # ... forward / backward / optimizer ...
>     nvtx.range_pop()
> ```

---

## Step 2 — Export to SQLite

```bash
nsys export --sqlite train_rank0.sqlite train_rank0.nsys-rep
```

This creates `train_rank0.sqlite` — the file all commands below read from.

---

## Step 3 — Start here: run the all-in-one report

```bash
myutils-profile nsys-analyze \
  --sqlite train_rank0.sqlite \
  --format markdown
```

This runs every built-in analysis in one shot and prints a Markdown report covering:

- GPU utilization (busy % vs idle %)
- Top time-consuming kernels
- NCCL collective breakdown
- Compute / comm overlap
- Per-iteration statistics
- Synchronization overhead
- Memory copy bandwidth

**Read the report top to bottom.** The numbers tell you where to drill next.

---

## Step 4 — Drill down with targeted skills

Once you know which area needs investigation, use `nsys-sql-skill` to run a single
parameterized query with full control over limits and filters.

```bash
myutils-profile nsys-sql-skill \
  --sqlite train_rank0.sqlite \
  --skill <skill_name> \
  --pretty
```

### Use-case guide

| I want to know... | Use this skill | Key output |
|---|---|---|
| Which kernels take the most GPU time? | `top_kernels` | `kernel_name`, `total_ms`, `invocations` |
| Full stats per kernel (min/max/avg)? | `aggregate_kernels` | `min_ms`, `max_ms`, `avg_ms` |
| Which NCCL collectives are slowest? | `nccl_breakdown` | `kernel_name`, `total_ms`, `count` |
| Are cudaDeviceSync calls blocking me? | `sync_breakdown` | `sync_type`, `total_ms`, `count` |
| Is PCIe or NVLink the bottleneck? | `memcpy_bandwidth_analysis` | `copy_kind`, `avg_gbps`, `total_gb` |
| How much time is buffer zeroing? | `memset_breakdown` | `fill_value`, `total_ms`, `total_gb` |
| Are my kernels underutilizing SMs? | `kernel_occupancy_estimate` | `occupancy_pct_estimate`, `threads_per_block` |
| Is multi-stream overlap happening? | `stream_parallelism` | `pct_time_multi_stream` |
| Show all NVTX names with nesting? | `nvtx_ranges_hierarchy` | `nvtx_text`, `depth`, `parent_nvtx_text` |
| Which training phase moves most data? | `nvtx_memcpy_breakdown` | `nvtx_text`, `total_gb` |
| Where are GPU idle bubbles? | `gpu_idle_gaps` | `gap_ms`, `before_kernel`, `after_kernel` |
| How long does kernel launch take? | `kernel_launch_overhead` | `api_name`, `overhead_us`, `api_ms` |
| What kernels are launch-attributed to a phase? | `nvtx_kernel_sm_detail` | `kernel_name`, `kind`, `occupancy_pct_estimate` |

---

## Reading the key numbers

### GPU utilization

```
busy_ms / span_ms × 100 = utilization_pct
```

- **> 80%** — good; the GPU is mostly working.
- **50–80%** — investigate idle gaps (`gpu_idle_gaps`) and sync overhead (`sync_breakdown`).
- **< 50%** — serious pipeline stall. Check sync, NCCL, and CPU-side data loading.

### Compute / comm overlap

```
overlap_ms / comm_total_ms × 100 = overlap_pct_of_comm
```

- **> 80%** — communication is well-hidden behind compute.
- **< 30%** — communication is exposed (blocking). Consider tensor parallelism or
  pipeline scheduling tuning.

### NCCL breakdown copy kinds

| `copy_kind` value | Direction | Typical cause |
|---|---|---|
| 1 | Host → Device (H2D) | Data loading, optimizer state restore |
| 2 | Device → Host (D2H) | Loss extraction, metric logging |
| 8 | Device → Device (D2D) | NVLink, within-node all-reduce |

High H2D bandwidth with low D2D bandwidth can indicate the system is bottlenecked
on PCIe rather than NVLink.

### SM occupancy estimate

The `occupancy_pct_estimate` column uses the simplified formula:

```
warps_per_block = ceil(threads_per_block / 32)
occupancy_pct   = MIN(64, warps_per_block * 4) / 64 * 100
```

- **100%** — launch configuration fully saturates the SM warp scheduler.
- **< 50%** — consider increasing `threads_per_block` (up to 256 or 512 is typical).
- Very large `registersPerThread` (> 64) or large `total_shared_bytes` will also
  reduce real occupancy beyond this estimate.

---

## Detecting slow training steps

### Check for outlier steps first

```bash
myutils-profile nsys-iter-outliers \
  --sqlite train_rank0.sqlite \
  --iteration-marker sample_0 \
  --sigma 2.0
```

Output:
```json
{
  "stats": {
    "count": 100,
    "mean_ms": 1234.5,
    "median_ms": 1230.1,
    "std_ms": 45.2,
    "p95_ms": 1310.0,
    "p99_ms": 1450.0
  },
  "outliers": [
    {"iteration": 7,  "duration_ms": 1890.0, "deviation_sigma": 14.5},
    {"iteration": 23, "duration_ms": 1542.0, "deviation_sigma": 6.9}
  ]
}
```

- **`outliers` empty** — step times are stable. Focus on average performance.
- **A few high-sigma outliers** — likely system noise, GC pauses, or NCCL straggler.
- **Many outliers** — possible learning rate instability or irregular data batches.

### Inspect per-step compute / comm breakdown

```bash
myutils-profile nsys-iter-overlap \
  --sqlite train_rank0.sqlite \
  --iteration-marker sample_0
```

Each row in the output is one training step:

```json
[
  {
    "iteration": 0,
    "duration_ms": 1230.1,
    "compute_ms": 980.4,
    "comm_ms": 420.5,
    "overlap_ms": 350.2,
    "comm_pct": 34.2,
    "kernel_count": 1847
  },
  ...
]
```

> **Important:** `compute_ms` here is **wall-clock** (overlapping streams counted once).
> The same field in `nsys-analyze` → `iterations[]` is a **naive SUM** (parallel streams
> counted separately). Use `nsys-iter-overlap` when you care about actual wall-clock cost.

---

## Inspecting a specific training phase

If you have NVTX markers for forward / backward / optimizer, you can zoom into one phase:

```bash
# List all kernel SM details inside the "forward" phase
myutils-profile nsys-sql-skill \
  --sqlite train_rank0.sqlite \
  --skill nvtx_kernel_sm_detail \
  --param nvtx_text=%forward% \
  --pretty

# List all NVTX ranges (including nested parent/child ranges)
myutils-profile nsys-sql-skill \
  --sqlite train_rank0.sqlite \
  --skill nvtx_ranges_hierarchy \
  --param nvtx_text=% \
  --param top_level_only=false \
  --pretty

# How much data is copied during the optimizer step?
myutils-profile nsys-sql-skill \
  --sqlite train_rank0.sqlite \
  --skill nvtx_memcpy_breakdown \
  --param nvtx_text=%optimizer% \
  --pretty
```

The `nvtx_text` parameter is a SQL `LIKE` pattern — `%` matches any substring.

---

## Comparing two runs (before / after optimization)

```bash
myutils-profile nsys-diff \
  --before-sqlite baseline.sqlite \
  --after-sqlite optimized.sqlite \
  --format markdown
```

Output highlights:
- `utilization_pct` delta
- `overlap_ms` delta
- Top kernels that got faster or slower (`kernel_diff_top`)
- NVTX range duration changes (`nvtx_diff_top`)

---

## Exporting raw data for custom analysis

```bash
# CSV with one row per kernel, annotated with iteration index
myutils-profile nsys-export \
  --sqlite train_rank0.sqlite \
  --format csv \
  --attach-iteration \
  --output kernels.csv

# HTML interactive timeline
myutils-profile nsys-timeline-html \
  --sqlite train_rank0.sqlite \
  --output timeline.html
```

Open `timeline.html` in a browser to visually inspect the kernel timeline with zoom and hover.

---

## Common parameters

| Parameter | Flag | Default | Effect |
|---|---|---|---|
| Device filter | `--device-id 0` | -1 (all) | Restrict to one GPU |
| Row limit | `--limit 50` | skill-default | Fewer rows = faster query |
| Time window | `--start-ns N --end-ns N` | -1 (full) | Analyze a sub-window |
| Iteration marker | `--iteration-marker sample_0` | `sample_0` | NVTX text pattern |
| Output file | `--output result.json` | stdout | Save to file |
| Pretty-print | `--pretty` | off | Indent JSON output |

---

## Quick diagnostic checklist

Copy this checklist and work through it top to bottom:

```
[ ] 1. Run nsys-analyze --format markdown
        Read: utilization_pct, overlap_pct_of_comm, top 3 kernels

[ ] 2. If utilization_pct < 70%:
        Run nsys-sql-skill --skill gpu_idle_gaps
        Run nsys-sql-skill --skill sync_breakdown

[ ] 3. If overlap_pct_of_comm < 50%:
        Run nsys-iter-overlap  → confirm which steps have low overlap
        Check nccl_breakdown for large all-reduce kernels

[ ] 4. If outlier steps suspected:
        Run nsys-iter-outliers --sigma 2.0

[ ] 5. If a specific kernel is suspicious:
        Run nsys-sql-skill --skill kernel_occupancy_estimate
        Look for low occupancy_pct_estimate or high registersPerThread

[ ] 6. If memory bandwidth seems low:
        Run nsys-sql-skill --skill memcpy_bandwidth_analysis
        Compare avg_gbps vs hardware peak (H100 NVLink ~900 GB/s, PCIe ~64 GB/s)

[ ] 7. If you changed something and want a before/after:
        Run nsys-diff --before-sqlite A.sqlite --after-sqlite B.sqlite
```

---

## MFU (Model FLOP Utilization)

If you know the theoretical FLOPs per step for your model, you can compute MFU directly:

```bash
myutils-profile nsys-analyze \
  --sqlite train_rank0.sqlite \
  --model-flops-per-step 1.2e15 \
  --peak-tflops 989
```

The report will include:

```json
{
  "step_time_s": 1.231,
  "achieved_model_tflops": 974.8,
  "mfu_pct": 98.6
}
```

If `--peak-tflops` is omitted, the system reads the GPU name from the SQLite metadata
and looks it up in a built-in table (H100=989, A100=312, etc.).

MFU > 50% is generally considered good for large model training.

---

## Python API (programmatic use)

If you prefer to work in Python instead of the CLI:

```python
from my_utils.profiling.sources.nsys_analyze import analyze_nsys_sqlite

result = analyze_nsys_sqlite(
    "train_rank0.sqlite",
    device_id=0,
    iteration_marker="sample_0",
    model_flops_per_step=1.2e15,
)

# Top 3 most expensive kernels
for k in result["top_kernels"][:3]:
    print(k["kernel_name"], k["total_ms"])

# Utilization
print(result["summary"]["timing"]["utilization_pct"])

# Overlap
print(result["overlap"]["overlap_pct_of_comm"])
```

Or use the engine directly for individual skills:

```python
import sqlite3
from my_utils.profiling.sources.nsys_sql_skills import NsysSqlSkillEngine

conn = sqlite3.connect("train_rank0.sqlite")
conn.row_factory = sqlite3.Row
engine = NsysSqlSkillEngine(conn)

# List available skills
print(engine.list_skills())

# Run a skill
rows = engine.execute("sync_breakdown", device_id=0, limit=20)
for row in rows:
    print(row)

# Per-iteration overlap (interval-merged, stream-correct)
iters = engine.analyze_per_iteration_overlap(marker="sample_0", device_id=0)
for it in iters[:5]:
    print(it["iteration"], it["compute_ms"], it["comm_ms"], it["overlap_ms"])

conn.close()
```

---

## Glossary

| Term | Meaning |
|---|---|
| `.nsys-rep` | Raw Nsight Systems profile (binary, not directly readable) |
| `.sqlite` | SQL database exported from `.nsys-rep`; what this system reads |
| Kernel | A GPU function dispatched to run on SMs |
| NCCL kernel | A kernel whose name contains `nccl` — a collective communication operation |
| NVTX range | A named time range annotated by the user in training code |
| Iteration / step | One training step, detected from NVTX markers |
| SM occupancy | Fraction of the SM warp scheduler that a kernel fills (higher = better) |
| compute_ms | Wall-clock time the GPU spends on non-NCCL kernels (interval-merged) |
| comm_ms | Wall-clock time the GPU spends on NCCL kernels (interval-merged) |
| overlap_ms | Time when compute and comm are running simultaneously |
| MFU | Model FLOP Utilization — how much of the GPU's peak TFLOPS you achieve |
| H2D / D2H / D2D | Host-to-Device / Device-to-Host / Device-to-Device memory copies |
