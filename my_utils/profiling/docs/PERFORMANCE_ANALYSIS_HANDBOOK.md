# Performance Analysis Handbook

**The authoritative end-to-end guide: collect every metric that matters, then turn it
into a diagnosis.** Written for both humans and agents picking this repo up cold.

If you read only one thing, read [§0 The 60-second version](#0-the-60-second-version).

---

## Table of contents

- [0. The 60-second version](#0-the-60-second-version)
- [1. Which tool answers which question](#1-which-tool-answers-which-question)
- [2. Collection: nsys](#2-collection-nsys)
- [3. Collection: ncu](#3-collection-ncu)
- [4. Analysis: the top-down triage](#4-analysis-the-top-down-triage)
- [5. Analysis: per-kernel diagnosis](#5-analysis-per-kernel-diagnosis)
- [6. The metric catalog](#6-the-metric-catalog)
- [7. Interpretation reference](#7-interpretation-reference)
- [8. Hardware ceilings](#8-hardware-ceilings)
- [9. Kernel taxonomy](#9-kernel-taxonomy)
- [10. Traps that make numbers lie](#10-traps-that-make-numbers-lie)
- [11. Python API reference](#11-python-api-reference)
- [12. Extending the engine](#12-extending-the-engine)

---

## 0. The 60-second version

```bash
# 1. Where does the step time go?  (whole-training view, ~5% overhead)
nsys profile -o run -t cuda,nvtx,osrt --cuda-graph-trace=node \
    --gpu-metrics-devices=all python train.py
nsys-analyze --input run.nsys-rep --format md

# 2. Why is the hot kernel slow?  (single-kernel view, heavy replay overhead)
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
    --config my_utils/profiling/ncu/ncu_full_collection.yaml
ncu-diagnose --report logs/ncu/ncu_full_collection.ncu-rep --gpu "H100 SXM5" --format md
```

The order is not optional. `nsys` first tells you whether the kernels are even the
problem; running `ncu` on a run that is actually dataloader-bound optimises the
wrong thing.

---

## 1. Which tool answers which question

| Question | Tool | Why |
|---|---|---|
| Is the GPU even busy? | nsys | Kernel-boundary timeline over the whole step |
| Is communication exposed? | nsys | Needs concurrent streams; ncu serialises them |
| Is the dataloader stalling? | nsys (+ `-t osrt`) | Host-side blocking calls are only in the CPU trace |
| Which kernels dominate? | nsys | `cuda_gpu_kern_sum` ranks by total GPU time |
| Why is *this* kernel slow? | ncu | Hardware counters, per-kernel |
| Is it memory or compute bound? | ncu | Speed-of-light section |
| Is it uncoalesced / bank-conflicted / spilling? | ncu | Memory + source sections |
| What does the inside of a fused kernel look like? | ncu PM sampling | Whole-kernel counters cannot show a timeline |
| What bandwidth did that allreduce get? | nsys + NCCL parsing | Needs message size and rank count |

**Critical property of ncu:** it replays each kernel many times with caches flushed
and clocks locked. That makes counters reproducible but destroys any measurement
that depends on concurrency or cache state carried in from neighbouring kernels.
Never use ncu to answer "do these two kernels overlap" — that is an nsys question.

---

## 2. Collection: nsys

### 2.1 The standard training capture

```bash
nsys profile \
  -o run --force-overwrite true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  --cuda-graph-trace=node \
  --gpu-metrics-devices=all \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python train.py
```

| Flag | What it buys you | Cost |
|---|---|---|
| `-t osrt` | Host blocking calls, so idle gaps can be attributed to `read`/`poll`/`pthread_cond_wait` | low |
| `--gpu-metrics-devices=all` | SM Active / SM Issue / Tensor Active / DRAM BW sampling; the only way to see tensor-core usage without ncu | low; needs elevated permissions |
| `--cuda-graph-trace=node` | Per-node visibility inside CUDA graphs; without it a graph is one opaque blob | low |
| `--cuda-memory-usage=true` | Allocation events, so steady-state `cudaMalloc` churn is detectable | low |
| `--capture-range=cudaProfilerApi` | Profile only the steady-state steps you mark | avoids huge traces |
| `--python-sampling=true` | Python-level stacks, for when the host is the bottleneck and you need to know *which* Python | medium |
| `--pytorch=autograd-shapes-nvtx` | Automatic NVTX ranges with tensor shapes | medium |
| `--nccl-trace` | NCCL API/collective events (needs NCCL >= 2.23) | low |

### 2.2 Restrict to steady state

Warm-up iterations are not representative: CUDA context init, autotuning caches and
allocator growth all land in the first few steps. Mark the region:

```python
import torch
for step, batch in enumerate(loader):
    if step == 10:
        torch.cuda.cudart().cudaProfilerStart()
    train_step(batch)
    if step == 13:
        torch.cuda.cudart().cudaProfilerStop()
        break
```

Megatron-LM and NeMo already do this: `--profile --profile-step-start 10
--profile-step-end 12 --profile-ranks 0`, and `NsysCallback(start_step=10,
end_step=15, ranks=[0])` respectively. TransformerEngine's NVTX ranges are
**no-ops unless `NVTE_NVTX_ENABLED=1`** is set.

### 2.3 Multi-rank

```bash
srun nsys profile -o report_%q{SLURM_PROCID} --session=rank%q{SLURM_PROCID} ... python train.py
```

One report per rank. `nsys` clock-syncs across nodes, so recipes can align them.
**Profiling only rank 0 is how stragglers get misdiagnosed as slow networks** — a
rank that arrives late at a collective stretches everyone else's collective, and
from inside rank 0's trace that is indistinguishable from a bandwidth problem.

### 2.4 Analysis commands

```bash
nsys-analyze --input run.nsys-rep --format md          # our comprehensive analysis
nsys stats --report cuda_gpu_kern_sum run.nsys-rep     # NVIDIA's kernel ranking
nsys analyze run.nsys-rep                              # NVIDIA's expert-system rules
nsys recipe nccl_gpu_time_util_map --input rank*.nsys-rep   # multi-rank comm heatmap
nsys-diff --baseline before.sqlite --target after.sqlite     # regression comparison
```

NVIDIA's `nsys analyze` rules and their defaults: `cuda_memcpy_async` (async copy
from pageable memory, which is secretly synchronous), `cuda_memcpy_sync`,
`cuda_memset_sync`, `cuda_api_sync`, `gpu_gaps` (**default 500 ms — far too coarse
for a 200 ms training step; pass `gap=1` for ML work**), `gpu_time_util`
(under 50% across 30 chunks).

---

## 3. Collection: ncu

### 3.1 One-pass complete collection

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
    --config my_utils/profiling/ncu/ncu_full_collection.yaml
```

`ncu_full_collection.yaml` pins **every section the diagnosis engine reads**. This
is deliberate belt-and-braces: `--set full` covers them today, but NVIDIA has moved
sections between sets before (SchedulerStats and WarpStateStats left `--set
detailed` in a recent release, silently removing stall analysis from anyone relying
on it).

The equivalent raw command:

```bash
ncu --set full --import-source yes \
    --target-processes all \
    --kernel-name "regex:gemm|flash|elementwise" \
    --launch-count 3 --launch-skip 10 \
    -o report python train.py
```

| Section | Unlocks |
|---|---|
| `SpeedOfLight` | compute/memory bound classification |
| `SpeedOfLight_RooflineChart` | roofline placement |
| `ComputeWorkloadAnalysis` | pipe utilisation, tensor-core engagement, IPC |
| `MemoryWorkloadAnalysis` | DRAM/L1/L2 traffic and hit rates |
| `MemoryWorkloadAnalysis_Tables` | per-space hit rates, bank conflicts, coalescing ratios |
| `SchedulerStats` | issue-slot utilisation, eligible warps |
| `WarpStateStats` | the 19 stall reasons — the backbone of latency diagnosis |
| `Occupancy` | achieved vs theoretical, and the binding limiter |
| `LaunchStats` | grid/block/registers/waves, tail-effect input |
| `InstructionStats` | FLOP counters (roofline), instruction mix, spilling |
| `SourceCounters` | per-line coalescing and divergence (needs `-lineinfo`) |
| `WorkloadDistribution` | SM/L2/DRAM load imbalance |

Build with `-lineinfo` (`nvcc --generate-line-info`) or SourceCounters findings can
only point at SASS addresses, not source lines.

### 3.2 Scope the capture

`--set full` replays each kernel tens of times. Always narrow it:

| Flag | Use |
|---|---|
| `-k "regex:..."` / `--kernel-name` | Only the kernels you care about |
| `-c 3 -s 10` | 3 launches, after skipping 10 (skip warm-up) |
| `--nvtx --nvtx-include "step/"` | Only kernels inside an NVTX range |
| `--filter-mode per-launch-config` | One sample per distinct grid/block shape |
| `--replay-mode range` | Whole-range metrics for a marked region |
| `--graph-profiling graph` | Treat a CUDA graph as one workload |

### 3.3 Intra-kernel timelines (separate run)

Whole-kernel counters are aggregates. To see utilisation *over time inside* a
kernel — flat-low vs long tail vs sawtooth — use PM sampling, in its own session
because it contends for the same counter hardware:

```bash
ncu --set pmsampling --pm-sampling-interval 100000 -o pmsample python train.py
```

This is the only option for persistent/megakernel designs (ThunderKittens
interpreter, Triton-distributed `mega_kernel_*`, whole-step CUDA graphs), where one
launch covers the entire step and per-kernel attribution stops being meaningful.
`classify_kernel(name).megakernel` flags these so a report can say so instead of
reporting "one kernel took 100% of the time".

### 3.4 Analysis commands

```bash
ncu-diagnose --report report.ncu-rep --gpu "H100 SXM5" --format md   # full diagnosis
ncu-report-analyze --report report.ncu-rep --format md               # metric summary
ncu -i report.ncu-rep --page raw --csv > all_metrics.csv             # bulk dump
```

`--gpu` is what unlocks absolute roofline ceilings. Without it the engine still
computes arithmetic intensity but has no peak to compare against. Accepted names
are anything `lookup_gpu_spec` matches — see [§8](#8-hardware-ceilings).

---

## 4. Analysis: the top-down triage

`my_utils/profiling/analyzers/triage.py` answers **one** question: where did the
step's time actually go? It runs an ordered tree and returns a single verdict.

```python
from my_utils.profiling.analyzers.triage import triage_step, TriageThresholds

verdict = triage_step(
    wall_ns=step_end - step_start,
    compute_intervals=[(k.start, k.end) for k in compute_kernels],
    comm_intervals=[(k.start, k.end) for k in nccl_kernels],
    memcpy_intervals=[(m.start, m.end) for m in memcpies],
    launch_api_ns=total_cuda_launch_kernel_ns,
    sync_api_ns=total_blocking_sync_ns,
    kernel_durations_ns=[k.end - k.start for k in all_kernels],
    thresholds=TriageThresholds(cuda_graphs=True),   # optional
)
print(verdict.verdict)     # host_bound | communication_bound | transfer_bound |
                           # launch_bound | kernel_bound
print(verdict.summary)
for step in verdict.next_steps:
    print(" -", step)
```

### The tree

```
1. >= 2 host signals crossed?          -> host_bound
     (idle share, launch-API share, GPU utilisation)
2. exposed comm / wall > 15%?          -> communication_bound
3. H2D+D2H / wall > 10%?               -> transfer_bound
4. > 25% of kernels under 10 us?       -> launch_bound
5. otherwise                           -> kernel_bound
```

Two design decisions worth understanding:

**Communication is judged on exposure, not volume.** The same 75 ms of collectives
fully hidden under compute is *not* a finding; exposed for 75 ms it is the finding.
`exposed_comm = comm_union - (comm ∩ compute)`.

**Host-bound needs two signals.** A single crossed threshold is noise. This
reproduces NVIDIA's own rule from their calibrated perf-analysis constants.

### Thresholds and where they come from

| Threshold | Default | Source |
|---|---|---|
| `gpu_idle_ratio` | 0.30 (0.15 with CUDA graphs) | NVIDIA perf-host-analysis M1 |
| `launch_overhead_ratio` | 0.10 | NVIDIA M2 |
| `gpu_utilization_floor` | 0.60 (0.80 graphs) | NVIDIA M4 |
| `comm_ratio` | 0.20 | NVIDIA M5 |
| `host_bound_signals_required` | 2 | NVIDIA ">= 2 metrics crossing" |
| `exposed_comm_ratio` | 0.15 | practitioner consensus |
| `short_kernel_us` | 10.0 | HTA short-kernel definition |
| `launch_delay_outlier_us` | 100.0 | HTA `launch_delay_cutoff` |
| `runtime_outlier_us` | 50.0 | HTA `runtime_cutoff` |
| `max_launch_queue` | 1024 | HTA `CUDA_MAX_LAUNCH_QUEUE_PER_STREAM` |
| `gap_threshold_us` | 1000 | nsys default is 500 **ms**; tightened for ML steps |

NVIDIA calibrated theirs on LLM *inference* on a B200 and explicitly says to loosen
them where GPU time legitimately dominates. Every threshold is a field on
`TriageThresholds` — override rather than fork.

### Secondary signals

Reported alongside the verdict, never as the verdict:

- **Launch queue at ~1024** — the host is blocked *in* launch calls, which means
  the GPU is the constraint, not the CPU. This is the opposite conclusion from an
  idle GPU and is easy to get backwards.
- **Steady-state `cudaMalloc`/`cudaFree`** — the caching allocator is missing
  (shape churn or fragmentation), and `cudaFree` synchronises the device.
- **Launch-delay outliers** — kernels waiting > 100 us between launch and start.

---

## 5. Analysis: per-kernel diagnosis

`my_utils/profiling/ncu/ncu_diagnostics.py`.

```python
from my_utils.profiling.ncu.ncu_diagnostics import diagnose_kernel
from my_utils.profiling.hardware.gpu_specs import lookup_gpu_spec

result = diagnose_kernel(
    metrics,                                   # {ncu metric name: value}
    kernel_name="ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn",
    gpu_spec=lookup_gpu_spec("H100 SXM5"),
    top_k=10,
)
result["verdict"]                # compute_bound | memory_bound | latency_bound |
                                 # small_grid | launch_bound | balanced | *_leaning
result["findings"]               # ranked, each with evidence + actions + ceiling
result["sections"]["stalls"]     # full stall ranking and bucket rollup
result["sections"]["roofline"]   # AI, achieved/attainable TFLOP/s, ridge point
```

Or straight from a report:

```python
from my_utils.profiling.ncu.ncu_report_tools import diagnose_ncu_report
payload = diagnose_ncu_report("report.ncu-rep", gpu_name="H100 SXM5", top_kernels=10)
```

### What each analysis contributes

| Function | Detects |
|---|---|
| `classify_bottleneck` | 4-way SOL classification + which section to read next |
| `analyze_stalls` | Ranked stall reasons, DrGPU-style bucket rollup, per-reason fixes |
| `compute_roofline` | AI, achieved vs attainable TFLOP/s, memory vs compute side |
| `analyze_occupancy` | Binding limiter (registers/smem/blocks/warps/barriers), achieved-vs-theoretical gap |
| `analyze_launch_config` | Block size not warp-multiple, small grid, tail wave |
| `analyze_coalescing` | Excess sectors, sectors/request, bytes/sector, cache locality |
| `analyze_shared_memory` | Bank conflicts as a share of wavefronts, n-way factor |
| `analyze_divergence` | Threads-per-instruction, branch efficiency |
| `analyze_spilling` | Local memory traffic, register pressure, L1 local hit rate |
| `analyze_pipes` | Busiest pipe, unexpected FP64 |
| `analyze_imbalance` | SM / L2-slice / DRAM-partition imbalance |

### Findings carry evidence and a ceiling, not a prediction

```python
{
  "category": "uncoalesced_global_access",
  "severity": "high",
  "summary": "860,000 sectors above the ideal (86% of all global sectors). ...",
  "evidence": {"sectors_actual": 1000000, "sectors_ideal": 140000, ...},
  "actions": ["Make consecutive threads read consecutive addresses.", ...],
  "speedup_ceiling": 7.14,     # UPPER BOUND, following GPA's T/(T-M) model
}
```

`speedup_ceiling` is what you get if the problem vanishes entirely — it is a
ceiling, never a forecast. Latency-hiding fixes are capped at 2x because you can
only fill issue slots you already have.

### It refuses to conclude when the data cannot support it

If the tensor pipe was busy but no `sm__ops_path_tensor_*` metric was collected,
the FLOP total counts CUDA-core instructions only. Rather than reporting a
tensor-core GEMM at "13% of roofline", the engine emits
`roofline_needs_tensor_counters` and names the section to collect. Any analyzer
that does not do this will confidently mis-grade every tensor-core kernel.

---

## 6. The metric catalog

`my_utils/profiling/ncu/metric_catalog.py` — 103 metrics keyed by stable short
names, each with per-architecture spelling candidates.

```python
from my_utils.profiling.ncu.metric_catalog import (
    METRIC_CATALOG, STALL_REASONS, resolve_metric, metrics_for_category, describe_arch)

resolve_metric("achieved_occupancy").names
# ('sm__warps_active.avg.pct_of_peak_sustained_active',)
metrics_for_category("coalescing")     # every coalescing metric
describe_arch(9, 0, 132)               # {'family': 'hopper', 'alias': 'h100/sm_90', ...}
```

**Never hard-code a metric string.** Ada appends `_v2` to the tensor pipe metrics,
Blackwell renames `dram__bytes_read` to `dram__bytes_op_read`, and Hopper adds
`sm__pipe_tma_*` that older parts lack. `MetricView.get("pipe_tensor_util")` tries
every known spelling; `view.raw("sm__pipe_tensor_cycles_active...")` does not.

### Key metric cheat sheet

| Quantity | Metric |
|---|---|
| Compute SOL % | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| Memory SOL % | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` |
| DRAM SOL % | `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Theoretical occupancy | `sm__maximum_warps_per_active_cycle_pct` |
| Issue-slot utilisation | `smsp__issue_active.avg.per_cycle_active` |
| Warp stall (per reason) | `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` |
| Coalescing | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` (4 ideal fp32, 32 worst) |
| Wasted bandwidth | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio` (32 ideal) |
| Bank conflicts | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` |
| Tensor-core usage | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` |
| FP32 FLOPs | `smsp__sass_thread_inst_executed_op_{fadd,fmul,ffma}_pred_on.sum` (add + mul + 2*fma) |
| Divergence | `smsp__thread_inst_executed_per_inst_executed.ratio` (32 ideal) |
| Waves per SM | `launch__waves_per_multiprocessor` |
| Register spilling | `smsp__sass_inst_executed_op_local_{ld,st}.sum` |

### The 19 stall reasons

Complete set, bucketed as in DrGPU's top-down decomposition:

| Bucket | Reasons |
|---|---|
| device_memory | `long_scoreboard`, `lg_throttle`, `tex_throttle`, `drain` |
| shared_memory | `short_scoreboard`, `mio_throttle` |
| synchronization | `barrier`, `membar` |
| instruction | `wait`, `math_pipe_throttle`, `no_instruction`, `branch_resolving`, `dispatch_stall`, `gmma` |
| other | `not_selected`*, `selected`*, `imc_miss`, `sleeping`, `misc` |

`*` benign — `selected` is productive time and `not_selected` means you have
*surplus* warps. An analyzer that reports these as bottlenecks is wrong.

```python
from my_utils.profiling.ncu.metric_catalog import STALL_REASONS
r = STALL_REASONS["long_scoreboard"]
r.meaning       # "Waiting on an L1TEX ... data return - i.e. memory latency."
r.fixes         # concrete actions
r.metric_name   # exact ncu metric
```

---

## 7. Interpretation reference

### 7.1 Bottleneck classification (NVIDIA's SOLBottleneck thresholds)

| Compute SOL | Memory SOL | Verdict | Read next |
|---|---|---|---|
| >= 80 | — | compute_bound | ComputeWorkloadAnalysis |
| — | >= 80 | memory_bound | MemoryWorkloadAnalysis |
| < 60 | < 60, waves < 1 | small_grid | LaunchStats |
| < 60 | < 60 | latency_bound | WarpStateStats |
| diff >= 10 | | compute/memory_leaning | the leading side |
| diff < 10 | | balanced | roofline |

Grading: **> 80 excellent, 60-80 good, 40-60 fair, < 40 poor.**

### 7.2 Symptom to cause

| Symptom | Likely cause | Fix |
|---|---|---|
| `long_scoreboard` dominant | Global memory latency | Coalescing, shared-memory staging, more ILP |
| `short_scoreboard` dominant | Shared-memory dependency | Remove bank conflicts, use registers |
| `mio_throttle` dominant | Too many shared/special-math ops | Fewer, wider shared accesses |
| `lg_throttle` dominant | Local/global queue full | Kill spills, vectorise loads |
| `barrier` dominant | Uneven work before `__syncthreads` | Balance paths, split large blocks |
| `math_pipe_throttle` | One pipe oversubscribed | Rebalance instruction mix |
| `not_selected` dominant | *Surplus* parallelism | Nothing — possibly fewer warps for cache locality |
| Sectors/request ~32 | Uncoalesced | Fix lane-to-address mapping |
| Bytes/sector < 16 | Strided/predicated access | Compact or gather via shared memory |
| Bank conflicts >= 10% | Shared layout | Pad leading dimension, or XOR swizzle |
| Achieved << theoretical occupancy | Tail wave or warp imbalance | Size grid to whole waves |
| `launch__occupancy_limit_registers` binding | Register pressure | `__launch_bounds__`, shorter live ranges |
| Local ld/st > 0 | Register spilling | Reduce per-thread tile; watch L1 local hit rate |
| Threads/inst < 24 of 32 | Divergence | Sort/bucket by branch condition, predicate |
| Waves 1.14 | Tail effect | Size grid to a whole number of waves |
| Tensor pipe ~0 on a GEMM | Not on tensor cores | AMP, align M/N/K, check for `simt`/`sgemm` in the name |
| FP64 pipe active in a BF16 model | Stray double literals | Audit `1.0` vs `1.0f`, `exp` vs `expf` |

### 7.3 Per-category expectations

A finding must be relative to what that *kind* of kernel should achieve. An
elementwise kernel not using tensor cores is correct; a GEMM not using them is a
bug.

| Category | Should be | Floor | Note |
|---|---|---|---|
| matmul | compute bound | 60% compute SOL | Large aligned GEMM: 85-95% of cuBLAS |
| attention | compute bound (prefill) | 50% compute SOL | **Decode is memory bound by definition** — grade on KV bytes/s |
| elementwise | memory bound | 70% DRAM SOL | Should reach 80-95% |
| normalization | memory bound | 60% DRAM SOL | Fuse with neighbours |
| softmax | memory bound | 55% DRAM SOL | SFU-limited: H100 exp is 3.9 vs 989 TFLOPS matmul |
| memory_ops | memory bound | 70% DRAM SOL | Usually removable entirely |
| communication | network bound | — | Grade against bus bandwidth, not SM/DRAM SOL |

Encoded in `CATEGORY_EXPECTATIONS` in `sources/kernel_taxonomy.py`.

### 7.4 GEMM-specific

- **Tile quantisation:** waste = `1 - (M*N) / (ceil(M/Tm)*Tm * ceil(N/Tn)*Tn)`. Flag > 10%.
- **Wave quantisation:** `waves = tiles / SMs`. A fractional tail costs a whole
  wave of time. 108 tiles on A100 = 1 clean wave; 117 tiles = 2 waves with the
  second at 8% occupancy, roughly halving throughput.
- **Alignment:** multiples of 8 (fp16), 16 (int8), 4 (tf32); on A100 best at 64/128/32.
- **Tile preference:** 256x128 / 128x256 > 128x128 > 256x64 / 64x256 > 64x64.

### 7.5 Communication

`busbw = algbw * factor`, `algbw = message_bytes / time`:

| Collective | Factor |
|---|---|
| AllReduce | `2(n-1)/n` |
| AllGather / ReduceScatter / AllToAll | `(n-1)/n` |
| Broadcast / Reduce | `1` |

Ceilings: 8xA100 NVLink3 ~230-240 GB/s; 8xH100 NVLink4 ring ~360 GB/s, ~480 GB/s
with NVLS (NVLS busbw legitimately exceeds the 450 GB/s link spec — it is a
normalised figure, not an error). Inter-node per rail: IB HDR 25 GB/s, NDR 50 GB/s.
Flag when large-message busbw is below ~70-80% of the ceiling.

Protocols, from the kernel name: **LL** (low latency, ~half bandwidth), **LL128**
(~95% bandwidth, ~2 us), **Simple** (full bandwidth, ~6 us). **PAT** (NCCL 2.23.4+)
is AllGather/ReduceScatter only and requires one GPU per node.

**Overlapped NCCL steals SMs** — roughly 1.3x slowdown on both the compute and the
comm. "Overlap enabled but the GEMMs got slower" is a real and expected outcome.

---

## 8. Hardware ceilings

`my_utils/profiling/hardware/gpu_specs.py` — 25 SKUs, Volta through Blackwell.

```python
from my_utils.profiling.hardware.gpu_specs import lookup_gpu_spec
spec = lookup_gpu_spec("NVIDIA H100 80GB HBM3")   # fuzzy match on nsys/ncu names
spec.peak_tflops("bf16")            # 989.4  (DENSE)
spec.peak_tflops("bf16", sparse=True)  # 1978.8
spec.ridge_point("bf16")            # 295.3 FLOP/byte
spec.attainable_tflops(ai=50, dtype="bf16")   # roofline ceiling at that AI
spec.effective_hbm_gbps()           # measured where known, else 85% of spec
```

**All peaks are dense.** Datasheets quote tensor numbers *with sparsity*; grading
achieved FLOP/s against a 2x-inflated peak silently halves every utilisation
number. L40S is the classic trap: the datasheet says 362 TFLOPS BF16, the dense
figure is 181.

### Ridge points (FLOP/byte where memory-bound becomes compute-bound)

| GPU | BF16 | FP8 | HBM |
|---|---|---|---|
| H100 SXM5 | 295 | 591 | 3.35 TB/s |
| H100 PCIe | 378 | 757 | 2.0 TB/s |
| H200 SXM | 206 | 412 | 4.8 TB/s |
| H800 SXM | 295 | 591 | 3.35 TB/s (NVLink cut to 400 GB/s) |
| H20 | 37 | 74 | 4.0 TB/s |
| A100 80GB | 153 | — | 2.04 TB/s |
| B200 | 281 | 563 | 8.0 TB/s |
| L40S | 209 | 419 | 0.864 TB/s |

`AI < ridge` -> memory bound, ceiling is `AI * bandwidth`. `AI > ridge` -> compute
bound, ceiling is the dtype peak. H200 has the *lowest* Hopper ridge, so it turns
more kernels compute-bound than an H100 does; H20 at 37 is memory-rich and
compute-poor, so almost everything on it is compute bound.

### Cost-model constants worth carrying

- A100: a non-matmul FLOP costs ~**16x** a tensor-core FLOP (312 vs 19.5 TFLOPS).
- H100: `MUFU.EX2` (exp) runs at **3.9 TFLOPS** vs 989 for FP16 matmul — 256x. This
  is why softmax can eat ~50% of an attention kernel's cycles.
- H100: plain `mma` caps around **63% of peak**; `wgmma` is required for the rest.
- Latency budget (cycles): shared ~19-29, L1 ~28-33, L2 ~200-260 near / ~360-414
  far, HBM ~466-656.

---

## 9. Kernel taxonomy

`my_utils/profiling/sources/kernel_taxonomy.py`.

```python
from my_utils.profiling.sources.kernel_taxonomy import (
    classify_kernel, parse_gemm_kernel, parse_nccl_kernel, is_megakernel)

info = classify_kernel("triton_poi_fused_add_mul_silu_23")
info.category      # 'activation'
info.framework     # 'triton'
info.triton_kind   # 'pointwise'
info.fused_ops     # ('add', 'mul', 'silu')
info.tensor_cores  # None  (name carries no evidence)
info.megakernel    # False
```

`tensor_cores` is tri-state on purpose: `None` means the name says nothing.
"Not using tensor cores" is only actionable when the name actually says so
(`simt`, `sgemm`).

### Framework fingerprints

| Framework | Signature |
|---|---|
| torch.compile | `triton_{poi,red,per,tem,for}_fused_*` |
| CUTLASS | `cutlass`, `sm90_`, `sm100_`, `nvjet`, `cute::`, `tcgen05` |
| cuBLAS | `ampere_`, `turing_`, `volta_`, `sm*_xmma` |
| FlashAttention | `flash_fwd`, `flash_bwd`, `flash::` |
| TransformerEngine | `te_`, `cast_transpose`, `layernorm_geglu` |
| ThunderKittens | `kittens::`, `_ZN7kittens`, `fwd_attend_ker`, `layernorm_tk` |
| Triton-distributed | `kernel_dispatch_token`, `kernel_gemm_rs_producer`, `moe_grouped_gemm` |
| NCCL | `ncclDevKernel_*` |

ThunderKittens note: `rt_bf`/`st_bf` are `using` aliases and **never** survive
mangling — searching a trace for them finds nothing. Match `kittens::` or the
mangled `_ZN7kittens` instead. Set `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` to make
Inductor embed the fused ATen op names.

### Kernel-name decoder

| Pattern | Diagnosis | Fix |
|---|---|---|
| Swarms of `vectorized_elementwise_kernel` between GEMMs | Unfused eager pointwise | `torch.compile` |
| `CatArrayBatchedCopy`, `direct_copy_kernel`, `nhwcToNchw` | Layout churn | Consistent memory format |
| `Memcpy DtoH` + `cudaStreamSynchronize` mid-step | Host sync (`.item()`, `print`, nonzero) | `torch.where`, defer logging |
| `volta_sgemm`, anything `simt` | Tensor cores unused | AMP, align dims |
| `ncclDevKernel_*` alone on its stream | Exposed communication | Overlap flags |
| One kernel spanning the whole step | Megakernel | Switch to PM sampling |

---

## 10. Traps that make numbers lie

Read this before trusting any number.

1. **ncu flushes caches and locks clocks.** L2 hit rates under ncu are *not* what
   production sees, because each replay starts cold. `--cache-control none`
   preserves cache state but costs reproducibility.
2. **ncu serialises kernels.** Any conclusion about overlap, concurrency or stream
   parallelism from an ncu report is invalid. Use nsys.
3. **Sparsity-inflated peaks.** Datasheet tensor TFLOPS are usually "with
   sparsity". Our tables store dense; if you bring your own peak, halve it.
4. **Zero-initialised benchmark data inflates tensor-core throughput.** Operand
   values change power draw and therefore sustained clocks. Benchmark with random
   data.
5. **Sustained clocks are not boost clocks.** An H800 under sustained `wgmma` drops
   from 1755 MHz to 1200-1275 MHz at its power cap. "% of peak" computed against
   boost-clock peak will read low for reasons that are not the kernel's fault.
6. **`nvidia-smi` utilisation is time-utilisation.** It reports the fraction of
   time *any* kernel was resident. A deadlocked NCCL kernel shows 100%. The ladder
   is: allocation -> kernel-time -> SM occupancy -> MFU.
7. **Warm-up is not steady state.** CUDA context init, autotuning and allocator
   growth all land in the first iterations. Profile from step ~10.
8. **Profiling only rank 0 misattributes stragglers.** A late rank stretches every
   rank's collective, which looks exactly like a slow network from inside one rank.
9. **Occupancy is not performance.** CUTLASS deliberately runs GEMMs at low
   occupancy with deep software pipelines. Low occupancy plus high tensor-pipe
   utilisation is a *good* kernel.
10. **Per-kernel metrics are meaningless for a megakernel.** Check
    `classify_kernel(name).megakernel` and switch to PM sampling.
11. **Averaging across shapes hides everything.** A GEMM at three different shapes
    is three kernels; averaging their SOL produces a number describing none of them.
    Use `--filter-mode per-launch-config`.
12. **Profiler overhead distorts gaps.** CUPTI buffer flushes appear as GPU idle.
    Subtract `PROFILER_OVERHEAD` and `CUPTI_ACTIVITY_KIND_OVERHEAD` intervals before
    attributing idle time — NVIDIA's own `gpu_gaps` rule does.
13. **`CUDA_LAUNCH_BLOCKING=1` poisons every trace.** It serialises launches, so
    every async behaviour you are trying to measure disappears.

---

## 11. Python API reference

```
my_utils/profiling/
├── hardware/gpu_specs.py            GpuSpec, lookup_gpu_spec, list_known_gpus
├── sources/kernel_taxonomy.py       classify_kernel, parse_gemm_kernel,
│                                    parse_nccl_kernel, is_megakernel,
│                                    uses_tensor_cores, CATEGORY_EXPECTATIONS
├── ncu/metric_catalog.py            METRIC_CATALOG, STALL_REASONS, resolve_metric,
│                                    describe_arch, SECTION_SETS
├── ncu/ncu_diagnostics.py           diagnose_kernel, classify_bottleneck,
│                                    analyze_stalls, compute_roofline, MetricView,
│                                    SOL_THRESHOLDS, Finding
├── ncu/ncu_report_tools.py          diagnose_ncu_report, analyze_ncu_report,
│                                    NcuReportSkillEngine
├── analyzers/triage.py              triage_step, TriageThresholds, TriageVerdict,
│                                    merge_intervals, interval_overlap_ns
├── sources/nsys_analyze.py          analyze_nsys_sqlite
├── sources/nsys_auto_analysis.py    build_comprehensive_analysis
└── sources/nsys_sql_skills.py       NsysSqlSkillEngine (29 SQL skills)
```

These modules are **pure analysis** — no torch, no CUDA. They import and run on a
laptop with nothing installed, which is what makes them testable.

`tests/profiling/test_analysis_engine.py` (80 tests) is the executable spec.

### Common recipes

```python
# Rank kernels by category to see the shape of a workload
from my_utils.profiling.sources.kernel_taxonomy import summarize_categories
summarize_categories([k.name for k in kernels])
# {'matmul': 412, 'elementwise': 3311, 'communication': 96, ...}
#  ^ 3311 elementwise kernels against 412 matmuls is a fusion problem

# Is this kernel worth optimising at all?
from my_utils.profiling.ncu.ncu_diagnostics import diagnose_kernel
d = diagnose_kernel(metrics, kernel_name=name, gpu_spec=spec)
best = max((f["speedup_ceiling"] or 1.0) for f in d["findings"]) if d["findings"] else 1.0
# best is the optimistic ceiling; if it is 1.05 the kernel is already fine
```

---

## 12. Extending the engine

**Add a GPU:** append a `GpuSpec` to `_GPU_SPECS` in `hardware/gpu_specs.py`. Keep
the list most-specific-first (`h100 sxm` before `h100`). Use dense peaks. Add a
lookup test.

**Add a kernel pattern:** add to `KERNEL_CATEGORIES` (ordered, first match wins) or
`_FRAMEWORK_PATTERNS` in `sources/kernel_taxonomy.py`. Demangled names start with a
return type, so anchor with `\b`, not `^`. Mangled symbols are case-sensitive and
are matched against the raw name.

**Add a metric:** add a `MetricSpec` to `_CATALOG_LIST` in `ncu/metric_catalog.py`
with every architecture spelling in `names`, most-preferred first.

**Add a rule:** write an `analyze_*(view: MetricView) -> dict` in
`ncu/ncu_diagnostics.py` returning `{"findings": [Finding(...)], ...}`, and register
it in `diagnose_kernel`. Rules must:
- read metrics through `view.get("catalog_key")`, never a raw string;
- put the numbers that produced the finding in `evidence`;
- emit `speedup_ceiling` only as an upper bound;
- stay silent when the required metrics are absent, rather than assuming zero.

**Change a threshold:** they live in `SOL_THRESHOLDS` (ncu) and `TriageThresholds`
(nsys). Both are documented with their source. Prefer overriding at the call site.

---

## Sources

Thresholds and formulas are taken from:

- Nsight Compute shipped rule sources (`<install>/sections/*.py`) — SOLBottleneck,
  CPIStall, Occupancy, LaunchConfiguration, UncoalescedAccess,
  SharedMemoryConflicts, ThreadDivergence, WorkloadImbalance
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) ·
  [CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) ·
  [Python Report Interface](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) ·
  [Analysis Guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [NVIDIA perf-analysis skills in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills) — the M1-M5 host-bound constants
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) — idle breakdown, launch stats, queue length, stragglers
- [NVIDIA Matrix Multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) — tile/wave quantisation
- [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) — busbw factors
- GPA (CGO 2021, [arXiv:2009.04061](https://arxiv.org/abs/2009.04061)) — matched-stall speedup model
- DrGPU (ICPE 2023) — top-down stall decomposition
- Luo et al., [arXiv:2402.13499](https://arxiv.org/abs/2402.13499) — Hopper microbenchmarks
- Jia et al., [arXiv:1804.06826](https://arxiv.org/abs/1804.06826) — Volta microbenchmarks
- [Horace He, "Making Deep Learning Go Brrrr"](https://horace.io/brrr_intro.html) — the compute/memory/overhead trichotomy
- [Simon Boehm, CUDA matmul worklog](https://siboehm.com/articles/22/CUDA-MMM) — the optimisation ladder
- [stas00/ml-engineering](https://github.com/stas00/ml-engineering) — MFU/HFU, network expectations
