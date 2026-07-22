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
- [5b. The 14 axes: what "complete" means](#5b-the-14-axes-what-complete-means)
- [5c. Cross-checking against NVIDIA's own rules](#5c-cross-checking-against-nvidias-own-rules)
- [6. The metric catalog](#6-the-metric-catalog)
- [6b. Every metric in the report, not just the catalogued ones](#6b-every-metric-in-the-report-not-just-the-catalogued-ones)
- [7. Interpretation reference](#7-interpretation-reference)
- [8. Hardware ceilings](#8-hardware-ceilings)
- [9. Kernel taxonomy](#9-kernel-taxonomy)
- [9b. Never conclude from a kernel name alone](#9b-never-conclude-from-a-kernel-name-alone)
- [9c. Trace validity: refuse before you analyse](#9c-trace-validity-refuse-before-you-analyse)
- [9d. Clock throttling](#9d-clock-throttling)
- [9e. Where, not what: source and SASS attribution](#9e-where-not-what-source-and-sass-attribution)
- [9f. Sampling validity: PC and PM sampling](#9f-sampling-validity-pc-and-pm-sampling)
- [9g. How a measurement was taken](#9g-how-a-measurement-was-taken)
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
result["coverage"]               # WHICH analyses ran, and which could not
result["axes"]                   # WHICH of the 14 axes were examined (see 5b)
result["metric_inventory"]       # every metric present, interpreted or not (6b)
result["corroboration"]          # agreement/conflict with NVIDIA's rules (5c)
result["throttling"]             # populated only when telemetry is supplied (9d)
```

Optional inputs, each of which closes a gap that is otherwise reported honestly
as unexamined rather than silently skipped:

```python
diagnose_kernel(
    metrics,
    kernel_name=name,
    gpu_spec=spec,
    shipped_rules=load_ncu_report_rule_rows("profile.ncu-rep"),  # 5c
    throttling={"clock_event_mask": mask},                       # 9d
    problem_shape={"m": 4096, "n": 4096, "k": 1024},             # tile quantisation
    collection={"cache_control": "none", "clocks_locked": True}, # 9g
)
```

`collection` changes no finding. It records what the numbers may be compared
against, and defaults to ncu's own defaults - cold cache, serialised execution -
because that is what almost every run uses:

```python
result["measurement_context"]["cache_state"]      # 'cold'
result["measurement_context"]["cannot_answer"]    # pipeline speed, overlap
```

`problem_shape` deserves a note: the kernel symbol encodes the *tile* shape and
never the *problem* shape, so tile quantisation cannot be checked without it.
When it is absent the check reports itself as unasked rather than passing.

### Read `coverage` before you read `findings`

This is the single most important habit with this tool. An analysis that never
ran, because its section was not collected, produces exactly what a healthy
analysis produces: **nothing**. Two findings can mean two problems, or two
problems plus nine questions nobody asked.

```python
cov = result["coverage"]
print(cov["summary"])
# 2 of 17 analyses ran. 15 could not: the required metrics are absent from this
# report, so those questions were not asked - this is missing coverage, not a
# clean result.

for skipped in cov["skipped"]:
    print(skipped["analysis"], "needs", skipped["needs_section"])
# stalls          needs WarpStateStats
# shared_memory   needs MemoryWorkloadAnalysis_Tables
# spilling        needs SourceCounters
```

Measured against a report carrying only Speed-of-Light and launch metrics -
what `--set basic` gives you - coverage is **2 of 15**. The thirteen that cannot
run include stalls, shared memory and source attribution: the first things you
would want for a fused kernel. `--set full` closes most of it; source
correlation additionally needs `-lineinfo` at build time (see 9e).

The number moved from "6 of 11" to "2 of 15" between versions of this handbook,
and both parts of that are worth knowing. Four analyses were added. But the
older figure was also **wrong**: `_ANALYSIS_REQUIREMENTS` gated four analyses on
catalog keys that did not exist (`warp_cycles_per_issue` for what is really
`warp_cycles_per_issued_inst`, and six more). `MetricView.get` returns `None`
for an unknown key rather than raising, so stalls, coalescing, divergence and
spilling were reported as uncollected on *every* report, including full ones
where those rules had run and emitted findings. A coverage report that lies is
worse than no coverage report, so a test now pins every key to the catalog.

### Every finding states its evidence

A conclusion without the numbers that produced it is an assertion. Each finding
carries the metric values it was derived from, what it means, an action, and an
optimistic ceiling:

```python
for f in result["findings"]:
    print(f"[{f['severity']}/{f['confidence']}] {f['title']}")
    print(" ", f["summary"])
    print("  evidence:", f["evidence"])
    print("  actions :", f["actions"])
    print("  ceiling :", f["speedup_ceiling"])
```

Real output from the fused-kernel report above:

```
[low/medium] SM load imbalance of 8%
  The busiest sm unit is active for 305,542 cycles against an average of 282,228.
  evidence: {'sm_cycles_avg': 282228, 'sm_cycles_max': 305542, 'imbalance': 0.0763}
  ceiling : 1.08x
```

`confidence` is not decoration. It is capped by the quality of the evidence
underneath: a finding that depends on the kernel *name* being right can never
exceed `medium`, and drops to `low` when the name is advisory (a truncated
symbol, or CUTLASS code that may have come from cuBLASLt).

A finding whose category is `measurement_caveat` means a number in the report is
known to be wrong - for example FP32 FLOPs on a CC 10.x report that lacks the
packed-FP32 counters, which undercounts by up to 2x. Fix the collection before
using that number.

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
| `analysis_coverage` | **Which analyses could not run**, and the section each needs |
| `analyze_coalescing` | Excess sectors, sectors/request, bytes/sector, cache locality |
| `analyze_shared_memory` | Bank conflicts as a share of wavefronts, n-way factor |
| `analyze_divergence` | Threads-per-instruction, branch efficiency |
| `analyze_spilling` | Local memory traffic, register pressure, L1 local hit rate |
| `analyze_pipes` | Busiest pipe, unexpected FP64 |
| `analyze_imbalance` | SM / L2-slice / DRAM-partition imbalance |

### Findings carry evidence and a ceiling, not a prediction

```jsonc
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

## 5b. The 14 axes: what "complete" means

A findings list cannot tell you what was *not* examined. Silence from the
communication axis means either "collectives are fine" or "nobody looked", and
those call for opposite next steps. `analyzers/axes.py` is the checklist that
makes the difference visible.

```python
from my_utils.profiling.analyzers.axes import AXES, axis_coverage, axis_for_category

result["axes"]["summary"]
# 4 of 14 axes examined. Not examined: shared_memory, stall, divergence,
# registers, communication, latency_launch, host_pipeline, power_clock,
# numerics, multi_gpu. Those axes produced no findings because they were never
# checked, which is not the same as being clean.

for axis in result["axes"]["axes"]:
    if not axis["examined"]:
        print(axis["axis"], "->", axis["remedy"])
# stall        -> --section WarpStateStats
# power_clock  -> Sample nvmlDeviceGetCurrentClocksEventReasons or DCGM 100/112/155/240/241
```

| Axis | Question it answers | Where it comes from |
|---|---|---|
| `compute` | Are the math pipes the limit, and the right pipe? | ncu SOL, pipes, roofline |
| `memory_bandwidth` | Which level is saturated, and is the traffic avoidable? | ncu memory sections |
| `shared_memory` | Is shared memory serialising on bank conflicts? | MemoryWorkloadAnalysis_Tables |
| `scheduler` | Do schedulers have warps to issue; is the grid shaped right? | Occupancy, LaunchStats, SchedulerStats |
| `stall` | When warps cannot issue, what are they waiting on? | WarpStateStats |
| `divergence` | Are threads in a warp doing the same work? | InstructionStats |
| `registers` | Is register pressure spilling to local memory? | LaunchStats, SourceCounters |
| `communication` | Are collectives at achievable bus bandwidth; who is late? | nsys + NCCL flight recorder |
| `latency_launch` | Is the GPU idle waiting for the host? | nsys timeline |
| `host_pipeline` | Is the input pipeline or Python the limit? | nsys with `--trace=osrt` |
| `power_clock` | Did the GPU run at the clock the numbers assume? | NVML / DCGM telemetry |
| `numerics` | Is the kernel using the narrowest precision allowed? | tensor-op counters |
| `multi_gpu` | Are all ranks doing equal work at the same time? | per-rank traces |
| `measurement` | Is this data trustworthy enough to conclude anything? | always available |

**Three axes can never be closed by ncu alone.** `communication`,
`latency_launch` and `host_pipeline` need a timeline; `power_clock` needs
external telemetry, because Nsight Compute reports no metric for the clock a
kernel actually ran at. A per-kernel report showing those as gaps is correct
behaviour, not a missing feature.

Why axes exist at all: three vocabularies described the same things. Our engine
emitted `uncoalesced_global_access`, NVIDIA's shipped rules emit
`UncoalescedGlobalAccess` under `MemoryWorkloadAnalysis`, and the nsys side said
`memcpy_bound`. Cross-checking silently matched nothing until all three mapped
onto one vocabulary.

---

## 5c. Cross-checking against NVIDIA's own rules

An `.ncu-rep` carries NVIDIA's own findings. Every action exposes
`rule_results_as_dicts()`: message, section, focus metrics, often a speedup
estimate. We were extracting them and throwing them away while running our own
re-implementation of the same rules.

```python
from my_utils.profiling.ncu.ncu_diagnostics import diagnose_kernel
from my_utils.profiling.ncu.ncu_report_tools import load_ncu_report_rule_rows

shipped = load_ncu_report_rule_rows("profile.ncu-rep")
result = diagnose_kernel(metrics, kernel_name=name, shipped_rules=shipped)

result["corroboration"]["corroborated"]    # axes where we and NVIDIA agree
result["corroboration"]["conflicts"]       # where we disagree -- the valuable part
result["corroboration"]["ncu_only"]        # rules NVIDIA fired that we lack
```

Four outcomes, each treated differently:

- **Agreement** promotes the finding to high confidence and names the
  corroborating rule in its evidence.
- **Disagreement on the bottleneck verdict** becomes its own high-severity
  finding. If we say compute-bound and `SOLBottleneck` says memory-bound, one of
  us is reading the wrong number - most often a Speed-of-Light comparison mixing
  `_active` and `_elapsed` denominators.
- **NVIDIA-only rules** are added rather than dropped, so nothing shipped in the
  report is lost.
- **Absent shipped rules weaken nothing.** Missing corroboration is not evidence
  against a finding, and the result says so explicitly rather than silently
  leaving findings unmarked.

### The LOCAL/GLOBAL speedup trap

Nsight Compute reports estimated speedups as either `GLOBAL` (relative to the
whole kernel) or `LOCAL` (relative to the section it came from). A 40% LOCAL
estimate on a section that is 5% of runtime is worth about 2%. The rule result
does not carry that section's share of runtime, so a LOCAL estimate is **never**
promoted to a kernel-level ceiling here - `speedup_ceiling` stays `None` and the
finding says why.

---

## 6. The metric catalog

`my_utils/profiling/ncu/metric_catalog.py` — 179 metrics keyed by stable short
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

## 6b. Every metric in the report, not just the catalogued ones

The curated catalog interprets 179 metrics. A `--set full` collection carries
thousands. Everything outside the catalog used to be loaded and then touched by
nothing - a report could hold the exact counter that explained a kernel and the
analysis would never mention it existed.

Metric names are not opaque. They follow the PerfWorks grammar:

```
<unit>__<counter>[.<rollup>][.<submetric>]

sm__throughput.avg.pct_of_peak_sustained_elapsed
|        |      |            |
unit   counter rollup     submetric
```

```python
from my_utils.profiling.ncu.section_index import (
    decode_metric_name, axis_for_metric_name, denominator_of, group_report_metrics,
)

decode_metric_name("pmsampling:smsp__warps_issue_stalled_barrier.avg")
# {'unit': 'smsp', 'quantity': 'warps_issue_stalled_barrier',
#  'rollup': 'avg', 'submetric': '', 'prefix': 'pmsampling'}

axis_for_metric_name("lts__t_sectors.sum")     # 'memory_bandwidth'
denominator_of("sm__throughput.avg.pct_of_peak_sustained_active")   # 'active'

result["metric_inventory"]["summary"]
# 3421 metrics present. 179 are interpreted by a rule; 3244 are decoded and
# placed on an axis but carry no threshold, so nothing judged them.
```

The unit prefix alone places a metric on an axis, so an uncatalogued counter is
still named, decoded and grouped rather than dropped. An unrecognised unit is
reported as unrecognised - a metric filed under the wrong axis is worse than one
filed under none.

### Checking the catalog against a real install

If Nsight Compute is installed locally, the shipped `.section` files are ground
truth for which metrics exist and which sections request them:

```python
from my_utils.profiling.ncu.section_index import audit_catalog_against_sections
from my_utils.profiling.ncu.metric_catalog import METRIC_CATALOG

audit_catalog_against_sections(METRIC_CATALOG)["summary"]
# 147 catalog spellings are requested by a shipped section; 37 exist under a
# different rollup and need an explicit --metrics request; 32 were not found in
# the shipped sections at all (candidates only ...)
```

That third bucket is **candidates, not errors**. Sections request only a subset
of what a device exposes, so `--query-metrics` on the target GPU is the only
authority. Several device attributes and `dram__bytes_read.sum` land there and
are perfectly real.

The distinction matters when building a collection command: `explicit_only` and
`unknown` metrics will not arrive from `--section` alone.

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

**Provenance of these four numbers.** They are read verbatim from the
`SpeedOfLight` rule shipped with Nsight Compute 2026.1.1
(`sections/SpeedOfLight.py`), where they appear as `balanced_threshold=10`,
`latency_bound_threshold=60`, `no_bound_threshold=80`, `waves_threshold=1`. Its
actual decision tree:

```
if sm < 80 and mem < 80:
    if sm < 60 and mem < 60:   -> small grid (waves < 1) else latency issue
    elif |sm - mem| >= 10:     -> whichever is higher dominates
    else:                      -> balanced
else:                          -> saturated; shift work between units
```

Three classification schemes circulate in blog posts and vendor material: a
single `max(sm,mem) < 60` latency gate, a banded >80 / 60-80 / <60 table, and a
60/40 two-axis table. **The first two are not rival schemes** - they are partial
descriptions of the one rule above. The third appears in no shipped rule; it
was present in `SOL_THRESHOLDS` as `compute_bound_compute`/`compute_bound_memory`,
was read by nothing, and has been removed. An unverified threshold sitting
beside verified ones invites equal trust.

One deliberate divergence: at >= 80 NVIDIA's rule says "shift work from the most
utilised unit" without naming a bound. We label it `compute_bound` /
`memory_bound`, which is more directly actionable and routes to the same section
NVIDIA's rule points at.

### 7.1b Hierarchical roofline: why, not just whether

The stock roofline plots one point per kernel - FLOPs over DRAM bytes - which
says whether a kernel is memory bound and nothing about why. Computing the same
FLOP count against traffic at each cache level gives three points sharing a
numerator, and the horizontal spread reads directly as locality.

```python
from my_utils.profiling.ncu.ncu_diagnostics import hierarchical_roofline

h = hierarchical_roofline(view, gpu_spec)
h["summary"]     # L1 AI=0.03 FLOP/byte | L2 AI=0.04 | DRAM AI=1.00
h["locality_verdict"]   # 'healthy' | 'leaking'
```

| Observation | Meaning | Lever |
|---|---|---|
| L1 and L2 intensities close (ratio < 1.5) | L1 passes traffic through rather than serving it | block for L1 |
| L2 and DRAM close (ratio < 1.5) | working set does not fit L2 at this tile size | shrink tile, reorder traversal |
| DRAM/L1 spread >= 4x | caches already absorbing reuse | **do not** spend effort on tiling |

That last row is the one most tools cannot produce: a finding that tells you
what *not* to do.

Two things it refuses to smooth over:

- **`l1tex__t_bytes` excludes shared-memory traffic.** So the L1 intensity is an
  *overestimate* precisely for tiled GEMM and attention - the kernels whose whole
  design is staging through shared memory, and the ones most likely to be
  profiled. Reported as a caveat whenever the shared pipe shows activity.
- **SASS FLOP counters miss tensor-core math**, making all three intensities
  floors rather than measurements.

The direct byte counters are not requested by any shipped section, so the sector
counters are the fallback at 32 bytes/sector, and `byte_source` records which
was used rather than presenting derived and measured bytes identically.

**Nsight Compute ships this too**, as four chart sections — one per precision:

```
--section SpeedOfLight_HierarchicalSingleRooflineChart   # fp32
--section SpeedOfLight_HierarchicalHalfRooflineChart     # fp16
--section SpeedOfLight_HierarchicalDoubleRooflineChart   # fp64
--section SpeedOfLight_HierarchicalTensorRooflineChart   # tensor
```

They render in the ncu UI; `hierarchical_roofline` gives you the same three
points programmatically, plus the level-to-level ratios and the shared-memory
caveat. Collect the matching chart section when you want to see it plotted.

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

## 9b. Never conclude from a kernel name alone

`my_utils/profiling/analyzers/evidence.py`.

A kernel symbol is the weakest evidence a profiler has, and in named cases it is
not merely vague but **wrong**:

* **NCCL names lie about the algorithm.** `generate.py::best_kernel()` collapses
  ~670 device functions onto ~40 launch stubs with the algo and protocol
  hard-coded to `RING`/`TREE` + `LL`; the real algorithm is dispatched at runtime
  through `ncclDevFuncTable`. A ReduceScatter genuinely running **NVLS + Simple**
  appears as `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`.
* **CuTe DSL names carry nothing.** The default is `kernel_kernel_<args>_0`, and
  `mangle_name` skips every IR value, so no shape or dtype survives.
  FlashAttention-4 ships exactly this way.
* **A `cutlass3x_*` symbol does not mean the user wrote CUTLASS.** cuBLASLt
  embeds CUTLASS-generated kernels.

So conclusions are fused from six sources, ranked by how much each can be
trusted:

| Provenance | Rank | Why |
|---|---|---|
| `HW_COUNTER` | 100 | Records what the silicon actually did |
| `NVTX` | 80 | Authored by the framework that issued the work |
| `SOURCE` | 75 | Static analysis of the code that ran |
| `CUDA_API` | 60 | `cuLaunchKernel` vs `cudaLaunchKernel` separates JIT from static |
| `LAUNCH_CONFIG` | 50 | Structural, but says nothing about the kernel body |
| `KERNEL_NAME` | 20 | See above |

```python
from my_utils.profiling.analyzers import attribute_kernel

fused, warnings = attribute_kernel(
    "ampere_sgemm_128x64_nn",
    metrics={"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 0.0},
)
fused["uses_tensor_cores"].value        # False
fused["uses_tensor_cores"].provenance   # 'hw_counter'
warnings
# ['Kernel name says matmul but the tensor pipe never activated - either it is an
#   FP32/SIMT fallback, the dtype is unsupported on this arch, or the shapes are
#   too small to tile. This is a finding, not a naming quirk.']
```

**The contradictions are the point.** A name that disagrees with a counter is
usually the most useful line in the report. And a name that says *nothing*
produces its own finding rather than silence - otherwise an unlabelled CuTe DSL
kernel falls through to "no expectations matched", which reads as "nothing
wrong".

### CUTLASS symbols are the exception

`parse_cutlass_symbol()` is the one place where a name is strong evidence,
because the template arguments **are** the compiled configuration:

```python
from my_utils.profiling.sources.kernel_taxonomy import parse_cutlass_symbol

cfg = parse_cutlass_symbol(demangled_symbol)
cfg.stages       # 9        - mainloop pipeline depth
cfg.cluster      # (1,1,1)  - thread block cluster
cfg.tile         # (128,64,64)
cfg.mma_shape    # (64,64,16)
cfg.schedule     # 'pingpong'
cfg.copy_atom    # 'SM90_TMA_LOAD'
cfg.swizzle      # (3,4,3)
cfg.observations()
# ['Cluster shape is 1x1x1, so no multicast across the cluster. ...']
```

`cfg.truncated` matters: nsys clips long symbols, and an absent field then means
"not visible", not "not used".

---

## 9c. Trace validity: refuse before you analyse

`my_utils/profiling/analyzers/trace_quality.py`. Every check returns issues with
a `blocks` flag - `True` means refuse the affected conclusion rather than caveat
it.

| Check | Refuses when |
|---|---|
| `check_warmup` | Too few iterations for a steady-state claim |
| `check_autotuning` | One name covers many launch configs - an autotune sweep |
| `check_kernel_name_uniqueness` | One symbol covers different shapes, so per-name stats blend them |
| `check_cuda_graphs` | Per-kernel host attribution does not exist under graph replay |
| `check_rank_completeness` | Rank files are missing - a **collection failure**, never "those ranks were idle" |
| `check_gpu_metric_gaps` | "Missing Data" is sampler exhaustion, not GPU idle |
| `check_profiler_overhead` | The gap you are measuring is the profiler's own flush |
| `check_clock_alignment` | Cross-rank claim below the ~10 ms floor that NTP alignment supports |
| `check_nvlink_utilization_validity` | NVLink reads ~100% but may simply be **inactive** |
| `check_multi_tenancy` | MPS (no per-client attribution), MIG, vGPU, NVLink-centric scheduling |
| `check_diagnostic_events` | The report's own diagnostics say data was dropped |
| `check_dataloader_attribution` | No worker threads in the trace, so the dataloader verdict is a guess |
| `check_derived_metric_invariants` | MFU > 100%, HFU < MFU, sparse peak, unknown dtype |

Two worth spelling out:

**Cross-node timestamps are not synchronised.** nsys aligns reports from
different hosts by UTC captured at collection start, with an error NVIDIA
documents as one to tens of milliseconds - while a typical collective runs
0.1-5 ms. So "rank 3 arrived 4 ms late" is indistinguishable from clock skew.
Same-host reports use TSC and are precise to nanoseconds; check the
`Report alignment source` field on the Analysis Summary tab.

**Naming a straggler needs entry times, not durations.** In a synchronous
collective the slow rank shows a long compute phase and a *short* wait, while
every fast rank waits inside NCCL - so ranking by time-in-NCCL names the victims:

```python
from my_utils.profiling.analyzers import detect_straggler_from_traces

# PyTorch Flight Recorder: TORCH_NCCL_TRACE_BUFFER_SIZE=2000
result = detect_straggler_from_traces(fr_entries, collective_seq_id=7,
                                      clock_alignment="UTC")
result["worst_rank"]           # 5
result["worst_vs_median_ms"]   # 40.0  - vs median, since the straggler
                               #         poisons the mean it is measured against
```

---

## 9d. Clock throttling

`my_utils/profiling/hardware/throttling.py`. A throttled run makes every derived
number wrong in the same direction, while looking exactly like a code regression.

```python
from my_utils.profiling.hardware import analyze_throttling

r = analyze_throttling(clock_event_mask=0x4, sm_clock_mhz=1395,
                       boost_clock_mhz=1755, thermal_violation_ns=3e8,
                       window_ns=1e10)
r["throttling"]     # True
r["invalidates"]    # ['mfu', 'achieved_tflops', 'achieved_bandwidth',
                    #  'pct_of_peak', 'regression_comparison']
```

**The trap this encodes:** `GpuIdle` (0x1) and `ApplicationsClocksSetting` (0x2)
share the NVML clock-event field with the real throttle reasons but are **not
limits being hit**. A bare `mask != 0` test therefore reports every idle GPU in
the fleet as throttled - and every deliberately clock-pinned benchmark, which is
the one case where the pinning is exactly what you wanted. Test against
`0x4|0x8|0x10|0x20|0x40|0x80|0x100`.

Prefer DCGM's accumulating violation counters (`POWER_VIOLATION` 240,
`THERMAL_VIOLATION` 241) over the instantaneous mask: throttling is bursty and a
1 Hz sample steps over most of it. Match DCGM fields on **numeric ID**, not name
- `CLOCK_THROTTLE_REASONS` was renamed to `CLOCKS_EVENT_REASONS` keeping id 112.

Note also that ncu **cannot set clocks on a MIG Compute Instance** at all: pass
`--clock-control none` and lock externally with
`nvidia-smi --lock-gpu-clocks=tdp,tdp`.

---

## 9e. Where, not what: source and SASS attribution

Every other analysis here says *what* is wrong. For a fused kernel that is often
not enough: "this kernel stalls on long-scoreboard" does not say which of six
fused stages stalls, and no whole-kernel counter can - the question is
structurally unanswerable from aggregates.

```python
from my_utils.profiling.ncu.source_correlation import (
    attribute_stalls_to_source, correlate_metric_to_source,
    pc_sampling_timeline, source_availability,
)

out = attribute_stalls_to_source(action, top_k=15)
for line in out["source_lines"]:
    print(f"{line['file_name']}:{line['line']}  {line['samples']:5d}  "
          f"{line['dominant_stall_reason']:18s} {line['source_text']}")
# attn.cu:2    400  MIO_THROTTLE       softmax
# attn.cu:1    300  LONG_SCOREBOARD    load q
```

That is the whole point of the module: two lines of a fused kernel, two
different bottlenecks, two different fixes.

`ncu-diagnose` includes this per kernel; the skill below is for when you want
it on its own or with a specific metric attributed:

```bash
python -m my_utils.profiling.cli ncu-diagnose --report profile.ncu-rep
# ...
# ### Where it stalls
# | source        | samples | dominant stall  | line       |
# | `attn.cu:2`   | 600     | MIO_THROTTLE    | `softmax`  |
```

Pass `--no-source` to skip it; it re-reads the report for PC samples.

### The pipeline `ncu-diagnose` runs

One command, one pass over the report, three stages:

`walk_report_once` opens the report once and visits each kernel launch a single
time, gathering its metrics, NVIDIA's shipped rule results, the action object
and its source attribution together. (It was four separate loaders, each doing
its own full traversal.) The three stages then run per launch:

1. **Collect** — every metric in the report, not the catalogued subset.
   `metric_inventory` accounts for all of them.
2. **Reason** — the 15 curated analyses, plus `scan_all_signals` over
   everything else. The scan reads metrics by name grammar, so it can say "the
   constant cache is at 91% of peak" for a unit nobody wrote a rule for. Its
   thresholds are ours and conservative; it points at a section to read, while
   the curated rules are what know the fix.
3. **Localise** — `link_findings_to_source` joins each finding to the source
   lines whose sampled stalls explain it.

```
### Signal to source

**Global loads touch 28.0 sectors per request**
- correlated via `LONG_SCOREBOARD, LG_THROTTLE` (concentrated, 100% of those samples)
  - `attn.cu:1` 700 samples (100%) `q = load_qkv(...)`
```

**The join is by stall reason and is a correlation, not a proof.** A line can
stall for several reasons at once, and a finding can have causes the sampler
cannot observe. Each link reports how concentrated the evidence is, so a signal
spread thinly across many lines looks different from one sitting on a single
line.

**Some findings are deliberately never linked.** Grid shape, occupancy limits,
tile quantisation and measurement faults are properties of the launch or of the
data, not of any line. So are the generic scan findings: a saturated constant
cache has no known relationship to any stall reason, and attaching one would
put a real source line next to an invented mechanism. That reads as evidence
and is not.

### The API this rests on

Verified by reading the `ncu_report` module shipped with Nsight Compute 2026.1.1,
not from documentation:

| Call | Returns |
|---|---|
| `IMetric.has_correlation_ids()` | whether a metric carries per-instruction values |
| `IMetric.correlation_ids()` | parallel metric whose instance values are addresses |
| `IAction.source_info(addr)` | `ISourceInfo` with `file_name()` / `line()` |
| `IAction.sass_by_pc(addr)` | SASS text, `""` when unavailable |
| `IAction.ptx_by_pc(addr)` | PTX text |
| `IAction.source_files()` | `{filename: content}`, empty content when not imported |
| `IAction.timed_warp_samples()` | raw PC-sample records -- often empty; see below |

**Two attribution paths, and the fallback is the one to expect.** A standard
`--set full` collection does *not* retain raw sample records: `timed_warp_samples()`
comes back empty. What it does retain is one
`smsp__pcsamp_warps_issue_stalled_<reason>` metric per stall reason, each with a
value per instruction and correlation IDs mapping those values to addresses.
`attribute_stalls_to_source` tries that path first and falls back to raw samples.

The `_not_issued` variants of those metrics count the same stalls on cycles
where no warp issued at all; folding both in double-counts every instruction.

### Three reasons it returns nothing, with three different fixes

```python
source_availability(action)["reasons_unavailable"]
```

1. **A bare `ncu` run collects no source metrics at all.** `SourceCounters`
   ships only in the `detailed` and `full` sets; the default is `basic`. This is
   the most common cause and has nothing to do with how the code was built.
2. **No `-lineinfo`.** Addresses exist but map to no source line. SASS-level
   attribution still works; source-level does not.
3. **File property mismatch.** Nsight Compute checks the source file's
   modification time and size against what the compiler recorded. Re-saving a
   `.cu` after compiling is enough to break source display. `--import-source
   yes` embeds the source and sidesteps this entirely.

### The timeline is not a summary

`pc_sampling_timeline` buckets samples by timestamp rather than aggregating
them. Sorting a time series by magnitude - which is what a naive summary does -
destroys the one dimension sampling adds over `WarpStateStats`:

```python
tl = pc_sampling_timeline(action, bucket_ns=100_000)
tl["phase_sequence"]      # ['LONG_SCOREBOARD', 'MATH_PIPE_THROTTLE']
tl["note"]
# The dominant stall reason changes 1 time(s) across the kernel
# (LONG_SCOREBOARD -> MATH_PIPE_THROTTLE). A single averaged stall breakdown
# would report a blend of these and suggest the wrong fix for each phase.
```

A kernel that is memory bound then compute bound averages to uniformly
mediocre, and the averaged answer is wrong for both halves.

---

## 9f. Sampling validity: PC and PM sampling

Sampled data looks identical whether it is sound or badly undersampled: a list
of samples with a distribution. Eleven samples and eleven thousand render the
same way and mean entirely different things.

```python
from my_utils.profiling.ncu.sampling_validity import (
    check_pc_sampling_validity, check_pm_sampling_validity,
)

v = check_pc_sampling_validity(
    sample_count=metrics["smsp__pcsamp_sample_count"],
    interval_cycles=metrics["smsp__pcsamp_interval_cycles"],
    dropped_bytes=metrics["smsp__pcsamp_dropped_bytes"],
    buffer_overflow=metrics["smsp__pcsamp_buffer_overflow"],
)
v["usable"], v["blocked_conclusions"]
```

Checks and thresholds mirror NVIDIA's shipped `PCSamplingData` and
`PMSamplingData` rules, read from the rule sources rather than invented.

| Signal | Meaning | Fix |
|---|---|---|
| `pcsamp_dropped_bytes > 0` | samples dropped under backpressure | raise `--warp-sampling-interval` |
| `pcsamp_buffer_overflow` | buffer filled; later samples missing | raise `--warp-sampling-buffer-size` |
| `pcsamp_sample_count == 0` | nothing collected (often kernel < one interval) | shorter interval, or longer kernel |
| PM `interval/duration >= 1` | at most one sample | reduce `--pm-sampling-interval` |
| PM `interval/duration > 0.1` | very few samples; short phases invisible | reduce interval |
| PM on CC < 7.5 | unsupported entirely | no configuration helps |

**Drops and overflows are the dangerous case.** They do not add noise, they
**bias**: drops correlate with the busiest periods, so the hottest code is
exactly what goes missing, and an overflow truncates everything after a point.
Both block attribution outright.

Sample counts block *per conclusion* rather than wholesale. Below ~200 samples,
ranking one source line above another is not supported, while the overall stall
distribution still is - different claims need different support. That 200 floor
is ours, not NVIDIA's, and is marked as such in the code.

### These five counters need an explicit `--metrics`

`smsp__pcsamp_sample_count`, `_interval_cycles`, `_buffer_overflow`,
`_buffer_size_bytes`, `_dropped_bytes` are requested by NVIDIA's *rule*, not by
any section's `Metrics` block. Collecting `SourceCounters` does **not** collect
them. `ncu_full_collection.yaml` requests them explicitly. Without them, a
biased sample set is indistinguishable from a sound one.

---

## 9g. How a measurement was taken

Two correct measurements of the same kernel routinely disagree by 2x, and the
reason is almost never the kernel.

**Nsight Compute flushes every GPU cache before each replay pass by default**
(`--cache-control all`). An ncu duration is a *cold-cache* number. A wall-clock
timing of the same kernel in a pipeline, where the previous kernel left its
output in L2, is a *warm* number. Neither is wrong. Comparing them is.

```python
from my_utils.profiling.analyzers.measurement_context import (
    describe_collection_mode, compare_measurements,
)

ncu_run   = describe_collection_mode(source="ncu")            # cold, serialised
wallclock = describe_collection_mode(source="wallclock")      # warm

cmp = compare_measurements(ncu_run, wallclock,
                           baseline_value=2.0, candidate_value=1.0)
cmp["comparable"]   # False
cmp["ratio"]        # None  <- deliberately not 0.5
cmp["verdict"]
# Not comparable. Cache state differs (cold vs warm). A cold-cache measurement
# and a warm one are different quantities; the difference between them is not a
# change in the code. Re-measure both sides the same way ...
```

`ratio` stays `None` when the modes differ. The raw number is available as
`uncomparable_raw_ratio` - a name that cannot be mistaken for a result.
Reporting a 2x "regression" that is entirely cache state sends someone to
optimise a kernel that did not change.

### What each mode structurally cannot answer

These are not caveats to weigh. They are questions where the measurement
contains no information:

| Mode | Cannot answer |
|---|---|
| ncu (default) | how fast the kernel runs in a pipeline - caches were flushed |
| ncu (any) | whether work overlaps - execution is serialised to profile it |
| unknown cache state | comparison with anything, since cache state moves durations more than most optimisations |
| clocks unlocked | run-to-run comparison at face value |
| synthetic inputs | performance on real data |

Use `--cache-control none` for pipeline-realistic ncu numbers. Stream
concurrency has to come from Nsight Systems; it is invisible to ncu by
construction.

**Thermal drift from long profiling loops** is real and documented: a published
H100 result was revised from 74% to 84% of peak after the iteration count was
found to be heating the part. `describe_collection_mode(iterations=..., 
clocks_locked=...)` warns when a long loop runs without pinned clocks.

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
├── ncu/section_index.py             decode_metric_name, axis_for_metric_name,
│                                    denominator_of, group_report_metrics,
│                                    audit_catalog_against_sections
├── ncu/ncu_diagnostics.py           diagnose_kernel, classify_bottleneck,
│                                    analyze_stalls, compute_roofline, MetricView,
│                                    analyze_memory_hierarchy, analyze_issue_efficiency,
│                                    analyze_instruction_mix, hierarchical_roofline,
│                                    analysis_coverage, SOL_THRESHOLDS, Finding
├── ncu/shipped_rules.py             normalize_shipped_rules, ShippedRule,
│                                    reconcile_with_shipped_rules
├── ncu/source_correlation.py        attribute_stalls_to_source, source_availability,
│                                    correlate_metric_to_source, pc_sampling_timeline,
│                                    summarize_warp_samples
├── ncu/sampling_validity.py         check_pc_sampling_validity,
│                                    check_pm_sampling_validity
├── ncu/ncu_report_tools.py          diagnose_ncu_report, analyze_ncu_report,
│                                    load_ncu_report_rule_rows, NcuReportSkillEngine
├── analyzers/axes.py                AXES, axis_coverage, axis_for_category,
│                                    axis_for_shipped_rule
├── analyzers/measurement_context.py describe_collection_mode, compare_measurements,
│                                    MeasurementContext, CacheState
├── analyzers/trace_quality.py       assess_trace_quality, check_clock_alignment,
│                                    check_derived_metric_invariants, ...
├── analyzers/nccl_bandwidth.py      analyze_collective, detect_straggler_from_traces
├── hardware/throttling.py           analyze_throttling, decode_clock_event_mask
├── analyzers/triage.py              triage_step, TriageThresholds, TriageVerdict,
│                                    merge_intervals, interval_overlap_ns
├── sources/nsys_analyze.py          analyze_nsys_sqlite
├── sources/nsys_auto_analysis.py    build_comprehensive_analysis
└── sources/nsys_sql_skills.py       NsysSqlSkillEngine (29 SQL skills)
```

These modules are **pure analysis** — no torch, no CUDA. They import and run on a
laptop with nothing installed, which is what makes them testable.

`tests/profiling/test_analysis_engine.py` (297 tests) is the executable spec.
Several tests exist specifically to stop this handbook drifting from the code:
`TestHandbookExamplesAreReal` checks every documented import resolves,
`TestCoverageKeysAreReal` pins coverage tables to the metric catalog, and
`TestSolThresholdsMatchShippedRule` pins the SOL constants to NVIDIA's rule.

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
