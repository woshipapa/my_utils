# References

Sources this toolkit's analysis methodology draws on, with what each one
contributed. Kept honest: entries state what we actually borrowed, not
just what we read.

## Methodology adopted into the engine

- **GCStack + GCScaler (ISCA 2025)** — closed stall accounting. Their full
  mechanism needs a cycle-level simulator; what transfers to ncu data is the
  *closure discipline*: a stall stack that does not sum to total cycles cannot
  say how much of the runtime it explained. Our stall accounting checks
  closure in both directions (<90% and >102% both raise findings), and uses
  ncu's `_per_issue_active` metrics, which already fractionally split
  concurrent stalls.
- **Modal GPU Glossary** (https://modal.com/gpu-glossary/perf) — the
  three-way bottleneck taxonomy (compute- / memory- / **overhead-bound**),
  banded SOL thresholds (>80 / 60–80 / <60) instead of a single cutoff, and
  the "single-digit occupancy is normal for Hopper GEMMs" calibration that
  gates our warp-specialization-aware rules.
- **Horace He, "Making Deep Learning Go Brrrr"**
  (https://horace.io/brrr_intro.html) — overhead-bound as a first-class
  bottleneck class, peer to compute- and memory-bound.
- **NVIDIA Nsight Compute shipped rules** (BSD-3-Clause, redistributed —
  see NOTICE) — the `sections/*.py` advisory rules ship inside Nsight
  Compute; our engine reconciles its own findings against them and passes
  their advice through, attributed.

## Official NVIDIA documentation

- Nsight Compute Kernel Profiling Guide —
  https://docs.nvidia.com/nsight-compute/ProfilingGuide — metric semantics,
  replay model, PC/PM sampling behaviour.
- Nsight Compute CLI reference —
  https://docs.nvidia.com/nsight-compute/NsightComputeCli
- Nsight Compute release notes —
  https://docs.nvidia.com/nsight-compute/ReleaseNotes — per-version metric
  and section changes.
- Nsight Systems User Guide —
  https://docs.nvidia.com/nsight-systems/UserGuide — sqlite export schema,
  recipes, multi-report analysis.
- CUTLASS documentation — https://docs.nvidia.com/cutlass/latest — warp
  specialization, ping-pong scheduling, TMA/WGMMA pipeline structure.
- NCCL documentation — https://docs.nvidia.com/deeplearning/nccl — collective
  algorithm and bandwidth model background for the NCCL analyzers.

## Practitioner writeups

- Colfax Research CUTLASS tutorials
  (https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel,
  and the FP8 FlashAttention / matrix-transpose installments) — Hopper GEMM
  design vocabulary used by our warp-specialization case analysis.
- PyTorch blog, "CUTLASS Ping-Pong GEMM Kernel"
  (https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) — the pingpong
  scheduling model our producer-warp vs CTA fusion case study builds on.
- PyTorch blog, "Trace Analysis for the Masses" (Meta HTA,
  https://pytorch.org/blog/trace-analysis-for-masses/ and
  https://github.com/facebookresearch/HolisticTraceAnalysis) — trace-level
  analysis dimensions (temporal breakdown, idle-time classification) that
  informed the nsys-side analyzers.
- Aleksa Gordić, "Inside NVIDIA GPUs / matmul"
  (https://www.aleksagordic.com/blog/matmul) — worked Hopper matmul
  optimization narrative with the metrics an expert actually watches.
- spatters.ca, "MMA matmul" (https://www.spatters.ca/mma-matmul) — staged
  Hopper matmul optimization walkthrough.
- Lei Mao's blog (https://leimao.github.io) — reproducible ncu-in-Docker
  workflows.
- Stas Bekman, ML Engineering (https://github.com/stas00/ml-engineering) —
  training-side performance accounting conventions (MFU reporting).

## Related tools studied

- NVIDIA nsight-training (https://github.com/NVIDIA/nsight-training) —
  official analysis exercises; sanity reference for report interpretation.
- CUTLASS profiler (https://github.com/NVIDIA/cutlass) — kernel taxonomy and
  performance-sweep conventions.
- Holistic Trace Analysis (Meta) — see above.
- DCGM / NVML (https://github.com/NVIDIA/DCGM,
  https://docs.nvidia.com/deploy/nvml-api) — system-level GPU telemetry
  complementing per-kernel data.

## Forum threads that settled specific metric semantics

NVIDIA developer-forum threads we relied on for behaviour that is not in the
manuals (all under https://forums.developer.nvidia.com/t/):

- `what-is-the-different-between-sm-pipe-tc-cycles-active-and-sm-pipe-tensor-cycles-active-in-nsight-compute` — tensor-pipe metric disambiguation.
- `why-low-tensor-pipe-utilization` — interpreting low tensor-pipe SOL on
  warp-specialized kernels.
- `question-about-warp-stalls-observed-in-gemm-profiling-on-h100` — Long
  Scoreboard absorbing warpgroup synchronization on Hopper WGMMA kernels.
- `sm90-setmaxnreg-will-change-occupancy-dynamically`, `setmaxnreg-purpose`,
  `about-setmaxnreg`, `hopper-launch-bounds-and-setmaxnreg-conflicts` —
  asymmetric per-warpgroup register budgets (`reg_alloc`/`reg_dealloc`), and
  why the kernel-level register figure cannot name the pressured warpgroup.
- `nsight-compute-clock-speed-during-profiling`,
  `sm-frequency-reported-in-nsight-compute`,
  `nsight-systems-gpc-frequency-profiling` — clock behaviour under replay;
  basis for our clock-confound guards.
- `the-mechanism-behind-mbarrier-try-wait` — mbarrier wait semantics behind
  barrier-stall interpretation.
- `how-many-flops-does-one-tensor-op-hmma-instruction-do`,
  `tensor-core-flops`,
  `how-to-measure-flops-of-a-cuda-kernel-function-by-using-nsight-compute-on-a100-gpu`,
  `discrepancy-in-tensor-core-fp16-performance-ceiling-on-h100-sxm-observed-in-nsight-compute` —
  FLOP counting and roofline ceiling calibration.
- `kernel-performance-discrepancy-in-nsight-compute-and-systems` — why ncu
  and nsys durations differ for the same kernel.
- `question-about-nsys-sqlite3-schema-on-analysis-details-table`,
  `getting-full-kernel-name-from-nsys` — nsys sqlite schema details.
- `error-some-events-were-lost-how-do-i-fix-this`,
  `nsys-profile-can-hang-for-a-long-time-when-profiling-pytorch-distributed-training-runs`,
  `multi-node-profiling-with-nsight-systems`,
  `nsight-compute-failed-to-profile-with-nvtx-ranges-in-pytorch`,
  `nsight-compute-profiling-challenges-with-flashattention-kernels-in-vllm`,
  `nsight-compute-hangs-on-instructionstats-warpstatestats-when-profiling-tma-mbarrier-kernels-on-blackwell`,
  `nsight-compute-profile-run-with-nan-value-in-multi-process-service-mps` —
  capture-side failure modes reflected in our trace-quality gating and docs.
- `switch-from-sm90-xmma-gemm-cublas-...-to-nvjet-tst-kernels-with-cuda-12-8`,
  `what-is-this-kernel-nvjet-tst-112x64-64x9-1x2-h-bz-bias-tnn-which-cuda-api-do-i-need` —
  cuBLAS kernel-family naming used by the kernel taxonomy.
- `maximum-tensor-core-utilization`,
  `questions-about-sm-pipe-cycles-active-metrics`,
  `ncu-performance-counters-for-profiling-tensor-ops-for-trt-int8-loop-rnn` —
  pipe-utilization metric semantics.

## Papers surveyed during development

Titles intentionally omitted where we did not re-verify them; links only:

- https://dl.acm.org/doi/10.1145/2503210.2503299
- https://dl.acm.org/doi/10.1145/3168831
- https://dl.acm.org/doi/10.1145/3392717.3392752
- https://dl.acm.org/doi/10.1145/3503222.3507708
- https://dl.acm.org/doi/10.1145/3582016.3582044
- https://dl.acm.org/doi/10.1145/3624062.3624208
- https://dl.acm.org/doi/10.1145/3767295.3803568
- https://dl.acm.org/doi/abs/10.1109/CGO51591.2021.9370339

## Ecosystem survey (2026-07)

Four-track survey (NCU tools, NSYS tools, methodology writeups, academic/HPC
toolchains) run against this toolkit's feature set. Entries note what each
tool offers that we lacked at survey time; items marked *(adopted)* have
since been integrated. **What exactly was borrowed from each source, and how
it was adapted to this engine's honesty rules, is recorded per capability in
`my_utils/profiling/docs/CAPABILITY_EVOLUTION.md`, Era 5.**

### Nsight Compute analysis tools

- NVIDIA nsight-python — https://github.com/NVIDIA/nsight-python — official
  Python capture + pandas extraction; no interpretation layer.
- NVIDIA nsight-training, Python Report Interface notebooks —
  https://github.com/NVIDIA/nsight-training — NVTX-range filtering,
  opcode-instanced metrics, Report2Json.
- NVIDIA TensorRT-LLM `perf-nsight-compute-analysis` skill —
  https://github.com/NVIDIA/TensorRT-LLM — staged section escalation and a
  profile→fix→re-profile loop.
- GPUscout (TUM, SC'23 workshop) — https://github.com/caps-tum/GPUscout —
  static SASS dataflow analysis fused with stall sampling and metrics; the
  one surveyed tool with analysis machinery deeper than ours (static
  channel).
- DrGPU (ICPE'23) — https://github.com/FindHao/drgpu — top-down stall
  decomposition tree rendering.
- KuangjuX/ncu-cli — https://github.com/KuangjuX/ncu-cli — threshold-rule
  diagnostics from ncu CSV; first-class A/B `diff` command *(adopted:
  `ncu-diff`)*.
- mit-han-lab/ncu-report-skill — https://github.com/mit-han-lab/ncu-report-skill —
  harness generation + agent playbook; cruder PM-sampling handling than ours.
- NERSC roofline-on-nvidia-gpus — https://gitlab.com/NERSC/roofline-on-nvidia-gpus —
  hierarchical-roofline collection section files.
- Giotyp/GPU-Roofline-Python — https://github.com/Giotyp/GPU-Roofline-Python —
  Ding–Williams instruction roofline and FP8/FP4 precision ceilings *(adopted:
  instruction roofline + precision-aware ceilings)*.
- LLNL Thicket NCU reader — https://thicket.readthedocs.io/en/latest/nsight_compute.html —
  calltree contextualization + multi-run ensemble statistics.
- meta-pytorch/tritonbench ncu_analyzer —
  https://github.com/meta-pytorch/tritonbench — whole-workload rollups
  (duration-weighted TFLOPS) for CI tracking.
- uliegecsm/reprospect — https://github.com/uliegecsm/reprospect — profiling
  metrics as pytest-style CI assertions.
- PerfDigest-MCP — https://github.com/onlyxItachi/PerfDigest-MCP —
  token-efficient agent digests, cross-vendor contract, before/after compare.
- Accel-Sim correlator — https://accel-sim.github.io/ — hardware-vs-model
  per-metric correlation harness.

### Nsight Systems analysis tools

- nsys bundled recipes + expert systems —
  https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html — 50+
  multi-report recipes: pacing (`*_pace`), time-binned per-rank utilization
  heatmaps, `gpu_gaps`, network/file-access summaries; rule-based
  anti-pattern advice (`nsys analyze`).
- Holistic Trace Analysis (Meta) —
  https://github.com/facebookresearch/HolisticTraceAnalysis — idle-time
  cause attribution (host wait vs dependency), CUPTI counter→op roofline.
- nsys-jax / nsys-jax-combine —
  https://github.com/NVIDIA/JAX-Toolbox/blob/main/docs/nsys-jax.md — XLA
  metadata-to-source provenance; multi-rep merge with dedup.
- NSYS-Analyzer-and-Visualizer —
  https://github.com/eshama1/NSYS-Analyzer-and-Visualizer — confidence
  intervals on comparisons; chunked sqlite reading for multi-GB traces.
- nsys2json — https://github.com/chenyu-jiang/nsys2json — Chrome-trace
  conversion (we have an equivalent).
- nsys2prv (BSC) — https://pypi.org/project/nsys2prv — bridge into the
  Paraver/Dimemas HPC analysis stack.
- hyxcl/nsys_recipes — https://github.com/hyxcl/nsys_recipes — full overlap
  matrices (comm-comm/comm-compute/compute-compute) with same-name grouping.
- GPU-Perf-Analyzer — https://github.com/mingxu1067/GPU-Perf-Analyzer —
  user-editable kernel taxonomy config; bubble-time metrics.
- torch-tb-profiler — https://github.com/pytorch/kineto/tree/main/tb_plugin —
  per-worker load-balance view; automatic recommendations.
- NVIDIA DLProf (deprecated) —
  https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/index.html —
  Tensor-Core-eligible-but-unused reporting.
- MLCommons Chakra — https://github.com/mlcommons/chakra — standardized
  execution-trace DAG feeding what-if simulators.

### Methodology sources surveyed

- KernelPro — https://arxiv.org/html/2606.26453 — 15 codified expert rules
  with published thresholds and per-rule hit rates; external calibration
  data for our rule engine *(adopted: threshold calibration)*.
- Chopper — https://arxiv.org/pdf/2512.08242 — kernel→op→cluster
  inefficiency ladder.
- "Measuring GPU Utilization One Level Deeper" —
  https://arxiv.org/html/2501.16909v1 — util→occupancy→per-pipe hierarchy;
  low-occupancy kernels can still saturate a pipe.
- "Dissecting the NVIDIA Hopper Architecture" —
  https://arxiv.org/pdf/2501.12084 — per-instruction-mix frequency variance;
  never derive throughput from raw cycles.
- "Microbenchmarking NVIDIA Blackwell" — https://arxiv.org/pdf/2512.02189 —
  tcgen05/TMEM as a new occupancy-limiting resource.
- FlashAttention-3 — https://pytorch.org/blog/flashattention-3/ — SFU/exp as
  attention critical path (989 vs 3.9 TFLOP/s ratio on H100) *(adopted:
  SFU/softmax-bound rule)*.
- PyTorch Hopper TMA deep dive — https://pytorch.org/blog/hopper-tma-unit/ —
  DRAM-throughput delta as TMA-adoption success metric.
- NVIDIA Analysis-Driven Optimization series —
  https://developer.nvidia.com/blog/analysis-driven-optimization-preparing-for-analysis-with-nvidia-nsight-compute-part-1/ —
  canonical SOL→stall→memory→source loop.
- NVIDIA DL performance guide (matmul) —
  https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html —
  tile/wave quantization arithmetic.
- Volkov, "Better Performance at Lower Occupancy" (GTC 2010) —
  https://www.nvidia.com/content/gtc-2010/pdfs/2238_gtc2010.pdf — ILP
  substitutes for occupancy *(adopted: occupancy-advice suppression guard)*.
- GPUs Go Brrr (Hazy Research) —
  https://hazyresearch.stanford.edu/blog/2024-05-12-tk — async-engine
  busyness as the Hopper mental model.
- SemiAnalysis H100/GB200 training benchmarks —
  https://newsletter.semianalysis.com/p/h100-vs-gb200-nvl72-training-benchmarks —
  MFU-at-scale reference bands and goodput framing.

### Academic / HPC toolchains

- HPCToolkit — https://hpctoolkit.org/ — heterogeneous calling-context
  trees (CPU launch stack + reconstructed device call tree); GPU-idleness
  blame shifting.
- GPA (CGO'21) — https://github.com/Jokeren/GPA — per-suggestion estimated
  speedups from PC sampling *(adopted: speedup upper-bound model)*.
- LEO — https://arxiv.org/html/2604.20032 — backward slicing from stalled
  instructions to producer instructions via SASS def-use chains.
- Scalasca / Score-P / Cube — https://www.vi-hps.org/tools/score-p.html —
  wait-state taxonomy, root-cause delay analysis, cross-rank critical path.
- Hatchet / Thicket (LLNL) — https://github.com/LLNL/hatchet —
  call-tree-indexed DataFrames with run-diff algebra; ensembles.
- Extra-P / Extra-Deep — https://github.com/extra-p/extrap — empirical
  scaling models f(p, n); scalability-bug flagging before scale-out.
- Omnitrace / ROCm Systems Profiler — https://github.com/ROCm/omnitrace —
  causal profiling (COZ-style virtual speedups).
- Intel VTune + Advisor — offload/what-if cross-device projection models.
- DeepContext (ASPLOS'25) — https://arxiv.org/abs/2411.02797 — rules that
  condition jointly on framework context and kernel metrics.
- GCStack + GCScaler (ISCA'25) —
  https://dl.acm.org/doi/10.1145/3695053.3731068 — already our stall-closure
  basis; GCScaler's what-if scaling not yet borrowed.

### Survey verdict (2026-07)

Single-report NCU interpretation depth: ours exceeded everything surveyed.
Confirmed unique at survey time: closed two-sided stall accounting,
PM-sampling multi-pass replay-group handling, clock-confound guards,
trace-quality gating, nsys schema-version adaptation, nccl-inspector
integration. Confirmed gaps clustered in: prescriptive/quantified advice,
A/B diff product, calling-context attribution, multi-run ensembles,
static-SASS evidence channel, idle root-cause attribution, and
visual/computational scalability to hundreds of ranks.
