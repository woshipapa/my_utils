# Progress Log — Profiling Analysis Engine Overhaul

Living checklist. **Tick and strike through when done.** Anyone (human or agent)
picking this up: read [`AGENTS.md`](AGENTS.md) first, then this file for what is
already done and what remains.

Started 2026-07-18. Last updated 2026-07-19.

**Status: 85 tests passing. Phase 1, 2, 4 done; Phase 3 partly done; Phase 5 open.**

---

## Phase 1 — Audit & research ✅

- [x] ~~Audit current ncu/nsys analysis coverage~~ — found: stall matching by
      substring only, no FLOP counters (so no real roofline), arch detection knew
      only Hopper/Blackwell, no per-workload expectations, no unified triage.
- [x] ~~Research NCU metric taxonomy~~ — ground truth from local Nsight Compute
      2026.1.1 install (`<install>/sections/*.py`), so thresholds are exact.
- [x] ~~Research nsys schema / recipes / rules~~ — ground truth from local Nsight
      Systems 2026.2.1 install; export schema 3.25.0.
- [x] ~~Research ecosystem analyzers~~ — HTA, torch-tb-profiler, omniperf, VTune,
      HPCToolkit blame-shifting, GPA speedup model, DrGPU stall tree.
- [x] ~~Research expert triage flows~~ — NVIDIA's calibrated M1-M5 host-bound
      constants (shipped as Claude skills in TensorRT-LLM), GEMM tile/wave
      quantisation, FA2/FA3 expectations, NCCL busbw.
- [x] ~~Research modern DSLs~~ — ThunderKittens, Triton-distributed, CuTe TMA
      atoms, Blackwell tcgen05/UMMA/TMEM, NCCL PAT.
- [x] ~~Completeness audit for blind spots~~ — measurement traps, green contexts,
      CUDA-graph attribution, Triton autotuning contamination, occupancy myth.

## Phase 2 — Core engine ✅ (pushed: `88d4f62`)

- [x] ~~`hardware/gpu_specs.py`~~ — 25 SKUs Volta→Blackwell, all **dense** peaks,
      measured vs spec bandwidth, ridge points matching published values.
- [x] ~~`sources/kernel_taxonomy.py`~~ — 20 categories, framework fingerprints
      (incl. mangled `_ZN7kittens`), GEMM tile parsing, NCCL algo/protocol parsing,
      megakernel detection.
- [x] ~~`ncu/metric_catalog.py`~~ — 105 metrics with per-arch spelling candidates,
      complete 19-reason stall taxonomy, arch detection 7.0→12.1.
- [x] ~~`ncu/ncu_diagnostics.py`~~ — rule engine mirroring NVIDIA's exact
      thresholds + roofline from FLOP counters + per-category expectations.
- [x] ~~`analyzers/triage.py`~~ — top-down decision tree, one verdict.
- [x] ~~`diagnose_ncu_report()` + `ncu-diagnose` CLI~~ — per-kernel diagnosis
      ranked by duration, markdown renderer.
- [x] ~~One-pass collection YAML~~ — pins every section the engine reads.
- [x] ~~80 tests, no torch/GPU needed~~ — `tests/profiling/test_analysis_engine.py`.

## Phase 3 — Correctness hardening (from the blind-spot audit) 🔄

These prevent *wrong conclusions*, which is worse than no conclusion.

- [x] ~~Roofline refuses to conclude when tensor-core FLOPs weren't collected~~ —
      CUDA-core counters don't see MMA; reports "roofline unreliable" instead of
      a bogus percentage.
- [x] ~~Occupancy findings gated on latency-hiding failure~~ — Volkov measured
      CUBLAS getting *faster* as occupancy fell 67%→33%. Now requires schedulers
      to actually be starving AND no unit saturated.
- [x] ~~Warp-specialized kernels excluded from the occupancy model~~ — `setmaxnreg`
      makes registers-per-thread a weighted artifact (FA3: 24-56 producer vs
      160-256 consumer).
- [x] ~~Green-context SM denominator~~ — small-grid and waves checks now prefer
      `launch__sm_count` (the launch's own view) over the device attribute, so a
      grid that fills its partition is no longer flagged as too small.
- [x] ~~Physical-limit sanity check~~ — any achieved FLOP/s or bandwidth above
      hardware peak is reported as a measurement bug and stops the roofline
      analysis, instead of becoming a confident percentage.
- [ ] Warm-up / autotuning contamination guard — Triton autotuning burns seconds
      of non-representative GPU work into iteration 0 (`do_bench` defaults
      25 ms warmup + 100 ms rep *per config*); refuse steady-state claims from a
      single iteration.
- [ ] CUDA-graph attribution mode — one `cudaGraphLaunch` correlation ID fans out
      to N kernels; per-kernel CPU attribution is unavailable. Detect and say so.
- [ ] Inductor `unique_kernel_names=0` detection — every kernel is named
      `triton_`; refuse name-based attribution and tell the user to set
      `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`.
- [ ] nsys data-quality gate — GPU-metrics gaps labelled "Missing Data" are
      *sampler buffer exhaustion*, not idle GPU; check Diagnostics Summary before
      trusting a trace.
- [ ] Multi-rank completeness check — refuse aggregate conclusions when the rank
      set has holes (a glob matching 6 of 8 ranks silently biases every average).

## Phase 4 — Documentation ✅

- [x] ~~`docs/PERFORMANCE_ANALYSIS_HANDBOOK.md`~~ — end-to-end: collection commands,
      analysis APIs, interpretation thresholds with sources, hardware ceilings,
      traps. Every API claim machine-verified.
- [x] ~~`AGENTS.md`~~ — orientation for agents: invariants, how to extend, the
      Python 3.10 constraint.
- [x] ~~Linked from README, profiling README, docs index~~.
- [x] ~~This progress log~~.

## Phase 5 — Remaining work 📋

- [ ] Wire triage into the nsys analysis path (`nsys_auto_analysis`) so
      `nsys-analyze` leads with a verdict.
- [ ] NCCL bus-bandwidth analysis — parse collective + size, compute busbw,
      compare against per-platform ceilings (ring ~360 GB/s / NVLS ~480 GB/s on
      8xH100; note NVLS legitimately exceeds the link spec).
- [ ] Retire the duplicate GPU tables — `nsys_auto_analysis._GPU_DB` and
      `nsys_mfu._PEAK_TFLOPS_*` should delegate to `hardware/gpu_specs.py`.
- [ ] Framework-specific profiling recipes (how to profile CuTe DSL / TK / Triton
      / FA3 kernels, which metrics matter for warp-specialized and TMA kernels).
- [ ] HTA-style analyses over nsys SQLite: launch-delay outliers, queue length,
      idle classification into host_wait / kernel_wait / other.

---

## Known-good facts worth not re-deriving

| Fact | Value |
|---|---|
| Ridge points (dense bf16) | H100 295, H200 206, A100 153, B200 281, L40S 209 |
| NVIDIA SOL thresholds | latency-bound <60, saturated >=80, balanced delta 10 |
| CPIStall gates | issue_active < 0.8 AND stall share > 0.3 |
| Occupancy rule gates | theoretical < 80%, achieved-vs-theoretical gap > 10pp |
| Bank conflicts | >= 10% of wavefronts |
| Divergence | < 24 of 32 threads per instruction |
| Tail effect | 1/(1+full_waves) >= 20% |
| HTA constants | short kernel 10us, launch delay 100us, runtime 50us, queue cap 1024 |
| NVIDIA host-bound | idle > 0.30 (0.15 graphs), launch > 0.10, util < 0.60, nccl > 0.20, **>=2 must cross** |
| busbw factors | AllReduce 2(n-1)/n; AllGather/ReduceScatter/AlltoAll (n-1)/n; Broadcast/Reduce 1 |
| FA3 build | ships `-lineinfo` by default; FA2 has it commented out |
| FA3 kernel filter | `-k regex:FlashAttnFwdSm90` (everything launches via `cutlass::device_kernel<T>`) |
| TK mangled symbol | `_ZN7kittens`; `rt_bf`/`st_bf` are aliases and never appear |

## Traps that cost real time

1. ncu flushes caches and locks clocks — L2 hit rates are not production values.
2. ncu serialises kernels — never conclude anything about overlap from ncu.
3. Datasheet tensor TFLOPS are usually **with sparsity**; halve them.
4. Zero-initialised benchmark data inflates tensor-core throughput.
5. Sustained clocks != boost clocks (H800 drops 1755 -> ~1200 MHz at its cap).
6. `nvidia-smi` utilisation is time-utilisation; a deadlocked NCCL kernel reads 100%.
7. Profiling only rank 0 misattributes stragglers as slow networks.
8. Occupancy is not performance — CUTLASS GEMMs run at low occupancy by design.
9. Per-kernel metrics are meaningless for a megakernel.
10. `CUDA_LAUNCH_BLOCKING=1` poisons every trace.
