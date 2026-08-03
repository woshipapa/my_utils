# Capability evolution: what was added, from where, and why

This is the provenance record for `my_utils/profiling`. For each capability:
what it does, **where the knowledge came from**, and **what problem forced it**.

Capabilities are grouped by the era that produced them. Within each era the
ordering is roughly chronological. Every claim here is traceable to a commit;
where a capability exists because something was *wrong*, that is stated rather
than smoothed over, because the defect is usually the most useful part of the
story.

A note on sourcing. Three kinds of provenance appear below, and they are not
equally strong:

- **VERIFIED LOCALLY** — read from the Nsight Compute install on this machine
  (2026.1.1: shipped `.section` files, rule sources, `ncu_report.py`). Strongest.
- **VERIFIED FROM SOURCE** — read from a primary source: NVIDIA documentation,
  PyTorch/NCCL source, a published paper.
- **DERIVED** — our own reasoning, not sourced from a vendor. Marked as ours
  wherever a threshold or heuristic is involved, so nobody mistakes it for
  vendor guidance.

---

## Era 1 — Capture plumbing (2026-01 to 2026-03)

The original purpose: start and stop profiling around the right part of a
training run.

| Capability | Source | Why |
|---|---|---|
| `ProfileManager`, `CaptureController` | — | Profiling a whole training run produces an unreadable multi-GB trace. Needed to capture specific iterations. |
| Range profiling, iteration/micro-batch windows | — | The interesting window is a steady-state step, not startup. |
| `profile_wrapper.sh`, launcher YAMLs | — | The nsys/ncu command lines are long and easy to get subtly wrong. |
| Framework adapters (torchtitan, verl, slime, roll, sglang, vllm) | Each framework's source | Every stack starts its step loop differently; there is no generic hook. |

**What this era did not do:** anything with the data afterwards. It produced
traces; reading them was manual.

---

## Era 2 — Reading the trace (2026-03 to 2026-05)

| Capability | Source | Why |
|---|---|---|
| `nsys_sqlite_provider`, `nsys_sql_skills` (29 SQL skills) | nsys SQLite schema, VERIFIED FROM SOURCE | The `.nsys-rep` export schema changes between versions; hard-coded queries broke. |
| `nsys_schema_adapter` | Same | Column names differ across nsys releases; resolve them at runtime instead of pinning a version. |
| `nsys_module_kernel_compare`, `nsys_diff` | — | "Did my change help?" needs two traces compared, not one read. |
| `nsys_timeline_html`, visualization layer | — | A table of kernels does not show gaps; a timeline does. |
| NCU CSV + report tooling, `NcuReportSkillEngine` | `ncu_report` Python API | Reading `.ncu-rep` by eye does not scale past a few kernels. |
| NCCL inspector tools | NCCL source | Collective behaviour is invisible in a plain kernel trace. |
| H100 metric aliases | ncu metric listings | Metric names changed on Hopper; lookups silently returned nothing. |

**The pattern that starts here:** a lookup that fails silently is worse than one
that raises. It recurs through the entire history.

---

## Era 3 — From measurement to diagnosis (2026-07)

The shift: stop reporting numbers, start reporting conclusions — and make every
conclusion carry its evidence.

### 3.1 The rule engine and metric catalog

| Capability | Source | Why |
|---|---|---|
| `METRIC_CATALOG` (182 metrics) | ncu metric reference + shipped sections | Raw metric names are meaningless without units, ideal values and interpretation. |
| Arch-variant name resolution | Observed renames across Ampere/Hopper/Blackwell | `_v2` suffixes and `dram__bytes_op_read` spellings meant a single hard-coded name returned `None` on half the fleet. |
| `classify_bottleneck` + `SOL_THRESHOLDS` | **VERIFIED LOCALLY**: `sections/SpeedOfLight.py` | The four constants (10/60/80/1) are read verbatim from NVIDIA's shipped rule, not from blog posts. |
| `analyze_stalls`, 19 stall reasons | ncu `WarpStateStats` + shipped `CPIStall` rule | "Latency bound" is not actionable; the stall reason is. |
| `compute_roofline` | NVIDIA roofline sections; FLOP weights `add + mul + 2*fma` | Arithmetic intensity decides whether a kernel *can* go faster. |
| Speedup ceilings from stall share | GPA (CGO 2021), VERIFIED FROM SOURCE | An estimate labelled as a ceiling is honest; labelled as a prediction it is not. |

**Blackwell packed-FP32 correction** (CC 10.0/10.3): `fadd2`/`fmul2`/`ffma2`
count two ops per instruction. Omitting them undercounts FP32 by up to 2x.
Source: NVIDIA's `SOLFPRoofline` rule. Gated on compute capability, and when the
packed counters are absent on a CC 10.x part it emits a caveat rather than
silently reporting half.

### 3.2 Evidence and provenance — the architectural turn

**Why it exists:** the request "不能只凭符号名字来判断" — do not judge by symbol
names alone. A kernel called `..._gemm_...` that never activates the tensor pipe
is either a fallback path, an unsupported dtype, or too small to tile. Trusting
the name hides all three.

`analyzers/evidence.py` ranks every claim by where it came from:

```
HW_COUNTER 100 > NVTX 80 > SOURCE 75 > CUDA_API 60 > LAUNCH_CONFIG 50 > KERNEL_NAME 20
```

A counter beats a name. When they disagree, that contradiction becomes a
finding — the disagreement is more informative than either claim alone.

| Capability | Source | Why |
|---|---|---|
| Provenance ranking, `fuse_claims`, `attribute_kernel` | DERIVED (ours) | Symbol names are the weakest evidence in a report and were being treated as the strongest. |
| CUTLASS symbol parsing | CUTLASS naming conventions | The exception that proves the rule: CUTLASS symbols *do* encode tile/stages/cluster reliably. |
| NCCL algorithm caveat | **VERIFIED FROM SOURCE**: NCCL `best_kernel()` | ~670 device functions collapse onto ~40 kernel symbols with hard-coded RING/TREE+LL. The symbol is reliable for collective *kind*, never for algorithm or protocol. |
| CuTe DSL / FlashAttention-4 naming | CuTe `mangle_name` | `kernel_kernel_<args>_0` carries no shape or dtype at all — an uninformative name is a finding, not a quirk. |
| Triton/Inductor fused-op names | `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES` | Only set when that env var is on; otherwise names are opaque. |

### 3.3 Refusing to conclude

The counterpart to evidence: knowing when to say nothing.

| Capability | Source | Why |
|---|---|---|
| `trace_quality.py` (13 checks) | Multiple, per check | Warmup, autotuning, CUDA-graph opacity, rank completeness, GPU-metric gaps, profiler overhead each invalidate specific conclusions. |
| `check_clock_alignment` | NTP/UTC error bounds vs collective durations | Cross-host clock error (1–10 ms) exceeds collective durations (0.1–5 ms), so cross-rank timing claims below a 10 ms floor are noise. |
| `check_derived_metric_invariants` | Arithmetic identities | MFU > 100%, HFU < MFU, or a sparsity-inflated peak are wrong denominators, not interesting results. |
| Straggler/victim inversion | DERIVED, from collective semantics | The slow rank waits *least*; fast ranks wait inside NCCL. Naive "who waited longest" identifies victims, not culprits. |
| `analyze_throttling` | NVML clock-event bitmask; DCGM fields 112/240/241 | **The trap:** `GpuIdle (0x1)` and `ApplicationsClocksSetting (0x2)` are not throttling. A bare `mask != 0` reports every idle GPU as throttled. |

### 3.4 A fabrication incident, and what it changed

During research an agent produced **fabricated statistics** — specific-sounding
figures attributed to real organisations, and in one case a fabricated claim
that another statistic was fabricated. These were committed before being caught.

Response: purge from code and docs, record the retraction itself in
`docs/research/`, and adopt a standing rule — **an unverifiable claim is marked
UNVERIFIED, never rounded up to true**. `docs/research/` now tags every finding
VERIFIED / UNVERIFIED / CORRECTED / RETRACTED.

The one still-open item from that era: whether nsys records `pt_data_*` in
`ThreadNames`. Marked UNVERIFIED in code. Needs a real `--trace=osrt` report.

---

## Era 4 — Completeness and ground truth (2026-07, current)

Driven by three requests: cover **all axes**, get **all ncu metrics**, and add
**PC/PM sampling and source correlation**.

### 4.1 The coverage bug

`_ANALYSIS_REQUIREMENTS` gated four analyses on catalog keys that did not exist
(`warp_cycles_per_issue` for `warp_cycles_per_issued_inst`, `threads_per_inst`
for `warp_exec_efficiency`, and five more). `MetricView.get` returns `None` for
an unknown key rather than raising, so **stalls, coalescing, divergence and
spilling were reported as uncollected on every report**, including full ones
where those rules had run and emitted findings.

This is exactly the confusion `analysis_coverage` exists to prevent. Fixed;
`TestCoverageKeysAreReal` now pins every key to the catalog.

### 4.2 Capabilities added

| Capability | Source | Why |
|---|---|---|
| `analyzers/axes.py` — 14 axes | DERIVED (ours) | Three vocabularies described the same things (`uncoalesced_global_access` / `UncoalescedGlobalAccess` / `memcpy_bound`), so cross-checking matched nothing. Also: an unexamined axis and a clean axis looked identical. |
| `shipped_rules.py` — reconciliation | `IAction.rule_results_as_dicts()` | NVIDIA's own findings were being extracted and thrown away while we ran a re-implementation of the same rules. Agreement raises confidence; disagreement is the valuable output. |
| LOCAL vs GLOBAL speedup handling | ncu speedup-estimation semantics | A 40% LOCAL estimate on a 5%-of-runtime section is worth ~2%. Never promoted to a kernel-level ceiling. |
| `group_report_metrics`, `UNIT_AXIS` | PerfWorks grammar, shipped sections | Catalog interprets 177 metrics; `--set full` carries thousands. The rest were loaded and touched by nothing. |
| `audit_catalog_against_sections` | **VERIFIED LOCALLY**: shipped `.section` files | 147 spellings section-backed, 37 need explicit `--metrics`, 32 unknown. The last bucket is *candidates*, not errors — sections request a subset of what a device exposes. |
| `analyze_memory_hierarchy` | ncu memory counters | SOL says "memory bound" but not *which level*. An L2-bound kernel does not get faster with more DRAM bandwidth. Sysmem/peer aperture misses reclassify the problem entirely. |
| `analyze_issue_efficiency` | ncu `SchedulerStats` | Occupancy counts resident warps; `warps_eligible` counts usable ones. High occupancy with no eligible warps means adding warps adds stalled warps. |
| `analyze_instruction_mix` | 13 `inst_pipe_*` metrics, previously unread | Compute SOL is a **max over pipes**, so an XU- or LSU-bound kernel shows modest SOL and reads as latency bound. |
| `hierarchical_roofline` | LBNL hierarchical-roofline method | One DRAM point says *whether*; three points sharing a numerator say *why*. The spread between levels is a direct read on locality. |
| `measurement_context.py` | **VERIFIED FROM SOURCE**: ncu `--cache-control` default | ncu is **cold-cache by default**; wall-clock is warm. A 2x difference read as a regression sends someone to optimise a kernel that did not change. |
| `source_correlation.py` | **VERIFIED LOCALLY**: `ncu_report.py` 2026.1.1 | For a fused kernel, "stalls on long-scoreboard" does not say *which stage*. No whole-kernel counter can. |
| `sampling_validity.py` | **VERIFIED LOCALLY**: `PCSamplingData.py`, `PMSamplingData.py` | Eleven samples and eleven thousand render identically. Drops and overflows **bias** rather than add noise. |
| Roofline dtype basis from counters | Per-precision tensor-op counters | Was `fp16 if tensor else fp32`. On Hopper the FP8 peak is 2x the FP16 peak, so FP8 kernels were graded as twice as efficient as they are. |
| Tile quantisation | DERIVED (ours) | Needs the *problem* shape; the symbol encodes only the *tile* shape. When absent, reports itself unasked rather than passing. |

### 4.3 The local install

Midway through this era I reported that no Nsight Compute was installed. **That
was wrong** — a zsh glob failure aborted the check before it tested
`/Applications`. A full 2026.1.1 install was present.

This changed the sourcing of everything after it. Shipped `.section` files, rule
sources and `ncu_report.py` became directly readable, which is why the later
capabilities are marked VERIFIED LOCALLY rather than inferred from
documentation. It also settled the SOL-threshold question (below) that had been
open as a judgment call.

Had it been found at the start, several earlier guesses would have been
unnecessary.

### 4.4 Threshold reconciliation

Research reported three "conflicting" SOL classification schemes. Reading
`SpeedOfLight.py` showed **there was no conflict**: the single `max(sm,mem) < 60`
gate and the banded >80/60–80/<60 table are partial descriptions of one rule
with four thresholds, and our values already matched it.

The genuine finding: a 60/40 two-axis table sat in `SOL_THRESHOLDS`, appeared in
no shipped rule, and was read by nothing. Removed — an unverified threshold
beside verified ones invites equal trust.

---

## Era 5 — Borrowing from the ecosystem (2026-07)

A four-track survey (NCU tools, NSYS tools, methodology writeups,
academic/HPC toolchains — full list in `docs/REFERENCES.md`) compared this
toolkit against everything findable. Verdict: single-report interpretation
depth exceeded the surveyed field, but several mechanisms around it had a
decade of prior art we lacked. The ones below were adopted. Each entry says
what was borrowed and — because none of them fit unchanged — how it was
adapted to this engine's honesty rules.

| Capability | Borrowed from | Adaptation |
|---|---|---|
| Speedup upper bounds on stall findings (`ncu/speedup_model.py`) | GPA (CGO'21, github.com/Jokeren/GPA): every suggestion carries an estimated speedup, VERIFIED FROM SOURCE | GPA models instruction-level dataflow; we have kernel-level closed stall shares. So ours is a *share-removal upper bound* — 1/(1−share), clipped by three ceilings we already compute (SOL of the busiest unit, DRAM roofline for memory stalls, occupancy headroom for latency-hiding stalls) — and is labelled "upper bound, not a prediction". Withheld entirely when the stall stack fails its closure check, with the reason stated. |
| Stall severity tiers 40%/60%, barrier gate 30% | KernelPro (arXiv:2606.26453), which published per-rule hit rates — the only external calibration data found for rules like ours, VERIFIED FROM SOURCE | Tiers adopted where ours were undocumented. The general 30% stall gate was *kept* despite KernelPro's 40%, because ours traces to NVIDIA's shipped CPIStall rule — a stricter threshold with a stronger source outranks a looser one with hit-rate data. |
| Occupancy-advice suppression | Volkov, "Better Performance at Lower Occupancy" (GTC 2010); corroborated by arXiv:2501.16909, VERIFIED FROM SOURCE | When any `sm__pipe_*` or issue-slot utilization is ≥80%, "increase occupancy" advice is replaced — not deleted — by a statement that a saturated pipe means more warps cannot help. The finding stays visible so the low occupancy itself is still on record. |
| A/B report diff (`ncu-diff`, `ncu/report_diff.py`) | KuangjuX/ncu-cli (`diff` command) and PerfDigest-MCP (`compare_metrics`), VERIFIED FROM SOURCE | Their shape (per-metric deltas, severity coding); our physics. The diff leads with the clock-confound guard (reusing `compare_measurements`, not reimplementing), refuses raw-time "speedups" when clocks differ >1%, adds a findings-level diff (appeared/disappeared/escalated), noise floors per metric kind, and marks the stall-delta section unreliable when either side failed closure. |
| Replay clock-drift check (`check_replay_clock_drift`) | Thermal-throttle cautions in the CUTLASS profiler docs and Nsight Compute Profiling Guide, VERIFIED FROM SOURCE; detection mechanism DERIVED | The sources say "cap iterations to avoid throttle"; they offer no detection. Ours derives per-replay-pass effective SM clocks from PM-sampling bucket data and flags ≥2% spread — with estimate *typing* (elapsed-cycle clocks are measured; active-cycle clocks are lower bounds; a drift claim is made only when the direction is provable). |
| Clock-control bias caveat | Nsight Compute Profiling Guide: base-clock lock cannot lower HBM clock on H100-class parts, VERIFIED FROM SOURCE; symptom detection DERIVED | ncu-rep records no clock-control flag, so detection is by symptom (SM ≤92% of rated while DRAM ≥97%). Stated as a bias direction on compute:memory verdicts, never as "data is wrong"; explicitly notes power/thermal capping produces the same signature. |
| SFU/softmax-bound rule (`analyze_sfu_pressure`) | FlashAttention-3 (pytorch.org/blog/flashattention-3): special-function throughput (3.9 TFLOP/s on H100, vs 989 tensor) as the attention critical path, VERIFIED FROM SOURCE | Their observation, our gating: fires only with tensor pipe ≥20% (a GEMM worth overlapping), XU ≥60% of peak (real pressure), tensor <80% (not itself the limiter) — so ordinary GEMMs and light epilogues never fire it. On our fused GEMM+RMSNorm report it correctly stays silent (XU at 0.2%) and says so. |
| Hierarchical arithmetic intensity (L1/L2/DRAM) | NERSC roofline-on-nvidia-gpus methodology; hierarchical-roofline NVIDIA blog, VERIFIED FROM SOURCE | Fixed FLOP numerator over per-level bytes; missing levels reported as absent, never invented (our reports lack `l1tex__t_sectors.sum` — the output says "not collected: L1"). Cache-blocking finding fires only on the clear signal (DRAM-compute-bound AND AI_L2 < 0.8× L2 ridge). |
| Instruction roofline + precision-aware ceilings | Ding & Williams 2019 (instruction roofline); Giotyp/GPU-Roofline-Python (multi-precision ceilings), VERIFIED FROM SOURCE | Warp-GIPS against a first-principles issue ceiling (validated: matches the report's own `pct_of_peak` to four decimals); its 4-schedulers/SM assumption stated in evidence. Precision peaks are dense datasheet numbers — sparsity values never pre-doubled into a ceiling. |

Two survey findings were recorded but *not* adopted, deliberately: GPUscout's
static-SASS evidence channel (the one surveyed mechanism deeper than ours —
needs a disassembly pipeline; queued, not rushed) and causal profiling
(Omnitrace/COZ — requires runtime perturbation, does not map onto offline
report data; a trace-replay approximation is queued instead).

---

## Recurring principles

Each of these exists because its absence produced a specific wrong answer.

1. **Silence must be attributable.** An unexamined axis and a clean axis are
   different facts. `coverage`, `axes` and `metric_inventory` all exist to keep
   them distinguishable.
2. **Never conclude from a name.** Provenance ranking, with contradictions
   surfaced rather than resolved silently.
3. **Refuse rather than guess.** Bus bandwidth is not derivable from kernel
   timing; cold- and warm-cache numbers are not comparable. Both return an
   explicit refusal, not a plausible number.
4. **A lookup that fails silently is a bug.** Both the coverage-key defect and
   the arch-variant metric misses were this.
5. **Mark our own thresholds as ours.** The 200-sample ranking floor and the
   axis table are DERIVED; the SOL constants and sampling checks are NVIDIA's.
   Mixing them without labels makes all of them look equally authoritative.
6. **Bias is worse than noise.** Dropped PC samples correlate with the busiest
   periods, so the hottest code goes missing first. That blocks attribution
   outright, where a merely small sample only blocks ranking.

---

## Still open

| Item | State |
|---|---|
| `pt_data_*` in nsys `ThreadNames` | UNVERIFIED in code. Needs a real `--trace=osrt` report. |
| Hopper/Blackwell metric counts | Unverified — the local install ships Linux binaries that cannot run on this host, so `--query-metrics` is unavailable. |
| Counter-based `smsp__average_warps_issue_stalled_*` spelling | Unverified. |
| Inline-function attribution | Nsight Compute's Inline Functions Table semantics are undocumented on the API side. |
| Overhead-bound as a top-level verdict | Not implemented. Needs cross-source (nsys launch gaps + ncu), which the two-source architecture supports but no rule yet uses. |

---

## Where to read next

- `PERFORMANCE_ANALYSIS_HANDBOOK.md` — how to use all of this
- `docs/research/2026-08-ncu-audit/` — the current audit record, including
  verified real-report evidence and open validation boundaries
- `tests/profiling/` — the distributed executable specification (597 tests in
  the final 2026-08 audit validation); `_synthetic_loader.py` keeps pure analysis tests
  independent of the package-level torch import
