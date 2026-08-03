# NCU Capability Coverage Matrix

Status: evidence ledger for the 2026-08 audit. This is deliberately a matrix of
what is proved, rather than a feature list.

## Definitions

Three percentages in this repository answer different questions and must not be
substituted for one another:

| measure | result | denominator | meaning |
|---|---:|---|---|
| Diagnostic-rule coverage | 11/11 | `ncu_diagnostics` rule requirements | All implemented per-kernel rules had their required inputs in the baseline report. It says nothing about rules that do not exist. |
| Logical catalog resolution | 146/182 (80.2%) | stable `METRIC_CATALOG` keys | The baseline H100 report resolves that many logical quantities through one of their candidate spellings. |
| NCU full-section name coverage | 168/395 (42.5%) | base metric names requested by local NCU 2026.1.1 `--set full` sections | The catalog has a stable interpretation for that many requested hardware/derived counter families. The other 227 are present in the collection plan but not individually judged by a repository rule. |

The report has 2,160 unique metric records, so neither catalog percentage is a
claim that the reader discarded unrecognised data. `group_report_metrics` keeps
uninterpreted metrics in the inventory. The exact 42.5% figure is from
`verify_catalog_coverage()` against the locally installed section files; it is
version-specific and must be refreshed whenever Nsight Compute changes.

## Real-report Evidence

Source corpus: the six H100 SXM5 reports documented in
[`desktop_h100_report_corpus.md`](desktop_h100_report_corpus.md). They are a
single kernel shape and collection configuration, not a general H100 coverage
claim.

| Capability | Code path | H100 evidence | Test status | Audit state |
|---|---|---|---|---|
| Static counters and NVIDIA rule results | `walk_report_once`, `diagnose_kernel` | 2,160 metric records and 11 rule results in each report | Synthetic report-reader tests | Verified for this corpus |
| Source and PC-sampling attribution | `source_correlation` | Baseline maps 9,559 samples across 183 source lines | Validity and source-correlation tests | Verified for baseline source mapping |
| PM timeline rendering | `analyze_pm_sampling` | 4 reports have no drops; 2 report dropped samples | Sampling-validity and report-reader tests | Only no-drop timelines are admissible |
| PM phase conclusions | `check_pm_sampling_validity` | `mode1` has 390 drops; `mode3` has 187 | Drop/merge regression tests | Correctly blocked for affected reports |
| Kernel Replay identification | `_infer_collection_context` | All six have nonzero memory backup and 44 or 46 replay passes | Manifest/conflict tests | Verified from report evidence |
| Application Replay identification and matching | sidecar manifest | No report with an application-replay manifest | Sidecar preserves replay mode, `app-replay-match`, and strict/relaxed mode | Unverified on a real report |
| Range and Application Range Replay | sidecar manifest and `MeasurementContext` | No range-replay report | Context tests preserve concurrency but block per-kernel attribution; sidecar preserves range/graph options | Unverified on a real report |
| Cache-control interpretation | collection sidecar | Historical report invocations are unknown | YAML sidecar test; missing sidecar fails closed | Historical durations are `NOT_COMPARABLE`, not assumed cold-cache |
| Clock-control interpretation | collection sidecar plus measured clocks | Historical report invocation unknown; clock guard rejected four comparisons | YAML sidecar and diff tests | Correct behavior verified, collection provenance missing |
| Differential performance claim | `ncu-diff`, provenance guard, and repeat statistics | CTA ping-pong is diagnostic-only: no sidecars, 6.2% clock/cycle disagreement | Diff, coverage, alias, PM/PC, and repeat tests | Repeat trials with explicit sidecars still required |
| Cross-architecture aliases | `MetricView.get` candidates | H100 only | Catalog alias tests | A100/Ada/B200 reports required |
| Current-version SASS instruction-size, Function Statistics, and HW warp-ID surfaces | `current_report_surfaces`, `ncu-audit` | The 2026.1.1 H100 corpus does not expose any of the 2026.2 surfaces | Synthetic discovery/renderer tests and one real-report absence check | Awaiting a 2026.2+ report; raw names and units are intentionally not guessed |
| MPS/MIG PM-sampling context scope | sidecar v2 plus `check_pm_sampling_validity` | Historical reports have no MPS/MIG metadata | Scope-gating, provenance, and measurement-context tests | Validate on a controlled MPS/MIG collection with context-switch data |

## Baseline Catalog Gaps

The baseline report resolves 146 keys and does not resolve 36. An unresolved
logical key is not automatically a bug: a counter can be inapplicable to this
instruction mix, unavailable on this GPU, omitted by the selected sections, or
renamed beyond the catalog aliases. The classification below is a collection
plan, not a claim of hardware support.

| class | keys | interpretation and next validation |
|---|---|---|
| Inactive instruction paths | 12 scalar FLOP keys plus `tensor_ops_fp8` | This BF16 tensor kernel need not execute scalar or FP8 operations. Collect a scalar and an FP8 workload before judging these aliases. |
| Unsupported architecture feature | `inst_pipe_tmem` | TMEM is a Blackwell feature; H100 is not evidence for its metric spelling. Validate on B200. |
| Raw traffic / pipeline representation | `dram_bytes`, global-sector, shared-bank, `sm_busy`, `pipe_tc_util`, L1/L2 traffic and sector keys | The report contains related derived/alternate metrics, but no candidate matched these stable keys. Use H100 `--query-metrics` and the exact section command to distinguish a renamed counter from an unrequested one. |
| Launch/context metadata | `green_context_id`, `uses_mps`, `func_cache_config` | The kernel did not establish that the relevant CUDA context features were active. Exercise MPS/green-context and cache-config workloads. |
| Opcode and spilling detail | `inst_executed_per_opcode_category`, `spill_local_inst`, `local_spill_requests_pct` | Baseline carries a high-level spilling signal but not each requested detail. Profile a controlled spilling kernel with `InstructionStats` and `SourceCounters`. |
| Sampler health/configuration | `pcsamp_buffer_overflow`, `pmsampler_interval_cycles` | Absence does not mean zero loss or a safe interval. The new dropped/merged gates are authoritative when those profiler counters exist; test controlled buffer pressure and explicit intervals. |

## Required Evidence Before Calling the Analysis Complete

1. Capture a controlled Application Replay report and a Range Replay report,
   each with the generated collection sidecar, then verify diagnosis wording and
   diff compatibility end to end.
2. On an otherwise idle H100 (GPU 6 or 7 only if the remote host is used), run
   `ncu --query-metrics` and a minimal section collection for each unresolved
   H100 key group. Do not infer an unsupported name from this one full report.
3. Add representative A100, Ada, and B200 reports. For every alias selected by
   `MetricView.get`, check both the value and its unit/rollup semantics.
4. Repeat each candidate optimization at least twice per side under an explicit
   sidecar (`replay_mode`, cache and clock controls, warmups, input
   distribution, workload/input identity) and pass the files with
   `--report-a-repeat` / `--report-b-repeat`. The tool uses median/MAD
   intervals; it does not promote a single observation to `VALID_SPEEDUP`.
5. Build at least one source-enabled collection with `-lineinfo` from the
   target build, rather than relying only on the existing absolute source paths.

Until these items are done, the implementation is a capable H100 diagnostic
tool with bounded evidence, not a complete cross-version or cross-architecture
NCU analysis system.

## Current Documentation Baseline

The current documentation baseline is Nsight Compute 2026.2.1. The local macOS
bundle is 2026.1.1.0, and `ncu-audit` therefore reports `upgrade_required`.
The audit reads the installed `.section` files separately from feature support:
the current local `--set full` denominator remains 395 base metric families,
of which the curated catalog covers 168 (42.5%). This is still a coverage
measure for local 2026.1.1 sections, not support for 2026.2.1 features.

When a 2026.2+ report becomes available, run `ncu-audit` with its version and
the wrapper's explicitly enumerated options, then parse one source-enabled
report. `current_report_surfaces` carries the report-provided raw names/API
members into JSON; only then should a stable catalog key or diagnostic threshold
be added.

## External Evidence Boundary

NVIDIA's current [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
is the normative source for Kernel, Application, Range, and Application Range
Replay. It establishes the hardware-counter pass limit, Kernel Replay memory
backup/restore, Application Replay determinism and matching, and range-level
counter attribution. The current [CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
is the normative source for the collection switches recorded in the sidecar.

Public command wrappers are useful only as compatibility evidence. For example,
an older [NCU wrapper](https://gist.github.com/dhy2000/349ec82ff2c29d68d935976a6a4e3591)
hard-codes `--clock-control base`, demonstrating why an unversioned default must
not be inferred from a report. It is not used to define current behavior.

The related systems literature reinforces the uncertainty policy but is not
treated as an NCU specification: [Tintin (OSDI 2025)](https://www.usenix.org/conference/osdi25/presentation/li)
and [BayesPerf (ASPLOS 2021)](https://dl.acm.org/doi/10.1145/3445814.3446739)
both address limited counter resources and measurement error. Their portable
lesson here is narrow: expose collection loss/uncertainty and refuse unsupported
attribution. NVIDIA documentation remains the source for CUDA profiler semantics.
