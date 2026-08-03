# NCU Analysis Audit Status

Date: 2026-08-03

## Conclusion

The repository is **not yet a complete NCU analysis system**. It is now a
well-tested, evidence-bounded H100 diagnostic path: it can parse the supplied
Kernel Replay reports, reject known-incomplete PM data, preserve collection
provenance when the YAML runner created it, and refuse several invalid timing
comparisons. Cross-architecture compatibility and real Application/Range Replay
behavior remain unproved.

## Resolved Findings

| priority | finding | resolution | evidence |
|---|---|---|---|
| P0 | PM timelines stayed usable after `profiler__pmsampler_dropped_samples` was nonzero. | Dropped samples block timeline and phase conclusions; merged samples block phase conclusions. | H100 `mode1` (390 drops) and `mode3` (187 drops) now render PM sampling unavailable; synthetic regressions pass. |
| P1 | PM output always described the collection as Kernel Replay. | Report memory-backup evidence identifies Kernel Replay; otherwise mode stays unknown unless a sidecar records it; conflicts fail closed. | All six supplied reports have nonzero `profiler__replayer_bytes_mem_backed_up`; no unsupported application/range claim is made. |
| P1 | Collection provenance discarded Application Replay matching and Range/graph controls. | The YAML sidecar now preserves replay mode, application matching/strictness, range options, graph mode, cache control, and clock control. | Manifest and diagnosis tests. |
| P1 | Measurement context claimed all NCU runs serialised kernels. | Range and Application Range Replay preserve concurrency but explicitly block per-kernel attribution from range totals. | Context and PM-rendering tests. |
| P2 | Documentation linked deleted research notes and a deleted monolithic test. | Documentation now points to the checked-in audit record and the real distributed suite. | Repository path check. |
| P2 | NCU 2026.1+ clock default was documented as `base`. | Documentation and comments state the 2026.1+ `boost` default; sidecars record an explicit final option. | NVIDIA CLI/release documentation and regression tests. |
| P0 | Imported reports with no sidecar were silently treated as default cold-cache collections. | Missing command provenance now makes cache state `unknown` and blocks a duration/speedup conclusion. YAML-created sidecars assert that the default is known. | `mode0` versus CTA ping-pong guarded diff and synthetic regression. |
| P1 | Finding disappearance could mean B did not collect the required section. | Finding identity includes source/rule/title; missing coverage renders `not evaluated`, never `disappeared`. The diff requests all findings rather than a display-truncated prefix. | Coverage regression test. |
| P1 | Kernel diffs paired only by compiler symbol and obscured launch changes. | Sidecars accept logical kernel aliases; the result exposes pairing confidence and A/B grid/block changes. | Alias/launch-change regression. |
| P1 | Raw counts and PM/PC data had no differential audit path. | Diff now shows work-normalised counts, all catalog coverage, every changed raw numeric metric in JSON, same-pass PM aggregates, and validity-gated PC source shares. | Synthetic PM/PC and Desktop CTA diff. |
| P2 | A single pair could look like a speedup despite run noise. | Optional repeat reports use median and MAD intervals. A single pair is `INCONCLUSIVE`; only matched provenance, locked clocks, and a stable repeated improvement produce `VALID_SPEEDUP`. | Repeat-statistics regression. |
| P1 | A report parsed successfully could be mistaken for support for the latest NCU documentation. | `ncu-audit` compares an installed version to the 2026.2.1 documentation baseline; report readers discover new surfaces only when their raw metric/API evidence is present. | Local 2026.1.1 audit and Desktop H100 report regression. |
| P1 | Sidecars could omit target identity and sampling/context controls needed to reproduce a diff. | Manifest schema v2 records effective target/filter/sampling/communicator controls and best-effort host, driver, GPU, MIG, and MPS identity. | YAML/measurement-context/audit regressions. |

## Open Validation Items

| priority | gap | why it matters | required evidence |
|---|---|---|---|
| P1 | No real Application Replay or Application Range Replay report with sidecar. | Deterministic matching, host-interacting kernels, and sidecar-to-report behavior are only unit-tested. | One controlled report for each mode, with strict and relaxed matching cases. |
| P1 | No real Range Replay report. | Range metrics are aggregate; the current range guard needs API-level validation with concurrent kernels. | A two-stream range whose concurrent behavior is visible in Nsight Systems. |
| P2 | 36 of 182 H100 logical keys do not resolve in the baseline report. | They may be inactive, unavailable, unrequested, or renamed. | H100 `--query-metrics`, targeted sections, and controlled scalar/FP8/spilling/MPS workloads. |
| P2 | Only H100 SXM5 reports are available. | Alias fallbacks and rollup/unit semantics can change on A100, Ada, and B200. | At least one source-enabled report per architecture. |
| P2 | Six comparison reports are single observations. | A code change cannot be separated reliably from clock/cache/run variation. | Repeated trials under an explicit sidecar and stable inputs. |
| P3 | Existing source paths point into an external build tree. | Source attribution works for the supplied reports but not necessarily for a reproducible target build. | Recollection from a `-lineinfo` build, preferably with `--import-source yes`. |
| P1 | No report collected with NCU 2026.2 or 2026.2.1. | The local desktop install and all six supplied reports are 2026.1.1-era evidence. New instruction-size, Function Statistics, hardware warp-ID, and injection surfaces must remain evidence-discovered rather than assumed. | Upgrade one H100 host and retain a controlled report plus schema/API inventory. |

## Collection Contract For Follow-up Runs

The YAML runner writes `<report>.ncu-rep.collection.json` only after a successful
collection. For any manual invocation, either use that runner or write the same
sidecar from the final effective command. At a minimum record:

```text
--replay-mode
--app-replay-match
--app-replay-mode
--range-replay-options
--graph-profiling
--cache-control
--clock-control
```

For an A/B comparison, also put workload identity under
`ncu.collection_metadata`: `workload_id`, `problem_shape`, `dtype`,
`input_hash`, optionally `output_hash`, plus `logical_kernel_id` or a
`kernel_aliases` mapping when schedule changes rename the compiled kernel.
The runner persists these fields in the sidecar. Different workload/input
identities block the comparison; missing identities leave it diagnostic-only.

For a noise-resistant result, collect at least two independent reports per side
and pass the additional files with `ncu-diff --report-a-repeat` and
`--report-b-repeat`. The implementation uses the median and MAD-derived
intervals; no single-run output is labelled a validated speedup.

When using the remote H100 host, use only the user-designated physical GPUs 6
or 7, and wait until the chosen GPU is idle before collecting. For example, the
application must see only physical GPU 6:

```bash
CUDA_VISIBLE_DEVICES=6 ncu --replay-mode application --app-replay-match all \
  --app-replay-mode strict --cache-control none --clock-control none \
  --export reports/app_replay_case --set full -- ./target_app
```

This command is a collection template, not a claim that its resulting duration
is comparable with the existing cold-cache Kernel Replay reports. The sidecars
and `ncu-diff` must reject that comparison unless all collection context and
measured clocks are compatible.

## Verification

- Targeted provenance/diff regression set: 59 passed.
- Full profiling regression suite: 597 passed, with one pre-existing legacy
  alias deprecation warning.
- `git diff --check`: passed.
- Desktop `mode0` versus CTA ping-pong was regenerated as
  `mode0_vs_cta_pingpong_provenance_guarded.diff.md`: it preserves the counter
  diagnosis, but correctly returns `NOT_COMPARABLE` because both historical
  reports lack collection sidecars and the baseline has a 6.6% internal
  GPC/SM-clock disagreement.

For detailed corpus evidence and coverage denominators, read
[`desktop_h100_report_corpus.md`](desktop_h100_report_corpus.md) and
[`capability_coverage_matrix.md`](capability_coverage_matrix.md).

## 2026.2.1 Documentation Delta

Current NVIDIA release notes add SASS instruction-size metrics, Function
Statistics line/time ranges, hardware warp-ID data in Instruction Statistics,
the dynamic CUDA injection API, `--process-id`, and injection-path listing
options. The implementation records them as versioned capability surfaces:

- `ncu-audit --version 2026.1.1` correctly returns `upgrade_required` against
  the `2026.2.1` baseline instead of claiming current coverage.
- The existing YAML wrapper can serialize the new CLI options but a local
  2026.1.1 target cannot exercise them.
- Existing reports contain no 2026.2 instruction-size/warp-ID/Function
  Statistics evidence; the reader reports those surfaces as `not_observed`.
  It still displays the existing `sass__*per_opcode*` distributions, without
  treating them as a substitute for the newer instruction-size metrics.

The feature inventory is intentionally a validation queue, not a compatibility
claim. Mark a feature `validated` only after a 2026.2+ controlled report has
been parsed and its raw names, units, and API shape are retained in a test.
