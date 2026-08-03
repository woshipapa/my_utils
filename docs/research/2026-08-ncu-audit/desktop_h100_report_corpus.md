# Desktop H100 NCU Report Corpus

Status: first real-report corpus for the 2026-08 Nsight Compute audit.

## Scope and provenance

- Reports remain read-only at `/Users/papa/Desktop/kernel_traces/ncu_kernels/`.
- The derived analysis is under
  `profile/ncu_audit_desktop_reports_2026-08-02/analysis/`; no report binary was
  copied into this repository.
- Reader: the local Nsight Compute Python Report Interface bundled with Nsight
  Compute 2026.1.1.0. The report's collection command/version is not encoded in
  the parsed metrics and therefore remains unknown.
- All six reports have one `pod_fused_device_kernel` launch on `H100 SXM5`, grid
  132 and block 384. They carry 2,160 metric records each, PC samples, PM
  sampling, and NVIDIA rule output.

| report | size (bytes) | SHA-256 | replay passes | PM pass groups | PM dropped samples |
|---|---:|---|---:|---:|---:|
| `cta_pingpong_128x128x64_auto_no_barrier_rms.ncu-rep` | 75,720,913 | `df3cf80f39edc087f156b57631aa6fd8a5200a7c1a7a6a4ef98eeeee78e9b997` | 46 | 7 | 0 |
| `mode0_register_baseline.ncu-rep` | 96,261,074 | `4b09bf2ca5b7ed4f6e3634e5b36ef0842ecd72e5d8ad6102f358b2ef367c4cea` | 46 | 7 | 0 |
| `mode1_single_buffer.ncu-rep` | 96,483,481 | `68ebd95fcad549aebd462ad00dce0dc64a71d8f829a5141774b7c18224d94a72` | 46 | 7 | 390 |
| `mode2_double_buffer.ncu-rep` | 96,337,873 | `a79818467e8c60eb606e0e13b16887433d698ffeb5da186778f7264f27efb401` | 46 | 7 | 0 |
| `mode3_chunk_ring.ncu-rep` | 96,507,457 | `095d2a2a66e604638dbd884fbaa8955a75f8a588d11fe3f1e331102ef16bf024` | 46 | 7 | 187 |
| `producer_warp_128x128x64_auto_w3.ncu-rep` | 87,316,559 | `9912ba58f503a1a066e9bb1491e37fdba9ddc3dae403c003e59cbe414434f93b` | 44 | 7 | 0 |

`profiler__replayer_bytes_mem_backed_up` equals 140,727,661 bytes per pass for
every report. Together with non-zero `profiler__replayer_passes`, this is direct
evidence that these are Kernel Replay collections, not an inference from the
report names. NVIDIA defines the replayer metrics and PM drop counter in its
[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

## Initial differential observations

The existing `ncu-diff` clock guard is useful and fired on four of five baseline
comparisons. `mode1` and `mode2` have lower raw durations than `mode0`, but their
SM clocks are 7.8% and 6.7% higher respectively, so raw time is not a valid
speedup claim. `mode3` is the only pair whose clocks agree within 1%; its elapsed
GPC-cycle ratio is 0.991, which is too small to call an optimization without
repeat measurements.

The CTA ping-pong variant is materially different in the same report corpus:
elapsed GPC cycles are 0.817x of baseline, long-scoreboard cycles per issue slot
fall by 50.1%, and shared-store bank conflicts fall from 20,103 to zero. The
clock-normalized and cycle ratios disagree by 6.2%, so the result is promising,
not a final performance claim.

## Catalog coverage is not completeness

Against `mode0_register_baseline`, 146 of the repository's 182 logical catalog
keys resolve to an actual report metric. The remaining 36 keys are not evidence
of a parser failure by themselves: they may be unsupported on H100, absent from
the chosen section set, or represent a feature this kernel does not exercise.
The next audit pass must classify each absent key by that cause before assigning
a coverage score. The current documentation's claim of a complete common-triage
surface is therefore provisional, not validated by this corpus.

## Resolved audit finding: PM sample loss was not gated

Original severity: P0 (the output labelled known-incomplete PM timelines as
usable). Resolved in this audit pass.

`mode1_single_buffer` records 390 and `mode3_chunk_ring` records 187
`profiler__pmsampler_dropped_samples`. NVIDIA defines this metric as samples
dropped because the PM-sampling buffer was insufficient. Before the repair,
both report diagnoses rendered their PM timeline normally and
`pm_sampling_validity.usable` was true.

The cause is directly traceable:

- `sampling_validity.check_pm_sampling_validity` accepts only architecture,
  interval, duration and pass-group count; it cannot represent lost or merged
  samples.
- Both report parsing paths in `ncu_report_tools.py` call it without
  `profiler__pmsampler_dropped_samples` or
  `profiler__pmsampler_merged_samples`.
- `metric_catalog.py` recognizes interval and pass-group metrics but not either
  data-loss metric.

The repair added dropped, merged, and buffer-size metrics to the catalog;
non-zero drops now block PM timeline and phase conclusions, while merged samples
retain the timeline but block phase claims. `ncu_report_tools` withholds the
timeline when blocked. The synthetic report-reader regression and both real
reports now produce `usable=False` and `pm_sampling.available=False` for the
two dropped-sample variants. NVIDIA's shipped `PMSamplingData` rule was checked
for the existing interval/duration logic; the extra data-loss gate is based on
the profiler-metric definition, because the shipped rule does not check it.

## Resolved audit finding: replay semantics were not report-derived

Original severity: P1 (application/range replay reports could be described with
the wrong semantics). Resolved in this audit pass.

`source_correlation.analyze_pm_sampling` unconditionally returns
`span_covers_replays: True` and describes the span as Kernel Replay. This corpus
makes that statement true because it records memory backup and replay passes.
However, the same function receives no `MeasurementContext` or invocation
manifest, and `diagnose_ncu_report` likewise does not derive the replay mode
from report metadata. For an Application Replay or Range Replay report, the
output would make an unsupported collection-mode claim.

The repair adds a versioned `<report>.ncu-rep.collection.json` sidecar. The YAML
collector writes the effective `replay-mode`, `cache-control`, and
`clock-control` arguments from the final command, plus Application Replay
matching/strictness and Range Replay/graph options. `ncu-diagnose` accepts
`--collection-manifest` and otherwise discovers that sidecar automatically.
Without one, it infers `kernel` only when
`profiler__replayer_bytes_mem_backed_up` is non-zero; otherwise replay mode is
reported as unknown. The PM timeline wording now follows the verified mode and
does not call an unknown or application/range collection Kernel Replay. If a
sidecar claims a mode that contradicts direct report evidence, the diagnosis
sets replay mode to unknown and records the conflict rather than trusting either
source.

Range Replay and Application Range Replay now have a separate context rule:
they can preserve concurrent execution, but their counters describe the range
and cannot be attributed to one kernel. The former blanket `serialized` NCU
context would have hidden that distinction.

## Documentation integrity finding

Severity: P2 (the audit trail cannot be followed from the checked-in docs).

Resolved in this audit pass. `CAPABILITY_EVOLUTION.md` used to link to
`docs/research/01` through `07`, and several documents cited the absent
`tests/profiling/test_analysis_engine.py`. Those references now point at this
checked-in audit record and the actual distributed `tests/profiling/` suite.
The current final audit validation has 597 passing tests. This is a
traceability repair only; it does not establish uncollected metric support.

## Resolved audit finding: outdated clock-control default

The historical comments stated `--clock-control=base` as the Nsight Compute
default. Current Nsight Compute 2026.2.1 documents `boost` in both the
[CLI reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/) and
[release notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/). The
collection context and catalog comments now state the version boundary: 2026.1+
defaults to `boost`, while older releases used `base`. The historical reports'
invocation remains unknown because the `.ncu-rep` does not retain it; newly
collected YAML runs record the explicit final `clock-control` option in the
sidecar.
