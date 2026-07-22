# SPDX-License-Identifier: Apache-2.0
"""Trace-validity checks: can this profile be trusted before it is analysed?

Every check here answers one question - *is the data good enough to draw the
conclusion the analyzer is about to draw?*  They run before the analysis, not
after, because a confident wrong answer is worse than a refusal.

The failure modes are all silent by construction, which is what makes them
dangerous:

* Triton autotuning burns seconds of unrepresentative GPU work into the first
  iteration, under the same kernel name as the tuned kernel.
* A CUDA graph replays N kernels under one ``cudaGraphLaunch``, so per-kernel
  host attribution simply does not exist - but the kernels are still there,
  looking attributable.
* ``TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0`` names every compiled kernel
  ``triton_``, so a per-kernel breakdown silently merges unrelated work.
* An nsys GPU-metrics gap labelled "Missing Data" is sampler-buffer exhaustion,
  not an idle GPU - and it looks exactly like the idle time you were hunting.
* A glob that matches 6 of 8 rank files produces a plausible per-rank aggregate
  computed over a biased sample, with the missing ranks being precisely the
  crashed or straggling ones.

Each check returns a :class:`QualityIssue` with a ``blocks`` flag: ``True``
means the analyzer should refuse the affected conclusion rather than caveat it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "QualityIssue",
    "check_warmup",
    "check_autotuning",
    "check_cuda_graphs",
    "check_kernel_name_uniqueness",
    "check_rank_completeness",
    "check_gpu_metric_gaps",
    "check_profiler_overhead",
    "assess_trace_quality",
    "check_multi_tenancy",
    "check_dataloader_attribution",
    "check_clock_alignment",
    "check_nvlink_utilization_validity",
    "check_diagnostic_events",
    "group_kernels_by_shape",
    "check_derived_metric_invariants",
    "DATALOADER_ATTRIBUTION_SQL",
]


@dataclass
class QualityIssue:
    """One reason to distrust part of a trace."""

    key: str
    title: str
    detail: str
    # What the analyzer must not conclude while this holds.
    invalidates: Tuple[str, ...] = ()
    blocks: bool = False  # True => refuse, False => caveat
    severity: str = "medium"  # info | low | medium | high
    evidence: Dict[str, Any] = field(default_factory=dict)
    remedy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "detail": self.detail,
            "invalidates": list(self.invalidates),
            "blocks": self.blocks,
            "severity": self.severity,
            "evidence": self.evidence,
            "remedy": self.remedy,
        }


# ---------------------------------------------------------------------------
# Warm-up and autotuning
# ---------------------------------------------------------------------------

MIN_STEADY_STATE_ITERATIONS = 3


def check_warmup(
    iteration_count: Optional[int], *, profiled_from_iteration: Optional[int] = None
) -> List[QualityIssue]:
    """Reject steady-state claims made from too few iterations.

    The first iterations carry CUDA context creation, library autotuning and
    allocator growth. None of that recurs, so timing them and calling the result
    "the step time" overstates it, sometimes by seconds.
    """
    issues: List[QualityIssue] = []
    if iteration_count is None:
        return issues
    if iteration_count < MIN_STEADY_STATE_ITERATIONS:
        issues.append(
            QualityIssue(
                key="insufficient_iterations",
                title=f"Only {iteration_count} iteration(s) captured",
                detail=(
                    f"Steady-state numbers need at least {MIN_STEADY_STATE_ITERATIONS} iterations. "
                    "The first ones include CUDA context init, library autotuning and allocator "
                    "growth, none of which happens again."
                ),
                invalidates=("step_time", "throughput", "mfu", "kernel_averages"),
                blocks=True,
                severity="high",
                evidence={"iteration_count": iteration_count},
                remedy="Profile a window starting around iteration 10 (cudaProfilerStart/Stop or --capture-range).",
            )
        )
    elif profiled_from_iteration is not None and profiled_from_iteration < 5:
        issues.append(
            QualityIssue(
                key="warmup_included",
                title=f"Capture starts at iteration {profiled_from_iteration}",
                detail="Early iterations are not representative of steady state.",
                invalidates=("step_time", "throughput"),
                severity="medium",
                evidence={"profiled_from_iteration": profiled_from_iteration},
                remedy="Start the capture around iteration 10.",
            )
        )
    return issues


# Distinct launch configurations of one kernel name, above which the run looks
# like an autotuning sweep rather than steady-state execution.
_AUTOTUNE_CONFIG_THRESHOLD = 5


def check_autotuning(
    launches: Sequence[Mapping[str, Any]],
    *,
    name_key: str = "kernel_name",
    config_keys: Sequence[str] = ("grid_size", "block_size", "shared_mem"),
) -> List[QualityIssue]:
    """Detect a Triton/library autotuning sweep hiding inside the trace.

    Triton's autotuner benchmarks every surviving config on a cache miss, with
    ``do_bench`` defaulting to 25 ms of warm-up plus 100 ms of measurement *per
    config*. Twenty configs is therefore about 2.5 s of real GPU work, all
    recorded under one kernel name. Averaging over it produces a number that
    describes none of the variants.
    """
    by_name: Dict[str, set] = {}
    for launch in launches:
        name = str(launch.get(name_key) or "")
        if not name:
            continue
        config = tuple(launch.get(k) for k in config_keys)
        by_name.setdefault(name, set()).add(config)

    suspects = {
        name: len(cfgs)
        for name, cfgs in by_name.items()
        if len(cfgs) >= _AUTOTUNE_CONFIG_THRESHOLD
    }
    if not suspects:
        return []

    worst = max(suspects, key=suspects.get)
    return [
        QualityIssue(
            key="autotuning_in_trace",
            title=f"{len(suspects)} kernel name(s) ran under many launch configurations",
            detail=(
                f"'{worst[:60]}' appears with {suspects[worst]} distinct launch configurations. "
                "That is the signature of an autotuning sweep: the same name covers genuinely "
                "different compiled variants, so any average over it describes none of them."
            ),
            invalidates=("kernel_averages", "kernel_ranking"),
            severity="high",
            evidence={"kernels_with_many_configs": suspects},
            remedy=(
                "Profile after autotuning has settled, key kernels on (name, grid, block, "
                "num_warps) rather than name alone, or check TRITON_PRINT_AUTOTUNING output."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# CUDA graphs
# ---------------------------------------------------------------------------


def check_cuda_graphs(
    *,
    graph_launch_count: int = 0,
    graph_kernel_count: int = 0,
    total_kernel_count: int = 0,
) -> List[QualityIssue]:
    """Flag that per-kernel host attribution is unavailable under graph replay.

    A graph replay submits every kernel in the graph with a single
    ``cudaGraphLaunch``, and CUPTI gives all of those kernels that one call's
    correlation id. A launch-to-kernel join that assumes 1:1 silently becomes
    1:N, so "CPU launch overhead per kernel" and "gap before this kernel" stop
    meaning anything for graph-internal work.
    """
    if not graph_launch_count and not graph_kernel_count:
        return []
    share = (graph_kernel_count / total_kernel_count) if total_kernel_count else None
    return [
        QualityIssue(
            key="cuda_graph_attribution",
            title="CUDA graph replay detected: per-kernel host attribution unavailable",
            detail=(
                f"{graph_kernel_count} kernel(s) executed inside CUDA graphs"
                + (f" ({share * 100:.0f}% of all kernels)" if share else "")
                + f" from {graph_launch_count} graph launch(es). Every kernel in a graph shares the "
                "single cudaGraphLaunch correlation id, so launch overhead and launch delay cannot "
                "be attributed per kernel. This is the point of graphs, not a defect."
            ),
            invalidates=(
                "per_kernel_launch_overhead",
                "per_kernel_launch_delay",
                "launch_gap_attribution",
            ),
            blocks=False,
            severity="medium",
            evidence={
                "graph_launch_count": graph_launch_count,
                "graph_kernel_count": graph_kernel_count,
                "graph_kernel_share": share,
            },
            remedy=(
                "Attribute by (graph id, node id) instead of by launching call. Collect with "
                "--cuda-graph-trace=node to see individual nodes, accepting the extra overhead."
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Kernel-name uniqueness
# ---------------------------------------------------------------------------

_BARE_TRITON = re.compile(r"^triton_?$")


def check_kernel_name_uniqueness(kernel_names: Iterable[str]) -> List[QualityIssue]:
    """Detect Inductor running with non-unique kernel names.

    ``TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0`` names every generated kernel
    ``triton_`` to maximise compilation caching. A per-kernel breakdown then
    merges unrelated fused regions under one row.
    """
    names = [str(n or "") for n in kernel_names]
    bare = [n for n in names if _BARE_TRITON.match(n.strip())]
    if len(bare) <= 1:
        return []
    return [
        QualityIssue(
            key="non_unique_kernel_names",
            title=f"{len(bare)} kernels all named 'triton_'",
            detail=(
                "TorchInductor was built with TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0, so every "
                "generated kernel carries the same name. Any breakdown keyed on kernel name is "
                "merging unrelated fused regions."
            ),
            invalidates=("kernel_ranking", "kernel_averages", "name_based_attribution"),
            blocks=True,
            severity="high",
            evidence={"bare_triton_kernels": len(bare)},
            remedy="Re-profile with TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1.",
        )
    ]


# ---------------------------------------------------------------------------
# Multi-rank completeness
# ---------------------------------------------------------------------------

_RANK_PATTERNS = (
    re.compile(r"rank[_\-]?(\d+)", re.IGNORECASE),
    re.compile(r"_(\d+)\.(?:nsys-rep|sqlite|json|gz)$", re.IGNORECASE),
)


def _rank_of(path: str) -> Optional[int]:
    for pattern in _RANK_PATTERNS:
        match = pattern.search(str(path))
        if match:
            return int(match.group(1))
    return None


def check_rank_completeness(
    report_paths: Sequence[str],
    *,
    expected_world_size: Optional[int] = None,
) -> List[QualityIssue]:
    """Refuse cross-rank aggregates when the rank set has holes.

    A glob that matches 6 of 8 rank files produces a per-rank heatmap with 6
    rows and no warning. The missing ranks are disproportionately the crashed or
    straggling ones, which is exactly what the aggregate was meant to find.
    """
    if not report_paths:
        return []
    ranks = sorted({r for r in (_rank_of(p) for p in report_paths) if r is not None})
    if not ranks:
        return [
            QualityIssue(
                key="rank_ids_unrecoverable",
                title="Could not identify rank ids from the report filenames",
                detail=(
                    f"{len(report_paths)} report(s) supplied but none carries a recognisable rank "
                    "id. Per-rank conclusions cannot be labelled reliably."
                ),
                invalidates=("straggler_detection", "per_rank_comparison"),
                severity="medium",
                evidence={"report_count": len(report_paths)},
                remedy="Name reports with the rank, e.g. -o report_%q{SLURM_PROCID}.",
            )
        ]

    expected = expected_world_size or (max(ranks) + 1)
    missing = [r for r in range(expected) if r not in ranks]
    if not missing:
        return []
    return [
        QualityIssue(
            key="incomplete_rank_set",
            title=f"{len(missing)} of {expected} ranks missing from the trace set",
            detail=(
                f"Ranks present: {len(ranks)}; missing: {missing[:12]}"
                + ("..." if len(missing) > 12 else "")
                + ". Aggregates over an incomplete rank set are biased, and a rank that failed to "
                "produce a report is more likely than average to be the one that was slow."
            ),
            invalidates=("straggler_detection", "per_rank_aggregate", "comm_analysis"),
            blocks=True,
            severity="high",
            evidence={
                "ranks_present": ranks,
                "ranks_missing": missing,
                "expected_world_size": expected,
            },
            remedy="Collect every rank, or state explicitly that the analysis covers a subset.",
        )
    ]


# ---------------------------------------------------------------------------
# nsys data quality
# ---------------------------------------------------------------------------

# nsys surfaces these in Diagnostics Summary rather than on stdout.
_DIAGNOSTIC_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
    (
        "buffer_overflow",
        r"buffer overflow",
        "GPU-metrics sampler buffer overflowed: timeline gaps are lost samples, not idle GPU.",
    ),
    (
        "trace_size_limit",
        r"size limit on recording trace events",
        "The trace hit its event-size limit, so events after that point are missing.",
    ),
    (
        "cupti_buffer",
        r"couldn'?t allocate cupti buf",
        "CUPTI could not allocate buffers; some CUDA events are missing.",
    ),
    (
        "event_order",
        r"wrong event order",
        "Event ordering broke; a large fraction of CUDA events may be absent.",
    ),
)


def check_gpu_metric_gaps(
    diagnostic_messages: Sequence[str] = (),
    *,
    missing_data_ranges: int = 0,
) -> List[QualityIssue]:
    """Interpret nsys diagnostics, especially gaps that are *not* idle GPU.

    A gap in the GPU-metrics rows labelled "Missing Data" means the sampler ran
    out of buffer, which happens more readily on big chips at high sampling
    frequency. Reading it as GPU idleness inverts the conclusion.
    """
    issues: List[QualityIssue] = []
    blob = " ".join(str(m).lower() for m in diagnostic_messages)

    for key, pattern, detail in _DIAGNOSTIC_PATTERNS:
        if re.search(pattern, blob):
            issues.append(
                QualityIssue(
                    key=f"nsys_{key}",
                    title=f"nsys reported a data-loss condition: {key.replace('_', ' ')}",
                    detail=detail,
                    invalidates=("idle_analysis", "kernel_totals", "gap_attribution"),
                    blocks=(key in ("trace_size_limit", "event_order")),
                    severity="high",
                    evidence={"matched": key},
                    remedy=(
                        "Lower --gpu-metrics-frequency, shorten the capture, or reduce traced "
                        "features, then re-collect."
                    ),
                )
            )

    if missing_data_ranges > 0:
        issues.append(
            QualityIssue(
                key="gpu_metrics_missing_data",
                title=f"{missing_data_ranges} 'Missing Data' range(s) in the GPU metrics",
                detail=(
                    "These are sampler-buffer exhaustion, not GPU idleness. This blocks rather than "
                    "caveats because reading them as idle time inverts the conclusion: the analyzer "
                    "would report a starved GPU that was in fact fully busy."
                ),
                invalidates=("idle_analysis", "gpu_utilization", "host_bound_verdict"),
                blocks=True,
                severity="high",
                evidence={"missing_data_ranges": missing_data_ranges},
                remedy="Reduce --gpu-metrics-frequency (default 10000 Hz) and re-collect.",
            )
        )
    return issues


def check_profiler_overhead(
    *,
    overhead_ns: Optional[float] = None,
    wall_ns: Optional[float] = None,
    threshold: float = 0.02,
) -> List[QualityIssue]:
    """Flag when the profiler's own overhead is large enough to distort gaps.

    CUPTI buffer flushes appear as GPU idle. NVIDIA's own gap rule subtracts
    them; an analyzer that does not will attribute its own instrumentation to
    the workload.
    """
    if not overhead_ns or not wall_ns:
        return []
    share = overhead_ns / wall_ns
    if share < threshold:
        return []
    return [
        QualityIssue(
            key="profiler_overhead_significant",
            title=f"Profiler overhead is {share * 100:.1f}% of the window",
            detail=(
                "CUPTI buffer flushes and instrumentation appear as GPU idle time. Subtract the "
                "overhead intervals before attributing idle time to the workload."
            ),
            invalidates=("idle_analysis", "gap_attribution"),
            severity="medium",
            evidence={
                "overhead_ns": overhead_ns,
                "wall_ns": wall_ns,
                "overhead_share": share,
            },
            remedy="Trace fewer features, or raise --cuda-flush-interval to batch flushes.",
        )
    ]


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def assess_trace_quality(
    *,
    iteration_count: Optional[int] = None,
    profiled_from_iteration: Optional[int] = None,
    launches: Sequence[Mapping[str, Any]] = (),
    kernel_names: Sequence[str] = (),
    graph_launch_count: int = 0,
    graph_kernel_count: int = 0,
    total_kernel_count: int = 0,
    report_paths: Sequence[str] = (),
    expected_world_size: Optional[int] = None,
    diagnostic_messages: Sequence[str] = (),
    missing_data_ranges: int = 0,
    overhead_ns: Optional[float] = None,
    wall_ns: Optional[float] = None,
) -> Dict[str, Any]:
    """Run every validity check and summarise what the trace cannot support.

    Returns ``blocked_conclusions``: the set of analyses the caller should
    refuse outright. Anything else is safe to report with the caveats attached.
    """
    issues: List[QualityIssue] = []
    issues += check_warmup(
        iteration_count, profiled_from_iteration=profiled_from_iteration
    )
    issues += check_autotuning(launches)
    issues += check_kernel_name_uniqueness(kernel_names)
    issues += check_cuda_graphs(
        graph_launch_count=graph_launch_count,
        graph_kernel_count=graph_kernel_count,
        total_kernel_count=total_kernel_count,
    )
    issues += check_rank_completeness(
        report_paths, expected_world_size=expected_world_size
    )
    issues += check_gpu_metric_gaps(
        diagnostic_messages, missing_data_ranges=missing_data_ranges
    )
    issues += check_profiler_overhead(overhead_ns=overhead_ns, wall_ns=wall_ns)

    blocked: set = set()
    for issue in issues:
        if issue.blocks:
            blocked.update(issue.invalidates)

    severity_rank = {"high": 0, "medium": 1, "low": 2, "info": 3}
    issues.sort(key=lambda i: (not i.blocks, severity_rank.get(i.severity, 9)))

    return {
        "trustworthy": not issues,
        "issue_count": len(issues),
        "blocking_count": sum(1 for i in issues if i.blocks),
        "blocked_conclusions": sorted(blocked),
        "issues": [i.to_dict() for i in issues],
    }


# ---------------------------------------------------------------------------
# Guards that only a trace can supply - a kernel name never reveals these
# ---------------------------------------------------------------------------

# Nsight Systems aligns reports from different hosts using UTC captured around
# collection start, and documents the error as "on the scale of one to tens of
# milliseconds" because it inherits NTP's precision. A typical AllReduce runs
# 0.1-5 ms, so the alignment error exceeds the thing being measured. TSC
# alignment (nanosecond precision) is selected automatically, but only for
# reports from the same target host.
UTC_ALIGNMENT_FLOOR_MS = 10.0


def check_clock_alignment(
    alignment_source: str = "",
    *,
    report_count: int = 1,
    claim_magnitude_ms: Optional[float] = None,
) -> List[QualityIssue]:
    """Refuse cross-node timing claims finer than the clock alignment supports.

    ``alignment_source`` is the ``Report alignment source`` property from the
    Analysis Summary tab: ``"TSC"`` (same host, nanosecond precision) or
    ``"UTC"`` (cross host, NTP precision).
    """
    if report_count < 2:
        return []
    source = str(alignment_source or "").strip().upper()
    if source.startswith("TSC"):
        return []

    detail = (
        "Reports are aligned by UTC wall-clock captured at collection start, whose "
        f"error is 1-10 ms. Claims finer than ~{UTC_ALIGNMENT_FLOOR_MS:.0f} ms across "
        "ranks are indistinguishable from clock skew."
    )
    blocks = (
        claim_magnitude_ms is not None and claim_magnitude_ms < UTC_ALIGNMENT_FLOOR_MS
    )
    if blocks:
        detail += (
            f" The pending claim is {claim_magnitude_ms:.2f} ms, which is below that floor: "
            "it is noise, not an observation."
        )
    return [
        QualityIssue(
            key="cross_node_clock_skew",
            title="Cross-node timestamps are NTP-aligned, not synchronised",
            detail=detail,
            invalidates=("straggler_rank", "arrival_skew", "cross_rank_latency"),
            blocks=blocks,
            severity="high" if blocks else "medium",
            evidence={
                "alignment_source": source or "unknown",
                "report_count": report_count,
                "floor_ms": UTC_ALIGNMENT_FLOOR_MS,
                "claim_ms": claim_magnitude_ms,
            },
            remedy=(
                "Anchor on a collective that is simultaneous by construction (match one "
                "AllReduce instance across ranks), derive per-rank offsets, and apply them "
                "with 'nsys export --ts-shift'. Otherwise keep conclusions within a rank. "
                "PTP gives sub-microsecond alignment; NTP does not."
            ),
        )
    ]


def check_nvlink_utilization_validity(
    nvlink_util_pct: Optional[float],
    *,
    links_active: Optional[bool] = None,
    nvlink_bytes: Optional[float] = None,
) -> List[QualityIssue]:
    """Reject the NVLink-saturated verdict when the links may simply be idle.

    NVIDIA documents this directly: "If metric sets with NVLink are used but the
    links are not active, they may appear as fully utilized." So the most obvious
    heuristic - high nvlrx/nvltx percent means communication-bound - inverts on
    exactly the machines where NVLink is absent or disabled.
    """
    if nvlink_util_pct is None or nvlink_util_pct < 90.0:
        return []
    moved_bytes = nvlink_bytes is not None and nvlink_bytes > 0
    if links_active is True or moved_bytes:
        return []
    return [
        QualityIssue(
            key="nvlink_util_ambiguous",
            title="NVLink reads ~100% but may be inactive rather than saturated",
            detail=(
                f"NVLink utilisation is {nvlink_util_pct:.1f}%, but inactive links report as "
                "fully utilised in Nsight Systems GPU metrics. Without a nonzero byte count "
                "or a topology check, saturated and absent are indistinguishable here."
            ),
            invalidates=("nvlink_saturated", "communication_bound"),
            blocks=True,
            severity="high",
            evidence={"nvlink_util_pct": nvlink_util_pct, "nvlink_bytes": nvlink_bytes},
            remedy=(
                "Confirm the links carry traffic before concluding saturation: check "
                "'nvidia-smi topo -p2p p' or 'nvidia-smi nvlink -s', or read the nvlink_sum "
                "recipe, which reports bytes rather than a percentage."
            ),
        )
    ]


def check_diagnostic_events(
    events: Iterable[Mapping[str, Any]] = (),
    *,
    log_text: str = "",
) -> List[QualityIssue]:
    """Gate every conclusion on the report's own diagnostics.

    Nsight Systems records collection-time problems in the ``DIAGNOSTIC_EVENT``
    table and surfaces them on the Diagnostics Summary page. A report that hit a
    buffer limit or dropped samples still loads and still looks complete, so this
    has to be read before the timeline, not after a result looks surprising.
    """
    issues: List[QualityIssue] = []
    serious = [
        row
        for row in (events or ())
        if isinstance(row, Mapping)
        and str(row.get("severity", "")).lower() in ("warning", "error", "fatal")
    ]

    # The literal strings the collector emits when it drops data.
    text = str(log_text or "")
    lost_markers = [
        marker
        for marker in (
            "were lost",
            "Reached the size limit on recording trace events",
            "throttled the collection of sampling data",
            "Buffer overflow",
        )
        if marker in text
    ]

    if serious:
        issues.append(
            QualityIssue(
                key="diagnostic_events",
                title=f"Report carries {len(serious)} collection diagnostic(s) at warning or above",
                detail=(
                    "The profiler recorded problems during collection. Timeline data may be "
                    "incomplete in ways that are not visually obvious: "
                    + "; ".join(
                        str(r.get("text") or r.get("message") or "?")[:90]
                        for r in serious[:3]
                    )
                ),
                invalidates=("any_timeline_conclusion",),
                blocks=True,
                severity="high",
                evidence={"count": len(serious)},
                remedy="Resolve the diagnostics and re-collect before drawing conclusions.",
            )
        )

    if lost_markers:
        issues.append(
            QualityIssue(
                key="dropped_events",
                title="Collector reported dropped events",
                detail=(
                    "The collection log contains "
                    + ", ".join(repr(m) for m in lost_markers)
                    + ". Any count, sum, or rate derived from this trace is a lower bound."
                ),
                invalidates=("kernel_counts", "api_counts", "utilization"),
                blocks=False,
                severity="high",
                evidence={"markers": lost_markers},
                remedy=(
                    "Reduce the sampling rate, narrow --trace, or shorten the capture window. "
                    "Captures beyond ~5 minutes are not officially supported."
                ),
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Multi-tenancy: MPS, MIG, vGPU
# ---------------------------------------------------------------------------

# Emitted verbatim by ncu when a metric needs a unit shared with another MIG
# instance. Greppable, so a collection log is enough to detect the condition.
MIG_SHARED_UNIT_ERROR = (
    "When profiling on a MIG instance, it is not possible to collect metrics "
    "from GPU units that are shared with other MIG instances"
)


def check_multi_tenancy(
    metrics: Optional[Mapping[str, Any]] = None,
    *,
    collection_log: str = "",
    mps_primary_client: Optional[bool] = None,
) -> List[QualityIssue]:
    """Detect tenancy modes that change what a measurement even means.

    Each of these is a documented reason a well-written kernel measures badly,
    or a reason an attribution cannot be made at all. They are invisible in the
    numbers themselves - only a launch attribute or a collection-log string
    reveals them - which is exactly why they get missed.
    """
    issues: List[QualityIssue] = []
    view = metrics or {}

    def _flag(name: str) -> bool:
        value = view.get(name)
        try:
            return value is not None and float(value) > 0
        except (TypeError, ValueError):
            return False

    if _flag("launch__uses_mps"):
        # NVIDIA is explicit that ncu "does generally not support isolating the
        # performance of individual clients" under MPS. Anything per-client is
        # therefore not a weak conclusion, it is an unavailable one.
        primary_only = mps_primary_client is False
        issues.append(
            QualityIssue(
                key="mps_shared_measurement",
                title="Kernel ran under MPS; per-client attribution is not available",
                detail=(
                    "Nsight Compute profiles how the GPU is utilised across all MPS clients "
                    "concurrently and does not isolate individual clients. Measured throughput "
                    "includes work from every co-resident client."
                    + (
                        " Instruction-level source and warp sampling are attributed to the "
                        "primary client only, and this kernel is not it."
                        if primary_only
                        else ""
                    )
                ),
                invalidates=(
                    "per_client_attribution",
                    "kernel_throughput",
                    "sol_classification",
                ),
                blocks=True,
                severity="high",
                evidence={"launch__uses_mps": view.get("launch__uses_mps")},
                remedy=(
                    "Profile the client alone, or accept whole-GPU attribution. If MPS must "
                    "stay on, use --replay-mode range (kernel mode lets each MPS client "
                    "contribute only a single launch) and --primary-client to narrow the window."
                ),
            )
        )

    if _flag("launch__uses_vgpu"):
        issues.append(
            QualityIssue(
                key="vgpu_counters_shared",
                title="Kernel ran on a vGPU; counters may include other VMs",
                detail=(
                    "Enabling profiling for a VM grants access to the GPU's global performance "
                    "counters, which may include activity from other VMs on the same physical "
                    "GPU. That VM can also lock clocks for everyone else."
                ),
                invalidates=(
                    "kernel_throughput",
                    "dram_bandwidth",
                    "sol_classification",
                ),
                blocks=False,
                severity="high",
                evidence={"launch__uses_vgpu": view.get("launch__uses_vgpu")},
                remedy="Confirm exclusive use of the physical GPU before trusting counter-derived numbers.",
            )
        )

    if _flag("launch__uses_nvlink_centric_scheduling"):
        # A documented reason a good kernel measures badly, in the same family as
        # green contexts: the denominator is the whole device but the kernel was
        # never given the whole device.
        issues.append(
            QualityIssue(
                key="nvlink_centric_scheduling",
                title="NVLink-centric scheduling was active; SM utilisation reads low by design",
                detail=(
                    "Some SM resources are not available to a workload under NVLink-centric "
                    "scheduling, which NVIDIA documents as producing lower-than-expected "
                    "measured utilisation. A low SM SOL here is not necessarily a defect."
                ),
                invalidates=("sm_utilization_verdict", "occupancy_verdict"),
                blocks=False,
                severity="medium",
                evidence={"launch__uses_nvlink_centric_scheduling": 1},
                remedy="Discount the SM SOL accordingly, or re-measure without NVLink-centric scheduling.",
            )
        )

    log = str(collection_log or "")
    if MIG_SHARED_UNIT_ERROR in log or "MIG instance" in log:
        issues.append(
            QualityIssue(
                key="mig_shared_units",
                title="MIG instance shares GPU units; some metrics could not be collected",
                detail=(
                    "Profiling on a shared Compute Instance cannot read units owned by other "
                    "MIG instances. Metrics from exclusively-owned units are still valid, so "
                    "this is a partial collection rather than a failed one - but any absent "
                    "metric here means 'not permitted', not 'zero'."
                ),
                invalidates=("dram_bandwidth", "l2_metrics"),
                blocks=False,
                severity="medium",
                evidence={"marker": MIG_SHARED_UNIT_ERROR},
                remedy=(
                    "Use an isolated Compute Instance for full coverage. Note also that ncu "
                    "cannot set clocks on any Compute Instance: pass --clock-control none and "
                    "lock externally with 'nvidia-smi --lock-gpu-clocks=tdp,tdp'."
                ),
            )
        )
    return issues


# ---------------------------------------------------------------------------
# Dataloader attribution
# ---------------------------------------------------------------------------

# PyTorch names its DataLoader worker and pin-memory threads at the OS level.
# Confirmed by reading the installed source, not by report:
#   worker.py     torch.multiprocessing._set_thread_name("pt_data_worker")
#   pin_memory.py torch.multiprocessing._set_thread_name("pt_data_pin")
# Line numbers move between releases, so match on the literal names.
#
# UNVERIFIED, and load-bearing for the SQL below: that nsys actually records
# these names in its ThreadNames table. The join is plausible - nsys does
# populate ThreadNames from OS thread names - but it has not been run against a
# real report here. Treat DATALOADER_ATTRIBUTION_SQL as a starting point to
# check, not as a result. The motivation is sound either way: torch.profiler
# cannot see into worker *processes* at all, so if nsys does capture them this
# is the only route to worker-side attribution.
DATALOADER_THREAD_PREFIXES = ("pt_data_worker", "pt_data_pin")

DATALOADER_ATTRIBUTION_SQL = """
-- Blocking time inside PyTorch DataLoader worker / pin-memory threads.
SELECT names.value AS thread_name,
       COUNT(*) AS call_count,
       SUM(osrt.end - osrt.start) / 1e6 AS blocked_ms
FROM OSRT_API AS osrt
JOIN ThreadNames AS names ON names.globalTid = osrt.globalTid
WHERE names.value LIKE 'pt_data_%'
GROUP BY names.value
ORDER BY blocked_ms DESC;
""".strip()


def check_dataloader_attribution(
    thread_names: Iterable[str] = (),
    *,
    gpu_idle_ms: Optional[float] = None,
    dataloader_blocked_ms: Optional[float] = None,
) -> List[QualityIssue]:
    """Say whether a GPU gap can be blamed on the dataloader, or only guessed at.

    Deliberately carries no prior about how often input pipelines are the cause.
    Figures of that kind circulate widely, but the ones this repo went looking
    for could not be traced to a source that was actually read, so none are
    encoded here. The check demands per-trace evidence instead: without worker
    threads in the trace, the dataloader-bound verdict is refused rather than
    assigned a default likelihood.
    """
    names = [str(n) for n in (thread_names or ())]
    have_worker_threads = any(n.startswith(DATALOADER_THREAD_PREFIXES) for n in names)

    if not have_worker_threads:
        if gpu_idle_ms and gpu_idle_ms > 0:
            return [
                QualityIssue(
                    key="dataloader_unattributable",
                    title="GPU idle time cannot be attributed to the dataloader",
                    detail=(
                        "No PyTorch DataLoader worker threads were found in the trace, so "
                        "worker-side blocking is unmeasured. Attributing idle time to input "
                        "pipeline here would be a guess."
                    ),
                    invalidates=("dataloader_bound",),
                    blocks=True,
                    severity="medium",
                    evidence={"gpu_idle_ms": gpu_idle_ms},
                    remedy=(
                        "Collect with --trace=osrt so worker threads appear, then join "
                        "ThreadNames.value LIKE 'pt_data_%' to OSRT_API by globalTid. "
                        "torch.profiler cannot see into worker processes; nsys can."
                    ),
                )
            ]
        return []

    if (
        gpu_idle_ms
        and dataloader_blocked_ms is not None
        and gpu_idle_ms > 0
        and dataloader_blocked_ms < 0.25 * gpu_idle_ms
    ):
        return [
            QualityIssue(
                key="dataloader_not_the_cause",
                title="Dataloader threads are present but are not explaining the idle time",
                detail=(
                    f"Worker threads blocked for {dataloader_blocked_ms:.0f} ms against "
                    f"{gpu_idle_ms:.0f} ms of GPU idle. The input pipeline accounts for a "
                    "minority of the gap; look elsewhere."
                ),
                invalidates=("dataloader_bound",),
                blocks=True,
                severity="medium",
                evidence={
                    "gpu_idle_ms": gpu_idle_ms,
                    "dataloader_blocked_ms": dataloader_blocked_ms,
                },
                remedy="Check host-side Python work, blocking syncs, and launch overhead instead.",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Kernel grouping and derived-metric invariants
# ---------------------------------------------------------------------------


def group_kernels_by_shape(
    launches: Iterable[Mapping[str, Any]],
    *,
    name_key: str = "kernel_name",
    shape_keys: Sequence[str] = ("grid_size", "block_size", "shared_mem", "dtype"),
    duration_key: str = "duration_ns",
) -> Dict[str, Any]:
    """Group launches by (name, shape, dtype) instead of by name alone.

    One kernel name covers genuinely different work. Grouping by name alone
    averages across shapes whose durations are not even monotonic in size: tile
    and wave quantisation mean a slightly larger N can take twice as long, so the
    mean describes no real configuration. Reports median, p10/p90 and min per
    group, and flags any group too dispersed to summarise with one number at all.
    """
    groups: Dict[Tuple, List[float]] = {}
    for launch in launches or ():
        if not isinstance(launch, Mapping):
            continue
        name = str(launch.get(name_key) or "")
        if not name:
            continue
        duration = launch.get(duration_key)
        if duration is None:
            continue
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            continue
        key = (name,) + tuple(launch.get(k) for k in shape_keys)
        groups.setdefault(key, []).append(duration)

    def _pct(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    summaries: List[Dict[str, Any]] = []
    non_stationary: List[str] = []
    for key, values in groups.items():
        values.sort()
        n = len(values)
        median = (
            values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
        )
        mean = sum(values) / n
        # Coefficient of variation over ~20%, or a max more than 3x the median,
        # means the group is not one population - reporting a single number for
        # it would be an average over unlike things.
        var = sum((v - mean) ** 2 for v in values) / n
        cv = (var**0.5 / mean) if mean else 0.0
        dispersed = n >= 3 and (cv > 0.20 or (median and values[-1] > 3 * median))
        summary = {
            "kernel": key[0][:120],
            "shape": dict(zip(shape_keys, key[1:])),
            "count": n,
            "median_ns": median,
            "min_ns": values[0],
            "p10_ns": _pct(values, 0.10),
            "p90_ns": _pct(values, 0.90),
            "max_ns": values[-1],
            "cv": round(cv, 3),
            "non_stationary": dispersed,
        }
        summaries.append(summary)
        if dispersed:
            non_stationary.append(key[0][:80])

    summaries.sort(key=lambda s: s["median_ns"] * s["count"], reverse=True)
    distinct_names = len({k[0] for k in groups})
    return {
        "groups": summaries,
        "group_count": len(groups),
        "distinct_names": distinct_names,
        "non_stationary": non_stationary,
        "note": (
            f"{len(groups)} (name, shape) groups across {distinct_names} distinct names. "
            "Grouping by name alone would have merged them."
            if len(groups) > distinct_names
            else ""
        ),
        "warning": (
            f"{len(non_stationary)} group(s) are too dispersed to summarise with a single "
            "number; use the median with p10/p90, or treat min as the achievable figure."
            if non_stationary
            else ""
        ),
    }


def check_derived_metric_invariants(
    *,
    mfu: Optional[float] = None,
    hfu: Optional[float] = None,
    efficiency_pct: Optional[float] = None,
    dtype: str = "",
    peak_is_sparse: Optional[bool] = None,
    activation_checkpointing: Optional[bool] = None,
) -> List[QualityIssue]:
    """Hard-fail the arithmetic identities a correct MFU/HFU computation obeys.

    A violated invariant means a wrong denominator, not an interesting result.
    The usual culprits each move the answer by a clean factor: a sparsity-inflated
    peak halves it, the wrong dtype's peak can quarter it, counting an FMA as one
    FLOP halves it again - and two such errors can cancel into a plausible number
    that is wrong twice.
    """
    issues: List[QualityIssue] = []

    for label, value in (("MFU", mfu), ("HFU", hfu), ("efficiency", efficiency_pct)):
        if value is not None and value > 100.0:
            issues.append(
                QualityIssue(
                    key=f"{label.lower()}_above_peak",
                    title=f"{label} exceeds 100% of peak",
                    detail=(
                        f"{label} computed as {value:.1f}%, which the hardware cannot do. This is a "
                        "denominator or FLOP-model error, not a result: check for a sparsity-inflated "
                        "peak, the wrong dtype's peak, or an FMA counted as one FLOP instead of two."
                    ),
                    invalidates=(label.lower(), "throughput_comparison"),
                    blocks=True,
                    severity="high",
                    evidence={label.lower(): value, "dtype": dtype},
                    remedy="Recompute against the dense peak for the dtype the kernel actually ran.",
                )
            )

    if mfu is not None and hfu is not None and hfu < mfu - 1e-6:
        issues.append(
            QualityIssue(
                key="hfu_below_mfu",
                title="HFU is below MFU, which is impossible",
                detail=(
                    f"HFU {hfu:.1f}% < MFU {mfu:.1f}%. Hardware FLOPs include every operation the "
                    "implementation performed, model FLOPs only those the model requires, so HFU is "
                    "always at least MFU. The two are being computed from different denominators."
                ),
                invalidates=("mfu", "hfu"),
                blocks=True,
                severity="high",
                evidence={"mfu": mfu, "hfu": hfu},
                remedy="Use one peak value and one FLOP convention for both.",
            )
        )

    if peak_is_sparse:
        issues.append(
            QualityIssue(
                key="sparse_peak_denominator",
                title="Efficiency computed against a sparsity-inflated peak",
                detail=(
                    "The denominator is the 2:4-sparse peak, which is double the dense figure and is "
                    "essentially never reached in production LLM work. Every efficiency number here "
                    "is half what it should be."
                ),
                invalidates=("mfu", "hfu", "pct_of_peak"),
                blocks=True,
                severity="high",
                evidence={"dtype": dtype},
                remedy="Use the dense peak. NVIDIA's spec tables print the sparse figure by default.",
            )
        )

    if activation_checkpointing and mfu is not None and hfu is None:
        issues.append(
            QualityIssue(
                key="mfu_without_recompute_model",
                title="MFU reported with activation checkpointing but no HFU",
                detail=(
                    "Recomputation performs extra forward passes that model FLOPs deliberately "
                    "exclude, so MFU alone hides real work the hardware did. Report HFU alongside it "
                    "or the two are not comparable across configurations."
                ),
                invalidates=("mfu",),
                blocks=False,
                severity="medium",
                evidence={"activation_checkpointing": True},
                remedy="Report HFU as a separate labelled field, with the recompute factor used.",
            )
        )

    if not dtype and (mfu is not None or efficiency_pct is not None):
        issues.append(
            QualityIssue(
                key="unknown_dtype_denominator",
                title="Efficiency computed without a known dtype",
                detail=(
                    "Peak FLOPs differ by up to 8x across bf16, fp8 and fp4, so an efficiency figure "
                    "without a stated dtype is not interpretable."
                ),
                invalidates=("mfu", "hfu", "pct_of_peak"),
                blocks=True,
                severity="high",
                evidence={},
                remedy="Determine the dominant dtype from the kernel mix, not from config.",
            )
        )
    return issues
