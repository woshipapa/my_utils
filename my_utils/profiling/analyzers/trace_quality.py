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
]


@dataclass
class QualityIssue:
    """One reason to distrust part of a trace."""

    key: str
    title: str
    detail: str
    # What the analyzer must not conclude while this holds.
    invalidates: Tuple[str, ...] = ()
    blocks: bool = False          # True => refuse, False => caveat
    severity: str = "medium"      # info | low | medium | high
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


def check_warmup(iteration_count: Optional[int], *, profiled_from_iteration: Optional[int] = None) -> List[QualityIssue]:
    """Reject steady-state claims made from too few iterations.

    The first iterations carry CUDA context creation, library autotuning and
    allocator growth. None of that recurs, so timing them and calling the result
    "the step time" overstates it, sometimes by seconds.
    """
    issues: List[QualityIssue] = []
    if iteration_count is None:
        return issues
    if iteration_count < MIN_STEADY_STATE_ITERATIONS:
        issues.append(QualityIssue(
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
        ))
    elif profiled_from_iteration is not None and profiled_from_iteration < 5:
        issues.append(QualityIssue(
            key="warmup_included",
            title=f"Capture starts at iteration {profiled_from_iteration}",
            detail="Early iterations are not representative of steady state.",
            invalidates=("step_time", "throughput"),
            severity="medium",
            evidence={"profiled_from_iteration": profiled_from_iteration},
            remedy="Start the capture around iteration 10.",
        ))
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

    suspects = {name: len(cfgs) for name, cfgs in by_name.items() if len(cfgs) >= _AUTOTUNE_CONFIG_THRESHOLD}
    if not suspects:
        return []

    worst = max(suspects, key=suspects.get)
    return [QualityIssue(
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
    )]


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
    return [QualityIssue(
        key="cuda_graph_attribution",
        title="CUDA graph replay detected: per-kernel host attribution unavailable",
        detail=(
            f"{graph_kernel_count} kernel(s) executed inside CUDA graphs"
            + (f" ({share * 100:.0f}% of all kernels)" if share else "")
            + f" from {graph_launch_count} graph launch(es). Every kernel in a graph shares the "
            "single cudaGraphLaunch correlation id, so launch overhead and launch delay cannot "
            "be attributed per kernel. This is the point of graphs, not a defect."
        ),
        invalidates=("per_kernel_launch_overhead", "per_kernel_launch_delay", "launch_gap_attribution"),
        blocks=False,
        severity="medium",
        evidence={"graph_launch_count": graph_launch_count,
                  "graph_kernel_count": graph_kernel_count,
                  "graph_kernel_share": share},
        remedy=(
            "Attribute by (graph id, node id) instead of by launching call. Collect with "
            "--cuda-graph-trace=node to see individual nodes, accepting the extra overhead."
        ),
    )]


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
    return [QualityIssue(
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
    )]


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
        return [QualityIssue(
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
        )]

    expected = expected_world_size or (max(ranks) + 1)
    missing = [r for r in range(expected) if r not in ranks]
    if not missing:
        return []
    return [QualityIssue(
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
        evidence={"ranks_present": ranks, "ranks_missing": missing, "expected_world_size": expected},
        remedy="Collect every rank, or state explicitly that the analysis covers a subset.",
    )]


# ---------------------------------------------------------------------------
# nsys data quality
# ---------------------------------------------------------------------------

# nsys surfaces these in Diagnostics Summary rather than on stdout.
_DIAGNOSTIC_PATTERNS: Tuple[Tuple[str, str, str], ...] = (
    ("buffer_overflow", r"buffer overflow",
     "GPU-metrics sampler buffer overflowed: timeline gaps are lost samples, not idle GPU."),
    ("trace_size_limit", r"size limit on recording trace events",
     "The trace hit its event-size limit, so events after that point are missing."),
    ("cupti_buffer", r"couldn'?t allocate cupti buf",
     "CUPTI could not allocate buffers; some CUDA events are missing."),
    ("event_order", r"wrong event order",
     "Event ordering broke; a large fraction of CUDA events may be absent."),
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
            issues.append(QualityIssue(
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
            ))

    if missing_data_ranges > 0:
        issues.append(QualityIssue(
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
        ))
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
    return [QualityIssue(
        key="profiler_overhead_significant",
        title=f"Profiler overhead is {share * 100:.1f}% of the window",
        detail=(
            "CUPTI buffer flushes and instrumentation appear as GPU idle time. Subtract the "
            "overhead intervals before attributing idle time to the workload."
        ),
        invalidates=("idle_analysis", "gap_attribution"),
        severity="medium",
        evidence={"overhead_ns": overhead_ns, "wall_ns": wall_ns, "overhead_share": share},
        remedy="Trace fewer features, or raise --cuda-flush-interval to batch flushes.",
    )]


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
    issues += check_warmup(iteration_count, profiled_from_iteration=profiled_from_iteration)
    issues += check_autotuning(launches)
    issues += check_kernel_name_uniqueness(kernel_names)
    issues += check_cuda_graphs(
        graph_launch_count=graph_launch_count,
        graph_kernel_count=graph_kernel_count,
        total_kernel_count=total_kernel_count,
    )
    issues += check_rank_completeness(report_paths, expected_world_size=expected_world_size)
    issues += check_gpu_metric_gaps(diagnostic_messages, missing_data_ranges=missing_data_ranges)
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
