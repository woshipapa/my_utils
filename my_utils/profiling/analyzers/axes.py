"""The canonical performance axes, and what it takes to have analysed each one.

Every part of this package produced its own vocabulary: our rule engine emits
``uncoalesced_global_access``, Nsight Compute's shipped rules emit
``UncoalescedGlobalAccess``, the nsys side talks about ``memcpy_bound``. Three
names for one axis means cross-checking silently fails to match, and a coverage
report cannot say which axes were actually examined.

This module is the single vocabulary all of them map onto. It exists to answer
two questions a performance report must be able to answer:

1. **Did we look at this axis at all?** Not "did we find a problem" -- an axis
   we never examined and an axis we examined and found clean look identical in a
   findings list, and they are not the same thing. Silence must be attributable.
2. **What would it take to look?** Each axis names the metrics or trace tables
   it needs, so a report can tell the user exactly which collection flag would
   close the gap instead of just reporting an absence.

The axis set is deliberately coarse. Finer distinctions belong in the finding
categories; the axes are the top-level checklist a reviewer runs down to decide
whether an analysis was complete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Axis",
    "AXES",
    "AXIS_IDS",
    "axis_for_category",
    "axis_for_shipped_rule",
    "axis_coverage",
    "summarize_axis_coverage",
]


@dataclass(frozen=True)
class Axis:
    """One top-level dimension of performance analysis."""

    axis_id: str
    title: str
    question: str
    # Finding categories (ours, and nsys's) that land on this axis.
    categories: Tuple[str, ...] = ()
    # Normalised substrings of Nsight Compute shipped rule identifiers/sections.
    shipped_rule_hints: Tuple[str, ...] = ()
    # Any one of these metric groups is enough to examine the axis. Each group
    # is a tuple of catalog keys; a group is satisfied when all of its keys are
    # present. Empty means the axis is not driven by ncu metrics.
    metric_groups: Tuple[Tuple[str, ...], ...] = ()
    # What to collect when the axis is uncovered.
    remedy: str = ""
    # Where the analysis lives, so a reader can go check it.
    implemented_by: Tuple[str, ...] = ()


AXES: Tuple[Axis, ...] = (
    Axis(
        axis_id="compute",
        title="Compute throughput",
        question="Are the math pipes the limit, and is the right pipe being used?",
        categories=(
            "bottleneck", "compute_bound_kernel_below_expectation", "pipe_saturated",
            "tensor_cores_idle", "unexpected_fp64", "below_roofline",
            "roofline_needs_tensor_counters",
        ),
        shipped_rule_hints=("solbottleneck", "speedoflight", "sol", "compute", "pipe",
                            "tensor", "fp64", "fp16", "roofline", "fproofline"),
        metric_groups=(
            ("compute_sol",),
            ("pipe_tensor_util", "pipe_fma_util"),
            ("flop_ffma", "duration_ns"),
        ),
        remedy="--section SpeedOfLight --section ComputeWorkloadAnalysis",
        implemented_by=("ncu_diagnostics.classify_bottleneck",
                        "ncu_diagnostics.analyze_pipes",
                        "ncu_diagnostics.compute_roofline"),
    ),
    Axis(
        axis_id="memory_bandwidth",
        title="Memory bandwidth and cache behaviour",
        question="Is DRAM, L2 or L1 the limit, and is the traffic avoidable?",
        categories=(
            "memory_bound_kernel_below_expectation", "uncoalesced_global_access",
            "poor_cache_locality", "memory", "memcpy_bound",
        ),
        shipped_rule_hints=("memory", "dram", "l2", "l1", "tex", "uncoalesced",
                            "coalesc", "aperture", "sector", "cache"),
        metric_groups=(
            ("memory_sol",),
            ("dram_sol",),
            ("l2_hit_rate", "l1_hit_rate"),
            ("dram_bytes", "duration_ns"),
        ),
        remedy="--section MemoryWorkloadAnalysis --section SpeedOfLight",
        implemented_by=("ncu_diagnostics.classify_bottleneck",
                        "ncu_diagnostics.analyze_coalescing",
                        "ncu_diagnostics.compute_roofline"),
    ),
    Axis(
        axis_id="shared_memory",
        title="Shared memory and bank conflicts",
        question="Is shared memory serialising on bank conflicts?",
        categories=("shared_memory", "bank_conflicts"),
        shipped_rule_hints=("shared", "bankconflict", "conflict"),
        metric_groups=(
            ("shared_bank_conflicts_ld", "shared_bank_conflicts_st"),
            ("shared_pipe_util",),
        ),
        remedy="--section MemoryWorkloadAnalysis_Tables",
        implemented_by=("ncu_diagnostics.analyze_shared_memory",),
    ),
    Axis(
        axis_id="scheduler",
        title="Scheduler, occupancy and quantisation",
        question="Do the schedulers have warps to issue, and is the grid the right shape?",
        categories=(
            "occupancy_achieved_gap", "small_grid", "tail_wave_quantization",
            "block_size_not_warp_multiple", "occupancy", "launch_config",
            "tile_quantization", "wave_quantization", "imbalance",
        ),
        shipped_rule_hints=("occupancy", "launch", "gridsize", "wave", "tail",
                            "issueslot", "scheduler", "imbalance", "balance"),
        metric_groups=(
            ("achieved_occupancy", "theoretical_occupancy"),
            ("warps_active_per_scheduler", "warps_eligible_per_scheduler"),
            ("grid_size", "block_size"),
        ),
        remedy="--section Occupancy --section LaunchStats --section SchedulerStats",
        implemented_by=("ncu_diagnostics.analyze_occupancy",
                        "ncu_diagnostics.analyze_launch_config",
                        "ncu_diagnostics.analyze_imbalance"),
    ),
    Axis(
        axis_id="stall",
        title="Warp stall reasons",
        question="When warps cannot issue, what are they waiting on?",
        categories=("stalls", "stall_reason"),
        shipped_rule_hints=("stall", "cpistall", "warpstate"),
        metric_groups=(("warp_cycles_per_issued_inst",), ("issue_active",)),
        remedy="--section WarpStateStats",
        implemented_by=("ncu_diagnostics.analyze_stalls",),
    ),
    Axis(
        axis_id="divergence",
        title="Branch divergence and predication",
        question="Are threads in a warp doing the same work?",
        categories=("thread_divergence", "divergence", "predication"),
        shipped_rule_hints=("divergen", "branch", "predicat"),
        metric_groups=(("warp_exec_efficiency",), ("branch_efficiency",)),
        remedy="--section InstructionStats --section SourceCounters",
        implemented_by=("ncu_diagnostics.analyze_divergence",),
    ),
    Axis(
        axis_id="registers",
        title="Registers and local-memory spilling",
        question="Is register pressure pushing traffic into local memory?",
        categories=("register_spilling", "spilling", "local_memory"),
        shipped_rule_hints=("spill", "register", "localmemory"),
        metric_groups=(("registers_per_thread",), ("local_ld_inst", "local_st_inst")),
        remedy="--section LaunchStats --section MemoryWorkloadAnalysis_Tables, "
               "and build with -Xptxas -v",
        implemented_by=("ncu_diagnostics.analyze_spilling",),
    ),
    Axis(
        axis_id="communication",
        title="Collective communication",
        question="Are collectives at achievable bus bandwidth, and who is late?",
        categories=(
            "communication", "nccl", "collective_bandwidth", "straggler",
            "comm_bound", "comm_exposed",
        ),
        shipped_rule_hints=("nvlink", "pcie", "systemmemory", "interconnect"),
        metric_groups=(),   # nsys-side, not an ncu per-kernel axis
        remedy="nsys profile --trace=cuda,nvtx,nccl (and NCCL flight recorder "
               "for entry timestamps)",
        implemented_by=("analyzers.nccl_bandwidth", "analyzers.distributed_alignment"),
    ),
    Axis(
        axis_id="latency_launch",
        title="Launch overhead and gaps",
        question="Is the GPU idle waiting for the host to launch work?",
        categories=(
            "launch_overhead", "launch_storm", "gpu_idle", "host_bound",
            "small_kernel", "cuda_graph",
        ),
        shipped_rule_hints=(),
        metric_groups=(),   # needs a timeline, not per-kernel counters
        remedy="nsys profile --trace=cuda,osrt,nvtx",
        implemented_by=("analyzers.triage", "sources.nsys_auto_analysis"),
    ),
    Axis(
        axis_id="host_pipeline",
        title="Host-side pipeline",
        question="Is the input pipeline or Python the limit rather than the GPU?",
        categories=("dataloader", "host_pipeline", "h2d", "d2h", "pageable_memcpy",
                    "sync_blocking"),
        shipped_rule_hints=(),
        metric_groups=(),
        remedy="nsys profile --trace=cuda,osrt,nvtx --python-sampling=true",
        implemented_by=("analyzers.trace_quality.check_dataloader_attribution",
                        "sources.nsys_auto_analysis"),
    ),
    Axis(
        axis_id="power_clock",
        title="Clocks, power and thermals",
        question="Did the GPU run at the clock the numbers were normalised against?",
        categories=("throttling", "clock_throttle", "power_cap", "thermal"),
        shipped_rule_hints=(),
        # No ncu metric reports the clock the kernel actually ran at, so this
        # axis can only be examined from external telemetry. Declaring no metric
        # group is what makes it show up as an honest gap rather than as clean.
        metric_groups=(),
        remedy="Sample nvmlDeviceGetCurrentClocksEventReasons or DCGM fields "
               "100/112/155/240/241 during the run",
        implemented_by=("hardware.throttling.analyze_throttling",),
    ),
    Axis(
        axis_id="numerics",
        title="Precision and tensor-core eligibility",
        question="Is the kernel using the narrowest precision it is allowed to?",
        categories=("numerics", "precision", "dtype", "tensor_eligibility"),
        shipped_rule_hints=("fp16", "fp8", "tf32", "bf16", "sparsity", "precision"),
        metric_groups=(("pipe_tensor_util",), ("tensor_ops_fp16", "tensor_ops_bf16")),
        remedy="--section ComputeWorkloadAnalysis --section InstructionStats",
        implemented_by=("ncu_diagnostics._expectation_findings",
                        "sources.kernel_taxonomy"),
    ),
    Axis(
        axis_id="multi_gpu",
        title="Multi-GPU and multi-node skew",
        question="Are all ranks doing the same amount of work at the same time?",
        categories=("rank_skew", "multi_gpu", "clock_alignment", "mfu", "hfu"),
        shipped_rule_hints=(),
        metric_groups=(),
        remedy="Profile every rank, and align traces on a common clock before "
               "comparing across hosts",
        implemented_by=("analyzers.distributed_alignment",
                        "analyzers.trace_quality.check_clock_alignment"),
    ),
    Axis(
        axis_id="measurement",
        title="Measurement validity",
        question="Is this data trustworthy enough to draw any conclusion from?",
        categories=("measurement_caveat", "measurement_above_physical_limit",
                    "evidence_conflict", "uninformative_name", "unattributable_kernel",
                    "coverage", "trace_quality"),
        shipped_rule_hints=(),
        metric_groups=(),
        remedy="Always available: this axis is checked from whatever data exists",
        implemented_by=("analyzers.trace_quality", "analyzers.evidence",
                        "ncu_diagnostics.analysis_coverage"),
    ),
)

AXIS_IDS: Tuple[str, ...] = tuple(a.axis_id for a in AXES)

_CATEGORY_TO_AXIS: Dict[str, str] = {}
for _axis in AXES:
    for _cat in _axis.categories:
        _CATEGORY_TO_AXIS[_cat] = _axis.axis_id


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def axis_for_category(category: str) -> str:
    """Map a finding category onto an axis id, or "" when it belongs to none.

    Exact match first, then a substring fallback so a category we forgot to list
    still lands somewhere sensible rather than vanishing from the coverage
    report. An unmapped category is returned as "", and callers must surface it
    rather than dropping it.
    """
    key = str(category or "").strip()
    if key in _CATEGORY_TO_AXIS:
        return _CATEGORY_TO_AXIS[key]
    blob = _norm(key)
    if not blob:
        return ""
    for axis in AXES:
        for cat in axis.categories:
            normalized = _norm(cat)
            if normalized and (normalized in blob or blob in normalized):
                return axis.axis_id
    return ""


def axis_for_shipped_rule(*identifiers: Any) -> str:
    """Map a Nsight Compute shipped rule onto an axis id by substring match.

    Order matters: the more specific hints are checked before the generic ones
    so ``UncoalescedGlobalAccess`` lands on memory rather than on compute via a
    stray substring. Axes are declared in that order.
    """
    blob = " ".join(_norm(i) for i in identifiers if i)
    if not blob:
        return ""
    best: Tuple[int, str] = (0, "")
    for axis in AXES:
        for hint in axis.shipped_rule_hints:
            if hint and hint in blob and len(hint) > best[0]:
                best = (len(hint), axis.axis_id)
    return best[1]


@dataclass
class AxisStatus:
    """Whether one axis was examined, and what came of it."""

    axis_id: str
    title: str
    question: str
    examined: bool
    finding_count: int = 0
    categories_seen: Tuple[str, ...] = ()
    corroborated_by_ncu: bool = False
    reason_not_examined: str = ""
    remedy: str = ""
    implemented_by: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis": self.axis_id,
            "title": self.title,
            "question": self.question,
            "examined": self.examined,
            "finding_count": self.finding_count,
            "categories_seen": list(self.categories_seen),
            "corroborated_by_ncu": self.corroborated_by_ncu,
            "reason_not_examined": self.reason_not_examined,
            "remedy": self.remedy,
            "implemented_by": list(self.implemented_by),
        }


def axis_coverage(
    findings: Sequence[Any] = (),
    *,
    metric_present: Optional[Any] = None,
    shipped_rule_axes: Sequence[str] = (),
    axes_examined: Sequence[str] = (),
) -> Dict[str, Any]:
    """Report, per axis, whether it was examined and what it produced.

    ``findings`` are dicts or objects with a ``category``. ``metric_present`` is
    a predicate (or a container) answering whether a catalog key exists in the
    report; it decides whether a metric-driven axis *could* be examined.
    ``axes_examined`` lets a caller assert an axis was checked even though it
    produced nothing -- which is the whole point: a clean axis and an unchecked
    axis must not look the same.

    An axis with no findings is only reported as examined when either its
    metrics are present or the caller vouched for it. Anything else is reported
    as a gap with the collection flag that would close it.
    """
    present = _presence_predicate(metric_present)
    asserted = {str(a) for a in axes_examined}
    shipped_axes = {str(a) for a in shipped_rule_axes if a}

    by_axis: Dict[str, List[str]] = {a.axis_id: [] for a in AXES}
    unmapped: List[str] = []
    for finding in findings or ():
        category = _category_of(finding)
        axis_id = axis_for_category(category)
        if axis_id:
            by_axis[axis_id].append(category)
        elif category:
            unmapped.append(category)

    statuses: List[AxisStatus] = []
    for axis in AXES:
        categories = tuple(dict.fromkeys(by_axis[axis.axis_id]))
        has_findings = bool(categories)

        metrics_available: Optional[bool] = None
        if axis.metric_groups and present is not None:
            metrics_available = any(
                all(present(key) for key in group) for group in axis.metric_groups
            )

        examined = has_findings or axis.axis_id in asserted or bool(metrics_available)

        reason = ""
        if not examined:
            if axis.metric_groups and metrics_available is False:
                needed = " or ".join("+".join(g) for g in axis.metric_groups)
                reason = (
                    f"none of the metric groups this axis needs are in the report "
                    f"({needed})"
                )
            elif not axis.metric_groups:
                reason = (
                    "this axis is driven by a timeline or by system telemetry, "
                    "neither of which was supplied to this analysis"
                )
            else:
                reason = "no metrics were supplied, so presence could not be checked"

        statuses.append(AxisStatus(
            axis_id=axis.axis_id,
            title=axis.title,
            question=axis.question,
            examined=examined,
            finding_count=len(by_axis[axis.axis_id]),
            categories_seen=categories,
            corroborated_by_ncu=axis.axis_id in shipped_axes,
            reason_not_examined=reason,
            remedy="" if examined else axis.remedy,
            implemented_by=axis.implemented_by,
        ))

    examined_ids = [s.axis_id for s in statuses if s.examined]
    gaps = [s for s in statuses if not s.examined]

    return {
        "axes": [s.to_dict() for s in statuses],
        "examined": examined_ids,
        "not_examined": [s.axis_id for s in gaps],
        "examined_count": len(examined_ids),
        "axis_count": len(statuses),
        # A category we failed to map is a bug in this table, not a clean result.
        # Reporting it is how the table gets fixed.
        "unmapped_categories": sorted(set(unmapped)),
        "summary": summarize_axis_coverage(statuses),
    }


def summarize_axis_coverage(statuses: Sequence[AxisStatus]) -> str:
    """One sentence a reader can act on, naming the unexamined axes."""
    total = len(statuses)
    gaps = [s for s in statuses if not s.examined]
    if not gaps:
        return f"All {total} performance axes were examined."
    names = ", ".join(s.axis_id for s in gaps)
    return (
        f"{total - len(gaps)} of {total} axes examined. Not examined: {names}. "
        "Those axes produced no findings because they were never checked, which "
        "is not the same as being clean."
    )


def _category_of(finding: Any) -> str:
    if isinstance(finding, Mapping):
        return str(finding.get("category", "") or "")
    return str(getattr(finding, "category", "") or "")


def _presence_predicate(metric_present: Any):
    """Accept a callable, a container, or None and return a predicate or None."""
    if metric_present is None:
        return None
    if callable(metric_present):
        return metric_present
    # A MetricView exposes .get(); a set or dict supports __contains__.
    get = getattr(metric_present, "get", None)
    if callable(get) and not isinstance(metric_present, (set, frozenset)):
        return lambda key: get(key) is not None
    try:
        return lambda key: key in metric_present
    except TypeError:
        return None
