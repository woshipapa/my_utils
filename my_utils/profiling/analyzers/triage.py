"""Top-down triage: one verdict for where a training step's time actually goes.

The analyzers in this package each answer a narrow question.  This module runs
the ordered decision tree that experts follow, so a report can lead with a
single attribution instead of a list of independent observations:

    1. Is the GPU idle?            -> host / dataloader / launch bound
    2. Is communication exposed?   -> comm bound
    3. Are transfers on the path?  -> H2D/D2H bound
    4. Otherwise                   -> kernel bound, and here are the kernels

The thresholds come from NVIDIA's own calibrated numbers (the perf-analysis
skills shipped in TensorRT-LLM) and from PyTorch's Holistic Trace Analysis.
They are defaults, not laws: NVIDIA calibrated theirs on LLM *inference* and
says to loosen them where GPU time legitimately dominates, so every threshold
is overridable through :class:`TriageThresholds`.

The verdict is deliberately conservative.  A single metric crossing its
threshold is a signal; NVIDIA's own rule is that two or more must cross before
declaring a run host-bound, and that is reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "TriageThresholds",
    "TriageSignal",
    "TriageVerdict",
    "triage_step",
    "merge_intervals",
    "interval_union_ns",
    "interval_overlap_ns",
]


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

def merge_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping [start, end) intervals into a disjoint, sorted list."""
    clean = [(float(s), float(e)) for s, e in intervals if e is not None and s is not None and e > s]
    if not clean:
        return []
    clean.sort()
    merged: List[Tuple[float, float]] = [clean[0]]
    for start, end in clean[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            if end > last_end:
                merged[-1] = (last_start, end)
        else:
            merged.append((start, end))
    return merged


def interval_union_ns(intervals: Sequence[Tuple[float, float]]) -> float:
    """Total time covered by a set of intervals, counting overlaps once."""
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_overlap_ns(
    a: Sequence[Tuple[float, float]],
    b: Sequence[Tuple[float, float]],
) -> float:
    """Time during which both interval sets are simultaneously active.

    This is the primitive behind comm/compute overlap: NVIDIA's nsys recipes and
    HTA both define overlap as the intersection of the merged communication
    intervals with the merged compute intervals.
    """
    merged_a = merge_intervals(a)
    merged_b = merge_intervals(b)
    if not merged_a or not merged_b:
        return 0.0
    total = 0.0
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        start = max(merged_a[i][0], merged_b[j][0])
        end = min(merged_a[i][1], merged_b[j][1])
        if end > start:
            total += end - start
        if merged_a[i][1] < merged_b[j][1]:
            i += 1
        else:
            j += 1
    return total


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriageThresholds:
    """Tunable gates for the triage tree.

    Defaults follow NVIDIA's perf-analysis skill constants for the host-bound
    metrics and HTA for the trace-level ones.  ``cuda_graphs`` tightens the idle
    and utilisation gates, because a graph-launched step should have almost no
    launch gaps.
    """

    gpu_idle_ratio: float = 0.30            # M1: idle share above this is suspicious
    gpu_idle_ratio_graphs: float = 0.15     # ... when CUDA graphs are in use
    launch_overhead_ratio: float = 0.10     # M2: cudaLaunchKernel time / wall
    gpu_utilization_floor: float = 0.60     # M4
    gpu_utilization_floor_graphs: float = 0.80
    comm_ratio: float = 0.20                # M5: nccl time / gpu active
    host_bound_signals_required: int = 2    # NVIDIA's ">= 2 metrics crossing" rule

    exposed_comm_ratio: float = 0.15        # exposed comm / step time
    memcpy_ratio: float = 0.10              # H2D+D2H time / step time
    sync_ratio: float = 0.05                # blocking sync API time / wall

    # HTA constants
    short_kernel_us: float = 10.0           # kernels below this are launch-dominated
    short_kernel_share: float = 0.25        # ... and this share of launches trips a finding
    launch_delay_outlier_us: float = 100.0  # HTA launch_delay_cutoff
    runtime_outlier_us: float = 50.0        # HTA runtime_cutoff
    max_launch_queue: int = 1024            # HTA CUDA_MAX_LAUNCH_QUEUE_PER_STREAM

    # nsys expert-system defaults, tightened for ML steps (500 ms is far too
    # coarse to see a dataloader stall inside a 200 ms iteration).
    gap_threshold_us: float = 1000.0

    cuda_graphs: bool = False

    def idle_gate(self) -> float:
        return self.gpu_idle_ratio_graphs if self.cuda_graphs else self.gpu_idle_ratio

    def utilization_gate(self) -> float:
        return self.gpu_utilization_floor_graphs if self.cuda_graphs else self.gpu_utilization_floor


# ---------------------------------------------------------------------------
# Signals and verdict
# ---------------------------------------------------------------------------

@dataclass
class TriageSignal:
    """One measured quantity, its gate, and whether it crossed."""

    key: str
    label: str
    value: Optional[float]
    threshold: float
    crossed: bool
    direction: str          # "above" | "below"
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "threshold": self.threshold,
            "crossed": self.crossed,
            "direction": self.direction,
            "detail": self.detail,
        }


@dataclass
class TriageVerdict:
    """The single attribution, plus the evidence and what to do next."""

    verdict: str
    confidence: str
    summary: str
    signals: List[TriageSignal] = field(default_factory=list)
    next_steps: Tuple[str, ...] = ()
    breakdown: Dict[str, Any] = field(default_factory=dict)
    secondary: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "signals": [s.to_dict() for s in self.signals],
            "crossed_signals": [s.key for s in self.signals if s.crossed],
            "next_steps": list(self.next_steps),
            "breakdown": self.breakdown,
            "secondary": self.secondary,
        }


def _ratio(part: Optional[float], whole: Optional[float]) -> Optional[float]:
    if part is None or not whole:
        return None
    return float(part) / float(whole)


def triage_step(
    *,
    wall_ns: float,
    compute_intervals: Sequence[Tuple[float, float]] = (),
    comm_intervals: Sequence[Tuple[float, float]] = (),
    memcpy_intervals: Sequence[Tuple[float, float]] = (),
    launch_api_ns: Optional[float] = None,
    sync_api_ns: Optional[float] = None,
    kernel_durations_ns: Sequence[float] = (),
    launch_delays_ns: Sequence[float] = (),
    max_queue_length: Optional[int] = None,
    steady_state_allocs: Optional[int] = None,
    thresholds: Optional[TriageThresholds] = None,
) -> TriageVerdict:
    """Attribute one profiling window to a single dominant cause.

    ``compute_intervals``, ``comm_intervals`` and ``memcpy_intervals`` are
    ``(start_ns, end_ns)`` pairs from the nsys kernel/memcpy tables.  Anything
    not supplied is simply skipped: the verdict degrades to whatever evidence is
    available and says so in ``confidence``.
    """
    t = thresholds or TriageThresholds()

    gpu_intervals = list(compute_intervals) + list(comm_intervals) + list(memcpy_intervals)
    gpu_active_ns = interval_union_ns(gpu_intervals)
    compute_ns = interval_union_ns(compute_intervals)
    comm_ns = interval_union_ns(comm_intervals)
    memcpy_ns = interval_union_ns(memcpy_intervals)

    overlap_ns = interval_overlap_ns(comm_intervals, compute_intervals)
    exposed_comm_ns = max(0.0, comm_ns - overlap_ns)

    idle_ns = max(0.0, wall_ns - gpu_active_ns)
    idle_ratio = _ratio(idle_ns, wall_ns)
    utilization = _ratio(gpu_active_ns, wall_ns)
    launch_ratio = _ratio(launch_api_ns, wall_ns)
    sync_ratio = _ratio(sync_api_ns, wall_ns)
    comm_ratio = _ratio(comm_ns, gpu_active_ns)
    exposed_comm_ratio = _ratio(exposed_comm_ns, wall_ns)
    memcpy_ratio = _ratio(memcpy_ns, wall_ns)
    overlap_pct_of_comm = _ratio(overlap_ns, comm_ns)

    short_kernels = [d for d in kernel_durations_ns if d is not None and d < t.short_kernel_us * 1000]
    short_share = _ratio(len(short_kernels), len(kernel_durations_ns)) if kernel_durations_ns else None
    delay_outliers = [d for d in launch_delays_ns if d is not None and d > t.launch_delay_outlier_us * 1000]
    delay_outlier_share = _ratio(len(delay_outliers), len(launch_delays_ns)) if launch_delays_ns else None

    signals: List[TriageSignal] = [
        TriageSignal("gpu_idle_ratio", "GPU idle share of wall time",
                     idle_ratio, t.idle_gate(),
                     bool(idle_ratio is not None and idle_ratio > t.idle_gate()), "above",
                     "Time with no kernel, memcpy or collective resident."),
        TriageSignal("launch_overhead_ratio", "cudaLaunchKernel time / wall",
                     launch_ratio, t.launch_overhead_ratio,
                     bool(launch_ratio is not None and launch_ratio > t.launch_overhead_ratio), "above",
                     "Host time spent purely issuing work."),
        TriageSignal("gpu_utilization", "GPU busy share of wall time",
                     utilization, t.utilization_gate(),
                     bool(utilization is not None and utilization < t.utilization_gate()), "below",
                     "Fraction of the window with the GPU doing something."),
        TriageSignal("comm_ratio", "Collective time / GPU busy time",
                     comm_ratio, t.comm_ratio,
                     bool(comm_ratio is not None and comm_ratio > t.comm_ratio), "above",
                     "How much of the GPU's work is communication."),
        TriageSignal("exposed_comm_ratio", "Non-overlapped collective time / wall",
                     exposed_comm_ratio, t.exposed_comm_ratio,
                     bool(exposed_comm_ratio is not None and exposed_comm_ratio > t.exposed_comm_ratio),
                     "above", "Communication with no compute running underneath it."),
        TriageSignal("memcpy_ratio", "H2D/D2H time / wall",
                     memcpy_ratio, t.memcpy_ratio,
                     bool(memcpy_ratio is not None and memcpy_ratio > t.memcpy_ratio), "above",
                     "Host-device transfer time."),
        TriageSignal("sync_ratio", "Blocking sync API time / wall",
                     sync_ratio, t.sync_ratio,
                     bool(sync_ratio is not None and sync_ratio > t.sync_ratio), "above",
                     "cudaDeviceSynchronize / cudaStreamSynchronize / event syncs."),
        TriageSignal("short_kernel_share", "Share of kernels under the launch-overhead floor",
                     short_share, t.short_kernel_share,
                     bool(short_share is not None and short_share > t.short_kernel_share), "above",
                     f"Kernels shorter than {t.short_kernel_us:.0f} us cost more to launch than to run."),
    ]

    host_signal_keys = ("gpu_idle_ratio", "launch_overhead_ratio", "gpu_utilization")
    host_crossed = [s for s in signals if s.key in host_signal_keys and s.crossed]

    breakdown = {
        "wall_ns": wall_ns,
        "gpu_active_ns": gpu_active_ns,
        "idle_ns": idle_ns,
        "compute_ns": compute_ns,
        "comm_ns": comm_ns,
        "memcpy_ns": memcpy_ns,
        "comm_compute_overlap_ns": overlap_ns,
        "exposed_comm_ns": exposed_comm_ns,
        "overlap_pct_of_comm": (overlap_pct_of_comm * 100.0) if overlap_pct_of_comm is not None else None,
        "kernel_count": len(kernel_durations_ns) or None,
        "short_kernel_count": len(short_kernels) or None,
        "launch_delay_outliers": len(delay_outliers) or None,
        "launch_delay_outlier_share": delay_outlier_share,
    }

    secondary: List[str] = []
    if max_queue_length is not None and max_queue_length >= t.max_launch_queue:
        secondary.append(
            f"Launch queue reached {max_queue_length} (cap ~{t.max_launch_queue}); the host was "
            "blocked in launch calls, which means the GPU is the constraint, not the CPU."
        )
    if steady_state_allocs:
        secondary.append(
            f"{steady_state_allocs} cudaMalloc/cudaFree calls after warm-up: the caching allocator "
            "is missing (shape churn or fragmentation), and cudaFree synchronises the device."
        )
    if delay_outlier_share is not None and delay_outlier_share > 0.05:
        secondary.append(
            f"{delay_outlier_share * 100:.0f}% of launches waited over "
            f"{t.launch_delay_outlier_us:.0f} us before starting on the GPU."
        )

    # ---- the ordered decision tree -------------------------------------
    # An absent GPU timeline and a genuinely idle GPU are indistinguishable from
    # the arithmetic alone: both leave gpu_active_ns at zero, and the idle-share
    # signals then cross every threshold. Reaching "host_bound" from that is the
    # exact failure this module exists to prevent, so the whole idle-driven
    # branch is gated on having actually measured GPU activity.
    measured_gpu_activity = gpu_active_ns > 0
    if len(host_crossed) >= t.host_bound_signals_required and not measured_gpu_activity:
        return TriageVerdict(
            verdict="undetermined",
            confidence="low",
            summary=(
                "No GPU activity intervals were supplied, so GPU idle time could not be "
                "measured. The host-side signals that fired are derived from that missing "
                "measurement and would read identically for a fully busy GPU, so no verdict "
                "is issued."
            ),
            signals=signals,
            next_steps=(
                "Supply kernel and memcpy (start_ns, end_ns) intervals - from the nsys "
                "CUPTI_ACTIVITY_KIND_KERNEL and _MEMCPY tables rather than the summary "
                "rollups, which carry totals only.",
            ),
            breakdown=breakdown,
        )

    if len(host_crossed) >= t.host_bound_signals_required:
        crossed_names = ", ".join(s.label for s in host_crossed)
        sync_hint = ""
        if sync_ratio is not None and sync_ratio > t.sync_ratio:
            sync_hint = (
                f" Blocking sync calls account for {sync_ratio * 100:.0f}% of the window, so look "
                "for .item(), .cpu(), print(tensor) or data-dependent indexing in the step."
            )
        verdict = TriageVerdict(
            verdict="host_bound",
            confidence="high" if len(host_crossed) >= 3 else "medium",
            summary=(
                f"The GPU is idle {(idle_ratio or 0) * 100:.0f}% of the window and "
                f"{len(host_crossed)} host-side signals crossed their thresholds ({crossed_names}). "
                "The host cannot feed the GPU fast enough." + sync_hint
            ),
            signals=signals,
            next_steps=(
                "Check whether the idle gaps line up with iteration boundaries (dataloader) or are "
                "spread evenly between kernels (per-launch dispatch overhead).",
                "For dataloader stalls: raise num_workers, enable pin_memory plus non_blocking copies, "
                "and set prefetch_factor.",
                "For dispatch overhead: torch.compile with mode='reduce-overhead', or CUDA graphs.",
                "Confirm with nsys: near-zero queue time on every kernel means the GPU is starved.",
            ),
            breakdown=breakdown,
            secondary=secondary,
        )
        return verdict

    if exposed_comm_ratio is not None and exposed_comm_ratio > t.exposed_comm_ratio:
        overlap_text = (
            f"Only {overlap_pct_of_comm * 100:.0f}% of collective time overlaps compute."
            if overlap_pct_of_comm is not None else ""
        )
        return TriageVerdict(
            verdict="communication_bound",
            confidence="high" if (comm_ratio or 0) > t.comm_ratio else "medium",
            summary=(
                f"Collectives occupy {(comm_ratio or 0) * 100:.0f}% of GPU time and "
                f"{exposed_comm_ratio * 100:.0f}% of the window is communication with no compute "
                f"underneath it. {overlap_text}"
            ),
            signals=signals,
            next_steps=(
                "Enable the framework's overlap knobs: DDP bucketing, FSDP backward_prefetch, "
                "Megatron --overlap-grad-reduce / --overlap-param-gather / --tp-comm-overlap.",
                "Compute bus bandwidth per collective and compare against the interconnect ceiling: "
                "low busbw means topology or protocol, high busbw means you are simply moving too much.",
                "Rule out stragglers first - a slow rank stretches every rank's collective, and that "
                "looks identical to a slow network from inside one rank's trace.",
                "Watch for SM contention: overlapped NCCL steals SMs and can slow the compute it hides behind.",
            ),
            breakdown=breakdown,
            secondary=secondary,
        )

    if memcpy_ratio is not None and memcpy_ratio > t.memcpy_ratio:
        return TriageVerdict(
            verdict="transfer_bound",
            confidence="medium",
            summary=(
                f"Host-device transfers take {memcpy_ratio * 100:.0f}% of the window. "
                "Transfers on the critical path usually mean unpinned memory or no double buffering."
            ),
            signals=signals,
            next_steps=(
                "Use pinned host memory and non_blocking=True so copies are genuinely asynchronous.",
                "Overlap transfers with compute on a separate stream, or prefetch the next batch.",
                "Check nsys' cuda_memcpy_async rule: an async copy from pageable memory is secretly synchronous.",
            ),
            breakdown=breakdown,
            secondary=secondary,
        )

    if short_share is not None and short_share > t.short_kernel_share:
        return TriageVerdict(
            verdict="launch_bound",
            confidence="medium",
            summary=(
                f"{short_share * 100:.0f}% of kernels run for under {t.short_kernel_us:.0f} us. "
                "The GPU is busy, but mostly with kernels too small to amortise their own launch."
            ),
            signals=signals,
            next_steps=(
                "Fuse the elementwise chains: torch.compile turns them into triton_poi_fused_* kernels.",
                "Use fused/foreach optimizers so the step is not one kernel per parameter.",
                "Capture the steady-state region into a CUDA graph.",
            ),
            breakdown=breakdown,
            secondary=secondary,
        )

    have_evidence = gpu_active_ns > 0
    return TriageVerdict(
        verdict="kernel_bound",
        confidence="medium" if have_evidence else "low",
        summary=(
            f"The GPU is busy {(utilization or 0) * 100:.0f}% of the window with no dominant host, "
            "communication or transfer problem. Time is going into the kernels themselves, so the "
            "next step is per-kernel analysis."
        ),
        signals=signals,
        next_steps=(
            "Rank kernels by total GPU time and profile the top few with Nsight Compute.",
            "Use the speed-of-light classification to route each kernel: compute bound, memory bound, "
            "latency bound or launch bound.",
            "Grade each kernel against what its category should achieve, not against a single global target.",
        ),
        breakdown=breakdown,
        secondary=secondary,
    )
