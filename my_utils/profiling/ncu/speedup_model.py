# SPDX-License-Identifier: Apache-2.0
"""GPA-style speedup upper bounds for stall-backed findings.

Following GPA (CGO 2021), each stall finding gets an *upper bound* on the
whole-kernel speedup that removing its stall class could produce. The model is
deliberately simple and deliberately labelled:

* The closed stall stack gives each reason a fractional share of the
  issue-active stall cycles (the ``..._per_issue_active`` counters are average
  warp counts, so concurrent stalls are already split proportionally). Removing
  a class with share ``s`` rescales predicted kernel cycles to ``1 - s``, an
  upper-bound speedup of ``1 / (1 - s)``.
* The bound then has to respect ceilings the engine already computes. Removing
  a stall does not remove the work: no speedup can push a throughput unit past
  its speed-of-light, a device-memory stall cannot push achieved bandwidth past
  the DRAM roofline, and a latency-hiding fix (more warps in flight) is bounded
  by the occupancy headroom the occupancy-limiter data reports.

Honesty constraints, enforced here rather than left to the renderer:

* These are **upper bounds, not predictions**. Every ``speedup_basis`` sentence
  says so, and the markdown renderer repeats it.
* When the stall stack failed its closure check (the reasons do not sum to the
  reported warp latency within tolerance), **no bound is emitted** -- the
  shares would be fractions of a denominator the report cannot support -- and
  the basis states why instead.
* Only stall-backed findings (category ``stall_*``, produced from the closed
  stall stack) are ever annotated. A bound on anything else would not have a
  stall share behind it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .metric_catalog import STALL_REASONS

__all__ = [
    "CLOSURE_LOW",
    "CLOSURE_HIGH",
    "stall_stack_closure",
    "estimate_stall_speedup_bound",
    "annotate_stall_speedup_bounds",
]

# Closure tolerance, matching the gates `analyze_stalls` uses for its own
# closure findings: below CLOSURE_LOW the report under-accounts (stall reasons
# are missing), above CLOSURE_HIGH the disjoint states exceed their own total
# (cross-pass disagreement). Either way the shares are unreliable.
CLOSURE_LOW = 0.90
CLOSURE_HIGH = 1.02

# Stall buckets whose canonical fix is latency hiding (more warps or more ILP
# in flight while the wait completes). Only these get the occupancy-headroom
# ceiling: a synchronization stall is not fixed by more warps at all.
_LATENCY_HIDING_BUCKETS = frozenset({"device_memory", "shared_memory"})


def stall_stack_closure(stalls: Mapping[str, Any]) -> Tuple[bool, str]:
    """Whether the closed stall accounting is trustworthy enough to model.

    Returns ``(closed, reason_if_not)``. The reason is written to be shown to
    the reader in place of a bound.
    """
    explained = stalls.get("explained_share") if isinstance(stalls, Mapping) else None
    if explained is None:
        return False, (
            "No speedup bound: the report carries no warp-latency total, so the "
            "stall stack has no closure check and the shares have no denominator."
        )
    if explained < CLOSURE_LOW:
        return False, (
            f"No speedup bound: the stall stack failed its closure check -- the "
            f"stall reasons present account for only {explained * 100:.0f}% of "
            "warp latency, so a share-based bound would be computed against a "
            "total the report cannot break down."
        )
    if explained > CLOSURE_HIGH:
        return False, (
            f"No speedup bound: the stall stack failed its closure check -- the "
            f"disjoint stall states sum to {explained * 100:.0f}% of the reported "
            "warp latency (two replay passes disagree), so every share is "
            "computed against a total its own components exceed."
        )
    return True, ""


def _sol_ceiling(view: Any) -> Optional[Tuple[str, float, float]]:
    """Ceiling from the busiest throughput unit: ``(unit, sol_pct, max_speedup)``.

    Removing a stall does not remove the instructions or the traffic, so the
    kernel cannot finish faster than its busiest unit's active time allows:
    speedup <= 100 / SOL%.
    """
    candidates = [
        ("SM", view.get("compute_sol")),
        ("memory system", view.get("memory_sol")),
    ]
    present = [(name, v) for name, v in candidates if v is not None and v > 0]
    if not present:
        return None
    name, sol = max(present, key=lambda kv: kv[1])
    return name, float(sol), 100.0 / float(sol)


def _dram_roofline_ceiling(
    view: Any, gpu_spec: Any
) -> Optional[Tuple[float, float, float]]:
    """DRAM-roofline ceiling: ``(achieved_gbps, peak_gbps, max_speedup)``.

    A kernel still has to move the same DRAM bytes after its memory stalls are
    gone, so it cannot speed up past the point where achieved bandwidth reaches
    the DRAM roofline.
    """
    peak = getattr(gpu_spec, "hbm_bandwidth_gbps", None) if gpu_spec else None
    if not peak:
        return None
    duration_ns = view.get("duration_ns")
    dram_bytes = view.get("dram_bytes")
    if dram_bytes is None:
        read = view.get("dram_bytes_read")
        write = view.get("dram_bytes_write")
        if read is not None or write is not None:
            dram_bytes = (read or 0.0) + (write or 0.0)
    if not dram_bytes or not duration_ns:
        return None
    achieved_gbps = dram_bytes / (duration_ns * 1e-9) / 1e9
    if achieved_gbps <= 0:
        return None
    return achieved_gbps, float(peak), float(peak) / achieved_gbps


def _occupancy_headroom_ceiling(
    occupancy: Optional[Mapping[str, Any]],
) -> Optional[Tuple[float, float, float]]:
    """Occupancy-headroom ceiling: ``(achieved_pct, theoretical_pct, max_speedup)``.

    Latency-hiding gains come from more warps in flight, and the
    occupancy-limiter data caps how many more warps there can be. Not applied
    when the occupancy model itself does not apply (warp-specialized kernels).
    """
    if not isinstance(occupancy, Mapping):
        return None
    if occupancy.get("occupancy_model_applicable") is False:
        return None
    achieved = occupancy.get("achieved_occupancy_pct")
    theoretical = occupancy.get("theoretical_occupancy_pct")
    if not achieved or not theoretical or achieved <= 0:
        return None
    if theoretical <= achieved:
        return float(achieved), float(theoretical), 1.0
    return float(achieved), float(theoretical), float(theoretical) / float(achieved)


def estimate_stall_speedup_bound(
    share: float,
    *,
    stall_key: str,
    view: Any,
    gpu_spec: Any = None,
    occupancy: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[float], str]:
    """Upper-bound speedup for removing one stall class entirely.

    ``share`` is the class's fraction of warp cycles per issued instruction
    from the closed stall stack. Returns ``(bound, basis_sentence)``; the bound
    is ``None`` only when the share itself is unusable.
    """
    if share is None or share <= 0.0 or share >= 1.0:
        return None, (
            "No speedup bound: the stall share is outside (0, 1), which the "
            "share-removal model cannot represent."
        )

    raw = 1.0 / (1.0 - share)
    bound = raw
    binding = (
        f"removing this stall's {share * 100:.0f}% share of issue-active stall "
        f"cycles rescales predicted kernel cycles to at best 1/(1-{share:.2f}) = "
        f"{raw:.2f}x"
    )

    reason = STALL_REASONS.get(stall_key)
    bucket = reason.bucket if reason else ""

    sol = _sol_ceiling(view)
    if sol is not None:
        unit, sol_pct, cap = sol
        if cap < bound:
            bound = cap
            binding = (
                f"the share-removal model allows {raw:.2f}x but the {unit} is "
                f"already at {sol_pct:.0f}% of peak, capping any speedup at "
                f"{cap:.2f}x"
            )

    if bucket == "device_memory":
        roofline = _dram_roofline_ceiling(view, gpu_spec)
        if roofline is not None:
            achieved_gbps, peak_gbps, cap = roofline
            if cap < bound:
                bound = cap
                binding = (
                    f"the share-removal model allows {raw:.2f}x but achieved DRAM "
                    f"bandwidth ({achieved_gbps:.0f} GB/s) cannot exceed the "
                    f"{peak_gbps:.0f} GB/s roofline, capping the speedup at {cap:.2f}x"
                )

    if bucket in _LATENCY_HIDING_BUCKETS:
        headroom = _occupancy_headroom_ceiling(occupancy)
        if headroom is not None:
            achieved_pct, theoretical_pct, cap = headroom
            if cap < bound:
                bound = cap
                binding = (
                    f"the share-removal model allows {raw:.2f}x but this stall is "
                    f"hidden by adding warps, and occupancy headroom "
                    f"({achieved_pct:.0f}% achieved of {theoretical_pct:.0f}% "
                    f"theoretical) caps a latency-hiding gain at {cap:.2f}x"
                )

    # One sentence: the model, then its assumptions. The "upper bound, not a
    # prediction" label is carried by the field name and restated by the
    # markdown renderer next to the number.
    basis = (
        f"Share-removal model on the closed stall stack: {binding}; assumes the "
        "stall is entirely removable, the stack's fractional shares are exact, "
        "and no new limiter appears."
    )
    return round(bound, 2), basis


def annotate_stall_speedup_bounds(
    findings: Iterable[Any],
    *,
    stalls: Mapping[str, Any],
    view: Any,
    gpu_spec: Any = None,
    occupancy: Optional[Mapping[str, Any]] = None,
) -> None:
    """Attach ``estimated_speedup_upper_bound`` / ``speedup_basis`` in place.

    Only findings whose category is ``stall_*`` -- the ones produced from the
    closed stall stack -- are ever annotated. When the stack failed its closure
    check, the bound is withheld and the basis says why.
    """
    closed, why_not = stall_stack_closure(stalls or {})
    for finding in findings:
        category = str(getattr(finding, "category", "") or "")
        if not category.startswith("stall_"):
            # Never emit a bound for findings that are not stall-backed.
            continue
        if not closed:
            finding.estimated_speedup_upper_bound = None
            finding.speedup_basis = why_not
            continue
        evidence: Dict[str, Any] = getattr(finding, "evidence", None) or {}
        share = evidence.get("share_of_warp_latency")
        stall_key = category[len("stall_") :]
        bound, basis = estimate_stall_speedup_bound(
            share if share is None else float(share),
            stall_key=stall_key,
            view=view,
            gpu_spec=gpu_spec,
            occupancy=occupancy,
        )
        finding.estimated_speedup_upper_bound = bound
        finding.speedup_basis = basis
