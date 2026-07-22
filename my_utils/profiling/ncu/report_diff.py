# SPDX-License-Identifier: Apache-2.0
"""Structured A/B comparison of two Nsight Compute reports.

The single-report pipeline in this package answers "what is wrong with this
kernel". This module answers the other question people profile for: "did my
change help, and what exactly moved". It is deliberately built on top of the
existing machinery rather than beside it:

* The clock-confound guard is
  :func:`~my_utils.profiling.analyzers.measurement_context.compare_measurements`,
  imported, not reimplemented. Two durations measured at SM clocks more than 1%
  apart are two different quantities; when that happens the diff refuses to
  present the raw-time delta as a speedup and leads with clock-normalised and
  elapsed-cycle figures instead.
* The "what changed the verdict" section diffs the FINDINGS of
  :func:`~my_utils.profiling.ncu.ncu_report_tools.diagnose_ncu_report` run on
  both reports -- findings that appeared, disappeared, or changed severity.
  "The bank-conflict finding disappeared and a barrier-stall finding appeared"
  is usually worth more than any single metric delta.
* Metric deltas are grouped by axis and use clock-independent quantities
  wherever one exists (counts, cycles per issue-slot, percentages), so the
  clock correction problem does not silently re-enter through a side door.

Honesty rules this module enforces:

* No causality claims. Two deltas moving together is correlation with the same
  code change, nothing more, and the output says so.
* Deltas smaller than measurement noise are reported as unchanged, not as
  improvements (rates and percentages below 2% relative movement are noise).
* Hit-rate deltas are traffic-weighted: a hit rate swinging on negligible
  traffic is labelled negligible instead of being severity-coded.
* Where stall accounting failed closure in either report, the stall-delta
  section is flagged unreliable rather than printed as if trustworthy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..analyzers.measurement_context import (
    compare_measurements,
    describe_collection_mode,
)
from .metric_catalog import STALL_REASONS
from .ncu_diagnostics import MetricView
from .ncu_report_tools import diagnose_ncu_report, walk_report_once

__all__ = ["diff_ncu_reports", "diff_result_to_markdown"]


# ---------------------------------------------------------------------------
# Noise thresholds. A diff that flags every third decimal place trains its
# reader to ignore it; these floors mark the moves worth a human's attention.
# ---------------------------------------------------------------------------

#: Relative movement below this on rates/percentages/counts is noise, not signal.
NOISE_REL = 0.02
#: Percentage-point metrics additionally need to move this many points.
PERCENT_POINT_FLOOR = 0.5
#: Cycles-per-issue-slot deltas below this are noise regardless of ratio.
STALL_CYCLES_FLOOR = 0.05
#: A hit rate whose underlying traffic is below this share of the level's
#: total sectors on both sides is not severity-coded, whatever it did.
NEGLIGIBLE_TRAFFIC_SHARE = 0.01

_SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3}

_NO_CAUSALITY_NOTE = (
    "Deltas in this report share one cause -- the code change between A and B -- "
    "but the diff does not establish causality between any two deltas. A stall "
    "that fell and a hit rate that rose moved together; whether one produced "
    "the other is a question for the source, not for this table."
)


# ---------------------------------------------------------------------------
# Axis tables. Every entry is a clock-independent quantity (count, percentage,
# or cycles per issue-slot), so no per-metric clock correction is needed here;
# the one clock-dependent quantity in the diff -- duration -- is handled by
# compare_measurements explicitly.
#
# Entry: (catalog key or raw metric name, label, unit, direction, abs_floor)
# direction: "higher_better" | "lower_better" | None (a shift, not a verdict).
# ---------------------------------------------------------------------------

_SOL_ENTRIES: List[Tuple[str, str, str, Optional[str], float]] = [
    # Utilisation shifting is a change of regime, not intrinsically good or bad:
    # compute SOL falling can mean "starved" or "the same work finished sooner".
    ("compute_sol", "SM compute throughput", "% of peak", None, PERCENT_POINT_FLOOR),
    ("memory_sol", "Compute-memory throughput", "% of peak", None, PERCENT_POINT_FLOOR),
    ("dram_sol", "DRAM throughput", "% of peak", None, PERCENT_POINT_FLOOR),
    ("l1_sol", "L1/TEX throughput", "% of peak", None, PERCENT_POINT_FLOOR),
    ("l2_sol", "L2 throughput", "% of peak", None, PERCENT_POINT_FLOOR),
]

_OCCUPANCY_ENTRIES: List[Tuple[str, str, str, Optional[str], float]] = [
    ("achieved_occupancy", "Achieved occupancy", "%", "higher_better", 1.0),
    ("theoretical_occupancy", "Theoretical occupancy", "%", "higher_better", 1.0),
    ("waves_per_sm", "Waves per SM", "waves", None, 0.0),
    ("registers_per_thread", "Registers per thread", "regs", None, 0.0),
]

_INSTRUCTION_ENTRIES: List[Tuple[str, str, str, Optional[str], float]] = [
    ("inst_executed", "Instructions executed (warp)", "inst", None, 0.0),
    ("executed_ipc", "Executed IPC", "inst/cycle", "higher_better", 0.0),
    ("issue_active", "Issue-active fraction", "of cycles", "higher_better", 0.0),
    ("branch_pct", "Branch instruction share", "%", None, PERCENT_POINT_FLOOR),
    # Reported at whatever aggregation the report used; no good/bad direction
    # is claimed for it here, only the shift.
    ("avg_thread_executed", "Avg threads active per inst", "threads", None, 0.5),
    ("pipe_fma_util", "FMA pipe utilisation", "% of peak", None, PERCENT_POINT_FLOOR),
    ("pipe_alu_util", "ALU pipe utilisation", "% of peak", None, PERCENT_POINT_FLOOR),
    ("pipe_lsu_util", "LSU pipe utilisation", "% of peak", None, PERCENT_POINT_FLOOR),
    (
        "pipe_tensor_hmma_util",
        "Tensor (HMMA) pipe utilisation",
        "% of peak",
        None,
        PERCENT_POINT_FLOOR,
    ),
]

_SPILL_ENTRIES: List[Tuple[str, str, str, Optional[str], float]] = [
    (
        "spill_local_inst",
        "Register-spill instructions (SASS)",
        "inst",
        "lower_better",
        0.0,
    ),
    (
        "local_spill_requests_pct",
        "Local traffic that is spilling",
        "%",
        "lower_better",
        PERCENT_POINT_FLOOR,
    ),
    ("local_ld_inst", "Local-load instructions", "inst", "lower_better", 0.0),
    ("local_st_inst", "Local-store instructions", "inst", "lower_better", 0.0),
]

_SHARED_ENTRIES: List[Tuple[str, str, str, Optional[str], float]] = [
    (
        "shared_bank_conflicts_ld",
        "Shared-mem load bank conflicts",
        "conflicts",
        "lower_better",
        32.0,
    ),
    (
        "shared_bank_conflicts_st",
        "Shared-mem store bank conflicts",
        "conflicts",
        "lower_better",
        32.0,
    ),
    (
        "shared_wavefronts_excessive",
        "Excessive shared wavefronts",
        "wavefronts",
        "lower_better",
        32.0,
    ),
]

# Hit rates carry their traffic so a swing on nothing is not a finding.
# Entry: (label, hit-rate key, traffic key, total-sectors key for the level)
_HIT_RATE_ENTRIES: List[Tuple[str, str, str, Optional[str]]] = [
    ("L1/TEX sector hit rate", "l1_hit_rate", "l1_sectors_total", None),
    ("L2 sector hit rate", "l2_hit_rate", "l2_sectors_total", None),
    ("L2 read hit rate", "l2_read_hit_rate", "l2_sectors_read", "l2_sectors_total"),
    ("L2 write hit rate", "l2_write_hit_rate", "l2_sectors_write", "l2_sectors_total"),
    (
        "Local-load L1TEX hit rate",
        "local_ld_hit_rate",
        "l1_local_ld_sectors",
        "l1_sectors_total",
    ),
    (
        "Local-store L1TEX hit rate",
        "l1tex__t_sector_pipe_lsu_mem_local_op_st_hit_rate.pct",
        "l1_local_st_sectors",
        "l1_sectors_total",
    ),
]

# Miss/traffic counts: the traffic-weighted complement of the hit rates above.
_TRAFFIC_ENTRIES: List[Tuple[str, str, str, Optional[str], float]] = [
    (
        "l2_miss_sectors",
        "L2 miss sectors (to DRAM/sysmem)",
        "sectors",
        "lower_better",
        512.0,
    ),
    (
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss.sum",
        "Local-load sectors missing L1TEX (sent to L2)",
        "sectors",
        "lower_better",
        512.0,
    ),
    (
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum",
        "Local-store sectors missing L1TEX (sent to L2)",
        "sectors",
        "lower_better",
        512.0,
    ),
    ("dram_bytes", "DRAM traffic", "bytes", None, 4096.0),
    ("l2_sectors_total", "L2 sector traffic", "sectors", None, 512.0),
    ("l1_sectors_total", "L1/TEX sector traffic", "sectors", None, 512.0),
]


# ---------------------------------------------------------------------------
# Delta rows
# ---------------------------------------------------------------------------


def _delta_row(
    label: str,
    a: Optional[float],
    b: Optional[float],
    *,
    unit: str = "",
    direction: Optional[str] = None,
    noise_rel: float = NOISE_REL,
    abs_floor: float = 0.0,
    metric: str = "",
    note: str = "",
) -> Optional[Dict[str, Any]]:
    """One metric's A/B values, delta, and severity-coded status.

    Returns None when neither side carries the metric -- an absent counter is
    not a delta. ``status`` is one of ``improved | regressed | changed |
    unchanged | a_only | b_only``; ``changed`` is used when the metric has no
    intrinsic good direction (a utilisation shifting regimes, a traffic count).
    """
    if a is None and b is None:
        return None
    row: Dict[str, Any] = {
        "label": label,
        "metric": metric,
        "unit": unit,
        "a": a,
        "b": b,
        "delta": None,
        "rel_change": None,
        "status": "",
        "note": note,
    }
    if a is None or b is None:
        row["status"] = "b_only" if a is None else "a_only"
        row["note"] = (
            note or "present in only one report; no delta can be formed"
        ).strip()
        return row

    delta = float(b) - float(a)
    row["delta"] = delta
    rel: Optional[float] = None
    if a:
        rel = delta / abs(float(a))
        row["rel_change"] = rel

    within_floor = abs_floor > 0.0 and abs(delta) < abs_floor
    within_rel = rel is not None and abs(rel) < noise_rel
    baseline_zero_still_zero = a == 0 and b == 0

    if baseline_zero_still_zero or within_floor or within_rel:
        row["status"] = "unchanged"
        return row

    if direction == "higher_better":
        row["status"] = "improved" if delta > 0 else "regressed"
    elif direction == "lower_better":
        row["status"] = "improved" if delta < 0 else "regressed"
    else:
        row["status"] = "changed"
    if a == 0 and b != 0:
        row["note"] = (note + " (baseline is zero; no ratio)").strip()
    return row


def _axis_rows(
    view_a: MetricView,
    view_b: MetricView,
    entries: Sequence[Tuple[str, str, str, Optional[str], float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, label, unit, direction, abs_floor in entries:
        row = _delta_row(
            label,
            view_a.get(key),
            view_b.get(key),
            unit=unit,
            direction=direction,
            abs_floor=abs_floor,
            metric=key,
        )
        if row is not None:
            rows.append(row)
    return rows


def _stall_rows(view_a: MetricView, view_b: MetricView) -> List[Dict[str, Any]]:
    """Per-stall-class deltas in cycles per issue-slot, largest movement first."""
    rows: List[Dict[str, Any]] = []
    for key, reason in STALL_REASONS.items():
        row = _delta_row(
            reason.display,
            view_a.stall(key),
            view_b.stall(key),
            unit="cycles/issue-slot",
            direction="lower_better",
            abs_floor=STALL_CYCLES_FLOOR,
            metric=reason.metric_name,
        )
        if row is None:
            continue
        row["bucket"] = reason.bucket
        rows.append(row)
    rows.sort(key=lambda r: -abs(r["delta"] or 0.0))
    return rows


def _hit_rate_rows(view_a: MetricView, view_b: MetricView) -> List[Dict[str, Any]]:
    """Hit-rate deltas, each carrying and weighted by its own traffic."""
    rows: List[Dict[str, Any]] = []
    for label, hit_key, traffic_key, total_key in _HIT_RATE_ENTRIES:
        row = _delta_row(
            label,
            view_a.get(hit_key),
            view_b.get(hit_key),
            unit="%",
            direction="higher_better",
            abs_floor=PERCENT_POINT_FLOOR,
            metric=hit_key,
        )
        if row is None:
            continue
        traffic_a = view_a.get(traffic_key)
        traffic_b = view_b.get(traffic_key)
        row["traffic_a"] = traffic_a
        row["traffic_b"] = traffic_b
        row["traffic_metric"] = traffic_key
        if total_key:
            total_a = view_a.get(total_key)
            total_b = view_b.get(total_key)
            # No total means the share is unknown, not zero. Demoting a hit-rate
            # swing to "negligible" on unknown traffic would hide real moves.
            share_a = (
                (traffic_a / total_a) if traffic_a is not None and total_a else None
            )
            share_b = (
                (traffic_b / total_b) if traffic_b is not None and total_b else None
            )
            row["traffic_share_a"] = share_a
            row["traffic_share_b"] = share_b
            if (
                row["status"] in ("improved", "regressed")
                and share_a is not None
                and share_b is not None
                and share_a < NEGLIGIBLE_TRAFFIC_SHARE
                and share_b < NEGLIGIBLE_TRAFFIC_SHARE
            ):
                row["status"] = "negligible_traffic"
                row["note"] = (
                    f"under {NEGLIGIBLE_TRAFFIC_SHARE * 100:.0f}% of this level's "
                    "sector traffic on both sides; the swing is real but weightless"
                )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Kernel matching
# ---------------------------------------------------------------------------


def _launch_sig(
    metrics: Mapping[str, float],
) -> Tuple[Optional[float], Optional[float]]:
    view = MetricView(metrics)
    return (view.get("grid_size"), view.get("block_size"))


def _match_kernels(
    bundles_a: Mapping[Tuple[int, int], Any],
    bundles_b: Mapping[Tuple[int, int], Any],
) -> Tuple[
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """Pair launches by demangled name, then by launch config when duplicated.

    Order-based pairing is the last resort for same-name-same-config repeats
    (e.g. the same kernel profiled at several iterations); anything left over
    on either side is reported unmatched rather than force-paired.
    """
    by_name_a: Dict[str, List[Tuple[int, int]]] = {}
    by_name_b: Dict[str, List[Tuple[int, int]]] = {}
    for key in sorted(bundles_a):
        by_name_a.setdefault(bundles_a[key].kernel_name, []).append(key)
    for key in sorted(bundles_b):
        by_name_b.setdefault(bundles_b[key].kernel_name, []).append(key)

    matches: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    unmatched_a: List[Tuple[int, int]] = []
    unmatched_b: List[Tuple[int, int]] = []

    for name, keys_a in by_name_a.items():
        keys_b = list(by_name_b.get(name, []))
        if not keys_b:
            unmatched_a.extend(keys_a)
            continue
        remaining_a = list(keys_a)
        if len(remaining_a) > 1 or len(keys_b) > 1:
            # Same name on both sides more than once: prefer identical launch
            # configuration before falling back to encounter order.
            paired_a: List[Tuple[int, int]] = []
            for key_a in list(remaining_a):
                sig_a = _launch_sig(bundles_a[key_a].metrics)
                for key_b in list(keys_b):
                    if _launch_sig(bundles_b[key_b].metrics) == sig_a:
                        matches.append((key_a, key_b))
                        keys_b.remove(key_b)
                        paired_a.append(key_a)
                        break
            remaining_a = [k for k in remaining_a if k not in paired_a]
        while remaining_a and keys_b:
            matches.append((remaining_a.pop(0), keys_b.pop(0)))
        unmatched_a.extend(remaining_a)
        unmatched_b.extend(keys_b)

    for name, keys_b in by_name_b.items():
        if name not in by_name_a:
            unmatched_b.extend(keys_b)

    matches.sort()
    return matches, sorted(unmatched_a), sorted(unmatched_b)


# ---------------------------------------------------------------------------
# Findings diff -- what changed the verdict
# ---------------------------------------------------------------------------


def _index_findings(
    findings: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for finding in findings or ():
        if not isinstance(finding, Mapping):
            continue
        category = str(finding.get("category") or "")
        if not category:
            continue
        previous = out.get(category)
        if previous is None or _SEVERITY_RANK.get(
            str(finding.get("severity")), 0
        ) > _SEVERITY_RANK.get(str(previous.get("severity")), 0):
            out[category] = dict(finding)
    return out


def _finding_brief(finding: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "category": finding.get("category"),
        "title": finding.get("title"),
        "severity": finding.get("severity"),
        "summary": finding.get("summary"),
        "source": finding.get("source"),
    }


def _diff_findings(
    findings_a: Optional[Sequence[Mapping[str, Any]]],
    findings_b: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Findings that appeared, disappeared, or changed severity from A to B."""
    index_a = _index_findings(findings_a)
    index_b = _index_findings(findings_b)

    appeared = [_finding_brief(index_b[cat]) for cat in index_b if cat not in index_a]
    disappeared = [
        _finding_brief(index_a[cat]) for cat in index_a if cat not in index_b
    ]
    severity_changed: List[Dict[str, Any]] = []
    unchanged = 0
    for category in index_a:
        if category not in index_b:
            continue
        sev_a = str(index_a[category].get("severity") or "info")
        sev_b = str(index_b[category].get("severity") or "info")
        if sev_a == sev_b:
            unchanged += 1
            continue
        severity_changed.append(
            {
                "category": category,
                "title": index_b[category].get("title"),
                "severity_a": sev_a,
                "severity_b": sev_b,
                "direction": (
                    "escalated"
                    if _SEVERITY_RANK.get(sev_b, 0) > _SEVERITY_RANK.get(sev_a, 0)
                    else "eased"
                ),
            }
        )

    rank = lambda f: -_SEVERITY_RANK.get(str(f.get("severity")), 0)  # noqa: E731
    appeared.sort(key=rank)
    disappeared.sort(key=rank)
    return {
        "appeared": appeared,
        "disappeared": disappeared,
        "severity_changed": severity_changed,
        "unchanged_count": unchanged,
    }


def _stall_reliability(diag: Optional[Mapping[str, Any]], side: str) -> Dict[str, Any]:
    """Whether one report's stall accounting closed well enough to diff."""
    stalls: Mapping[str, Any] = {}
    if isinstance(diag, Mapping):
        sections = diag.get("sections")
        if isinstance(sections, Mapping) and isinstance(
            sections.get("stalls"), Mapping
        ):
            stalls = sections["stalls"]
    share = stalls.get("explained_share")
    if share is None:
        return {
            "side": side,
            "reliable": False,
            "explained_share": None,
            "note": f"report {side} carries no stall-closure data (warp latency total absent)",
        }
    share = float(share)
    if share < 0.9:
        return {
            "side": side,
            "reliable": False,
            "explained_share": share,
            "note": (
                f"stall accounting failed closure in report {side}: the reasons "
                f"present explain only {share * 100:.0f}% of warp latency"
            ),
        }
    if share > 1.02:
        return {
            "side": side,
            "reliable": False,
            "explained_share": share,
            "note": (
                f"stall accounting failed closure in report {side}: disjoint stall "
                f"states sum to {share * 100:.1f}% of the total they partition, so "
                "the replay passes disagree"
            ),
        }
    return {"side": side, "reliable": True, "explained_share": share, "note": ""}


# ---------------------------------------------------------------------------
# Duration and the clock guard
# ---------------------------------------------------------------------------


def _duration_block(
    view_a: MetricView, view_b: MetricView, comparison: Mapping[str, Any]
) -> Dict[str, Any]:
    dur_a = view_a.get("duration_ns")
    dur_b = view_b.get("duration_ns")
    cyc_a = view_a.get("gpc_cycles_elapsed")
    cyc_b = view_b.get("gpc_cycles_elapsed")
    sm_a = view_a.get("sm_clock_hz")
    sm_b = view_b.get("sm_clock_hz")

    raw_ratio = (dur_b / dur_a) if dur_a and dur_b else None
    clock_ratio = (sm_b / sm_a) if sm_a and sm_b else None
    # Same correction compare_measurements applies for a duration: multiply.
    normalised = comparison.get("clock_normalised_ratio")
    if normalised is None and raw_ratio is not None and clock_ratio is not None:
        normalised = raw_ratio * clock_ratio
    cycles_ratio = (cyc_b / cyc_a) if cyc_a and cyc_b else None

    comparable = bool(comparison.get("comparable"))
    if raw_ratio is None:
        headline = "duration missing on at least one side; no time delta can be formed"
    elif comparable:
        headline = (
            f"SM clocks agree ({(sm_a or 0) / 1e6:.0f} vs {(sm_b or 0) / 1e6:.0f} MHz), "
            f"so the raw-time ratio stands: B runs at {raw_ratio:.3f}x of A"
            + (
                f"; the elapsed-cycle ratio {cycles_ratio:.3f}x cross-checks it"
                if cycles_ratio is not None
                else ""
            )
            + "."
        )
    else:
        pieces = [
            "raw durations are NOT comparable as a speedup"
            + (
                f" (SM clocks {(sm_a or 0) / 1e6:.0f} vs {(sm_b or 0) / 1e6:.0f} MHz)"
                if sm_a and sm_b
                else ""
            )
        ]
        if normalised is not None:
            pieces.append(
                f"clock-normalised, B runs at {normalised:.3f}x of A "
                f"(raw {raw_ratio:.3f}x contains the clock change)"
            )
        if cycles_ratio is not None:
            pieces.append(
                f"the clock-independent elapsed-cycle ratio is {cycles_ratio:.3f}x"
            )
        headline = "; ".join(pieces) + "."

    # The two clock-corrected estimates should agree. When they do not, the
    # clock moved between replay passes (duration and cycles come from
    # different passes), and neither figure deserves more trust than their gap.
    if (
        normalised is not None
        and cycles_ratio is not None
        and cycles_ratio
        and abs(normalised / cycles_ratio - 1.0) > 0.02
    ):
        headline += (
            f" NOTE: the clock-normalised duration ratio ({normalised:.3f}x) and the "
            f"elapsed-cycle ratio ({cycles_ratio:.3f}x) disagree by "
            f"{abs(normalised / cycles_ratio - 1.0) * 100:.1f}%, which means the clock "
            "varied between replay passes; trust neither figure to better than that."
        )

    return {
        "a_ns": dur_a,
        "b_ns": dur_b,
        "raw_ratio": raw_ratio,
        "sm_clock_a_hz": sm_a,
        "sm_clock_b_hz": sm_b,
        "clock_ratio": clock_ratio,
        "clock_normalised_ratio": normalised,
        "elapsed_cycles_a": cyc_a,
        "elapsed_cycles_b": cyc_b,
        "cycles_ratio": cycles_ratio,
        "comparable_as_raw_time": comparable,
        "headline": headline,
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


def diff_ncu_reports(
    report_a: str,
    report_b: str,
    *,
    kernel_like: str = "%",
    findings_per_kernel: int = 24,
    ncu_report_module: Any = None,
) -> Dict[str, Any]:
    """Compare two .ncu-rep files kernel-by-kernel.

    A is the baseline, B the candidate; every ratio in the result is B over A.
    ``kernel_like`` filters kernels by the same LIKE pattern the rest of the
    package uses. ``ncu_report_module`` is the usual injection point for tests.
    """
    bundles_a = walk_report_once(
        report_a,
        kernel_like=kernel_like,
        include_source=False,
        ncu_report_module=ncu_report_module,
    )
    bundles_b = walk_report_once(
        report_b,
        kernel_like=kernel_like,
        include_source=False,
        ncu_report_module=ncu_report_module,
    )

    # The findings diff runs the same pipeline `ncu-diagnose` runs -- signal
    # scan, shipped-rule reconciliation and all -- so what appeared or
    # disappeared here is exactly what the single-report tool would have said.
    def _diagnoses(path: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
        payload = diagnose_ncu_report(
            path,
            kernel_like=kernel_like,
            top_kernels=len(bundles_a) + len(bundles_b) + 1,
            findings_per_kernel=findings_per_kernel,
            include_source=False,
            ncu_report_module=ncu_report_module,
        )
        out: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for kernel in payload.get("kernels") or []:
            if isinstance(kernel, dict):
                out[(kernel.get("range_index"), kernel.get("action_index"))] = kernel
        out["__payload__"] = payload  # type: ignore[index]
        return out

    diag_a = _diagnoses(report_a)
    diag_b = _diagnoses(report_b)
    payload_a: Dict[str, Any] = diag_a.pop("__payload__")  # type: ignore[assignment]
    payload_b: Dict[str, Any] = diag_b.pop("__payload__")  # type: ignore[assignment]

    matches, unmatched_a, unmatched_b = _match_kernels(bundles_a, bundles_b)

    kernels: List[Dict[str, Any]] = []
    blocked_pairs: List[str] = []
    for key_a, key_b in matches:
        bundle_a = bundles_a[key_a]
        bundle_b = bundles_b[key_b]
        view_a = MetricView(bundle_a.metrics)
        view_b = MetricView(bundle_b.metrics)

        # Clock guard first. This is compare_measurements' decision, not ours.
        ctx_a = describe_collection_mode(
            source="ncu",
            sm_clock_hz=view_a.get("sm_clock_hz"),
            gpc_clock_hz=view_a.get("gpc_clock_hz"),
        )
        ctx_b = describe_collection_mode(
            source="ncu",
            sm_clock_hz=view_b.get("sm_clock_hz"),
            gpc_clock_hz=view_b.get("gpc_clock_hz"),
        )
        comparison = compare_measurements(
            ctx_a,
            ctx_b,
            baseline_value=view_a.get("duration_ns"),
            candidate_value=view_b.get("duration_ns"),
            metric="duration_ns",
        )
        if not comparison.get("comparable"):
            blocked_pairs.append(bundle_a.kernel_name)

        kernel_diag_a = diag_a.get(key_a)
        kernel_diag_b = diag_b.get(key_b)
        reliability = [
            _stall_reliability(kernel_diag_a, "A"),
            _stall_reliability(kernel_diag_b, "B"),
        ]
        stall_notes = [r["note"] for r in reliability if r["note"]]

        verdict_a = str((kernel_diag_a or {}).get("verdict") or "")
        verdict_b = str((kernel_diag_b or {}).get("verdict") or "")

        grid_a, block_a = _launch_sig(bundle_a.metrics)
        grid_b, block_b = _launch_sig(bundle_b.metrics)

        kernels.append(
            {
                "kernel_name": bundle_a.kernel_name,
                "launch_a": {"grid": grid_a, "block": block_a},
                "launch_b": {"grid": grid_b, "block": block_b},
                "clock_comparison": dict(comparison),
                "duration": _duration_block(view_a, view_b, comparison),
                "verdict": {
                    "a": verdict_a,
                    "b": verdict_b,
                    "changed": bool(verdict_a and verdict_b and verdict_a != verdict_b),
                },
                "findings_diff": _diff_findings(
                    (kernel_diag_a or {}).get("findings"),
                    (kernel_diag_b or {}).get("findings"),
                ),
                "stall_delta_reliable": all(r["reliable"] for r in reliability),
                "stall_reliability": reliability,
                "stall_reliability_note": "; ".join(stall_notes),
                "axes": {
                    "speed_of_light": _axis_rows(view_a, view_b, _SOL_ENTRIES),
                    "stall_composition": _stall_rows(view_a, view_b),
                    "occupancy": _axis_rows(view_a, view_b, _OCCUPANCY_ENTRIES),
                    "memory_hierarchy": _hit_rate_rows(view_a, view_b)
                    + _axis_rows(view_a, view_b, _TRAFFIC_ENTRIES),
                    "instruction_mix": _axis_rows(view_a, view_b, _INSTRUCTION_ENTRIES),
                    "spills": _axis_rows(view_a, view_b, _SPILL_ENTRIES),
                    "shared_memory": _axis_rows(view_a, view_b, _SHARED_ENTRIES),
                },
            }
        )

    def _unmatched(bundles, keys) -> List[Dict[str, Any]]:
        out = []
        for key in keys:
            grid, block = _launch_sig(bundles[key].metrics)
            out.append(
                {
                    "kernel_name": bundles[key].kernel_name,
                    "range_index": key[0],
                    "action_index": key[1],
                    "grid": grid,
                    "block": block,
                }
            )
        return out

    if blocked_pairs:
        guard_summary = (
            f"SM clocks differ by more than 1% on {len(blocked_pairs)} of "
            f"{len(matches)} matched kernel(s). Raw-time deltas for those kernels "
            "are NOT speedups; read the clock-normalised and elapsed-cycle "
            "figures instead."
        )
    elif matches:
        guard_summary = (
            "SM clocks agree within 1% on every matched kernel; raw-duration "
            "ratios are presented with elapsed-cycle ratios as cross-check."
        )
    else:
        guard_summary = "No kernels matched between the two reports."

    notes = [_NO_CAUSALITY_NOTE]
    for kernel in kernels:
        if not kernel["stall_delta_reliable"]:
            notes.append(
                f"`{kernel['kernel_name']}`: {kernel['stall_reliability_note']}; "
                "the stall-composition deltas for this kernel are unreliable."
            )

    return {
        "report_a": str(report_a),
        "report_b": str(report_b),
        "kernel_filter": kernel_like,
        "gpu_a": payload_a.get("gpu") or payload_a.get("gpu_detected_from_report"),
        "gpu_b": payload_b.get("gpu") or payload_b.get("gpu_detected_from_report"),
        "clock_guard": {
            "all_comparable": not blocked_pairs,
            "blocked_kernels": blocked_pairs,
            "summary": guard_summary,
        },
        "matched_kernel_count": len(matches),
        "kernels": kernels,
        "unmatched_a": _unmatched(bundles_a, unmatched_a),
        "unmatched_b": _unmatched(bundles_b, unmatched_b),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

_STATUS_MARK = {
    "improved": "improved",
    "regressed": "REGRESSED",
    "changed": "changed",
    "unchanged": "unchanged",
    "negligible_traffic": "negligible traffic",
    "a_only": "A only",
    "b_only": "B only",
}


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value != int(value):
        if abs(value) >= 100:
            return f"{value:,.0f}"
        return f"{value:.3g}" if abs(value) < 10 else f"{value:.2f}"
    return f"{int(value):,}"


def _fmt_ratio(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.3f}"


def _fmt_rel(row: Mapping[str, Any]) -> str:
    rel = row.get("rel_change")
    if rel is None:
        if row.get("a") == 0 and (row.get("b") or 0) != 0:
            return "from 0"
        return "-"
    return f"{rel * 100:+.1f}%"


def _axis_table(
    lines: List[str], title: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    shown = [r for r in rows if r.get("status") not in ("unchanged", "")]
    hidden = len(rows) - len(shown)
    if not rows:
        return
    lines.append(f"### {title}")
    lines.append("")
    if not shown:
        lines.append(f"All {len(rows)} tracked metrics unchanged within noise.")
        lines.append("")
        return
    has_traffic = any("traffic_a" in r for r in shown)
    header = "| metric | A | B | delta | rel | status |"
    sep = "|---|---|---|---|---|---|"
    if has_traffic:
        header = "| metric | A | B | delta | rel | traffic A -> B | status |"
        sep = "|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for row in shown:
        status = _STATUS_MARK.get(str(row.get("status")), str(row.get("status")))
        note = str(row.get("note") or "")
        cells = [
            f"{row['label']} ({row['unit']})" if row.get("unit") else str(row["label"]),
            _fmt(row.get("a")),
            _fmt(row.get("b")),
            _fmt(row.get("delta")),
            _fmt_rel(row),
        ]
        if has_traffic:
            if "traffic_a" in row:
                cells.append(
                    f"{_fmt(row.get('traffic_a'))} -> {_fmt(row.get('traffic_b'))}"
                )
            else:
                cells.append("-")
        cells.append(status + (f" ({note})" if note else ""))
        lines.append("| " + " | ".join(cells) + " |")
    if hidden:
        lines.append("")
        lines.append(f"_{hidden} further metric(s) unchanged within noise._")
    lines.append("")


def diff_result_to_markdown(payload: Mapping[str, Any]) -> str:
    """Render :func:`diff_ncu_reports` output as a readable report."""
    if not isinstance(payload, Mapping):
        return "# NCU Report Diff\n\n(no payload)\n"

    lines: List[str] = ["# NCU Report Diff (A -> B)", ""]
    lines.append(f"- A (baseline): `{payload.get('report_a', '')}`")
    lines.append(f"- B (candidate): `{payload.get('report_b', '')}`")
    gpu_a, gpu_b = payload.get("gpu_a") or "?", payload.get("gpu_b") or "?"
    lines.append(
        f"- GPU: {gpu_a}"
        + ("" if gpu_a == gpu_b else f" vs {gpu_b} -- DIFFERENT DEVICES")
    )
    lines.append(f"- matched kernels: {payload.get('matched_kernel_count', 0)}")
    for side, key in (("A", "unmatched_a"), ("B", "unmatched_b")):
        unmatched = payload.get(key) or []
        if unmatched:
            names = ", ".join(f"`{u.get('kernel_name')}`" for u in unmatched)
            lines.append(f"- only in {side} (no counterpart to diff against): {names}")
    lines.append("")

    # The clock guard leads. Everything below it is read in its light.
    guard = payload.get("clock_guard") or {}
    lines.append("## Clock guard")
    lines.append("")
    if guard.get("all_comparable"):
        lines.append(str(guard.get("summary") or ""))
    else:
        lines.append(f"**WARNING: {guard.get('summary') or 'clocks differ'}**")
    lines.append("")

    for kernel in payload.get("kernels") or []:
        if not isinstance(kernel, Mapping):
            continue
        launch_a = kernel.get("launch_a") or {}
        lines.append(
            f"## `{kernel.get('kernel_name', '?')}` "
            f"(grid {_fmt(launch_a.get('grid'))}, block {_fmt(launch_a.get('block'))})"
        )
        lines.append("")

        duration = kernel.get("duration") or {}
        comparison = kernel.get("clock_comparison") or {}
        lines.append("### Duration")
        lines.append("")
        lines.append("| quantity | A | B | B/A |")
        lines.append("|---|---|---|---|")
        raw_label = (
            "raw duration (ns)"
            if duration.get("comparable_as_raw_time")
            else "raw duration (ns) -- confounded by clock, NOT a speedup"
        )
        lines.append(
            f"| {raw_label} | {_fmt(duration.get('a_ns'))} | {_fmt(duration.get('b_ns'))} "
            f"| {_fmt_ratio(duration.get('raw_ratio'))} |"
        )
        sm_a, sm_b = duration.get("sm_clock_a_hz"), duration.get("sm_clock_b_hz")
        lines.append(
            f"| SM clock (MHz) | {_fmt(sm_a / 1e6 if sm_a else None)} "
            f"| {_fmt(sm_b / 1e6 if sm_b else None)} | {_fmt_ratio(duration.get('clock_ratio'))} |"
        )
        if duration.get("clock_normalised_ratio") is not None:
            lines.append(
                f"| clock-normalised duration ratio | - | - "
                f"| {_fmt_ratio(duration.get('clock_normalised_ratio'))} |"
            )
        lines.append(
            f"| elapsed GPC cycles (clock-independent) | {_fmt(duration.get('elapsed_cycles_a'))} "
            f"| {_fmt(duration.get('elapsed_cycles_b'))} | {_fmt_ratio(duration.get('cycles_ratio'))} |"
        )
        lines.append("")
        lines.append(str(duration.get("headline") or ""))
        lines.append("")
        if not comparison.get("comparable"):
            for blocker in comparison.get("blockers") or []:
                lines.append(f"- guard: {blocker}")
            lines.append("")
        for caveat in comparison.get("caveats") or []:
            lines.append(f"- caveat: {caveat}")
        if comparison.get("caveats"):
            lines.append("")

        verdict = kernel.get("verdict") or {}
        findings = kernel.get("findings_diff") or {}
        lines.append("### What changed the verdict")
        lines.append("")
        if verdict.get("changed"):
            lines.append(
                f"- bottleneck verdict: `{verdict.get('a')}` -> `{verdict.get('b')}`"
            )
        elif verdict.get("a"):
            lines.append(f"- bottleneck verdict unchanged: `{verdict.get('a')}`")
        for finding in findings.get("disappeared") or []:
            lines.append(
                f"- disappeared ({finding.get('severity')}): "
                f"**{finding.get('title')}** [{finding.get('category')}]"
            )
        for finding in findings.get("appeared") or []:
            lines.append(
                f"- appeared ({finding.get('severity')}): "
                f"**{finding.get('title')}** [{finding.get('category')}]"
            )
        for change in findings.get("severity_changed") or []:
            lines.append(
                f"- {change.get('direction')}: {change.get('title')} "
                f"[{change.get('category')}] {change.get('severity_a')} -> "
                f"{change.get('severity_b')}"
            )
        if not (
            findings.get("appeared")
            or findings.get("disappeared")
            or findings.get("severity_changed")
        ):
            lines.append(
                f"- no findings appeared or disappeared "
                f"({findings.get('unchanged_count', 0)} present on both sides)"
            )
        else:
            lines.append(
                f"- {findings.get('unchanged_count', 0)} finding(s) unchanged on both sides"
            )
        lines.append("")

        axes = kernel.get("axes") or {}
        if not kernel.get("stall_delta_reliable"):
            lines.append(
                f"**Stall deltas unreliable: {kernel.get('stall_reliability_note')}.**"
            )
            lines.append("")
        _axis_table(
            lines,
            "Stall composition (cycles per issue-slot)",
            axes.get("stall_composition") or [],
        )
        _axis_table(lines, "Speed of light", axes.get("speed_of_light") or [])
        _axis_table(lines, "Occupancy", axes.get("occupancy") or [])
        _axis_table(
            lines,
            "Memory hierarchy (traffic-weighted)",
            axes.get("memory_hierarchy") or [],
        )
        _axis_table(lines, "Instruction mix", axes.get("instruction_mix") or [])
        _axis_table(lines, "Spills", axes.get("spills") or [])
        _axis_table(lines, "Shared memory", axes.get("shared_memory") or [])

    notes = payload.get("notes") or []
    if notes:
        lines.append("## Honesty notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines) + "\n"
