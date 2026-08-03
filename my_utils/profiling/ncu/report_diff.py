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

import math
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..analyzers.measurement_context import (
    compare_measurements,
    describe_collection_mode,
    measurement_collection_context,
)
from .metric_catalog import METRIC_CATALOG, STALL_REASONS
from .ncu_diagnostics import MetricView
from .ncu_report_tools import (
    _effective_collection_context,
    _resolve_collection_context,
    diagnose_ncu_report,
    walk_report_once,
)

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

# Raw counts are meaningful only after dividing by work.  These ratios are
# intentionally conservative: a schedule may legitimately alter instruction
# count or traffic, so they make the denominator explicit rather than turning
# every lower raw total into an "improvement".
_NORMALISED_RATIO_ENTRIES: List[Tuple[str, str, str, str, Optional[str], float]] = [
    (
        "shared_bank_conflicts_ld_per_wavefront",
        "Shared-load bank conflicts per wavefront",
        "shared_bank_conflicts_ld",
        "shared_wavefronts_ld",
        "lower_better",
        0.0,
    ),
    (
        "shared_bank_conflicts_st_per_wavefront",
        "Shared-store bank conflicts per wavefront",
        "shared_bank_conflicts_st",
        "shared_wavefronts_st",
        "lower_better",
        0.0,
    ),
    (
        "local_loads_per_instruction",
        "Local-load instructions per executed instruction",
        "local_ld_inst",
        "inst_executed",
        "lower_better",
        0.0,
    ),
    (
        "local_stores_per_instruction",
        "Local-store instructions per executed instruction",
        "local_st_inst",
        "inst_executed",
        "lower_better",
        0.0,
    ),
    (
        "dram_bytes_per_instruction",
        "DRAM bytes per executed instruction",
        "dram_bytes",
        "inst_executed",
        None,
        0.0,
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


def _ratio_rows(view_a: MetricView, view_b: MetricView) -> List[Dict[str, Any]]:
    """Compare counters per explicit work denominator, never raw totals alone."""
    rows: List[Dict[str, Any]] = []
    for key, label, numerator, denominator, direction, abs_floor in _NORMALISED_RATIO_ENTRIES:
        a_num, a_den = view_a.get(numerator), view_a.get(denominator)
        b_num, b_den = view_b.get(numerator), view_b.get(denominator)
        a = a_num / a_den if a_num is not None and a_den else None
        b = b_num / b_den if b_num is not None and b_den else None
        row = _delta_row(
            label,
            a,
            b,
            unit=f"{numerator}/{denominator}",
            direction=direction,
            abs_floor=abs_floor,
            metric=key,
            note=(
                f"normalised by {denominator}; raw numerator is not a workload-"
                "independent verdict"
            ),
        )
        if row is not None:
            row["numerator"] = numerator
            row["denominator"] = denominator
            row["numerator_a"] = a_num
            row["numerator_b"] = b_num
            row["denominator_a"] = a_den
            row["denominator_b"] = b_den
            rows.append(row)
    return rows


def _catalog_rows(view_a: MetricView, view_b: MetricView) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Expose all catalog coverage, while only listing changed/one-sided keys."""
    counts = Counter()
    changed: List[Dict[str, Any]] = []
    for key, spec in sorted(METRIC_CATALOG.items()):
        a, b = view_a.get(key), view_b.get(key)
        if a is not None and b is not None:
            counts["both"] += 1
        elif a is not None:
            counts["a_only"] += 1
        elif b is not None:
            counts["b_only"] += 1
        else:
            counts["neither"] += 1
        row = _delta_row(
            spec.description or key,
            a,
            b,
            unit=spec.unit,
            direction=(
                "higher_better"
                if spec.higher_is_better is True
                else "lower_better"
                if spec.higher_is_better is False
                else None
            ),
            metric=key,
        )
        if row is not None and row["status"] != "unchanged":
            row["category"] = spec.category
            row["section"] = spec.section
            changed.append(row)
    counts["total"] = len(METRIC_CATALOG)
    changed.sort(
        key=lambda row: (
            row["status"] in {"a_only", "b_only"},
            -abs(float(row.get("rel_change") or 0.0)),
            str(row["metric"]),
        )
    )
    return dict(counts), changed


def _raw_metric_diff(
    metrics_a: Mapping[str, float], metrics_b: Mapping[str, float], *, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Inventory every numeric metric and surface the largest raw changes.

    Raw metric names are deliberately not severity-coded. They are an audit
    appendix that prevents a collected counter from being silently discarded;
    the curated and normalised axes carry the interpretation.
    """
    names_a, names_b = set(metrics_a), set(metrics_b)
    common = names_a & names_b
    changes: List[Dict[str, Any]] = []
    changed_count = 0
    for name in common:
        a, b = metrics_a[name], metrics_b[name]
        if not (math.isfinite(float(a)) and math.isfinite(float(b))):
            continue
        if a == b:
            continue
        changed_count += 1
        row = _delta_row(name, a, b, metric=name)
        if row is not None:
            changes.append(row)
    changes.sort(
        key=lambda row: (-abs(float(row.get("rel_change") or 0.0)), str(row["metric"]))
    )
    return {
        "numeric_a": len(names_a),
        "numeric_b": len(names_b),
        "common": len(common),
        "a_only": len(names_a - names_b),
        "b_only": len(names_b - names_a),
        "changed_common": changed_count,
        "changes_truncated": max(0, len(changes) - limit) if limit is not None else 0,
        "changes": changes[:limit] if limit is not None else changes,
    }


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
    """Hit-rate shifts with traffic context, never severity-coded in isolation."""
    rows: List[Dict[str, Any]] = []
    for label, hit_key, traffic_key, total_key in _HIT_RATE_ENTRIES:
        row = _delta_row(
            label,
            view_a.get(hit_key),
            view_b.get(hit_key),
            unit="%",
            # A lower hit rate can be harmless (or even expected) when a
            # schedule moves less traffic or changes the cache's role.  Miss
            # counts and work-normalised traffic decide the cost; hit-rate is
            # context, not a standalone regression verdict.
            direction=None,
            abs_floor=PERCENT_POINT_FLOOR,
            metric=hit_key,
            note="interpret with miss and traffic rows; hit rate alone is not a performance verdict",
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
                row["status"] == "changed"
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
    *,
    aliases_a: Optional[Mapping[str, Any]] = None,
    aliases_b: Optional[Mapping[str, Any]] = None,
    logical_kernel_id_a: str = "",
    logical_kernel_id_b: str = "",
) -> Tuple[
    List[Tuple[Tuple[int, int], Tuple[int, int]]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """Pair launches by logical alias/name, then launch config when duplicated.

    Order-based pairing is the last resort for same-name-same-config repeats
    (e.g. the same kernel profiled at several iterations); anything left over
    on either side is reported unmatched rather than force-paired.
    """
    def _identity(
        key: Tuple[int, int], bundles: Mapping[Tuple[int, int], Any], aliases, logical_id
    ) -> str:
        name = str(bundles[key].kernel_name)
        if isinstance(aliases, Mapping):
            alias = aliases.get(name)
            if alias is not None and str(alias).strip():
                return str(alias).strip()
        # A one-kernel report can use a simple logical id instead of repeating
        # the compiler-generated kernel symbol in an aliases map.
        if logical_id and len(bundles) == 1:
            return logical_id
        return name

    by_name_a: Dict[str, List[Tuple[int, int]]] = {}
    by_name_b: Dict[str, List[Tuple[int, int]]] = {}
    for key in sorted(bundles_a):
        by_name_a.setdefault(_identity(key, bundles_a, aliases_a, logical_kernel_id_a), []).append(key)
    for key in sorted(bundles_b):
        by_name_b.setdefault(_identity(key, bundles_b, aliases_b, logical_kernel_id_b), []).append(key)

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


def _match_metadata(bundle_a: Any, bundle_b: Any, collection_a: Mapping[str, Any], collection_b: Mapping[str, Any]) -> Dict[str, Any]:
    """Explain why the two launches were paired and what changed structurally."""
    aliases_a = collection_a.get("kernel_aliases")
    aliases_b = collection_b.get("kernel_aliases")
    name_a, name_b = str(bundle_a.kernel_name), str(bundle_b.kernel_name)
    alias_a = aliases_a.get(name_a) if isinstance(aliases_a, Mapping) else None
    alias_b = aliases_b.get(name_b) if isinstance(aliases_b, Mapping) else None
    same_name = name_a == name_b
    alias_match = bool(alias_a and alias_b and str(alias_a) == str(alias_b))
    grid_a, block_a = _launch_sig(bundle_a.metrics)
    grid_b, block_b = _launch_sig(bundle_b.metrics)
    return {
        "method": "demangled_name" if same_name else "logical_kernel_alias",
        "confidence": "high" if same_name and (grid_a, block_a) == (grid_b, block_b) else "medium",
        "alias": str(alias_a or alias_b or ""),
        "raw_name_a": name_a,
        "raw_name_b": name_b,
        "launch_signature_changed": (grid_a, block_a) != (grid_b, block_b),
        "note": (
            "launch grid or block changed; this diff explains the measured change "
            "but does not prove equal work without matching workload provenance"
            if (grid_a, block_a) != (grid_b, block_b)
            else ""
        ),
        "alias_match": alias_match,
    }


# ---------------------------------------------------------------------------
# Findings diff -- what changed the verdict
# ---------------------------------------------------------------------------


def _finding_identity(finding: Mapping[str, Any]) -> str:
    """Stable identity that does not collapse distinct shipped NCU rules."""
    evidence = finding.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    rule = str(evidence.get("ncu_rule") or "")
    metric = str(
        evidence.get("metric")
        or evidence.get("metric_name")
        or evidence.get("focus_metric")
        or ""
    )
    return "|".join(
        (
            str(finding.get("source") or "heuristic"),
            str(finding.get("category") or ""),
            rule,
            metric,
            str(finding.get("title") or ""),
        )
    )


def _analysis_for_finding(finding: Mapping[str, Any]) -> str:
    """Map a finding to the coverage gate that could have produced it."""
    category = str(finding.get("category") or "")
    source = str(finding.get("source") or "")
    evidence = finding.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    # Corroboration promotes a heuristic finding's source to ``ncu_rule`` even
    # when it did not originate from a specific stored rule row.  Such a
    # finding is still covered by its normal analysis rather than the optional
    # shipped-rule export.
    if source == "ncu_rule" and evidence.get("ncu_rule"):
        return "ncu_rules"
    if category.startswith("stall_"):
        return "stalls"
    prefixes = (
        ("occupancy_", "occupancy"),
        ("block_size_", "launch"),
        ("small_grid", "launch"),
        ("tail_wave", "launch"),
        ("tile_quantization", "launch"),
        ("uncoalesced_", "coalescing"),
        ("sparse_global_", "coalescing"),
        ("shared_bank_", "shared_memory"),
        ("thread_divergence", "divergence"),
        ("register_spilling", "spilling"),
        ("pipe_", "pipes"),
        ("unexpected_fp64", "instruction_mix"),
        ("poor_cache_", "memory_hierarchy"),
        ("memory", "memory_hierarchy"),
        ("below_roofline", "roofline"),
        ("tensor_cores_idle", "instruction_mix"),
        ("compute_bound_", "bottleneck"),
        ("memory_bound_", "bottleneck"),
    )
    for prefix, analysis in prefixes:
        if category.startswith(prefix):
            return analysis
    return ""


def _coverage_ran(diag: Optional[Mapping[str, Any]], analysis: str) -> Optional[bool]:
    if not analysis:
        return None
    if analysis == "ncu_rules":
        corroboration = (diag or {}).get("corroboration")
        if not isinstance(corroboration, Mapping):
            return None
        available = corroboration.get("shipped_rules_available")
        return bool(available) if available is not None else None
    coverage = (diag or {}).get("coverage")
    if not isinstance(coverage, Mapping):
        return None
    ran = coverage.get("ran")
    if not isinstance(ran, Sequence) or isinstance(ran, (str, bytes)):
        return None
    return analysis in ran


def _index_findings(
    findings: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for finding in findings or ():
        if not isinstance(finding, Mapping):
            continue
        identity = _finding_identity(finding)
        if not str(finding.get("category") or ""):
            continue
        previous = out.get(identity)
        if previous is None or _SEVERITY_RANK.get(
            str(finding.get("severity")), 0
        ) > _SEVERITY_RANK.get(str(previous.get("severity")), 0):
            out[identity] = dict(finding)
    return out


def _finding_brief(finding: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "identity": _finding_identity(finding),
        "category": finding.get("category"),
        "title": finding.get("title"),
        "severity": finding.get("severity"),
        "summary": finding.get("summary"),
        "source": finding.get("source"),
    }


def _diff_findings(
    findings_a: Optional[Sequence[Mapping[str, Any]]],
    findings_b: Optional[Sequence[Mapping[str, Any]]],
    *,
    diag_a: Optional[Mapping[str, Any]] = None,
    diag_b: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Diff findings without treating missing diagnostic coverage as a fix."""
    index_a = _index_findings(findings_a)
    index_b = _index_findings(findings_b)

    appeared: List[Dict[str, Any]] = []
    disappeared: List[Dict[str, Any]] = []
    not_evaluated_in_a: List[Dict[str, Any]] = []
    not_evaluated_in_b: List[Dict[str, Any]] = []
    for identity, finding in index_b.items():
        if identity in index_a:
            continue
        brief = _finding_brief(finding)
        analysis = _analysis_for_finding(finding)
        brief["analysis"] = analysis
        ran = _coverage_ran(diag_a, analysis)
        if ran is True:
            appeared.append(brief)
        else:
            brief["coverage"] = "unknown" if ran is None else "not_collected"
            not_evaluated_in_a.append(brief)
    for identity, finding in index_a.items():
        if identity in index_b:
            continue
        brief = _finding_brief(finding)
        analysis = _analysis_for_finding(finding)
        brief["analysis"] = analysis
        ran = _coverage_ran(diag_b, analysis)
        if ran is True:
            disappeared.append(brief)
        else:
            brief["coverage"] = "unknown" if ran is None else "not_collected"
            not_evaluated_in_b.append(brief)
    severity_changed: List[Dict[str, Any]] = []
    unchanged = 0
    for identity in index_a:
        if identity not in index_b:
            continue
        sev_a = str(index_a[identity].get("severity") or "info")
        sev_b = str(index_b[identity].get("severity") or "info")
        if sev_a == sev_b:
            unchanged += 1
            continue
        severity_changed.append(
            {
                "identity": identity,
                "category": index_b[identity].get("category"),
                "title": index_b[identity].get("title"),
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
    not_evaluated_in_a.sort(key=rank)
    not_evaluated_in_b.sort(key=rank)
    return {
        "appeared": appeared,
        "disappeared": disappeared,
        "severity_changed": severity_changed,
        "not_evaluated_in_a": not_evaluated_in_a,
        "not_evaluated_in_b": not_evaluated_in_b,
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


def _same_provenance_value(left: Any, right: Any) -> bool:
    """Compare JSON-like sidecar values without depending on dict ordering."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return dict(left) == dict(right)
    return left == right


def _report_compatibility_guards(
    comparison: Mapping[str, Any],
    view_a: MetricView,
    view_b: MetricView,
    collection_a: Mapping[str, Any],
    collection_b: Mapping[str, Any],
    *,
    gpu_a: str = "",
    gpu_b: str = "",
) -> Dict[str, Any]:
    """Add device and workload identity guards to collection-mode guards."""
    blockers = list(comparison.get("blockers") or ())
    report_blockers: List[str] = []
    caveats = list(comparison.get("caveats") or ())

    cc_a = (view_a.get("cc_major"), view_a.get("cc_minor"))
    cc_b = (view_b.get("cc_major"), view_b.get("cc_minor"))
    if all(value is not None for value in (*cc_a, *cc_b)) and cc_a != cc_b:
        report_blockers.append(
            f"GPU compute capability differs ({cc_a[0]}.{cc_a[1]} vs {cc_b[0]}.{cc_b[1]}), "
            "so instruction throughput, cache hierarchy, and scheduler behavior differ"
        )
    elif gpu_a and gpu_b and gpu_a != gpu_b:
        report_blockers.append(
            f"GPU identity differs ({gpu_a} vs {gpu_b}); compare only after recollecting "
            "on one device"
        )

    for key, label in (
        ("workload_id", "workload id"),
        ("problem_shape", "problem shape"),
        ("dtype", "dtype"),
        ("input_hash", "input hash"),
        ("output_hash", "output hash"),
    ):
        left, right = collection_a.get(key), collection_b.get(key)
        if left not in (None, "") and right not in (None, ""):
            if not _same_provenance_value(left, right):
                report_blockers.append(
                    f"{label} differs between reports ({left!r} vs {right!r}); the "
                    "kernel may have executed different work"
                )
        else:
            caveats.append(
                f"{label} was not recorded on both sides; equal workload is unproven"
            )

    for key, label in (("kernel_config", "kernel configuration"), ("build_id", "build id"), ("git_commit", "git commit")):
        left, right = collection_a.get(key), collection_b.get(key)
        if left not in (None, "") and right not in (None, "") and not _same_provenance_value(left, right):
            caveats.append(f"{label} changed ({left!r} -> {right!r}), as expected for this A/B run")

    if report_blockers:
        blockers.extend(report_blockers)
    out = dict(comparison)
    out["blockers"] = blockers
    out["caveats"] = caveats
    out["report_blockers"] = report_blockers
    out["comparable"] = not blockers
    if blockers:
        out["ratio"] = None
        out["uncomparable_raw_ratio"] = comparison.get("uncomparable_raw_ratio") or (
            (comparison.get("candidate_value") / comparison.get("baseline_value"))
            if comparison.get("candidate_value") and comparison.get("baseline_value")
            else None
        )
        out["verdict"] = (
            "Not comparable. " + " Also, ".join(blockers).capitalize() + ". "
            "Re-measure both sides with matching provenance before drawing a conclusion."
        )
    return out


def _pm_sampling_diff(source_a: Optional[Mapping[str, Any]], source_b: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare only stable PM aggregate features, never cross-pass timelines."""
    source_a, source_b = source_a or {}, source_b or {}
    valid_a = source_a.get("pm_sampling_validity")
    valid_b = source_b.get("pm_sampling_validity")
    valid_a = valid_a if isinstance(valid_a, Mapping) else {}
    valid_b = valid_b if isinstance(valid_b, Mapping) else {}
    if valid_a.get("usable") is not True or valid_b.get("usable") is not True:
        return {
            "available": False,
            "reason": "PM sampling is unavailable or invalid on at least one side",
            "validity_a": dict(valid_a),
            "validity_b": dict(valid_b),
            "features": [],
        }
    pm_a, pm_b = source_a.get("pm_sampling"), source_b.get("pm_sampling")
    pm_a = pm_a if isinstance(pm_a, Mapping) else {}
    pm_b = pm_b if isinstance(pm_b, Mapping) else {}
    if pm_a.get("available") is not True or pm_b.get("available") is not True:
        return {
            "available": False,
            "reason": "the report does not contain a usable PM sampling summary on both sides",
            "validity_a": dict(valid_a),
            "validity_b": dict(valid_b),
            "features": [],
        }

    def _series_index(pm: Mapping[str, Any]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
        out: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for item in pm.get("series") or ():
            if not isinstance(item, Mapping):
                continue
            metric, group = str(item.get("metric") or ""), str(item.get("pass_group") or "")
            if metric and group:
                out[(metric, group)] = item
        return out

    series_a, series_b = _series_index(pm_a), _series_index(pm_b)
    rows: List[Dict[str, Any]] = []
    for key in sorted(series_a.keys() & series_b.keys()):
        for feature, label in (
            ("duty_cycle", "duty cycle"),
            ("mean_in_active_window", "mean in active window"),
            ("peak", "peak"),
            ("peak_to_mean", "peak-to-mean"),
        ):
            a, b = series_a[key].get(feature), series_b[key].get(feature)
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            row = _delta_row(
                f"{key[0]} [{key[1]}] {label}",
                float(a),
                float(b),
                metric=f"{key[0]}:{key[1]}:{feature}",
                note="same PM pass group only; timeline buckets are never diffed across replay passes",
            )
            if row is not None:
                rows.append(row)
    return {
        "available": True,
        "reason": "",
        "validity_a": dict(valid_a),
        "validity_b": dict(valid_b),
        "matched_series": len(series_a.keys() & series_b.keys()),
        "unmatched_series_a": sorted(f"{m} [{p}]" for m, p in series_a.keys() - series_b.keys()),
        "unmatched_series_b": sorted(f"{m} [{p}]" for m, p in series_b.keys() - series_a.keys()),
        "features": rows,
    }


def _pc_hotspot_diff(source_a: Optional[Mapping[str, Any]], source_b: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Diff PC-sampling hotspot shares, gated on each report's validity check."""
    source_a, source_b = source_a or {}, source_b or {}
    valid_a = source_a.get("sampling_validity")
    valid_b = source_b.get("sampling_validity")
    valid_a = valid_a if isinstance(valid_a, Mapping) else {}
    valid_b = valid_b if isinstance(valid_b, Mapping) else {}
    if valid_a.get("usable") is not True or valid_b.get("usable") is not True:
        return {
            "available": False,
            "reason": "PC sampling is unavailable or invalid on at least one side",
            "validity_a": dict(valid_a),
            "validity_b": dict(valid_b),
            "hotspots": [],
        }

    def _lines(source: Mapping[str, Any]) -> Dict[Tuple[str, int], Mapping[str, Any]]:
        attribution = source.get("stall_attribution")
        attribution = attribution if isinstance(attribution, Mapping) else {}
        out: Dict[Tuple[str, int], Mapping[str, Any]] = {}
        for item in attribution.get("source_lines") or ():
            if not isinstance(item, Mapping):
                continue
            filename, line = str(item.get("file_name") or ""), item.get("line")
            if filename and isinstance(line, (int, float)):
                out[(filename, int(line))] = item
        return out

    lines_a, lines_b = _lines(source_a), _lines(source_b)
    rows: List[Dict[str, Any]] = []
    for key in sorted(lines_a.keys() | lines_b.keys()):
        a, b = lines_a.get(key), lines_b.get(key)
        row = _delta_row(
            f"{key[0]}:{key[1]}",
            float(a.get("share_of_samples")) if a and isinstance(a.get("share_of_samples"), (int, float)) else None,
            float(b.get("share_of_samples")) if b and isinstance(b.get("share_of_samples"), (int, float)) else None,
            unit="share of PC samples",
            metric=f"{key[0]}:{key[1]}",
        )
        if row is not None:
            row["dominant_stall_a"] = a.get("dominant_stall_reason") if a else ""
            row["dominant_stall_b"] = b.get("dominant_stall_reason") if b else ""
            rows.append(row)
    rows.sort(key=lambda row: -abs(float(row.get("delta") or 0.0)))
    return {
        "available": True,
        "reason": "",
        "validity_a": dict(valid_a),
        "validity_b": dict(valid_b),
        "hotspots": rows[:50],
        "hotspots_truncated": max(0, len(rows) - 50),
    }


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


def _comparison_status(
    comparison: Mapping[str, Any], match: Mapping[str, Any], collection_a: Mapping[str, Any], collection_b: Mapping[str, Any]
) -> Tuple[str, str]:
    """Classify what the A/B result is allowed to claim, before repetition stats."""
    if not comparison.get("comparable"):
        return "NOT_COMPARABLE", "collection/device/workload guards block a speed claim"
    workload_keys = ("workload_id", "problem_shape", "dtype", "input_hash")
    has_workload_proof = all(
        collection_a.get(key) not in (None, "")
        and collection_b.get(key) not in (None, "")
        and _same_provenance_value(collection_a.get(key), collection_b.get(key))
        for key in workload_keys
    )
    if match.get("launch_signature_changed") and not has_workload_proof:
        return (
            "DIAGNOSTIC_ONLY",
            "launch geometry changed but workload provenance does not prove equal work",
        )
    if not has_workload_proof:
        return "DIAGNOSTIC_ONLY", "workload provenance is incomplete; metric deltas are diagnostic only"
    return "INCONCLUSIVE", "one collection per side cannot establish a noise-resistant speedup"


def _robust_duration_summary(values: Sequence[float]) -> Dict[str, Any]:
    """Median/MAD summary that is not dominated by one throttled collection."""
    clean = sorted(float(value) for value in values if math.isfinite(float(value)) and value > 0)
    if not clean:
        return {"count": 0, "values_ns": [], "median_ns": None, "mad_ns": None}
    median = statistics.median(clean)
    mad = statistics.median(abs(value - median) for value in clean)
    sigma = 1.4826 * mad
    return {
        "count": len(clean),
        "values_ns": clean,
        "median_ns": median,
        "mad_ns": mad,
        "robust_sigma_ns": sigma,
        "interval_95_ns": [max(0.0, median - 2.0 * sigma), median + 2.0 * sigma],
    }


def _repeat_duration_samples(
    report_paths: Iterable[str],
    *,
    kernel_name: str,
    kernel_alias: str = "",
    kernel_like: str = "%",
    ncu_report_module: Any = None,
) -> Dict[str, Any]:
    """Extract same-logical-kernel duration observations from repeat reports."""
    values: List[float] = []
    missing: List[str] = []
    for path in report_paths:
        collection, _manifest = _resolve_collection_context(path)
        bundles = walk_report_once(
            path,
            kernel_like=kernel_like,
            include_source=False,
            collection=collection,
            ncu_report_module=ncu_report_module,
        )
        matches = []
        aliases = collection.get("kernel_aliases")
        for bundle in bundles.values():
            alias = aliases.get(bundle.kernel_name) if isinstance(aliases, Mapping) else ""
            if bundle.kernel_name == kernel_name or (kernel_alias and str(alias) == kernel_alias):
                duration = MetricView(bundle.metrics).get("duration_ns")
                if duration is not None:
                    matches.append(float(duration))
        if len(matches) != 1:
            missing.append(str(path))
        else:
            values.append(matches[0])
    out = _robust_duration_summary(values)
    out["missing_or_ambiguous_reports"] = missing
    return out


def _repeat_statistics(
    report_paths_a: Sequence[str],
    report_paths_b: Sequence[str],
    kernels: Sequence[Mapping[str, Any]],
    *,
    kernel_like: str,
    ncu_report_module: Any = None,
) -> Dict[str, Any]:
    """Judge whether repeated A/B durations have separated robust intervals."""
    if len(report_paths_a) < 2 or len(report_paths_b) < 2:
        return {
            "available": False,
            "reason": "at least two reports per side are required for a repeatability judgement",
            "reports_a": list(report_paths_a),
            "reports_b": list(report_paths_b),
            "kernels": [],
        }
    rows: List[Dict[str, Any]] = []
    for kernel in kernels:
        match = kernel.get("match") or {}
        a = _repeat_duration_samples(
            report_paths_a,
            kernel_name=str(match.get("raw_name_a") or kernel.get("kernel_name") or ""),
            kernel_alias=str(match.get("alias") or ""),
            kernel_like=kernel_like,
            ncu_report_module=ncu_report_module,
        )
        b = _repeat_duration_samples(
            report_paths_b,
            kernel_name=str(match.get("raw_name_b") or kernel.get("kernel_name") or ""),
            kernel_alias=str(match.get("alias") or ""),
            kernel_like=kernel_like,
            ncu_report_module=ncu_report_module,
        )
        outcome = "inconclusive"
        ratio = None
        if a["count"] >= 2 and b["count"] >= 2 and a["median_ns"] and b["median_ns"]:
            ratio = b["median_ns"] / a["median_ns"]
            a_low, a_high = a["interval_95_ns"]
            b_low, b_high = b["interval_95_ns"]
            if b_high < a_low:
                outcome = "stable_improvement"
            elif b_low > a_high:
                outcome = "stable_regression"
            else:
                outcome = "overlapping_intervals"
        rows.append(
            {
                "kernel_name": kernel.get("kernel_name"),
                "a": a,
                "b": b,
                "median_ratio_b_over_a": ratio,
                "outcome": outcome,
            }
        )
    return {
        "available": True,
        "method": "median and MAD-derived interval (median +/- 2 * 1.4826 * MAD)",
        "reports_a": list(report_paths_a),
        "reports_b": list(report_paths_b),
        "kernels": rows,
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
    collection_manifest_a: str = "",
    collection_manifest_b: str = "",
    repeat_reports_a: Sequence[str] = (),
    repeat_reports_b: Sequence[str] = (),
    ncu_report_module: Any = None,
) -> Dict[str, Any]:
    """Compare two .ncu-rep files kernel-by-kernel.

    A is the baseline, B the candidate; every ratio in the result is B over A.
    ``kernel_like`` filters kernels by the same LIKE pattern the rest of the
    package uses. ``repeat_reports_a`` and ``repeat_reports_b`` add independent
    reports to a median/MAD repeatability check. ``ncu_report_module`` is the
    usual injection point for tests.
    """
    collection_a, manifest_a = _resolve_collection_context(
        report_a, collection_manifest=collection_manifest_a
    )
    collection_b, manifest_b = _resolve_collection_context(
        report_b, collection_manifest=collection_manifest_b
    )
    bundles_a = walk_report_once(
        report_a,
        kernel_like=kernel_like,
        include_source=True,
        source_top_k=50,
        collection=collection_a,
        ncu_report_module=ncu_report_module,
    )
    bundles_b = walk_report_once(
        report_b,
        kernel_like=kernel_like,
        include_source=True,
        source_top_k=50,
        collection=collection_b,
        ncu_report_module=ncu_report_module,
    )

    # The findings diff runs the same pipeline `ncu-diagnose` runs -- signal
    # scan, shipped-rule reconciliation and all -- so what appeared or
    # disappeared here is exactly what the single-report tool would have said.
    def _diagnoses(
        path: str, collection: Mapping[str, Any]
    ) -> Dict[Tuple[int, int], Dict[str, Any]]:
        payload = diagnose_ncu_report(
            path,
            kernel_like=kernel_like,
            top_kernels=len(bundles_a) + len(bundles_b) + 1,
            # A diff must not convert an omitted tail finding into a
            # disappearance.  Source attribution is disabled here, so this
            # does not increase report traversal cost.
            findings_per_kernel=max(int(findings_per_kernel), 1024),
            include_source=False,
            collection=collection,
            ncu_report_module=ncu_report_module,
        )
        out: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for kernel in payload.get("kernels") or []:
            if isinstance(kernel, dict):
                out[(kernel.get("range_index"), kernel.get("action_index"))] = kernel
        out["__payload__"] = payload  # type: ignore[index]
        return out

    diag_a = _diagnoses(report_a, collection_a)
    diag_b = _diagnoses(report_b, collection_b)
    payload_a: Dict[str, Any] = diag_a.pop("__payload__")  # type: ignore[assignment]
    payload_b: Dict[str, Any] = diag_b.pop("__payload__")  # type: ignore[assignment]

    matches, unmatched_a, unmatched_b = _match_kernels(
        bundles_a,
        bundles_b,
        aliases_a=collection_a.get("kernel_aliases") if isinstance(collection_a.get("kernel_aliases"), Mapping) else None,
        aliases_b=collection_b.get("kernel_aliases") if isinstance(collection_b.get("kernel_aliases"), Mapping) else None,
        logical_kernel_id_a=str(collection_a.get("logical_kernel_id") or ""),
        logical_kernel_id_b=str(collection_b.get("logical_kernel_id") or ""),
    )

    kernels: List[Dict[str, Any]] = []
    blocked_pairs: List[Dict[str, Any]] = []
    for key_a, key_b in matches:
        bundle_a = bundles_a[key_a]
        bundle_b = bundles_b[key_b]
        view_a = MetricView(bundle_a.metrics)
        view_b = MetricView(bundle_b.metrics)
        effective_a, _evidence_a = _effective_collection_context(
            collection_a, bundle_a.metrics
        )
        effective_b, _evidence_b = _effective_collection_context(
            collection_b, bundle_b.metrics
        )

        # Clock guard first. This is compare_measurements' decision, not ours.
        ctx_a = describe_collection_mode(
            source="ncu",
            sm_clock_hz=view_a.get("sm_clock_hz"),
            gpc_clock_hz=view_a.get("gpc_clock_hz"),
            **measurement_collection_context(effective_a),
        )
        ctx_b = describe_collection_mode(
            source="ncu",
            sm_clock_hz=view_b.get("sm_clock_hz"),
            gpc_clock_hz=view_b.get("gpc_clock_hz"),
            **measurement_collection_context(effective_b),
        )
        comparison = compare_measurements(
            ctx_a,
            ctx_b,
            baseline_value=view_a.get("duration_ns"),
            candidate_value=view_b.get("duration_ns"),
            metric="duration_ns",
        )
        comparison = _report_compatibility_guards(
            comparison,
            view_a,
            view_b,
            effective_a,
            effective_b,
            gpu_a=str(payload_a.get("gpu") or payload_a.get("gpu_detected_from_report") or ""),
            gpu_b=str(payload_b.get("gpu") or payload_b.get("gpu_detected_from_report") or ""),
        )
        if not comparison.get("comparable"):
            blocked_pairs.append(
                {"kernel_name": bundle_a.kernel_name, "blockers": list(comparison.get("blockers") or ())}
            )

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
        match = _match_metadata(bundle_a, bundle_b, effective_a, effective_b)
        result_status, result_status_reason = _comparison_status(
            comparison, match, effective_a, effective_b
        )
        catalog_coverage, catalog_changes = _catalog_rows(view_a, view_b)

        kernels.append(
            {
                "kernel_name": bundle_a.kernel_name,
                "match": match,
                "launch_a": {"grid": grid_a, "block": block_a},
                "launch_b": {"grid": grid_b, "block": block_b},
                "clock_comparison": dict(comparison),
                "duration": _duration_block(view_a, view_b, comparison),
                "result_status": result_status,
                "result_status_reason": result_status_reason,
                "verdict": {
                    "a": verdict_a,
                    "b": verdict_b,
                    "changed": bool(verdict_a and verdict_b and verdict_a != verdict_b),
                },
                "findings_diff": _diff_findings(
                    (kernel_diag_a or {}).get("findings"),
                    (kernel_diag_b or {}).get("findings"),
                    diag_a=kernel_diag_a,
                    diag_b=kernel_diag_b,
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
                    "normalised_work": _ratio_rows(view_a, view_b),
                },
                "catalog_coverage": catalog_coverage,
                "catalog_changes": catalog_changes,
                "raw_metric_inventory": _raw_metric_diff(bundle_a.metrics, bundle_b.metrics),
                "pm_sampling": _pm_sampling_diff(bundle_a.source, bundle_b.source),
                "pc_sampling": _pc_hotspot_diff(bundle_a.source, bundle_b.source),
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
        reasons = Counter(
            blocker for item in blocked_pairs for blocker in item.get("blockers") or ()
        )
        guard_summary = (
            f"{len(blocked_pairs)} of {len(matches)} matched kernel(s) fail one or more "
            "compatibility guards; raw-duration deltas are not speedups. "
            + "; ".join(f"{count}x {reason}" for reason, count in reasons.most_common(3))
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

    report_paths_a = tuple(dict.fromkeys((str(report_a), *map(str, repeat_reports_a))))
    report_paths_b = tuple(dict.fromkeys((str(report_b), *map(str, repeat_reports_b))))
    repeat_stats = _repeat_statistics(
        report_paths_a,
        report_paths_b,
        kernels,
        kernel_like=kernel_like,
        ncu_report_module=ncu_report_module,
    )
    if repeat_stats.get("available"):
        rows = list(repeat_stats.get("kernels") or ())
        for kernel, stats in zip(kernels, rows):
            kernel["repeat_statistics"] = stats
            comparison = kernel.get("clock_comparison") or {}
            baseline = comparison.get("baseline") or {}
            candidate = comparison.get("candidate") or {}
            clocks_locked = baseline.get("clocks_locked") is True and candidate.get("clocks_locked") is True
            if (
                kernel.get("result_status") == "INCONCLUSIVE"
                and clocks_locked
                and stats.get("outcome") == "stable_improvement"
            ):
                kernel["result_status"] = "VALID_SPEEDUP"
                kernel["result_status_reason"] = (
                    "matched provenance, locked clocks, and non-overlapping repeat "
                    "duration intervals support a speedup"
                )
            elif kernel.get("result_status") == "INCONCLUSIVE":
                kernel["result_status_reason"] = (
                    f"repeat result: {stats.get('outcome')}; this is not a validated "
                    "speedup without a stable improvement under locked clocks"
                )

    return {
        "report_a": str(report_a),
        "report_b": str(report_b),
        "collection_manifests": {"a": manifest_a, "b": manifest_b},
        "kernel_filter": kernel_like,
        "gpu_a": payload_a.get("gpu") or payload_a.get("gpu_detected_from_report"),
        "gpu_b": payload_b.get("gpu") or payload_b.get("gpu_detected_from_report"),
        "clock_guard": {
            "all_comparable": not blocked_pairs,
            "blocked_kernels": [item["kernel_name"] for item in blocked_pairs],
            "blocked_pairs": blocked_pairs,
            "summary": guard_summary,
        },
        "matched_kernel_count": len(matches),
        "kernels": kernels,
        "repeat_statistics": repeat_stats,
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

    # The compatibility guard leads. Everything below it is read in its light.
    guard = payload.get("clock_guard") or {}
    lines.append("## Clock guard and comparability")
    lines.append("")
    if guard.get("all_comparable"):
        lines.append(str(guard.get("summary") or ""))
    else:
        lines.append(f"**WARNING: {guard.get('summary') or 'clocks differ'}**")
    lines.append("")

    repeats = payload.get("repeat_statistics") or {}
    lines.append("## Repeatability")
    lines.append("")
    if not repeats.get("available"):
        lines.append(str(repeats.get("reason") or "repeatability was not evaluated"))
    else:
        lines.append(str(repeats.get("method") or ""))
        for row in repeats.get("kernels") or []:
            if not isinstance(row, Mapping):
                continue
            a, b = row.get("a") or {}, row.get("b") or {}
            lines.append(
                f"- `{row.get('kernel_name')}`: {row.get('outcome')}; median B/A "
                f"{_fmt_ratio(row.get('median_ratio_b_over_a'))}; n={a.get('count', 0)}/"
                f"{b.get('count', 0)}, MAD ns={_fmt(a.get('mad_ns'))}/{_fmt(b.get('mad_ns'))}."
            )
    lines.append("")

    for kernel in payload.get("kernels") or []:
        if not isinstance(kernel, Mapping):
            continue
        launch_a = kernel.get("launch_a") or {}
        launch_b = kernel.get("launch_b") or {}
        match = kernel.get("match") or {}
        lines.append(
            f"## `{kernel.get('kernel_name', '?')}` "
            f"(grid {_fmt(launch_a.get('grid'))} -> {_fmt(launch_b.get('grid'))}, "
            f"block {_fmt(launch_a.get('block'))} -> {_fmt(launch_b.get('block'))})"
        )
        lines.append("")
        lines.append(
            f"- result: **{kernel.get('result_status', 'INCONCLUSIVE')}** - "
            f"{kernel.get('result_status_reason', '')}"
        )
        lines.append(
            f"- match: {match.get('method', 'unknown')} "
            f"(confidence {match.get('confidence', 'unknown')})"
            + (f", alias `{match.get('alias')}`" if match.get("alias") else "")
        )
        if match.get("note"):
            lines.append(f"- match caveat: {match.get('note')}")
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
        for finding in findings.get("not_evaluated_in_b") or []:
            lines.append(
                f"- not evaluated in B, not claimed disappeared: **{finding.get('title')}** "
                f"[{finding.get('category')}; {finding.get('analysis') or 'unknown coverage'}]"
            )
        for finding in findings.get("not_evaluated_in_a") or []:
            lines.append(
                f"- not evaluated in A, not claimed appeared: **{finding.get('title')}** "
                f"[{finding.get('category')}; {finding.get('analysis') or 'unknown coverage'}]"
            )
        if not (
            findings.get("appeared")
            or findings.get("disappeared")
            or findings.get("severity_changed")
            or findings.get("not_evaluated_in_a")
            or findings.get("not_evaluated_in_b")
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
        _axis_table(lines, "Work-normalised counters", axes.get("normalised_work") or [])

        pm = kernel.get("pm_sampling") or {}
        lines.append("### PM sampling aggregates")
        lines.append("")
        if pm.get("available"):
            _axis_table(lines, "Same-pass PM aggregate features", pm.get("features") or [])
            if pm.get("unmatched_series_a") or pm.get("unmatched_series_b"):
                lines.append(
                    "- unmatched PM series were not compared because pass-group identity changed."
                )
                lines.append("")
        else:
            lines.append(f"Unavailable: {pm.get('reason') or 'no valid PM sampling data'}.")
            lines.append("")

        pc = kernel.get("pc_sampling") or {}
        lines.append("### PC sampling hotspots")
        lines.append("")
        if pc.get("available"):
            _axis_table(lines, "Source-line sample-share changes", (pc.get("hotspots") or [])[:20])
            if pc.get("hotspots_truncated"):
                lines.append(f"_{pc.get('hotspots_truncated')} additional hotspot rows in JSON output._")
                lines.append("")
        else:
            lines.append(f"Unavailable: {pc.get('reason') or 'no valid PC sampling data'}.")
            lines.append("")

        coverage = kernel.get("catalog_coverage") or {}
        if coverage:
            lines.append("### Metric catalog coverage")
            lines.append("")
            lines.append(
                f"{coverage.get('both', 0)} of {coverage.get('total', 0)} catalog keys are "
                f"present on both sides; A-only {coverage.get('a_only', 0)}, "
                f"B-only {coverage.get('b_only', 0)}, absent on both {coverage.get('neither', 0)}."
            )
            lines.append("")
            _axis_table(lines, "Changed or one-sided catalog metrics", kernel.get("catalog_changes") or [])

        raw = kernel.get("raw_metric_inventory") or {}
        if raw:
            lines.append("### Raw metric audit appendix")
            lines.append("")
            lines.append(
                f"numeric metrics A/B/common: {raw.get('numeric_a', 0)}/"
                f"{raw.get('numeric_b', 0)}/{raw.get('common', 0)}; changed common: "
                f"{raw.get('changed_common', 0)}; A-only/B-only: {raw.get('a_only', 0)}/"
                f"{raw.get('b_only', 0)}. JSON retains every changed raw metric; "
                "the Markdown appendix shows the largest 25 relative changes."
            )
            lines.append("")
            _axis_table(lines, "Largest raw metric changes", (raw.get("changes") or [])[:25])

    notes = payload.get("notes") or []
    if notes:
        lines.append("## Honesty notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines) + "\n"
