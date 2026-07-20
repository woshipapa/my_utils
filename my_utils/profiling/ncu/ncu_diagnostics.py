"""Rule engine mapping Nsight Compute metrics onto named performance problems.

The rules below mirror the ones Nsight Compute ships (``SOLBottleneck``,
``CPIStall``, ``LaunchConfiguration``, ``UncoalescedGlobalAccess``, ...) using
their exact thresholds, and add the analyses NVIDIA does not ship: roofline
placement computed from FLOP counters, tile/wave-quantisation checks driven by
kernel names, and per-workload expectations (a GEMM that misses tensor cores is
a finding; an elementwise kernel that does is not).

Two design choices worth knowing:

* **Every finding carries its evidence.** ``Finding.evidence`` holds the metric
  values that produced it, so a report can always show its work rather than
  asserting a conclusion.
* **Speedup estimates are stall-share based**, following GPA (CGO 2021): for a
  stall-elimination fix the ceiling is ``T/(T-M)`` where ``M`` is the matched
  stall time; for a latency-hiding fix the ceiling is capped at 2x because you
  can only fill issue slots you already have. Estimates are labelled as ceilings,
  never as predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .metric_catalog import (
    BENIGN_STALL_KEYS,
    METRIC_CATALOG,
    STALL_REASONS,
    describe_arch,
)

__all__ = [
    "Finding",
    "MetricView",
    "build_metric_view",
    "classify_bottleneck",
    "analyze_stalls",
    "compute_roofline",
    "analyze_occupancy",
    "analyze_launch_config",
    "analyze_coalescing",
    "analyze_shared_memory",
    "analyze_divergence",
    "analyze_spilling",
    "analyze_pipes",
    "analyze_imbalance",
    "analyze_memory_hierarchy",
    "analyze_issue_efficiency",
    "analyze_instruction_mix",
    "hierarchical_roofline",
    "diagnose_kernel",
    "analysis_coverage",
    "SOL_THRESHOLDS",
]


# NVIDIA's own constants, from the shipped rule sources (Nsight Compute 2026.1.1).
SOL_THRESHOLDS = {
    # These four are read verbatim from the SpeedOfLight rule shipped with
    # Nsight Compute 2026.1.1 (`sections/SpeedOfLight.py`), where they appear as
    # balanced_threshold / latency_bound_threshold / no_bound_threshold /
    # waves_threshold. Three different classification schemes circulate in blog
    # posts and vendor docs -- a single `max(sm,mem) < 60` latency gate, a
    # banded >80 / 60-80 / <60 table, and a 60/40 two-axis table. The first two
    # are not rival schemes: they are partial descriptions of this one rule.
    # Its actual decision tree is:
    #
    #   if sm < 80 and mem < 80:
    #       if sm < 60 and mem < 60:      -> small grid (waves < 1) else latency
    #       elif |sm - mem| >= 10:        -> whichever is higher dominates
    #       else:                         -> balanced
    #   else:                             -> saturated; shift work between units
    #
    # This is the primary scheme. The 60/40 two-axis table was here as
    # `compute_bound_compute`/`compute_bound_memory` and has been removed: it
    # appears in no shipped rule, nothing read it, and keeping an unverified
    # threshold beside verified ones invites someone to trust it equally.
    "balanced_delta": 10.0,      # |compute - memory| below this => balanced
    "latency_bound": 60.0,       # both SOL below this => latency issue
    "saturated": 80.0,           # either SOL at/above this => genuinely bound
    "waves_small_grid": 1.0,     # below one wave => small grid
    # CPIStall
    "issue_active_stall_gate": 0.8,
    "stall_share_gate": 0.3,
    # IssueSlotUtilization
    "issue_active_low": 0.6,
    "warp_ratio_gate": 0.8,
    # Occupancy
    "achieved_vs_theoretical_gap_pp": 10.0,
    "theoretical_occupancy_low": 80.0,
    # LaunchConfiguration
    "tail_effect_min_speedup": 0.2,
    # SharedMemoryConflicts
    "bank_conflict_share": 0.10,
    # ThreadDivergence
    "threads_per_inst_low": 24.0,
    # LocalMemoryUsage
    "local_inst_share": 0.10,
    "local_hit_rate_low": 80.0,
    # WorkloadImbalance
    "imbalance_min_speedup": 0.05,
    # Short-kernel / launch-overhead gate (microseconds)
    "short_kernel_us": 10.0,
    # FP64 pipe activity that is suspicious in a BF16/FP16 model
    "fp64_warn": 15.0,
}


@dataclass
class Finding:
    """One diagnosed problem, its evidence, and what to do about it."""

    category: str
    title: str
    summary: str
    severity: str = "info"           # info | low | medium | high
    confidence: str = "medium"       # low | medium | high
    evidence: Dict[str, Any] = field(default_factory=dict)
    actions: Tuple[str, ...] = ()
    # Upper bound on the win, as a multiplier (1.25 == "up to 25% faster").
    speedup_ceiling: Optional[float] = None
    source: str = "heuristic"        # heuristic | ncu_rule | expectation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "title": self.title,
            "summary": self.summary,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "actions": list(self.actions),
            "speedup_ceiling": self.speedup_ceiling,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Metric access
# ---------------------------------------------------------------------------

class MetricView:
    """Read-only view over one kernel's metrics, resolving catalog keys to values.

    Nsight Compute renames metrics across architectures (``_v2`` suffixes on Ada,
    ``dram__bytes_op_read`` on Blackwell), so lookups go through the catalog's
    candidate list rather than a single hard-coded string.
    """

    def __init__(self, metrics: Mapping[str, float]):
        # Normalise once: strip whitespace, keep the raw spelling as the key.
        self._raw: Dict[str, float] = {}
        for name, value in metrics.items():
            if value is None:
                continue
            try:
                self._raw[str(name).strip()] = float(value)
            except (TypeError, ValueError):
                continue
        self._lower = {k.lower(): v for k, v in self._raw.items()}

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def raw(self, metric_name: str) -> Optional[float]:
        """Value of an exact ncu metric name."""
        if metric_name in self._raw:
            return self._raw[metric_name]
        return self._lower.get(metric_name.lower())

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Value for a catalog key, trying each candidate spelling in order."""
        spec = METRIC_CATALOG.get(key)
        if spec is None:
            value = self.raw(key)
            return default if value is None else value
        for name in spec.names:
            value = self.raw(name)
            if value is not None:
                return value
        return default

    def stall(self, reason_key: str) -> Optional[float]:
        """Warp-cycles-per-issued-instruction attributable to one stall reason."""
        reason = STALL_REASONS.get(reason_key)
        if reason is None:
            return None
        return self.raw(reason.metric_name)

    def present_keys(self) -> List[str]:
        """Catalog keys that this report actually carries a value for."""
        return [key for key in METRIC_CATALOG if self.get(key) is not None]

    def metric_names(self) -> List[str]:
        return list(self._raw)


def build_metric_view(metric_stats: Iterable[Mapping[str, Any]], *, stat: str = "avg") -> MetricView:
    """Build a :class:`MetricView` from ``ncu_report_tools`` metric-stat rows.

    Rows look like ``{"metric_name": ..., "avg": ..., "max": ...}``; ``value`` is
    accepted as a fallback for single-valued rows.
    """
    flat: Dict[str, float] = {}
    for row in metric_stats or ():
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("metric_name") or "").strip()
        if not name:
            continue
        for candidate in (stat, "value", "avg", "sum", "max"):
            if candidate in row and row[candidate] is not None:
                try:
                    flat[name] = float(row[candidate])
                except (TypeError, ValueError):
                    continue
                break
    return MetricView(flat)


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Bottleneck classification
# ---------------------------------------------------------------------------

def classify_bottleneck(view: MetricView) -> Dict[str, Any]:
    """Four-way roofline classification following NVIDIA's SOLBottleneck rule.

    Returns a verdict plus the routing hint for which section to read next.
    """
    compute = _pct(view.get("compute_sol"))
    memory = _pct(view.get("memory_sol"))
    waves = view.get("waves_per_sm")
    occupancy = view.get("achieved_occupancy")
    duration_ns = view.get("duration_ns")

    result: Dict[str, Any] = {
        "compute_sol_pct": compute,
        "memory_sol_pct": memory,
        "waves_per_sm": waves,
        "verdict": "unknown",
        "explanation": "",
        "next_section": "",
        "grade": "",
    }

    if compute is None or memory is None:
        result["explanation"] = (
            "Speed-of-light metrics missing; collect the SpeedOfLight section "
            "(it is in every ncu set, including --set basic)."
        )
        return result

    best = max(compute, memory)
    result["grade"] = (
        "excellent" if best >= 80 else
        "good" if best >= 60 else
        "fair" if best >= 40 else
        "poor"
    )

    t = SOL_THRESHOLDS
    if best >= t["saturated"]:
        if compute >= memory:
            result.update(
                verdict="compute_bound",
                explanation=f"SM throughput {compute:.1f}% of peak - the SM pipelines are saturated.",
                next_section="ComputeWorkloadAnalysis",
            )
        else:
            result.update(
                verdict="memory_bound",
                explanation=f"Memory throughput {memory:.1f}% of peak - the memory system is saturated.",
                next_section="MemoryWorkloadAnalysis",
            )
        return result

    if compute < t["latency_bound"] and memory < t["latency_bound"]:
        if waves is not None and waves < t["waves_small_grid"]:
            result.update(
                verdict="small_grid",
                explanation=(
                    f"Only {waves:.2f} waves per SM: the grid cannot fill the GPU, so neither "
                    "compute nor memory can approach peak."
                ),
                next_section="LaunchStats",
            )
            return result
        if duration_ns is not None and duration_ns < t["short_kernel_us"] * 1000.0:
            result.update(
                verdict="launch_bound",
                explanation=(
                    f"Kernel runs for {duration_ns / 1000.0:.1f} us with both throughputs below "
                    f"{t['latency_bound']:.0f}%. At this duration launch overhead dominates; "
                    "diagnose in Nsight Systems, not here."
                ),
                next_section="LaunchStats",
            )
            return result
        detail = ""
        if occupancy is not None and occupancy >= 50.0:
            detail = (
                f" Occupancy is {occupancy:.0f}%, so this is not a warp-supply problem - "
                "look at dependency chains and instruction mix."
            )
        result.update(
            verdict="latency_bound",
            explanation=(
                f"Compute {compute:.1f}% and memory {memory:.1f}% are both below "
                f"{t['latency_bound']:.0f}% of peak: the kernel is waiting, not working.{detail}"
            ),
            next_section="WarpStateStats",
        )
        return result

    if abs(compute - memory) >= t["balanced_delta"]:
        if compute > memory:
            result.update(
                verdict="compute_leaning",
                explanation=f"Compute {compute:.1f}% leads memory {memory:.1f}% but neither is saturated.",
                next_section="ComputeWorkloadAnalysis",
            )
        else:
            result.update(
                verdict="memory_leaning",
                explanation=f"Memory {memory:.1f}% leads compute {compute:.1f}% but neither is saturated.",
                next_section="MemoryWorkloadAnalysis",
            )
        return result

    result.update(
        verdict="balanced",
        explanation=(
            f"Compute {compute:.1f}% and memory {memory:.1f}% are within "
            f"{t['balanced_delta']:.0f} points of each other; both must come down to go faster."
        ),
        next_section="SpeedOfLight_RooflineChart",
    )
    return result


# ---------------------------------------------------------------------------
# Warp stalls
# ---------------------------------------------------------------------------

def analyze_stalls(view: MetricView, *, top_k: int = 6) -> Dict[str, Any]:
    """Rank warp-stall reasons and turn the dominant ones into findings.

    Mirrors NVIDIA's CPIStall rule: a reason is reported when issue activity is
    below 0.8 and the reason accounts for more than 30% of the warp cycles spent
    per issued instruction.
    """
    total_latency = view.get("warp_cycles_per_issued_inst")
    issue_active = view.get("issue_active")

    rows: List[Dict[str, Any]] = []
    for key, reason in STALL_REASONS.items():
        value = view.stall(key)
        if value is None:
            continue
        share = (value / total_latency) if total_latency else None
        rows.append({
            "key": key,
            "display": reason.display,
            "bucket": reason.bucket,
            "cycles_per_issued_inst": value,
            "share_of_warp_latency": share,
            "meaning": reason.meaning,
            "fixes": list(reason.fixes),
            "benign": key in BENIGN_STALL_KEYS,
        })
    rows.sort(key=lambda r: r["cycles_per_issued_inst"], reverse=True)

    actionable = [r for r in rows if not r["benign"]]
    findings: List[Finding] = []

    gate_issue = SOL_THRESHOLDS["issue_active_stall_gate"]
    gate_share = SOL_THRESHOLDS["stall_share_gate"]
    issue_gate_open = issue_active is None or issue_active < gate_issue

    for row in actionable[:top_k]:
        share = row["share_of_warp_latency"]
        if share is None or not issue_gate_open or share <= gate_share:
            continue
        # Stall-elimination ceiling: removing a stall that occupies `share` of
        # warp latency can at best speed the kernel by 1/(1-share).
        ceiling = 1.0 / (1.0 - share) if share < 0.95 else None
        findings.append(Finding(
            category=f"stall_{row['key']}",
            title=f"Warp stalls dominated by {row['display']}",
            summary=(
                f"{row['display']} accounts for {share * 100:.0f}% of warp cycles per issued "
                f"instruction ({row['cycles_per_issued_inst']:.2f} cycles). {row['meaning']}"
            ),
            severity="high" if share > 0.5 else "medium",
            confidence="high",
            evidence={
                "stall_metric": STALL_REASONS[row["key"]].metric_name,
                "cycles_per_issued_inst": row["cycles_per_issued_inst"],
                "share_of_warp_latency": share,
                "warp_cycles_per_issued_inst": total_latency,
                "issue_active": issue_active,
            },
            actions=tuple(row["fixes"]),
            speedup_ceiling=ceiling,
            source="ncu_rule",
        ))

    # Bucket rollup gives the DrGPU-style top-down view.
    buckets: Dict[str, float] = {}
    for row in actionable:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0.0) + row["cycles_per_issued_inst"]

    # Closed accounting, after GCStack (ISCA 2025). Its central argument is
    # that a cycle stack which does not sum to the total cannot say how much of
    # the runtime it has explained, and that priority-based schemes -- which
    # assign a stall cycle to one event by a fixed ranking -- exaggerate memory
    # stalls and hide concurrent ones. That critique does not apply to these
    # counters: each `..._per_issue_active` metric is the average number of
    # warps in that state, so a cycle with several warps stalled for different
    # reasons is already split between them proportionally, which is what
    # GCStack's fractional attribution achieves in simulation.
    #
    # What does apply is the closure check. If the report is missing stall
    # metrics, the shares silently under-account and nothing says so. Measured
    # on a real H100 report the 19 reasons sum to 21.682 against a total of
    # 21.664 -- 0.08% apart.
    accounted = sum(row["cycles_per_issued_inst"] for row in rows
                    if row["cycles_per_issued_inst"] is not None)
    residual = (total_latency - accounted) if total_latency else None
    explained = (accounted / total_latency) if total_latency else None

    # A large residual means reasons are missing from the report, not that the
    # kernel has unexplained stalls. Saying which it is matters.
    if explained is not None and explained < 0.9:
        findings.append(Finding(
            category="measurement_caveat",
            title="Stall accounting does not close",
            summary=(
                f"The stall reasons present account for {explained * 100:.0f}% of "
                f"warp latency ({accounted:.2f} of {total_latency:.2f} cycles per "
                "issued instruction). The rest is not unexplained stalling -- it is "
                "stall reasons this report does not carry, so every share below is "
                "computed against a total the report cannot fully break down."
            ),
            severity="medium",
            confidence="high",
            evidence={"accounted": accounted, "total": total_latency,
                      "explained_share": explained,
                      "reasons_present": len(rows)},
            actions=("Collect --section WarpStateStats in full; partial stall "
                     "coverage makes every stall share an underestimate.",),
            source="heuristic",
        ))

    return {
        "warp_cycles_per_issued_inst": total_latency,
        "issue_active": issue_active,
        "stalls": rows[:top_k],
        "all_stalls": rows,
        "buckets": dict(sorted(buckets.items(), key=lambda kv: kv[1], reverse=True)),
        "dominant_bucket": max(buckets, key=buckets.get) if buckets else "",
        # Closure, so a reader can tell "explained and small" from "not measured".
        "accounted_cycles_per_issued_inst": accounted,
        "residual_cycles_per_issued_inst": residual,
        "explained_share": explained,
        "reasons_present": len(rows),
        "top_stalls_explain": (
            sum(r["cycles_per_issued_inst"] or 0.0 for r in rows[:top_k]) / total_latency
            if total_latency else None
        ),
        "accounting_note": (
            (f"The {len(rows)} stall reasons in this report account for "
             f"{explained * 100:.1f}% of warp latency; the top {min(top_k, len(rows))} "
             f"account for "
             f"{sum(r['cycles_per_issued_inst'] or 0.0 for r in rows[:top_k]) / total_latency * 100:.1f}%.")
            if explained is not None else
            "Warp latency total is absent, so the shares below have no denominator "
            "and cannot be read as fractions of anything."
        ),
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# Roofline
# ---------------------------------------------------------------------------

def _roofline_dtype_basis(
    view: MetricView, *, fp16_flops: Optional[float], fp64_flops: Optional[float],
) -> Tuple[str, str, Tuple[str, ...]]:
    """Decide which precision's peak the roofline is measured against.

    Returns ``(dtype, source, ambiguous_candidates)``. ``source`` records how the
    choice was made, because a dtype read from a hardware counter and one
    inferred from the absence of other evidence deserve different trust.

    The tensor-op counters are per-precision and are the only authoritative
    signal here. When the tensor pipe was busy but no per-precision counter was
    collected, the precision is genuinely unknown: the candidates are returned so
    the caller can report the spread instead of asserting one.
    """
    tensor_ops = {
        "fp8": view.get("tensor_ops_fp8"),
        "fp16": view.get("tensor_ops_fp16"),
        "bf16": view.get("tensor_ops_bf16"),
    }
    observed = {d: v for d, v in tensor_ops.items() if v}
    if observed:
        # More than one can be non-zero in a mixed kernel; the dominant one sets
        # the ceiling, and the others are reported as alternatives.
        dominant = max(observed, key=lambda d: observed[d])
        others = tuple(d for d in observed if d != dominant)
        return dominant, "tensor_op_counters", others

    if fp64_flops:
        return "fp64", "fp64_flop_counters", ()

    # The tensor pipe was busy but no per-precision counter was collected. This
    # is the ambiguous case: `tensor_present` cannot express it, because it is
    # true only when such a counter exists. `pipe_tensor_util` is the signal that
    # MMA work happened at all.
    tensor_util = view.get("pipe_tensor_util")
    if tensor_util and tensor_util > 1.0:
        # Every dtype the MMA pipes support is a candidate, and on Hopper their
        # peaks differ by 4x between fp8 and tf32.
        return "fp16", "tensor_pipe_active_precision_unknown", ("fp8", "fp16", "bf16", "tf32")

    if fp16_flops:
        # Half-precision in the CUDA cores, not on tensor cores.
        return "fp16", "fp16_flop_counters", ()

    return "fp32", "fp32_default_no_tensor_activity", ()


def compute_roofline(view: MetricView, gpu_spec: Any = None) -> Dict[str, Any]:
    """Derive achieved FLOPs, arithmetic intensity and roofline placement.

    FLOP counting follows the weights Nsight Compute's own roofline sections use:
    ``add + mul + 2*fma`` per precision.  ``gpu_spec`` is an optional
    :class:`~my_utils.profiling.hardware.gpu_specs.GpuSpec`; without it the
    absolute ceilings are unknown and only arithmetic intensity is reported.
    """
    duration_ns = view.get("duration_ns")
    dram_bytes = view.get("dram_bytes")
    if dram_bytes is None:
        read = view.get("dram_bytes_read")
        write = view.get("dram_bytes_write")
        if read is not None or write is not None:
            dram_bytes = (read or 0.0) + (write or 0.0)

    def _flops(prefix: str) -> Optional[float]:
        add = view.get(f"flop_{prefix}add")
        mul = view.get(f"flop_{prefix}mul")
        fma = view.get(f"flop_{prefix}fma")
        if add is None and mul is None and fma is None:
            return None
        return (add or 0.0) + (mul or 0.0) + 2.0 * (fma or 0.0)

    fp32_flops = _flops("f")

    # Blackwell packed-FP32 correction, following NVIDIA's SOLFPRoofline rule.
    # Gated on compute capability because the counters exist only on CC 10.0 and
    # 10.3; adding them elsewhere would double-count nothing, but reading them as
    # present on other architectures would misreport why FP32 looks the way it does.
    cc_major = view.get("cc_major")
    cc_minor = view.get("cc_minor")
    packed_fp32_applied = False
    if cc_major is not None and int(cc_major) == 10 and (cc_minor is None or int(cc_minor) in (0, 3)):
        fadd2 = view.get("flop_fadd2")
        fmul2 = view.get("flop_fmul2")
        ffma2 = view.get("flop_ffma2")
        if fadd2 is not None or fmul2 is not None or ffma2 is not None:
            packed = 2.0 * (fadd2 or 0.0) + 2.0 * (fmul2 or 0.0) + 4.0 * (ffma2 or 0.0)
            if packed > 0:
                fp32_flops = (fp32_flops or 0.0) + packed
                packed_fp32_applied = True
    fp16_flops = _flops("h")
    fp64_flops = _flops("d")

    tensor_ops = 0.0
    tensor_present = False
    for key in ("tensor_ops_fp16", "tensor_ops_bf16", "tensor_ops_fp8"):
        value = view.get(key)
        if value is not None:
            tensor_ops += value
            tensor_present = True

    components = [v for v in (fp32_flops, fp16_flops, fp64_flops) if v is not None]
    total_flops: Optional[float] = None
    if components or tensor_present:
        total_flops = sum(components) + (tensor_ops if tensor_present else 0.0)

    result: Dict[str, Any] = {
        "duration_ns": duration_ns,
        "dram_bytes": dram_bytes,
        "flops_fp32": fp32_flops,
        "flops_fp16": fp16_flops,
        "flops_fp64": fp64_flops,
        "tensor_ops": tensor_ops if tensor_present else None,
        "total_flops": total_flops,
        "arithmetic_intensity": None,
        "achieved_tflops": None,
        "achieved_dram_gbps": None,
        "ridge_point": None,
        "roofline_side": "unknown",
        "pct_of_attainable": None,
        # Whether the Blackwell packed-FP32 counters were folded in. Without
        # this a reader cannot distinguish a correct FP32 figure from one that
        # is silently half of what the hardware actually did.
        "packed_fp32_applied": packed_fp32_applied,
        "caveats": [],
        "findings": [],
    }

    if cc_major is not None and int(cc_major) == 10 and not packed_fp32_applied and fp32_flops:
        result["caveats"].append(
            "Compute capability 10.x packs two FP32 ops per instruction, but the packed "
            "counters (fadd2/fmul2/ffma2) are absent from this report, so FP32 FLOPs may be "
            "undercounted by up to 2x. Collect them explicitly with --metrics to close this."
        )

    if total_flops is not None and dram_bytes:
        result["arithmetic_intensity"] = total_flops / dram_bytes
    if total_flops is not None and duration_ns:
        result["achieved_tflops"] = total_flops / (duration_ns * 1e-9) / 1e12
    if dram_bytes and duration_ns:
        result["achieved_dram_gbps"] = dram_bytes / (duration_ns * 1e-9) / 1e9

    if gpu_spec is None:
        return result

    # Physical-limit sanity check. A measurement above hardware peak is always a
    # measurement bug - wrong GPU spec, wrong FLOP model, a kernel that did not
    # do the work it claimed - and reporting it as a result is worse than
    # reporting nothing. Publicised "100x speedup" results have failed exactly
    # here: the number exceeded the hardware ceiling and nobody checked.
    sanity: List[str] = []
    achieved_tflops = result.get("achieved_tflops")
    achieved_gbps = result.get("achieved_dram_gbps")

    best_peak = max(
        (p for p in (gpu_spec.peak_tflops(d) for d in ("fp8", "fp16", "bf16", "tf32", "fp32"))
         if p is not None),
        default=None,
    )
    if achieved_tflops and best_peak and achieved_tflops > best_peak * 1.05:
        sanity.append(
            f"Measured {achieved_tflops:.1f} TFLOP/s exceeds the {gpu_spec.name} peak of "
            f"{best_peak:.1f} TFLOP/s. The FLOP model or the GPU identification is wrong."
        )
    if achieved_gbps and gpu_spec.hbm_bandwidth_gbps and achieved_gbps > gpu_spec.hbm_bandwidth_gbps * 1.05:
        sanity.append(
            f"Measured {achieved_gbps:.0f} GB/s exceeds the {gpu_spec.name} HBM peak of "
            f"{gpu_spec.hbm_bandwidth_gbps:.0f} GB/s. Traffic served from cache is not DRAM "
            "traffic - check which counter produced this."
        )
    result["sanity_violations"] = sanity

    # Which precision's peak to measure against. Getting this wrong scales the
    # efficiency figure directly: on Hopper the FP8 peak is twice the FP16 peak,
    # so grading an FP8 kernel against FP16 reports it as twice as efficient as
    # it is. The tensor-op counters say which pipe actually ran, so they decide
    # this whenever they are present -- the kernel name does not.
    dtype, dtype_source, dtype_alternatives = _roofline_dtype_basis(
        view, fp16_flops=fp16_flops, fp64_flops=fp64_flops,
    )
    ridge = gpu_spec.ridge_point(dtype)
    result["ridge_point"] = ridge
    result["dtype_basis"] = dtype
    result["dtype_basis_source"] = dtype_source

    # When the precision is ambiguous, report the spread rather than picking one
    # and presenting it as fact. A reader who sees a single number will use it.
    if dtype_alternatives:
        spread = {}
        for candidate in dtype_alternatives:
            peak = gpu_spec.peak_tflops(candidate)
            if peak:
                spread[candidate] = peak
        if len(spread) > 1:
            lo, hi = min(spread.values()), max(spread.values())
            result["dtype_ambiguous"] = True
            result["dtype_candidate_peaks_tflops"] = spread
            if hi > lo * 1.1:
                result["caveats"].append(
                    "The precision this kernel ran at could not be determined from the "
                    f"counters, and the candidate peaks differ by {hi / lo:.1f}x "
                    f"({', '.join(f'{k}={v:.0f}' for k, v in sorted(spread.items()))} "
                    f"TFLOP/s). Efficiency is reported against {dtype} and would scale "
                    "inversely with a different choice. Collect the sm__ops_path_tensor_* "
                    "counters, or supply the dtype, to remove the ambiguity."
                )

    ai = result["arithmetic_intensity"]
    if ai is not None and ridge is not None:
        result["roofline_side"] = "compute_bound" if ai > ridge else "memory_bound"
        attainable = gpu_spec.attainable_tflops(ai, dtype)
        result["attainable_tflops"] = attainable
        achieved = result["achieved_tflops"]
        if attainable and achieved:
            result["pct_of_attainable"] = 100.0 * achieved / attainable

    # The FLOP counters above are CUDA-core (SASS) counters. A tensor-core kernel
    # does almost all of its math through the MMA pipes, which these do not see,
    # so the FLOP total is a floor rather than the truth whenever the tensor pipe
    # was busy but no sm__ops_path_tensor_* metric was collected.
    tensor_util = view.get("pipe_tensor_util")
    flops_undercounted = bool(tensor_util and tensor_util > 1.0 and not tensor_present)
    result["flops_undercounted"] = flops_undercounted
    if flops_undercounted:
        result["flops_undercount_reason"] = (
            f"Tensor pipe was {tensor_util:.0f}% active but no sm__ops_path_tensor_* metric was "
            "collected, so the FLOP total counts only CUDA-core instructions and is a lower bound."
        )

    findings: List[Finding] = []
    if sanity:
        findings.append(Finding(
            category="measurement_above_physical_limit",
            title="Measurement exceeds hardware limits - do not trust this number",
            summary=(
                " ".join(sanity)
                + " Every downstream conclusion drawn from this measurement is suspect until "
                "the discrepancy is explained."
            ),
            severity="high",
            confidence="high",
            evidence={"achieved_tflops": achieved_tflops, "achieved_dram_gbps": achieved_gbps,
                      "gpu": gpu_spec.name, "peak_tflops": best_peak,
                      "hbm_peak_gbps": gpu_spec.hbm_bandwidth_gbps},
            actions=(
                "Confirm the GPU identification matches the machine that produced the report.",
                "Check the FLOP model: tensor-core work is not visible to the CUDA-core counters.",
                "For bandwidth, confirm the counter measures DRAM rather than cache traffic.",
            ),
        ))
        # Refuse to draw roofline conclusions on top of an impossible measurement.
        result["findings"] = findings
        return result

    pct = result.get("pct_of_attainable")
    compute_sol = view.get("compute_sol")
    memory_sol = view.get("memory_sol")
    # Never claim a kernel is far below its ceiling when speed-of-light already
    # says a unit is saturated - in that case the FLOP model is what is wrong,
    # not the kernel.
    sol_says_saturated = max(
        [v for v in (compute_sol, memory_sol) if v is not None], default=0.0
    ) >= SOL_THRESHOLDS["latency_bound"]

    if pct is not None and pct < 50.0 and not flops_undercounted and not sol_says_saturated:
        findings.append(Finding(
            category="below_roofline",
            title="Kernel sits well below its roofline ceiling",
            summary=(
                f"Achieved {result['achieved_tflops']:.1f} TFLOP/s against an attainable "
                f"{result['attainable_tflops']:.1f} TFLOP/s at arithmetic intensity "
                f"{ai:.1f} FLOP/byte ({pct:.0f}% of ceiling). Being far below *both* roofs "
                "points at latency or occupancy rather than bandwidth or math."
            ),
            severity="medium",
            confidence="medium",
            evidence={k: result[k] for k in
                      ("achieved_tflops", "attainable_tflops", "arithmetic_intensity", "ridge_point")},
            actions=(
                "Check warp stalls and occupancy before optimising math or memory layout.",
                "Confirm the kernel is large enough to fill the GPU (waves per SM >= 1).",
            ),
        ))
    elif pct is not None and pct < 50.0 and flops_undercounted:
        findings.append(Finding(
            category="roofline_needs_tensor_counters",
            title="Roofline is unreliable: tensor-core FLOPs were not collected",
            summary=(
                f"The measured {result['achieved_tflops']:.1f} TFLOP/s counts only CUDA-core "
                f"instructions while the tensor pipe ran at {tensor_util:.0f}%. Collect "
                "sm__ops_path_tensor_* (the SpeedOfLight_HierarchicalTensorRooflineChart section) "
                "before drawing any roofline conclusion for this kernel."
            ),
            severity="info",
            confidence="high",
            evidence={"achieved_tflops_cuda_core_only": result["achieved_tflops"],
                      "pipe_tensor_util_pct": tensor_util},
            actions=("Re-profile with --set full or add the hierarchical tensor roofline section.",),
        ))
    result["findings"] = findings
    return result


# ---------------------------------------------------------------------------
# Occupancy and launch configuration
# ---------------------------------------------------------------------------

_LIMITER_ADVICE = {
    "registers": (
        "Occupancy is register limited.",
        ("Add __launch_bounds__ or -maxrregcount to cap register usage.",
         "Shorten live ranges; recompute cheap values instead of holding them.",
         "Watch for spilling after capping registers - trading spills for occupancy usually loses."),
    ),
    "shared_mem": (
        "Occupancy is shared-memory limited.",
        ("Shrink the tile, or reuse one shared buffer for multiple phases.",
         "Check the carveout: cudaFuncAttributePreferredSharedMemoryCarveout."),
    ),
    "blocks": (
        "Occupancy is limited by the blocks-per-SM hardware cap.",
        ("Use larger blocks so fewer of them fill the SM.",),
    ),
    "warps": (
        "Occupancy is limited by block size (warp slots).",
        ("Raise threads per block; 128-256 is the usual sweet spot.",),
    ),
    "barriers": (
        "Occupancy is limited by barrier resources.",
        ("Reduce the number of distinct barriers or cooperating groups per block.",),
    ),
}


def _latency_hiding_is_failing(view: MetricView) -> Optional[bool]:
    """Whether low occupancy is actually costing this kernel anything.

    Low occupancy is only a problem when the schedulers are running out of work.
    CUTLASS-class GEMMs deliberately run at 1-2 CTAs per SM with deep software
    pipelines and hit peak anyway, and Volkov's measurements showed CUBLAS
    getting *faster* while occupancy dropped from 67% to 33%. So an occupancy
    finding is gated on two independent signals:

    * the schedulers are failing to issue (``issue_active`` below its gate), and
    * no throughput unit is already saturated.

    Returns ``None`` when neither signal was collected, in which case the caller
    should stay quiet rather than guess.
    """
    issue_active = view.get("issue_active")
    compute_sol = view.get("compute_sol")
    memory_sol = view.get("memory_sol")

    sols = [v for v in (compute_sol, memory_sol) if v is not None]
    if sols and max(sols) >= SOL_THRESHOLDS["latency_bound"]:
        # A unit is already near its ceiling; more warps cannot help.
        return False
    if issue_active is not None:
        return issue_active < SOL_THRESHOLDS["issue_active_low"]
    if sols:
        return True  # nothing saturated, but we cannot see issue activity
    return None


def _is_warp_specialized(view: MetricView) -> bool:
    """Detect a kernel that reallocates registers between warpgroups.

    Hopper/Blackwell warp-specialized kernels call ``setmaxnreg`` so the producer
    and consumer warpgroups hold very different register counts (FlashAttention-3
    uses 24-56 for producers and 160-256 for consumers). The single
    ``launch__registers_per_thread`` a profiler reports is a weighted artifact of
    those, so the standard occupancy model does not apply.
    """
    if view.get("inst_pipe_gmma") or view.get("pipe_tma_util"):
        return True
    return False


def analyze_occupancy(view: MetricView) -> Dict[str, Any]:
    """Identify the binding occupancy limiter and the achieved/theoretical gap."""
    achieved = view.get("achieved_occupancy")
    theoretical = view.get("theoretical_occupancy")

    limiters = {
        "blocks": view.get("occupancy_limit_blocks"),
        "registers": view.get("occupancy_limit_registers"),
        "shared_mem": view.get("occupancy_limit_shared_mem"),
        "warps": view.get("occupancy_limit_warps"),
        "barriers": view.get("occupancy_limit_barriers"),
    }
    present = {k: v for k, v in limiters.items() if v is not None and v > 0}
    binding = min(present, key=present.get) if present else ""

    findings: List[Finding] = []
    t = SOL_THRESHOLDS

    latency_failing = _latency_hiding_is_failing(view)
    warp_specialized = _is_warp_specialized(view)

    if warp_specialized:
        # setmaxnreg makes registers-per-thread a weighted average across
        # warpgroups, so neither the occupancy model nor register advice holds.
        return {
            "achieved_occupancy_pct": achieved,
            "theoretical_occupancy_pct": theoretical,
            "limiters": present,
            "binding_limiter": binding,
            "warp_specialized": True,
            "occupancy_model_applicable": False,
            "note": (
                "Warp-specialized kernel: producer and consumer warpgroups reallocate registers "
                "with setmaxnreg, so the reported registers-per-thread is a weighted artifact and "
                "the occupancy model does not apply. Judge this kernel on pipe utilisation instead."
            ),
            "findings": [],
        }

    if (
        theoretical is not None
        and theoretical < t["theoretical_occupancy_low"]
        and binding
        and latency_failing is not False
    ):
        title, actions = _LIMITER_ADVICE.get(binding, ("Occupancy is limited.", ()))
        hedge = "" if latency_failing else (
            " Note that no scheduler-starvation signal was collected, so this may be a deliberate "
            "design point rather than a defect."
        )
        findings.append(Finding(
            category=f"occupancy_limited_{binding}",
            title=f"Theoretical occupancy capped at {theoretical:.0f}% by {binding}",
            summary=(
                f"{title} The launch configuration alone caps occupancy at {theoretical:.0f}%; "
                f"the binding resource allows {present[binding]:.0f} warps.{hedge}"
            ),
            severity="medium" if theoretical < 50 else "low",
            confidence="high" if latency_failing else "low",
            evidence={
                "theoretical_occupancy_pct": theoretical,
                "achieved_occupancy_pct": achieved,
                "limiters": present,
                "binding_limiter": binding,
                "registers_per_thread": view.get("registers_per_thread"),
                "shared_mem_per_block": view.get("shared_mem_per_block"),
                "block_size": view.get("block_size"),
            },
            actions=actions,
            source="ncu_rule",
        ))

    if achieved is not None and theoretical is not None:
        gap = theoretical - achieved
        if gap > t["achieved_vs_theoretical_gap_pp"]:
            findings.append(Finding(
                category="occupancy_achieved_gap",
                title="Achieved occupancy trails theoretical occupancy",
                summary=(
                    f"Achieved {achieved:.0f}% against a theoretical {theoretical:.0f}% "
                    f"({gap:.0f} points short). The launch config is not the constraint - "
                    "this is scheduling overhead, a tail wave, or imbalance between warps."
                ),
                severity="medium",
                confidence="high",
                evidence={"achieved_occupancy_pct": achieved,
                          "theoretical_occupancy_pct": theoretical,
                          "gap_pp": gap,
                          "waves_per_sm": view.get("waves_per_sm")},
                actions=(
                    "Check waves per SM for a partial tail wave.",
                    "Look for per-block work imbalance (variable-length sequences, ragged batches).",
                ),
                speedup_ceiling=(theoretical / achieved) if achieved > 0 else None,
                source="ncu_rule",
            ))

    return {
        "achieved_occupancy_pct": achieved,
        "theoretical_occupancy_pct": theoretical,
        "limiters": present,
        "binding_limiter": binding,
        "findings": findings,
    }


def analyze_launch_config(
    view: MetricView, *, gemm_shape: Any = None,
    problem_shape: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Block-size, small-grid, tail-effect and tile-quantisation checks.

    ``gemm_shape`` is an optional
    :class:`~my_utils.profiling.sources.kernel_taxonomy.GemmShape`; when present
    the tile dimensions enable tile-quantisation reasoning that grid size alone
    cannot support.
    """
    grid = view.get("grid_size")
    block = view.get("block_size")
    waves = view.get("waves_per_sm")
    achieved = view.get("achieved_occupancy")
    theoretical = view.get("theoretical_occupancy")

    # A green context owns only a subset of the device's SMs, so comparing the
    # grid against the *device* SM count would understate how full the machine
    # is by the partition ratio. launch__sm_count is the launch's own view and is
    # therefore preferred over the device attribute.
    launch_sm_count = view.raw("launch__sm_count")
    device_sm_count = view.get("device_sm_count")
    sm_count = launch_sm_count if launch_sm_count else device_sm_count
    in_green_context = bool(view.get("uses_green_context"))
    partitioned = bool(
        in_green_context
        or (launch_sm_count and device_sm_count and launch_sm_count < device_sm_count)
    )

    findings: List[Finding] = []
    t = SOL_THRESHOLDS

    if block is not None and block % 32 != 0:
        findings.append(Finding(
            category="block_size_not_warp_multiple",
            title=f"Block size {int(block)} is not a multiple of 32",
            summary=(
                f"A block of {int(block)} threads leaves {32 - int(block) % 32} lanes of the last "
                "warp masked off for the whole kernel."
            ),
            severity="medium",
            confidence="high",
            evidence={"block_size": block},
            actions=("Round the block size to a multiple of 32; 128-256 threads is the usual range.",),
            source="ncu_rule",
        ))

    if grid is not None and sm_count and grid < sm_count:
        scope = (
            f"its green-context partition of {int(sm_count)} SMs"
            if partitioned else f"a {int(sm_count)}-SM GPU"
        )
        findings.append(Finding(
            category="small_grid",
            title=f"Grid of {int(grid)} blocks cannot fill {int(sm_count)} SMs",
            summary=(
                f"With {int(grid)} blocks on {scope}, at least "
                f"{int(sm_count - grid)} SMs are idle for the entire kernel."
            ),
            severity="high",
            confidence="high",
            evidence={"grid_size": grid, "sm_count": sm_count, "waves_per_sm": waves,
                      "sm_count_source": "launch" if launch_sm_count else "device",
                      "green_context": in_green_context},
            actions=(
                "Expose more parallelism (larger batch, split reductions, persistent kernels).",
                "Fuse with neighbouring kernels so one launch covers more work.",
            ),
            speedup_ceiling=(sm_count / grid) if grid else None,
            source="ncu_rule",
        ))

    # Tail effect: a partial final wave costs a whole wave of time.
    if waves is not None and waves >= 1.0:
        whole = math.floor(waves)
        frac = waves - whole
        if frac > 0:
            potential = 1.0 / (whole + 1.0)
            occupancy_confirms = (
                achieved is not None and theoretical is not None and achieved < 0.8 * theoretical
            )
            if potential >= t["tail_effect_min_speedup"]:
                findings.append(Finding(
                    category="tail_wave_quantization",
                    title=f"Partial tail wave after {whole} full wave(s)",
                    summary=(
                        f"The grid is {waves:.2f} waves: {whole} full plus a {frac:.2f} tail. "
                        f"The tail occupies a whole wave of time while using only {frac * 100:.0f}% "
                        f"of the machine, so up to {potential * 100:.0f}% of the runtime is wasted."
                        + ("" if occupancy_confirms else
                           " (Achieved occupancy does not clearly confirm this - treat as a hypothesis.)")
                    ),
                    severity="medium" if potential >= 0.25 else "low",
                    confidence="high" if occupancy_confirms else "medium",
                    evidence={"waves_per_sm": waves, "full_waves": whole, "tail_fraction": frac,
                              "achieved_occupancy_pct": achieved,
                              "theoretical_occupancy_pct": theoretical},
                    actions=(
                        "Size the grid to a whole number of waves.",
                        "Shrink the tile so more, smaller blocks divide evenly across SMs.",
                        "Consider a persistent kernel with a grid of exactly one wave.",
                    ),
                    speedup_ceiling=1.0 / (1.0 - potential),
                    source="ncu_rule",
                ))

    result: Dict[str, Any] = {
        "grid_size": grid,
        "block_size": block,
        "waves_per_sm": waves,
        "sm_count": sm_count,
        "findings": findings,
    }

    # Tile quantisation needs the GEMM tile shape from the kernel name.
    if gemm_shape is not None and getattr(gemm_shape, "tile_m", None) and getattr(gemm_shape, "tile_n", None):
        result["tile_m"] = gemm_shape.tile_m
        result["tile_n"] = gemm_shape.tile_n
        if grid and sm_count:
            tiles = float(grid)
            waves_from_tiles = tiles / float(sm_count)
            result["tiles"] = tiles
            result["waves_from_tiles"] = waves_from_tiles

        # Tile quantisation: the problem shape does not divide evenly by the
        # tile shape, so the boundary tiles compute mostly padding. This is
        # distinct from wave quantisation (which is about tiles vs SMs) and it
        # is why a GEMM's throughput is not monotonic in its size -- growing M
        # from 4096 to 4097 adds a whole row of tiles doing 1/128th of a tile's
        # useful work.
        #
        # The kernel symbol carries the TILE shape and never the PROBLEM shape,
        # so this cannot be computed from the name alone. `problem_shape` has to
        # come from NVTX, from the framework, or from the caller. Without it we
        # say what is missing rather than quietly skipping the check.
        if not problem_shape:
            findings.append(Finding(
                category="tile_quantization",
                title="Tile quantisation could not be checked",
                summary=(
                    f"The symbol gives a {gemm_shape.tile_m}x{gemm_shape.tile_n} tile but "
                    "carries no problem shape, and no problem shape was supplied. Whether "
                    "the boundary tiles are computing padding is therefore unknown -- this "
                    "is an unasked question, not a clean result."
                ),
                severity="info",
                confidence="high",
                evidence={"tile_m": gemm_shape.tile_m, "tile_n": gemm_shape.tile_n,
                          "problem_shape": None},
                actions=(
                    "Pass problem_shape={'m': M, 'n': N, 'k': K}, or wrap the call in an "
                    "NVTX range carrying the shape, to enable this check.",
                ),
                source="heuristic",
            ))
        for dim, tile in (("m", gemm_shape.tile_m), ("n", gemm_shape.tile_n)):
            problem = (problem_shape or {}).get(dim)
            if not problem or not tile:
                continue
            tiles_needed = math.ceil(float(problem) / float(tile))
            covered = tiles_needed * float(tile)
            waste = (covered - float(problem)) / covered if covered else 0.0
            result[f"tile_{dim}_waste"] = waste
            # One-eighth wasted is roughly where the padding costs more than the
            # simplest fix (padding the problem up to a tile multiple) saves.
            if waste > 0.125:
                findings.append(Finding(
                    category="tile_quantization",
                    title=f"Problem dimension {dim.upper()} does not fill its tiles",
                    summary=(
                        f"{dim.upper()}={int(problem)} against a tile of {int(tile)} needs "
                        f"{tiles_needed} tiles covering {int(covered)}, so "
                        f"{waste * 100:.0f}% of the work in that dimension is padding. "
                        "Throughput is not monotonic in problem size when this happens."
                    ),
                    severity="medium" if waste > 0.25 else "low",
                    confidence="medium",
                    evidence={
                        f"{dim}": problem, f"tile_{dim}": tile,
                        "tiles_needed": tiles_needed, "covered": covered,
                        "waste_fraction": round(waste, 4),
                        "shape_source": "parsed from the kernel symbol",
                    },
                    actions=(
                        f"Pad {dim.upper()} up to a multiple of {int(tile)}, or pick a "
                        f"kernel whose tile divides {int(problem)} evenly.",
                    ),
                    # The ceiling is the padding you stop computing.
                    speedup_ceiling=round(1.0 / (1.0 - waste), 3) if waste < 0.9 else None,
                    source="heuristic",
                ))
    return result


# ---------------------------------------------------------------------------
# Memory access patterns
# ---------------------------------------------------------------------------

def analyze_coalescing(view: MetricView) -> Dict[str, Any]:
    """Global-access coalescing: sectors per request, bytes per sector, excess sectors."""
    findings: List[Finding] = []

    excessive = view.get("l2_sectors_global_excessive")
    actual = view.get("l2_sectors_global_actual")
    ideal = view.get("l2_sectors_global_ideal")
    if excessive is not None and excessive > 0:
        share = (excessive / actual) if actual else None
        findings.append(Finding(
            category="uncoalesced_global_access",
            title="Uncoalesced global accesses",
            summary=(
                f"{excessive:,.0f} sectors above the ideal"
                + (f" ({share * 100:.0f}% of all global sectors)" if share else "")
                + ". Each excess sector is bandwidth spent moving bytes the warp never uses."
            ),
            severity="high" if (share or 0) > 0.3 else "medium",
            confidence="high",
            evidence={"sectors_actual": actual, "sectors_ideal": ideal, "sectors_excessive": excessive},
            actions=(
                "Make consecutive threads read consecutive addresses.",
                "Change the data layout (AoS to SoA) or transpose during a shared-memory staging step.",
                "Vectorise to 128-bit accesses once the pattern is contiguous.",
            ),
            speedup_ceiling=(1.0 / (1.0 - share)) if share and share < 0.95 else None,
            source="ncu_rule",
        ))

    for key, label in (("global_ld_sectors_per_request", "load"),
                       ("global_st_sectors_per_request", "store")):
        value = view.get(key)
        if value is None:
            continue
        spec = METRIC_CATALOG[key]
        if spec.warn_above is not None and value > spec.warn_above:
            findings.append(Finding(
                category=f"uncoalesced_global_{label}",
                title=f"Global {label}s touch {value:.1f} sectors per request",
                summary=(
                    f"A fully coalesced fp32 warp {label} touches {spec.ideal:.0f} sectors; this kernel "
                    f"touches {value:.1f}, i.e. {value / (spec.ideal or 1):.1f}x the necessary traffic."
                ),
                severity="high" if value > 16 else "medium",
                confidence="high",
                evidence={"metric": spec.names[0], "sectors_per_request": value, "ideal": spec.ideal},
                actions=("Rework the lane-to-address mapping so a warp covers contiguous 128-byte lines.",),
            ))

    for key, label in (("global_ld_bytes_per_sector", "load"),
                       ("global_st_bytes_per_sector", "store")):
        value = view.get(key)
        if value is None:
            continue
        spec = METRIC_CATALOG[key]
        if spec.warn_below is not None and 0 < value < spec.warn_below:
            waste = 1.0 - value / 32.0
            findings.append(Finding(
                category=f"sparse_global_{label}",
                title=f"Only {value:.1f} of 32 bytes per {label} sector are used",
                summary=(
                    f"{waste * 100:.0f}% of the bandwidth spent on global {label}s moves bytes the "
                    "kernel never reads. Typical of strided or predicated access."
                ),
                severity="medium",
                confidence="high",
                evidence={"metric": spec.names[0], "bytes_per_sector": value, "wasted_fraction": waste},
                actions=("Coalesce or compact the access; consider gathering into shared memory first.",),
            ))

    l1_hit = view.get("l1_hit_rate")
    l2_hit = view.get("l2_hit_rate")
    # TMA moves data from global straight into shared memory, bypassing L1. A
    # correctly TMA-driven kernel therefore reads 0% L1 hit rate, and a real
    # CUTLASS Hopper GEMM does exactly that - so the cache-locality finding has
    # to be suppressed when the TMA pipe was active, or it fires on the
    # best-tuned kernels in the trace.
    tma_active = view.get("pipe_tma_util") or 0.0
    if (
        l1_hit is not None and l2_hit is not None
        and l1_hit < 50.0 and l2_hit < 50.0
        and tma_active <= 0.0
    ):
        findings.append(Finding(
            category="poor_cache_locality",
            title="Both L1 and L2 hit rates are low",
            summary=(
                f"L1 {l1_hit:.0f}% / L2 {l2_hit:.0f}%: nearly every access reaches DRAM. "
                "Either the working set genuinely exceeds cache, or the traversal order destroys locality."
            ),
            severity="medium",
            confidence="medium",
            evidence={"l1_hit_rate_pct": l1_hit, "l2_hit_rate_pct": l2_hit},
            actions=("Tile the computation so the working set fits L2.",
                     "Reorder loops/blocks for locality (threadblock swizzle for GEMM-like kernels)."),
        ))

    return {
        "sectors_per_ld_request": view.get("global_ld_sectors_per_request"),
        "bytes_per_ld_sector": view.get("global_ld_bytes_per_sector"),
        "l1_hit_rate_pct": l1_hit,
        "l2_hit_rate_pct": l2_hit,
        "excessive_sectors": excessive,
        "findings": findings,
    }


def analyze_shared_memory(view: MetricView) -> Dict[str, Any]:
    """Shared-memory bank conflicts, using NVIDIA's 10%-of-wavefronts gate."""
    findings: List[Finding] = []
    details: Dict[str, Any] = {}

    for op in ("ld", "st"):
        conflicts = view.get(f"shared_bank_conflicts_{op}")
        wavefronts = view.get(f"shared_wavefronts_{op}")
        if conflicts is None or not wavefronts:
            continue
        share = conflicts / wavefronts
        details[f"{op}_conflicts"] = conflicts
        details[f"{op}_wavefronts"] = wavefronts
        details[f"{op}_conflict_share"] = share
        if share >= SOL_THRESHOLDS["bank_conflict_share"]:
            findings.append(Finding(
                category=f"shared_bank_conflicts_{op}",
                title=f"Shared-memory {op} bank conflicts on {share * 100:.0f}% of wavefronts",
                summary=(
                    f"{conflicts:,.0f} conflicts across {wavefronts:,.0f} {op} wavefronts. Conflicting "
                    "lanes serialise, so the shared-memory pipe delivers a fraction of its bandwidth."
                ),
                severity="medium",
                confidence="high",
                evidence={f"bank_conflicts_{op}": conflicts, f"wavefronts_{op}": wavefronts,
                          "conflict_share": share},
                actions=(
                    "Pad the shared array's leading dimension (classic +1 element padding).",
                    "Swizzle the shared layout so a warp's lanes hit distinct banks.",
                    "For tensor-core tiles, use the CUTLASS-style XOR swizzle instead of padding.",
                ),
                speedup_ceiling=1.0 / (1.0 - min(share, 0.9)),
                source="ncu_rule",
            ))

    nway = view.get("shared_conflicts_nway")
    if nway is not None and nway > 1.5:
        details["nway"] = nway

    details["findings"] = findings
    return details


def analyze_divergence(view: MetricView) -> Dict[str, Any]:
    """Thread divergence and predication, using NVIDIA's 24-of-32 gate."""
    findings: List[Finding] = []
    threads_per_inst = view.get("warp_exec_efficiency")
    pred_on = view.get("warp_exec_efficiency_pred_on")
    branch_eff = view.get("branch_efficiency")

    gate = SOL_THRESHOLDS["threads_per_inst_low"]
    worst = min([v for v in (threads_per_inst, pred_on) if v is not None], default=None)
    if worst is not None and worst < gate:
        findings.append(Finding(
            category="thread_divergence",
            title=f"Only {worst:.1f} of 32 threads active per instruction",
            summary=(
                f"Warps execute with {worst:.1f}/32 lanes active on average "
                f"({worst / 32 * 100:.0f}% efficiency). Divergent branches or predication are "
                "wasting most of each warp's width."
            ),
            severity="high" if worst < 16 else "medium",
            confidence="high",
            evidence={"threads_per_inst": threads_per_inst,
                      "threads_per_inst_pred_on": pred_on,
                      "branch_efficiency_pct": branch_eff},
            actions=(
                "Group data so threads in a warp take the same path (sort/bucket by branch condition).",
                "Replace short divergent branches with predicated arithmetic or select().",
                "Move the branch up to block granularity where possible.",
            ),
            speedup_ceiling=32.0 / worst if worst > 0 else None,
            source="ncu_rule",
        ))

    return {
        "threads_per_inst": threads_per_inst,
        "threads_per_inst_pred_on": pred_on,
        "branch_efficiency_pct": branch_eff,
        "findings": findings,
    }


def analyze_spilling(view: MetricView) -> Dict[str, Any]:
    """Register spilling to local memory."""
    findings: List[Finding] = []
    local_ld = view.get("local_ld_inst") or 0.0
    local_st = view.get("local_st_inst") or 0.0
    total_inst = view.get("inst_executed")
    regs = view.get("registers_per_thread")
    local_hit = view.get("local_ld_hit_rate")

    local_total = local_ld + local_st
    share = (local_total / total_inst) if total_inst else None

    if local_total > 0:
        severity = "medium"
        extra = ""
        if share is not None and share >= SOL_THRESHOLDS["local_inst_share"]:
            severity = "high"
            extra = f" Local traffic is {share * 100:.0f}% of all executed instructions."
        if local_hit is not None and local_hit < SOL_THRESHOLDS["local_hit_rate_low"]:
            severity = "high"
            extra += (
                f" Worse, only {local_hit:.0f}% of local loads hit L1, so spills are reaching "
                "L2 or DRAM."
            )
        findings.append(Finding(
            category="register_spilling",
            title="Register spilling to local memory",
            summary=(
                f"{local_ld:,.0f} local loads and {local_st:,.0f} local stores with "
                f"{regs:.0f} registers/thread." if regs else
                f"{local_ld:,.0f} local loads and {local_st:,.0f} local stores."
            ) + extra,
            severity=severity,
            confidence="high",
            evidence={"local_ld_inst": local_ld, "local_st_inst": local_st,
                      "local_share_of_inst": share, "registers_per_thread": regs,
                      "local_ld_hit_rate_pct": local_hit},
            actions=(
                "Shorten live ranges or recompute values instead of keeping them alive.",
                "Reduce the per-thread tile so the working set fits in registers.",
                "If __launch_bounds__ is capping registers, weigh the spills against the occupancy gained.",
                "Check for dynamically indexed local arrays - those always spill.",
            ),
            source="ncu_rule",
        ))

    return {"local_ld_inst": local_ld, "local_st_inst": local_st,
            "local_share_of_inst": share, "registers_per_thread": regs,
            "findings": findings}


def analyze_pipes(view: MetricView) -> Dict[str, Any]:
    """Which execution pipe is busiest, plus FP64-in-a-BF16-model detection."""
    pipes = {
        "alu": view.get("pipe_alu_util"),
        "fma": view.get("pipe_fma_util"),
        "fp64": view.get("pipe_fp64_util"),
        "tensor": view.get("pipe_tensor_util"),
        "shared": view.get("shared_pipe_util"),
        "lsu": view.get("pipe_lsu_util"),
        "tma": view.get("pipe_tma_util"),
        "tc": view.get("pipe_tc_util"),
    }
    present = {k: v for k, v in pipes.items() if v is not None}
    findings: List[Finding] = []
    busiest = max(present, key=present.get) if present else ""

    fp64 = pipes.get("fp64")
    if fp64 is not None and fp64 > SOL_THRESHOLDS.get("fp64_warn", 15.0):
        findings.append(Finding(
            category="unexpected_fp64",
            title=f"FP64 pipe is {fp64:.0f}% utilised",
            summary=(
                "Significant FP64 activity. In a BF16/FP16 training kernel this usually comes from "
                "stray double literals, math functions without the f suffix, or an accidental "
                "float64 tensor - and FP64 runs at a small fraction of the FP32 rate on data-centre parts."
            ),
            severity="medium",
            confidence="medium",
            evidence={"pipe_fp64_util_pct": fp64, "pipe_fma_util_pct": pipes.get("fma")},
            actions=("Audit floating-point literals (1.0 vs 1.0f) and math calls (exp vs expf).",
                     "Check for a float64 tensor sneaking in from numpy or a Python scalar."),
        ))

    if present and busiest and present[busiest] >= 80.0:
        findings.append(Finding(
            category="pipe_saturated",
            title=f"{busiest.upper()} pipe is saturated at {present[busiest]:.0f}%",
            summary=f"The {busiest} pipeline is the throughput limiter for this kernel.",
            severity="info",
            confidence="high",
            evidence={"pipes": present},
            actions=(f"Move work off the {busiest} pipe or reduce the instruction count it sees.",),
        ))

    return {"pipes": present, "busiest": busiest, "findings": findings}


# Each instruction pipe, and what work landing there means. The XU and ADU
# entries are the ones that pay for themselves: transcendental and
# address-arithmetic pressure is invisible in a SpeedOfLight rollup, because
# neither pipe is what "compute throughput" measures.
_INSTRUCTION_PIPES: Tuple[Tuple[str, str, str], ...] = (
    ("inst_pipe_fma", "FMA (fp32/int)", ""),
    ("inst_pipe_alu", "ALU (integer, logic)", ""),
    ("inst_pipe_fp64", "FP64", "FP64 is 1/64 rate on consumer and 1/2 on datacenter parts; "
                              "unintended doubles are a common accidental slowdown."),
    ("inst_pipe_fp16", "FP16 in the FMA pipe", "Half precision running in FMA rather than "
                                               "on tensor cores leaves most of the peak unused."),
    ("inst_pipe_xu", "XU (transcendental: rsqrt, exp, sin)",
     "The XU pipe is roughly a quarter rate. Softmax and normalisation land here, "
     "and the cost does not appear in the compute SOL figure."),
    ("inst_pipe_lsu", "LSU (load/store issue)",
     "A saturated LSU is an instruction-issue limit, not a bandwidth limit: the fix "
     "is fewer, wider accesses, not faster memory."),
    ("inst_pipe_adu", "ADU (address arithmetic)",
     "Address computation dominating usually means indexing that could be strength-"
     "reduced or hoisted out of the inner loop."),
    ("inst_pipe_tex", "TEX (texture/surface)", ""),
    ("inst_pipe_uniform", "Uniform datapath", ""),
    ("inst_pipe_cbu", "CBU (convergence/branch)",
     "Convergence-unit pressure accompanies heavy branching or warp synchronisation."),
    ("inst_pipe_tmem", "Tensor memory (Blackwell)", ""),
    ("inst_pipe_gmma", "GMMA (Hopper warpgroup MMA)", ""),
    ("inst_pipe_tensor_dmma", "DMMA (fp64 tensor)", ""),
)


def analyze_instruction_mix(view: MetricView) -> Dict[str, Any]:
    """Which instruction pipe the work actually lands in.

    The SpeedOfLight compute figure is a max over pipes, so a kernel bound on
    the transcendental unit or on address arithmetic shows a low compute SOL and
    reads as latency bound. Naming the busiest pipe changes the advice
    completely: a saturated XU is fixed by cutting transcendental calls, a
    saturated LSU by widening accesses, and neither by anything that helps a
    FMA-bound kernel.

    Every one of these metrics was in the catalog and read by nothing.
    """
    findings: List[Finding] = []
    result: Dict[str, Any] = {"findings": findings}

    pipes: List[Tuple[str, str, float, str]] = []
    for key, label, note in _INSTRUCTION_PIPES:
        value = view.get(key)
        if value is not None:
            pipes.append((key, label, float(value), note))
    if not pipes:
        return result

    pipes.sort(key=lambda item: item[2], reverse=True)
    result["pipes"] = {key: value for key, _label, value, _note in pipes}
    top_key, top_label, top_value, top_note = pipes[0]
    result["busiest_pipe"] = top_label
    result["busiest_pipe_pct"] = top_value

    # These percentages are pct_of_peak_sustained_ACTIVE. They say how hard a
    # pipe worked while the SM was active, which is the right question for "is
    # this pipe the limit" but is NOT comparable with the _elapsed figures that
    # classify_bottleneck uses. The finding says so rather than inviting the
    # reader to rank them against the SOL numbers.
    if top_value >= 60.0:
        findings.append(Finding(
            category="pipe_saturated",
            title=f"{top_label} is the busiest instruction pipe",
            summary=(
                f"The {top_label} pipe runs at {top_value:.0f}% of its peak while the SM "
                f"is active. " + (top_note + " " if top_note else "") +
                "Speed-of-light compute throughput is a maximum over pipes, so a kernel "
                "limited by this pipe can still show a modest compute SOL and be "
                "misread as latency bound."
            ),
            severity="medium" if top_value >= 80.0 else "low",
            confidence="high",
            evidence={
                "pipe": top_key,
                "pct_of_peak_active": top_value,
                "all_pipes": {label: value for _k, label, value, _n in pipes},
                "denominator": "active (not comparable with _elapsed SOL figures)",
            },
            actions=(
                _PIPE_ACTIONS.get(top_key,
                                  "Reduce the instruction count landing in this pipe."),
            ),
            source="heuristic",
        ))

    # FP64 where nobody asked for it. Distinct from the pipe-saturation finding
    # above: even a few percent of FP64 on a part with a 1/64 rate is a large
    # share of the time.
    fp64 = view.get("inst_pipe_fp64")
    if fp64 is not None and fp64 > 1.0 and top_key != "inst_pipe_fp64":
        findings.append(Finding(
            category="unexpected_fp64",
            title="FP64 instructions are executing",
            summary=(
                f"The FP64 pipe is at {fp64:.1f}% of peak. On parts where FP64 runs at "
                "1/64 rate, even a small instruction share can dominate the time. This "
                "is usually an unintended double: a literal without an f suffix, or a "
                "math function that defaulted to the double overload."
            ),
            severity="medium",
            confidence="medium",
            evidence={"inst_pipe_fp64_pct_active": fp64},
            actions=(
                "Check for double literals (1.0 vs 1.0f) and double-precision math "
                "calls (exp vs expf) in the kernel.",
            ),
            source="heuristic",
        ))
    return result


_PIPE_ACTIONS: Dict[str, str] = {
    "inst_pipe_xu": "Cut transcendental calls: reuse computed values, use the fast-math "
                    "intrinsics (__expf, __frcp_rn) where the accuracy allows, or "
                    "restructure softmax/normalisation to evaluate fewer of them.",
    "inst_pipe_lsu": "Issue fewer, wider memory instructions: vectorise to 128-bit "
                     "accesses so the same bytes move in a quarter of the instructions.",
    "inst_pipe_adu": "Hoist address arithmetic out of the inner loop, or strength-reduce "
                     "the indexing to pointer increments.",
    "inst_pipe_fp64": "Remove unintended double precision: check for literals without an "
                      "f suffix and for double-overload math calls.",
    "inst_pipe_fp16": "Half-precision work is running in the FMA pipe rather than on "
                      "tensor cores. Check the shapes and the layout permit an MMA path.",
    "inst_pipe_alu": "Reduce integer and logic work in the inner loop; it is often index "
                     "arithmetic that can be hoisted or precomputed.",
    "inst_pipe_cbu": "Reduce branching and warp-level synchronisation in the hot path.",
}


# A sector is 32 bytes on every architecture Nsight Compute supports. Used to
# derive byte counts from the sector counters, which sections do collect, when
# the direct byte counters (which they do not) are absent.
_BYTES_PER_SECTOR = 32.0


def hierarchical_roofline(view: MetricView, gpu_spec: Any = None) -> Dict[str, Any]:
    """Arithmetic intensity at L1, L2 and DRAM, not just DRAM.

    The stock roofline plots one point per kernel: FLOPs over DRAM bytes. That
    answers "is this kernel memory bound" and nothing about *why*. Computing the
    same FLOP count against traffic at each level gives three points that share
    a numerator, and the horizontal spread between them is a direct read on
    cache locality:

    * **L1 and L2 intensities close together** -- L1 is not catching reuse, so
      nearly everything that misses L1 also reaches L2. Blocking for L1 is the
      lever.
    * **L2 and DRAM intensities close together** -- L2 is not catching reuse
      either, and the working set genuinely does not fit. Blocking helps only
      if the tile size can come down.
    * **Wide spread at both** -- the cache hierarchy is already doing its job,
      and locality is not where the remaining time is.

    Following the LBNL hierarchical-roofline method, which specifies this as a
    single-pass collection.

    Two things this deliberately does not paper over:

    **Shared memory is not counted in the L1 byte total.** ``l1tex__t_bytes``
    excludes shared-memory traffic, so for a tiled GEMM or an attention kernel
    -- exactly the kernels whose whole design is staging through shared memory
    -- the L1 intensity is an overestimate. Reported as a caveat whenever the
    shared-memory pipe shows activity, rather than left for the reader to know.

    **FLOP counts from SASS counters miss tensor-core work**, which is most of
    the math in the kernels most worth analysing. Inherited from
    ``compute_roofline`` and surfaced here too, because a numerator that is a
    floor makes all three intensities floors.
    """
    findings: List[Finding] = []
    caveats: List[str] = []
    result: Dict[str, Any] = {"findings": findings, "caveats": caveats}

    roofline = compute_roofline(view, gpu_spec)
    flops = roofline.get("total_flops")
    if not flops:
        result["available"] = False
        result["reason"] = (
            "No FLOP counters in this report, so no arithmetic intensity can be "
            "computed at any level. Collect the SASS FLOP counters "
            "(regex:smsp__sass_thread_inst_executed_op_[dhf](add|mul|fma)_pred_on.sum)."
        )
        return result

    def _bytes(direct_key: str, sector_key: str) -> Tuple[Optional[float], str]:
        direct = view.get(direct_key)
        if direct is not None:
            return direct, "byte counter"
        sectors = view.get(sector_key)
        if sectors is not None:
            return sectors * _BYTES_PER_SECTOR, f"{sector_key} x {_BYTES_PER_SECTOR:.0f}"
        return None, ""

    l1_bytes, l1_source = _bytes("l1_bytes_total", "l1_sectors_total")
    l2_bytes, l2_source = _bytes("l2_bytes_total", "l2_sectors_total")
    dram_bytes = view.get("dram_bytes")
    if dram_bytes is None:
        read, write = view.get("dram_bytes_read"), view.get("dram_bytes_write")
        if read is not None or write is not None:
            dram_bytes = (read or 0.0) + (write or 0.0)

    levels: Dict[str, Dict[str, Any]] = {}
    for name, byte_count, source in (
        ("l1", l1_bytes, l1_source),
        ("l2", l2_bytes, l2_source),
        ("dram", dram_bytes, "dram__bytes"),
    ):
        if byte_count:
            levels[name] = {
                "bytes": byte_count,
                "arithmetic_intensity": flops / byte_count,
                "byte_source": source,
            }

    result["available"] = bool(levels)
    result["total_flops"] = flops
    result["levels"] = levels
    if not levels:
        result["reason"] = (
            "No traffic counters at any level. Collect --section MemoryWorkloadAnalysis."
        )
        return result

    # Shared memory is invisible to the L1 byte counter, and the kernels this
    # matters most for are precisely the ones worth profiling.
    shared_active = (
        view.get("shared_wavefronts_ld") or view.get("shared_wavefronts_st")
        or view.get("shared_pipe_util")
    )
    if "l1" in levels and shared_active:
        caveats.append(
            "This kernel uses shared memory, and the L1 byte counter does not include "
            "shared-memory traffic. The L1 arithmetic intensity above is therefore an "
            "overestimate -- real bytes moved at that level are higher. This affects "
            "tiled GEMM and attention kernels most, since staging through shared memory "
            "is the point of their design."
        )
    if roofline.get("flops_undercounted"):
        caveats.append(
            "The FLOP total comes from SASS counters, which do not see tensor-core math. "
            "Every intensity here is a floor, not a measurement."
        )

    # The gaps between levels are the actual diagnostic.
    ai_l1 = levels.get("l1", {}).get("arithmetic_intensity")
    ai_l2 = levels.get("l2", {}).get("arithmetic_intensity")
    ai_dram = levels.get("dram", {}).get("arithmetic_intensity")

    if ai_l1 and ai_l2:
        ratio = ai_l2 / ai_l1
        result["l1_to_l2_intensity_ratio"] = ratio
        if ratio < 1.5:
            findings.append(Finding(
                category="poor_cache_locality",
                title="L1 is not capturing reuse",
                summary=(
                    f"Arithmetic intensity is {ai_l1:.2f} FLOP/byte at L1 and {ai_l2:.2f} "
                    f"at L2, a ratio of {ratio:.2f}. Nearly every byte that leaves L1 "
                    "reaches L2, so L1 is passing traffic through rather than serving it. "
                    "The stock DRAM-only roofline cannot show this."
                ),
                severity="medium",
                confidence="high",
                evidence={"ai_l1": ai_l1, "ai_l2": ai_l2, "ratio": ratio,
                          "l1_byte_source": levels["l1"]["byte_source"]},
                actions=(
                    "Block for L1: shrink the working set of the inner loop so each "
                    "fetched line is reused before it is evicted.",
                ),
                source="heuristic",
            ))

    if ai_l2 and ai_dram:
        ratio = ai_dram / ai_l2
        result["l2_to_dram_intensity_ratio"] = ratio
        if ratio < 1.5:
            findings.append(Finding(
                category="poor_cache_locality",
                title="L2 is not capturing reuse either",
                summary=(
                    f"Arithmetic intensity is {ai_l2:.2f} FLOP/byte at L2 and "
                    f"{ai_dram:.2f} at DRAM, a ratio of {ratio:.2f}. Traffic missing L2 "
                    "goes almost entirely to device memory, so the working set does not "
                    "fit in L2 at the current tile size."
                ),
                severity="medium",
                confidence="high",
                evidence={"ai_l2": ai_l2, "ai_dram": ai_dram, "ratio": ratio},
                actions=(
                    "Reduce the tile or batch size so the working set fits in L2, or "
                    "reorder the traversal so reuse happens before eviction.",
                ),
                source="heuristic",
            ))

    if ai_l1 and ai_dram and ai_dram / ai_l1 >= 4.0:
        result["locality_verdict"] = "healthy"
        findings.append(Finding(
            category="poor_cache_locality",
            title="Cache hierarchy is already capturing most reuse",
            summary=(
                f"Arithmetic intensity rises from {ai_l1:.2f} FLOP/byte at L1 to "
                f"{ai_dram:.2f} at DRAM, a {ai_dram / ai_l1:.1f}x spread. The caches are "
                "absorbing the traffic they should. Locality is not where the remaining "
                "time is; look elsewhere before restructuring for cache behaviour."
            ),
            severity="info",
            confidence="high",
            evidence={"ai_l1": ai_l1, "ai_dram": ai_dram},
            actions=("Do not spend effort on blocking or tiling; it is already working.",),
            source="heuristic",
        ))
    elif ai_l1 and ai_dram:
        result["locality_verdict"] = "leaking"

    result["summary"] = " | ".join(
        f"{name.upper()} AI={info['arithmetic_intensity']:.2f} FLOP/byte"
        for name, info in levels.items()
    )
    return result


def analyze_memory_hierarchy(view: MetricView) -> Dict[str, Any]:
    """Where in the hierarchy the traffic actually lands, and whether it should.

    `classify_bottleneck` says "memory bound" from the SpeedOfLight rollup, and
    `analyze_coalescing` says whether requests are shaped well. Neither says
    which level is saturated, whether reads and writes are balanced, or whether
    traffic is leaving the device entirely. Those are separate questions with
    separate fixes, and the counters for all of them were being collected and
    read by nothing.
    """
    findings: List[Finding] = []
    caveats: List[str] = []
    result: Dict[str, Any] = {"findings": findings, "caveats": caveats}

    l2_sol = view.get("l2_sol")
    l1_sol = view.get("l1_sol")
    dram_sol = view.get("dram_sol")
    for label, value in (("L2", l2_sol), ("L1/TEX", l1_sol), ("DRAM", dram_sol)):
        if value is not None:
            result[f"{label.split('/')[0].lower()}_sol"] = value

    # Which level is the tightest. Only _elapsed percentages are comparable, and
    # every SOL key above resolves to an _elapsed spelling for exactly that
    # reason -- mixing in an _active figure would rank the idlest unit first.
    levels = [(n, v) for n, v in (("L2", l2_sol), ("DRAM", dram_sol), ("L1/TEX", l1_sol))
              if v is not None]
    if levels:
        tightest, tightest_val = max(levels, key=lambda kv: kv[1])
        result["tightest_level"] = tightest
        result["tightest_level_pct"] = tightest_val
        if tightest_val >= 70.0 and tightest != "DRAM":
            findings.append(Finding(
                category="memory",
                title=f"{tightest} is the saturated level, not DRAM",
                summary=(
                    f"{tightest} runs at {tightest_val:.0f}% of peak while DRAM is at "
                    f"{dram_sol:.0f}%. " if dram_sol is not None else
                    f"{tightest} runs at {tightest_val:.0f}% of peak. "
                ) + (
                    "Adding DRAM bandwidth will not help; the fix is to cut requests "
                    "at this level -- larger tiles, better reuse, or vectorised access."
                ),
                severity="medium",
                confidence="high",
                evidence={n: v for n, v in levels},
                actions=(
                    f"Reduce {tightest} traffic rather than DRAM traffic: increase reuse "
                    "per byte fetched, or widen each access so fewer requests move the "
                    "same bytes.",
                ),
                source="heuristic",
            ))

    # Read/write asymmetry. A write-heavy kernel behaves differently from a
    # read-heavy one at the same total bandwidth: writes cannot be prefetched,
    # and read-modify-write of partial sectors doubles the traffic.
    rd = view.get("dram_bytes_read")
    wr = view.get("dram_bytes_write")
    if rd is not None and wr is not None and (rd + wr) > 0:
        write_share = wr / (rd + wr)
        result["dram_read_bytes"] = rd
        result["dram_write_bytes"] = wr
        result["dram_write_share"] = write_share
        if write_share > 0.6:
            findings.append(Finding(
                category="memory",
                title="DRAM traffic is write-dominated",
                summary=(
                    f"{write_share * 100:.0f}% of device-memory traffic is writes "
                    f"({wr / 1e6:.1f} MB written against {rd / 1e6:.1f} MB read). Writes "
                    "cannot be prefetched, and a partially-written sector costs a read "
                    "as well, so write-heavy kernels reach a lower fraction of peak "
                    "bandwidth than read-heavy ones."
                ),
                severity="low",
                confidence="high",
                evidence={"dram_bytes_read": rd, "dram_bytes_write": wr,
                          "write_share": round(write_share, 3)},
                actions=(
                    "Write whole sectors: make the store pattern contiguous and aligned "
                    "so no sector is partially written.",
                ),
                source="heuristic",
            ))

    # Read vs write hit rate. One low half is invisible in the aggregate.
    rd_hit = view.get("l2_read_hit_rate")
    wr_hit = view.get("l2_write_hit_rate")
    if rd_hit is not None and wr_hit is not None:
        result["l2_read_hit_rate"] = rd_hit
        result["l2_write_hit_rate"] = wr_hit
        if abs(rd_hit - wr_hit) > 30.0:
            low, high = ("read", "write") if rd_hit < wr_hit else ("write", "read")
            findings.append(Finding(
                category="poor_cache_locality",
                title=f"L2 {low} hit rate is far below the {high} hit rate",
                summary=(
                    f"L2 hits {rd_hit:.0f}% of reads and {wr_hit:.0f}% of writes. The "
                    f"aggregate hit rate hides this: the {low} path is missing to DRAM "
                    "while the other is served from cache."
                ),
                severity="medium",
                confidence="high",
                evidence={"l2_read_hit_rate": rd_hit, "l2_write_hit_rate": wr_hit},
                actions=(
                    f"Look at the {low} access pattern specifically -- the aggregate "
                    "hit rate will not show the improvement or the regression.",
                ),
                source="heuristic",
            ))

    # Traffic leaving the device. This is the one that turns a "slow kernel"
    # into "this kernel is touching host or peer memory", which is a different
    # problem by two orders of magnitude.
    sysmem = view.get("l2_aperture_sysmem_miss")
    peer = view.get("l2_aperture_peer_miss")
    for label, value, detail in (
        ("system", sysmem, "host memory over PCIe or NVLink-C2C"),
        ("peer", peer, "another GPU's memory over NVLink or PCIe"),
    ):
        if value is None or value <= 0:
            continue
        result[f"{label}_aperture_sectors"] = value
        findings.append(Finding(
            category="memory",
            title=f"Kernel is reaching {label} memory, not just device memory",
            summary=(
                f"{value:,.0f} L2 sectors missed to the {label} aperture, meaning the "
                f"kernel accessed {detail}. That path is roughly an order of magnitude "
                "slower than HBM, so a small number of these accesses can dominate the "
                "kernel's time while barely showing up in DRAM throughput."
            ),
            severity="high",
            confidence="high",
            evidence={f"l2_aperture_{label}_miss_sectors": value},
            actions=(
                "Check for unified/managed memory that has not migrated, or a pointer "
                "that should have been a device allocation. cudaMemPrefetchAsync or an "
                "explicit device copy usually removes this entirely.",
            ),
            source="heuristic",
        ))

    # No caveat when the counters are absent: `analysis_coverage` already
    # reports memory_hierarchy as skipped, and saying it twice turns a real
    # signal into noise.
    return result


def analyze_issue_efficiency(view: MetricView) -> Dict[str, Any]:
    """Whether the schedulers had anything to issue, and why not.

    Occupancy says how many warps are resident. This says how many were
    *eligible* -- the number that matters. High occupancy with no eligible warps
    means every resident warp is stalled, and adding more warps will not help;
    that distinction decides whether the fix is occupancy or latency.
    """
    findings: List[Finding] = []
    result: Dict[str, Any] = {"findings": findings}

    issue = view.get("issue_slot_util")
    eligible = view.get("warps_eligible_per_scheduler")
    active = view.get("warps_active_per_scheduler")
    for key, value in (("issue_slot_util", issue),
                       ("warps_eligible_per_scheduler", eligible),
                       ("warps_active_per_scheduler", active)):
        if value is not None:
            result[key] = value

    # NVIDIA's own SchedulerStats guidance: below roughly one eligible warp per
    # scheduler per cycle, the scheduler idles whenever the chosen warp stalls.
    if eligible is not None and eligible < 1.0:
        detail = ""
        if active is not None and active > 4.0:
            detail = (
                f" {active:.1f} warps were resident, so the problem is not occupancy: "
                "the resident warps are stalled rather than absent. Raising occupancy "
                "would add more stalled warps."
            )
        findings.append(Finding(
            category="occupancy",
            title="Schedulers had less than one eligible warp per cycle",
            summary=(
                f"{eligible:.2f} warps were eligible to issue per scheduler per active "
                f"cycle. Below 1.0 the scheduler has nothing to switch to the moment "
                f"the selected warp stalls, so stall latency is exposed directly."
                + detail
            ),
            severity="medium",
            confidence="high",
            evidence={"warps_eligible_per_scheduler": eligible,
                      "warps_active_per_scheduler": active,
                      "issue_slot_util_pct": issue},
            actions=(
                "Read the stall breakdown before changing occupancy: it names what the "
                "resident warps are waiting on."
                if (active or 0) > 4.0 else
                "Increase occupancy so the scheduler has warps to switch to, or shorten "
                "the dependency chains that make each warp ineligible.",
            ),
            source="ncu_rule",
        ))
    return result


def analyze_imbalance(view: MetricView) -> Dict[str, Any]:
    """SM / L2-slice / DRAM-partition load imbalance (NVIDIA's WorkloadImbalance rule)."""
    findings: List[Finding] = []
    details: Dict[str, Any] = {}

    for unit, avg_key, max_key in (
        ("sm", "sm_cycles_active", "sm_cycles_active_max"),
        ("l2", "l2_cycles_active_avg", "l2_cycles_active_max"),
        ("dram", "dram_cycles_active_avg", "dram_cycles_active_max"),
    ):
        avg = view.get(avg_key)
        peak = view.get(max_key)
        if avg is None or peak is None or peak <= 0:
            continue
        imbalance = 1.0 - (avg / peak)
        details[f"{unit}_imbalance"] = imbalance
        if imbalance >= SOL_THRESHOLDS["imbalance_min_speedup"]:
            findings.append(Finding(
                category=f"{unit}_load_imbalance",
                title=f"{unit.upper()} load imbalance of {imbalance * 100:.0f}%",
                summary=(
                    f"The busiest {unit} unit is active for {peak:,.0f} cycles against an average of "
                    f"{avg:,.0f}. Work is unevenly spread, so the kernel waits on its slowest unit."
                ),
                severity="medium" if imbalance > 0.2 else "low",
                confidence="medium",
                evidence={f"{unit}_cycles_avg": avg, f"{unit}_cycles_max": peak, "imbalance": imbalance},
                actions=(
                    "Use grid-stride loops so blocks self-balance."
                    if unit == "sm" else
                    "Check for hot spots in the address pattern (bank/partition camping).",
                ),
                speedup_ceiling=1.0 / (1.0 - imbalance) if imbalance < 0.95 else None,
                source="ncu_rule",
            ))

    details["findings"] = findings
    return details


# ---------------------------------------------------------------------------
# Workload expectations
# ---------------------------------------------------------------------------

def _article(word: str) -> str:
    """Pick "a" or "an" so generated sentences read naturally."""
    return "an" if str(word)[:1].lower() in "aeiou" else "a"


def _expectation_findings(
    view: MetricView,
    kernel_info: Any,
    *,
    category_confidence: str = "low",
    name_is_informative: bool = True,
) -> List[Finding]:
    """Compare a kernel against what its *category* should achieve.

    Every finding here rests on the category being right, and the category comes
    from the kernel symbol - the weakest evidence available. So the confidence of
    each finding is capped by the confidence of the classification itself, and an
    uninformative name produces an explicit finding rather than silence: a
    ``kernel_kernel_0`` from CuTe DSL would otherwise fall through to "no
    expectations matched", which reads exactly like "nothing is wrong".
    """
    from ..sources.kernel_taxonomy import CATEGORY_EXPECTATIONS  # local import: avoid cycle

    findings: List[Finding] = []

    if not name_is_informative:
        return [Finding(
            category="unattributable_kernel",
            title="Kernel name carries no workload information",
            summary=(
                "This symbol identifies no workload category, so the checks that compare a "
                "kernel against what its category should achieve could not run. That is a gap "
                "in coverage, not a clean result. CuTe DSL kernels default to this shape and "
                "FlashAttention-4 ships that way."
            ),
            severity="info",
            confidence="high",
            evidence={"kernel_name": getattr(kernel_info, "name", "")},
            actions=(
                "Call set_name_prefix() at the launch site to make the kernel attributable.",
                "Until then, judge this kernel from counters alone: tensor pipe, DRAM SOL, "
                "and the stall profile do not depend on the name.",
            ),
            source="expectation",
        )]

    if kernel_info is None:
        return []
    expectation = CATEGORY_EXPECTATIONS.get(getattr(kernel_info, "category", ""), {})
    if not expectation:
        return []

    category = kernel_info.category
    # A name-derived category never supports better than medium confidence, and
    # an advisory one (truncated symbol, cuBLASLt-embedded CUTLASS) drops to low.
    capped = {"high": "medium", "medium": "medium", "low": "low"}.get(category_confidence, "low")

    want_tc = expectation.get("expect_tensor_cores")
    tensor_util = view.get("pipe_tensor_util")
    if want_tc is True and tensor_util is not None and tensor_util < 1.0:
        findings.append(Finding(
            category="tensor_cores_idle",
            title=f"{category} kernel is not using tensor cores",
            summary=(
                f"The tensor pipe is {tensor_util:.1f}% utilised on a {category} kernel that should be "
                "running MMA instructions. Check dtype (fp32 instead of bf16/fp16), operand alignment, "
                "and whether a fallback path was selected."
            ),
            severity="high",
            confidence=capped,
            evidence={"pipe_tensor_util_pct": tensor_util,
                      "category_provenance": "kernel_name",
                      "pipe_fma_util_pct": view.get("pipe_fma_util"),
                      "kernel_category": category,
                      "kernel_name": getattr(kernel_info, "name", "")},
            actions=(
                "Enable AMP / cast to bf16 or fp16.",
                "Align M/N/K: multiples of 8 for fp16, 16 for int8, 4 for tf32 (64/128/32 for best A100 rates).",
                "Confirm the library picked a tensor-core kernel - a *simt* or sgemm name means it did not.",
            ),
            source="expectation",
        ))

    dram_floor = expectation.get("dram_sol_pct")
    dram_sol = view.get("dram_sol")
    if isinstance(dram_floor, (int, float)) and dram_sol is not None and dram_sol < dram_floor:
        duration_ns = view.get("duration_ns")
        waves = view.get("waves_per_sm")
        reason = ""
        if duration_ns is not None and duration_ns < SOL_THRESHOLDS["short_kernel_us"] * 1000:
            reason = " The kernel is very short, so launch and ramp-up dominate."
        elif waves is not None and waves < 1:
            reason = " The grid is under one wave, so most SMs never participate."
        findings.append(Finding(
            category="memory_bound_kernel_below_expectation",
            title=f"{category} kernel reaches only {dram_sol:.0f}% of DRAM peak",
            summary=(
                f"{_article(category).capitalize()} {category} kernel is bandwidth bound by nature and should approach "
                f"{dram_floor:.0f}%+ of DRAM speed-of-light.{reason} "
                + str(expectation.get("advice") or "")
            ),
            severity="medium",
            confidence="medium",
            evidence={"dram_sol_pct": dram_sol, "expected_min_pct": dram_floor,
                      "duration_ns": duration_ns, "waves_per_sm": waves,
                      "kernel_category": category},
            actions=(
                "Fuse with neighbouring elementwise work to amortise the traffic.",
                "Check coalescing and vectorisation of the access pattern.",
            ),
            source="expectation",
        ))

    compute_floor = expectation.get("compute_sol_pct")
    compute_sol = view.get("compute_sol")
    if isinstance(compute_floor, (int, float)) and compute_sol is not None and compute_sol < compute_floor:
        findings.append(Finding(
            category="compute_bound_kernel_below_expectation",
            title=f"{category} kernel reaches only {compute_sol:.0f}% of compute peak",
            summary=(
                f"{_article(category).capitalize()} {category} kernel should be compute bound and clear {compute_floor:.0f}% SM "
                f"throughput. " + str(expectation.get("advice") or "")
            ),
            severity="medium",
            confidence="medium",
            evidence={"compute_sol_pct": compute_sol, "expected_min_pct": compute_floor,
                      "kernel_category": category},
            actions=("Check tile/wave quantisation and tensor-core engagement first.",),
            source="expectation",
        ))

    return findings


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}



# Which catalog metric each analysis needs before it can say anything at all.
# Used to report what was *not* checked: an analysis that silently skips looks
# identical to one that ran and found nothing, and those mean opposite things.
# The keys here MUST be ones the corresponding rule actually reads. Seven of
# them once were not - `warp_cycles_per_issue` instead of
# `warp_cycles_per_issued_inst`, `threads_per_inst` instead of
# `warp_exec_efficiency`, and so on. `MetricView.get` falls through to a raw
# lookup for an unknown key and returns None, so those analyses were reported as
# "skipped, metrics absent" on every report, including full ones where the rule
# had run and emitted findings. A coverage report that lies is worse than none,
# so `test_coverage_keys_exist_in_catalog` now pins every key to the catalog.
_ANALYSIS_REQUIREMENTS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "bottleneck":    (("compute_sol", "memory_sol"), "SpeedOfLight"),
    "roofline":      (("dram_bytes", "duration_ns"), "SpeedOfLight + MemoryWorkloadAnalysis"),
    "stalls":        (("warp_cycles_per_issued_inst", "issue_active"), "WarpStateStats"),
    "occupancy":     (("achieved_occupancy", "theoretical_occupancy"), "Occupancy"),
    "launch":        (("grid_size", "block_size"), "LaunchStats"),
    "coalescing":    (("global_ld_sectors_per_request", "global_ld_bytes_per_sector",
                       "l2_sectors_global_excessive"),
                      "MemoryWorkloadAnalysis_Tables"),
    "shared_memory": (("shared_bank_conflicts_ld", "shared_bank_conflicts_st"),
                      "MemoryWorkloadAnalysis_Tables"),
    "divergence":    (("warp_exec_efficiency", "branch_efficiency"),
                      "InstructionStats / SourceCounters"),
    "spilling":      (("local_ld_inst", "local_st_inst", "registers_per_thread"),
                      "SourceCounters + LaunchStats"),
    "pipes":         (("pipe_tensor_util", "pipe_fma_util"), "ComputeWorkloadAnalysis"),
    "imbalance":     (("sm_cycles_active", "sm_cycles_active_max"), "WorkloadDistribution"),
    "memory_hierarchy": (("l2_sol", "l1_sol", "dram_bytes_read", "l2_read_hit_rate"),
                         "MemoryWorkloadAnalysis"),
    "issue_efficiency": (("warps_eligible_per_scheduler", "issue_slot_util"),
                         "SchedulerStats"),
    "instruction_mix": (("inst_pipe_fma", "inst_pipe_alu", "inst_pipe_xu",
                         "inst_pipe_lsu"), "ComputeWorkloadAnalysis"),
    "hierarchical_roofline": (("l1_sectors_total", "l2_sectors_total", "l1_bytes_total"),
                              "MemoryWorkloadAnalysis + FLOP counters"),
}


def analysis_coverage(view: MetricView) -> Dict[str, Any]:
    """Report which analyses could run against this report, and which could not.

    An analysis that never ran because its section was not collected produces no
    findings - exactly like an analysis that ran and found nothing healthy. A
    report showing two findings may therefore be two problems, or two problems
    plus eight unasked questions. This makes the difference explicit.
    """
    ran: List[str] = []
    skipped: List[Dict[str, str]] = []
    for analysis, (keys, section) in sorted(_ANALYSIS_REQUIREMENTS.items()):
        if any(view.get(key) is not None for key in keys):
            ran.append(analysis)
        else:
            skipped.append({
                "analysis": analysis,
                "needs_section": section,
                "missing_metrics": ", ".join(keys),
            })
    total = len(_ANALYSIS_REQUIREMENTS)
    return {
        "ran": ran,
        "skipped": skipped,
        "coverage_pct": round(100.0 * len(ran) / total, 1) if total else 0.0,
        "summary": (
            f"{len(ran)} of {total} analyses ran. "
            + (
                f"{len(skipped)} could not: the required metrics are absent from this report, "
                "so those questions were not asked - this is missing coverage, not a clean result."
                if skipped else "Full coverage."
            )
        ),
        "remedy": (
            "Collect with --set full, or add the named sections with --section, to close the gap."
            if skipped else ""
        ),
    }


def diagnose_kernel(
    metrics: Mapping[str, float],
    *,
    kernel_name: str = "",
    gpu_spec: Any = None,
    top_k: int = 10,
    shipped_rules: Optional[Iterable[Any]] = None,
    string_metrics: Optional[Mapping[str, str]] = None,
    throttling: Optional[Mapping[str, Any]] = None,
    problem_shape: Optional[Mapping[str, int]] = None,
    collection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run every rule over one kernel's metrics and return a ranked diagnosis.

    ``metrics`` maps exact ncu metric names to values.  ``gpu_spec`` unlocks the
    absolute roofline; without it the analysis still works but reports
    arithmetic intensity without a ceiling.

    ``collection`` describes how the report was collected and is forwarded to
    :func:`~my_utils.profiling.analyzers.measurement_context.describe_collection_mode`
    (``cache_control``, ``replay_mode``, ``iterations``, ``clocks_locked``,
    ``input_distribution``). It does not change any finding; it records what
    these numbers can and cannot be compared against. Nsight Compute flushes
    every cache before each replay pass by default, so an ncu duration is a
    cold-cache number and comparing it with a wall-clock timing measures the
    cache state rather than the code. Omitting it assumes ncu defaults, which is
    what almost every run uses.

    ``problem_shape`` is the GEMM's logical shape (``{"m": M, "n": N, "k": K}``).
    The kernel symbol encodes the TILE shape and never the problem shape, so
    tile quantisation cannot be checked without it; when it is absent the check
    reports itself as unasked rather than passing silently.

    ``throttling`` accepts the keyword arguments of
    :func:`~my_utils.profiling.hardware.analyze_throttling` -- a clock-event
    bitmask, an SM clock, DCGM violation counters. Nsight Compute reports no
    metric for the clock a kernel actually ran at, so without this the
    ``power_clock`` axis is honestly reported as unexamined. When throttling is
    detected, every derived figure it invalidates is demoted to low confidence:
    a throttled run measures the clock, not the code.

    ``shipped_rules`` accepts the rule results Nsight Compute stored in the
    report (``rule_results_as_dicts()``, or the flattened rows from
    :func:`~my_utils.profiling.ncu.ncu_report_tools.collect_rule_results`).
    Passing them cross-checks our engine against NVIDIA's: agreeing categories
    are promoted to high confidence with the corroborating rule named,
    disagreements on the bottleneck verdict are raised as their own high-severity
    finding, and rules NVIDIA fired that we have no analysis for are added rather
    than dropped.  Omitting them changes nothing except that the findings say so.
    """
    from ..analyzers.evidence import attribute_kernel  # local import: avoid cycle
    from ..sources.kernel_taxonomy import classify_kernel, parse_gemm_kernel  # local import

    view = metrics if isinstance(metrics, MetricView) else MetricView(metrics)

    kernel_info = classify_kernel(kernel_name) if kernel_name else None
    gemm_shape = parse_gemm_kernel(kernel_name) if kernel_name else None

    arch = describe_arch(
        view.get("cc_major"),
        view.get("cc_minor"),
        view.get("sm_count") or view.get("device_sm_count"),
    )

    bottleneck = classify_bottleneck(view)
    stalls = analyze_stalls(view)
    roofline = compute_roofline(view, gpu_spec)
    occupancy = analyze_occupancy(view)
    launch = analyze_launch_config(view, gemm_shape=gemm_shape, problem_shape=problem_shape)
    coalescing = analyze_coalescing(view)
    shared = analyze_shared_memory(view)
    divergence = analyze_divergence(view)
    spilling = analyze_spilling(view)
    pipes = analyze_pipes(view)
    imbalance = analyze_imbalance(view)
    hierarchy = analyze_memory_hierarchy(view)
    issue = analyze_issue_efficiency(view)
    inst_mix = analyze_instruction_mix(view)
    hier_roofline = hierarchical_roofline(view, gpu_spec)

    sections = {
        "bottleneck": bottleneck,
        "stalls": stalls,
        "roofline": roofline,
        "occupancy": occupancy,
        "launch": launch,
        "coalescing": coalescing,
        "shared_memory": shared,
        "divergence": divergence,
        "spilling": spilling,
        "pipes": pipes,
        "imbalance": imbalance,
        "memory_hierarchy": hierarchy,
        "issue_efficiency": issue,
        "instruction_mix": inst_mix,
        "hierarchical_roofline": hier_roofline,
    }

    findings: List[Finding] = []
    for section_name, section in sections.items():
        if not isinstance(section, dict):
            continue
        findings.extend(section.get("findings", []))
        # A caveat that stays inside its section is invisible: callers read the
        # findings list. A measurement known to be wrong must appear there, or a
        # silently halved FP32 figure reads as a clean result.
        for caveat in section.get("caveats") or ():
            findings.append(Finding(
                category="measurement_caveat",
                title=f"{section_name} measurement is known to be incomplete",
                summary=str(caveat),
                severity="medium",
                confidence="high",
                evidence={"section": section_name},
                actions=("Collect the missing counters before using this number.",),
                source="heuristic",
            ))
    # Route the classification through the evidence layer so a name that
    # disagrees with the counters is reported as a contradiction rather than
    # silently winning.
    fused, name_warnings = attribute_kernel(kernel_name or "", metrics=view)
    category_attr = fused.get("category")
    informative_attr = fused.get("name_is_informative")
    findings.extend(_expectation_findings(
        view, kernel_info,
        category_confidence=(category_attr.confidence if category_attr else "low"),
        name_is_informative=(informative_attr.value is not False) if informative_attr else True,
    ))
    for warning in name_warnings:
        # Two different situations arrive here: the name asserting something the
        # counters contradict, and the name asserting nothing at all. Only the
        # first is a conflict, and calling the second one that would be wrong.
        uninformative = "carries no shape or dtype" in warning
        findings.append(Finding(
            category="uninformative_name" if uninformative else "evidence_conflict",
            title=(
                "Kernel name carries no shape or dtype information"
                if uninformative
                else "Kernel name disagrees with the measured counters"
            ),
            summary=warning,
            severity="low" if uninformative else "medium",
            confidence="high",
            evidence={"kernel_name": kernel_name},
            actions=(
                ("Read shape and dtype from the counters or from NVTX; the symbol has neither.",)
                if uninformative
                else ("Trust the counters. The symbol is the weakest evidence in the report.",)
            ),
            source="evidence",
        ))

    # Clock throttling first: it decides whether any of the above is attributable
    # to the code at all. Nsight Compute has no metric for the clock a kernel ran
    # at, so this can only come from external telemetry the caller supplies.
    throttle_result: Dict[str, Any] = {}
    if throttling:
        from ..hardware.throttling import analyze_throttling  # local: optional dep

        throttle_result = analyze_throttling(**dict(throttling))
        invalidated = list(throttle_result.get("invalidates") or ())
        if invalidated:
            findings.append(Finding(
                category="throttling",
                title="The GPU was throttled during this measurement",
                summary=(
                    f"{throttle_result['clock_events'].get('detail', '')} "
                    f"Derived figures that assume a steady clock are not "
                    f"attributable to the code: {', '.join(invalidated)}."
                ).strip(),
                severity="high",
                confidence="high",
                evidence={"throttling": throttle_result},
                actions=tuple(throttle_result["clock_events"].get("remedies") or ())
                        or ("Lock clocks with nvidia-smi -lgc before re-measuring.",),
                source="heuristic",
            ))
            # A throttled run measures the clock, not the kernel. Any finding
            # that rests on an invalidated figure must stop claiming certainty.
            demoted = {"compute", "memory_bandwidth"}
            from ..analyzers.axes import axis_for_category as _axis_of

            findings = [
                f if _axis_of(f.category) not in demoted or f.confidence == "low"
                else Finding(
                    category=f.category, title=f.title, summary=f.summary,
                    severity=f.severity, confidence="low",
                    evidence={**f.evidence, "throttled": True,
                              "confidence_reduced_because":
                                  "the GPU was throttled, so this figure reflects "
                                  "the clock rather than the code"},
                    actions=f.actions, speedup_ceiling=f.speedup_ceiling, source=f.source,
                )
                for f in findings
            ]

    # Cross-check against the rules Nsight Compute ran itself. This must happen
    # before the sort, because it both adds findings and rewrites confidence.
    from .shipped_rules import (  # local import: avoid a cycle at module load
        normalize_shipped_rules,
        reconcile_with_shipped_rules,
    )
    normalized_shipped = normalize_shipped_rules(shipped_rules or ())
    reconciliation = reconcile_with_shipped_rules(
        findings, normalized_shipped, our_verdict=str(bottleneck.get("verdict") or ""),
    )
    findings = list(reconciliation["findings"])

    findings.sort(key=lambda f: (
        _SEVERITY_ORDER.get(f.severity, 9),
        -(f.speedup_ceiling or 1.0),
    ))

    coverage = analysis_coverage(view)

    # Per-axis coverage: a report must be able to say which dimensions of
    # performance it examined, not only which ones produced findings. An axis we
    # never looked at and an axis that came back clean are indistinguishable in a
    # findings list, and they mean opposite things.
    from ..analyzers.axes import axis_coverage  # local import: avoid a cycle

    axes = axis_coverage(
        findings,
        metric_present=view,
        shipped_rule_axes=[r.axis for r in normalized_shipped if r.axis],
        # These ran against this kernel's data whether or not they spoke up. An
        # axis that was checked and came back clean must not read as a gap.
        axes_examined=(["measurement"] + (["power_clock"] if throttling else [])),
    )

    # Account for every metric the report carried, not just the catalogued ones.
    # A --set full collection holds thousands; the catalog interprets under two
    # hundred. The rest used to be loaded and silently ignored, so a report could
    # contain the counter that explained the kernel and never mention it. This
    # does not invent thresholds for them - it names them and places them on an
    # axis, so "we have no data" and "we had data and nothing read it" stop
    # looking the same.
    from .section_index import group_report_metrics  # local import: optional dep

    inventory = group_report_metrics(view.metric_names(), catalog=METRIC_CATALOG)
    # String-valued metrics never reach the numeric view, so without this they
    # are absent from an inventory that claims to account for everything.
    if string_metrics:
        inventory["string_valued"] = sorted(string_metrics)
        inventory["string_valued_count"] = len(string_metrics)
        inventory["total_including_string"] = inventory["total"] + len(string_metrics)
        inventory["summary"] = (
            inventory["summary"]
            + f" A further {len(string_metrics)} metrics are string-valued "
            "(the GPU model, launch policy, and the constituent lists behind each "
            "Speed-of-Light rollup); they carry no number to judge but are not lost."
        )

    # Record how this was measured. Not a finding: a property of the numbers
    # above, which decides what they may be compared against.
    from ..analyzers.measurement_context import describe_collection_mode

    context = describe_collection_mode(**{
        "source": "ncu",
        # The clock the kernel actually ran at. A duration measured at one
        # clock is not comparable with one measured at another, and on a real
        # pair of reports half an apparent 12.4% speedup was the clock.
        "sm_clock_hz": view.get("sm_clock_hz"),
        "gpc_clock_hz": view.get("gpc_clock_hz"),
        **dict(collection or {}),
    })

    return {
        "kernel_name": kernel_name,
        "coverage": coverage,
        "axes": axes,
        "metric_inventory": inventory,
        "throttling": throttle_result,
        "measurement_context": context.to_dict(),
        "corroboration": {
            key: reconciliation[key]
            for key in ("shipped_rules_available", "corroborated", "uncorroborated",
                        "ncu_only", "conflicts", "note")
        },
        "kernel_category": getattr(kernel_info, "category", "") if kernel_info else "",
        "kernel_framework": getattr(kernel_info, "framework", "") if kernel_info else "",
        "architecture": arch,
        "gpu": getattr(gpu_spec, "name", "") if gpu_spec else "",
        "verdict": bottleneck.get("verdict"),
        "sections": sections,
        "findings": [f.to_dict() for f in findings[:top_k]],
        "finding_count": len(findings),
        "metrics_present": len(view.present_keys()),
    }
