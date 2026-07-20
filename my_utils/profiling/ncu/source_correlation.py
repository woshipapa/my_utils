"""Per-instruction and per-source-line attribution from an ncu report.

Every analysis elsewhere in this package answers "what is wrong with this
kernel". This module answers "where" -- which source line, which SASS
instruction. For a fused kernel that is often the only useful answer: knowing a
kernel is bound on long-scoreboard stalls does not tell you which of the six
fused stages is doing the stalling, and no whole-kernel counter can.

The mechanism, verified against the ``ncu_report`` module shipped with Nsight
Compute 2026.1.1 rather than from documentation:

* Source-correlated metrics carry *instance values* -- one per instruction --
  and ``IMetric.correlation_ids()`` returns a parallel metric whose instance
  values are the instruction addresses. ``has_correlation_ids()`` says whether
  a given metric works this way.
* ``IAction.source_info(address)`` maps an address to an ``ISourceInfo`` with
  ``file_name()`` and ``line()``.
* ``IAction.sass_by_pc(address)`` and ``ptx_by_pc(address)`` return the SASS and
  PTX text at an address, or ``""`` when unavailable.
* ``IAction.source_files()`` returns ``{filename: content}``, empty content when
  the source was not imported.
* ``IAction.timed_warp_samples()`` returns the PC-sampling series: dicts of
  ``timestamp`` (ns), ``pc``, ``stall_reason`` (a ``StallReason`` enum) and
  ``not_issued``.

Three things routinely make this return nothing, and each is reported as a
distinct cause rather than as an empty result:

1. **A bare ``ncu`` run collects no source metrics at all.** ``SourceCounters``
   ships only in the ``detailed`` and ``full`` sets, and the default is
   ``basic``. This is the most common reason and has nothing to do with the
   build.
2. **No ``-lineinfo``.** Without it the addresses exist but map to no source
   line, so SASS-level attribution works and source-level does not.
3. **File property mismatch.** Nsight Compute checks the source file's
   modification time and size against what the compiler recorded. Re-saving a
   ``.cu`` after compiling silently breaks source display. ``--import-source
   yes`` embeds the source and sidesteps this.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SourceLineAttribution",
    "InstructionAttribution",
    "correlate_metric_to_source",
    "attribute_stalls_to_source",
    "attribute_stalls_via_metrics",
    "top_stalling_instructions",
    "analyze_pm_sampling",
    "pc_sampling_timeline",
    "summarize_warp_samples",
    "source_availability",
    "link_findings_to_source",
    "STALL_REASON_BY_CATEGORY",
    "SOURCE_METRIC_HINTS",
]


# Metrics that carry per-instruction instance values. Verified spellings; the
# `smsp__sass_inst_executed_memdesc_explicit_*` family was renamed in 2024.3
# from `smsp__inst_executed_memdesc_explicit_*`, and only stale UI-guide tables
# still show the old form.
SOURCE_METRIC_HINTS: Tuple[str, ...] = (
    "sass__inst_executed",
    "sass__thread_inst_executed",
    "smsp__sass_inst_executed",
    "smsp__pcsamp_warps_issue_stalled",
    "memory_l1_wavefronts_shared",
    "memory_l2_theoretical_sectors_global",
    "derived__memory_l1_wavefronts_shared_excessive",
)


@dataclass
class InstructionAttribution:
    """One instruction address, with whatever could be resolved about it."""

    address: int
    value: float = 0.0
    file_name: str = ""
    line: Optional[int] = None
    sass: str = ""
    ptx: str = ""

    @property
    def located(self) -> bool:
        return bool(self.file_name) and self.line is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "address_hex": f"0x{self.address:x}",
            "value": self.value,
            "file_name": self.file_name,
            "line": self.line,
            "sass": self.sass,
            "ptx": self.ptx,
            "located": self.located,
        }


@dataclass
class SourceLineAttribution:
    """One source line, with everything attributed to it summed."""

    file_name: str
    line: int
    value: float = 0.0
    instruction_count: int = 0
    sass_samples: Tuple[str, ...] = ()
    source_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "line": self.line,
            "value": self.value,
            "instruction_count": self.instruction_count,
            "sass_samples": list(self.sass_samples),
            "source_text": self.source_text,
        }


# Which sampled stall reasons explain which finding categories. This is the
# join between "what is wrong" and "where it happens": a finding says the kernel
# is bound on memory latency, and the sampler says which lines were stalled on
# LONG_SCOREBOARD. Reasons are the names of ``ncu_report.StallReason`` members,
# read from the shipped module rather than guessed.
STALL_REASON_BY_CATEGORY: Dict[str, Tuple[str, ...]] = {
    # Memory latency and throughput.
    "memory_bound_kernel_below_expectation": ("LONG_SCOREBOARD", "LG_THROTTLE",
                                              "MIO_THROTTLE", "DRAIN"),
    "uncoalesced_global_access": ("LONG_SCOREBOARD", "LG_THROTTLE"),
    "poor_cache_locality": ("LONG_SCOREBOARD",),
    "memory": ("LONG_SCOREBOARD", "LG_THROTTLE", "MIO_THROTTLE"),
    # Shared memory and its queues.
    "shared_memory": ("MIO_THROTTLE", "SHORT_SCOREBOARD"),
    "bank_conflicts": ("MIO_THROTTLE", "SHORT_SCOREBOARD"),
    # Compute pipes.
    "compute_bound_kernel_below_expectation": ("MATH_PIPE_THROTTLE", "WAIT"),
    "pipe_saturated": ("MATH_PIPE_THROTTLE", "WAIT"),
    "unexpected_fp64": ("MATH_PIPE_THROTTLE",),
    "tensor_cores_idle": ("MATH_PIPE_THROTTLE", "WAIT"),
    # Scheduling and synchronisation.
    "occupancy": ("NOT_SELECTED", "NO_INSTRUCTIONS"),
    "occupancy_achieved_gap": ("NOT_SELECTED", "NO_INSTRUCTIONS"),
    "imbalance": ("BARRIER", "MEMBAR", "DRAIN"),
    # Control flow.
    "thread_divergence": ("BRANCH_RESOLVING",),
    # Register pressure spills to local memory, which is L1TEX traffic.
    "register_spilling": ("LONG_SCOREBOARD", "LG_THROTTLE"),
}

# Categories assembled with an f-string, matched by prefix rather than by
# spelling out every interpolated value. Writing the variants by hand went wrong
# twice: `uncoalesced_global_{label}` interpolates "load"/"store", not "ld"/"st",
# so hand-written `uncoalesced_global_ld` entries matched nothing and the
# highest-severity finding in a test report went unlinked while noisier ones did
# not. Prefixes cannot be wrong about a suffix they never name.
_CATEGORY_PREFIX_REASONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("uncoalesced_", ("LONG_SCOREBOARD", "LG_THROTTLE")),
    ("sparse_global_", ("LONG_SCOREBOARD", "LG_THROTTLE")),
    ("shared_bank_conflicts_", ("MIO_THROTTLE", "SHORT_SCOREBOARD")),
    ("occupancy_limited_", ("NOT_SELECTED", "NO_INSTRUCTIONS")),
    ("sm_load_imbalance", ("BARRIER", "MEMBAR", "DRAIN")),
    ("l2_load_imbalance", ("LONG_SCOREBOARD",)),
    ("dram_load_imbalance", ("LONG_SCOREBOARD",)),
)

# Stall categories localise themselves: a `stall_long_scoreboard` finding is
# explained by exactly the LONG_SCOREBOARD samples.
_STALL_CATEGORY_PREFIX = "stall_"

# Categories deliberately NOT linked. Each is a property of the launch or of the
# measurement rather than of any line, or -- for the generic scan findings --
# names a unit whose relationship to any stall reason is unknown. Linking them
# anyway would attach a real source line to an invented mechanism, which reads
# as evidence and is not.
_UNLINKABLE = frozenset({
    "unit_saturated", "unit_duty_cycle", "unit_hit_rate",   # generic scan
    "small_grid", "tail_wave_quantization", "block_size_not_warp_multiple",
    "tile_quantization", "wave_quantization",
    "measurement_caveat", "measurement_above_physical_limit",
    "evidence_conflict", "uninformative_name", "unattributable_kernel",
    "throttling", "coverage",
})


def link_findings_to_source(
    findings: Sequence[Any],
    action: Any,
    *,
    attribution: Optional[Mapping[str, Any]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Attach the source lines that explain each finding.

    A finding says *what*; the sampler says *where*. Joining them is the point
    of the whole pipeline: "memory bound" plus "line 412 accounts for 60% of
    LONG_SCOREBOARD samples" is actionable in a way neither half is alone.

    The join is by stall reason, via :data:`STALL_REASON_BY_CATEGORY`. It is a
    *correlation*, not a proof -- the same line can stall for several reasons and
    a finding can have causes the sampler cannot see -- so each link carries the
    share of that reason's samples the line accounts for, and the caller can see
    how concentrated the evidence is. A finding whose reasons are absent from
    the samples gets no link rather than a weak one.

    ``attribution`` is the output of :func:`attribute_stalls_to_source`; passing
    it avoids re-walking the samples when the caller already has it.
    """
    if attribution is None:
        attribution = attribute_stalls_to_source(action, top_k=64)
    if not isinstance(attribution, Mapping) or not attribution.get("available"):
        return {
            "linked": [],
            "available": False,
            "reason": (attribution or {}).get(
                "reason", "no stall attribution available for this launch"),
        }

    lines = attribution.get("source_lines") or []
    totals = attribution.get("stall_reasons") or {}

    linked: List[Dict[str, Any]] = []
    seen_signatures: Dict[Any, str] = {}
    duplicates: List[Dict[str, Any]] = []
    for finding in findings:
        category = (finding.get("category") if isinstance(finding, Mapping)
                    else getattr(finding, "category", "")) or ""
        if category in _UNLINKABLE:
            continue
        reasons = STALL_REASON_BY_CATEGORY.get(category)
        if reasons is None and category.startswith(_STALL_CATEGORY_PREFIX):
            # `stall_long_scoreboard` -> ("LONG_SCOREBOARD",)
            reasons = (category[len(_STALL_CATEGORY_PREFIX):].upper(),)
        if reasons is None:
            for prefix, prefix_reasons in _CATEGORY_PREFIX_REASONS:
                if category.startswith(prefix):
                    reasons = prefix_reasons
                    break
        if not reasons:
            continue

        hits: List[Dict[str, Any]] = []
        for row in lines:
            per_reason = row.get("stall_reasons") or {}
            matched = {r: per_reason[r] for r in reasons if per_reason.get(r)}
            if not matched:
                continue
            samples = sum(matched.values())
            # Share of all samples for these reasons that this line accounts for.
            reason_total = sum(int(totals.get(r, 0)) for r in reasons) or 1
            hits.append({
                "file_name": row.get("file_name"),
                "line": row.get("line"),
                "source_text": row.get("source_text"),
                "samples": samples,
                "share_of_reason": samples / reason_total,
                "matched_stall_reasons": matched,
                "sass_samples": (row.get("sass_samples") or [])[:2],
            })
        if not hits:
            continue
        hits.sort(key=lambda h: h["samples"], reverse=True)
        top = hits[: int(top_k)]
        covered = sum(h["share_of_reason"] for h in top)

        title = (finding.get("title") if isinstance(finding, Mapping)
                 else getattr(finding, "title", "")) or ""

        # Two findings that resolve to the same lines via the same reasons are
        # one piece of evidence, not two. Repeating it under a second heading
        # makes a single observation look corroborated.
        signature = (tuple(reasons),
                     tuple((h["file_name"], h["line"]) for h in top))
        if signature in seen_signatures:
            duplicates.append({"category": category, "finding_title": title,
                               "same_lines_as": seen_signatures[signature]})
            continue
        seen_signatures[signature] = title

        linked.append({
            "category": category,
            "finding_title": title,
            "matched_on_stall_reasons": list(reasons),
            "source_lines": top,
            "share_explained": covered,
            "concentration": (
                "concentrated" if top and top[0]["share_of_reason"] >= 0.5 else
                "spread" if covered < 0.5 else "moderate"
            ),
            "caveat": (
                "Correlation by stall reason, not proof of cause. A line can stall "
                "for several reasons at once, and a finding can have causes the "
                "sampler cannot observe."
            ),
        })

    return {
        "available": True,
        "linked": linked,
        "linked_count": len(linked),
        "duplicate_links": duplicates,
        "duplicate_note": (
            f"{len(duplicates)} finding(s) resolved to lines already shown under "
            "another heading and were folded away; repeating one observation "
            "makes it look like two." if duplicates else ""
        ),
        "unlinkable_note": (
            "Findings absent from this list have no stall reason that would "
            "localise them -- launch geometry, occupancy limits and measurement "
            "caveats are properties of the kernel, not of any one line."
        ),
    }


def _as_dict(value: Any) -> Dict[str, str]:
    """Coerce ncu_report's SWIG map into a real dict.

    `IAction.source_files()` returns a `map_string_string`, which supports
    `keys()`/`__getitem__` but is **not** a `collections.abc.Mapping` subclass.
    An `isinstance(..., Mapping)` guard therefore discarded all 64 source files
    on a real report and reported "the kernel was built without -lineinfo" for a
    kernel that had full source. Duck-typing is the only safe test here.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    try:
        return {str(k): str(value[k]) for k in value.keys()}
    except Exception:
        pass
    try:
        return {str(k): str(v) for k, v in value.items()}
    except Exception:
        return {}


def _maybe(obj: Any, name: str, *args: Any) -> Any:
    """Call ``obj.name(*args)`` if it exists, else return None. Never raises.

    The ncu_report API is a SWIG binding whose surface has changed across
    releases. A missing method must degrade to "this could not be read", never
    to a traceback that loses the rest of the analysis.
    """
    fn = getattr(obj, name, None)
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:
        return None


def _instance_values(metric: Any) -> List[float]:
    """Read a metric's per-instruction instance values."""
    count = _maybe(metric, "num_instances") or 0
    values: List[float] = []
    for index in range(int(count)):
        value = _maybe(metric, "as_double", index)
        if value is None:
            value = _maybe(metric, "as_uint64", index)
        values.append(float(value) if value is not None else 0.0)
    return values


def source_availability(action: Any) -> Dict[str, Any]:
    """Report whether *metric* source correlation is possible, and if not, why.

    Scope note: this describes correlating per-instruction **metric** values to
    source, which needs metrics carrying correlation IDs.
    :func:`attribute_stalls_to_source` runs off ``timed_warp_samples`` instead
    and needs none of that -- so a report can attribute stalls to source lines
    while this function reports correlation as impossible. Both are true; they
    are answering different questions.

    The three causes are independent and have different fixes, so they are
    distinguished rather than collapsed into "no source data".
    """
    files = _as_dict(_maybe(action, "source_files"))

    metric_names = list(_maybe(action, "metric_names") or ())
    source_metrics = [
        name for name in metric_names
        if any(hint in str(name) for hint in SOURCE_METRIC_HINTS)
    ]

    # Do any of them actually carry per-instruction correlation IDs?
    correlated: List[str] = []
    for name in source_metrics:
        metric = _maybe(action, "metric_by_name", name)
        if metric is not None and _maybe(metric, "has_correlation_ids"):
            correlated.append(str(name))

    with_content = {k: v for k, v in files.items() if v}
    without_content = sorted(k for k, v in files.items() if not v)

    reasons: List[str] = []
    if not source_metrics:
        reasons.append(
            "No source-correlated metrics in this report. A bare `ncu` run collects "
            "the `basic` set, which does not include SourceCounters -- that section "
            "ships only in `detailed` and `full`. Re-collect with "
            "`--section SourceCounters` or `--set full`. This is unrelated to how "
            "the code was built."
        )
    elif not correlated:
        reasons.append(
            "Source metrics are present but carry no correlation IDs, so their "
            "values cannot be tied to instructions."
        )
    if source_metrics and not files:
        reasons.append(
            "No source files are associated with this action. The kernel was most "
            "likely built without `-lineinfo`, so addresses exist but map to no "
            "source line. SASS-level attribution still works; source-level does not."
        )
    elif without_content:
        reasons.append(
            f"{len(without_content)} source file(s) are named but carry no content, "
            "so lines cannot be shown. Either the source was not imported "
            "(`--import-source yes`), or the file's modification time or size no "
            "longer matches what the compiler recorded -- re-saving a .cu after "
            "compiling is enough to break this."
        )

    return {
        "source_correlation_possible": bool(correlated),
        "source_lines_available": bool(with_content),
        "sass_available": bool(_maybe(action, "sass_by_pc", 0) is not None),
        "source_metric_count": len(source_metrics),
        "correlated_metric_count": len(correlated),
        "correlated_metrics": sorted(correlated),
        "files_with_content": sorted(with_content),
        "files_without_content": without_content,
        "reasons_unavailable": reasons,
    }


def correlate_metric_to_source(
    action: Any,
    metric_name: str,
    *,
    top_k: int = 20,
    include_sass: bool = True,
    include_ptx: bool = False,
) -> Dict[str, Any]:
    """Attribute one source-correlated metric to instructions and source lines.

    Returns both views. The instruction view survives a build without
    ``-lineinfo``; the source-line view does not, and the result says which one
    it managed to produce rather than returning an empty list either way.
    """
    metric = _maybe(action, "metric_by_name", metric_name)
    if metric is None:
        return {
            "metric": metric_name,
            "available": False,
            "reason": f"metric '{metric_name}' is not in this report",
            "instructions": [], "source_lines": [],
        }

    if not _maybe(metric, "has_correlation_ids"):
        return {
            "metric": metric_name,
            "available": False,
            "reason": (
                f"'{metric_name}' has no correlation IDs, so its value cannot be "
                "attributed to individual instructions. It is a whole-kernel total."
            ),
            "instructions": [], "source_lines": [],
        }

    correlation = _maybe(metric, "correlation_ids")
    if correlation is None:
        return {
            "metric": metric_name, "available": False,
            "reason": "correlation IDs were advertised but could not be read",
            "instructions": [], "source_lines": [],
        }

    values = _instance_values(metric)
    addresses = _instance_values(correlation)
    if not values or not addresses:
        return {
            "metric": metric_name, "available": False,
            "reason": "the metric carries no instance values",
            "instructions": [], "source_lines": [],
        }

    # Lengths should match one-to-one; if a release ever disagrees, use the
    # shorter and say so rather than zipping silently past the end.
    truncated = len(values) != len(addresses)
    pairs = list(zip(addresses, values))

    attributions: List[InstructionAttribution] = []
    for address, value in pairs:
        addr = int(address)
        entry = InstructionAttribution(address=addr, value=float(value))
        info = _maybe(action, "source_info", addr)
        if info is not None:
            entry.file_name = str(_maybe(info, "file_name") or "")
            line = _maybe(info, "line")
            entry.line = int(line) if line is not None else None
        if include_sass:
            entry.sass = str(_maybe(action, "sass_by_pc", addr) or "")
        if include_ptx:
            entry.ptx = str(_maybe(action, "ptx_by_pc", addr) or "")
        attributions.append(entry)

    attributions.sort(key=lambda a: a.value, reverse=True)
    total = sum(a.value for a in attributions) or 1.0

    # Roll up to source lines. Many instructions map to one line, and the line
    # is what a reader can act on.
    files = _as_dict(_maybe(action, "source_files"))
    by_line: Dict[Tuple[str, int], SourceLineAttribution] = {}
    for entry in attributions:
        if not entry.located:
            continue
        key = (entry.file_name, int(entry.line))
        rolled = by_line.get(key)
        if rolled is None:
            rolled = SourceLineAttribution(file_name=entry.file_name, line=int(entry.line))
            by_line[key] = rolled
        rolled.value += entry.value
        rolled.instruction_count += 1
        if entry.sass and len(rolled.sass_samples) < 4:
            rolled.sass_samples = rolled.sass_samples + (entry.sass,)

    for (file_name, line), rolled in by_line.items():
        content = files.get(file_name)
        if content:
            lines = content.splitlines()
            if 0 < line <= len(lines):
                rolled.source_text = lines[line - 1].strip()

    ranked_lines = sorted(by_line.values(), key=lambda r: r.value, reverse=True)
    unlocated = sum(a.value for a in attributions if not a.located)

    return {
        "metric": metric_name,
        "available": True,
        "instruction_count": len(attributions),
        "total_value": total,
        "instructions": [a.to_dict() for a in attributions[: int(top_k)]],
        "source_lines": [r.to_dict() for r in ranked_lines[: int(top_k)]],
        "unlocated_value": unlocated,
        "unlocated_share": unlocated / total,
        "truncated_instance_mismatch": truncated,
        "note": (
            f"{unlocated / total * 100:.0f}% of the total could not be tied to a "
            "source line. Those instructions are usually compiler-generated or "
            "inlined from a header without line info; their SASS is still shown."
            if unlocated else
            "Every instruction resolved to a source line."
        ),
    }


# The per-instruction PC-sampling stall metrics. Verified present on a real
# --set full Hopper report: 578 instances each, all carrying correlation IDs.
PCSAMP_STALL_PREFIX = "smsp__pcsamp_warps_issue_stalled_"


def attribute_stalls_via_metrics(
    action: Any,
    *,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Rank source lines by stalls, using the aggregated per-instruction metrics.

    This is the path a normal ``--set full`` report supports, and it is the one
    that matters: ``timed_warp_samples()`` returns the raw sample records, which
    a standard collection does not retain. What it retains instead is one
    ``smsp__pcsamp_warps_issue_stalled_<reason>`` metric per stall reason, each
    carrying a value per instruction plus correlation IDs mapping those values
    to addresses.

    Discovered by running against a real report where every stall metric was
    present with 578 correlated instances and `timed_warp_samples()` returned
    nothing -- the analysis reported "no PC sampling in this report" while
    sitting on a complete per-instruction breakdown.
    """
    names = [str(n) for n in (_maybe(action, "metric_names") or ())
             if str(n).startswith(PCSAMP_STALL_PREFIX)]
    if not names:
        return {"available": False,
                "reason": (
                    "No per-instruction PC-sampling stall metrics "
                    f"(`{PCSAMP_STALL_PREFIX}*`). Collect --section SourceCounters "
                    "or --set full."),
                "source_lines": [], "stall_reasons": {}}

    files = _as_dict(_maybe(action, "source_files"))
    by_line: Dict[Tuple[str, int], Dict[str, Any]] = {}
    reason_totals: Counter = Counter()
    unlocated = 0.0
    address_cache: Dict[int, Optional[Tuple[str, int]]] = {}

    for metric_name in names:
        # `..._not_issued` counts the same stall on cycles no warp issued at
        # all; folding both in would double-count the same instruction.
        if metric_name.endswith("_not_issued"):
            continue
        reason = metric_name[len(PCSAMP_STALL_PREFIX):].upper()
        metric = _maybe(action, "metric_by_name", metric_name)
        if metric is None or not _maybe(metric, "has_correlation_ids"):
            continue
        correlation = _maybe(metric, "correlation_ids")
        if correlation is None:
            continue

        count = int(_maybe(metric, "num_instances") or 0)
        for index in range(count):
            value = _maybe(metric, "as_double", index)
            if not value or value <= 0:
                continue
            reason_totals[reason] += value

            raw_address = _maybe(correlation, "as_uint64", index)
            if raw_address is None:
                unlocated += value
                continue
            address = int(raw_address)
            if address not in address_cache:
                info = _maybe(action, "source_info", address)
                located: Optional[Tuple[str, int]] = None
                if info is not None:
                    file_name = str(_maybe(info, "file_name") or "")
                    line = _maybe(info, "line")
                    if file_name and line is not None:
                        located = (file_name, int(line))
                address_cache[address] = located
            located = address_cache[address]
            if located is None:
                unlocated += value
                continue

            entry = by_line.setdefault(located, {
                "file_name": located[0], "line": located[1],
                "samples": 0.0, "stall_reasons": Counter(), "sass_samples": [],
            })
            entry["samples"] += value
            entry["stall_reasons"][reason] += value
            if len(entry["sass_samples"]) < 3:
                sass = str(_maybe(action, "sass_by_pc", address) or "")
                if sass:
                    entry["sass_samples"].append(sass)

    total = sum(reason_totals.values())
    if total <= 0:
        return {"available": False,
                "reason": "PC-sampling stall metrics are present but all zero",
                "source_lines": [], "stall_reasons": {}}

    ranked: List[Dict[str, Any]] = []
    for (file_name, line_no), entry in by_line.items():
        text = ""
        content = files.get(file_name)
        if content:
            lines = content.splitlines()
            if 0 < line_no <= len(lines):
                text = lines[line_no - 1].strip()
        dominant = entry["stall_reasons"].most_common(1)
        ranked.append({
            "file_name": file_name,
            "line": line_no,
            "samples": int(entry["samples"]),
            "share_of_samples": entry["samples"] / total,
            "dominant_stall_reason": dominant[0][0] if dominant else "",
            "stall_reasons": {k: int(v) for k, v in entry["stall_reasons"].most_common()},
            "sass_samples": entry["sass_samples"],
            "source_text": text,
        })
    ranked.sort(key=lambda row: row["samples"], reverse=True)

    return {
        "available": True,
        "source": "correlated pcsamp metrics",
        "total_samples": int(total),
        "stall_reasons": {k: int(v) for k, v in reason_totals.most_common()},
        "source_lines": ranked[: int(top_k)],
        "unlocated_samples": int(unlocated),
        "unlocated_share": unlocated / total,
        "confidence_note": (
            f"{int(total)} sampled stall cycles attributed across "
            f"{len(ranked)} source lines. PC sampling is statistical: a difference "
            "of a few samples between lines is noise."
            + (f" {unlocated / total * 100:.0f}% could not be tied to a source line, "
               "usually inlined or compiler-generated code."
               if unlocated else "")
        ),
    }


def attribute_stalls_to_source(
    action: Any,
    *,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Rank source lines by sampled stalls, using PC sampling.

    This is the analysis that makes a fused kernel tractable. Whole-kernel stall
    percentages say the kernel waits on long-scoreboard; this says *which line*
    waits, which is the difference between a diagnosis and an action.

    Uses ``timed_warp_samples()``, whose samples carry a PC and a stall reason.
    Sample counts are statistical: a line with three samples is not meaningfully
    worse than one with two, and the result reports the sample total so a reader
    can judge whether the ranking is supported.
    """
    # Correlated metrics first: that is what a standard --set full collection
    # retains. Raw sample records are the exception, not the rule.
    via_metrics = attribute_stalls_via_metrics(action, top_k=top_k)
    if via_metrics.get("available"):
        return via_metrics

    samples = _maybe(action, "timed_warp_samples")
    if not samples:
        return {
            "available": False,
            "reason": (
                "Neither per-instruction PC-sampling stall metrics nor raw warp "
                "samples are in this report. Collect --section SourceCounters or "
                "--set full. ("
                + str(via_metrics.get("reason", "")) + ")"
            ),
            "source_lines": [], "stall_reasons": {},
        }

    by_pc: Dict[int, Counter] = defaultdict(Counter)
    reason_totals: Counter = Counter()
    issued = 0
    not_issued = 0

    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        pc = sample.get("pc")
        reason = sample.get("stall_reason")
        name = getattr(reason, "name", None) or str(reason)
        if sample.get("not_issued"):
            not_issued += 1
        else:
            issued += 1
        reason_totals[name] += 1
        if pc is not None:
            by_pc[int(pc)][name] += 1

    total_samples = sum(reason_totals.values()) or 1

    # Roll up PCs to source lines.
    files = _as_dict(_maybe(action, "source_files"))
    by_line: Dict[Tuple[str, int], Dict[str, Any]] = {}
    unlocated_samples = 0

    for pc, reasons in by_pc.items():
        info = _maybe(action, "source_info", pc)
        file_name = str(_maybe(info, "file_name") or "") if info is not None else ""
        line_no = _maybe(info, "line") if info is not None else None
        count = sum(reasons.values())
        if not file_name or line_no is None:
            unlocated_samples += count
            continue
        key = (file_name, int(line_no))
        entry = by_line.setdefault(key, {
            "file_name": file_name, "line": int(line_no),
            "samples": 0, "stall_reasons": Counter(), "sass_samples": [],
        })
        entry["samples"] += count
        entry["stall_reasons"].update(reasons)
        if len(entry["sass_samples"]) < 3:
            sass = str(_maybe(action, "sass_by_pc", pc) or "")
            if sass:
                entry["sass_samples"].append(sass)

    ranked: List[Dict[str, Any]] = []
    for (file_name, line_no), entry in by_line.items():
        content = files.get(file_name)
        text = ""
        if content:
            lines = content.splitlines()
            if 0 < line_no <= len(lines):
                text = lines[line_no - 1].strip()
        dominant = entry["stall_reasons"].most_common(1)
        ranked.append({
            "file_name": file_name,
            "line": line_no,
            "samples": entry["samples"],
            "share_of_samples": entry["samples"] / total_samples,
            "dominant_stall_reason": dominant[0][0] if dominant else "",
            "stall_reasons": dict(entry["stall_reasons"].most_common()),
            "sass_samples": entry["sass_samples"],
            "source_text": text,
        })
    ranked.sort(key=lambda r: r["samples"], reverse=True)

    return {
        "available": True,
        "total_samples": total_samples,
        "issued_samples": issued,
        "not_issued_samples": not_issued,
        "stall_reasons": dict(reason_totals.most_common()),
        "source_lines": ranked[: int(top_k)],
        "unlocated_samples": unlocated_samples,
        "unlocated_share": unlocated_samples / total_samples,
        "confidence_note": (
            f"{total_samples} samples total. PC sampling is statistical: differences "
            "of a few samples between lines are noise, and a line needs on the order "
            "of tens of samples before its rank is meaningful."
            + (
                f" {unlocated_samples / total_samples * 100:.0f}% of samples could not "
                "be tied to a source line, most often inlined or compiler-generated code."
                if unlocated_samples else ""
            )
        ),
    }


PM_SAMPLING_PREFIX = "pmsampling:"


def analyze_pm_sampling(action: Any, *, top_k: int = 8) -> Dict[str, Any]:
    """Read the PM-sampling timeline: how utilisation moved across the kernel.

    PM sampling is a different instrument from PC sampling. PC sampling says
    *where* (which instruction stalled); PM sampling says *when* (what the
    machine was doing at each point in the kernel's execution). The metrics
    arrive as ``pmsampling:<metric>`` with one value per time bucket and
    correlation IDs carrying the GPU timestamp of each bucket.

    The reason it matters, and the reason a whole-kernel average cannot replace
    it: on a real report the tensor pipe read 33.7% for the kernel as a whole
    and peaked at 93.8% inside a window covering under a fifth of the samples.
    Those are not the same machine. An average across a burst and a long idle
    tail describes neither.
    """
    names = [str(n) for n in (_maybe(action, "metric_names") or ())
             if str(n).startswith(PM_SAMPLING_PREFIX)]
    if not names:
        return {"available": False,
                "reason": (
                    "No `pmsampling:` metrics. PM sampling ships in the `full` and "
                    "`pmsampling` sets; a `basic` or `detailed` run has none."),
                "series": []}

    # PM sampling is multiplexed across replay passes: `pmsampler_pass_groups`
    # says how many. Each pass re-runs the kernel with its own capture window,
    # so series from different passes have different timestamp bases and
    # different bucket counts, and share no timestamps at all. On the test
    # report there are seven groups whose start times differ by milliseconds.
    #
    # Bucket index N therefore means a different moment in different passes, and
    # an active window derived from one pass cannot be applied to another. That
    # is what the first version of this function did -- it took the window from
    # whichever pass held `sm__cycles_active` and used those indices for
    # everything.
    by_pass: Dict[Tuple[int, int, int], List[Tuple[str, List[float]]]] = {}
    raw_series: List[Tuple[str, List[float]]] = []
    for name in names:
        metric = _maybe(action, "metric_by_name", name)
        if metric is None:
            continue
        count = int(_maybe(metric, "num_instances") or 0)
        if count <= 0:
            continue
        values = [float(_maybe(metric, "as_double", i) or 0.0) for i in range(count)]
        raw_series.append((name, values))

        correlation = _maybe(metric, "correlation_ids")
        first = step = 0
        if correlation is not None and count >= 2:
            first = int(_maybe(correlation, "as_uint64", 0) or 0)
            second = int(_maybe(correlation, "as_uint64", 1) or 0)
            step = second - first
        by_pass.setdefault((first, step, count), []).append((name, values))

    if not raw_series:
        return {"available": False,
                "reason": "`pmsampling:` metrics are present but carry no samples",
                "series": []}

    # Per pass, the kernel-active window comes from that pass's own series --
    # they were captured together, so they share a clock.
    pass_windows: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
    for key, members in by_pass.items():
        length = key[2]
        anchor = next((v for n, v in members if "sm__cycles_active" in n), None)
        if anchor is None:
            anchor = [max(v[i] if i < len(v) else 0.0 for _n, v in members)
                      for i in range(length)]
        busy = [i for i, value in enumerate(anchor) if value > 0.0]
        pass_windows[key] = (min(busy), max(busy)) if busy else (0, length - 1)

    pass_ids = {key: index for index, key in enumerate(sorted(by_pass))}
    bucket_count = max(len(values) for _n, values in raw_series)

    series: List[Dict[str, Any]] = []
    for key, members in by_pass.items():
        window_lo, window_hi = pass_windows[key]
        for name, values in members:
            window = values[window_lo:window_hi + 1]
            active_in_window = [v for v in window if v > 0.0]
            active = [v for v in values if v > 0.0]
            peak = max(values) if values else 0.0
            short = name[len(PM_SAMPLING_PREFIX):]
            is_pct = short.endswith(".pct") or "pct_of_peak" in short
            entry: Dict[str, Any] = {
                "metric": short,
                "is_percentage": is_pct,
                "unit": "%" if is_pct else "count",
                "pass_group": pass_ids[key],
                "buckets": len(values),
                "active_buckets": len(active_in_window),
                "active_buckets_whole_series": len(active),
                "duty_cycle": (len(active_in_window) / len(window)) if window else 0.0,
                "peak": peak,
                "mean_in_active_window": (sum(window) / len(window)) if window else 0.0,
                "mean_all_buckets": (sum(values) / len(values)) if values else 0.0,
                "mean_while_nonzero": (sum(active) / len(active)) if active else 0.0,
            }
            if peak > 0 and entry["mean_in_active_window"] > 0:
                entry["peak_to_mean"] = peak / entry["mean_in_active_window"]
            series.append(entry)

    series.sort(key=lambda e: (e["is_percentage"], e["peak"]), reverse=True)

    # Reported from the pass that carries the SM activity anchor, since that is
    # the one whose window is known to be the kernel rather than an envelope.
    anchor_key = next((k for k, m in by_pass.items()
                       if any("sm__cycles_active" in n for n, _v in m)),
                      next(iter(by_pass)))
    anchor_lo, anchor_hi = pass_windows[anchor_key]
    step_ns = anchor_key[1]
    duration_ns = step_ns * (anchor_key[2] - 1) if step_ns else None
    window_len = anchor_hi - anchor_lo + 1

    # A "bursty" verdict compares a peak against a ceiling, which only exists
    # for percentages. For a raw count there is no bar to be near.
    bursty = [e for e in series
              if e["is_percentage"] and e.get("peak_to_mean", 1.0) >= 2.0
              and e["peak"] >= 50.0]

    return {
        "available": True,
        "metric_count": len(series),
        "bucket_count": bucket_count,
        "pass_group_count": len(by_pass),
        "active_window_buckets": [anchor_lo, anchor_hi],
        "active_window_length": window_len,
        "active_window_ns": (step_ns * (window_len - 1)) if step_ns else None,
        "cross_pass_warning": (
            f"These {len(series)} series come from {len(by_pass)} replay passes, "
            "each a separate execution with its own capture window and bucket "
            "count. Series within one pass share a clock and can be compared "
            "bucket for bucket; series from different passes cannot. Comparing "
            "them by bucket index compares different moments of different runs."
        ) if len(by_pass) > 1 else "",
        "window_source": ("sm__cycles_active" if any(
            "sm__cycles_active" in n for n, _v in by_pass[anchor_key])
            else "union of that pass's own series"),
        "denominator_note": (
            "`mean_in_active_window` divides by the buckets in which the kernel "
            f"was running, bounded per pass group ({window_len} of "
            f"{bucket_count} buckets). "
            "`mean_all_buckets` divides by the sampler's whole session, most of "
            "which the kernel was not running in -- it is reported for "
            "transparency, not for comparison."
        ),
        "sampled_span_ns": duration_ns,
        # The sampler runs across the whole profiling session, which under
        # kernel replay covers several executions of the same kernel. A span
        # several times the kernel's duration is expected and does not mean the
        # kernel ran that long; the caller is given both so the difference is
        # visible rather than silently misread as one execution.
        "span_covers_replays": True,
        "bucket_interval_ns": (duration_ns / (series[0]["buckets"] - 1)
                               if duration_ns and series[0]["buckets"] > 1 else None),
        "series": series[: int(top_k)],
        "bursty": [
            {"metric": e["metric"], "peak": e["peak"],
             "mean_in_active_window": e["mean_in_active_window"],
             "active_share": e["duty_cycle"],
             # Carried so a consumer can tell a percentage from a count without
             # re-deriving it from the metric name.
             "is_percentage": e["is_percentage"], "unit": e["unit"]}
            for e in bursty
        ],
        "note": (
            "; ".join(
                f"{e['metric'].split('.')[0]} peaks at {e['peak']:.0f}% and averages "
                f"{e['mean_in_active_window']:.0f}% across the kernel's active "
                f"window (non-zero in {e['duty_cycle'] * 100:.0f}% of that window)"
                for e in bursty[:3]
            )
            or "No percentage-valued unit shows a large gap between its peak and "
               "its kernel average."
        ),
        "span_note": (
            "The sampled span covers the whole profiling session. Under kernel "
            "replay that is several executions of this kernel, so a span larger "
            "than the kernel duration is expected."
        ),
        "counts_note": (
            "Series marked `unit: count` are raw per-bucket counts, not "
            "percentages; they have no ceiling to be measured against."
        ),
    }


def top_stalling_instructions(
    action: Any, *, top_k: int = 12,
) -> Dict[str, Any]:
    """Rank individual SASS instructions by sampled stall cycles.

    Source-line attribution answers "which line"; a line compiles to many
    instructions and they do not stall for the same reason. This goes one level
    finer -- the exact instruction at the exact address -- which is what tells
    you a bf16 conversion is three PRMT/IMAD instructions rather than a memory
    access.
    """
    names = [str(n) for n in (_maybe(action, "metric_names") or ())
             if str(n).startswith(PCSAMP_STALL_PREFIX) and not str(n).endswith("_not_issued")]
    if not names:
        return {"available": False,
                "reason": f"no `{PCSAMP_STALL_PREFIX}*` metrics in this report",
                "instructions": []}

    files = _as_dict(_maybe(action, "source_files"))
    by_address: Dict[int, Dict[str, Any]] = {}

    for metric_name in names:
        reason = metric_name[len(PCSAMP_STALL_PREFIX):].upper()
        metric = _maybe(action, "metric_by_name", metric_name)
        if metric is None or not _maybe(metric, "has_correlation_ids"):
            continue
        correlation = _maybe(metric, "correlation_ids")
        if correlation is None:
            continue
        for index in range(int(_maybe(metric, "num_instances") or 0)):
            value = _maybe(metric, "as_double", index)
            if not value or value <= 0:
                continue
            raw = _maybe(correlation, "as_uint64", index)
            if raw is None:
                continue
            address = int(raw)
            entry = by_address.setdefault(address, {
                "address": address, "samples": 0.0, "stall_reasons": Counter(),
            })
            entry["samples"] += value
            entry["stall_reasons"][reason] += value

    if not by_address:
        return {"available": False,
                "reason": "stall metrics carry no non-zero per-instruction values",
                "instructions": []}

    ranked = sorted(by_address.values(), key=lambda e: e["samples"], reverse=True)
    total = sum(e["samples"] for e in ranked) or 1.0

    out: List[Dict[str, Any]] = []
    for entry in ranked[: int(top_k)]:
        address = entry["address"]
        info = _maybe(action, "source_info", address)
        file_name = str(_maybe(info, "file_name") or "") if info is not None else ""
        line = _maybe(info, "line") if info is not None else None
        text = ""
        if file_name and line is not None:
            content = files.get(file_name)
            if content:
                lines = content.splitlines()
                if 0 < int(line) <= len(lines):
                    text = lines[int(line) - 1].strip()
        dominant = entry["stall_reasons"].most_common(1)
        out.append({
            "address": address,
            "address_hex": f"0x{address:x}",
            "sass": str(_maybe(action, "sass_by_pc", address) or "").strip(),
            "samples": int(entry["samples"]),
            "share": entry["samples"] / total,
            "dominant_stall_reason": dominant[0][0] if dominant else "",
            "stall_reasons": {k: int(v) for k, v in entry["stall_reasons"].most_common(4)},
            "file_name": file_name,
            "line": int(line) if line is not None else None,
            "source_text": text,
        })

    return {
        "available": True,
        "total_samples": int(total),
        "distinct_instructions": len(by_address),
        "instructions": out,
        "note": (
            f"{len(by_address)} instructions carried a non-zero stall sample. One "
            "source line compiles to many instructions and they do not stall for "
            "the same reason, which is why this is a level finer than the line view."
        ),
    }


def pc_sampling_timeline(
    action: Any,
    *,
    bucket_ns: int = 100_000,
) -> Dict[str, Any]:
    """Bucket warp samples over time to show how stalls evolve within a kernel.

    Sorting samples by magnitude, which is what a naive summary does, destroys
    the one dimension PC sampling adds over the counter-based WarpStateStats
    section: *when*. A kernel whose first half is memory bound and second half
    compute bound averages to something that looks uniformly mediocre and
    suggests the wrong fix for both halves.
    """
    samples = _maybe(action, "timed_warp_samples")
    if not samples:
        return {"available": False, "reason": "no timed warp samples in this report",
                "buckets": []}

    stamped = [
        (int(s["timestamp"]), getattr(s.get("stall_reason"), "name", None)
                              or str(s.get("stall_reason")))
        for s in samples
        if isinstance(s, Mapping) and s.get("timestamp") is not None
    ]
    if not stamped:
        return {"available": False, "reason": "samples carry no timestamps", "buckets": []}

    stamped.sort()
    start = stamped[0][0]
    end = stamped[-1][0]
    width = max(int(bucket_ns), 1)

    buckets: Dict[int, Counter] = defaultdict(Counter)
    for timestamp, reason in stamped:
        buckets[(timestamp - start) // width][reason] += 1

    out = []
    for index in sorted(buckets):
        counts = buckets[index]
        total = sum(counts.values())
        dominant = counts.most_common(1)
        out.append({
            "bucket_index": index,
            "offset_ns": index * width,
            "samples": total,
            "dominant_stall_reason": dominant[0][0] if dominant else "",
            "dominant_share": (dominant[0][1] / total) if dominant and total else 0.0,
            "stall_reasons": dict(counts.most_common(6)),
        })

    # A phase change is what makes the timeline worth reading.
    phases = [b["dominant_stall_reason"] for b in out]
    distinct = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]

    return {
        "available": True,
        "bucket_ns": width,
        "duration_ns": end - start,
        "bucket_count": len(out),
        "buckets": out,
        "phase_sequence": distinct,
        "phase_change_count": max(0, len(distinct) - 1),
        "note": (
            "The dominant stall reason changes "
            f"{max(0, len(distinct) - 1)} time(s) across the kernel "
            f"({' -> '.join(distinct[:8])}). A single averaged stall breakdown would "
            "report a blend of these and suggest the wrong fix for each phase."
            if len(distinct) > 1 else
            f"One dominant stall reason throughout ({distinct[0] if distinct else 'none'}); "
            "the averaged breakdown is representative here."
        ),
    }


def summarize_warp_samples(action: Any) -> Dict[str, Any]:
    """Whole-kernel view of PC sampling, with its relationship to WarpStateStats.

    The two are different instruments measuring the same thing. WarpStateStats
    counts every cycle in hardware; PC sampling takes periodic samples and can
    attribute them to instructions. They should broadly agree, and when they do
    not, the counter-based figure is the one to trust for magnitude while the
    sampled one is the only one that can say where.
    """
    samples = _maybe(action, "timed_warp_samples")
    if not samples:
        return {"available": False,
                "reason": "no timed warp samples in this report",
                "stall_reasons": {}}

    reasons: Counter = Counter()
    not_issued = 0
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        name = (getattr(sample.get("stall_reason"), "name", None)
                or str(sample.get("stall_reason")))
        reasons[name] += 1
        if sample.get("not_issued"):
            not_issued += 1

    total = sum(reasons.values()) or 1
    return {
        "available": True,
        "total_samples": total,
        "not_issued_samples": not_issued,
        "not_issued_share": not_issued / total,
        "stall_reasons": {k: v for k, v in reasons.most_common()},
        "stall_shares": {k: v / total for k, v in reasons.most_common()},
        "comparison_note": (
            "These are sampled counts, not cycle counts. WarpStateStats measures the "
            "same stalls in hardware over every cycle and is the better source for "
            "magnitude; PC sampling is the only source that can attribute a stall to "
            "an instruction. Disagreement between them is expected at small sample "
            "counts and worth investigating at large ones."
        ),
    }
