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
    "pc_sampling_timeline",
    "summarize_warp_samples",
    "source_availability",
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
    files = _maybe(action, "source_files") or {}
    if not isinstance(files, Mapping):
        files = {}

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
    files = _maybe(action, "source_files") or {}
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
        content = files.get(file_name) if isinstance(files, Mapping) else None
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
    samples = _maybe(action, "timed_warp_samples")
    if not samples:
        return {
            "available": False,
            "reason": (
                "No timed warp samples in this report. PC sampling is collected by "
                "the SourceCounters/`full` path; a `basic` run has none."
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
    files = _maybe(action, "source_files") or {}
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
        content = files.get(file_name) if isinstance(files, Mapping) else None
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
