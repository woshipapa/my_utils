"""Complete metric index generated from the Nsight Compute ``.section`` files.

:mod:`metric_catalog` is hand-written: it gives a stable short key, a threshold
and an interpretation to the metrics the rules reason about.  That is the right
shape for diagnosis, but it can never be exhaustive - NVIDIA adds metrics every
release, and ``--set full`` requests several hundred.

This module covers the rest.  It parses the ``.section`` files shipped with the
installed Nsight Compute and produces, for **every** metric those sections
request:

* the metric's exact name,
* the ``Label`` NVIDIA gives it in the UI,
* which section(s) request it and which sets those sections belong to,
* the decoded unit / quantity / rollup / submetric from the name grammar.

So a report can always say what a metric *is*, even when no rule knows what a
good value for it would be.  The two layers answer different questions:
``metric_catalog`` answers "is this value bad and what do I do", this answers
"what am I even looking at".

The index is built on demand and cached, because parsing ~25 protobuf-text files
costs a few milliseconds and the answer only changes when ncu is upgraded.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MetricEntry",
    "SectionInfo",
    "SectionIndex",
    "build_section_index",
    "find_sections_dir",
    "decode_metric_name",
    "UNIT_MEANINGS",
    "SUBMETRIC_MEANINGS",
    "UNIT_AXIS",
    "axis_for_metric_name",
    "denominator_of",
    "group_report_metrics",
    "audit_catalog_against_sections",
    "pm_sampling_groups",
]


# ---------------------------------------------------------------------------
# What the name grammar means
# ---------------------------------------------------------------------------

# Hardware unit prefixes. Getting these wrong is the usual cause of a
# dimensionally invalid comparison, e.g. treating an SM-level time fraction as
# though it were a device-level throughput.
UNIT_MEANINGS: Dict[str, str] = {
    "sm": "Streaming Multiprocessor - per-SM counters; .avg is across SMs, .sum totals them.",
    "smsp": "SM sub-partition (warp scheduler) - four per SM. Warp stalls and issue live here.",
    "l1tex": "L1 data cache / texture unit, per SM. Sector and wavefront counters live here.",
    "lts": "L2 cache slice. .sum aggregates all slices; imbalance across slices shows in .max vs .avg.",
    "ltc": "L2 cache as a whole (aggregated across slices).",
    "lrc": "L2 request coalescer / inline compression (GH100 and later).",
    "syslts": "System-side L2, used for sysmem and peer apertures (CC 10.0+).",
    "dram": "Device memory (HBM/GDDR) controller.",
    "dramc": "Device memory controller channel.",
    "fbpa": "Frame-buffer partition, the DRAM-side aggregation point.",
    "fbp": "Frame-buffer partition group.",
    "mcc": "Memory controller channel.",
    "gpu": "Whole-device aggregate; the only level at which 'the GPU is X% busy' is meaningful.",
    "gpc": "Graphics Processing Cluster. gpc__cycles_elapsed is the kernel's wall clock in cycles.",
    "tpc": "Texture Processing Cluster - a pair of SMs.",
    "gr": "Graphics/compute front end, including work distribution.",
    "pcie": "PCIe interface counters.",
    "nvlrx": "NVLink receive.",
    "nvltx": "NVLink transmit.",
    "c2c": "Chip-to-chip interconnect (Grace-Hopper / Grace-Blackwell).",
    "numa": "NUMA affinity metadata, not a performance counter.",
    "launch": "Launch configuration - static properties of the launch, not measurements.",
    "device": "Device attributes queried from the driver, not measured.",
    "profiler": "The profiler's own state (sampling interval, buffer size).",
    "sass": "SASS-patched software counters; require instrumentation, not hardware counters.",
    "derived": "Computed by Nsight Compute from other metrics rather than read from hardware.",
    "memory": "Per-source-line memory counters from SourceCounters; need -lineinfo to attribute.",
}

# Rollups across unit instances.
ROLLUP_MEANINGS: Dict[str, str] = {
    "sum": "Total across all instances of the unit.",
    "avg": "Mean across instances - hides imbalance; compare against .max to detect it.",
    "min": "Least-loaded instance.",
    "max": "Most-loaded instance. The gap to .avg is the load-imbalance signal.",
}

# Submetrics. The active/elapsed distinction is the one that silently changes
# conclusions: 'active' divides by cycles the unit was doing something, so an
# idle unit can still read 100%.
SUBMETRIC_MEANINGS: Dict[str, str] = {
    "peak_sustained": "The hardware's sustainable peak rate for this quantity.",
    "peak_sustained_active": "Peak rate, counted over cycles the unit was active.",
    "peak_sustained_elapsed": "Peak rate, counted over the whole kernel duration.",
    "per_cycle_active": "Rate per cycle in which the unit was active.",
    "per_cycle_elapsed": "Rate per cycle of kernel duration.",
    "per_second": "Absolute rate in Hz / bytes per second.",
    "pct_of_peak_sustained_active": (
        "Percent of peak, over ACTIVE cycles only. A unit that was mostly idle can still "
        "read high here - it says 'when working, how hard', not 'how much work'."
    ),
    "pct_of_peak_sustained_elapsed": (
        "Percent of peak, over the WHOLE kernel. This is the one to use for speed-of-light "
        "and for anything compared against a device peak."
    ),
    "ratio": "A dimensionless ratio between two counters.",
    "pct": "A percentage computed by Nsight Compute.",
    "max_rate": "The maximum value the corresponding ratio can take (e.g. 32 bytes/sector).",
}

_ROLLUPS = tuple(ROLLUP_MEANINGS)


@dataclass(frozen=True)
class MetricEntry:
    """One metric requested by at least one section."""

    name: str
    label: str = ""
    sections: Tuple[str, ...] = ()
    sets: Tuple[str, ...] = ()
    unit: str = ""
    quantity: str = ""
    rollup: str = ""
    submetric: str = ""

    @property
    def unit_meaning(self) -> str:
        return UNIT_MEANINGS.get(self.unit, "")

    @property
    def submetric_meaning(self) -> str:
        return SUBMETRIC_MEANINGS.get(self.submetric, "")

    @property
    def rollup_meaning(self) -> str:
        return ROLLUP_MEANINGS.get(self.rollup, "")

    def describe(self) -> str:
        """A one-paragraph explanation assembled from the name grammar and Label."""
        parts: List[str] = []
        if self.label:
            parts.append(f"{self.label}.")
        if self.unit_meaning:
            parts.append(self.unit_meaning)
        if self.rollup_meaning:
            parts.append(self.rollup_meaning)
        if self.submetric_meaning:
            parts.append(self.submetric_meaning)
        if self.sections:
            parts.append(f"Collected by: {', '.join(self.sections)}.")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "sections": list(self.sections),
            "sets": list(self.sets),
            "unit": self.unit,
            "quantity": self.quantity,
            "rollup": self.rollup,
            "submetric": self.submetric,
            "description": self.describe(),
        }


@dataclass(frozen=True)
class SectionInfo:
    """One ``.section`` file."""

    identifier: str
    display_name: str = ""
    description: str = ""
    sets: Tuple[str, ...] = ()
    metric_count: int = 0
    filename: str = ""


@dataclass
class SectionIndex:
    """Every metric the installed Nsight Compute's sections request."""

    sections_dir: str
    sections: Dict[str, SectionInfo] = field(default_factory=dict)
    metrics: Dict[str, MetricEntry] = field(default_factory=dict)

    def in_set(self, set_name: str = "full") -> List[MetricEntry]:
        """Metrics requested by any section belonging to ``set_name``."""
        return [m for m in self.metrics.values() if set_name in m.sets]

    def by_unit(self, unit: str) -> List[MetricEntry]:
        return [m for m in self.metrics.values() if m.unit == unit]

    def search(self, pattern: str) -> List[MetricEntry]:
        """Metrics whose name or label matches a regular expression."""
        regex = re.compile(pattern, re.IGNORECASE)
        return [m for m in self.metrics.values()
                if regex.search(m.name) or (m.label and regex.search(m.label))]

    def explain(self, metric_name: str) -> Optional[MetricEntry]:
        """Look up one metric, tolerating a missing or differing submetric suffix."""
        if metric_name in self.metrics:
            return self.metrics[metric_name]
        base = metric_name.split(".")[0]
        for name, entry in self.metrics.items():
            if name.split(".")[0] == base:
                return entry
        return None

    def unit_summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.metrics.values():
            counts[entry.unit] = counts.get(entry.unit, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_DEFAULT_SECTION_DIRS: Tuple[str, ...] = (
    "/Applications/NVIDIA Nsight Compute.app/Contents/Resources/sections",
    "/opt/nvidia/nsight-compute/*/sections",
    "/usr/local/cuda/nsight-compute-*/sections",
    "/usr/local/NVIDIA-Nsight-Compute*/sections",
)


def find_sections_dir(explicit: str = "") -> Optional[Path]:
    """Locate the Nsight Compute sections directory, or ``None``."""
    candidates: Sequence[str] = (explicit,) if explicit else _DEFAULT_SECTION_DIRS
    for candidate in candidates:
        if not candidate:
            continue
        if "*" in candidate:
            matches = sorted(glob.glob(candidate))
            if matches:
                return Path(matches[-1])
        elif os.path.isdir(candidate):
            return Path(candidate)
    return None


def decode_metric_name(name: str) -> Dict[str, str]:
    """Split a metric name into unit / quantity / rollup / submetric.

    ``sm__throughput.avg.pct_of_peak_sustained_elapsed`` decomposes into
    unit ``sm``, quantity ``throughput``, rollup ``avg``, submetric
    ``pct_of_peak_sustained_elapsed``.
    """
    raw = str(name or "").strip()
    if not raw:
        return {"unit": "", "quantity": "", "rollup": "", "submetric": ""}

    # Strip collection prefixes (pmsampling:, group:, breakdown:, regex:).
    prefix = ""
    if ":" in raw:
        prefix, _, raw = raw.partition(":")

    head, _, tail = raw.partition(".")
    if "__" in head:
        unit, _, quantity = head.partition("__")
    else:
        unit, quantity = "", head

    rollup = ""
    submetric = ""
    if tail:
        pieces = tail.split(".")
        if pieces and pieces[0] in _ROLLUPS:
            rollup = pieces[0]
            submetric = ".".join(pieces[1:])
        else:
            submetric = tail

    return {
        "unit": unit,
        "quantity": quantity,
        "rollup": rollup,
        "submetric": submetric,
        "prefix": prefix,
    }


# Which analysis axis each hardware unit belongs to. This is what lets a metric
# nobody catalogued still be placed on an axis: the unit prefix alone says which
# part of the machine it measures. Units absent from this table map to "", and
# callers must report that rather than silently bucketing them somewhere.
UNIT_AXIS: Dict[str, str] = {
    "sm": "compute",
    "smsp": "scheduler",
    "sass": "compute",
    "gpc": "compute",
    "tpc": "compute",
    "gr": "scheduler",
    "gpu": "compute",
    "l1tex": "memory_bandwidth",
    "lts": "memory_bandwidth",
    "ltc": "memory_bandwidth",
    "lrc": "memory_bandwidth",
    "syslts": "memory_bandwidth",
    "dram": "memory_bandwidth",
    "dramc": "memory_bandwidth",
    "fbpa": "memory_bandwidth",
    "fbp": "memory_bandwidth",
    "mcc": "memory_bandwidth",
    "memory": "memory_bandwidth",
    "pcie": "communication",
    "nvlrx": "communication",
    "nvltx": "communication",
    "c2c": "communication",
    "launch": "scheduler",
    "device": "measurement",
    "profiler": "measurement",
    "derived": "measurement",
    "numa": "measurement",
}


def axis_for_metric_name(name: str) -> str:
    """The analysis axis a metric belongs to, from its unit prefix alone.

    Returns "" for a unit this module does not recognise. That is a gap in
    :data:`UNIT_AXIS`, and reporting it is how the table gets extended -- a
    metric assigned to the wrong axis is worse than one assigned to none.
    """
    return UNIT_AXIS.get(decode_metric_name(name).get("unit", ""), "")


def denominator_of(name: str) -> str:
    """``active``, ``elapsed``, or "" -- the distinction that flips verdicts.

    A percentage over ACTIVE cycles divides by cycles the unit was busy; over
    ELAPSED cycles it divides by the kernel's whole duration. A unit busy for 3%
    of a kernel but saturated while busy reads ~100% active and ~3% elapsed.
    Ranking units by a mixture of the two puts the idlest unit at the top, which
    is exactly how an L1 gets reported as the bottleneck of a DRAM-bound kernel.
    """
    submetric = decode_metric_name(name).get("submetric", "")
    if submetric.endswith("_active"):
        return "active"
    if submetric.endswith("_elapsed"):
        return "elapsed"
    return ""


# Metric families that predate the unit__counter grammar and carry no "__".
# The SourceCounters section emits these; they are real metrics, and lumping
# them in with display names like "Duration" would hide per-source-line data.
_LEGACY_PREFIX_UNIT: Tuple[Tuple[str, str], ...] = (
    ("memory_l1_", "memory"),
    ("memory_l2_", "memory"),
    ("memory_shared_", "memory"),
    ("memory_", "memory"),
    ("derived_", "derived"),
)


def _legacy_unit(name: str) -> str:
    """Unit for a metric name that has no ``__`` separator, or ""."""
    low = str(name or "").strip().lower()
    for prefix, unit in _LEGACY_PREFIX_UNIT:
        if low.startswith(prefix):
            return unit
    return ""


def audit_catalog_against_sections(
    catalog: Mapping[str, Any],
    *,
    sections_dir: str = "",
) -> Dict[str, Any]:
    """Check catalog metric names against a local Nsight Compute install.

    Returns three buckets, and the distinction between the last two matters for
    building a collection command:

    ``section_backed``
        The name is requested by at least one shipped section, so collecting
        that section collects it.
    ``explicit_only``
        The base name appears in a shipped section under a different
        rollup/submetric. The metric is real; this spelling has to be asked for
        with ``--metrics`` because no section requests it.
    ``unknown``
        Neither the name nor its base appears anywhere in the shipped sections.
        This is a *candidate* typo and nothing stronger: sections request only a
        subset of what a device exposes, and `--query-metrics` on the target GPU
        is the only authority. Reporting these as errors would be wrong.

    Returns ``{"available": False}`` when no install is found, which is the
    normal case on a machine that only reads reports.
    """
    index = build_section_index(sections_dir) if sections_dir else build_section_index()
    if index is None:
        return {
            "available": False,
            "note": (
                "No local Nsight Compute install found, so catalog names could not be "
                "checked against the shipped sections. This is not a failure -- it "
                "means the check did not run."
            ),
        }

    known = set(index.metrics)
    known_base = {name.split(":")[-1].split(".")[0] for name in known}

    section_backed: List[str] = []
    explicit_only: List[str] = []
    unknown: List[str] = []
    for spec in catalog.values():
        for name in getattr(spec, "names", ()) or ():
            text = str(name)
            if text in known:
                section_backed.append(text)
            elif text.split(":")[-1].split(".")[0] in known_base:
                explicit_only.append(text)
            else:
                unknown.append(text)

    return {
        "available": True,
        "sections_dir": str(index.sections_dir) if hasattr(index, "sections_dir") else "",
        "shipped_metric_count": len(known),
        "section_backed": sorted(set(section_backed)),
        "explicit_only": sorted(set(explicit_only)),
        "unknown": sorted(set(unknown)),
        "summary": (
            f"{len(set(section_backed))} catalog spellings are requested by a shipped "
            f"section; {len(set(explicit_only))} exist under a different rollup and need "
            f"an explicit --metrics request; {len(set(unknown))} were not found in the "
            f"shipped sections at all (candidates only -- sections cover a subset of "
            f"what a device exposes, so --query-metrics on the target GPU decides)."
        ),
    }


@lru_cache(maxsize=4)
def pm_sampling_groups(sections_dir: str = "") -> Dict[str, str]:
    """Map each PM-sampling metric to the pass group Nsight Compute assigns it.

    PM sampling is multiplexed: metrics that cannot share a pass are split
    across replay passes, each a separate execution with its own clock. The
    shipped ``PmSampling_WarpStates.section`` states this outright -- "Metrics
    in different groups come from different passes" -- and declares the grouping
    with a ``Groups: "sampling_wsN"`` line per metric.

    Reading it beats inferring the grouping from correlation-ID timestamps: it
    is declarative, available without a report, and survives a report where two
    passes happen to start close together. Verified against a real H100 capture,
    where the five declared groups matched the five timestamp clusters exactly.

    Returns ``{}`` when no install is present, in which case the caller should
    fall back to clustering by timestamp.
    """
    directory = Path(sections_dir) if sections_dir else find_sections_dir()
    if not directory or not Path(directory).is_dir():
        return {}

    mapping: Dict[str, str] = {}
    pattern = re.compile(r'Name:\s*"(pmsampling:[^"]+)"\s*\n\s*Groups:\s*"([^"]+)"')
    for path in sorted(Path(directory).glob("*.section")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for match in pattern.finditer(text):
            mapping.setdefault(match.group(1), match.group(2))
    return mapping


def group_report_metrics(
    names: Iterable[str],
    *,
    catalog: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Account for every metric in a report, catalogued or not.

    A ``--set full`` collection carries thousands of metrics; the curated
    catalog interprets fewer than two hundred. Previously the rest were loaded
    and then ignored, so a report could contain the counter that explained a
    kernel and never mention it. This does not invent thresholds for them -- it
    names them, decodes them, and places them on an axis, so a reader can see
    what data is present and unexamined instead of assuming absence.

    ``catalog`` should be ``METRIC_CATALOG``; its per-entry ``names`` tuples mark
    which spellings a rule already interprets.
    """
    known: set = set()
    if catalog:
        for spec in catalog.values():
            for candidate in getattr(spec, "names", ()) or ():
                known.add(str(candidate))

    by_unit: Dict[str, List[str]] = {}
    by_axis: Dict[str, List[str]] = {}
    uncatalogued: List[str] = []
    unknown_units: Dict[str, int] = {}
    undecodable: List[str] = []
    total = 0

    for raw in names or ():
        text = str(raw or "").strip()
        if not text:
            continue
        total += 1
        parts = decode_metric_name(text)
        unit = parts.get("unit", "") or _legacy_unit(text)
        if not unit:
            # Display names ("Duration", "Block Size") are not PerfWorks
            # counters. They are real data, so they are kept, not dropped.
            undecodable.append(text)
            continue
        by_unit.setdefault(unit, []).append(text)
        axis = UNIT_AXIS.get(unit, "")
        if axis:
            by_axis.setdefault(axis, []).append(text)
        else:
            unknown_units[unit] = unknown_units.get(unit, 0) + 1
        if catalog and text not in known:
            uncatalogued.append(text)

    interpreted = total - len(uncatalogued) if catalog else 0
    return {
        "total": total,
        "by_unit": {u: sorted(v) for u, v in sorted(by_unit.items())},
        "by_axis": {a: sorted(v) for a, v in sorted(by_axis.items())},
        "unit_counts": {u: len(v) for u, v in sorted(by_unit.items())},
        "axis_counts": {a: len(v) for a, v in sorted(by_axis.items())},
        "uncatalogued": sorted(uncatalogued),
        "uncatalogued_count": len(uncatalogued),
        "interpreted_count": interpreted,
        "unknown_units": dict(sorted(unknown_units.items())),
        "undecodable": sorted(undecodable),
        "summary": (
            f"{total} metrics present. {interpreted} are interpreted by a rule; "
            f"{len(uncatalogued)} are decoded and placed on an axis but carry no "
            f"threshold, so nothing judged them."
            if catalog else
            f"{total} metrics across {len(by_unit)} hardware units."
        ),
    }


_IDENT_RE = re.compile(r'Identifier:\s*"([^"]+)"')
_DISPLAY_RE = re.compile(r'DisplayName:\s*"([^"]+)"')
_DESC_RE = re.compile(r'Description:\s*"([^"]*)"')
_SETS_RE = re.compile(r'Sets\s*\{[^}]*?Identifier:\s*"([^"]+)"', re.S)
# Metrics blocks pair an optional Label with a Name.
_METRIC_BLOCK_RE = re.compile(
    r'(?:Label:\s*"([^"]*)"\s*)?Name:\s*"([a-zA-Z_][a-zA-Z0-9_.:%|+\-]*)"'
)


@lru_cache(maxsize=4)
def build_section_index(sections_dir: str = "") -> Optional[SectionIndex]:
    """Parse every ``.section`` file and index the metrics they request.

    Returns ``None`` when no Nsight Compute installation can be found - callers
    should degrade to :mod:`metric_catalog` alone rather than fabricate.
    """
    root = find_sections_dir(sections_dir)
    if root is None:
        return None

    index = SectionIndex(sections_dir=str(root))
    accumulated: Dict[str, Dict[str, object]] = {}

    for path in sorted(root.glob("*.section")):
        text = path.read_text(errors="ignore")

        ident_match = _IDENT_RE.search(text)
        identifier = ident_match.group(1) if ident_match else path.stem
        display_match = _DISPLAY_RE.search(text)
        desc_match = _DESC_RE.search(text)
        sets = tuple(sorted(set(_SETS_RE.findall(text))))

        names_here = 0
        for label, name in _METRIC_BLOCK_RE.findall(text):
            # Skip protobuf field values that are not metric names.
            if "__" not in name and not name.startswith(
                ("derived__", "memory_", "sass__", "pmsampling:", "group:", "breakdown:")
            ):
                continue
            names_here += 1
            slot = accumulated.setdefault(
                name, {"label": "", "sections": set(), "sets": set()}
            )
            if label and not slot["label"]:
                slot["label"] = label
            slot["sections"].add(identifier)          # type: ignore[union-attr]
            slot["sets"].update(sets)                 # type: ignore[union-attr]

        index.sections[identifier] = SectionInfo(
            identifier=identifier,
            display_name=display_match.group(1) if display_match else "",
            description=desc_match.group(1) if desc_match else "",
            sets=sets,
            metric_count=names_here,
            filename=path.name,
        )

    for name, slot in accumulated.items():
        decoded = decode_metric_name(name)
        index.metrics[name] = MetricEntry(
            name=name,
            label=str(slot["label"]),
            sections=tuple(sorted(slot["sections"])),      # type: ignore[arg-type]
            sets=tuple(sorted(slot["sets"])),              # type: ignore[arg-type]
            unit=decoded["unit"],
            quantity=decoded["quantity"],
            rollup=decoded["rollup"],
            submetric=decoded["submetric"],
        )

    return index
