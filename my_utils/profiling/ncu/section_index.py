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
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MetricEntry",
    "SectionInfo",
    "SectionIndex",
    "build_section_index",
    "find_sections_dir",
    "decode_metric_name",
    "UNIT_MEANINGS",
    "SUBMETRIC_MEANINGS",
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
