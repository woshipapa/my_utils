"""Reason over every metric in the report, not only the curated ones.

The rule engine interprets ~177 metrics chosen because each has a threshold and
a known fix. A ``--set full`` report carries thousands. `metric_inventory` names
and files the rest, which stops them being invisible, but naming is not
reasoning: a counter can sit in the inventory at 97% of peak and nothing says so.

This module closes that gap using the PerfWorks grammar rather than a lookup
table. A metric ending in ``pct_of_peak_sustained_elapsed`` is a utilisation
percentage whatever unit it belongs to, and a metric ending in ``hit_rate.pct``
is a hit rate. That is enough to say "this unit is saturated" or "this cache is
missing" for units nobody wrote a rule for.

Three classes of signal come out of it:

**Saturated units** -- anything at or above the Speed-of-Light saturation
threshold. For units the curated rules already cover this is redundant and is
suppressed; for the rest it is the only thing that would have reported them.

**Duty cycle** -- the ``_active`` vs ``_elapsed`` pair for the same counter.
``_active`` divides by cycles the unit was busy, ``_elapsed`` by cycles the
kernel ran, so their ratio is how much of the kernel the unit was engaged at
all. A unit at 95% active and 4% elapsed is saturated whenever it runs and
almost never runs -- which reads as "nearly idle" or "nearly maxed out"
depending on which number you happened to look at. This is the single most
common way two metrics from the same report appear to contradict each other,
and neither is wrong.

**Internal contradictions** -- percentages above 100, and hit rates that exceed
their own request counts. These are measurement faults, not results.

Every threshold here is ours, not NVIDIA's, and is marked as such in the
findings. They are deliberately conservative: this scan runs over thousands of
metrics, so a rule that is merely usually right produces more noise than signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ncu_diagnostics import Finding

__all__ = [
    "scan_all_signals",
    "SATURATED_PCT",
    "LOW_HIT_RATE_PCT",
    "BURSTY_DUTY_CYCLE",
]

# Matches the Speed-of-Light rule's `no_bound_threshold`, so a unit nobody wrote
# a rule for is judged by the same bar as the ones that have rules.
SATURATED_PCT = 80.0

# DERIVED (ours). Below this a cache is returning less than a third of what it
# is asked for, which is worth a look at any level.
LOW_HIT_RATE_PCT = 30.0

# DERIVED (ours). active/elapsed above this means the unit is engaged for a
# small fraction of the kernel but saturated while engaged.
BURSTY_DUTY_CYCLE = 4.0

# Units the curated rules already report on. A generic finding for these would
# duplicate a better-informed one.
_COVERED_UNITS = frozenset({"sm", "smsp", "gpu", "dram", "lts", "l1tex", "gpc"})

# Counters whose "percentage" is not a utilisation and must not be read as one.
_NON_UTILISATION = ("hit_rate", "success_rate", "efficiency", "ratio")

_PCT_ELAPSED = ".pct_of_peak_sustained_elapsed"
_PCT_ACTIVE = ".pct_of_peak_sustained_active"


def _decode(name: str) -> Dict[str, str]:
    from .section_index import decode_metric_name

    return decode_metric_name(name)


def _unit_label(unit: str) -> str:
    from .section_index import UNIT_MEANINGS

    text = UNIT_MEANINGS.get(unit, "")
    return text.split(" - ")[0].split(".")[0] if text else unit


def _traffic_behind(hit_rate_metric: str, traffic: Mapping[str, float]) -> Optional[float]:
    """Requests or sectors behind a hit-rate metric, or None if undeterminable.

    `l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct` pairs with
    `l1tex__t_requests_pipe_lsu_mem_global_op_st.sum`. The names are derived
    rather than tabulated, and None is returned when no companion is present --
    a hit rate whose volume is unknown cannot be judged, and guessing that it
    matters is how a scan over thousands of metrics turns into noise.
    """
    stem = hit_rate_metric[: -len("_hit_rate.pct")] if hit_rate_metric.endswith(
        "_hit_rate.pct") else None
    if not stem:
        return None
    for counter in ("t_requests", "t_sectors"):
        for source in ("t_sector", "t_sectors", "t_requests"):
            candidate = stem.replace(source, counter, 1) + ".sum"
            if candidate in traffic:
                return traffic[candidate]
    return None


def scan_all_signals(
    metrics: Mapping[str, float],
    *,
    covered_units: Iterable[str] = _COVERED_UNITS,
    max_findings: int = 25,
) -> Dict[str, Any]:
    """Scan every metric for anomalies its own name makes interpretable.

    ``metrics`` maps raw ncu metric names to values -- the full set from the
    report, not the catalogued subset. Returns findings plus a utilisation
    profile, which is what makes "underutilised" a meaningful statement: a unit
    is not underused in the abstract, only relative to the units that are busy.
    """
    covered = frozenset(covered_units)
    findings: List[Finding] = []

    elapsed: Dict[str, Tuple[str, float]] = {}   # counter -> (raw name, value)
    active: Dict[str, Tuple[str, float]] = {}
    hit_rates: List[Tuple[str, str, float]] = []
    traffic: Dict[str, float] = {}
    impossible: List[Tuple[str, float]] = []

    for raw, value in (metrics or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number:            # NaN
            continue
        name = str(raw)
        parts = _decode(name)
        unit = parts.get("unit", "")
        counter = parts.get("quantity", "")
        if not unit or not counter:
            continue
        key = f"{unit}__{counter}"
        low = name.lower()

        if name.endswith(_PCT_ELAPSED):
            elapsed[key] = (name, number)
        elif name.endswith(_PCT_ACTIVE):
            active[key] = (name, number)

        if "hit_rate" in low and low.endswith(".pct"):
            hit_rates.append((name, unit, number))
        # Traffic counters, so a hit rate can be weighed by whether anything
        # actually went through that path.
        if low.endswith(".sum") and ("t_sectors" in low or "t_requests" in low):
            traffic[name] = number

        # A percentage above 100 is a measurement fault wherever it appears.
        if (name.endswith(_PCT_ELAPSED) or name.endswith(_PCT_ACTIVE)
                or low.endswith(".pct")) and number > 100.5:
            impossible.append((name, number))

    # ---- saturated units the curated rules do not cover --------------------
    saturated = sorted(
        ((k, n, v) for k, (n, v) in elapsed.items()
         if v >= SATURATED_PCT
         and k.split("__")[0] not in covered
         and not any(t in n.lower() for t in _NON_UTILISATION)),
        key=lambda item: item[2], reverse=True,
    )
    for key, name, value in saturated[:5]:
        unit = key.split("__")[0]
        findings.append(Finding(
            # Deliberately NOT `pipe_saturated`: that category means the math
            # pipe, and the linkage layer joins it to MATH_PIPE_THROTTLE
            # samples. A saturated constant cache linked to the math pipe would
            # be a fabricated connection presented with full confidence.
            category="unit_saturated",
            title=f"{_unit_label(unit)} is at {value:.0f}% of peak",
            summary=(
                f"`{name}` reads {value:.1f}% of peak over elapsed cycles. No curated "
                f"rule covers the {unit} unit, so this was found by scanning the "
                "report rather than by a rule that knows what to do about it -- "
                "treat it as a pointer to the right section, not a diagnosis."
            ),
            severity="medium",
            confidence="medium",
            evidence={"metric": name, "pct_of_peak_elapsed": value,
                      "threshold": SATURATED_PCT,
                      "threshold_source": "matches NVIDIA's SOL saturation bar"},
            actions=(f"Read the section that owns `{name}` before acting.",),
            source="heuristic",
        ))

    # ---- duty cycle: the same counter seen two ways ------------------------
    bursty: List[Dict[str, Any]] = []
    for key in set(elapsed) & set(active):
        elapsed_name, elapsed_value = elapsed[key]
        active_name, active_value = active[key]
        if elapsed_value <= 0.0 or active_value < SATURATED_PCT:
            continue
        ratio = active_value / elapsed_value
        if ratio < BURSTY_DUTY_CYCLE:
            continue
        bursty.append({
            "counter": key, "active_pct": active_value,
            "elapsed_pct": elapsed_value, "duty_cycle": 1.0 / ratio,
        })
        findings.append(Finding(
            category="unit_duty_cycle",
            title=f"{_unit_label(key.split('__')[0])} is bursty, not idle",
            summary=(
                f"`{key}` reads {active_value:.0f}% of peak over ACTIVE cycles but "
                f"{elapsed_value:.0f}% over ELAPSED cycles. The unit is saturated "
                f"whenever it runs and runs for roughly {100.0 / ratio:.0f}% of the "
                "kernel. Reading only the elapsed figure makes it look idle; reading "
                "only the active figure makes it look like the bottleneck. Both "
                "numbers are correct."
            ),
            severity="low",
            confidence="high",
            evidence={"active_metric": active_name, "active_pct": active_value,
                      "elapsed_metric": elapsed_name, "elapsed_pct": elapsed_value,
                      "duty_cycle_pct": 100.0 / ratio},
            actions=(
                "Decide which question you are asking: the elapsed figure answers "
                "'is this unit the kernel's limit', the active figure answers 'is "
                "this unit efficient when used'.",
            ),
            source="heuristic",
        ))

    # ---- low hit rates, weighed by whether that path carried traffic -------
    #
    # A hit rate is a ratio, and a ratio over zero requests is not a result. On
    # a real H100 report this rule fired four times at "0%" -- one of them for
    # reduction operations the kernel never issued (0 requests), and the text
    # said "more than 100% of requests miss", which is arithmetic nonsense from
    # rendering 100 - 0. Three of the four crowded out genuine findings.
    no_traffic = 0
    unknown_traffic = 0
    reported_units: set = set()
    for name, unit, value in sorted(hit_rates, key=lambda item: item[2]):
        if value >= LOW_HIT_RATE_PCT:
            continue
        volume = _traffic_behind(name, traffic)
        if volume is not None and volume <= 0.0:
            no_traffic += 1
            continue
        if volume is None:
            unknown_traffic += 1
            continue
        # One finding per unit: four spellings of "L1TEX misses" are one fact.
        if unit in reported_units:
            continue
        reported_units.add(unit)
        findings.append(Finding(
            # Not `poor_cache_locality`: that category is linked to source lines
            # via memory stall reasons, and this finding is about one specific
            # path (atomics, stores, surface ops) whose lines the scan cannot
            # identify. Linking it reproduced the top LONG_SCOREBOARD lines
            # verbatim under a different heading -- duplication wearing the
            # costume of a second piece of evidence.
            category="unit_hit_rate",
            title=f"{_unit_label(unit)} hit rate is {value:.0f}%",
            summary=(
                f"`{name}` reads {value:.1f}%, so {100.0 - value:.0f}% of the "
                f"{volume:,.0f} requests on this path miss and go further out."
            ),
            severity="medium" if value < 15.0 else "low",
            confidence="medium",
            evidence={"metric": name, "hit_rate_pct": value,
                      "requests_behind_it": volume,
                      "threshold": LOW_HIT_RATE_PCT, "threshold_source": "ours"},
            actions=("Compare this against the traffic volume: a low hit rate on a "
                     "small number of requests costs little.",),
            source="heuristic",
        ))
        if len(reported_units) >= 3:
            break

    # ---- internal contradictions ------------------------------------------
    for name, value in impossible[:5]:
        findings.append(Finding(
            category="measurement_above_physical_limit",
            title="A percentage metric exceeds 100%",
            summary=(
                f"`{name}` reads {value:.1f}%. A percentage of peak cannot exceed "
                "100, so this is a measurement fault -- a counter multiplexed across "
                "replay passes that did not agree, or a peak the tool could not "
                "determine for this part."
            ),
            severity="high",
            confidence="high",
            evidence={"metric": name, "value": value},
            actions=("Re-collect this metric on its own, without multiplexing, "
                     "before using it or anything derived from it.",),
            source="heuristic",
        ))

    # ---- utilisation profile ----------------------------------------------
    # "Underutilised" only means something relative to what is busy, so the
    # profile is reported rather than a finding emitted per idle unit.
    profile = sorted(
        ((name, value) for name, value in
         ((n, v) for _k, (n, v) in elapsed.items())
         if not any(t in name.lower() for t in _NON_UTILISATION)),
        key=lambda item: item[1], reverse=True,
    )

    findings.sort(key=lambda f: ({"high": 0, "medium": 1, "low": 2, "info": 3}
                                 .get(f.severity, 9),))
    return {
        "findings": findings[: int(max_findings)],
        "finding_count": len(findings),
        "metrics_scanned": len(metrics or {}),
        "utilisation_profile": [
            {"metric": n, "pct_of_peak_elapsed": v} for n, v in profile[:15]
        ],
        "idle_units": [
            {"metric": n, "pct_of_peak_elapsed": v} for n, v in profile if v < 5.0
        ][:10],
        "saturated_units": [
            {"metric": n, "pct_of_peak_elapsed": v} for _k, n, v in saturated
        ],
        "bursty_units": bursty,
        "hit_rates_skipped_no_traffic": no_traffic,
        "hit_rates_skipped_unknown_traffic": unknown_traffic,
        "note": (
            (f"{no_traffic} low hit rate(s) were skipped because nothing went "
             "through that path, and "
             f"{unknown_traffic} because their traffic volume could not be "
             "established. " if (no_traffic or unknown_traffic) else "")
            + f"Scanned {len(metrics or {})} metrics by name grammar. Thresholds here "
            "are ours, not NVIDIA's, and deliberately conservative: a rule applied to "
            "thousands of metrics produces noise unless it is nearly always right. "
            "These findings point at a section to read; the curated rules are what "
            "know the fix."
        ),
    }
