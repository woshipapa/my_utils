# SPDX-License-Identifier: Apache-2.0
"""Whether sampled data is trustworthy before anything is concluded from it.

PC sampling and PM sampling both produce data that looks identical whether it is
sound or badly undersampled: a list of samples with a distribution. The
distribution of eleven samples and the distribution of eleven thousand render
the same way and mean entirely different things.

Nsight Compute ships rules for exactly this (``PCSamplingData``,
``PMSamplingData``). Their checks and thresholds are mirrored here rather than
invented, and read from the shipped rule sources of 2026.1.1 rather than from
documentation:

**PC sampling** (``PCSamplingData.py``)

* ``smsp__pcsamp_dropped_bytes`` non-zero -- samples were dropped under
  backpressure. Fix: raise ``--warp-sampling-interval``.
* ``smsp__pcsamp_buffer_overflow`` non-zero -- the buffer overflowed. Fix:
  raise ``--warp-sampling-buffer-size``.
* ``smsp__pcsamp_sample_count == 0`` -- nothing collected, and NVIDIA's rule
  specifically notes the case where the kernel is shorter than one sampling
  interval.

**PM sampling** (``PMSamplingData.py``)

* Supported from CC 7.5. Below that the section produces nothing.
* CC > 8.0 uses a time-based interval (``profiler__pmsampler_interval_time``,
  minimum 1000); CC 7.5-8.0 uses cycles (``profiler__pmsampler_interval_cycles``,
  minimum 20000).
* ``interval / duration >= 1`` -- no or almost no samples.
* ``interval / duration > 0.1`` -- very few samples.

The reason to mirror these rather than rely on the report carrying them: the
rules only run when their section was collected and ``--apply-rules`` was on. A
report can contain the sampling counters and none of the warnings, and then the
sampled data is used with nothing flagging that it is unusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "SamplingIssue",
    "check_pc_sampling_validity",
    "check_pm_sampling_validity",
    "check_replay_clock_drift",
    "MIN_SAMPLES_FOR_RANKING",
    "PM_SAMPLING_MIN_CC",
    "REPLAY_CLOCK_DRIFT_THRESHOLD",
]

# Below this many samples, ranking one line above another is not supported by
# the data. Not from NVIDIA: a floor for the rank-ordering claims this package
# makes, which their rules do not make.
MIN_SAMPLES_FOR_RANKING = 200

# PM sampling is supported starting with SM 7.5 (PMSamplingData.py).
PM_SAMPLING_MIN_CC = 75

# Spread between per-pass effective SM clocks above which the collection is
# flagged as having drifted. Not NVIDIA's number -- their rules do not check
# this at all. 2% is chosen to sit above the noise of the estimators used
# (bucket quantisation is well under 0.1%) and below the sag observed on a
# real H100 collection (4.6% between two passes of one report).
REPLAY_CLOCK_DRIFT_THRESHOLD = 0.02

# Interval floors below which NVIDIA's rule stops advising a smaller interval,
# because it is already at the practical minimum.
_PM_MIN_INTERVAL_TIME = 1000  # CC > 8.0, time-based
_PM_MIN_INTERVAL_CYCLES = 20000  # CC 7.5-8.0, cycle-based


@dataclass
class SamplingIssue:
    """One reason sampled data should not be trusted as-is."""

    key: str
    title: str
    detail: str
    severity: str = "medium"  # high | medium | low | info
    #: True when the data must not be used for the listed conclusions at all.
    blocks: bool = False
    invalidates: Tuple[str, ...] = ()
    remedy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "detail": self.detail,
            "severity": self.severity,
            "blocks": self.blocks,
            "invalidates": list(self.invalidates),
            "remedy": self.remedy,
        }


def _num(value: Any) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def check_pc_sampling_validity(
    *,
    sample_count: Optional[float] = None,
    interval_cycles: Optional[float] = None,
    kernel_duration_cycles: Optional[float] = None,
    dropped_bytes: Optional[float] = None,
    buffer_overflow: Optional[float] = None,
    buffer_size_bytes: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate PC-sampling data, mirroring NVIDIA's ``PCSamplingData`` rule.

    Returns ``blocked_conclusions``: what the data cannot support. Attributing
    stalls to source lines needs far more samples than reporting an overall
    distribution, so the two are blocked independently.
    """
    issues: List[SamplingIssue] = []

    count = _num(sample_count)
    interval = _num(interval_cycles)
    duration = _num(kernel_duration_cycles)
    dropped = _num(dropped_bytes)
    overflow = _num(buffer_overflow)
    buffer_size = _num(buffer_size_bytes)

    if interval is None or interval == 0:
        return {
            "checked": False,
            "usable": None,
            "issues": [],
            "blocked_conclusions": [],
            "note": (
                "No PC-sampling interval in this report, so PC sampling was most "
                "likely not supported or not collected. Nothing was validated -- "
                "this is not a clean result."
            ),
        }

    if dropped:
        issues.append(
            SamplingIssue(
                key="pcsamp_dropped_samples",
                title="PC samples were dropped under backpressure",
                detail=(
                    f"{dropped:,.0f} bytes of samples were dropped at a sampling interval "
                    f"of {interval:,.0f} cycles. The surviving samples are not a uniform "
                    "sample of the kernel: drops correlate with the busiest periods, so "
                    "the hottest code is the most likely to be under-represented."
                ),
                severity="high",
                blocks=True,
                invalidates=(
                    "stall_attribution",
                    "hot_line_ranking",
                    "pc_sampling_timeline",
                ),
                remedy="Increase --warp-sampling-interval so fewer samples are produced.",
            )
        )

    if overflow:
        issues.append(
            SamplingIssue(
                key="pcsamp_buffer_overflow",
                title="The PC-sampling buffer overflowed",
                detail=(
                    "A buffer overflow occurred"
                    + (
                        f" with a buffer of {buffer_size:,.0f} bytes"
                        if buffer_size
                        else ""
                    )
                    + ". Samples after the overflow point are missing, which biases the "
                    "distribution toward whatever ran early in the kernel."
                ),
                severity="high",
                blocks=True,
                invalidates=(
                    "stall_attribution",
                    "hot_line_ranking",
                    "pc_sampling_timeline",
                ),
                remedy="Increase --warp-sampling-buffer-size.",
            )
        )

    if count is not None and count == 0:
        detail = "The PC sampler ran against this kernel and came back empty."
        remedy = "Reduce --warp-sampling-interval, or profile a longer-running kernel."
        if duration and interval >= duration:
            detail += (
                f" The kernel ran for {duration:,.0f} cycles, which is shorter than the "
                f"{interval:,.0f}-cycle sampling interval, so there was no opportunity "
                "to take a sample."
            )
        issues.append(
            SamplingIssue(
                key="pcsamp_no_samples",
                title="No PC samples were collected",
                detail=detail,
                severity="high",
                blocks=True,
                invalidates=(
                    "stall_attribution",
                    "hot_line_ranking",
                    "pc_sampling_timeline",
                    "sampled_stall_distribution",
                ),
                remedy=remedy,
            )
        )
    elif count is not None and 0 < count < MIN_SAMPLES_FOR_RANKING:
        # Not NVIDIA's check. Their rule is silent here, but this package ranks
        # source lines against each other, and that claim needs a sample floor.
        issues.append(
            SamplingIssue(
                key="pcsamp_too_few_for_ranking",
                title="Too few PC samples to rank source lines",
                detail=(
                    f"{count:,.0f} samples were collected. That is enough to see which "
                    "stall reason dominates overall, but not to say one source line is "
                    f"hotter than another -- below roughly {MIN_SAMPLES_FOR_RANKING} "
                    "samples the per-line counts are dominated by sampling noise."
                ),
                severity="medium",
                blocks=True,
                invalidates=("hot_line_ranking",),
                remedy=(
                    "Reduce --warp-sampling-interval, or profile more launches of the "
                    "same kernel, to raise the sample count."
                ),
            )
        )

    blocked: set = set()
    for issue in issues:
        if issue.blocks:
            blocked.update(issue.invalidates)

    return {
        "checked": True,
        "usable": not any(i.blocks for i in issues),
        "sample_count": count,
        "interval_cycles": interval,
        "issues": [i.to_dict() for i in issues],
        "blocked_conclusions": sorted(blocked),
        "note": (
            f"{count:,.0f} samples collected." if count else "Sample count unavailable."
        )
        + ("" if not blocked else f" Blocked: {', '.join(sorted(blocked))}."),
    }


def check_pm_sampling_validity(
    *,
    cc_major: Optional[float] = None,
    cc_minor: Optional[float] = None,
    interval: Optional[float] = None,
    duration: Optional[float] = None,
    pass_groups: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate PM-sampling data, mirroring NVIDIA's ``PMSamplingData`` rule.

    ``interval`` and ``duration`` must be in matching units. NVIDIA's rule picks
    them by architecture: above CC 8.0 it reads
    ``profiler__pmsampler_interval_time`` against ``gpu__time_duration.sum``;
    at CC 7.5-8.0 it reads ``profiler__pmsampler_interval_cycles`` against
    ``gpc__cycles_elapsed.max``. ``compute_capability`` decides which floor
    applies to the advice.
    """
    issues: List[SamplingIssue] = []

    major = _num(cc_major)
    minor = _num(cc_minor)
    cc = None
    if major is not None:
        cc = int(major) * 10 + int(minor or 0)

    if cc is not None and cc < PM_SAMPLING_MIN_CC:
        return {
            "checked": True,
            "usable": False,
            "supported": False,
            "issues": [
                SamplingIssue(
                    key="pm_sampling_unsupported",
                    title="PM sampling is not supported on this architecture",
                    detail=(
                        f"PM sampling requires compute capability {PM_SAMPLING_MIN_CC / 10:.1f} "
                        f"or later; this device reports {cc / 10:.1f}. No timeline data can "
                        "be collected here by any configuration."
                    ),
                    severity="info",
                    blocks=True,
                    invalidates=("pm_sampling_timeline",),
                    remedy="",
                ).to_dict()
            ],
            "blocked_conclusions": ["pm_sampling_timeline"],
            "note": "PM sampling unsupported on this architecture.",
        }

    interval_value = _num(interval)
    duration_value = _num(duration)

    if not interval_value or not duration_value:
        return {
            "checked": False,
            "usable": None,
            "supported": True,
            "issues": [],
            "blocked_conclusions": [],
            "note": (
                "PM-sampling interval or duration missing, so nothing was validated. "
                "Collect --section PmSampling (it ships in the `full` and `pmsampling` "
                "sets, not in `basic`)."
            ),
        }

    # Time-based above CC 8.0, cycle-based at 7.5-8.0 -- this only changes which
    # floor the advice mentions, not the ratio test.
    time_based = cc is None or cc > 80
    min_interval = _PM_MIN_INTERVAL_TIME if time_based else _PM_MIN_INTERVAL_CYCLES

    ratio = interval_value / duration_value
    if ratio >= 1.0:
        detail = (
            f"One sampling period is {ratio:.1f}x as long as the whole workload, "
            "leaving at most a single chance to sample. Any timeline built from "
            "this describes the sampler, not the workload."
        )
        severity, blocks = "high", True
    elif ratio > 0.1:
        detail = (
            f"One sampling period covers {ratio * 100:.0f}% of the workload, so only "
            "a handful of samples exist. A phase change shorter than one interval "
            "is invisible, and the timeline may show phases that are sampling artefacts."
        )
        severity, blocks = "medium", True
    else:
        detail = ""
        severity, blocks = "info", False

    if detail:
        remedy = ""
        if interval_value > min_interval:
            remedy = (
                "Reduce --pm-sampling-interval. Either raise --pm-sampling-buffer-size "
                "to match, or leave the buffer size unset and let the tool size it."
            )
        else:
            remedy = (
                f"The interval is already at the practical floor ({min_interval}); "
                "profile a longer-running workload instead."
            )
        issues.append(
            SamplingIssue(
                key="pm_sampling_interval_too_coarse",
                title="PM-sampling interval is too coarse for this workload",
                detail=detail,
                severity=severity,
                blocks=blocks,
                invalidates=("pm_sampling_timeline", "phase_detection"),
                remedy=remedy,
            )
        )

    blocked: set = set()
    for issue in issues:
        if issue.blocks:
            blocked.update(issue.invalidates)

    estimated = duration_value / interval_value if interval_value else 0.0
    return {
        "checked": True,
        "usable": not any(i.blocks for i in issues),
        "supported": True,
        "interval": interval_value,
        "duration": duration_value,
        "interval_duration_ratio": ratio,
        "estimated_sample_count": estimated,
        "pass_groups": _num(pass_groups),
        "issues": [i.to_dict() for i in issues],
        "blocked_conclusions": sorted(blocked),
        "note": (
            f"Roughly {estimated:.0f} samples span the workload "
            f"(interval/duration = {ratio:.3f})."
        ),
    }


def check_replay_clock_drift(
    clock_estimates: Optional[Mapping[str, Mapping[str, Any]]],
    *,
    threshold: float = REPLAY_CLOCK_DRIFT_THRESHOLD,
) -> Dict[str, Any]:
    """Detect intra-collection SM clock drift across ncu's replay passes.

    ncu replays the kernel many times per collection and multiplexes the
    metric set across those passes. On high-TDP parts the clock can sag
    between early and late passes -- the profiling loop itself heats the part
    -- so two counters that read as one coherent measurement were in fact
    taken at different frequencies. Nothing in the report flags this.

    ``clock_estimates`` maps a label to ``{"clock_hz": float, "kind": ...}``
    where ``kind`` is:

    * ``"measured"`` -- an exact clock for its pass (``<unit>__cycles_elapsed``
      per bucket, or a whole-pass ``.per_second`` rate). Elapsed cycles tick
      every cycle, so cycles over time *is* the clock.
    * ``"lower_bound"`` -- derived from an activity counter
      (``<unit>__cycles_active`` per bucket). Active cycles can undercount a
      bucket the unit was partly idle in, so the value bounds the clock from
      below and never from above.

    The asymmetry decides what a gap can prove. An estimate *above* a measured
    clock proves drift of at least that gap, whatever its kind. A lower bound
    *below* a measured clock proves nothing -- the unit may simply have idled
    -- and two lower bounds prove nothing against each other. Only the
    provable direction triggers; the raw spread is reported either way.

    This is a caveat about the collection, not a claim that any counter is
    wrong: every counter is a faithful reading of its own pass.
    """
    valid: Dict[str, Dict[str, Any]] = {}
    for label, entry in (clock_estimates or {}).items():
        if not isinstance(entry, Mapping):
            continue
        clock = _num(entry.get("clock_hz"))
        if not clock or clock <= 0:
            continue
        kind = str(entry.get("kind") or "measured")
        valid[str(label)] = {"clock_hz": clock, "kind": kind}

    if len(valid) < 2:
        return {
            "checked": False,
            "drifted": None,
            "supported_drift": None,
            "raw_spread": None,
            "threshold": threshold,
            "estimates": valid,
            "issues": [],
            "note": (
                "Fewer than two per-pass clock estimates, so drift across replay "
                "passes could not be assessed. Nothing was validated -- this is "
                "not a clean result."
            ),
        }

    clocks = [e["clock_hz"] for e in valid.values()]
    raw_spread = max(clocks) / min(clocks) - 1.0

    # The largest gap whose direction the estimator kinds can actually support.
    best: Optional[Tuple[float, str, str]] = None  # (ratio-1, hi_label, lo_label)
    for hi_label, hi in valid.items():
        for lo_label, lo in valid.items():
            if hi_label == lo_label or hi["clock_hz"] <= lo["clock_hz"]:
                continue
            if lo["kind"] != "measured":
                # Anything sitting above a lower bound proves nothing: the
                # bound may simply undercount its own pass's clock.
                continue
            gap = hi["clock_hz"] / lo["clock_hz"] - 1.0
            if best is None or gap > best[0]:
                best = (gap, hi_label, lo_label)

    supported = best[0] if best else 0.0
    drifted = supported >= threshold

    issues: List[SamplingIssue] = []
    if drifted and best is not None:
        gap, hi_label, lo_label = best
        hi_mhz = valid[hi_label]["clock_hz"] / 1e6
        lo_mhz = valid[lo_label]["clock_hz"] / 1e6
        qualifier = (
            " (a lower-bound estimate, so the true gap is at least this)"
            if valid[hi_label]["kind"] == "lower_bound"
            else ""
        )
        issues.append(
            SamplingIssue(
                key="replay_clock_drift",
                title="SM clock drifted across the replay passes of this collection",
                detail=(
                    f"The effective SM clock differs by at least {gap * 100:.1f}% "
                    f"between replay passes of this one collection: {lo_label} at "
                    f"{lo_mhz:.0f} MHz against {hi_label} at {hi_mhz:.0f} MHz"
                    f"{qualifier}. Metrics multiplexed across passes therefore mix "
                    "different clock states, and PM-sampling series from different "
                    "passes are additionally skewed relative to each other. Every "
                    "counter is a faithful reading of its own pass; it is the "
                    "combination that mixes clock states."
                ),
                severity="medium",
                blocks=False,
                remedy=(
                    "Pin the SM clock externally before profiling "
                    "(nvidia-smi -lgc <clock>,<clock>, with --clock-control none), "
                    "or collect fewer sections so the collection spans fewer "
                    "replay passes."
                ),
            )
        )

    if drifted and best is not None:
        note = (
            f"Effective SM clock spans {supported * 100:.1f}% across replay "
            f"passes ({best[2]} vs {best[1]})."
        )
    elif raw_spread >= threshold:
        note = (
            f"Estimates spread {raw_spread * 100:.1f}%, but the spread is not "
            "attributable to the clock: the low side is an activity-derived "
            "lower bound that may undercount its pass. Inconclusive."
        )
    else:
        note = f"Per-pass clock estimates agree within {raw_spread * 100:.1f}%."

    return {
        "checked": True,
        "drifted": drifted,
        "supported_drift": supported,
        "raw_spread": raw_spread,
        "threshold": threshold,
        "estimates": valid,
        "issues": [i.to_dict() for i in issues],
        "note": note,
    }
