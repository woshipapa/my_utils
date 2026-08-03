# SPDX-License-Identifier: Apache-2.0
"""How a measurement was taken, and what that makes it unable to answer.

Two correct measurements of the same kernel disagree by 2x routinely, and the
reason is almost never the kernel. Nsight Compute flushes every GPU cache before
each replay pass by default (``--cache-control all``), so an ncu duration is a
*cold-cache* number. A wall-clock benchmark of the same kernel in a real
pipeline, where the previous kernel left its output in L2, is a *warm* number.
Neither is wrong. Comparing them is.

This module makes the collection mode a property of the measurement rather than
something the reader is assumed to know. Each mode carries what it cannot
answer, and :func:`compare_measurements` refuses comparisons across incompatible
modes instead of reporting the resulting difference as a finding.

The failures this prevents are all ones where the tool is confidently wrong:

* An ncu number and a wall-clock number differing by 2x, reported as a
  regression, when the difference is entirely cache state.
* A stream-overlap improvement measured under ncu, which serialises kernel
  execution to profile each one -- the gain is structurally invisible there and
  must come from a timeline.
* A long profiling loop thermally throttling the GPU and reporting the resulting
  decline as a property of the code. Colfax published a correction of exactly
  this shape, revising a 74% figure to 84% after finding the iteration count was
  the cause.
* Benchmarking on unrealistic inputs. Random +/-1 values are cheap for the
  hardware in ways real data is not.

None of these degrade a conclusion. Each one inverts it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "CacheState",
    "MeasurementContext",
    "describe_collection_mode",
    "compare_measurements",
    "assess_clock_control_bias",
    "measurement_collection_context",
    "NCU_DEFAULT_CACHE_CONTROL",
]

# Nsight Compute's default. Documented in the CLI reference under
# --cache-control; the default flushes all caches before each replay pass so
# that replays are reproducible, at the cost of never measuring a warm cache.
NCU_DEFAULT_CACHE_CONTROL = "all"

# These are the sidecar fields that affect a MeasurementContext.  NCU report
# sidecars also carry workload/build provenance; keeping that separate avoids
# silently accepting arbitrary metadata as a collection-mode argument.
_MEASUREMENT_COLLECTION_FIELDS = frozenset(
    {
        "cache_control",
        "replay_mode",
        "app_replay_match",
        "app_replay_mode",
        "range_replay_options",
        "graph_profiling",
        "iterations",
        "warmup_iterations",
        "clocks_locked",
        "input_distribution",
        "clock_control",
        "pipeline_boost_state",
        "mps_active",
        "mig_instance_id",
        "ncu_defaults_known",
    }
)


def measurement_collection_context(collection: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return only collection-mode fields consumed by this module.

    A collection sidecar deliberately has both collection controls and broader
    workload provenance (build id, input hash, logical kernel aliases).  The
    latter belongs to report-level comparability checks, not this constructor.
    """
    return {
        key: value
        for key, value in dict(collection or {}).items()
        if key in _MEASUREMENT_COLLECTION_FIELDS
    }

# Thresholds for the clock-control symptom (see assess_clock_control_bias).
# DRAM at >=97% of its rated clock is "running at full clock" -- on a real
# H100 report it sat at 99.97%. SM at <=92% of its rated clock is "held below
# it": a boost-clocked SM runs at 95-100% of the rated (peak) value, while a
# base-locked H100 sits near 80% and the real report measured 88%.
_DRAM_FULL_CLOCK_SHARE = 0.97
_SM_REDUCED_CLOCK_SHARE = 0.92


class CacheState:
    """Cache state a measurement was taken under."""

    COLD = "cold"  # caches flushed before the kernel (ncu default)
    WARM = "warm"  # whatever the preceding work left behind
    UNKNOWN = "unknown"


@dataclass
class MeasurementContext:
    """Everything about how a number was produced that changes what it means."""

    source: str = ""  # "ncu" | "nsys" | "wallclock" | ...
    cache_state: str = CacheState.UNKNOWN
    #: ncu serialises kernels to profile them; overlap cannot be observed.
    serialized_execution: bool = False
    #: Replays re-run the kernel; side-effecting kernels measure differently.
    replay_mode: str = ""  # kernel | application | range | app-range
    #: Application Replay identity/matching policy. Empty means unrecorded.
    app_replay_match: str = ""
    app_replay_mode: str = ""
    #: Range Replay and CUDA graph options affect the measurement entity.
    range_replay_options: str = ""
    graph_profiling: str = ""
    iterations: Optional[int] = None
    warmup_iterations: Optional[int] = None
    clocks_locked: Optional[bool] = None
    input_distribution: str = ""  # "real" | "random" | "ones" | ...
    #: Measured SM clock, Hz. Nsight Compute 2026.1+ defaults to
    #: ``--clock-control=boost`` while older releases used ``base``; the report
    #: does not record which was used. A duration measured at one clock is not
    #: comparable with one measured at another.
    sm_clock_hz: Optional[float] = None
    gpc_clock_hz: Optional[float] = None
    dram_clock_hz: Optional[float] = None
    #: Rated (peak) clocks from the device attributes, for judging how far the
    #: measured clocks sat below them. In Hz, not the kHz the attribute uses.
    rated_sm_clock_hz: Optional[float] = None
    rated_dram_clock_hz: Optional[float] = None
    #: ncu's --clock-control value when the caller knows it ("" when unknown --
    #: the report does not record it, so it usually is unknown).
    clock_control: str = ""
    #: Nsight Compute's scheduler boost policy (normally ``stable`` or
    #: ``dynamic``).  Dynamic boosting may vary between replay passes.
    pipeline_boost_state: str = ""
    #: Multiprocess/MIG state changes the resource and PM-sampling scope.  An
    #: empty value is unknown, rather than a claim that the GPU was exclusive.
    mps_active: Optional[bool] = None
    mig_instance_id: str = ""
    #: Output of :func:`assess_clock_control_bias`, or None when unassessed.
    clock_control_bias: Optional[Dict[str, Any]] = None
    notes: Tuple[str, ...] = ()

    @property
    def answers(self) -> Tuple[str, ...]:
        """Questions this measurement can answer."""
        out: List[str] = []
        if self.cache_state == CacheState.COLD:
            out.append("kernel cost in isolation, independent of what ran before it")
        if self.cache_state == CacheState.WARM:
            out.append("kernel cost in the pipeline it actually runs in")
        if not self.serialized_execution:
            out.append("concurrency and overlap between streams")
        if self.clocks_locked:
            out.append("run-to-run comparison, since the clock is pinned")
        return tuple(out)

    @property
    def cannot_answer(self) -> Tuple[str, ...]:
        """Questions this measurement structurally cannot answer.

        These are not caveats to weigh. They are questions where the measurement
        contains no information, and any number it produces is an artefact of
        the collection mode.
        """
        out: List[str] = []
        if self.cache_state == CacheState.COLD:
            out.append(
                "how fast this kernel runs in a pipeline: caches were flushed before "
                "it, so any reuse from a preceding kernel is excluded by construction"
            )
        if self.serialized_execution:
            out.append(
                "whether work overlaps: execution was serialised for profiling, so "
                "concurrency gains are invisible and stream parallelism cannot be seen"
            )
        if self.cache_state == CacheState.UNKNOWN:
            out.append(
                "comparison against any other measurement, since the cache state is "
                "unrecorded and cache state alone moves a duration by more than most "
                "optimisations do"
            )
        if self.clocks_locked is False:
            out.append(
                "run-to-run comparison at face value: clocks were not locked, so part "
                "of any difference is the clock rather than the code"
            )
        if self.clock_disagreement and self.clock_disagreement > 1.02:
            out.append(
                f"anything derived from cycles: the GPC and SM clocks in this "
                f"report differ by {(self.clock_disagreement - 1) * 100:.1f}%, so the "
                "two domains were not measured over the same window"
            )
        if self.input_distribution in ("random", "ones", "zeros"):
            out.append(
                f"performance on real data: inputs were '{self.input_distribution}', "
                "which can be materially cheaper for the hardware than real values"
            )
        if self.replay_mode.strip().lower() in {"range", "app-range"}:
            out.append(
                "per-kernel attribution from this collection: range replay counters "
                "describe the complete range, which can include concurrent kernels"
            )
        return tuple(out)

    @property
    def clock_disagreement(self) -> Optional[float]:
        """Ratio between the GPC and SM clocks, or None.

        They should agree. A gap means the two domains were not measured over
        the same window, which makes every cycle-derived figure in the report
        suspect. Seen at 5% on a real report whose other counters were also
        internally inconsistent.
        """
        if not self.sm_clock_hz or not self.gpc_clock_hz:
            return None
        return max(self.sm_clock_hz, self.gpc_clock_hz) / min(
            self.sm_clock_hz, self.gpc_clock_hz
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "sm_clock_hz": self.sm_clock_hz,
            "gpc_clock_hz": self.gpc_clock_hz,
            "dram_clock_hz": self.dram_clock_hz,
            "clock_control": self.clock_control,
            "pipeline_boost_state": self.pipeline_boost_state,
            "mps_active": self.mps_active,
            "mig_instance_id": self.mig_instance_id,
            "clock_control_bias": self.clock_control_bias,
            "clock_disagreement": self.clock_disagreement,
            "cache_state": self.cache_state,
            "serialized_execution": self.serialized_execution,
            "replay_mode": self.replay_mode,
            "app_replay_match": self.app_replay_match,
            "app_replay_mode": self.app_replay_mode,
            "range_replay_options": self.range_replay_options,
            "graph_profiling": self.graph_profiling,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "clocks_locked": self.clocks_locked,
            "input_distribution": self.input_distribution,
            "answers": list(self.answers),
            "cannot_answer": list(self.cannot_answer),
            "notes": list(self.notes),
        }


def assess_clock_control_bias(
    *,
    sm_clock_hz: Optional[float] = None,
    dram_clock_hz: Optional[float] = None,
    rated_sm_clock_hz: Optional[float] = None,
    rated_dram_clock_hz: Optional[float] = None,
    clock_control: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect the compute:memory bias of profiling with a lowered SM clock.

    Documented behaviour (Nsight Compute Profiling Guide 2026): explicitly
    selecting ``--clock-control base`` locks the SM clock to base but **cannot lower the
    HBM clock** on H100/B200-class parts. The kernel then computes at a reduced
    rate against memory running at full rate, so every compute-vs-memory
    verdict built from the report -- Speed-of-Light compute against memory
    throughput, roofline side -- reads more memory-rich (compute-poor) than
    the same kernel at full clock. The numbers are not wrong; the *balance*
    between them is a property of the clock state, and this names it.

    Two detection paths, tried in order:

    * **Declared**: the caller knows ``--clock-control`` was ``base``. The
      report itself does not record the option, so this path is only available
      when the invocation is known.
    * **Symptom**: the measured SM clock sits well below the device's rated
      (peak) clock while the measured DRAM clock sits at its rated value.
      Rated clocks come from ``device__attribute_clock_rate`` and
      ``device__attribute_memory_clock_rate`` (kHz in the report; pass Hz
      here). Limits: the symptom cannot distinguish ``--clock-control base``
      from power or thermal capping -- but either way the SM:DRAM clock ratio
      during the measurement was below the device's rated ratio, and the bias
      direction is the same, so the caveat holds regardless of cause.

    This is an interpretation caveat, not a finding: it does not claim any
    counter is wrong, only that the compute:memory balance leans a known way.
    """
    control = str(clock_control or "").strip().lower()

    remedy = (
        "For an unbiased balance, pin the SM clock externally "
        "(nvidia-smi -lgc <clock>,<clock>) and profile with "
        "--clock-control none."
    )
    tail = (
        "Nsight Compute's --clock-control base lowers the SM clock but cannot "
        "lower the HBM clock on H100/B200-class parts, so the measured "
        "compute:memory balance is biased toward looking memory-rich "
        "(compute-poor) relative to full-clock operation. The counters are not "
        "wrong; verdicts that weigh compute against memory bandwidth "
        "(Speed-of-Light compute vs memory, roofline side) lean memory-rich. " + remedy
    )

    if control == "base":
        return {
            "checked": True,
            "biased": True,
            "method": "declared",
            "sm_share_of_rated": None,
            "dram_share_of_rated": None,
            "caveat": (
                "--clock-control base was in effect: the SM clock is "
                "locked at base while HBM runs at full clock. " + tail
            ),
            "limits": "",
        }

    values = (sm_clock_hz, dram_clock_hz, rated_sm_clock_hz, rated_dram_clock_hz)
    if not all(v and v > 0 for v in values):
        return {
            "checked": False,
            "biased": None,
            "method": "",
            "sm_share_of_rated": None,
            "dram_share_of_rated": None,
            "caveat": "",
            "limits": (
                "Clock-control bias was not assessed: it needs the measured SM "
                "and DRAM clocks plus the device's rated clocks "
                "(device__attribute_clock_rate, "
                "device__attribute_memory_clock_rate), and at least one is "
                "missing. Unassessed is not the same as unbiased."
            ),
        }

    sm_share = sm_clock_hz / rated_sm_clock_hz
    dram_share = dram_clock_hz / rated_dram_clock_hz
    biased = (
        dram_share >= _DRAM_FULL_CLOCK_SHARE and sm_share <= _SM_REDUCED_CLOCK_SHARE
    )

    caveat = ""
    limits = ""
    if biased:
        caveat = (
            f"The SM ran at {sm_share * 100:.0f}% of its rated clock "
            f"({sm_clock_hz / 1e6:.0f} of {rated_sm_clock_hz / 1e6:.0f} MHz) "
            f"while DRAM ran at {dram_share * 100:.0f}% of its rated clock "
            f"({dram_clock_hz / 1e6:.0f} of {rated_dram_clock_hz / 1e6:.0f} MHz). "
            + tail
        )
        limits = (
            "Detected from the clock symptom, not from recorded profiler "
            "options: the same signature is produced by power or thermal "
            "capping. Either way the SM:DRAM clock ratio during the "
            "measurement was below the device's rated ratio, and the bias "
            "direction is the same."
        )

    return {
        "checked": True,
        "biased": biased,
        "method": "symptom",
        "sm_share_of_rated": sm_share,
        "dram_share_of_rated": dram_share,
        "caveat": caveat,
        "limits": limits,
    }


def describe_collection_mode(
    *,
    source: str = "ncu",
    cache_control: Optional[str] = None,
    ncu_defaults_known: bool = True,
    replay_mode: str = "",
    app_replay_match: str = "",
    app_replay_mode: str = "",
    range_replay_options: str = "",
    graph_profiling: str = "",
    iterations: Optional[int] = None,
    warmup_iterations: Optional[int] = None,
    clocks_locked: Optional[bool] = None,
    input_distribution: str = "",
    sm_clock_hz: Optional[float] = None,
    gpc_clock_hz: Optional[float] = None,
    dram_clock_hz: Optional[float] = None,
    rated_sm_clock_hz: Optional[float] = None,
    rated_dram_clock_hz: Optional[float] = None,
    clock_control: Optional[str] = None,
    pipeline_boost_state: str = "",
    mps_active: Optional[bool] = None,
    mig_instance_id: str = "",
) -> MeasurementContext:
    """Build a :class:`MeasurementContext` from how the tool was invoked.

    ``cache_control`` is ncu's ``--cache-control`` value. Passing ``None`` for an
    ncu run means the default was in effect only when ``ncu_defaults_known`` is
    true.  The YAML collector can assert that because it wrote the command;
    imported historical reports cannot, and must remain unknown rather than
    being silently labelled cold-cache.
    """
    src = (source or "").strip().lower()
    notes: List[str] = []

    if src == "ncu":
        effective = (
            (cache_control or NCU_DEFAULT_CACHE_CONTROL).strip().lower()
            if cache_control is not None or ncu_defaults_known
            else ""
        )
        if effective in ("all", "system"):
            cache_state = CacheState.COLD
            if cache_control is None:
                notes.append(
                    "--cache-control was not specified, so Nsight Compute's default "
                    "('all') applied and every cache was flushed before each replay "
                    "pass. These durations are cold-cache and will read slower than "
                    "the same kernel in a pipeline. Use --cache-control none for "
                    "pipeline-realistic numbers."
                )
        elif effective == "none":
            cache_state = CacheState.WARM
            notes.append(
                "--cache-control none: caches were not flushed, so these numbers "
                "include reuse from preceding work and are comparable with "
                "wall-clock timings rather than with default ncu runs."
            )
        else:
            cache_state = CacheState.UNKNOWN
            if cache_control is None and not ncu_defaults_known:
                notes.append(
                    "The report has no collection sidecar, so its NCU command and "
                    "cache-control setting are unknown. Do not treat its duration as "
                    "a cold-cache default or compare it as a speedup until provenance "
                    "is supplied."
                )
        replay = str(replay_mode or "").strip().lower()
        # Range replay captures the range as an entity so concurrent launches
        # can execute together. Kernel and application replay are still
        # interpreted as per-launch, serialised measurements.
        serialized = replay not in {"range", "app-range"}
        if serialized:
            notes.append(
                "Nsight Compute serialises kernel execution to profile each launch, so "
                "no overlap between streams is observable here. Stream concurrency has "
                "to be measured with Nsight Systems."
            )
        else:
            notes.append(
                "Range Replay preserves the range as the measurement entity and can "
                "include concurrent launches. Its aggregate counters cannot be "
                "assigned to an individual kernel."
            )
    elif src in ("nsys", "wallclock", "timer", "cuda_event"):
        cache_state = CacheState.WARM
        serialized = False
    else:
        cache_state = CacheState.UNKNOWN
        serialized = False

    if iterations is not None and iterations > 100 and clocks_locked is not True:
        notes.append(
            f"{iterations} iterations without locked clocks: a long profiling loop "
            "heats the part and the later iterations can run at a lower clock. A "
            "published H100 result was revised from 74% to 84% of peak after exactly "
            "this was found to be the cause."
        )
    if warmup_iterations == 0 or (warmup_iterations is None and src != "ncu"):
        notes.append(
            "No warmup iterations recorded. The first launch pays JIT, module load "
            "and allocator costs that belong to neither the kernel nor the steady "
            "state."
        )
    boost = str(pipeline_boost_state or "").strip().lower()
    if boost == "dynamic":
        notes.append(
            "--pipeline-boost-state dynamic lets the profiler adjust pipeline "
            "boosting during collection. Treat small replay-to-replay differences "
            "as scheduling-sensitive unless repeat runs agree."
        )
    if mps_active is True:
        notes.append(
            "MPS was active. Resource availability and PM-sampling scope can be "
            "shared with other clients; compare only against the same MPS setup."
        )
    if mig_instance_id:
        notes.append(
            "This report ran on MIG instance "
            f"{mig_instance_id!r}; its resource partition is part of the measurement."
        )

    if sm_clock_hz and gpc_clock_hz:
        ratio = max(sm_clock_hz, gpc_clock_hz) / min(sm_clock_hz, gpc_clock_hz)
        if ratio > 1.02:
            notes.append(
                f"GPC clock {gpc_clock_hz / 1e6:.0f} MHz against SM clock "
                f"{sm_clock_hz / 1e6:.0f} MHz -- a {(ratio - 1) * 100:.1f}% gap between "
                "domains that should agree. Treat cycle-derived figures in this "
                "report as suspect and re-collect."
            )

    # A lowered SM clock against full-clock HBM biases every compute:memory
    # verdict; an interpretation caveat, carried with the context because it is
    # a property of how the numbers were taken, not of the kernel.
    bias = assess_clock_control_bias(
        sm_clock_hz=sm_clock_hz,
        dram_clock_hz=dram_clock_hz,
        rated_sm_clock_hz=rated_sm_clock_hz,
        rated_dram_clock_hz=rated_dram_clock_hz,
        clock_control=clock_control,
    )
    if bias.get("biased"):
        notes.append(
            bias["caveat"] + ((" " + bias["limits"]) if bias.get("limits") else "")
        )

    return MeasurementContext(
        source=src,
        sm_clock_hz=sm_clock_hz,
        gpc_clock_hz=gpc_clock_hz,
        dram_clock_hz=dram_clock_hz,
        rated_sm_clock_hz=rated_sm_clock_hz,
        rated_dram_clock_hz=rated_dram_clock_hz,
        clock_control=str(clock_control or ""),
        pipeline_boost_state=boost,
        mps_active=mps_active,
        mig_instance_id=str(mig_instance_id or ""),
        clock_control_bias=bias if bias.get("checked") else None,
        cache_state=cache_state,
        serialized_execution=serialized,
        replay_mode=replay_mode,
        app_replay_match=app_replay_match,
        app_replay_mode=app_replay_mode,
        range_replay_options=range_replay_options,
        graph_profiling=graph_profiling,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        clocks_locked=clocks_locked,
        input_distribution=input_distribution,
        notes=tuple(notes),
    )


# How a metric responds to the clock. Anything not recognised is treated as
# clock-independent, because inventing a correction is worse than omitting one.
_DURATION_HINTS = ("duration", "time", "latency", "elapsed_ns", "_ns", "us", "ms")
_RATE_HINTS = (
    "throughput",
    "flops",
    "flop_s",
    "tflop",
    "bandwidth",
    "per_second",
    "gbps",
    "bytes_per",
    "rate",
)


def _metric_kind(metric: str) -> str:
    """`duration`, `rate`, or `clock_independent`."""
    low = str(metric or "").lower()
    if any(h in low for h in _RATE_HINTS):
        return "rate"
    if any(h in low for h in _DURATION_HINTS):
        return "duration"
    return "clock_independent"


def compare_measurements(
    baseline: MeasurementContext,
    candidate: MeasurementContext,
    *,
    baseline_value: Optional[float] = None,
    candidate_value: Optional[float] = None,
    metric: str = "duration",
) -> Dict[str, Any]:
    """Decide whether two measurements may be compared at all.

    Returns ``comparable: False`` with a reason when the collection modes differ
    in a way that changes the number. The refusal is the point -- reporting a
    2x "regression" that is entirely cache state is worse than reporting
    nothing, because it sends someone to optimise a kernel that did not change.
    """
    blockers: List[str] = []

    if CacheState.UNKNOWN in (baseline.cache_state, candidate.cache_state):
        blockers.append(
            "at least one measurement has an unrecorded cache state, and cache state "
            "moves a duration by more than most optimisations do"
        )
    elif baseline.cache_state != candidate.cache_state:
        blockers.append(
            f"cache state differs ({baseline.cache_state} vs {candidate.cache_state}). "
            "A cold-cache measurement and a warm one are different quantities; the "
            "difference between them is not a change in the code"
        )

    if baseline.serialized_execution != candidate.serialized_execution:
        blockers.append(
            "one measurement serialised execution and the other did not, so any "
            "difference includes overlap that only one of them could observe"
        )

    if (
        baseline.replay_mode
        and candidate.replay_mode
        and baseline.replay_mode != candidate.replay_mode
    ):
        blockers.append(
            f"replay modes differ ({baseline.replay_mode} vs {candidate.replay_mode}); "
            "kernel, application, and range replay do not measure the same entity"
        )

    for field, label in (
        ("app_replay_match", "application-replay matching"),
        ("app_replay_mode", "application-replay mode"),
        ("range_replay_options", "range-replay options"),
        ("graph_profiling", "graph-profiling mode"),
        ("pipeline_boost_state", "pipeline boost state"),
        ("mig_instance_id", "MIG instance"),
    ):
        left, right = getattr(baseline, field), getattr(candidate, field)
        if left and right and left != right:
            blockers.append(
                f"{label} differs ({left!r} vs {right!r}), so launch identity or "
                "the measured execution scope can differ"
            )

    if (
        baseline.mps_active is not None
        and candidate.mps_active is not None
        and baseline.mps_active != candidate.mps_active
    ):
        blockers.append(
            "MPS state differs between reports, so the SM partition and PM-sampling "
            "scope are not the same measurement"
        )

    if (
        baseline.input_distribution
        and candidate.input_distribution
        and baseline.input_distribution != candidate.input_distribution
    ):
        blockers.append(
            f"input distributions differ ('{baseline.input_distribution}' vs "
            f"'{candidate.input_distribution}'), which alone can move throughput by "
            "20% or more"
        )

    for side, context in (("baseline", baseline), ("candidate", candidate)):
        if context.clock_disagreement and context.clock_disagreement > 1.02:
            blockers.append(
                f"the {side} report's GPC and SM clocks disagree by "
                f"{(context.clock_disagreement - 1) * 100:.1f}%, indicating replay "
                "passes were not measured over a stable clock window"
            )

    ratio = None
    if baseline_value and candidate_value:
        ratio = candidate_value / baseline_value

    # Duration measured at two different clocks is two different quantities.
    # On a real pair of reports the wall-clock gap was 12.4% and the
    # cycle-normalised gap 5.2% -- half the apparent speedup was the clock.
    # How the clock enters depends on what is being compared. A duration scales
    # inversely with the clock; a throughput scales with it; a byte count and a
    # cycle count do not depend on it at all. Applying the duration correction
    # to everything invented a 6.9% difference between two runs that moved
    # byte-identical traffic, and inverted the correction for throughputs.
    kind = _metric_kind(metric)
    clock_ratio = None
    if baseline.sm_clock_hz and candidate.sm_clock_hz:
        clock_ratio = candidate.sm_clock_hz / baseline.sm_clock_hz
        if kind == "clock_independent":
            pass  # bytes, counts, cycles: the clock does not enter
        elif abs(clock_ratio - 1.0) > 0.01:
            blockers.append(
                f"the two measurements ran at different SM clocks "
                f"({baseline.sm_clock_hz / 1e6:.0f} vs "
                f"{candidate.sm_clock_hz / 1e6:.0f} MHz, {(clock_ratio - 1) * 100:+.1f}%). "
                f"A {kind.replace('_', '-')} ratio therefore mixes the change in the "
                "code with the change in the clock; compare cycles, or lock the "
                "clock and re-measure"
            )
    elif (
        not (baseline.sm_clock_hz and candidate.sm_clock_hz)
        and kind != "clock_independent"
    ):
        # Failing open here let a comparison through with no clock evidence at
        # all, which is the case most likely to be confounded.
        blockers.append(
            "the SM clock is unrecorded on at least one side, so whether the two "
            "ran at the same frequency is unknown"
        )

    result: Dict[str, Any] = {
        "metric": metric,
        "comparable": not blockers,
        "blockers": blockers,
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "baseline_value": baseline_value,
        "candidate_value": candidate_value,
    }

    if blockers:
        result["ratio"] = None
        result["verdict"] = (
            "Not comparable. " + " Also, ".join(blockers).capitalize() + ". "
            "Re-measure both sides the same way before drawing a conclusion."
        )
        # The raw ratio is still returned under a name that cannot be mistaken
        # for a result, because someone will want it for debugging.
        result["uncomparable_raw_ratio"] = ratio
        if clock_ratio and ratio and kind != "clock_independent":
            # A duration shrinks as the clock rises, so multiply; a throughput
            # grows with it, so divide.
            result["clock_ratio"] = clock_ratio
            result["metric_kind"] = kind
            result["clock_normalised_ratio"] = (
                ratio * clock_ratio if kind == "duration" else ratio / clock_ratio
            )
            result["verdict"] += (
                f" Normalising for the clock leaves "
                f"{ratio * clock_ratio:.3f}x of the {ratio:.3f}x observed."
            )
    else:
        result["ratio"] = ratio
        result["verdict"] = (
            f"Comparable: both measured {baseline.cache_state}-cache from "
            f"{baseline.source}." + (f" Ratio {ratio:.2f}x." if ratio else "")
        )

    # Warnings that do not block but change how much the difference means.
    soft: List[str] = []
    if baseline.clocks_locked is not True or candidate.clocks_locked is not True:
        soft.append(
            "Clocks were not confirmed locked on both sides, so small differences "
            "may be clock rather than code."
        )
    result["caveats"] = soft
    return result
