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

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "CacheState",
    "MeasurementContext",
    "describe_collection_mode",
    "compare_measurements",
    "NCU_DEFAULT_CACHE_CONTROL",
]

# Nsight Compute's default. Documented in the CLI reference under
# --cache-control; the default flushes all caches before each replay pass so
# that replays are reproducible, at the cost of never measuring a warm cache.
NCU_DEFAULT_CACHE_CONTROL = "all"


class CacheState:
    """Cache state a measurement was taken under."""

    COLD = "cold"          # caches flushed before the kernel (ncu default)
    WARM = "warm"          # whatever the preceding work left behind
    UNKNOWN = "unknown"


@dataclass
class MeasurementContext:
    """Everything about how a number was produced that changes what it means."""

    source: str = ""                       # "ncu" | "nsys" | "wallclock" | ...
    cache_state: str = CacheState.UNKNOWN
    #: ncu serialises kernels to profile them; overlap cannot be observed.
    serialized_execution: bool = False
    #: Replays re-run the kernel; side-effecting kernels measure differently.
    replay_mode: str = ""                  # kernel | application | range
    iterations: Optional[int] = None
    warmup_iterations: Optional[int] = None
    clocks_locked: Optional[bool] = None
    input_distribution: str = ""           # "real" | "random" | "ones" | ...
    #: Measured SM clock, Hz. Nsight Compute defaults to --clock-control=base
    #: but the clock it *achieves* still varies run to run, and a duration
    #: measured at one clock is not comparable with one measured at another.
    sm_clock_hz: Optional[float] = None
    gpc_clock_hz: Optional[float] = None
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
            self.sm_clock_hz, self.gpc_clock_hz)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "sm_clock_hz": self.sm_clock_hz,
            "gpc_clock_hz": self.gpc_clock_hz,
            "clock_disagreement": self.clock_disagreement,
            "cache_state": self.cache_state,
            "serialized_execution": self.serialized_execution,
            "replay_mode": self.replay_mode,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "clocks_locked": self.clocks_locked,
            "input_distribution": self.input_distribution,
            "answers": list(self.answers),
            "cannot_answer": list(self.cannot_answer),
            "notes": list(self.notes),
        }


def describe_collection_mode(
    *,
    source: str = "ncu",
    cache_control: Optional[str] = None,
    replay_mode: str = "",
    iterations: Optional[int] = None,
    warmup_iterations: Optional[int] = None,
    clocks_locked: Optional[bool] = None,
    input_distribution: str = "",
    sm_clock_hz: Optional[float] = None,
    gpc_clock_hz: Optional[float] = None,
) -> MeasurementContext:
    """Build a :class:`MeasurementContext` from how the tool was invoked.

    ``cache_control`` is ncu's ``--cache-control`` value. Passing ``None`` for an
    ncu run means the default was in effect, which is ``all`` -- caches flushed.
    That default is the single most common reason an ncu duration disagrees with
    a wall-clock one, and assuming it rather than recording "unknown" is
    deliberate: the default is what almost every run uses.
    """
    src = (source or "").strip().lower()
    notes: List[str] = []

    if src == "ncu":
        effective = (cache_control or NCU_DEFAULT_CACHE_CONTROL).strip().lower()
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
        # ncu profiles one kernel at a time; concurrent execution is serialised.
        serialized = True
        notes.append(
            "Nsight Compute serialises kernel execution to profile each launch, so "
            "no overlap between streams is observable here. Stream concurrency has "
            "to be measured with Nsight Systems."
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

    if sm_clock_hz and gpc_clock_hz:
        ratio = max(sm_clock_hz, gpc_clock_hz) / min(sm_clock_hz, gpc_clock_hz)
        if ratio > 1.02:
            notes.append(
                f"GPC clock {gpc_clock_hz / 1e6:.0f} MHz against SM clock "
                f"{sm_clock_hz / 1e6:.0f} MHz -- a {(ratio - 1) * 100:.1f}% gap between "
                "domains that should agree. Treat cycle-derived figures in this "
                "report as suspect and re-collect."
            )

    return MeasurementContext(
        source=src,
        sm_clock_hz=sm_clock_hz,
        gpc_clock_hz=gpc_clock_hz,
        cache_state=cache_state,
        serialized_execution=serialized,
        replay_mode=replay_mode,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        clocks_locked=clocks_locked,
        input_distribution=input_distribution,
        notes=tuple(notes),
    )


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

    if (baseline.input_distribution and candidate.input_distribution
            and baseline.input_distribution != candidate.input_distribution):
        blockers.append(
            f"input distributions differ ('{baseline.input_distribution}' vs "
            f"'{candidate.input_distribution}'), which alone can move throughput by "
            "20% or more"
        )

    ratio = None
    if baseline_value and candidate_value:
        ratio = candidate_value / baseline_value

    # Duration measured at two different clocks is two different quantities.
    # On a real pair of reports the wall-clock gap was 12.4% and the
    # cycle-normalised gap 5.2% -- half the apparent speedup was the clock.
    clock_ratio = None
    if baseline.sm_clock_hz and candidate.sm_clock_hz:
        clock_ratio = candidate.sm_clock_hz / baseline.sm_clock_hz
        if abs(clock_ratio - 1.0) > 0.01:
            blockers.append(
                f"the two measurements ran at different SM clocks "
                f"({baseline.sm_clock_hz / 1e6:.0f} vs "
                f"{candidate.sm_clock_hz / 1e6:.0f} MHz, {(clock_ratio - 1) * 100:+.1f}%). "
                "A duration ratio therefore mixes the change in the code with the "
                "change in the clock; compare cycles, or lock the clock and "
                "re-measure"
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
        if clock_ratio and ratio:
            # The part of the difference that is not the clock.
            result["clock_ratio"] = clock_ratio
            result["clock_normalised_ratio"] = ratio * clock_ratio
            result["verdict"] += (
                f" Normalising for the clock leaves "
                f"{ratio * clock_ratio:.3f}x of the {ratio:.3f}x observed."
            )
    else:
        result["ratio"] = ratio
        result["verdict"] = (
            f"Comparable: both measured {baseline.cache_state}-cache from "
            f"{baseline.source}."
            + (f" Ratio {ratio:.2f}x." if ratio else "")
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
