"""step_timer.py — Per-training-step timing with rolling statistics and anomaly detection.

This module is **framework-agnostic** and works with any training loop
(PyTorch DDP, DeepSpeed, Megatron-LM, FSDP, plain inference loops, etc.).
It does NOT require CUDA, torch.distributed, or any optional dependencies.

Typical usage::

    from my_utils import StepTimer

    timer = StepTimer(window_size=100, anomaly_threshold=2.5, rank=rank)

    for step in range(max_steps):
        with timer.step(step):
            loss = model(batch)
            loss.backward()
            optimizer.step()

    if rank == 0:
        print(timer.summary())
        for a in timer.get_anomalies():
            print(f"  slow step {a.step}: {a.duration*1000:.1f}ms ({a.ratio:.2f}x mean)")
"""

import logging
import math
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StepStats:
    """Summary statistics for a collection of step timings."""
    count: int
    total: float    # seconds
    mean: float     # seconds
    std: float      # seconds
    min: float      # seconds
    max: float      # seconds
    p50: float      # seconds
    p95: float      # seconds
    p99: float      # seconds


@dataclass
class AnomalyRecord:
    """Record of a single anomalously-slow training step."""
    step: int
    duration: float         # seconds
    mean_at_detection: float  # rolling mean at the moment of detection
    ratio: float            # duration / mean_at_detection


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StepTimer:
    """Per-training-step wall-clock timer with rolling statistics and anomaly detection.

    Designed to be as lightweight as possible — pure Python, no external deps.

    Args:
        window_size: Number of recent steps used for the rolling mean/std.
            Larger values give smoother statistics; smaller values react faster
            to training speed changes (e.g., after a checkpoint load).
        anomaly_threshold: A step is flagged when
            ``duration > rolling_mean * anomaly_threshold``.
            2.5 is a good default (flags steps >2.5x slower than the mean).
        warmup_steps: Steps that are excluded from anomaly detection.
            GPU/framework warmup artificially inflates the first few steps.
        rank: Current process rank. Only rank 0 emits anomaly warnings by
            default (pass ``rank=None`` to always emit).
        enabled: When ``False``, all operations are no-ops (zero overhead).
            Useful for controlled A/B benchmarks.
    """

    def __init__(
        self,
        window_size: int = 100,
        anomaly_threshold: float = 2.5,
        warmup_steps: int = 5,
        rank: int = 0,
        enabled: bool = True,
    ) -> None:
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.warmup_steps = warmup_steps
        self.rank = rank
        self.enabled = enabled

        self._durations: List[float] = []
        self._steps: List[int] = []
        self._anomalies: List[AnomalyRecord] = []
        self._start_time: Optional[float] = None
        self._current_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Context manager API (recommended)
    # ------------------------------------------------------------------

    @contextmanager
    def step(self, step_idx: int) -> Generator[None, None, None]:
        """Context manager that times a single training step.

        Example::

            with timer.step(global_step):
                forward_backward_and_step()
        """
        if not self.enabled:
            yield
            return
        self._begin(step_idx)
        try:
            yield
        finally:
            self._end()

    # ------------------------------------------------------------------
    # Manual start / stop API (for code that cannot use context managers)
    # ------------------------------------------------------------------

    def start(self, step_idx: int) -> None:
        """Begin timing step *step_idx*.  Must be paired with :meth:`stop`."""
        if self.enabled:
            self._begin(step_idx)

    def stop(self) -> float:
        """End timing the current step.

        Returns:
            Duration in seconds, or 0.0 if not enabled / not started.
        """
        if not self.enabled or self._start_time is None:
            return 0.0
        return self._end()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self, window: Optional[int] = None) -> Optional[StepStats]:
        """Compute timing statistics over recorded steps.

        Args:
            window: If provided, restrict analysis to the last *window* steps.

        Returns:
            :class:`StepStats` or ``None`` if no data has been recorded yet.
        """
        durations = self._durations[-window:] if window else self._durations
        if not durations:
            return None
        sorted_d = sorted(durations)
        n = len(sorted_d)

        def _pct(p: float) -> float:
            idx = max(min(int(math.ceil(p / 100.0 * n)) - 1, n - 1), 0)
            return sorted_d[idx]

        return StepStats(
            count=n,
            total=sum(durations),
            mean=statistics.mean(durations),
            std=statistics.stdev(durations) if n > 1 else 0.0,
            min=sorted_d[0],
            max=sorted_d[-1],
            p50=_pct(50),
            p95=_pct(95),
            p99=_pct(99),
        )

    def get_anomalies(self) -> List[AnomalyRecord]:
        """Return all detected anomalous steps (copies of internal records)."""
        return list(self._anomalies)

    def last_duration(self) -> Optional[float]:
        """Duration of the most recently completed step in seconds."""
        return self._durations[-1] if self._durations else None

    def throughput(self, batch_size: int, window: Optional[int] = None) -> Optional[float]:
        """Compute samples-per-second throughput.

        Args:
            batch_size: Number of samples (tokens, images, …) per step.
            window: If provided, use only the last *window* steps.

        Returns:
            Samples per second, or ``None`` if no data available.
        """
        stats = self.get_stats(window=window)
        if stats is None or stats.mean == 0.0:
            return None
        return batch_size / stats.mean

    def per_step_report(self) -> Dict[int, float]:
        """Return a ``{step_idx: duration_seconds}`` dict for all recorded steps."""
        return dict(zip(self._steps, self._durations))

    def summary(self, title: str = "StepTimer Summary") -> str:
        """Return a human-readable summary table."""
        stats = self.get_stats()
        if stats is None:
            return f"{title}: no data recorded."
        sep = "=" * 52
        rows = [
            sep,
            f"  {title}",
            sep,
            f"  Steps recorded  : {stats.count}",
            f"  Total wall time : {stats.total:.3f} s",
            f"  Mean / step     : {stats.mean * 1e3:.2f} ms",
            f"  Std  / step     : {stats.std  * 1e3:.2f} ms",
            f"  Min  / step     : {stats.min  * 1e3:.2f} ms",
            f"  Max  / step     : {stats.max  * 1e3:.2f} ms",
            f"  p50  / step     : {stats.p50  * 1e3:.2f} ms",
            f"  p95  / step     : {stats.p95  * 1e3:.2f} ms",
            f"  p99  / step     : {stats.p99  * 1e3:.2f} ms",
            f"  Anomalies       : {len(self._anomalies)} step(s)",
            sep,
        ]
        return "\n".join(rows)

    def reset(self) -> None:
        """Clear all recorded data (statistics and anomalies)."""
        self._durations.clear()
        self._steps.clear()
        self._anomalies.clear()
        self._start_time = None
        self._current_step = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _begin(self, step_idx: int) -> None:
        self._current_step = step_idx
        self._start_time = time.perf_counter()

    def _end(self) -> float:
        duration = time.perf_counter() - self._start_time  # type: ignore[operator]
        self._start_time = None
        step_idx = self._current_step

        self._durations.append(duration)
        self._steps.append(step_idx)  # type: ignore[arg-type]

        self._check_anomaly(step_idx, duration)  # type: ignore[arg-type]
        return duration

    def _check_anomaly(self, step_idx: int, duration: float) -> None:
        """Flag and log the step if it is anomalously slow."""
        n = len(self._durations)
        if n <= self.warmup_steps:
            return  # skip warmup steps

        window = self._durations[-self.window_size:]
        if not window:
            return

        mean = statistics.mean(window)
        if mean > 0.0 and duration > mean * self.anomaly_threshold:
            record = AnomalyRecord(
                step=step_idx,
                duration=duration,
                mean_at_detection=mean,
                ratio=duration / mean,
            )
            self._anomalies.append(record)
            if self.rank == 0:
                logger.warning(
                    "[StepTimer] Slow step detected — step=%d, duration=%.3fs "
                    "(%.2fx rolling mean=%.3fs)",
                    step_idx, duration, record.ratio, mean,
                )
