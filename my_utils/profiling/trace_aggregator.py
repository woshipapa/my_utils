"""trace_aggregator.py — Merge per-rank GlobalLogger CSV files into a single Chrome Trace JSON.

GlobalLogger writes one ``profile_rank_<N>.csv`` per process.  When training
finishes you can use :class:`TraceAggregator` to:

1. Load all per-rank CSV files from a directory.
2. Apply per-rank clock offsets (produced by :class:`~my_utils.ClockSynchronizer`)
   so that events from different machines/GPUs share a common time base.
3. Export a single ``trace.json`` that can be opened in:
   - **Perfetto UI** — https://ui.perfetto.dev  (recommended, handles large traces)
   - **chrome://tracing** — paste the JSON path in the URL bar

CSV column order (written by ``GlobalLogger.log_profile_event``)::

    timestamp_unix, readable_time, machine_id, step, event_name, event_type,
    duration_ms, metadata

``event_type`` is either ``START`` or ``END``.

Example::

    from my_utils.profiling import TraceAggregator

    agg = TraceAggregator()
    agg.load_directory("/path/to/logs", pattern="profile_rank_*.csv")

    # Optional: apply clock offsets from ClockSynchronizer
    agg.set_clock_offset(rank=1, offset_seconds=-0.003)   # rank-1 is 3 ms ahead

    agg.export_chrome_trace("/path/to/logs/merged_trace.json")
    print(agg.rank_summary())
"""

import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class _RawEvent:
    """One CSV row parsed into a structured object."""
    timestamp_unix: float   # seconds, wall clock
    machine_id: str
    step: int
    event_name: str
    event_type: str         # "START" or "END"
    duration_ms: float
    metadata: str
    rank: int               # inferred from filename


@dataclass
class RankStat:
    """Per-rank event statistics."""
    rank: int
    machine_id: str
    num_events: int
    first_ts: float         # earliest timestamp (seconds)
    last_ts: float          # latest timestamp (seconds)
    clock_offset: float     # applied offset (seconds)
    unique_event_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TraceAggregator:
    """Merge GlobalLogger per-rank CSV files into a Chrome Trace JSON.

    Args:
        default_pid_label: When ``True``, Chrome Trace ``pid`` is set to the
            rank integer so that different ranks appear as separate processes
            in the viewer.  When ``False``, all ranks share pid=0 and are
            separated only by ``tid``.
    """

    def __init__(self, default_pid_label: bool = True) -> None:
        self._events: List[_RawEvent] = []
        self._clock_offsets: Dict[int, float] = {}   # rank → offset in seconds
        self._rank_meta: Dict[int, RankStat] = {}
        self._default_pid_label = default_pid_label

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_file(self, csv_path: str, rank: Optional[int] = None) -> int:
        """Load a single ``profile_rank_<N>.csv`` file.

        Args:
            csv_path: Absolute path to the CSV file.
            rank: Override the rank inferred from the filename.  Useful if
                your files follow a non-standard naming convention.

        Returns:
            Number of events loaded from this file.
        """
        if rank is None:
            rank = _infer_rank(csv_path)

        loaded = 0
        with open(csv_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh):
                line = line.strip()
                if not line or lineno == 0 and line.startswith("timestamp"):
                    continue  # skip header
                event = _parse_csv_line(line, rank)
                if event is not None:
                    self._events.append(event)
                    loaded += 1

        logger.debug("Loaded %d events from %s (rank=%d)", loaded, csv_path, rank)
        return loaded

    def load_directory(
        self,
        log_dir: str,
        pattern: str = "profile_rank_*.csv",
    ) -> int:
        """Scan *log_dir* for CSV files matching *pattern* and load them all.

        Args:
            log_dir: Directory that contains the per-rank CSV files.
            pattern: Glob pattern relative to *log_dir*.

        Returns:
            Total number of events loaded across all ranks.
        """
        paths = sorted(glob.glob(os.path.join(log_dir, pattern)))
        if not paths:
            logger.warning("No CSV files found in %s matching %s", log_dir, pattern)
            return 0

        total = 0
        for p in paths:
            total += self.load_file(p)
        logger.info("Loaded %d events from %d file(s) in %s", total, len(paths), log_dir)
        return total

    # ------------------------------------------------------------------
    # Clock offset correction
    # ------------------------------------------------------------------

    def set_clock_offset(self, rank: int, offset_seconds: float) -> None:
        """Apply a time correction to all events from *rank*.

        The offset is added to each event's timestamp::

            corrected_ts = raw_ts + offset_seconds

        Use negative offsets for clocks that run ahead.  Offsets are typically
        produced by :class:`~my_utils.ClockSynchronizer`.

        Args:
            rank: Target rank.
            offset_seconds: Signed time delta in seconds.
        """
        self._clock_offsets[rank] = offset_seconds

    def set_clock_offsets(self, offsets: Dict[int, float]) -> None:
        """Bulk-apply clock offsets from a ``{rank: offset_seconds}`` dict."""
        self._clock_offsets.update(offsets)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_chrome_trace(
        self,
        output_path: str,
        step_filter: Optional[Tuple[int, int]] = None,
        name_filter: Optional[str] = None,
    ) -> int:
        """Write a Chrome Trace JSON file.

        The file can be opened in Perfetto UI (https://ui.perfetto.dev) or
        chrome://tracing.

        Args:
            output_path: Destination ``.json`` file path.
            step_filter: If given as ``(start_step, end_step)``, only include
                events whose ``step`` is in ``[start_step, end_step)``.
            name_filter: If given, only include events whose ``event_name``
                matches this regex pattern.

        Returns:
            Number of Chrome Trace events written (each START/END becomes one
            "B"/"E" event, so the count is the same as input events).
        """
        _name_re = re.compile(name_filter) if name_filter else None

        trace_events: List[Dict] = []
        for ev in self._events:
            # Apply optional filters
            if step_filter and not (step_filter[0] <= ev.step < step_filter[1]):
                continue
            if _name_re and not _name_re.search(ev.event_name):
                continue

            offset = self._clock_offsets.get(ev.rank, 0.0)
            ts_us = (ev.timestamp_unix + offset) * 1e6  # → microseconds

            phase = "B" if ev.event_type == "START" else "E"

            entry: Dict = {
                "name": ev.event_name,
                "ph": phase,
                "ts": ts_us,
                "pid": ev.rank if self._default_pid_label else 0,
                "tid": 0,
                "args": {
                    "step": ev.step,
                    "machine_id": ev.machine_id,
                },
            }
            if ev.metadata:
                entry["args"]["metadata"] = ev.metadata
            if phase == "E" and ev.duration_ms > 0:
                entry["args"]["duration_ms"] = ev.duration_ms

            trace_events.append(entry)

        # Sort by timestamp for viewers that are sensitive to ordering
        trace_events.sort(key=lambda e: e["ts"])

        # Build metadata events (process/thread names)
        metadata_events = self._build_metadata_events()

        payload = {"traceEvents": metadata_events + trace_events}

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

        logger.info("Chrome Trace written to %s (%d events)", output_path, len(trace_events))
        return len(trace_events)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def rank_summary(self) -> str:
        """Return a human-readable per-rank event summary."""
        stats = self._compute_rank_stats()
        if not stats:
            return "TraceAggregator: no events loaded."

        lines = ["=" * 64, "  TraceAggregator — Per-rank summary", "=" * 64]
        for rank in sorted(stats):
            s = stats[rank]
            offset = self._clock_offsets.get(rank, 0.0)
            span = s.last_ts - s.first_ts
            lines.append(
                f"  Rank {rank:3d}  machine={s.machine_id:<20s}  "
                f"events={s.num_events:6d}  span={span:.3f}s  "
                f"offset={offset:+.6f}s"
            )
        lines.append("=" * 64)
        return "\n".join(lines)

    def num_events(self) -> int:
        """Total number of events currently loaded."""
        return len(self._events)

    def clear(self) -> None:
        """Remove all loaded events and offsets."""
        self._events.clear()
        self._clock_offsets.clear()
        self._rank_meta.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_rank_stats(self) -> Dict[int, RankStat]:
        agg: Dict[int, Dict] = {}
        for ev in self._events:
            r = ev.rank
            if r not in agg:
                agg[r] = {
                    "machine_id": ev.machine_id,
                    "num_events": 0,
                    "first_ts": ev.timestamp_unix,
                    "last_ts": ev.timestamp_unix,
                    "names": set(),
                }
            d = agg[r]
            d["num_events"] += 1
            d["first_ts"] = min(d["first_ts"], ev.timestamp_unix)
            d["last_ts"] = max(d["last_ts"], ev.timestamp_unix)
            d["names"].add(ev.event_name)

        return {
            r: RankStat(
                rank=r,
                machine_id=d["machine_id"],
                num_events=d["num_events"],
                first_ts=d["first_ts"],
                last_ts=d["last_ts"],
                clock_offset=self._clock_offsets.get(r, 0.0),
                unique_event_names=sorted(d["names"]),
            )
            for r, d in agg.items()
        }

    def _build_metadata_events(self) -> List[Dict]:
        """Emit Chrome Trace metadata events for human-readable pid/tid names."""
        stats = self._compute_rank_stats()
        meta: List[Dict] = []
        for rank, s in stats.items():
            pid = rank if self._default_pid_label else 0
            meta.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"Rank {rank} ({s.machine_id})"},
            })
            meta.append({
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": "main"},
            })
        return meta


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _infer_rank(csv_path: str) -> int:
    """Extract rank integer from a filename like ``profile_rank_3.csv``."""
    name = os.path.basename(csv_path)
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    logger.warning("Cannot infer rank from filename %s, defaulting to 0", name)
    return 0


def _parse_csv_line(line: str, rank: int) -> Optional[_RawEvent]:
    """Parse one CSV data row.  Returns ``None`` on malformed lines."""
    parts = line.split(",", maxsplit=7)
    if len(parts) < 7:
        return None
    try:
        return _RawEvent(
            timestamp_unix=float(parts[0]),
            machine_id=parts[2].strip(),
            step=int(parts[3]),
            event_name=parts[4].strip(),
            event_type=parts[5].strip().upper(),
            duration_ms=float(parts[6]),
            metadata=parts[7].strip() if len(parts) > 7 else "",
            rank=rank,
        )
    except (ValueError, IndexError):
        logger.debug("Skipping malformed CSV line: %r", line)
        return None
