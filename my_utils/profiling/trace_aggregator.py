"""trace_aggregator.py — Merge per-rank GlobalLogger CSV files into a Chrome Trace JSON.

Overview
--------
``MyTimer.synchronize_and_log()`` calls ``GlobalLogger.log_profile_event()`` at the
end of every training step, emitting one ``START`` and one ``END`` CSV row per stage::

    timestamp_unix,readable_time,machine_id,step,event_name,event_type,duration_ms,metadata
    1748000000.123456,14:30:00.123,Rank_0,5,forward,START,0.000,node_id=3
    1748000000.456789,14:30:00.456,Rank_0,5,forward,END,333.666,node_id=3

This module:

1. **Loads** all ``profile_rank_*.csv`` files from a log directory.
2. **Pairs** every START row with its matching END row (using ``node_id`` from the
   metadata field when present; falls back to a name-based stack for older logs).
3. **Corrects** timestamps with per-rank clock offsets
   (from :class:`~my_utils.ClockSynchronizer`).
4. **Exports** a single Chrome Trace JSON file, viewable at:

   - **https://ui.perfetto.dev** (recommended — handles large traces, better UX)
   - **chrome://tracing** — paste the ``.json`` path

   Visual hierarchy is **implicit**: because parent stages start before and end after
   their children, Perfetto nests them automatically without needing explicit
   parent-child links.

5. **Reports** per-stage/per-rank statistics (mean, std, p50, p95, p99, count) as a
   formatted text table.

6. Provides a **CLI** for post-training use::

       python -m my_utils.profiling.trace_aggregator \\
           --log-dir ./logs \\
           --output  ./logs/merged_trace.json \\
           --steps   10:50 \\
           --stats   ./logs/stage_stats.txt

Example Python usage::

    from my_utils.profiling import TraceAggregator

    agg = TraceAggregator()
    agg.load_directory("./logs")

    # Optional: apply clock offsets from ClockSynchronizer
    agg.set_clock_offsets({1: -0.003, 2: 0.001})   # rank→offset in seconds

    n = agg.export_chrome_trace("./logs/merged_trace.json")
    print(f"Wrote {n} trace events")
    print(agg.rank_summary())
    print(agg.stage_stats_table())
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chrome Trace colour palette
# Perfetto supports a fixed set of colour name strings.
# We cycle through them deterministically for consistent cross-run colouring.
# ---------------------------------------------------------------------------
_CHROME_COLOURS = [
    "rail_response",        # blue
    "rail_animation",       # green
    "rail_idle",            # grey
    "rail_load",            # yellow
    "good",                 # light-green
    "bad",                  # orange
    "terrible",             # red
    "generic_work",         # purple
    "background_memory_dump",
    "light_memory_dump",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "heap_dump_stack_frame",
    "thread_state_runnable",
    "thread_state_sleeping",
]


# ---------------------------------------------------------------------------
# Internal data containers
# ---------------------------------------------------------------------------

@dataclass
class _RawRow:
    """One parsed CSV row from GlobalLogger output."""
    rank: int
    timestamp: float    # wall-clock seconds (before offset correction)
    machine_id: str
    step: int
    event_name: str
    event_type: str     # "START" or "END"
    duration_ms: float  # only meaningful on END rows (CUDA elapsed)
    node_id: Optional[int]  # parsed from metadata "node_id=N"


@dataclass
class CompleteEvent:
    """A fully paired START+END event ready for Chrome Trace export."""
    rank: int
    machine_id: str
    step: int
    event_name: str
    start_ts: float     # wall-clock seconds, already clock-offset corrected
    end_ts: float       # wall-clock seconds, already clock-offset corrected
    duration_ms: float  # CUDA elapsed from END row (0 if unavailable)
    node_id: Optional[int]

    @property
    def wall_dur_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1e3


@dataclass
class StageStat:
    """Per-stage timing statistics aggregated across all steps and ranks."""
    event_name: str
    rank: int
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class RankStat:
    """High-level per-rank event summary."""
    rank: int
    machine_id: str
    num_raw_rows: int
    num_complete_events: int
    num_steps: int
    earliest_ts: float      # wall-clock seconds (original, before offset)
    latest_ts: float        # wall-clock seconds (original, before offset)
    clock_offset: float     # applied offset (seconds)
    unique_stages: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TraceAggregator:
    """Aggregate MyTimer per-rank CSV logs into a Chrome Trace JSON.

    Typical usage pattern::

        agg = TraceAggregator()
        agg.load_directory("./logs")
        agg.set_clock_offsets({1: -0.003})
        agg.export_chrome_trace("./logs/merged.json")
        print(agg.rank_summary())
        print(agg.stage_stats_table())
    """

    def __init__(self) -> None:
        # raw rows per rank, in file order
        self._raw: Dict[int, List[_RawRow]] = defaultdict(list)
        # paired complete events per rank (populated lazily)
        self._events: Dict[int, List[CompleteEvent]] = {}
        self._dirty: bool = False   # set True whenever raw data changes
        # clock offsets: rank → seconds (added to raw timestamp)
        self._offsets: Dict[int, float] = {}
        # stable colour map: event_name → colour string
        self._colours: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_file(self, csv_path: str, rank: Optional[int] = None) -> int:
        """Load a single ``profile_rank_<N>.csv`` file.

        Args:
            csv_path: Path to the CSV file.
            rank: Override rank inferred from the filename.

        Returns:
            Number of raw rows loaded (header excluded).
        """
        if rank is None:
            rank = _infer_rank_from_path(csv_path)

        loaded = 0
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for lineno, row in enumerate(reader):
                if lineno == 0 and row and row[0].strip().lower().startswith("timestamp"):
                    continue  # skip header
                parsed = _parse_row(row, rank)
                if parsed is not None:
                    self._raw[rank].append(parsed)
                    loaded += 1

        self._dirty = True
        logger.debug("Loaded %d rows from %s (rank=%d)", loaded, csv_path, rank)
        return loaded

    def load_directory(
        self,
        log_dir: str,
        pattern: str = "profile_rank_*.csv",
    ) -> int:
        """Scan *log_dir* for matching CSV files and load them all.

        Args:
            log_dir: Directory containing the per-rank CSVs.
            pattern: Glob pattern relative to *log_dir*.

        Returns:
            Total raw rows loaded across all ranks.
        """
        paths = sorted(glob.glob(os.path.join(log_dir, pattern)))
        if not paths:
            logger.warning(
                "No CSV files found in %s matching pattern %r", log_dir, pattern
            )
            return 0

        total = sum(self.load_file(p) for p in paths)
        logger.info(
            "Loaded %d rows from %d file(s) in %s", total, len(paths), log_dir
        )
        return total

    # ------------------------------------------------------------------
    # Clock offset correction
    # ------------------------------------------------------------------

    def set_clock_offset(self, rank: int, offset_seconds: float) -> None:
        """Apply an additive time correction to all events from *rank*.

        ``corrected_ts = raw_ts + offset_seconds``

        Use a **negative** offset for a clock that runs ahead.

        Args:
            rank: Target rank index.
            offset_seconds: Signed time delta in seconds.
        """
        self._offsets[rank] = offset_seconds
        self._dirty = True

    def set_clock_offsets(self, offsets: Dict[int, float]) -> None:
        """Bulk-apply clock offsets.  See :meth:`set_clock_offset`."""
        self._offsets.update(offsets)
        self._dirty = True

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_chrome_trace(
        self,
        output_path: str,
        step_range: Optional[Tuple[int, int]] = None,
        name_filter: Optional[str] = None,
        include_step_markers: bool = True,
    ) -> int:
        """Write a Chrome Trace JSON file.

        The result can be opened in Perfetto UI (https://ui.perfetto.dev)
        or chrome://tracing.

        Hierarchy is visualised **automatically**: parent stages (which
        encompass child stages in wall-clock time) will appear visually
        nested in the viewer without any extra work.

        Args:
            output_path: Destination ``.json`` path.
            step_range: ``(first_step, last_step_inclusive)`` — only include
                events whose ``step`` falls within this range.
            name_filter: If given, only include events whose ``event_name``
                matches this Python regex (via ``re.search``).
            include_step_markers: If ``True``, add an instant marker "I"
                event at the start of each step for easy navigation.

        Returns:
            Number of "X" complete events written.
        """
        self._ensure_paired()

        name_re = re.compile(name_filter) if name_filter else None

        trace_events: List[Dict] = []
        written = 0

        for rank, events in sorted(self._events.items()):
            for ev in events:
                if step_range and not (step_range[0] <= ev.step <= step_range[1]):
                    continue
                if name_re and not name_re.search(ev.event_name):
                    continue

                ts_us = ev.start_ts * 1e6
                dur_us = (ev.end_ts - ev.start_ts) * 1e6

                trace_events.append({
                    "name": ev.event_name,
                    "ph": "X",          # complete event (best for Perfetto)
                    "ts": ts_us,
                    "dur": max(dur_us, 0.0),
                    "pid": rank,
                    "tid": 0,
                    "cname": self._colour(ev.event_name),
                    "args": _build_args(ev),
                })
                written += 1

        # Step boundary instant markers (shown as vertical lines in Perfetto)
        if include_step_markers:
            step_starts: Dict[Tuple[int, int], float] = {}  # (rank, step) → ts_us
            for rank, events in self._events.items():
                for ev in events:
                    key = (rank, ev.step)
                    if key not in step_starts or ev.start_ts < step_starts[key] / 1e6:
                        step_starts[key] = ev.start_ts * 1e6
            for (rank, step), ts_us in step_starts.items():
                trace_events.append({
                    "name": f"step_{step}",
                    "ph": "I",
                    "ts": ts_us,
                    "pid": rank,
                    "tid": 0,
                    "s": "t",   # thread-scope instant
                    "args": {"step": step},
                })

        # Sort by timestamp so viewers don't need to re-sort
        trace_events.sort(key=lambda e: e["ts"])

        # Metadata events: process/thread names
        meta = self._build_metadata_events()

        payload = {
            "traceEvents": meta + trace_events,
            "displayTimeUnit": "ms",
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))

        logger.info(
            "Chrome Trace written → %s  (%d complete events, %d ranks)",
            output_path, written, len(self._events),
        )
        return written

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stage_stats(self) -> List[StageStat]:
        """Compute per-stage, per-rank timing statistics.

        Duration used is the **wall-clock** span of each paired event
        (i.e. ``end_ts - start_ts`` in ms), which equals the CUDA elapsed
        time recorded by MyTimer when CUDA timing is enabled.

        Returns:
            A list of :class:`StageStat` objects, one per (stage, rank) pair.
        """
        self._ensure_paired()

        # Collect durations: (event_name, rank) → [duration_ms, ...]
        buckets: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for rank, events in self._events.items():
            for ev in events:
                buckets[(ev.event_name, rank)].append(ev.wall_dur_ms)

        results: List[StageStat] = []
        for (name, rank), durs in sorted(buckets.items()):
            results.append(_compute_stat(name, rank, durs))
        return results

    def stage_stats_table(
        self,
        sort_by: str = "mean_ms",
        top_n: Optional[int] = None,
    ) -> str:
        """Return a formatted per-stage statistics table string.

        Args:
            sort_by: Column to sort by — one of
                ``"mean_ms"``, ``"std_ms"``, ``"max_ms"``, ``"count"``.
            top_n: If given, show only the top N stages by *sort_by*.

        Returns:
            A multi-line table string ready to ``print()``.
        """
        stats = self.get_stage_stats()
        if not stats:
            return "TraceAggregator: no events to report."

        stats.sort(key=lambda s: getattr(s, sort_by, 0.0), reverse=True)
        if top_n:
            stats = stats[:top_n]

        col_w = max(len(s.event_name) for s in stats)
        col_w = max(col_w, 12)
        sep = "=" * (col_w + 72)
        header = (
            f"  {'Stage':<{col_w}}  {'Rank':>4}  {'Count':>5}  "
            f"{'Mean(ms)':>9}  {'Std(ms)':>8}  {'Min(ms)':>8}  "
            f"{'Max(ms)':>8}  {'p50(ms)':>8}  {'p95(ms)':>8}  {'p99(ms)':>8}"
        )
        lines = [sep, f"  Stage Statistics  (sorted by {sort_by})", sep, header, sep]
        for s in stats:
            lines.append(
                f"  {s.event_name:<{col_w}}  {s.rank:>4}  {s.count:>5}  "
                f"{s.mean_ms:>9.2f}  {s.std_ms:>8.2f}  {s.min_ms:>8.2f}  "
                f"{s.max_ms:>8.2f}  {s.p50_ms:>8.2f}  {s.p95_ms:>8.2f}  {s.p99_ms:>8.2f}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def rank_summary(self) -> str:
        """Return a concise per-rank event summary string."""
        self._ensure_paired()
        stats = self._compute_rank_stats()
        if not stats:
            return "TraceAggregator: no events loaded."

        sep = "=" * 80
        lines = [sep, "  TraceAggregator — Per-rank summary", sep]
        for rank in sorted(stats):
            s = stats[rank]
            span = s.latest_ts - s.earliest_ts
            lines.append(
                f"  rank={rank:3d}  machine={s.machine_id:<18s}  "
                f"rows={s.num_raw_rows:6d}  events={s.num_complete_events:6d}  "
                f"steps={s.num_steps:4d}  span={span:.3f}s  "
                f"offset={s.clock_offset:+.6f}s"
            )
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def num_events(self) -> int:
        """Total number of complete events across all ranks."""
        self._ensure_paired()
        return sum(len(v) for v in self._events.values())

    def clear(self) -> None:
        """Remove all loaded data, offsets, and derived state."""
        self._raw.clear()
        self._events.clear()
        self._offsets.clear()
        self._colours.clear()
        self._dirty = False

    # ------------------------------------------------------------------
    # Internal: pairing
    # ------------------------------------------------------------------

    def _ensure_paired(self) -> None:
        """Pair START/END rows into CompleteEvents (lazy, cached)."""
        if not self._dirty and self._events:
            return
        self._events = {}
        for rank, rows in self._raw.items():
            offset = self._offsets.get(rank, 0.0)
            self._events[rank] = _pair_rows(rows, offset)
        self._dirty = False

    # ------------------------------------------------------------------
    # Internal: Chrome Trace helpers
    # ------------------------------------------------------------------

    def _colour(self, event_name: str) -> str:
        if event_name not in self._colours:
            idx = len(self._colours) % len(_CHROME_COLOURS)
            self._colours[event_name] = _CHROME_COLOURS[idx]
        return self._colours[event_name]

    def _build_metadata_events(self) -> List[Dict]:
        """Emit Chrome Trace 'M' metadata events to name each process/thread."""
        stats = self._compute_rank_stats()
        meta: List[Dict] = []
        for rank, s in stats.items():
            meta.append({
                "name": "process_name",
                "ph": "M",
                "pid": rank,
                "tid": 0,
                "args": {"name": f"Rank {rank} ({s.machine_id})"},
            })
            meta.append({
                "name": "process_sort_index",
                "ph": "M",
                "pid": rank,
                "tid": 0,
                "args": {"sort_index": rank},
            })
            meta.append({
                "name": "thread_name",
                "ph": "M",
                "pid": rank,
                "tid": 0,
                "args": {"name": "main"},
            })
        return meta

    def _compute_rank_stats(self) -> Dict[int, RankStat]:
        result: Dict[int, RankStat] = {}
        for rank, rows in self._raw.items():
            if not rows:
                continue
            tss = [r.timestamp for r in rows]
            steps = {r.step for r in rows}
            stages = sorted({r.event_name for r in rows})
            events = self._events.get(rank, [])
            result[rank] = RankStat(
                rank=rank,
                machine_id=rows[0].machine_id,
                num_raw_rows=len(rows),
                num_complete_events=len(events),
                num_steps=len(steps),
                earliest_ts=min(tss),
                latest_ts=max(tss),
                clock_offset=self._offsets.get(rank, 0.0),
                unique_stages=stages,
            )
        return result


# ---------------------------------------------------------------------------
# Pairing algorithm
# ---------------------------------------------------------------------------

def _pair_rows(rows: List[_RawRow], clock_offset: float) -> List[CompleteEvent]:
    """Match every START row with its corresponding END row.

    Matching priority:
    1. By ``node_id`` (exact, O(1) lookup) — used when MyTimer metadata is
       present (``node_id=N``).
    2. By name-based LIFO stack — fallback for logs without node_id.

    Within each ``(step,)`` boundary, unpaired STARTs are discarded with a
    warning.
    """
    # Group by step
    by_step: Dict[int, List[_RawRow]] = defaultdict(list)
    for row in rows:
        by_step[row.step].append(row)

    complete: List[CompleteEvent] = []
    for step, step_rows in sorted(by_step.items()):
        complete.extend(_pair_step_rows(step_rows, clock_offset, step))
    return complete


def _pair_step_rows(
    rows: List[_RawRow],
    clock_offset: float,
    step: int,
) -> List[CompleteEvent]:
    """Pair START/END rows for one training step."""
    # Separate starts and ends
    starts: List[_RawRow] = [r for r in rows if r.event_type == "START"]
    ends: List[_RawRow] = [r for r in rows if r.event_type == "END"]

    # --- Strategy 1: node_id-based exact matching ---
    has_node_id = any(s.node_id is not None for s in starts)
    if has_node_id:
        return _pair_by_node_id(starts, ends, clock_offset)

    # --- Strategy 2: name-based LIFO stack ---
    return _pair_by_stack(rows, clock_offset)


def _pair_by_node_id(
    starts: List[_RawRow],
    ends: List[_RawRow],
    clock_offset: float,
) -> List[CompleteEvent]:
    """Exact pairing: match each START to the END with the same node_id."""
    end_by_node: Dict[int, _RawRow] = {}
    for e in ends:
        if e.node_id is not None:
            end_by_node[e.node_id] = e

    complete: List[CompleteEvent] = []
    for s in starts:
        nid = s.node_id
        e = end_by_node.get(nid) if nid is not None else None
        if e is None:
            # Fallback: try matching by name if node_id is missing on END
            logger.debug(
                "No END found for START node_id=%s event=%r step=%d — skipping",
                nid, s.event_name, s.step,
            )
            continue
        start_ts = s.timestamp + clock_offset
        end_ts = e.timestamp + clock_offset
        if end_ts < start_ts:
            # Can happen if clock offset was applied incorrectly; swap gracefully
            start_ts, end_ts = end_ts, start_ts
        complete.append(CompleteEvent(
            rank=s.rank,
            machine_id=s.machine_id,
            step=s.step,
            event_name=s.event_name,
            start_ts=start_ts,
            end_ts=end_ts,
            duration_ms=e.duration_ms,
            node_id=s.node_id,
        ))
    return complete


def _pair_by_stack(rows: List[_RawRow], clock_offset: float) -> List[CompleteEvent]:
    """LIFO stack-based pairing when node_id is not available."""
    # stack: Dict[event_name → List[_RawRow]] acting as a LIFO per name
    stack: Dict[str, List[_RawRow]] = defaultdict(list)
    complete: List[CompleteEvent] = []

    for row in rows:  # already in file (time) order
        if row.event_type == "START":
            stack[row.event_name].append(row)
        elif row.event_type == "END":
            name_stack = stack.get(row.event_name)
            if not name_stack:
                logger.debug(
                    "Unmatched END for event=%r step=%d — skipping",
                    row.event_name, row.step,
                )
                continue
            s = name_stack.pop()
            start_ts = s.timestamp + clock_offset
            end_ts = row.timestamp + clock_offset
            complete.append(CompleteEvent(
                rank=row.rank,
                machine_id=row.machine_id,
                step=row.step,
                event_name=row.event_name,
                start_ts=start_ts,
                end_ts=end_ts,
                duration_ms=row.duration_ms,
                node_id=None,
            ))

    # Warn about unclosed starts
    for name, leftover in stack.items():
        if leftover:
            logger.debug(
                "%d unclosed START for event=%r step=%d",
                len(leftover), name, leftover[0].step,
            )
    return complete


# ---------------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------------

def _compute_stat(name: str, rank: int, durs: List[float]) -> StageStat:
    n = len(durs)
    sorted_d = sorted(durs)

    def _pct(p: float) -> float:
        idx = max(min(int(math.ceil(p / 100.0 * n)) - 1, n - 1), 0)
        return sorted_d[idx]

    return StageStat(
        event_name=name,
        rank=rank,
        count=n,
        mean_ms=statistics.mean(durs),
        std_ms=statistics.stdev(durs) if n > 1 else 0.0,
        min_ms=sorted_d[0],
        max_ms=sorted_d[-1],
        p50_ms=_pct(50),
        p95_ms=_pct(95),
        p99_ms=_pct(99),
    )


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _infer_rank_from_path(csv_path: str) -> int:
    """Extract the rank integer from a filename like ``profile_rank_3.csv``."""
    name = os.path.basename(csv_path)
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    logger.warning("Cannot infer rank from filename %r — defaulting to 0", name)
    return 0


def _parse_node_id(metadata: str) -> Optional[int]:
    """Parse ``node_id=<N>`` out of the metadata field."""
    m = re.search(r"node_id\s*=\s*(\d+)", metadata)
    return int(m.group(1)) if m else None


def _parse_row(row: List[str], rank: int) -> Optional[_RawRow]:
    """Parse one CSV row into a :class:`_RawRow`.  Returns ``None`` on errors."""
    # Expected columns (GlobalLogger format):
    # 0: timestamp_unix  1: readable_time  2: machine_id  3: step
    # 4: event_name      5: event_type     6: duration_ms  7: metadata (optional)
    if len(row) < 7:
        return None
    try:
        ts = float(row[0])
        machine_id = row[2].strip()
        step = int(row[3])
        event_name = row[4].strip()
        event_type = row[5].strip().upper()
        duration_ms = float(row[6])
        metadata = row[7].strip() if len(row) > 7 else ""
        node_id = _parse_node_id(metadata)

        if event_type not in ("START", "END"):
            return None

        return _RawRow(
            rank=rank,
            timestamp=ts,
            machine_id=machine_id,
            step=step,
            event_name=event_name,
            event_type=event_type,
            duration_ms=duration_ms,
            node_id=node_id,
        )
    except (ValueError, IndexError):
        logger.debug("Skipping malformed CSV row: %r", row)
        return None


def _build_args(ev: CompleteEvent) -> Dict:
    """Build the 'args' dict for a Chrome Trace 'X' event."""
    args: Dict = {
        "step": ev.step,
        "rank": ev.rank,
        "machine_id": ev.machine_id,
        "wall_dur_ms": round(ev.wall_dur_ms, 3),
    }
    if ev.duration_ms > 0:
        args["cuda_dur_ms"] = round(ev.duration_ms, 3)
    if ev.node_id is not None:
        args["node_id"] = ev.node_id
    return args


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m my_utils.profiling.trace_aggregator",
        description=(
            "Merge per-rank GlobalLogger CSV files into a Chrome Trace JSON.\n"
            "Open the output at https://ui.perfetto.dev"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--log-dir", "-d",
        required=True,
        metavar="DIR",
        help="Directory containing profile_rank_*.csv files",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        metavar="FILE",
        help="Output .json path (default: <log-dir>/merged_trace.json)",
    )
    p.add_argument(
        "--pattern",
        default="profile_rank_*.csv",
        metavar="GLOB",
        help="Glob pattern for CSV files (default: profile_rank_*.csv)",
    )
    p.add_argument(
        "--steps",
        default=None,
        metavar="START:END",
        help="Inclusive step range to export, e.g. '10:50'",
    )
    p.add_argument(
        "--name-filter",
        default=None,
        metavar="REGEX",
        help="Only export events whose name matches this regex",
    )
    p.add_argument(
        "--offsets",
        default=None,
        metavar="RANK:OFFSET,...",
        help="Clock offsets in seconds, e.g. '1:-0.003,2:0.001'",
    )
    p.add_argument(
        "--stats",
        default=None,
        metavar="FILE",
        help="Write stage statistics table to this file (default: print to stdout)",
    )
    p.add_argument(
        "--sort-stats-by",
        default="mean_ms",
        choices=["mean_ms", "std_ms", "max_ms", "count"],
        help="Sort column for statistics table (default: mean_ms)",
    )
    p.add_argument(
        "--top-n-stats",
        type=int,
        default=None,
        metavar="N",
        help="Show only top N stages in statistics table",
    )
    p.add_argument(
        "--no-step-markers",
        action="store_true",
        help="Suppress per-step instant markers in the trace",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def main() -> None:  # pragma: no cover
    """CLI entry point (``python -m my_utils.profiling.trace_aggregator``)."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    agg = TraceAggregator()

    # Load
    total_rows = agg.load_directory(args.log_dir, pattern=args.pattern)
    if total_rows == 0:
        logger.error("No data loaded — aborting.")
        return

    # Clock offsets
    if args.offsets:
        for token in args.offsets.split(","):
            token = token.strip()
            if ":" not in token:
                continue
            rank_str, off_str = token.split(":", 1)
            try:
                agg.set_clock_offset(int(rank_str), float(off_str))
            except ValueError:
                logger.warning("Could not parse offset token %r — skipping", token)

    # Step range
    step_range: Optional[Tuple[int, int]] = None
    if args.steps:
        parts = args.steps.split(":")
        if len(parts) == 2:
            try:
                step_range = (int(parts[0]), int(parts[1]))
            except ValueError:
                logger.warning("Invalid --steps %r — ignoring", args.steps)

    # Export trace
    output = args.output or os.path.join(args.log_dir, "merged_trace.json")
    n = agg.export_chrome_trace(
        output,
        step_range=step_range,
        name_filter=args.name_filter,
        include_step_markers=not args.no_step_markers,
    )
    print(f"✓ Wrote {n} events → {output}")
    print(f"  Open at: https://ui.perfetto.dev  (File → Open trace file)")
    print()

    # Rank summary
    print(agg.rank_summary())

    # Stage stats
    table = agg.stage_stats_table(
        sort_by=args.sort_stats_by,
        top_n=args.top_n_stats,
    )
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as fh:
            fh.write(table + "\n")
        print(f"✓ Stage stats → {args.stats}")
    else:
        print(table)


if __name__ == "__main__":
    main()
