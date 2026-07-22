# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..metrics.metrics_store import MetricsStore
from ..metrics.metrics_types import MetricEvent


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _to_us(value: float, unit: str) -> Optional[float]:
    unit_l = str(unit or "").strip().lower()
    if unit_l in ("us", "microsecond", "microseconds"):
        return float(value)
    if unit_l in ("ms", "millisecond", "milliseconds"):
        return float(value) * 1000.0
    if unit_l in ("ns", "nanosecond", "nanoseconds"):
        return float(value) / 1000.0
    if unit_l in ("s", "sec", "second", "seconds"):
        return float(value) * 1_000_000.0
    return None


def _event_prefix(event_name: str) -> str:
    if "." not in event_name:
        return event_name or "metric"
    return event_name.split(".", 1)[0]


def _first_str(tags: Mapping[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in tags and tags[key] not in (None, ""):
            return str(tags[key])
    return None


def _extract_rank(event: MetricEvent) -> Optional[str]:
    value = _first_str(event.tags, ("rank", "pid", "global_pid"))
    if value is None:
        return None
    return str(value)


def _event_name_for_trace(event: MetricEvent) -> str:
    tag_name = _first_str(
        event.tags, ("stage", "op", "kernel", "module", "func", "api", "name")
    )
    if tag_name:
        device = event.tags.get("device")
        if device:
            return f"{tag_name} [{device}]"
        return tag_name
    return event.name


def _event_end_timestamp_sec(event: MetricEvent) -> Optional[float]:
    attrs = event.attributes or {}
    for key in ("end_timestamp", "end_ts", "abs_end_time", "event_end_time"):
        if key in attrs:
            value = _safe_float(attrs.get(key))
            if value is not None:
                return value
    if "end_ns" in attrs:
        value = _safe_float(attrs.get("end_ns"))
        if value is not None:
            return value / 1_000_000_000.0
    value = _safe_float(event.timestamp)
    return value


def _event_start_timestamp_sec(event: MetricEvent) -> Optional[float]:
    attrs = event.attributes or {}
    for key in ("start_timestamp", "start_ts", "abs_start_time", "event_start_time"):
        if key in attrs:
            value = _safe_float(attrs.get(key))
            if value is not None:
                return value
    if "start_ns" in attrs:
        value = _safe_float(attrs.get("start_ns"))
        if value is not None:
            return value / 1_000_000_000.0
    return None


def _duration_us_from_event(event: MetricEvent) -> Optional[float]:
    value = _safe_float(event.value)
    if value is None:
        return None
    duration_us = _to_us(value, event.unit)
    return duration_us


def _build_duration_window(event: MetricEvent) -> Optional[Tuple[float, float]]:
    """
    Returns `(start_ts_us, duration_us)` for Chrome Trace `ph='X'`.
    """
    duration_us = _duration_us_from_event(event)
    if duration_us is None or duration_us <= 0:
        return None

    start_s = _event_start_timestamp_sec(event)
    end_s = _event_end_timestamp_sec(event)

    if start_s is None and end_s is None:
        return None
    if start_s is None and end_s is not None:
        start_s = end_s - duration_us / 1_000_000.0
    elif start_s is not None and end_s is None:
        end_s = start_s + duration_us / 1_000_000.0
    elif start_s is not None and end_s is not None and end_s > start_s:
        duration_us = (end_s - start_s) * 1_000_000.0
    elif start_s is not None and end_s is not None and end_s <= start_s:
        return None

    if start_s is None:
        return None
    return start_s * 1_000_000.0, float(duration_us)


def estimate_rank_time_offsets(
    events: Iterable[MetricEvent],
    *,
    reference_rank: str = "",
    anchor_metric_prefixes: Sequence[str] = ("latency",),
) -> Dict[str, float]:
    """
    Estimate per-rank clock offsets (in seconds), so adding the offset aligns ranks.

    Strategy:
    - use per-step earliest anchor timestamp per rank
    - offset(rank) = median(ref_step_anchor - rank_step_anchor) across common steps
    """
    prefixes = tuple(str(item) for item in anchor_metric_prefixes if item)
    if not prefixes:
        prefixes = ("latency",)

    rank_step_anchor: Dict[str, Dict[int, float]] = {}
    rank_first_anchor: Dict[str, float] = {}
    for event in events:
        if not event.name:
            continue
        prefix = _event_prefix(event.name)
        if prefix not in prefixes:
            continue
        rank = _extract_rank(event)
        if rank is None:
            continue
        anchor_ts = _event_start_timestamp_sec(event)
        if anchor_ts is None:
            anchor_ts = _event_end_timestamp_sec(event)
        if anchor_ts is None:
            continue

        step = None
        if "step" in event.tags:
            try:
                step = int(event.tags["step"])
            except Exception:
                step = None

        if step is not None:
            rank_step_anchor.setdefault(rank, {})
            current = rank_step_anchor[rank].get(step)
            if current is None or anchor_ts < current:
                rank_step_anchor[rank][step] = anchor_ts

        current_first = rank_first_anchor.get(rank)
        if current_first is None or anchor_ts < current_first:
            rank_first_anchor[rank] = anchor_ts

    all_ranks = sorted(set(rank_step_anchor.keys()) | set(rank_first_anchor.keys()))
    if not all_ranks:
        return {}

    ref_rank = str(reference_rank).strip() if reference_rank else all_ranks[0]
    if ref_rank not in all_ranks:
        ref_rank = all_ranks[0]

    offsets: Dict[str, float] = {rank: 0.0 for rank in all_ranks}
    ref_steps = rank_step_anchor.get(ref_rank, {})
    ref_first = rank_first_anchor.get(ref_rank)
    for rank in all_ranks:
        if rank == ref_rank:
            continue
        deltas: List[float] = []
        rank_steps = rank_step_anchor.get(rank, {})
        for step, ref_ts in ref_steps.items():
            if step in rank_steps:
                deltas.append(ref_ts - rank_steps[step])
        if deltas:
            offsets[rank] = float(statistics.median(deltas))
            continue
        if ref_first is not None and rank in rank_first_anchor:
            offsets[rank] = float(ref_first - rank_first_anchor[rank])
    return offsets


@dataclass
class ChromeTraceExportConfig:
    include_metric_prefixes: List[str] = field(default_factory=lambda: ["latency"])
    include_non_duration_metrics: bool = False
    auto_align_ranks: bool = True
    reference_rank: str = ""
    rank_clock_offsets_sec: Dict[str, float] = field(default_factory=dict)


def metric_events_to_chrome_trace(
    events: Iterable[MetricEvent],
    *,
    config: Optional[ChromeTraceExportConfig] = None,
) -> Dict[str, Any]:
    cfg = config or ChromeTraceExportConfig()
    event_list = list(events)

    include_prefixes = {
        str(item) for item in (cfg.include_metric_prefixes or ["latency"])
    }
    rank_offsets = {
        str(k): float(v) for k, v in (cfg.rank_clock_offsets_sec or {}).items()
    }
    if cfg.auto_align_ranks:
        estimated = estimate_rank_time_offsets(
            event_list,
            reference_rank=cfg.reference_rank,
            anchor_metric_prefixes=tuple(include_prefixes),
        )
        for rank, offset in estimated.items():
            rank_offsets[rank] = rank_offsets.get(rank, 0.0) + float(offset)

    process_id_map: Dict[str, int] = {}
    process_name_map: Dict[int, str] = {}
    thread_id_map: Dict[Tuple[int, str], int] = {}
    thread_name_map: Dict[Tuple[int, int], str] = {}
    pid_next_tid: Dict[int, int] = {}

    def resolve_pid(event: MetricEvent) -> Tuple[int, Optional[str]]:
        rank = _extract_rank(event)
        if rank is not None:
            try:
                pid = int(rank)
            except Exception:
                pid = process_id_map.setdefault(
                    f"rank:{rank}", 10_000 + len(process_id_map)
                )
            process_name_map[pid] = f"rank_{rank}"
            return pid, rank
        tag_pid = _first_str(event.tags, ("pid", "global_pid"))
        if tag_pid is not None:
            try:
                pid = int(tag_pid)
            except Exception:
                pid = process_id_map.setdefault(
                    f"pid:{tag_pid}", 20_000 + len(process_id_map)
                )
            process_name_map.setdefault(pid, f"pid_{tag_pid}")
            return pid, None
        provider_key = f"provider:{event.provider_id or 'unknown'}"
        pid = process_id_map.setdefault(provider_key, 30_000 + len(process_id_map))
        process_name_map.setdefault(pid, provider_key)
        return pid, None

    def resolve_tid(pid: int, event: MetricEvent) -> int:
        label = (
            _first_str(
                event.tags, ("stage", "op", "kernel", "module", "func", "api", "device")
            )
            or event.name
        )
        key = (pid, label)
        if key not in thread_id_map:
            thread_id = int(pid_next_tid.get(pid, 1))
            thread_id_map[key] = thread_id
            thread_name_map[(pid, thread_id)] = label
            pid_next_tid[pid] = thread_id + 1
        return thread_id_map[key]

    trace_events: List[Dict[str, Any]] = []
    for event in event_list:
        prefix = _event_prefix(event.name)
        if prefix not in include_prefixes:
            continue
        window = _build_duration_window(event)
        if window is None:
            if not cfg.include_non_duration_metrics:
                continue
            end_ts = _event_end_timestamp_sec(event)
            if end_ts is None:
                continue
            window = (end_ts * 1_000_000.0, 1.0)

        start_ts_us, dur_us = window
        pid, rank = resolve_pid(event)
        if rank is not None and rank in rank_offsets:
            start_ts_us += rank_offsets[rank] * 1_000_000.0
        tid = resolve_tid(pid, event)

        args: Dict[str, Any] = {
            "metric_name": event.name,
            "provider_id": event.provider_id,
            "value": event.value,
            "unit": event.unit,
            "tags": dict(event.tags),
        }
        if event.attributes:
            safe_attrs: Dict[str, Any] = {}
            for key, value in event.attributes.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    safe_attrs[str(key)] = value
                else:
                    safe_attrs[str(key)] = repr(value)
            args["attributes"] = safe_attrs

        trace_events.append(
            {
                "name": _event_name_for_trace(event),
                "cat": prefix,
                "ph": "X",
                "ts": float(start_ts_us),
                "dur": float(max(0.0, dur_us)),
                "pid": int(pid),
                "tid": int(tid),
                "args": args,
            }
        )

    # Metadata events
    metadata_events: List[Dict[str, Any]] = []
    for pid, process_name in sorted(process_name_map.items(), key=lambda item: item[0]):
        metadata_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": int(pid),
                "tid": 0,
                "args": {"name": process_name},
            }
        )
    for (pid, tid), thread_name in sorted(
        thread_name_map.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        metadata_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": int(pid),
                "tid": int(tid),
                "args": {"name": thread_name},
            }
        )

    trace_events.sort(
        key=lambda item: (item.get("ts", 0.0), item.get("pid", 0), item.get("tid", 0))
    )
    payload = {
        "traceEvents": metadata_events + trace_events,
        "displayTimeUnit": "ms",
        "metadata": {
            "event_count": len(trace_events),
            "process_count": len(process_name_map),
            "thread_count": len(thread_name_map),
            "auto_aligned": bool(cfg.auto_align_ranks),
            "rank_clock_offsets_sec": rank_offsets,
            "included_prefixes": sorted(include_prefixes),
        },
    }
    return payload


def write_chrome_trace(
    events: Iterable[MetricEvent],
    *,
    output_path: str,
    config: Optional[ChromeTraceExportConfig] = None,
) -> str:
    payload = metric_events_to_chrome_trace(events, config=config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def export_events_file_to_chrome_trace(
    events_path: str,
    *,
    output_path: str,
    config: Optional[ChromeTraceExportConfig] = None,
) -> str:
    events = MetricsStore.read_events_file(events_path)
    return write_chrome_trace(events, output_path=output_path, config=config)
