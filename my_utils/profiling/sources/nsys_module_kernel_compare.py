from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(v)


def _maybe_float(value: object) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _pick_first(row: Dict[str, object], keys: Sequence[str]) -> object:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _kind_from_row(row: Dict[str, object], kernel_name: str) -> str:
    kind = str(row.get("kind") or "").strip().lower()
    if kind in {"compute", "comm"}:
        return kind
    return "comm" if "nccl" in str(kernel_name or "").lower() else "compute"


def _normalize_like_pattern(text: str) -> Optional[Pattern[str]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    normalized = raw.replace("*", "%")
    escaped: List[str] = []
    for ch in normalized:
        if ch == "%":
            escaped.append(".*")
        elif ch == "_":
            escaped.append(".")
        else:
            escaped.append(re.escape(ch))
    return re.compile("^" + "".join(escaped) + "$", flags=re.IGNORECASE)


def _matches_nvtx(nvtx_text: str, pattern: Optional[Pattern[str]]) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(str(nvtx_text or "")))


def _merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    items: List[Tuple[int, int]] = []
    for start_ns, end_ns in intervals:
        s = _to_int(start_ns, -1)
        e = _to_int(end_ns, -1)
        if s < 0 or e <= s:
            continue
        items.append((s, e))
    if not items:
        return []
    items.sort(key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, int]] = []
    cur_s, cur_e = items[0]
    for s, e in items[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out


def _covered_ns(intervals: Iterable[Tuple[int, int]]) -> int:
    merged = _merge_intervals(intervals)
    return int(sum(max(0, e - s) for s, e in merged))


def _weighted_mean(pairs: Iterable[Tuple[Optional[float], float]]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for value, weight in pairs:
        if value is None:
            continue
        w = _to_float(weight, 0.0)
        if w <= 0:
            continue
        num += float(value) * w
        den += w
    if den <= 0:
        return None
    return float(num / den)


def _short_kernel_name(name: str, width: int = 64) -> str:
    text = str(name or "")
    if len(text) <= int(width):
        return text
    if int(width) <= 3:
        return text[: max(0, int(width))]
    return text[: int(width) - 3] + "..."


def _safe_ratio(numer: float, denom: float) -> Optional[float]:
    d = _to_float(denom, 0.0)
    if d <= 0:
        return None
    return float(_to_float(numer, 0.0) / d)


_KERNEL_RESOURCE_KEYS: Tuple[str, ...] = (
    "threads_per_block",
    "total_blocks",
    "grid_x",
    "grid_y",
    "grid_z",
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "total_shared_bytes",
    "occupancy_pct",
)

_GEOMETRY_KEYS: Tuple[str, ...] = (
    "total_blocks",
    "grid_x",
    "grid_y",
    "grid_z",
)


def _canon_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(float(value), 6)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, bool):
        return bool(value)
    try:
        i = int(value)
        if str(i) == str(value):
            return int(i)
    except Exception:
        pass
    try:
        f = float(value)
        if math.isfinite(f):
            return round(float(f), 6)
    except Exception:
        pass
    return str(value)


def _sorted_values(values: Iterable[object]) -> List[object]:
    normed = {_canon_value(v) for v in values if _canon_value(v) is not None}
    return sorted(normed, key=lambda x: (str(type(x)), str(x)))


@dataclass
class _KernelEvent:
    nvtx_text: str
    device_id: int
    stream_id: int
    kind: str
    kernel_name: str
    start_ns: int
    end_ns: int
    duration_ms: float
    threads_per_block: Optional[int]
    total_blocks: Optional[int]
    grid_x: Optional[int]
    grid_y: Optional[int]
    grid_z: Optional[int]
    registers_per_thread: Optional[int]
    static_shared_bytes: Optional[int]
    dynamic_shared_bytes: Optional[int]
    total_shared_bytes: Optional[int]
    occupancy_pct: Optional[float]


def _normalize_event(row: Dict[str, object]) -> Optional[_KernelEvent]:
    kernel_name = str(_pick_first(row, ["kernel_name", "name"]) or "").strip()
    if not kernel_name:
        return None
    start_ns = _to_int(_pick_first(row, ["kernel_start_ns", "start_ns", "start"]), -1)
    end_ns = _to_int(_pick_first(row, ["kernel_end_ns", "end_ns", "end"]), -1)
    if start_ns < 0 or end_ns <= start_ns:
        return None
    duration_ms = _to_float(row.get("duration_ms"), (end_ns - start_ns) / 1e6)
    tpb = _pick_first(row, ["threads_per_block", "threadsPerBlock", "blockX"])
    total_blocks = _pick_first(row, ["total_blocks", "blocks", "gridX"])
    grid_x = _pick_first(row, ["gridX", "grid_x"])
    grid_y = _pick_first(row, ["gridY", "grid_y"])
    grid_z = _pick_first(row, ["gridZ", "grid_z"])
    regs = _pick_first(row, ["registersPerThread", "registers_per_thread"])
    static_shared = _pick_first(row, ["static_shared_bytes", "staticSharedMemory"])
    dynamic_shared = _pick_first(row, ["dynamic_shared_bytes", "dynamicSharedMemory"])
    total_shared = _pick_first(row, ["total_shared_bytes"])
    if total_shared is None:
        total_shared = _to_int(static_shared, 0) + _to_int(dynamic_shared, 0)
    occ = _maybe_float(_pick_first(row, ["occupancy_pct_h100_estimate", "occupancy_pct_estimate"]))
    return _KernelEvent(
        nvtx_text=str(row.get("nvtx_text") or row.get("nvtx_name") or ""),
        device_id=_to_int(_pick_first(row, ["device_id", "deviceId"]), -1),
        stream_id=_to_int(_pick_first(row, ["stream_id", "streamId"]), -1),
        kind=_kind_from_row(row, kernel_name),
        kernel_name=kernel_name,
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ms=duration_ms,
        threads_per_block=_to_int(tpb, -1) if tpb is not None else None,
        total_blocks=_to_int(total_blocks, -1) if total_blocks is not None else None,
        grid_x=_to_int(grid_x, -1) if grid_x is not None else None,
        grid_y=_to_int(grid_y, -1) if grid_y is not None else None,
        grid_z=_to_int(grid_z, -1) if grid_z is not None else None,
        registers_per_thread=_to_int(regs, -1) if regs is not None else None,
        static_shared_bytes=_to_int(static_shared, -1) if static_shared is not None else None,
        dynamic_shared_bytes=_to_int(dynamic_shared, -1) if dynamic_shared is not None else None,
        total_shared_bytes=_to_int(total_shared, -1) if total_shared is not None else None,
        occupancy_pct=occ,
    )


def _load_rows(path: str) -> List[Dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list in {path}")
    out: List[Dict[str, object]] = []
    for row in payload:
        if isinstance(row, dict):
            out.append(dict(row))
    return out


def _build_profile(
    rows: Sequence[Dict[str, object]],
    *,
    label: str,
    source_path: str,
    nvtx_text: str,
    device_id: int,
    stream_ids: Sequence[int],
    top_k: int,
    timeline_limit_per_stream: int,
) -> Dict[str, object]:
    nvtx_pat = _normalize_like_pattern(nvtx_text)
    stream_filter = {int(v) for v in stream_ids}
    events: List[_KernelEvent] = []
    for row in rows:
        ev = _normalize_event(row)
        if ev is None:
            continue
        if not _matches_nvtx(ev.nvtx_text, nvtx_pat):
            continue
        if int(device_id) >= 0 and int(ev.device_id) != int(device_id):
            continue
        if stream_filter and int(ev.stream_id) not in stream_filter:
            continue
        events.append(ev)

    events.sort(key=lambda x: (x.start_ns, x.stream_id, x.end_ns, x.kernel_name))
    if not events:
        return {
            "label": str(label),
            "source_path": str(source_path),
            "filters": {
                "nvtx_text": str(nvtx_text or ""),
                "device_id": int(device_id),
                "stream_ids": sorted(stream_filter),
            },
            "summary": {
                "event_count": 0,
                "stream_count": 0,
                "kernel_kind_counts": {},
                "kernel_set_size": 0,
                "window_start_ns": None,
                "window_end_ns": None,
                "window_span_ms": 0.0,
                "total_kernel_ms": 0.0,
                "busy_union_ms": 0.0,
                "weighted_occupancy_pct": None,
            },
            "top_kernels": [],
            "kernel_stats_all": [],
            "kernel_set": [],
            "streams": [],
        }

    window_start_ns = min(ev.start_ns for ev in events)
    window_end_ns = max(ev.end_ns for ev in events)
    window_span_ms = float(max(0, window_end_ns - window_start_ns) / 1e6)
    total_kernel_ms = float(sum(ev.duration_ms for ev in events))
    busy_union_ms = float(_covered_ns((ev.start_ns, ev.end_ns) for ev in events) / 1e6)
    occ_pairs = [(ev.occupancy_pct, ev.duration_ms) for ev in events]
    weighted_occ = _weighted_mean(occ_pairs)

    kernel_kind_counts = Counter(ev.kind for ev in events)
    kernel_set = sorted({ev.kernel_name for ev in events})

    kernel_stats: Dict[str, Dict[str, object]] = {}
    stream_stats: Dict[int, Dict[str, object]] = {}
    for ev in events:
        k = kernel_stats.setdefault(
            ev.kernel_name,
            {
                "kernel_name": ev.kernel_name,
                "kind_counts": Counter(),
                "invocations": 0,
                "total_ms": 0.0,
                "streams": set(),
                "weighted_occ_num": 0.0,
                "weighted_occ_den": 0.0,
                "resource_sets": {key: set() for key in _KERNEL_RESOURCE_KEYS},
            },
        )
        k["invocations"] = int(k["invocations"]) + 1
        k["total_ms"] = float(k["total_ms"]) + float(ev.duration_ms)
        k["kind_counts"][ev.kind] += 1
        k["streams"].add(ev.stream_id)
        k["resource_sets"]["threads_per_block"].add(ev.threads_per_block)
        k["resource_sets"]["total_blocks"].add(ev.total_blocks)
        k["resource_sets"]["grid_x"].add(ev.grid_x)
        k["resource_sets"]["grid_y"].add(ev.grid_y)
        k["resource_sets"]["grid_z"].add(ev.grid_z)
        k["resource_sets"]["registers_per_thread"].add(ev.registers_per_thread)
        k["resource_sets"]["static_shared_bytes"].add(ev.static_shared_bytes)
        k["resource_sets"]["dynamic_shared_bytes"].add(ev.dynamic_shared_bytes)
        k["resource_sets"]["total_shared_bytes"].add(ev.total_shared_bytes)
        k["resource_sets"]["occupancy_pct"].add(ev.occupancy_pct)
        if ev.occupancy_pct is not None and ev.duration_ms > 0:
            k["weighted_occ_num"] = float(k["weighted_occ_num"]) + float(ev.occupancy_pct) * float(ev.duration_ms)
            k["weighted_occ_den"] = float(k["weighted_occ_den"]) + float(ev.duration_ms)

        s = stream_stats.setdefault(
            int(ev.stream_id),
            {
                "stream_id": int(ev.stream_id),
                "events": [],
                "kernel_set": set(),
                "kernel_kind_counts": Counter(),
                "total_kernel_ms": 0.0,
                "weighted_occ_num": 0.0,
                "weighted_occ_den": 0.0,
            },
        )
        s["events"].append(ev)
        s["kernel_set"].add(ev.kernel_name)
        s["kernel_kind_counts"][ev.kind] += 1
        s["total_kernel_ms"] = float(s["total_kernel_ms"]) + float(ev.duration_ms)
        if ev.occupancy_pct is not None and ev.duration_ms > 0:
            s["weighted_occ_num"] = float(s["weighted_occ_num"]) + float(ev.occupancy_pct) * float(ev.duration_ms)
            s["weighted_occ_den"] = float(s["weighted_occ_den"]) + float(ev.duration_ms)

    kernel_rows_all: List[Dict[str, object]] = []
    for name, stats in kernel_stats.items():
        inv = int(stats.get("invocations", 0))
        total_ms = float(stats.get("total_ms", 0.0))
        occ_den = float(stats.get("weighted_occ_den", 0.0))
        occ_num = float(stats.get("weighted_occ_num", 0.0))
        resource_sets = dict(stats.get("resource_sets") or {})
        resource_signatures = {key: _sorted_values(resource_sets.get(key, set())) for key in _KERNEL_RESOURCE_KEYS}
        kernel_rows_all.append(
            {
                "kernel_name": name,
                "kind_counts": dict(stats.get("kind_counts", {})),
                "invocations": inv,
                "total_ms": round(total_ms, 6),
                "avg_ms": round(total_ms / max(1, inv), 6),
                "stream_count": len(stats.get("streams", set())),
                "weighted_occupancy_pct": round(occ_num / occ_den, 6) if occ_den > 0 else None,
                "resource_signatures": resource_signatures,
            }
        )
    kernel_rows_all.sort(key=lambda x: (-_to_float(x.get("total_ms"), 0.0), str(x.get("kernel_name") or "")))

    stream_rows: List[Dict[str, object]] = []
    for stream_id, stats in stream_stats.items():
        s_events: List[_KernelEvent] = sorted(
            list(stats.get("events") or []),
            key=lambda ev: (ev.start_ns, ev.end_ns, ev.kernel_name),
        )
        first_start_ns = min(ev.start_ns for ev in s_events)
        last_end_ns = max(ev.end_ns for ev in s_events)
        span_ms = float(max(0, last_end_ns - first_start_ns) / 1e6)
        busy_ms = float(_covered_ns((ev.start_ns, ev.end_ns) for ev in s_events) / 1e6)
        total_ms = float(stats.get("total_kernel_ms", 0.0))
        occ_den = float(stats.get("weighted_occ_den", 0.0))
        occ_num = float(stats.get("weighted_occ_num", 0.0))
        timeline_rows: List[Dict[str, object]] = []
        for idx, ev in enumerate(s_events):
            if int(timeline_limit_per_stream) > 0 and idx >= int(timeline_limit_per_stream):
                break
            timeline_rows.append(
                {
                    "timeline_index": int(idx),
                    "kernel_name": ev.kernel_name,
                    "kernel_name_short": _short_kernel_name(ev.kernel_name),
                    "kind": ev.kind,
                    "start_offset_ms": round((ev.start_ns - window_start_ns) / 1e6, 6),
                    "end_offset_ms": round((ev.end_ns - window_start_ns) / 1e6, 6),
                    "duration_ms": round(ev.duration_ms, 6),
                    "threads_per_block": ev.threads_per_block,
                    "total_blocks": ev.total_blocks,
                    "grid_x": ev.grid_x,
                    "grid_y": ev.grid_y,
                    "grid_z": ev.grid_z,
                    "registers_per_thread": ev.registers_per_thread,
                    "static_shared_bytes": ev.static_shared_bytes,
                    "dynamic_shared_bytes": ev.dynamic_shared_bytes,
                    "total_shared_bytes": ev.total_shared_bytes,
                    "occupancy_pct": round(ev.occupancy_pct, 6) if ev.occupancy_pct is not None else None,
                }
            )
        stream_rows.append(
            {
                "stream_id": int(stream_id),
                "event_count": len(s_events),
                "kernel_set_size": len(stats.get("kernel_set", set())),
                "kernel_kind_counts": dict(stats.get("kernel_kind_counts", {})),
                "first_start_offset_ms": round((first_start_ns - window_start_ns) / 1e6, 6),
                "last_end_offset_ms": round((last_end_ns - window_start_ns) / 1e6, 6),
                "span_ms": round(span_ms, 6),
                "busy_union_ms": round(busy_ms, 6),
                "total_kernel_ms": round(total_ms, 6),
                "busy_pct_of_stream_span": round((busy_ms / span_ms) * 100.0, 6) if span_ms > 0 else None,
                "weighted_occupancy_pct": round(occ_num / occ_den, 6) if occ_den > 0 else None,
                "kernel_set": sorted(stats.get("kernel_set", set())),
                "kernel_sequence": [ev.kernel_name for ev in s_events],
                "timeline_sample": timeline_rows,
            }
        )
    stream_rows.sort(key=lambda x: int(x.get("stream_id", -1)))

    return {
        "label": str(label),
        "source_path": str(source_path),
        "filters": {
            "nvtx_text": str(nvtx_text or ""),
            "device_id": int(device_id),
            "stream_ids": sorted(stream_filter),
        },
        "summary": {
            "event_count": len(events),
            "stream_count": len(stream_rows),
            "kernel_kind_counts": dict(kernel_kind_counts),
            "kernel_set_size": len(kernel_set),
            "window_start_ns": int(window_start_ns),
            "window_end_ns": int(window_end_ns),
            "window_span_ms": round(window_span_ms, 6),
            "total_kernel_ms": round(total_kernel_ms, 6),
            "busy_union_ms": round(busy_union_ms, 6),
            "weighted_occupancy_pct": round(weighted_occ, 6) if weighted_occ is not None else None,
        },
        "top_kernels": kernel_rows_all[: max(1, int(top_k))],
        "kernel_stats_all": kernel_rows_all,
        "kernel_set": kernel_set,
        "streams": stream_rows,
    }


def _diff_kernel_totals(
    base_profile: Dict[str, object],
    target_profile: Dict[str, object],
    *,
    top_k: int,
) -> List[Dict[str, object]]:
    def _index_all(profile: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for row in profile.get("kernel_stats_all", []):
            name = str((row or {}).get("kernel_name") or "")
            if not name:
                continue
            out[name] = dict(row or {})
        return out

    base_index = _index_all(base_profile)
    target_index = _index_all(target_profile)
    names = sorted(set(base_index.keys()) | set(target_index.keys()))
    rows: List[Dict[str, object]] = []
    for name in names:
        b = base_index.get(name, {})
        t = target_index.get(name, {})
        b_ms = _to_float(b.get("total_ms"), 0.0)
        t_ms = _to_float(t.get("total_ms"), 0.0)
        ratio = _safe_ratio(t_ms, b_ms)
        rows.append(
            {
                "kernel_name": name,
                "base_total_ms": round(b_ms, 6),
                "target_total_ms": round(t_ms, 6),
                "delta_ms": round(t_ms - b_ms, 6),
                "ratio_target_over_base": round(ratio, 6) if ratio is not None else None,
                "base_invocations": _to_int(b.get("invocations"), 0),
                "target_invocations": _to_int(t.get("invocations"), 0),
                "base_occ_pct": b.get("weighted_occupancy_pct"),
                "target_occ_pct": t.get("weighted_occupancy_pct"),
            }
        )
    rows.sort(key=lambda x: (-abs(_to_float(x.get("delta_ms"), 0.0)), str(x.get("kernel_name") or "")))
    return rows[: max(1, int(top_k))]


def _first_divergence(a: Sequence[str], b: Sequence[str]) -> Dict[str, object]:
    i = 0
    n = min(len(a), len(b))
    while i < n and str(a[i]) == str(b[i]):
        i += 1
    return {
        "index": int(i),
        "base_kernel": str(a[i]) if i < len(a) else None,
        "target_kernel": str(b[i]) if i < len(b) else None,
    }


def _diff_streams(
    base_profile: Dict[str, object],
    target_profile: Dict[str, object],
) -> List[Dict[str, object]]:
    base_index = {int((row or {}).get("stream_id", -1)): dict(row or {}) for row in base_profile.get("streams", [])}
    target_index = {int((row or {}).get("stream_id", -1)): dict(row or {}) for row in target_profile.get("streams", [])}
    stream_ids = sorted(set(base_index.keys()) | set(target_index.keys()))
    rows: List[Dict[str, object]] = []
    for stream_id in stream_ids:
        b = base_index.get(stream_id, {})
        t = target_index.get(stream_id, {})
        b_seq = [str(x) for x in (b.get("kernel_sequence") or [])]
        t_seq = [str(x) for x in (t.get("kernel_sequence") or [])]
        seq_ratio = SequenceMatcher(None, b_seq, t_seq, autojunk=False).ratio() if (b_seq or t_seq) else 1.0
        b_set = set(b.get("kernel_set") or [])
        t_set = set(t.get("kernel_set") or [])
        rows.append(
            {
                "stream_id": int(stream_id),
                "base_event_count": _to_int(b.get("event_count"), 0),
                "target_event_count": _to_int(t.get("event_count"), 0),
                "base_total_kernel_ms": _to_float(b.get("total_kernel_ms"), 0.0),
                "target_total_kernel_ms": _to_float(t.get("total_kernel_ms"), 0.0),
                "delta_total_kernel_ms": round(
                    _to_float(t.get("total_kernel_ms"), 0.0) - _to_float(b.get("total_kernel_ms"), 0.0), 6
                ),
                "base_occ_pct": b.get("weighted_occupancy_pct"),
                "target_occ_pct": t.get("weighted_occupancy_pct"),
                "delta_occ_pct": round(
                    _to_float(t.get("weighted_occupancy_pct"), 0.0) - _to_float(b.get("weighted_occupancy_pct"), 0.0),
                    6,
                )
                if (b.get("weighted_occupancy_pct") is not None or t.get("weighted_occupancy_pct") is not None)
                else None,
                "kernel_set_added": sorted(t_set - b_set),
                "kernel_set_removed": sorted(b_set - t_set),
                "kernel_set_common_count": len(b_set & t_set),
                "sequence_similarity": round(float(seq_ratio), 6),
                "first_divergence": _first_divergence(b_seq, t_seq),
                "base_timeline_sample": list(b.get("timeline_sample") or []),
                "target_timeline_sample": list(t.get("timeline_sample") or []),
            }
        )
    rows.sort(key=lambda x: (-abs(_to_float(x.get("delta_total_kernel_ms"), 0.0)), int(x.get("stream_id", -1))))
    return rows


def _diff_kernel_resources(
    base_profile: Dict[str, object],
    target_profile: Dict[str, object],
) -> Dict[str, object]:
    def _index_all(profile: Dict[str, object]) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for row in profile.get("kernel_stats_all", []) or []:
            name = str((row or {}).get("kernel_name") or "")
            if not name:
                continue
            out[name] = dict(row or {})
        return out

    base_index = _index_all(base_profile)
    target_index = _index_all(target_profile)
    common = sorted(set(base_index.keys()) & set(target_index.keys()))
    changed_rows: List[Dict[str, object]] = []
    unchanged_names: List[str] = []
    geometry_only = 0
    resource_changed = 0

    for name in common:
        b = base_index.get(name, {})
        t = target_index.get(name, {})
        b_sig = dict(b.get("resource_signatures") or {})
        t_sig = dict(t.get("resource_signatures") or {})
        diff_map: Dict[str, Dict[str, object]] = {}
        changed_keys: List[str] = []
        for key in _KERNEL_RESOURCE_KEYS:
            bv = list(b_sig.get(key) or [])
            tv = list(t_sig.get(key) or [])
            if bv != tv:
                changed_keys.append(key)
                diff_map[key] = {"base": bv, "target": tv}
        if not changed_keys:
            unchanged_names.append(name)
            continue
        only_geometry = all(key in _GEOMETRY_KEYS for key in changed_keys)
        if only_geometry:
            geometry_only += 1
        else:
            resource_changed += 1
        changed_rows.append(
            {
                "kernel_name": name,
                "change_type": "geometry_only" if only_geometry else "resource_or_impl_change",
                "changed_keys": changed_keys,
                "resource_diffs": diff_map,
                "base_invocations": _to_int(b.get("invocations"), 0),
                "target_invocations": _to_int(t.get("invocations"), 0),
                "base_total_ms": _to_float(b.get("total_ms"), 0.0),
                "target_total_ms": _to_float(t.get("total_ms"), 0.0),
                "base_stream_count": _to_int(b.get("stream_count"), 0),
                "target_stream_count": _to_int(t.get("stream_count"), 0),
            }
        )

    changed_rows.sort(
        key=lambda x: (
            0 if str(x.get("change_type")) == "resource_or_impl_change" else 1,
            -abs(_to_float(x.get("target_total_ms"), 0.0) - _to_float(x.get("base_total_ms"), 0.0)),
            str(x.get("kernel_name") or ""),
        )
    )
    return {
        "common_kernel_count": len(common),
        "changed_kernel_count": len(changed_rows),
        "unchanged_kernel_count": len(unchanged_names),
        "geometry_only_changed_count": int(geometry_only),
        "resource_or_impl_changed_count": int(resource_changed),
        "changed_kernels": changed_rows,
        "unchanged_kernels": unchanged_names,
    }


def compare_module_kernel_json(
    *,
    base_json: str,
    target_json: str,
    base_label: str = "base",
    target_label: str = "target",
    nvtx_text: str = "",
    device_id: int = -1,
    stream_ids: Optional[Sequence[int]] = None,
    top_k: int = 20,
    timeline_limit_per_stream: int = 40,
) -> Dict[str, object]:
    base_rows = _load_rows(base_json)
    target_rows = _load_rows(target_json)
    streams = list(stream_ids or [])
    base_profile = _build_profile(
        base_rows,
        label=base_label,
        source_path=base_json,
        nvtx_text=nvtx_text,
        device_id=device_id,
        stream_ids=streams,
        top_k=top_k,
        timeline_limit_per_stream=timeline_limit_per_stream,
    )
    target_profile = _build_profile(
        target_rows,
        label=target_label,
        source_path=target_json,
        nvtx_text=nvtx_text,
        device_id=device_id,
        stream_ids=streams,
        top_k=top_k,
        timeline_limit_per_stream=timeline_limit_per_stream,
    )

    base_summary = dict(base_profile.get("summary") or {})
    target_summary = dict(target_profile.get("summary") or {})
    base_kernel_names = {str(v) for v in (base_profile.get("kernel_set") or [])}
    target_kernel_names = {str(v) for v in (target_profile.get("kernel_set") or [])}
    base_kernel_names.discard("")
    target_kernel_names.discard("")
    kernel_added = sorted(target_kernel_names - base_kernel_names)
    kernel_removed = sorted(base_kernel_names - target_kernel_names)
    kernel_resource_diff = _diff_kernel_resources(base_profile, target_profile)

    compare_payload = {
        "module_delta": {
            "event_count_delta": _to_int(target_summary.get("event_count"), 0) - _to_int(base_summary.get("event_count"), 0),
            "stream_count_delta": _to_int(target_summary.get("stream_count"), 0) - _to_int(base_summary.get("stream_count"), 0),
            "total_kernel_ms_delta": round(
                _to_float(target_summary.get("total_kernel_ms"), 0.0) - _to_float(base_summary.get("total_kernel_ms"), 0.0),
                6,
            ),
            "busy_union_ms_delta": round(
                _to_float(target_summary.get("busy_union_ms"), 0.0) - _to_float(base_summary.get("busy_union_ms"), 0.0),
                6,
            ),
            "window_span_ms_delta": round(
                _to_float(target_summary.get("window_span_ms"), 0.0) - _to_float(base_summary.get("window_span_ms"), 0.0),
                6,
            ),
            "weighted_occupancy_pct_delta": round(
                _to_float(target_summary.get("weighted_occupancy_pct"), 0.0)
                - _to_float(base_summary.get("weighted_occupancy_pct"), 0.0),
                6,
            )
            if (base_summary.get("weighted_occupancy_pct") is not None or target_summary.get("weighted_occupancy_pct") is not None)
            else None,
        },
        "kernel_set_diff": {
            "added_count": len(kernel_added),
            "removed_count": len(kernel_removed),
            "common_count": len(base_kernel_names & target_kernel_names),
            "added": kernel_added,
            "removed": kernel_removed,
        },
        "kernel_resource_diff": kernel_resource_diff,
        "top_kernel_duration_deltas": _diff_kernel_totals(base_profile, target_profile, top_k=top_k),
        "stream_deltas": _diff_streams(base_profile, target_profile),
    }

    return {
        "base": base_profile,
        "target": target_profile,
        "compare": compare_payload,
    }


def module_kernel_compare_to_markdown(payload: Dict[str, object]) -> str:
    base = dict(payload.get("base") or {})
    target = dict(payload.get("target") or {})
    compare = dict(payload.get("compare") or {})
    module_delta = dict(compare.get("module_delta") or {})
    kernel_set = dict(compare.get("kernel_set_diff") or {})
    kernel_resource_diff = dict(compare.get("kernel_resource_diff") or {})

    lines: List[str] = []
    lines.append("# NSYS Module Kernel Compare")
    lines.append("")
    lines.append(f"- base: `{base.get('label', 'base')}` ({base.get('source_path', '')})")
    lines.append(f"- target: `{target.get('label', 'target')}` ({target.get('source_path', '')})")
    lines.append("")
    lines.append("## Module Delta")
    lines.append("")
    lines.append(f"- event_count delta: `{module_delta.get('event_count_delta', 0)}`")
    lines.append(f"- stream_count delta: `{module_delta.get('stream_count_delta', 0)}`")
    lines.append(f"- total_kernel_ms delta: `{module_delta.get('total_kernel_ms_delta', 0.0)}`")
    lines.append(f"- busy_union_ms delta: `{module_delta.get('busy_union_ms_delta', 0.0)}`")
    lines.append(f"- window_span_ms delta: `{module_delta.get('window_span_ms_delta', 0.0)}`")
    lines.append(f"- weighted_occupancy_pct delta: `{module_delta.get('weighted_occupancy_pct_delta')}`")
    lines.append("")
    lines.append("## Kernel Set Diff")
    lines.append("")
    lines.append(f"- added_count: `{kernel_set.get('added_count', 0)}`")
    lines.append(f"- removed_count: `{kernel_set.get('removed_count', 0)}`")
    lines.append(f"- common_count: `{kernel_set.get('common_count', 0)}`")
    lines.append("")
    lines.append("## Same-Kernel Resource Diff")
    lines.append("")
    lines.append(f"- common_kernel_count: `{kernel_resource_diff.get('common_kernel_count', 0)}`")
    lines.append(f"- changed_kernel_count: `{kernel_resource_diff.get('changed_kernel_count', 0)}`")
    lines.append(f"- unchanged_kernel_count: `{kernel_resource_diff.get('unchanged_kernel_count', 0)}`")
    lines.append(f"- geometry_only_changed_count: `{kernel_resource_diff.get('geometry_only_changed_count', 0)}`")
    lines.append(
        f"- resource_or_impl_changed_count: `{kernel_resource_diff.get('resource_or_impl_changed_count', 0)}`"
    )
    lines.append("")
    changed_kernels = list(kernel_resource_diff.get("changed_kernels") or [])
    if changed_kernels:
        lines.append("| kernel | change_type | changed_keys | base_inv | target_inv |")
        lines.append("|---|---|---|---:|---:|")
        for row in changed_kernels:
            lines.append(
                "| {name} | {typ} | {keys} | {b} | {t} |".format(
                    name=_short_kernel_name(str((row or {}).get("kernel_name") or ""), width=72),
                    typ=row.get("change_type", ""),
                    keys=",".join(str(x) for x in ((row or {}).get("changed_keys") or [])),
                    b=_to_int((row or {}).get("base_invocations"), 0),
                    t=_to_int((row or {}).get("target_invocations"), 0),
                )
            )
        lines.append("")
    lines.append("## Top Kernel Duration Deltas")
    lines.append("")
    lines.append("| kernel | base_ms | target_ms | delta_ms | ratio |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in compare.get("top_kernel_duration_deltas", []) or []:
        lines.append(
            "| {name} | {b} | {t} | {d} | {r} |".format(
                name=_short_kernel_name(str((row or {}).get("kernel_name") or ""), width=72),
                b=row.get("base_total_ms", 0.0),
                t=row.get("target_total_ms", 0.0),
                d=row.get("delta_ms", 0.0),
                r=row.get("ratio_target_over_base"),
            )
        )
    lines.append("")
    lines.append("## Stream Deltas")
    lines.append("")
    for stream in compare.get("stream_deltas", []) or []:
        stream_id = int((stream or {}).get("stream_id", -1))
        lines.append(f"### Stream {stream_id}")
        lines.append(f"- event_count: `{stream.get('base_event_count', 0)} -> {stream.get('target_event_count', 0)}`")
        lines.append(
            "- total_kernel_ms: `{}` -> `{}` (delta `{}`)".format(
                stream.get("base_total_kernel_ms", 0.0),
                stream.get("target_total_kernel_ms", 0.0),
                stream.get("delta_total_kernel_ms", 0.0),
            )
        )
        lines.append(
            "- weighted_occ_pct: `{}` -> `{}` (delta `{}`)".format(
                stream.get("base_occ_pct"),
                stream.get("target_occ_pct"),
                stream.get("delta_occ_pct"),
            )
        )
        lines.append(f"- sequence_similarity: `{stream.get('sequence_similarity', 0.0)}`")
        div = dict(stream.get("first_divergence") or {})
        lines.append(
            "- first_divergence: idx `{}` | base `{}` | target `{}`".format(
                div.get("index"),
                _short_kernel_name(str(div.get("base_kernel") or ""), width=56),
                _short_kernel_name(str(div.get("target_kernel") or ""), width=56),
            )
        )
        lines.append(f"- kernel_set_added: `{len(stream.get('kernel_set_added', []) or [])}`")
        lines.append(f"- kernel_set_removed: `{len(stream.get('kernel_set_removed', []) or [])}`")
        lines.append("")
    return "\n".join(lines)
