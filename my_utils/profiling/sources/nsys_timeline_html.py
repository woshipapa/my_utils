from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .nsys_flat_export import collect_kernel_rows
from .nsys_schema_adapter import NsightSchema
from .nsys_sql_skills import calculate_h100_occupancy
from .nsys_sqlite_provider import NsysSqliteMetricsProvider

_RANK_RE = re.compile(r"\brank(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE)
_GPU_IDX_RE = re.compile(r"(?:gpu|device)\s*[_:#=\- ]?\s*(\d+)", re.IGNORECASE)
_DEFAULT_FOCUS_METRIC_TOKENS: Tuple[str, ...] = (
    "compute warps in flight",
    "warps in flight",
    "dram read bandwidth",
    "dram write bandwidth",
    "dram read",
    "dram write",
    "dram__bytes_read",
    "dram__bytes_write",
    "unallocated warps in active sms",
    "unallocated warps",
    "sm issue",
    "issue active",
    "smsp__issue",
    "sm active",
    "sm__active",
    "tensor active",
    "tensor__active",
)


def _ident(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("empty SQL identifier")
    for ch in text:
        if not (ch.isalnum() or ch == "_"):
            raise ValueError(f"unsafe SQL identifier: {name}")
    return text


def _color_for_name(name: str) -> str:
    digest = int(hashlib.md5((name or "").encode(), usedforsecurity=False).hexdigest(), 16)
    r = 70 + (digest >> 16 & 0xFF) * 130 // 255
    g = 70 + (digest >> 8 & 0xFF) * 130 // 255
    b = 70 + (digest & 0xFF) * 130 // 255
    return f"rgb({r},{g},{b})"


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_occupancy_pct_estimate(
    occupancy_pct_estimate: object,
    *,
    threads_per_block: object,
    registers_per_thread: object,
    total_shared_bytes: object,
) -> Optional[float]:
    try:
        if occupancy_pct_estimate is not None and str(occupancy_pct_estimate).strip() != "":
            occ = float(occupancy_pct_estimate)
            if math.isfinite(occ):
                return float(occ)
    except Exception:
        pass
    occ_h100 = calculate_h100_occupancy(
        threads_per_block,
        registers_per_thread,
        total_shared_bytes,
    )
    if occ_h100 is None:
        return None
    try:
        if math.isfinite(float(occ_h100)):
            return float(occ_h100)
    except Exception:
        return None
    return None


def _parse_rank_from_text(text: object) -> Optional[int]:
    s = str(text or "")
    m = _RANK_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return int(end_a) > int(start_b) and int(start_a) < int(end_b)


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    items: List[Tuple[int, int]] = []
    for s, e in intervals:
        ss = _to_int(s, -1)
        ee = _to_int(e, -1)
        if ss < 0 or ee <= ss:
            continue
        items.append((ss, ee))
    if not items:
        return merged
    items.sort(key=lambda x: (x[0], x[1]))
    cur_s, cur_e = items[0]
    for s, e in items[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
            continue
        merged.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _noop_debug(_: str) -> None:
    return


def _build_debug_logger(
    *,
    enabled: bool,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Callable[[str], None]:
    if not bool(enabled):
        return _noop_debug
    sink = log_fn or print

    def _emit(message: str) -> None:
        try:
            sink(f"[nsys-timeline-html][debug] {message}")
        except Exception:
            pass

    return _emit


def _preview_dict_rows(
    rows: Sequence[Dict[str, object]],
    *,
    keys: Sequence[str],
    limit: int = 3,
) -> str:
    try:
        max_rows = int(limit)
    except Exception:
        max_rows = 3
    out: List[Dict[str, object]] = []
    selected = list(rows) if max_rows <= 0 else list(rows)[:max_rows]
    for row in selected:
        item: Dict[str, object] = {}
        for key in keys:
            item[key] = row.get(key)
        out.append(item)
    return json.dumps(out, ensure_ascii=False)


def _safe_table_count(conn: sqlite3.Connection, table_name: str) -> Optional[int]:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {_ident(table_name)}").fetchone()
    except Exception:
        return None
    if not row:
        return None
    try:
        return int(row[0] or 0)
    except Exception:
        return None


def _select_nvtx_windows(
    provider: NsysSqliteMetricsProvider,
    *,
    nvtx_text: str,
) -> List[Dict[str, object]]:
    rows = provider.run_sql_skill(
        "nvtx_ranges_hierarchy",
        nvtx_text=str(nvtx_text or "%"),
        top_level_only=0,
        limit=500000,
    )
    if not rows:
        return []
    items = []
    for row in rows:
        s = _to_int(row.get("start_ns"), -1)
        e = _to_int(row.get("end_ns"), -1)
        if s < 0 or e <= s:
            continue
        items.append(
            {
                "nvtx_text": str(row.get("nvtx_text") or ""),
                "start_ns": s,
                "end_ns": e,
                "duration_ms": round((e - s) / 1e6, 3),
            }
        )
    if not items:
        return []
    items.sort(key=lambda x: (int(x["start_ns"]), int(x["end_ns"])))
    return items


def _pick_nvtx_windows(
    windows: Sequence[Dict[str, object]],
    *,
    nvtx_index: int,
) -> List[Dict[str, object]]:
    if not windows:
        return []
    # -1 means all matched scopes (default behavior).
    if int(nvtx_index) < 0:
        return list(windows)
    idx = max(0, min(int(nvtx_index), len(windows) - 1))
    return [dict(windows[idx])]


def _collect_kernels_in_window(
    provider: NsysSqliteMetricsProvider,
    *,
    start_ns: int,
    end_ns: int,
    nvtx_text: str,
    nvtx_windows: Optional[Sequence[Dict[str, object]]] = None,
    device_id: int,
    limit: int,
    debug_log: Optional[Callable[[str], None]] = None,
    debug_rows: int = 3,
) -> List[Dict[str, object]]:
    debug = debug_log or _noop_debug
    try:
        debug_rows_i = int(debug_rows)
    except Exception:
        debug_rows_i = 3
    rows: List[Dict[str, object]] = []
    attributed_kernel_keys = set()
    selected_windows: List[Tuple[int, int]] = []
    selected_pairs = set()
    if nvtx_windows:
        for item in nvtx_windows:
            ws = _to_int(item.get("start_ns"), -1)
            we = _to_int(item.get("end_ns"), -1)
            if we <= ws:
                continue
            selected_pairs.add((ws, we))
            selected_windows.append((ws, we))
    debug(
        "kernel query window start_ns={} end_ns={} device_id={} selected_nvtx_windows={}".format(
            int(start_ns),
            int(end_ns),
            int(device_id),
            len(selected_windows),
        )
    )

    # Prefer launch-attribution view (NVTX -> Runtime -> correlationId -> Kernel).
    skills = set(provider.list_sql_skills())
    debug("available skills={} has_nvtx_kernel_sm_detail={}".format(len(skills), "nvtx_kernel_sm_detail" in skills))
    detailed_kept = 0
    if "nvtx_kernel_sm_detail" in skills:
        detailed = provider.run_sql_skill(
            "nvtx_kernel_sm_detail",
            nvtx_text=str(nvtx_text or "%"),
            device_id=int(device_id),
            limit=int(limit),
        )
        debug("skill nvtx_kernel_sm_detail returned {} rows before filtering".format(len(detailed)))
        for row in detailed:
            ns = _to_int(row.get("nvtx_start_ns"), -1)
            ne = _to_int(row.get("nvtx_end_ns"), -1)
            if selected_pairs:
                if (ns, ne) not in selected_pairs:
                    continue
            # Robust fallback: older payload variants may expose generic start/end names.
            ks = _to_int(row.get("kernel_start_ns", row.get("start_ns")), -1)
            ke = _to_int(row.get("kernel_end_ns", row.get("end_ns")), -1)
            if ks < 0 or ke <= ks:
                continue
            if not selected_pairs and int(start_ns) >= 0 and int(end_ns) > int(start_ns):
                # Timeline window filtering must follow GPU execution timestamps.
                # Using NVTX CPU-side timestamps here can drop valid kernels due to
                # async CPU launch vs GPU execution lag.
                if not _intervals_overlap(ks, ke, int(start_ns), int(end_ns)):
                    continue
            uniq = (ks, ke, _to_int(row.get("stream_id"), 0), str(row.get("kernel_name") or ""))
            attributed_kernel_keys.add(uniq)
            detailed_kept += 1
            threads_per_block = row.get("threads_per_block")
            registers_per_thread = row.get("registersPerThread")
            total_shared_bytes = row.get("total_shared_bytes")
            rows.append(
                {
                    "stream_id": _to_int(row.get("stream_id"), 0),
                    "device_id": _to_int(row.get("device_id"), int(device_id)),
                    "kernel_name": str(row.get("kernel_name") or ""),
                    "start_ns": ks,
                    "end_ns": ke,
                    "duration_ms": float(row.get("duration_ms") or round((ke - ks) / 1e6, 6)),
                    "kind": str(row.get("kind") or "compute"),
                    "registers_per_thread": row.get("registersPerThread"),
                    "threads_per_block": row.get("threads_per_block"),
                    "static_shared_bytes": row.get("static_shared_bytes"),
                    "dynamic_shared_bytes": row.get("dynamic_shared_bytes"),
                    "total_shared_bytes": row.get("total_shared_bytes"),
                    "occupancy_pct_estimate": _resolve_occupancy_pct_estimate(
                        row.get("occupancy_pct_estimate"),
                        threads_per_block=threads_per_block,
                        registers_per_thread=registers_per_thread,
                        total_shared_bytes=total_shared_bytes,
                    ),
                    "nvtx_text": str(row.get("nvtx_text") or ""),
                    "nvtx_start_ns": ns,
                    "nvtx_end_ns": ne,
                    "rank": _parse_rank_from_text(row.get("nvtx_text")),
                }
            )
        debug("skill nvtx_kernel_sm_detail kept {} rows after filtering".format(detailed_kept))
    else:
        debug("skill nvtx_kernel_sm_detail unavailable, fallback to kernel_map only")

    # Fallback/补齐: plain kernel rows in time window.
    # This keeps timeline complete when launch-attribution misses some kernels.
    fallback = collect_kernel_rows(
        provider,
        device_id=int(device_id),
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        limit=int(limit),
        attach_iteration=False,
    )
    fallback_kept = 0
    debug("skill kernel_map returned {} rows before fallback filtering".format(len(fallback)))
    for row in fallback:
        ks = _to_int(row.get("start_ns"), -1)
        ke = _to_int(row.get("end_ns"), -1)
        if ks < 0 or ke <= ks:
            continue
        if selected_windows:
            if not any(_intervals_overlap(ks, ke, ws, we) for ws, we in selected_windows):
                continue
        elif int(start_ns) >= 0 and int(end_ns) > int(start_ns):
            if not _intervals_overlap(ks, ke, int(start_ns), int(end_ns)):
                continue
        uniq = (ks, ke, _to_int(row.get("stream_id"), 0), str(row.get("kernel_name") or ""))
        if uniq in attributed_kernel_keys:
            continue
        fallback_kept += 1
        threads_per_block = row.get("threads_per_block")
        registers_per_thread = row.get("registersPerThread")
        total_shared_bytes = row.get("total_shared_bytes")
        rows.append(
            {
                "stream_id": _to_int(row.get("stream_id"), 0),
                "device_id": _to_int(row.get("device_id"), int(device_id)),
                "kernel_name": str(row.get("kernel_name") or ""),
                "start_ns": ks,
                "end_ns": ke,
                "duration_ms": float(row.get("duration_ms") or 0.0),
                "kind": "comm" if bool(row.get("is_nccl")) else "compute",
                "registers_per_thread": None,
                "threads_per_block": None,
                "static_shared_bytes": None,
                "dynamic_shared_bytes": None,
                "total_shared_bytes": None,
                "occupancy_pct_estimate": _resolve_occupancy_pct_estimate(
                    row.get("occupancy_pct_estimate"),
                    threads_per_block=threads_per_block,
                    registers_per_thread=registers_per_thread,
                    total_shared_bytes=total_shared_bytes,
                ),
                "nvtx_text": None,
                "nvtx_start_ns": None,
                "nvtx_end_ns": None,
                "rank": None,
            }
        )
    rows.sort(key=lambda x: (_to_int(x.get("start_ns"), 0), _to_int(x.get("end_ns"), 0), _to_int(x.get("stream_id"), 0)))
    debug("kernel rows final={} (from detail={} + fallback_added={})".format(len(rows), detailed_kept, fallback_kept))
    stream_count = len(
        {
            (_to_int(item.get("device_id"), -1), _to_int(item.get("stream_id"), 0))
            for item in rows
        }
    )
    unique_gpu_kernel_count = len(
        {
            (
                _to_int(item.get("start_ns"), -1),
                _to_int(item.get("end_ns"), -1),
                _to_int(item.get("device_id"), -1),
                _to_int(item.get("stream_id"), 0),
                str(item.get("kernel_name") or ""),
            )
            for item in rows
        }
    )
    debug(
        "matched kernels total={} unique_gpu_kernels={} stream_count={} selected_nvtx_windows={}".format(
            len(rows),
            unique_gpu_kernel_count,
            stream_count,
            len(selected_windows),
        )
    )
    if rows:
        debug(
            "kernel sample={}".format(
                _preview_dict_rows(
                    rows,
                    keys=("kernel_name", "stream_id", "device_id", "start_ns", "end_ns", "kind"),
                    limit=debug_rows_i,
                )
            )
        )
    return rows


def _downsample_points(points: Sequence[Tuple[int, float]], max_points: int = 2000) -> List[Tuple[int, float]]:
    if int(max_points) <= 0:
        return [(int(t), float(v)) for t, v in points]
    if len(points) <= max_points:
        return [(int(t), float(v)) for t, v in points]
    step = max(1, int(math.ceil(len(points) / float(max_points))))
    sampled = [(int(points[i][0]), float(points[i][1])) for i in range(0, len(points), step)]
    last_t, last_v = points[-1]
    if sampled[-1][0] != int(last_t):
        sampled.append((int(last_t), float(last_v)))
    return sampled


def _metric_device_tag(device_id: object, source_name: object) -> Tuple[str, str]:
    dev = _to_int(device_id, -1)
    if dev >= 0:
        return "gpu", str(dev)
    src = str(source_name or "").strip()
    if src:
        m = _GPU_IDX_RE.search(src)
        if m:
            return "gpu", str(m.group(1))
        return "source", src
    return "unknown", "unknown"


def _metric_series_name(metric_name: str, tag_kind: str, tag_value: str) -> str:
    if tag_kind == "gpu":
        return f"{metric_name} [gpu {tag_value}]"
    if tag_kind == "source":
        return f"{metric_name} [src {tag_value}]"
    return f"{metric_name} [unknown]"


def _normalize_metric_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ").strip().lower())


def _is_percent_throughput_metric_name(metric_name: object) -> bool:
    raw = str(metric_name or "").lower()
    norm = _normalize_metric_text(metric_name)
    if "%" in raw:
        return True
    if "throughput" in norm:
        return True
    if "pct" in norm:
        return True
    if "percent" in norm:
        return True
    return False


def _is_default_focus_metric_name(metric_name: object) -> bool:
    norm = _normalize_metric_text(metric_name)
    if not norm:
        return False
    # Keep warps focus on throughput/% variants only, excluding avg/cycle-style variants.
    if ("warps in flight" in norm or "unallocated warps" in norm) and not _is_percent_throughput_metric_name(
        metric_name
    ):
        return False
    for token in _DEFAULT_FOCUS_METRIC_TOKENS:
        if token in norm:
            return True
    return False


def _collect_metric_samples(
    sqlite_path: str,
    *,
    start_ns: int,
    end_ns: int,
    metric_name_like: str = "%",
    include_all_sources: bool = False,
    device_id: int = -1,
    limit: int = -1,
    max_points_per_series: int = -1,
    apply_default_focus_filter: bool = False,
    restrict_to_intervals: Optional[Sequence[Tuple[int, int]]] = None,
    debug_log: Optional[Callable[[str], None]] = None,
    debug_rows: int = 3,
) -> List[Dict[str, object]]:
    debug = debug_log or _noop_debug
    try:
        debug_rows_i = int(debug_rows)
    except Exception:
        debug_rows_i = 3
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        schema = NsightSchema(conn)
        if not schema.metrics_table:
            debug("metrics table not detected in schema")
            return []
        metrics_table = _ident(schema.metrics_table)
        ts_col = schema.resolve_column(metrics_table, ("timestamp",))
        id_col = schema.metrics_id_col or schema.resolve_column(metrics_table, ("metricId", "nameId", "eventId"))
        val_col = schema.metrics_value_col or schema.resolve_column(metrics_table, ("value", "metricValue", "val"))
        if not ts_col:
            debug(
                "metrics timestamp unresolved table={} required_col=timestamp (rawTimestamp fallback disabled)".format(
                    metrics_table
                )
            )
            return []
        if not id_col or not val_col:
            debug(
                "metrics columns unresolved table={} ts_col={} id_col={} val_col={}".format(
                    metrics_table, ts_col, id_col, val_col
                )
            )
            return []
        debug(
            "metrics source table={} ts_col={} id_col={} val_col={} device_id={} include_all_sources={} metric_like={}".format(
                metrics_table,
                ts_col,
                id_col,
                val_col,
                int(device_id),
                int(bool(include_all_sources)),
                str(metric_name_like or "%"),
            )
        )

        name_expr = ""
        name_not_null = ""
        joins: List[str] = []

        if schema.table_exists("TARGET_INFO_GPU_METRICS"):
            gm_tbl = _ident("TARGET_INFO_GPU_METRICS")
            gm_id_col = schema.resolve_column(gm_tbl, ("metricId", "id", "metric_id"))
            gm_name_col = schema.resolve_column(gm_tbl, ("name", "metricName", "metric_name", "value"))
            g_type_col = schema.resolve_column(metrics_table, ("typeId", "type_id", "eventType", "event_type"))
            gm_type_col = schema.resolve_column(gm_tbl, ("typeId", "type_id", "eventType", "event_type"))
            if gm_id_col and gm_name_col:
                if g_type_col and gm_type_col:
                    joins.append(
                        f"JOIN {gm_tbl} gm "
                        f"ON g.{_ident(id_col)} = gm.{_ident(gm_id_col)} "
                        f"AND g.{_ident(g_type_col)} = gm.{_ident(gm_type_col)}"
                    )
                else:
                    joins.append(
                        f"JOIN {gm_tbl} gm ON g.{_ident(id_col)} = gm.{_ident(gm_id_col)}"
                    )
                name_expr = f"gm.{_ident(gm_name_col)}"
                name_not_null = f"AND gm.{_ident(gm_name_col)} IS NOT NULL "
        has_gpu_info_mapping = bool(name_expr.startswith("gm."))
        if has_gpu_info_mapping:
            debug("metrics name mapping=TARGET_INFO_GPU_METRICS")

        if not name_expr and schema.string_table:
            string_table = _ident(schema.string_table)
            joins.append(f"JOIN {string_table} s ON g.{_ident(id_col)} = s.id")
            name_expr = "s.value"
            name_not_null = "AND s.value IS NOT NULL "
            debug("metrics name mapping=StringIds")

        if not name_expr:
            name_expr = f"CAST(g.{_ident(id_col)} AS TEXT)"
            name_not_null = ""
            debug("metrics name mapping=CAST(metricId AS TEXT)")

        source_join = ""
        source_where = ""
        source_name_expr = "NULL"
        source_key_expr = ""
        source_col = schema.resolve_column(metrics_table, ("sourceId", "source_id"))
        if source_col:
            source_key_expr = f"g.{_ident(source_col)}"
        elif has_gpu_info_mapping:
            gm_source_col = schema.resolve_column("TARGET_INFO_GPU_METRICS", ("sourceId", "source_id"))
            if gm_source_col:
                source_key_expr = f"gm.{_ident(gm_source_col)}"

        if source_key_expr and schema.table_exists("GENERIC_EVENT_SOURCES"):
            ges_tbl = _ident("GENERIC_EVENT_SOURCES")
            ges_id_col = schema.resolve_column(ges_tbl, ("sourceId", "id", "source_id"))
            ges_name_col = schema.resolve_column(ges_tbl, ("name", "source", "sourceName"))
            ges_name_id_col = schema.resolve_column(ges_tbl, ("nameId", "name_id"))
            if ges_id_col:
                source_join = (
                    f"LEFT JOIN {ges_tbl} gs ON {source_key_expr} = gs.{_ident(ges_id_col)}"
                )
                if ges_name_col:
                    source_name_expr = f"gs.{_ident(ges_name_col)}"
                elif ges_name_id_col and schema.string_table:
                    source_join += (
                        f" LEFT JOIN {_ident(schema.string_table)} sgs "
                        f"ON gs.{_ident(ges_name_id_col)} = sgs.id"
                    )
                    source_name_expr = "sgs.value"
                if source_name_expr != "NULL":
                    source_where = (
                        "AND (? = 1 "
                        f"OR {source_name_expr} IS NULL "
                        f"OR LOWER({source_name_expr}) LIKE '%gpu%metric%' "
                        f"OR LOWER({source_name_expr}) = 'gpumetrics') "
                    )
        debug(
            "metrics source mapping key_expr={} source_name_expr={}".format(
                source_key_expr or "NULL",
                source_name_expr,
            )
        )

        device_where = ""
        device_expr = "NULL"
        filter_params: List[object] = [
            int(start_ns),
            int(end_ns),
            str(metric_name_like or "%"),
            str(metric_name_like or "%"),
        ]
        device_col = schema.resolve_column(metrics_table, ("deviceId", "gpuId", "device", "gpu"))
        if device_col:
            device_expr = f"CAST(g.{_ident(device_col)} AS INTEGER)"
            device_where = f"AND (? < 0 OR g.{_ident(device_col)} = ?) "
            filter_params.extend([int(device_id), int(device_id)])

        if source_where:
            filter_params.append(1 if bool(include_all_sources) else 0)

        from_join_expr = (
            f"FROM {metrics_table} g "
            + " ".join(joins)
            + " "
            + source_join
            + " "
        )

        limit_i = int(limit)
        tmp_tbl = "_myu_metrics_base"
        def _materialize_rows() -> Tuple[int, List[sqlite3.Row]]:
            ts_ident = _ident(str(ts_col))
            base_where = (
                "WHERE g.{ts} >= ? AND g.{ts} <= ? ".format(ts=ts_ident)
                + "AND (? = '%' OR {name_expr} LIKE ?) ".format(name_expr=name_expr)
                + name_not_null
                + device_where
                + source_where
                + f"AND g.{_ident(val_col)} IS NOT NULL "
            )
            select_expr = (
                f"SELECT g.{ts_ident} AS ts_ns, "
                f"{name_expr} AS metric_name, "
                f"{device_expr} AS metric_device_id, "
                f"{source_name_expr} AS metric_source_name, "
                f"CAST(g.{_ident(val_col)} AS REAL) AS metric_value "
            )
            conn.execute(f"DROP TABLE IF EXISTS {tmp_tbl}")
            create_tmp_sql = (
                f"CREATE TEMP TABLE {tmp_tbl} AS "
                + select_expr
                + from_join_expr
                + base_where
            )
            conn.execute(create_tmp_sql, filter_params)
            total = int(conn.execute(f"SELECT COUNT(*) FROM {tmp_tbl}").fetchone()[0] or 0)
            debug("metrics temp rows={} ts_col={} sampling_limit={}".format(total, ts_col, int(limit_i)))
            if total > 0:
                sample_sql = (
                    f"SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value "
                    f"FROM {tmp_tbl} ORDER BY ts_ns ASC"
                )
                if debug_rows_i > 0:
                    sample_sql += f" LIMIT {int(debug_rows_i)}"
                sample_rows = conn.execute(sample_sql).fetchall()
                debug("metrics temp sample ts_col={} data={}".format(ts_col, json.dumps([dict(x) for x in sample_rows], ensure_ascii=False)))

            if total <= 0:
                return total, []
            if limit_i <= 0 or total <= limit_i:
                data_rows = conn.execute(
                    f"SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value "
                    f"FROM {tmp_tbl} ORDER BY ts_ns ASC"
                ).fetchall()
                return total, data_rows

            # Distribute sampling budget by metric/device series, so each series keeps timeline coverage.
            series_rows = conn.execute(
                f"SELECT metric_name, metric_device_id, metric_source_name, COUNT(*) AS cnt "
                f"FROM {tmp_tbl} "
                f"GROUP BY metric_name, metric_device_id, metric_source_name "
                f"ORDER BY metric_name, metric_device_id, metric_source_name"
            ).fetchall()
            series_count = max(1, len(series_rows))
            target_per_series = max(2, int(math.floor(float(limit_i) / float(series_count))))
            collected: List[sqlite3.Row] = []
            for sr in series_rows:
                metric_name = sr["metric_name"]
                metric_device_id = sr["metric_device_id"]
                metric_source_name = sr["metric_source_name"]
                cnt = int(sr["cnt"] or 0)
                if cnt <= target_per_series:
                    stride = 1
                else:
                    stride = max(1, int(math.ceil(float(cnt) / float(target_per_series))))
                sampled_sql = (
                    "WITH numbered AS ("
                    f"SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value, "
                    "ROW_NUMBER() OVER (ORDER BY ts_ns ASC) AS rn, "
                    "COUNT(*) OVER () AS total_rows "
                    f"FROM {tmp_tbl} "
                    "WHERE metric_name = ? "
                    "AND metric_device_id IS ? "
                    "AND metric_source_name IS ?"
                    ") "
                    "SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value "
                    "FROM numbered "
                    "WHERE rn = 1 OR rn = total_rows OR ((rn - 1) % ?) = 0 "
                    "ORDER BY ts_ns ASC"
                )
                sampled_rows = conn.execute(
                    sampled_sql,
                    [metric_name, metric_device_id, metric_source_name, int(stride)],
                ).fetchall()
                collected.extend(sampled_rows)
            data_rows = sorted(collected, key=lambda r: (_to_int(r["ts_ns"], 0), str(r["metric_name"] or "")))
            return total, data_rows

        total_rows, rows = _materialize_rows()
        if restrict_to_intervals:
            merged_intervals = _merge_intervals(restrict_to_intervals)
            before = len(rows)
            if not merged_intervals:
                rows = []
            else:
                filtered_rows: List[sqlite3.Row] = []
                idx = 0
                for row in rows:
                    ts = _to_int(row["ts_ns"], -1)
                    if ts < 0:
                        continue
                    while idx < len(merged_intervals) and ts > merged_intervals[idx][1]:
                        idx += 1
                    if idx >= len(merged_intervals):
                        break
                    lo, hi = merged_intervals[idx]
                    if ts >= lo and ts <= hi:
                        filtered_rows.append(row)
                rows = filtered_rows
            debug(
                "metrics interval filter applied intervals={} rows_before={} rows_after={}".format(
                    len(merged_intervals) if restrict_to_intervals else 0,
                    int(before),
                    int(len(rows)),
                )
            )
        debug("metrics rows after sampling={} using_ts_col={}".format(len(rows), ts_col))
    finally:
        conn.close()

    grouped: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}
    for row in rows:
        name = str(row["metric_name"] or "")
        if not name:
            continue
        tag_kind, tag_value = _metric_device_tag(row["metric_device_id"], row["metric_source_name"])
        ts_ns = _to_int(row["ts_ns"], -1)
        if ts_ns < 0:
            continue
        try:
            value = float(row["metric_value"])
        except Exception:
            continue
        grouped.setdefault((name, tag_kind, tag_value), []).append((ts_ns, value))

    if bool(apply_default_focus_filter):
        before_count = len(grouped)
        grouped = {k: v for k, v in grouped.items() if _is_default_focus_metric_name(k[0])}
        debug(
            "metrics default focus filter enabled tokens={} before_series={} after_series={}".format(
                len(_DEFAULT_FOCUS_METRIC_TOKENS),
                int(before_count),
                int(len(grouped)),
            )
        )

    series: List[Dict[str, object]] = []
    for name, tag_kind, tag_value in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2])):
        points = _downsample_points(
            grouped[(name, tag_kind, tag_value)],
            max_points=int(max_points_per_series),
        )
        display_name = _metric_series_name(str(name), str(tag_kind), str(tag_value))
        series.append(
            {
                "name": display_name,
                "color": _color_for_name(display_name),
                "points": [[int(t), float(v)] for t, v in points],
            }
        )
    total_points = sum(len(item.get("points", [])) for item in series)
    debug(
        "metrics series={} total_points={} max_points_per_series={}".format(
            len(series),
            int(total_points),
            int(max_points_per_series),
        )
    )
    if series:
        debug(
            "metrics series sample={}".format(
                _preview_dict_rows(series, keys=("name", "color"), limit=debug_rows_i)
            )
        )
    return series


def _render_html(
    *,
    sqlite_path: str,
    kernels: List[Dict[str, object]],
    metric_series: List[Dict[str, object]],
    window_start_ns: int,
    window_end_ns: int,
    nvtx_windows: Optional[Sequence[Dict[str, object]]],
    width_px: int,
    include_metrics: bool,
    overlay_metrics_per_track: int,
) -> str:
    span = max(1, int(window_end_ns) - int(window_start_ns))
    grouped: Dict[Tuple[Optional[int], int], Dict[int, List[Dict[str, object]]]] = {}
    for row in kernels:
        rv = row.get("rank")
        rank = int(rv) if isinstance(rv, int) else None
        dev = _to_int(row.get("device_id"), -1)
        sid = _to_int(row.get("stream_id"), 0)
        key = (rank, dev)
        grouped.setdefault(key, {}).setdefault(sid, []).append(row)
    group_keys = sorted(
        grouped.keys(),
        key=lambda k: (k[0] is None, int(k[0]) if k[0] is not None else 10**9, int(k[1])),
    )
    stream_count = sum(len(streams) for streams in grouped.values())
    known_ranks = sorted({int(k[0]) for k in group_keys if k[0] is not None})
    has_unknown_rank = any(k[0] is None for k in group_keys)

    nvtx_count = len(list(nvtx_windows or []))
    meta_nvtx = f" | nvtx_scopes={nvtx_count}" if nvtx_count > 0 else ""
    rank_meta = f" | ranks={len(known_ranks) + (1 if has_unknown_rank else 0)}"

    all_stream_groups: List[Dict[str, object]] = []
    for rank, dev in group_keys:
        stream_rows: List[Dict[str, object]] = []
        streams = grouped.get((rank, dev), {})
        for sid in sorted(streams.keys()):
            kernels_payload: List[Dict[str, object]] = []
            for row in streams[sid]:
                ks = _to_int(row.get("start_ns"), 0)
                ke = _to_int(row.get("end_ns"), 0)
                kname = str(row.get("kernel_name") or "")
                kernels_payload.append(
                    {
                        "kernel_name": kname,
                        "start_ns": ks,
                        "end_ns": ke,
                        "kind": str(row.get("kind") or ""),
                        "duration_ms": row.get("duration_ms"),
                        "occupancy_pct_estimate": row.get("occupancy_pct_estimate"),
                        "color": _color_for_name(kname),
                    }
                )
            stream_rows.append(
                {
                    "stream_id": int(sid),
                    "kernels": kernels_payload,
                }
            )
        all_stream_groups.append(
            {
                "rank": rank,
                "device_id": int(dev),
                "streams": stream_rows,
            }
        )

    payload = {
        "window_start_ns": int(window_start_ns),
        "window_end_ns": int(window_end_ns),
        "span_ns": int(span),
        "metrics": metric_series if include_metrics else [],
        "chart_width": int(width_px),
        "all_stream_groups": all_stream_groups,
        "overlay_metrics_per_track": int(max(0, int(overlay_metrics_per_track))),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    lines: List[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<title>NSYS NVTX Timeline</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#0f1116;color:#e7e9ee;}",
        ".meta{margin-bottom:12px;color:#a8afbf;font-size:13px;}",
        ".card{background:#171c28;border:1px solid #2a3243;border-radius:8px;padding:12px;margin-bottom:14px;}",
        ".panel-title{font-size:13px;color:#c8d0df;margin:0 0 8px 0;}",
        ".legend{display:flex;flex-wrap:wrap;gap:8px 12px;margin-top:8px;font-size:12px;color:#b7c0d0;}",
        ".legend-item{display:flex;align-items:center;gap:6px;}",
        ".swatch{width:10px;height:10px;border-radius:2px;}",
        ".metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:10px;}",
        ".metric-panel{background:#121826;border:1px solid #2a3243;border-radius:6px;padding:8px;}",
        ".metric-title{font-size:12px;color:#d6deef;margin-bottom:4px;word-break:break-word;}",
        ".metric-sub{font-size:11px;color:#95a4bf;margin:4px 0 0 0;}",
        ".row{display:flex;align-items:center;margin:8px 0;}",
        ".label{width:140px;color:#a8afbf;font-size:12px;}",
        f".track{{position:relative;height:24px;width:{int(width_px)}px;background:#1a1f2b;border-radius:4px;overflow:hidden;}}",
        ".bar{position:absolute;height:18px;top:3px;border-radius:3px;}",
        ".bar:hover{outline:1px solid #fff;}",
        ".stream-track{height:56px;background:#171d2b;}",
        ".stream-track .bar{top:34px;height:18px;}",
        ".metric-overlay{position:absolute;left:0;top:2px;width:100%;height:30px;pointer-events:none;}",
        ".metric-overlay line{stroke:#5b6c8e;stroke-width:1;opacity:0.7;}",
        ".metric-overlay polyline{fill:none;stroke-width:1.2;stroke-linejoin:round;stroke-linecap:round;opacity:0.92;}",
        ".stream-overlay-hint{font-size:11px;color:#94a3bd;margin-top:8px;}",
        ".allstream-root{display:flex;flex-direction:column;gap:10px;}",
        ".allstream-panel{position:relative;background:#121826;border:1px solid #2a3243;border-radius:6px;padding:8px;}",
        ".allstream-title{font-size:12px;color:#d6deef;margin:0 0 6px 0;}",
        ".allstream-scroll{overflow-x:auto;overflow-y:hidden;border:1px solid #29344a;border-radius:4px;background:#0f1522;}",
        ".allstream-controls{display:flex;flex-wrap:wrap;gap:6px 10px;margin-bottom:6px;}",
        ".shift-row{display:flex;align-items:center;gap:6px;font-size:11px;color:#9fb0ca;background:#101725;border:1px solid #2a3243;border-radius:4px;padding:3px 6px;}",
        ".shift-btn{background:#1a2538;color:#dbe6ff;border:1px solid #3d4f70;border-radius:3px;padding:1px 6px;cursor:pointer;font-size:11px;}",
        ".shift-btn:hover{background:#22314a;}",
        ".allstream-svg{display:block;}",
        ".allstream-hover-tip{position:absolute;z-index:20;min-width:220px;max-width:460px;background:rgba(10,16,28,0.96);border:1px solid #47608c;border-radius:6px;padding:6px 8px;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,0.35);font-size:11px;color:#dce7ff;}",
        ".allstream-hover-head{font-size:11px;color:#a9bce0;margin-bottom:4px;font-family:Consolas,Monaco,monospace;}",
        ".allstream-hover-row{display:flex;align-items:center;gap:6px;line-height:1.25;margin:2px 0;}",
        ".allstream-hover-swatch{width:8px;height:8px;border-radius:2px;flex:none;}",
        ".allstream-hover-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}",
        ".allstream-hover-val{font-family:Consolas,Monaco,monospace;color:#f3f7ff;}",
        ".axis-note{font-size:11px;color:#8792a8;margin-top:6px;}",
        ".empty{color:#8f9ab0;font-size:12px;padding:6px 0;}",
        ".group-title{font-size:12px;color:#c9d3e8;margin:10px 0 4px 0;}",
        "</style></head><body>",
        "<h2>NSYS NVTX Window Timeline</h2>",
        (
            f"<div class='meta'>sqlite={html.escape(str(sqlite_path))} | "
            f"kernels={len(kernels)} | streams={stream_count}{rank_meta}{meta_nvtx}</div>"
        ),
    ]

    if include_metrics:
        lines.extend(
            [
                "<div class='card'>",
                "<h3 class='panel-title'>GPU Metrics In Window</h3>",
                "<div id='metrics-grid' class='metrics-grid'></div>",
                "<div class='axis-note'>Each metric/device is rendered in an independent panel (own Y-axis). X-axis is shared ns timeline.</div>",
                "</div>",
            ]
        )

    lines.extend(
        [
            "<div class='card'>",
            "<h3 class='panel-title'>All Streams Overlap + Metrics Alignment</h3>",
            "<div id='allstream-root' class='allstream-root'></div>",
            (
                "<div class='axis-note'>"
                "All streams for the same rank/device are merged into one wide panel. "
                "Kernel Theoretical Occupancy Sum [%] is auto-derived by summing overlapping kernels, and "
                "you can use +/- controls to shift each metric curve vertically while keeping the same shared X-axis."
                "</div>"
            ),
            "</div>",
        ]
    )

    if nvtx_count > 0:
        lines.extend(["<div class='card'>", "<h3 class='panel-title'>Matched NVTX Scopes</h3>"])
        lines.append("<div class='row'>")
        lines.append("<div class='label'>nvtx</div>")
        lines.append("<div class='track'>")
        for i, scope in enumerate(nvtx_windows or []):
            s = _to_int(scope.get("start_ns"), 0)
            e = _to_int(scope.get("end_ns"), 0)
            if e <= s:
                continue
            left = (s - int(window_start_ns)) / float(span)
            width = max((e - s) / float(span), 1.0 / float(width_px))
            left_px = int(left * width_px)
            width_bar = max(1, int(width * width_px))
            name = str(scope.get("nvtx_text") or f"scope_{i}")
            color = _color_for_name(f"nvtx::{name}")
            title = f"{name} | start_ns={s} | end_ns={e} | duration_ms={scope.get('duration_ms')}"
            lines.append(
                f"<div class='bar' style='left:{left_px}px;width:{width_bar}px;background:{color};opacity:0.85;' "
                f"title='{html.escape(title)}'></div>"
            )
        lines.append("</div></div>")
        lines.append("</div>")

    lines.extend(["<div class='card'>", "<h3 class='panel-title'>Kernel Timeline By Stream</h3>"])
    if not kernels:
        lines.append("<div class='empty'>No kernels in selected window.</div>")
    else:
        for rank, dev in group_keys:
            if rank is None:
                group_title = "Rank Unknown"
            else:
                group_title = f"Rank {rank}"
            if int(dev) >= 0:
                group_title += f" | Device {int(dev)}"
            lines.append(f"<div class='group-title'>{html.escape(group_title)}</div>")
            streams = grouped.get((rank, dev), {})
            for sid in sorted(streams.keys()):
                rank_attr = "" if rank is None else str(rank)
                lines.append("<div class='row'>")
                lines.append(f"<div class='label'>stream {sid}</div>")
                lines.append(
                    "<div class='track stream-track' "
                    f"data-rank='{html.escape(rank_attr)}' "
                    f"data-device='{int(dev)}' "
                    f"data-stream='{int(sid)}'>"
                )
                for row in streams[sid]:
                    s = _to_int(row.get("start_ns"), 0)
                    e = _to_int(row.get("end_ns"), 0)
                    left = (s - int(window_start_ns)) / float(span)
                    width = max((e - s) / float(span), 1.0 / float(width_px))
                    left_px = int(left * width_px)
                    width_bar = max(1, int(width * width_px))
                    name = str(row.get("kernel_name") or "")
                    color = _color_for_name(name)
                    title = (
                        f"{name} | kind={row.get('kind')} | dur_ms={row.get('duration_ms')} | "
                        f"start_ns={s} | end_ns={e} | stream={sid} | device={dev} | rank={rank} | "
                        f"tpb={row.get('threads_per_block')} | regs={row.get('registers_per_thread')} | "
                        f"smem={row.get('total_shared_bytes')} | occ_theoretical_pct={row.get('occupancy_pct_estimate')}"
                    )
                    lines.append(
                        f"<div class='bar' style='left:{left_px}px;width:{width_bar}px;background:{color};' "
                        f"title='{html.escape(title)}'></div>"
                    )
                lines.append("</div></div>")
        if include_metrics and int(overlay_metrics_per_track) > 0:
            lines.append(
                "<div class='stream-overlay-hint'>"
                "Per-stream attribution overlay: metric curves are stacked above kernel bars in each stream lane "
                "(metrics are selected per device by highest variation)."
                "</div>"
            )
    lines.extend(["</div>"])

    lines.extend(
        [
            "<script>",
            f"const TIMELINE_DATA = {payload_json};",
            "(function(){",
            "  const d = TIMELINE_DATA;",
            "  const grid = document.getElementById('metrics-grid');",
            "  const series = [...(Array.isArray(d.metrics) ? d.metrics : [])].sort((a,b) => String(a.name||'').localeCompare(String(b.name||'')));",
            "  const span = Math.max(1, Number(d.span_ns || 1));",
            "  const startNs = Number(d.window_start_ns || 0);",
            "  const mk = (tag, attrs) => {",
            "    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);",
            "    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v));",
            "    return el;",
            "  };",
            "  const fmt = (v) => {",
            "    const x = Number(v);",
            "    if (!Number.isFinite(x)) return 'nan';",
            "    const ax = Math.abs(x);",
            "    if (ax >= 1e6 || (ax > 0 && ax < 1e-3)) return x.toExponential(2);",
            "    return x.toFixed(3);",
            "  };",
            "  if (grid) {",
            "    if (!series.length) {",
            "      grid.innerHTML = \"<div class='empty'>No GPU metric samples in this window.</div>\";",
            "    } else {",
            "      const W = Math.max(480, Math.min(920, Number(d.chart_width || 1200)));",
            "      const H = 190;",
            "      const padL = 52, padR = 16, padT = 12, padB = 26;",
            "      const x0 = padL, y0 = padT, cw = W - padL - padR, ch = H - padT - padB;",
            "      const x = (t) => x0 + ((Number(t) - startNs) / span) * cw;",
            "      for (const s of series) {",
            "        const ptsRaw = Array.isArray(s.points) ? s.points : [];",
            "        if (!ptsRaw.length) continue;",
            "        let minV = Number.POSITIVE_INFINITY;",
            "        let maxV = Number.NEGATIVE_INFINITY;",
            "        for (const p of ptsRaw) {",
            "          const v = Number(p[1]);",
            "          if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "        }",
            "        if (!Number.isFinite(minV) || !Number.isFinite(maxV)) { minV = 0; maxV = 1; }",
            "        if (Math.abs(maxV - minV) < 1e-12) { maxV = minV + 1.0; }",
            "        const y = (v) => y0 + (1.0 - ((Number(v) - minV) / (maxV - minV))) * ch;",
            "        const card = document.createElement('div'); card.className = 'metric-panel';",
            "        const title = document.createElement('div'); title.className = 'metric-title'; title.textContent = String(s.name || 'metric');",
            "        const svg = mk('svg', {width:W, height:H, viewBox:`0 0 ${W} ${H}`});",
            "        svg.appendChild(mk('rect', {x:x0, y:y0, width:cw, height:ch, fill:'#111725', stroke:'#32405b'}));",
            "        svg.appendChild(mk('line', {x1:x0, y1:y0+ch, x2:x0+cw, y2:y0+ch, stroke:'#55637f'}));",
            "        svg.appendChild(mk('line', {x1:x0, y1:y0, x2:x0, y2:y0+ch, stroke:'#55637f'}));",
            "        const yMinText = mk('text', {x:6, y:y0+ch, fill:'#9aa7be', 'font-size':11}); yMinText.textContent = fmt(minV); svg.appendChild(yMinText);",
            "        const yMaxText = mk('text', {x:6, y:y0+10, fill:'#9aa7be', 'font-size':11}); yMaxText.textContent = fmt(maxV); svg.appendChild(yMaxText);",
            "        const xStartText = mk('text', {x:x0, y:y0+ch+16, fill:'#7f8daa', 'font-size':10}); xStartText.textContent = String(Math.round(startNs)); svg.appendChild(xStartText);",
            "        const xEndText = mk('text', {x:x0+cw-92, y:y0+ch+16, fill:'#7f8daa', 'font-size':10}); xEndText.textContent = String(Math.round(startNs + span)); svg.appendChild(xEndText);",
            "        const pts = ptsRaw.map((p) => `${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ');",
            "        if (pts) {",
            "          svg.appendChild(mk('polyline', {points:pts, fill:'none', stroke:s.color || '#88c', 'stroke-width':1.4, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "        }",
            "        const sub = document.createElement('div'); sub.className = 'metric-sub';",
            "        sub.textContent = `samples=${ptsRaw.length} | min=${fmt(minV)} | max=${fmt(maxV)}`;",
            "        card.appendChild(title);",
            "        card.appendChild(svg);",
            "        card.appendChild(sub);",
            "        grid.appendChild(card);",
            "      }",
            "    }",
            "  }",
            "  const parseGpuTag = (name) => {",
            "    const m = String(name || '').match(/\\[gpu\\s+(\\d+)\\]/i);",
            "    return m ? Number(m[1]) : null;",
            "  };",
            "  const normalizeMetricName = (name) => {",
            "    let s = String(name || '').toLowerCase();",
            "    s = s.replace(/\\[[^\\]]*\\]/g, ' ');",
            "    s = s.replace(/_/g, ' ');",
            "    s = s.replace(/\\s+/g, ' ').trim();",
            "    return s;",
            "  };",
            "  const isThroughputPercentName = (name) => {",
            "    const raw = String(name || '').toLowerCase();",
            "    const n = normalizeMetricName(name);",
            "    return raw.includes('%') || n.includes('throughput') || n.includes('pct') || n.includes('percent');",
            "  };",
            "  const isComputeWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('compute warps in flight') || (n.includes('warps in flight') && !n.includes('unallocated'))) && isThroughputPercentName(name);",
            "  };",
            "  const isUnallocatedWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('unallocated warps in active sms') || n.includes('unallocated warps')) && isThroughputPercentName(name);",
            "  };",
            "  const toNumericPoints = (seriesLike) => {",
            "    const ptsRaw = Array.isArray(seriesLike && seriesLike.points) ? seriesLike.points : [];",
            "    const out = [];",
            "    for (const p of ptsRaw) {",
            "      const t = Number(p[0]);",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(t) && Number.isFinite(v)) out.push([t, v]);",
            "    }",
            "    out.sort((a,b) => a[0] - b[0]);",
            "    return out;",
            "  };",
            "  const interpAt = (pts, t) => {",
            "    if (!pts.length) return 0;",
            "    const tt = Number(t);",
            "    if (tt <= pts[0][0]) return Number(pts[0][1]) || 0;",
            "    const last = pts[pts.length - 1];",
            "    if (tt >= last[0]) return Number(last[1]) || 0;",
            "    let lo = 0, hi = pts.length - 1;",
            "    while (lo + 1 < hi) {",
            "      const mid = (lo + hi) >> 1;",
            "      if (pts[mid][0] <= tt) lo = mid; else hi = mid;",
            "    }",
            "    const t0 = Number(pts[lo][0]);",
            "    const t1 = Number(pts[hi][0]);",
            "    const v0 = Number(pts[lo][1]);",
            "    const v1 = Number(pts[hi][1]);",
            "    if (!Number.isFinite(t0) || !Number.isFinite(t1) || Math.abs(t1 - t0) < 1e-12) return Number.isFinite(v1) ? v1 : 0;",
            "    const a = (tt - t0) / (t1 - t0);",
            "    const vv0 = Number.isFinite(v0) ? v0 : 0;",
            "    const vv1 = Number.isFinite(v1) ? v1 : 0;",
            "    return vv0 + (vv1 - vv0) * a;",
            "  };",
            "  const seriesScore = (s) => {",
            "    const ptsRaw = Array.isArray(s.points) ? s.points : [];",
            "    if (!ptsRaw.length) return 0;",
            "    let minV = Number.POSITIVE_INFINITY;",
            "    let maxV = Number.NEGATIVE_INFINITY;",
            "    for (const p of ptsRaw) {",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "    }",
            "    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return 0;",
            "    return Math.abs(maxV - minV);",
            "  };",
            "  const overlayCount = Math.max(0, Number(d.overlay_metrics_per_track || 0));",
            "  const tracks = Array.from(document.querySelectorAll('.stream-track'));",
            "  if (overlayCount > 0 && tracks.length > 0) {",
            "    for (const track of tracks) {",
            "      const dev = Number(track.dataset.device || -1);",
            "      let selected = series.filter((s) => {",
            "        const g = parseGpuTag(s.name);",
            "        return g !== null && g === dev;",
            "      });",
            "      if (selected.length === 0) selected = series;",
            "      selected = [...selected].sort((a,b) => seriesScore(b) - seriesScore(a));",
            "      if (overlayCount > 0) {",
            "        let trimmed = selected.slice(0, overlayCount);",
            "        if (overlayCount >= 2) {",
            "          const computePreferred = selected.find((s) => isComputeWarpsSeries(s.name));",
            "          const unallocPreferred = selected.find((s) => isUnallocatedWarpsSeries(s.name));",
            "          if (computePreferred && unallocPreferred) {",
            "            trimmed = [computePreferred, unallocPreferred, ...trimmed.filter((s) => s !== computePreferred && s !== unallocPreferred)].slice(0, overlayCount);",
            "          }",
            "        }",
            "        selected = trimmed;",
            "      } else {",
            "        selected = [];",
            "      }",
            "      if (!selected.length) continue;",
            "      const tw = Math.max(1, Number(track.clientWidth || d.chart_width || 1));",
            "      const svgH = 30;",
            "      const yTop = 3;",
            "      const yBottom = 27;",
            "      const svg = mk('svg', {class:'metric-overlay', width:tw, height:svgH, viewBox:`0 0 ${tw} ${svgH}`, preserveAspectRatio:'none'});",
            "      svg.appendChild(mk('line', {x1:0, y1:yBottom, x2:tw, y2:yBottom}));",
            "      const xTrack = (t) => ((Number(t) - startNs) / span) * tw;",
            "      const clampPct = (v) => Math.max(0, Math.min(100, Number(v) || 0));",
            "      const yPct = (vPct) => yBottom - (clampPct(vPct) / 100.0) * (yBottom - yTop);",
            "      const drawArea = (topPts, basePts, color, opacity) => {",
            "        if (topPts.length < 2 || basePts.length < 2) return;",
            "        const poly = [];",
            "        for (const p of topPts) poly.push(`${p[0].toFixed(2)},${p[1].toFixed(2)}`);",
            "        for (let i = basePts.length - 1; i >= 0; i--) poly.push(`${basePts[i][0].toFixed(2)},${basePts[i][1].toFixed(2)}`);",
            "        svg.appendChild(mk('polygon', {points:poly.join(' '), fill:String(color || '#88c'), opacity:String(opacity)}));",
            "      };",
            "      const computeSeries = selected.find((s) => isComputeWarpsSeries(s.name));",
            "      const unallocSeries = selected.find((s) => isUnallocatedWarpsSeries(s.name));",
            "      const handled = new Set();",
            "      if (computeSeries && unallocSeries) {",
            "        const cPts = toNumericPoints(computeSeries);",
            "        const uPts = toNumericPoints(unallocSeries);",
            "        if (cPts.length >= 2 && uPts.length >= 2) {",
            "          const cTop = [];",
            "          const cBase = [];",
            "          for (const p of cPts) {",
            "            const tx = xTrack(p[0]);",
            "            if (!Number.isFinite(tx)) continue;",
            "            cTop.push([tx, yPct(p[1])]);",
            "            cBase.push([tx, yBottom]);",
            "          }",
            "          drawArea(cTop, cBase, String(computeSeries.color || '#4fa5ff'), 0.42);",
            "          const uTop = [];",
            "          const uBase = [];",
            "          for (const p of uPts) {",
            "            const tx = xTrack(p[0]);",
            "            if (!Number.isFinite(tx)) continue;",
            "            const base = clampPct(interpAt(cPts, p[0]));",
            "            const top = clampPct(base + p[1]);",
            "            uTop.push([tx, yPct(top)]);",
            "            uBase.push([tx, yPct(base)]);",
            "          }",
            "          drawArea(uTop, uBase, String(unallocSeries.color || '#ffae57'), 0.58);",
            "          const cLine = cTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "          if (cLine) svg.appendChild(mk('polyline', {points:cLine, fill:'none', stroke:String(computeSeries.color || '#4fa5ff'), 'stroke-width':1.05, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "          const uLine = uTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "          if (uLine) svg.appendChild(mk('polyline', {points:uLine, fill:'none', stroke:String(unallocSeries.color || '#ffae57'), 'stroke-width':1.05, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "          handled.add(computeSeries);",
            "          handled.add(unallocSeries);",
            "        }",
            "      }",
            "      for (const ms of selected) {",
            "        if (handled.has(ms)) continue;",
            "        const ptsRaw = toNumericPoints(ms);",
            "        if (!ptsRaw.length) continue;",
            "        let minV = Number.POSITIVE_INFINITY;",
            "        let maxV = Number.NEGATIVE_INFINITY;",
            "        for (const p of ptsRaw) {",
            "          const v = Number(p[1]);",
            "          if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "        }",
            "        if (!Number.isFinite(minV) || !Number.isFinite(maxV)) continue;",
            "        if (Math.abs(maxV - minV) < 1e-12) maxV = minV + 1.0;",
            "        const pts = [];",
            "        for (const p of ptsRaw) {",
            "          const tx = xTrack(p[0]);",
            "          const v = Number(p[1]);",
            "          if (!Number.isFinite(tx) || !Number.isFinite(v)) continue;",
            "          const ty = yTop + (1.0 - ((v - minV) / (maxV - minV))) * (yBottom - yTop);",
            "          pts.push(`${tx.toFixed(2)},${ty.toFixed(2)}`);",
            "        }",
            "        if (pts.length >= 2) {",
            "          svg.appendChild(mk('polyline', {points:pts.join(' '), stroke:ms.color || '#88c'}));",
            "        }",
            "      }",
            "      track.appendChild(svg);",
            "      const names = selected.map((s) => String(s.name || '')).join(' | ');",
            "      if (names) {",
            "        const oldTitle = String(track.getAttribute('title') || '');",
            "        const title = oldTitle ? `${oldTitle} || overlay_metrics=${names}` : `overlay_metrics=${names}`;",
            "        track.setAttribute('title', title);",
            "      }",
            "    }",
            "  }",
            "})();",
            "</script>",
            "<script>",
            "(function(){",
            "  const d = TIMELINE_DATA;",
            "  const root = document.getElementById('allstream-root');",
            "  if (!root) return;",
            "  const groups = Array.isArray(d.all_stream_groups) ? d.all_stream_groups : [];",
            "  if (!groups.length) {",
            "    root.innerHTML = \"<div class='empty'>No kernels available for all-stream view.</div>\";",
            "    return;",
            "  }",
            "  const metricsAll = Array.isArray(d.metrics) ? d.metrics : [];",
            "  const overlayCount = Math.max(0, Number(d.overlay_metrics_per_track || 0));",
            "  const span = Math.max(1, Number(d.span_ns || 1));",
            "  const startNs = Number(d.window_start_ns || 0);",
            "  const basePlotW = Math.max(2200, Math.round(Number(d.chart_width || 1800) * 1.35));",
            "  const parseGpuTag = (name) => {",
            "    const m = String(name || '').match(/\\[gpu\\s+(\\d+)\\]/i);",
            "    return m ? Number(m[1]) : null;",
            "  };",
            "  const normalizeMetricName = (name) => {",
            "    let s = String(name || '').toLowerCase();",
            "    s = s.replace(/\\[[^\\]]*\\]/g, ' ');",
            "    s = s.replace(/_/g, ' ');",
            "    s = s.replace(/\\s+/g, ' ').trim();",
            "    return s;",
            "  };",
            "  const isThroughputPercentName = (name) => {",
            "    const raw = String(name || '').toLowerCase();",
            "    const n = normalizeMetricName(name);",
            "    return raw.includes('%') || n.includes('throughput') || n.includes('pct') || n.includes('percent');",
            "  };",
            "  const isComputeWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('compute warps in flight') || (n.includes('warps in flight') && !n.includes('unallocated'))) && isThroughputPercentName(name);",
            "  };",
            "  const isUnallocatedWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('unallocated warps in active sms') || n.includes('unallocated warps')) && isThroughputPercentName(name);",
            "  };",
            "  const toNumericPoints = (seriesLike) => {",
            "    const ptsRaw = Array.isArray(seriesLike && seriesLike.points) ? seriesLike.points : [];",
            "    const out = [];",
            "    for (const p of ptsRaw) {",
            "      const t = Number(p[0]);",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(t) && Number.isFinite(v)) out.push([t, v]);",
            "    }",
            "    out.sort((a,b) => a[0] - b[0]);",
            "    return out;",
            "  };",
            "  const interpAt = (pts, t) => {",
            "    if (!pts.length) return 0;",
            "    const tt = Number(t);",
            "    if (tt <= pts[0][0]) return Number(pts[0][1]) || 0;",
            "    const last = pts[pts.length - 1];",
            "    if (tt >= last[0]) return Number(last[1]) || 0;",
            "    let lo = 0, hi = pts.length - 1;",
            "    while (lo + 1 < hi) {",
            "      const mid = (lo + hi) >> 1;",
            "      if (pts[mid][0] <= tt) lo = mid; else hi = mid;",
            "    }",
            "    const t0 = Number(pts[lo][0]);",
            "    const t1 = Number(pts[hi][0]);",
            "    const v0 = Number(pts[lo][1]);",
            "    const v1 = Number(pts[hi][1]);",
            "    if (!Number.isFinite(t0) || !Number.isFinite(t1) || Math.abs(t1 - t0) < 1e-12) return Number.isFinite(v1) ? v1 : 0;",
            "    const a = (tt - t0) / (t1 - t0);",
            "    const vv0 = Number.isFinite(v0) ? v0 : 0;",
            "    const vv1 = Number.isFinite(v1) ? v1 : 0;",
            "    return vv0 + (vv1 - vv0) * a;",
            "  };",
            "  const mkSvg = (tag, attrs) => {",
            "    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);",
            "    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v));",
            "    return el;",
            "  };",
            "  const metricRange = (series) => {",
            "    const pts = Array.isArray(series.points) ? series.points : [];",
            "    let minV = Number.POSITIVE_INFINITY;",
            "    let maxV = Number.NEGATIVE_INFINITY;",
            "    for (const p of pts) {",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "    }",
            "    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return 0;",
            "    return Math.abs(maxV - minV);",
            "  };",
            "  const shortName = (name) => {",
            "    const s = String(name || 'metric');",
            "    return s.length <= 52 ? s : (s.slice(0, 49) + '...');",
            "  };",
            "  const fmtVal = (v) => {",
            "    const x = Number(v);",
            "    if (!Number.isFinite(x)) return 'nan';",
            "    const ax = Math.abs(x);",
            "    if (ax >= 1e6 || (ax > 0 && ax < 1e-3)) return x.toExponential(2);",
            "    return x.toFixed(3);",
            "  };",
            "  const stepSampleAt = (pts, t) => {",
            "    if (!Array.isArray(pts) || pts.length === 0) return null;",
            "    const tt = Number(t);",
            "    if (!Number.isFinite(tt)) return null;",
            "    let lo = 0;",
            "    let hi = pts.length;",
            "    while (lo < hi) {",
            "      const mid = (lo + hi) >> 1;",
            "      if (Number(pts[mid][0]) <= tt) lo = mid + 1;",
            "      else hi = mid;",
            "    }",
            "    const idx = Math.max(0, lo - 1);",
            "    const ts0 = Number(pts[idx][0]);",
            "    const value = Number(pts[idx][1]);",
            "    if (!Number.isFinite(ts0) || !Number.isFinite(value)) return null;",
            "    const ts1 = (idx + 1 < pts.length) ? Number(pts[idx + 1][0]) : null;",
            "    return {value:value, start_ns:ts0, end_ns:Number.isFinite(ts1) ? ts1 : null};",
            "  };",
            "  const buildKernelOccSumSeries = (streamRows) => {",
            "    const events = new Map();",
            "    const addEvent = (ts, delta) => {",
            "      const t = Number(ts);",
            "      const d = Number(delta);",
            "      if (!Number.isFinite(t) || !Number.isFinite(d)) return;",
            "      events.set(t, Number(events.get(t) || 0) + d);",
            "    };",
            "    for (const srow of (Array.isArray(streamRows) ? streamRows : [])) {",
            "      const kernels = Array.isArray(srow.kernels) ? srow.kernels : [];",
            "      for (const k of kernels) {",
            "        const s = Number(k.start_ns);",
            "        const e = Number(k.end_ns);",
            "        const occRaw = k.occupancy_pct_estimate;",
            "        if (occRaw === null || occRaw === undefined || occRaw === '') continue;",
            "        const occ = Number(occRaw);",
            "        if (!Number.isFinite(s) || !Number.isFinite(e) || !(e > s) || !Number.isFinite(occ)) continue;",
            "        addEvent(s, occ);",
            "        addEvent(e, -occ);",
            "      }",
            "    }",
            "    const times = [...events.keys()].sort((a,b) => a - b);",
            "    const points = [];",
            "    let running = 0.0;",
            "    const wndStart = Number(startNs);",
            "    const wndEnd = Number(startNs + span);",
            "    if (times.length > 0 && wndStart < times[0]) points.push([wndStart, 0.0]);",
            "    for (const t of times) {",
            "      running += Number(events.get(t) || 0);",
            "      if (Math.abs(running) < 1e-9) running = 0.0;",
            "      points.push([t, running]);",
            "    }",
            "    if (times.length > 0 && times[times.length - 1] < wndEnd) points.push([wndEnd, running]);",
            "    return {",
            "      name: 'Kernel Theoretical Occupancy Sum [%]',",
            "      color: '#7fd96b',",
            "      points: points,",
            "      metric_kind: 'kernel_occ_sum'",
            "    };",
            "  };",
            "  for (const g of groups) {",
                "    const panel = document.createElement('div');",
                "    panel.className = 'allstream-panel';",
            "    const title = document.createElement('div');",
            "    title.className = 'allstream-title';",
            "    const rankText = (g.rank === null || g.rank === undefined) ? 'Rank Unknown' : `Rank ${g.rank}`;",
            "    const devText = Number(g.device_id) >= 0 ? `Device ${g.device_id}` : 'Device Unknown';",
            "    title.textContent = `${rankText} | ${devText}`;",
            "    panel.appendChild(title);",
            "",
            "    const streamRows = Array.isArray(g.streams) ? g.streams : [];",
            "    if (!streamRows.length) {",
            "      const empty = document.createElement('div');",
            "      empty.className = 'empty';",
            "      empty.textContent = 'No stream rows in this group.';",
            "      panel.appendChild(empty);",
            "      root.appendChild(panel);",
            "      continue;",
            "    }",
            "",
            "    let selectedMetrics = [];",
            "    if (overlayCount > 0 && metricsAll.length > 0) {",
            "      const dev = Number(g.device_id);",
            "      const exact = metricsAll.filter((s) => parseGpuTag(s.name) === dev);",
            "      selectedMetrics = (exact.length > 0 ? exact : metricsAll).slice();",
            "      selectedMetrics.sort((a,b) => metricRange(b) - metricRange(a));",
            "      if (overlayCount > 0) {",
            "        let trimmed = selectedMetrics.slice(0, overlayCount);",
            "        if (overlayCount >= 2) {",
            "          const computePreferred = selectedMetrics.find((s) => isComputeWarpsSeries(s.name));",
            "          const unallocPreferred = selectedMetrics.find((s) => isUnallocatedWarpsSeries(s.name));",
            "          if (computePreferred && unallocPreferred) {",
            "            trimmed = [computePreferred, unallocPreferred, ...trimmed.filter((s) => s !== computePreferred && s !== unallocPreferred)].slice(0, overlayCount);",
            "          }",
            "        }",
            "        selectedMetrics = trimmed;",
            "      } else {",
            "        selectedMetrics = [];",
            "      }",
            "    }",
            "",
            "    const occSumSeries = buildKernelOccSumSeries(streamRows);",
            "    const metricState = [];",
            "    if (Array.isArray(occSumSeries.points) && occSumSeries.points.length > 0) {",
            "      metricState.push({series:occSumSeries, offset:0, visible:true});",
            "    }",
            "    for (const s of selectedMetrics) metricState.push({series:s, offset:0, visible:true});",
            "    for (const st of metricState) st._pts = toNumericPoints(st.series);",
            "    if (metricState.length > 0) {",
                "      const ctrls = document.createElement('div');",
                "      ctrls.className = 'allstream-controls';",
            "      panel.appendChild(ctrls);",
            "      for (const st of metricState) {",
            "        const row = document.createElement('div');",
            "        row.className = 'shift-row';",
            "        const sw = document.createElement('span');",
            "        sw.style.width = '10px';",
            "        sw.style.height = '10px';",
            "        sw.style.borderRadius = '2px';",
            "        sw.style.display = 'inline-block';",
            "        sw.style.background = String(st.series.color || '#88c');",
            "        row.appendChild(sw);",
            "        const lbl = document.createElement('span');",
            "        lbl.textContent = shortName(st.series.name);",
            "        row.appendChild(lbl);",
            "        const dy = document.createElement('span');",
            "        dy.textContent = 'dy=0';",
            "        row.appendChild(dy);",
            "        const mkBtn = (text) => {",
            "          const b = document.createElement('button');",
            "          b.type = 'button';",
            "          b.className = 'shift-btn';",
            "          b.textContent = text;",
            "          return b;",
            "        };",
            "        const up = mkBtn('up');",
            "        const down = mkBtn('down');",
            "        const reset = mkBtn('reset');",
            "        const toggle = mkBtn('hide');",
            "        row.appendChild(up);",
            "        row.appendChild(down);",
            "        row.appendChild(reset);",
            "        row.appendChild(toggle);",
            "        ctrls.appendChild(row);",
            "        st._dyEl = dy;",
            "        st._toggleEl = toggle;",
            "        st._btnUp = up;",
            "        st._btnDown = down;",
            "        st._btnReset = reset;",
            "      }",
            "    }",
            "",
            "    const scroll = document.createElement('div');",
            "    scroll.className = 'allstream-scroll';",
            "    panel.appendChild(scroll);",
            "    const laneH = 18;",
            "    const laneGap = 6;",
            "    const labelW = 88;",
            "    const rightPad = 18;",
            "    const metricBandH = metricState.length > 0 ? 132 : 0;",
            "    const lanesH = streamRows.length * (laneH + laneGap);",
            "    const svgH = metricBandH + lanesH + 38;",
            "    const svgW = labelW + basePlotW + rightPad;",
            "    const svg = mkSvg('svg', {class:'allstream-svg', width:svgW, height:svgH, viewBox:`0 0 ${svgW} ${svgH}`});",
            "    scroll.appendChild(svg);",
            "    const x = (t) => labelW + ((Number(t) - startNs) / span) * basePlotW;",
            "    const yMetricTop = 8;",
            "    const yMetricBottom = metricBandH > 0 ? (metricBandH - 14) : 0;",
            "    const hoverTip = document.createElement('div');",
            "    hoverTip.className = 'allstream-hover-tip';",
            "    hoverTip.style.display = 'none';",
            "    panel.appendChild(hoverTip);",
            "    let hoverNs = null;",
            "    let hoverLine = null;",
            "    const updateHoverLine = () => {",
            "      if (!hoverLine || hoverNs === null || metricBandH <= 0) {",
            "        if (hoverLine) hoverLine.setAttribute('visibility', 'hidden');",
            "        return;",
            "      }",
            "      const hx = Math.max(labelW, Math.min(labelW + basePlotW, x(hoverNs)));",
            "      hoverLine.setAttribute('x1', String(hx));",
            "      hoverLine.setAttribute('x2', String(hx));",
            "      hoverLine.setAttribute('y1', String(yMetricTop));",
            "      hoverLine.setAttribute('y2', String(yMetricBottom));",
            "      hoverLine.setAttribute('visibility', 'visible');",
            "    };",
            "    const hideHover = () => {",
            "      hoverNs = null;",
            "      hoverTip.style.display = 'none';",
            "      if (hoverLine) hoverLine.setAttribute('visibility', 'hidden');",
            "    };",
            "    const updateHoverAtEvent = (ev) => {",
            "      if (metricBandH <= 0) return;",
            "      const rect = svg.getBoundingClientRect();",
            "      const px = Number(ev.clientX) - rect.left;",
            "      const py = Number(ev.clientY) - rect.top;",
            "      if (!Number.isFinite(px) || !Number.isFinite(py) || py < 2 || py > (metricBandH - 2)) {",
            "        hideHover();",
            "        return;",
            "      }",
            "      const plotX = Math.max(labelW, Math.min(labelW + basePlotW, px));",
            "      hoverNs = startNs + ((plotX - labelW) / basePlotW) * span;",
            "      updateHoverLine();",
            "      const visibleMetricState = metricState.filter((st) => st.visible);",
            "      if (!visibleMetricState.length) {",
            "        hideHover();",
            "        return;",
            "      }",
            "      const rows = [];",
            "      for (const st of visibleMetricState) {",
            "        const sample = stepSampleAt(st._pts, hoverNs);",
            "        if (!sample) continue;",
            "        const endText = sample.end_ns === null ? 'end' : String(Math.round(sample.end_ns));",
            "        rows.push(`<div class='allstream-hover-row'><span class='allstream-hover-swatch' style='background:${String(st.series.color || '#88c')}'></span><span class='allstream-hover-name'>${shortName(st.series.name)}</span><span class='allstream-hover-val'>${fmtVal(sample.value)} @ [${Math.round(sample.start_ns)}, ${endText})</span></div>`);",
            "      }",
            "      if (!rows.length) {",
            "        hideHover();",
            "        return;",
            "      }",
            "      hoverTip.innerHTML = `<div class='allstream-hover-head'>t=${Math.round(hoverNs)} ns</div>${rows.join('')}`;",
            "      hoverTip.style.display = 'block';",
            "      const panelRect = panel.getBoundingClientRect();",
            "      let tx = Number(ev.clientX) - panelRect.left + 12;",
            "      let ty = Number(ev.clientY) - panelRect.top + 12;",
            "      const maxX = Math.max(6, panel.clientWidth - hoverTip.offsetWidth - 6);",
            "      const maxY = Math.max(6, panel.clientHeight - hoverTip.offsetHeight - 6);",
            "      if (tx > maxX) tx = maxX;",
            "      if (ty > maxY) ty = maxY;",
            "      hoverTip.style.left = `${Math.round(tx)}px`;",
            "      hoverTip.style.top = `${Math.round(ty)}px`;",
            "    };",
            "",
            "    const render = () => {",
                "      while (svg.firstChild) svg.removeChild(svg.firstChild);",
            "      svg.appendChild(mkSvg('rect', {x:0, y:0, width:svgW, height:svgH, fill:'#0f1522'}));",
            "      if (metricBandH > 0) {",
            "        svg.appendChild(mkSvg('rect', {x:labelW, y:2, width:basePlotW, height:metricBandH-4, fill:'#111a2b', stroke:'#30405c'}));",
            "      }",
            "      const yStreamsTop = metricBandH + 10;",
            "      svg.appendChild(mkSvg('rect', {x:labelW, y:yStreamsTop-2, width:basePlotW, height:lanesH+8, fill:'#131c2a', stroke:'#30405c'}));",
            "      svg.appendChild(mkSvg('line', {x1:labelW, y1:yStreamsTop-2, x2:labelW+basePlotW, y2:yStreamsTop-2, stroke:'#55637f'}));",
            "      const tStart = mkSvg('text', {x:labelW, y:svgH-6, fill:'#7f8daa', 'font-size':10});",
            "      tStart.textContent = String(Math.round(startNs));",
            "      svg.appendChild(tStart);",
            "      const tEnd = mkSvg('text', {x:labelW+basePlotW-95, y:svgH-6, fill:'#7f8daa', 'font-size':10});",
            "      tEnd.textContent = String(Math.round(startNs + span));",
            "      svg.appendChild(tEnd);",
            "",
            "      if (metricBandH > 0) {",
            "        const visibleMetricState = metricState.filter((st) => st.visible);",
            "        const computeState = visibleMetricState.find((st) => isComputeWarpsSeries(st.series && st.series.name));",
            "        const unallocState = visibleMetricState.find((st) => isUnallocatedWarpsSeries(st.series && st.series.name));",
            "        const handled = new Set();",
            "        const clampPct = (v) => Math.max(0, Math.min(100, Number(v) || 0));",
            "        const pairOffset = (computeState && unallocState) ? (Number(computeState.offset || 0) + Number(unallocState.offset || 0)) : 0;",
            "        const yPct = (vPct) => yMetricBottom - (clampPct(vPct) / 100.0) * (yMetricBottom - yMetricTop) + pairOffset;",
            "        const drawArea = (topPts, basePts, color, opacity) => {",
            "          if (topPts.length < 2 || basePts.length < 2) return;",
            "          const poly = [];",
            "          for (const p of topPts) poly.push(`${p[0].toFixed(2)},${p[1].toFixed(2)}`);",
            "          for (let i = basePts.length - 1; i >= 0; i--) poly.push(`${basePts[i][0].toFixed(2)},${basePts[i][1].toFixed(2)}`);",
            "          svg.appendChild(mkSvg('polygon', {points:poly.join(' '), fill:String(color || '#88c'), opacity:String(opacity)}));",
            "        };",
            "        if (computeState && unallocState) {",
            "          const cPts = computeState._pts;",
            "          const uPts = unallocState._pts;",
            "          if (cPts.length >= 2 && uPts.length >= 2) {",
            "            const cTop = [];",
            "            const cBase = [];",
            "            for (const p of cPts) {",
            "              const tx = x(p[0]);",
            "              if (!Number.isFinite(tx)) continue;",
            "              cTop.push([tx, yPct(p[1])]);",
            "              cBase.push([tx, yMetricBottom + pairOffset]);",
            "            }",
            "            drawArea(cTop, cBase, String(computeState.series.color || '#4fa5ff'), 0.42);",
            "            const uTop = [];",
            "            const uBase = [];",
            "            for (const p of uPts) {",
            "              const tx = x(p[0]);",
            "              if (!Number.isFinite(tx)) continue;",
            "              const base = clampPct(interpAt(cPts, p[0]));",
            "              const top = clampPct(base + p[1]);",
            "              uTop.push([tx, yPct(top)]);",
            "              uBase.push([tx, yPct(base)]);",
            "            }",
            "            drawArea(uTop, uBase, String(unallocState.series.color || '#ffae57'), 0.58);",
            "            const cLine = cTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "            if (cLine) svg.appendChild(mkSvg('polyline', {points:cLine, fill:'none', stroke:String(computeState.series.color || '#4fa5ff'), 'stroke-width':1.1, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "            const uLine = uTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "            if (uLine) svg.appendChild(mkSvg('polyline', {points:uLine, fill:'none', stroke:String(unallocState.series.color || '#ffae57'), 'stroke-width':1.1, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "            handled.add(computeState);",
            "            handled.add(unallocState);",
            "          }",
            "        }",
            "        for (const st of visibleMetricState) {",
            "          if (handled.has(st)) continue;",
            "          const ptsRaw = st._pts;",
            "          if (!ptsRaw.length) continue;",
            "          let minV = Number.POSITIVE_INFINITY;",
            "          let maxV = Number.NEGATIVE_INFINITY;",
            "          for (const p of ptsRaw) {",
            "            const v = Number(p[1]);",
            "            if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "          }",
            "          if (!Number.isFinite(minV) || !Number.isFinite(maxV)) continue;",
            "          if (Math.abs(maxV - minV) < 1e-12) maxV = minV + 1.0;",
            "          const points = [];",
            "          for (const p of ptsRaw) {",
            "            const tx = x(p[0]);",
            "            const v = Number(p[1]);",
            "            if (!Number.isFinite(tx) || !Number.isFinite(v)) continue;",
            "            const tyNorm = yMetricTop + (1.0 - ((v - minV) / (maxV - minV))) * (yMetricBottom - yMetricTop);",
            "            const ty = tyNorm + st.offset;",
            "            points.push(`${tx.toFixed(2)},${ty.toFixed(2)}`);",
            "          }",
            "          if (points.length >= 2) {",
            "            svg.appendChild(mkSvg('polyline', {points:points.join(' '), fill:'none', stroke:String(st.series.color || '#88c'), 'stroke-width':1.2, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "          }",
            "        }",
            "      }",
            "",
            "      for (let i = 0; i < streamRows.length; i++) {",
            "        const srow = streamRows[i];",
            "        const laneY = yStreamsTop + i * (laneH + laneGap);",
            "        const lbl = mkSvg('text', {x:4, y:laneY + 13, fill:'#98a9c5', 'font-size':11});",
            "        lbl.textContent = `stream ${srow.stream_id}`;",
            "        svg.appendChild(lbl);",
            "        svg.appendChild(mkSvg('line', {x1:labelW, y1:laneY + laneH + 1, x2:labelW + basePlotW, y2:laneY + laneH + 1, stroke:'#2d3a52'}));",
            "        const kernels = Array.isArray(srow.kernels) ? srow.kernels : [];",
            "        for (const k of kernels) {",
            "          const s = Number(k.start_ns || 0);",
            "          const e = Number(k.end_ns || 0);",
            "          if (!(e > s)) continue;",
            "          const left = x(s);",
            "          const width = Math.max(1, ((e - s) / span) * basePlotW);",
            "          const rect = mkSvg('rect', {x:left, y:laneY, width:width, height:laneH, rx:2, ry:2, fill:String(k.color || '#6699cc'), opacity:0.92});",
            "          const ttl = mkSvg('title', {});",
            "          ttl.textContent = `${k.kernel_name || ''} | stream=${srow.stream_id} | start_ns=${s} | end_ns=${e} | kind=${k.kind || ''} | occ_theoretical_pct=${k.occupancy_pct_estimate}`;",
            "          rect.appendChild(ttl);",
            "          svg.appendChild(rect);",
            "        }",
            "      }",
            "      if (metricBandH > 0) {",
            "        hoverLine = mkSvg('line', {x1:labelW, y1:yMetricTop, x2:labelW, y2:yMetricBottom, stroke:'#dbe6ff', 'stroke-width':1, opacity:0.92, visibility:'hidden'});",
            "        svg.appendChild(hoverLine);",
            "        updateHoverLine();",
            "      } else {",
            "        hoverLine = null;",
            "        hoverTip.style.display = 'none';",
            "      }",
            "    };",
            "",
            "    if (metricBandH > 0) {",
            "      svg.addEventListener('mousemove', updateHoverAtEvent);",
            "      svg.addEventListener('mouseleave', hideHover);",
            "    }",
            "    for (const st of metricState) {",
                "      if (st._btnUp) st._btnUp.onclick = () => { st.offset -= 10; st._dyEl.textContent = `dy=${st.offset}`; render(); };",
            "      if (st._btnDown) st._btnDown.onclick = () => { st.offset += 10; st._dyEl.textContent = `dy=${st.offset}`; render(); };",
            "      if (st._btnReset) st._btnReset.onclick = () => { st.offset = 0; st.visible = true; st._dyEl.textContent = 'dy=0'; if (st._toggleEl) st._toggleEl.textContent = 'hide'; render(); };",
            "      if (st._toggleEl) st._toggleEl.onclick = () => { st.visible = !st.visible; st._toggleEl.textContent = st.visible ? 'hide' : 'show'; render(); };",
            "    }",
            "    render();",
            "    root.appendChild(panel);",
            "  }",
            "})();",
            "</script>",
            "</body></html>",
        ]
    )
    return "\n".join(lines)


def export_timeline_html(
    sqlite_path: str,
    *,
    output_path: str,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 100000,
    width_px: int = 1800,
    nvtx_text: str = "",
    nvtx_index: int = -1,
    include_metrics: bool = False,
    metric_name_like: str = "%",
    metrics_limit: int = -1,
    metrics_max_points: int = -1,
    overlay_metrics_per_track: int = 7,
    default_focus_metrics: bool = False,
    include_all_metric_sources: bool = False,
    debug: bool = False,
    debug_rows: int = -1,
    debug_log_fn: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Export an interactive HTML timeline from an nsys SQLite file.

    Parameters
    ----------
    progress_cb:
        Optional callback for per-phase progress messages printed to the caller.
        Each call receives a string like ``"  done: collect_kernels  [234 ms]"``.
        Pass ``lambda msg: print(msg, file=sys.stderr)`` for CLI progress output.
    """
    debug_log = _build_debug_logger(enabled=bool(debug), log_fn=debug_log_fn)

    _phase_timings: List[Dict[str, float]] = []

    def _emit_phase(name: str, elapsed_ms: float) -> None:
        _phase_timings.append({"phase": name, "elapsed_ms": elapsed_ms})
        if progress_cb:
            progress_cb(f"  done:     {name}  [{elapsed_ms} ms]")
    try:
        debug_rows_i = int(debug_rows)
    except Exception:
        debug_rows_i = -1
    debug_log(
        "start sqlite={} output={} device_id={} start_ns={} end_ns={} nvtx_text={} nvtx_index={} include_metrics={} metrics_limit={} metrics_max_points={} overlay_metrics_per_track={} default_focus_metrics={} debug_rows={}".format(
            str(sqlite_path),
            str(output_path),
            int(device_id),
            int(start_ns),
            int(end_ns),
            str(nvtx_text or ""),
            int(nvtx_index),
            int(bool(include_metrics)),
            int(metrics_limit),
            int(metrics_max_points),
            int(overlay_metrics_per_track),
            int(bool(default_focus_metrics)),
            int(debug_rows_i),
        )
    )
    _t0 = time.perf_counter()
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            schema = NsightSchema(conn)
            debug_log(
                "schema tables kernel={} runtime={} nvtx={} metrics={} string={}".format(
                    schema.kernel_table,
                    schema.runtime_table,
                    schema.nvtx_table,
                    schema.metrics_table,
                    schema.string_table,
                )
            )
            debug_log(
                "metrics columns timestamp={} id={} value={}".format(
                    schema.metrics_timestamp_col,
                    schema.metrics_id_col,
                    schema.metrics_value_col,
                )
            )
            for table_name in (
                schema.kernel_table,
                schema.runtime_table,
                schema.nvtx_table,
                schema.metrics_table,
                "TARGET_INFO_GPU_METRICS",
                "GENERIC_EVENT_SOURCES",
            ):
                if not table_name:
                    continue
                count = _safe_table_count(conn, table_name)
                debug_log("table count {}={}".format(table_name, "n/a" if count is None else count))
        finally:
            conn.close()
    except Exception as exc:
        debug_log("schema probe failed: {}".format(exc))
    _emit_phase("schema_probe", round((time.perf_counter() - _t0) * 1000, 1))

    provider = NsysSqliteMetricsProvider(sqlite_path)

    _t0 = time.perf_counter()
    selected_nvtx_windows: List[Dict[str, object]] = []
    effective_start_ns = int(start_ns)
    effective_end_ns = int(end_ns)
    if str(nvtx_text or "").strip():
        matched_nvtx = _select_nvtx_windows(
            provider,
            nvtx_text=str(nvtx_text),
        )
        selected_nvtx_windows = _pick_nvtx_windows(matched_nvtx, nvtx_index=int(nvtx_index))
        debug_log(
            "nvtx match count={} selected_count={}".format(
                len(matched_nvtx),
                len(selected_nvtx_windows),
            )
        )
        if selected_nvtx_windows:
            debug_log(
                "selected nvtx sample={}".format(
                    _preview_dict_rows(
                        selected_nvtx_windows,
                        keys=("nvtx_text", "start_ns", "end_ns", "duration_ms"),
                        limit=debug_rows_i,
                    )
                )
            )
        if selected_nvtx_windows:
            effective_start_ns = min(_to_int(item.get("start_ns"), -1) for item in selected_nvtx_windows)
            effective_end_ns = max(_to_int(item.get("end_ns"), -1) for item in selected_nvtx_windows)
            debug_log(
                "effective window from nvtx start_ns={} end_ns={}".format(
                    int(effective_start_ns),
                    int(effective_end_ns),
                )
            )

    _emit_phase("nvtx_window_selection", round((time.perf_counter() - _t0) * 1000, 1))

    _t0 = time.perf_counter()
    if effective_start_ns < 0 or effective_end_ns < 0 or effective_end_ns <= effective_start_ns:
        # If no explicit valid window, infer from full kernel rows.
        base_rows = collect_kernel_rows(
            provider,
            device_id=int(device_id),
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            limit=int(limit),
            attach_iteration=False,
        )
        debug_log("infer window from kernel_map rows={}".format(len(base_rows)))
        if not base_rows:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("<html><body><h2>No kernels</h2></body></html>", encoding="utf-8")
            debug_log("no kernels found; wrote empty html")
            return str(out)
        effective_start_ns = min(int(item["start_ns"]) for item in base_rows)
        effective_end_ns = max(int(item["end_ns"]) for item in base_rows)
        debug_log(
            "effective window from kernels start_ns={} end_ns={}".format(
                int(effective_start_ns),
                int(effective_end_ns),
            )
        )
    _emit_phase("window_inference", round((time.perf_counter() - _t0) * 1000, 1))

    _t0 = time.perf_counter()
    # 按照nvtx range内的runtime correlationId到具体的kernel，得到kernel window
    kernels = _collect_kernels_in_window(
        provider,
        start_ns=int(effective_start_ns),
        end_ns=int(effective_end_ns),
        nvtx_text=str(nvtx_text or "%"),
        nvtx_windows=selected_nvtx_windows or None,
        device_id=int(device_id),
        limit=int(limit),
        debug_log=debug_log,
        debug_rows=debug_rows_i,
    )

    render_start_ns = int(effective_start_ns)
    render_end_ns = int(effective_end_ns)
    kernel_intervals = _merge_intervals(
        [
            (_to_int(row.get("start_ns"), -1), _to_int(row.get("end_ns"), -1))
            for row in kernels
        ]
    )
    if kernel_intervals:
        kernel_span_start = kernel_intervals[0][0]
        kernel_span_end = kernel_intervals[-1][1]
        new_render_start = min(int(render_start_ns), int(kernel_span_start))
        new_render_end = max(int(render_end_ns), int(kernel_span_end))
        if new_render_start != int(render_start_ns) or new_render_end != int(render_end_ns):
            debug_log(
                "extend render window with kernel span start_ns={} end_ns={} (old=[{}, {}], new=[{}, {}])".format(
                    int(kernel_span_start),
                    int(kernel_span_end),
                    int(render_start_ns),
                    int(render_end_ns),
                    int(new_render_start),
                    int(new_render_end),
                )
            )
            render_start_ns = int(new_render_start)
            render_end_ns = int(new_render_end)
    kernel_stream_count = len(
        {
            (_to_int(row.get("device_id"), -1), _to_int(row.get("stream_id"), 0))
            for row in kernels
        }
    )
    debug_log(
        "collect_kernels matched_kernels={} streams={} merged_gpu_intervals={}".format(
            len(kernels),
            kernel_stream_count,
            len(kernel_intervals),
        )
    )
    if progress_cb:
        progress_cb(
            "  info:     matched_kernels={}  streams={}  merged_gpu_intervals={}".format(
                len(kernels),
                kernel_stream_count,
                len(kernel_intervals),
            )
        )

    metric_series: List[Dict[str, object]] = []
    if bool(include_metrics):
        metric_query_start_ns = int(render_start_ns)
        metric_query_end_ns = int(render_end_ns)
        metric_restrict_intervals: Optional[List[Tuple[int, int]]] = None
        debug_log(
            "metrics query uses render window start_ns={} end_ns={} (no kernel-interval restriction, kernel_intervals={})".format(
                int(metric_query_start_ns),
                int(metric_query_end_ns),
                len(kernel_intervals),
            )
        )
        apply_focus_filter = bool(default_focus_metrics) and str(metric_name_like or "%").strip() in ("", "%")
        debug_log(
            "metrics focus filter active={} metric_name_like={}".format(
                int(apply_focus_filter),
                str(metric_name_like or "%"),
            )
        )
        metric_series = _collect_metric_samples(
            sqlite_path,
            start_ns=int(metric_query_start_ns),
            end_ns=int(metric_query_end_ns),
            metric_name_like=str(metric_name_like or "%"),
            include_all_sources=bool(include_all_metric_sources),
            device_id=int(device_id),
            limit=int(metrics_limit),
            max_points_per_series=int(metrics_max_points),
            apply_default_focus_filter=bool(apply_focus_filter),
            restrict_to_intervals=metric_restrict_intervals,
            debug_log=debug_log,
            debug_rows=debug_rows_i,
        )
    else:
        debug_log("metrics disabled (include_metrics=0)")
    _emit_phase("collect_kernels_and_metrics", round((time.perf_counter() - _t0) * 1000, 1))

    _t0 = time.perf_counter()
    text = _render_html(
        sqlite_path=str(sqlite_path),
        kernels=kernels,
        metric_series=metric_series,
        window_start_ns=int(render_start_ns),
        window_end_ns=int(render_end_ns),
        nvtx_windows=selected_nvtx_windows,
        width_px=int(width_px),
        include_metrics=bool(include_metrics),
        overlay_metrics_per_track=int(overlay_metrics_per_track),
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    _emit_phase("render_and_write", round((time.perf_counter() - _t0) * 1000, 1))
    debug_log(
        "done output={} kernels={} metric_series={} window=[{}, {}]".format(
            str(out),
            len(kernels),
            len(metric_series),
            int(render_start_ns),
            int(render_end_ns),
        )
    )
    if progress_cb:
        total_ms = round(sum(float(p.get("elapsed_ms") or 0) for p in _phase_timings), 1)
        progress_cb(f"  total:    {total_ms} ms  (phases: {len(_phase_timings)})")
    return str(out)


def export_timeline_compare_html(
    sqlite_paths: Sequence[str],
    *,
    output_path: str,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 100000,
    width_px: int = 1800,
    nvtx_text: str = "",
    nvtx_index: int = -1,
    include_metrics: bool = False,
    metric_name_like: str = "%",
    metrics_limit: int = -1,
    metrics_max_points: int = -1,
    overlay_metrics_per_track: int = 7,
    default_focus_metrics: bool = False,
    include_all_metric_sources: bool = False,
    debug: bool = False,
    debug_rows: int = -1,
    debug_log_fn: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    items = [str(p) for p in sqlite_paths if str(p or "").strip()]
    if len(items) < 2:
        raise ValueError("export_timeline_compare_html requires at least two sqlite paths")

    compare_debug = _build_debug_logger(enabled=bool(debug), log_fn=debug_log_fn)
    rendered: List[Dict[str, str]] = []
    total = len(items)

    def _build_section_srcdoc(full_html: str, *, panel_title: str) -> Optional[str]:
        if str(panel_title or "").strip() not in str(full_html or ""):
            return None
        injected = "\n".join(
            [
                "<style>",
                "body{margin:0;padding:0;background:#0f1116;overflow:hidden;}",
                "h2,.meta{display:none !important;}",
                ".card{margin:0 !important;}",
                "</style>",
                "<script>",
                "(function(){",
                f"  const target = {json.dumps(str(panel_title), ensure_ascii=False)};",
                "  const cards = Array.from(document.querySelectorAll('.card'));",
                "  let kept = 0;",
                "  for (const card of cards) {",
                "    const titleEl = card.querySelector('.panel-title');",
                "    const title = titleEl ? String(titleEl.textContent || '').trim() : '';",
                "    if (title === target) {",
                "      kept += 1;",
                "      continue;",
                "    }",
                "    card.remove();",
                "  }",
                "  const h2 = document.querySelector('h2');",
                "  if (h2) h2.remove();",
                "  const meta = document.querySelector('.meta');",
                "  if (meta) meta.remove();",
                "  if (!kept && document.body) {",
                "    document.body.innerHTML = \"<div class='card'><div class='empty'>Section not available.</div></div>\";",
                "  }",
                "})();",
                "</script>",
            ]
        )
        text = str(full_html or "")
        if "</body>" in text:
            return text.replace("</body>", injected + "\n</body>", 1)
        return text + injected

    with tempfile.TemporaryDirectory(prefix="nsys_timeline_compare_") as tmpdir:
        for idx, sqlite_path in enumerate(items):
            label = Path(str(sqlite_path)).name or f"sqlite_{idx}"
            compare_debug(f"compare child start index={idx} sqlite={sqlite_path}")

            def _child_progress(msg: str, *, _idx: int = idx, _label: str = label) -> None:
                if progress_cb:
                    progress_cb(f"[{_idx + 1}/{total}] {_label} {str(msg or '').strip()}")

            def _child_debug(msg: str, *, _idx: int = idx, _label: str = label) -> None:
                compare_debug(f"[{_idx + 1}/{total}] {_label} {msg}")

            tmp_out = Path(tmpdir) / f"timeline_{idx:03d}.html"
            export_timeline_html(
                str(sqlite_path),
                output_path=str(tmp_out),
                device_id=int(device_id),
                start_ns=int(start_ns),
                end_ns=int(end_ns),
                limit=int(limit),
                width_px=int(width_px),
                nvtx_text=str(nvtx_text or ""),
                nvtx_index=int(nvtx_index),
                include_metrics=bool(include_metrics),
                metric_name_like=str(metric_name_like or "%"),
                metrics_limit=int(metrics_limit),
                metrics_max_points=int(metrics_max_points),
                overlay_metrics_per_track=int(overlay_metrics_per_track),
                default_focus_metrics=bool(default_focus_metrics),
                include_all_metric_sources=bool(include_all_metric_sources),
                debug=bool(debug),
                debug_rows=int(debug_rows),
                debug_log_fn=_child_debug,
                progress_cb=_child_progress,
            )
            rendered.append(
                {
                    "label": label,
                    "sqlite_path": str(sqlite_path),
                    "srcdoc": tmp_out.read_text(encoding="utf-8"),
                }
            )
            compare_debug(f"compare child done index={idx} sqlite={sqlite_path}")

    section_titles: List[str] = [
        "All Streams Overlap + Metrics Alignment",
        "Matched NVTX Scopes",
        "Kernel Timeline By Stream",
    ]
    if bool(include_metrics):
        section_titles.append("GPU Metrics In Window")

    section_blocks: List[str] = []
    for section_title in section_titles:
        section_cards: List[str] = []
        for idx, item in enumerate(rendered):
            section_srcdoc = _build_section_srcdoc(str(item["srcdoc"]), panel_title=str(section_title))
            if not section_srcdoc:
                continue
            srcdoc_attr = html.escape(str(section_srcdoc), quote=True)
            sqlite_display = html.escape(str(item["sqlite_path"]))
            label_display = html.escape(str(item["label"]))
            section_cards.extend(
                [
                    "<section class='compare-card'>",
                    "<div class='compare-head'>",
                    f"<div class='compare-index'>#{idx + 1}</div>",
                    "<div class='compare-meta'>",
                    f"<div class='compare-title'>{label_display}</div>",
                    f"<div class='compare-path'>{sqlite_display}</div>",
                    "</div>",
                    "</div>",
                    (
                        "<iframe class='compare-frame' loading='lazy' "
                        "sandbox='allow-scripts allow-same-origin' "
                        f"srcdoc=\"{srcdoc_attr}\"></iframe>"
                    ),
                    "</section>",
                ]
            )
        if not section_cards:
            continue
        section_blocks.extend(
            [
                "<section class='compare-section'>",
                f"<h3 class='compare-section-title'>{html.escape(str(section_title))}</h3>",
                "<div class='compare-section-stack'>",
                *section_cards,
                "</div>",
                "</section>",
            ]
        )

    note = (
        "Each compare section groups the same timeline panel across all sqlite files. "
        "Each embedded timeline still keeps its own local matched window and its own kernel/metrics X-axis."
    )
    if str(nvtx_text or "").strip():
        note += f" Shared NVTX filter: {str(nvtx_text)}."
    page = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'/>",
            "<title>NSYS NVTX Timeline Compare</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:20px;background:#0f1116;color:#e7e9ee;}",
            ".meta{margin-bottom:14px;color:#a8afbf;font-size:13px;}",
            ".compare-root{display:flex;flex-direction:column;gap:18px;align-items:center;}",
            ".compare-section{width:min(100%, 1900px);}",
            ".compare-section-title{font-size:15px;color:#dbe6ff;margin:0 0 10px 0;}",
            ".compare-section-stack{display:flex;flex-direction:column;gap:14px;align-items:stretch;}",
            ".compare-card{width:min(100%, 1900px);background:#171c28;border:1px solid #2a3243;border-radius:8px;padding:12px;box-sizing:border-box;}",
            ".compare-head{display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;}",
            ".compare-index{font-size:12px;color:#9bb1d6;background:#121826;border:1px solid #34415a;border-radius:999px;padding:2px 8px;line-height:18px;}",
            ".compare-title{font-size:14px;color:#e7eefc;font-weight:600;word-break:break-word;}",
            ".compare-path{font-size:11px;color:#8fa1bf;word-break:break-all;margin-top:2px;}",
            ".compare-frame{display:block;width:100%;min-height:140px;border:1px solid #2f3b53;border-radius:6px;background:#0c111c;box-sizing:border-box;}",
            "</style></head><body>",
            "<h2>NSYS NVTX Timeline Compare</h2>",
            (
                f"<div class='meta'>sqlite_count={len(rendered)} | include_metrics={int(bool(include_metrics))} | "
                f"device_id={int(device_id)} | note={html.escape(note)}</div>"
            ),
            "<div class='compare-root'>",
            *section_blocks,
            "</div>",
            "<script>",
            "(function(){",
            "  const frames = Array.from(document.querySelectorAll('.compare-frame'));",
            "  const resizeFrame = (frame) => {",
            "    try {",
            "      const doc = frame.contentDocument;",
            "      if (!doc || !doc.documentElement) return;",
            "      const body = doc.body;",
            "      const h = Math.max(",
            "        body ? body.scrollHeight : 0,",
            "        doc.documentElement.scrollHeight || 0,",
            "        140",
            "      );",
            "      frame.style.height = `${Math.min(Math.max(h + 12, 140), 6000)}px`;",
            "    } catch (_) {}",
            "  };",
            "  for (const frame of frames) {",
            "    frame.addEventListener('load', () => {",
            "      resizeFrame(frame);",
            "      setTimeout(() => resizeFrame(frame), 100);",
            "      setTimeout(() => resizeFrame(frame), 500);",
            "      setTimeout(() => resizeFrame(frame), 1500);",
            "    });",
            "  }",
            "})();",
            "</script>",
            "</body></html>",
        ]
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    compare_debug("compare html wrote output={} sqlite_count={}".format(str(out), len(rendered)))
    return str(out)
