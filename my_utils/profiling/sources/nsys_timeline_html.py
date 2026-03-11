from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .nsys_flat_export import collect_kernel_rows
from .nsys_schema_adapter import NsightSchema
from .nsys_sqlite_provider import NsysSqliteMetricsProvider

_RANK_RE = re.compile(r"\brank(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE)
_GPU_IDX_RE = re.compile(r"(?:gpu|device)\s*[_:#=\- ]?\s*(\d+)", re.IGNORECASE)


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


def _parse_rank_from_text(text: object) -> Optional[int]:
    s = str(text or "")
    m = _RANK_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
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
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    selected_pairs = set()
    if nvtx_windows:
        for item in nvtx_windows:
            selected_pairs.add((_to_int(item.get("start_ns"), -1), _to_int(item.get("end_ns"), -1)))

    # Prefer launch-attribution view (NVTX -> Runtime -> correlationId -> Kernel).
    if "nvtx_kernel_sm_detail" in set(provider.list_sql_skills()):
        detailed = provider.run_sql_skill(
            "nvtx_kernel_sm_detail",
            nvtx_text=str(nvtx_text or "%"),
            device_id=int(device_id),
            limit=int(limit),
        )
        seen = set()
        for row in detailed:
            ns = _to_int(row.get("nvtx_start_ns"), -1)
            ne = _to_int(row.get("nvtx_end_ns"), -1)
            if selected_pairs:
                if (ns, ne) not in selected_pairs:
                    continue
            else:
                if ns < int(start_ns) or ne > int(end_ns):
                    continue
            ks = _to_int(row.get("kernel_start_ns"), -1)
            ke = _to_int(row.get("kernel_end_ns"), -1)
            if ks < 0 or ke <= ks:
                continue
            uniq = (ks, ke, _to_int(row.get("stream_id"), 0), str(row.get("kernel_name") or ""))
            if uniq in seen:
                continue
            seen.add(uniq)
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
                    "occupancy_pct_estimate": row.get("occupancy_pct_estimate"),
                    "nvtx_text": str(row.get("nvtx_text") or ""),
                    "nvtx_start_ns": ns,
                    "nvtx_end_ns": ne,
                    "rank": _parse_rank_from_text(row.get("nvtx_text")),
                }
            )
        if rows:
            return rows

    # Fallback: plain kernel rows in time window.
    fallback = collect_kernel_rows(
        provider,
        device_id=int(device_id),
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        limit=int(limit),
        attach_iteration=False,
    )
    for row in fallback:
        rows.append(
            {
                "stream_id": _to_int(row.get("stream_id"), 0),
                "device_id": _to_int(row.get("device_id"), int(device_id)),
                "kernel_name": str(row.get("kernel_name") or ""),
                "start_ns": _to_int(row.get("start_ns"), 0),
                "end_ns": _to_int(row.get("end_ns"), 0),
                "duration_ms": float(row.get("duration_ms") or 0.0),
                "kind": "comm" if bool(row.get("is_nccl")) else "compute",
                "registers_per_thread": None,
                "threads_per_block": None,
                "static_shared_bytes": None,
                "dynamic_shared_bytes": None,
                "total_shared_bytes": None,
                "occupancy_pct_estimate": None,
                "nvtx_text": None,
                "nvtx_start_ns": None,
                "nvtx_end_ns": None,
                "rank": None,
            }
        )
    return rows


def _downsample_points(points: Sequence[Tuple[int, float]], max_points: int = 2000) -> List[Tuple[int, float]]:
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


def _collect_metric_samples(
    sqlite_path: str,
    *,
    start_ns: int,
    end_ns: int,
    metric_name_like: str = "%",
    include_all_sources: bool = False,
    device_id: int = -1,
    limit: int = 300000,
) -> List[Dict[str, object]]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        schema = NsightSchema(conn)
        if not schema.metrics_table:
            return []
        metrics_table = _ident(schema.metrics_table)
        ts_col = schema.metrics_timestamp_col or schema.resolve_column(metrics_table, ("timestamp", "start", "time"))
        id_col = schema.metrics_id_col or schema.resolve_column(metrics_table, ("metricId", "nameId", "eventId"))
        val_col = schema.metrics_value_col or schema.resolve_column(metrics_table, ("value", "metricValue", "val"))
        if not ts_col or not id_col or not val_col:
            return []

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

        if not name_expr and schema.string_table:
            string_table = _ident(schema.string_table)
            joins.append(f"JOIN {string_table} s ON g.{_ident(id_col)} = s.id")
            name_expr = "s.value"
            name_not_null = "AND s.value IS NOT NULL "

        if not name_expr:
            name_expr = f"CAST(g.{_ident(id_col)} AS TEXT)"
            name_not_null = ""

        source_join = ""
        source_where = ""
        source_name_expr = "NULL"
        source_col = schema.resolve_column(metrics_table, ("sourceId", "source_id"))
        if source_col and schema.table_exists("GENERIC_EVENT_SOURCES"):
            ges_tbl = _ident("GENERIC_EVENT_SOURCES")
            ges_id_col = schema.resolve_column(ges_tbl, ("sourceId", "id", "source_id"))
            ges_name_col = schema.resolve_column(ges_tbl, ("name", "source", "sourceName"))
            if ges_id_col and ges_name_col:
                source_join = (
                    f"LEFT JOIN {ges_tbl} gs ON g.{_ident(source_col)} = gs.{_ident(ges_id_col)}"
                )
                source_name_expr = f"gs.{_ident(ges_name_col)}"
                source_where = (
                    "AND (? = 1 "
                    f"OR gs.{_ident(ges_name_col)} IS NULL "
                    f"OR LOWER(gs.{_ident(ges_name_col)}) LIKE '%gpu%metric%' "
                    f"OR LOWER(gs.{_ident(ges_name_col)}) = 'gpumetrics') "
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

        base_where = (
            "WHERE g.{ts} >= ? AND g.{ts} <= ? ".format(ts=_ident(ts_col))
            + "AND (? = '%' OR {name_expr} LIKE ?) ".format(name_expr=name_expr)
            + name_not_null
            + device_where
            + source_where
            + f"AND g.{_ident(val_col)} IS NOT NULL "
        )
        select_expr = (
            f"SELECT g.{_ident(ts_col)} AS ts_ns, "
            f"{name_expr} AS metric_name, "
            f"{device_expr} AS metric_device_id, "
            f"{source_name_expr} AS metric_source_name, "
            f"CAST(g.{_ident(val_col)} AS REAL) AS metric_value "
        )
        from_join_expr = (
            f"FROM {metrics_table} g "
            + " ".join(joins)
            + " "
            + source_join
            + " "
        )

        limit_i = int(limit)
        tmp_tbl = "_myu_metrics_base"
        conn.execute(f"DROP TABLE IF EXISTS {tmp_tbl}")
        create_tmp_sql = (
            f"CREATE TEMP TABLE {tmp_tbl} AS "
            + select_expr
            + from_join_expr
            + base_where
        )
        conn.execute(create_tmp_sql, filter_params)
        total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {tmp_tbl}").fetchone()[0] or 0)

        if total_rows <= 0:
            rows = []
        elif limit_i <= 0 or total_rows <= limit_i:
            rows = conn.execute(
                f"SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value "
                f"FROM {tmp_tbl} ORDER BY ts_ns ASC"
            ).fetchall()
        else:
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
            rows = sorted(collected, key=lambda r: (_to_int(r["ts_ns"], 0), str(r["metric_name"] or "")))
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

    series: List[Dict[str, object]] = []
    for name, tag_kind, tag_value in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2])):
        points = _downsample_points(grouped[(name, tag_kind, tag_value)], max_points=2000)
        display_name = _metric_series_name(str(name), str(tag_kind), str(tag_value))
        series.append(
            {
                "name": display_name,
                "color": _color_for_name(display_name),
                "points": [[int(t), float(v)] for t, v in points],
            }
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

    payload = {
        "window_start_ns": int(window_start_ns),
        "window_end_ns": int(window_end_ns),
        "span_ns": int(span),
        "metrics": metric_series if include_metrics else [],
        "chart_width": int(width_px),
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
                lines.append("<div class='row'>")
                lines.append(f"<div class='label'>stream {sid}</div>")
                lines.append("<div class='track'>")
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
                        f"smem={row.get('total_shared_bytes')} | occ={row.get('occupancy_pct_estimate')}"
                    )
                    lines.append(
                        f"<div class='bar' style='left:{left_px}px;width:{width_bar}px;background:{color};' "
                        f"title='{html.escape(title)}'></div>"
                    )
                lines.append("</div></div>")
    lines.extend(["</div>"])

    lines.extend(
        [
            "<script>",
            f"const TIMELINE_DATA = {payload_json};",
            "(function(){",
            "  const d = TIMELINE_DATA;",
            "  const grid = document.getElementById('metrics-grid');",
            "  if (!grid) return;",
            "  if (!Array.isArray(d.metrics) || d.metrics.length === 0) {",
            "    grid.innerHTML = \"<div class='empty'>No GPU metric samples in this window.</div>\";",
            "    return;",
            "  }",
            "  const W = Math.max(480, Math.min(920, Number(d.chart_width || 1200)));",
            "  const H = 190;",
            "  const padL = 52, padR = 16, padT = 12, padB = 26;",
            "  const x0 = padL, y0 = padT, cw = W - padL - padR, ch = H - padT - padB;",
            "  const span = Math.max(1, Number(d.span_ns || 1));",
            "  const startNs = Number(d.window_start_ns || 0);",
            "  const x = (t) => x0 + ((Number(t) - startNs) / span) * cw;",
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
            "  const series = [...d.metrics].sort((a,b) => String(a.name||'').localeCompare(String(b.name||'')));",
            "  for (const s of series) {",
            "    const ptsRaw = Array.isArray(s.points) ? s.points : [];",
            "    if (!ptsRaw.length) continue;",
            "    let minV = Number.POSITIVE_INFINITY;",
            "    let maxV = Number.NEGATIVE_INFINITY;",
            "    for (const p of ptsRaw) {",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "    }",
            "    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) { minV = 0; maxV = 1; }",
            "    if (Math.abs(maxV - minV) < 1e-12) { maxV = minV + 1.0; }",
            "    const y = (v) => y0 + (1.0 - ((Number(v) - minV) / (maxV - minV))) * ch;",
            "    const card = document.createElement('div'); card.className = 'metric-panel';",
            "    const title = document.createElement('div'); title.className = 'metric-title'; title.textContent = String(s.name || 'metric');",
            "    const svg = mk('svg', {width:W, height:H, viewBox:`0 0 ${W} ${H}`});",
            "    svg.appendChild(mk('rect', {x:x0, y:y0, width:cw, height:ch, fill:'#111725', stroke:'#32405b'}));",
            "    svg.appendChild(mk('line', {x1:x0, y1:y0+ch, x2:x0+cw, y2:y0+ch, stroke:'#55637f'}));",
            "    svg.appendChild(mk('line', {x1:x0, y1:y0, x2:x0, y2:y0+ch, stroke:'#55637f'}));",
            "    const yMinText = mk('text', {x:6, y:y0+ch, fill:'#9aa7be', 'font-size':11}); yMinText.textContent = fmt(minV); svg.appendChild(yMinText);",
            "    const yMaxText = mk('text', {x:6, y:y0+10, fill:'#9aa7be', 'font-size':11}); yMaxText.textContent = fmt(maxV); svg.appendChild(yMaxText);",
            "    const xStartText = mk('text', {x:x0, y:y0+ch+16, fill:'#7f8daa', 'font-size':10}); xStartText.textContent = String(Math.round(startNs)); svg.appendChild(xStartText);",
            "    const xEndText = mk('text', {x:x0+cw-92, y:y0+ch+16, fill:'#7f8daa', 'font-size':10}); xEndText.textContent = String(Math.round(startNs + span)); svg.appendChild(xEndText);",
            "    const pts = ptsRaw.map((p) => `${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ');",
            "    if (pts) {",
            "      svg.appendChild(mk('polyline', {points:pts, fill:'none', stroke:s.color || '#88c', 'stroke-width':1.4, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "    }",
            "    const sub = document.createElement('div'); sub.className = 'metric-sub';",
            "    sub.textContent = `samples=${ptsRaw.length} | min=${fmt(minV)} | max=${fmt(maxV)}`;",
            "    card.appendChild(title);",
            "    card.appendChild(svg);",
            "    card.appendChild(sub);",
            "    grid.appendChild(card);",
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
    metrics_limit: int = 300000,
    include_all_metric_sources: bool = False,
) -> str:
    provider = NsysSqliteMetricsProvider(sqlite_path)

    selected_nvtx_windows: List[Dict[str, object]] = []
    effective_start_ns = int(start_ns)
    effective_end_ns = int(end_ns)
    if str(nvtx_text or "").strip():
        matched_nvtx = _select_nvtx_windows(
            provider,
            nvtx_text=str(nvtx_text),
        )
        selected_nvtx_windows = _pick_nvtx_windows(matched_nvtx, nvtx_index=int(nvtx_index))
        if selected_nvtx_windows:
            effective_start_ns = min(_to_int(item.get("start_ns"), -1) for item in selected_nvtx_windows)
            effective_end_ns = max(_to_int(item.get("end_ns"), -1) for item in selected_nvtx_windows)

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
        if not base_rows:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("<html><body><h2>No kernels</h2></body></html>", encoding="utf-8")
            return str(out)
        effective_start_ns = min(int(item["start_ns"]) for item in base_rows)
        effective_end_ns = max(int(item["end_ns"]) for item in base_rows)

    kernels = _collect_kernels_in_window(
        provider,
        start_ns=int(effective_start_ns),
        end_ns=int(effective_end_ns),
        nvtx_text=str(nvtx_text or "%"),
        nvtx_windows=selected_nvtx_windows or None,
        device_id=int(device_id),
        limit=int(limit),
    )

    metric_series: List[Dict[str, object]] = []
    if bool(include_metrics):
        metric_series = _collect_metric_samples(
            sqlite_path,
            start_ns=int(effective_start_ns),
            end_ns=int(effective_end_ns),
            metric_name_like=str(metric_name_like or "%"),
            include_all_sources=bool(include_all_metric_sources),
            device_id=int(device_id),
            limit=int(metrics_limit),
        )

    text = _render_html(
        sqlite_path=str(sqlite_path),
        kernels=kernels,
        metric_series=metric_series,
        window_start_ns=int(effective_start_ns),
        window_end_ns=int(effective_end_ns),
        nvtx_windows=selected_nvtx_windows,
        width_px=int(width_px),
        include_metrics=bool(include_metrics),
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return str(out)
