from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

from .nsys_schema_adapter import NsightSchema


_STEP_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\bstep(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE),
    re.compile(r"\biter(?:ation)?(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE),
)
_RANK_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\brank(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE),
)


def _is_nccl_kernel(name: str) -> bool:
    return "nccl" in str(name or "").lower()


def _extract_step_rank(text: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for pattern in _STEP_PATTERNS:
        match = pattern.search(text or "")
        if match:
            tags["step"] = str(match.group(1))
            break
    for pattern in _RANK_PATTERNS:
        match = pattern.search(text or "")
        if match:
            tags["rank"] = str(match.group(1))
            break
    return tags


def _kernel_name_expr(schema: NsightSchema, kernel_alias: str = "k") -> Tuple[str, str]:
    kernel_table = str(schema.kernel_table or "")
    if not kernel_table:
        return ("''", "")
    short_col = schema.resolve_column(kernel_table, ("shortName",))
    demangled_col = schema.resolve_column(kernel_table, ("demangledName",))
    if schema.string_table is None:
        if demangled_col:
            return (f"CAST({kernel_alias}.{demangled_col} AS TEXT)", "")
        if short_col:
            return (f"CAST({kernel_alias}.{short_col} AS TEXT)", "")
        return ("''", "")

    joins = ""
    name_expr = "''"
    sid_table = schema.string_table
    if short_col:
        joins += f" LEFT JOIN {sid_table} s_short ON {kernel_alias}.{short_col} = s_short.id "
        name_expr = "s_short.value"
    if demangled_col:
        joins += f" LEFT JOIN {sid_table} s_dem ON {kernel_alias}.{demangled_col} = s_dem.id "
        name_expr = "COALESCE(s_dem.value, s_short.value)" if short_col else "s_dem.value"
    return name_expr, joins


def detect_iterations(
    conn: sqlite3.Connection,
    *,
    schema: Optional[NsightSchema] = None,
    marker: str = "sample_0",
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    top_level_only: bool = True,
    limit: int = 2000,
) -> List[Dict[str, object]]:
    resolved_schema = schema or NsightSchema(conn)
    kernel_table = resolved_schema.kernel_table
    runtime_table = resolved_schema.runtime_table
    nvtx_table = resolved_schema.nvtx_table
    if not (kernel_table and runtime_table and nvtx_table):
        return []

    nvtx_cols = set(resolved_schema.columns(nvtx_table))
    runtime_cols = set(resolved_schema.columns(runtime_table))
    kernel_cols = set(resolved_schema.columns(kernel_table))
    required_nvtx = {"start", "end"}
    required_runtime = {"start", "end", "correlationId"}
    required_kernel = {"start", "end", "correlationId"}
    if not (required_nvtx.issubset(nvtx_cols) and required_runtime.issubset(runtime_cols) and required_kernel.issubset(kernel_cols)):
        return []

    text_expr = "n.text"
    nvtx_join = ""
    if resolved_schema.string_table and "textId" in nvtx_cols:
        nvtx_join = f" LEFT JOIN {resolved_schema.string_table} s_nvtx ON n.textId = s_nvtx.id "
        text_expr = "COALESCE(n.text, s_nvtx.value)"
    elif "nameId" in nvtx_cols and resolved_schema.string_table:
        nvtx_join = f" LEFT JOIN {resolved_schema.string_table} s_nvtx ON n.nameId = s_nvtx.id "
        text_expr = "COALESCE(n.text, s_nvtx.value)"

    kernel_name_expr, kernel_name_join = _kernel_name_expr(resolved_schema, "k")

    like_text = str(marker or "").strip()
    if not like_text:
        like_text = "sample_0"
    if "%" not in like_text:
        like_text = f"%{like_text}%"

    where_parts = [
        f"{text_expr} IS NOT NULL",
        "n.[end] > n.start",
        f"{text_expr} LIKE :marker",
    ]
    params: Dict[str, object] = {
        "marker": like_text,
        "device_id": int(device_id),
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "limit": int(limit),
    }
    if start_ns >= 0:
        where_parts.append("n.start >= :start_ns")
    if end_ns >= 0:
        where_parts.append("n.[end] <= :end_ns")

    query_nvtx = (
        f"SELECT {text_expr} AS nvtx_text, n.start AS start_ns, n.[end] AS end_ns, "
        + ("n.globalTid AS global_tid, " if "globalTid" in nvtx_cols else "NULL AS global_tid, ")
        + "n.rowid AS _rowid "
        + f"FROM {nvtx_table} n "
        + nvtx_join
        + " WHERE "
        + " AND ".join(where_parts)
        + " ORDER BY n.start ASC LIMIT :limit"
    )
    candidates = [dict(row) for row in conn.execute(query_nvtx, params).fetchall()]
    if not candidates:
        return []

    if top_level_only:
        filtered: List[Dict[str, object]] = []
        last_end = -1
        for row in candidates:
            s = int(row.get("start_ns") or 0)
            e = int(row.get("end_ns") or 0)
            if e <= s:
                continue
            if s >= last_end:
                filtered.append(row)
                last_end = e
        candidates = filtered

    kernel_where = ["k.[end] > k.start"]
    if device_id >= 0 and "deviceId" in kernel_cols:
        kernel_where.append("k.deviceId = :device_id")
    if start_ns >= 0:
        kernel_where.append("k.start >= :global_start_ns")
        params["global_start_ns"] = int(start_ns)
    if end_ns >= 0:
        kernel_where.append("k.[end] <= :global_end_ns")
        params["global_end_ns"] = int(end_ns)

    query_kernels = (
        "SELECT k.start AS start_ns, k.[end] AS end_ns, "
        + ("k.streamId AS stream_id, " if "streamId" in kernel_cols else "NULL AS stream_id, ")
        + f"{kernel_name_expr} AS kernel_name "
        + f"FROM {kernel_table} k "
        + kernel_name_join
        + " WHERE "
        + " AND ".join(kernel_where)
        + " ORDER BY k.start ASC"
    )
    kernel_rows = [dict(row) for row in conn.execute(query_kernels, params).fetchall()]

    # Runtime rows are optional for narrowing kernels to the NVTX owner thread.
    runtime_rows: List[Dict[str, object]] = []
    if "globalTid" in runtime_cols:
        runtime_where = ["r.[end] > r.start"]
        if start_ns >= 0:
            runtime_where.append("r.start >= :global_start_ns")
        if end_ns >= 0:
            runtime_where.append("r.[end] <= :global_end_ns")
        query_runtime = (
            "SELECT r.start AS start_ns, r.[end] AS end_ns, r.correlationId AS correlation_id, r.globalTid AS global_tid "
            + f"FROM {runtime_table} r WHERE "
            + " AND ".join(runtime_where)
            + " ORDER BY r.start ASC"
        )
        runtime_rows = [dict(row) for row in conn.execute(query_runtime, params).fetchall()]

    runtime_by_tid: Dict[int, List[Dict[str, object]]] = {}
    corr_to_kernel: Dict[int, List[Dict[str, object]]] = {}
    if runtime_rows:
        for item in runtime_rows:
            tid = int(item.get("global_tid") or 0)
            runtime_by_tid.setdefault(tid, []).append(item)
        # Build map with a cheap window match on correlation id.
        if "correlationId" in kernel_cols:
            query_k_corr = (
                "SELECT k.correlationId AS correlation_id, k.start AS start_ns, k.[end] AS end_ns, "
                + ("k.streamId AS stream_id, " if "streamId" in kernel_cols else "NULL AS stream_id, ")
                + f"{kernel_name_expr} AS kernel_name "
                + f"FROM {kernel_table} k "
                + kernel_name_join
                + " WHERE "
                + " AND ".join(kernel_where)
                + " ORDER BY k.start ASC"
            )
            for row in conn.execute(query_k_corr, params).fetchall():
                corr = int(row["correlation_id"])
                corr_to_kernel.setdefault(corr, []).append(dict(row))

    results: List[Dict[str, object]] = []
    for idx, row in enumerate(candidates):
        iter_start = int(row.get("start_ns") or 0)
        iter_end = int(row.get("end_ns") or 0)
        if iter_end <= iter_start:
            continue

        chosen_kernels: List[Dict[str, object]] = []
        global_tid = row.get("global_tid")
        if runtime_by_tid and corr_to_kernel and global_tid is not None:
            tid = int(global_tid)
            rt_rows = runtime_by_tid.get(tid, [])
            for rt in rt_rows:
                rs = int(rt["start_ns"])
                re = int(rt["end_ns"])
                if rs < iter_start or re > iter_end:
                    continue
                corr = int(rt["correlation_id"])
                chosen_kernels.extend(corr_to_kernel.get(corr, []))
        else:
            # Fallback: pure time-window assignment.
            for kernel in kernel_rows:
                ks = int(kernel["start_ns"])
                ke = int(kernel["end_ns"])
                if ks >= iter_start and ke <= iter_end:
                    chosen_kernels.append(kernel)

        kernel_count = 0
        nccl_count = 0
        compute_ns = 0
        comm_ns = 0
        gpu_min_start: Optional[int] = None
        gpu_max_end: Optional[int] = None
        for kernel in chosen_kernels:
            ks = int(kernel["start_ns"])
            ke = int(kernel["end_ns"])
            if ke <= ks:
                continue
            kernel_count += 1
            dur = int(ke - ks)
            if _is_nccl_kernel(str(kernel.get("kernel_name") or "")):
                nccl_count += 1
                comm_ns += dur
            else:
                compute_ns += dur
            gpu_min_start = ks if gpu_min_start is None else min(gpu_min_start, ks)
            gpu_max_end = ke if gpu_max_end is None else max(gpu_max_end, ke)

        text = str(row.get("nvtx_text") or "")
        item = {
            "iteration": int(idx),
            "marker_text": text,
            "start_ns": iter_start,
            "end_ns": iter_end,
            "duration_ms": round((iter_end - iter_start) / 1e6, 6),
            "kernel_count": int(kernel_count),
            "nccl_kernel_count": int(nccl_count),
            "compute_ms": round(compute_ns / 1e6, 6),
            "comm_ms": round(comm_ns / 1e6, 6),
            "gpu_start_ns": int(gpu_min_start or iter_start),
            "gpu_end_ns": int(gpu_max_end or iter_end),
            "gpu_duration_ms": round(((gpu_max_end or iter_end) - (gpu_min_start or iter_start)) / 1e6, 6),
        }
        item.update(_extract_step_rank(text))
        results.append(item)
    return results
