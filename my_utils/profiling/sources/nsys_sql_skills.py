from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .nsys_iterations import detect_iterations
from .nsys_schema_adapter import NsightSchema


def _ident(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("empty SQL identifier")
    for ch in text:
        if not (ch.isalnum() or ch == "_"):
            raise ValueError(f"unsafe SQL identifier: {name}")
    return text


@dataclass
class SqlSkillParam:
    name: str
    description: str
    type: str = "str"
    required: bool = False
    default: object = None


@dataclass
class SqlSkill:
    name: str
    title: str
    description: str
    category: str
    sql: str
    params: List[SqlSkillParam] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    format_fn: Optional[Callable[[List[Dict[str, object]]], str]] = None

    def _resolve_params(self, kwargs: Dict[str, object]) -> Dict[str, object]:
        resolved: Dict[str, object] = {}
        for param in self.params:
            if param.name in kwargs:
                value = kwargs[param.name]
            elif param.default is not None:
                value = param.default
            elif param.required:
                raise ValueError(f"skill '{self.name}' missing parameter '{param.name}'")
            else:
                continue

            ptype = str(param.type or "str").lower()
            if ptype == "int":
                value = int(value)
            elif ptype == "float":
                value = float(value)
            elif ptype == "bool":
                value = 1 if bool(value) else 0
            else:
                value = str(value).replace("'", "''")
            resolved[param.name] = value
        return resolved

    def execute(self, conn: sqlite3.Connection, **kwargs) -> List[Dict[str, object]]:
        resolved = self._resolve_params(dict(kwargs))
        sql = self.sql.format(**resolved) if resolved else self.sql
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in (cursor.description or [])]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def run(self, conn: sqlite3.Connection, **kwargs) -> object:
        rows = self.execute(conn, **kwargs)
        if self.format_fn is not None:
            return self.format_fn(rows)
        return rows


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    valid = [(int(s), int(e)) for s, e in intervals if e > s]
    if not valid:
        return []
    valid.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int]] = [valid[0]]
    for start, end in valid[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _covered_ns(intervals: Sequence[Tuple[int, int]]) -> int:
    return int(sum((end - start) for start, end in intervals))


def _intersection_coverage_ns(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> int:
    if not a or not b:
        return 0
    i = 0
    j = 0
    total = 0
    while i < len(a) and j < len(b):
        a_start, a_end = a[i]
        b_start, b_end = b[j]
        left = max(a_start, b_start)
        right = min(a_end, b_end)
        if right > left:
            total += int(right - left)
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return int(total)


def _is_nccl_kernel(name: str) -> bool:
    text = str(name or "").lower()
    if not text:
        return False
    return "nccl" in text


def calculate_h100_occupancy(
    threads_per_block: object,
    registers_per_thread: object,
    shared_bytes: object,
) -> Optional[float]:
    """
    Estimate theoretical occupancy for NVIDIA H100 (sm_90) strictly matching
    the official NVIDIA CUDA Occupancy Calculator allocation rules.
    """
    try:
        tpb = int(threads_per_block)
    except Exception:
        return None

    # CUDA hard limits: at least 1 thread, max 1024 threads per block
    if tpb <= 0 or tpb > 1024:
        return None

    # Warps per block (warp size is always 32)
    warps_per_block = int(math.ceil(tpb / 32.0))

    # --- Limit 1: Threads / Warps Capacity ---
    # H100 supports max 64 warps per SM
    blocks_limit_threads = int(math.floor(64 / warps_per_block))

    # --- Limit 2: Register Capacity ---
    regs = None
    try:
        if registers_per_thread is not None:
            regs = int(registers_per_thread)
    except Exception:
        pass

    if regs is None or regs <= 0:
        blocks_limit_regs = 32  # Not bottlenecked by registers
    elif regs > 255:
        return 0.0  # Launch failure: exceeds Hopper absolute max registers per thread
    else:
        # A. Register allocation granularity is 256 per warp
        regs_per_warp = int(math.ceil((regs * 32) / 256.0) * 256)

        # B. Calculate max total warps the SM's 64K register file can physically hold
        max_warps_by_regs = math.floor(65536 / regs_per_warp)

        # C. CRITICAL: Warp allocation granularity for sm_90 is 4.
        # The hardware warp schedulers allocate warps in physical chunks of 4.
        max_warps_by_regs_rounded = int(math.floor(max_warps_by_regs / 4.0) * 4)

        # D. Translate the available physical warp slots into max active blocks
        blocks_limit_regs = int(math.floor(max_warps_by_regs_rounded / warps_per_block))

    # --- Limit 3: Shared Memory Capacity ---
    smem = None
    try:
        if shared_bytes is not None:
            smem = int(shared_bytes)
    except Exception:
        pass

    if smem is None or smem <= 0:
        blocks_limit_smem = 32  # Not bottlenecked by shared memory
    else:
        # Hopper Hard Limit: A single block CANNOT exceed 227 KB (232,448 bytes)
        # Even though the SM has 228 KB, launching a 228KB block will fail.
        if smem > 232448:
            return 0.0

        # Hopper SMEM allocation granularity is 128 bytes
        smem_per_block = int(math.ceil(smem / 128.0) * 128)

        # Hopper max shared memory per SM is 228 KB (233,472 bytes)
        blocks_limit_smem = int(math.floor(233472 / smem_per_block))

    # --- Final calculation ---
    # Hopper max blocks per SM is 32. Actual active blocks is the bottleneck of all limits.
    active_blocks = min(blocks_limit_threads, blocks_limit_regs, blocks_limit_smem, 32)

    if active_blocks <= 0:
        return 0.0

    # Theoretical Occupancy = Active Warps / Max Supported Warps (64)
    occupancy_pct = (active_blocks * warps_per_block) / 64.0 * 100.0

    return round(float(occupancy_pct), 1)


def _build_builtin_skills(schema: NsightSchema) -> Dict[str, SqlSkill]:
    kernel_table = _ident(schema.kernel_table or "CUPTI_ACTIVITY_KIND_KERNEL")
    runtime_table = _ident(schema.runtime_table or "CUPTI_ACTIVITY_KIND_RUNTIME")
    nvtx_table = _ident(schema.nvtx_table or "NVTX_EVENTS")
    memcpy_table = _ident(schema.memcpy_table or "CUPTI_ACTIVITY_KIND_MEMCPY")

    string_table = schema.string_table
    short_col = schema.resolve_column(kernel_table, ("shortName",))
    demangled_col = schema.resolve_column(kernel_table, ("demangledName",))
    corr_col = schema.resolve_column(kernel_table, ("correlationId",))
    device_col = schema.resolve_column(kernel_table, ("deviceId",))
    stream_col = schema.resolve_column(kernel_table, ("streamId",))
    start_col = schema.resolve_column(kernel_table, ("start",))
    end_col = schema.resolve_column(kernel_table, ("end",))

    if not all([start_col, end_col]):
        return {}

    name_expr = "CAST(k.shortName AS TEXT)"
    name_join = ""
    if string_table and short_col:
        string_table_ident = _ident(string_table)
        name_join = f" JOIN {string_table_ident} s ON k.{_ident(short_col)} = s.id "
        name_expr = "s.value"
    if string_table and demangled_col:
        string_table_ident = _ident(string_table)
        name_join += f" LEFT JOIN {string_table_ident} d ON k.{_ident(demangled_col)} = d.id "
        if short_col:
            name_expr = "COALESCE(d.value, s.value)"
        else:
            name_expr = "d.value"

    kernel_where_device = ""
    if device_col:
        kernel_where_device = f" AND ({{device_id}} < 0 OR k.{_ident(device_col)} = {{device_id}})"

    skill_map: Dict[str, SqlSkill] = {}

    # 1) Aggregate kernels
    skill_map["aggregate_kernels"] = SqlSkill(
        name="aggregate_kernels",
        title="Aggregate Kernels",
        description="Group kernel executions by demangled/short name and report latency stats.",
        category="kernels",
        sql=(
            f"SELECT {name_expr} AS kernel_name, "
            f"COUNT(*) AS invocations, "
            f"ROUND(SUM(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS total_ms, "
            f"ROUND(AVG(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS avg_ms, "
            f"ROUND(MIN(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS min_ms, "
            f"ROUND(MAX(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS max_ms "
            f"FROM {kernel_table} k "
            f"{name_join} "
            f"WHERE 1=1 "
            f"{kernel_where_device} "
            f"GROUP BY {name_expr} "
            f"ORDER BY total_ms DESC "
            f"LIMIT {{limit}}"
        ),
        params=[
            SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
            SqlSkillParam("limit", "max rows", "int", False, 20),
        ],
        tags=["kernel", "aggregate", "hotspot"],
    )

    # 2) Top kernels alias to aggregate (same SQL, different default limit)
    skill_map["top_kernels"] = SqlSkill(
        name="top_kernels",
        title="Top Kernels",
        description="Top kernels ranked by cumulative execution time.",
        category="kernels",
        sql=skill_map["aggregate_kernels"].sql,
        params=[
            SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
            SqlSkillParam("limit", "max rows", "int", False, 15),
        ],
        tags=["kernel", "hotspot", "top"],
    )

    # 3) NVTX range aggregate
    if schema.table_exists(nvtx_table):
        nvtx_text_expr = "n.text"
        nvtx_join = ""
        text_id_col = schema.resolve_column(nvtx_table, ("textId",))
        if string_table and text_id_col:
            nvtx_join = f" LEFT JOIN {_ident(string_table)} s ON n.{_ident(text_id_col)} = s.id "
            nvtx_text_expr = "COALESCE(n.text, s.value)"
        skill_map["aggregate_nvtx_ranges"] = SqlSkill(
            name="aggregate_nvtx_ranges",
            title="Aggregate NVTX Ranges",
            description="Group NVTX ranges by text and report total/avg duration.",
            category="nvtx",
            sql=(
                f"SELECT {nvtx_text_expr} AS nvtx_name, "
                "COUNT(*) AS range_count, "
                "ROUND(SUM(n.[end] - n.start) / 1e6, 3) AS total_ms, "
                "ROUND(AVG(n.[end] - n.start) / 1e6, 3) AS avg_ms "
                f"FROM {nvtx_table} n "
                f"{nvtx_join} "
                f"WHERE {nvtx_text_expr} IS NOT NULL "
                "AND n.[end] > n.start "
                "GROUP BY nvtx_name "
                "ORDER BY total_ms DESC "
                "LIMIT {limit}"
            ),
            params=[SqlSkillParam("limit", "max rows", "int", False, 30)],
            tags=["nvtx", "range", "aggregate"],
        )

    # 4) Memcpy by kind in time window
    if schema.table_exists(memcpy_table):
        copy_kind_col = schema.resolve_column(memcpy_table, ("copyKind",))
        memcpy_device_col = schema.resolve_column(memcpy_table, ("deviceId",))
        memcpy_start_col = schema.resolve_column(memcpy_table, ("start",))
        memcpy_end_col = schema.resolve_column(memcpy_table, ("end",))
        if copy_kind_col and memcpy_start_col and memcpy_end_col:
            memcpy_where_device = ""
            if memcpy_device_col:
                memcpy_where_device = f" AND ({{device_id}} < 0 OR m.{_ident(memcpy_device_col)} = {{device_id}})"
            skill_map["memcpy_in_window"] = SqlSkill(
                name="memcpy_in_window",
                title="Memcpy In Window",
                description="Aggregate memcpy count/time by copyKind in optional time window.",
                category="memory",
                sql=(
                    f"SELECT m.{_ident(copy_kind_col)} AS copy_kind, "
                    "COUNT(*) AS memcpy_count, "
                    f"ROUND(SUM(m.[{_ident(memcpy_end_col)}] - m.{_ident(memcpy_start_col)}) / 1e6, 3) AS total_ms "
                    f"FROM {memcpy_table} m "
                    "WHERE 1=1 "
                    f"{memcpy_where_device} "
                    f"AND ({{start_ns}} < 0 OR m.{_ident(memcpy_start_col)} >= {{start_ns}}) "
                    f"AND ({{end_ns}} < 0 OR m.[{_ident(memcpy_end_col)}] <= {{end_ns}}) "
                    f"GROUP BY m.{_ident(copy_kind_col)} "
                    "ORDER BY total_ms DESC"
                ),
                params=[
                    SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                    SqlSkillParam("start_ns", "window start ns, -1 to disable", "int", False, -1),
                    SqlSkillParam("end_ns", "window end ns, -1 to disable", "int", False, -1),
                ],
                tags=["memcpy", "copykind", "window"],
            )

    # 5) Kernel map by correlation id
    if corr_col and stream_col:
        skill_map["kernel_map"] = SqlSkill(
            name="kernel_map",
            title="Kernel Correlation Map",
            description="Return kernel timeline rows keyed by correlationId.",
            category="kernels",
            sql=(
                f"SELECT k.{_ident(corr_col)} AS correlation_id, "
                f"k.{_ident(start_col)} AS start_ns, "
                f"k.[{_ident(end_col)}] AS end_ns, "
                f"k.{_ident(stream_col)} AS stream_id, "
                + (f"k.{_ident(device_col)} AS device_id, " if device_col else "NULL AS device_id, ")
                + f"{name_expr} AS kernel_name "
                f"FROM {kernel_table} k "
                f"{name_join} "
                "WHERE 1=1 "
                f"{kernel_where_device} "
                f"AND ({{start_ns}} < 0 OR k.{_ident(start_col)} >= {{start_ns}}) "
                f"AND ({{end_ns}} < 0 OR k.[{_ident(end_col)}] <= {{end_ns}}) "
                f"ORDER BY k.{_ident(start_col)} ASC "
                "LIMIT {limit}"
            ),
            params=[
                SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                SqlSkillParam("start_ns", "window start ns, -1 to disable", "int", False, -1),
                SqlSkillParam("end_ns", "window end ns, -1 to disable", "int", False, -1),
                SqlSkillParam("limit", "max rows", "int", False, 2000),
            ],
            tags=["kernel", "correlation", "map"],
        )

    # 6) GPU idle gaps
    if stream_col:
        skill_map["gpu_idle_gaps"] = SqlSkill(
            name="gpu_idle_gaps",
            title="GPU Idle Gaps",
            description="Detect gaps between consecutive kernels on each stream.",
            category="kernels",
            sql=(
                "WITH ordered AS ( "
                f"  SELECT k.{_ident(stream_col)} AS stream_id, "
                f"         k.{_ident(start_col)} AS start_ns, "
                f"         k.[{_ident(end_col)}] AS end_ns, "
                f"         {name_expr} AS kernel_name, "
                f"         LAG(k.[{_ident(end_col)}]) OVER (PARTITION BY k.{_ident(stream_col)} ORDER BY k.{_ident(start_col)}) AS prev_end_ns, "
                f"         LAG({name_expr}) OVER (PARTITION BY k.{_ident(stream_col)} ORDER BY k.{_ident(start_col)}) AS prev_kernel_name "
                f"  FROM {kernel_table} k "
                f"  {name_join} "
                "  WHERE 1=1 "
                f"  {kernel_where_device} "
                ") "
                "SELECT stream_id, "
                "       ROUND((start_ns - prev_end_ns) / 1e6, 3) AS gap_ms, "
                "       prev_kernel_name AS before_kernel, "
                "       kernel_name AS after_kernel "
                "FROM ordered "
                "WHERE prev_end_ns IS NOT NULL "
                "  AND (start_ns - prev_end_ns) > {min_gap_ns} "
                "ORDER BY gap_ms DESC "
                "LIMIT {limit}"
            ),
            params=[
                SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                SqlSkillParam("min_gap_ns", "minimum gap ns", "int", False, 1_000_000),
                SqlSkillParam("limit", "max rows", "int", False, 20),
            ],
            tags=["idle", "gap", "bubble", "stream"],
        )

    # 7) Kernel launch overhead
    runtime_corr_col = schema.resolve_column(runtime_table, ("correlationId",))
    runtime_start_col = schema.resolve_column(runtime_table, ("start",))
    runtime_end_col = schema.resolve_column(runtime_table, ("end",))
    if all([runtime_corr_col, runtime_start_col, runtime_end_col, corr_col]):
        runtime_api_name_expr = "NULL"
        runtime_api_name_join = ""
        runtime_name_id_col = schema.resolve_column(runtime_table, ("nameId", "name_id"))
        if string_table and runtime_name_id_col:
            runtime_api_name_join = (
                f" LEFT JOIN {_ident(string_table)} rs ON r.{_ident(runtime_name_id_col)} = rs.id "
            )
            runtime_api_name_expr = "rs.value"
        else:
            runtime_cbid_col = schema.resolve_column(runtime_table, ("cbid", "apiId", "functionId"))
            if runtime_cbid_col:
                runtime_api_name_expr = f"CAST(r.{_ident(runtime_cbid_col)} AS TEXT)"
        skill_map["kernel_launch_overhead"] = SqlSkill(
            name="kernel_launch_overhead",
            title="Kernel Launch Overhead",
            description="CPU runtime API latency and launch overhead to kernel start by correlationId, including API name when available.",
            category="kernels",
            sql=(
                f"SELECT {name_expr} AS kernel_name, "
                f"{runtime_api_name_expr} AS api_name, "
                f"ROUND((r.[{_ident(runtime_end_col)}] - r.{_ident(runtime_start_col)}) / 1e6, 3) AS api_ms, "
                f"ROUND((k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS kernel_ms, "
                f"ROUND((k.{_ident(start_col)} - r.{_ident(runtime_start_col)}) / 1e3, 3) AS overhead_us "
                f"FROM {_ident(runtime_table)} r "
                f"{runtime_api_name_join} "
                f"JOIN {kernel_table} k ON r.{_ident(runtime_corr_col)} = k.{_ident(corr_col)} "
                f"{name_join} "
                "WHERE 1=1 "
                f"{kernel_where_device} "
                "ORDER BY overhead_us DESC "
                "LIMIT {limit}"
            ),
            params=[
                SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                SqlSkillParam("limit", "max rows", "int", False, 30),
            ],
            tags=["launch", "overhead", "cpu_gpu_latency"],
        )

    # 8) NCCL breakdown
    skill_map["nccl_breakdown"] = SqlSkill(
        name="nccl_breakdown",
        title="NCCL Breakdown",
        description="Aggregate NCCL kernels by name and report count/time distribution.",
        category="communication",
        sql=(
            f"SELECT {name_expr} AS kernel_name, "
            "COUNT(*) AS count, "
            f"ROUND(SUM(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS total_ms, "
            f"ROUND(AVG(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS avg_ms, "
            f"ROUND(MIN(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS min_ms, "
            f"ROUND(MAX(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS max_ms "
            f"FROM {kernel_table} k "
            f"{name_join} "
            "WHERE 1=1 "
            f"{kernel_where_device} "
            f"AND LOWER({name_expr}) LIKE '%nccl%' "
            f"GROUP BY {name_expr} "
            "ORDER BY total_ms DESC "
            "LIMIT {limit}"
        ),
        params=[
            SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
            SqlSkillParam("limit", "max rows", "int", False, 30),
        ],
        tags=["nccl", "communication", "collective"],
    )

    # 9) NVTX -> kernel map
    runtime_global_tid_col = schema.resolve_column(runtime_table, ("globalTid",))
    nvtx_global_tid_col = schema.resolve_column(nvtx_table, ("globalTid",))
    nvtx_start_col = schema.resolve_column(nvtx_table, ("start",))
    nvtx_end_col = schema.resolve_column(nvtx_table, ("end",))
    runtime_start_for_nvtx = schema.resolve_column(runtime_table, ("start",))
    runtime_end_for_nvtx = schema.resolve_column(runtime_table, ("end",))
    if all([runtime_corr_col, corr_col, nvtx_start_col, nvtx_end_col, runtime_start_for_nvtx, runtime_end_for_nvtx]):
        nvtx_text_expr = "n.text"
        nvtx_join = ""
        if schema.string_table:
            text_id_col = schema.resolve_column(nvtx_table, ("textId", "nameId"))
            if text_id_col:
                nvtx_join = f" LEFT JOIN {_ident(schema.string_table)} s_nvtx ON n.{_ident(text_id_col)} = s_nvtx.id "
                nvtx_text_expr = "COALESCE(n.text, s_nvtx.value)"
        tid_join = ""
        if runtime_global_tid_col and nvtx_global_tid_col:
            tid_join = f" AND n.{_ident(nvtx_global_tid_col)} = r.{_ident(runtime_global_tid_col)} "
        skill_map["nvtx_kernel_map"] = SqlSkill(
            name="nvtx_kernel_map",
            title="NVTX Kernel Map",
            description="Map NVTX ranges to kernels launched inside each range.",
            category="nvtx",
            sql=(
                f"SELECT {nvtx_text_expr} AS nvtx_text, "
                f"{name_expr} AS kernel_name, "
                f"k.{_ident(start_col)} AS start_ns, "
                f"k.[{_ident(end_col)}] AS end_ns, "
                f"ROUND(k.{_ident(start_col)} / 1e6, 3) AS start_ms, "
                f"ROUND(k.[{_ident(end_col)}] / 1e6, 3) AS end_ms "
                f"FROM {nvtx_table} n "
                f"{nvtx_join} "
                f"JOIN {_ident(runtime_table)} r ON n.{_ident(nvtx_start_col)} <= r.{_ident(runtime_start_for_nvtx)} "
                f"AND n.[{_ident(nvtx_end_col)}] >= r.[{_ident(runtime_end_for_nvtx)}] "
                f"{tid_join} "
                f"JOIN {kernel_table} k ON r.{_ident(runtime_corr_col)} = k.{_ident(corr_col)} "
                f"{name_join} "
                "WHERE 1=1 "
                f"{kernel_where_device} "
                f"AND {nvtx_text_expr} IS NOT NULL "
                f"AND ({{start_ns}} < 0 OR k.{_ident(start_col)} >= {{start_ns}}) "
                f"AND ({{end_ns}} < 0 OR k.[{_ident(end_col)}] <= {{end_ns}}) "
                f"ORDER BY k.{_ident(start_col)} ASC "
                "LIMIT {limit}"
            ),
            params=[
                SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                SqlSkillParam("start_ns", "window start ns, -1 to disable", "int", False, -1),
                SqlSkillParam("end_ns", "window end ns, -1 to disable", "int", False, -1),
                SqlSkillParam("limit", "max rows", "int", False, 1000),
            ],
            tags=["nvtx", "kernel", "mapping", "attribution"],
        )

    # 10) Schema inspect
    skill_map["schema_inspect"] = SqlSkill(
        name="schema_inspect",
        title="Schema Inspect",
        description="Inspect SQLite tables and column schema.",
        category="utility",
        sql=(
            "SELECT m.name AS table_name, p.name AS column_name, p.type AS column_type, p.pk AS is_pk "
            "FROM sqlite_master m "
            "JOIN pragma_table_info(m.name) p "
            "WHERE m.type = 'table' "
            "AND m.name NOT LIKE 'sqlite_%' "
            "AND m.name LIKE '{table_like}' "
            "ORDER BY m.name, p.cid "
            "LIMIT {limit}"
        ),
        params=[
            SqlSkillParam("table_like", "SQL LIKE expression for table name", "str", False, "%"),
            SqlSkillParam("limit", "max rows", "int", False, 10000),
        ],
        tags=["schema", "tables", "columns"],
    )

    # 11) GPU metrics aggregate (nsys --gpu-metrics-device style hardware sampling)
    if schema.metrics_table:
        metrics_table = _ident(schema.metrics_table)
        metrics_ts_col = schema.metrics_timestamp_col or schema.resolve_column(metrics_table, ("timestamp", "start", "time"))
        metrics_id_col = schema.metrics_id_col or schema.resolve_column(metrics_table, ("metricId", "nameId", "eventId"))
        metrics_value_col = schema.metrics_value_col or schema.resolve_column(metrics_table, ("value", "metricValue", "val"))
        if metrics_ts_col and metrics_id_col and metrics_value_col:
            metrics_name_expr = ""
            metrics_name_join = ""
            metrics_name_not_null = ""

            # Prefer dedicated GPU metric dictionary table when available.
            gpu_info_table = "TARGET_INFO_GPU_METRICS" if schema.table_exists("TARGET_INFO_GPU_METRICS") else None
            if gpu_info_table:
                gpu_info_tbl = _ident(gpu_info_table)
                gpu_info_id_col = schema.resolve_column(gpu_info_tbl, ("metricId", "id", "metric_id"))
                gpu_info_name_col = schema.resolve_column(gpu_info_tbl, ("name", "metricName", "metric_name", "value"))
                metrics_type_col = schema.resolve_column(metrics_table, ("typeId", "type_id", "eventType", "event_type"))
                gpu_info_type_col = schema.resolve_column(gpu_info_tbl, ("typeId", "type_id", "eventType", "event_type"))
                if gpu_info_id_col and gpu_info_name_col:
                    if metrics_type_col and gpu_info_type_col:
                        metrics_name_join = (
                            f"JOIN {gpu_info_tbl} gm "
                            f"ON g.{_ident(metrics_id_col)} = gm.{_ident(gpu_info_id_col)} "
                            f"AND g.{_ident(metrics_type_col)} = gm.{_ident(gpu_info_type_col)} "
                        )
                    else:
                        metrics_name_join = (
                            f"JOIN {gpu_info_tbl} gm "
                            f"ON g.{_ident(metrics_id_col)} = gm.{_ident(gpu_info_id_col)} "
                        )
                    metrics_name_expr = f"gm.{_ident(gpu_info_name_col)}"
                    metrics_name_not_null = f"AND gm.{_ident(gpu_info_name_col)} IS NOT NULL "
            metrics_has_gpu_info_mapping = bool(metrics_name_expr.startswith("gm."))

            # Fallback to StringIds only when GPU metric dictionary is unavailable.
            if not metrics_name_expr and schema.string_table:
                metrics_string_table = _ident(schema.string_table)
                metrics_name_expr = "s.value"
                metrics_name_join = f"JOIN {metrics_string_table} s ON g.{_ident(metrics_id_col)} = s.id "
                metrics_name_not_null = "AND s.value IS NOT NULL "

            if not metrics_name_expr:
                metrics_name_expr = f"CAST(g.{_ident(metrics_id_col)} AS TEXT)"
                metrics_name_not_null = ""

            # Filter non-GPU generic sources when sourceId dictionary exists.
            metrics_source_join = ""
            metrics_source_where = ""
            metrics_source_name_expr = "NULL"
            metrics_source_col = schema.resolve_column(metrics_table, ("sourceId", "source_id"))
            if metrics_source_col and schema.table_exists("GENERIC_EVENT_SOURCES") and (not metrics_has_gpu_info_mapping):
                ges_tbl = _ident("GENERIC_EVENT_SOURCES")
                ges_id_col = schema.resolve_column(ges_tbl, ("sourceId", "id", "source_id"))
                ges_name_col = schema.resolve_column(ges_tbl, ("name", "source", "sourceName"))
                if ges_id_col and ges_name_col:
                    metrics_source_join = (
                        f"LEFT JOIN {ges_tbl} gs "
                        f"ON g.{_ident(metrics_source_col)} = gs.{_ident(ges_id_col)} "
                    )
                    metrics_source_name_expr = f"gs.{_ident(ges_name_col)}"
                    metrics_source_where = (
                        "AND ("
                        "{include_all_sources} = 1 "
                        f"OR gs.{_ident(ges_name_col)} IS NULL "
                        f"OR LOWER(gs.{_ident(ges_name_col)}) LIKE '%gpu%metric%' "
                        f"OR LOWER(gs.{_ident(ges_name_col)}) = 'gpumetrics'"
                        ") "
                    )

            metrics_device_expr = "'unknown'"
            metrics_device_where = ""
            metrics_device_col = schema.resolve_column(metrics_table, ("deviceId", "gpuId", "device", "gpu"))
            if metrics_device_col:
                metrics_device_expr = f"CAST(g.{_ident(metrics_device_col)} AS TEXT)"
                metrics_device_where = f"AND ({{device_id}} < 0 OR g.{_ident(metrics_device_col)} = {{device_id}}) "
            elif metrics_source_name_expr != "NULL":
                metrics_device_expr = f"COALESCE({metrics_source_name_expr}, 'unknown')"

            skill_map["gpu_metrics_aggregate"] = SqlSkill(
                name="gpu_metrics_aggregate",
                title="GPU Metrics Aggregate",
                description=(
                    "Aggregate Nsight GPU hardware sampling metrics (for example sm__active, "
                    "tensor__active, dram bandwidth counters) by metric name."
                ),
                category="metrics",
                sql=(
                    f"SELECT {metrics_name_expr} AS metric_name, "
                    f"{metrics_device_expr} AS metric_device, "
                    "COUNT(*) AS sample_count, "
                    f"ROUND(AVG(CAST(g.{_ident(metrics_value_col)} AS REAL)), 3) AS avg_value, "
                    f"ROUND(MIN(CAST(g.{_ident(metrics_value_col)} AS REAL)), 3) AS min_value, "
                    f"ROUND(MAX(CAST(g.{_ident(metrics_value_col)} AS REAL)), 3) AS max_value "
                    f"FROM {metrics_table} g "
                    f"{metrics_name_join} "
                    f"{metrics_source_join} "
                    "WHERE 1=1 "
                    f"AND ({{start_ns}} < 0 OR g.{_ident(metrics_ts_col)} >= {{start_ns}}) "
                    f"AND ({{end_ns}} < 0 OR g.{_ident(metrics_ts_col)} <= {{end_ns}}) "
                    f"{metrics_source_where}"
                    f"{metrics_device_where}"
                    f"AND ('{{metric_name_like}}' = '%' OR {metrics_name_expr} LIKE '{{metric_name_like}}') "
                    f"{metrics_name_not_null}"
                    f"AND g.{_ident(metrics_value_col)} IS NOT NULL "
                    f"GROUP BY {metrics_name_expr}, {metrics_device_expr} "
                    "ORDER BY sample_count DESC, metric_name ASC, metric_device ASC "
                    "LIMIT {limit}"
                ),
                params=[
                    SqlSkillParam("start_ns", "window start ns, -1 to disable", "int", False, -1),
                    SqlSkillParam("end_ns", "window end ns, -1 to disable", "int", False, -1),
                    SqlSkillParam("metric_name_like", "metric name SQL LIKE pattern", "str", False, "%"),
                    SqlSkillParam("device_id", "CUDA deviceId if metrics table provides it, -1 means all", "int", False, -1),
                    SqlSkillParam(
                        "include_all_sources",
                        "1 to include all generic sources (ETW/FTrace/etc), 0 to prefer GPU metrics sources only",
                        "bool",
                        False,
                        False,
                    ),
                    SqlSkillParam("limit", "max rows", "int", False, 500),
                ],
                tags=["gpu_metrics", "sampling", "sm_active", "tensor_active", "dram"],
            )

    # 12) Thread utilization
    if schema.table_exists("COMPOSITE_EVENTS"):
        ce_cpu_col = schema.resolve_column("COMPOSITE_EVENTS", ("cpuCycles", "cycles", "CpuCycles"))
        ce_tid_col = schema.resolve_column("COMPOSITE_EVENTS", ("globalTid", "global_tid"))
        tn_tid_col = schema.resolve_column("ThreadNames", ("globalTid", "global_tid"))
        tn_name_col = schema.resolve_column("ThreadNames", ("nameId", "name_id"))
        if ce_cpu_col and ce_tid_col:
            thread_name_expr = "NULL"
            thread_name_join = ""
            if schema.table_exists("ThreadNames") and schema.string_table and tn_tid_col and tn_name_col:
                thread_name_join = (
                    f" LEFT JOIN ThreadNames tn ON ce.{_ident(ce_tid_col)} = tn.{_ident(tn_tid_col)} "
                    f" LEFT JOIN {_ident(schema.string_table)} ts ON tn.{_ident(tn_name_col)} = ts.id "
                )
                thread_name_expr = "ts.value"
            skill_map["thread_utilization"] = SqlSkill(
                name="thread_utilization",
                title="Thread Utilization",
                description="Aggregate CPU cycles by thread and estimate utilization percentage.",
                category="system",
                sql=(
                    f"SELECT ce.{_ident(ce_tid_col)} AS global_tid, "
                    f"({thread_name_expr}) AS thread_name, "
                    f"ROUND(SUM(ce.{_ident(ce_cpu_col)}) * 100.0 / "
                    f"(SELECT CASE WHEN SUM({_ident(ce_cpu_col)}) > 0 THEN SUM({_ident(ce_cpu_col)}) ELSE 1 END "
                    "FROM COMPOSITE_EVENTS), 4) AS cpu_pct, "
                    f"SUM(ce.{_ident(ce_cpu_col)}) AS cpu_cycles "
                    "FROM COMPOSITE_EVENTS ce "
                    f"{thread_name_join} "
                    f"GROUP BY ce.{_ident(ce_tid_col)}, thread_name "
                    "ORDER BY cpu_pct DESC "
                    "LIMIT {limit}"
                ),
                params=[SqlSkillParam("limit", "max rows", "int", False, 30)],
                tags=["thread", "cpu", "utilization"],
            )

    # 12) Memcpy bandwidth analysis
    if schema.table_exists(memcpy_table):
        bw_bytes_col = schema.resolve_column(memcpy_table, ("bytes", "srcSize", "numBytes"))
        bw_ck_col = schema.resolve_column(memcpy_table, ("copyKind",))
        bw_dev_col = schema.resolve_column(memcpy_table, ("deviceId",))
        bw_start_col = schema.resolve_column(memcpy_table, ("start",))
        bw_end_col = schema.resolve_column(memcpy_table, ("end",))
        if bw_bytes_col and bw_ck_col and bw_start_col and bw_end_col:
            bw_device_where = ""
            if bw_dev_col:
                bw_device_where = f" AND ({{device_id}} < 0 OR m.{_ident(bw_dev_col)} = {{device_id}})"
            skill_map["memcpy_bandwidth_analysis"] = SqlSkill(
                name="memcpy_bandwidth_analysis",
                title="Memcpy Bandwidth Analysis",
                description="Aggregate memcpy bandwidth (GB/s) by copyKind direction (H2D/D2H/D2D).",
                category="memory",
                sql=(
                    f"SELECT m.{_ident(bw_ck_col)} AS copy_kind, "
                    "COUNT(*) AS count, "
                    f"ROUND(SUM(m.{_ident(bw_bytes_col)}) / 1.0e9, 3) AS total_gb, "
                    f"ROUND(SUM(m.[{_ident(bw_end_col)}] - m.{_ident(bw_start_col)}) / 1.0e6, 3) AS total_ms, "
                    f"ROUND(CAST(SUM(m.{_ident(bw_bytes_col)}) AS REAL) / "
                    f"NULLIF(SUM(m.[{_ident(bw_end_col)}] - m.{_ident(bw_start_col)}), 0), 3) AS avg_gbps, "
                    f"ROUND(MIN(CAST(m.{_ident(bw_bytes_col)} AS REAL) / "
                    f"NULLIF(m.[{_ident(bw_end_col)}] - m.{_ident(bw_start_col)}, 0)), 3) AS min_gbps, "
                    f"ROUND(MAX(CAST(m.{_ident(bw_bytes_col)} AS REAL) / "
                    f"NULLIF(m.[{_ident(bw_end_col)}] - m.{_ident(bw_start_col)}, 0)), 3) AS max_gbps "
                    f"FROM {memcpy_table} m "
                    "WHERE 1=1 "
                    f"{bw_device_where} "
                    f"GROUP BY m.{_ident(bw_ck_col)} "
                    "ORDER BY total_ms DESC"
                ),
                params=[
                    SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                ],
                tags=["memcpy", "bandwidth", "pcie", "nvlink", "memory"],
            )

    # 13) Sync breakdown
    if schema.sync_table and schema.table_exists(schema.sync_table):
        sync_tbl = _ident(schema.sync_table)
        sync_type_col = schema.resolve_column(sync_tbl, ("syncType", "type"))
        sync_start_col = schema.resolve_column(sync_tbl, ("start",))
        sync_end_col = schema.resolve_column(sync_tbl, ("end",))
        sync_dev_col = schema.resolve_column(sync_tbl, ("deviceId",))
        if sync_type_col and sync_start_col and sync_end_col:
            sync_device_where = ""
            if sync_dev_col:
                sync_device_where = f" AND ({{device_id}} < 0 OR s.{_ident(sync_dev_col)} = {{device_id}})"
            skill_map["sync_breakdown"] = SqlSkill(
                name="sync_breakdown",
                title="Sync Breakdown",
                description="Aggregate CUDA synchronization events by type, reporting count and overhead.",
                category="pipeline",
                sql=(
                    f"SELECT s.{_ident(sync_type_col)} AS sync_type, "
                    "COUNT(*) AS count, "
                    f"ROUND(SUM(s.[{_ident(sync_end_col)}] - s.{_ident(sync_start_col)}) / 1e6, 3) AS total_ms, "
                    f"ROUND(AVG(s.[{_ident(sync_end_col)}] - s.{_ident(sync_start_col)}) / 1e6, 3) AS avg_ms, "
                    f"ROUND(MAX(s.[{_ident(sync_end_col)}] - s.{_ident(sync_start_col)}) / 1e6, 3) AS max_ms "
                    f"FROM {sync_tbl} s "
                    "WHERE 1=1 "
                    f"{sync_device_where} "
                    f"GROUP BY s.{_ident(sync_type_col)} "
                    "ORDER BY total_ms DESC "
                    "LIMIT {limit}"
                ),
                params=[
                    SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                    SqlSkillParam("limit", "max rows", "int", False, 50),
                ],
                tags=["sync", "synchronization", "overhead", "pipeline"],
            )

    # 14) Memset breakdown
    if schema.memset_table and schema.table_exists(schema.memset_table):
        memset_tbl = _ident(schema.memset_table)
        memset_bytes_col = schema.resolve_column(memset_tbl, ("bytes", "numBytes"))
        memset_value_col = schema.resolve_column(memset_tbl, ("value",))
        memset_start_col = schema.resolve_column(memset_tbl, ("start",))
        memset_end_col = schema.resolve_column(memset_tbl, ("end",))
        memset_dev_col = schema.resolve_column(memset_tbl, ("deviceId",))
        if memset_bytes_col and memset_start_col and memset_end_col:
            memset_device_where = ""
            if memset_dev_col:
                memset_device_where = f" AND ({{device_id}} < 0 OR ms.{_ident(memset_dev_col)} = {{device_id}})"
            if memset_value_col:
                value_select = f"ms.{_ident(memset_value_col)} AS fill_value, "
                group_by_expr = f"ms.{_ident(memset_value_col)}"
            else:
                value_select = ""
                group_by_expr = "'all'"
            skill_map["memset_breakdown"] = SqlSkill(
                name="memset_breakdown",
                title="Memset Breakdown",
                description="Aggregate memset ops by fill value (0=zero-init vs custom fill), reporting bytes and bandwidth.",
                category="memory",
                sql=(
                    f"SELECT {value_select}"
                    "COUNT(*) AS count, "
                    f"ROUND(SUM(ms.{_ident(memset_bytes_col)}) / 1.0e9, 3) AS total_gb, "
                    f"ROUND(SUM(ms.[{_ident(memset_end_col)}] - ms.{_ident(memset_start_col)}) / 1e6, 3) AS total_ms, "
                    f"ROUND(CAST(SUM(ms.{_ident(memset_bytes_col)}) AS REAL) / "
                    f"NULLIF(SUM(ms.[{_ident(memset_end_col)}] - ms.{_ident(memset_start_col)}), 0), 3) AS avg_gbps "
                    f"FROM {memset_tbl} ms "
                    "WHERE 1=1 "
                    f"{memset_device_where} "
                    f"GROUP BY {group_by_expr} "
                    "ORDER BY total_ms DESC "
                    "LIMIT {limit}"
                ),
                params=[
                    SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                    SqlSkillParam("limit", "max rows", "int", False, 20),
                ],
                tags=["memset", "zero-init", "memory", "overhead"],
            )

    # 15) Kernel occupancy estimate (raw launch metrics; occupancy computed in Python)
    occ_block_x = schema.resolve_column(kernel_table, ("blockX",))
    occ_block_y = schema.resolve_column(kernel_table, ("blockY",))
    occ_block_z = schema.resolve_column(kernel_table, ("blockZ",))
    occ_reg_col = schema.resolve_column(kernel_table, ("registersPerThread",))
    occ_static_smem = schema.resolve_column(kernel_table, ("staticSharedMemory",))
    occ_dyn_smem = schema.resolve_column(kernel_table, ("dynamicSharedMemory",))
    occ_theoretical_col = schema.resolve_column(
        kernel_table,
        ("theoreticalOccupancyPct", "theoreticalOccupancy", "theoretical_occupancy_pct", "theoretical_occupancy"),
    )
    if occ_block_x and occ_block_y and occ_block_z:
        if occ_static_smem and occ_dyn_smem:
            occ_shared_expr = f"(COALESCE(k.{_ident(occ_static_smem)}, 0) + COALESCE(k.{_ident(occ_dyn_smem)}, 0))"
        elif occ_static_smem:
            occ_shared_expr = f"COALESCE(k.{_ident(occ_static_smem)}, 0)"
        elif occ_dyn_smem:
            occ_shared_expr = f"COALESCE(k.{_ident(occ_dyn_smem)}, 0)"
        else:
            occ_shared_expr = "NULL"
        occ_static_expr = f"k.{_ident(occ_static_smem)}" if occ_static_smem else "NULL"
        occ_dyn_expr = f"k.{_ident(occ_dyn_smem)}" if occ_dyn_smem else "NULL"
        occ_reg_expr = f"k.{_ident(occ_reg_col)}" if occ_reg_col else "NULL"
        occ_tpb_expr = f"(k.{_ident(occ_block_x)} * k.{_ident(occ_block_y)} * k.{_ident(occ_block_z)})"
        if occ_theoretical_col:
            occ_estimate_expr = f"ROUND(AVG(CAST(k.{_ident(occ_theoretical_col)} AS REAL)), 3)"
        else:
            occ_estimate_expr = "NULL"
        skill_map["kernel_occupancy_estimate"] = SqlSkill(
            name="kernel_occupancy_estimate",
            title="Kernel Occupancy Estimate",
            description=(
                "Export raw launch-config metrics for occupancy analysis: threads_per_block, "
                "registersPerThread, static/dynamic shared memory and total_shared_bytes. "
                "If sqlite includes theoretical occupancy, it is returned in occupancy_pct_estimate; "
                "otherwise use H100(sm_90) occupancy from calculate_h100_occupancy(...)."
            ),
            category="compute",
            sql=(
                f"SELECT {name_expr} AS kernel_name, "
                f"{occ_tpb_expr} AS threads_per_block, "
                f"{occ_reg_expr} AS registersPerThread, "
                f"{occ_static_expr} AS static_shared_bytes, "
                f"{occ_dyn_expr} AS dynamic_shared_bytes, "
                f"{occ_shared_expr} AS total_shared_bytes, "
                "COUNT(*) AS invocations, "
                f"ROUND(SUM(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS total_ms, "
                f"ROUND(AVG(k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS avg_ms, "
                f"{occ_estimate_expr} AS occupancy_pct_estimate "
                f"FROM {kernel_table} k "
                f"{name_join} "
                "WHERE 1=1 "
                f"{kernel_where_device} "
                "GROUP BY 1,2,3,4,5,6 "
                "ORDER BY total_ms DESC "
                "LIMIT {limit}"
            ),
            params=[
                SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                SqlSkillParam("limit", "max rows", "int", False, 30),
            ],
            tags=["occupancy", "compute", "launch_config", "register", "shared_memory"],
        )

    # 16) Stream parallelism
    if stream_col:
        sp_device_where = ""
        if device_col:
            sp_device_where = f" AND ({{device_id}} < 0 OR k.{_ident(device_col)} = {{device_id}})"
        skill_map["stream_parallelism"] = SqlSkill(
            name="stream_parallelism",
            title="Stream Parallelism",
            description=(
                "Analyze concurrent stream utilization via time buckets. "
                "Each kernel contributes to all buckets it overlaps (not only its start bucket)."
            ),
            category="pipeline",
            sql=(
                "WITH kernels AS ( "
                f"  SELECT CAST(k.{_ident(start_col)} / {{bucket_ns}} AS INTEGER) AS start_bucket, "
                f"         CAST((k.[{_ident(end_col)}] - 1) / {{bucket_ns}} AS INTEGER) AS end_bucket, "
                f"         k.{_ident(stream_col)} AS stream_id "
                f"  FROM {kernel_table} k "
                "  WHERE 1=1 "
                f"  {sp_device_where} "
                f"  AND k.[{_ident(end_col)}] > k.{_ident(start_col)} "
                "), "
                "expanded(bucket, end_bucket, stream_id) AS ( "
                "  SELECT start_bucket, end_bucket, stream_id FROM kernels "
                "  UNION ALL "
                "  SELECT bucket + 1, end_bucket, stream_id "
                "  FROM expanded "
                "  WHERE bucket < end_bucket "
                "), "
                "bucket_streams AS ( "
                "  SELECT bucket, COUNT(DISTINCT stream_id) AS concurrent_streams "
                "  FROM expanded "
                "  GROUP BY bucket "
                ") "
                "SELECT "
                "  MAX(concurrent_streams) AS max_concurrent_streams, "
                "  ROUND(AVG(CAST(concurrent_streams AS REAL)), 2) AS avg_concurrent_streams, "
                "  COUNT(*) AS total_buckets, "
                "  SUM(CASE WHEN concurrent_streams > 1 THEN 1 ELSE 0 END) AS multi_stream_buckets, "
                "  ROUND(100.0 * SUM(CASE WHEN concurrent_streams > 1 THEN 1 ELSE 0 END) / MAX(1, COUNT(*)), 1) AS pct_time_multi_stream "
                "FROM bucket_streams"
            ),
            params=[
                SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                SqlSkillParam("bucket_ns", "time bucket size in nanoseconds", "int", False, 1_000_000),
            ],
            tags=["stream", "parallelism", "concurrent", "pipeline", "overlap"],
        )

    # 17) NVTX memcpy breakdown (per-phase data movement)
    if schema.table_exists(nvtx_table) and schema.table_exists(memcpy_table):
        nm_nvtx_start = schema.resolve_column(nvtx_table, ("start",))
        nm_nvtx_end = schema.resolve_column(nvtx_table, ("end",))
        nm_bytes_col = schema.resolve_column(memcpy_table, ("bytes", "srcSize", "numBytes"))
        nm_mc_start = schema.resolve_column(memcpy_table, ("start",))
        nm_mc_end = schema.resolve_column(memcpy_table, ("end",))
        nm_ck_col = schema.resolve_column(memcpy_table, ("copyKind",))
        if nm_nvtx_start and nm_nvtx_end and nm_bytes_col and nm_mc_start and nm_mc_end:
            nm_text_expr = "n.text"
            nm_nvtx_join = ""
            if string_table:
                nm_text_id = schema.resolve_column(nvtx_table, ("textId", "nameId"))
                if nm_text_id:
                    nm_nvtx_join = f" LEFT JOIN {_ident(string_table)} s_n ON n.{_ident(nm_text_id)} = s_n.id "
                    nm_text_expr = "COALESCE(n.text, s_n.value)"
            nm_ck_select = f"m.{_ident(nm_ck_col)} AS copy_kind, " if nm_ck_col else ""
            nm_ck_group = f", m.{_ident(nm_ck_col)}" if nm_ck_col else ""
            skill_map["nvtx_memcpy_breakdown"] = SqlSkill(
                name="nvtx_memcpy_breakdown",
                title="NVTX Memcpy Breakdown",
                description="Aggregate memcpy bytes and duration within each NVTX range to identify per-phase data movement.",
                category="memory",
                sql=(
                    f"SELECT {nm_text_expr} AS nvtx_text, "
                    f"{nm_ck_select}"
                    "COUNT(*) AS memcpy_count, "
                    f"ROUND(SUM(m.{_ident(nm_bytes_col)}) / 1.0e9, 3) AS total_gb, "
                    f"ROUND(SUM(m.[{_ident(nm_mc_end)}] - m.{_ident(nm_mc_start)}) / 1e6, 3) AS total_ms, "
                    f"ROUND(CAST(SUM(m.{_ident(nm_bytes_col)}) AS REAL) / "
                    f"NULLIF(SUM(m.[{_ident(nm_mc_end)}] - m.{_ident(nm_mc_start)}), 0), 3) AS avg_gbps "
                    f"FROM {nvtx_table} n "
                    f"{nm_nvtx_join} "
                    f"JOIN {memcpy_table} m "
                    f"  ON m.{_ident(nm_mc_start)} >= n.{_ident(nm_nvtx_start)} "
                    f"  AND m.[{_ident(nm_mc_end)}] <= n.[{_ident(nm_nvtx_end)}] "
                    f"WHERE {nm_text_expr} IS NOT NULL "
                    f"AND n.[{_ident(nm_nvtx_end)}] > n.{_ident(nm_nvtx_start)} "
                    f"GROUP BY {nm_text_expr}{nm_ck_group} "
                    "ORDER BY total_ms DESC "
                    "LIMIT {limit}"
                ),
                params=[
                    SqlSkillParam("limit", "max rows", "int", False, 50),
                ],
                tags=["nvtx", "memcpy", "bandwidth", "phase", "memory"],
            )

    # 18) NVTX GPU metrics breakdown (per-range hardware sampling)
    if schema.table_exists(nvtx_table) and schema.metrics_table:
        ngm_metrics_table = _ident(schema.metrics_table)
        ngm_ts_col = schema.metrics_timestamp_col or schema.resolve_column(ngm_metrics_table, ("timestamp", "start", "time"))
        ngm_id_col = schema.metrics_id_col or schema.resolve_column(ngm_metrics_table, ("metricId", "nameId", "eventId"))
        ngm_val_col = schema.metrics_value_col or schema.resolve_column(ngm_metrics_table, ("value", "metricValue", "val"))
        ngm_nvtx_start = schema.resolve_column(nvtx_table, ("start",))
        ngm_nvtx_end = schema.resolve_column(nvtx_table, ("end",))
        if ngm_ts_col and ngm_id_col and ngm_val_col and ngm_nvtx_start and ngm_nvtx_end:
            ngm_nvtx_text_expr = "n.text"
            ngm_nvtx_text_join = ""
            if string_table:
                ngm_text_id = schema.resolve_column(nvtx_table, ("textId", "nameId"))
                if ngm_text_id:
                    ngm_nvtx_text_join = f" LEFT JOIN {_ident(string_table)} sn ON n.{_ident(ngm_text_id)} = sn.id "
                    ngm_nvtx_text_expr = "COALESCE(n.text, sn.value)"

            ngm_metric_name_expr = ""
            ngm_metric_name_join = ""
            ngm_metric_name_not_null = ""

            ngm_gpu_info_table = "TARGET_INFO_GPU_METRICS" if schema.table_exists("TARGET_INFO_GPU_METRICS") else None
            if ngm_gpu_info_table:
                ngm_gi_tbl = _ident(ngm_gpu_info_table)
                ngm_gi_id_col = schema.resolve_column(ngm_gi_tbl, ("metricId", "id", "metric_id"))
                ngm_gi_name_col = schema.resolve_column(ngm_gi_tbl, ("name", "metricName", "metric_name", "value"))
                ngm_type_col = schema.resolve_column(ngm_metrics_table, ("typeId", "type_id", "eventType", "event_type"))
                ngm_gi_type_col = schema.resolve_column(ngm_gi_tbl, ("typeId", "type_id", "eventType", "event_type"))
                if ngm_gi_id_col and ngm_gi_name_col:
                    if ngm_type_col and ngm_gi_type_col:
                        ngm_metric_name_join = (
                            f"JOIN {ngm_gi_tbl} gm "
                            f"ON g.{_ident(ngm_id_col)} = gm.{_ident(ngm_gi_id_col)} "
                            f"AND g.{_ident(ngm_type_col)} = gm.{_ident(ngm_gi_type_col)} "
                        )
                    else:
                        ngm_metric_name_join = (
                            f"JOIN {ngm_gi_tbl} gm "
                            f"ON g.{_ident(ngm_id_col)} = gm.{_ident(ngm_gi_id_col)} "
                        )
                    ngm_metric_name_expr = f"gm.{_ident(ngm_gi_name_col)}"
                    ngm_metric_name_not_null = f"AND gm.{_ident(ngm_gi_name_col)} IS NOT NULL "
            ngm_has_gpu_info_mapping = bool(ngm_metric_name_expr.startswith("gm."))

            if not ngm_metric_name_expr and schema.string_table:
                ngm_string_table = _ident(schema.string_table)
                ngm_metric_name_expr = "s.value"
                ngm_metric_name_join = f"JOIN {ngm_string_table} s ON g.{_ident(ngm_id_col)} = s.id "
                ngm_metric_name_not_null = "AND s.value IS NOT NULL "

            if not ngm_metric_name_expr:
                ngm_metric_name_expr = f"CAST(g.{_ident(ngm_id_col)} AS TEXT)"
                ngm_metric_name_not_null = ""

            ngm_source_join = ""
            ngm_source_where = ""
            ngm_source_name_expr = "NULL"
            ngm_source_col = schema.resolve_column(ngm_metrics_table, ("sourceId", "source_id"))
            if ngm_source_col and schema.table_exists("GENERIC_EVENT_SOURCES") and (not ngm_has_gpu_info_mapping):
                ngm_ges_tbl = _ident("GENERIC_EVENT_SOURCES")
                ngm_ges_id_col = schema.resolve_column(ngm_ges_tbl, ("sourceId", "id", "source_id"))
                ngm_ges_name_col = schema.resolve_column(ngm_ges_tbl, ("name", "source", "sourceName"))
                if ngm_ges_id_col and ngm_ges_name_col:
                    ngm_source_join = (
                        f"LEFT JOIN {ngm_ges_tbl} gs2 "
                        f"ON g.{_ident(ngm_source_col)} = gs2.{_ident(ngm_ges_id_col)} "
                    )
                    ngm_source_name_expr = f"gs2.{_ident(ngm_ges_name_col)}"
                    ngm_source_where = (
                        "AND ("
                        "{include_all_sources} = 1 "
                        f"OR gs2.{_ident(ngm_ges_name_col)} IS NULL "
                        f"OR LOWER(gs2.{_ident(ngm_ges_name_col)}) LIKE '%gpu%metric%' "
                        f"OR LOWER(gs2.{_ident(ngm_ges_name_col)}) = 'gpumetrics'"
                        ") "
                    )

            ngm_device_where = ""
            ngm_device_col = schema.resolve_column(ngm_metrics_table, ("deviceId", "gpuId", "device", "gpu"))
            if ngm_device_col:
                ngm_device_where = f"AND ({{device_id}} < 0 OR g.{_ident(ngm_device_col)} = {{device_id}}) "
                ngm_metric_device_expr = f"CAST(g.{_ident(ngm_device_col)} AS TEXT)"
            elif ngm_source_name_expr != "NULL":
                ngm_metric_device_expr = f"COALESCE({ngm_source_name_expr}, 'unknown')"
            else:
                ngm_metric_device_expr = "'unknown'"

            skill_map["nvtx_gpu_metrics_breakdown"] = SqlSkill(
                name="nvtx_gpu_metrics_breakdown",
                title="NVTX GPU Metrics Breakdown",
                description=(
                    "For NVTX ranges matching nvtx_text, aggregate GPU hardware sampling points "
                    "whose timestamps fall inside each range. Useful to inspect sm/tensor/dram "
                    "metrics per training phase (forward/backward/sample_x)."
                ),
                category="metrics",
                sql=(
                    f"SELECT {ngm_nvtx_text_expr} AS nvtx_text, "
                    f"n.{_ident(ngm_nvtx_start)} AS nvtx_start_ns, "
                    f"n.[{_ident(ngm_nvtx_end)}] AS nvtx_end_ns, "
                    f"{ngm_metric_name_expr} AS metric_name, "
                    f"{ngm_metric_device_expr} AS metric_device, "
                    "COUNT(*) AS sample_count, "
                    f"ROUND(AVG(CAST(g.{_ident(ngm_val_col)} AS REAL)), 3) AS avg_value, "
                    f"ROUND(MIN(CAST(g.{_ident(ngm_val_col)} AS REAL)), 3) AS min_value, "
                    f"ROUND(MAX(CAST(g.{_ident(ngm_val_col)} AS REAL)), 3) AS max_value "
                    f"FROM {nvtx_table} n "
                    f"{ngm_nvtx_text_join} "
                    f"JOIN {ngm_metrics_table} g "
                    f"  ON g.{_ident(ngm_ts_col)} >= n.{_ident(ngm_nvtx_start)} "
                    f" AND g.{_ident(ngm_ts_col)} <= n.[{_ident(ngm_nvtx_end)}] "
                    f"{ngm_metric_name_join} "
                    f"{ngm_source_join} "
                    "WHERE 1=1 "
                    f"AND ('{{nvtx_text}}' = '%' OR {ngm_nvtx_text_expr} LIKE '{{nvtx_text}}') "
                    f"AND ('{{metric_name_like}}' = '%' OR {ngm_metric_name_expr} LIKE '{{metric_name_like}}') "
                    f"AND {ngm_nvtx_text_expr} IS NOT NULL "
                    f"AND n.[{_ident(ngm_nvtx_end)}] > n.{_ident(ngm_nvtx_start)} "
                    f"AND ({{start_ns}} < 0 OR n.{_ident(ngm_nvtx_start)} >= {{start_ns}}) "
                    f"AND ({{end_ns}} < 0 OR n.[{_ident(ngm_nvtx_end)}] <= {{end_ns}}) "
                    f"{ngm_source_where}"
                    f"{ngm_device_where}"
                    f"{ngm_metric_name_not_null}"
                    f"AND g.{_ident(ngm_val_col)} IS NOT NULL "
                    f"GROUP BY {ngm_nvtx_text_expr}, n.{_ident(ngm_nvtx_start)}, n.[{_ident(ngm_nvtx_end)}], {ngm_metric_name_expr}, {ngm_metric_device_expr} "
                    f"ORDER BY n.{_ident(ngm_nvtx_start)} ASC, sample_count DESC, metric_name ASC, metric_device ASC "
                    "LIMIT {limit}"
                ),
                params=[
                    SqlSkillParam("nvtx_text", "NVTX text LIKE pattern, default %", "str", False, "%"),
                    SqlSkillParam("metric_name_like", "metric name SQL LIKE pattern", "str", False, "%"),
                    SqlSkillParam("device_id", "CUDA deviceId if metric table provides it, -1 means all", "int", False, -1),
                    SqlSkillParam("start_ns", "NVTX window start ns, -1 to disable", "int", False, -1),
                    SqlSkillParam("end_ns", "NVTX window end ns, -1 to disable", "int", False, -1),
                    SqlSkillParam(
                        "include_all_sources",
                        "1 to include all generic sources (ETW/FTrace/etc), 0 to prefer GPU metrics sources only",
                        "bool",
                        False,
                        False,
                    ),
                    SqlSkillParam("limit", "max rows", "int", False, 5000),
                ],
                tags=["nvtx", "gpu_metrics", "sampling", "phase", "sm_active", "tensor_active", "dram"],
            )

    # 19) NVTX kernel SM/memory detail
    # For a given NVTX range pattern, list every kernel that ran inside it with
    # full SM launch config and theoretical occupancy.
    if (
        schema.table_exists(nvtx_table)
        and schema.runtime_table
        and nvtx_start_col
        and nvtx_end_col
        and runtime_corr_col
        and runtime_start_for_nvtx
        and runtime_end_for_nvtx
        and corr_col
    ):
        # NVTX text expression — use alias "sv" to avoid collision with name_join aliases s/d
        sk18_text_expr = "n.text"
        sk18_nvtx_join = ""
        if string_table:
            sk18_text_id = schema.resolve_column(nvtx_table, ("textId", "nameId"))
            if sk18_text_id:
                sk18_nvtx_join = f" LEFT JOIN {_ident(string_table)} sv ON n.{_ident(sk18_text_id)} = sv.id "
                sk18_text_expr = "COALESCE(n.text, sv.value)"
        sk18_tid_join = ""
        if runtime_global_tid_col and nvtx_global_tid_col:
            sk18_tid_join = (
                f" AND n.{_ident(nvtx_global_tid_col)} = r.{_ident(runtime_global_tid_col)} "
            )

        # SM columns — resolve fresh (may be None on older nsys exports)
        sk18_bx = schema.resolve_column(kernel_table, ("blockX",))
        sk18_by = schema.resolve_column(kernel_table, ("blockY",))
        sk18_bz = schema.resolve_column(kernel_table, ("blockZ",))
        sk18_gx = schema.resolve_column(kernel_table, ("gridX",))
        sk18_gy = schema.resolve_column(kernel_table, ("gridY",))
        sk18_gz = schema.resolve_column(kernel_table, ("gridZ",))
        sk18_reg = schema.resolve_column(kernel_table, ("registersPerThread",))
        sk18_static = schema.resolve_column(kernel_table, ("staticSharedMemory",))
        sk18_dyn = schema.resolve_column(kernel_table, ("dynamicSharedMemory",))
        sk18_local = schema.resolve_column(kernel_table, ("localMemoryPerThread",))
        sk18_theoretical_occ = schema.resolve_column(
            kernel_table,
            ("theoreticalOccupancyPct", "theoreticalOccupancy", "theoretical_occupancy_pct", "theoretical_occupancy"),
        )

        if sk18_bx and sk18_by and sk18_bz:
            sk18_tpb = f"(k.{_ident(sk18_bx)} * k.{_ident(sk18_by)} * k.{_ident(sk18_bz)})"
        else:
            sk18_tpb = "NULL"
        # Prefer sqlite theoretical occupancy when present; otherwise Python side can estimate.
        if sk18_theoretical_occ:
            sk18_occ = f"ROUND(CAST(k.{_ident(sk18_theoretical_occ)} AS REAL), 3)"
        else:
            sk18_occ = "NULL"

        sk18_static_expr = f"k.{_ident(sk18_static)}" if sk18_static else "NULL"
        sk18_dyn_expr = f"k.{_ident(sk18_dyn)}" if sk18_dyn else "NULL"

        if sk18_static and sk18_dyn:
            sk18_smem = f"(COALESCE(k.{_ident(sk18_static)}, 0) + COALESCE(k.{_ident(sk18_dyn)}, 0))"
        elif sk18_static:
            sk18_smem = f"COALESCE(k.{_ident(sk18_static)}, 0)"
        elif sk18_dyn:
            sk18_smem = f"COALESCE(k.{_ident(sk18_dyn)}, 0)"
        else:
            sk18_smem = "NULL"

        sk18_device_where = (
            f"AND ({{device_id}} < 0 OR k.{_ident(device_col)} = {{device_id}}) "
            if device_col else ""
        )

        # Optional columns: grid dims, local memory (present only in real nsys exports)
        optional_cols = ""
        if sk18_gx and sk18_gy and sk18_gz:
            optional_cols += (
                f"k.{_ident(sk18_gx)} * k.{_ident(sk18_gy)} * k.{_ident(sk18_gz)} AS total_blocks, "
                f"k.{_ident(sk18_gx)} AS gridX, "
                f"k.{_ident(sk18_gy)} AS gridY, "
                f"k.{_ident(sk18_gz)} AS gridZ, "
            )
        if sk18_reg:
            optional_cols += f"k.{_ident(sk18_reg)} AS registersPerThread, "
        else:
            optional_cols += "NULL AS registersPerThread, "
        if sk18_local:
            optional_cols += f"k.{_ident(sk18_local)} AS localMemoryPerThread, "

        skill_map["nvtx_kernel_sm_detail"] = SqlSkill(
            name="nvtx_kernel_sm_detail",
            title="NVTX Kernel SM/Memory Detail",
            description=(
                "For each NVTX range matching the text pattern, list every kernel whose launch "
                "runtime API call falls inside the NVTX range, then join to GPU execution by "
                "correlationId. Includes full SM launch config (threads/block, grid, registers, "
                "shared memory, local memory). Kernels are labelled 'compute' or 'comm' (nccl). "
                "If sqlite includes per-kernel theoretical occupancy it is returned; otherwise "
                "use Python-side calculate_h100_occupancy(...) for H100."
            ),
            category="compute",
            sql=(
                f"SELECT {sk18_text_expr} AS nvtx_text, "
                f"n.{_ident(nvtx_start_col)} AS nvtx_start_ns, "
                f"n.[{_ident(nvtx_end_col)}] AS nvtx_end_ns, "
                f"{name_expr} AS kernel_name, "
                f"CASE WHEN LOWER({name_expr}) LIKE '%nccl%' THEN 'comm' ELSE 'compute' END AS kind, "
                f"k.{_ident(start_col)} AS kernel_start_ns, "
                f"k.[{_ident(end_col)}] AS kernel_end_ns, "
                f"ROUND((k.[{_ident(end_col)}] - k.{_ident(start_col)}) / 1e6, 3) AS duration_ms, "
                + (f"k.{_ident(stream_col)} AS stream_id, " if stream_col else "NULL AS stream_id, ")
                + (f"k.{_ident(device_col)} AS device_id, " if device_col else "NULL AS device_id, ")
                + f"{sk18_tpb} AS threads_per_block, "
                + optional_cols
                + f"{sk18_static_expr} AS static_shared_bytes, "
                + f"{sk18_dyn_expr} AS dynamic_shared_bytes, "
                + f"{sk18_smem} AS total_shared_bytes, "
                + f"{sk18_occ} AS occupancy_pct_estimate "
                + f"FROM {nvtx_table} n "
                + f"{sk18_nvtx_join} "
                + f"JOIN {_ident(runtime_table)} r "
                + f"  ON n.{_ident(nvtx_start_col)} <= r.{_ident(runtime_start_for_nvtx)} "
                + f"  AND n.[{_ident(nvtx_end_col)}] >= r.[{_ident(runtime_end_for_nvtx)}] "
                + sk18_tid_join
                + f"JOIN {kernel_table} k "
                + f"  ON r.{_ident(runtime_corr_col)} = k.{_ident(corr_col)} "
                + f"{name_join} "
                + f"WHERE {sk18_text_expr} LIKE '{{nvtx_text}}' "
                + f"AND {sk18_text_expr} IS NOT NULL "
                + f"AND n.[{_ident(nvtx_end_col)}] > n.{_ident(nvtx_start_col)} "
                + sk18_device_where
                + f"ORDER BY n.{_ident(nvtx_start_col)} ASC, k.{_ident(start_col)} ASC "
                + "LIMIT {limit}"
            ),
            params=[
                SqlSkillParam(
                    "nvtx_text",
                    "NVTX range text LIKE pattern, e.g. %forward% or %sample_0%",
                    "str",
                    True,
                ),
                SqlSkillParam("device_id", "CUDA deviceId, -1 means all devices", "int", False, -1),
                SqlSkillParam("limit", "max rows", "int", False, 2000),
            ],
            tags=["nvtx", "kernel", "sm", "occupancy", "shared_memory", "launch_config", "detail"],
        )

    # 20) NVTX ranges hierarchy (raw rows + parent/child relation)
    if schema.table_exists(nvtx_table):
        sk19_start_col = schema.resolve_column(nvtx_table, ("start",))
        sk19_end_col = schema.resolve_column(nvtx_table, ("end",))
        sk19_tid_col = schema.resolve_column(nvtx_table, ("globalTid", "global_tid"))
        sk19_event_type_col = schema.resolve_column(nvtx_table, ("eventType", "event_type", "type"))
        if sk19_start_col and sk19_end_col:
            sk19_text_expr = "n.text"
            sk19_nvtx_join = ""
            if string_table:
                sk19_text_id = schema.resolve_column(nvtx_table, ("textId", "nameId"))
                if sk19_text_id:
                    sk19_nvtx_join = f" LEFT JOIN {_ident(string_table)} sv19 ON n.{_ident(sk19_text_id)} = sv19.id "
                    sk19_text_expr = "COALESCE(n.text, sv19.value)"

            sk19_tid_select = f"n.{_ident(sk19_tid_col)} AS global_tid, " if sk19_tid_col else "NULL AS global_tid, "
            sk19_event_type_select = (
                f"n.{_ident(sk19_event_type_col)} AS event_type " if sk19_event_type_col else "NULL AS event_type "
            )

            skill_map["nvtx_ranges_hierarchy"] = SqlSkill(
                name="nvtx_ranges_hierarchy",
                title="NVTX Ranges Hierarchy",
                description=(
                    "List NVTX ranges as raw rows (sorted). Parent/child hierarchy is derived "
                    "in Python with an O(N) stack algorithm."
                ),
                category="nvtx",
                sql=(
                    f"SELECT n.rowid AS nvtx_rowid, "
                    f"       {sk19_text_expr} AS nvtx_text, "
                    f"       n.{_ident(sk19_start_col)} AS start_ns, "
                    f"       n.[{_ident(sk19_end_col)}] AS end_ns, "
                    f"       ROUND((n.[{_ident(sk19_end_col)}] - n.{_ident(sk19_start_col)}) / 1e6, 3) AS duration_ms, "
                    f"       {sk19_tid_select}"
                    f"       {sk19_event_type_select}"
                    f"FROM {nvtx_table} n "
                    f"{sk19_nvtx_join} "
                    f"WHERE {sk19_text_expr} IS NOT NULL "
                    f"  AND n.[{_ident(sk19_end_col)}] > n.{_ident(sk19_start_col)} "
                    f"  AND ('{{nvtx_text}}' = '%' OR {sk19_text_expr} LIKE '{{nvtx_text}}') "
                    "ORDER BY global_tid ASC, start_ns ASC, end_ns DESC "
                    "LIMIT {limit}"
                ),
                params=[
                    SqlSkillParam("nvtx_text", "NVTX text LIKE pattern, default % for all ranges", "str", False, "%"),
                    SqlSkillParam("top_level_only", "1 to keep only root ranges, 0 for full hierarchy", "bool", False, False),
                    SqlSkillParam("limit", "max rows", "int", False, 5000),
                ],
                tags=["nvtx", "hierarchy", "parent", "child", "ranges", "raw"],
            )

    return skill_map


class NsysSqlSkillEngine:
    """Built-in SQL skill runner for Nsight Systems SQLite exports."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.schema = NsightSchema(conn)
        self._skills = _build_builtin_skills(self.schema)
        # Cache schema fragments for internal kernel queries (avoids rebuilding per call)
        self._kernel_table = _ident(self.schema.kernel_table or "CUPTI_ACTIVITY_KIND_KERNEL")
        self._start_col = self.schema.resolve_column(self._kernel_table, ("start",))
        self._end_col = self.schema.resolve_column(self._kernel_table, ("end",))
        self._device_col = self.schema.resolve_column(self._kernel_table, ("deviceId",))
        self._stream_col = self.schema.resolve_column(self._kernel_table, ("streamId",))
        _string_table = self.schema.string_table
        _short_col = self.schema.resolve_column(self._kernel_table, ("shortName",))
        _demangled_col = self.schema.resolve_column(self._kernel_table, ("demangledName",))
        self._name_expr = "CAST(k.shortName AS TEXT)"
        self._name_join = ""
        if _string_table and _short_col:
            _st = _ident(_string_table)
            self._name_join = f"JOIN {_st} s ON k.{_ident(_short_col)} = s.id"
            self._name_expr = "s.value"
        if _string_table and _demangled_col:
            _st = _ident(_string_table)
            self._name_join += f" LEFT JOIN {_st} d ON k.{_ident(_demangled_col)} = d.id"
            self._name_expr = "COALESCE(d.value, s.value)" if _short_col else "d.value"

    def list_skills(self) -> List[str]:
        return sorted(self._skills.keys())

    def get_skill(self, name: str) -> Optional[SqlSkill]:
        return self._skills.get(str(name))

    def describe_skill(self, name: str) -> Optional[Dict[str, object]]:
        skill = self.get_skill(name)
        if skill is None:
            return None
        return {
            "name": skill.name,
            "title": skill.title,
            "description": skill.description,
            "category": skill.category,
            "tags": list(skill.tags),
            "params": [
                {
                    "name": p.name,
                    "description": p.description,
                    "type": p.type,
                    "required": bool(p.required),
                    "default": p.default,
                }
                for p in skill.params
            ],
        }

    def describe_skills(self) -> List[Dict[str, object]]:
        names = self.list_skills()
        result: List[Dict[str, object]] = []
        for name in names:
            payload = self.describe_skill(name)
            if payload is not None:
                result.append(payload)
        return result

    @staticmethod
    def _as_bool(value: object, default: bool = False) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    def _build_nvtx_hierarchy(self, rows: Sequence[Dict[str, object]], *, top_level_only: bool) -> List[Dict[str, object]]:
        """
        Build NVTX parent/child hierarchy in O(N) using per-thread stacks.
        Input rows must be sorted by (global_tid, start_ns asc, end_ns desc).
        """
        stacks: Dict[str, List[Dict[str, object]]] = {}
        out: List[Dict[str, object]] = []

        for raw in rows:
            row = dict(raw)
            tid_key = str(row.get("global_tid"))
            if not tid_key:
                tid_key = "__none__"
            stack = stacks.setdefault(tid_key, [])

            start_ns = int(row.get("start_ns") or 0)
            end_ns = int(row.get("end_ns") or 0)
            if end_ns <= start_ns:
                continue

            # Pop finished/non-containing ancestors.
            while stack and (start_ns >= int(stack[-1]["end_ns"]) or end_ns > int(stack[-1]["end_ns"])):
                stack.pop()

            parent = stack[-1] if stack else None
            depth = len(stack)
            row["depth"] = depth
            row["parent_nvtx_text"] = parent.get("nvtx_text") if parent else None
            row["parent_start_ns"] = parent.get("start_ns") if parent else None
            row["parent_end_ns"] = parent.get("end_ns") if parent else None

            if (not top_level_only) or depth == 0:
                out.append(row)

            stack.append(
                {
                    "nvtx_text": row.get("nvtx_text"),
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                }
            )
        return out

    def execute(self, name: str, **kwargs) -> List[Dict[str, object]]:
        skill_name = str(name)
        skill = self.get_skill(skill_name)
        if skill is None:
            raise KeyError(f"Unknown SQL skill: {name}")
        rows = skill.execute(self.conn, **kwargs)
        if skill_name == "nvtx_ranges_hierarchy":
            return self._build_nvtx_hierarchy(
                rows,
                top_level_only=self._as_bool(kwargs.get("top_level_only"), default=False),
            )
        return rows

    def execute_kernel_occupancy_estimate_h100(self, **kwargs) -> List[Dict[str, object]]:
        """
        Execute `kernel_occupancy_estimate` and attach H100(sm_90) occupancy estimate in Python.
        """
        rows = self.execute("kernel_occupancy_estimate", **kwargs)
        result: List[Dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["occupancy_pct_h100_estimate"] = calculate_h100_occupancy(
                item.get("threads_per_block"),
                item.get("registersPerThread"),
                item.get("total_shared_bytes"),
            )
            result.append(item)
        return result

    def execute_nvtx_kernel_sm_detail_h100(self, **kwargs) -> List[Dict[str, object]]:
        """
        Execute `nvtx_kernel_sm_detail` and attach H100(sm_90) occupancy estimate in Python.
        """
        rows = self.execute("nvtx_kernel_sm_detail", **kwargs)
        result: List[Dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["occupancy_pct_h100_estimate"] = calculate_h100_occupancy(
                item.get("threads_per_block"),
                item.get("registersPerThread"),
                item.get("total_shared_bytes"),
            )
            result.append(item)
        return result

    def _kernel_rows_for_window(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        limit: int = 2_000_000,
    ) -> List[Dict[str, object]]:
        skill = self.get_skill("kernel_map")
        if skill is None:
            return []
        return skill.execute(
            self.conn,
            device_id=int(device_id),
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            limit=int(limit),
        )

    def _build_kernel_where(
        self,
        device_id: int,
        start_ns: int,
        end_ns: int,
    ) -> Tuple[str, List[object]]:
        """Build WHERE clause fragments and positional params for kernel queries."""
        parts = ["1=1"]
        params: List[object] = []
        if self._device_col and device_id >= 0:
            parts.append(f"k.{_ident(self._device_col)} = ?")
            params.append(device_id)
        if start_ns >= 0:
            parts.append(f"k.{_ident(self._start_col)} >= ?")
            params.append(start_ns)
        if end_ns >= 0:
            parts.append(f"k.[{_ident(self._end_col)}] <= ?")
            params.append(end_ns)
        return " AND ".join(parts), params

    def _fetch_minimal_kernel_intervals(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        limit: int = 2_000_000,
    ) -> List[Tuple[int, int, str]]:
        """Fetch (start_ns, end_ns, kernel_name) tuples — 3-column query for overlap analysis."""
        if not self._start_col or not self._end_col:
            return []
        where, params = self._build_kernel_where(device_id, start_ns, end_ns)
        params.append(limit)
        sql = (
            f"SELECT k.{_ident(self._start_col)}, k.[{_ident(self._end_col)}], {self._name_expr} "
            f"FROM {self._kernel_table} k {self._name_join} "
            f"WHERE {where} "
            f"ORDER BY k.{_ident(self._start_col)} ASC LIMIT ?"
        )
        result: List[Tuple[int, int, str]] = []
        for row in self.conn.execute(sql, params):
            s, e, name = int(row[0] or 0), int(row[1] or 0), str(row[2] or "")
            if e > s:
                result.append((s, e, name))
        return result

    def _fetch_kernel_spans(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        limit: int = 2_000_000,
    ) -> List[Tuple[int, int]]:
        """Fetch (start_ns, end_ns) pairs — minimal 2-column query for utilization interval merge."""
        if not self._start_col or not self._end_col:
            return []
        where, params = self._build_kernel_where(device_id, start_ns, end_ns)
        params.append(limit)
        sql = (
            f"SELECT k.{_ident(self._start_col)}, k.[{_ident(self._end_col)}] "
            f"FROM {self._kernel_table} k "
            f"WHERE {where} "
            f"ORDER BY k.{_ident(self._start_col)} ASC LIMIT ?"
        )
        result: List[Tuple[int, int]] = []
        for row in self.conn.execute(sql, params):
            s, e = int(row[0] or 0), int(row[1] or 0)
            if e > s:
                result.append((s, e))
        return result

    def _query_kernel_aggregate(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
    ) -> Dict[str, object]:
        """Single SQL aggregate: kernel_count, min/max timestamps, sum duration, stream count."""
        if not self._start_col or not self._end_col:
            return {}
        where, params = self._build_kernel_where(device_id, start_ns, end_ns)
        stream_select = (
            f", COUNT(DISTINCT k.{_ident(self._stream_col)}) AS stream_count"
            if self._stream_col
            else ""
        )
        sql = (
            f"SELECT COUNT(*) AS kernel_count, "
            f"MIN(k.{_ident(self._start_col)}) AS min_start_ns, "
            f"MAX(k.[{_ident(self._end_col)}]) AS max_end_ns, "
            f"SUM(k.[{_ident(self._end_col)}] - k.{_ident(self._start_col)}) AS sum_duration_ns"
            f"{stream_select} "
            f"FROM {self._kernel_table} k "
            f"WHERE {where}"
        )
        row = self.conn.execute(sql, params).fetchone()
        if row is None:
            return {}
        result: Dict[str, object] = {
            "kernel_count": int(row[0] or 0),
            "min_start_ns": int(row[1] or 0),
            "max_end_ns": int(row[2] or 0),
            "sum_duration_ns": int(row[3] or 0),
        }
        if self._stream_col:
            result["stream_count"] = int(row[4] or 0)
        return result

    def analyze_compute_comm_overlap(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        limit: int = 2_000_000,
    ) -> Dict[str, object]:
        triples = self._fetch_minimal_kernel_intervals(
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
            limit=limit,
        )
        compute_iv: List[Tuple[int, int]] = []
        comm_iv: List[Tuple[int, int]] = []
        for s, e, name in triples:
            if _is_nccl_kernel(name):
                comm_iv.append((s, e))
            else:
                compute_iv.append((s, e))

        compute_m = _merge_intervals(compute_iv)
        comm_m = _merge_intervals(comm_iv)
        compute_ns = _covered_ns(compute_m)
        comm_ns = _covered_ns(comm_m)
        overlap_ns = _intersection_coverage_ns(compute_m, comm_m)

        compute_only_ns = max(0, compute_ns - overlap_ns)
        comm_only_ns = max(0, comm_ns - overlap_ns)
        overlap_pct_of_comm = (100.0 * overlap_ns / comm_ns) if comm_ns > 0 else 0.0
        overlap_pct_of_compute = (100.0 * overlap_ns / compute_ns) if compute_ns > 0 else 0.0

        return {
            "device_id": int(device_id),
            "window": {"start_ns": int(start_ns), "end_ns": int(end_ns)},
            "kernel_rows": len(triples),
            "compute_intervals": len(compute_m),
            "comm_intervals": len(comm_m),
            "compute_only_ms": round(compute_only_ns / 1e6, 6),
            "comm_only_ms": round(comm_only_ns / 1e6, 6),
            "overlap_ms": round(overlap_ns / 1e6, 6),
            "compute_total_ms": round(compute_ns / 1e6, 6),
            "comm_total_ms": round(comm_ns / 1e6, 6),
            "overlap_pct_of_comm": round(overlap_pct_of_comm, 4),
            "overlap_pct_of_compute": round(overlap_pct_of_compute, 4),
        }

    def summarize_gpu_kernels(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        top_k: int = 10,
        limit: int = 2_000_000,
    ) -> Dict[str, object]:
        agg = self._query_kernel_aggregate(
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        kernel_count = int(agg.get("kernel_count") or 0)
        if kernel_count == 0:
            return {
                "device_id": int(device_id),
                "window": {"start_ns": int(start_ns), "end_ns": int(end_ns)},
                "kernel_rows": 0,
                "timing": {},
                "top_kernels": [],
                "stream_count": 0,
            }

        # Fetch minimal (start, end) spans for interval-merge utilization calculation
        spans = self._fetch_kernel_spans(
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
            limit=limit,
        )

        span_ns = int(agg.get("max_end_ns", 0)) - int(agg.get("min_start_ns", 0))
        sum_kernel_ns = int(agg.get("sum_duration_ns") or 0)
        stream_count = int(agg.get("stream_count") or 0)

        merged = _merge_intervals(spans)
        busy_ns = _covered_ns(merged)
        idle_ns = max(0, span_ns - busy_ns)
        util_pct = (100.0 * busy_ns / span_ns) if span_ns > 0 else 0.0

        top_rows = self.execute("top_kernels", device_id=device_id, limit=max(1, int(top_k)))

        return {
            "device_id": int(device_id),
            "window": {"start_ns": int(start_ns), "end_ns": int(end_ns)},
            "kernel_rows": kernel_count,
            "stream_count": stream_count,
            "timing": {
                "span_ms": round(span_ns / 1e6, 6),
                "busy_ms": round(busy_ns / 1e6, 6),
                "idle_ms": round(idle_ns / 1e6, 6),
                "utilization_pct": round(util_pct, 4),
                "sum_kernel_ms": round(sum_kernel_ns / 1e6, 6),
            },
            "top_kernels": top_rows,
        }

    def detect_iterations(
        self,
        *,
        marker: str = "sample_0",
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        top_level_only: bool = True,
        limit: int = 2000,
    ) -> List[Dict[str, object]]:
        return detect_iterations(
            self.conn,
            schema=self.schema,
            marker=marker,
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
            top_level_only=top_level_only,
            limit=limit,
        )

    def analyze_per_iteration_overlap(
        self,
        *,
        marker: str = "sample_0",
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        top_level_only: bool = True,
        limit: int = 2000,
    ) -> List[Dict[str, object]]:
        """Per-iteration compute/comm/overlap breakdown.

        Returns one dict per iteration with fields from detect_iterations() extended by:
        compute_ms, comm_ms, overlap_ms, comm_pct, kernel_count.
        """
        iterations = self.detect_iterations(
            marker=marker,
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
            top_level_only=top_level_only,
            limit=limit,
        )
        result: List[Dict[str, object]] = []
        for it in iterations:
            it_start = int(it.get("start_ns") or 0)
            it_end = int(it.get("end_ns") or 0)
            if it_end <= it_start:
                continue
            triples = self._fetch_minimal_kernel_intervals(
                device_id=device_id,
                start_ns=it_start,
                end_ns=it_end,
            )
            compute_iv: List[Tuple[int, int]] = []
            comm_iv: List[Tuple[int, int]] = []
            for s, e, name in triples:
                if _is_nccl_kernel(name):
                    comm_iv.append((s, e))
                else:
                    compute_iv.append((s, e))
            compute_m = _merge_intervals(compute_iv)
            comm_m = _merge_intervals(comm_iv)
            compute_ns_val = _covered_ns(compute_m)
            comm_ns_val = _covered_ns(comm_m)
            overlap_ns_val = _intersection_coverage_ns(compute_m, comm_m)
            total_active_ns = compute_ns_val + comm_ns_val - overlap_ns_val
            entry = dict(it)
            entry.update({
                "compute_ms": round(compute_ns_val / 1e6, 3),
                "comm_ms": round(comm_ns_val / 1e6, 3),
                "overlap_ms": round(overlap_ns_val / 1e6, 3),
                "comm_pct": round(100.0 * comm_ns_val / total_active_ns, 2) if total_active_ns > 0 else 0.0,
                "kernel_count": len(triples),
            })
            result.append(entry)
        return result

    def detect_iteration_outliers(
        self,
        *,
        marker: str = "sample_0",
        device_id: int = -1,
        threshold_sigma: float = 2.0,
        limit: int = 2000,
    ) -> Dict[str, object]:
        """Statistical outlier detection on iteration durations.

        Returns:
            {
                "stats": {count, mean_ms, median_ms, std_ms, p95_ms, p99_ms},
                "outliers": [{iteration, duration_ms, deviation_sigma}, ...]
            }
        """
        iterations = self.detect_iterations(
            marker=marker,
            device_id=device_id,
            start_ns=-1,
            end_ns=-1,
            top_level_only=True,
            limit=limit,
        )
        durations = [
            float(it.get("duration_ms") or 0)
            for it in iterations
            if (it.get("duration_ms") or 0) > 0
        ]
        if not durations:
            return {"stats": {}, "outliers": []}

        n = len(durations)
        mean = sum(durations) / n
        sorted_d = sorted(durations)
        if n % 2 == 0:
            median = (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2.0
        else:
            median = sorted_d[n // 2]
        variance = sum((x - mean) ** 2 for x in durations) / max(1, n - 1)
        std = variance ** 0.5
        p95 = sorted_d[min(int(0.95 * n), n - 1)]
        p99 = sorted_d[min(int(0.99 * n), n - 1)]

        stats: Dict[str, object] = {
            "count": n,
            "mean_ms": round(mean, 3),
            "median_ms": round(median, 3),
            "std_ms": round(std, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
        }

        outliers: List[Dict[str, object]] = []
        for i, it in enumerate(iterations):
            d = float(it.get("duration_ms") or 0)
            if d <= 0:
                continue
            sigma = abs(d - median) / std if std > 0 else 0.0
            if sigma > threshold_sigma:
                outliers.append({
                    "iteration": i,
                    "duration_ms": round(d, 3),
                    "deviation_sigma": round(sigma, 3),
                })
        return {"stats": stats, "outliers": outliers}
