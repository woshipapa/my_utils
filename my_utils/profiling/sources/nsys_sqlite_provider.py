from __future__ import annotations

import json
import re
import sqlite3
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..metrics.metrics_provider import BaseMetricsProvider, ProviderCapabilities
from ..metrics.metrics_types import MetricEvent
from .nsys_mfu import compute_mfu_single, infer_peak_tflops
from .nsys_schema_adapter import NsightSchema, NsysVersionInfo, detect_nsys_version
from .nsys_sql_skills import NsysSqlSkillEngine


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _decode_global_tid(global_tid: int) -> Tuple[int, int]:
    # Nsight Systems official SQLite examples:
    # pid = globalTid / 0x1000000 % 0x1000000
    # tid = globalTid % 0x1000000
    pid = (int(global_tid) // 0x1000000) % 0x1000000
    tid = int(global_tid) % 0x1000000
    return int(pid), int(tid)


def _decode_global_pid(global_pid: int) -> int:
    return int((int(global_pid) // 0x1000000) % 0x1000000)


CUDA_MEMCPY_KIND_MAP: Dict[int, str] = {
    0: "unknown",
    1: "htod",
    2: "dtoh",
    3: "htoa",
    4: "atoh",
    5: "atoa",
    6: "atod",
    7: "dtoa",
    8: "dtod",
    9: "htoh",
    10: "ptop",
    11: "uvm_htod",
    12: "uvm_dtoh",
    13: "uvm_dtod",
}


CUDA_MEMORY_KIND_MAP: Dict[int, str] = {
    0: "pageable",
    1: "pinned",
    2: "device",
    3: "array",
    4: "managed",
    5: "device_static",
    6: "managed_static",
    7: "unknown",
}


CUDA_MEMPOOL_TYPE_MAP: Dict[int, str] = {
    0: "invalid",
    1: "local",
    2: "imported",
}


CUDA_MEMPOOL_OPERATION_TYPE_MAP: Dict[int, str] = {
    0: "invalid",
    1: "created",
    2: "destroyed",
    3: "trimmed",
}


CUDA_EVENT_CLASS_MAP: Dict[int, str] = {
    0: "cuda_runtime",
    1: "cuda_driver",
    13: "cuda_egl_driver",
    28: "cudnn",
    29: "cublas",
    33: "cudnn_start",
    34: "cudnn_finish",
    35: "cublas_start",
    36: "cublas_finish",
    67: "cudabacktrace",
    77: "cuda_graph_node_creation",
}


NVTX_EVENT_TYPE_MAP: Dict[int, str] = {
    33: "category",
    34: "mark",
    39: "thread",
    59: "push_pop_range",
    60: "start_end_range",
    75: "domain_create",
    76: "domain_destroy",
}


GPU_METRIC_NAME_HINTS: Dict[str, str] = {
    "sms_active": "compute.gpu.sm.active",
    "sm_active": "compute.gpu.sm.active",
    "sms_throughput": "compute.gpu.sm.throughput",
    "sm_throughput": "compute.gpu.sm.throughput",
    "dram_throughput": "memory.gpu.dram.throughput",
    "dram_read_throughput": "memory.gpu.dram.read_throughput",
    "dram_write_throughput": "memory.gpu.dram.write_throughput",
    "pcie_tx_throughput": "io.gpu.pcie.tx_throughput",
    "pcie_rx_throughput": "io.gpu.pcie.rx_throughput",
    "nvlink_tx_throughput": "io.gpu.nvlink.tx_throughput",
    "nvlink_rx_throughput": "io.gpu.nvlink.rx_throughput",
    "tensor_active": "compute.gpu.tensor.active",
    "tensor_throughput": "compute.gpu.tensor.throughput",
    "gpu_utilization": "compute.gpu.utilization",
    "gpu_active": "compute.gpu.active",
}


AUTO_DURATION_SKIP_TABLES = {
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
    "CUPTI_ACTIVITY_KIND_MEMSET",
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
    "CUDA_GPU_MEMORY_USAGE_EVENTS",
    "CUDA_GPU_MEMORY_POOL_EVENTS",
    "NVTX_EVENTS",
    "OSRT_API",
    "GPU_METRICS",
    "GENERIC_EVENTS",
    "GENERIC_EVENT_TYPES",
    "GENERIC_EVENT_SOURCES",
    "TARGET_INFO_GPU_METRICS",
    "TARGET_INFO_NETWORK_METRICS",
    "NET_NIC_METRIC",
    "NET_IB_SWITCH_METRIC",
    "PMU_EVENTS",
    "PMU_EVENT_COUNTERS",
    "PMU_EVENT_REQUESTS",
    "StringIds",
    "ThreadNames",
    "PROCESSES",
    "META_DATA_EXPORT",
    "EXPORT_META_DATA",
}


class NsysSqliteMetricsProvider(BaseMetricsProvider):
    """
    Parse Nsight Systems SQLite export into canonical metric events.

    Based on official SQLite schema docs and common query examples:
    - CUPTI runtime/kernel/memcpy/memset/sync tables
    - CUDA memory usage/pool tables
    - GPU_METRICS and GENERIC_EVENTS
    - network metrics (NIC / IB switch)
    - PMU and UM page-fault tables
    - NVTX_EVENTS and optional OSRT_API
    - StringIds for name resolution
    - globalTid/globalPid decode for pid/tid tags
    """

    provider_id = "nsys_sqlite"
    _capabilities = ProviderCapabilities(
        provider_type="nsys_sqlite",
        source_mode="offline",
        metric_prefixes=["latency", "memory", "io", "compute", "comm", "perf", "calls"],
        dimensions=["pid", "tid", "rank", "step", "stream", "kernel", "runtime_api"],
        supports_incremental=True,
        supports_rank_scope=True,
        supports_step_scope=True,
        notes="Schema-adaptive parser for Nsight Systems SQLite exports.",
    )

    _STEP_PATTERNS = (
        re.compile(r"\bstep(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE),
        re.compile(r"\biter(?:ation)?(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE),
    )
    _RANK_PATTERNS = (re.compile(r"\brank(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE),)

    def __init__(
        self,
        sqlite_path: str,
        *,
        provider_id: Optional[str] = None,
        enabled: bool = True,
        include_runtime: bool = True,
        include_kernels: bool = True,
        include_memcpy: bool = True,
        include_memset: bool = True,
        include_sync: bool = True,
        include_nvtx: bool = True,
        include_memory_usage: bool = True,
        include_osrt: bool = False,
        include_gpu_metrics: bool = True,
        include_generic_events: bool = True,
        include_network_metrics: bool = True,
        include_pmu_metrics: bool = True,
        include_cuda_um: bool = True,
        include_auto_duration_tables: bool = True,
        parse_step_from_nvtx: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.sqlite_path = Path(sqlite_path)

        self.include_runtime = bool(include_runtime)
        self.include_kernels = bool(include_kernels)
        self.include_memcpy = bool(include_memcpy)
        self.include_memset = bool(include_memset)
        self.include_sync = bool(include_sync)
        self.include_nvtx = bool(include_nvtx)
        self.include_memory_usage = bool(include_memory_usage)
        self.include_osrt = bool(include_osrt)
        self.include_gpu_metrics = bool(include_gpu_metrics)
        self.include_generic_events = bool(include_generic_events)
        self.include_network_metrics = bool(include_network_metrics)
        self.include_pmu_metrics = bool(include_pmu_metrics)
        self.include_cuda_um = bool(include_cuda_um)
        self.include_auto_duration_tables = bool(include_auto_duration_tables)
        self.parse_step_from_nvtx = bool(parse_step_from_nvtx)

        self._last_rowid: Dict[str, int] = {}
        self._full_table_consumed: set[str] = set()

        self._table_columns_cache: Dict[str, set[str]] = {}
        self._string_id_cache: Optional[Dict[int, str]] = None
        self._thread_name_cache: Optional[Dict[int, str]] = None
        self._process_name_cache: Optional[Dict[int, str]] = None

        # (pid, correlationId) -> runtime api info
        # pid can be None when not available in source row.
        self._runtime_by_correlation: Dict[
            Tuple[Optional[str], int], Dict[str, str]
        ] = {}
        self._schema_meta: Optional[Dict[str, str]] = None
        self._version_info: Optional[NsysVersionInfo] = None
        self._schema_tags_cache: Optional[Dict[str, str]] = None
        self._gpu_metric_info_cache: Optional[
            Dict[Tuple[Optional[int], Optional[int]], Dict[str, str]]
        ] = None
        self._network_metric_info_cache: Optional[
            Dict[Tuple[int, int], Dict[str, str]]
        ] = None
        self._generic_event_type_cache: Optional[Dict[int, Dict[str, str]]] = None

    def _load_schema_meta(self, conn: sqlite3.Connection) -> Dict[str, str]:
        if self._schema_meta is not None:
            return self._schema_meta

        result: Dict[str, str] = {}
        for table_name in ("META_DATA_EXPORT", "EXPORT_META_DATA"):
            if not self._table_exists(conn, table_name):
                continue
            cols = self._table_columns(conn, table_name)
            if not {"name", "value"}.issubset(cols):
                continue
            try:
                rows = conn.execute(f"SELECT name, value FROM {table_name};").fetchall()
            except sqlite3.Error:
                continue
            if not rows:
                continue
            result["meta_table"] = table_name
            for row in rows:
                key = str(row["name"] or "").strip()
                if not key:
                    continue
                result[key] = str(row["value"] or "")
            break

        self._schema_meta = result
        self._version_info = detect_nsys_version(result)
        self._schema_tags_cache = None  # invalidate on schema reload
        return result

    def _schema_tags(self) -> Dict[str, str]:
        if self._schema_tags_cache is not None:
            return self._schema_tags_cache
        meta = self._schema_meta or {}
        tags: Dict[str, str] = {}
        if "EXPORT_SCHEMA_VERSION" in meta:
            tags["export_schema_version"] = meta["EXPORT_SCHEMA_VERSION"]
        if "EXPORT_SCHEMA_VERSION_MAJOR" in meta:
            tags["export_schema_major"] = meta["EXPORT_SCHEMA_VERSION_MAJOR"]
        if "EXPORT_SCHEMA_VERSION_MINOR" in meta:
            tags["export_schema_minor"] = meta["EXPORT_SCHEMA_VERSION_MINOR"]
        if "EXPORT_SCHEMA_VERSION_MICRO" in meta:
            tags["export_schema_micro"] = meta["EXPORT_SCHEMA_VERSION_MICRO"]
        if "meta_table" in meta:
            tags["export_meta_table"] = meta["meta_table"]
        version_info = self._version_info
        if version_info is not None:
            if version_info.exporter_version:
                tags["nsys_exporter_version"] = version_info.exporter_version
            if version_info.export_schema_version:
                tags["nsys_export_schema_version"] = version_info.export_schema_version
            tags["nsys_adapter_family"] = version_info.adapter_family
        self._schema_tags_cache = tags
        return tags

    @staticmethod
    def _row_int(row: sqlite3.Row, *keys: str) -> Optional[int]:
        for key in keys:
            if key in row.keys():
                value = _safe_int(row[key])
                if value is not None:
                    return value
        return None

    @staticmethod
    def _row_float(row: sqlite3.Row, *keys: str) -> Optional[float]:
        for key in keys:
            if key in row.keys():
                value = _safe_float(row[key])
                if value is not None:
                    return value
        return None

    @staticmethod
    def _get_node_id(row: sqlite3.Row) -> Optional[str]:
        if "_rowid" in row.keys() and row["_rowid"] is not None:
            return str(row["_rowid"])
        return None

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
            (table_name,),
        ).fetchone()
        return row is not None

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        key = table_name.lower()
        if key in self._table_columns_cache:
            return self._table_columns_cache[key]
        if not self._table_exists(conn, table_name):
            self._table_columns_cache[key] = set()
            return set()
        rows = conn.execute(f"PRAGMA table_info({table_name});").fetchall()
        cols = {str(row["name"]) for row in rows}
        self._table_columns_cache[key] = cols
        return cols

    def _ensure_string_cache(self, conn: sqlite3.Connection) -> None:
        if self._string_id_cache is not None:
            return
        self._string_id_cache = {}
        if not self._table_exists(conn, "StringIds"):
            return
        try:
            for row in conn.execute("SELECT id, value FROM StringIds;").fetchall():
                sid = _safe_int(row["id"])
                if sid is None:
                    continue
                self._string_id_cache[sid] = str(row["value"] or "")
        except sqlite3.Error:
            self._string_id_cache = {}

    def _resolve_string(self, conn: sqlite3.Connection, raw_value) -> str:
        if raw_value is None:
            return ""
        if isinstance(raw_value, bytes):
            try:
                return raw_value.decode("utf-8", errors="replace")
            except Exception:
                return str(raw_value)
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if text and not text.isdigit():
                return text
        maybe_id = _safe_int(raw_value)
        if maybe_id is None:
            return str(raw_value)
        self._ensure_string_cache(conn)
        if self._string_id_cache is None:
            return str(raw_value)
        return self._string_id_cache.get(maybe_id, str(raw_value))

    def _ensure_thread_process_names(self, conn: sqlite3.Connection) -> None:
        if self._thread_name_cache is not None and self._process_name_cache is not None:
            return

        self._thread_name_cache = {}
        self._process_name_cache = {}

        if self._table_exists(conn, "ThreadNames"):
            cols = self._table_columns(conn, "ThreadNames")
            if {"globalTid", "nameId"}.issubset(cols):
                for row in conn.execute(
                    "SELECT globalTid, nameId FROM ThreadNames;"
                ).fetchall():
                    gtid = _safe_int(row["globalTid"])
                    if gtid is None:
                        continue
                    name = self._resolve_string(conn, row["nameId"])
                    if name:
                        self._thread_name_cache[gtid] = name

        if self._table_exists(conn, "PROCESSES"):
            cols = self._table_columns(conn, "PROCESSES")
            pid_col = (
                "globalPid"
                if "globalPid" in cols
                else ("pid" if "pid" in cols else None)
            )
            name_col = (
                "name" if "name" in cols else ("nameId" if "nameId" in cols else None)
            )
            if pid_col and name_col:
                for row in conn.execute(
                    f"SELECT {pid_col} AS _pid, {name_col} AS _name FROM PROCESSES;"
                ).fetchall():
                    gpid = _safe_int(row["_pid"])
                    if gpid is None:
                        continue
                    name = self._resolve_string(conn, row["_name"])
                    if name:
                        self._process_name_cache[gpid] = name

    def _rows_incremental(
        self,
        conn: sqlite3.Connection,
        table: str,
        *,
        required_columns: Sequence[str],
    ) -> List[sqlite3.Row]:
        if not self._table_exists(conn, table):
            return []
        cols = self._table_columns(conn, table)
        select_cols = [col for col in required_columns if col in cols]
        if not select_cols:
            return []

        col_sql = ", ".join(select_cols)
        last = self._last_rowid.get(table, 0)
        try:
            query = f"SELECT rowid AS _rowid, {col_sql} FROM {table} WHERE rowid > ? ORDER BY rowid ASC;"
            rows = conn.execute(query, (last,)).fetchall()
            if rows:
                self._last_rowid[table] = int(rows[-1]["_rowid"])
            return rows
        except sqlite3.Error:
            if table in self._full_table_consumed:
                return []
            rows = conn.execute(f"SELECT {col_sql} FROM {table};").fetchall()
            self._full_table_consumed.add(table)
            return rows

    def _list_tables(self, conn: sqlite3.Connection) -> List[str]:
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
            ).fetchall()
            return [str(row["name"]) for row in rows if row["name"]]
        except sqlite3.Error:
            return []

    @staticmethod
    def _normalize_token(raw: str) -> str:
        text = str(raw or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_") or "unknown"

    @staticmethod
    def _infer_unit_from_name(raw: str) -> str:
        text = NsysSqliteMetricsProvider._normalize_token(raw)
        if "percent" in text or text.endswith("_pct") or "_pct_" in text:
            return "percent"
        if "bytes_per_second" in text:
            return "bytes/s"
        if "byte" in text and "second" in text:
            return "bytes/s"
        if "bytes" in text or text.endswith("_bytes"):
            return "bytes"
        if "throughput" in text and "gb" in text:
            return "GB/s"
        return ""

    @staticmethod
    def _canonical_gpu_metric_name(raw_name: str) -> str:
        token = NsysSqliteMetricsProvider._normalize_token(raw_name)
        if token in GPU_METRIC_NAME_HINTS:
            return GPU_METRIC_NAME_HINTS[token]
        if "dram" in token and (
            "throughput" in token or "bandwidth" in token or token.endswith("_bw")
        ):
            return "memory.gpu.dram.throughput"
        if "sm" in token and (
            "active" in token or "throughput" in token or "utilization" in token
        ):
            return "compute.gpu.sm.utilization"
        if "pcie" in token and ("throughput" in token or "bandwidth" in token):
            if "tx" in token:
                return "io.gpu.pcie.tx_throughput"
            if "rx" in token:
                return "io.gpu.pcie.rx_throughput"
            return "io.gpu.pcie.throughput"
        if "nvlink" in token and ("throughput" in token or "bandwidth" in token):
            if "tx" in token:
                return "io.gpu.nvlink.tx_throughput"
            if "rx" in token:
                return "io.gpu.nvlink.rx_throughput"
            return "io.gpu.nvlink.throughput"
        if "tensor" in token and ("active" in token or "throughput" in token):
            return "compute.gpu.tensor.throughput"
        return f"external.nsys.gpu_metric.{token}"

    @staticmethod
    def _safe_json_load(raw) -> Optional[dict]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            try:
                raw = raw.decode("utf-8", errors="replace")
            except Exception:
                return None
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
        except Exception:
            return None
        if isinstance(obj, dict):
            return obj
        return None

    @staticmethod
    def _flatten_numeric_payload(payload, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                sub_key = NsysSqliteMetricsProvider._normalize_token(str(key))
                next_prefix = f"{prefix}.{sub_key}" if prefix else sub_key
                out.update(
                    NsysSqliteMetricsProvider._flatten_numeric_payload(
                        value, next_prefix
                    )
                )
            return out
        if isinstance(payload, list):
            # Keep compact output by skipping array expansion.
            return out
        value = _safe_float(payload)
        if value is None:
            return out
        name = prefix or "value"
        out[name] = float(value)
        return out

    def _ensure_gpu_metric_info(self, conn: sqlite3.Connection) -> None:
        if self._gpu_metric_info_cache is not None:
            return
        self._gpu_metric_info_cache = {}
        if not self._table_exists(conn, "TARGET_INFO_GPU_METRICS"):
            return

        cols = self._table_columns(conn, "TARGET_INFO_GPU_METRICS")
        type_col = "typeId" if "typeId" in cols else None
        metric_col = "metricId" if "metricId" in cols else None
        name_col = (
            "metricName"
            if "metricName" in cols
            else ("name" if "name" in cols else None)
        )
        if name_col is None:
            return

        select_cols = []
        if type_col:
            select_cols.append(f"{type_col} AS _type_id")
        if metric_col:
            select_cols.append(f"{metric_col} AS _metric_id")
        select_cols.append(f"{name_col} AS _name")
        query = f"SELECT {', '.join(select_cols)} FROM TARGET_INFO_GPU_METRICS;"
        try:
            rows = conn.execute(query).fetchall()
        except sqlite3.Error:
            return

        for row in rows:
            type_id = _safe_int(row["_type_id"]) if "_type_id" in row.keys() else None
            metric_id = (
                _safe_int(row["_metric_id"]) if "_metric_id" in row.keys() else None
            )
            metric_name = str(row["_name"] or "").strip()
            if not metric_name:
                continue
            self._gpu_metric_info_cache[(type_id, metric_id)] = {
                "metric_name": metric_name
            }
            if metric_id is not None:
                self._gpu_metric_info_cache.setdefault(
                    (None, metric_id), {"metric_name": metric_name}
                )

    def _resolve_gpu_metric_name(
        self, conn: sqlite3.Connection, type_id: Optional[int], metric_id: Optional[int]
    ) -> str:
        self._ensure_gpu_metric_info(conn)
        cache = self._gpu_metric_info_cache or {}
        if (type_id, metric_id) in cache:
            return cache[(type_id, metric_id)]["metric_name"]
        if (None, metric_id) in cache:
            return cache[(None, metric_id)]["metric_name"]
        t = "none" if type_id is None else str(type_id)
        m = "none" if metric_id is None else str(metric_id)
        return f"type_{t}_metric_{m}"

    def _ensure_network_metric_info(self, conn: sqlite3.Connection) -> None:
        if self._network_metric_info_cache is not None:
            return
        self._network_metric_info_cache = {}
        if not self._table_exists(conn, "TARGET_INFO_NETWORK_METRICS"):
            return

        cols = self._table_columns(conn, "TARGET_INFO_NETWORK_METRICS")
        required = {"metricsListId", "metricsIdx"}
        if not required.issubset(cols):
            return
        name_col = "name" if "name" in cols else None
        unit_col = "unit" if "unit" in cols else None
        if name_col is None:
            return
        select_cols = [
            "metricsListId AS _list_id",
            "metricsIdx AS _idx",
            f"{name_col} AS _name",
        ]
        if unit_col:
            select_cols.append(f"{unit_col} AS _unit")
        query = f"SELECT {', '.join(select_cols)} FROM TARGET_INFO_NETWORK_METRICS;"
        try:
            rows = conn.execute(query).fetchall()
        except sqlite3.Error:
            return
        for row in rows:
            list_id = _safe_int(row["_list_id"])
            idx = _safe_int(row["_idx"])
            if list_id is None or idx is None:
                continue
            name = str(row["_name"] or "").strip()
            unit = str(row["_unit"] or "").strip() if "_unit" in row.keys() else ""
            if not name:
                continue
            self._network_metric_info_cache[(list_id, idx)] = {
                "name": name,
                "unit": unit,
            }

    def _resolve_network_metric(
        self, conn: sqlite3.Connection, list_id: Optional[int], idx: Optional[int]
    ) -> Dict[str, str]:
        self._ensure_network_metric_info(conn)
        if list_id is None or idx is None:
            return {}
        cache = self._network_metric_info_cache or {}
        return cache.get((list_id, idx), {})

    def _ensure_generic_event_type_cache(self, conn: sqlite3.Connection) -> None:
        if self._generic_event_type_cache is not None:
            return
        self._generic_event_type_cache = {}

        source_map: Dict[int, str] = {}
        if self._table_exists(conn, "GENERIC_EVENT_SOURCES"):
            cols = self._table_columns(conn, "GENERIC_EVENT_SOURCES")
            if {"sourceId", "name"}.issubset(cols):
                try:
                    rows = conn.execute(
                        "SELECT sourceId, name FROM GENERIC_EVENT_SOURCES;"
                    ).fetchall()
                    for row in rows:
                        sid = _safe_int(row["sourceId"])
                        if sid is None:
                            continue
                        source_map[sid] = str(row["name"] or "")
                except sqlite3.Error:
                    pass

        if not self._table_exists(conn, "GENERIC_EVENT_TYPES"):
            return
        cols = self._table_columns(conn, "GENERIC_EVENT_TYPES")
        if "typeId" not in cols:
            return
        select_cols = ["typeId"]
        if "sourceId" in cols:
            select_cols.append("sourceId")
        if "data" in cols:
            select_cols.append("data")
        query = f"SELECT {', '.join(select_cols)} FROM GENERIC_EVENT_TYPES;"
        try:
            rows = conn.execute(query).fetchall()
        except sqlite3.Error:
            return

        for row in rows:
            type_id = _safe_int(row["typeId"])
            if type_id is None:
                continue
            source_id = _safe_int(row["sourceId"]) if "sourceId" in row.keys() else None
            source_name = source_map.get(source_id, "")
            data = self._safe_json_load(row["data"]) if "data" in row.keys() else None
            type_name = ""
            if data:
                for key in ("name", "Name", "metricName", "typeName", "displayName"):
                    if key in data and data[key]:
                        type_name = str(data[key]).strip()
                        break
            if not type_name:
                type_name = f"type_{type_id}"
            self._generic_event_type_cache[type_id] = {
                "type_name": type_name,
                "source_name": source_name,
            }

    def _resolve_generic_event_type(
        self, conn: sqlite3.Connection, type_id: Optional[int]
    ) -> Dict[str, str]:
        self._ensure_generic_event_type_cache(conn)
        if type_id is None:
            return {}
        cache = self._generic_event_type_cache or {}
        return cache.get(type_id, {})

    def _extract_step_rank_from_nvtx(self, text: str) -> Dict[str, str]:
        tags: Dict[str, str] = {}
        if not self.parse_step_from_nvtx or not text:
            return tags
        for pattern in self._STEP_PATTERNS:
            match = pattern.search(text)
            if match:
                tags["step"] = match.group(1)
                break
        for pattern in self._RANK_PATTERNS:
            match = pattern.search(text)
            if match:
                tags["rank"] = match.group(1)
                break
        return tags

    def _enrich_common_tags(
        self, conn: sqlite3.Connection, row: sqlite3.Row
    ) -> Dict[str, str]:
        self._ensure_thread_process_names(conn)

        tags: Dict[str, str] = {}
        tags.update(self._schema_tags())
        if "globalTid" in row.keys():
            gtid = _safe_int(row["globalTid"])
            if gtid is not None:
                pid, tid = _decode_global_tid(gtid)
                tags["pid"] = str(pid)
                tags["tid"] = str(tid)
                tags["global_tid"] = str(gtid)
                if self._thread_name_cache and gtid in self._thread_name_cache:
                    tags["thread"] = self._thread_name_cache[gtid]

        if "globalPid" in row.keys():
            gpid = _safe_int(row["globalPid"])
            if gpid is not None:
                tags.setdefault("pid", str(_decode_global_pid(gpid)))
                tags["global_pid"] = str(gpid)
                if self._process_name_cache and gpid in self._process_name_cache:
                    tags["process"] = self._process_name_cache[gpid]

        for key in (
            "deviceId",
            "contextId",
            "greenContextId",
            "streamId",
            "correlationId",
        ):
            if key in row.keys():
                val = _safe_int(row[key])
                if val is not None:
                    tags[key.lower()] = str(val)

        if "eventClass" in row.keys():
            ec = _safe_int(row["eventClass"])
            if ec is not None:
                tags["event_class"] = CUDA_EVENT_CLASS_MAP.get(ec, str(ec))
        return tags

    def _attach_runtime_api_tag(self, tags: Dict[str, str], row: sqlite3.Row) -> None:
        corr = (
            _safe_int(row["correlationId"]) if "correlationId" in row.keys() else None
        )
        if corr is None:
            return
        pid = tags.get("pid")
        runtime = self._runtime_by_correlation.get((pid, corr))
        if runtime is None:
            runtime = self._runtime_by_correlation.get((None, corr))
        if runtime and runtime.get("api"):
            tags["runtime_api"] = runtime["api"]

    def _make_duration_event(
        self,
        *,
        now: float,
        row: sqlite3.Row,
        name: str,
        tags: Dict[str, str],
    ) -> Optional[MetricEvent]:
        start = _safe_float(row["start"]) if "start" in row.keys() else None
        end = _safe_float(row["end"]) if "end" in row.keys() else None
        if start is None or end is None or end <= start:
            return None
        duration_us = (end - start) / 1000.0
        node_id = self._get_node_id(row)
        return MetricEvent(
            timestamp=now,
            name=name,
            value=duration_us,
            unit="us",
            provider_id=self.provider_id,
            tags=tags,
            attributes={"start_ns": start, "end_ns": end},
            node_id=node_id,
        )

    def _read_runtime(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            required_columns=(
                "start",
                "end",
                "nameId",
                "eventClass",
                "globalTid",
                "correlationId",
            ),
        )
        if not rows:
            return []
        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            api_name = (
                self._resolve_string(conn, row["nameId"])
                if "nameId" in row.keys()
                else "unknown_api"
            )
            tags["api"] = api_name
            evt = self._make_duration_event(
                now=now, row=row, name="latency.cuda.runtime_api", tags=tags
            )
            if evt is not None:
                events.append(evt)
            corr = (
                _safe_int(row["correlationId"])
                if "correlationId" in row.keys()
                else None
            )
            if corr is not None:
                pid = tags.get("pid")
                self._runtime_by_correlation[(pid, corr)] = {"api": api_name}
                if pid is None:
                    self._runtime_by_correlation[(None, corr)] = {"api": api_name}
        return events

    def _read_kernels(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "CUPTI_ACTIVITY_KIND_KERNEL",
            required_columns=(
                "start",
                "end",
                "demangledName",
                "shortName",
                "eventClass",
                "globalPid",
                "deviceId",
                "contextId",
                "streamId",
                "correlationId",
                "registersPerThread",
                "gridX",
                "gridY",
                "gridZ",
                "blockX",
                "blockY",
                "blockZ",
                "staticSharedMemory",
                "dynamicSharedMemory",
                "localMemoryTotal",
                "localMemoryPerThread",
            ),
        )
        if not rows:
            return []
        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            if "demangledName" in row.keys():
                kernel_raw = row["demangledName"]
            elif "shortName" in row.keys():
                kernel_raw = row["shortName"]
            else:
                kernel_raw = None
            tags["kernel"] = self._resolve_string(conn, kernel_raw) or "unknown_kernel"
            self._attach_runtime_api_tag(tags, row)

            evt = self._make_duration_event(
                now=now, row=row, name="latency.kernel.cuda", tags=tags
            )
            if evt is not None:
                events.append(evt)

            node_id = self._get_node_id(row)
            for col, metric_name, unit in (
                ("registersPerThread", "compute.kernel.registers_per_thread", "count"),
                ("staticSharedMemory", "memory.kernel.shared_static_bytes", "bytes"),
                ("dynamicSharedMemory", "memory.kernel.shared_dynamic_bytes", "bytes"),
                ("localMemoryTotal", "memory.kernel.local_total_bytes", "bytes"),
                (
                    "localMemoryPerThread",
                    "memory.kernel.local_per_thread_bytes",
                    "bytes",
                ),
            ):
                if col not in row.keys():
                    continue
                val = _safe_float(row[col])
                if val is None:
                    continue
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name=metric_name,
                        value=val,
                        unit=unit,
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )

            if {"blockX", "blockY", "blockZ"}.issubset(set(row.keys())):
                bx = _safe_int(row["blockX"]) or 1
                by = _safe_int(row["blockY"]) or 1
                bz = _safe_int(row["blockZ"]) or 1
                for axis, value in (("x", bx), ("y", by), ("z", bz)):
                    events.append(
                        MetricEvent(
                            timestamp=now,
                            name=f"compute.kernel.block_{axis}",
                            value=int(max(1, value)),
                            unit="count",
                            provider_id=self.provider_id,
                            tags=tags,
                            node_id=node_id,
                        )
                    )
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name="compute.kernel.threads_per_block",
                        value=int(max(1, bx * by * bz)),
                        unit="count",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
            if {"gridX", "gridY", "gridZ"}.issubset(set(row.keys())):
                gx = _safe_int(row["gridX"]) or 1
                gy = _safe_int(row["gridY"]) or 1
                gz = _safe_int(row["gridZ"]) or 1
                for axis, value in (("x", gx), ("y", gy), ("z", gz)):
                    events.append(
                        MetricEvent(
                            timestamp=now,
                            name=f"compute.kernel.grid_{axis}",
                            value=int(max(1, value)),
                            unit="count",
                            provider_id=self.provider_id,
                            tags=tags,
                            node_id=node_id,
                        )
                    )
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name="compute.kernel.blocks_per_grid",
                        value=int(max(1, gx * gy * gz)),
                        unit="count",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
        return events

    def _read_memcpy(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            required_columns=(
                "start",
                "end",
                "eventClass",
                "globalPid",
                "deviceId",
                "contextId",
                "streamId",
                "correlationId",
                "bytes",
                "copyKind",
                "srcKind",
                "dstKind",
            ),
        )
        if not rows:
            return []
        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            copy_kind = _safe_int(row["copyKind"]) if "copyKind" in row.keys() else None
            src_kind = _safe_int(row["srcKind"]) if "srcKind" in row.keys() else None
            dst_kind = _safe_int(row["dstKind"]) if "dstKind" in row.keys() else None
            if copy_kind is not None:
                tags["copy_kind"] = CUDA_MEMCPY_KIND_MAP.get(copy_kind, str(copy_kind))
            if src_kind is not None:
                tags["src_kind"] = CUDA_MEMORY_KIND_MAP.get(src_kind, str(src_kind))
            if dst_kind is not None:
                tags["dst_kind"] = CUDA_MEMORY_KIND_MAP.get(dst_kind, str(dst_kind))
            self._attach_runtime_api_tag(tags, row)

            dur_evt = self._make_duration_event(
                now=now, row=row, name="latency.memcpy.cuda", tags=tags
            )
            if dur_evt is not None:
                events.append(dur_evt)

            bytes_val = _safe_float(row["bytes"]) if "bytes" in row.keys() else None
            node_id = self._get_node_id(row)
            if bytes_val is not None:
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name="io.memcpy.bytes",
                        value=bytes_val,
                        unit="bytes",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
                if dur_evt is not None and float(dur_evt.value) > 0:
                    bw_gbps = float(bytes_val) / (float(dur_evt.value) * 1000.0)
                    events.append(
                        MetricEvent(
                            timestamp=now,
                            name="io.memcpy.bandwidth_gbps",
                            value=bw_gbps,
                            unit="GB/s",
                            provider_id=self.provider_id,
                            tags=tags,
                            node_id=node_id,
                        )
                    )
        return events

    def _read_memset(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "CUPTI_ACTIVITY_KIND_MEMSET",
            required_columns=(
                "start",
                "end",
                "eventClass",
                "globalPid",
                "deviceId",
                "contextId",
                "streamId",
                "correlationId",
                "bytes",
                "memKind",
            ),
        )
        if not rows:
            return []
        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            mem_kind = _safe_int(row["memKind"]) if "memKind" in row.keys() else None
            if mem_kind is not None:
                tags["mem_kind"] = CUDA_MEMORY_KIND_MAP.get(mem_kind, str(mem_kind))
            self._attach_runtime_api_tag(tags, row)

            dur_evt = self._make_duration_event(
                now=now, row=row, name="latency.memset.cuda", tags=tags
            )
            if dur_evt is not None:
                events.append(dur_evt)

            bytes_val = _safe_float(row["bytes"]) if "bytes" in row.keys() else None
            node_id = self._get_node_id(row)
            if bytes_val is not None:
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name="io.memset.bytes",
                        value=bytes_val,
                        unit="bytes",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
        return events

    def _read_sync(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
            required_columns=(
                "start",
                "end",
                "eventClass",
                "globalPid",
                "deviceId",
                "contextId",
                "streamId",
                "correlationId",
                "syncType",
                "eventId",
            ),
        )
        if not rows:
            return []
        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            sync_type = _safe_int(row["syncType"]) if "syncType" in row.keys() else None
            event_id = _safe_int(row["eventId"]) if "eventId" in row.keys() else None
            if sync_type is not None:
                tags["sync_type"] = str(sync_type)
            if event_id is not None:
                tags["sync_event_id"] = str(event_id)
            self._attach_runtime_api_tag(tags, row)

            dur_evt = self._make_duration_event(
                now=now, row=row, name="latency.cuda.sync", tags=tags
            )
            if dur_evt is not None:
                events.append(dur_evt)
        return events

    def _read_memory_usage(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        if not self._table_exists(conn, "CUDA_GPU_MEMORY_USAGE_EVENTS"):
            return []

        usage_cols = self._table_columns(conn, "CUDA_GPU_MEMORY_USAGE_EVENTS")
        # Some very old exports/docs may represent memory-pool records in this table shape.
        # If regular usage fields are absent, skip here and let memory-pool parser handle it.
        if (
            "bytes" not in usage_cols
            and "memoryOperationType" not in usage_cols
            and "operationType" in usage_cols
            and "poolType" in usage_cols
        ):
            return []

        rows = self._rows_incremental(
            conn,
            "CUDA_GPU_MEMORY_USAGE_EVENTS",
            required_columns=(
                "start",
                "timestamp",
                "bytes",
                "memoryOperationType",
                "operationType",
                "memoryPoolConfigType",
                "memoryPoolOperationType",
                "poolOperationType",
                "memoryPoolType",
                "poolType",
                "memKind",
                "memoryKind",
                "deviceId",
                "contextId",
                "greenContextId",
                "streamId",
                "correlationId",
                "globalPid",
            ),
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            mem_kind = self._row_int(row, "memKind", "memoryKind")
            if mem_kind is not None:
                tags["mem_kind"] = CUDA_MEMORY_KIND_MAP.get(mem_kind, str(mem_kind))

            mem_pool_type = self._row_int(row, "memoryPoolType", "poolType")
            if mem_pool_type is not None:
                tags["memory_pool_type"] = CUDA_MEMPOOL_TYPE_MAP.get(
                    mem_pool_type, str(mem_pool_type)
                )

            mem_pool_op = self._row_int(
                row, "memoryPoolOperationType", "poolOperationType"
            )
            if mem_pool_op is not None:
                tags["memory_pool_operation"] = CUDA_MEMPOOL_OPERATION_TYPE_MAP.get(
                    mem_pool_op, str(mem_pool_op)
                )

            op_type = self._row_int(row, "memoryOperationType", "operationType")
            if op_type is not None:
                tags["memory_operation_type"] = (
                    "allocate"
                    if op_type == 0
                    else ("deallocate" if op_type == 1 else str(op_type))
                )

            self._attach_runtime_api_tag(tags, row)
            bytes_val = self._row_float(row, "bytes")
            if bytes_val is None:
                continue

            node_id = self._get_node_id(row)
            if op_type == 0:
                name = "memory.cuda.alloc.bytes"
            elif op_type == 1:
                name = "memory.cuda.free.bytes"
            else:
                name = "memory.cuda.usage.bytes"

            events.append(
                MetricEvent(
                    timestamp=now,
                    name=name,
                    value=bytes_val,
                    unit="bytes",
                    provider_id=self.provider_id,
                    tags=tags,
                    node_id=node_id,
                )
            )
        return events

    def _read_memory_pool_events(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        table_name: Optional[str] = None
        if self._table_exists(conn, "CUDA_GPU_MEMORY_POOL_EVENTS"):
            table_name = "CUDA_GPU_MEMORY_POOL_EVENTS"
        elif self._table_exists(conn, "CUDA_GPU_MEMORY_USAGE_EVENTS"):
            usage_cols = self._table_columns(conn, "CUDA_GPU_MEMORY_USAGE_EVENTS")
            if {"operationType", "poolType"}.issubset(
                usage_cols
            ) and "bytes" not in usage_cols:
                # Compatibility fallback for legacy exports where pool events are in usage table shape.
                table_name = "CUDA_GPU_MEMORY_USAGE_EVENTS"
        if table_name is None:
            return []

        rows = self._rows_incremental(
            conn,
            table_name,
            required_columns=(
                "start",
                "timestamp",
                "globalPid",
                "deviceId",
                "correlationId",
                "operationType",
                "memoryPoolOperationType",
                "poolOperationType",
                "poolType",
                "memoryPoolType",
                "minBytesToKeep",
                "localMemoryPoolReleaseThreshold",
                "localMemoryPoolSize",
                "localMemoryPoolUtilizedSize",
            ),
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            pool_op = self._row_int(
                row, "operationType", "memoryPoolOperationType", "poolOperationType"
            )
            pool_type = self._row_int(row, "poolType", "memoryPoolType")
            if pool_op is not None:
                tags["memory_pool_operation"] = CUDA_MEMPOOL_OPERATION_TYPE_MAP.get(
                    pool_op, str(pool_op)
                )
            if pool_type is not None:
                tags["memory_pool_type"] = CUDA_MEMPOOL_TYPE_MAP.get(
                    pool_type, str(pool_type)
                )

            self._attach_runtime_api_tag(tags, row)
            node_id = self._get_node_id(row)
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="calls.cuda.memory_pool_event",
                    value=1,
                    unit="count",
                    provider_id=self.provider_id,
                    tags=tags,
                    node_id=node_id,
                )
            )

            for col, metric_name in (
                ("minBytesToKeep", "memory.cuda.pool.min_bytes_to_keep"),
                (
                    "localMemoryPoolReleaseThreshold",
                    "memory.cuda.pool.local_release_threshold_bytes",
                ),
                ("localMemoryPoolSize", "memory.cuda.pool.local_size_bytes"),
                (
                    "localMemoryPoolUtilizedSize",
                    "memory.cuda.pool.local_utilized_bytes",
                ),
            ):
                val = self._row_float(row, col)
                if val is None:
                    continue
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name=metric_name,
                        value=val,
                        unit="bytes",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
        return events

    def _read_gpu_metrics(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "GPU_METRICS",
            required_columns=(
                "timestamp",
                "rawTimestamp",
                "typeId",
                "metricId",
                "value",
            ),
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            value = self._row_float(row, "value")
            if value is None:
                continue
            type_id = self._row_int(row, "typeId")
            metric_id = self._row_int(row, "metricId")
            metric_name = self._resolve_gpu_metric_name(conn, type_id, metric_id)
            canonical_name = self._canonical_gpu_metric_name(metric_name)
            unit = self._infer_unit_from_name(metric_name)
            tags = dict(self._schema_tags())
            tags["metric_name"] = metric_name
            if type_id is not None:
                tags["type_id"] = str(type_id)
                # NVIDIA docs describe lower 8 bits as GPU id for generic metric typeId.
                tags["gpu_id"] = str(int(type_id) & 0xFF)
            if metric_id is not None:
                tags["metric_id"] = str(metric_id)
            ts_ns = self._row_int(row, "timestamp", "rawTimestamp")
            if ts_ns is not None:
                tags["sample_timestamp_ns"] = str(ts_ns)
            node_id = self._get_node_id(row)
            events.append(
                MetricEvent(
                    timestamp=now,
                    name=canonical_name,
                    value=value,
                    unit=unit,
                    provider_id=self.provider_id,
                    tags=tags,
                    node_id=node_id,
                )
            )
        return events

    def _read_generic_events(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "GENERIC_EVENTS",
            required_columns=(
                "timestamp",
                "rawTimestamp",
                "typeId",
                "data",
            ),
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            type_id = self._row_int(row, "typeId")
            type_info = self._resolve_generic_event_type(conn, type_id)
            type_name = type_info.get(
                "type_name", f"type_{type_id}" if type_id is not None else "unknown"
            )
            source_name = type_info.get("source_name", "")

            payload = (
                self._safe_json_load(row["data"]) if "data" in row.keys() else None
            )
            if not payload:
                continue
            flat = self._flatten_numeric_payload(payload)
            if not flat:
                continue

            base_tags = dict(self._schema_tags())
            base_tags["generic_type"] = self._normalize_token(type_name)
            if source_name:
                base_tags["generic_source"] = source_name
            if type_id is not None:
                base_tags["type_id"] = str(type_id)
                base_tags["gpu_id"] = str(int(type_id) & 0xFF)
            ts_ns = self._row_int(row, "timestamp", "rawTimestamp")
            if ts_ns is not None:
                base_tags["sample_timestamp_ns"] = str(ts_ns)
            node_id = self._get_node_id(row)

            type_token = self._normalize_token(type_name)
            for key, value in flat.items():
                metric_key = self._normalize_token(key)
                if type_token == "gpu_metrics":
                    metric_name = self._canonical_gpu_metric_name(metric_key)
                else:
                    metric_name = f"external.nsys.generic.{type_token}.{metric_key}"
                unit = self._infer_unit_from_name(metric_key)
                tags = dict(base_tags)
                tags["metric_key"] = metric_key
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name=metric_name,
                        value=value,
                        unit=unit,
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
        return events

    def _read_network_metric_table(
        self, conn: sqlite3.Connection, table_name: str, metric_prefix: str
    ) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            table_name,
            required_columns=(
                "start",
                "end",
                "metricsListId",
                "metricsIdx",
                "value",
                "globalId",
            ),
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            value = self._row_float(row, "value")
            if value is None:
                continue

            list_id = self._row_int(row, "metricsListId")
            idx = self._row_int(row, "metricsIdx")
            info = self._resolve_network_metric(conn, list_id, idx)
            metric_name_raw = info.get(
                "name", f"{table_name}_metric_{idx if idx is not None else 'unknown'}"
            )
            metric_name = f"{metric_prefix}.{self._normalize_token(metric_name_raw)}"
            unit = info.get("unit", "") or self._infer_unit_from_name(metric_name_raw)

            tags = dict(self._schema_tags())
            tags["source_table"] = table_name
            tags["metric_name"] = metric_name_raw
            if list_id is not None:
                tags["metrics_list_id"] = str(list_id)
            if idx is not None:
                tags["metrics_idx"] = str(idx)
            global_id = self._row_int(row, "globalId")
            if global_id is not None:
                tags["global_id"] = str(global_id)
            node_id = self._get_node_id(row)
            events.append(
                MetricEvent(
                    timestamp=now,
                    name=metric_name,
                    value=value,
                    unit=unit,
                    provider_id=self.provider_id,
                    tags=tags,
                    node_id=node_id,
                )
            )

            # Keep sampling-window latency for diagnostics.
            if "start" in row.keys() and "end" in row.keys():
                dur = self._make_duration_event(
                    now=now,
                    row=row,
                    name=f"latency.nsys.{self._normalize_token(table_name)}",
                    tags=tags,
                )
                if dur is not None:
                    events.append(dur)
        return events

    def _read_network_metrics(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        events: List[MetricEvent] = []
        events.extend(
            self._read_network_metric_table(conn, "NET_NIC_METRIC", "comm.nic")
        )
        events.extend(
            self._read_network_metric_table(
                conn, "NET_IB_SWITCH_METRIC", "comm.ib.switch"
            )
        )
        return events

    def _read_pmu_metrics(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        if not self._table_exists(conn, "PMU_EVENT_COUNTERS"):
            return []
        counter_cols = self._table_columns(conn, "PMU_EVENT_COUNTERS")
        if not {"id", "idx", "value"}.issubset(counter_cols):
            return []

        last = self._last_rowid.get("PMU_EVENT_COUNTERS", 0)
        select_cols = [
            "c.rowid AS _rowid",
            "c.id AS counter_id",
            "c.idx AS request_idx",
            "c.value AS value",
        ]
        join_sql = ""
        where_sql = "WHERE c.rowid > ?"
        if self._table_exists(conn, "PMU_EVENTS"):
            pmu_cols = self._table_columns(conn, "PMU_EVENTS")
            join_sql += " LEFT JOIN PMU_EVENTS e ON e.counter_id = c.id"
            for col in ("start", "end", "cpu", "globalVm"):
                if col in pmu_cols:
                    select_cols.append(f"e.{col} AS {col}")
        if self._table_exists(conn, "PMU_EVENT_REQUESTS"):
            req_cols = self._table_columns(conn, "PMU_EVENT_REQUESTS")
            join_sql += " LEFT JOIN PMU_EVENT_REQUESTS r ON r.id = c.idx"
            for col in ("event_name", "source", "unit_type"):
                if col in req_cols:
                    select_cols.append(f"r.{col} AS {col}")

        query = (
            f"SELECT {', '.join(select_cols)} "
            f"FROM PMU_EVENT_COUNTERS c"
            f"{join_sql} "
            f"{where_sql} "
            f"ORDER BY c.rowid ASC;"
        )
        try:
            rows = conn.execute(query, (last,)).fetchall()
        except sqlite3.Error:
            return []
        if not rows:
            return []
        self._last_rowid["PMU_EVENT_COUNTERS"] = int(rows[-1]["_rowid"])

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            value = self._row_float(row, "value")
            if value is None:
                continue
            event_name = (
                str(row["event_name"] or "").strip()
                if "event_name" in row.keys()
                else ""
            )
            metric_name = (
                f"perf.pmu.{self._normalize_token(event_name)}"
                if event_name
                else "perf.pmu.value"
            )
            tags = dict(self._schema_tags())
            if event_name:
                tags["pmu_event"] = event_name
            if "source" in row.keys() and row["source"] is not None:
                tags["pmu_source"] = str(row["source"])
            if "unit_type" in row.keys() and row["unit_type"] is not None:
                tags["pmu_unit_type"] = str(row["unit_type"])
            if "cpu" in row.keys() and row["cpu"] is not None:
                tags["cpu"] = str(row["cpu"])
            if "globalVm" in row.keys() and row["globalVm"] is not None:
                tags["global_vm"] = str(row["globalVm"])

            node_id = self._get_node_id(row)
            events.append(
                MetricEvent(
                    timestamp=now,
                    name=metric_name,
                    value=value,
                    unit="",
                    provider_id=self.provider_id,
                    tags=tags,
                    node_id=node_id,
                )
            )

            if "start" in row.keys() and "end" in row.keys():
                dur = self._make_duration_event(
                    now=now, row=row, name="latency.pmu.sample_window", tags=tags
                )
                if dur is not None:
                    events.append(dur)
        return events

    def _read_cuda_um_events(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        now = time.time()
        events: List[MetricEvent] = []

        cpu_rows = self._rows_incremental(
            conn,
            "CUDA_UM_CPU_PAGE_FAULT_EVENTS",
            required_columns=("start", "globalPid", "address", "CpuInstruction"),
        )
        for row in cpu_rows:
            tags = self._enrich_common_tags(conn, row)
            if "address" in row.keys() and row["address"] is not None:
                tags["address"] = str(row["address"])
            if "CpuInstruction" in row.keys() and row["CpuInstruction"] is not None:
                tags["cpu_instruction"] = str(row["CpuInstruction"])
            node_id = self._get_node_id(row)
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="calls.cuda.um.cpu_page_fault",
                    value=1,
                    unit="count",
                    provider_id=self.provider_id,
                    tags=tags,
                    node_id=node_id,
                )
            )

        gpu_rows = self._rows_incremental(
            conn,
            "CUDA_UM_GPU_PAGE_FAULT_EVENTS",
            required_columns=(
                "start",
                "end",
                "globalPid",
                "address",
                "numberOfPageFaults",
                "migrationCause",
                "accessType",
            ),
        )
        for row in gpu_rows:
            tags = self._enrich_common_tags(conn, row)
            if "address" in row.keys() and row["address"] is not None:
                tags["address"] = str(row["address"])
            if "migrationCause" in row.keys() and row["migrationCause"] is not None:
                tags["migration_cause"] = str(row["migrationCause"])
            if "accessType" in row.keys() and row["accessType"] is not None:
                tags["access_type"] = str(row["accessType"])
            node_id = self._get_node_id(row)
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="calls.cuda.um.gpu_page_fault_event",
                    value=1,
                    unit="count",
                    provider_id=self.provider_id,
                    tags=tags,
                    node_id=node_id,
                )
            )
            fault_count = self._row_float(row, "numberOfPageFaults")
            if fault_count is not None:
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name="memory.cuda.um.gpu_page_faults",
                        value=fault_count,
                        unit="count",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
            dur = self._make_duration_event(
                now=now, row=row, name="latency.cuda.um.gpu_page_fault", tags=tags
            )
            if dur is not None:
                events.append(dur)
        return events

    def _read_auto_duration_tables(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        tables = self._list_tables(conn)
        if not tables:
            return []
        now = time.time()
        events: List[MetricEvent] = []
        for table in tables:
            if table in AUTO_DURATION_SKIP_TABLES:
                continue
            if (
                table.startswith("ENUM_")
                or table.startswith("TARGET_INFO_")
                or table.startswith("EXPORT_")
            ):
                continue
            cols = self._table_columns(conn, table)
            if not {"start", "end"}.issubset(cols):
                continue

            rows = self._rows_incremental(
                conn,
                table,
                required_columns=(
                    "start",
                    "end",
                    "nameId",
                    "textId",
                    "eventClass",
                    "eventType",
                    "globalTid",
                    "globalPid",
                    "deviceId",
                    "contextId",
                    "greenContextId",
                    "streamId",
                    "correlationId",
                    "size",
                    "bytes",
                    "numberOfPageFaults",
                ),
            )
            if not rows:
                continue

            table_token = self._normalize_token(table)
            for row in rows:
                tags = self._enrich_common_tags(conn, row)
                tags["source_table"] = table

                if "nameId" in row.keys() and row["nameId"] is not None:
                    name = self._resolve_string(conn, row["nameId"])
                    if name:
                        tags["name"] = name
                if "textId" in row.keys() and row["textId"] is not None:
                    text = self._resolve_string(conn, row["textId"])
                    if text:
                        tags["text"] = text

                node_id = self._get_node_id(row)
                if table.startswith("MPI_"):
                    latency_name = f"latency.mpi.{table_token}"
                else:
                    latency_name = f"latency.nsys.{table_token}"
                evt = self._make_duration_event(
                    now=now, row=row, name=latency_name, tags=tags
                )
                if evt is not None:
                    events.append(evt)

                bytes_val = self._row_float(row, "bytes", "size")
                if bytes_val is not None:
                    if table.startswith("MPI_"):
                        bytes_metric_name = "comm.mpi.bytes"
                    else:
                        bytes_metric_name = f"io.nsys.{table_token}.bytes"
                    events.append(
                        MetricEvent(
                            timestamp=now,
                            name=bytes_metric_name,
                            value=bytes_val,
                            unit="bytes",
                            provider_id=self.provider_id,
                            tags=tags,
                            node_id=node_id,
                        )
                    )

                fault_count = self._row_float(row, "numberOfPageFaults")
                if fault_count is not None:
                    events.append(
                        MetricEvent(
                            timestamp=now,
                            name="memory.cuda.um.page_faults",
                            value=fault_count,
                            unit="count",
                            provider_id=self.provider_id,
                            tags=tags,
                            node_id=node_id,
                        )
                    )
        return events

    def _read_nvtx(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "NVTX_EVENTS",
            required_columns=(
                "start",
                "end",
                "eventType",
                "text",
                "textId",
                "name",
                "nameId",
                "jsonText",
                "domainId",
                "category",
                "globalTid",
                "color",
            ),
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            event_type = (
                _safe_int(row["eventType"]) if "eventType" in row.keys() else None
            )
            if event_type is not None:
                tags["nvtx_event_type"] = NVTX_EVENT_TYPE_MAP.get(
                    event_type, str(event_type)
                )
            if "domainId" in row.keys() and row["domainId"] is not None:
                tags["nvtx_domain_id"] = str(row["domainId"])
            if "category" in row.keys() and row["category"] is not None:
                tags["nvtx_category"] = str(row["category"])
            if "color" in row.keys() and row["color"] is not None:
                tags["nvtx_color"] = str(row["color"])

            text = ""
            if "text" in row.keys() and row["text"] is not None:
                text = self._resolve_string(conn, row["text"])
            if not text and "textId" in row.keys():
                text = self._resolve_string(conn, row["textId"])
            if not text and "nameId" in row.keys():
                text = self._resolve_string(conn, row["nameId"])
            if not text and "name" in row.keys() and row["name"] is not None:
                maybe_name = row["name"]
                sid = _safe_int(maybe_name)
                text = (
                    self._resolve_string(conn, sid)
                    if sid is not None
                    else str(maybe_name)
                )
            if not text and "jsonText" in row.keys() and row["jsonText"] is not None:
                text = str(row["jsonText"])
            if text:
                tags["nvtx_text"] = text
                # Expose NVTX custom label as standard "name" dimension for analyzer grouping.
                tags.setdefault("name", text)
                tags.update(self._extract_step_rank_from_nvtx(text))

            start = _safe_float(row["start"]) if "start" in row.keys() else None
            end = _safe_float(row["end"]) if "end" in row.keys() else None
            node_id = self._get_node_id(row)
            if start is not None and end is not None and end > start:
                duration_us = (end - start) / 1000.0
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name="latency.nvtx.range",
                        value=duration_us,
                        unit="us",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
            else:
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name="calls.nvtx.mark",
                        value=1,
                        unit="count",
                        provider_id=self.provider_id,
                        tags=tags,
                        node_id=node_id,
                    )
                )
        return events

    def _read_osrt(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        rows = self._rows_incremental(
            conn,
            "OSRT_API",
            required_columns=("start", "end", "nameId", "eventClass", "globalTid"),
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            tags = self._enrich_common_tags(conn, row)
            api_name = (
                self._resolve_string(conn, row["nameId"])
                if "nameId" in row.keys()
                else "unknown_osrt_api"
            )
            tags["api"] = api_name
            evt = self._make_duration_event(
                now=now, row=row, name="latency.osrt.api", tags=tags
            )
            if evt is not None:
                events.append(evt)
        return events

    def describe_schema(self) -> Dict[str, object]:
        if not self.sqlite_path.exists():
            return {"sqlite_path": str(self.sqlite_path), "exists": False}
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            self._load_schema_meta(conn)
            schema = NsightSchema(conn, meta=self._schema_meta)
            tables = list(schema.tables)
            columns: Dict[str, List[str]] = {
                table: schema.columns(table) for table in tables
            }
            summary = schema.summary()
            return {
                "sqlite_path": str(self.sqlite_path),
                "exists": True,
                "table_count": len(tables),
                "tables": tables,
                "columns": columns,
                "schema_meta": dict(self._schema_meta or {}),
                "canonical_tables": dict(summary.get("canonical_tables", {})),
                "version_info": {
                    "exporter_version": getattr(
                        self._version_info, "exporter_version", ""
                    ),
                    "export_schema_version": getattr(
                        self._version_info, "export_schema_version", ""
                    ),
                    "adapter_family": getattr(
                        self._version_info, "adapter_family", "generic"
                    ),
                    "known_version": bool(
                        getattr(self._version_info, "known_version", False)
                    ),
                },
            }
        finally:
            conn.close()

    def list_sql_skills(self) -> List[str]:
        """
        Return built-in SQL skill names available for this SQLite schema.
        """
        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.list_skills()
        finally:
            conn.close()

    def describe_sql_skills(self) -> List[Dict[str, object]]:
        """
        Return built-in SQL skills with metadata and parameters.
        """
        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.describe_skills()
        finally:
            conn.close()

    def run_sql_skill(self, skill_name: str, **params: Any) -> List[Dict[str, object]]:
        """
        Execute one built-in SQL skill and return rows.

        Example:
            provider.run_sql_skill("top_kernels", device_id=0, limit=20)
        """
        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.execute(skill_name, **params)
        finally:
            conn.close()

    def analyze_compute_comm_overlap(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        limit: int = 2_000_000,
    ) -> Dict[str, object]:
        """
        Analyze compute (non-NCCL) and communication (NCCL-like) overlap from kernel timeline.
        """
        if not self.sqlite_path.exists():
            return {}
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.analyze_compute_comm_overlap(
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
            )
        finally:
            conn.close()

    def summarize_gpu_kernels(
        self,
        *,
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        top_k: int = 10,
        limit: int = 2_000_000,
    ) -> Dict[str, object]:
        """
        Produce GPU kernel time summary (span/busy/idle/utilization/top kernels).
        """
        if not self.sqlite_path.exists():
            return {}
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.summarize_gpu_kernels(
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                top_k=top_k,
                limit=limit,
            )
        finally:
            conn.close()

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
        """
        Detect iteration windows from NVTX marker ranges.
        """
        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.detect_iterations(
                marker=marker,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                top_level_only=top_level_only,
                limit=limit,
            )
        finally:
            conn.close()

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
        """
        Per-iteration compute/comm/overlap breakdown.

        Returns one dict per iteration with all detect_iterations() fields plus:
        compute_ms, comm_ms, overlap_ms, comm_pct, kernel_count.
        """
        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.analyze_per_iteration_overlap(
                marker=marker,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                top_level_only=top_level_only,
                limit=limit,
            )
        finally:
            conn.close()

    def detect_iteration_outliers(
        self,
        *,
        marker: str = "sample_0",
        device_id: int = -1,
        threshold_sigma: float = 2.0,
        limit: int = 2000,
    ) -> Dict[str, object]:
        """
        Statistical outlier detection on iteration step durations.

        Returns:
            {
                "stats":    {count, mean_ms, median_ms, std_ms, p95_ms, p99_ms},
                "outliers": [{iteration, duration_ms, deviation_sigma}, ...]
            }
        """
        if not self.sqlite_path.exists():
            return {"stats": {}, "outliers": []}
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.detect_iteration_outliers(
                marker=marker,
                device_id=device_id,
                threshold_sigma=threshold_sigma,
                limit=limit,
            )
        finally:
            conn.close()

    def analyze_rank_straggler(
        self,
        *,
        marker: str = "sample_0",
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        limit: int = 2000,
    ) -> Dict[str, object]:
        """Detect straggler ranks in multi-GPU training.

        Groups iterations by rank tag extracted from NVTX text, computes per-rank
        duration statistics, and flags ranks whose median step time exceeds the
        global median by more than 10 %.

        Returns:
            {
                "global_stats": {count, mean_ms, median_ms, std_ms},
                "per_rank":     [{rank, count, mean_ms, median_ms, std_ms,
                                  delta_vs_global_pct, is_straggler}, ...],
                "stragglers":   [list of straggler rank IDs]
            }
        """
        if not self.sqlite_path.exists():
            return {"global_stats": {}, "per_rank": [], "stragglers": []}
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.analyze_rank_straggler(
                marker=marker,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
            )
        finally:
            conn.close()

    def classify_gpu_bottleneck(
        self,
        *,
        nvtx_text: str = "%",
        device_id: int = -1,
        start_ns: int = -1,
        end_ns: int = -1,
        limit: int = 2000,
    ) -> List[Dict[str, object]]:
        """Classify GPU bottleneck per NVTX phase using hardware metric sampling.

        Classifies each NVTX range as one of: tensor_bound, compute_bound,
        memory_bound, or latency_bound based on SM active, tensor active and
        DRAM throughput averages sampled inside the range.

        Returns:
            List of dicts with keys:
                nvtx_text, nvtx_start_ns, nvtx_end_ns,
                sm_active_avg, tensor_active_avg, dram_throughput_avg,
                bottleneck, bottleneck_confidence
        """
        if not self.sqlite_path.exists():
            return []
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            return engine.classify_gpu_bottleneck(
                nvtx_text=nvtx_text,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
            )
        finally:
            conn.close()

    def first_gpu_name(self) -> str:
        if not self.sqlite_path.exists():
            return ""
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            if not self._table_exists(conn, "TARGET_INFO_GPU"):
                return ""
            row = conn.execute(
                "SELECT name FROM TARGET_INFO_GPU ORDER BY id ASC LIMIT 1;"
            ).fetchone()
            return str(row["name"] or "").strip() if row else ""
        finally:
            conn.close()

    def compute_mfu(
        self,
        *,
        step_time_s: float,
        model_flops_per_step: float,
        peak_tflops: Optional[float] = None,
        precision: str = "fp16",
    ) -> Dict[str, object]:
        """
        Compute MFU from step time and model FLOPs.
        """
        resolved_peak = _safe_float(peak_tflops)
        if resolved_peak is None:
            gpu_name = self.first_gpu_name()
            resolved_peak = infer_peak_tflops(gpu_name, precision=precision)
            if resolved_peak is None:
                return {
                    "error": "peak_tflops is required when GPU name mapping is unavailable.",
                    "gpu_name": gpu_name,
                    "precision": precision,
                }
        payload = compute_mfu_single(
            step_time_s=float(step_time_s),
            model_flops_per_step=float(model_flops_per_step),
            peak_tflops=float(resolved_peak),
        )
        if "error" not in payload:
            payload["gpu_name"] = self.first_gpu_name()
            payload["precision"] = precision
        return payload

    def get_metrics(self) -> List[MetricEvent]:
        if not self.sqlite_path.exists():
            return []

        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            self._load_schema_meta(conn)
            result: List[MetricEvent] = []
            if self.include_runtime:
                result.extend(self._read_runtime(conn))
            if self.include_kernels:
                result.extend(self._read_kernels(conn))
            if self.include_memcpy:
                result.extend(self._read_memcpy(conn))
            if self.include_memset:
                result.extend(self._read_memset(conn))
            if self.include_sync:
                result.extend(self._read_sync(conn))
            if self.include_memory_usage:
                result.extend(self._read_memory_usage(conn))
                result.extend(self._read_memory_pool_events(conn))
            if self.include_gpu_metrics:
                result.extend(self._read_gpu_metrics(conn))
            if self.include_generic_events:
                result.extend(self._read_generic_events(conn))
            if self.include_network_metrics:
                result.extend(self._read_network_metrics(conn))
            if self.include_pmu_metrics:
                result.extend(self._read_pmu_metrics(conn))
            if self.include_cuda_um:
                result.extend(self._read_cuda_um_events(conn))
            if self.include_nvtx:
                result.extend(self._read_nvtx(conn))
            if self.include_osrt:
                result.extend(self._read_osrt(conn))
            if self.include_auto_duration_tables:
                result.extend(self._read_auto_duration_tables(conn))
            return result
        finally:
            conn.close()


class NsysSqliteGlobMetricsProvider(BaseMetricsProvider):
    """
    Aggregate multiple Nsight Systems SQLite files matched by a glob pattern.

    This enables one config provider spec to cover multi-rank outputs like:
    ./logs/light_bagel_pretrain/xxxprefix_rank_*.sqlite
    """

    provider_id = "nsys_sqlite_glob"
    _capabilities = ProviderCapabilities(
        provider_type="nsys_sqlite_glob",
        source_mode="offline",
        metric_prefixes=["latency", "memory", "io", "compute", "comm", "perf", "calls"],
        dimensions=["rank", "pid", "tid", "sqlite_file"],
        supports_incremental=True,
        supports_rank_scope=True,
        supports_step_scope=True,
        notes="Glob-based multi-file wrapper for NsysSqliteMetricsProvider.",
    )

    def __init__(
        self,
        *,
        sqlite_glob: str,
        provider_id: Optional[str] = None,
        enabled: bool = True,
        recursive: bool = False,
        parse_rank_from_filename: bool = True,
        file_rank_regex: str = r"rank[_-]?(\d+)",
        **provider_kwargs: Any,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)

        self.sqlite_glob = str(sqlite_glob)
        self.recursive = bool(recursive)
        self.parse_rank_from_filename = bool(parse_rank_from_filename)
        self._file_rank_pattern = re.compile(str(file_rank_regex), re.IGNORECASE)
        self._provider_kwargs = dict(provider_kwargs)
        self._children: Dict[str, NsysSqliteMetricsProvider] = {}

    def _iter_paths(self) -> List[Path]:
        matched = sorted(glob(self.sqlite_glob, recursive=self.recursive))
        return [Path(item) for item in matched if item]

    def _get_child(self, path: Path) -> NsysSqliteMetricsProvider:
        key = str(path.resolve())
        child = self._children.get(key)
        if child is None:
            # Keep child provider_id stable but unify exported provider_id on events.
            child = NsysSqliteMetricsProvider(
                sqlite_path=str(path),
                provider_id=f"{self.provider_id}:{path.stem}",
                enabled=True,
                **self._provider_kwargs,
            )
            self._children[key] = child
        return child

    def _infer_rank_from_filename(self, path: Path) -> Optional[str]:
        if not self.parse_rank_from_filename:
            return None
        match = self._file_rank_pattern.search(path.name)
        if not match:
            return None
        return str(match.group(1))

    def get_metrics(self) -> List[MetricEvent]:
        paths = self._iter_paths()
        if not paths:
            return []

        events: List[MetricEvent] = []
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            try:
                child = self._get_child(path)
                child_events = child.get_metrics()
            except Exception:
                # Skip broken/non-sqlite files in a broad glob match.
                continue

            rank_from_name = self._infer_rank_from_filename(path)
            for event in child_events:
                event.provider_id = self.provider_id
                event.tags.setdefault("sqlite_file", path.name)
                event.tags.setdefault("sqlite_path", str(path))
                if rank_from_name is not None and "rank" not in event.tags:
                    event.tags["rank"] = rank_from_name
                events.append(event)
        return events
