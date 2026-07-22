# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import sqlite3


@dataclass
class NsysVersionInfo:
    exporter_version: str = ""
    export_schema_version: str = ""
    adapter_family: str = "generic"
    known_version: bool = False
    parsed_version: List[int] = field(default_factory=list)


def _pick_first(meta: Dict[str, str], keys: List[str]) -> str:
    for key in keys:
        if key in meta and meta[key]:
            return str(meta[key])
    return ""


def _parse_version(version_text: str) -> List[int]:
    matches = re.findall(r"\d+", version_text or "")
    return [int(token) for token in matches[:4]]


def _resolve_family(version_nums: List[int]) -> str:
    if not version_nums:
        return "generic"
    year = version_nums[0]
    if year >= 2024:
        return "nsys_2024_plus"
    if year == 2023:
        return "nsys_2023"
    if year == 2022:
        return "nsys_2022"
    return "generic"


def detect_nsys_version(meta: Dict[str, str]) -> NsysVersionInfo:
    exporter_version = _pick_first(
        meta,
        [
            "NSIGHT_SYSTEMS_VERSION",
            "EXPORTER_VERSION",
            "EXPORT_VERSION",
            "TOOL_VERSION",
        ],
    )
    export_schema_version = _pick_first(
        meta,
        [
            "EXPORT_SCHEMA_VERSION",
            "EXPORT_SCHEMA",
        ],
    )
    parsed = _parse_version(exporter_version)
    family = _resolve_family(parsed)
    return NsysVersionInfo(
        exporter_version=exporter_version,
        export_schema_version=export_schema_version,
        adapter_family=family,
        known_version=bool(parsed),
        parsed_version=parsed,
    )


class NsightSchema:
    """
    Schema/metadata helper for Nsight Systems SQLite exports.

    Key goals:
    - auto-detect major table variants across exporter versions
    - expose canonical table choices (kernel/runtime/nvtx/string/meta)
    - provide robust column alias resolution for query builders
    """

    def __init__(
        self, conn: sqlite3.Connection, *, meta: Optional[Dict[str, str]] = None
    ) -> None:
        self._conn = conn
        self.tables = self._load_tables()
        self._columns_cache: Dict[str, set[str]] = {}
        self.meta = dict(meta or self._load_meta())
        self.version = detect_nsys_version(self.meta)

        self.meta_table = self._detect_first_existing(
            ("META_DATA_EXPORT", "EXPORT_META_DATA", "META_DATA_CAPTURE")
        )
        self.string_table = self._detect_first_existing(("StringIds", "STRINGIDS"))
        self.kernel_table = self._detect_kernel_table()
        self.runtime_table = self._detect_first_existing(
            ("CUPTI_ACTIVITY_KIND_RUNTIME",)
        )
        self.nvtx_table = self._detect_first_existing(("NVTX_EVENTS",))
        self.memcpy_table = self._detect_first_existing(("CUPTI_ACTIVITY_KIND_MEMCPY",))
        self.memset_table = self._detect_first_existing(("CUPTI_ACTIVITY_KIND_MEMSET",))
        self.sync_table = self._detect_first_existing(
            ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",)
        )
        self.metrics_table = self._detect_metrics_table()
        self.metrics_timestamp_col = (
            self.resolve_column(
                self.metrics_table, ("timestamp", "rawTimestamp", "start", "time")
            )
            if self.metrics_table
            else None
        )
        self.metrics_id_col = (
            self.resolve_column(self.metrics_table, ("metricId", "nameId", "eventId"))
            if self.metrics_table
            else None
        )
        self.metrics_value_col = (
            self.resolve_column(self.metrics_table, ("value", "metricValue", "val"))
            if self.metrics_table
            else None
        )

    def _load_tables(self) -> List[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        result = [str(row[0]) for row in rows if row and row[0]]
        result.sort()
        return result

    def _load_meta(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for table_name in ("META_DATA_EXPORT", "EXPORT_META_DATA", "META_DATA_CAPTURE"):
            if table_name not in self.tables:
                continue
            cols = self._table_columns(table_name)
            if not {"name", "value"}.issubset(cols):
                continue
            try:
                rows = self._conn.execute(
                    f"SELECT name, value FROM {table_name};"
                ).fetchall()
            except sqlite3.Error:
                continue
            out["meta_table"] = table_name
            for row in rows:
                key = str(row[0] or "").strip()
                if key:
                    out[key] = str(row[1] or "")
            if out:
                break
        return out

    def _detect_first_existing(self, candidates: Iterable[str]) -> Optional[str]:
        table_set = set(self.tables)
        for table in candidates:
            if table in table_set:
                return table
        return None

    def _table_has_rows(self, table_name: str) -> bool:
        try:
            row = self._conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1;").fetchone()
            return row is not None
        except sqlite3.Error:
            return False

    def _is_metrics_fact_table(self, table_name: str) -> bool:
        return bool(
            self.resolve_column(
                table_name, ("timestamp", "rawTimestamp", "start", "time")
            )
            and self.resolve_column(table_name, ("metricId", "nameId", "eventId"))
            and self.resolve_column(table_name, ("value", "metricValue", "val"))
        )

    def _detect_metrics_table(self) -> Optional[str]:
        candidates = (
            "GPU_METRICS",
            "CUPTI_ACTIVITY_KIND_GPU_METRIC",
            "CUPTI_ACTIVITY_KIND_METRIC",
            "TARGET_INFO_GPU_METRICS",
        )
        first_existing: Optional[str] = None
        # Prefer a non-empty fact table to avoid selecting empty compatibility tables.
        for table_name in candidates:
            if not self.table_exists(table_name):
                continue
            if first_existing is None:
                first_existing = table_name
            if self._is_metrics_fact_table(table_name) and self._table_has_rows(
                table_name
            ):
                return table_name
        # Fallback to structurally valid metrics fact table even if empty.
        for table_name in candidates:
            if self.table_exists(table_name) and self._is_metrics_fact_table(
                table_name
            ):
                return table_name
        return first_existing

    def _detect_kernel_table(self) -> Optional[str]:
        if "CUPTI_ACTIVITY_KIND_KERNEL" in self.tables:
            return "CUPTI_ACTIVITY_KIND_KERNEL"
        candidates = [
            t
            for t in self.tables
            if "KERNEL" in t.upper() and not t.upper().startswith("ENUM_")
        ]
        if not candidates:
            return None
        candidates.sort()
        return candidates[0]

    def table_exists(self, table_name: str) -> bool:
        return str(table_name) in set(self.tables)

    def _table_columns(self, table_name: str) -> set[str]:
        key = str(table_name)
        if key in self._columns_cache:
            return self._columns_cache[key]
        if key not in set(self.tables):
            self._columns_cache[key] = set()
            return set()
        try:
            rows = self._conn.execute(f"PRAGMA table_info({key});").fetchall()
            cols = {str(row[1]) for row in rows if len(row) > 1}
        except sqlite3.Error:
            cols = set()
        self._columns_cache[key] = cols
        return cols

    def columns(self, table_name: str) -> List[str]:
        cols = sorted(self._table_columns(table_name))
        return cols

    def has_columns(self, table_name: str, required: Sequence[str]) -> bool:
        cols = self._table_columns(table_name)
        return all(col in cols for col in required)

    def resolve_column(self, table_name: str, aliases: Sequence[str]) -> Optional[str]:
        cols = self._table_columns(table_name)
        for alias in aliases:
            if alias in cols:
                return alias
        return None

    def summary(self) -> Dict[str, object]:
        return {
            "table_count": len(self.tables),
            "tables": list(self.tables),
            "meta": dict(self.meta),
            "version": {
                "exporter_version": self.version.exporter_version,
                "export_schema_version": self.version.export_schema_version,
                "adapter_family": self.version.adapter_family,
                "known_version": self.version.known_version,
                "parsed_version": list(self.version.parsed_version),
            },
            "canonical_tables": {
                "meta": self.meta_table,
                "string_ids": self.string_table,
                "kernel": self.kernel_table,
                "runtime": self.runtime_table,
                "nvtx": self.nvtx_table,
                "memcpy": self.memcpy_table,
                "memset": self.memset_table,
                "sync": self.sync_table,
                "metrics": self.metrics_table,
            },
            "metrics_columns": {
                "timestamp": self.metrics_timestamp_col,
                "metric_id": self.metrics_id_col,
                "value": self.metrics_value_col,
            },
        }
