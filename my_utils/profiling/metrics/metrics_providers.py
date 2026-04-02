from __future__ import annotations

import csv
import json
import os
import pstats
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .metrics_provider import BaseMetricsProvider, ProviderCapabilities
from .metrics_types import MetricEvent


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


def _normalize_name(raw: str) -> str:
    text = str(raw).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _safe_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_timestamp_seconds(value, *, default_now: Optional[float] = None) -> float:
    fallback = time.time() if default_now is None else float(default_now)
    if value is None:
        return fallback

    if isinstance(value, (int, float)):
        numeric = float(value)
        abs_numeric = abs(numeric)
        if abs_numeric > 1e16:  # ns
            return numeric / 1e9
        if abs_numeric > 1e13:  # us
            return numeric / 1e6
        if abs_numeric > 1e11:  # ms
            return numeric / 1e3
        if abs_numeric > 1e9:  # epoch seconds
            return numeric
        return fallback

    text = str(value).strip()
    if not text:
        return fallback
    maybe_number = _safe_float(text)
    if maybe_number is not None:
        return _parse_timestamp_seconds(maybe_number, default_now=fallback)

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
        return float(dt.timestamp())
    except Exception:
        return fallback


def _duration_to_ms(value: float, unit: str) -> Optional[float]:
    u = str(unit or "").lower()
    if u in ("ms", "millisecond", "milliseconds"):
        return float(value)
    if u in ("us", "microsecond", "microseconds"):
        return float(value) / 1000.0
    if u in ("ns", "nanosecond", "nanoseconds"):
        return float(value) / 1_000_000.0
    if u in ("s", "sec", "second", "seconds"):
        return float(value) * 1000.0
    return None


def _read_incremental_text_lines(
    path: Path,
    *,
    file_pos: int,
    remainder: str,
) -> Tuple[List[str], int, str]:
    if not path.exists():
        return [], 0, ""

    try:
        file_size = int(path.stat().st_size)
    except OSError:
        return [], 0, remainder

    start_pos = max(0, int(file_pos))
    carry = str(remainder or "")
    if file_size < start_pos:
        start_pos = 0
        carry = ""

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(start_pos)
        chunk = handle.read()
        next_pos = handle.tell()

    if not chunk:
        return [], int(next_pos), carry

    text = carry + chunk
    ends_with_newline = text.endswith("\n") or text.endswith("\r")
    lines = text.splitlines()
    next_remainder = ""
    if not ends_with_newline and lines:
        next_remainder = lines.pop()

    return lines, int(next_pos), next_remainder


def _flatten_numeric_fields(payload: object, *, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = _normalize_name(f"{prefix}_{key}" if prefix else str(key))
            if isinstance(value, dict):
                out.update(_flatten_numeric_fields(value, prefix=name))
                continue
            if isinstance(value, list):
                continue
            fv = _safe_float(value)
            if fv is None:
                continue
            out[name] = fv
    return out


def _read_incremental_csv_rows(
    csv_path: Path,
    *,
    file_pos: int,
    fieldnames: Optional[List[str]],
) -> Tuple[List[Dict[str, str]], int, Optional[List[str]]]:
    if not csv_path.exists():
        return [], 0, fieldnames

    try:
        file_size = int(csv_path.stat().st_size)
    except OSError:
        return [], 0, fieldnames

    start_pos = max(0, int(file_pos))
    current_fieldnames = list(fieldnames) if fieldnames else None
    if file_size < start_pos:
        # file truncated/rotated; restart from beginning
        start_pos = 0
        current_fieldnames = None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        if start_pos <= 0:
            reader = csv.DictReader(handle)
            current_fieldnames = list(reader.fieldnames or [])
        else:
            handle.seek(start_pos)
            if current_fieldnames:
                reader = csv.DictReader(handle, fieldnames=current_fieldnames)
            else:
                handle.seek(0)
                reader = csv.DictReader(handle)
                current_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
        next_pos = handle.tell()

    return rows, int(next_pos), current_fieldnames


class MyTimerMetricsProvider(BaseMetricsProvider):
    provider_id = "my_timer"
    _capabilities = ProviderCapabilities(
        provider_type="my_timer",
        source_mode="online",
        metric_prefixes=["latency"],
        dimensions=["stage", "rank", "step", "device"],
        supports_incremental=True,
        supports_step_scope=True,
        supports_rank_scope=True,
    )

    def __init__(
        self,
        timer,
        *,
        include_cpu: bool = True,
        include_cuda: bool = True,
        provider_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.timer = timer
        self.include_cpu = include_cpu
        self.include_cuda = include_cuda
        self._cursor = 0

    def get_metrics(self) -> List[MetricEvent]:
        if self.timer is None:
            return []

        records = list(getattr(self.timer, "records", []) or [])
        if len(records) < self._cursor:
            self._cursor = 0

        selected = records[self._cursor :]
        self._cursor = len(records)

        now = time.time()
        events: List[MetricEvent] = []
        for record in selected:
            stage = str(record.get("stage", "unknown"))
            rank = str(record.get("rank", "0"))
            step = str(record.get("iteration", ""))
            node_id = record.get("node_id")
            parent_id = record.get("parent_id")

            base_tags = {"stage": stage, "rank": rank}
            if step:
                base_tags["step"] = step

            cpu_ms = _safe_float(record.get("cpu_duration_ms"))
            cuda_ms = _safe_float(record.get("cuda_duration_ms"))
            abs_start = _safe_float(record.get("abs_start_time"))
            abs_end = _safe_float(record.get("abs_end_time"))

            base_attrs: Dict[str, object] = {}
            if abs_start is not None:
                base_attrs["start_timestamp"] = abs_start
            if abs_end is not None:
                base_attrs["end_timestamp"] = abs_end

            if self.include_cpu and cpu_ms is not None:
                event_ts = abs_end
                if event_ts is None and abs_start is not None:
                    event_ts = abs_start + cpu_ms / 1000.0
                if event_ts is None:
                    event_ts = now
                events.append(
                    MetricEvent(
                        timestamp=event_ts,
                        name="latency.stage",
                        value=cpu_ms,
                        unit="ms",
                        provider_id=self.provider_id,
                        tags={**base_tags, "device": "cpu"},
                        attributes=dict(base_attrs),
                        node_id=str(node_id) if node_id is not None else None,
                        parent_id=str(parent_id) if parent_id is not None else None,
                    )
                )

            if self.include_cuda and cuda_ms is not None:
                event_ts = abs_end
                if event_ts is None and abs_start is not None:
                    event_ts = abs_start + cuda_ms / 1000.0
                if event_ts is None:
                    event_ts = now
                events.append(
                    MetricEvent(
                        timestamp=event_ts,
                        name="latency.stage",
                        value=cuda_ms,
                        unit="ms",
                        provider_id=self.provider_id,
                        tags={**base_tags, "device": "cuda"},
                        attributes=dict(base_attrs),
                        node_id=str(node_id) if node_id is not None else None,
                        parent_id=str(parent_id) if parent_id is not None else None,
                    )
                )

        return events


class TorchProfilerMetricsProvider(BaseMetricsProvider):
    provider_id = "torch_profiler"
    _capabilities = ProviderCapabilities(
        provider_type="torch_profiler",
        source_mode="online",
        metric_prefixes=["latency", "memory", "compute"],
        dimensions=["op", "step", "device"],
        supports_incremental=True,
        supports_step_scope=True,
    )

    def __init__(
        self,
        profiler,
        *,
        include_memory: bool = True,
        include_flops: bool = True,
        provider_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.profiler = profiler
        self.include_memory = include_memory
        self.include_flops = include_flops
        self._cursor = 0

    def get_metrics(self) -> List[MetricEvent]:
        if self.profiler is None or not hasattr(self.profiler, "events"):
            return []

        profiler_events = list(self.profiler.events() or [])
        if len(profiler_events) < self._cursor:
            self._cursor = 0
        selected = profiler_events[self._cursor :]
        self._cursor = len(profiler_events)

        now = time.time()
        metrics: List[MetricEvent] = []
        for evt in selected:
            op_name = str(getattr(evt, "name", "unknown"))
            tags = {
                "op": op_name,
                "device": "cuda" if bool(getattr(evt, "is_cuda", False)) else "cpu",
            }

            step_num = getattr(evt, "step_num", None)
            if step_num is not None:
                tags["step"] = str(step_num)

            for attr, event_name in (
                ("self_cpu_time_total", "latency.op.self_cpu"),
                ("cpu_time_total", "latency.op.total_cpu"),
                ("self_cuda_time_total", "latency.op.self_cuda"),
                ("cuda_time_total", "latency.op.total_cuda"),
            ):
                value = _safe_float(getattr(evt, attr, None))
                if value is None or value <= 0:
                    continue
                metrics.append(
                    MetricEvent(
                        timestamp=now,
                        name=event_name,
                        value=value,
                        unit="us",
                        provider_id=self.provider_id,
                        tags=tags,
                    )
                )

            if self.include_memory:
                for attr, event_name in (
                    ("cpu_memory_usage", "memory.op.cpu"),
                    ("cuda_memory_usage", "memory.op.cuda"),
                    ("self_cpu_memory_usage", "memory.op.self_cpu"),
                    ("self_cuda_memory_usage", "memory.op.self_cuda"),
                ):
                    value = _safe_float(getattr(evt, attr, None))
                    if value is None:
                        continue
                    metrics.append(
                        MetricEvent(
                            timestamp=now,
                            name=event_name,
                            value=value,
                            unit="bytes",
                            provider_id=self.provider_id,
                            tags=tags,
                        )
                    )

            if self.include_flops:
                flops = _safe_float(getattr(evt, "flops", None))
                if flops is not None and flops > 0:
                    metrics.append(
                        MetricEvent(
                            timestamp=now,
                            name="compute.op.flops",
                            value=flops,
                            unit="flops",
                            provider_id=self.provider_id,
                            tags=tags,
                        )
                    )

        return metrics


class ModuleProfilerMetricsProvider(BaseMetricsProvider):
    provider_id = "module_profiler"
    _capabilities = ProviderCapabilities(
        provider_type="module_profiler",
        source_mode="online",
        metric_prefixes=["latency"],
        dimensions=["module", "run_count"],
        supports_incremental=False,
    )

    def __init__(self, module_profiler, *, provider_id: Optional[str] = None, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.module_profiler = module_profiler
        self._last_signature: Optional[str] = None

    def get_metrics(self) -> List[MetricEvent]:
        if self.module_profiler is None or not hasattr(self.module_profiler, "summary"):
            return []

        try:
            df = self.module_profiler.summary()
        except Exception:
            return []

        if df is None or len(df) == 0:
            return []

        signature = f"{len(df)}-{float(df['run_count'].sum()) if 'run_count' in df.columns else len(df)}"
        if signature == self._last_signature:
            return []
        self._last_signature = signature

        now = time.time()
        events: List[MetricEvent] = []
        for _, row in df.iterrows():
            module_name = str(row.get("module_name", "unknown"))
            tags = {"module": module_name}
            run_count = row.get("run_count", None)
            if run_count is not None:
                tags["run_count"] = str(int(run_count))

            for col, metric_name in (
                ("mean_ms", "latency.module.mean"),
                ("median_ms", "latency.module.median"),
                ("std_ms", "latency.module.std"),
                ("total_ms", "latency.module.total"),
                ("percentage", "latency.module.share_percent"),
            ):
                if col not in row:
                    continue
                value = _safe_float(row[col])
                if value is None:
                    continue
                unit = "ms" if col != "percentage" else "percent"
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name=metric_name,
                        value=value,
                        unit=unit,
                        provider_id=self.provider_id,
                        tags=tags,
                    )
                )

        return events


class TableCsvMetricsProvider(BaseMetricsProvider):
    """Generic table CSV provider used for external tool outputs."""

    provider_id = "csv_table"
    _capabilities = ProviderCapabilities(
        provider_type="table_csv",
        source_mode="offline",
        metric_prefixes=["external"],
        dimensions=["csv_columns"],
        supports_incremental=True,
    )

    def __init__(
        self,
        csv_path: str,
        *,
        value_column: str,
        name_column: Optional[str] = None,
        tag_columns: Optional[Sequence[str]] = None,
        unit: str = "",
        event_name_prefix: str = "external",
        provider_id: str = "csv_table",
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        self.csv_path = Path(csv_path)
        self.value_column = value_column
        self.name_column = name_column
        self.tag_columns = list(tag_columns or [])
        self.unit = unit
        self.event_name_prefix = event_name_prefix
        self.provider_id = provider_id
        self._file_pos = 0
        self._csv_fieldnames: Optional[List[str]] = None

    def get_metrics(self) -> List[MetricEvent]:
        if not self.csv_path.exists():
            self._file_pos = 0
            self._csv_fieldnames = None
            return []

        rows, self._file_pos, self._csv_fieldnames = _read_incremental_csv_rows(
            self.csv_path,
            file_pos=self._file_pos,
            fieldnames=self._csv_fieldnames,
        )
        if not rows:
            return []

        events: List[MetricEvent] = []
        now = time.time()
        for row in rows:
            value = _safe_float(row.get(self.value_column))
            if value is None:
                continue

            raw_name = row.get(self.name_column, self.value_column) if self.name_column else self.value_column
            event_name = f"{self.event_name_prefix}.{_normalize_name(raw_name)}"
            tags = {col: str(row.get(col, "")) for col in self.tag_columns if row.get(col) is not None}
            events.append(
                MetricEvent(
                    timestamp=now,
                    name=event_name,
                    value=value,
                    unit=self.unit,
                    provider_id=self.provider_id,
                    tags=tags,
                )
            )

        return events


class NcuCsvMetricsProvider(BaseMetricsProvider):
    provider_id = "ncu_csv"
    _capabilities = ProviderCapabilities(
        provider_type="ncu_csv",
        source_mode="offline",
        metric_prefixes=["gpu", "compute", "memory"],
        dimensions=["kernel"],
        supports_incremental=True,
    )

    def __init__(
        self,
        csv_path: str,
        *,
        metrics_allowlist: Optional[Iterable[str]] = None,
        provider_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.csv_path = Path(csv_path)
        self.metrics_allowlist = set(metrics_allowlist or [])
        self._file_pos = 0
        self._csv_fieldnames: Optional[List[str]] = None

    def get_metrics(self) -> List[MetricEvent]:
        if not self.csv_path.exists():
            self._file_pos = 0
            self._csv_fieldnames = None
            return []

        pending, self._file_pos, self._csv_fieldnames = _read_incremental_csv_rows(
            self.csv_path,
            file_pos=self._file_pos,
            fieldnames=self._csv_fieldnames,
        )
        if not pending:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in pending:
            metric_name = str(row.get("Metric Name", "")).strip()
            if not metric_name:
                continue
            if self.metrics_allowlist and metric_name not in self.metrics_allowlist:
                continue

            value = _safe_float(row.get("Metric Value"))
            if value is None:
                continue

            tags = {}
            kernel = row.get("Kernel Name")
            if kernel:
                tags["kernel"] = str(kernel)

            events.append(
                MetricEvent(
                    timestamp=now,
                    name=f"gpu.ncu.{_normalize_name(metric_name)}",
                    value=value,
                    unit=str(row.get("Metric Unit", "")),
                    provider_id=self.provider_id,
                    tags=tags,
                )
            )
        return events


class _LegacyNsysSqliteMetricsProvider(BaseMetricsProvider):
    provider_id = "nsys_sqlite"
    _capabilities = ProviderCapabilities(
        provider_type="nsys_sqlite_legacy",
        source_mode="offline",
        metric_prefixes=["latency", "io"],
        dimensions=["kernel"],
        supports_incremental=True,
    )

    def __init__(self, sqlite_path: str, *, provider_id: Optional[str] = None, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.sqlite_path = Path(sqlite_path)
        self._last_rowid: Dict[str, int] = {}

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        query = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;"
        row = conn.execute(query, (table_name,)).fetchone()
        return row is not None

    def _read_kernels(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        table = "CUPTI_ACTIVITY_KIND_KERNEL"
        if not self._table_exists(conn, table):
            return []

        last = self._last_rowid.get(table, 0)
        query = (
            f"SELECT rowid, start, end, demangledName "
            f"FROM {table} WHERE rowid > ? ORDER BY rowid ASC"
        )
        rows = conn.execute(query, (last,)).fetchall()
        if not rows:
            return []

        self._last_rowid[table] = int(rows[-1][0])
        now = time.time()
        events: List[MetricEvent] = []
        for rowid, start, end, kernel_name in rows:
            start_f = _safe_float(start)
            end_f = _safe_float(end)
            if start_f is None or end_f is None or end_f <= start_f:
                continue
            duration_us = (end_f - start_f) / 1000.0
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="latency.kernel.cuda",
                    value=duration_us,
                    unit="us",
                    provider_id=self.provider_id,
                    tags={"kernel": str(kernel_name or "unknown")},
                    node_id=str(rowid),
                )
            )
        return events

    def _read_memcpy(self, conn: sqlite3.Connection) -> List[MetricEvent]:
        table = "CUPTI_ACTIVITY_KIND_MEMCPY"
        if not self._table_exists(conn, table):
            return []

        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table});").fetchall()]
        required = {"rowid", "start", "end", "bytes"}
        if not {"start", "end", "bytes"}.issubset(set(columns)):
            return []

        last = self._last_rowid.get(table, 0)
        query = f"SELECT rowid, start, end, bytes FROM {table} WHERE rowid > ? ORDER BY rowid ASC"
        rows = conn.execute(query, (last,)).fetchall()
        if not rows:
            return []
        self._last_rowid[table] = int(rows[-1][0])

        now = time.time()
        events: List[MetricEvent] = []
        for rowid, start, end, nbytes in rows:
            start_f = _safe_float(start)
            end_f = _safe_float(end)
            bytes_f = _safe_float(nbytes)
            if start_f is None or end_f is None or bytes_f is None or end_f <= start_f:
                continue

            duration_us = (end_f - start_f) / 1000.0
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="latency.memcpy.cuda",
                    value=duration_us,
                    unit="us",
                    provider_id=self.provider_id,
                    tags={},
                    node_id=str(rowid),
                )
            )
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="io.memcpy.bytes",
                    value=bytes_f,
                    unit="bytes",
                    provider_id=self.provider_id,
                    tags={},
                    node_id=str(rowid),
                )
            )
        return events

    def get_metrics(self) -> List[MetricEvent]:
        if not self.sqlite_path.exists():
            return []

        conn = sqlite3.connect(str(self.sqlite_path))
        try:
            result: List[MetricEvent] = []
            result.extend(self._read_kernels(conn))
            result.extend(self._read_memcpy(conn))
            return result
        finally:
            conn.close()


# Use the schema-adaptive parser implementation.
from ..sources.nsys_sqlite_provider import (  # noqa: E402
    NsysSqliteGlobMetricsProvider as NsysSqliteGlobMetricsProvider,
)
from ..sources.nsys_sqlite_provider import NsysSqliteMetricsProvider as NsysSqliteMetricsProvider  # noqa: E402


class CProfileStatsProvider(BaseMetricsProvider):
    provider_id = "cprofile"
    _capabilities = ProviderCapabilities(
        provider_type="cprofile",
        source_mode="offline",
        metric_prefixes=["latency", "calls"],
        dimensions=["file", "line", "func"],
        supports_incremental=True,
    )

    def __init__(self, stats_path: str, *, provider_id: Optional[str] = None, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.stats_path = Path(stats_path)
        self._last_mtime = 0.0

    def get_metrics(self) -> List[MetricEvent]:
        if not self.stats_path.exists():
            return []

        mtime = self.stats_path.stat().st_mtime
        if mtime <= self._last_mtime:
            return []
        self._last_mtime = mtime

        stats = pstats.Stats(str(self.stats_path))
        now = time.time()
        events: List[MetricEvent] = []
        for (filename, line, func_name), values in stats.stats.items():
            cc, nc, tt, ct, _ = values
            tags = {
                "file": str(filename),
                "line": str(line),
                "func": str(func_name),
            }
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="latency.python.self",
                    value=float(tt) * 1000.0,
                    unit="ms",
                    provider_id=self.provider_id,
                    tags=tags,
                )
            )
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="latency.python.total",
                    value=float(ct) * 1000.0,
                    unit="ms",
                    provider_id=self.provider_id,
                    tags=tags,
                )
            )
            events.append(
                MetricEvent(
                    timestamp=now,
                    name="calls.python",
                    value=int(nc),
                    unit="count",
                    provider_id=self.provider_id,
                    tags={**tags, "primitive_calls": str(int(cc))},
                )
            )
        return events


class PerfStatTextProvider(BaseMetricsProvider):
    provider_id = "perf_stat"
    _capabilities = ProviderCapabilities(
        provider_type="perf_stat",
        source_mode="offline",
        metric_prefixes=["perf"],
        dimensions=["counter_name"],
        supports_incremental=True,
    )
    _LINE = re.compile(r"^\s*([0-9][0-9,\.]*)\s+([A-Za-z0-9_\-.:/]+).*$")

    def __init__(self, stat_path: str, *, provider_id: Optional[str] = None, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.stat_path = Path(stat_path)
        self._last_mtime = 0.0

    def get_metrics(self) -> List[MetricEvent]:
        if not self.stat_path.exists():
            return []
        mtime = self.stat_path.stat().st_mtime
        if mtime <= self._last_mtime:
            return []
        self._last_mtime = mtime

        now = time.time()
        events: List[MetricEvent] = []
        with self.stat_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                match = self._LINE.match(line)
                if not match:
                    continue
                value = _safe_float(match.group(1))
                metric = match.group(2)
                if value is None:
                    continue
                events.append(
                    MetricEvent(
                        timestamp=now,
                        name=f"perf.{_normalize_name(metric)}",
                        value=value,
                        unit="",
                        provider_id=self.provider_id,
                        tags={},
                        attributes={"raw_line": line},
                    )
                )
        return events


class DcgmCsvMetricsProvider(BaseMetricsProvider):
    provider_id = "dcgm_csv"
    _capabilities = ProviderCapabilities(
        provider_type="dcgm_csv",
        source_mode="offline",
        metric_prefixes=["gpu", "dcgm", "thermals", "power"],
        dimensions=["gpu", "metric_name", "rank"],
        supports_incremental=True,
        supports_rank_scope=True,
    )

    _DEFAULT_VALUE_COLUMNS = ("value", "metric_value", "val", "avg", "average")
    _DEFAULT_METRIC_COLUMNS = (
        "metric_name",
        "metric",
        "field_name",
        "field",
        "fieldtag",
        "name",
        "counter",
        "key",
    )
    _DEFAULT_TS_COLUMNS = ("timestamp", "ts", "time", "timestamp_ns", "timestamp_ms")
    _DEFAULT_GPU_COLUMNS = ("gpu", "gpu_id", "device_id", "device", "entityid", "entity_id", "uuid")
    _DEFAULT_UNIT_COLUMNS = ("unit", "metric_unit")
    _DEFAULT_RANK_COLUMNS = ("rank", "global_rank", "local_rank")

    def __init__(
        self,
        csv_path: str,
        *,
        value_column: str = "",
        metric_column: str = "",
        timestamp_column: str = "",
        gpu_column: str = "",
        rank_column: str = "",
        unit_column: str = "",
        tag_columns: Optional[Sequence[str]] = None,
        metric_allowlist: Optional[Iterable[str]] = None,
        event_name_prefix: str = "gpu.dcgm",
        provider_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.csv_path = Path(csv_path)
        self.value_column = str(value_column or "").strip()
        self.metric_column = str(metric_column or "").strip()
        self.timestamp_column = str(timestamp_column or "").strip()
        self.gpu_column = str(gpu_column or "").strip()
        self.rank_column = str(rank_column or "").strip()
        self.unit_column = str(unit_column or "").strip()
        self.tag_columns = [str(col) for col in (tag_columns or []) if str(col).strip()]
        self.metric_allowlist = {str(name).strip() for name in (metric_allowlist or []) if str(name).strip()}
        self.event_name_prefix = str(event_name_prefix or "gpu.dcgm").strip().rstrip(".")
        self._file_pos = 0
        self._csv_fieldnames: Optional[List[str]] = None

    @staticmethod
    def _pick_column(
        row: Dict[str, str],
        explicit: str,
        defaults: Sequence[str],
    ) -> Optional[str]:
        if explicit and explicit in row:
            return explicit
        for name in defaults:
            if name in row:
                return name
        return None

    def get_metrics(self) -> List[MetricEvent]:
        if not self.csv_path.exists():
            self._file_pos = 0
            self._csv_fieldnames = None
            return []

        rows, self._file_pos, self._csv_fieldnames = _read_incremental_csv_rows(
            self.csv_path,
            file_pos=self._file_pos,
            fieldnames=self._csv_fieldnames,
        )
        if not rows:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for row in rows:
            value_col = self._pick_column(row, self.value_column, self._DEFAULT_VALUE_COLUMNS)
            metric_col = self._pick_column(row, self.metric_column, self._DEFAULT_METRIC_COLUMNS)
            if not value_col or not metric_col:
                continue

            metric_raw = str(row.get(metric_col, "")).strip()
            if not metric_raw:
                field_id = row.get("fieldId") or row.get("field_id")
                metric_raw = f"field_{field_id}" if field_id is not None else ""
            if not metric_raw:
                continue
            if self.metric_allowlist and metric_raw not in self.metric_allowlist:
                continue

            value = _safe_float(row.get(value_col))
            if value is None:
                continue

            ts_col = self._pick_column(row, self.timestamp_column, self._DEFAULT_TS_COLUMNS)
            unit_col = self._pick_column(row, self.unit_column, self._DEFAULT_UNIT_COLUMNS)
            gpu_col = self._pick_column(row, self.gpu_column, self._DEFAULT_GPU_COLUMNS)
            rank_col = self._pick_column(row, self.rank_column, self._DEFAULT_RANK_COLUMNS)

            tags: Dict[str, str] = {}
            if gpu_col and row.get(gpu_col) not in (None, ""):
                tags["gpu"] = str(row.get(gpu_col))
            if rank_col and row.get(rank_col) not in (None, ""):
                tags["rank"] = str(row.get(rank_col))
            for col in self.tag_columns:
                if col in row and row.get(col) not in (None, ""):
                    tags[col] = str(row.get(col))

            timestamp = _parse_timestamp_seconds(row.get(ts_col), default_now=now) if ts_col else now
            unit = str(row.get(unit_col) or "") if unit_col else ""
            events.append(
                MetricEvent(
                    timestamp=timestamp,
                    name=f"{self.event_name_prefix}.{_normalize_name(metric_raw)}",
                    value=value,
                    unit=unit,
                    provider_id=self.provider_id,
                    tags=tags,
                )
            )

        return events


class NcclLogMetricsProvider(BaseMetricsProvider):
    provider_id = "nccl_log"
    _capabilities = ProviderCapabilities(
        provider_type="nccl_log",
        source_mode="offline",
        metric_prefixes=["comm", "nccl"],
        dimensions=["rank", "op", "level"],
        supports_incremental=True,
        supports_rank_scope=True,
    )

    _ISO_TS = re.compile(
        r"^\s*(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    )
    _RANK = re.compile(r"(?:\brank\b\s*[:=]?\s*(\d+))|\[(\d+)\]")
    _LEVEL = re.compile(r"\b(INFO|WARN|WARNING|ERROR|TRACE|FATAL)\b", re.IGNORECASE)
    _OP = re.compile(
        r"\b(all_?reduce|all_?gather|reduce_?scatter|broadcast|sendrecv|send|recv)\b",
        re.IGNORECASE,
    )
    _DURATION = re.compile(
        r"(?:time|duration|latency)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*(ns|us|ms|s)\b",
        re.IGNORECASE,
    )
    _BUSBW = re.compile(r"\bbusbw\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    _ALGBW = re.compile(r"\balgbw\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    _BYTES = re.compile(r"\b(?:nbytes|bytes|size)\s*[:=]?\s*([0-9]+)\b", re.IGNORECASE)

    def __init__(
        self,
        log_path: str,
        *,
        event_name_prefix: str = "comm.nccl",
        levels: Optional[Iterable[str]] = None,
        provider_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.log_path = Path(log_path)
        self.event_name_prefix = str(event_name_prefix or "comm.nccl").strip().rstrip(".")
        self.levels = {str(level).upper() for level in (levels or ["INFO", "WARN", "WARNING", "ERROR", "FATAL"])}
        self._file_pos = 0
        self._tail = ""
        self._path_rank: Optional[int] = None
        m = re.search(r"rank[_\-]?(\d+)", self.log_path.name, re.IGNORECASE)
        if m:
            self._path_rank = _safe_int(m.group(1))

    def _extract_rank(self, line: str) -> Optional[int]:
        match = self._RANK.search(str(line or ""))
        if match:
            for group in match.groups():
                rank = _safe_int(group)
                if rank is not None:
                    return rank
        return self._path_rank

    def get_metrics(self) -> List[MetricEvent]:
        if not self.log_path.exists():
            self._file_pos = 0
            self._tail = ""
            return []

        lines, self._file_pos, self._tail = _read_incremental_text_lines(
            self.log_path,
            file_pos=self._file_pos,
            remainder=self._tail,
        )
        if not lines:
            return []

        now = time.time()
        events: List[MetricEvent] = []
        for line in lines:
            if "nccl" not in line.lower():
                continue
            level_match = self._LEVEL.search(line)
            level = str(level_match.group(1).upper()) if level_match else "INFO"
            if self.levels and level not in self.levels:
                continue

            tags: Dict[str, str] = {"level": level.lower()}
            rank = self._extract_rank(line)
            if rank is not None:
                tags["rank"] = str(rank)

            op_match = self._OP.search(line)
            if op_match:
                tags["op"] = _normalize_name(op_match.group(1))

            ts_match = self._ISO_TS.search(line)
            timestamp = _parse_timestamp_seconds(ts_match.group(1), default_now=now) if ts_match else now

            duration_match = self._DURATION.search(line)
            if duration_match:
                duration_value = _safe_float(duration_match.group(1))
                unit = str(duration_match.group(2)).lower()
                if duration_value is not None:
                    duration_ms = _duration_to_ms(float(duration_value), unit)
                    if duration_ms is not None:
                        events.append(
                            MetricEvent(
                                timestamp=timestamp,
                                name=f"{self.event_name_prefix}.duration",
                                value=duration_ms,
                                unit="ms",
                                provider_id=self.provider_id,
                                tags=tags,
                                attributes={"raw_line": line},
                            )
                        )

            busbw_match = self._BUSBW.search(line)
            if busbw_match:
                busbw = _safe_float(busbw_match.group(1))
                if busbw is not None:
                    events.append(
                        MetricEvent(
                            timestamp=timestamp,
                            name=f"{self.event_name_prefix}.busbw",
                            value=busbw,
                            unit="GB/s",
                            provider_id=self.provider_id,
                            tags=tags,
                            attributes={"raw_line": line},
                        )
                    )

            algbw_match = self._ALGBW.search(line)
            if algbw_match:
                algbw = _safe_float(algbw_match.group(1))
                if algbw is not None:
                    events.append(
                        MetricEvent(
                            timestamp=timestamp,
                            name=f"{self.event_name_prefix}.algbw",
                            value=algbw,
                            unit="GB/s",
                            provider_id=self.provider_id,
                            tags=tags,
                            attributes={"raw_line": line},
                        )
                    )

            bytes_match = self._BYTES.search(line)
            if bytes_match:
                nbytes = _safe_float(bytes_match.group(1))
                if nbytes is not None:
                    events.append(
                        MetricEvent(
                            timestamp=timestamp,
                            name=f"{self.event_name_prefix}.bytes",
                            value=nbytes,
                            unit="bytes",
                            provider_id=self.provider_id,
                            tags=tags,
                            attributes={"raw_line": line},
                        )
                    )

            if level in {"WARN", "WARNING", "ERROR", "FATAL"}:
                events.append(
                    MetricEvent(
                        timestamp=timestamp,
                        name=f"{self.event_name_prefix}.issue.count",
                        value=1,
                        unit="count",
                        provider_id=self.provider_id,
                        tags=tags,
                        attributes={"raw_line": line},
                    )
                )

        return events


class RasJsonMetricsProvider(BaseMetricsProvider):
    provider_id = "ras_json"
    _capabilities = ProviderCapabilities(
        provider_type="ras_json",
        source_mode="offline",
        metric_prefixes=["ras", "health", "error"],
        dimensions=["component", "severity", "rank", "gpu"],
        supports_incremental=True,
        supports_rank_scope=True,
    )

    _DEFAULT_TIMESTAMP_KEYS = ("timestamp", "ts", "time", "time_ns", "event_time", "created_at")
    _DEFAULT_SEVERITY_KEYS = ("severity", "level", "priority")
    _DEFAULT_COMPONENT_KEYS = ("component", "subsystem", "module", "source")
    _DEFAULT_RANK_KEYS = ("rank", "global_rank", "local_rank")
    _DEFAULT_GPU_KEYS = ("gpu", "gpu_id", "device", "device_id")

    def __init__(
        self,
        json_path: str,
        *,
        event_name_prefix: str = "ras",
        json_lines: Optional[bool] = None,
        timestamp_key: str = "",
        severity_key: str = "",
        component_key: str = "",
        rank_key: str = "",
        gpu_key: str = "",
        provider_id: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__(enabled=enabled)
        if provider_id:
            self.provider_id = str(provider_id)
        self.json_path = Path(json_path)
        self.event_name_prefix = str(event_name_prefix or "ras").strip().rstrip(".")
        self.json_lines = json_lines
        self.timestamp_key = str(timestamp_key or "").strip()
        self.severity_key = str(severity_key or "").strip()
        self.component_key = str(component_key or "").strip()
        self.rank_key = str(rank_key or "").strip()
        self.gpu_key = str(gpu_key or "").strip()

        self._file_pos = 0
        self._tail = ""
        self._last_mtime = 0.0
        self._seen_event_keys: set[str] = set()

    @staticmethod
    def _pick_key(payload: Dict[str, object], explicit: str, defaults: Sequence[str]) -> Optional[str]:
        if explicit and explicit in payload:
            return explicit
        for key in defaults:
            if key in payload:
                return key
        return None

    @staticmethod
    def _as_records(payload: object) -> List[Dict[str, object]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            events = payload.get("events")
            if isinstance(events, list):
                return [item for item in events if isinstance(item, dict)]
            return [payload]
        return []

    def _record_key(self, record: Dict[str, object]) -> str:
        event_id = record.get("event_id") or record.get("id")
        if event_id is not None:
            return f"id:{event_id}"
        signature = [
            str(record.get("timestamp", "")),
            str(record.get("time", "")),
            str(record.get("severity", "")),
            str(record.get("component", "")),
            str(record.get("message", "")),
            str(record.get("rank", "")),
            str(record.get("gpu", "")),
        ]
        return "|".join(signature)

    def _record_to_events(self, record: Dict[str, object], *, now: float) -> List[MetricEvent]:
        ts_key = self._pick_key(record, self.timestamp_key, self._DEFAULT_TIMESTAMP_KEYS)
        sev_key = self._pick_key(record, self.severity_key, self._DEFAULT_SEVERITY_KEYS)
        comp_key = self._pick_key(record, self.component_key, self._DEFAULT_COMPONENT_KEYS)
        rank_key = self._pick_key(record, self.rank_key, self._DEFAULT_RANK_KEYS)
        gpu_key = self._pick_key(record, self.gpu_key, self._DEFAULT_GPU_KEYS)

        timestamp = _parse_timestamp_seconds(record.get(ts_key), default_now=now) if ts_key else now
        severity = str(record.get(sev_key, "")).strip().lower() if sev_key else ""
        component = str(record.get(comp_key, "")).strip() if comp_key else ""

        tags: Dict[str, str] = {}
        if severity:
            tags["severity"] = severity
        if component:
            tags["component"] = component
        if rank_key and record.get(rank_key) not in (None, ""):
            tags["rank"] = str(record.get(rank_key))
        if gpu_key and record.get(gpu_key) not in (None, ""):
            tags["gpu"] = str(record.get(gpu_key))

        numeric_fields = _flatten_numeric_fields(record)
        for key in list(numeric_fields.keys()):
            if "timestamp" in key or key.endswith("_time") or key.endswith("_ts"):
                numeric_fields.pop(key, None)

        events: List[MetricEvent] = []
        if not numeric_fields:
            events.append(
                MetricEvent(
                    timestamp=timestamp,
                    name=f"{self.event_name_prefix}.event.count",
                    value=1,
                    unit="count",
                    provider_id=self.provider_id,
                    tags=tags,
                    attributes={"message": str(record.get("message") or "")},
                )
            )
        else:
            for key, value in sorted(numeric_fields.items()):
                unit = "count" if any(token in key for token in ("count", "error", "fault", "fail", "retry", "timeout")) else ""
                events.append(
                    MetricEvent(
                        timestamp=timestamp,
                        name=f"{self.event_name_prefix}.{key}",
                        value=value,
                        unit=unit,
                        provider_id=self.provider_id,
                        tags=tags,
                        attributes={"message": str(record.get("message") or "")},
                    )
                )

        if severity in {"warn", "warning", "error", "critical", "fatal"}:
            events.append(
                MetricEvent(
                    timestamp=timestamp,
                    name=f"{self.event_name_prefix}.severity.{_normalize_name(severity)}.count",
                    value=1,
                    unit="count",
                    provider_id=self.provider_id,
                    tags=tags,
                    attributes={"message": str(record.get("message") or "")},
                )
            )

        return events

    def _parse_json_lines(self) -> List[Dict[str, object]]:
        lines, self._file_pos, self._tail = _read_incremental_text_lines(
            self.json_path,
            file_pos=self._file_pos,
            remainder=self._tail,
        )
        records: List[Dict[str, object]] = []
        for line in lines:
            text = str(line).strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except Exception:
                continue
            records.extend(self._as_records(parsed))
        return records

    def _parse_json_document(self) -> List[Dict[str, object]]:
        mtime = self.json_path.stat().st_mtime
        if mtime <= self._last_mtime:
            return []
        self._last_mtime = mtime
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        self._file_pos = 0
        self._tail = ""
        return self._as_records(payload)

    def get_metrics(self) -> List[MetricEvent]:
        if not self.json_path.exists():
            self._file_pos = 0
            self._tail = ""
            self._last_mtime = 0.0
            self._seen_event_keys.clear()
            return []

        now = time.time()
        if self.json_lines is None:
            suffix = self.json_path.suffix.lower()
            is_json_lines = suffix in (".jsonl", ".ndjson")
        else:
            is_json_lines = bool(self.json_lines)

        records = self._parse_json_lines() if is_json_lines else self._parse_json_document()
        if not records:
            return []

        events: List[MetricEvent] = []
        for record in records:
            key = self._record_key(record)
            if key in self._seen_event_keys:
                continue
            self._seen_event_keys.add(key)
            events.extend(self._record_to_events(record, now=now))

        if len(self._seen_event_keys) > 200_000:
            # Keep memory bounded for very long-running collectors.
            self._seen_event_keys.clear()

        return events
