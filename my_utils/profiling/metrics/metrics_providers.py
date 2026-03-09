from __future__ import annotations

import csv
import os
import pstats
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

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
        self._processed_lines = 0

    def get_metrics(self) -> List[MetricEvent]:
        if not self.csv_path.exists():
            return []

        events: List[MetricEvent] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if self._processed_lines > len(rows):
                self._processed_lines = 0
            pending = rows[self._processed_lines :]
            self._processed_lines = len(rows)

            now = time.time()
            for row in pending:
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
        self._processed_lines = 0

    def get_metrics(self) -> List[MetricEvent]:
        if not self.csv_path.exists():
            return []

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if self._processed_lines > len(rows):
            self._processed_lines = 0
        pending = rows[self._processed_lines :]
        self._processed_lines = len(rows)

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
