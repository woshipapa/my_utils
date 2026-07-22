# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def _to_number(value: object) -> Optional[float]:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text.lower() in {"nan", "na", "none", "null", "-"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_int(value: object) -> Optional[int]:
    number = _to_number(value)
    if number is None:
        return None
    return int(number)


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _stats(values: Iterable[object]) -> Dict[str, object]:
    nums = sorted(float(x) for x in (_to_number(v) for v in values) if x is not None)
    if not nums:
        return {"count": 0, "sum": 0.0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    total = float(sum(nums))
    return {
        "count": len(nums),
        "sum": total,
        "avg": total / float(len(nums)),
        "p50": _percentile(nums, 50),
        "p90": _percentile(nums, 90),
        "max": float(nums[-1]),
    }


def _like_match(text: str, pattern: str) -> bool:
    p = str(pattern or "").strip()
    if not p or p in {"%", "*"}:
        return True
    p = p.replace("*", "%")
    parts: List[str] = ["^"]
    for ch in p:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    parts.append("$")
    return re.match("".join(parts), str(text or ""), flags=re.IGNORECASE) is not None


def _message_size_bucket(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    unit_idx = 0
    while value >= 1024.0 and unit_idx < len(units) - 1:
        value /= 1024.0
        unit_idx += 1
    lower = max(1, int(math.floor(value)))
    return f"{lower}-{lower + 1}{units[unit_idx]}"


def _iter_input_files(path_text: str, *, prometheus: bool = False) -> List[Path]:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"NCCL inspector input not found: {path_text}")
    if path.is_file():
        return [path]
    if prometheus:
        suffixes = {".prom", ".txt", ".metrics", ".out"}
    else:
        suffixes = {".json", ".jsonl", ".log", ".out", ".txt"}
    files = [
        p
        for p in path.rglob("*")
        if p.is_file() and (p.suffix.lower() in suffixes or not p.suffix)
    ]
    return sorted(files)


def _load_json_objects(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            return [payload]
    except Exception:
        pass

    out: List[Dict[str, object]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or not raw.startswith("{"):
            continue
        try:
            item = json.loads(raw)
        except Exception:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _kernel_span_us(perf: Dict[str, object]) -> Optional[float]:
    trace = perf.get("event_trace_ts")
    if not isinstance(trace, dict):
        return None
    events = trace.get("kernel_events")
    if not isinstance(events, list):
        return None
    starts: List[int] = []
    stops: List[int] = []
    for item in events:
        if not isinstance(item, dict):
            continue
        start = _to_int(item.get("kernel_start_ts"))
        stop = _to_int(item.get("kernel_stop_ts"))
        if start is not None and stop is not None and stop >= start:
            starts.append(start)
            stops.append(stop)
    if not starts or not stops:
        return None
    return float(max(stops) - min(starts))


@dataclass
class NcclInspectorEvent:
    source_file: str
    kind: str
    comm_id: str = ""
    comm_name: str = ""
    rank: int = -1
    nranks: int = -1
    nnodes: int = -1
    hostname: str = ""
    pid: int = -1
    dump_timestamp_us: int = -1
    op: str = ""
    seq: int = -1
    peer: int = -1
    msg_size_bytes: int = 0
    exec_time_us: float = 0.0
    algobw_gbs: float = 0.0
    busbw_gbs: float = 0.0
    timing_source: str = ""
    kernel_span_us: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_file": self.source_file,
            "kind": self.kind,
            "comm_id": self.comm_id,
            "comm_name": self.comm_name,
            "rank": self.rank,
            "nranks": self.nranks,
            "nnodes": self.nnodes,
            "hostname": self.hostname,
            "pid": self.pid,
            "dump_timestamp_us": self.dump_timestamp_us,
            "op": self.op,
            "seq": self.seq,
            "peer": self.peer,
            "msg_size_bytes": self.msg_size_bytes,
            "message_size_bucket": _message_size_bucket(self.msg_size_bytes),
            "exec_time_us": self.exec_time_us,
            "algobw_gbs": self.algobw_gbs,
            "busbw_gbs": self.busbw_gbs,
            "timing_source": self.timing_source,
            "kernel_span_us": self.kernel_span_us,
        }


@dataclass
class NcclPrometheusMetric:
    source_file: str
    metric: str
    labels: Dict[str, str]
    value: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_file": self.source_file,
            "metric": self.metric,
            "labels": dict(self.labels),
            "value": self.value,
        }


@dataclass
class SkillParam:
    name: str
    description: str
    type: str = "str"
    required: bool = False
    default: object = None


@dataclass
class NcclInspectorSkill:
    name: str
    title: str
    description: str
    category: str
    params: List[SkillParam] = field(default_factory=list)
    run_fn: Optional[Callable[..., object]] = None


def _event_from_json_obj(
    obj: Dict[str, object], source_file: str
) -> Optional[NcclInspectorEvent]:
    header = obj.get("header")
    metadata = obj.get("metadata")
    if not isinstance(header, dict):
        header = {}
    if not isinstance(metadata, dict):
        metadata = {}

    if isinstance(obj.get("coll_perf"), dict):
        kind = "collective"
        perf = obj["coll_perf"]  # type: ignore[index]
        assert isinstance(perf, dict)
        op = str(perf.get("coll", ""))
        seq = _to_int(perf.get("coll_sn")) or -1
        peer = -1
        msg_size = _to_int(perf.get("coll_msg_size_bytes")) or 0
        exec_time = _to_number(perf.get("coll_exec_time_us")) or 0.0
        algobw = _to_number(perf.get("coll_algobw_gbs")) or 0.0
        busbw = _to_number(perf.get("coll_busbw_gbs")) or 0.0
        timing_source = str(perf.get("coll_timing_source", ""))
    elif isinstance(obj.get("p2p_perf"), dict):
        kind = "p2p"
        perf = obj["p2p_perf"]  # type: ignore[index]
        assert isinstance(perf, dict)
        op = str(perf.get("p2p", ""))
        seq = _to_int(perf.get("p2p_sn")) or -1
        peer = _to_int(perf.get("p2p_peer")) or -1
        msg_size = _to_int(perf.get("p2p_msg_size_bytes")) or 0
        exec_time = _to_number(perf.get("p2p_exec_time_us")) or 0.0
        algobw = _to_number(perf.get("p2p_algobw_gbs")) or 0.0
        busbw = _to_number(perf.get("p2p_busbw_gbs")) or 0.0
        timing_source = str(perf.get("p2p_timing_source", ""))
    else:
        return None

    return NcclInspectorEvent(
        source_file=source_file,
        kind=kind,
        comm_id=str(header.get("id", "")),
        comm_name=str(header.get("comm_name", "")),
        rank=_to_int(header.get("rank"))
        if _to_int(header.get("rank")) is not None
        else -1,
        nranks=_to_int(header.get("n_ranks"))
        if _to_int(header.get("n_ranks")) is not None
        else -1,
        nnodes=_to_int(header.get("nnodes"))
        if _to_int(header.get("nnodes")) is not None
        else -1,
        hostname=str(metadata.get("hostname", "")),
        pid=_to_int(metadata.get("pid"))
        if _to_int(metadata.get("pid")) is not None
        else -1,
        dump_timestamp_us=_to_int(metadata.get("dump_timestamp_us"))
        if _to_int(metadata.get("dump_timestamp_us")) is not None
        else -1,
        op=op,
        seq=seq,
        peer=peer,
        msg_size_bytes=msg_size,
        exec_time_us=float(exec_time),
        algobw_gbs=float(algobw),
        busbw_gbs=float(busbw),
        timing_source=timing_source,
        kernel_span_us=_kernel_span_us(perf),
    )


def load_nccl_inspector_events(path_text: str) -> List[NcclInspectorEvent]:
    events: List[NcclInspectorEvent] = []
    for path in _iter_input_files(path_text):
        for obj in _load_json_objects(path):
            event = _event_from_json_obj(obj, str(path))
            if event is not None:
                events.append(event)
    return events


_PROM_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{(.*)\}\s+([-+0-9.eE]+)\s*$")
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def _parse_prom_labels(text: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for match in _LABEL_RE.finditer(text):
        key = match.group(1)
        value = (
            match.group(2).replace(r"\"", '"').replace(r"\\", "\\").replace(r"\n", "\n")
        )
        labels[key] = value
    return labels


def load_nccl_prometheus_metrics(path_text: str) -> List[NcclPrometheusMetric]:
    metrics: List[NcclPrometheusMetric] = []
    for path in _iter_input_files(path_text, prometheus=True):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            match = _PROM_RE.match(raw)
            if not match:
                continue
            value = _to_number(match.group(3))
            if value is None:
                continue
            metrics.append(
                NcclPrometheusMetric(
                    source_file=str(path),
                    metric=match.group(1),
                    labels=_parse_prom_labels(match.group(2)),
                    value=float(value),
                )
            )
    return metrics


def _filter_events(
    events: Sequence[NcclInspectorEvent],
    *,
    kind: str = "",
    op_like: str = "%",
    comm_like: str = "%",
    min_msg_size_bytes: int = 0,
) -> List[NcclInspectorEvent]:
    out: List[NcclInspectorEvent] = []
    for event in events:
        if kind and event.kind != kind:
            continue
        if not _like_match(event.op, op_like):
            continue
        if not _like_match(event.comm_name, comm_like):
            continue
        if int(event.msg_size_bytes) < int(min_msg_size_bytes):
            continue
        out.append(event)
    return out


def _group_events(
    events: Sequence[NcclInspectorEvent],
    keys: Sequence[str],
) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[object, ...], List[NcclInspectorEvent]] = {}
    for event in events:
        row = event.to_dict()
        key = tuple(row.get(k, "") for k in keys)
        buckets.setdefault(key, []).append(event)

    rows: List[Dict[str, object]] = []
    for key, items in buckets.items():
        row = {name: value for name, value in zip(keys, key)}
        row.update(
            {
                "count": len(items),
                "exec_time_us": _stats(x.exec_time_us for x in items),
                "busbw_gbs": _stats(x.busbw_gbs for x in items),
                "algobw_gbs": _stats(x.algobw_gbs for x in items),
                "msg_size_bytes": _stats(x.msg_size_bytes for x in items),
                "ranks": sorted({x.rank for x in items if x.rank >= 0}),
                "hosts": sorted({x.hostname for x in items if x.hostname}),
            }
        )
        rows.append(row)
    rows.sort(
        key=lambda x: float(x.get("exec_time_us", {}).get("sum", 0.0)), reverse=True
    )  # type: ignore[union-attr]
    return rows


def _timing_source_summary(
    events: Sequence[NcclInspectorEvent],
) -> List[Dict[str, object]]:
    buckets: Dict[str, List[NcclInspectorEvent]] = {}
    for event in events:
        buckets.setdefault(event.timing_source or "unknown", []).append(event)
    rows = []
    total = max(1, len(events))
    for source, items in buckets.items():
        rows.append(
            {
                "timing_source": source,
                "count": len(items),
                "pct": 100.0 * float(len(items)) / float(total),
                "exec_time_us": _stats(x.exec_time_us for x in items),
            }
        )
    rows.sort(key=lambda x: int(x["count"]), reverse=True)
    return rows


def _rank_skew(
    events: Sequence[NcclInspectorEvent], *, top_k: int = 20
) -> List[Dict[str, object]]:
    seq_groups: Dict[Tuple[str, str, int], List[NcclInspectorEvent]] = {}
    for event in events:
        if event.seq < 0 or event.rank < 0:
            continue
        seq_groups.setdefault((event.kind, event.op, event.seq), []).append(event)

    rows: List[Dict[str, object]] = []
    for (kind, op, seq), items in seq_groups.items():
        if len({x.rank for x in items}) < 2:
            continue
        times = [float(x.exec_time_us) for x in items]
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / float(len(times))
        slow = max(items, key=lambda x: x.exec_time_us)
        fast = min(items, key=lambda x: x.exec_time_us)
        rows.append(
            {
                "kind": kind,
                "op": op,
                "seq": seq,
                "rank_count": len({x.rank for x in items}),
                "avg_exec_time_us": avg_time,
                "min_exec_time_us": min_time,
                "max_exec_time_us": max_time,
                "skew_ratio": (max_time / min_time) if min_time > 0 else 0.0,
                "skew_us": max_time - min_time,
                "slow_rank": slow.rank,
                "fast_rank": fast.rank,
                "msg_size_bytes": max(x.msg_size_bytes for x in items),
            }
        )
    rows.sort(key=lambda x: (float(x["skew_ratio"]), float(x["skew_us"])), reverse=True)
    return rows[: int(top_k)]


def _prometheus_summary(
    metrics: Sequence[NcclPrometheusMetric], *, top_k: int = 20
) -> Dict[str, object]:
    by_metric: Dict[str, List[NcclPrometheusMetric]] = {}
    for item in metrics:
        by_metric.setdefault(item.metric, []).append(item)

    metric_rows = []
    for name, items in by_metric.items():
        metric_rows.append(
            {
                "metric": name,
                "count": len(items),
                "value": _stats(x.value for x in items),
                "labels": sorted({k for x in items for k in x.labels.keys()}),
            }
        )
    metric_rows.sort(key=lambda x: int(x["count"]), reverse=True)

    top_rows = sorted(
        (item.to_dict() for item in metrics),
        key=lambda x: float(x.get("value", 0.0)),
        reverse=True,
    )[: int(top_k)]
    return {
        "metric_count": len(metrics),
        "source_files": sorted({x.source_file for x in metrics}),
        "metric_summary": metric_rows,
        "top_values": top_rows,
    }


class NcclInspectorSkillEngine:
    def __init__(self, path: str, prometheus_path: str = "") -> None:
        self.path = str(path)
        self.prometheus_path = str(prometheus_path or "")
        self.events = load_nccl_inspector_events(self.path) if self.path else []
        self.prometheus_metrics = (
            load_nccl_prometheus_metrics(self.prometheus_path)
            if self.prometheus_path
            else []
        )
        self._skills = self._build_skills()

    def list_skills(self) -> List[str]:
        return sorted(self._skills.keys())

    def describe_skills(self) -> Dict[str, object]:
        out = []
        for name in self.list_skills():
            item = self._skills[name]
            out.append(
                {
                    "name": item.name,
                    "title": item.title,
                    "description": item.description,
                    "category": item.category,
                    "params": [param.__dict__ for param in item.params],
                }
            )
        return {"skills": out}

    def run_skill(self, name: str, **params: object) -> object:
        skill = self._skills.get(str(name))
        if skill is None or skill.run_fn is None:
            raise KeyError(f"unknown NCCL inspector skill: {name}")
        return skill.run_fn(**params)

    def _selected_events(
        self,
        kind: str = "",
        op_like: str = "%",
        comm_like: str = "%",
        min_msg_size_bytes: int = 0,
    ) -> List[NcclInspectorEvent]:
        return _filter_events(
            self.events,
            kind=str(kind or ""),
            op_like=str(op_like or "%"),
            comm_like=str(comm_like or "%"),
            min_msg_size_bytes=int(min_msg_size_bytes or 0),
        )

    def _summary(self, **params: object) -> Dict[str, object]:
        events = self._selected_events(**params)
        return {
            "event_count": len(events),
            "collective_count": sum(1 for x in events if x.kind == "collective"),
            "p2p_count": sum(1 for x in events if x.kind == "p2p"),
            "source_files": sorted({x.source_file for x in events}),
            "communicators": len({x.comm_id for x in events if x.comm_id}),
            "comm_names": sorted({x.comm_name for x in events if x.comm_name}),
            "hosts": sorted({x.hostname for x in events if x.hostname}),
            "ranks": sorted({x.rank for x in events if x.rank >= 0}),
            "ops": sorted({x.op for x in events if x.op}),
            "exec_time_us": _stats(x.exec_time_us for x in events),
            "busbw_gbs": _stats(x.busbw_gbs for x in events),
            "algobw_gbs": _stats(x.algobw_gbs for x in events),
            "msg_size_bytes": _stats(x.msg_size_bytes for x in events),
            "timing_sources": _timing_source_summary(events),
        }

    def _top_ops(
        self, kind: str, top_k: int = 20, **params: object
    ) -> List[Dict[str, object]]:
        events = self._selected_events(kind=kind, **params)
        rows = _group_events(events, ["kind", "op", "message_size_bucket", "comm_name"])
        return rows[: int(top_k)]

    def _comm_summary(
        self, top_k: int = 50, **params: object
    ) -> List[Dict[str, object]]:
        events = self._selected_events(**params)
        rows = _group_events(events, ["comm_id", "comm_name", "kind", "op"])
        return rows[: int(top_k)]

    def _event_rows(
        self, top_k: int = 100, sort_by: str = "exec_time_us", **params: object
    ) -> List[Dict[str, object]]:
        events = self._selected_events(**params)
        key = str(sort_by or "exec_time_us")
        rows = [event.to_dict() for event in events]
        rows.sort(key=lambda x: float(x.get(key, 0.0) or 0.0), reverse=True)
        return rows[: int(top_k)]

    def _build_skills(self) -> Dict[str, NcclInspectorSkill]:
        common_params = [
            SkillParam(
                "kind",
                "filter by event kind: collective, p2p, or empty for all",
                default="",
            ),
            SkillParam(
                "op_like", "operation LIKE pattern (%/_/* supported)", default="%"
            ),
            SkillParam(
                "comm_like",
                "communicator name LIKE pattern (%/_/* supported)",
                default="%",
            ),
            SkillParam(
                "min_msg_size_bytes",
                "minimum message size in bytes",
                type="int",
                default=0,
            ),
        ]
        top_params = common_params + [
            SkillParam("top_k", "number of rows to return", type="int", default=20)
        ]
        return {
            "summary": NcclInspectorSkill(
                name="summary",
                title="NCCL Inspector Summary",
                description="Summarize NCCL Inspector JSON/JSONL events.",
                category="summary",
                params=common_params,
                run_fn=self._summary,
            ),
            "top_collectives": NcclInspectorSkill(
                name="top_collectives",
                title="Top NCCL Collectives",
                description="Group collective events by op, message-size bucket, and communicator.",
                category="collective",
                params=common_params[1:]
                + [
                    SkillParam(
                        "top_k", "number of rows to return", type="int", default=20
                    )
                ],
                run_fn=lambda top_k=20, **kw: self._top_ops(
                    "collective",
                    top_k=int(top_k),
                    op_like=str(kw.get("op_like", "%")),
                    comm_like=str(kw.get("comm_like", "%")),
                    min_msg_size_bytes=int(kw.get("min_msg_size_bytes", 0) or 0),
                ),
            ),
            "top_p2p": NcclInspectorSkill(
                name="top_p2p",
                title="Top NCCL P2P",
                description="Group P2P events by op, message-size bucket, and communicator.",
                category="p2p",
                params=common_params[1:]
                + [
                    SkillParam(
                        "top_k", "number of rows to return", type="int", default=20
                    )
                ],
                run_fn=lambda top_k=20, **kw: self._top_ops(
                    "p2p",
                    top_k=int(top_k),
                    op_like=str(kw.get("op_like", "%")),
                    comm_like=str(kw.get("comm_like", "%")),
                    min_msg_size_bytes=int(kw.get("min_msg_size_bytes", 0) or 0),
                ),
            ),
            "comm_summary": NcclInspectorSkill(
                name="comm_summary",
                title="Communicator Summary",
                description="Aggregate events by communicator, kind, and operation.",
                category="summary",
                params=top_params,
                run_fn=self._comm_summary,
            ),
            "rank_skew": NcclInspectorSkill(
                name="rank_skew",
                title="Rank Skew",
                description="Find sequence numbers whose per-rank execution time differs most.",
                category="diagnostic",
                params=top_params,
                run_fn=lambda top_k=20, **kw: _rank_skew(
                    self._selected_events(**kw), top_k=int(top_k)
                ),
            ),
            "events": NcclInspectorSkill(
                name="events",
                title="Raw Event Rows",
                description="Return normalized event rows sorted by a numeric column.",
                category="raw",
                params=top_params
                + [
                    SkillParam(
                        "sort_by",
                        "numeric event field to sort by",
                        default="exec_time_us",
                    )
                ],
                run_fn=self._event_rows,
            ),
            "prometheus_summary": NcclInspectorSkill(
                name="prometheus_summary",
                title="Prometheus Textfile Summary",
                description="Summarize NCCL Inspector Prometheus textfile metrics.",
                category="prometheus",
                params=[
                    SkillParam(
                        "top_k",
                        "number of top metric samples to return",
                        type="int",
                        default=20,
                    )
                ],
                run_fn=lambda top_k=20: _prometheus_summary(
                    self.prometheus_metrics, top_k=int(top_k)
                ),
            ),
        }


def analyze_nccl_inspector(
    path: str,
    *,
    prometheus_path: str = "",
    top_k: int = 20,
    op_like: str = "%",
    comm_like: str = "%",
    min_msg_size_bytes: int = 0,
) -> Dict[str, object]:
    engine = NcclInspectorSkillEngine(path, prometheus_path=prometheus_path)
    params = {
        "op_like": op_like,
        "comm_like": comm_like,
        "min_msg_size_bytes": min_msg_size_bytes,
    }
    summary = engine.run_skill("summary", **params)
    assert isinstance(summary, dict)
    payload: Dict[str, object] = {
        "summary": summary,
        "top_collectives": engine.run_skill("top_collectives", top_k=top_k, **params),
        "top_p2p": engine.run_skill("top_p2p", top_k=top_k, **params),
        "comm_summary": engine.run_skill("comm_summary", top_k=top_k, **params),
        "rank_skew": engine.run_skill("rank_skew", top_k=top_k, **params),
    }
    if prometheus_path:
        payload["prometheus_summary"] = engine.run_skill(
            "prometheus_summary", top_k=top_k
        )
    payload["recommendations"] = _recommendations(payload)
    return payload


def _recommendations(payload: Dict[str, object]) -> List[str]:
    recs: List[str] = []
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return recs
    timing = summary.get("timing_sources")
    if isinstance(timing, list):
        non_gpu = [
            x
            for x in timing
            if isinstance(x, dict)
            and str(x.get("timing_source", "")).lower()
            not in {"kernel_gpu", "gpu", "unknown"}
        ]
        if non_gpu:
            recs.append(
                "Some events used CPU or fallback timing. For newest NCCL Inspector, keep "
                "NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=1 when you need GPU-timed-only samples."
            )
    rank_skew = payload.get("rank_skew")
    if isinstance(rank_skew, list) and rank_skew:
        first = rank_skew[0]
        if (
            isinstance(first, dict)
            and float(first.get("skew_ratio", 0.0) or 0.0) >= 1.5
        ):
            recs.append(
                "Large per-rank skew detected for at least one NCCL sequence; inspect slow_rank, "
                "node placement, and overlap with compute/host stalls."
            )
    if int(summary.get("event_count", 0) or 0) == 0:
        recs.append(
            "No NCCL Inspector JSON events were parsed. Check NCCL_INSPECTOR_ENABLE=1 and "
            "NCCL_INSPECTOR_DUMP_DIR, or pass --prometheus-path for textfile metrics."
        )
    return recs


def analyze_nccl_inspector_to_markdown(payload: Dict[str, object]) -> str:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    lines = [
        "# NCCL Inspector Analysis",
        "",
        "## Summary",
        "",
        f"- events: {summary.get('event_count', 0)}",
        f"- collectives: {summary.get('collective_count', 0)}",
        f"- p2p: {summary.get('p2p_count', 0)}",
        f"- communicators: {summary.get('communicators', 0)}",
        f"- hosts: {', '.join(str(x) for x in summary.get('hosts', []) or []) or '(none)'}",
        "",
        "## Top Collectives",
        "",
    ]
    for row in list(payload.get("top_collectives", []) or [])[:10]:
        if not isinstance(row, dict):
            continue
        exec_stats = row.get("exec_time_us", {})
        bw_stats = row.get("busbw_gbs", {})
        if not isinstance(exec_stats, dict):
            exec_stats = {}
        if not isinstance(bw_stats, dict):
            bw_stats = {}
        lines.append(
            f"- {row.get('op')} {row.get('message_size_bucket')} comm={row.get('comm_name')}: "
            f"count={row.get('count')}, exec_sum_us={exec_stats.get('sum', 0):.3f}, "
            f"busbw_avg_gbs={bw_stats.get('avg', 0):.3f}"
        )
    lines.extend(["", "## Rank Skew", ""])
    for row in list(payload.get("rank_skew", []) or [])[:10]:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"- {row.get('kind')} {row.get('op')} seq={row.get('seq')}: "
            f"skew_ratio={float(row.get('skew_ratio', 0.0) or 0.0):.3f}, "
            f"slow_rank={row.get('slow_rank')}, fast_rank={row.get('fast_rank')}"
        )
    recs = payload.get("recommendations", [])
    if isinstance(recs, list) and recs:
        lines.extend(["", "## Recommendations", ""])
        for item in recs:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def inspector_result_to_json(payload: object, *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
