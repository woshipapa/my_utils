from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence


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


def _to_number(value: object) -> Optional[float]:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.lower() in {"nan", "na", "none", "null", "-"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


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


def _load_ncu_report_module(ncu_report_module: Any = None) -> ModuleType:
    if ncu_report_module is not None:
        return ncu_report_module
    try:
        import ncu_report as mod  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "ncu_report module is required for .ncu-rep direct parsing. "
            "Install/use Nsight Compute Python Report Interface."
        ) from exc
    return mod  # type: ignore[return-value]


def _maybe_call(obj: object, name: str, default: Any = None) -> Any:
    if not hasattr(obj, name):
        return default
    value = getattr(obj, name)
    if callable(value):
        try:
            return value()
        except Exception:
            return default
    return value


def _iter_ranges(ctx: object) -> List[object]:
    ranges: List[object] = []
    try:
        for item in ctx:  # type: ignore[operator]
            ranges.append(item)
        if ranges:
            return ranges
    except Exception:
        pass
    num_ranges = _maybe_call(ctx, "num_ranges", 0) or 0
    for i in range(int(num_ranges)):
        try:
            item = ctx.range_by_idx(i)  # type: ignore[attr-defined]
        except Exception:
            item = None
        if item is not None:
            ranges.append(item)
    return ranges


def _iter_actions(range_obj: object) -> List[object]:
    actions: List[object] = []
    try:
        for item in range_obj:  # type: ignore[operator]
            actions.append(item)
        if actions:
            return actions
    except Exception:
        pass
    num_actions = _maybe_call(range_obj, "num_actions", 0) or 0
    for i in range(int(num_actions)):
        try:
            item = range_obj.action_by_idx(i)  # type: ignore[attr-defined]
        except Exception:
            item = None
        if item is not None:
            actions.append(item)
    return actions


def _metric_value(metric: object) -> Any:
    for name in ("value", "as_double", "as_uint64", "as_string"):
        if not hasattr(metric, name):
            continue
        attr = getattr(metric, name)
        try:
            if callable(attr):
                value = attr()
            else:
                value = attr
            if value is not None and str(value) != "":
                return value
        except Exception:
            continue
    return None


@dataclass
class NcuReportMetricRecord:
    kernel_name: str
    metric_name: str
    raw_value: object
    numeric_value: Optional[float]
    unit: str
    range_index: int
    action_index: int


@dataclass
class SkillParam:
    name: str
    description: str
    type: str = "str"
    required: bool = False
    default: object = None


@dataclass
class ReportSkill:
    name: str
    title: str
    description: str
    category: str
    params: List[SkillParam] = field(default_factory=list)
    run_fn: Optional[Callable[..., object]] = None


class NcuReportSkillEngine:
    def __init__(
        self,
        report_path: str,
        *,
        ncu_report_module: Any = None,
    ) -> None:
        self.report_path = str(report_path)
        self._records = load_ncu_report_records(
            report_path,
            metric_like="%",
            kernel_like="%",
            ncu_report_module=ncu_report_module,
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
                    "params": [
                        {
                            "name": p.name,
                            "description": p.description,
                            "type": p.type,
                            "required": p.required,
                            "default": p.default,
                        }
                        for p in item.params
                    ],
                }
            )
        return {
            "report_path": self.report_path,
            "metric_records": len(self._records),
            "skills": out,
        }

    def run_skill(self, name: str, **kwargs: Any) -> object:
        skill = self._skills.get(str(name))
        if not skill or not skill.run_fn:
            raise ValueError(f"unknown ncu report skill: {name}")
        params = self._resolve_params(skill, kwargs)
        return skill.run_fn(**params)

    def _resolve_params(self, skill: ReportSkill, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in skill.params:
            if p.name in kwargs:
                value = kwargs[p.name]
            elif p.default is not None:
                value = p.default
            elif p.required:
                raise ValueError(f"skill '{skill.name}' missing required param '{p.name}'")
            else:
                continue
            t = str(p.type).lower()
            if t == "int":
                value = int(value)
            elif t == "float":
                value = float(value)
            elif t == "bool":
                value = bool(value)
            else:
                value = str(value)
            out[p.name] = value
        return out

    def _filtered_records(self, *, metric_like: str = "%", kernel_like: str = "%") -> List[NcuReportMetricRecord]:
        out: List[NcuReportMetricRecord] = []
        for rec in self._records:
            if not _like_match(rec.metric_name, metric_like):
                continue
            if not _like_match(rec.kernel_name, kernel_like):
                continue
            out.append(rec)
        return out

    def _skill_summary(self, *, metric_like: str = "%", kernel_like: str = "%", top_k: int = 20) -> Dict[str, object]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        by_metric: Dict[str, int] = {}
        by_kernel: Dict[str, int] = {}
        values: List[float] = []
        non_numeric = 0
        for rec in rows:
            by_metric[rec.metric_name] = by_metric.get(rec.metric_name, 0) + 1
            by_kernel[rec.kernel_name] = by_kernel.get(rec.kernel_name, 0) + 1
            if rec.numeric_value is None:
                non_numeric += 1
            else:
                values.append(float(rec.numeric_value))
        values_sorted = sorted(values)
        return {
            "report_path": self.report_path,
            "metric_records": len(self._records),
            "filtered_records": len(rows),
            "filters": {"metric_like": metric_like, "kernel_like": kernel_like},
            "unique_metrics": len(by_metric),
            "unique_kernels": len(by_kernel),
            "numeric_values": len(values),
            "non_numeric_values": non_numeric,
            "value_stats": {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": (sum(values) / len(values)) if values else None,
                "p50": _percentile(values_sorted, 50) if values else None,
                "p90": _percentile(values_sorted, 90) if values else None,
                "p99": _percentile(values_sorted, 99) if values else None,
            },
            "top_metrics": sorted(by_metric.items(), key=lambda x: x[1], reverse=True)[: int(top_k)],
            "top_kernels": sorted(by_kernel.items(), key=lambda x: x[1], reverse=True)[: int(top_k)],
        }

    def _skill_per_metric_stats(self, *, metric_like: str = "%", kernel_like: str = "%") -> List[Dict[str, object]]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        grouped: Dict[str, List[NcuReportMetricRecord]] = {}
        for rec in rows:
            grouped.setdefault(rec.metric_name, []).append(rec)
        out: List[Dict[str, object]] = []
        for metric_name, recs in grouped.items():
            values = [float(r.numeric_value) for r in recs if r.numeric_value is not None]
            values_sorted = sorted(values)
            out.append(
                {
                    "metric_name": metric_name,
                    "samples": len(recs),
                    "numeric_samples": len(values),
                    "non_numeric_samples": len(recs) - len(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "avg": (sum(values) / len(values)) if values else None,
                    "p50": _percentile(values_sorted, 50) if values else None,
                    "p90": _percentile(values_sorted, 90) if values else None,
                    "p99": _percentile(values_sorted, 99) if values else None,
                }
            )
        out.sort(key=lambda x: int(x.get("samples", 0)), reverse=True)
        return out

    def _skill_top_kernels(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        top_k: int = 20,
        score: str = "sum",
    ) -> List[Dict[str, object]]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        grouped: Dict[str, List[float]] = {}
        for rec in rows:
            if rec.numeric_value is None:
                continue
            grouped.setdefault(rec.kernel_name, []).append(float(rec.numeric_value))
        score_name = str(score or "sum").lower()
        out: List[Dict[str, object]] = []
        for kernel_name, vals in grouped.items():
            total = sum(vals)
            avg = total / len(vals)
            mx = max(vals)
            mn = min(vals)
            if score_name == "avg":
                score_val = avg
            elif score_name == "max":
                score_val = mx
            elif score_name == "min":
                score_val = mn
            else:
                score_val = total
            out.append(
                {
                    "kernel_name": kernel_name,
                    "samples": len(vals),
                    "sum": total,
                    "avg": avg,
                    "max": mx,
                    "min": mn,
                    "score": score_val,
                    "score_mode": score_name,
                }
            )
        out.sort(key=lambda x: float(x["score"]), reverse=True)
        return out[: int(top_k)]

    def _skill_all_metrics(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        limit: int = 20000,
    ) -> List[Dict[str, object]]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        out: List[Dict[str, object]] = []
        for rec in rows[: int(limit)]:
            out.append(
                {
                    "range_index": rec.range_index,
                    "action_index": rec.action_index,
                    "kernel_name": rec.kernel_name,
                    "metric_name": rec.metric_name,
                    "raw_value": rec.raw_value,
                    "numeric_value": rec.numeric_value,
                    "unit": rec.unit,
                }
            )
        return out

    def _build_skills(self) -> Dict[str, ReportSkill]:
        return {
            "summary": ReportSkill(
                name="summary",
                title="Report Summary",
                description="Summarize parsed metric records and value stats.",
                category="overview",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern (%/_/*)", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern (%/_/*)", "str", False, "%"),
                    SkillParam("top_k", "top rows limit", "int", False, 20),
                ],
                run_fn=self._skill_summary,
            ),
            "per_metric_stats": ReportSkill(
                name="per_metric_stats",
                title="Per Metric Stats",
                description="Compute count/min/max/avg/p50/p90/p99 for each metric.",
                category="metrics",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                ],
                run_fn=self._skill_per_metric_stats,
            ),
            "top_kernels": ReportSkill(
                name="top_kernels",
                title="Top Kernels",
                description="Aggregate selected numeric metric values by kernel.",
                category="kernels",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                    SkillParam("top_k", "top rows limit", "int", False, 20),
                    SkillParam("score", "score mode: sum|avg|max|min", "str", False, "sum"),
                ],
                run_fn=self._skill_top_kernels,
            ),
            "all_metrics": ReportSkill(
                name="all_metrics",
                title="All Metrics Rows",
                description="Dump all parsed metric rows (with optional limit).",
                category="inspect",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                    SkillParam("limit", "max output rows", "int", False, 20000),
                ],
                run_fn=self._skill_all_metrics,
            ),
        }


def load_ncu_report_records(
    report_path: str,
    *,
    metric_like: str = "%",
    kernel_like: str = "%",
    limit_actions: int = -1,
    ncu_report_module: Any = None,
) -> List[NcuReportMetricRecord]:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")

    mod = _load_ncu_report_module(ncu_report_module)
    ctx = mod.load_report(str(path))

    out: List[NcuReportMetricRecord] = []
    action_counter = 0
    ranges = _iter_ranges(ctx)
    for range_idx, range_obj in enumerate(ranges):
        actions = _iter_actions(range_obj)
        for action_idx, action in enumerate(actions):
            action_counter += 1
            if int(limit_actions) > 0 and action_counter > int(limit_actions):
                return out
            kernel_name = str(_maybe_call(action, "name", "") or "")
            if not _like_match(kernel_name, kernel_like):
                continue
            try:
                metric_names = list(action.metric_names())  # type: ignore[attr-defined]
            except Exception:
                metric_names = []
            for metric_name in metric_names:
                name = str(metric_name)
                if not _like_match(name, metric_like):
                    continue
                try:
                    metric = action.metric_by_name(name)  # type: ignore[attr-defined]
                except Exception:
                    continue
                raw_value = _metric_value(metric)
                unit = str(_maybe_call(metric, "unit", "") or "")
                out.append(
                    NcuReportMetricRecord(
                        kernel_name=kernel_name,
                        metric_name=name,
                        raw_value=raw_value,
                        numeric_value=_to_number(raw_value),
                        unit=unit,
                        range_index=range_idx,
                        action_index=action_idx,
                    )
                )
    return out


def analyze_ncu_report(
    report_path: str,
    *,
    top_k: int = 20,
    metric_like: str = "",
    kernel_like: str = "%",
    include_all_metrics: bool = True,
    all_metrics_limit: int = 20000,
    ncu_report_module: Any = None,
) -> Dict[str, object]:
    engine = NcuReportSkillEngine(report_path, ncu_report_module=ncu_report_module)
    metric_pattern = str(metric_like or "").strip() or "%"
    summary = engine.run_skill("summary", metric_like=metric_pattern, kernel_like=kernel_like, top_k=top_k)
    per_metric_stats = engine.run_skill("per_metric_stats", metric_like=metric_pattern, kernel_like=kernel_like)

    selected_metric = metric_pattern
    if selected_metric in {"%", "*"} and isinstance(per_metric_stats, list) and per_metric_stats:
        preferred_keywords = ("duration", "elapsed", "time", "cycles", "throughput")
        for row in per_metric_stats:
            name = str(row.get("metric_name", "")).lower() if isinstance(row, dict) else ""
            if any(k in name for k in preferred_keywords):
                selected_metric = str(row.get("metric_name"))
                break
        if selected_metric in {"%", "*"}:
            first = per_metric_stats[0]
            if isinstance(first, dict):
                selected_metric = str(first.get("metric_name", selected_metric))

    top_kernels = engine.run_skill(
        "top_kernels",
        metric_like=selected_metric,
        kernel_like=kernel_like,
        top_k=top_k,
        score="sum",
    )
    payload: Dict[str, object] = {
        "summary": summary,
        "selected_metric_like": selected_metric,
        "per_metric_stats": per_metric_stats,
        "top_kernels": top_kernels,
        "available_skills": engine.list_skills(),
    }
    if bool(include_all_metrics):
        payload["all_metrics"] = engine.run_skill(
            "all_metrics",
            metric_like=metric_pattern,
            kernel_like=kernel_like,
            limit=int(all_metrics_limit),
        )
    return payload


def analyze_ncu_report_to_markdown(payload: Dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    selected_metric_like = str(payload.get("selected_metric_like", "")) if isinstance(payload, dict) else ""
    top_kernels = payload.get("top_kernels", []) if isinstance(payload, dict) else []
    metric_stats = payload.get("per_metric_stats", []) if isinstance(payload, dict) else []
    lines: List[str] = []
    lines.append("# NCU Report Analyze")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if isinstance(summary, dict):
        lines.append(f"- metric_records: {summary.get('metric_records')}")
        lines.append(f"- filtered_records: {summary.get('filtered_records')}")
        lines.append(f"- unique_metrics: {summary.get('unique_metrics')}")
        lines.append(f"- unique_kernels: {summary.get('unique_kernels')}")
        lines.append(f"- numeric_values: {summary.get('numeric_values')}")
        lines.append(f"- non_numeric_values: {summary.get('non_numeric_values')}")
    lines.append("")
    lines.append("## Top Kernels")
    lines.append("")
    lines.append(f"- metric_like: `{selected_metric_like}`")
    lines.append("")
    lines.append("| kernel | samples | score | avg | max |")
    lines.append("|---|---:|---:|---:|---:|")
    if isinstance(top_kernels, list):
        for row in top_kernels:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {kernel} | {samples} | {score} | {avg} | {mx} |".format(
                    kernel=str(row.get("kernel_name", "")),
                    samples=row.get("samples", 0),
                    score=round(float(row.get("score", 0.0)), 6),
                    avg=round(float(row.get("avg", 0.0)), 6),
                    mx=round(float(row.get("max", 0.0)), 6),
                )
            )
    lines.append("")
    lines.append("## Per Metric Stats")
    lines.append("")
    lines.append("| metric | samples | numeric | p50 | p90 | p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    if isinstance(metric_stats, list):
        for row in metric_stats[:100]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {metric} | {samples} | {num} | {p50} | {p90} | {p99} |".format(
                    metric=str(row.get("metric_name", "")),
                    samples=row.get("samples", 0),
                    num=row.get("numeric_samples", 0),
                    p50=row.get("p50", ""),
                    p90=row.get("p90", ""),
                    p99=row.get("p99", ""),
                )
            )
    lines.append("")
    return "\n".join(lines)


def report_result_to_json(payload: object, *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
