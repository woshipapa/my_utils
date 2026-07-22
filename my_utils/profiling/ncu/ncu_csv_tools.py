from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


def _normalize_key(text: str) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


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


def _like_match(text: str, pattern: str) -> bool:
    if not pattern:
        return True
    p = str(pattern).strip()
    if p in {"%", "*"}:
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
    regex = "".join(parts)
    return re.match(regex, str(text or ""), flags=re.IGNORECASE) is not None


def _detect_key(candidates: Sequence[str], keys: Sequence[str]) -> str:
    key_set = set(keys)
    for item in candidates:
        if item in key_set:
            return item
    return ""


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


@dataclass
class NcuCsvMetricRecord:
    kernel_name: str
    metric_name: str
    value: float
    unit: str = ""
    row_index: int = -1
    source_format: str = "unknown"


@dataclass
class SkillParam:
    name: str
    description: str
    type: str = "str"
    required: bool = False
    default: object = None


@dataclass
class CsvSkill:
    name: str
    title: str
    description: str
    category: str
    params: List[SkillParam] = field(default_factory=list)
    run_fn: Optional[Callable[..., object]] = None


class NcuCsvSkillEngine:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = str(csv_path)
        self._rows = self._load_rows(self.csv_path)
        self._records = self._normalize_records(self._rows)
        self._skills = self._build_skills()

    @staticmethod
    def _load_rows(csv_path: str) -> List[Dict[str, str]]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header row: {csv_path}")
            rows = []
            for row in reader:
                rows.append(
                    {str(k or "").strip(): str(v or "").strip() for k, v in row.items()}
                )
        return rows

    @staticmethod
    def _normalize_records(rows: Sequence[Dict[str, str]]) -> List[NcuCsvMetricRecord]:
        if not rows:
            return []
        norm_rows: List[Dict[str, str]] = []
        for row in rows:
            norm = {
                _normalize_key(k): v for k, v in row.items() if str(k or "").strip()
            }
            norm_rows.append(norm)

        keys = sorted({k for row in norm_rows for k in row.keys()})
        kernel_key = _detect_key(
            ["kernel_name", "kernel", "kernelname", "name"],
            keys,
        )
        metric_name_key = _detect_key(["metric_name", "metric"], keys)
        metric_value_key = _detect_key(["metric_value", "value"], keys)
        metric_unit_key = _detect_key(["metric_unit", "unit"], keys)

        records: List[NcuCsvMetricRecord] = []
        if metric_name_key and metric_value_key:
            for idx, row in enumerate(norm_rows):
                kernel_name = str(row.get(kernel_key, "")).strip() if kernel_key else ""
                metric_name = str(row.get(metric_name_key, "")).strip()
                value = _to_number(row.get(metric_value_key, ""))
                if not metric_name or value is None:
                    continue
                records.append(
                    NcuCsvMetricRecord(
                        kernel_name=kernel_name,
                        metric_name=metric_name,
                        value=float(value),
                        unit=str(row.get(metric_unit_key, "")).strip()
                        if metric_unit_key
                        else "",
                        row_index=idx,
                        source_format="long",
                    )
                )
            return records

        meta_hints = {
            "kernel",
            "name",
            "context",
            "stream",
            "process",
            "device",
            "grid",
            "block",
            "launch",
            "range",
            "invocation",
            "id",
            "section",
            "rule",
            "source",
            "file",
            "function",
            "module",
            "line",
        }
        for idx, row in enumerate(norm_rows):
            kernel_name = str(row.get(kernel_key, "")).strip() if kernel_key else ""
            for key, raw_value in row.items():
                if key == kernel_key:
                    continue
                if any(token in key for token in meta_hints):
                    continue
                number = _to_number(raw_value)
                if number is None:
                    continue
                records.append(
                    NcuCsvMetricRecord(
                        kernel_name=kernel_name,
                        metric_name=key,
                        value=float(number),
                        unit="",
                        row_index=idx,
                        source_format="wide",
                    )
                )
        return records

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
            "csv_path": self.csv_path,
            "raw_rows": len(self._rows),
            "metric_records": len(self._records),
            "skills": out,
        }

    def run_skill(self, name: str, **kwargs: Any) -> object:
        skill = self._skills.get(str(name))
        if not skill:
            raise ValueError(f"unknown ncu csv skill: {name}")
        params = self._resolve_params(skill, kwargs)
        if skill.run_fn is None:
            raise RuntimeError(f"skill '{name}' has no run function")
        return skill.run_fn(**params)

    def _resolve_params(
        self, skill: CsvSkill, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in skill.params:
            if p.name in kwargs:
                value = kwargs[p.name]
            elif p.default is not None:
                value = p.default
            elif p.required:
                raise ValueError(
                    f"skill '{skill.name}' missing required param '{p.name}'"
                )
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

    def _filtered_records(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
    ) -> List[NcuCsvMetricRecord]:
        out: List[NcuCsvMetricRecord] = []
        for rec in self._records:
            if not _like_match(rec.metric_name, metric_like):
                continue
            if not _like_match(rec.kernel_name, kernel_like):
                continue
            out.append(rec)
        return out

    def _skill_summary(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        top_k: int = 20,
    ) -> Dict[str, object]:
        records = self._filtered_records(
            metric_like=metric_like, kernel_like=kernel_like
        )
        values = [r.value for r in records]
        values_sorted = sorted(values)
        by_metric: Dict[str, int] = {}
        by_kernel: Dict[str, int] = {}
        for r in records:
            by_metric[r.metric_name] = by_metric.get(r.metric_name, 0) + 1
            by_kernel[r.kernel_name] = by_kernel.get(r.kernel_name, 0) + 1
        return {
            "csv_path": self.csv_path,
            "raw_rows": len(self._rows),
            "metric_records": len(self._records),
            "filtered_records": len(records),
            "filters": {"metric_like": metric_like, "kernel_like": kernel_like},
            "unique_metrics": len(by_metric),
            "unique_kernels": len(by_kernel),
            "value_stats": {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": (sum(values) / len(values)) if values else None,
                "p50": _percentile(values_sorted, 50) if values else None,
                "p90": _percentile(values_sorted, 90) if values else None,
                "p99": _percentile(values_sorted, 99) if values else None,
            },
            "top_metrics": sorted(by_metric.items(), key=lambda x: x[1], reverse=True)[
                : int(top_k)
            ],
            "top_kernels": sorted(by_kernel.items(), key=lambda x: x[1], reverse=True)[
                : int(top_k)
            ],
        }

    def _skill_top_kernels(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        top_k: int = 20,
        score: str = "sum",
    ) -> List[Dict[str, object]]:
        records = self._filtered_records(
            metric_like=metric_like, kernel_like=kernel_like
        )
        grouped: Dict[str, List[float]] = {}
        for r in records:
            grouped.setdefault(r.kernel_name, []).append(float(r.value))
        rows: List[Dict[str, object]] = []
        score_name = str(score or "sum").lower()
        for kernel, vals in grouped.items():
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
            rows.append(
                {
                    "kernel_name": kernel,
                    "samples": len(vals),
                    "sum": total,
                    "avg": avg,
                    "max": mx,
                    "min": mn,
                    "score": score_val,
                    "score_mode": score_name,
                }
            )
        rows.sort(key=lambda x: float(x["score"]), reverse=True)
        return rows[: int(top_k)]

    def _skill_top_metrics(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        top_k: int = 20,
    ) -> List[Dict[str, object]]:
        records = self._filtered_records(
            metric_like=metric_like, kernel_like=kernel_like
        )
        grouped: Dict[str, List[float]] = {}
        for r in records:
            grouped.setdefault(r.metric_name, []).append(float(r.value))
        rows: List[Dict[str, object]] = []
        for metric, vals in grouped.items():
            total = sum(vals)
            rows.append(
                {
                    "metric_name": metric,
                    "samples": len(vals),
                    "sum": total,
                    "avg": total / len(vals),
                    "max": max(vals),
                    "min": min(vals),
                }
            )
        rows.sort(key=lambda x: float(x["sum"]), reverse=True)
        return rows[: int(top_k)]

    def _skill_metric_percentiles(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
    ) -> List[Dict[str, object]]:
        records = self._filtered_records(
            metric_like=metric_like, kernel_like=kernel_like
        )
        grouped: Dict[str, List[float]] = {}
        for r in records:
            grouped.setdefault(r.metric_name, []).append(float(r.value))
        rows: List[Dict[str, object]] = []
        for metric, vals in grouped.items():
            arr = sorted(vals)
            rows.append(
                {
                    "metric_name": metric,
                    "samples": len(arr),
                    "p50": _percentile(arr, 50),
                    "p90": _percentile(arr, 90),
                    "p99": _percentile(arr, 99),
                    "max": arr[-1],
                    "min": arr[0],
                }
            )
        rows.sort(key=lambda x: float(x["p99"]), reverse=True)
        return rows

    def _skill_schema_inspect(self, *, limit: int = 50) -> Dict[str, object]:
        headers = sorted({key for row in self._rows for key in row.keys()})
        preview = self._rows[: int(limit)]
        return {
            "csv_path": self.csv_path,
            "raw_rows": len(self._rows),
            "headers": headers,
            "normalized_headers": [_normalize_key(h) for h in headers],
            "preview_rows": preview,
        }

    def _build_skills(self) -> Dict[str, CsvSkill]:
        return {
            "summary": CsvSkill(
                name="summary",
                title="CSV Summary",
                description="Summarize row counts, unique metrics/kernels and value stats.",
                category="overview",
                params=[
                    SkillParam(
                        "metric_like",
                        "metric name LIKE pattern (%/_/*)",
                        "str",
                        False,
                        "%",
                    ),
                    SkillParam(
                        "kernel_like",
                        "kernel name LIKE pattern (%/_/*)",
                        "str",
                        False,
                        "%",
                    ),
                    SkillParam("top_k", "top rows limit", "int", False, 20),
                ],
                run_fn=self._skill_summary,
            ),
            "top_kernels": CsvSkill(
                name="top_kernels",
                title="Top Kernels by Metric Value",
                description="Aggregate selected metric values by kernel and rank by score mode.",
                category="kernels",
                params=[
                    SkillParam(
                        "metric_like", "metric name LIKE pattern", "str", False, "%"
                    ),
                    SkillParam(
                        "kernel_like", "kernel name LIKE pattern", "str", False, "%"
                    ),
                    SkillParam("top_k", "top rows limit", "int", False, 20),
                    SkillParam(
                        "score", "score mode: sum|avg|max|min", "str", False, "sum"
                    ),
                ],
                run_fn=self._skill_top_kernels,
            ),
            "top_metrics": CsvSkill(
                name="top_metrics",
                title="Top Metrics",
                description="Aggregate values by metric name.",
                category="metrics",
                params=[
                    SkillParam(
                        "metric_like", "metric name LIKE pattern", "str", False, "%"
                    ),
                    SkillParam(
                        "kernel_like", "kernel name LIKE pattern", "str", False, "%"
                    ),
                    SkillParam("top_k", "top rows limit", "int", False, 20),
                ],
                run_fn=self._skill_top_metrics,
            ),
            "metric_percentiles": CsvSkill(
                name="metric_percentiles",
                title="Metric Percentiles",
                description="Compute p50/p90/p99 by metric.",
                category="metrics",
                params=[
                    SkillParam(
                        "metric_like", "metric name LIKE pattern", "str", False, "%"
                    ),
                    SkillParam(
                        "kernel_like", "kernel name LIKE pattern", "str", False, "%"
                    ),
                ],
                run_fn=self._skill_metric_percentiles,
            ),
            "schema_inspect": CsvSkill(
                name="schema_inspect",
                title="CSV Schema Inspect",
                description="Inspect headers and preview rows.",
                category="inspect",
                params=[SkillParam("limit", "preview row count", "int", False, 50)],
                run_fn=self._skill_schema_inspect,
            ),
        }


def analyze_ncu_csv(
    csv_path: str,
    *,
    top_k: int = 20,
    metric_like: str = "",
    kernel_like: str = "%",
) -> Dict[str, object]:
    engine = NcuCsvSkillEngine(csv_path)
    summary = engine.run_skill(
        "summary",
        metric_like=metric_like or "%",
        kernel_like=kernel_like,
        top_k=top_k,
    )
    all_metrics = (
        [row[0] for row in summary.get("top_metrics", [])]
        if isinstance(summary, dict)
        else []
    )

    selected_metric = str(metric_like or "").strip()
    if not selected_metric:
        metric_keywords = ["duration", "time", "elapsed", "cycles", "throughput"]
        for item in all_metrics:
            low = str(item).lower()
            if any(token in low for token in metric_keywords):
                selected_metric = item
                break
    if not selected_metric:
        selected_metric = "%"

    top_kernels = engine.run_skill(
        "top_kernels",
        metric_like=selected_metric,
        kernel_like=kernel_like,
        top_k=top_k,
        score="sum",
    )
    metric_percentiles = engine.run_skill(
        "metric_percentiles",
        metric_like=selected_metric,
        kernel_like=kernel_like,
    )
    return {
        "summary": summary,
        "selected_metric_like": selected_metric,
        "top_kernels": top_kernels,
        "metric_percentiles": metric_percentiles,
        "available_skills": engine.list_skills(),
    }


def analyze_ncu_to_markdown(payload: Dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    selected_metric_like = (
        str(payload.get("selected_metric_like", ""))
        if isinstance(payload, dict)
        else ""
    )
    top_kernels = payload.get("top_kernels", []) if isinstance(payload, dict) else []
    lines: List[str] = []
    lines.append("# NCU CSV Analyze Report")
    lines.append("")
    if isinstance(summary, dict):
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- raw_rows: {summary.get('raw_rows')}")
        lines.append(f"- metric_records: {summary.get('metric_records')}")
        lines.append(f"- filtered_records: {summary.get('filtered_records')}")
        lines.append(f"- unique_metrics: {summary.get('unique_metrics')}")
        lines.append(f"- unique_kernels: {summary.get('unique_kernels')}")
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
    return "\n".join(lines)


def skill_result_to_json(payload: object, *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
