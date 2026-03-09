from __future__ import annotations

import json
import sqlite3
import statistics
from typing import Dict, List, Optional

from .nsys_mfu import compute_mfu_single, infer_peak_tflops
from .nsys_sqlite_provider import NsysSqliteMetricsProvider


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _first_gpu_name(sqlite_path: str) -> str:
    try:
        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        try:
            tables = {
                str(row["name"])
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
            }
            if "TARGET_INFO_GPU" not in tables:
                return ""
            row = conn.execute("SELECT name FROM TARGET_INFO_GPU ORDER BY id LIMIT 1;").fetchone()
            return str(row["name"] or "").strip() if row else ""
        finally:
            conn.close()
    except Exception:
        return ""


def analyze_nsys_sqlite(
    sqlite_path: str,
    *,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    top_k: int = 10,
    iteration_marker: str = "sample_0",
    model_flops_per_step: Optional[float] = None,
    peak_tflops: Optional[float] = None,
    peak_precision: str = "fp16",
    limit: int = 500000,
) -> Dict[str, object]:
    provider = NsysSqliteMetricsProvider(sqlite_path)
    schema = provider.describe_schema()
    summary = provider.summarize_gpu_kernels(
        device_id=device_id,
        start_ns=start_ns,
        end_ns=end_ns,
        top_k=top_k,
        limit=limit,
    )
    overlap = provider.analyze_compute_comm_overlap(
        device_id=device_id,
        start_ns=start_ns,
        end_ns=end_ns,
        limit=limit,
    )
    top_kernels = provider.run_sql_skill(
        "top_kernels",
        device_id=device_id,
        limit=max(int(top_k), 1),
    )
    nccl_breakdown = provider.run_sql_skill(
        "nccl_breakdown",
        device_id=device_id,
        limit=max(int(top_k), 1),
    )
    iterations = provider.detect_iterations(
        marker=iteration_marker,
        device_id=device_id,
        start_ns=start_ns,
        end_ns=end_ns,
        top_level_only=True,
        limit=10000,
    )

    warnings: List[str] = []
    mfu: Optional[Dict[str, object]] = None
    if model_flops_per_step is not None:
        step_time_s: Optional[float] = None
        if iterations:
            iter_ms = [_to_float(item.get("duration_ms")) for item in iterations]
            iter_ms = [x for x in iter_ms if x and x > 0]
            if iter_ms:
                step_time_s = float(statistics.median(iter_ms)) / 1000.0
        if step_time_s is None:
            span_ms = _to_float(((summary or {}).get("timing") or {}).get("span_ms"))
            if span_ms and span_ms > 0:
                step_time_s = float(span_ms) / 1000.0
        if step_time_s is None:
            warnings.append("MFU skipped: no usable step time from iterations or summary span.")
        else:
            resolved_peak = peak_tflops
            if resolved_peak is None:
                gpu_name = _first_gpu_name(sqlite_path)
                resolved_peak = infer_peak_tflops(gpu_name, precision=peak_precision)
                if resolved_peak is None:
                    warnings.append("MFU peak_tflops not provided and GPU name mapping failed.")
            if resolved_peak is not None:
                mfu = compute_mfu_single(
                    step_time_s=float(step_time_s),
                    model_flops_per_step=float(model_flops_per_step),
                    peak_tflops=float(resolved_peak),
                )

    return {
        "sqlite_path": sqlite_path,
        "device_id": int(device_id),
        "window": {"start_ns": int(start_ns), "end_ns": int(end_ns)},
        "schema": schema,
        "summary": summary,
        "overlap": overlap,
        "top_kernels": top_kernels,
        "nccl_breakdown": nccl_breakdown,
        "iterations": iterations,
        "mfu": mfu,
        "warnings": warnings,
    }


def analyze_to_markdown(result: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# NSYS Analyze Report")
    lines.append("")
    lines.append(f"- sqlite: `{result.get('sqlite_path', '')}`")
    lines.append(f"- device_id: `{result.get('device_id', -1)}`")
    lines.append("")

    summary = result.get("summary") or {}
    timing = summary.get("timing") or {}
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- span_ms: `{timing.get('span_ms', 0)}`")
    lines.append(f"- busy_ms: `{timing.get('busy_ms', 0)}`")
    lines.append(f"- idle_ms: `{timing.get('idle_ms', 0)}`")
    lines.append(f"- utilization_pct: `{timing.get('utilization_pct', 0)}`")
    lines.append("")

    lines.append("## Overlap")
    lines.append("")
    overlap = result.get("overlap") or {}
    lines.append(f"- overlap_ms: `{overlap.get('overlap_ms', 0)}`")
    lines.append(f"- comm_total_ms: `{overlap.get('comm_total_ms', 0)}`")
    lines.append(f"- compute_total_ms: `{overlap.get('compute_total_ms', 0)}`")
    lines.append("")

    lines.append("## Top Kernels")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.get("top_kernels", []), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## NCCL Breakdown")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.get("nccl_breakdown", []), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Iterations")
    lines.append("")
    lines.append(f"- count: `{len(result.get('iterations', []) or [])}`")
    lines.append("```json")
    lines.append(json.dumps(result.get("iterations", []), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    mfu = result.get("mfu")
    if mfu:
        lines.append("## MFU")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(mfu, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for item in warnings:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)

