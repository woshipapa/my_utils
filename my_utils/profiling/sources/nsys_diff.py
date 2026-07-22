from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Tuple

from .nsys_sql_skills import NsysSqlSkillEngine


def _index_by_name(
    rows: List[Dict[str, object]], key: str
) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        name = str(row.get(key) or "")
        if not name:
            continue
        out[name] = row
    return out


def _diff_named_rows(
    before_rows: List[Dict[str, object]],
    after_rows: List[Dict[str, object]],
    *,
    name_key: str,
    value_key: str,
    top_k: int = 20,
) -> List[Dict[str, object]]:
    before = _index_by_name(before_rows, name_key)
    after = _index_by_name(after_rows, name_key)
    names = sorted(set(before.keys()) | set(after.keys()))
    diffs: List[Dict[str, object]] = []
    for name in names:
        b = float(before.get(name, {}).get(value_key, 0.0) or 0.0)
        a = float(after.get(name, {}).get(value_key, 0.0) or 0.0)
        delta = a - b
        ratio = (a / b) if b > 0 else None
        diffs.append(
            {
                "name": name,
                "before": round(b, 6),
                "after": round(a, 6),
                "delta": round(delta, 6),
                "ratio": round(ratio, 6) if ratio is not None else None,
            }
        )
    diffs.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return diffs[: max(1, int(top_k))]


def _analyze_side(
    sqlite_path: str,
    *,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    top_k: int = 20,
) -> Tuple[
    Dict[str, object],
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """Open one SQLite file, run all needed analyses, return (summary, overlap, kernels, nvtx)."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        engine = NsysSqlSkillEngine(conn)
        summary = engine.summarize_gpu_kernels(
            device_id=device_id, start_ns=start_ns, end_ns=end_ns, top_k=top_k
        )
        overlap = engine.analyze_compute_comm_overlap(
            device_id=device_id, start_ns=start_ns, end_ns=end_ns
        )
        kernel_rows = engine.execute(
            "aggregate_kernels", device_id=device_id, limit=5000
        )
        try:
            nvtx_rows = engine.execute("aggregate_nvtx_ranges", limit=5000)
        except Exception:
            nvtx_rows = []
        return summary, overlap, kernel_rows, nvtx_rows
    finally:
        conn.close()


def diff_nsys_sqlite(
    before_sqlite: str,
    after_sqlite: str,
    *,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    top_k: int = 20,
) -> Dict[str, object]:
    before_summary, before_overlap, before_k, before_nvtx = _analyze_side(
        before_sqlite,
        device_id=device_id,
        start_ns=start_ns,
        end_ns=end_ns,
        top_k=top_k,
    )
    after_summary, after_overlap, after_k, after_nvtx = _analyze_side(
        after_sqlite, device_id=device_id, start_ns=start_ns, end_ns=end_ns, top_k=top_k
    )

    kernel_diff = _diff_named_rows(
        before_k,
        after_k,
        name_key="kernel_name",
        value_key="total_ms",
        top_k=top_k,
    )
    nvtx_diff = _diff_named_rows(
        before_nvtx,
        after_nvtx,
        name_key="nvtx_name",
        value_key="total_ms",
        top_k=top_k,
    )

    before_util = float(
        ((before_summary.get("timing") or {}).get("utilization_pct") or 0.0)
    )
    after_util = float(
        ((after_summary.get("timing") or {}).get("utilization_pct") or 0.0)
    )
    before_overlap_ms = float((before_overlap.get("overlap_ms") or 0.0))
    after_overlap_ms = float((after_overlap.get("overlap_ms") or 0.0))

    return {
        "before_sqlite": before_sqlite,
        "after_sqlite": after_sqlite,
        "device_id": int(device_id),
        "window": {"start_ns": int(start_ns), "end_ns": int(end_ns)},
        "summary": {
            "utilization_pct": {
                "before": round(before_util, 6),
                "after": round(after_util, 6),
                "delta": round(after_util - before_util, 6),
            },
            "overlap_ms": {
                "before": round(before_overlap_ms, 6),
                "after": round(after_overlap_ms, 6),
                "delta": round(after_overlap_ms - before_overlap_ms, 6),
            },
        },
        "kernel_diff_top": kernel_diff,
        "nvtx_diff_top": nvtx_diff,
    }


def diff_to_markdown(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# NSYS Diff Report")
    lines.append("")
    lines.append(f"- before: `{payload.get('before_sqlite', '')}`")
    lines.append(f"- after: `{payload.get('after_sqlite', '')}`")
    lines.append(f"- device_id: `{payload.get('device_id', -1)}`")
    lines.append("")
    lines.append("## Summary Deltas")
    lines.append("")
    summary = payload.get("summary") or {}
    util = summary.get("utilization_pct") or {}
    overlap = summary.get("overlap_ms") or {}
    lines.append(f"- utilization_pct delta: `{util.get('delta', 0)}`")
    lines.append(f"- overlap_ms delta: `{overlap.get('delta', 0)}`")
    lines.append("")

    lines.append("## Kernel Diff Top")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(payload.get("kernel_diff_top", []), ensure_ascii=False, indent=2)
    )
    lines.append("```")
    lines.append("")

    lines.append("## NVTX Diff Top")
    lines.append("")
    lines.append("```json")
    lines.append(
        json.dumps(payload.get("nvtx_diff_top", []), ensure_ascii=False, indent=2)
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
