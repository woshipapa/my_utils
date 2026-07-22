# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from .nsys_sqlite_provider import NsysSqliteMetricsProvider


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def collect_kernel_rows(
    provider: NsysSqliteMetricsProvider,
    *,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 500000,
    attach_iteration: bool = False,
    iteration_marker: str = "sample_0",
) -> List[Dict[str, object]]:
    rows = provider.run_sql_skill(
        "kernel_map",
        device_id=int(device_id),
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        limit=int(limit),
    )

    enriched: List[Dict[str, object]] = []
    for row in rows:
        s = _to_int(row.get("start_ns"), 0)
        e = _to_int(row.get("end_ns"), 0)
        if e <= s:
            continue
        name = str(row.get("kernel_name") or "")
        enriched.append(
            {
                "device_id": _to_int(row.get("device_id"), int(device_id)),
                "stream_id": _to_int(row.get("stream_id"), 0),
                "correlation_id": _to_int(row.get("correlation_id"), 0),
                "kernel_name": name,
                "start_ns": s,
                "end_ns": e,
                "duration_us": round((e - s) / 1000.0, 6),
                "duration_ms": round((e - s) / 1e6, 6),
                "is_nccl": bool("nccl" in name.lower()),
            }
        )

    if attach_iteration and enriched:
        iters = provider.detect_iterations(
            marker=iteration_marker,
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
            top_level_only=True,
            limit=10000,
        )
        iter_windows = [
            (
                _to_int(item.get("start_ns"), -1),
                _to_int(item.get("end_ns"), -1),
                _to_int(item.get("iteration"), -1),
            )
            for item in iters
        ]
        for row in enriched:
            it_idx = -1
            s = _to_int(row["start_ns"])
            e = _to_int(row["end_ns"])
            for ws, we, wi in iter_windows:
                if ws < 0 or we < 0:
                    continue
                if s >= ws and e <= we:
                    it_idx = wi
                    break
            row["iteration"] = int(it_idx)

    return enriched


def write_rows_json(rows: List[Dict[str, object]], output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def write_rows_csv(rows: List[Dict[str, object]], output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return str(out)
    fieldnames = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(out)


def export_kernels_flat(
    sqlite_path: str,
    *,
    output_path: str,
    fmt: str = "json",
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 500000,
    attach_iteration: bool = False,
    iteration_marker: str = "sample_0",
) -> str:
    provider = NsysSqliteMetricsProvider(sqlite_path)
    rows = collect_kernel_rows(
        provider,
        device_id=device_id,
        start_ns=start_ns,
        end_ns=end_ns,
        limit=limit,
        attach_iteration=attach_iteration,
        iteration_marker=iteration_marker,
    )
    text = str(fmt or "json").strip().lower()
    if text in {"csv"}:
        return write_rows_csv(rows, output_path)
    return write_rows_json(rows, output_path)
