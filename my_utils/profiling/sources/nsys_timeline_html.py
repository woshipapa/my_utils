from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List

from .nsys_flat_export import collect_kernel_rows
from .nsys_sqlite_provider import NsysSqliteMetricsProvider


def _color_for_name(name: str) -> str:
    seed = sum(ord(ch) for ch in (name or ""))
    r = 70 + (seed * 37) % 130
    g = 70 + (seed * 53) % 130
    b = 70 + (seed * 71) % 130
    return f"rgb({r},{g},{b})"


def export_timeline_html(
    sqlite_path: str,
    *,
    output_path: str,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 100000,
    width_px: int = 1800,
) -> str:
    provider = NsysSqliteMetricsProvider(sqlite_path)
    rows = collect_kernel_rows(
        provider,
        device_id=device_id,
        start_ns=start_ns,
        end_ns=end_ns,
        limit=limit,
        attach_iteration=False,
    )
    if not rows:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("<html><body><h2>No kernels</h2></body></html>", encoding="utf-8")
        return str(out)

    min_start = min(int(item["start_ns"]) for item in rows)
    max_end = max(int(item["end_ns"]) for item in rows)
    span = max(1, max_end - min_start)

    by_stream: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        sid = int(row.get("stream_id") or 0)
        by_stream.setdefault(sid, []).append(row)
    stream_ids = sorted(by_stream.keys())

    header = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<title>NSYS Timeline</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#0f1116;color:#e7e9ee;}",
        ".row{display:flex;align-items:center;margin:8px 0;}",
        ".label{width:120px;color:#a8afbf;font-size:12px;}",
        f".track{{position:relative;height:24px;width:{int(width_px)}px;background:#1a1f2b;border-radius:4px;overflow:hidden;}}",
        ".bar{position:absolute;height:18px;top:3px;border-radius:3px;}",
        ".bar:hover{outline:1px solid #fff;}",
        ".meta{margin-bottom:12px;color:#a8afbf;}",
        "</style></head><body>",
        "<h2>NSYS Kernel Timeline (Static HTML)</h2>",
        f"<div class='meta'>sqlite={html.escape(str(sqlite_path))} | device_id={device_id} | kernels={len(rows)}</div>",
    ]

    body: List[str] = []
    for sid in stream_ids:
        body.append("<div class='row'>")
        body.append(f"<div class='label'>stream {sid}</div>")
        body.append("<div class='track'>")
        for row in by_stream[sid]:
            s = int(row["start_ns"])
            e = int(row["end_ns"])
            left = (s - min_start) / span
            width = max((e - s) / span, 1.0 / float(width_px))
            left_px = int(left * width_px)
            width_bar = max(1, int(width * width_px))
            name = str(row.get("kernel_name") or "")
            color = _color_for_name(name)
            title = (
                f"{name} | dur_ms={row.get('duration_ms', 0)} | "
                f"start_ns={row.get('start_ns', 0)} | end_ns={row.get('end_ns', 0)}"
            )
            body.append(
                f"<div class='bar' style='left:{left_px}px;width:{width_bar}px;background:{color};' title='{html.escape(title)}'></div>"
            )
        body.append("</div></div>")

    footer = ["</body></html>"]
    text = "\n".join(header + body + footer)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return str(out)

