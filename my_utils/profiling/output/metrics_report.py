from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Optional

from .metrics_diff import ReportDiff
from ..metrics.metrics_types import AnalysisReport


class MetricsReportRenderer:
    def to_json(self, report: AnalysisReport, *, indent: int = 2) -> str:
        return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)

    def to_markdown(self, report: AnalysisReport) -> str:
        lines = ["# Profiling Analysis Report", ""]
        lines.append(f"- Generated at: `{report.generated_at}`")
        lines.append(f"- Schema version: `{report.schema_version}`")
        if report.overall_score is not None:
            lines.append(f"- Overall score: `{report.overall_score}`")
        lines.append(f"- Total findings: `{len(report.findings)}`")
        lines.append("")

        if report.summary:
            lines.append("## Summary")
            for key, value in report.summary.items():
                lines.append(f"- `{key}`: `{value}`")
            lines.append("")

        lines.append("## Findings")
        if not report.findings:
            lines.append("- None")
        else:
            for item in report.findings:
                lines.append(f"- **{item.severity.upper()}** `{item.finding_type}`: {item.title}")
                lines.append(f"  - {item.description}")
                if item.data:
                    lines.append(f"  - data keys: {', '.join(item.data.keys())}")
        lines.append("")

        lines.append("## Recommendations")
        if not report.recommendations:
            lines.append("- None")
        else:
            for rec in report.recommendations:
                lines.append(f"- {rec}")
        lines.append("")
        return "\n".join(lines)

    def to_html(self, report: AnalysisReport) -> str:
        summary_items = "".join(
            f"<li><b>{escape(str(k))}</b>: {escape(str(v))}</li>" for k, v in report.summary.items()
        )
        findings_html = "".join(
            (
                "<div class='finding'>"
                f"<h3>{escape(item.title)} <span class='sev sev-{escape(item.severity)}'>{escape(item.severity.upper())}</span></h3>"
                f"<p><b>Type:</b> {escape(item.finding_type)}</p>"
                f"<p>{escape(item.description)}</p>"
                f"<pre>{escape(json.dumps(item.data, ensure_ascii=False, indent=2))}</pre>"
                "</div>"
            )
            for item in report.findings
        )
        rec_html = "".join(f"<li>{escape(rec)}</li>" for rec in report.recommendations)

        score_html = ""
        if report.overall_score is not None:
            score_html = f"<p><b>Overall Score:</b> {escape(str(report.overall_score))}</p>"

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Profiling Analysis Report</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #14233a;
      --muted: #5a6e8d;
      --warn: #b26a00;
      --high: #b42f2f;
      --info: #2a6ec4;
      --border: #d8e1f0;
    }}
    body {{ margin: 0; font-family: "Segoe UI", "Noto Sans", sans-serif; background: var(--bg); color: var(--text); }}
    main {{ max-width: 1080px; margin: 24px auto; padding: 0 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
    .finding {{ border-top: 1px solid var(--border); padding-top: 12px; margin-top: 12px; }}
    .sev {{ padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
    .sev-warning {{ background: #fff2d9; color: var(--warn); }}
    .sev-high {{ background: #ffe0df; color: var(--high); }}
    .sev-info {{ background: #dfeeff; color: var(--info); }}
    pre {{ white-space: pre-wrap; background: #f2f5fb; padding: 8px; border-radius: 8px; }}
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Profiling Analysis Report</h1>
      <p><b>Generated:</b> {escape(report.generated_at)}</p>
      <p><b>Schema:</b> {escape(report.schema_version)}</p>
      {score_html}
      <h2>Summary</h2>
      <ul>{summary_items}</ul>
    </section>
    <section class="card">
      <h2>Findings</h2>
      {findings_html or "<p>No findings.</p>"}
    </section>
    <section class="card">
      <h2>Recommendations</h2>
      <ul>{rec_html or "<li>No recommendations.</li>"}</ul>
    </section>
  </main>
</body>
</html>
"""

    def render(self, report: AnalysisReport, fmt: str) -> str:
        fmt_l = (fmt or "json").lower()
        if fmt_l == "json":
            return self.to_json(report)
        if fmt_l in ("md", "markdown"):
            return self.to_markdown(report)
        if fmt_l == "html":
            return self.to_html(report)
        raise ValueError(f"Unsupported report format: {fmt}")

    def write(self, report: AnalysisReport, output_path: str, *, fmt: Optional[str] = None) -> str:
        path = Path(output_path)
        target_fmt = fmt or path.suffix.lstrip(".") or "json"
        content = self.render(report, target_fmt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def diff_to_markdown(self, diff: ReportDiff) -> str:
        lines = ["# Profiling Diff Report", ""]
        lines.append(f"- Base: `{diff.base_path}`")
        lines.append(f"- Target: `{diff.target_path}`")
        if diff.score_delta is not None:
            lines.append(f"- Score delta: `{diff.score_delta:+.2f}`")
        lines.append("")
        lines.append("## Finding Delta")
        total = diff.finding_delta.get("total", {})
        if total:
            lines.append(
                f"- base={total.get('base', 0)} target={total.get('target', 0)} delta={total.get('delta', 0)}"
            )
        for key, item in (diff.finding_delta.get("by_type", {}) or {}).items():
            lines.append(f"- `{key}`: {item.get('base', 0)} -> {item.get('target', 0)} (delta {item.get('delta', 0)})")
        lines.append("")
        return "\n".join(lines)

