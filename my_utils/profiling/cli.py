from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .analyzers.metrics_analyzer import MetricsAnalyzer
from .pipeline.metrics_collector import MetricsCollector
from .output.metrics_diff import compare_reports, write_diff
from .output.metrics_report import MetricsReportRenderer
from .metrics.metrics_store import MetricsStore
from .output.metrics_trace import ChromeTraceExportConfig, export_events_file_to_chrome_trace
from .metrics.metrics_types import AnalysisReport
from .metrics.provider_registry import DEFAULT_PROVIDER_REGISTRY


def _parse_tags(values: Iterable[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        tags[key] = value.strip()
    return tags


def _formats(value: str) -> List[str]:
    items = [x.strip().lower() for x in str(value or "").split(",")]
    return [x for x in items if x]


def _parse_rank_offsets(values: Iterable[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for item in values:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        try:
            result[str(key)] = float(value.strip())
        except Exception:
            continue
    return result


def cmd_ingest(args: argparse.Namespace) -> int:
    collector = MetricsCollector.from_config(args.config)
    tags = _parse_tags(args.tags)
    total_written = 0
    collector.start()
    try:
        step = args.step_start
        for _ in range(args.collect_times):
            total_written += collector.collect(step=step, tags=tags)
            step += args.step_stride
    finally:
        collector.stop()
    print(f"[ingest] providers={collector.list_providers()} written_events={total_written}")
    if args.analyze:
        report = collector.analyze()
        for fmt in _formats(args.report_formats):
            suffix = "md" if fmt in ("md", "markdown") else fmt
            output = str(Path(collector.output_dir) / f"analysis_report.{suffix}")
            collector.export_report(fmt=fmt, output_path=output, report=report)
            print(f"[ingest] wrote report: {output}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    events = MetricsStore.read_events_file(args.events)
    analyzer = MetricsAnalyzer(workload_profile=args.workload, enable_advanced_rules=not args.disable_advanced_rules)
    report = analyzer.analyze(events)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = MetricsReportRenderer()
    for fmt in _formats(args.report_formats):
        suffix = "md" if fmt in ("md", "markdown") else fmt
        path = output_dir / f"analysis_report.{suffix}"
        renderer.write(report, output_path=str(path), fmt=fmt)
        print(f"[analyze] wrote report: {path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.report_json).read_text(encoding="utf-8"))
    report = AnalysisReport.from_dict(payload)
    renderer = MetricsReportRenderer()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in _formats(args.report_formats):
        suffix = "md" if fmt in ("md", "markdown") else fmt
        path = output_dir / f"rendered_report.{suffix}"
        renderer.write(report, output_path=str(path), fmt=fmt)
        print(f"[report] wrote: {path}")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    diff = compare_reports(args.base_report, args.target_report)
    output = write_diff(diff, args.output)
    print(f"[diff] wrote: {output}")
    if args.markdown:
        md_path = Path(args.markdown)
        md = MetricsReportRenderer().diff_to_markdown(diff)
        md_path.write_text(md, encoding="utf-8")
        print(f"[diff] wrote markdown: {md_path}")
    return 0


def cmd_list_providers(args: argparse.Namespace) -> int:
    for provider_type in DEFAULT_PROVIDER_REGISTRY.list_types():
        print(provider_type)
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    cfg = ChromeTraceExportConfig(
        include_metric_prefixes=[item.strip() for item in (args.include_metric_prefixes or "").split(",") if item.strip()],
        include_non_duration_metrics=bool(args.include_non_duration_metrics),
        auto_align_ranks=bool(args.auto_align_ranks),
        reference_rank=str(args.reference_rank or ""),
        rank_clock_offsets_sec=_parse_rank_offsets(args.rank_clock_offset),
    )
    output = export_events_file_to_chrome_trace(
        args.events,
        output_path=args.output,
        config=cfg,
    )
    print(f"[trace] wrote chrome trace: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified profiling metrics CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="collect metrics from configured providers")
    ingest.add_argument("--config", required=True, help="collector config YAML/JSON")
    ingest.add_argument("--collect-times", type=int, default=1)
    ingest.add_argument("--step-start", type=int, default=0)
    ingest.add_argument("--step-stride", type=int, default=1)
    ingest.add_argument("--tag", dest="tags", action="append", default=[])
    ingest.add_argument("--analyze", action="store_true")
    ingest.add_argument("--report-formats", default="json,markdown,html")
    ingest.set_defaults(func=cmd_ingest)

    analyze = sub.add_parser("analyze", help="analyze events JSONL and generate reports")
    analyze.add_argument("--events", required=True, help="metrics events jsonl path")
    analyze.add_argument("--output-dir", default="./metrics_cli_output")
    analyze.add_argument("--workload", default="default")
    analyze.add_argument("--disable-advanced-rules", action="store_true")
    analyze.add_argument("--report-formats", default="json,markdown,html")
    analyze.set_defaults(func=cmd_analyze)

    report = sub.add_parser("report", help="render report json to markdown/html/json")
    report.add_argument("--report-json", required=True)
    report.add_argument("--output-dir", default="./metrics_cli_output")
    report.add_argument("--report-formats", default="markdown,html")
    report.set_defaults(func=cmd_report)

    diff = sub.add_parser("diff", help="diff two report json files")
    diff.add_argument("--base-report", required=True)
    diff.add_argument("--target-report", required=True)
    diff.add_argument("--output", required=True, help="diff json output path")
    diff.add_argument("--markdown", default="", help="optional diff markdown output path")
    diff.set_defaults(func=cmd_diff)

    list_providers = sub.add_parser("list-providers", help="list available provider types")
    list_providers.set_defaults(func=cmd_list_providers)

    trace = sub.add_parser("trace", help="convert metrics events jsonl to Chrome Trace JSON")
    trace.add_argument("--events", required=True, help="metrics events jsonl path")
    trace.add_argument("--output", required=True, help="chrome trace json output path")
    trace.add_argument(
        "--include-metric-prefixes",
        default="latency",
        help="comma-separated metric prefixes to include (default: latency)",
    )
    trace.add_argument(
        "--include-non-duration-metrics",
        action="store_true",
        help="also include non-duration metrics as tiny span events",
    )
    trace.add_argument(
        "--auto-align-ranks",
        action="store_true",
        help="estimate clock offsets between ranks using shared steps",
    )
    trace.add_argument(
        "--reference-rank",
        default="",
        help="reference rank for auto alignment (default: first rank in events)",
    )
    trace.add_argument(
        "--rank-clock-offset",
        action="append",
        default=[],
        help="manual rank clock offset in seconds, format rank=offset_sec (can repeat)",
    )
    trace.set_defaults(func=cmd_trace)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
