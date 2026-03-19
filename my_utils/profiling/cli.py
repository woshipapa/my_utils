from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .analyzers.metrics_analyzer import MetricsAnalyzer
from .pipeline.metrics_collector import MetricsCollector
from .output.metrics_diff import compare_reports, write_diff
from .output.metrics_report import MetricsReportRenderer
from .metrics.metrics_store import MetricsStore
from .output.metrics_trace import ChromeTraceExportConfig, export_events_file_to_chrome_trace
from .metrics.metrics_types import AnalysisReport
from .metrics.provider_registry import DEFAULT_PROVIDER_REGISTRY
from .sources.nsys_analyze import analyze_nsys_sqlite, analyze_to_markdown
from .sources.nsys_diff import diff_nsys_sqlite, diff_to_markdown
from .sources.nsys_flat_export import export_kernels_flat
from .sources.nsys_sql_skills import NsysSqlSkillEngine
from .sources.nsys_sqlite_provider import NsysSqliteMetricsProvider
from .sources.nsys_timeline_html import export_timeline_compare_html, export_timeline_html


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


def _auto_parse_value(text: str) -> Any:
    raw = str(text).strip()
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except Exception:
        return raw


def _parse_kv_params(values: Iterable[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = _auto_parse_value(value)
    return out


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
        (str(table_name),),
    ).fetchone()
    return bool(row)


def _detect_gpu_name_from_sqlite(sqlite_path: str) -> str:
    """
    Best-effort GPU name detection from nsys sqlite.
    Returns empty string when unavailable.
    """
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
    except Exception:
        return ""
    try:
        if _sqlite_table_exists(conn, "TARGET_INFO_GPU"):
            row = conn.execute(
                "SELECT name FROM TARGET_INFO_GPU ORDER BY id ASC LIMIT 1;"
            ).fetchone()
            if row and row[0]:
                return str(row[0])
        if _sqlite_table_exists(conn, "TARGET_INFO_CUDA_GPU"):
            row = conn.execute(
                "SELECT name FROM TARGET_INFO_CUDA_GPU ORDER BY id ASC LIMIT 1;"
            ).fetchone()
            if row and row[0]:
                return str(row[0])
    except Exception:
        return ""
    finally:
        conn.close()
    return ""


def _infer_unavailable_skill_reason(skill_name: str, schema_info: Dict[str, Any]) -> str:
    canonical = dict(schema_info.get("canonical_tables") or {})
    columns = dict(schema_info.get("columns") or {})
    skill = str(skill_name or "").strip()

    if skill == "gpu_metrics_aggregate":
        metrics_table = canonical.get("metrics")
        if not metrics_table:
            return (
                "no GPU metrics table detected in sqlite "
                "(expected one of GPU_METRICS / CUPTI_ACTIVITY_KIND_GPU_METRIC / "
                "CUPTI_ACTIVITY_KIND_METRIC / TARGET_INFO_GPU_METRICS). "
                "Likely profile/export did not include GPU metrics sampling."
            )
        metric_cols = set(columns.get(str(metrics_table), []) or [])
        missing_aliases: List[str] = []
        if "timestamp" not in metric_cols:
            missing_aliases.append("timestamp")
        if not any(x in metric_cols for x in ("metricId", "nameId", "eventId")):
            missing_aliases.append("metricId/nameId/eventId")
        if not any(x in metric_cols for x in ("value", "metricValue", "val")):
            missing_aliases.append("value/metricValue/val")
        string_table = canonical.get("string_ids")
        if not string_table:
            missing_aliases.append("StringIds/STRINGIDS")
        if missing_aliases:
            return "metrics schema columns/tables missing: " + ", ".join(missing_aliases)
        return "skill is schema-guarded and unavailable under current sqlite schema."

    return "skill is schema-guarded and unavailable under current sqlite schema."


def _schema_tables_grouped(schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns = dict(schema_info.get("columns") or {})
    out: List[Dict[str, Any]] = []
    for table_name in sorted(columns.keys()):
        cols = list(columns.get(table_name) or [])
        out.append(
            {
                "table_name": str(table_name),
                "column_count": len(cols),
                "columns": cols,
            }
        )
    return out


def _schema_relations(schema_info: Dict[str, Any]) -> List[Dict[str, str]]:
    tables = set((schema_info.get("tables") or []))
    columns = {str(k): set(v or []) for k, v in dict(schema_info.get("columns") or {}).items()}
    canonical = dict(schema_info.get("canonical_tables") or {})
    relations: List[Dict[str, str]] = []
    seen = set()

    def _add(frm: str, to: str, on: str, rel_type: str) -> None:
        if not frm or not to or frm == to:
            return
        if frm not in tables or to not in tables:
            return
        key = (frm, to, on, rel_type)
        if key in seen:
            return
        seen.add(key)
        relations.append(
            {
                "from_table": frm,
                "to_table": to,
                "on": on,
                "type": rel_type,
            }
        )

    kernel = str(canonical.get("kernel") or "")
    runtime = str(canonical.get("runtime") or "")
    nvtx = str(canonical.get("nvtx") or "")
    string_ids = str(canonical.get("string_ids") or "")
    metrics = str(canonical.get("metrics") or "")
    memcpy = str(canonical.get("memcpy") or "")
    memset = str(canonical.get("memset") or "")
    sync = str(canonical.get("sync") or "")

    if kernel and runtime:
        if "correlationId" in columns.get(kernel, set()) and "correlationId" in columns.get(runtime, set()):
            _add(runtime, kernel, "correlationId", "id_join")
    if nvtx and runtime:
        if "start" in columns.get(nvtx, set()) and "end" in columns.get(nvtx, set()) and "start" in columns.get(runtime, set()):
            if "globalTid" in columns.get(nvtx, set()) and "globalTid" in columns.get(runtime, set()):
                _add(nvtx, runtime, "start/end window + globalTid", "time_window")
            else:
                _add(nvtx, runtime, "start/end window", "time_window")
    if nvtx and kernel and runtime:
        if (
            "start" in columns.get(nvtx, set())
            and "end" in columns.get(nvtx, set())
            and "start" in columns.get(runtime, set())
            and "correlationId" in columns.get(runtime, set())
            and "correlationId" in columns.get(kernel, set())
        ):
            _add(nvtx, kernel, "NVTX->Runtime->correlationId", "derived")
    if metrics and nvtx:
        ts_candidates = {"timestamp", "start", "time"}
        if ts_candidates.intersection(columns.get(metrics, set())) and "start" in columns.get(nvtx, set()) and "end" in columns.get(nvtx, set()):
            _add(nvtx, metrics, "timestamp in [nvtx.start, nvtx.end]", "time_window")
    if string_ids:
        if kernel:
            if "shortName" in columns.get(kernel, set()):
                _add(kernel, string_ids, "shortName -> id", "id_join")
            if "demangledName" in columns.get(kernel, set()):
                _add(kernel, string_ids, "demangledName -> id", "id_join")
        if runtime and "nameId" in columns.get(runtime, set()):
            _add(runtime, string_ids, "nameId -> id", "id_join")
        if nvtx:
            if "textId" in columns.get(nvtx, set()):
                _add(nvtx, string_ids, "textId -> id", "id_join")
            if "nameId" in columns.get(nvtx, set()):
                _add(nvtx, string_ids, "nameId -> id", "id_join")
        if metrics:
            for col in ("metricId", "nameId", "eventId"):
                if col in columns.get(metrics, set()):
                    _add(metrics, string_ids, f"{col} -> id", "id_join")
                    break
    if metrics and "TARGET_INFO_GPU_METRICS" in tables:
        metric_cols = columns.get(metrics, set())
        gm_cols = columns.get("TARGET_INFO_GPU_METRICS", set())
        if "metricId" in metric_cols and "metricId" in gm_cols:
            if "typeId" in metric_cols and "typeId" in gm_cols:
                _add(metrics, "TARGET_INFO_GPU_METRICS", "metricId + typeId", "id_join")
            else:
                _add(metrics, "TARGET_INFO_GPU_METRICS", "metricId", "id_join")
    if metrics and "GENERIC_EVENT_SOURCES" in tables:
        metric_cols = columns.get(metrics, set())
        ges_cols = columns.get("GENERIC_EVENT_SOURCES", set())
        if ("sourceId" in metric_cols or "source_id" in metric_cols) and ("sourceId" in ges_cols or "id" in ges_cols or "source_id" in ges_cols):
            _add(metrics, "GENERIC_EVENT_SOURCES", "sourceId", "id_join")
    if "COMPOSITE_EVENTS" in tables and "ThreadNames" in tables:
        ce_cols = columns.get("COMPOSITE_EVENTS", set())
        tn_cols = columns.get("ThreadNames", set())
        if "globalTid" in ce_cols and "globalTid" in tn_cols:
            _add("COMPOSITE_EVENTS", "ThreadNames", "globalTid", "id_join")
    if "ThreadNames" in tables and string_ids and "nameId" in columns.get("ThreadNames", set()):
        _add("ThreadNames", string_ids, "nameId -> id", "id_join")

    # Helpful same-key hints for tables that share core ids.
    shared_keys = ("correlationId", "globalTid", "deviceId", "streamId")
    table_list = sorted(tables)
    for i, a in enumerate(table_list):
        for b in table_list[i + 1 :]:
            ca = columns.get(a, set())
            cb = columns.get(b, set())
            for key in shared_keys:
                if key in ca and key in cb:
                    _add(a, b, key, "shared_key_hint")
                    break

    return relations


def _schema_mermaid(schema_info: Dict[str, Any], relations: List[Dict[str, str]]) -> str:
    tables = sorted(set((schema_info.get("tables") or [])))
    if not tables:
        return "flowchart LR\n  EMPTY[\"No tables\"]\n"
    node_ids: Dict[str, str] = {table: f"T{i}" for i, table in enumerate(tables)}
    lines: List[str] = ["flowchart LR"]
    for table in tables:
        nid = node_ids[table]
        label = str(table).replace('"', '\\"')
        lines.append(f'  {nid}["{label}"]')
    for rel in relations:
        frm = str(rel.get("from_table") or "")
        to = str(rel.get("to_table") or "")
        if frm not in node_ids or to not in node_ids:
            continue
        on = str(rel.get("on") or "").replace('"', '\\"')
        lines.append(f'  {node_ids[frm]} -->|"{on}"| {node_ids[to]}')
    return "\n".join(lines) + "\n"


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


def cmd_nsys_sql_skill(args: argparse.Namespace) -> int:
    provider = NsysSqliteMetricsProvider(args.sqlite)
    if args.list_skills:
        payload = provider.describe_sql_skills()
        print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0

    if not args.skill:
        print("[nsys-sql-skill] --skill is required unless --list-skills is set")
        return 2

    params = _parse_kv_params(args.param)
    exec_params = dict(params)
    debug_enabled = bool(getattr(args, "debug", True))
    exec_params["debug"] = bool(debug_enabled)
    if debug_enabled:
        exec_params["debug_rows"] = int(getattr(args, "debug_rows", -1) or -1)
    skill_name = str(args.skill or "").strip()
    available_skills = set(provider.list_sql_skills())
    if skill_name not in available_skills:
        print(
            f"[nsys-sql-skill] skill '{skill_name}' is unavailable for this sqlite.",
            file=sys.stderr,
        )
        schema_info = provider.describe_schema()
        reason = _infer_unavailable_skill_reason(skill_name, schema_info)
        if reason:
            print(f"[nsys-sql-skill] reason: {reason}", file=sys.stderr)
        print(
            "[nsys-sql-skill] hint: run with --list-skills to see skills supported by current sqlite.",
            file=sys.stderr,
        )
        return 2

    occ_arch = str(getattr(args, "occupancy_arch", "auto") or "auto").strip().lower()
    use_h100_occupancy = False
    if skill_name in {"kernel_occupancy_estimate", "nvtx_kernel_sm_detail"} and occ_arch != "none":
        if occ_arch == "h100":
            use_h100_occupancy = True
        elif occ_arch == "auto":
            gpu_name = _detect_gpu_name_from_sqlite(args.sqlite)
            use_h100_occupancy = "h100" in str(gpu_name).lower()

    if use_h100_occupancy:
        conn = sqlite3.connect(str(args.sqlite))
        conn.row_factory = sqlite3.Row
        try:
            engine = NsysSqlSkillEngine(conn)
            if skill_name == "kernel_occupancy_estimate":
                rows = engine.execute_kernel_occupancy_estimate_h100(**exec_params)
            else:
                rows = engine.execute_nvtx_kernel_sm_detail_h100(**exec_params)
        finally:
            conn.close()
    else:
        rows = provider.run_sql_skill(skill_name, **exec_params)

    if not rows:
        print(
            f"[nsys-sql-skill] warning: skill '{skill_name}' returned 0 rows.",
            file=sys.stderr,
        )
        if skill_name == "gpu_metrics_aggregate":
            print(
                "[nsys-sql-skill] hint: check metric_name_like/start_ns/end_ns filters, "
                "and confirm profile/export included GPU metrics sampling.",
                file=sys.stderr,
            )

    if skill_name in {"kernel_occupancy_estimate", "nvtx_kernel_sm_detail"} and not use_h100_occupancy:
        print(
            "[nsys-sql-skill] note: occupancy_pct_estimate depends on sqlite theoretical occupancy columns; "
            "if absent it will be NULL. Use --occupancy-arch h100 (or auto on H100) to attach "
            "occupancy_pct_h100_estimate.",
            file=sys.stderr,
        )

    payload: Any = rows
    if skill_name == "schema_inspect":
        schema_view = str(getattr(args, "schema_view", "both") or "both").strip().lower()
        schema_info = provider.describe_schema()
        grouped = _schema_tables_grouped(schema_info)
        relations = _schema_relations(schema_info)
        mermaid = _schema_mermaid(schema_info, relations)
        if schema_view == "flat":
            payload = rows
        elif schema_view == "grouped":
            payload = {"tables": grouped}
        elif schema_view == "mermaid":
            payload = {"relations": relations, "mermaid": mermaid}
        else:
            payload = {"tables": grouped, "relations": relations, "mermaid": mermaid}

    text = json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[nsys-sql-skill] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_nsys_export(args: argparse.Namespace) -> int:
    output = export_kernels_flat(
        args.sqlite,
        output_path=args.output,
        fmt=args.format,
        device_id=args.device_id,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        limit=args.limit,
        attach_iteration=bool(args.attach_iteration),
        iteration_marker=args.iteration_marker,
    )
    print(f"[nsys-export] wrote: {output}")
    return 0


def cmd_nsys_analyze(args: argparse.Namespace) -> int:
    def _progress(msg: str) -> None:
        print(f"[nsys-analyze]{msg}", file=sys.stderr)

    result = analyze_nsys_sqlite(
        args.sqlite,
        device_id=args.device_id,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        nvtx_text=str(args.nvtx_text).strip() or None,
        top_k=args.top_k,
        iteration_marker=args.iteration_marker,
        model_flops_per_step=args.model_flops_per_step,
        peak_tflops=args.peak_tflops,
        peak_precision=args.peak_precision,
        limit=args.limit,
        progress_cb=_progress,
    )
    if str(args.format).lower() in ("md", "markdown"):
        text = analyze_to_markdown(result)
    else:
        text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[nsys-analyze] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_nsys_diff(args: argparse.Namespace) -> int:
    result = diff_nsys_sqlite(
        args.before_sqlite,
        args.after_sqlite,
        device_id=args.device_id,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        top_k=args.top_k,
    )
    if str(args.format).lower() in ("md", "markdown"):
        text = diff_to_markdown(result)
    else:
        text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[nsys-diff] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_nsys_iter_overlap(args: argparse.Namespace) -> int:
    provider = NsysSqliteMetricsProvider(args.sqlite)
    rows = provider.analyze_per_iteration_overlap(
        marker=args.iteration_marker,
        device_id=args.device_id,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        top_level_only=not args.include_nested,
        limit=args.limit,
    )
    text = json.dumps(rows, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[nsys-iter-overlap] wrote: {path}  ({len(rows)} iterations)")
    else:
        print(text)
    return 0


def cmd_nsys_iter_outliers(args: argparse.Namespace) -> int:
    provider = NsysSqliteMetricsProvider(args.sqlite)
    result = provider.detect_iteration_outliers(
        marker=args.iteration_marker,
        device_id=args.device_id,
        threshold_sigma=args.sigma,
        limit=args.limit,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        outlier_count = len((result.get("outliers") or []))
        print(f"[nsys-iter-outliers] wrote: {path}  ({outlier_count} outliers)")
    else:
        print(text)
    return 0


def cmd_nsys_timeline_html(args: argparse.Namespace) -> int:
    def _progress(msg: str) -> None:
        print(f"[nsys-timeline-html]{msg}", file=sys.stderr)

    output = export_timeline_html(
        args.sqlite,
        output_path=args.output,
        device_id=args.device_id,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        limit=args.limit,
        width_px=args.width_px,
        nvtx_text=args.nvtx_text,
        nvtx_index=args.nvtx_index,
        include_metrics=bool(args.include_metrics),
        metric_name_like=args.metric_name_like,
        metrics_limit=args.metrics_limit,
        metrics_max_points=args.metrics_max_points,
        overlay_metrics_per_track=args.overlay_metrics_per_track,
        kernel_category_map_json=args.kernel_category_map_json,
        kernel_category_engine=args.kernel_category_engine,
        kernel_category_model=args.kernel_category_model,
        enable_kernel_category_breakdown=bool(args.enable_kernel_category_breakdown),
        kernel_category_table_output=args.kernel_category_table_output,
        nvtx_category_stats_output=args.nvtx_category_stats_output,
        default_focus_metrics=bool(args.default_focus_metrics),
        include_all_metric_sources=bool(args.include_all_metric_sources),
        debug=bool(args.debug),
        debug_rows=int(args.debug_rows),
        progress_cb=_progress,
    )
    print(f"[nsys-timeline-html] wrote: {output}")
    return 0


def cmd_nsys_timeline_compare_html(args: argparse.Namespace) -> int:
    def _progress(msg: str) -> None:
        print(f"[nsys-timeline-compare-html] {msg}", file=sys.stderr)

    output = export_timeline_compare_html(
        args.sqlite,
        output_path=args.output,
        device_id=args.device_id,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        limit=args.limit,
        width_px=args.width_px,
        nvtx_text=args.nvtx_text,
        nvtx_index=args.nvtx_index,
        include_metrics=bool(args.include_metrics),
        metric_name_like=args.metric_name_like,
        metrics_limit=args.metrics_limit,
        metrics_max_points=args.metrics_max_points,
        overlay_metrics_per_track=args.overlay_metrics_per_track,
        kernel_category_map_json=args.kernel_category_map_json,
        kernel_category_engine=args.kernel_category_engine,
        kernel_category_model=args.kernel_category_model,
        enable_kernel_category_breakdown=bool(args.enable_kernel_category_breakdown),
        kernel_category_table_output=args.kernel_category_table_output,
        nvtx_category_stats_output=args.nvtx_category_stats_output,
        default_focus_metrics=bool(args.default_focus_metrics),
        include_all_metric_sources=bool(args.include_all_metric_sources),
        debug=bool(args.debug),
        debug_rows=int(args.debug_rows),
        progress_cb=_progress,
    )
    print(f"[nsys-timeline-compare-html] wrote: {output}")
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

    nsys_sql = sub.add_parser("nsys-sql-skill", help="run built-in Nsight SQLite SQL skills")
    nsys_sql.add_argument("--sqlite", required=True, help="nsys exported sqlite path")
    nsys_sql.add_argument("--list-skills", action="store_true", help="list available skills and parameters")
    nsys_sql.add_argument("--skill", default="", help="skill name, e.g. top_kernels")
    nsys_sql.add_argument(
        "--param",
        action="append",
        default=[],
        help="skill parameter in key=value format (can repeat), e.g. --param device_id=0 --param limit=20",
    )
    nsys_sql.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=True,
        help="enable per-skill query debug logs (default: enabled)",
    )
    nsys_sql.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="disable per-skill query debug logs",
    )
    nsys_sql.add_argument(
        "--debug-rows",
        type=int,
        default=-1,
        help="sample row count for debug logs; <=0 means no limit",
    )
    nsys_sql.add_argument(
        "--occupancy-arch",
        default="auto",
        choices=["auto", "none", "h100"],
        help=(
            "occupancy enrichment mode for occupancy-related skills: "
            "auto=detect H100 from sqlite and compute in Python, "
            "none=keep SQL field (NULL), h100=force H100 occupancy computation"
        ),
    )
    nsys_sql.add_argument("--output", default="", help="optional json output path")
    nsys_sql.add_argument(
        "--schema-view",
        default="both",
        choices=["flat", "grouped", "mermaid", "both"],
        help=(
            "schema_inspect output mode: "
            "flat=raw rows, grouped=all columns grouped by table, "
            "mermaid=relations+mermaid graph, both=grouped tables + relations + mermaid"
        ),
    )
    nsys_sql.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    nsys_sql.set_defaults(func=cmd_nsys_sql_skill)

    nsys_export = sub.add_parser("nsys-export", help="export sqlite kernel timeline rows to json/csv")
    nsys_export.add_argument("--sqlite", required=True, help="nsys exported sqlite path")
    nsys_export.add_argument("--output", required=True, help="output file path")
    nsys_export.add_argument("--format", default="json", choices=["json", "csv"])
    nsys_export.add_argument("--device-id", type=int, default=-1)
    nsys_export.add_argument("--start-ns", type=int, default=-1)
    nsys_export.add_argument("--end-ns", type=int, default=-1)
    nsys_export.add_argument("--limit", type=int, default=500000)
    nsys_export.add_argument("--attach-iteration", action="store_true")
    nsys_export.add_argument("--iteration-marker", default="sample_0")
    nsys_export.set_defaults(func=cmd_nsys_export)

    nsys_analyze = sub.add_parser("nsys-analyze", help="run nsys-oriented summary/overlap/nccl/iterations/mfu analysis")
    nsys_analyze.add_argument("--sqlite", required=True, help="nsys exported sqlite path")
    nsys_analyze.add_argument("--device-id", type=int, default=-1)
    nsys_analyze.add_argument("--start-ns", type=int, default=-1)
    nsys_analyze.add_argument("--end-ns", type=int, default=-1)
    nsys_analyze.add_argument("--top-k", type=int, default=10)
    nsys_analyze.add_argument("--limit", type=int, default=500000)
    nsys_analyze.add_argument("--iteration-marker", default="sample_0")
    nsys_analyze.add_argument(
        "--nvtx-text",
        default="",
        help=(
            "NVTX text LIKE pattern to restrict analysis to the union time window of all "
            "matching NVTX ranges (e.g. 'forward', '%%step_%%', 'backward'). "
            "When set, start_ns/end_ns are derived automatically from matching ranges "
            "unless --start-ns/--end-ns are explicitly provided."
        ),
    )
    nsys_analyze.add_argument("--model-flops-per-step", type=float, default=None)
    nsys_analyze.add_argument("--peak-tflops", type=float, default=None)
    nsys_analyze.add_argument("--peak-precision", default="fp16")
    nsys_analyze.add_argument("--format", default="json", choices=["json", "markdown", "md"])
    nsys_analyze.add_argument("--output", default="")
    nsys_analyze.add_argument("--pretty", action="store_true")
    nsys_analyze.set_defaults(func=cmd_nsys_analyze)

    nsys_diff = sub.add_parser("nsys-diff", help="diff two nsys sqlite profiles by kernel/nvtx aggregates")
    nsys_diff.add_argument("--before-sqlite", required=True)
    nsys_diff.add_argument("--after-sqlite", required=True)
    nsys_diff.add_argument("--device-id", type=int, default=-1)
    nsys_diff.add_argument("--start-ns", type=int, default=-1)
    nsys_diff.add_argument("--end-ns", type=int, default=-1)
    nsys_diff.add_argument("--top-k", type=int, default=20)
    nsys_diff.add_argument("--format", default="json", choices=["json", "markdown", "md"])
    nsys_diff.add_argument("--output", default="")
    nsys_diff.add_argument("--pretty", action="store_true")
    nsys_diff.set_defaults(func=cmd_nsys_diff)

    nsys_timeline = sub.add_parser(
        "nsys-timeline-html",
        help="export static html timeline from nsys sqlite",
        description=(
            "Export one interactive HTML timeline from a single nsys SQLite.\n"
            "Window selection order:\n"
            "1) If --nvtx-text is set, match NVTX ranges first (and optionally pick one by --nvtx-index).\n"
            "2) Otherwise use explicit --start-ns/--end-ns if valid.\n"
            "3) Otherwise infer from collected kernel GPU execution timestamps.\n"
            "Kernel/category ratios are computed from GPU kernel execution time (start/end), not CPU NVTX duration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nsys_timeline.add_argument(
        "--sqlite",
        required=True,
        help="path to one nsys-exported SQLite file",
    )
    nsys_timeline.add_argument(
        "--output",
        required=True,
        help="output HTML file path",
    )
    nsys_timeline.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="GPU device filter for kernels/metrics; -1 means all devices (default: -1)",
    )
    nsys_timeline.add_argument(
        "--start-ns",
        type=int,
        default=-1,
        help="timeline start timestamp (ns); <0 means unset (default: -1)",
    )
    nsys_timeline.add_argument(
        "--end-ns",
        type=int,
        default=-1,
        help="timeline end timestamp (ns); <0 means unset (default: -1)",
    )
    nsys_timeline.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="max kernel rows to collect per query path (default: 100000)",
    )
    nsys_timeline.add_argument(
        "--width-px",
        type=int,
        default=1800,
        help="base plot width in pixels for stream/all-stream panels (default: 1800)",
    )
    nsys_timeline.add_argument(
        "--nvtx-text",
        default="",
        help=(
            "SQL LIKE pattern for NVTX text matching (example: 'qwen_layer_%%'). "
            "When set, timeline focuses on matched NVTX scopes."
        ),
    )
    nsys_timeline.add_argument(
        "--nvtx-index",
        type=int,
        default=-1,
        help=(
            "index of matched NVTX scope after start-time sort when --nvtx-text is set. "
            "-1 means use all matched scopes (default: -1)"
        ),
    )
    nsys_timeline.add_argument(
        "--include-metrics",
        action="store_true",
        help="enable GPU metrics collection/rendering in the selected timeline window",
    )
    nsys_timeline.add_argument(
        "--metric-name-like",
        default="%",
        help=(
            "SQL LIKE filter on metric names when --include-metrics is enabled "
            "(default: '%%', i.e. no name filter)"
        ),
    )
    nsys_timeline.add_argument(
        "--metrics-limit",
        type=int,
        default=-1,
        help=(
            "global metric row cap before series-level sampling. "
            "<=0 disables this cap and keeps all fetched rows (default: -1)"
        ),
    )
    nsys_timeline.add_argument(
        "--metrics-max-points",
        type=int,
        default=-1,
        help=(
            "max rendered points per metric series. "
            "<=0 keeps all points in each series (default: -1)"
        ),
    )
    nsys_timeline.add_argument(
        "--overlay-metrics-per-track",
        type=int,
        default=7,
        help=(
            "number of metric series overlaid onto each stream lane for attribution view. "
            "0 disables per-track metric overlays (default: 7)"
        ),
    )
    nsys_timeline.add_argument(
        "--default-focus-metrics",
        dest="default_focus_metrics",
        action="store_true",
        default=False,
        help=(
            "when --metric-name-like is '%%' (or empty), keep only built-in focus metrics "
            "for attribution (default: disabled)"
        ),
    )
    nsys_timeline.add_argument(
        "--no-default-focus-metrics",
        dest="default_focus_metrics",
        action="store_false",
        help="explicitly disable built-in focus-metric filtering",
    )
    nsys_timeline.add_argument(
        "--include-all-metric-sources",
        action="store_true",
        help=(
            "include non-GPU generic metric sources (for example ETW/FTrace) "
            "in addition to GPU metric sources"
        ),
    )
    nsys_timeline.add_argument(
        "--kernel-category-map-json",
        default="",
        help=(
            "optional JSON file for kernel-category rules. "
            "Supported formats: {pattern: category} or nested "
            "{engine:{model:{pattern:category}}}."
        ),
    )
    nsys_timeline.add_argument(
        "--kernel-category-engine",
        default="sglang",
        help="engine key when using nested kernel-category mapping (default: sglang)",
    )
    nsys_timeline.add_argument(
        "--kernel-category-model",
        default="llama",
        help="model key when using nested kernel-category mapping (default: llama)",
    )
    nsys_timeline.add_argument(
        "--disable-kernel-category-breakdown",
        dest="enable_kernel_category_breakdown",
        action="store_false",
        default=True,
        help="disable the overlap-aware kernel-category breakdown panel in HTML output",
    )
    nsys_timeline.add_argument(
        "--enable-kernel-category-breakdown",
        dest="enable_kernel_category_breakdown",
        action="store_true",
        help="enable overlap-aware kernel-category breakdown panel (default: enabled)",
    )
    nsys_timeline.add_argument(
        "--kernel-category-table-output",
        default="",
        help=(
            "optional output path for kernel-category membership table "
            "(which kernels belong to which category). "
            "Use .csv for CSV, otherwise JSON."
        ),
    )
    nsys_timeline.add_argument(
        "--nvtx-category-stats-output",
        default="",
        help=(
            "optional JSON output path for per-matched-NVTX category stability stats: "
            "per-window category ratio, aggregate avg/std/min/max, and outlier scopes."
        ),
    )
    nsys_timeline.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=True,
        help="enable debug diagnostics for timeline collection/rendering (default: enabled)",
    )
    nsys_timeline.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="disable timeline debug diagnostics",
    )
    nsys_timeline.add_argument(
        "--debug-rows",
        type=int,
        default=-1,
        help=(
            "row preview limit in debug logs. "
            "<=0 means no row-preview limit (default: -1)"
        ),
    )
    nsys_timeline.set_defaults(func=cmd_nsys_timeline_html)

    nsys_timeline_compare = sub.add_parser(
        "nsys-timeline-compare-html",
        help="export a single html page that compares multiple nsys sqlite timelines",
        description=(
            "Export one compare HTML page for multiple nsys SQLite files.\n"
            "Each sqlite is collected independently, then rendered in aligned sections "
            "(all-stream overlap, matched NVTX scopes, stream timeline, summaries).\n"
            "Category ratios are still computed from GPU kernel execution time, not CPU NVTX duration."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nsys_timeline_compare.add_argument(
        "--sqlite",
        required=True,
        action="append",
        help=(
            "path to one nsys SQLite; repeat this option to include multiple profiles "
            "(at least two required)"
        ),
    )
    nsys_timeline_compare.add_argument("--output", required=True, help="output compare HTML file path")
    nsys_timeline_compare.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="GPU device filter for kernels/metrics; -1 means all devices (default: -1)",
    )
    nsys_timeline_compare.add_argument(
        "--start-ns",
        type=int,
        default=-1,
        help="timeline start timestamp (ns) per sqlite when no NVTX override is used (default: -1)",
    )
    nsys_timeline_compare.add_argument(
        "--end-ns",
        type=int,
        default=-1,
        help="timeline end timestamp (ns) per sqlite when no NVTX override is used (default: -1)",
    )
    nsys_timeline_compare.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="max kernel rows to collect per sqlite/query path (default: 100000)",
    )
    nsys_timeline_compare.add_argument(
        "--width-px",
        type=int,
        default=1800,
        help="base plot width in pixels for each compared panel (default: 1800)",
    )
    nsys_timeline_compare.add_argument(
        "--nvtx-text",
        default="",
        help=(
            "SQL LIKE pattern for NVTX text matching. "
            "Each sqlite uses its own matched NVTX scope(s)."
        ),
    )
    nsys_timeline_compare.add_argument(
        "--nvtx-index",
        type=int,
        default=-1,
        help=(
            "index of matched NVTX scope after start-time sort per sqlite. "
            "-1 means use all matched scopes (default: -1)"
        ),
    )
    nsys_timeline_compare.add_argument(
        "--include-metrics",
        action="store_true",
        help="enable GPU metrics collection/rendering for each compared sqlite",
    )
    nsys_timeline_compare.add_argument(
        "--metric-name-like",
        default="%",
        help=(
            "SQL LIKE filter on metric names when --include-metrics is enabled "
            "(default: '%%', i.e. no name filter)"
        ),
    )
    nsys_timeline_compare.add_argument(
        "--metrics-limit",
        type=int,
        default=-1,
        help=(
            "global metric row cap before series-level sampling. "
            "<=0 disables this cap (default: -1)"
        ),
    )
    nsys_timeline_compare.add_argument(
        "--metrics-max-points",
        type=int,
        default=-1,
        help=(
            "max rendered points per metric series. "
            "<=0 keeps all points in each series (default: -1)"
        ),
    )
    nsys_timeline_compare.add_argument(
        "--overlay-metrics-per-track",
        type=int,
        default=7,
        help=(
            "number of metric series overlaid onto each stream lane per sqlite. "
            "0 disables per-track overlays (default: 7)"
        ),
    )
    nsys_timeline_compare.add_argument(
        "--default-focus-metrics",
        dest="default_focus_metrics",
        action="store_true",
        default=False,
        help=(
            "when --metric-name-like is '%%' (or empty), keep only built-in focus metrics "
            "for attribution (default: disabled)"
        ),
    )
    nsys_timeline_compare.add_argument(
        "--no-default-focus-metrics",
        dest="default_focus_metrics",
        action="store_false",
        help="explicitly disable built-in focus-metric filtering",
    )
    nsys_timeline_compare.add_argument(
        "--include-all-metric-sources",
        action="store_true",
        help="include non-GPU generic metric sources (ETW/FTrace/etc)",
    )
    nsys_timeline_compare.add_argument(
        "--kernel-category-map-json",
        default="",
        help=(
            "optional JSON file for kernel-category rules. "
            "Supported formats: {pattern: category} or nested "
            "{engine:{model:{pattern:category}}}."
        ),
    )
    nsys_timeline_compare.add_argument(
        "--kernel-category-engine",
        default="sglang",
        help="engine key when using nested kernel-category mapping (default: sglang)",
    )
    nsys_timeline_compare.add_argument(
        "--kernel-category-model",
        default="llama",
        help="model key when using nested kernel-category mapping (default: llama)",
    )
    nsys_timeline_compare.add_argument(
        "--disable-kernel-category-breakdown",
        dest="enable_kernel_category_breakdown",
        action="store_false",
        default=True,
        help="disable overlap-aware kernel-category breakdown panels in compare HTML",
    )
    nsys_timeline_compare.add_argument(
        "--enable-kernel-category-breakdown",
        dest="enable_kernel_category_breakdown",
        action="store_true",
        help="enable overlap-aware kernel-category breakdown panels (default: enabled)",
    )
    nsys_timeline_compare.add_argument(
        "--kernel-category-table-output",
        default="",
        help=(
            "optional output path for merged kernel-category membership table across sqlites. "
            "Use .csv for CSV, otherwise JSON."
        ),
    )
    nsys_timeline_compare.add_argument(
        "--nvtx-category-stats-output",
        default="",
        help=(
            "optional JSON output path for per-sqlite NVTX category stability stats "
            "(per-window ratios, aggregate stats, and outlier windows)."
        ),
    )
    nsys_timeline_compare.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=True,
        help="enable debug diagnostics for compare collection/rendering (default: enabled)",
    )
    nsys_timeline_compare.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="disable compare debug diagnostics",
    )
    nsys_timeline_compare.add_argument(
        "--debug-rows",
        type=int,
        default=-1,
        help=(
            "row preview limit in debug logs. "
            "<=0 means no row-preview limit (default: -1)"
        ),
    )
    nsys_timeline_compare.set_defaults(func=cmd_nsys_timeline_compare_html)

    nsys_iter_overlap = sub.add_parser(
        "nsys-iter-overlap",
        help="per-iteration compute/comm/overlap breakdown from NVTX markers",
        description=(
            "Detect training iterations via NVTX marker ranges, then for each iteration "
            "compute the compute-only / comm-only / overlapping duration breakdown. "
            "Kernels whose name contains 'nccl' (case-insensitive) are classified as comm; "
            "all others are compute. Overlap is measured by intersecting merged interval sets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nsys_iter_overlap.add_argument("--sqlite", required=True, help="nsys exported sqlite path")
    nsys_iter_overlap.add_argument(
        "--iteration-marker", default="sample_0",
        help="NVTX text substring that marks iteration boundaries (default: sample_0)",
    )
    nsys_iter_overlap.add_argument("--device-id", type=int, default=-1, help="CUDA device ID, -1 = all")
    nsys_iter_overlap.add_argument("--start-ns", type=int, default=-1, help="global window start ns, -1 = no filter")
    nsys_iter_overlap.add_argument("--end-ns", type=int, default=-1, help="global window end ns, -1 = no filter")
    nsys_iter_overlap.add_argument(
        "--include-nested", action="store_true",
        help="include nested NVTX ranges with same marker (default: top-level only)",
    )
    nsys_iter_overlap.add_argument("--limit", type=int, default=2000, help="max iterations to process (default: 2000)")
    nsys_iter_overlap.add_argument("--output", default="", help="optional JSON output path")
    nsys_iter_overlap.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    nsys_iter_overlap.set_defaults(func=cmd_nsys_iter_overlap)

    nsys_iter_outliers = sub.add_parser(
        "nsys-iter-outliers",
        help="detect statistically anomalous training iterations by step duration",
        description=(
            "Detect training iterations via NVTX marker ranges, compute step duration statistics "
            "(mean, median, std, p95, p99), and flag iterations whose duration deviates from the "
            "median by more than --sigma standard deviations. "
            "Useful for identifying GC pauses, NCCL stragglers, or system noise."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nsys_iter_outliers.add_argument("--sqlite", required=True, help="nsys exported sqlite path")
    nsys_iter_outliers.add_argument(
        "--iteration-marker", default="sample_0",
        help="NVTX text substring that marks iteration boundaries (default: sample_0)",
    )
    nsys_iter_outliers.add_argument("--device-id", type=int, default=-1, help="CUDA device ID, -1 = all")
    nsys_iter_outliers.add_argument(
        "--sigma", type=float, default=2.0,
        help="flag iterations deviating more than this many σ from median (default: 2.0)",
    )
    nsys_iter_outliers.add_argument("--limit", type=int, default=2000, help="max iterations to scan (default: 2000)")
    nsys_iter_outliers.add_argument("--output", default="", help="optional JSON output path")
    nsys_iter_outliers.add_argument("--pretty", action="store_true", help="pretty-print JSON output")
    nsys_iter_outliers.set_defaults(func=cmd_nsys_iter_outliers)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _run_nsys_alias(subcommand: str) -> int:
    return main([str(subcommand)] + list(sys.argv[1:]))


def entry_nsys_sql_skill() -> int:
    return _run_nsys_alias("nsys-sql-skill")


def entry_nsys_export() -> int:
    return _run_nsys_alias("nsys-export")


def entry_nsys_analyze() -> int:
    return _run_nsys_alias("nsys-analyze")


def entry_nsys_diff() -> int:
    return _run_nsys_alias("nsys-diff")


def entry_nsys_timeline_html() -> int:
    return _run_nsys_alias("nsys-timeline-html")


def entry_nsys_iter_overlap() -> int:
    return _run_nsys_alias("nsys-iter-overlap")


def entry_nsys_iter_outliers() -> int:
    return _run_nsys_alias("nsys-iter-outliers")


if __name__ == "__main__":
    raise SystemExit(main())
