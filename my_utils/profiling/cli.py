# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import shlex
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .analyzers.metrics_analyzer import MetricsAnalyzer
from .pipeline.metrics_collector import MetricsCollector
from .output.metrics_diff import compare_reports, write_diff
from .output.metrics_report import MetricsReportRenderer
from .metrics.metrics_store import MetricsStore
from .output.metrics_trace import (
    ChromeTraceExportConfig,
    export_events_file_to_chrome_trace,
)
from .metrics.metrics_types import AnalysisReport
from .metrics.provider_registry import DEFAULT_PROVIDER_REGISTRY
from .sources.nsys_analyze import analyze_nsys_sqlite, analyze_to_markdown
from .sources.nsys_diff import diff_nsys_sqlite, diff_to_markdown
from .sources.nsys_flat_export import export_kernels_flat
from .sources.nsys_module_kernel_compare import (
    compare_module_kernel_json,
    compare_module_kernel_rows,
    module_kernel_compare_to_html,
    module_kernel_compare_to_markdown,
)
from .sources.nsys_sql_skills import NsysSqlSkillEngine
from .sources.nsys_sqlite_provider import NsysSqliteMetricsProvider
from .sources.nsys_timeline_html import (
    export_timeline_compare_html,
    export_timeline_html,
)
from .ncu.ncu_csv_tools import (
    NcuCsvSkillEngine,
    analyze_ncu_csv,
    analyze_ncu_to_markdown,
    skill_result_to_json,
)
from .ncu.ncu_report_tools import (
    diagnose_ncu_report,
    diagnose_result_to_markdown,
    NcuReportSkillEngine,
    analyze_ncu_report,
    analyze_ncu_report_to_markdown,
    report_result_to_json,
)
from .ncu.report_diff import diff_ncu_reports, diff_result_to_markdown
from .nccl.nccl_inspector_tools import (
    NcclInspectorSkillEngine,
    analyze_nccl_inspector,
    analyze_nccl_inspector_to_markdown,
    inspector_result_to_json,
)


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


def _choose_opt_name(action: argparse.Action) -> str:
    for opt in action.option_strings:
        if str(opt).startswith("--"):
            return str(opt)
    if action.option_strings:
        return str(action.option_strings[0])
    return str(action.dest)


def _action_help_text(action: argparse.Action) -> str:
    text = str(action.help or "").strip()
    return text if text else "(no description)"


def _action_meta_text(action: argparse.Action) -> str:
    parts: List[str] = []
    if action.required:
        parts.append("required")
    if action.choices:
        parts.append("choices=" + ",".join(str(x) for x in action.choices))
    if action.default not in (None, argparse.SUPPRESS):
        parts.append(f"default={action.default}")
    return "; ".join(parts)


def _is_bool_action(action: argparse.Action) -> bool:
    return isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))


def _is_append_action(action: argparse.Action) -> bool:
    return isinstance(action, argparse._AppendAction)


def _split_user_values(text: str) -> List[str]:
    raw = [x.strip() for x in str(text or "").split(",")]
    return [x for x in raw if x]


def _collect_subparsers(
    parser: argparse.ArgumentParser,
) -> Dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return {str(name): sub for name, sub in dict(action.choices).items()}
    return {}


def _collect_subparser_help(parser: argparse.ArgumentParser) -> Dict[str, str]:
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        out: Dict[str, str] = {}
        for choice_action in action._choices_actions:
            out[str(choice_action.dest)] = str(choice_action.help or "").strip()
        return out
    return {}


def _iter_user_actions(parser: argparse.ArgumentParser) -> List[argparse.Action]:
    out: List[argparse.Action] = []
    for action in parser._actions:
        if not action.option_strings:
            continue
        if action.dest in {"help", "func", "command"}:
            continue
        out.append(action)
    return out


def _build_args_from_action(action: argparse.Action, raw_value: str) -> List[str]:
    name = _choose_opt_name(action)
    value = str(raw_value).strip()
    if not value:
        return []
    if _is_bool_action(action):
        low = value.lower()
        if low in {"", "default", "keep", "skip"}:
            return []
        if low in {"1", "y", "yes", "true", "on", "enable", "enabled"}:
            target = True
        elif low in {"0", "n", "no", "false", "off", "disable", "disabled"}:
            target = False
        else:
            return []
        if isinstance(action, argparse._StoreTrueAction):
            return [name] if target else []
        # store_false: option means disable (set False)
        return [name] if (not target) else []
    if _is_append_action(action):
        parts = _split_user_values(value)
        out: List[str] = []
        for item in parts:
            out.extend([name, item])
        return out
    nargs = action.nargs
    if nargs in ("+", "*"):
        parts = _split_user_values(value)
        if not parts:
            return []
        return [name] + parts
    return [name, value]


def _prompt_bool(question: str, *, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    raw = input(f"{question} [{suffix}]: ").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"y", "yes", "1", "true", "on"}


def _normalize_bool_text(raw: str) -> Optional[bool]:
    low = str(raw or "").strip().lower()
    if not low:
        return None
    if low in {"1", "y", "yes", "true", "on", "enable", "enabled"}:
        return True
    if low in {"0", "n", "no", "false", "off", "disable", "disabled"}:
        return False
    if low in {"default", "keep", "skip", "s", "k", "d"}:
        return None
    return None


def _group_actions_by_dest(
    actions: List[argparse.Action],
) -> List[List[argparse.Action]]:
    by_dest: Dict[str, List[argparse.Action]] = {}
    for action in actions:
        by_dest.setdefault(str(action.dest), []).append(action)
    grouped: List[List[argparse.Action]] = []
    seen: set[str] = set()
    for action in actions:
        dest = str(action.dest)
        if dest in seen:
            continue
        seen.add(dest)
        grouped.append(list(by_dest.get(dest, [action])))
    return grouped


def _bool_group_default(actions: List[argparse.Action]) -> bool:
    for action in actions:
        if action.default not in (None, argparse.SUPPRESS):
            return bool(action.default)
    # Heuristic fallback based on action type.
    if all(isinstance(action, argparse._StoreFalseAction) for action in actions):
        return True
    return False


def _bool_group_option_for_target(actions: List[argparse.Action], target: bool) -> str:
    if target:
        for action in actions:
            if isinstance(action, argparse._StoreTrueAction):
                return _choose_opt_name(action)
    else:
        for action in actions:
            if isinstance(action, argparse._StoreFalseAction):
                return _choose_opt_name(action)
    return ""


def _coerce_value_for_action(action: argparse.Action, raw: str) -> Any:
    value = str(raw or "").strip()
    if not value:
        return None
    if _is_append_action(action):
        return _split_user_values(value)
    if action.nargs in ("+", "*"):
        return _split_user_values(value)
    if action.type is not None:
        try:
            return action.type(value)
        except Exception:
            return value
    return value


def _nsys_panel_skip_reason(
    command_name: str,
    action_group: List[argparse.Action],
    selected_values: Dict[str, Any],
) -> str:
    if not action_group:
        return ""
    dest = str(action_group[0].dest or "")
    command = str(command_name or "")

    if dest == "debug_rows" and selected_values.get("debug") is False:
        return "requires debug mode enabled"

    if command in {"nsys-timeline-html", "nsys-timeline-compare-html"}:
        metric_related = {
            "metric_name_like",
            "metrics_limit",
            "metrics_max_points",
            "overlay_metrics_per_track",
            "default_focus_metrics",
            "include_all_metric_sources",
        }
        if dest in metric_related and not bool(
            selected_values.get("include_metrics", False)
        ):
            return "requires --include-metrics"
        if (
            dest == "nvtx_index"
            and not str(selected_values.get("nvtx_text") or "").strip()
        ):
            return "requires --nvtx-text"

    if command == "nsys-sql-skill":
        if bool(selected_values.get("list_skills", False)):
            if dest in {
                "skill",
                "param",
                "debug",
                "debug_rows",
                "occupancy_arch",
                "schema_view",
                "output",
            }:
                return "ignored when --list-skills is enabled"
        skill_name = str(selected_values.get("skill") or "").strip().lower()
        if dest == "schema_view" and skill_name not in {"schema_inspect"}:
            return "only used when --skill schema_inspect"
        if dest == "occupancy_arch" and skill_name not in {
            "kernel_occupancy_estimate",
            "nvtx_kernel_sm_detail",
        }:
            return "only used for occupancy-related skills"

    if command == "nsys-analyze":
        if dest in {"peak_tflops", "peak_precision"} and selected_values.get(
            "model_flops_per_step"
        ) in (None, "", 0):
            return "only useful when --model-flops-per-step is set"

    if command == "nsys-module-kernel-compare":
        has_json = bool(str(selected_values.get("base_json") or "").strip()) or bool(
            str(selected_values.get("target_json") or "").strip()
        )
        has_sqlite = bool(
            str(selected_values.get("base_sqlite") or "").strip()
        ) or bool(str(selected_values.get("target_sqlite") or "").strip())
        has_nvtx_text = bool(str(selected_values.get("nvtx_text") or "").strip())
        if has_json and dest in {
            "base_sqlite",
            "target_sqlite",
            "sqlite_limit",
            "occupancy_arch",
        }:
            return "ignored in JSON input mode"
        if has_sqlite and dest in {"base_json", "target_json"}:
            return "ignored in sqlite input mode"
        if dest in {"sqlite_limit", "occupancy_arch"} and not has_sqlite:
            return "only used in sqlite input mode"
        if dest == "nvtx_index" and not has_nvtx_text:
            return "requires --nvtx-text"

    return ""


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


def _infer_unavailable_skill_reason(
    skill_name: str, schema_info: Dict[str, Any]
) -> str:
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
            return "metrics schema columns/tables missing: " + ", ".join(
                missing_aliases
            )
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
    columns = {
        str(k): set(v or []) for k, v in dict(schema_info.get("columns") or {}).items()
    }
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
        if "correlationId" in columns.get(
            kernel, set()
        ) and "correlationId" in columns.get(runtime, set()):
            _add(runtime, kernel, "correlationId", "id_join")
    if nvtx and runtime:
        if (
            "start" in columns.get(nvtx, set())
            and "end" in columns.get(nvtx, set())
            and "start" in columns.get(runtime, set())
        ):
            if "globalTid" in columns.get(nvtx, set()) and "globalTid" in columns.get(
                runtime, set()
            ):
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
        if (
            ts_candidates.intersection(columns.get(metrics, set()))
            and "start" in columns.get(nvtx, set())
            and "end" in columns.get(nvtx, set())
        ):
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
        if ("sourceId" in metric_cols or "source_id" in metric_cols) and (
            "sourceId" in ges_cols or "id" in ges_cols or "source_id" in ges_cols
        ):
            _add(metrics, "GENERIC_EVENT_SOURCES", "sourceId", "id_join")
    if "COMPOSITE_EVENTS" in tables and "ThreadNames" in tables:
        ce_cols = columns.get("COMPOSITE_EVENTS", set())
        tn_cols = columns.get("ThreadNames", set())
        if "globalTid" in ce_cols and "globalTid" in tn_cols:
            _add("COMPOSITE_EVENTS", "ThreadNames", "globalTid", "id_join")
    if (
        "ThreadNames" in tables
        and string_ids
        and "nameId" in columns.get("ThreadNames", set())
    ):
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


def _schema_mermaid(
    schema_info: Dict[str, Any], relations: List[Dict[str, str]]
) -> str:
    tables = sorted(set((schema_info.get("tables") or [])))
    if not tables:
        return 'flowchart LR\n  EMPTY["No tables"]\n'
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
    print(
        f"[ingest] providers={collector.list_providers()} written_events={total_written}"
    )
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
    analyzer = MetricsAnalyzer(
        workload_profile=args.workload,
        enable_advanced_rules=not args.disable_advanced_rules,
    )
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
        include_metric_prefixes=[
            item.strip()
            for item in (args.include_metric_prefixes or "").split(",")
            if item.strip()
        ],
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


def cmd_nsys_panel(args: argparse.Namespace) -> int:
    parser = build_parser()
    subparsers = _collect_subparsers(parser)
    help_map = _collect_subparser_help(parser)
    profile_cmds = sorted(
        [
            name
            for name in subparsers.keys()
            if (name.startswith("nsys-") or name.startswith("ncu-"))
            and name != "nsys-panel"
        ]
    )
    if not profile_cmds:
        print("[nsys-panel] no profiling subcommands found")
        return 2

    print("=== Profiling Interactive Panel (NSYS + NCU) ===")
    print("Select one command by index or name:")
    for idx, name in enumerate(profile_cmds, start=1):
        desc = (
            help_map.get(name)
            or str(getattr(subparsers[name], "description", "") or "").strip()
        )
        print(f"  {idx}. {name} - {desc}")

    selected_name = ""
    while not selected_name:
        raw = input("command > ").strip()
        if not raw:
            continue
        if raw in subparsers and raw in profile_cmds:
            selected_name = raw
            break
        try:
            idx = int(raw)
        except Exception:
            idx = -1
        if 1 <= idx <= len(profile_cmds):
            selected_name = profile_cmds[idx - 1]
            break
        print("Invalid selection, try again.")

    selected = subparsers[selected_name]
    desc = str(getattr(selected, "description", "") or "").strip()
    if desc:
        print(f"\n[{selected_name}] {desc}\n")
    else:
        print(f"\n[{selected_name}]\n")

    user_actions = _iter_user_actions(selected)
    required_groups = [
        group
        for group in _group_actions_by_dest(
            [a for a in user_actions if bool(a.required)]
        )
        if group
    ]
    optional_groups = [
        group
        for group in _group_actions_by_dest(
            [a for a in user_actions if not bool(a.required)]
        )
        if group
    ]

    cmd_tokens: List[str] = [selected_name]
    selected_values: Dict[str, Any] = {}

    if required_groups:
        print("Required arguments:")
        for group in required_groups:
            action = group[0]
            opt = _choose_opt_name(action)
            info = _action_help_text(action)
            meta = _action_meta_text(action)
            prompt = f"  {opt} ({info}"
            if meta:
                prompt += f"; {meta}"
            prompt += ") > "
            while True:
                value = input(prompt).strip()
                if not value:
                    print("    this argument is required")
                    continue
                args_from_value = _build_args_from_action(action, value)
                if not args_from_value:
                    print("    invalid value, try again")
                    continue
                cmd_tokens.extend(args_from_value)
                selected_values[str(action.dest)] = _coerce_value_for_action(
                    action, value
                )
                break

    if optional_groups and _prompt_bool("Configure optional arguments?", default=False):
        print("\nOptional arguments (press Enter to skip):")
        for group in optional_groups:
            action = group[0]
            skip_reason = _nsys_panel_skip_reason(selected_name, group, selected_values)
            if skip_reason:
                opt = _choose_opt_name(action)
                print(f"  {opt} [skip] {skip_reason}")
                continue

            bool_group = all(_is_bool_action(item) for item in group)
            if bool_group:
                default_bool = _bool_group_default(group)
                enable_opt = _bool_group_option_for_target(group, True)
                disable_opt = _bool_group_option_for_target(group, False)
                info_items = sorted(
                    {
                        _action_help_text(item)
                        for item in group
                        if _action_help_text(item)
                    }
                )
                info = " | ".join(info_items) if info_items else "(no description)"
                default_text = "on" if default_bool else "off"
                toggle_hint = "on/off/skip"
                shown_opt = enable_opt or disable_opt or _choose_opt_name(action)
                prompt = f"  {shown_opt} ({info}; default={default_text}; input {toggle_hint}) > "
                while True:
                    raw = input(prompt).strip()
                    target = _normalize_bool_text(raw)
                    if raw and target is None:
                        print("    invalid input, use on/off/skip")
                        continue
                    break
                if target is None:
                    selected_values[str(action.dest)] = default_bool
                    continue
                selected_values[str(action.dest)] = bool(target)
                if bool(target) == bool(default_bool):
                    continue
                opt = _bool_group_option_for_target(group, bool(target))
                if opt:
                    cmd_tokens.append(opt)
                continue

            opt = _choose_opt_name(action)
            info = _action_help_text(action)
            meta = _action_meta_text(action)
            prompt = f"  {opt} ({info}"
            if meta:
                prompt += f"; {meta}"
            prompt += ") > "
            value = input(prompt).strip()
            if not value:
                continue
            args_from_value = _build_args_from_action(action, value)
            if not args_from_value:
                print("    invalid value, skipped")
                continue
            cmd_tokens.extend(args_from_value)
            selected_values[str(action.dest)] = _coerce_value_for_action(action, value)

    cmd_display = "myutils-profile " + " ".join(shlex.quote(x) for x in cmd_tokens)
    print("\nGenerated command:")
    print(cmd_display)

    if _prompt_bool("Execute now?", default=False):
        return int(main(cmd_tokens))
    return 0


def cmd_nsys_sql_skill(args: argparse.Namespace) -> int:
    provider = NsysSqliteMetricsProvider(args.sqlite)
    if args.list_skills:
        payload = provider.describe_sql_skills()
        print(
            json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None)
        )
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
    if (
        skill_name in {"kernel_occupancy_estimate", "nvtx_kernel_sm_detail"}
        and occ_arch != "none"
    ):
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

    if (
        skill_name in {"kernel_occupancy_estimate", "nvtx_kernel_sm_detail"}
        and not use_h100_occupancy
    ):
        print(
            "[nsys-sql-skill] note: occupancy_pct_estimate depends on sqlite theoretical occupancy columns; "
            "if absent it will be NULL. Use --occupancy-arch h100 (or auto on H100) to attach "
            "occupancy_pct_h100_estimate.",
            file=sys.stderr,
        )

    payload: Any = rows
    if skill_name == "schema_inspect":
        schema_view = (
            str(getattr(args, "schema_view", "both") or "both").strip().lower()
        )
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


def cmd_ncu_csv_skill(args: argparse.Namespace) -> int:
    engine = NcuCsvSkillEngine(args.csv)
    if args.list_skills:
        payload = engine.describe_skills()
        text = skill_result_to_json(payload, pretty=bool(args.pretty))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"[ncu-csv-skill] wrote: {path}")
        else:
            print(text)
        return 0

    if not args.skill:
        print(
            "[ncu-csv-skill] --skill is required unless --list-skills is set",
            file=sys.stderr,
        )
        return 2

    params = _parse_kv_params(args.param)
    payload = engine.run_skill(str(args.skill), **params)
    text = skill_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[ncu-csv-skill] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_ncu_csv_analyze(args: argparse.Namespace) -> int:
    payload = analyze_ncu_csv(
        args.csv,
        top_k=int(args.top_k),
        metric_like=str(args.metric_like or ""),
        kernel_like=str(args.kernel_like or "%"),
    )
    if str(args.format).lower() in {"md", "markdown"}:
        text = analyze_ncu_to_markdown(payload)
    else:
        text = skill_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[ncu-csv-analyze] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_ncu_report_skill(args: argparse.Namespace) -> int:
    engine = NcuReportSkillEngine(args.report)
    if args.list_skills:
        payload = engine.describe_skills()
        text = report_result_to_json(payload, pretty=bool(args.pretty))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"[ncu-report-skill] wrote: {path}")
        else:
            print(text)
        return 0

    if not args.skill:
        print(
            "[ncu-report-skill] --skill is required unless --list-skills is set",
            file=sys.stderr,
        )
        return 2

    params = _parse_kv_params(args.param)
    payload = engine.run_skill(str(args.skill), **params)
    text = report_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[ncu-report-skill] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_ncu_report_analyze(args: argparse.Namespace) -> int:
    payload = analyze_ncu_report(
        args.report,
        top_k=int(args.top_k),
        metric_like=str(args.metric_like or ""),
        kernel_like=str(args.kernel_like or "%"),
        include_all_metrics=bool(args.include_all_metrics),
        all_metrics_limit=int(args.all_metrics_limit),
    )
    if str(args.format).lower() in {"md", "markdown"}:
        text = analyze_ncu_report_to_markdown(payload)
    else:
        text = report_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[ncu-report-analyze] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_ncu_diagnose(args: argparse.Namespace) -> int:
    payload = diagnose_ncu_report(
        args.report,
        kernel_like=str(args.kernel_like or "%"),
        top_kernels=int(args.top_kernels),
        findings_per_kernel=int(args.findings_per_kernel),
        gpu_name=str(args.gpu or ""),
        include_source=not bool(getattr(args, "no_source", False)),
    )
    if str(args.format).lower() in {"md", "markdown"}:
        text = diagnose_result_to_markdown(payload)
    else:
        text = report_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[ncu-diagnose] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_ncu_diff(args: argparse.Namespace) -> int:
    payload = diff_ncu_reports(
        args.report_a,
        args.report_b,
        kernel_like=str(args.kernel or "%"),
        findings_per_kernel=int(args.findings_per_kernel),
    )
    if str(args.format).lower() in {"md", "markdown"}:
        text = diff_result_to_markdown(payload)
    else:
        text = report_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[ncu-diff] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_ncu_metrics(args: argparse.Namespace) -> int:
    from .ncu.metric_catalog import explain_metric, verify_catalog_coverage
    from .ncu.section_index import build_section_index

    if args.coverage:
        payload = verify_catalog_coverage(str(args.sections_dir or ""))
        print(report_result_to_json(payload, pretty=True))
        return 0

    index = build_section_index(str(args.sections_dir or ""))
    if args.search:
        if index is None:
            print(
                "[ncu-metrics] no Nsight Compute installation found; pass --sections-dir"
            )
            return 1
        hits = index.search(args.search)
        print(f"# {len(hits)} metric(s) matching {args.search!r}\n")
        for entry in sorted(hits, key=lambda e: e.name)[: int(args.limit)]:
            label = f"  [{entry.label}]" if entry.label else ""
            print(f"{entry.name}{label}")
            if args.verbose:
                print(f"    {entry.describe()}")
        return 0

    if args.metric:
        payload = explain_metric(args.metric, args.value)
        print(report_result_to_json(payload, pretty=True))
        return 0

    if index is None:
        print("[ncu-metrics] no Nsight Compute installation found; pass --sections-dir")
        return 1
    print("# Nsight Compute metric index\n")
    print(f"sections dir : {index.sections_dir}")
    print(f"sections     : {len(index.sections)}")
    print(
        f"metrics      : {len(index.metrics)}  ({len(index.in_set('full'))} in --set full)\n"
    )
    print("metrics per hardware unit:")
    for unit, count in index.unit_summary().items():
        print(f"  {unit or '(none)':12s} {count:4d}")
    return 0


def cmd_nccl_inspector_skill(args: argparse.Namespace) -> int:
    engine = NcclInspectorSkillEngine(
        args.input, prometheus_path=str(args.prometheus_path or "")
    )
    if args.list_skills:
        payload = engine.describe_skills()
        text = inspector_result_to_json(payload, pretty=bool(args.pretty))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"[nccl-inspector-skill] wrote: {path}")
        else:
            print(text)
        return 0

    if not args.skill:
        print(
            "[nccl-inspector-skill] --skill is required unless --list-skills is set",
            file=sys.stderr,
        )
        return 2

    params = _parse_kv_params(args.param)
    payload = engine.run_skill(str(args.skill), **params)
    text = inspector_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[nccl-inspector-skill] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_nccl_inspector_analyze(args: argparse.Namespace) -> int:
    payload = analyze_nccl_inspector(
        args.input,
        prometheus_path=str(args.prometheus_path or ""),
        top_k=int(args.top_k),
        op_like=str(args.op_like or "%"),
        comm_like=str(args.comm_like or "%"),
        min_msg_size_bytes=int(args.min_msg_size_bytes),
    )
    if str(args.format).lower() in {"md", "markdown"}:
        text = analyze_nccl_inspector_to_markdown(payload)
    else:
        text = inspector_result_to_json(payload, pretty=bool(args.pretty))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[nccl-inspector-analyze] wrote: {path}")
    else:
        print(text)
    return 0


def cmd_nsys_module_kernel_compare(args: argparse.Namespace) -> int:
    def _load_rows_from_sqlite(sqlite_path: str) -> List[Dict[str, object]]:
        provider = NsysSqliteMetricsProvider(str(sqlite_path))
        skills = set(provider.list_sql_skills())
        if "nvtx_kernel_sm_detail" not in skills:
            raise ValueError(
                f"sqlite '{sqlite_path}' does not support skill 'nvtx_kernel_sm_detail' under current schema"
            )

        exec_params: Dict[str, Any] = {
            "nvtx_text": str(args.nvtx_text or "").strip() or "%",
            "device_id": int(args.device_id),
            "limit": int(args.sqlite_limit),
        }
        occ_arch = (
            str(getattr(args, "occupancy_arch", "auto") or "auto").strip().lower()
        )
        use_h100_occupancy = False
        if occ_arch == "h100":
            use_h100_occupancy = True
        elif occ_arch == "auto":
            gpu_name = _detect_gpu_name_from_sqlite(str(sqlite_path))
            use_h100_occupancy = "h100" in str(gpu_name).lower()

        if use_h100_occupancy:
            conn = sqlite3.connect(str(sqlite_path))
            conn.row_factory = sqlite3.Row
            try:
                engine = NsysSqlSkillEngine(conn)
                rows = engine.execute_nvtx_kernel_sm_detail_h100(**exec_params)
            finally:
                conn.close()
            return list(rows or [])
        return list(
            provider.run_sql_skill("nvtx_kernel_sm_detail", **exec_params) or []
        )

    has_json_pair = bool(str(args.base_json or "").strip()) and bool(
        str(args.target_json or "").strip()
    )
    has_sqlite_pair = bool(str(args.base_sqlite or "").strip()) and bool(
        str(args.target_sqlite or "").strip()
    )
    has_any_json = bool(str(args.base_json or "").strip()) or bool(
        str(args.target_json or "").strip()
    )
    has_any_sqlite = bool(str(args.base_sqlite or "").strip()) or bool(
        str(args.target_sqlite or "").strip()
    )
    if has_any_json and has_any_sqlite:
        print(
            "[nsys-module-kernel-compare] choose one input mode: JSON pair or sqlite pair, do not mix.",
            file=sys.stderr,
        )
        return 2
    if has_any_json and not has_json_pair:
        print(
            "[nsys-module-kernel-compare] JSON mode requires both --base-json and --target-json.",
            file=sys.stderr,
        )
        return 2
    if has_any_sqlite and not has_sqlite_pair:
        print(
            "[nsys-module-kernel-compare] sqlite mode requires both --base-sqlite and --target-sqlite.",
            file=sys.stderr,
        )
        return 2
    if not has_json_pair and not has_sqlite_pair:
        print(
            "[nsys-module-kernel-compare] provide either (--base-json, --target-json) "
            "or (--base-sqlite, --target-sqlite).",
            file=sys.stderr,
        )
        return 2

    stream_ids = [int(v) for v in (args.stream_id or [])]
    if has_sqlite_pair:
        try:
            base_rows = _load_rows_from_sqlite(str(args.base_sqlite))
            target_rows = _load_rows_from_sqlite(str(args.target_sqlite))
        except Exception as exc:
            print(
                f"[nsys-module-kernel-compare] failed to load sqlite rows: {exc}",
                file=sys.stderr,
            )
            return 2
        payload = compare_module_kernel_rows(
            base_rows=base_rows,
            target_rows=target_rows,
            base_label=str(args.base_label or "base"),
            target_label=str(args.target_label or "target"),
            base_source_path=str(args.base_sqlite),
            target_source_path=str(args.target_sqlite),
            nvtx_text=str(args.nvtx_text or ""),
            nvtx_index=int(args.nvtx_index),
            device_id=int(args.device_id),
            stream_ids=stream_ids,
            top_k=int(args.top_k),
            timeline_limit_per_stream=int(args.timeline_limit_per_stream),
        )
    else:
        payload = compare_module_kernel_json(
            base_json=str(args.base_json),
            target_json=str(args.target_json),
            base_label=str(args.base_label or "base"),
            target_label=str(args.target_label or "target"),
            nvtx_text=str(args.nvtx_text or ""),
            nvtx_index=int(args.nvtx_index),
            device_id=int(args.device_id),
            stream_ids=stream_ids,
            top_k=int(args.top_k),
            timeline_limit_per_stream=int(args.timeline_limit_per_stream),
        )
    fmt = str(args.format or "json").strip().lower()
    if fmt in {"md", "markdown"}:
        text = module_kernel_compare_to_markdown(payload)
    elif fmt == "html":
        text = module_kernel_compare_to_html(payload)
    else:
        text = json.dumps(
            payload, ensure_ascii=False, indent=2 if args.pretty else None
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"[nsys-module-kernel-compare] wrote: {path}")
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

    analyze = sub.add_parser(
        "analyze", help="analyze events JSONL and generate reports"
    )
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
    diff.add_argument(
        "--markdown", default="", help="optional diff markdown output path"
    )
    diff.set_defaults(func=cmd_diff)

    list_providers = sub.add_parser(
        "list-providers", help="list available provider types"
    )
    list_providers.set_defaults(func=cmd_list_providers)

    trace = sub.add_parser(
        "trace", help="convert metrics events jsonl to Chrome Trace JSON"
    )
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

    nsys_panel = sub.add_parser(
        "nsys-panel",
        help="interactive panel to choose nsys command and fill args",
        description=(
            "Interactive panel for nsys commands.\n"
            "You can pick one nsys subcommand, see its description, fill required/optional args, "
            "then choose to run it immediately or only print the generated command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    nsys_panel.set_defaults(func=cmd_nsys_panel)

    nsys_sql = sub.add_parser(
        "nsys-sql-skill", help="run built-in Nsight SQLite SQL skills"
    )
    nsys_sql.add_argument("--sqlite", required=True, help="nsys exported sqlite path")
    nsys_sql.add_argument(
        "--list-skills",
        action="store_true",
        help="list available skills and parameters",
    )
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
    nsys_sql.add_argument(
        "--pretty", action="store_true", help="pretty-print JSON output"
    )
    nsys_sql.set_defaults(func=cmd_nsys_sql_skill)

    nsys_export = sub.add_parser(
        "nsys-export", help="export sqlite kernel timeline rows to json/csv"
    )
    nsys_export.add_argument(
        "--sqlite", required=True, help="nsys exported sqlite path"
    )
    nsys_export.add_argument("--output", required=True, help="output file path")
    nsys_export.add_argument("--format", default="json", choices=["json", "csv"])
    nsys_export.add_argument("--device-id", type=int, default=-1)
    nsys_export.add_argument("--start-ns", type=int, default=-1)
    nsys_export.add_argument("--end-ns", type=int, default=-1)
    nsys_export.add_argument("--limit", type=int, default=500000)
    nsys_export.add_argument("--attach-iteration", action="store_true")
    nsys_export.add_argument("--iteration-marker", default="sample_0")
    nsys_export.set_defaults(func=cmd_nsys_export)

    nsys_analyze = sub.add_parser(
        "nsys-analyze",
        help="run nsys-oriented summary/overlap/nccl/iterations/mfu analysis",
    )
    nsys_analyze.add_argument(
        "--sqlite", required=True, help="nsys exported sqlite path"
    )
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
            "matching NVTX ranges. No implicit wildcard is added; use %%/_ (or *) explicitly "
            "for fuzzy matching (e.g. '%%step_%%'). "
            "When set, start_ns/end_ns are derived automatically from matching ranges "
            "unless --start-ns/--end-ns are explicitly provided."
        ),
    )
    nsys_analyze.add_argument("--model-flops-per-step", type=float, default=None)
    nsys_analyze.add_argument("--peak-tflops", type=float, default=None)
    nsys_analyze.add_argument("--peak-precision", default="fp16")
    nsys_analyze.add_argument(
        "--format", default="json", choices=["json", "markdown", "md"]
    )
    nsys_analyze.add_argument("--output", default="")
    nsys_analyze.add_argument("--pretty", action="store_true")
    nsys_analyze.set_defaults(func=cmd_nsys_analyze)

    nsys_diff = sub.add_parser(
        "nsys-diff", help="diff two nsys sqlite profiles by kernel/nvtx aggregates"
    )
    nsys_diff.add_argument("--before-sqlite", required=True)
    nsys_diff.add_argument("--after-sqlite", required=True)
    nsys_diff.add_argument("--device-id", type=int, default=-1)
    nsys_diff.add_argument("--start-ns", type=int, default=-1)
    nsys_diff.add_argument("--end-ns", type=int, default=-1)
    nsys_diff.add_argument("--top-k", type=int, default=20)
    nsys_diff.add_argument(
        "--format", default="json", choices=["json", "markdown", "md"]
    )
    nsys_diff.add_argument("--output", default="")
    nsys_diff.add_argument("--pretty", action="store_true")
    nsys_diff.set_defaults(func=cmd_nsys_diff)

    ncu_csv_skill = sub.add_parser(
        "ncu-csv-skill", help="run built-in ncu CSV parsing skills"
    )
    ncu_csv_skill.add_argument("--csv", required=True, help="ncu csv path")
    ncu_csv_skill.add_argument(
        "--list-skills", action="store_true", help="list skills and params"
    )
    ncu_csv_skill.add_argument("--skill", default="", help="skill name")
    ncu_csv_skill.add_argument(
        "--param",
        action="append",
        default=[],
        help="skill parameter in key=value format (can repeat)",
    )
    ncu_csv_skill.add_argument("--output", default="", help="optional json output path")
    ncu_csv_skill.add_argument(
        "--pretty", action="store_true", help="pretty-print json output"
    )
    ncu_csv_skill.set_defaults(func=cmd_ncu_csv_skill)

    ncu_csv_analyze = sub.add_parser(
        "ncu-csv-analyze", help="run summarized analysis for ncu csv"
    )
    ncu_csv_analyze.add_argument("--csv", required=True, help="ncu csv path")
    ncu_csv_analyze.add_argument(
        "--metric-like", default="", help="metric LIKE pattern (%%/_/*)"
    )
    ncu_csv_analyze.add_argument(
        "--kernel-like", default="%", help="kernel LIKE pattern (%%/_/*)"
    )
    ncu_csv_analyze.add_argument("--top-k", type=int, default=20)
    ncu_csv_analyze.add_argument(
        "--format", default="json", choices=["json", "markdown", "md"]
    )
    ncu_csv_analyze.add_argument("--output", default="")
    ncu_csv_analyze.add_argument("--pretty", action="store_true")
    ncu_csv_analyze.set_defaults(func=cmd_ncu_csv_analyze)

    ncu_report_skill = sub.add_parser(
        "ncu-report-skill", help="run built-in ncu .ncu-rep parsing skills"
    )
    ncu_report_skill.add_argument(
        "--report", required=True, help="ncu report path (.ncu-rep)"
    )
    ncu_report_skill.add_argument(
        "--list-skills", action="store_true", help="list skills and params"
    )
    ncu_report_skill.add_argument("--skill", default="", help="skill name")
    ncu_report_skill.add_argument(
        "--param",
        action="append",
        default=[],
        help="skill parameter in key=value format (can repeat)",
    )
    ncu_report_skill.add_argument(
        "--output", default="", help="optional json output path"
    )
    ncu_report_skill.add_argument(
        "--pretty", action="store_true", help="pretty-print json output"
    )
    ncu_report_skill.set_defaults(func=cmd_ncu_report_skill)

    ncu_metrics = sub.add_parser(
        "ncu-metrics",
        help="explain any ncu metric, search the index, or report catalog coverage",
    )
    ncu_metrics.add_argument("--metric", default="", help="explain one metric by name")
    ncu_metrics.add_argument(
        "--value",
        type=float,
        default=None,
        help="a measured value, to get a verdict where a rule exists",
    )
    ncu_metrics.add_argument(
        "--search", default="", help="regex over metric names and labels"
    )
    ncu_metrics.add_argument(
        "--coverage",
        action="store_true",
        help="report how much of --set full the curated catalog covers",
    )
    ncu_metrics.add_argument(
        "--sections-dir",
        default="",
        help="Nsight Compute sections directory (auto-detected by default)",
    )
    ncu_metrics.add_argument("--limit", type=int, default=60)
    ncu_metrics.add_argument("--verbose", action="store_true")
    ncu_metrics.set_defaults(func=cmd_ncu_metrics)

    ncu_diagnose = sub.add_parser(
        "ncu-diagnose",
        help="diagnose every kernel in a .ncu-rep: bottleneck class, stalls, roofline, fixes",
    )
    ncu_diagnose.add_argument(
        "--report", required=True, help="ncu report path (.ncu-rep)"
    )
    ncu_diagnose.add_argument(
        "--kernel-like", default="%", help="kernel LIKE pattern (%%/_/*)"
    )
    ncu_diagnose.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="how many kernels to report, ranked by duration",
    )
    ncu_diagnose.add_argument("--findings-per-kernel", type=int, default=8)
    ncu_diagnose.add_argument(
        "--gpu",
        default="",
        help="GPU name (e.g. 'H100 SXM5') to unlock absolute roofline ceilings",
    )
    ncu_diagnose.add_argument(
        "--no-source",
        action="store_true",
        help="skip source-line attribution (it re-reads the report for PC samples)",
    )
    ncu_diagnose.add_argument(
        "--format", default="md", choices=["json", "md", "markdown"]
    )
    ncu_diagnose.add_argument("--output", default="")
    ncu_diagnose.add_argument("--pretty", action="store_true")
    ncu_diagnose.set_defaults(func=cmd_ncu_diagnose)

    ncu_diff = sub.add_parser(
        "ncu-diff",
        help="A/B diff of two .ncu-rep reports: clock guard, per-axis metric deltas, findings that appeared/disappeared",
    )
    ncu_diff.add_argument(
        "--report-a", required=True, help="baseline ncu report path (.ncu-rep)"
    )
    ncu_diff.add_argument(
        "--report-b", required=True, help="candidate ncu report path (.ncu-rep)"
    )
    ncu_diff.add_argument(
        "--kernel", default="%", help="kernel LIKE pattern (%%/_/*) to diff"
    )
    ncu_diff.add_argument(
        "--findings-per-kernel",
        type=int,
        default=24,
        help="findings kept per kernel per side before the findings diff",
    )
    ncu_diff.add_argument("--format", default="md", choices=["json", "md", "markdown"])
    ncu_diff.add_argument("--output", default="")
    ncu_diff.add_argument("--pretty", action="store_true")
    ncu_diff.set_defaults(func=cmd_ncu_diff)

    ncu_report_analyze = sub.add_parser(
        "ncu-report-analyze",
        help="run summarized analysis for ncu .ncu-rep (includes per-metric stats)",
    )
    ncu_report_analyze.add_argument(
        "--report", required=True, help="ncu report path (.ncu-rep)"
    )
    ncu_report_analyze.add_argument(
        "--metric-like", default="", help="metric LIKE pattern (%%/_/*)"
    )
    ncu_report_analyze.add_argument(
        "--kernel-like", default="%", help="kernel LIKE pattern (%%/_/*)"
    )
    ncu_report_analyze.add_argument("--top-k", type=int, default=20)
    ncu_report_analyze.add_argument(
        "--include-all-metrics",
        action="store_true",
        default=True,
        help="include all parsed metric rows in output payload",
    )
    ncu_report_analyze.add_argument(
        "--no-include-all-metrics",
        dest="include_all_metrics",
        action="store_false",
        help="omit all_metrics rows and keep aggregated analysis only",
    )
    ncu_report_analyze.add_argument(
        "--all-metrics-limit",
        type=int,
        default=20000,
        help="max rows in all_metrics payload",
    )
    ncu_report_analyze.add_argument(
        "--format", default="json", choices=["json", "markdown", "md"]
    )
    ncu_report_analyze.add_argument("--output", default="")
    ncu_report_analyze.add_argument("--pretty", action="store_true")
    ncu_report_analyze.set_defaults(func=cmd_ncu_report_analyze)

    nccl_skill = sub.add_parser(
        "nccl-inspector-skill",
        help="run built-in NCCL Inspector JSON/Prometheus parsing skills",
    )
    nccl_skill.add_argument(
        "--input",
        required=True,
        help="NCCL Inspector JSON/JSONL file or dump directory",
    )
    nccl_skill.add_argument(
        "--prometheus-path",
        default="",
        help="optional NCCL Inspector Prometheus textfile or directory",
    )
    nccl_skill.add_argument(
        "--list-skills", action="store_true", help="list skills and params"
    )
    nccl_skill.add_argument("--skill", default="", help="skill name")
    nccl_skill.add_argument(
        "--param",
        action="append",
        default=[],
        help="skill parameter in key=value format (can repeat)",
    )
    nccl_skill.add_argument("--output", default="", help="optional json output path")
    nccl_skill.add_argument(
        "--pretty", action="store_true", help="pretty-print json output"
    )
    nccl_skill.set_defaults(func=cmd_nccl_inspector_skill)

    nccl_analyze = sub.add_parser(
        "nccl-inspector-analyze",
        help="summarize NCCL Inspector JSON dumps and optional Prometheus textfiles",
    )
    nccl_analyze.add_argument(
        "--input",
        required=True,
        help="NCCL Inspector JSON/JSONL file or dump directory",
    )
    nccl_analyze.add_argument(
        "--prometheus-path",
        default="",
        help="optional NCCL Inspector Prometheus textfile or directory",
    )
    nccl_analyze.add_argument(
        "--op-like", default="%", help="operation LIKE pattern (%%/_/*)"
    )
    nccl_analyze.add_argument(
        "--comm-like", default="%", help="communicator name LIKE pattern (%%/_/*)"
    )
    nccl_analyze.add_argument("--min-msg-size-bytes", type=int, default=0)
    nccl_analyze.add_argument("--top-k", type=int, default=20)
    nccl_analyze.add_argument(
        "--format", default="json", choices=["json", "markdown", "md"]
    )
    nccl_analyze.add_argument("--output", default="")
    nccl_analyze.add_argument("--pretty", action="store_true")
    nccl_analyze.set_defaults(func=cmd_nccl_inspector_analyze)

    nsys_module_kernel_compare = sub.add_parser(
        "nsys-module-kernel-compare",
        help=(
            "compare two profiles for one module/NVTX scope (stream timeline + resource deltas). "
            "Supports JSON exports or direct sqlite inputs."
        ),
    )
    nsys_module_kernel_compare.add_argument(
        "--base-json", default="", help="baseline kernel JSON path"
    )
    nsys_module_kernel_compare.add_argument(
        "--target-json", default="", help="target kernel JSON path"
    )
    nsys_module_kernel_compare.add_argument(
        "--base-sqlite", default="", help="baseline nsys sqlite path"
    )
    nsys_module_kernel_compare.add_argument(
        "--target-sqlite", default="", help="target nsys sqlite path"
    )
    nsys_module_kernel_compare.add_argument(
        "--base-label", default="base", help="display label for baseline"
    )
    nsys_module_kernel_compare.add_argument(
        "--target-label", default="target", help="display label for target"
    )
    nsys_module_kernel_compare.add_argument(
        "--nvtx-text",
        default="",
        help=(
            "optional SQL-LIKE NVTX filter (% and _ supported; * also accepted). "
            "No implicit wildcard is added. "
            "Empty means no NVTX filter."
        ),
    )
    nsys_module_kernel_compare.add_argument(
        "--nvtx-index",
        type=int,
        default=-1,
        help=(
            "index of matched NVTX scope after start-time sort. "
            "-1 means keep all matched scopes (default: -1)"
        ),
    )
    nsys_module_kernel_compare.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="device filter; -1 means all devices (default: -1)",
    )
    nsys_module_kernel_compare.add_argument(
        "--stream-id",
        action="append",
        type=int,
        default=[],
        help="stream filter, repeatable (default: keep all streams)",
    )
    nsys_module_kernel_compare.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="top delta rows for kernel-duration compare (default: 20)",
    )
    nsys_module_kernel_compare.add_argument(
        "--timeline-limit-per-stream",
        type=int,
        default=40,
        help="timeline sample rows kept per stream for each workload (default: 40)",
    )
    nsys_module_kernel_compare.add_argument(
        "--sqlite-limit",
        type=int,
        default=500000,
        help="max rows fetched from nvtx_kernel_sm_detail per sqlite in sqlite mode (default: 500000)",
    )
    nsys_module_kernel_compare.add_argument(
        "--occupancy-arch",
        default="auto",
        choices=["auto", "h100", "none"],
        help=(
            "sqlite mode only: occupancy attachment policy. "
            "'auto' uses H100 estimate when TARGET_INFO reports H100."
        ),
    )
    nsys_module_kernel_compare.add_argument(
        "--format", default="json", choices=["json", "markdown", "md", "html"]
    )
    nsys_module_kernel_compare.add_argument("--output", default="")
    nsys_module_kernel_compare.add_argument("--pretty", action="store_true")
    nsys_module_kernel_compare.set_defaults(func=cmd_nsys_module_kernel_compare)

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
            "No implicit wildcard is added; use %%/_ (or *) explicitly. "
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
        default=True,
        help=(
            "when --metric-name-like is '%%' (or empty), keep only built-in focus metrics "
            "for attribution (default: enabled)"
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
    nsys_timeline_compare.add_argument(
        "--output", required=True, help="output compare HTML file path"
    )
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
            "No implicit wildcard is added; use %%/_ (or *) explicitly. "
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
        default=True,
        help=(
            "when --metric-name-like is '%%' (or empty), keep only built-in focus metrics "
            "for attribution (default: enabled)"
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
    nsys_iter_overlap.add_argument(
        "--sqlite", required=True, help="nsys exported sqlite path"
    )
    nsys_iter_overlap.add_argument(
        "--iteration-marker",
        default="sample_0",
        help="NVTX text substring that marks iteration boundaries (default: sample_0)",
    )
    nsys_iter_overlap.add_argument(
        "--device-id", type=int, default=-1, help="CUDA device ID, -1 = all"
    )
    nsys_iter_overlap.add_argument(
        "--start-ns",
        type=int,
        default=-1,
        help="global window start ns, -1 = no filter",
    )
    nsys_iter_overlap.add_argument(
        "--end-ns", type=int, default=-1, help="global window end ns, -1 = no filter"
    )
    nsys_iter_overlap.add_argument(
        "--include-nested",
        action="store_true",
        help="include nested NVTX ranges with same marker (default: top-level only)",
    )
    nsys_iter_overlap.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="max iterations to process (default: 2000)",
    )
    nsys_iter_overlap.add_argument(
        "--output", default="", help="optional JSON output path"
    )
    nsys_iter_overlap.add_argument(
        "--pretty", action="store_true", help="pretty-print JSON output"
    )
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
    nsys_iter_outliers.add_argument(
        "--sqlite", required=True, help="nsys exported sqlite path"
    )
    nsys_iter_outliers.add_argument(
        "--iteration-marker",
        default="sample_0",
        help="NVTX text substring that marks iteration boundaries (default: sample_0)",
    )
    nsys_iter_outliers.add_argument(
        "--device-id", type=int, default=-1, help="CUDA device ID, -1 = all"
    )
    nsys_iter_outliers.add_argument(
        "--sigma",
        type=float,
        default=2.0,
        help="flag iterations deviating more than this many σ from median (default: 2.0)",
    )
    nsys_iter_outliers.add_argument(
        "--limit", type=int, default=2000, help="max iterations to scan (default: 2000)"
    )
    nsys_iter_outliers.add_argument(
        "--output", default="", help="optional JSON output path"
    )
    nsys_iter_outliers.add_argument(
        "--pretty", action="store_true", help="pretty-print JSON output"
    )
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


def entry_nsys_panel() -> int:
    return _run_nsys_alias("nsys-panel")


def entry_nsys_export() -> int:
    return _run_nsys_alias("nsys-export")


def entry_nsys_analyze() -> int:
    return _run_nsys_alias("nsys-analyze")


def entry_nsys_diff() -> int:
    return _run_nsys_alias("nsys-diff")


def entry_nsys_module_kernel_compare() -> int:
    return _run_nsys_alias("nsys-module-kernel-compare")


def entry_nsys_timeline_html() -> int:
    return _run_nsys_alias("nsys-timeline-html")


def entry_nsys_iter_overlap() -> int:
    return _run_nsys_alias("nsys-iter-overlap")


def entry_nsys_iter_outliers() -> int:
    return _run_nsys_alias("nsys-iter-outliers")


def entry_ncu_csv_skill() -> int:
    return _run_nsys_alias("ncu-csv-skill")


def entry_ncu_csv_analyze() -> int:
    return _run_nsys_alias("ncu-csv-analyze")


def entry_ncu_report_skill() -> int:
    return _run_nsys_alias("ncu-report-skill")


def entry_ncu_report_analyze() -> int:
    return _run_nsys_alias("ncu-report-analyze")


def entry_ncu_metrics() -> int:
    return _run_nsys_alias("ncu-metrics")


def entry_ncu_diagnose() -> int:
    return _run_nsys_alias("ncu-diagnose")


def entry_ncu_diff() -> int:
    return _run_nsys_alias("ncu-diff")


def entry_nccl_inspector_skill() -> int:
    return _run_nsys_alias("nccl-inspector-skill")


def entry_nccl_inspector_analyze() -> int:
    return _run_nsys_alias("nccl-inspector-analyze")


if __name__ == "__main__":
    raise SystemExit(main())
