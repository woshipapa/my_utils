from __future__ import annotations

import json
import sqlite3
import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

from .nsys_auto_analysis import build_comprehensive_analysis, comprehensive_to_markdown
from .nsys_mfu import compute_mfu_single, infer_peak_tflops
from .nsys_schema_adapter import NsightSchema
from .nsys_sql_skills import NsysSqlSkillEngine


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _first_gpu_name_from_conn(conn: sqlite3.Connection) -> str:
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        }
        if "TARGET_INFO_GPU" not in tables:
            return ""
        row = conn.execute("SELECT name FROM TARGET_INFO_GPU ORDER BY id LIMIT 1;").fetchone()
        return str(row["name"] or "").strip() if row else ""
    except Exception:
        return ""


def _resolve_nvtx_text(
    conn: sqlite3.Connection,
    schema: NsightSchema,
    nvtx_text: str,
    *,
    limit: int = 10000,
) -> Dict[str, object]:
    """Resolve a NVTX text LIKE pattern to a time window and individual range list.

    Returns a dict with:
        pattern          – the original pattern string
        matched_ranges   – [{nvtx_text, start_ns, end_ns, duration_ms}, ...]
        count            – number of matched ranges
        window_start_ns  – min(start_ns) across all matches, or -1 if none
        window_end_ns    – max(end_ns)   across all matches, or -1 if none
        total_duration_ms – sum of individual range durations
        warning          – non-empty string if no ranges matched
    """
    nvtx_table = schema.nvtx_table
    if not nvtx_table or not schema.table_exists(nvtx_table):
        return {
            "pattern": nvtx_text,
            "matched_ranges": [],
            "count": 0,
            "window_start_ns": -1,
            "window_end_ns": -1,
            "total_duration_ms": 0.0,
            "warning": "NVTX table not found in sqlite.",
        }

    # Build text expression (handle textId / inline text)
    nvtx_cols = set(schema.columns(nvtx_table))
    text_expr = "n.text"
    nvtx_join = ""
    if schema.string_table:
        if "textId" in nvtx_cols:
            nvtx_join = f" LEFT JOIN {schema.string_table} s_nsc ON n.textId = s_nsc.id "
            text_expr = "COALESCE(n.text, s_nsc.value)"
        elif "nameId" in nvtx_cols:
            nvtx_join = f" LEFT JOIN {schema.string_table} s_nsc ON n.nameId = s_nsc.id "
            text_expr = "COALESCE(n.text, s_nsc.value)"

    start_col = "start" if "start" in nvtx_cols else None
    end_col = "end" if "end" in nvtx_cols else None
    if not start_col or not end_col:
        return {
            "pattern": nvtx_text,
            "matched_ranges": [],
            "count": 0,
            "window_start_ns": -1,
            "window_end_ns": -1,
            "total_duration_ms": 0.0,
            "warning": "NVTX table missing start/end columns.",
        }

    like_pattern = str(nvtx_text).strip()
    if "%" not in like_pattern:
        like_pattern = f"%{like_pattern}%"

    sql = (
        f"SELECT {text_expr} AS nvtx_text, "
        f"n.{start_col} AS start_ns, "
        f"n.[{end_col}] AS end_ns, "
        f"ROUND((n.[{end_col}] - n.{start_col}) / 1e6, 3) AS duration_ms "
        f"FROM {nvtx_table} n {nvtx_join} "
        f"WHERE {text_expr} IS NOT NULL "
        f"AND n.[{end_col}] > n.{start_col} "
        f"AND {text_expr} LIKE ? "
        f"ORDER BY n.{start_col} ASC "
        f"LIMIT ?"
    )
    try:
        rows = [dict(r) for r in conn.execute(sql, (like_pattern, int(limit))).fetchall()]
    except Exception as exc:
        return {
            "pattern": nvtx_text,
            "matched_ranges": [],
            "count": 0,
            "window_start_ns": -1,
            "window_end_ns": -1,
            "total_duration_ms": 0.0,
            "warning": f"NVTX query failed: {exc}",
        }

    if not rows:
        return {
            "pattern": nvtx_text,
            "matched_ranges": [],
            "count": 0,
            "window_start_ns": -1,
            "window_end_ns": -1,
            "total_duration_ms": 0.0,
            "warning": f"No NVTX ranges matched pattern '{like_pattern}'.",
        }

    window_start = min(int(r["start_ns"]) for r in rows)
    window_end   = max(int(r["end_ns"])   for r in rows)
    total_dur    = round(sum(float(r.get("duration_ms") or 0) for r in rows), 3)

    return {
        "pattern": nvtx_text,
        "matched_ranges": rows,
        "count": len(rows),
        "window_start_ns": window_start,
        "window_end_ns": window_end,
        "total_duration_ms": total_dur,
        "warning": "",
    }


def _safe_skill(engine: "NsysSqlSkillEngine", skill_name: str, warnings_out: List[str], **kwargs) -> List[Dict[str, object]]:
    """Execute a skill and return [] on failure, appending a warning instead of crashing."""
    skill = engine.get_skill(skill_name)
    if skill is None:
        return []
    try:
        return engine.execute(skill_name, **kwargs)
    except Exception as exc:
        warnings_out.append(f"[SKILL_ERROR] {skill_name} failed: {exc}")
        return []


def _resolve_nvtx_kernel_span(
    engine: "NsysSqlSkillEngine",
    *,
    nvtx_text: str,
    device_id: int,
    limit: int,
) -> Dict[str, object]:
    """Resolve attributed GPU kernel span for NVTX text via correlationId mapping."""
    like_pattern = str(nvtx_text or "").strip()
    if not like_pattern:
        return {
            "pattern": "",
            "count": 0,
            "kernel_start_ns": -1,
            "kernel_end_ns": -1,
            "warning": "empty nvtx_text pattern",
        }
    if "%" not in like_pattern:
        like_pattern = f"%{like_pattern}%"

    skill = engine.get_skill("nvtx_kernel_sm_detail")
    if skill is None:
        return {
            "pattern": like_pattern,
            "count": 0,
            "kernel_start_ns": -1,
            "kernel_end_ns": -1,
            "warning": "skill nvtx_kernel_sm_detail unavailable",
        }

    try:
        rows = engine.execute(
            "nvtx_kernel_sm_detail",
            nvtx_text=like_pattern,
            device_id=int(device_id),
            limit=int(limit),
        )
    except Exception as exc:
        return {
            "pattern": like_pattern,
            "count": 0,
            "kernel_start_ns": -1,
            "kernel_end_ns": -1,
            "warning": f"nvtx_kernel_sm_detail failed: {exc}",
        }

    starts: List[int] = []
    ends: List[int] = []
    for row in rows:
        try:
            ks = int(row.get("kernel_start_ns", row.get("start_ns")) or -1)
            ke = int(row.get("kernel_end_ns", row.get("end_ns")) or -1)
        except Exception:
            continue
        if ks < 0 or ke <= ks:
            continue
        starts.append(ks)
        ends.append(ke)

    if not starts:
        return {
            "pattern": like_pattern,
            "count": 0,
            "kernel_start_ns": -1,
            "kernel_end_ns": -1,
            "warning": "no attributed kernels found in nvtx_kernel_sm_detail",
        }

    return {
        "pattern": like_pattern,
        "count": len(starts),
        "kernel_start_ns": int(min(starts)),
        "kernel_end_ns": int(max(ends)),
        "warning": "",
    }


class _Phase:
    """Lightweight context manager that times a named analysis phase.

    On exit it appends ``{name: elapsed_ms}`` to *timings_out* and optionally
    calls *progress_cb* with a human-readable message.
    """

    def __init__(
        self,
        name: str,
        timings_out: List[Dict[str, float]],
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._name = name
        self._timings = timings_out
        self._cb = progress_cb
        self._t0: float = 0.0

    def __enter__(self) -> "_Phase":
        if self._cb:
            self._cb(f"  starting: {self._name}")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed_ms = round((time.perf_counter() - self._t0) * 1000, 1)
        self._timings.append({"phase": self._name, "elapsed_ms": elapsed_ms})
        if self._cb:
            self._cb(f"  done:     {self._name}  [{elapsed_ms} ms]")


def analyze_nsys_sqlite(
    sqlite_path: str,
    *,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    nvtx_text: Optional[str] = None,
    top_k: int = 10,
    iteration_marker: str = "sample_0",
    model_flops_per_step: Optional[float] = None,
    peak_tflops: Optional[float] = None,
    peak_precision: str = "fp16",
    limit: int = 500000,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Run the full nsys analysis pipeline and return a result dict.

    Parameters
    ----------
    progress_cb:
        Optional callback called at the start and end of every analysis phase.
        Receives a human-readable string like
        ``"  done: summarize_gpu_kernels  [123.4 ms]"``.
        Pass ``lambda msg: print(msg, file=sys.stderr)`` for CLI progress output.
    """
    phase_timings: List[Dict[str, float]] = []

    def _phase(name: str) -> _Phase:
        return _Phase(name, phase_timings, progress_cb)

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        with _phase("schema_and_nvtx_scope"):
            schema_obj = NsightSchema(conn)
            engine = NsysSqlSkillEngine(conn)
            explicit_start_given = int(start_ns) >= 0
            explicit_end_given = int(end_ns) >= 0

            tables = list(schema_obj.tables)
            schema: Dict[str, object] = {
                "sqlite_path": sqlite_path,
                "exists": True,
                "table_count": len(tables),
                "tables": tables,
                "columns": {t: schema_obj.columns(t) for t in tables},
                "schema_meta": dict(schema_obj.meta),
                "canonical_tables": dict(schema_obj.summary().get("canonical_tables", {})),
                "version_info": {
                    "exporter_version": schema_obj.version.exporter_version,
                    "export_schema_version": schema_obj.version.export_schema_version,
                    "adapter_family": schema_obj.version.adapter_family,
                    "known_version": schema_obj.version.known_version,
                },
            }

            # --- NVTX scope resolution ---
            # When nvtx_text is given, resolve matching NVTX ranges and use their
            # union time window to restrict ALL subsequent analyses.  Explicit
            # start_ns / end_ns from the caller take priority over the resolved window.
            nvtx_text_info: Optional[Dict[str, object]] = None
            nvtx_kernel_span_info: Optional[Dict[str, object]] = None
            if nvtx_text:
                nvtx_text_info = _resolve_nvtx_text(
                    conn, schema_obj, nvtx_text, limit=limit
                )
                if nvtx_text_info.get("warning"):
                    pass  # warn later via warnings list; don't abort
                else:
                    # Override time window only when caller did NOT supply explicit bounds
                    if start_ns < 0 and nvtx_text_info["window_start_ns"] >= 0:
                        start_ns = int(nvtx_text_info["window_start_ns"])
                    if end_ns < 0 and nvtx_text_info["window_end_ns"] >= 0:
                        end_ns = int(nvtx_text_info["window_end_ns"])
                    # Extend analysis window by attributed GPU kernel execution span
                    # (NVTX -> Runtime -> correlationId -> GPU kernel timestamps).
                    nvtx_kernel_span_info = _resolve_nvtx_kernel_span(
                        engine,
                        nvtx_text=str(nvtx_text),
                        device_id=int(device_id),
                        limit=int(limit),
                    )
                    if not str(nvtx_kernel_span_info.get("warning") or ""):
                        k_start = int(nvtx_kernel_span_info.get("kernel_start_ns") or -1)
                        k_end = int(nvtx_kernel_span_info.get("kernel_end_ns") or -1)
                        if k_start >= 0 and not explicit_start_given:
                            start_ns = k_start if int(start_ns) < 0 else min(int(start_ns), k_start)
                        if k_end >= 0 and not explicit_end_given:
                            end_ns = k_end if int(end_ns) < 0 else max(int(end_ns), k_end)

        with _phase("summarize_gpu_kernels"):
            summary = engine.summarize_gpu_kernels(
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                top_k=top_k,
                limit=limit,
            )
            # top_kernels is already computed inside summarize_gpu_kernels(); reuse it.
            top_kernels = list((summary or {}).get("top_kernels") or [])

        with _phase("compute_comm_overlap"):
            overlap = engine.analyze_compute_comm_overlap(
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=limit,
            )

        with _phase("nccl_breakdown"):
            nccl_breakdown = engine.execute(
                "nccl_breakdown",
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=max(int(top_k), 1),
            )

        with _phase("detect_iterations"):
            iterations = engine.detect_iterations(
                marker=iteration_marker,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                top_level_only=True,
                limit=10000,
            )

        warnings: List[str] = []

        # Surface NVTX scope warnings early
        if nvtx_text_info:
            scope_warn = str(nvtx_text_info.get("warning") or "")
            if scope_warn:
                warnings.append(f"[NVTX_SCOPE] {scope_warn}")
            else:
                n_ranges = int(nvtx_text_info.get("count") or 0)
                w_start = int(nvtx_text_info.get("window_start_ns") or 0)
                w_end   = int(nvtx_text_info.get("window_end_ns") or 0)
                warnings.append(
                    f"[NVTX_SCOPE] Scoped to pattern '{nvtx_text}': "
                    f"{n_ranges} range(s) matched, "
                    f"window [{round(w_start/1e6,1)} ms – {round(w_end/1e6,1)} ms] "
                    f"(span {round((w_end-w_start)/1e6,1)} ms)."
                )

        if nvtx_kernel_span_info:
            span_warn = str(nvtx_kernel_span_info.get("warning") or "")
            if span_warn:
                warnings.append(f"[NVTX_SCOPE_KERNEL_SPAN] {span_warn}")
            else:
                k_count = int(nvtx_kernel_span_info.get("count") or 0)
                k_start = int(nvtx_kernel_span_info.get("kernel_start_ns") or 0)
                k_end = int(nvtx_kernel_span_info.get("kernel_end_ns") or 0)
                warnings.append(
                    f"[NVTX_SCOPE_KERNEL_SPAN] Attributed kernels={k_count}, "
                    f"gpu_window [{round(k_start/1e6,1)} ms - {round(k_end/1e6,1)} ms]."
                )

        with _phase("mfu"):
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
                        gpu_name = _first_gpu_name_from_conn(conn)
                        resolved_peak = infer_peak_tflops(gpu_name, precision=peak_precision)
                        if resolved_peak is None:
                            warnings.append("MFU peak_tflops not provided and GPU name mapping failed.")
                    if resolved_peak is not None:
                        mfu = compute_mfu_single(
                            step_time_s=float(step_time_s),
                            model_flops_per_step=float(model_flops_per_step),
                            peak_tflops=float(resolved_peak),
                        )

        with _phase("sync_breakdown"):
            sync_breakdown = _safe_skill(
                engine, "sync_breakdown", warnings,
                device_id=device_id, limit=50, start_ns=start_ns, end_ns=end_ns,
            )

        with _phase("memcpy_bandwidth"):
            memcpy_bandwidth = _safe_skill(
                engine, "memcpy_bandwidth_analysis", warnings,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        with _phase("kernel_duration_stats"):
            kernel_duration_stats = _safe_skill(
                engine, "kernel_duration_stats", warnings,
                device_id=device_id, start_ns=start_ns, end_ns=end_ns, min_invocations=3, limit=20,
            )

        with _phase("short_kernels_overhead"):
            short_kernels = _safe_skill(
                engine, "short_kernels_overhead", warnings,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
            )

        with _phase("per_stream_utilization"):
            per_stream_utilization = _safe_skill(
                engine, "per_stream_utilization", warnings,
                device_id=device_id, start_ns=start_ns, end_ns=end_ns, limit=30,
            )

        with _phase("nvtx_wall_efficiency"):
            _eff_nvtx_pattern = (
                f"%{nvtx_text}%" if nvtx_text and "%" not in nvtx_text
                else (nvtx_text if nvtx_text else f"%{iteration_marker}%")
            )
            nvtx_wall_efficiency = _safe_skill(
                engine, "nvtx_wall_efficiency", warnings,
                nvtx_text=_eff_nvtx_pattern,
                device_id=device_id,
                start_ns=start_ns,
                end_ns=end_ns,
                limit=200,
            )

        with _phase("nvtx_subphase_timing"):
            nvtx_subphase_timing = _safe_skill(
                engine, "nvtx_subphase_timing", warnings,
                start_ns=start_ns, end_ns=end_ns,
            )

        with _phase("gpu_metrics"):
            gpu_metrics_aggregate = _safe_skill(
                engine, "gpu_metrics_aggregate", warnings,
                device_id=device_id, start_ns=start_ns, end_ns=end_ns,
            )
            gpu_metrics_percentiles_rows = _safe_skill(
                engine, "gpu_metrics_percentiles", warnings,
                device_id=device_id, start_ns=start_ns, end_ns=end_ns,
            )

        with _phase("memcpy_transfers_detail"):
            memcpy_transfers_detail_rows = _safe_skill(
                engine, "memcpy_transfers_detail", warnings,
                device_id=device_id, start_ns=start_ns, end_ns=end_ns,
            )

        with _phase("cpu_launch_gap"):
            cpu_launch_gap_rows = _safe_skill(
                engine, "cpu_launch_gap", warnings,
                device_id=device_id, start_ns=start_ns, end_ns=end_ns,
            )

        with _phase("comprehensive_analysis"):
            _gpu_name = _first_gpu_name_from_conn(conn)
            comprehensive_analysis = build_comprehensive_analysis(
                gpu_name=_gpu_name,
                summary=summary or {},
                overlap=overlap or {},
                nccl_breakdown=list(nccl_breakdown or []),
                aggregate_kernels=top_kernels,
                per_stream_utilization=list(per_stream_utilization or []),
                memcpy_bandwidth=list(memcpy_bandwidth or []),
                sync_breakdown=list(sync_breakdown or []),
                gpu_metrics_aggregate=list(gpu_metrics_aggregate or []),
                gpu_metrics_percentiles=list(gpu_metrics_percentiles_rows or []),
                memcpy_transfers_detail=list(memcpy_transfers_detail_rows or []),
                cpu_launch_gap=list(cpu_launch_gap_rows or []),
                short_kernels=list(short_kernels or []),
                kernel_duration_stats=list(kernel_duration_stats or []),
            )

        with _phase("rank_straggler"):
            try:
                rank_straggler = engine.analyze_rank_straggler(
                    marker=iteration_marker,
                    device_id=device_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
            except Exception as exc:
                warnings.append(f"[SKILL_ERROR] analyze_rank_straggler failed: {exc}")
                rank_straggler = {"global_stats": {}, "per_rank": [], "stragglers": []}

        with _phase("bottleneck_classification"):
            _bottleneck_pattern = (
                f"%{nvtx_text}%" if nvtx_text and "%" not in nvtx_text
                else (nvtx_text if nvtx_text else f"%{iteration_marker}%")
            )
            try:
                bottleneck_classification = engine.classify_gpu_bottleneck(
                    nvtx_text=_bottleneck_pattern,
                    device_id=device_id,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
            except Exception as exc:
                warnings.append(f"[SKILL_ERROR] classify_gpu_bottleneck failed: {exc}")
                bottleneck_classification = []

        # ---- Automatic problem detection warnings ----
        # 1. Low overall GPU utilization
        util_pct = _to_float(((summary or {}).get("timing") or {}).get("utilization_pct"))
        if util_pct is not None and util_pct < 70.0:
            warnings.append(
                f"[LOW_GPU_UTIL] GPU utilization is {util_pct:.1f}% (target ≥70%). "
                "Likely causes: excessive synchronisation, CPU-bound preprocessing, "
                "or sequential H2D memcpy before kernel launch. "
                "Check sync_breakdown and per_stream_utilization for idle time."
            )

        # 2. High-jitter kernels (CV > 50%)
        if kernel_duration_stats:
            jitter_kernels = [
                r for r in kernel_duration_stats
                if _to_float(r.get("cv_pct")) is not None and (_to_float(r.get("cv_pct")) or 0) > 50.0
            ]
            if jitter_kernels:
                top_jitter = jitter_kernels[0]
                warnings.append(
                    f"[KERNEL_JITTER] {len(jitter_kernels)} kernel(s) have execution-time CV >50%."
                    f" Worst: '{top_jitter.get('kernel_name')}' CV={top_jitter.get('cv_pct')}%"
                    f" (avg={top_jitter.get('avg_ms')}ms, stddev={top_jitter.get('stddev_ms')}ms). "
                    "Likely causes: wavefront imbalance, branch divergence, dynamic shared "
                    "memory sizing, or device-side queues with variable work."
                )

        # 3. Micro-kernel (launch) overhead
        if short_kernels:
            micro_row = next((r for r in short_kernels if str(r.get("duration_bracket", "")).startswith("b_")), None)
            submicro_row = next((r for r in short_kernels if str(r.get("duration_bracket", "")).startswith("a_")), None)
            short_total_pct = 0.0
            for r in short_kernels:
                bracket = str(r.get("duration_bracket", ""))
                if bracket.startswith("a_") or bracket.startswith("b_"):
                    short_total_pct += float(r.get("pct_count") or 0)
            if short_total_pct > 30.0:
                warnings.append(
                    f"[LAUNCH_OVERHEAD] {short_total_pct:.0f}% of kernel invocations are <10 µs. "
                    "This level of micro-kernel overhead wastes GPU cycles on launch latency. "
                    "Consider kernel fusion, cuBLAS/cuDNN operator merging, or CUDA Graphs "
                    "to amortise launch overhead."
                )

        # 4. Poor compute/comm overlap
        overlap_pct_comm = _to_float((overlap or {}).get("overlap_pct_of_comm"))
        comm_total_ms = _to_float((overlap or {}).get("comm_total_ms"))
        if comm_total_ms and comm_total_ms > 10.0 and overlap_pct_comm is not None and overlap_pct_comm < 10.0:
            warnings.append(
                f"[LOW_OVERLAP] Compute/comm overlap is only {overlap_pct_comm:.1f}% of comm time. "
                "Communication stalls compute for most of each step. Enable async collectives "
                "(NCCL async_op=True / ZeRO stage 2 overlap) or pipeline microbatches to "
                "improve overlap."
            )

        # 5. Straggler ranks
        if rank_straggler.get("stragglers"):
            slowest = rank_straggler["stragglers"]
            per_rank = rank_straggler.get("per_rank", [])
            worst_entry = next((r for r in per_rank if str(r.get("rank")) in slowest), None)
            delta = worst_entry.get("delta_vs_global_pct") if worst_entry else None
            warnings.append(
                f"[STRAGGLER] Rank(s) {slowest} are stragglers "
                + (f"(+{delta}% vs global median). " if delta is not None else ". ")
                + "Investigate per-rank data loading time, uneven tensor sizes, or "
                "NUMA / NVLink topology affinity."
            )

        # 6. Memory-bound phases
        mem_bound_phases = [
            r for r in bottleneck_classification
            if r.get("bottleneck") == "memory_bound"
        ]
        if mem_bound_phases:
            top_mem = mem_bound_phases[0]
            warnings.append(
                f"[MEMORY_BOUND] {len(mem_bound_phases)} NVTX phase(s) are memory-bandwidth bound "
                f"(DRAM throughput avg {top_mem.get('dram_throughput_avg')}%). "
                "Increase arithmetic intensity with larger tiles/microbatches, use "
                "Flash-Attention style fused kernels, or enable activation recomputation "
                "to trade memory for compute."
            )

        # 7. Latency-bound (low utilization everywhere)
        latency_bound_phases = [
            r for r in bottleneck_classification
            if r.get("bottleneck") == "latency_bound"
        ]
        if latency_bound_phases:
            warnings.append(
                f"[LATENCY_BOUND] {len(latency_bound_phases)} NVTX phase(s) show low SM active, "
                "tensor active AND DRAM throughput simultaneously. This typically indicates "
                "synchronisation barriers, PCIe transfers, or host-side preprocessing blocking "
                "GPU progress."
            )

        return {
            "sqlite_path": sqlite_path,
            "device_id": int(device_id),
            "window": {"start_ns": int(start_ns), "end_ns": int(end_ns)},
            "nvtx_text_info": nvtx_text_info,
            "nvtx_kernel_span_info": nvtx_kernel_span_info,
            "schema": schema,
            "summary": summary,
            "overlap": overlap,
            "top_kernels": top_kernels,
            "nccl_breakdown": nccl_breakdown,
            "iterations": iterations,
            "sync_breakdown": sync_breakdown,
            "memcpy_bandwidth": memcpy_bandwidth,
            "kernel_duration_stats": kernel_duration_stats,
            "short_kernels": short_kernels,
            "per_stream_utilization": per_stream_utilization,
            "nvtx_wall_efficiency": nvtx_wall_efficiency,
            "nvtx_subphase_timing": nvtx_subphase_timing,
            "rank_straggler": rank_straggler,
            "bottleneck_classification": bottleneck_classification,
            "gpu_metrics_aggregate": gpu_metrics_aggregate,
            "gpu_metrics_percentiles": gpu_metrics_percentiles_rows,
            "memcpy_transfers_detail": memcpy_transfers_detail_rows,
            "cpu_launch_gap": cpu_launch_gap_rows,
            "comprehensive_analysis": comprehensive_analysis,
            "mfu": mfu,
            "warnings": warnings,
            "phase_timings": phase_timings,
        }
    finally:
        conn.close()


def analyze_to_markdown(result: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# NSYS Analyze Report")
    lines.append("")
    lines.append(f"- sqlite: `{result.get('sqlite_path', '')}`")
    lines.append(f"- device_id: `{result.get('device_id', -1)}`")
    win = result.get("window") or {}
    if win.get("start_ns", -1) >= 0 or win.get("end_ns", -1) >= 0:
        lines.append(
            f"- window: `{round(int(win.get('start_ns',-1))/1e6,1)} ms` – "
            f"`{round(int(win.get('end_ns',-1))/1e6,1)} ms`"
        )
    lines.append("")

    # NVTX scope banner
    nsi = result.get("nvtx_text_info")
    if nsi:
        lines.append("## NVTX Scope")
        lines.append("")
        lines.append(f"- pattern: `{nsi.get('pattern', '')}`")
        lines.append(f"- matched ranges: `{nsi.get('count', 0)}`")
        lines.append(f"- total_duration_ms: `{nsi.get('total_duration_ms', 0)}`")
        w_s = int(nsi.get("window_start_ns") or 0)
        w_e = int(nsi.get("window_end_ns") or 0)
        lines.append(
            f"- analysis window: `{round(w_s/1e6,1)} ms` – `{round(w_e/1e6,1)} ms` "
            f"(span `{round((w_e-w_s)/1e6,1)} ms`)"
        )
        if nsi.get("warning"):
            lines.append(f"- **warning**: {nsi['warning']}")
        # Show up to 10 matched ranges
        matched = list(nsi.get("matched_ranges") or [])
        if matched:
            lines.append("")
            lines.append("| # | nvtx_text | start_ms | end_ms | duration_ms |")
            lines.append("|---|-----------|----------|--------|-------------|")
            for i, r in enumerate(matched[:10]):
                lines.append(
                    f"| {i+1} | {r.get('nvtx_text','')} "
                    f"| {round(int(r.get('start_ns',0))/1e6,1)} "
                    f"| {round(int(r.get('end_ns',0))/1e6,1)} "
                    f"| {r.get('duration_ms','')} |"
                )
            if len(matched) > 10:
                lines.append(f"| … | *(+{len(matched)-10} more)* | | | |")
        lines.append("")

    nks = result.get("nvtx_kernel_span_info")
    if nks:
        lines.append("## NVTX Kernel Span")
        lines.append("")
        lines.append(f"- pattern: `{nks.get('pattern', '')}`")
        lines.append(f"- attributed kernels: `{nks.get('count', 0)}`")
        if nks.get("warning"):
            lines.append(f"- **warning**: {nks.get('warning')}")
        else:
            k_s = int(nks.get("kernel_start_ns") or 0)
            k_e = int(nks.get("kernel_end_ns") or 0)
            lines.append(
                f"- gpu kernel window: `{round(k_s/1e6,1)} ms` - `{round(k_e/1e6,1)} ms` "
                f"(span `{round((k_e-k_s)/1e6,1)} ms`)"
            )
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

    sync_bd = result.get("sync_breakdown") or []
    if sync_bd:
        lines.append("## Sync Breakdown")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(sync_bd, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    memcpy_bw = result.get("memcpy_bandwidth") or []
    if memcpy_bw:
        lines.append("## Memcpy Bandwidth")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(memcpy_bw, ensure_ascii=False, indent=2))
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

    nwe = result.get("nvtx_wall_efficiency") or []
    if nwe:
        lines.append("## NVTX Wall-Clock GPU Efficiency")
        lines.append("")
        lines.append("| NVTX Phase | wall_ms | kernel_busy_ms | gpu_efficiency_pct |")
        lines.append("|------------|---------|----------------|--------------------|")
        for r in nwe[:20]:
            eff = r.get("gpu_efficiency_pct", "")
            flag = " ⚠" if isinstance(eff, (int, float)) and float(eff) < 60 else ""
            lines.append(
                f"| {r.get('nvtx_text', '')} "
                f"| {r.get('wall_ms', '')} "
                f"| {r.get('kernel_busy_ms', '')} "
                f"| {eff}%{flag} |"
            )
        lines.append("")

    spt = result.get("nvtx_subphase_timing") or []
    if spt:
        lines.append("## NVTX Sub-Phase Timing")
        lines.append("")
        win = result.get("window") or {}
        w_start = int(win.get("start_ns", -1))
        w_end = int(win.get("end_ns", -1))
        if w_start >= 0 and w_end > w_start:
            lines.append(f"*Window: {round(w_start/1e6,1)} ms – {round(w_end/1e6,1)} ms (span {round((w_end-w_start)/1e6,1)} ms)*")
            lines.append("")
        lines.append("| Sub-Phase | Count | total_ms | avg_ms | min_ms | max_ms | pct_of_window |")
        lines.append("|-----------|-------|----------|--------|--------|--------|---------------|")
        for r in spt[:50]:
            pct = r.get("pct_of_window")
            pct_str = f"{pct}%" if pct is not None else "n/a"
            lines.append(
                f"| {r.get('nvtx_name', '')} "
                f"| {r.get('range_count', '')} "
                f"| {r.get('total_ms', '')} "
                f"| {r.get('avg_ms', '')} "
                f"| {r.get('min_ms', '')} "
                f"| {r.get('max_ms', '')} "
                f"| {pct_str} |"
            )
        lines.append("")

    kds = result.get("kernel_duration_stats") or []
    if kds:
        lines.append("## Kernel Duration Jitter (Top by CV)")
        lines.append("")
        lines.append("| Kernel | Invocations | avg_ms | stddev_ms | cv_pct | total_ms |")
        lines.append("|--------|-------------|--------|-----------|--------|----------|")
        for r in kds[:10]:
            lines.append(
                f"| {r.get('kernel_name', '')} "
                f"| {r.get('invocations', '')} "
                f"| {r.get('avg_ms', '')} "
                f"| {r.get('stddev_ms', '')} "
                f"| {r.get('cv_pct', '')} % "
                f"| {r.get('total_ms', '')} |"
            )
        lines.append("")

    sk = result.get("short_kernels") or []
    if sk:
        lines.append("## Small-Kernel Launch Overhead Distribution")
        lines.append("")
        lines.append("| Duration Bracket | Count | total_ms | pct_count | pct_time |")
        lines.append("|------------------|-------|----------|-----------|----------|")
        for r in sk:
            lines.append(
                f"| {r.get('duration_bracket', '')} "
                f"| {r.get('kernel_count', '')} "
                f"| {r.get('total_ms', '')} "
                f"| {r.get('pct_count', '')} % "
                f"| {r.get('pct_time', '')} % |"
            )
        lines.append("")

    psu = result.get("per_stream_utilization") or []
    if psu:
        lines.append("## Per-Stream GPU Utilization")
        lines.append("")
        lines.append("| Stream | Kernels | busy_ms | span_ms | utilization_pct |")
        lines.append("|--------|---------|---------|---------|-----------------|")
        for r in psu[:15]:
            lines.append(
                f"| {r.get('stream_id', '')} "
                f"| {r.get('kernel_count', '')} "
                f"| {r.get('kernel_busy_ms', '')} "
                f"| {r.get('stream_span_ms', '')} "
                f"| {r.get('utilization_pct', '')} % |"
            )
        lines.append("")

    rs = result.get("rank_straggler") or {}
    if rs.get("per_rank"):
        lines.append("## Multi-Rank Straggler Analysis")
        lines.append("")
        gst = rs.get("global_stats") or {}
        lines.append(f"- Global median: `{gst.get('median_ms', 'N/A')}` ms  |  "
                     f"std: `{gst.get('std_ms', 'N/A')}` ms  |  "
                     f"count: `{gst.get('count', 'N/A')}`")
        lines.append("")
        lines.append("| Rank | Iterations | median_ms | std_ms | delta_vs_global_pct | straggler |")
        lines.append("|------|------------|-----------|--------|---------------------|-----------|")
        for r in rs["per_rank"]:
            tag = "**YES**" if r.get("is_straggler") else "no"
            lines.append(
                f"| {r.get('rank', '')} "
                f"| {r.get('count', '')} "
                f"| {r.get('median_ms', '')} "
                f"| {r.get('std_ms', '')} "
                f"| {r.get('delta_vs_global_pct', '')} % "
                f"| {tag} |"
            )
        lines.append("")

    bc = result.get("bottleneck_classification") or []
    if bc:
        lines.append("## GPU Bottleneck Classification (per NVTX Phase)")
        lines.append("")
        lines.append("| NVTX Phase | sm_active% | tensor_active% | dram_throughput% | bottleneck | confidence |")
        lines.append("|------------|-----------|----------------|------------------|------------|------------|")
        for r in bc[:20]:
            lines.append(
                f"| {r.get('nvtx_text', '')} "
                f"| {r.get('sm_active_avg', '')} "
                f"| {r.get('tensor_active_avg', '')} "
                f"| {r.get('dram_throughput_avg', '')} "
                f"| **{r.get('bottleneck', '')}** "
                f"| {r.get('bottleneck_confidence', '')} |"
            )
        lines.append("")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("## Problem Detections & Recommendations")
        lines.append("")
        for item in warnings:
            # Extract tag like [LOW_GPU_UTIL]
            tag_end = item.find("]")
            if item.startswith("[") and tag_end > 0:
                tag = item[1:tag_end]
                body = item[tag_end + 2:]
                lines.append(f"### `{tag}`")
                lines.append(body)
            else:
                lines.append(f"- {item}")
            lines.append("")

    # Comprehensive auto-analysis section
    comp = result.get("comprehensive_analysis")
    if comp:
        lines.append("")
        lines.append(comprehensive_to_markdown(comp))

    # Phase timing summary
    pt = result.get("phase_timings") or []
    if pt:
        lines.append("")
        lines.append("## Analysis Phase Timings")
        lines.append("")
        total_ms = sum(float(p.get("elapsed_ms") or 0) for p in pt)
        lines.append(f"*Total analysis time: {round(total_ms, 1)} ms*")
        lines.append("")
        lines.append("| Phase | elapsed_ms |")
        lines.append("|-------|------------|")
        for p in pt:
            lines.append(f"| {p.get('phase', '')} | {p.get('elapsed_ms', '')} |")
        lines.append("")

    return "\n".join(lines)

