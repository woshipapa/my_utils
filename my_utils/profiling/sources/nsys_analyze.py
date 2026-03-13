from __future__ import annotations

import json
import sqlite3
import statistics
from typing import Dict, List, Optional

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
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        schema_obj = NsightSchema(conn)
        engine = NsysSqlSkillEngine(conn)

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

        summary = engine.summarize_gpu_kernels(
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
            top_k=top_k,
            limit=limit,
        )
        overlap = engine.analyze_compute_comm_overlap(
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
            limit=limit,
        )
        # top_kernels is already computed inside summarize_gpu_kernels(); reuse it.
        top_kernels = list((summary or {}).get("top_kernels") or [])
        nccl_breakdown = engine.execute(
            "nccl_breakdown",
            device_id=device_id,
            limit=max(int(top_k), 1),
        )
        iterations = engine.detect_iterations(
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

        sync_breakdown = engine.execute("sync_breakdown", device_id=device_id, limit=50) if engine.get_skill("sync_breakdown") else []
        memcpy_bandwidth = engine.execute("memcpy_bandwidth_analysis", device_id=device_id) if engine.get_skill("memcpy_bandwidth_analysis") else []

        # --- New deeper analyses ---
        # Kernel jitter: sort by CV descending so worst-jitter kernels appear first
        kernel_duration_stats = (
            engine.execute("kernel_duration_stats", device_id=device_id, min_invocations=3, limit=20)
            if engine.get_skill("kernel_duration_stats") else []
        )

        # Small-kernel overhead bracketing
        short_kernels = (
            engine.execute("short_kernels_overhead", device_id=device_id)
            if engine.get_skill("short_kernels_overhead") else []
        )

        # Per-stream utilization
        per_stream_utilization = (
            engine.execute("per_stream_utilization", device_id=device_id, limit=30)
            if engine.get_skill("per_stream_utilization") else []
        )

        # Rank straggler analysis (only meaningful when NVTX encodes rank tags)
        rank_straggler = engine.analyze_rank_straggler(
            marker=iteration_marker,
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )

        # GPU bottleneck classification per NVTX range (requires GPU metrics)
        bottleneck_classification = engine.classify_gpu_bottleneck(
            nvtx_text=f"%{iteration_marker}%",
            device_id=device_id,
            start_ns=start_ns,
            end_ns=end_ns,
        )

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
            "rank_straggler": rank_straggler,
            "bottleneck_classification": bottleneck_classification,
            "mfu": mfu,
            "warnings": warnings,
        }
    finally:
        conn.close()


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
    return "\n".join(lines)

