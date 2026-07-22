# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import time
from pathlib import Path

from my_utils.profiling import DEFAULT_PROVIDER_REGISTRY, MetricEvent, MetricsAnalyzer
from my_utils.profiling.analyzers.analysis_rules import (
    CommunicationHealthRule,
    CrossLayerConsistencyRule,
    RooflineGapRule,
)
from my_utils.profiling.metrics.metrics_providers import (
    DcgmCsvMetricsProvider,
    NcclLogMetricsProvider,
    RasJsonMetricsProvider,
)
from my_utils.profiling.sources.nsys_auto_analysis import build_comprehensive_analysis
from my_utils.profiling.sources.nsys_timeline_html import (
    _extract_timeline_payload_from_html,
    _render_html,
)


def _evt(name: str, value: float, unit: str, **tags: str) -> MetricEvent:
    return MetricEvent(
        timestamp=time.time(),
        name=name,
        value=value,
        unit=unit,
        provider_id="ut",
        tags=tags,
    )


def test_provider_registry_has_new_observability_providers() -> None:
    types = set(DEFAULT_PROVIDER_REGISTRY.list_types())
    assert {"dcgm_csv", "nccl_log", "ras_json"}.issubset(types)


def test_dcgm_csv_provider_incremental_read(tmp_path: Path) -> None:
    csv_path = tmp_path / "dcgm.csv"
    csv_path.write_text(
        "timestamp,gpu_id,metric_name,value,unit,rank\n"
        "1712000000,0,sm_active,71.5,percent,0\n",
        encoding="utf-8",
    )
    provider = DcgmCsvMetricsProvider(str(csv_path))
    first = provider.get_metrics()
    assert len(first) == 1
    assert first[0].name == "gpu.dcgm.sm_active"
    assert first[0].tags.get("gpu") == "0"

    second = provider.get_metrics()
    assert second == []

    with csv_path.open("a", encoding="utf-8") as handle:
        handle.write("1712000001,1,dram_throughput,62.0,percent,1\n")
    third = provider.get_metrics()
    assert len(third) == 1
    assert third[0].tags.get("rank") == "1"


def test_nccl_log_provider_parses_duration_and_bandwidth(tmp_path: Path) -> None:
    log_path = tmp_path / "nccl.rank0.log"
    log_path.write_text(
        "2026-04-02 10:00:00 NCCL INFO rank 0 allreduce time 3.2 ms algbw 120.5 busbw 95.2 nBytes 1048576\n"
        "2026-04-02 10:00:01 NCCL WARN rank 0 timeout in allreduce\n",
        encoding="utf-8",
    )
    provider = NcclLogMetricsProvider(str(log_path))
    events = provider.get_metrics()
    names = {item.name for item in events}
    assert "comm.nccl.duration" in names
    assert "comm.nccl.busbw" in names
    assert "comm.nccl.algbw" in names
    assert "comm.nccl.bytes" in names
    assert "comm.nccl.issue.count" in names


def test_ras_json_provider_supports_jsonl_incremental(tmp_path: Path) -> None:
    path = tmp_path / "ras.jsonl"
    path.write_text(
        json.dumps(
            {
                "timestamp": 1712000000,
                "severity": "warning",
                "component": "nvlink",
                "correctable_errors": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = RasJsonMetricsProvider(str(path), json_lines=True)
    first = provider.get_metrics()
    assert any(item.name == "ras.correctable_errors" for item in first)
    assert any("severity.warning.count" in item.name for item in first)

    assert provider.get_metrics() == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "timestamp": 1712000001,
                    "severity": "error",
                    "component": "pcie",
                    "fatal_errors": 1,
                }
            )
            + "\n"
        )
    second = provider.get_metrics()
    assert any(item.name == "ras.fatal_errors" for item in second)


def test_new_analysis_rules_emit_findings() -> None:
    events = [
        _evt("comm.nccl.duration", 10.0, "ms", rank="0"),
        _evt("comm.nccl.duration", 40.0, "ms", rank="1"),
        _evt("comm.nccl.busbw", 12.0, "GB/s", rank="0"),
        _evt("comm.nccl.issue.count", 1.0, "count", rank="1", level="error"),
        _evt("compute.gpu.sm.active", 35.0, "percent"),
        _evt("gpu.ncu.dram_throughput", 82.0, "percent"),
        _evt("latency.step", 100.0, "ms", step="0"),
        _evt("latency.kernel.cuda", 12.0, "ms", step="0"),
    ]
    comm_finding = CommunicationHealthRule().apply(events, {})
    roofline_finding = RooflineGapRule().apply(events, {})
    consistency_finding = CrossLayerConsistencyRule().apply(events, {})
    assert comm_finding is not None
    assert roofline_finding is not None
    assert consistency_finding is not None


def test_pretrain_profile_includes_new_rules() -> None:
    analyzer = MetricsAnalyzer(workload_profile="pretrain", enable_advanced_rules=True)
    rules = set(analyzer.list_rules())
    assert {"comm_health", "roofline_gap", "cross_layer_consistency"}.issubset(rules)


def test_build_comprehensive_analysis_contains_new_sections() -> None:
    report = build_comprehensive_analysis(
        gpu_name="NVIDIA H100",
        summary={
            "timing": {"span_ms": 100.0, "busy_ms": 65.0, "utilization_pct": 65.0}
        },
        overlap={
            "comm_total_ms": 30.0,
            "compute_total_ms": 70.0,
            "overlap_ms": 10.0,
            "comm_only_ms": 20.0,
            "overlap_pct_of_comm": 33.3,
            "overlap_pct_of_compute": 14.2,
        },
        nccl_breakdown=[
            {
                "kernel_name": "ncclAllReduceRingLLKernel_sum_f16",
                "total_ms": 30.0,
                "invocations": 3,
                "avg_ms": 10.0,
            }
        ],
        aggregate_kernels=[
            {"kernel_name": "gemm_kernel", "total_ms": 50.0, "invocations": 5},
            {
                "kernel_name": "ncclAllReduceRingLLKernel_sum_f16",
                "total_ms": 30.0,
                "invocations": 3,
            },
        ],
        per_stream_utilization=[
            {
                "stream_id": 7,
                "kernel_count": 10,
                "kernel_busy_ms": 60.0,
                "stream_span_ms": 100.0,
                "utilization_pct": 60.0,
            },
            {
                "stream_id": 8,
                "kernel_count": 3,
                "kernel_busy_ms": 20.0,
                "stream_span_ms": 90.0,
                "utilization_pct": 22.2,
            },
        ],
        memcpy_bandwidth=[
            {
                "copy_kind": 1,
                "count": 4,
                "total_gb": 6.0,
                "total_ms": 20.0,
                "avg_gbps": 300.0,
            }
        ],
        sync_breakdown=[
            {
                "sync_type": "cudaStreamSynchronize",
                "count": 4,
                "total_ms": 12.0,
                "avg_ms": 3.0,
            }
        ],
        gpu_metrics_aggregate=[
            {
                "metric_name": "sm__active.avg.pct_of_peak_sustained_elapsed",
                "sample_count": 10,
                "avg_value": 45.0,
                "max_value": 80.0,
            },
            {
                "metric_name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "sample_count": 10,
                "avg_value": 75.0,
                "max_value": 95.0,
            },
        ],
        gpu_metrics_percentiles=[],
        memcpy_transfers_detail=[{"bytes": 128 * 1024}],
        cpu_launch_gap=[
            {"kernel_name": "gemm_kernel", "avg_gap_us": 80.0, "total_gap_ms": 8.0}
        ],
        short_kernels=[{"duration_bracket": "b_lt10us", "pct_count": 35.0}],
        kernel_duration_stats=[
            {
                "kernel_name": "gemm_kernel",
                "cv_pct": 12.0,
                "avg_ms": 10.0,
                "stddev_ms": 1.2,
            }
        ],
    )
    assert "communication_health" in report
    assert "roofline_gap" in report
    assert "cross_layer_consistency" in report


def test_timeline_html_contains_new_observability_panels() -> None:
    kernels = [
        {
            "rank": 0,
            "device_id": 0,
            "stream_id": 7,
            "kernel_name": "gemm_kernel",
            "start_ns": 0,
            "end_ns": 10_000,
            "duration_ms": 0.01,
            "kind": "compute",
            "occupancy_pct_estimate": 70.0,
        },
        {
            "rank": 1,
            "device_id": 0,
            "stream_id": 8,
            "kernel_name": "ncclAllReduce",
            "start_ns": 2_000,
            "end_ns": 12_000,
            "duration_ms": 0.01,
            "kind": "comm",
            "occupancy_pct_estimate": 60.0,
        },
    ]
    metric_series = [
        {
            "name": "sm__active.avg.pct_of_peak_sustained_elapsed [gpu 0]",
            "color": "#5fa",
            "points": [[0, 62.0], [10_000, 58.0]],
        },
        {
            "name": "dram__throughput.avg.pct_of_peak_sustained_elapsed [gpu 0]",
            "color": "#fa5",
            "points": [[0, 75.0], [10_000, 78.0]],
        },
        {
            "name": "python_gil_hold_pct",
            "color": "#acf",
            "points": [[0, 40.0], [10_000, 55.0]],
        },
    ]
    html_text = _render_html(
        sqlite_path="dummy.sqlite",
        kernels=kernels,
        metric_series=metric_series,
        kernel_category_summary={"rows": []},
        kernel_category_profile="custom",
        nvtx_window_category_stats={},
        window_start_ns=0,
        window_end_ns=10_000,
        display_span_ns=10_000,
        nvtx_windows=[],
        width_px=800,
        include_metrics=True,
        overlay_metrics_per_track=2,
    )
    assert "Rank Heatmap" in html_text
    assert "Roofline Proxy" in html_text
    assert "Python GIL Lane" in html_text

    payload = _extract_timeline_payload_from_html(html_text)
    assert payload is not None
    assert payload.get("rank_heatmap_rows")
    assert payload.get("roofline_proxy")
    assert payload.get("gil_lane_series")
