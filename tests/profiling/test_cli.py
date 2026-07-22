# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.cli (nsys-facing subcommands)."""

from __future__ import annotations


import json
import re
import sqlite3
from pathlib import Path


from my_utils.profiling.cli import main


from _synthetic_loader import _init_sqlite


def test_cli_timeline_metrics_defaults_keep_all_points(tmp_path: Path) -> None:
    db = tmp_path / "rank0_cli_default_full_metrics.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    extra_rows = []
    start_ns = 1_000_000
    for i in range(5000):
        ts = start_ns + i * 1000
        extra_rows.append((ts, 101, float(i % 100), 1))
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        extra_rows,
    )
    conn.commit()
    conn.close()

    out_html = tmp_path / "timeline_default_full_metrics.html"
    rc = main(
        [
            "nsys-timeline-html",
            "--sqlite",
            str(db),
            "--output",
            str(out_html),
            "--device-id",
            "0",
            "--start-ns",
            str(start_ns),
            "--end-ns",
            str(start_ns + 4_999_000),
            "--include-metrics",
            "--metric-name-like",
            "%sm__active%",
        ]
    )
    assert rc == 0
    text = out_html.read_text(encoding="utf-8")
    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", text, flags=re.S)
    assert m is not None
    payload = json.loads(m.group(1))
    series = payload.get("metrics") or []
    sm_series = next(
        (s for s in series if "sm__active" in str(s.get("name", ""))), None
    )
    assert sm_series is not None, series
    assert len(sm_series.get("points", [])) == 5000


def test_timeline_default_focus_metrics_filters_unrelated_series(
    tmp_path: Path,
) -> None:
    db = tmp_path / "timeline_focus_metrics.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        (777, "random_metric_should_be_filtered"),
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        [
            (2000, 777, 12.0, 1),
            (3000, 777, 15.0, 1),
        ],
    )
    conn.commit()
    conn.close()

    out_html = tmp_path / "timeline_focus_metrics.html"
    rc = main(
        [
            "nsys-timeline-html",
            "--sqlite",
            str(db),
            "--output",
            str(out_html),
            "--device-id",
            "0",
            "--include-metrics",
            "--default-focus-metrics",
        ]
    )
    assert rc == 0
    text = out_html.read_text(encoding="utf-8")
    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", text, flags=re.S)
    assert m is not None
    payload = json.loads(m.group(1))
    names = {str(s.get("name", "")) for s in (payload.get("metrics") or [])}
    assert any("tensor__active" in n for n in names), names
    assert all("random_metric_should_be_filtered" not in n for n in names), names


def test_timeline_focus_metrics_enabled_by_default(tmp_path: Path) -> None:
    db = tmp_path / "timeline_focus_metrics_default_on.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (7788, "random_metric_should_be_filtered"),
            (7789, "NVLINK TX Throughput"),
        ],
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        [
            (2000, 7788, 12.0, 1),
            (3000, 7788, 15.0, 1),
            (2500, 7789, 22.0, 1),
            (3500, 7789, 24.0, 1),
        ],
    )
    conn.commit()
    conn.close()

    out_html = tmp_path / "timeline_focus_metrics_default_on.html"
    rc = main(
        [
            "nsys-timeline-html",
            "--sqlite",
            str(db),
            "--output",
            str(out_html),
            "--device-id",
            "0",
            "--include-metrics",
        ]
    )
    assert rc == 0
    text = out_html.read_text(encoding="utf-8")
    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", text, flags=re.S)
    assert m is not None
    payload = json.loads(m.group(1))
    names = {str(s.get("name", "")) for s in (payload.get("metrics") or [])}
    assert any("tensor__active" in n for n in names), names
    assert any("dram" in n.lower() for n in names), names
    assert any("nvlink" in n.lower() for n in names), names
    assert all("random_metric_should_be_filtered" not in n for n in names), names


def test_timeline_default_focus_warps_metrics_keep_avg_throughput_and_cycle(
    tmp_path: Path,
) -> None:
    db = tmp_path / "timeline_focus_warps_throughput_only.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (778, "Compute Warps in Flight [Avg Warps Per Cycle]"),
            (779, "Compute Warps in Flight [Throughput %]"),
        ],
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        [
            (2100, 778, 3.25, 1),
            (2200, 778, 3.55, 1),
            (2100, 779, 61.0, 1),
            (2200, 779, 66.0, 1),
        ],
    )
    conn.commit()
    conn.close()

    out_html = tmp_path / "timeline_focus_warps_throughput_only.html"
    rc = main(
        [
            "nsys-timeline-html",
            "--sqlite",
            str(db),
            "--output",
            str(out_html),
            "--device-id",
            "0",
            "--include-metrics",
            "--default-focus-metrics",
        ]
    )
    assert rc == 0
    text = out_html.read_text(encoding="utf-8")
    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", text, flags=re.S)
    assert m is not None
    payload = json.loads(m.group(1))
    names = {str(s.get("name", "")).lower() for s in (payload.get("metrics") or [])}
    assert any("compute warps in flight" in n and "throughput" in n for n in names), (
        names
    )
    assert any(
        "compute warps in flight" in n and "avg warps per cycle" in n for n in names
    ), names


def test_cli_new_subcommands(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    overlap_json = tmp_path / "iter_overlap.json"
    outliers_json = tmp_path / "iter_outliers.json"
    occ_json = tmp_path / "occ_h100.json"

    # nsys-iter-overlap
    assert (
        main(
            [
                "nsys-iter-overlap",
                "--sqlite",
                str(db),
                "--iteration-marker",
                "sample_0",
                "--device-id",
                "0",
                "--limit",
                "100",
                "--output",
                str(overlap_json),
                "--pretty",
            ]
        )
        == 0
    )
    assert overlap_json.exists()
    data = json.loads(overlap_json.read_text())
    assert isinstance(data, list) and len(data) >= 1
    assert "compute_ms" in data[0]
    assert "comm_ms" in data[0]
    assert "overlap_ms" in data[0]
    assert "comm_pct" in data[0]
    print(f"\n[nsys-iter-overlap] {len(data)} iterations written to {overlap_json}")

    # nsys-iter-outliers
    assert (
        main(
            [
                "nsys-iter-outliers",
                "--sqlite",
                str(db),
                "--iteration-marker",
                "sample_0",
                "--device-id",
                "0",
                "--sigma",
                "0.5",
                "--output",
                str(outliers_json),
                "--pretty",
            ]
        )
        == 0
    )
    assert outliers_json.exists()
    data2 = json.loads(outliers_json.read_text())
    assert "stats" in data2 and "outliers" in data2
    assert data2["stats"]["count"] >= 1
    print(
        f"[nsys-iter-outliers] stats={data2['stats']}  outliers={len(data2['outliers'])}"
    )

    # nsys-sql-skill: occupancy should be enriched for H100 by default (--occupancy-arch auto)
    assert (
        main(
            [
                "nsys-sql-skill",
                "--sqlite",
                str(db),
                "--skill",
                "kernel_occupancy_estimate",
                "--param",
                "device_id=0",
                "--param",
                "limit=10",
                "--output",
                str(occ_json),
                "--pretty",
            ]
        )
        == 0
    )
    assert occ_json.exists()
    occ_rows = json.loads(occ_json.read_text())
    assert isinstance(occ_rows, list) and len(occ_rows) >= 1
    assert "occupancy_pct_h100_estimate" in occ_rows[0]
    assert occ_rows[0]["occupancy_pct_h100_estimate"] is not None


def test_cli_sql_skill_reports_missing_gpu_metrics(tmp_path: Path, capsys) -> None:
    db = tmp_path / "no_metrics.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_GPU_METRIC")
    conn.commit()
    conn.close()

    rc = main(
        [
            "nsys-sql-skill",
            "--sqlite",
            str(db),
            "--skill",
            "gpu_metrics_aggregate",
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    msg = (captured.err or "") + (captured.out or "")
    assert "unavailable" in msg.lower()
    assert "gpu metrics table" in msg.lower()


def test_cli_sql_skill_debug_logs(tmp_path: Path, capsys) -> None:
    db = tmp_path / "debug.sqlite"
    _init_sqlite(db)

    rc = main(
        [
            "nsys-sql-skill",
            "--sqlite",
            str(db),
            "--skill",
            "top_kernels",
            "--param",
            "device_id=0",
            "--param",
            "limit=5",
            "--debug",
            "--debug-rows",
            "2",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    msg = (captured.err or "") + (captured.out or "")
    assert "[nsys-sql-skill][debug]" in msg
    assert "skill=top_kernels" in msg
    assert "rows=" in msg


def test_cli_schema_inspect_grouped_and_mermaid(tmp_path: Path) -> None:
    db = tmp_path / "schema.sqlite"
    _init_sqlite(db)
    out_json = tmp_path / "schema_inspect.json"

    rc = main(
        [
            "nsys-sql-skill",
            "--sqlite",
            str(db),
            "--skill",
            "schema_inspect",
            "--output",
            str(out_json),
            "--pretty",
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "tables" in payload
    assert "relations" in payload
    assert "mermaid" in payload
    tables = payload["tables"]
    assert isinstance(tables, list) and len(tables) >= 1
    by_name = {t["table_name"]: t for t in tables}
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in by_name
    assert "correlationId" in set(by_name["CUPTI_ACTIVITY_KIND_KERNEL"]["columns"])
    mermaid = str(payload["mermaid"])
    assert "flowchart LR" in mermaid
    assert "CUPTI_ACTIVITY_KIND_RUNTIME" in mermaid
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in mermaid


def test_cli_nsys_commands(tmp_path: Path) -> None:
    db_a = tmp_path / "a.sqlite"
    db_b = tmp_path / "b.sqlite"
    _init_sqlite(db_a, scale=1.0)
    _init_sqlite(db_b, scale=1.2)
    # Inject one additional rank=1 scope + runtime + kernel to validate
    # multi-rank timeline rendering in a single HTML.
    conn = sqlite3.connect(str(db_a))
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        (37000, 47000, "sample_0 step=1 rank=1", None, 59, 32345678),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        (38000, 38200, 101, 3, 32345678),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (39000, 43000, 11, 101, 1, 1, 0, 128, 1, 1, 32, 4096, 0, 75.0),
    )
    conn.commit()
    conn.close()

    export_json = tmp_path / "kernels.json"
    export_csv = tmp_path / "kernels.csv"
    analyze_json = tmp_path / "analyze.json"
    diff_json = tmp_path / "diff.json"
    timeline_html = tmp_path / "timeline.html"
    timeline_nvtx_html = tmp_path / "timeline_nvtx_metrics.html"
    timeline_compare_html = tmp_path / "timeline_compare.html"

    assert (
        main(
            [
                "nsys-export",
                "--sqlite",
                str(db_a),
                "--output",
                str(export_json),
                "--format",
                "json",
                "--device-id",
                "0",
            ]
        )
        == 0
    )
    assert export_json.exists()

    assert (
        main(
            [
                "nsys-export",
                "--sqlite",
                str(db_a),
                "--output",
                str(export_csv),
                "--format",
                "csv",
                "--device-id",
                "0",
                "--attach-iteration",
            ]
        )
        == 0
    )
    assert export_csv.exists()

    assert (
        main(
            [
                "nsys-analyze",
                "--sqlite",
                str(db_a),
                "--device-id",
                "0",
                "--model-flops-per-step",
                "1e12",
                "--peak-tflops",
                "100",
                "--output",
                str(analyze_json),
            ]
        )
        == 0
    )
    assert analyze_json.exists()

    assert (
        main(
            [
                "nsys-diff",
                "--before-sqlite",
                str(db_a),
                "--after-sqlite",
                str(db_b),
                "--device-id",
                "0",
                "--output",
                str(diff_json),
            ]
        )
        == 0
    )
    assert diff_json.exists()

    assert (
        main(
            [
                "nsys-timeline-html",
                "--sqlite",
                str(db_a),
                "--output",
                str(timeline_html),
                "--device-id",
                "0",
            ]
        )
        == 0
    )
    assert timeline_html.exists()

    assert (
        main(
            [
                "nsys-timeline-html",
                "--sqlite",
                str(db_a),
                "--output",
                str(timeline_nvtx_html),
                "--device-id",
                "0",
                "--nvtx-text",
                "%sample_0%",
                "--include-metrics",
                "--metric-name-like",
                "%active%",
                "--metrics-limit",
                "10000",
            ]
        )
        == 0
    )
    assert timeline_nvtx_html.exists()
    timeline_text = timeline_nvtx_html.read_text(encoding="utf-8")
    assert "GPU Metrics In Window" in timeline_text
    assert "Kernel Timeline By Stream" in timeline_text
    assert "stream-track" in timeline_text
    assert "overlay_metrics_per_track" in timeline_text
    assert "Kernel Theoretical Occupancy Sum [%]" in timeline_text
    assert "occ_theoretical_pct=" in timeline_text
    assert "sample_0 step=1 rank=0" in timeline_text
    assert "sample_0 step=2 rank=0" in timeline_text

    assert (
        main(
            [
                "nsys-timeline-compare-html",
                "--sqlite",
                str(db_a),
                "--sqlite",
                str(db_b),
                "--output",
                str(timeline_compare_html),
                "--device-id",
                "0",
                "--nvtx-text",
                "%sample_0%",
                "--include-metrics",
                "--metric-name-like",
                "%active%",
            ]
        )
        == 0
    )
    assert timeline_compare_html.exists()
    compare_text = timeline_compare_html.read_text(encoding="utf-8")
    assert "NSYS NVTX Timeline Compare" in compare_text
    assert compare_text.count("<iframe") == 8, compare_text
    assert "Optimization Summary" in compare_text
    assert "Pairwise Delta Summary" in compare_text
    assert "All Streams Overlap + Metrics Alignment" in compare_text
    assert "Matched NVTX Scopes" in compare_text
    assert "sample_0 step=1 rank=1" in timeline_text
    assert "nvtx_scopes=3" in timeline_text
    assert "Rank 0 | Device 0" in timeline_text
    assert "Rank 1 | Device 0" in timeline_text
    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", timeline_text, flags=re.S)
    assert m is not None
    payload = json.loads(m.group(1))
    all_groups = payload.get("all_stream_groups") or []
    assert all_groups, payload
    first_group = all_groups[0]
    stream_rows = first_group.get("streams") or []
    assert stream_rows, first_group
    kernels = stream_rows[0].get("kernels") or []
    assert kernels, stream_rows[0]
    assert "occupancy_pct_estimate" in kernels[0], kernels[0]


def test_cli_nsys_module_kernel_compare_json_and_markdown(tmp_path: Path) -> None:
    base_json = tmp_path / "base_module_kernels.json"
    target_json = tmp_path / "target_module_kernels.json"
    out_json = tmp_path / "module_compare.json"
    out_md = tmp_path / "module_compare.md"
    out_html = tmp_path / "module_compare.html"

    base_rows = [
        {
            "nvtx_text": "dual_stream_wan_layer_22",
            "kernel_name": "gemm_A",
            "kind": "compute",
            "kernel_start_ns": 1000,
            "kernel_end_ns": 1600,
            "duration_ms": 0.6,
            "stream_id": 83,
            "device_id": 3,
            "threads_per_block": 128,
            "total_blocks": 120,
            "registersPerThread": 32,
            "total_shared_bytes": 0,
            "occupancy_pct_h100_estimate": 100.0,
        },
        {
            "nvtx_text": "dual_stream_wan_layer_22",
            "kernel_name": "flash_attn_fwd",
            "kind": "compute",
            "kernel_start_ns": 1800,
            "kernel_end_ns": 4200,
            "duration_ms": 2.4,
            "stream_id": 83,
            "device_id": 3,
            "threads_per_block": 384,
            "total_blocks": 132,
            "registersPerThread": 168,
            "total_shared_bytes": 216064,
            "occupancy_pct_h100_estimate": 18.8,
        },
        {
            "nvtx_text": "dual_stream_wan_layer_22",
            "kernel_name": "nccl_allgather",
            "kind": "comm",
            "kernel_start_ns": 1400,
            "kernel_end_ns": 3900,
            "duration_ms": 2.5,
            "stream_id": 40,
            "device_id": 3,
            "threads_per_block": 640,
            "total_blocks": 24,
            "registersPerThread": 96,
            "total_shared_bytes": 103808,
            "occupancy_pct_h100_estimate": 26.6,
        },
        {
            "nvtx_text": "other_nvtx_scope",
            "kernel_name": "should_be_filtered",
            "kind": "compute",
            "kernel_start_ns": 100,
            "kernel_end_ns": 200,
            "duration_ms": 0.1,
            "stream_id": 11,
            "device_id": 3,
        },
    ]
    target_rows = [
        {
            "nvtx_text": "dual_stream_wan_layer_22",
            "kernel_name": "gemm_A",
            "kind": "compute",
            "kernel_start_ns": 1000,
            "kernel_end_ns": 1750,
            "duration_ms": 0.75,
            "stream_id": 83,
            "device_id": 3,
            "threads_per_block": 128,
            "total_blocks": 150,
            "registersPerThread": 40,
            "total_shared_bytes": 0,
            "occupancy_pct_h100_estimate": 100.0,
        },
        {
            "nvtx_text": "dual_stream_wan_layer_22",
            "kernel_name": "flash_attn_fwd",
            "kind": "compute",
            "kernel_start_ns": 1820,
            "kernel_end_ns": 3520,
            "duration_ms": 1.7,
            "stream_id": 83,
            "device_id": 3,
            "threads_per_block": 384,
            "total_blocks": 132,
            "registersPerThread": 168,
            "total_shared_bytes": 216064,
            "occupancy_pct_h100_estimate": 18.8,
        },
        {
            "nvtx_text": "dual_stream_wan_layer_22",
            "kernel_name": "gelu_fused",
            "kind": "compute",
            "kernel_start_ns": 3550,
            "kernel_end_ns": 3950,
            "duration_ms": 0.4,
            "stream_id": 83,
            "device_id": 3,
            "threads_per_block": 128,
            "total_blocks": 90,
            "registersPerThread": 32,
            "total_shared_bytes": 0,
            "occupancy_pct_h100_estimate": 100.0,
        },
        {
            "nvtx_text": "dual_stream_wan_layer_22",
            "kernel_name": "nccl_allgather",
            "kind": "comm",
            "kernel_start_ns": 1450,
            "kernel_end_ns": 3150,
            "duration_ms": 1.7,
            "stream_id": 40,
            "device_id": 3,
            "threads_per_block": 640,
            "total_blocks": 24,
            "registersPerThread": 96,
            "total_shared_bytes": 103808,
            "occupancy_pct_h100_estimate": 26.6,
        },
    ]
    base_json.write_text(
        json.dumps(base_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    target_json.write_text(
        json.dumps(target_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rc_json = main(
        [
            "nsys-module-kernel-compare",
            "--base-json",
            str(base_json),
            "--target-json",
            str(target_json),
            "--nvtx-text",
            "dual_stream_wan_layer_22",
            "--device-id",
            "3",
            "--output",
            str(out_json),
            "--pretty",
        ]
    )
    assert rc_json == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    compare = payload.get("compare") or {}
    kernel_set = compare.get("kernel_set_diff") or {}
    assert "gelu_fused" in set(kernel_set.get("added") or [])
    kernel_resource_diff = compare.get("kernel_resource_diff") or {}
    assert int(kernel_resource_diff.get("changed_kernel_count", 0)) >= 1
    changed_rows = kernel_resource_diff.get("changed_kernels") or []
    gemm_change = next(
        (x for x in changed_rows if str((x or {}).get("kernel_name")) == "gemm_A"), None
    )
    assert gemm_change is not None
    assert "registers_per_thread" in set(gemm_change.get("changed_keys") or [])
    assert "top_kernel_duration_deltas" in compare
    stream_deltas = compare.get("stream_deltas") or []
    assert stream_deltas
    stream83 = next(
        (x for x in stream_deltas if int((x or {}).get("stream_id", -1)) == 83), None
    )
    assert stream83 is not None
    assert float(stream83.get("sequence_similarity", 0.0)) < 1.0
    assert isinstance(stream83.get("base_timeline_sample"), list)
    assert isinstance(stream83.get("target_timeline_sample"), list)

    rc_md = main(
        [
            "nsys-module-kernel-compare",
            "--base-json",
            str(base_json),
            "--target-json",
            str(target_json),
            "--nvtx-text",
            "dual_stream_wan_layer_22",
            "--device-id",
            "3",
            "--format",
            "markdown",
            "--output",
            str(out_md),
        ]
    )
    assert rc_md == 0
    md_text = out_md.read_text(encoding="utf-8")
    assert "NSYS Module Kernel Compare" in md_text
    assert "Same-Kernel Resource Diff" in md_text
    assert "Top Kernel Duration Deltas" in md_text
    assert "Stream 83" in md_text
    assert "kernel_set_added_list" in md_text
    assert "gelu_fused" in md_text

    rc_html = main(
        [
            "nsys-module-kernel-compare",
            "--base-json",
            str(base_json),
            "--target-json",
            str(target_json),
            "--nvtx-text",
            "dual_stream_wan_layer_22",
            "--device-id",
            "3",
            "--format",
            "html",
            "--output",
            str(out_html),
        ]
    )
    assert rc_html == 0
    assert out_html.exists()
    html_text = out_html.read_text(encoding="utf-8")
    assert "NSYS Module Kernel Compare" in html_text
    assert "Same-Kernel Resource Diff" in html_text
    assert "Top Kernel Duration Deltas" in html_text
    assert "Stream 83" in html_text
    assert "gelu_fused" in html_text
    assert (
        "possible_fusion_in_target" in html_text
        or "possible_split_in_target" in html_text
        or "kernel_set_changed" in html_text
    )


def test_cli_nsys_module_kernel_compare_sqlite_mode(tmp_path: Path) -> None:
    db_a = tmp_path / "module_compare_base.sqlite"
    db_b = tmp_path / "module_compare_target.sqlite"
    out_json = tmp_path / "module_compare_from_sqlite.json"
    _init_sqlite(db_a, scale=1.0)
    _init_sqlite(db_b, scale=1.25)

    rc = main(
        [
            "nsys-module-kernel-compare",
            "--base-sqlite",
            str(db_a),
            "--target-sqlite",
            str(db_b),
            "--nvtx-text",
            "%sample_0%",
            "--nvtx-index",
            "1",
            "--device-id",
            "0",
            "--sqlite-limit",
            "200000",
            "--format",
            "json",
            "--output",
            str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    base = dict(payload.get("base") or {})
    target = dict(payload.get("target") or {})
    compare = dict(payload.get("compare") or {})
    assert str(base.get("source_path") or "") == str(db_a)
    base_scope_sel = dict(base.get("nvtx_scope_selection") or {})
    target_scope_sel = dict(target.get("nvtx_scope_selection") or {})
    base_scope = dict(base_scope_sel.get("selected_scope") or {})
    target_scope = dict(target_scope_sel.get("selected_scope") or {})
    assert int(base_scope_sel.get("requested_nvtx_index") or -1) == 1
    assert int(target_scope_sel.get("requested_nvtx_index") or -1) == 1
    assert "sample_0 step=2 rank=0" in str(base_scope.get("nvtx_text") or ""), (
        base_scope
    )
    assert "sample_0 step=2 rank=0" in str(target_scope.get("nvtx_text") or ""), (
        target_scope
    )
    stream_deltas = list(compare.get("stream_deltas") or [])
    assert stream_deltas, compare
    first_stream = dict(stream_deltas[0] or {})
    assert "change_hint" in first_stream
    base_timeline = list(first_stream.get("base_timeline_sample") or [])
    assert base_timeline, first_stream
    assert "registers_per_thread" in dict(base_timeline[0] or {})
    assert "total_shared_bytes" in dict(base_timeline[0] or {})

    rc_mix = main(
        [
            "nsys-module-kernel-compare",
            "--base-json",
            "dummy.json",
            "--base-sqlite",
            str(db_a),
            "--target-sqlite",
            str(db_b),
            "--output",
            str(tmp_path / "should_not_exist.json"),
        ]
    )
    assert rc_mix == 2
