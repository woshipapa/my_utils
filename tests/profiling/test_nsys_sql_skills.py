# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.sources.nsys_sql_skills."""

from __future__ import annotations


import sqlite3
from pathlib import Path
from typing import List, Tuple


from my_utils.profiling.sources.nsys_sql_skills import (
    NsysSqlSkillEngine,
    calculate_h100_occupancy,
)


from my_utils.profiling.sources.nsys_timeline_html import _collect_metric_samples


from _synthetic_loader import _init_sqlite, _show


def test_all_skills_register_and_execute(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    skills = engine.list_skills()
    print(f"\n[skills] registered: {len(skills)}")
    for s in skills:
        print(f"  - {s}")

    EXPECTED_21 = {
        "aggregate_kernels",
        "top_kernels",
        "aggregate_nvtx_ranges",
        "memcpy_in_window",
        "kernel_map",
        "gpu_idle_gaps",
        "kernel_launch_overhead",
        "nccl_breakdown",
        "nvtx_kernel_map",
        "schema_inspect",
        "gpu_metrics_aggregate",
        "thread_utilization",
        "memcpy_bandwidth_analysis",
        "sync_breakdown",
        "memset_breakdown",
        "kernel_occupancy_estimate",
        "stream_parallelism",
        "nvtx_memcpy_breakdown",
        "nvtx_gpu_metrics_breakdown",
        "nvtx_kernel_sm_detail",
        "nvtx_ranges_hierarchy",
    }
    missing = EXPECTED_21 - set(skills)
    assert not missing, f"Missing skills: {missing}"

    # Default params for skills that have required parameters
    REQUIRED_DEFAULTS: dict = {
        "nvtx_kernel_sm_detail": {"nvtx_text": "%"},
    }

    errors: List[Tuple[str, str]] = []
    for name in skills:
        try:
            kwargs = REQUIRED_DEFAULTS.get(name, {})
            rows = engine.execute(name, **kwargs)
            print(f"  [{name}] OK  rows={len(rows)}")
        except Exception as exc:
            errors.append((name, str(exc)))
            print(f"  [{name}] ERROR: {exc}")

    conn.close()
    assert not errors, f"Skills raised errors: {errors}"


def test_new_skill_outputs(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    rows = engine.execute(
        "gpu_metrics_aggregate", metric_name_like="%active%", start_ns=-1, end_ns=-1
    )
    _show("Skill 11 - gpu_metrics_aggregate", rows)
    assert len(rows) >= 2
    names = {r["metric_name"] for r in rows}
    assert any("sm__active" in n for n in names)
    assert any("tensor__active" in n for n in names)
    for r in rows:
        assert "sample_count" in r
        assert "avg_value" in r
        assert "min_value" in r
        assert "max_value" in r
    # Default source filter should exclude non-GpuMetrics rows.
    sm_row = next(
        (r for r in rows if "sm__active" in str(r.get("metric_name", ""))), None
    )
    assert sm_row is not None
    assert sm_row["sample_count"] == 2

    rows_all_sources = engine.execute(
        "gpu_metrics_aggregate",
        metric_name_like="%active%",
        start_ns=-1,
        end_ns=-1,
        include_all_sources=1,
    )
    sm_rows_all = [
        r for r in rows_all_sources if "sm__active" in str(r.get("metric_name", ""))
    ]
    assert sm_rows_all
    assert sum(int(r.get("sample_count") or 0) for r in sm_rows_all) == 3

    rows_nvtx_metrics = engine.execute(
        "nvtx_gpu_metrics_breakdown",
        nvtx_text="%sample_0%",
        metric_name_like="%active%",
        include_all_sources=0,
        limit=100,
    )
    _show("Skill 18 - nvtx_gpu_metrics_breakdown", rows_nvtx_metrics)
    assert len(rows_nvtx_metrics) >= 2
    r0 = rows_nvtx_metrics[0]
    assert "nvtx_text" in r0
    assert "metric_name" in r0
    assert "sample_count" in r0
    assert "avg_value" in r0
    assert "min_value" in r0
    assert "max_value" in r0
    sm_nvtx = next(
        (r for r in rows_nvtx_metrics if "sm__active" in str(r.get("metric_name", ""))),
        None,
    )
    assert sm_nvtx is not None
    assert sm_nvtx["sample_count"] == 2

    # Skill 6: kernel_launch_overhead should include API name for attribution.
    rows_launch = engine.execute("kernel_launch_overhead", device_id=0, limit=20)
    _show("Skill 6 - kernel_launch_overhead", rows_launch)
    assert len(rows_launch) >= 1
    assert "api_name" in rows_launch[0]
    assert rows_launch[0]["api_name"] is not None
    assert "api_ms" in rows_launch[0]
    assert "kernel_ms" in rows_launch[0]
    assert "overhead_us" in rows_launch[0]

    # 鈹€鈹€ Skill 12: memcpy_bandwidth_analysis 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows = engine.execute("memcpy_bandwidth_analysis", device_id=0)
    _show("Skill 12 鈥?memcpy_bandwidth_analysis", rows)
    assert len(rows) == 3, f"expected 3 copyKind groups, got {len(rows)}"
    for r in rows:
        assert "copy_kind" in r
        assert "avg_gbps" in r
        assert r["avg_gbps"] is not None and r["avg_gbps"] > 0

    # 鈹€鈹€ Skill 13: sync_breakdown 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows = engine.execute("sync_breakdown", device_id=0)
    _show("Skill 13 鈥?sync_breakdown", rows)
    assert len(rows) >= 1
    assert "sync_type" in rows[0]
    assert "total_ms" in rows[0]
    total_sync_ms = sum(r["total_ms"] for r in rows)
    assert total_sync_ms > 0

    # 鈹€鈹€ Skill 14: memset_breakdown 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows = engine.execute("memset_breakdown", device_id=0)
    _show("Skill 14 鈥?memset_breakdown", rows)
    assert len(rows) >= 1
    assert "fill_value" in rows[0]
    assert "total_gb" in rows[0]
    # zero-init bytes = 8MB + 1MB = 9MB
    zero_rows = [r for r in rows if r["fill_value"] == 0]
    assert zero_rows, "Expected a zero-init (fill_value=0) row"
    assert abs(zero_rows[0]["total_gb"] - 9 / 1024) < 0.001

    # 鈹€鈹€ Skill 15: kernel_occupancy_estimate (raw metrics) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows = engine.execute("kernel_occupancy_estimate", device_id=0, limit=10)
    _show("Skill 15 鈥?kernel_occupancy_estimate", rows)
    assert len(rows) >= 1
    assert "threads_per_block" in rows[0]
    assert "registersPerThread" in rows[0]
    assert "static_shared_bytes" in rows[0]
    assert "dynamic_shared_bytes" in rows[0]
    assert "total_shared_bytes" in rows[0]
    assert "occupancy_pct_estimate" in rows[0]
    # sqlite theoretical occupancy is returned when available.
    assert all((r["occupancy_pct_estimate"] is not None) for r in rows)

    rows_occ_h100 = engine.execute_kernel_occupancy_estimate_h100(device_id=0, limit=10)
    _show("Skill 15 + H100 occupancy", rows_occ_h100)
    assert len(rows_occ_h100) == len(rows)
    assert "occupancy_pct_estimate" in rows_occ_h100[0]
    assert rows_occ_h100[0]["occupancy_pct_estimate"] is not None
    assert "occupancy_pct_h100_estimate" in rows_occ_h100[0]
    assert rows_occ_h100[0]["occupancy_pct_h100_estimate"] is not None

    # 鈹€鈹€ Skill 16: stream_parallelism 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows = engine.execute("stream_parallelism", device_id=0, bucket_ns=5000)
    _show("Skill 16 鈥?stream_parallelism", rows)
    assert len(rows) == 1  # single aggregate row
    r = rows[0]
    assert "max_concurrent_streams" in r
    assert "pct_time_multi_stream" in r
    assert r["max_concurrent_streams"] >= 2  # we have 3 streams
    # With cross-bucket expansion, long kernels should contribute beyond start bucket.
    assert r["total_buckets"] >= 5

    # 鈹€鈹€ Skill 17: nvtx_memcpy_breakdown 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows = engine.execute("nvtx_memcpy_breakdown", limit=20)
    _show("Skill 17 鈥?nvtx_memcpy_breakdown", rows)
    # memcpy rows fall inside NVTX ranges (forward/backward/step ranges)
    assert len(rows) >= 1
    assert "nvtx_text" in rows[0]
    assert "total_gb" in rows[0]

    # 鈹€鈹€ Skill 18: nvtx_kernel_sm_detail 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # sample_0 step=1 range [0, 22000] launch-attributed kernels by runtime->correlationId
    rows = engine.execute("nvtx_kernel_sm_detail", nvtx_text="%sample_0%", device_id=0)
    _show("Skill 18 鈥?nvtx_kernel_sm_detail (%sample_0%)", rows)
    assert len(rows) >= 1, "Expected launch-attributed kernels in sample_0 NVTX range"
    r = rows[0]
    assert "nvtx_text" in r
    assert "kernel_name" in r
    assert "kind" in r
    assert r["kind"] in ("compute", "comm")
    assert "duration_ms" in r
    assert "threads_per_block" in r
    assert "static_shared_bytes" in r
    assert "dynamic_shared_bytes" in r
    assert "total_shared_bytes" in r
    assert "occupancy_pct_estimate" in r
    assert r["occupancy_pct_estimate"] is not None

    # Python-side H100 occupancy estimation
    rows_h100 = engine.execute_nvtx_kernel_sm_detail_h100(
        nvtx_text="%sample_0%", device_id=0
    )
    _show("Skill 18 + H100 occupancy", rows_h100)
    assert len(rows_h100) == len(rows)
    assert "occupancy_pct_estimate" in rows_h100[0]
    assert rows_h100[0]["occupancy_pct_estimate"] is not None
    assert "occupancy_pct_h100_estimate" in rows_h100[0]
    assert rows_h100[0]["occupancy_pct_h100_estimate"] is not None

    # kind labelling: nccl kernel must be 'comm', others 'compute'
    kinds = {r["kernel_name"]: r["kind"] for r in rows}
    for name, kind in kinds.items():
        if "nccl" in name.lower():
            assert kind == "comm", f"nccl kernel '{name}' should be labelled 'comm'"
        else:
            assert kind == "compute", f"non-nccl kernel '{name}' should be 'compute'"

    # forward-only filter: based on runtime launch in NVTX window, not kernel end-time containment.
    # In fixture, NCCL launch runtime [7000,9000] is inside forward [0,10000], so comm can appear.
    rows_fwd = engine.execute(
        "nvtx_kernel_sm_detail", nvtx_text="%forward%", device_id=0
    )
    _show("Skill 18 鈥?nvtx_kernel_sm_detail (%forward%)", rows_fwd)
    assert len(rows_fwd) >= 1
    assert all((r["nvtx_text"] == "forward") for r in rows_fwd)
    assert any((r["kind"] == "comm") for r in rows_fwd)

    # 鈹€鈹€ Skill 19: nvtx_ranges_hierarchy 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows_all_nvtx = engine.execute(
        "nvtx_ranges_hierarchy", nvtx_text="%", top_level_only=False, limit=100
    )
    _show("Skill 19 鈥?nvtx_ranges_hierarchy (all)", rows_all_nvtx)
    assert len(rows_all_nvtx) >= 4
    by_text = {r["nvtx_text"]: r for r in rows_all_nvtx}
    assert "sample_0 step=1 rank=0" in by_text
    assert "forward" in by_text
    assert "backward" in by_text
    # nested ranges should point to sample_0 step=1
    assert by_text["forward"]["parent_nvtx_text"] == "sample_0 step=1 rank=0"
    assert by_text["backward"]["parent_nvtx_text"] == "sample_0 step=1 rank=0"
    assert by_text["forward"]["depth"] >= 1
    assert by_text["backward"]["depth"] >= 1

    rows_root_nvtx = engine.execute(
        "nvtx_ranges_hierarchy", nvtx_text="%", top_level_only=True, limit=100
    )
    _show("Skill 19 鈥?nvtx_ranges_hierarchy (top-level)", rows_root_nvtx)
    assert len(rows_root_nvtx) >= 1
    assert all((r["depth"] == 0) for r in rows_root_nvtx)

    conn.close()


def test_nvtx_gpu_metrics_breakdown_uses_correlation_gpu_windows(
    tmp_path: Path,
) -> None:
    db = tmp_path / "nvtx_gpu_window.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        (100_000, 100_200, "late_launch", None, 59, 12345678),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        (100_100, 100_180, 999, 3, 12345678),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (101_000, 101_500, 7, 999, 1, 1, 0, 128, 1, 1, 32, 4096, 0, 87.5),
    )
    # Outside NVTX CPU wall window, inside GPU kernel execution window.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        (101_100, 101, 77.0, 1),
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    rows = engine.execute(
        "nvtx_gpu_metrics_breakdown",
        nvtx_text="%late_launch%",
        metric_name_like="%sm__active%",
        include_all_sources=1,
        limit=100,
    )
    _show("Skill 18 - correlation mapped gpu window", rows)
    assert rows, rows
    sm_rows = [r for r in rows if "sm__active" in str(r.get("metric_name", ""))]
    assert sm_rows, rows
    sm = sm_rows[0]
    assert int(sm.get("sample_count") or 0) == 1, sm
    assert abs(float(sm.get("avg_value") or 0.0) - 77.0) < 1e-6, sm
    assert int(sm.get("nvtx_start_ns") or 0) == 100_000, sm
    assert int(sm.get("nvtx_end_ns") or 0) == 100_200, sm
    conn.close()


def test_nvtx_gpu_metrics_breakdown_overlapping_kernel_windows_no_double_count(
    tmp_path: Path,
) -> None:
    db = tmp_path / "nvtx_gpu_overlap_dedup.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        (1_000_000, 1_000_300, "overlap_case", None, 59, 12345678),
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (1_000_050, 1_000_080, 7001, 3, 12345678),
            (1_000_090, 1_000_120, 7002, 3, 12345678),
        ],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (1_001_000, 1_001_600, 7, 7001, 1, 1, 0, 128, 1, 1, 32, 4096, 0, 80.0),
            (1_001_400, 1_001_900, 8, 7002, 1, 1, 0, 128, 1, 1, 32, 4096, 0, 80.0),
        ],
    )
    # This point falls into both raw kernel windows; result must count it only once.
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        (1_001_500, 101, 66.0, 1),
    )
    conn.commit()
    conn.row_factory = sqlite3.Row

    engine = NsysSqlSkillEngine(conn)
    rows = engine.execute(
        "nvtx_gpu_metrics_breakdown",
        nvtx_text="%overlap_case%",
        metric_name_like="%sm__active%",
        include_all_sources=1,
        limit=100,
    )
    _show("Skill 18 - overlap windows dedup", rows)
    assert rows, rows
    sm = next((r for r in rows if "sm__active" in str(r.get("metric_name", ""))), None)
    assert sm is not None, rows
    assert int(sm.get("sample_count") or 0) == 1, sm
    assert abs(float(sm.get("avg_value") or 0.0) - 66.0) < 1e-6, sm
    conn.close()


def test_gpu_metric_name_mapping_prefers_target_info(tmp_path: Path) -> None:
    db = tmp_path / "rank0_target_info.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE TARGET_INFO_GPU_METRICS ("
        "typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT)"
    )
    cur.executemany(
        "INSERT INTO TARGET_INFO_GPU_METRICS(typeId, sourceId, typeName, metricId, metricName) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, "float", 101, "sm__active.avg.pct_of_peak_sustained_elapsed"),
            (1, 1, "float", 102, "tensor__active.avg.pct_of_peak_sustained_elapsed"),
            (1, 1, "float", 103, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ],
    )
    # Inject a conflicting StringIds mapping. TARGET_INFO names should still win.
    cur.execute("UPDATE StringIds SET value = 'BAD_STRING_METRIC_NAME' WHERE id = 101")
    conn.commit()
    conn.close()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)
    rows = engine.execute(
        "gpu_metrics_aggregate",
        metric_name_like="%",
        include_all_sources=0,
        start_ns=-1,
        end_ns=-1,
    )
    names = {str(r.get("metric_name", "")) for r in rows}
    assert any("sm__active" in n for n in names), names
    assert all("BAD_STRING_METRIC_NAME" not in n for n in names), names
    conn.close()

    timeline_rows = _collect_metric_samples(
        str(db),
        start_ns=-1,
        end_ns=10_000_000,
        metric_name_like="%",
        include_all_sources=False,
        device_id=-1,
        limit=10000,
    )
    tnames = {str(r.get("name", "")) for r in timeline_rows}
    assert any("sm__active" in n for n in tnames), tnames
    assert all("BAD_STRING_METRIC_NAME" not in n for n in tnames), tnames


def test_gpu_metrics_source_nameid_chain_maps_device(tmp_path: Path) -> None:
    db = tmp_path / "metrics_source_nameid_chain.sqlite"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (101, "sm__active.avg.pct_of_peak_sustained_elapsed"),
            (9001, "GPU 2 Metrics"),
        ],
    )
    cur.execute(
        "CREATE TABLE GPU_METRICS ("
        "timestamp INTEGER, metricId INTEGER, typeId INTEGER, value REAL)"
    )
    cur.execute(
        "CREATE TABLE TARGET_INFO_GPU_METRICS ("
        "metricId INTEGER, typeId INTEGER, sourceId INTEGER, metricName TEXT)"
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, [end] INTEGER)"
    )
    cur.execute("CREATE TABLE GENERIC_EVENT_SOURCES (sourceId INTEGER, nameId INTEGER)")
    cur.execute(
        "INSERT INTO TARGET_INFO_GPU_METRICS(metricId, typeId, sourceId, metricName) VALUES (?, ?, ?, ?)",
        (101, 1, 7, "sm__active.avg.pct_of_peak_sustained_elapsed"),
    )
    cur.execute(
        "INSERT INTO GENERIC_EVENT_SOURCES(sourceId, nameId) VALUES (?, ?)",
        (7, 9001),
    )
    cur.executemany(
        "INSERT INTO GPU_METRICS(timestamp, metricId, typeId, value) VALUES (?, ?, ?, ?)",
        [
            (1000, 101, 1, 50.0),
            (2000, 101, 1, 70.0),
        ],
    )
    conn.commit()
    conn.close()

    timeline_rows = _collect_metric_samples(
        str(db),
        start_ns=0,
        end_ns=10_000,
        metric_name_like="%active%",
        include_all_sources=False,
        device_id=-1,
        limit=-1,
        max_points_per_series=-1,
    )
    assert timeline_rows, "expected timeline metric rows"
    timeline_names = {str(r.get("name", "")) for r in timeline_rows}
    assert any("[gpu 2]" in n for n in timeline_names), timeline_names

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)
    rows = engine.execute(
        "gpu_metrics_aggregate",
        metric_name_like="%active%",
        include_all_sources=0,
        start_ns=0,
        end_ns=10_000,
        device_id=-1,
    )
    conn.close()
    assert rows, "expected gpu_metrics_aggregate rows"
    devices = {str(r.get("metric_device", "")) for r in rows}
    assert any("GPU 2 Metrics" in d for d in devices), devices


def test_gpu_metrics_split_by_device_dimension(tmp_path: Path) -> None:
    db = tmp_path / "rank0_multi_device.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute(
        "ALTER TABLE CUPTI_ACTIVITY_KIND_GPU_METRIC ADD COLUMN deviceId INTEGER"
    )
    cur.execute("UPDATE CUPTI_ACTIVITY_KIND_GPU_METRIC SET deviceId = 0")
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId, deviceId) VALUES (?, ?, ?, ?, ?)",
        [
            (7000, 101, 55.0, 1, 1),
            (8000, 101, 65.0, 1, 1),
            (9000, 102, 35.0, 1, 1),
        ],
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)
    rows = engine.execute(
        "gpu_metrics_aggregate",
        metric_name_like="%active%",
        include_all_sources=0,
        device_id=-1,
        start_ns=-1,
        end_ns=-1,
    )
    conn.close()
    dev_set = {str(r.get("metric_device", "")) for r in rows}
    assert "0" in dev_set and "1" in dev_set, rows

    timeline_rows = _collect_metric_samples(
        str(db),
        start_ns=-1,
        end_ns=20_000,
        metric_name_like="%active%",
        include_all_sources=False,
        device_id=-1,
        limit=1000,
    )
    names = {str(r.get("name", "")) for r in timeline_rows}
    assert any("[gpu 0]" in n for n in names), names
    assert any("[gpu 1]" in n for n in names), names


def test_calculate_h100_occupancy() -> None:
    # 128 threads/block -> 4 warps/block.
    # regs=32 => regs_per_warp=1024, regs/block=4096 => reg-limited blocks=16
    # smem=4096 => smem-limited blocks=57
    # thread-limited blocks=16, final active_blocks=min(16,16,57,32)=16
    # occupancy=16*4/64*100 = 100.0
    occ = calculate_h100_occupancy(128, 32, 4096)
    assert occ == 100.0

    # Invalid threads_per_block -> None
    assert calculate_h100_occupancy(None, 32, 0) is None
    assert calculate_h100_occupancy(0, 32, 0) is None

    # Very high registers/thread should make occupancy collapse to 0 on H100 rule.
    occ_zero = calculate_h100_occupancy(256, 512, 0)
    assert occ_zero == 0.0


def test_new_engine_methods(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    # 鈹€鈹€ analyze_per_iteration_overlap 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    per_iter = engine.analyze_per_iteration_overlap(
        marker="sample_0", device_id=0, top_level_only=True, limit=100
    )
    _show("Engine: analyze_per_iteration_overlap", per_iter)
    assert isinstance(per_iter, list)
    assert len(per_iter) >= 1
    for entry in per_iter:
        assert "compute_ms" in entry
        assert "comm_ms" in entry
        assert "overlap_ms" in entry
        assert "comm_pct" in entry
        assert "kernel_count" in entry
        assert entry["kernel_count"] >= 0
        assert 0.0 <= entry["comm_pct"] <= 100.0

    # 鈹€鈹€ detect_iteration_outliers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    result = engine.detect_iteration_outliers(
        marker="sample_0",
        device_id=0,
        threshold_sigma=0.0,  # sigma=0 鈫?all are outliers
    )
    _show("Engine: detect_iteration_outliers (sigma=0)", result)
    assert "stats" in result
    assert "outliers" in result
    stats = result["stats"]
    assert "count" in stats and stats["count"] >= 1
    assert "mean_ms" in stats
    assert "median_ms" in stats
    assert "std_ms" in stats
    assert "p95_ms" in stats
    assert "p99_ms" in stats
    # with 2 iterations and sigma=0, both should be flagged
    assert len(result["outliers"]) >= 1
    for o in result["outliers"]:
        assert "iteration" in o
        assert "duration_ms" in o
        assert "deviation_sigma" in o

    conn.close()
