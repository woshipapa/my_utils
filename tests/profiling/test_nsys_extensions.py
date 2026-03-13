from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple

from my_utils.profiling.cli import main
from my_utils.profiling.sources.nsys_sql_skills import (
    NsysSqlSkillEngine,
    calculate_h100_occupancy,
)
from my_utils.profiling.sources.nsys_timeline_html import (
    _collect_kernels_in_window,
    _collect_metric_samples,
    export_timeline_html,
)
from my_utils.profiling.sources.nsys_sqlite_provider import NsysSqliteMetricsProvider


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _init_sqlite(path: Path, *, scale: float = 1.0) -> None:
    """Build a minimal but complete nsys-like SQLite covering all current skills."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # 鈹€鈹€ meta 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    cur.executemany(
        "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
        [
            ("NSIGHT_SYSTEMS_VERSION", "2024.7.1"),
            ("EXPORT_SCHEMA_VERSION", "3.15.1"),
        ],
    )

    # 鈹€鈹€ string table 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (1, "void gemm_kernel()"),
            (2, "ncclAllReduceRingLLKernel_sum_f16"),
            (3, "cudaLaunchKernel"),
            (4, "worker_main"),
            (5, "void attention_kernel()"),
            (101, "sm__active.avg.pct_of_peak_sustained_elapsed"),
            (102, "tensor__active.avg.pct_of_peak_sustained_elapsed"),
            (103, "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ],
    )

    # 鈹€鈹€ kernel table (includes block/register columns for skill 15) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER, [end] INTEGER, streamId INTEGER, correlationId INTEGER, "
        "shortName INTEGER, demangledName INTEGER, deviceId INTEGER, "
        "blockX INTEGER, blockY INTEGER, blockZ INTEGER, "
        "registersPerThread INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, "
        "theoreticalOccupancyPct REAL)"
    )
    # rows: (start, end, stream, corr, short, demangled, dev, bx, by, bz, regs, static_smem, dyn_smem, theoretical_occupancy_pct)
    s = scale
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (0,           int(10000*s), 7, 1, 1, 1, 0, 128, 1, 1, 32, 4096, 0,    87.5),  # gemm stream7
            (int(8000*s), int(20000*s), 8, 2, 2, 2, 0, 256, 1, 1, 40, 0,    0,    62.5),  # nccl  stream8
            (int(25000*s),int(35000*s), 7, 3, 1, 1, 0, 128, 1, 1, 32, 4096, 0,    87.5),  # gemm stream7
            (int(5000*s), int(12000*s), 9, 4, 5, 5, 0,  64, 1, 1, 48, 8192, 2048, 50.0),  # attention stream9
        ],
    )

    # 鈹€鈹€ runtime table 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME ("
        "start INTEGER, [end] INTEGER, correlationId INTEGER, nameId INTEGER, globalTid INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (0,              int(2000*s),  1, 3, 12345678),
            (int(7000*s),    int(9000*s),  2, 3, 12345678),
            (int(24000*s),   int(24500*s), 3, 3, 12345678),
            (int(4500*s),    int(5000*s),  4, 3, 12345678),
        ],
    )

    # 鈹€鈹€ NVTX events 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE NVTX_EVENTS ("
        "start INTEGER, [end] INTEGER, text TEXT, textId INTEGER, eventType INTEGER, globalTid INTEGER)"
    )
    cur.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0,              int(22000*s), "sample_0 step=1 rank=0", None, 59, 12345678),
            (int(23000*s),   int(36000*s), "sample_0 step=2 rank=0", None, 59, 12345678),
            (0,              int(10000*s), "forward",                None, 59, 12345678),
            (int(10000*s),   int(20000*s), "backward",               None, 59, 12345678),
        ],
    )

    # 鈹€鈹€ memcpy table (skill 4, 12, 17) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
        "(start INTEGER, [end] INTEGER, copyKind INTEGER, bytes INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?)",
        [
            (0,            int(1000*s), 1, int(1024*1024),   0),   # H2D 1 MB
            (int(3000*s),  int(6000*s), 2, int(2*1024*1024), 0),   # D2H 2 MB
            (int(12000*s), int(15000*s),8, int(4*1024*1024), 0),   # D2D 4 MB
        ],
    )

    # 鈹€鈹€ memset table (skill 14) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET "
        "(start INTEGER, [end] INTEGER, bytes INTEGER, value INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?, ?, ?, ?)",
        [
            (0,           int(500*s),  int(8*1024*1024), 0, 0),   # zero-init 8 MB
            (int(500*s),  int(600*s),  int(1024*1024),   0, 0),   # zero-init 1 MB
            (int(1000*s), int(1100*s), int(512*1024),    1, 0),   # custom fill
        ],
    )

    # 鈹€鈹€ synchronization table (skill 13) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
        "(start INTEGER, [end] INTEGER, syncType INTEGER, streamId INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES (?, ?, ?, ?, ?)",
        [
            (int(20000*s), int(21000*s), 1, 7, 0),   # cudaStreamSync
            (int(35000*s), int(35500*s), 2, 0, 0),   # cudaDeviceSync
            (int(36000*s), int(36100*s), 1, 8, 0),   # cudaStreamSync
        ],
    )

    # 鈹€鈹€ CPU events (skill 11) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    cur.execute("CREATE TABLE COMPOSITE_EVENTS (globalTid INTEGER, cpuCycles INTEGER)")
    cur.executemany(
        "INSERT INTO COMPOSITE_EVENTS VALUES (?, ?)",
        [
            (12345678, int(1000*s)),
            (22345678, int(500*s)),
        ],
    )
    cur.execute("CREATE TABLE ThreadNames (globalTid INTEGER, nameId INTEGER)")
    cur.executemany(
        "INSERT INTO ThreadNames VALUES (?, ?)",
        [(12345678, 4), (22345678, 4)],
    )

    cur.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT)")
    cur.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H100')")
    cur.execute("CREATE TABLE GENERIC_EVENT_SOURCES (id INTEGER, name TEXT)")
    cur.executemany(
        "INSERT INTO GENERIC_EVENT_SOURCES VALUES (?, ?)",
        [
            (1, "GpuMetrics"),
            (2, "ETW"),
        ],
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_GPU_METRIC "
        "(timestamp INTEGER, metricId INTEGER, value REAL, sourceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC VALUES (?, ?, ?, ?)",
        [
            (int(1000 * s), 101, 62.5, 1),
            (int(2000 * s), 101, 70.0, 1),
            (int(3000 * s), 102, 41.0, 1),
            (int(4000 * s), 102, 44.5, 1),
            (int(5000 * s), 103, 57.25, 1),
            (int(6000 * s), 101, 99.0, 2),
        ],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helper: pretty-print a section header + rows
# ---------------------------------------------------------------------------

def _show(title: str, rows, *, limit: int = 5) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if isinstance(rows, list):
        for r in rows[:limit]:
            print(" ", r)
        if len(rows) > limit:
            print(f"  ... ({len(rows)} total)")
    elif isinstance(rows, dict):
        for k, v in rows.items():
            print(f"  {k}: {v}")
    else:
        print(" ", rows)


# ===========================================================================
# Test 1 鈥?all skills register and execute without error
# ===========================================================================

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
        "aggregate_kernels", "top_kernels", "aggregate_nvtx_ranges",
        "memcpy_in_window", "kernel_map", "gpu_idle_gaps",
        "kernel_launch_overhead", "nccl_breakdown", "nvtx_kernel_map",
        "schema_inspect", "gpu_metrics_aggregate", "thread_utilization",
        "memcpy_bandwidth_analysis", "sync_breakdown", "memset_breakdown",
        "kernel_occupancy_estimate", "stream_parallelism", "nvtx_memcpy_breakdown", "nvtx_gpu_metrics_breakdown",
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


# ===========================================================================
# Test 2 鈥?new skills return correct columns and values
# ===========================================================================

def test_new_skill_outputs(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    rows = engine.execute("gpu_metrics_aggregate", metric_name_like="%active%", start_ns=-1, end_ns=-1)
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
    sm_row = next((r for r in rows if "sm__active" in str(r.get("metric_name", ""))), None)
    assert sm_row is not None
    assert sm_row["sample_count"] == 2

    rows_all_sources = engine.execute(
        "gpu_metrics_aggregate",
        metric_name_like="%active%",
        start_ns=-1,
        end_ns=-1,
        include_all_sources=1,
    )
    sm_rows_all = [r for r in rows_all_sources if "sm__active" in str(r.get("metric_name", ""))]
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
    sm_nvtx = next((r for r in rows_nvtx_metrics if "sm__active" in str(r.get("metric_name", ""))), None)
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
    assert len(rows) == 1           # single aggregate row
    r = rows[0]
    assert "max_concurrent_streams" in r
    assert "pct_time_multi_stream" in r
    assert r["max_concurrent_streams"] >= 2   # we have 3 streams
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
    rows_h100 = engine.execute_nvtx_kernel_sm_detail_h100(nvtx_text="%sample_0%", device_id=0)
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
    rows_fwd = engine.execute("nvtx_kernel_sm_detail", nvtx_text="%forward%", device_id=0)
    _show("Skill 18 鈥?nvtx_kernel_sm_detail (%forward%)", rows_fwd)
    assert len(rows_fwd) >= 1
    assert all((r["nvtx_text"] == "forward") for r in rows_fwd)
    assert any((r["kind"] == "comm") for r in rows_fwd)

    # 鈹€鈹€ Skill 19: nvtx_ranges_hierarchy 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rows_all_nvtx = engine.execute("nvtx_ranges_hierarchy", nvtx_text="%", top_level_only=False, limit=100)
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

    rows_root_nvtx = engine.execute("nvtx_ranges_hierarchy", nvtx_text="%", top_level_only=True, limit=100)
    _show("Skill 19 鈥?nvtx_ranges_hierarchy (top-level)", rows_root_nvtx)
    assert len(rows_root_nvtx) >= 1
    assert all((r["depth"] == 0) for r in rows_root_nvtx)

    conn.close()


def test_nvtx_gpu_metrics_breakdown_uses_correlation_gpu_windows(tmp_path: Path) -> None:
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
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER, [end] INTEGER)"
    )
    cur.execute(
        "CREATE TABLE GENERIC_EVENT_SOURCES ("
        "sourceId INTEGER, nameId INTEGER)"
    )
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


def test_timeline_metric_sampling_spans_whole_window(tmp_path: Path) -> None:
    db = tmp_path / "rank0_sampling.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    extra_rows = []
    start_ns = 1_000_000
    for i in range(4000):
        ts = start_ns + i * 1000
        metric_id = 101 if (i % 2 == 0) else 102
        extra_rows.append((ts, metric_id, float(i % 100), 1))
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        extra_rows,
    )
    conn.commit()
    conn.close()

    series = _collect_metric_samples(
        str(db),
        start_ns=start_ns,
        end_ns=start_ns + 3_999_000,
        metric_name_like="%active%",
        include_all_sources=False,
        device_id=-1,
        limit=80,  # force sampling
    )
    all_ts = sorted(
        int(p[0])
        for s in series
        for p in s.get("points", [])
        if isinstance(p, list) and len(p) >= 2
    )
    names = {str(s.get("name", "")) for s in series}
    assert any("sm__active" in n for n in names), names
    assert any("tensor__active" in n for n in names), names
    assert all_ts, "expected sampled metric points"
    # Must cover both beginning and end of the selected window.
    assert all_ts[0] <= start_ns + 5_000
    assert all_ts[-1] >= start_ns + 3_990_000


def test_timeline_metrics_disable_per_series_downsample(tmp_path: Path) -> None:
    db = tmp_path / "rank0_no_downsample.sqlite"
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

    series = _collect_metric_samples(
        str(db),
        start_ns=start_ns,
        end_ns=start_ns + 4_999_000,
        metric_name_like="%sm__active%",
        include_all_sources=False,
        device_id=-1,
        limit=-1,                  # no global sampling
        max_points_per_series=-1,  # no per-series downsample
    )
    assert series, "expected at least one metric series"
    sm_series = next((s for s in series if "sm__active" in str(s.get("name", ""))), None)
    assert sm_series is not None, series
    # selected window covers only injected range, so all 5000 injected points should remain.
    assert len(sm_series.get("points", [])) == 5000


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
    sm_series = next((s for s in series if "sm__active" in str(s.get("name", ""))), None)
    assert sm_series is not None, series
    assert len(sm_series.get("points", [])) == 5000


def test_timeline_default_focus_metrics_filters_unrelated_series(tmp_path: Path) -> None:
    db = tmp_path / "timeline_focus_metrics.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("INSERT INTO StringIds(id, value) VALUES (?, ?)", (777, "random_metric_should_be_filtered"))
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
    assert any("sm__active" in n for n in names), names
    assert any("tensor__active" in n for n in names), names
    assert all("random_metric_should_be_filtered" not in n for n in names), names


def test_timeline_default_focus_warps_metrics_keep_throughput_only(tmp_path: Path) -> None:
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
    assert any("compute warps in flight" in n and "throughput" in n for n in names), names
    assert all("avg warps per cycle" not in n for n in names), names


def test_gpu_metrics_split_by_device_dimension(tmp_path: Path) -> None:
    db = tmp_path / "rank0_multi_device.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_GPU_METRIC ADD COLUMN deviceId INTEGER")
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


def test_timeline_metrics_require_timestamp_column(tmp_path: Path) -> None:
    db = tmp_path / "raw_ts.sqlite"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.execute(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        (101, "sm__active.avg.pct_of_peak_sustained_elapsed"),
    )
    # Keep CUPTI table present but empty; GPU_METRICS has only rawTimestamp.
    # Timeline metrics path is strict: timestamp-only (no rawTimestamp fallback).
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_GPU_METRIC "
        "(timestamp INTEGER, metricId INTEGER, value REAL, sourceId INTEGER)"
    )
    cur.execute(
        "CREATE TABLE GPU_METRICS "
        "(rawTimestamp INTEGER, metricId INTEGER, value REAL, sourceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO GPU_METRICS(rawTimestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        [
            (1000, 101, 11.0, 1),
            (2000, 101, 22.0, 1),
        ],
    )
    conn.commit()
    conn.close()

    rows = _collect_metric_samples(
        str(db),
        start_ns=0,
        end_ns=10_000,
        metric_name_like="%active%",
        include_all_sources=False,
        device_id=-1,
        limit=100,
    )
    assert not rows, "expected no rows when metrics table lacks timestamp column"


def test_timeline_metrics_no_raw_timestamp_fallback_when_timestamp_window_misses(tmp_path: Path) -> None:
    db = tmp_path / "raw_ts_fallback.sqlite"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.execute(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        (101, "sm__active.avg.pct_of_peak_sustained_elapsed"),
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_GPU_METRIC "
        "(timestamp INTEGER, rawTimestamp INTEGER, metricId INTEGER, value REAL, sourceId INTEGER)"
    )
    # timestamp is far away from analysis window, while rawTimestamp is inside window.
    # Timeline metrics path must not fallback to rawTimestamp.
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, rawTimestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?, ?)",
        [
            (10_000_000, 1000, 101, 11.0, 1),
            (10_000_500, 2000, 101, 22.0, 1),
        ],
    )
    conn.commit()
    conn.close()

    rows = _collect_metric_samples(
        str(db),
        start_ns=0,
        end_ns=5000,
        metric_name_like="%active%",
        include_all_sources=False,
        device_id=-1,
        limit=100,
    )
    assert not rows, "expected no rows when timestamp misses window and rawTimestamp fallback is disabled"


def test_timeline_metrics_use_gpu_kernel_window_not_nvtx_cpu_window(tmp_path: Path) -> None:
    db = tmp_path / "gpu_window_vs_nvtx.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    # CPU NVTX window is short; runtime launch is inside it, but GPU kernel executes later.
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        (100_000, 100_500, "late_launch rank=0", None, 59, 12345678),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        (100_100, 100_200, 999, 3, 12345678),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (101_000, 101_500, 7, 999, 1, 1, 0, 128, 1, 1, 32, 4096, 0, 75.0),
    )
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_GPU_METRIC(timestamp, metricId, value, sourceId) VALUES (?, ?, ?, ?)",
        (101_100, 101, 77.0, 1),
    )
    conn.commit()
    conn.close()

    out_html = tmp_path / "gpu_window_vs_nvtx.html"
    export_timeline_html(
        str(db),
        output_path=str(out_html),
        device_id=0,
        nvtx_text="%late_launch%",
        nvtx_index=0,
        include_metrics=True,
        metric_name_like="%sm__active%",
        metrics_limit=-1,
        metrics_max_points=-1,
        debug=False,
    )
    text = out_html.read_text(encoding="utf-8")
    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", text, flags=re.S)
    assert m is not None
    payload = json.loads(m.group(1))
    series = payload.get("metrics") or []
    assert series, "expected metrics from GPU execution window"
    ts = sorted(
        int(p[0])
        for s in series
        for p in s.get("points", [])
        if isinstance(p, list) and len(p) >= 2
    )
    assert 101_100 in ts, ts
    # Render window should be extended to include delayed GPU execution.
    assert int(payload.get("window_end_ns") or 0) >= 101_500


def test_timeline_kernel_occupancy_fallback_to_h100_formula_when_sqlite_missing(tmp_path: Path) -> None:
    db = tmp_path / "timeline_occ_fallback.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    # Keep launch config columns, but drop sqlite-provided theoretical occupancy values.
    conn.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET theoreticalOccupancyPct = NULL")
    conn.commit()
    conn.close()

    out_html = tmp_path / "timeline_occ_fallback.html"
    export_timeline_html(
        str(db),
        output_path=str(out_html),
        device_id=0,
        include_metrics=False,
        debug=False,
    )
    text = out_html.read_text(encoding="utf-8")
    assert "occ_theoretical_pct=" in text
    assert "occ_theoretical_pct=None" not in text
    assert "Kernel Theoretical Occupancy Sum [%]" in text

    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", text, flags=re.S)
    assert m is not None
    payload = json.loads(m.group(1))
    groups = payload.get("all_stream_groups") or []
    assert groups, payload
    occ_values = []
    for g in groups:
        for srow in (g.get("streams") or []):
            for k in (srow.get("kernels") or []):
                occ = k.get("occupancy_pct_estimate")
                if occ is None:
                    continue
                occ_values.append(float(occ))
    assert occ_values, payload
    # Expected from calculate_h100_occupancy for fixture launch configs.
    assert any(abs(v - 100.0) < 1e-6 for v in occ_values), occ_values
    assert any(abs(v - 75.0) < 1e-6 for v in occ_values), occ_values


def test_timeline_kernel_fallback_keeps_overlap_rows(tmp_path: Path) -> None:
    db = tmp_path / "kernel_overlap.sqlite"
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [(1, "k_overlap"), (2, "k_inside"), (3, "k_outside")],
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER, [end] INTEGER, streamId INTEGER, correlationId INTEGER, "
        "shortName INTEGER, demangledName INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (100, 220, 7, 1, 1, 1, 0),   # overlaps [150, 210]
            (160, 170, 7, 2, 2, 2, 0),   # inside [150, 210]
            (0, 50, 7, 3, 3, 3, 0),      # outside
        ],
    )
    conn.commit()
    conn.close()

    provider = NsysSqliteMetricsProvider(str(db))
    rows = _collect_kernels_in_window(
        provider,
        start_ns=150,
        end_ns=210,
        nvtx_text="%",
        nvtx_windows=None,
        device_id=0,
        limit=100,
    )
    names = {str(r.get("kernel_name", "")) for r in rows}
    assert "k_overlap" in names, names
    assert "k_inside" in names, names
    assert "k_outside" not in names, names
    assert any((int(r.get("start_ns") or 0), int(r.get("end_ns") or 0)) == (100, 220) for r in rows), rows


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


# ===========================================================================
# Test 3 鈥?new engine methods
# ===========================================================================

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
        marker="sample_0", device_id=0, threshold_sigma=0.0  # sigma=0 鈫?all are outliers
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


# ===========================================================================
# Test 4 鈥?analyze_nsys_sqlite result dict has new keys
# ===========================================================================

def test_analyze_nsys_sqlite_new_keys(tmp_path: Path) -> None:
    from my_utils.profiling.sources.nsys_analyze import analyze_nsys_sqlite, analyze_to_markdown

    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    result = analyze_nsys_sqlite(str(db), device_id=0, top_k=5)

    _show("analyze_nsys_sqlite keys", list(result.keys()))

    assert "sync_breakdown" in result,    "Missing key: sync_breakdown"
    assert "memcpy_bandwidth" in result,  "Missing key: memcpy_bandwidth"
    assert isinstance(result["sync_breakdown"], list)
    assert isinstance(result["memcpy_bandwidth"], list)
    assert len(result["sync_breakdown"]) >= 1,   "sync_breakdown should have rows"
    assert len(result["memcpy_bandwidth"]) >= 1, "memcpy_bandwidth should have rows"

    _show("sync_breakdown", result["sync_breakdown"])
    _show("memcpy_bandwidth", result["memcpy_bandwidth"])

    # markdown render must include the two new sections
    md = analyze_to_markdown(result)
    assert "## Sync Breakdown" in md,    "Markdown missing Sync Breakdown section"
    assert "## Memcpy Bandwidth" in md,  "Markdown missing Memcpy Bandwidth section"
    print("\n[markdown sections found]")
    for line in md.splitlines():
        if line.startswith("##"):
            print(" ", line)


def test_analyze_nsys_sqlite_window_scoping_for_new_skills(tmp_path: Path) -> None:
    from my_utils.profiling.sources.nsys_analyze import analyze_nsys_sqlite

    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    full = analyze_nsys_sqlite(str(db), device_id=0, top_k=10)
    win = analyze_nsys_sqlite(str(db), device_id=0, start_ns=0, end_ns=7000, top_k=10)

    # 1) nccl_breakdown / sync_breakdown / memcpy_bandwidth should honor the window.
    assert full["nccl_breakdown"], full["nccl_breakdown"]
    assert not win["nccl_breakdown"], win["nccl_breakdown"]

    assert full["sync_breakdown"], full["sync_breakdown"]
    assert not win["sync_breakdown"], win["sync_breakdown"]

    full_copy_kinds = {int(r["copy_kind"]) for r in (full["memcpy_bandwidth"] or [])}
    win_copy_kinds = {int(r["copy_kind"]) for r in (win["memcpy_bandwidth"] or [])}
    assert full_copy_kinds == {1, 2, 8}, full["memcpy_bandwidth"]
    assert win_copy_kinds == {1, 2}, win["memcpy_bandwidth"]

    # 2) short_kernels_overhead / per_stream_utilization should be window-scoped.
    assert any(r.get("duration_bracket") == "c_10-100us" for r in (full["short_kernels"] or []))
    assert not any(r.get("duration_bracket") == "c_10-100us" for r in (win["short_kernels"] or []))

    assert any(int(r.get("stream_id", -1)) == 8 for r in (full["per_stream_utilization"] or []))
    assert not any(int(r.get("stream_id", -1)) == 8 for r in (win["per_stream_utilization"] or []))

    # kernel_duration_stats uses min_invocations=3 in analyze path; validate its windowing directly.
    provider = NsysSqliteMetricsProvider(str(db))
    kds_full = provider.run_sql_skill(
        "kernel_duration_stats",
        device_id=0,
        start_ns=-1,
        end_ns=-1,
        min_invocations=1,
        limit=20,
    )
    kds_win = provider.run_sql_skill(
        "kernel_duration_stats",
        device_id=0,
        start_ns=0,
        end_ns=7000,
        min_invocations=1,
        limit=20,
    )
    by_name_full = {str(r.get("kernel_name")): r for r in kds_full}
    by_name_win = {str(r.get("kernel_name")): r for r in kds_win}
    assert "ncclAllReduceRingLLKernel_sum_f16" in by_name_full, kds_full
    assert "ncclAllReduceRingLLKernel_sum_f16" not in by_name_win, kds_win
    assert int(by_name_win["void gemm_kernel()"]["invocations"]) == 1, kds_win


# ===========================================================================
# Original tests (preserved, now also benefit from richer fixture)
# ===========================================================================

def test_nsys_new_skills_and_methods(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db, scale=1.0)
    provider = NsysSqliteMetricsProvider(str(db))
    skills = provider.list_sql_skills()
    for name in ("nccl_breakdown", "nvtx_kernel_map", "schema_inspect", "thread_utilization"):
        assert name in skills
    assert provider.run_sql_skill("nccl_breakdown", device_id=0, limit=10)
    assert provider.run_sql_skill("schema_inspect", table_like="CUPTI%", limit=100)
    assert provider.run_sql_skill("thread_utilization", limit=10)
    iters = provider.detect_iterations(marker="sample_0", device_id=0)
    assert len(iters) >= 1
    mfu = provider.compute_mfu(step_time_s=0.01, model_flops_per_step=1e12, peak_tflops=100)
    assert "mfu_pct" in mfu


def test_cli_new_subcommands(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    overlap_json = tmp_path / "iter_overlap.json"
    outliers_json = tmp_path / "iter_outliers.json"
    occ_json = tmp_path / "occ_h100.json"

    # nsys-iter-overlap
    assert main([
        "nsys-iter-overlap",
        "--sqlite", str(db),
        "--iteration-marker", "sample_0",
        "--device-id", "0",
        "--limit", "100",
        "--output", str(overlap_json),
        "--pretty",
    ]) == 0
    assert overlap_json.exists()
    data = json.loads(overlap_json.read_text())
    assert isinstance(data, list) and len(data) >= 1
    assert "compute_ms" in data[0]
    assert "comm_ms" in data[0]
    assert "overlap_ms" in data[0]
    assert "comm_pct" in data[0]
    print(f"\n[nsys-iter-overlap] {len(data)} iterations written to {overlap_json}")

    # nsys-iter-outliers
    assert main([
        "nsys-iter-outliers",
        "--sqlite", str(db),
        "--iteration-marker", "sample_0",
        "--device-id", "0",
        "--sigma", "0.5",
        "--output", str(outliers_json),
        "--pretty",
    ]) == 0
    assert outliers_json.exists()
    data2 = json.loads(outliers_json.read_text())
    assert "stats" in data2 and "outliers" in data2
    assert data2["stats"]["count"] >= 1
    print(f"[nsys-iter-outliers] stats={data2['stats']}  outliers={len(data2['outliers'])}")

    # nsys-sql-skill: occupancy should be enriched for H100 by default (--occupancy-arch auto)
    assert main([
        "nsys-sql-skill",
        "--sqlite", str(db),
        "--skill", "kernel_occupancy_estimate",
        "--param", "device_id=0",
        "--param", "limit=10",
        "--output", str(occ_json),
        "--pretty",
    ]) == 0
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
