from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Tuple

from my_utils.profiling.cli import main
from my_utils.profiling.sources.nsys_sql_skills import (
    NsysSqlSkillEngine,
    calculate_h100_occupancy,
)
from my_utils.profiling.sources.nsys_timeline_html import _collect_metric_samples
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
    sm_row_all = next((r for r in rows_all_sources if "sm__active" in str(r.get("metric_name", ""))), None)
    assert sm_row_all is not None
    assert sm_row_all["sample_count"] == 3

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
    assert all_ts, "expected sampled metric points"
    # Must cover both beginning and end of the selected window.
    assert all_ts[0] <= start_ns + 5_000
    assert all_ts[-1] >= start_ns + 3_990_000


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
    assert "sample_0 step=1 rank=0" in timeline_text
    assert "sample_0 step=2 rank=0" in timeline_text
    assert "sample_0 step=1 rank=1" in timeline_text
    assert "nvtx_scopes=3" in timeline_text
    assert "Rank 0 | Device 0" in timeline_text
    assert "Rank 1 | Device 0" in timeline_text
