from __future__ import annotations

import sqlite3
from pathlib import Path

from my_utils.profiling.cli import main
from my_utils.profiling.sources.nsys_sqlite_provider import NsysSqliteMetricsProvider


def _init_sqlite(path: Path, *, scale: float = 1.0) -> None:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    cur.executemany(
        "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
        [
            ("NSIGHT_SYSTEMS_VERSION", "2024.7.1"),
            ("EXPORT_SCHEMA_VERSION", "3.15.1"),
        ],
    )
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (1, "void gemm_kernel()"),
            (2, "ncclAllReduceRingLLKernel_sum_f16"),
            (3, "cudaLaunchKernel"),
            (4, "worker_main"),
        ],
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER, [end] INTEGER, streamId INTEGER, correlationId INTEGER, "
        "shortName INTEGER, demangledName INTEGER, deviceId INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (0, int(10000 * scale), 7, 1, 1, 1, 0),
            (int(8000 * scale), int(20000 * scale), 8, 2, 2, 2, 0),
            (int(25000 * scale), int(35000 * scale), 7, 3, 1, 1, 0),
        ],
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME ("
        "start INTEGER, [end] INTEGER, correlationId INTEGER, nameId INTEGER, globalTid INTEGER)"
    )
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (0, int(2000 * scale), 1, 3, 12345678),
            (int(7000 * scale), int(9000 * scale), 2, 3, 12345678),
            (int(24000 * scale), int(24500 * scale), 3, 3, 12345678),
        ],
    )
    cur.execute(
        "CREATE TABLE NVTX_EVENTS ("
        "start INTEGER, [end] INTEGER, text TEXT, textId INTEGER, eventType INTEGER, globalTid INTEGER)"
    )
    cur.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        [
            (0, int(22000 * scale), "sample_0 step=1 rank=0", None, 59, 12345678),
            (int(23000 * scale), int(36000 * scale), "sample_0 step=2 rank=0", None, 59, 12345678),
        ],
    )
    cur.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, [end] INTEGER, copyKind INTEGER, bytes INTEGER, deviceId INTEGER)")
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?)",
        [
            (0, int(1000 * scale), 1, 1024, 0),
            (int(3000 * scale), int(6000 * scale), 2, 2048, 0),
        ],
    )
    cur.execute("CREATE TABLE COMPOSITE_EVENTS (globalTid INTEGER, cpuCycles INTEGER)")
    cur.executemany(
        "INSERT INTO COMPOSITE_EVENTS VALUES (?, ?)",
        [
            (12345678, int(1000 * scale)),
            (22345678, int(500 * scale)),
        ],
    )
    cur.execute("CREATE TABLE ThreadNames (globalTid INTEGER, nameId INTEGER)")
    cur.executemany(
        "INSERT INTO ThreadNames VALUES (?, ?)",
        [
            (12345678, 4),
            (22345678, 4),
        ],
    )
    cur.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT)")
    cur.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H100')")
    conn.commit()
    conn.close()


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


def test_cli_nsys_commands(tmp_path: Path) -> None:
    db_a = tmp_path / "a.sqlite"
    db_b = tmp_path / "b.sqlite"
    _init_sqlite(db_a, scale=1.0)
    _init_sqlite(db_b, scale=1.2)

    export_json = tmp_path / "kernels.json"
    export_csv = tmp_path / "kernels.csv"
    analyze_json = tmp_path / "analyze.json"
    diff_json = tmp_path / "diff.json"
    timeline_html = tmp_path / "timeline.html"

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

