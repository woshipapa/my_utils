from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple

from my_utils.profiling.cli import main
from my_utils.profiling.sources.nsys_sql_skills import NsysSqlSkillEngine
from my_utils.profiling.sources.nsys_sqlite_provider import NsysSqliteMetricsProvider


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _init_sqlite(path: Path, *, scale: float = 1.0) -> None:
    """Build a minimal but complete nsys-like SQLite covering all 17 skills."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # ── meta ────────────────────────────────────────────────────────────────
    cur.execute("CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT)")
    cur.executemany(
        "INSERT INTO META_DATA_EXPORT VALUES (?, ?)",
        [
            ("NSIGHT_SYSTEMS_VERSION", "2024.7.1"),
            ("EXPORT_SCHEMA_VERSION", "3.15.1"),
        ],
    )

    # ── string table ─────────────────────────────────────────────────────────
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.executemany(
        "INSERT INTO StringIds(id, value) VALUES (?, ?)",
        [
            (1, "void gemm_kernel()"),
            (2, "ncclAllReduceRingLLKernel_sum_f16"),
            (3, "cudaLaunchKernel"),
            (4, "worker_main"),
            (5, "void attention_kernel()"),
        ],
    )

    # ── kernel table (includes block/register columns for skill 15) ──────────
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "start INTEGER, [end] INTEGER, streamId INTEGER, correlationId INTEGER, "
        "shortName INTEGER, demangledName INTEGER, deviceId INTEGER, "
        "blockX INTEGER, blockY INTEGER, blockZ INTEGER, "
        "registersPerThread INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER)"
    )
    # rows: (start, end, stream, corr, short, demangled, dev, bx, by, bz, regs, static_smem, dyn_smem)
    s = scale
    cur.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (0,           int(10000*s), 7, 1, 1, 1, 0, 128, 1, 1, 32, 4096, 0),   # gemm stream7
            (int(8000*s), int(20000*s), 8, 2, 2, 2, 0, 256, 1, 1, 40, 0,    0),   # nccl  stream8
            (int(25000*s),int(35000*s), 7, 3, 1, 1, 0, 128, 1, 1, 32, 4096, 0),   # gemm stream7
            (int(5000*s), int(12000*s), 9, 4, 5, 5, 0,  64, 1, 1, 48, 8192, 2048),# attention stream9
        ],
    )

    # ── runtime table ────────────────────────────────────────────────────────
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

    # ── NVTX events ──────────────────────────────────────────────────────────
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

    # ── memcpy table (skill 4, 12, 17) ───────────────────────────────────────
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

    # ── memset table (skill 14) ───────────────────────────────────────────────
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

    # ── synchronization table (skill 13) ─────────────────────────────────────
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

    # ── CPU events (skill 11) ─────────────────────────────────────────────────
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
# Test 1 – all 17 skills register and execute without error
# ===========================================================================

def test_all_17_skills_register_and_execute(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    skills = engine.list_skills()
    print(f"\n[skills] registered: {len(skills)}")
    for s in skills:
        print(f"  - {s}")

    EXPECTED_17 = {
        "aggregate_kernels", "top_kernels", "aggregate_nvtx_ranges",
        "memcpy_in_window", "kernel_map", "gpu_idle_gaps",
        "kernel_launch_overhead", "nccl_breakdown", "nvtx_kernel_map",
        "schema_inspect", "thread_utilization",
        "memcpy_bandwidth_analysis", "sync_breakdown", "memset_breakdown",
        "kernel_occupancy_estimate", "stream_parallelism", "nvtx_memcpy_breakdown",
    }
    missing = EXPECTED_17 - set(skills)
    assert not missing, f"Missing skills: {missing}"

    errors: List[Tuple[str, str]] = []
    for name in skills:
        try:
            rows = engine.execute(name)
            print(f"  [{name}] OK  rows={len(rows)}")
        except Exception as exc:
            errors.append((name, str(exc)))
            print(f"  [{name}] ERROR: {exc}")

    conn.close()
    assert not errors, f"Skills raised errors: {errors}"


# ===========================================================================
# Test 2 – new skills return correct columns and values
# ===========================================================================

def test_new_skill_outputs(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    # ── Skill 12: memcpy_bandwidth_analysis ─────────────────────────────────
    rows = engine.execute("memcpy_bandwidth_analysis", device_id=0)
    _show("Skill 12 – memcpy_bandwidth_analysis", rows)
    assert len(rows) == 3, f"expected 3 copyKind groups, got {len(rows)}"
    for r in rows:
        assert "copy_kind" in r
        assert "avg_gbps" in r
        assert r["avg_gbps"] is not None and r["avg_gbps"] > 0

    # ── Skill 13: sync_breakdown ─────────────────────────────────────────────
    rows = engine.execute("sync_breakdown", device_id=0)
    _show("Skill 13 – sync_breakdown", rows)
    assert len(rows) >= 1
    assert "sync_type" in rows[0]
    assert "total_ms" in rows[0]
    total_sync_ms = sum(r["total_ms"] for r in rows)
    assert total_sync_ms > 0

    # ── Skill 14: memset_breakdown ───────────────────────────────────────────
    rows = engine.execute("memset_breakdown", device_id=0)
    _show("Skill 14 – memset_breakdown", rows)
    assert len(rows) >= 1
    assert "fill_value" in rows[0]
    assert "total_gb" in rows[0]
    # zero-init bytes = 8MB + 1MB = 9MB
    zero_rows = [r for r in rows if r["fill_value"] == 0]
    assert zero_rows, "Expected a zero-init (fill_value=0) row"
    assert abs(zero_rows[0]["total_gb"] - 9 / 1024) < 0.001

    # ── Skill 15: kernel_occupancy_estimate ──────────────────────────────────
    rows = engine.execute("kernel_occupancy_estimate", device_id=0, limit=10)
    _show("Skill 15 – kernel_occupancy_estimate", rows)
    assert len(rows) >= 1
    assert "threads_per_block" in rows[0]
    assert "occupancy_pct_estimate" in rows[0]
    for r in rows:
        assert 0 < r["occupancy_pct_estimate"] <= 100

    # ── Skill 16: stream_parallelism ─────────────────────────────────────────
    rows = engine.execute("stream_parallelism", device_id=0, bucket_ns=5000)
    _show("Skill 16 – stream_parallelism", rows)
    assert len(rows) == 1           # single aggregate row
    r = rows[0]
    assert "max_concurrent_streams" in r
    assert "pct_time_multi_stream" in r
    assert r["max_concurrent_streams"] >= 2   # we have 3 streams

    # ── Skill 17: nvtx_memcpy_breakdown ──────────────────────────────────────
    rows = engine.execute("nvtx_memcpy_breakdown", limit=20)
    _show("Skill 17 – nvtx_memcpy_breakdown", rows)
    # memcpy rows fall inside NVTX ranges (forward/backward/step ranges)
    assert len(rows) >= 1
    assert "nvtx_text" in rows[0]
    assert "total_gb" in rows[0]

    conn.close()


# ===========================================================================
# Test 3 – new engine methods
# ===========================================================================

def test_new_engine_methods(tmp_path: Path) -> None:
    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    engine = NsysSqlSkillEngine(conn)

    # ── analyze_per_iteration_overlap ────────────────────────────────────────
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

    # ── detect_iteration_outliers ────────────────────────────────────────────
    result = engine.detect_iteration_outliers(
        marker="sample_0", device_id=0, threshold_sigma=0.0  # sigma=0 → all are outliers
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
# Test 4 – analyze_nsys_sqlite result dict has new keys
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

