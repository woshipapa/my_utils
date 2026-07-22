"""Tests for my_utils.profiling.sources.nsys_analyze."""

from __future__ import annotations


import sqlite3
from pathlib import Path


from my_utils.profiling.sources.nsys_sqlite_provider import NsysSqliteMetricsProvider


from _synthetic_loader import _init_sqlite, _show


def test_analyze_nsys_sqlite_new_keys(tmp_path: Path) -> None:
    from my_utils.profiling.sources.nsys_analyze import (
        analyze_nsys_sqlite,
        analyze_to_markdown,
    )

    db = tmp_path / "rank0.sqlite"
    _init_sqlite(db)

    result = analyze_nsys_sqlite(str(db), device_id=0, top_k=5)

    _show("analyze_nsys_sqlite keys", list(result.keys()))

    assert "sync_breakdown" in result, "Missing key: sync_breakdown"
    assert "memcpy_bandwidth" in result, "Missing key: memcpy_bandwidth"
    assert "subphase_timings" in result, "Missing key: subphase_timings"
    assert isinstance(result["sync_breakdown"], list)
    assert isinstance(result["memcpy_bandwidth"], list)
    assert isinstance(result["subphase_timings"], list)
    assert len(result["sync_breakdown"]) >= 1, "sync_breakdown should have rows"
    assert len(result["memcpy_bandwidth"]) >= 1, "memcpy_bandwidth should have rows"
    assert len(result["subphase_timings"]) >= 1, "subphase_timings should have rows"

    _show("sync_breakdown", result["sync_breakdown"])
    _show("memcpy_bandwidth", result["memcpy_bandwidth"])

    # markdown render must include the two new sections
    md = analyze_to_markdown(result)
    assert "## Sync Breakdown" in md, "Markdown missing Sync Breakdown section"
    assert "## Memcpy Bandwidth" in md, "Markdown missing Memcpy Bandwidth section"
    assert "## Analysis Subphase Timings" in md, (
        "Markdown missing subphase timings section"
    )
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
    assert any(
        r.get("duration_bracket") == "c_10-100us" for r in (full["short_kernels"] or [])
    )
    assert not any(
        r.get("duration_bracket") == "c_10-100us" for r in (win["short_kernels"] or [])
    )

    assert any(
        int(r.get("stream_id", -1)) == 8 for r in (full["per_stream_utilization"] or [])
    )
    assert not any(
        int(r.get("stream_id", -1)) == 8 for r in (win["per_stream_utilization"] or [])
    )

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


def test_analyze_nsys_sqlite_nvtx_scope_extends_to_gpu_kernel_span(
    tmp_path: Path,
) -> None:
    from my_utils.profiling.sources.nsys_analyze import analyze_nsys_sqlite

    db = tmp_path / "rank0.sqlite"
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
    conn.commit()
    conn.close()

    result = analyze_nsys_sqlite(
        str(db),
        device_id=0,
        nvtx_text="late_launch",
        top_k=10,
    )
    win = result.get("window") or {}
    assert int(win.get("start_ns") or -1) == 100_000, win
    assert int(win.get("end_ns") or -1) == 101_500, win

    span_info = result.get("nvtx_kernel_span_info") or {}
    assert not span_info.get("warning"), span_info
    assert int(span_info.get("kernel_start_ns") or -1) == 101_000, span_info
    assert int(span_info.get("kernel_end_ns") or -1) == 101_500, span_info
    assert int(span_info.get("count") or 0) >= 1, span_info

    summary = result.get("summary") or {}
    assert int(summary.get("kernel_rows") or 0) >= 1, summary
    warns = list(result.get("warnings") or [])
    assert any("NVTX_SCOPE_KERNEL_SPAN" in str(w) for w in warns), warns


def test_analyze_nvtx_text_requires_explicit_wildcard(tmp_path: Path) -> None:
    from my_utils.profiling.sources.nsys_analyze import analyze_nsys_sqlite

    db = tmp_path / "analyze_nvtx_explicit_wildcard.sqlite"
    _init_sqlite(db)

    exact = analyze_nsys_sqlite(
        str(db),
        device_id=0,
        nvtx_text="sample_0",
        top_k=10,
    )
    exact_info = dict(exact.get("nvtx_text_info") or {})
    assert int(exact_info.get("count") or 0) == 0, exact_info
    assert "No NVTX ranges matched pattern 'sample_0'" in str(
        exact_info.get("warning") or ""
    ), exact_info

    like = analyze_nsys_sqlite(
        str(db),
        device_id=0,
        nvtx_text="%sample_0%",
        top_k=10,
    )
    like_info = dict(like.get("nvtx_text_info") or {})
    assert int(like_info.get("count") or 0) >= 2, like_info
    assert not str(like_info.get("warning") or ""), like_info
