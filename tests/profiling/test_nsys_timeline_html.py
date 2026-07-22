"""Tests for my_utils.profiling.sources.nsys_timeline_html."""

from __future__ import annotations


import html
import json
import re
import sqlite3
from pathlib import Path
from typing import List


from my_utils.profiling.sources.nsys_timeline_html import (
    _collect_kernels_in_window,
    _collect_metric_samples,
    _pick_nvtx_windows,
    _select_nvtx_windows,
    export_timeline_compare_html,
    export_timeline_html,
)


from my_utils.profiling.sources.nsys_sqlite_provider import NsysSqliteMetricsProvider


from _synthetic_loader import _init_sqlite


def _extract_compare_payloads(html_text: str) -> List[dict]:
    payloads: List[dict] = []
    for srcdoc in re.findall(r'srcdoc="(.*?)"', html_text, flags=re.S):
        inner = html.unescape(srcdoc)
        m = re.search(r"const TIMELINE_DATA = (\{.*?\});", inner, flags=re.S)
        if not m:
            continue
        payload = json.loads(m.group(1))
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def test_timeline_kernel_collection_keeps_duplicate_nvtx_attribution_rows(
    tmp_path: Path,
) -> None:
    db = tmp_path / "timeline_keep_duplicate_nvtx_rows.sqlite"
    _init_sqlite(db)

    provider = NsysSqliteMetricsProvider(str(db))
    skill_rows = provider.run_sql_skill(
        "nvtx_kernel_sm_detail",
        nvtx_text="%",
        device_id=0,
        limit=1000,
    )
    windows = _pick_nvtx_windows(
        _select_nvtx_windows(provider, nvtx_text="%"), nvtx_index=-1
    )
    collected = _collect_kernels_in_window(
        provider,
        start_ns=0,
        end_ns=36_000,
        nvtx_text="%",
        nvtx_windows=windows,
        device_id=0,
        limit=1000,
    )

    assert len(skill_rows) > 0, skill_rows
    assert len(collected) == len(skill_rows), (len(collected), len(skill_rows))


def test_timeline_debug_logs_emit_matched_kernel_counts(tmp_path: Path) -> None:
    db = tmp_path / "timeline_debug_counts.sqlite"
    _init_sqlite(db)

    out = tmp_path / "timeline_debug_counts.html"
    debug_messages: List[str] = []
    progress_messages: List[str] = []

    export_timeline_html(
        str(db),
        output_path=str(out),
        device_id=0,
        nvtx_text="%sample_0%",
        include_metrics=False,
        debug=True,
        debug_log_fn=debug_messages.append,
        progress_cb=progress_messages.append,
    )

    assert any("matched kernels total=" in msg for msg in debug_messages), (
        debug_messages
    )
    assert any("collect_kernels matched_kernels=" in msg for msg in debug_messages), (
        debug_messages
    )
    assert any("matched_kernels=" in msg for msg in progress_messages), (
        progress_messages
    )


def test_timeline_progress_emits_selected_nvtx_full_name(tmp_path: Path) -> None:
    db = tmp_path / "timeline_progress_nvtx_name.sqlite"
    _init_sqlite(db)

    out = tmp_path / "timeline_progress_nvtx_name.html"
    progress_messages: List[str] = []

    export_timeline_html(
        str(db),
        output_path=str(out),
        device_id=0,
        nvtx_text="%sample_0%",
        nvtx_index=1,
        include_metrics=False,
        debug=False,
        progress_cb=progress_messages.append,
    )

    assert any("nvtx_match_count=" in msg for msg in progress_messages), (
        progress_messages
    )
    assert any(
        "selected_nvtx[0] full_name=sample_0 step=2 rank=0" in msg
        for msg in progress_messages
    ), progress_messages


def test_timeline_compare_progress_emits_selected_nvtx_full_name(
    tmp_path: Path,
) -> None:
    db_a = tmp_path / "timeline_compare_progress_a.sqlite"
    db_b = tmp_path / "timeline_compare_progress_b.sqlite"
    _init_sqlite(db_a)
    _init_sqlite(db_b, scale=1.2)

    out = tmp_path / "timeline_compare_progress.html"
    progress_messages: List[str] = []

    export_timeline_compare_html(
        [str(db_a), str(db_b)],
        output_path=str(out),
        device_id=0,
        nvtx_text="%sample_0%",
        nvtx_index=1,
        include_metrics=False,
        debug=False,
        progress_cb=progress_messages.append,
    )

    assert any(
        "[1/2]" in msg and "selected_nvtx[0] full_name=sample_0 step=2 rank=0" in msg
        for msg in progress_messages
    ), progress_messages
    assert any(
        "[2/2]" in msg and "selected_nvtx[0] full_name=sample_0 step=2 rank=0" in msg
        for msg in progress_messages
    ), progress_messages


def test_timeline_nvtx_text_requires_explicit_wildcard(tmp_path: Path) -> None:
    db = tmp_path / "timeline_nvtx_explicit_wildcard.sqlite"
    _init_sqlite(db)

    out_exact = tmp_path / "timeline_nvtx_exact.html"
    exact_progress: List[str] = []
    export_timeline_html(
        str(db),
        output_path=str(out_exact),
        device_id=0,
        nvtx_text="sample_0",
        include_metrics=False,
        debug=False,
        progress_cb=exact_progress.append,
    )
    assert any(
        "nvtx_match_count=0 selected_count=0" in msg for msg in exact_progress
    ), exact_progress

    out_like = tmp_path / "timeline_nvtx_like.html"
    like_progress: List[str] = []
    export_timeline_html(
        str(db),
        output_path=str(out_like),
        device_id=0,
        nvtx_text="%sample_0%",
        include_metrics=False,
        debug=False,
        progress_cb=like_progress.append,
    )
    assert any("nvtx_match_count=2 selected_count=2" in msg for msg in like_progress), (
        like_progress
    )


def test_timeline_allstream_js_not_blocked_when_metrics_disabled(
    tmp_path: Path,
) -> None:
    db = tmp_path / "timeline_no_metrics_allstream.sqlite"
    _init_sqlite(db)

    out = tmp_path / "timeline_no_metrics_allstream.html"
    export_timeline_html(
        str(db),
        output_path=str(out),
        device_id=0,
        include_metrics=False,
        debug=False,
    )

    text = out.read_text(encoding="utf-8")
    assert "const root = document.getElementById('allstream-root');" in text
    assert "if (!grid) return;" not in text
    assert "if (grid) {" in text


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
        limit=-1,  # no global sampling
        max_points_per_series=-1,  # no per-series downsample
    )
    assert series, "expected at least one metric series"
    sm_series = next(
        (s for s in series if "sm__active" in str(s.get("name", ""))), None
    )
    assert sm_series is not None, series
    # selected window covers only injected range, so all 5000 injected points should remain.
    assert len(sm_series.get("points", [])) == 5000


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


def test_timeline_metrics_no_raw_timestamp_fallback_when_timestamp_window_misses(
    tmp_path: Path,
) -> None:
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
    assert not rows, (
        "expected no rows when timestamp misses window and rawTimestamp fallback is disabled"
    )


def test_timeline_metrics_use_gpu_kernel_window_not_nvtx_cpu_window(
    tmp_path: Path,
) -> None:
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


def test_timeline_kernel_occupancy_fallback_to_h100_formula_when_sqlite_missing(
    tmp_path: Path,
) -> None:
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
        for srow in g.get("streams") or []:
            for k in srow.get("kernels") or []:
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
            (100, 220, 7, 1, 1, 1, 0),  # overlaps [150, 210]
            (160, 170, 7, 2, 2, 2, 0),  # inside [150, 210]
            (0, 50, 7, 3, 3, 3, 0),  # outside
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
    assert any(
        (int(r.get("start_ns") or 0), int(r.get("end_ns") or 0)) == (100, 220)
        for r in rows
    ), rows


def test_timeline_compare_html_embeds_multiple_sqlites(tmp_path: Path) -> None:
    db_a = tmp_path / "compare_a.sqlite"
    db_b = tmp_path / "compare_b.sqlite"
    _init_sqlite(db_a, scale=1.0)
    _init_sqlite(db_b, scale=1.2)

    out = tmp_path / "timeline_compare.html"
    export_timeline_compare_html(
        [str(db_a), str(db_b)],
        output_path=str(out),
        device_id=0,
        nvtx_text="%sample_0%",
        include_metrics=True,
        metric_name_like="%active%",
        metrics_limit=-1,
        metrics_max_points=-1,
    )

    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "NSYS NVTX Timeline Compare" in text
    assert text.count("<iframe") == 8, text
    assert str(db_a) in text, text
    assert str(db_b) in text, text
    assert (
        "Each compare section groups the same timeline panel across all sqlite files"
        in text
    )
    assert "equal-duration kernels render with equal widths" in text
    assert ".compare-root" in text
    assert "min-height:140px" in text
    assert "Math.min(Math.max(h + 12, 140), 6000)" in text
    assert "Optimization Summary" in text
    assert "Pairwise Delta Summary" in text
    assert "Kernel Hotspots" in text
    assert "Metric Snapshot" in text
    assert "All Streams Overlap + Metrics Alignment" in text
    assert "Matched NVTX Scopes" in text
    assert "Kernel Timeline By Stream" in text
    assert "GPU Metrics In Window" in text
    assert text.index("All Streams Overlap + Metrics Alignment") < text.index(
        "Matched NVTX Scopes"
    ), text
    payloads = _extract_compare_payloads(text)
    assert payloads, text
    display_spans = {
        int(p.get("display_span_ns") or p.get("span_ns") or 0) for p in payloads
    }
    assert len(display_spans) == 1, display_spans
    data_spans = {int(p.get("data_span_ns") or p.get("span_ns") or 0) for p in payloads}
    assert len(data_spans) > 1, data_spans


def test_timeline_compare_html_reports_fusion_candidates(tmp_path: Path) -> None:
    db_a = tmp_path / "fusion_base.sqlite"
    db_b = tmp_path / "fusion_target.sqlite"
    _init_sqlite(db_a)
    _init_sqlite(db_b)

    def _inject_fusion_case(db: Path, *, fused: bool) -> None:
        conn = sqlite3.connect(str(db))
        conn.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [
                (8011, "fusion_anchor_start"),
                (8012, "fusion_mid_1"),
                (8013, "fusion_mid_2"),
                (8014, "fusion_anchor_end"),
                (8015, "fusion_mid_fused"),
            ],
        )
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
            (200_000, 200_200, "fusion_case", None, 59, 12345678),
        )
        if not fused:
            runtimes = [
                (200_010, 200_020, 8101, 3, 12345678),
                (200_030, 200_040, 8102, 3, 12345678),
                (200_050, 200_060, 8103, 3, 12345678),
                (200_070, 200_080, 8104, 3, 12345678),
            ]
            kernels = [
                (
                    200_300,
                    200_360,
                    88,
                    8101,
                    8011,
                    8011,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    200_370,
                    200_430,
                    88,
                    8102,
                    8012,
                    8012,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    200_440,
                    200_520,
                    88,
                    8103,
                    8013,
                    8013,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    200_530,
                    200_590,
                    88,
                    8104,
                    8014,
                    8014,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
            ]
        else:
            runtimes = [
                (200_010, 200_020, 8201, 3, 12345678),
                (200_030, 200_040, 8202, 3, 12345678),
                (200_050, 200_060, 8203, 3, 12345678),
            ]
            kernels = [
                (
                    200_300,
                    200_360,
                    88,
                    8201,
                    8011,
                    8011,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    200_370,
                    200_520,
                    88,
                    8202,
                    8015,
                    8015,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    200_530,
                    200_590,
                    88,
                    8203,
                    8014,
                    8014,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
            ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", runtimes
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            kernels,
        )
        conn.commit()
        conn.close()

    _inject_fusion_case(db_a, fused=False)
    _inject_fusion_case(db_b, fused=True)

    out = tmp_path / "fusion_compare.html"
    export_timeline_compare_html(
        [str(db_a), str(db_b)],
        output_path=str(out),
        device_id=0,
        nvtx_text="%fusion_case%",
        include_metrics=False,
    )

    text = out.read_text(encoding="utf-8")
    assert "Potential Fusion Mapping" in text
    assert "Possible Fusion In Target" in text
    assert "score=" in text
    assert "strong-anchors" in text
    assert "fusion_mid_1" in text
    assert "fusion_mid_2" in text
    assert "fusion_mid_fused" in text
    assert "stream 88" in text


def test_timeline_compare_html_avoids_false_positive_when_kernel_is_only_removed(
    tmp_path: Path,
) -> None:
    db_a = tmp_path / "delete_base.sqlite"
    db_b = tmp_path / "delete_target.sqlite"
    _init_sqlite(db_a)
    _init_sqlite(db_b)

    def _inject_case(db: Path, *, target_only_removes_one_kernel: bool) -> None:
        conn = sqlite3.connect(str(db))
        conn.executemany(
            "INSERT INTO StringIds(id, value) VALUES (?, ?)",
            [
                (8111, "delete_anchor_start"),
                (8112, "delete_mid_keep"),
                (8113, "delete_mid_drop"),
                (8114, "delete_anchor_end"),
            ],
        )
        conn.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
            (210_000, 210_200, "delete_case", None, 59, 998877),
        )
        if not target_only_removes_one_kernel:
            runtimes = [
                (210_010, 210_020, 9101, 3, 998877),
                (210_030, 210_040, 9102, 3, 998877),
                (210_050, 210_060, 9103, 3, 998877),
                (210_070, 210_080, 9104, 3, 998877),
            ]
            kernels = [
                (
                    210_300,
                    210_360,
                    41,
                    9101,
                    8111,
                    8111,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    210_370,
                    210_430,
                    41,
                    9102,
                    8112,
                    8112,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    210_440,
                    210_500,
                    41,
                    9103,
                    8113,
                    8113,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    210_510,
                    210_570,
                    41,
                    9104,
                    8114,
                    8114,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
            ]
        else:
            runtimes = [
                (210_010, 210_020, 9201, 3, 998877),
                (210_030, 210_040, 9202, 3, 998877),
                (210_050, 210_060, 9203, 3, 998877),
            ]
            kernels = [
                (
                    210_300,
                    210_360,
                    41,
                    9201,
                    8111,
                    8111,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    210_370,
                    210_430,
                    41,
                    9202,
                    8112,
                    8112,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
                (
                    210_510,
                    210_570,
                    41,
                    9203,
                    8114,
                    8114,
                    0,
                    128,
                    1,
                    1,
                    32,
                    4096,
                    0,
                    87.5,
                ),
            ]
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", runtimes
        )
        conn.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            kernels,
        )
        conn.commit()
        conn.close()

    _inject_case(db_a, target_only_removes_one_kernel=False)
    _inject_case(db_b, target_only_removes_one_kernel=True)

    out = tmp_path / "delete_compare.html"
    export_timeline_compare_html(
        [str(db_a), str(db_b)],
        output_path=str(out),
        device_id=0,
        nvtx_text="%delete_case%",
        include_metrics=False,
    )

    text = out.read_text(encoding="utf-8")
    assert "Potential Fusion Mapping" in text
    assert "Possible Fusion In Target" not in text
    assert "Possible Split In Target" not in text
    assert "No strong fusion candidates detected" in text


def test_nvtx_kernel_sm_detail_cross_thread_runtime_fallback_keeps_kernels(
    tmp_path: Path,
) -> None:
    db = tmp_path / "cross_thread_nvtx.sqlite"
    _init_sqlite(db)

    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?)",
        (100_000, 100_200, "cross_thread_layer", None, 59, 111),
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
        [
            (100_010, 100_020, 7001, 3, 222),
            (100_040, 100_050, 7002, 3, 222),
        ],
    )
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (100_300, 100_500, 7, 7001, 1, 1, 0, 128, 1, 1, 32, 4096, 0, 87.5),
            (100_520, 100_780, 8, 7002, 5, 5, 0, 64, 1, 1, 48, 8192, 2048, 50.0),
        ],
    )
    conn.commit()
    conn.close()

    provider = NsysSqliteMetricsProvider(str(db))
    rows = provider.run_sql_skill(
        "nvtx_kernel_sm_detail",
        nvtx_text="%cross_thread_layer%",
        device_id=0,
        limit=100,
    )
    names = {str(r.get("kernel_name", "")) for r in rows}
    assert "void gemm_kernel()" in names, rows
    assert "void attention_kernel()" in names, rows
    assert len(rows) == 2, rows

    out = tmp_path / "cross_thread_timeline.html"
    export_timeline_html(
        str(db),
        output_path=str(out),
        device_id=0,
        nvtx_text="%cross_thread_layer%",
        include_metrics=False,
        debug=False,
    )
    text = out.read_text(encoding="utf-8")
    assert "void gemm_kernel()" in text
    assert "void attention_kernel()" in text
