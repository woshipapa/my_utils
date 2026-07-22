from __future__ import annotations

import json
from pathlib import Path

from my_utils.profiling import (
    ChromeTraceExportConfig,
    MetricEvent,
    estimate_rank_time_offsets,
    metric_events_to_chrome_trace,
    write_chrome_trace,
)


def _build_events() -> list[MetricEvent]:
    # Two ranks, same logical step, rank1 clock is +2s.
    return [
        MetricEvent(
            timestamp=1000.100,
            name="latency.stage",
            value=100.0,
            unit="ms",
            provider_id="ut",
            tags={"rank": "0", "step": "10", "stage": "forward"},
            attributes={"end_timestamp": 1000.100},
        ),
        MetricEvent(
            timestamp=1002.100,
            name="latency.stage",
            value=100.0,
            unit="ms",
            provider_id="ut",
            tags={"rank": "1", "step": "10", "stage": "forward"},
            attributes={"end_timestamp": 1002.100},
        ),
    ]


def test_estimate_rank_time_offsets() -> None:
    offsets = estimate_rank_time_offsets(_build_events(), reference_rank="0")
    # rank1 should be shifted by around -2s to align with rank0.
    assert "1" in offsets
    assert abs(offsets["1"] + 2.0) < 1e-6


def test_metric_events_to_chrome_trace_auto_align() -> None:
    payload = metric_events_to_chrome_trace(
        _build_events(),
        config=ChromeTraceExportConfig(auto_align_ranks=True, reference_rank="0"),
    )
    trace_events = [item for item in payload["traceEvents"] if item.get("ph") == "X"]
    assert len(trace_events) == 2

    # Compare start timestamps after alignment.
    by_pid = {item["pid"]: item for item in trace_events}
    # pid uses rank directly when available.
    ts0 = by_pid[0]["ts"]
    ts1 = by_pid[1]["ts"]
    assert abs(ts0 - ts1) < 1.0  # within 1us


def test_write_chrome_trace(tmp_path: Path) -> None:
    out = tmp_path / "trace.json"
    path = write_chrome_trace(
        _build_events(),
        output_path=str(out),
        config=ChromeTraceExportConfig(auto_align_ranks=True, reference_rank="0"),
    )
    assert Path(path).exists()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "traceEvents" in payload
    assert any(item.get("ph") == "X" for item in payload["traceEvents"])
