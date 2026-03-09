from __future__ import annotations

import json
import time
from pathlib import Path

from my_utils.profiling.cli import main
from my_utils.profiling.metrics_types import MetricEvent


def test_cli_analyze_and_diff(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    events_path = tmp_path / "events.jsonl"
    events = [
        MetricEvent(
            timestamp=time.time(),
            name="latency.stage",
            value=10.0,
            unit="ms",
            provider_id="ut",
            tags={"step": "0", "rank": "0", "stage": "forward"},
        ),
        MetricEvent(
            timestamp=time.time(),
            name="memory.gpu.allocated",
            value=1_000_000_000.0,
            unit="bytes",
            provider_id="ut",
            tags={"step": "0", "rank": "0"},
        ),
    ]
    lines = [json.dumps(item.to_dict(), ensure_ascii=False) for item in events]
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_a = tmp_path / "a"
    rc = main(
        [
            "analyze",
            "--events",
            str(events_path),
            "--output-dir",
            str(out_a),
            "--report-formats",
            "json,markdown",
        ]
    )
    assert rc == 0
    report_a = out_a / "analysis_report.json"
    assert report_a.exists()

    # Create a second report with a different workload profile for diff.
    out_b = tmp_path / "b"
    rc = main(
        [
            "analyze",
            "--events",
            str(events_path),
            "--output-dir",
            str(out_b),
            "--workload",
            "inference",
            "--report-formats",
            "json",
        ]
    )
    assert rc == 0
    report_b = out_b / "analysis_report.json"
    assert report_b.exists()

    diff_json = tmp_path / "diff.json"
    rc = main(
        [
            "diff",
            "--base-report",
            str(report_a),
            "--target-report",
            str(report_b),
            "--output",
            str(diff_json),
        ]
    )
    assert rc == 0
    assert diff_json.exists()

    trace_json = tmp_path / "trace.json"
    rc = main(
        [
            "trace",
            "--events",
            str(events_path),
            "--output",
            str(trace_json),
            "--auto-align-ranks",
        ]
    )
    assert rc == 0
    assert trace_json.exists()
    payload = json.loads(trace_json.read_text(encoding="utf-8"))
    assert "traceEvents" in payload
    assert any(item.get("ph") == "X" for item in payload["traceEvents"])
