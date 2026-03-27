from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import my_utils.profiling.cli as profiling_cli
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


def test_cli_alias_entry_forwards_subcommand(monkeypatch) -> None:
    captured = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(profiling_cli, "main", _fake_main)
    monkeypatch.setattr(sys, "argv", ["nsys-sql-skill", "--sqlite", "x.sqlite", "--list-skills"])
    rc = profiling_cli.entry_nsys_sql_skill()
    assert rc == 0
    assert captured["argv"][0] == "nsys-sql-skill"
    assert "--sqlite" in captured["argv"]
    assert "x.sqlite" in captured["argv"]


def test_nsys_panel_generate_command_without_execute(monkeypatch, capsys) -> None:
    answers = iter(
        [
            "nsys-sql-skill",   # choose command by name
            "demo.sqlite",      # required --sqlite
            "n",                # skip optional args
            "n",                # do not execute
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    rc = main(["nsys-panel"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NSYS Interactive Panel" in out
    assert "myutils-profile nsys-sql-skill --sqlite demo.sqlite" in out


def test_nsys_panel_execute_selected_command(monkeypatch) -> None:
    called = {}

    def _fake_main(argv=None):
        called["argv"] = list(argv or [])
        return 0

    answers = iter(
        [
            "nsys-export",      # choose command
            "trace.sqlite",     # required --sqlite
            "out.json",         # required --output
            "n",                # skip optional args
            "y",                # execute now
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(profiling_cli, "main", _fake_main)
    rc = profiling_cli.cmd_nsys_panel(args=None)
    assert rc == 0
    assert called["argv"] == [
        "nsys-export",
        "--sqlite",
        "trace.sqlite",
        "--output",
        "out.json",
    ]
