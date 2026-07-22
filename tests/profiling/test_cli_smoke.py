# SPDX-License-Identifier: Apache-2.0
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
    monkeypatch.setattr(
        sys, "argv", ["nsys-sql-skill", "--sqlite", "x.sqlite", "--list-skills"]
    )
    rc = profiling_cli.entry_nsys_sql_skill()
    assert rc == 0
    assert captured["argv"][0] == "nsys-sql-skill"
    assert "--sqlite" in captured["argv"]
    assert "x.sqlite" in captured["argv"]


def test_nsys_panel_generate_command_without_execute(monkeypatch, capsys) -> None:
    answers = iter(
        [
            "nsys-sql-skill",  # choose command by name
            "demo.sqlite",  # required --sqlite
            "n",  # skip optional args
            "n",  # do not execute
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    rc = main(["nsys-panel"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Profiling Interactive Panel (NSYS + NCU)" in out
    assert "myutils-profile nsys-sql-skill --sqlite demo.sqlite" in out


def test_nsys_panel_execute_selected_command(monkeypatch) -> None:
    called = {}

    def _fake_main(argv=None):
        called["argv"] = list(argv or [])
        return 0

    answers = iter(
        [
            "nsys-export",  # choose command
            "trace.sqlite",  # required --sqlite
            "out.json",  # required --output
            "n",  # skip optional args
            "y",  # execute now
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


def test_nsys_panel_bool_conflict_groups_and_semantic_skip(monkeypatch, capsys) -> None:
    prompts: list[str] = []

    def _fake_input(prompt: str) -> str:
        prompts.append(str(prompt))
        text = str(prompt)
        line = text.strip()
        if text.startswith("command >"):
            return "nsys-timeline-html"
        if line.startswith("--sqlite"):
            return "demo.sqlite"
        if line.startswith("--output"):
            return "out.html"
        if "Configure optional arguments?" in text:
            return "y"
        if line.startswith("--include-metrics"):
            return "on"
        if line.startswith("--default-focus-metrics"):
            return "off"
        if line.startswith("--debug"):
            return "off"
        if "Execute now?" in text:
            return "n"
        return ""

    monkeypatch.setattr("builtins.input", _fake_input)
    rc = main(["nsys-panel"])
    assert rc == 0
    out = capsys.readouterr().out
    assert (
        "myutils-profile nsys-timeline-html --sqlite demo.sqlite --output out.html --include-metrics --no-default-focus-metrics --no-debug"
        in out
    )
    assert "--debug --no-debug" not in out
    assert "--default-focus-metrics --no-default-focus-metrics" not in out

    debug_prompts = [p for p in prompts if "--debug" in p or "--no-debug" in p]
    assert len(debug_prompts) == 1, debug_prompts
    focus_prompts = [
        p
        for p in prompts
        if "--default-focus-metrics" in p or "--no-default-focus-metrics" in p
    ]
    assert len(focus_prompts) == 1, focus_prompts
    debug_rows_prompts = [p for p in prompts if "--debug-rows" in p]
    assert len(debug_rows_prompts) == 0, debug_rows_prompts


def test_nsys_panel_respects_list_skills_short_circuit(monkeypatch, capsys) -> None:
    prompts: list[str] = []

    def _fake_input(prompt: str) -> str:
        prompts.append(str(prompt))
        text = str(prompt)
        line = text.strip()
        if text.startswith("command >"):
            return "nsys-sql-skill"
        if line.startswith("--sqlite"):
            return "skills.sqlite"
        if "Configure optional arguments?" in text:
            return "y"
        if line.startswith("--list-skills"):
            return "on"
        if line.startswith("--pretty"):
            return "on"
        if "Execute now?" in text:
            return "n"
        return ""

    monkeypatch.setattr("builtins.input", _fake_input)
    rc = main(["nsys-panel"])
    assert rc == 0
    out = capsys.readouterr().out
    assert (
        "myutils-profile nsys-sql-skill --sqlite skills.sqlite --list-skills --pretty"
        in out
    )
    assert "--skill" not in out.split("Generated command:")[-1]
    assert "--param" not in out.split("Generated command:")[-1]

    skill_prompts = [p for p in prompts if "--skill" in p]
    param_prompts = [p for p in prompts if "--param" in p]
    debug_prompts = [p for p in prompts if "--debug" in p or "--no-debug" in p]
    assert len(skill_prompts) == 0, skill_prompts
    assert len(param_prompts) == 0, param_prompts
    assert len(debug_prompts) == 0, debug_prompts


def test_ncu_csv_skill_and_analyze(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Kernel Name,Metric Name,Metric Value",
                "k1,time_metric,10",
                "k2,time_metric,20",
                "k1,other_metric,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_skill = tmp_path / "skill.json"
    rc = main(
        [
            "ncu-csv-skill",
            "--csv",
            str(csv_path),
            "--skill",
            "summary",
            "--output",
            str(out_skill),
            "--pretty",
        ]
    )
    assert rc == 0
    assert out_skill.exists()

    out_analyze = tmp_path / "analyze.json"
    rc = main(
        [
            "ncu-csv-analyze",
            "--csv",
            str(csv_path),
            "--output",
            str(out_analyze),
            "--pretty",
        ]
    )
    assert rc == 0
    assert out_analyze.exists()


def test_ncu_alias_entry_forwards_subcommand(monkeypatch) -> None:
    captured = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(profiling_cli, "main", _fake_main)
    monkeypatch.setattr(
        sys, "argv", ["ncu-csv-skill", "--csv", "x.csv", "--list-skills"]
    )
    rc = profiling_cli.entry_ncu_csv_skill()
    assert rc == 0
    assert captured["argv"][0] == "ncu-csv-skill"
