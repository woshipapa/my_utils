# SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic current-version NCU report-surface discovery."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "my_utils"
        / "profiling"
        / "ncu"
        / "current_report_surfaces.py"
    )
    spec = importlib.util.spec_from_file_location("current_report_surfaces", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


surfaces = _load_module()


class _Metric:
    def __init__(self, values, labels=()):
        self._values = list(values)
        self._labels = list(labels)

    def num_instances(self):
        return len(self._values)

    def value(self, index=None):
        if index is None:
            return sum(self._values)
        return self._values[index]

    def correlation_ids(self):
        return _Metric(self._labels) if self._labels else None


class _Action:
    def __init__(self):
        self.metrics = {
            "sass__inst_executed_per_opcode": _Metric([90, 10], ["FFMA", "LDG"]),
            "sass__instruction_size_by_opcode": _Metric([16, 16], ["FFMA", "LDG"]),
            "smsp__pcsamp_warp_id_stall_samples": _Metric([7, 2], [0, 1]),
        }

    def metric_names(self):
        return tuple(self.metrics)

    def metric_by_name(self, name):
        return self.metrics.get(name)

    def function_statistics(self):
        return ()


def test_discovers_new_surfaces_only_when_the_report_exposes_them() -> None:
    result = surfaces.discover_current_report_surfaces(_Action())
    assert result["sass_instruction_size"]["observed"] is True
    assert result["instruction_stats_hw_warp_id"]["observed"] is True
    assert result["function_statistics_line_time_range"]["observed"] is True


def test_opcode_breakdown_preserves_report_instance_labels() -> None:
    result = surfaces.summarize_instruction_breakdowns(_Action(), top_k=1)
    row = result["breakdowns"][0]
    assert row["available"] is True
    assert row["total"] == 100
    assert row["entries"] == [{"label": "FFMA", "value": 90.0, "share": 0.9}]


def test_current_surface_summary_keeps_unknown_new_metrics_as_data_not_findings() -> None:
    result = surfaces.summarize_current_report_surfaces(_Action())
    size = result["surfaces"]["sass_instruction_size"]
    assert size["available"] is True
    assert size["metrics"][0]["entries"][0]["value"] == 16.0
