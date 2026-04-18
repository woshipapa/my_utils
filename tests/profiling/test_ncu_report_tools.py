from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_ncu_report_tools_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "my_utils"
        / "profiling"
        / "ncu"
        / "ncu_report_tools.py"
    )
    spec = importlib.util.spec_from_file_location("ncu_report_tools", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ncu_report_tools"] = module
    spec.loader.exec_module(module)
    return module


_MOD = _load_ncu_report_tools_module()
NcuReportSkillEngine = _MOD.NcuReportSkillEngine
analyze_ncu_report = _MOD.analyze_ncu_report
load_ncu_report_records = _MOD.load_ncu_report_records


class _FakeMetric:
    def __init__(self, value: object, unit: str = "") -> None:
        self._value = value
        self._unit = unit

    def value(self):
        return self._value

    def unit(self):
        return self._unit


class _FakeAction:
    def __init__(self, name: str, metrics: dict[str, _FakeMetric]) -> None:
        self._name = name
        self._metrics = dict(metrics)

    def name(self):
        return self._name

    def metric_names(self):
        return list(self._metrics.keys())

    def metric_by_name(self, key: str):
        return self._metrics[key]


class _FakeRange:
    def __init__(self, actions: list[_FakeAction]) -> None:
        self._actions = list(actions)

    def __iter__(self):
        return iter(self._actions)

    def num_actions(self):
        return len(self._actions)

    def action_by_idx(self, i: int):
        return self._actions[i]


class _FakeContext:
    def __init__(self, ranges: list[_FakeRange]) -> None:
        self._ranges = list(ranges)

    def __iter__(self):
        return iter(self._ranges)

    def num_ranges(self):
        return len(self._ranges)

    def range_by_idx(self, i: int):
        return self._ranges[i]


class _FakeNcuReportModule:
    def __init__(self, ctx: _FakeContext) -> None:
        self._ctx = ctx

    def load_report(self, _path: str):
        return self._ctx


def _fake_module() -> _FakeNcuReportModule:
    r0 = _FakeRange(
        [
            _FakeAction(
                "k1",
                {
                    "gpu__time_duration.sum": _FakeMetric("10", "ns"),
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric("50", "%"),
                },
            ),
            _FakeAction(
                "k2",
                {
                    "gpu__time_duration.sum": _FakeMetric("30", "ns"),
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric("20", "%"),
                },
            ),
        ]
    )
    return _FakeNcuReportModule(_FakeContext([r0]))


def test_load_ncu_report_records(tmp_path: Path) -> None:
    rep = tmp_path / "a.ncu-rep"
    rep.write_text("", encoding="utf-8")
    rows = load_ncu_report_records(
        str(rep),
        metric_like="%time%",
        ncu_report_module=_fake_module(),
    )
    assert len(rows) == 2
    assert rows[0].metric_name == "gpu__time_duration.sum"


def test_ncu_report_skill_engine_summary(tmp_path: Path) -> None:
    rep = tmp_path / "b.ncu-rep"
    rep.write_text("", encoding="utf-8")
    engine = NcuReportSkillEngine(str(rep), ncu_report_module=_fake_module())
    summary = engine.run_skill("summary", metric_like="%", top_k=10)
    assert isinstance(summary, dict)
    assert int(summary["metric_records"]) == 4
    per_metric = engine.run_skill("per_metric_stats", metric_like="%")
    assert isinstance(per_metric, list) and per_metric


def test_analyze_ncu_report(tmp_path: Path) -> None:
    rep = tmp_path / "c.ncu-rep"
    rep.write_text("", encoding="utf-8")
    payload = analyze_ncu_report(
        str(rep),
        top_k=10,
        metric_like="%",
        include_all_metrics=True,
        all_metrics_limit=100,
        ncu_report_module=_fake_module(),
    )
    assert isinstance(payload, dict)
    assert "per_metric_stats" in payload
    assert "all_metrics" in payload
