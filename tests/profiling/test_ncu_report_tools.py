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
load_ncu_report_rule_rows = _MOD.load_ncu_report_rule_rows


class _FakeMetric:
    def __init__(self, value: object, unit: str = "") -> None:
        self._value = value
        self._unit = unit

    def value(self):
        return self._value

    def unit(self):
        return self._unit


class _FakeAction:
    def __init__(self, name: str, metrics: dict[str, _FakeMetric], rules: list[dict[str, object]] | None = None) -> None:
        self._name = name
        self._metrics = dict(metrics)
        self._rules = list(rules or [])

    def name(self):
        return self._name

    def metric_names(self):
        return list(self._metrics.keys())

    def metric_by_name(self, key: str):
        return self._metrics[key]

    def rule_results_as_dicts(self):
        return list(self._rules)


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
    k1_rules = [
        {
            "rule_identifier": "SpeedOfLight_Memory",
            "name": "Memory Throughput",
            "section_identifier": "SpeedOfLight",
            "rule_message": {
                "title": "Memory Throughput is High",
                "message_type": "WARNING",
                "message": "DRAM throughput is close to peak.",
            },
            "speedup_estimation": {"type": "GLOBAL", "speedup": 21.5},
            "focus_metrics": [
                {"name": "dram__throughput.avg.pct_of_peak_sustained_elapsed", "value": 85.0, "severity": "HIGH"}
            ],
        }
    ]
    r0 = _FakeRange(
        [
            _FakeAction(
                "k1",
                {
                    "gpu__time_duration.sum": _FakeMetric("10", "ns"),
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric("35", "%"),
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric("85", "%"),
                    "smsp__issue_active.avg.pct_of_peak_sustained_active": _FakeMetric("40", "%"),
                    "smsp__warps_eligible.avg": _FakeMetric("0.9", ""),
                    "smsp__pcsamp_warps_issue_stalled_long_scoreboard": _FakeMetric("68", "%"),
                    "memory_ideal_l2_transactions_global": _FakeMetric("100", ""),
                    "memory_l2_transactions_global": _FakeMetric("180", ""),
                    "smsp__branch_divergence": _FakeMetric("32", "%"),
                    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": _FakeMetric("4", ""),
                },
                rules=k1_rules,
            ),
            _FakeAction(
                "k2",
                {
                    "gpu__time_duration.sum": _FakeMetric("30", "ns"),
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric("20", "%"),
                    "launch__occupancy_limit_registers": _FakeMetric("1", ""),
                },
            ),
        ]
    )
    return _FakeNcuReportModule(_FakeContext([r0]))


def _fake_dimension_module() -> _FakeNcuReportModule:
    r0 = _FakeRange(
        [
            _FakeAction(
                "dim_kernel",
                {
                    "launch__grid_size": _FakeMetric("64", ""),
                    "launch__block_size": _FakeMetric("128", ""),
                    "launch__waves_per_multiprocessor": _FakeMetric("0.5", ""),
                    "device__attribute_multiprocessor_count": _FakeMetric("148", ""),
                    "launch__registers_per_thread": _FakeMetric("160", ""),
                    "sm__maximum_warps_per_active_cycle_pct": _FakeMetric("100", "%"),
                    "sm__warps_active.avg.pct_of_peak_sustained_active": _FakeMetric("25", "%"),
                    "sm__cycles_active.avg": _FakeMetric("100", "cycle"),
                    "sm__cycles_active.max": _FakeMetric("300", "cycle"),
                    "sm__cycles_active.min": _FakeMetric("20", "cycle"),
                    "smsp__pcsamp_sample_count": _FakeMetric("100", ""),
                    "smsp__pcsamp_warps_issue_stalled_long_scoreboard": _FakeMetric("45", ""),
                    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": _FakeMetric("0", "%"),
                    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": _FakeMetric("60", "%"),
                    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": _FakeMetric("1", "%"),
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": _FakeMetric("600", ""),
                    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": _FakeMetric("100", ""),
                    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio": _FakeMetric("8", ""),
                    "smsp__sass_inst_executed_op_local_ld.sum": _FakeMetric("2", ""),
                    "smsp__sass_inst_executed_op_local_st.sum": _FakeMetric("0", ""),
                },
            )
        ]
    )
    return _FakeNcuReportModule(_FakeContext([r0]))


def _fake_h100_dimension_module() -> _FakeNcuReportModule:
    r0 = _FakeRange(
        [
            _FakeAction(
                "h100_kernel",
                {
                    "device__attribute_compute_capability_major": _FakeMetric("9", ""),
                    "device__attribute_compute_capability_minor": _FakeMetric("0", ""),
                    "device__attribute_multiprocessor_count": _FakeMetric("132", ""),
                    "device__attribute_max_warps_per_multiprocessor": _FakeMetric("64", ""),
                    "launch__grid_size": _FakeMetric("96", ""),
                    "launch__waves_per_multiprocessor": _FakeMetric("0.75", ""),
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric("8", "%"),
                    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": _FakeMetric("6.5", ""),
                    "smsp__inst_executed_op_global_ld.sum": _FakeMetric("200", ""),
                    "smsp__inst_executed_op_global_st.sum": _FakeMetric("50", ""),
                    "smsp__average_data_bytes_per_sector_mem_global_op_st.ratio": _FakeMetric("12", ""),
                    "smsp__inst_executed_op_local_ld.sum": _FakeMetric("1", ""),
                    "smsp__inst_executed_op_local_st.sum": _FakeMetric("1", ""),
                    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": _FakeMetric("4.5", ""),
                    "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg": _FakeMetric("3.0", ""),
                },
            )
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
    assert int(summary["metric_records"]) == 13
    per_metric = engine.run_skill("per_metric_stats", metric_like="%")
    assert isinstance(per_metric, list) and per_metric
    bottleneck = engine.run_skill("bottleneck_report", metric_like="%", top_k=10)
    assert isinstance(bottleneck, dict)
    assert "top_bottlenecks" in bottleneck
    heuristic = bottleneck.get("heuristic_findings", [])
    categories = {str(x.get("category")) for x in heuristic if isinstance(x, dict)}
    assert "global_memory_coalescing" in categories
    dimension_report = bottleneck.get("dimension_report", {})
    assert isinstance(dimension_report, dict)
    assert "dimensions" in dimension_report


def test_load_ncu_report_rule_rows(tmp_path: Path) -> None:
    rep = tmp_path / "r.ncu-rep"
    rep.write_text("", encoding="utf-8")
    rows = load_ncu_report_rule_rows(str(rep), ncu_report_module=_fake_module())
    assert isinstance(rows, list)
    assert rows
    assert rows[0]["rule_identifier"] == "SpeedOfLight_Memory"


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
    assert "bottleneck_report" in payload
    assert "rule_results" in payload


def test_ncu_dimension_report_detects_codex_skill_patterns(tmp_path: Path) -> None:
    rep = tmp_path / "dim.ncu-rep"
    rep.write_text("", encoding="utf-8")
    engine = NcuReportSkillEngine(str(rep), ncu_report_module=_fake_dimension_module())
    assert "dimension_report" in engine.list_skills()

    report = engine.run_skill("dimension_report", top_k=20)
    assert isinstance(report, dict)
    findings = {
        str(item.get("category"))
        for item in report.get("top_findings", [])
        if isinstance(item, dict)
    }
    assert "small_grid" in findings
    assert "dominant_stall" in findings
    assert "scalar_fma_no_tensor_core" in findings
    assert "fp64_activity" in findings
    assert "uncoalesced_global_loads" in findings
    assert "register_spill" in findings

    memory_dim = next(
        item
        for item in report["dimensions"]
        if item["key"] == "memory_access_cache"
    )
    sectors = memory_dim["signals"]["sectors_per_ld_request"]
    assert sectors["value"] == 6.0


def test_ncu_dimension_report_supports_h100_metric_aliases(tmp_path: Path) -> None:
    rep = tmp_path / "h100.ncu-rep"
    rep.write_text("", encoding="utf-8")
    engine = NcuReportSkillEngine(str(rep), ncu_report_module=_fake_h100_dimension_module())
    report = engine.run_skill("dimension_report", top_k=20)
    assert isinstance(report, dict)
    assert report["architecture"]["family"] == "hopper"
    assert report["architecture"]["alias"] == "h100/sm_90"
    assert report["architecture"]["compute_capability"] == "9.0"

    memory_dim = next(
        item
        for item in report["dimensions"]
        if item["key"] == "memory_access_cache"
    )
    signals = memory_dim["signals"]
    assert signals["sectors_per_ld_request"]["value"] == 6.5
    assert signals["global_ld_instructions"]["metric_name"] == "smsp__inst_executed_op_global_ld.sum"
    assert signals["local_ld"]["metric_name"] == "smsp__inst_executed_op_local_ld.sum"

    findings = {
        str(item.get("category"))
        for item in report.get("top_findings", [])
        if isinstance(item, dict)
    }
    assert "small_grid" in findings
    assert "uncoalesced_global_loads" in findings
    assert "register_spill" in findings
