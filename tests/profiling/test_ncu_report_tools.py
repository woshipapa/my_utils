# SPDX-License-Identifier: Apache-2.0
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
    def __init__(
        self,
        name: str,
        metrics: dict[str, _FakeMetric],
        rules: list[dict[str, object]] | None = None,
    ) -> None:
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
                {
                    "name": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "value": 85.0,
                    "severity": "HIGH",
                }
            ],
        }
    ]
    r0 = _FakeRange(
        [
            _FakeAction(
                "k1",
                {
                    "gpu__time_duration.sum": _FakeMetric("10", "ns"),
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                        "35", "%"
                    ),
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                        "85", "%"
                    ),
                    "smsp__issue_active.avg.pct_of_peak_sustained_active": _FakeMetric(
                        "40", "%"
                    ),
                    "smsp__warps_eligible.avg": _FakeMetric("0.9", ""),
                    "smsp__pcsamp_warps_issue_stalled_long_scoreboard": _FakeMetric(
                        "68", "%"
                    ),
                    "memory_ideal_l2_transactions_global": _FakeMetric("100", ""),
                    "memory_l2_transactions_global": _FakeMetric("180", ""),
                    "smsp__branch_divergence": _FakeMetric("32", "%"),
                    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum": _FakeMetric(
                        "4", ""
                    ),
                },
                rules=k1_rules,
            ),
            _FakeAction(
                "k2",
                {
                    "gpu__time_duration.sum": _FakeMetric("30", "ns"),
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                        "20", "%"
                    ),
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
                    "sm__warps_active.avg.pct_of_peak_sustained_active": _FakeMetric(
                        "25", "%"
                    ),
                    "sm__cycles_active.avg": _FakeMetric("100", "cycle"),
                    "sm__cycles_active.max": _FakeMetric("300", "cycle"),
                    "sm__cycles_active.min": _FakeMetric("20", "cycle"),
                    "smsp__pcsamp_sample_count": _FakeMetric("100", ""),
                    "smsp__pcsamp_warps_issue_stalled_long_scoreboard": _FakeMetric(
                        "45", ""
                    ),
                    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                        "0", "%"
                    ),
                    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": _FakeMetric(
                        "60", "%"
                    ),
                    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": _FakeMetric(
                        "1", "%"
                    ),
                    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": _FakeMetric(
                        "600", ""
                    ),
                    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": _FakeMetric(
                        "100", ""
                    ),
                    "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio": _FakeMetric(
                        "8", ""
                    ),
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
                    "device__attribute_max_warps_per_multiprocessor": _FakeMetric(
                        "64", ""
                    ),
                    "launch__grid_size": _FakeMetric("96", ""),
                    "launch__waves_per_multiprocessor": _FakeMetric("0.75", ""),
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed": _FakeMetric(
                        "8", "%"
                    ),
                    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": _FakeMetric(
                        "6.5", ""
                    ),
                    "smsp__inst_executed_op_global_ld.sum": _FakeMetric("200", ""),
                    "smsp__inst_executed_op_global_st.sum": _FakeMetric("50", ""),
                    "smsp__average_data_bytes_per_sector_mem_global_op_st.ratio": _FakeMetric(
                        "12", ""
                    ),
                    "smsp__inst_executed_op_local_ld.sum": _FakeMetric("1", ""),
                    "smsp__inst_executed_op_local_st.sum": _FakeMetric("1", ""),
                    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": _FakeMetric(
                        "4.5", ""
                    ),
                    "pmsampling:smsp__warps_issue_stalled_long_scoreboard.avg": _FakeMetric(
                        "3.0", ""
                    ),
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
        item for item in report["dimensions"] if item["key"] == "memory_access_cache"
    )
    sectors = memory_dim["signals"]["sectors_per_ld_request"]
    assert sectors["value"] == 6.0


def test_ncu_dimension_report_supports_h100_metric_aliases(tmp_path: Path) -> None:
    rep = tmp_path / "h100.ncu-rep"
    rep.write_text("", encoding="utf-8")
    engine = NcuReportSkillEngine(
        str(rep), ncu_report_module=_fake_h100_dimension_module()
    )
    report = engine.run_skill("dimension_report", top_k=20)
    assert isinstance(report, dict)
    assert report["architecture"]["family"] == "hopper"
    assert report["architecture"]["alias"] == "h100/sm_90"
    assert report["architecture"]["compute_capability"] == "9.0"

    memory_dim = next(
        item for item in report["dimensions"] if item["key"] == "memory_access_cache"
    )
    signals = memory_dim["signals"]
    assert signals["sectors_per_ld_request"]["value"] == 6.5
    assert (
        signals["global_ld_instructions"]["metric_name"]
        == "smsp__inst_executed_op_global_ld.sum"
    )
    assert signals["local_ld"]["metric_name"] == "smsp__inst_executed_op_local_ld.sum"

    findings = {
        str(item.get("category"))
        for item in report.get("top_findings", [])
        if isinstance(item, dict)
    }
    assert "small_grid" in findings
    assert "uncoalesced_global_loads" in findings
    assert "register_spill" in findings


# ---------------------------------------------------------------------------
# Diagnosis-path tests moved from test_analysis_engine.py.  These exercise
# the torch-free file-path-loaded module (_prof.ncu.ncu_report_tools) via
# the shared synthetic loader.
# ---------------------------------------------------------------------------

import types
import pytest

from _synthetic_loader import ncu_diagnostics, ncu_report_tools


class TestReportDiagnosisUsesShippedRules:
    """The CLI path must cross-check against NVIDIA's rules, not just the API.

    `diagnose_kernel` accepted `shipped_rules=` from the start, but
    `diagnose_ncu_report` -- what `mp ncu-diagnose` actually runs -- never
    passed them, so corroboration reported "no shipped rules" on every real
    report. A feature reachable only from a hand-written call is not reachable.
    """

    def _fake_module(self, with_rules=True):
        class M:
            def __init__(self, v):
                self.v = v

            def value(self):
                return self.v

            def as_double(self):
                return self.v

            def as_uint64(self):
                return int(self.v)

            def unit(self):
                return ""

            def has_correlation_ids(self):
                return False

        values = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        }

        class Action:
            def name(self):
                return "gemm_kernel"

            def metric_names(self):
                return list(values)

            def metric_by_name(self, k):
                return M(values.get(k, 0.0))

            def rule_results_as_dicts(self):
                if not with_rules:
                    return []
                return [
                    {
                        "rule_identifier": "SOLBottleneck",
                        "section_identifier": "SpeedOfLight",
                        "rule_message": {
                            "title": "Memory more utilized",
                            "message": "This kernel is memory bound.",
                            "message_type": "warning",
                        },
                        "speedup_estimation": {"type": "GLOBAL", "speedup": 25.0},
                    }
                ]

        class Rng:
            num_actions = 1

            def action_by_idx(self, i):
                return Action()

        class Ctx:
            num_ranges = 1

            def range_by_idx(self, i):
                return Rng()

        return types.SimpleNamespace(load_report=lambda p: Ctx())

    def _first(self, out):
        return (out.get("kernels") or out.get("diagnoses"))[0]

    def test_shipped_rules_reach_the_diagnosis(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._fake_module()
        )
        assert self._first(out)["corroboration"]["shipped_rules_available"] is True

    def test_disagreement_with_nvidia_surfaces_through_the_report_path(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._fake_module()
        )
        assert self._first(out)["corroboration"]["conflicts"]

    def test_report_without_rules_is_reported_honestly(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._fake_module(with_rules=False)
        )
        assert self._first(out)["corroboration"]["shipped_rules_available"] is False


class TestDiagnoseIsSelfContained:
    """`ncu-diagnose` must answer both what and where, in one command.

    Source attribution used to be reachable only through a separate SkillEngine
    call, so the question a fused kernel most needs answered -- which line
    stalls -- was absent from the command people actually run.
    """

    def _module(self, with_samples=True):
        class Stall:
            def __init__(self, n):
                self.name = n

        class M:
            def __init__(self, v):
                self.v = v

            def value(self):
                return self.v

            def as_double(self):
                return self.v

            def as_uint64(self):
                return int(self.v)

            def unit(self):
                return ""

            def has_correlation_ids(self):
                return False

        vals = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 32.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 28.0,
            "smsp__pcsamp_sample_count": 5000.0,
            "smsp__pcsamp_interval_cycles": 1000.0,
        }

        class Action:
            def name(self):
                return "fused_attn_fwd"

            def metric_names(self):
                return list(vals)

            def metric_by_name(self, k):
                return M(vals[k]) if k in vals else None

            def rule_results_as_dicts(self):
                return []

            def source_files(self):
                return {"attn.cu": "load\nsoftmax\nmm\n"}

            def source_info(self, a):
                table = {0x10: ("attn.cu", 1), 0x20: ("attn.cu", 2)}
                if a not in table:
                    return None
                fname, ln = table[a]

                class I:
                    def file_name(self):
                        return fname

                    def line(self):
                        return ln

                return I()

            def sass_by_pc(self, a):
                return ""

            def ptx_by_pc(self, a):
                return ""

            def timed_warp_samples(self):
                if not with_samples:
                    return []
                return [
                    {
                        "timestamp": i * 100,
                        "pc": 0x20,
                        "stall_reason": Stall("MIO_THROTTLE"),
                        "not_issued": True,
                    }
                    for i in range(600)
                ]

        class Rng:
            num_actions = 1

            def action_by_idx(self, i):
                return Action()

        class Ctx:
            num_ranges = 1

            def range_by_idx(self, i):
                return Rng()

        return types.SimpleNamespace(load_report=lambda p: Ctx())

    def test_source_attribution_is_in_the_diagnosis(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._module()
        )
        kernel = out["kernels"][0]
        assert "source_attribution" in kernel
        rows = kernel["source_attribution"]["stall_attribution"]["source_lines"]
        assert rows and rows[0]["line"] == 2

    def test_markdown_renders_where_it_stalls(self):
        text = ncu_report_tools.diagnose_result_to_markdown(
            ncu_report_tools.diagnose_ncu_report(
                "/dev/null", ncu_report_module=self._module()
            )
        )
        assert "### Where it stalls" in text
        assert "MIO_THROTTLE" in text

    def test_no_contradiction_when_attribution_succeeds(self):
        """Do not print 'no source data' directly beneath the source data."""
        text = ncu_report_tools.diagnose_result_to_markdown(
            ncu_report_tools.diagnose_ncu_report(
                "/dev/null", ncu_report_module=self._module()
            )
        )
        assert "### Where it stalls" in text
        assert "No source-correlated metrics" not in text

    def test_include_source_false_skips_it(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", include_source=False, ncu_report_module=self._module()
        )
        assert "source_attribution" not in out["kernels"][0]

    def test_absent_samples_do_not_break_the_diagnosis(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._module(with_samples=False)
        )
        kernel = out["kernels"][0]
        assert kernel["verdict"]
        assert kernel["source_attribution"]["stall_attribution"]["available"] is False


class TestNcuReportModuleDiscovery:
    """`ncu_report` ships with Nsight Compute, not on PyPI.

    The first-run failure for anyone with a real report is an ImportError whose
    fix is a PYTHONPATH entry, not a pip install. Discovery removes the step
    where it can be found; the error message covers when it cannot.
    """

    def test_discovery_returns_a_dir_or_none_never_a_guess(self):
        found = ncu_report_tools.find_ncu_report_dir()
        assert found is None or (found / "ncu_report.py").exists()

    def test_error_message_is_actionable(self):

        source = Path(ncu_report_tools.__file__).read_text()
        block = source.split("The `ncu_report` module is required")[1][:1200]
        assert "PYTHONPATH" in block
        assert "not on PyPI" in block or "nothing to" in block
        assert "find /" in block, "must tell the user how to locate it"


class TestReportIsReadOnce:
    """Four full traversals of a --set full report was three too many.

    `diagnose_ncu_report` called four loaders -- metrics, shipped rules, source
    attribution, and one just to retain the action objects -- each of which
    opened the report and walked every range and action. `walk_report_once`
    visits each action a single time and gathers all four.
    """

    def _counting_module(self):
        opens = {"n": 0}

        class Stall:
            def __init__(self, n):
                self.name = n

        class M:
            def __init__(self, v):
                self.v = v

            def value(self):
                return self.v

            def as_double(self):
                return self.v

            def as_uint64(self):
                return int(self.v)

            def unit(self):
                return ""

            def has_correlation_ids(self):
                return False

        vals = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 34.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 72.0,
            "smsp__pcsamp_sample_count": 8000.0,
            "smsp__pcsamp_interval_cycles": 1000.0,
            "gpu__time_duration.sum": 410000.0,
        }

        class Action:
            def name(self):
                return "k"

            def metric_names(self):
                return list(vals)

            def metric_by_name(self, k):
                return M(vals[k]) if k in vals else None

            def rule_results_as_dicts(self):
                return [
                    {
                        "rule_identifier": "SOLBottleneck",
                        "rule_message": {
                            "title": "t",
                            "message": "m",
                            "message_type": "optimization",
                        },
                        "speedup_estimation": {"type": "GLOBAL", "speedup": 20.0},
                    }
                ]

            def source_files(self):
                return {"k.cu": "a\nb\n"}

            def source_info(self, a):
                if a != 0x10:
                    return None

                class I:
                    def file_name(self):
                        return "k.cu"

                    def line(self):
                        return 1

                return I()

            def sass_by_pc(self, a):
                return ""

            def ptx_by_pc(self, a):
                return ""

            def timed_warp_samples(self):
                return [
                    {
                        "timestamp": i,
                        "pc": 0x10,
                        "stall_reason": Stall("LONG_SCOREBOARD"),
                        "not_issued": True,
                    }
                    for i in range(400)
                ]

        class Rng:
            num_actions = 1

            def action_by_idx(self, i):
                return Action()

        class Ctx:
            num_ranges = 1

            def range_by_idx(self, i):
                return Rng()

        def loader(path):
            opens["n"] += 1
            return Ctx()

        return types.SimpleNamespace(load_report=loader), opens

    def test_report_is_opened_exactly_once(self):
        module, opens = self._counting_module()
        ncu_report_tools.diagnose_ncu_report("/dev/null", ncu_report_module=module)
        assert opens["n"] == 1, f"report opened {opens['n']} times; expected 1"

    def test_single_pass_still_produces_every_section(self):
        module, _ = self._counting_module()
        kernel = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=module
        )["kernels"][0]
        for key in (
            "verdict",
            "coverage",
            "axes",
            "metric_inventory",
            "corroboration",
            "signal_scan",
            "source_attribution",
            "duration_ns",
        ):
            assert key in kernel, f"single-pass rewrite dropped `{key}`"
        assert kernel["corroboration"]["shipped_rules_available"] is True
        assert kernel["source_attribution"]["stall_attribution"]["available"] is True

    def test_no_source_still_reads_once(self):
        module, opens = self._counting_module()
        ncu_report_tools.diagnose_ncu_report(
            "/dev/null", include_source=False, ncu_report_module=module
        )
        assert opens["n"] == 1


class TestStringValuedMetrics:
    """21 metrics on a real report have string values, and they are not noise.

    They were dropped as unparseable. Among them: the GPU model (which the
    caller was being asked to supply by hand), the constituent lists behind each
    Speed-of-Light rollup, and the launch scheduling policy.
    """

    class _Action:
        _NUM = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 33.7,
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 33.7,
            "sm__issue_active.avg.pct_of_peak_sustained_elapsed": 14.1,
        }
        _STR = {
            "device__attribute_display_name": "NVIDIA H100 80GB HBM3",
            "launch__cluster_scheduling_policy": "PolicySpread",
            "breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
            "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
        }

        class _M:
            def __init__(self, v):
                self.v = v

            def value(self):
                return self.v

            def as_double(self):
                return self.v if isinstance(self.v, float) else None

            def as_string(self):
                return self.v if isinstance(self.v, str) else None

            def unit(self):
                return ""

        def name(self):
            return "k"

        def metric_names(self):
            return list(self._NUM) + list(self._STR)

        def metric_by_name(self, n):
            if n in self._NUM:
                return self._M(self._NUM[n])
            if n in self._STR:
                return self._M(self._STR[n])
            return None

    def test_string_metrics_are_kept_not_dropped(self):
        numeric, text = ncu_report_tools._metrics_for_action(self._Action())
        assert len(numeric) == 3 and len(text) == 3
        assert text["device__attribute_display_name"] == "NVIDIA H100 80GB HBM3"

    def test_gpu_name_comes_from_the_report(self):
        _, text = ncu_report_tools._metrics_for_action(self._Action())
        assert ncu_report_tools.gpu_name_from_report(text) == "NVIDIA H100 80GB HBM3"

    def test_sol_breakdown_names_the_driving_constituent(self):
        """A SOL throughput is a max over constituents, not an average."""
        numeric, text = ncu_report_tools._metrics_for_action(self._Action())
        out = ncu_report_tools.resolve_sol_breakdown(text, numeric)
        entry = out["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
        assert entry["rollup_value"] == pytest.approx(33.7)
        top = entry["top_constituents"][0]
        assert "pipe_tensor" in top["metric"], "the max constituent drives the rollup"
        assert "maximum over these" in entry["note"]

    def test_inventory_counts_string_metrics(self):
        result = ncu_diagnostics.diagnose_kernel(
            {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 33.7},
            string_metrics={"device__attribute_display_name": "NVIDIA H100"},
        )
        inventory = result["metric_inventory"]
        assert inventory["string_valued_count"] == 1
        assert inventory["total_including_string"] == inventory["total"] + 1
        assert "not lost" in inventory["summary"]

    def test_breakdown_with_unresolvable_constituents_is_skipped(self):
        out = ncu_report_tools.resolve_sol_breakdown(
            {"breakdown:x": "not_collected_a,not_collected_b"}, {}
        )
        assert out == {}
