# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.ncu.source_correlation, plus the report-rendering checks that reuse its fake-action machinery."""

from __future__ import annotations


import types
import pytest


from _synthetic_loader import (
    ncu_report_tools,
    section_index,
    signal_scan,
    source_correlation,
    source_correlation_mod,
)


class _FakeStall:
    def __init__(self, name):
        self.name = name


class _FakeMetric:
    """Mirrors the IMetric surface verified against ncu_report 2026.1.1."""

    def __init__(self, values, correlation=None):
        self._values = values
        self._correlation = correlation

    def num_instances(self):
        return len(self._values)

    def as_double(self, index):
        return float(self._values[index])

    def has_correlation_ids(self):
        return self._correlation is not None

    def correlation_ids(self):
        return _FakeMetric(self._correlation) if self._correlation else None


class _FakeAction:
    """Mirrors the IAction surface: source_info/sass_by_pc/source_files/..."""

    def __init__(
        self, *, metrics=None, sources=None, lines=None, sass=None, samples=None
    ):
        self._metrics = metrics or {}
        self._sources = sources or {}
        self._lines = lines or {}  # address -> (file, line)
        self._sass = sass or {}
        self._samples = samples or []

    def metric_names(self):
        return list(self._metrics)

    def metric_by_name(self, name):
        return self._metrics.get(name)

    def source_files(self):
        return dict(self._sources)

    def source_info(self, address):
        entry = self._lines.get(address)
        if entry is None:
            return None
        file_name, line = entry

        class _Info:
            def file_name(self_inner):
                return file_name

            def line(self_inner):
                return line

        return _Info()

    def sass_by_pc(self, address):
        return self._sass.get(address, "")

    def ptx_by_pc(self, address):
        return ""

    def timed_warp_samples(self):
        return list(self._samples)


class TestSourceAvailability:
    def test_basic_set_is_diagnosed_as_the_cause(self):
        """The most common cause: a bare ncu run collects no source metrics."""
        report = source_correlation.source_availability(_FakeAction())
        assert report["source_correlation_possible"] is False
        assert any(
            "basic" in r and "SourceCounters" in r
            for r in report["reasons_unavailable"]
        )

    def test_missing_lineinfo_is_distinguished_from_missing_section(self):
        action = _FakeAction(
            metrics={
                "sass__inst_executed": _FakeMetric([1.0], correlation=[0x10]),
            }
        )
        report = source_correlation.source_availability(action)
        assert report["source_correlation_possible"] is True
        assert any("-lineinfo" in r for r in report["reasons_unavailable"])

    def test_empty_file_content_suggests_import_source(self):
        action = _FakeAction(
            metrics={"sass__inst_executed": _FakeMetric([1.0], correlation=[0x10])},
            sources={"kernel.cu": ""},
        )
        report = source_correlation.source_availability(action)
        assert any("--import-source" in r for r in report["reasons_unavailable"])


class TestMetricToSourceCorrelation:
    def _action(self):
        return _FakeAction(
            metrics={
                "sass__inst_executed": _FakeMetric(
                    [10.0, 90.0, 5.0], correlation=[0x10, 0x20, 0x30]
                )
            },
            sources={"kernel.cu": "line one\nline two\nline three\n"},
            lines={0x10: ("kernel.cu", 1), 0x20: ("kernel.cu", 2)},
            sass={0x10: "LDG.E R0", 0x20: "FFMA R2, R0, R1", 0x30: "EXIT"},
        )

    def test_hot_line_is_ranked_first_with_its_source_text(self):
        out = source_correlation.correlate_metric_to_source(
            self._action(), "sass__inst_executed"
        )
        assert out["available"] is True
        top = out["source_lines"][0]
        assert top["line"] == 2 and top["source_text"] == "line two"
        assert "FFMA" in top["sass_samples"][0]

    def test_unlocated_instructions_are_counted_not_hidden(self):
        out = source_correlation.correlate_metric_to_source(
            self._action(), "sass__inst_executed"
        )
        assert out["unlocated_value"] == pytest.approx(5.0)
        assert "could not be tied to a source line" in out["note"]

    def test_whole_kernel_metric_says_so_rather_than_returning_empty(self):
        action = _FakeAction(metrics={"sm__throughput": _FakeMetric([50.0])})
        out = source_correlation.correlate_metric_to_source(action, "sm__throughput")
        assert out["available"] is False
        assert "whole-kernel total" in out["reason"]

    def test_absent_metric_is_reported(self):
        out = source_correlation.correlate_metric_to_source(_FakeAction(), "nope")
        assert out["available"] is False and "not in this report" in out["reason"]


class TestStallAttribution:
    def _action(self):
        samples = [
            {
                "timestamp": 1000 + i,
                "pc": 0x20,
                "stall_reason": _FakeStall("LONG_SCOREBOARD"),
                "not_issued": True,
            }
            for i in range(30)
        ] + [
            {
                "timestamp": 2000 + i,
                "pc": 0x10,
                "stall_reason": _FakeStall("WAIT"),
                "not_issued": False,
            }
            for i in range(5)
        ]
        return _FakeAction(
            sources={"kernel.cu": "load here\ncompute here\n"},
            lines={0x10: ("kernel.cu", 1), 0x20: ("kernel.cu", 2)},
            sass={0x20: "LDG.E.128 R4"},
            samples=samples,
        )

    def test_stalls_are_attributed_to_the_hottest_line(self):
        out = source_correlation.attribute_stalls_to_source(self._action())
        assert out["available"] is True
        top = out["source_lines"][0]
        assert top["line"] == 2
        assert top["dominant_stall_reason"] == "LONG_SCOREBOARD"
        assert top["source_text"] == "compute here"

    def test_sample_count_confidence_is_stated(self):
        out = source_correlation.attribute_stalls_to_source(self._action())
        assert "statistical" in out["confidence_note"]
        assert out["total_samples"] == 35

    def test_missing_samples_are_explained(self):
        out = source_correlation.attribute_stalls_to_source(_FakeAction())
        assert out["available"] is False
        assert "SourceCounters" in out["reason"] or "set full" in out["reason"]


class TestPcSamplingTimeline:
    def test_phase_change_is_detected_not_averaged_away(self):
        """A kernel memory-bound then compute-bound must not average to 'mediocre'."""
        samples = [
            {
                "timestamp": i * 1000,
                "pc": 0x10,
                "stall_reason": _FakeStall("LONG_SCOREBOARD"),
                "not_issued": True,
            }
            for i in range(100)
        ] + [
            {
                "timestamp": 200_000 + i * 1000,
                "pc": 0x20,
                "stall_reason": _FakeStall("MATH_PIPE_THROTTLE"),
                "not_issued": False,
            }
            for i in range(100)
        ]
        out = source_correlation.pc_sampling_timeline(
            _FakeAction(samples=samples), bucket_ns=50_000
        )
        assert out["available"] is True
        assert out["phase_change_count"] >= 1
        assert "LONG_SCOREBOARD" in out["phase_sequence"]
        assert "MATH_PIPE_THROTTLE" in out["phase_sequence"]
        assert "wrong fix for each phase" in out["note"]

    def test_uniform_kernel_says_the_average_is_representative(self):
        samples = [
            {
                "timestamp": i * 1000,
                "pc": 0x10,
                "stall_reason": _FakeStall("WAIT"),
                "not_issued": False,
            }
            for i in range(200)
        ]
        out = source_correlation.pc_sampling_timeline(
            _FakeAction(samples=samples), bucket_ns=50_000
        )
        assert out["phase_change_count"] == 0
        assert "representative" in out["note"]

    def test_summary_names_its_relationship_to_warpstatestats(self):
        samples = [
            {
                "timestamp": 1,
                "pc": 0x10,
                "stall_reason": _FakeStall("BARRIER"),
                "not_issued": True,
            }
        ]
        out = source_correlation.summarize_warp_samples(_FakeAction(samples=samples))
        assert "WarpStateStats" in out["comparison_note"]


class TestSignalToSourceLinkage:
    """Join what-is-wrong to where-it-happens, and refuse invented joins."""

    def _attribution(self):
        return {
            "available": True,
            "stall_reasons": {"LONG_SCOREBOARD": 700, "MATH_PIPE_THROTTLE": 100},
            "source_lines": [
                {
                    "file_name": "attn.cu",
                    "line": 1,
                    "source_text": "load_qkv",
                    "samples": 700,
                    "stall_reasons": {"LONG_SCOREBOARD": 700},
                    "sass_samples": ["LDG.E.128"],
                },
                {
                    "file_name": "attn.cu",
                    "line": 3,
                    "source_text": "mm(p,v)",
                    "samples": 100,
                    "stall_reasons": {"MATH_PIPE_THROTTLE": 100},
                    "sass_samples": [],
                },
            ],
        }

    def _link(self, category):
        return source_correlation_mod.link_findings_to_source(
            [{"category": category, "title": "t"}],
            None,
            attribution=self._attribution(),
        )

    def test_memory_finding_lands_on_the_loading_line(self):
        out = self._link("uncoalesced_global_load")
        assert out["linked"], "f-string category must match by prefix"
        row = out["linked"][0]["source_lines"][0]
        assert row["line"] == 1 and row["share_of_reason"] == pytest.approx(1.0)

    def test_stall_category_localises_itself(self):
        out = self._link("stall_long_scoreboard")
        assert out["linked"][0]["source_lines"][0]["line"] == 1

    def test_generic_scan_findings_are_not_linked(self):
        """A saturated constant cache has no known relation to any stall reason."""
        assert self._link("unit_saturated")["linked"] == []
        assert self._link("unit_duty_cycle")["linked"] == []

    def test_launch_geometry_findings_are_not_linked(self):
        """Grid shape is a property of the launch, not of a line."""
        assert self._link("small_grid")["linked"] == []
        assert self._link("tile_quantization")["linked"] == []

    def test_measurement_faults_are_not_linked(self):
        assert self._link("measurement_above_physical_limit")["linked"] == []

    def test_link_carries_its_own_caveat(self):
        out = self._link("uncoalesced_global_load")
        assert "not proof of cause" in out["linked"][0]["caveat"]

    def test_absent_attribution_is_reported_not_faked(self):
        out = source_correlation_mod.link_findings_to_source(
            [{"category": "uncoalesced_global_load", "title": "t"}],
            None,
            attribution={"available": False, "reason": "no samples"},
        )
        assert out["available"] is False and out["linked"] == []


class TestRealReportRegressions:
    """Three bugs a real H100 report exposed that no fixture had.

    Each survived because my fixtures happened to use the shape my code
    assumed: raw warp samples where a real report has aggregated metrics, and a
    plain dict where ncu_report hands back a SWIG map.
    """

    class _SwigLikeMap:
        """Supports keys()/__getitem__ but is NOT a collections.abc.Mapping.

        This is what `IAction.source_files()` actually returns. An
        `isinstance(..., Mapping)` guard discarded all 64 source files on a real
        report and then reported the kernel as built without -lineinfo.
        """

        def __init__(self, data):
            self._data = dict(data)

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    def test_swig_map_is_not_discarded(self):
        files = source_correlation._as_dict(
            self._SwigLikeMap({"k.cu": "line one\nline two\n"})
        )
        assert files == {"k.cu": "line one\nline two\n"}
        assert not isinstance(
            self._SwigLikeMap({}),
            __import__("collections.abc", fromlist=["Mapping"]).Mapping,
        ), "guard premise"

    class _CorrelatedAction:
        """A report with aggregated pcsamp metrics and no raw warp samples.

        This is the normal `--set full` shape: `timed_warp_samples()` empty,
        `smsp__pcsamp_warps_issue_stalled_*` present with correlation IDs.
        """

        _PREFIX = "smsp__pcsamp_warps_issue_stalled_"

        class _Metric:
            def __init__(self, values, correlation=None):
                self._values = values
                self._correlation = correlation

            def num_instances(self):
                return len(self._values)

            def as_double(self, i):
                return float(self._values[i])

            def as_uint64(self, i):
                return int(self._values[i])

            def has_correlation_ids(self):
                return self._correlation is not None

            def correlation_ids(self):
                if self._correlation is None:
                    return None
                return TestRealReportRegressions._CorrelatedAction._Metric(
                    self._correlation
                )

        def metric_names(self):
            return [
                self._PREFIX + "long_scoreboard",
                self._PREFIX + "long_scoreboard_not_issued",
                self._PREFIX + "barrier",
            ]

        def metric_by_name(self, name):
            M = TestRealReportRegressions._CorrelatedAction._Metric
            if name.endswith("_not_issued"):
                return M([999.0], correlation=[0x10])  # must be ignored
            if name.endswith("long_scoreboard"):
                return M([100.0, 20.0], correlation=[0x10, 0x20])
            if name.endswith("barrier"):
                return M([50.0], correlation=[0x20])
            return None

        def source_files(self):
            return TestRealReportRegressions._SwigLikeMap(
                {"k.cu": "load line\ncompute line\n"}
            )

        def source_info(self, address):
            table = {0x10: ("k.cu", 1), 0x20: ("k.cu", 2)}
            if address not in table:
                return None
            name, line = table[address]

            class _Info:
                def file_name(self):
                    return name

                def line(self):
                    return line

            return _Info()

        def sass_by_pc(self, address):
            return "LDG.E"

        def ptx_by_pc(self, address):
            return ""

        def timed_warp_samples(self):
            return []

    def test_attribution_works_without_raw_warp_samples(self):
        out = source_correlation.attribute_stalls_to_source(self._CorrelatedAction())
        assert out["available"] is True
        assert out["source"] == "correlated pcsamp metrics"
        top = out["source_lines"][0]
        assert top["line"] == 1 and top["dominant_stall_reason"] == "LONG_SCOREBOARD"
        assert top["source_text"] == "load line"

    def test_not_issued_variants_are_not_double_counted(self):
        out = source_correlation.attribute_stalls_to_source(self._CorrelatedAction())
        # 100 + 20 long_scoreboard + 50 barrier == 170; the 999 _not_issued
        # counts the same stalls again and must be excluded.
        assert out["total_samples"] == 170

    def test_hit_rate_with_no_traffic_is_suppressed(self):
        """0% hit rate over 0 requests is not a result."""
        out = signal_scan.scan_all_signals(
            {
                "l1tex__t_sector_pipe_lsu_mem_global_op_red_hit_rate.pct": 0.0,
                "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum": 0.0,
            }
        )
        assert not [f for f in out["findings"] if f.category == "unit_hit_rate"]
        assert out["hit_rates_skipped_no_traffic"] == 1

    def test_hit_rate_with_traffic_is_reported_with_its_volume(self):
        out = signal_scan.scan_all_signals(
            {
                "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct": 0.0,
                "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": 49152.0,
            }
        )
        finding = next(f for f in out["findings"] if f.category == "unit_hit_rate")
        assert finding.evidence["requests_behind_it"] == 49152.0
        assert "more than 100%" not in finding.summary, "100 - 0 rendered as prose"
        assert "100% of the 49,152 requests" in finding.summary

    def test_scan_hit_rate_findings_are_not_linked_to_source(self):
        """The scan knows a path missed, not which lines use that path."""
        out = source_correlation.link_findings_to_source(
            [{"category": "unit_hit_rate", "title": "t"}],
            None,
            attribution={
                "available": True,
                "stall_reasons": {"LONG_SCOREBOARD": 100},
                "source_lines": [
                    {
                        "file_name": "k.cu",
                        "line": 1,
                        "samples": 100,
                        "stall_reasons": {"LONG_SCOREBOARD": 100},
                    }
                ],
            },
        )
        assert out["linked"] == []

    def test_identical_linkages_are_folded(self):
        attribution = {
            "available": True,
            "stall_reasons": {"LONG_SCOREBOARD": 100},
            "source_lines": [
                {
                    "file_name": "k.cu",
                    "line": 1,
                    "samples": 100,
                    "stall_reasons": {"LONG_SCOREBOARD": 100},
                }
            ],
        }
        # Both resolve to exactly ("LONG_SCOREBOARD",), so same reasons AND
        # same lines. Findings whose reasons differ are NOT folded, because the
        # same line stalling for a second reason is a second fact.
        out = source_correlation.link_findings_to_source(
            [
                {"category": "poor_cache_locality", "title": "first"},
                {"category": "l2_load_imbalance", "title": "second"},
            ],
            None,
            attribution=attribution,
        )
        assert len(out["linked"]) == 1
        assert out["duplicate_links"] and out["duplicate_note"]

    def test_different_reasons_on_the_same_lines_are_kept(self):
        attribution = {
            "available": True,
            "stall_reasons": {"LONG_SCOREBOARD": 100, "LG_THROTTLE": 10},
            "source_lines": [
                {
                    "file_name": "k.cu",
                    "line": 1,
                    "samples": 110,
                    "stall_reasons": {"LONG_SCOREBOARD": 100, "LG_THROTTLE": 10},
                }
            ],
        }
        out = source_correlation.link_findings_to_source(
            [
                {"category": "poor_cache_locality", "title": "locality"},
                {"category": "register_spilling", "title": "spilling"},
            ],
            None,
            attribution=attribution,
        )
        assert len(out["linked"]) == 2, (
            "same lines but different mechanisms is two findings, not one"
        )


class TestInstructionAndPmSampling:
    """Instruction-level attribution, and the PM-sampling timeline."""

    class _Action:
        _P = "smsp__pcsamp_warps_issue_stalled_"
        _PM = "pmsampling:"

        class _M:
            def __init__(self, values, correlation=None):
                self.values = values
                self._c = correlation

            def num_instances(self):
                return len(self.values)

            def as_double(self, i):
                return float(self.values[i])

            def as_uint64(self, i):
                return int(self.values[i])

            def has_correlation_ids(self):
                return self._c is not None

            def correlation_ids(self):
                if self._c is None:
                    return None
                return TestInstructionAndPmSampling._Action._M(self._c)

        def metric_names(self):
            return [
                self._P + "long_scoreboard",
                self._PM
                + "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                self._PM + "sm__cycles_active.avg",
            ]

        def metric_by_name(self, n):
            M = TestInstructionAndPmSampling._Action._M
            if n.endswith("long_scoreboard"):
                return M([600.0, 100.0], correlation=[0x10, 0x20])
            if n.endswith("pct_of_peak_sustained_elapsed"):
                # bursty: high peak, low average
                return M(
                    [0.0, 0.0, 94.0, 90.0, 0.0, 0.0],
                    correlation=[1000, 2500, 4000, 5500, 7000, 8500],
                )
            if n.endswith("sm__cycles_active.avg"):
                return M(
                    [0.0, 2964.0, 100.0, 0.0, 0.0, 0.0],
                    correlation=[1000, 2500, 4000, 5500, 7000, 8500],
                )
            return None

        def source_files(self):
            return {"k.cu": "convert line\nload line\n"}

        def source_info(self, a):
            t = {0x10: ("k.cu", 1), 0x20: ("k.cu", 2)}
            if a not in t:
                return None
            f, l = t[a]

            class I:
                def file_name(self):
                    return f

                def line(self):
                    return l

            return I()

        def sass_by_pc(self, a):
            return {0x10: "PRMT R19, R8, 0x7732, RZ", 0x20: "LDG.E.128 R4"}.get(a, "")

        def ptx_by_pc(self, a):
            return ""

        def timed_warp_samples(self):
            return []

    def test_top_instruction_carries_its_sass(self):
        out = source_correlation.top_stalling_instructions(self._Action())
        assert out["available"] is True
        top = out["instructions"][0]
        assert top["sass"] == "PRMT R19, R8, 0x7732, RZ"
        assert top["line"] == 1 and top["dominant_stall_reason"] == "LONG_SCOREBOARD"
        assert top["address_hex"] == "0x10"

    def test_instruction_view_is_finer_than_the_line_view(self):
        out = source_correlation.top_stalling_instructions(self._Action())
        assert out["distinct_instructions"] == 2
        assert "finer than the line view" in out["note"]

    def test_pm_sampling_reports_peak_against_the_active_window(self):
        """The denominator must be the window the kernel ran in.

        Averaging over the sampler's whole session divides by a lot of time the
        kernel was not running. On a real report that turned an 81us kernel into
        a 216us "active window" and reported the tensor pipe at 8.7% when the
        figure over the kernel's own window is 32.3% -- which matches the
        whole-kernel counter, as it should.
        """
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert out["available"] is True
        tensor = next(e for e in out["series"] if "tensor" in e["metric"])
        assert tensor["peak"] == pytest.approx(94.0)
        assert tensor["mean_in_active_window"] > tensor["mean_all_buckets"], (
            "the window average must exceed the whole-session average"
        )
        # The window is derived per pass; with no kernel duration in this
        # fixture it falls back to that pass's own envelope.
        assert (
            "envelope" in out["window_source"]
            or "kernel duration" in out["window_source"]
        )

    def test_duty_cycle_is_counted_inside_the_window(self):
        """Counting non-zero buckets series-wide over a window denominator
        produced shares above 100% for DRAM, which is active either side of the
        launch."""
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert all(0.0 <= e["duty_cycle"] <= 1.0 for e in out["series"])

    def test_denominators_are_named_in_the_output(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert "mean_all_buckets" in out["denominator_note"]
        assert "not for comparison" in out["denominator_note"]

    def test_raw_counts_are_not_rendered_as_percentages(self):
        """`sm__cycles_active.avg` is a count; "2964%" is nonsense."""
        out = source_correlation.analyze_pm_sampling(self._Action())
        counts = [e for e in out["series"] if not e["is_percentage"]]
        assert counts and counts[0]["unit"] == "count"
        # and a count must never be called bursty, having no ceiling
        assert all(e["is_percentage"] for e in out["bursty"])

    def test_pm_sampling_absent_is_explained(self):
        class _Bare:
            def metric_names(self):
                return ["sm__throughput.avg"]

            def metric_by_name(self, n):
                return None

        out = source_correlation.analyze_pm_sampling(_Bare())
        assert out["available"] is False
        assert "pmsampling" in out["reason"]


class TestSamplingAppearsInTheReport:
    """PC and PM sampling must reach the rendered report, not just the payload."""

    def _module(self):
        A = TestInstructionAndPmSampling._Action

        class Action(A):
            def name(self):
                return "k"

            def rule_results_as_dicts(self):
                return []

            def metric_names(self):
                return A.metric_names(self) + [
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "smsp__pcsamp_sample_count",
                    "smsp__pcsamp_interval_cycles",
                ]

            def metric_by_name(self, n):
                simple = {
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed": 33.7,
                    "smsp__pcsamp_sample_count": 9244.0,
                    "smsp__pcsamp_interval_cycles": 2048.0,
                }
                if n in simple:

                    class _M:
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

                    return _M(simple[n])
                return A.metric_by_name(self, n)

        class Rng:
            num_actions = 1

            def action_by_idx(self, i):
                return Action()

        class Ctx:
            num_ranges = 1

            def range_by_idx(self, i):
                return Rng()

        return types.SimpleNamespace(load_report=lambda p: Ctx())

    def _markdown(self):
        return ncu_report_tools.diagnose_result_to_markdown(
            ncu_report_tools.diagnose_ncu_report(
                "/dev/null", ncu_report_module=self._module()
            )
        )

    def test_pc_sampling_section_is_rendered(self):
        text = self._markdown()
        assert "### PC sampling" in text
        assert "9,244 samples" in text and "2,048-cycle" in text

    def test_stalling_instructions_are_rendered_with_sass(self):
        text = self._markdown()
        assert "#### Stalling instructions" in text
        assert "PRMT R19, R8, 0x7732, RZ" in text

    def test_pm_sampling_section_is_rendered(self):
        text = self._markdown()
        assert "### PM sampling" in text
        assert "time buckets" in text

    def test_pm_span_does_not_invent_kernel_replay_without_metadata(self):
        """A multipass timeline is not proof of a particular replay mode."""
        text = self._markdown()
        assert "did not record its replay mode" in text
        assert "Under kernel replay" not in text

    def test_raw_counts_carry_no_percent_sign_in_the_table(self):
        text = self._markdown()
        assert "2964.0%" not in text and "2965%" not in text


class TestPmSamplingPassGroups:
    """PM sampling is multiplexed across replay passes.

    Each pass re-runs the kernel with its own capture window, so series from
    different passes have different timestamp bases and bucket counts and share
    no timestamps. On a real report there were seven groups whose start times
    differed by milliseconds. Bucket index N is a different moment in each, so
    a window derived from one pass must not be applied to another -- which is
    exactly what the first version of this function did.
    """

    class _Action:
        _PM = "pmsampling:"

        class _M:
            def __init__(self, values, t0, step):
                self.values, self.t0, self.step = values, t0, step

            def num_instances(self):
                return len(self.values)

            def as_double(self, i):
                return float(self.values[i])

            def as_uint64(self, i):
                return self.t0 + i * self.step

            def has_correlation_ids(self):
                return True

            def correlation_ids(self):
                return self

        def metric_names(self):
            return [
                self._PM + "sm__cycles_active.avg",
                self._PM
                + "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                self._PM + "smsp__warps_issue_stalled_barrier.avg",
            ]

        def metric_by_name(self, n):
            M = TestPmSamplingPassGroups._Action._M
            # pass A: 6 buckets from t=1000, step 1500
            if n.endswith("sm__cycles_active.avg"):
                return M([0.0, 100.0, 100.0, 100.0, 0.0, 0.0], 1_000, 1500)
            if n.endswith("pct_of_peak_sustained_elapsed"):
                return M([0.0, 90.0, 60.0, 30.0, 0.0, 0.0], 1_000, 1500)
            # pass B: a DIFFERENT execution -- different base, different length
            if n.endswith("barrier.avg"):
                return M([50.0, 400.0, 200.0, 0.0], 9_000_000, 1472)
            return None

    def test_passes_are_detected(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert out["pass_group_count"] == 2
        assert out["cross_pass_warning"]

    def test_each_series_records_its_pass(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        groups = {e["metric"]: e["pass_group"] for e in out["series"]}
        assert (
            groups["sm__cycles_active.avg"]
            == groups["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]
        )
        assert (
            groups["smsp__warps_issue_stalled_barrier.avg"]
            != groups["sm__cycles_active.avg"]
        )

    def test_window_is_computed_per_pass_not_borrowed(self):
        """Pass A is active in buckets 1..3; pass B in 0..2. Applying A's
        window to B would average in a bucket B never had."""
        out = source_correlation.analyze_pm_sampling(self._Action())
        barrier = next(e for e in out["series"] if "barrier" in e["metric"])
        # B's own window is buckets 0..2, mean (50+400+200)/3 = 216.7
        assert barrier["mean_in_active_window"] == pytest.approx(216.7, abs=1.0)

    def test_single_pass_report_has_no_warning(self):
        class _One(TestPmSamplingPassGroups._Action):
            def metric_names(self):
                return [self._PM + "sm__cycles_active.avg"]

        out = source_correlation.analyze_pm_sampling(_One())
        assert out["pass_group_count"] == 1
        assert out["cross_pass_warning"] == ""

    def test_all_series_are_returned_not_truncated(self):
        """`top_k` truncated the payload, so 26 series returned 8 and the new
        stall-reason series were invisible to any consumer."""
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert out["series_count"] == len(out["series"]) == 3

    def test_explicit_top_k_still_trims(self):
        out = source_correlation.analyze_pm_sampling(self._Action(), top_k=1)
        assert len(out["series"]) == 1 and out["series_count"] == 3


class TestPmSamplingDeclaredGroups:
    """Nsight Compute declares the pass grouping; we do not have to infer it.

    `PmSampling_WarpStates.section` states it outright -- "Metrics in different
    groups come from different passes" -- and carries a `Groups: "sampling_wsN"`
    line per metric. On a real H100 capture the five declared warp-state groups
    matched the five timestamp clusters exactly.

    Declared groups are a constraint, not the outcome: the scheduler may pack
    several compatible groups into one pass, and on that same report
    `sampling_2`+`sampling_3` and `sampling_0`+`sampling_1` were each merged.
    So timestamps stay authoritative for pass identity and the declared name is
    carried as the explanation.
    """

    def test_groups_are_read_from_the_install(self):
        groups = section_index.pm_sampling_groups()
        if not groups:
            pytest.skip("no local Nsight Compute install")
        warp_states = {
            name.replace("pmsampling:smsp__warps_issue_stalled_", "").replace(
                ".avg", ""
            ): group
            for name, group in groups.items()
            if "warps_issue_stalled" in name
        }
        assert warp_states.get("long_scoreboard") == "sampling_ws2"
        assert warp_states.get("barrier") == "sampling_ws0"
        # barrier and long_scoreboard in different groups is why "what was the
        # barrier doing when long_scoreboard peaked" is unanswerable.
        assert warp_states["barrier"] != warp_states["long_scoreboard"]

    def test_lookup_falls_back_to_the_base_name(self):
        """The section declares one submetric spelling and the report may carry
        another, which left real metrics unattributed on an exact-name match."""
        groups = section_index.pm_sampling_groups()
        if not groups:
            pytest.skip("no local Nsight Compute install")
        bases = {name.split(".")[0] for name in groups}
        assert "pmsampling:l1tex__data_pipe_lsu_wavefronts" in bases

    def test_absent_install_returns_empty_not_a_guess(self):
        assert section_index.pm_sampling_groups("/nonexistent/path") == {}

    def test_window_is_sized_from_the_kernel_duration(self):
        """Every pass runs the same kernel, so its duration is the same in all
        of them. The envelope of whatever counters a pass happens to carry is
        not: a pass holding DRAM counters sees activity either side of the
        launch, and a pass holding two sparse stall reasons sees almost none."""

        class _Sparse:
            _PM = "pmsampling:"

            class _M:
                def __init__(self, values, t0, step):
                    self.values, self.t0, self.step = values, t0, step

                def num_instances(self):
                    return len(self.values)

                def as_double(self, i=None):
                    return 8000.0 if i is None else float(self.values[i])

                def as_uint64(self, i):
                    return self.t0 + i * self.step

                def has_correlation_ids(self):
                    return True

                def correlation_ids(self):
                    return self

            def metric_names(self):
                return [self._PM + "smsp__warps_issue_stalled_imc_miss.avg"]

            def metric_by_name(self, n):
                M = _Sparse._M
                if n == "gpu__time_duration.sum":
                    return M([], 0, 1)  # as_double() -> 8000 ns
                # non-zero in 1 bucket of 10; the envelope would give a
                # 1-bucket window and inflate the mean tenfold
                return M([0, 0, 0, 100.0, 0, 0, 0, 0, 0, 0], 1000, 1000)

        out = source_correlation.analyze_pm_sampling(_Sparse())
        entry = out["series"][0]
        # 8000ns / 1000ns per bucket = 8 buckets, not the 1 the envelope gives
        assert entry["active_buckets"] == 1
        assert entry["mean_in_active_window"] == pytest.approx(100.0 / 8, abs=0.1)
        assert "kernel duration" in out["window_source"]

    def test_falls_back_to_the_envelope_without_a_duration(self):
        class _NoDuration(TestPmSamplingPassGroups._Action):
            def metric_by_name(self, n):
                if n == "gpu__time_duration.sum":
                    return None
                return TestPmSamplingPassGroups._Action.metric_by_name(self, n)

        out = source_correlation.analyze_pm_sampling(_NoDuration())
        assert out["available"] is True
        assert "envelope" in out["window_source"]


class TestLinkageDedupUsesContributingReasons:
    """A declared stall reason that carried no samples is not corroboration.

    `register_spilling` declares (LONG_SCOREBOARD, LG_THROTTLE) and
    `stall_long_scoreboard` declares (LONG_SCOREBOARD). On a real report where
    LG_THROTTLE was zero, those resolved to an identical set of lines but the
    dedup keyed on the *declared* tuple, so both were printed -- one observation
    rendered twice under different headings, which reads as two pieces of
    evidence.
    """

    _ATTRIBUTION = {
        "available": True,
        "stall_reasons": {"LONG_SCOREBOARD": 1000, "LG_THROTTLE": 0},
        "source_lines": [
            {
                "file_name": "k.cu",
                "line": 1,
                "samples": 600,
                "stall_reasons": {"LONG_SCOREBOARD": 600},
            },
            {
                "file_name": "k.cu",
                "line": 2,
                "samples": 400,
                "stall_reasons": {"LONG_SCOREBOARD": 400},
            },
        ],
    }

    def _link(self, categories):
        return source_correlation.link_findings_to_source(
            [{"category": c, "title": c} for c in categories],
            None,
            attribution=self._ATTRIBUTION,
        )

    def test_zero_sample_reason_does_not_make_a_finding_distinct(self):
        out = self._link(["stall_long_scoreboard", "register_spilling"])
        assert len(out["linked"]) == 1, "same lines, same contributing reason"
        assert out["duplicate_links"][0]["identical_via"] == ["LONG_SCOREBOARD"]

    def test_contributing_reasons_are_reported_separately(self):
        out = self._link(["register_spilling"])
        entry = out["linked"][0]
        assert entry["contributing_stall_reasons"] == ["LONG_SCOREBOARD"]
        assert entry["declared_but_absent"] == ["LG_THROTTLE"]
        # the declared list is still available, unchanged
        assert "LG_THROTTLE" in entry["matched_on_stall_reasons"]

    def test_genuinely_different_reasons_are_still_kept(self):
        attribution = {
            "available": True,
            "stall_reasons": {"LONG_SCOREBOARD": 500, "BARRIER": 500},
            "source_lines": [
                {
                    "file_name": "k.cu",
                    "line": 1,
                    "samples": 1000,
                    "stall_reasons": {"LONG_SCOREBOARD": 500, "BARRIER": 500},
                },
            ],
        }
        out = source_correlation.link_findings_to_source(
            [
                {"category": "stall_long_scoreboard", "title": "ls"},
                {"category": "stall_barrier", "title": "b"},
            ],
            None,
            attribution=attribution,
        )
        assert len(out["linked"]) == 2, (
            "same line, different mechanisms, both carrying samples"
        )

    def test_markdown_shows_the_absent_reason(self):
        text = ncu_report_tools.diagnose_result_to_markdown(
            {
                "kernels": [
                    {
                        "kernel_name": "k",
                        "signal_to_source": self._link(["register_spilling"]),
                    }
                ]
            }
        )
        assert "carried no samples in this kernel" in text


class TestPmSamplingReplayClockDrift:
    """Per-pass effective clocks, and the drift caveat built from them.

    Each PM pass group that carries a plain cycles counter yields an effective
    SM clock (cycles-per-bucket over the bucket width); the report-level
    `sm__cycles_elapsed.avg.per_second` -- itself collected in one replay pass
    -- joins them. A supported gap above the threshold is thermal sag inside
    ONE collection, which nothing in the report otherwise flags.
    """

    class _Scalar:
        def __init__(self, value):
            self._value = value

        def as_double(self):
            return self._value

    class _Action:
        _PM = "pmsampling:"

        class _M:
            def __init__(self, values, t0, step):
                self.values, self.t0, self.step = values, t0, step

            def num_instances(self):
                return len(self.values)

            def as_double(self, i):
                return float(self.values[i])

            def as_uint64(self, i):
                return self.t0 + i * self.step

            def has_correlation_ids(self):
                return True

            def correlation_ids(self):
                return self

        # pass A: sm__cycles_active at ~1.85 GHz effective (lower bound);
        # pass B: gpc__cycles_elapsed at ~1.75 GHz (measured). 5.7% apart.
        _FAST = [0.0] + [1500 * 1.85] * 8 + [0.0]
        _SLOW = [1500 * 1.75] * 10

        def metric_names(self):
            return [
                self._PM + "sm__cycles_active.avg",
                self._PM + "gpc__cycles_elapsed.avg",
            ]

        def metric_by_name(self, n):
            M = TestPmSamplingReplayClockDrift._Action._M
            if n == self._PM + "sm__cycles_active.avg":
                return M(list(self._FAST), 1_000, 1500)
            if n == self._PM + "gpc__cycles_elapsed.avg":
                return M(list(self._SLOW), 9_000_000, 1500)
            if n == "sm__cycles_elapsed.avg.per_second":
                return TestPmSamplingReplayClockDrift._Scalar(1.75e9)
            return None

    def test_per_pass_clocks_are_estimated_with_their_kind(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        clocks = out["pass_effective_clocks"]
        kinds = {label: e["kind"] for label, e in clocks.items()}
        active = next(label for label in clocks if "cycles_active" in label)
        elapsed = next(label for label in clocks if "gpc__cycles_elapsed" in label)
        # active cycles bound the clock from below; elapsed cycles ARE it
        assert kinds[active] == "lower_bound"
        assert kinds[elapsed] == "measured"
        assert clocks[active]["clock_hz"] == pytest.approx(1.85e9, rel=1e-6)
        assert clocks[elapsed]["clock_hz"] == pytest.approx(1.75e9, rel=1e-6)
        # the report-level clock joins as one more per-pass estimate
        assert "collection (sm__cycles_elapsed.avg.per_second)" in clocks

    def test_drift_across_passes_is_flagged_and_reaches_the_warning(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        drift = out["replay_clock_drift"]
        assert drift["drifted"] is True
        assert drift["supported_drift"] == pytest.approx(1.85 / 1.75 - 1, rel=1e-6)
        # the caveat rides the existing cross-pass channel, which the
        # markdown renderer already prints -- no new reporting path
        assert "SM clock drifted at least" in out["cross_pass_warning"]
        assert "mix different clock states" in out["cross_pass_warning"]

    def test_agreeing_passes_do_not_invent_a_caveat(self):
        class _Steady(TestPmSamplingReplayClockDrift._Action):
            _FAST = [0.0] + [1500 * 1.75] * 8 + [0.0]

        out = source_correlation.analyze_pm_sampling(_Steady())
        drift = out["replay_clock_drift"]
        assert drift["checked"] is True and drift["drifted"] is False
        assert "SM clock drifted" not in out["cross_pass_warning"]

    def test_too_few_busy_buckets_yield_no_estimate(self):
        """The two-pass fixture above (TestPmSamplingPassGroups) has under four
        non-zero cycle buckets per pass: an estimate from that would be noise,
        so the drift check must report unchecked rather than clean."""
        out = source_correlation.analyze_pm_sampling(TestPmSamplingPassGroups._Action())
        assert out["replay_clock_drift"]["checked"] is False
        assert out["pass_effective_clocks"] == {}

    def test_percentage_and_rate_spellings_are_not_mistaken_for_cycles(self):
        class _Pct(TestPmSamplingReplayClockDrift._Action):
            def metric_names(self):
                return [
                    self._PM
                    + "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                ]

            def metric_by_name(self, n):
                M = TestPmSamplingReplayClockDrift._Action._M
                if n.endswith("pct_of_peak_sustained_elapsed"):
                    return M([50.0] * 10, 1_000, 1500)
                return None

        out = source_correlation.analyze_pm_sampling(_Pct())
        assert out["pass_effective_clocks"] == {}

    def test_application_range_replay_is_not_described_as_kernel_replay(self):
        out = source_correlation.analyze_pm_sampling(
            self._Action(), replay_mode="app-range"
        )
        assert "application-range-replay" in out["span_note"]
        assert "Kernel Replay measurement" in out["span_note"]
