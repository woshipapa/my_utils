# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.ncu.report_diff -- the A/B report diff.

Two synthetic reports are built with the same fake ncu_report module shape the
ncu_report_tools tests use, keyed by file name so one module can serve both
sides of a diff. The scenarios mirror the real pairs the tool was validated
on: a comparable-clock pair where stalls and bank conflicts move, and a
clock-mismatch pair where the guard must refuse the raw-time delta.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from _synthetic_loader import metric_catalog, report_diff

STALL = metric_catalog.STALL_METRIC_TEMPLATE


# ---------------------------------------------------------------------------
# Fake ncu_report module, keyed by report file name
# ---------------------------------------------------------------------------


class _FakeMetric:
    def __init__(self, value: object, unit: str = "") -> None:
        self._value = value
        self._unit = unit

    def value(self):
        return self._value

    def unit(self):
        return self._unit


class _FakeAction:
    def __init__(self, name: str, metrics: dict) -> None:
        self._name = name
        self._metrics = {k: _FakeMetric(v) for k, v in metrics.items()}

    def name(self):
        return self._name

    def metric_names(self):
        return list(self._metrics)

    def metric_by_name(self, key: str):
        return self._metrics[key]

    def rule_results_as_dicts(self):
        return []


class _FakeRange:
    def __init__(self, actions: list) -> None:
        self._actions = list(actions)

    def num_actions(self):
        return len(self._actions)

    def action_by_idx(self, i: int):
        return self._actions[i]


class _FakeContext:
    def __init__(self, ranges: list) -> None:
        self._ranges = list(ranges)

    def num_ranges(self):
        return len(self._ranges)

    def range_by_idx(self, i: int):
        return self._ranges[i]


class _FakeNcuModule:
    """load_report keyed by file name, so one module serves both reports."""

    def __init__(self, contexts: dict) -> None:
        self._contexts = dict(contexts)

    def load_report(self, path: str):
        return self._contexts[Path(path).name]


def _kernel_metrics(
    *,
    dur_ns: float,
    sm_hz: float,
    cycles: float,
    stalls: dict,
    declared_total: float | None = None,
    bank_conflicts_st: float = 0.0,
    l2_hit: float = 70.0,
    l2_sectors: float = 1_000_000.0,
    block: float = 384.0,
    extra: dict | None = None,
) -> dict:
    total = declared_total if declared_total is not None else sum(stalls.values())
    metrics = {
        "gpu__time_duration.sum": dur_ns,
        "sm__cycles_elapsed.avg.per_second": sm_hz,
        "gpc__cycles_elapsed.avg.per_second": sm_hz * 1.001,
        "gpc__cycles_elapsed.max": cycles,
        "launch__grid_size": 132.0,
        "launch__block_size": block,
        # Below the 0.8 gate so dominant stalls become findings.
        "smsp__issue_active.avg.per_cycle_active": 0.2,
        "smsp__average_warp_latency_per_inst_issued.ratio": total,
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": bank_conflicts_st,
        "lts__t_sector_hit_rate.pct": l2_hit,
        "lts__t_sectors.sum": l2_sectors,
    }
    for reason, value in stalls.items():
        metrics[STALL.format(reason=reason)] = value
    metrics.update(extra or {})
    return metrics


def _write_pair(tmp_path: Path, metrics_a: dict, metrics_b: dict, name="kern"):
    rep_a = tmp_path / "a.ncu-rep"
    rep_b = tmp_path / "b.ncu-rep"
    rep_a.write_bytes(b"")
    rep_b.write_bytes(b"")
    # A normal collected report has command provenance.  Individual tests that
    # exercise missing/changed provenance remove or replace these sidecars.
    sidecar = json.dumps(
        {
            "schema_version": 1,
            "collection": {
                "ncu_defaults_known": True,
                "cache_control": "all",
            },
        }
    )
    Path(str(rep_a) + ".collection.json").write_text(sidecar, encoding="utf-8")
    Path(str(rep_b) + ".collection.json").write_text(sidecar, encoding="utf-8")
    module = _FakeNcuModule(
        {
            "a.ncu-rep": _FakeContext([_FakeRange([_FakeAction(name, metrics_a)])]),
            "b.ncu-rep": _FakeContext([_FakeRange([_FakeAction(name, metrics_b)])]),
        }
    )
    return str(rep_a), str(rep_b), module


def _rows_by_metric(rows: list) -> dict:
    return {r["metric"]: r for r in rows}


# ---------------------------------------------------------------------------
# Comparable clocks: deltas, severity coding, findings diff
# ---------------------------------------------------------------------------


class TestComparableClockDiff:
    def _payload(self, tmp_path: Path):
        metrics_a = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"long_scoreboard": 10.0, "barrier": 4.0, "wait": 2.0},
            bank_conflicts_st=15_000.0,
            l2_hit=70.0,
        )
        metrics_b = _kernel_metrics(
            dur_ns=85_000.0,
            sm_hz=1.81e9,  # 0.6% apart: inside the guard's 1% tolerance
            cycles=154_000.0,
            stalls={"long_scoreboard": 3.0, "barrier": 8.0, "wait": 2.0},
            bank_conflicts_st=0.0,
            l2_hit=70.3,  # below the 0.5pp floor: must read as unchanged
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics_a, metrics_b)
        return report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)

    def test_clock_guard_passes_and_duration_ratio_stands(self, tmp_path):
        payload = self._payload(tmp_path)
        assert payload["clock_guard"]["all_comparable"] is True
        assert payload["matched_kernel_count"] == 1
        kernel = payload["kernels"][0]
        assert kernel["clock_comparison"]["comparable"] is True
        duration = kernel["duration"]
        assert duration["comparable_as_raw_time"] is True
        assert abs(duration["raw_ratio"] - 0.85) < 1e-9
        # Cycles are clock-independent and must be carried as the cross-check.
        assert abs(duration["cycles_ratio"] - 154_000.0 / 180_000.0) < 1e-9

    def test_stall_deltas_severity_coded_in_cycles_per_issue_slot(self, tmp_path):
        kernel = self._payload(tmp_path)["kernels"][0]
        rows = _rows_by_metric(kernel["axes"]["stall_composition"])
        lsb = rows[STALL.format(reason="long_scoreboard")]
        barrier = rows[STALL.format(reason="barrier")]
        wait = rows[STALL.format(reason="wait")]
        assert lsb["status"] == "improved" and abs(lsb["delta"] + 7.0) < 1e-9
        assert barrier["status"] == "regressed" and abs(barrier["delta"] - 4.0) < 1e-9
        assert wait["status"] == "unchanged"
        assert lsb["unit"] == "cycles/issue-slot"

    def test_bank_conflict_elimination_and_noise_floor(self, tmp_path):
        kernel = self._payload(tmp_path)["kernels"][0]
        shared = _rows_by_metric(kernel["axes"]["shared_memory"])
        st = shared["shared_bank_conflicts_st"]
        assert st["status"] == "improved"
        assert st["a"] == 15_000.0 and st["b"] == 0.0
        # 70.0 -> 70.3 is below both the 2% relative and 0.5pp floors.
        hierarchy = _rows_by_metric(kernel["axes"]["memory_hierarchy"])
        assert hierarchy["l2_hit_rate"]["status"] == "unchanged"

    def test_hit_rate_shift_is_context_not_a_standalone_regression(self, tmp_path):
        metrics_a = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"wait": 2.0},
            l2_hit=80.0,
        )
        metrics_b = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"wait": 2.0},
            l2_hit=60.0,
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics_a, metrics_b)
        rows = _rows_by_metric(
            report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)["kernels"][0]["axes"]["memory_hierarchy"]
        )
        assert rows["l2_hit_rate"]["status"] == "changed"

    def test_findings_diff_reports_appear_and_disappear(self, tmp_path):
        kernel = self._payload(tmp_path)["kernels"][0]
        diff = kernel["findings_diff"]
        disappeared = {f["category"] for f in diff["disappeared"]}
        appeared = {f["category"] for f in diff["appeared"]}
        assert "stall_long_scoreboard" in disappeared
        assert "stall_barrier" in appeared

    def test_stall_accounting_closed_so_deltas_are_trusted(self, tmp_path):
        kernel = self._payload(tmp_path)["kernels"][0]
        assert kernel["stall_delta_reliable"] is True

    def test_markdown_renders_key_sections(self, tmp_path):
        payload = self._payload(tmp_path)
        text = report_diff.diff_result_to_markdown(payload)
        assert "## Clock guard" in text
        assert "What changed the verdict" in text
        assert "Stall composition (cycles per issue-slot)" in text
        assert "disappeared" in text and "stall_long_scoreboard" in text
        assert "Honesty notes" in text
        assert "does not establish causality" in text

    def test_sidecars_block_cold_warm_duration_comparison(self, tmp_path):
        metrics = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"long_scoreboard": 5.0},
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics, metrics)
        Path(rep_a + ".collection.json").write_text(
            json.dumps({"schema_version": 1, "collection": {"cache_control": "all"}}),
            encoding="utf-8",
        )
        Path(rep_b + ".collection.json").write_text(
            json.dumps({"schema_version": 1, "collection": {"cache_control": "none"}}),
            encoding="utf-8",
        )
        payload = report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)
        assert payload["clock_guard"]["all_comparable"] is False
        assert payload["collection_manifests"]["a"]["status"] == "loaded"
        blockers = payload["kernels"][0]["clock_comparison"]["blockers"]
        assert any("cache state" in blocker for blocker in blockers)

    def test_missing_sidecars_fail_closed_on_cache_state(self, tmp_path):
        metrics = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"long_scoreboard": 5.0},
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics, metrics)
        Path(rep_a + ".collection.json").unlink()
        Path(rep_b + ".collection.json").unlink()
        payload = report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)
        assert payload["clock_guard"]["all_comparable"] is False
        blockers = payload["kernels"][0]["clock_comparison"]["blockers"]
        assert any("unrecorded cache state" in blocker for blocker in blockers)

    def test_legacy_sidecar_without_cache_provenance_also_fails_closed(self, tmp_path):
        metrics = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"long_scoreboard": 5.0},
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics, metrics)
        legacy = json.dumps(
            {"schema_version": 1, "collection": {"replay_mode": "kernel"}}
        )
        Path(rep_a + ".collection.json").write_text(legacy, encoding="utf-8")
        Path(rep_b + ".collection.json").write_text(legacy, encoding="utf-8")
        payload = report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)
        assert payload["kernels"][0]["result_status"] == "NOT_COMPARABLE"


# ---------------------------------------------------------------------------
# Clock mismatch: the guard must engage and lead the output
# ---------------------------------------------------------------------------


class TestClockMismatchGuard:
    def _payload(self, tmp_path: Path):
        metrics_a = _kernel_metrics(
            dur_ns=87_520.0,
            sm_hz=1.891e9,
            cycles=156_558.0,
            stalls={"long_scoreboard": 5.0, "barrier": 3.0},
        )
        metrics_b = _kernel_metrics(
            dur_ns=81_952.0,
            sm_hz=2.039e9,  # +7.8%: two different quantities, not a speedup
            cycles=146_452.0,
            stalls={"long_scoreboard": 5.0, "barrier": 3.0},
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics_a, metrics_b)
        return report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)

    def test_guard_blocks_raw_time_comparison(self, tmp_path):
        payload = self._payload(tmp_path)
        assert payload["clock_guard"]["all_comparable"] is False
        assert payload["clock_guard"]["blocked_kernels"] == ["kern"]
        kernel = payload["kernels"][0]
        assert kernel["clock_comparison"]["comparable"] is False
        assert any(
            "different SM clocks" in blocker
            for blocker in kernel["clock_comparison"]["blockers"]
        )

    def test_clock_normalised_ratio_is_the_presented_figure(self, tmp_path):
        duration = self._payload(tmp_path)["kernels"][0]["duration"]
        assert duration["comparable_as_raw_time"] is False
        raw = 81_952.0 / 87_520.0
        clock = 2.039e9 / 1.891e9
        assert abs(duration["raw_ratio"] - raw) < 1e-9
        # Durations scale inversely with the clock, so the correction multiplies.
        assert abs(duration["clock_normalised_ratio"] - raw * clock) < 1e-9
        assert "NOT comparable" in duration["headline"]

    def test_markdown_leads_with_the_warning(self, tmp_path):
        text = report_diff.diff_result_to_markdown(self._payload(tmp_path))
        guard_pos = text.index("## Clock guard")
        assert "**WARNING:" in text
        assert text.index("**WARNING:") < text.index("### Duration")
        assert guard_pos < text.index("### Duration")
        assert "NOT a speedup" in text


# ---------------------------------------------------------------------------
# Stall closure failure poisons the stall-delta section
# ---------------------------------------------------------------------------


class TestStallClosureFailure:
    def test_unclosed_accounting_flags_the_section_unreliable(self, tmp_path):
        metrics_a = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"long_scoreboard": 4.0, "barrier": 4.0},
            declared_total=16.0,  # reasons explain only 50% of warp latency
        )
        metrics_b = _kernel_metrics(
            dur_ns=90_000.0,
            sm_hz=1.80e9,
            cycles=162_000.0,
            stalls={"long_scoreboard": 4.0, "barrier": 4.0},
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics_a, metrics_b)
        payload = report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)
        kernel = payload["kernels"][0]
        assert kernel["stall_delta_reliable"] is False
        assert "failed closure in report A" in kernel["stall_reliability_note"]
        assert any("unreliable" in note for note in payload["notes"])
        text = report_diff.diff_result_to_markdown(payload)
        assert "Stall deltas unreliable" in text


# ---------------------------------------------------------------------------
# Kernel matching
# ---------------------------------------------------------------------------


class TestKernelMatching:
    def test_unmatched_kernels_are_reported_not_force_paired(self, tmp_path):
        shared = dict(
            dur_ns=10_000.0, sm_hz=1.8e9, cycles=18_000.0, stalls={"wait": 2.0}
        )
        rep_a = tmp_path / "a.ncu-rep"
        rep_b = tmp_path / "b.ncu-rep"
        rep_a.write_bytes(b"")
        rep_b.write_bytes(b"")
        module = _FakeNcuModule(
            {
                "a.ncu-rep": _FakeContext(
                    [
                        _FakeRange(
                            [
                                _FakeAction("k_main", _kernel_metrics(**shared)),
                                _FakeAction("k_only_a", _kernel_metrics(**shared)),
                            ]
                        )
                    ]
                ),
                "b.ncu-rep": _FakeContext(
                    [
                        _FakeRange(
                            [
                                _FakeAction("k_main", _kernel_metrics(**shared)),
                                _FakeAction("k_only_b", _kernel_metrics(**shared)),
                            ]
                        )
                    ]
                ),
            }
        )
        payload = report_diff.diff_ncu_reports(
            str(rep_a), str(rep_b), ncu_report_module=module
        )
        assert payload["matched_kernel_count"] == 1
        assert [u["kernel_name"] for u in payload["unmatched_a"]] == ["k_only_a"]
        assert [u["kernel_name"] for u in payload["unmatched_b"]] == ["k_only_b"]
        text = report_diff.diff_result_to_markdown(payload)
        assert "k_only_a" in text and "k_only_b" in text

    def test_duplicate_names_pair_by_launch_config(self):
        def bundle(name, block):
            return SimpleNamespace(
                kernel_name=name,
                metrics={"launch__grid_size": 132.0, "launch__block_size": block},
            )

        bundles_a = {(0, 0): bundle("k", 128.0), (0, 1): bundle("k", 256.0)}
        # Reversed encounter order on the B side: config must win over order.
        bundles_b = {(0, 0): bundle("k", 256.0), (0, 1): bundle("k", 128.0)}
        matches, unmatched_a, unmatched_b = report_diff._match_kernels(
            bundles_a, bundles_b
        )
        assert ((0, 0), (0, 1)) in matches
        assert ((0, 1), (0, 0)) in matches
        assert not unmatched_a and not unmatched_b

    def test_logical_alias_matches_renamed_kernel_and_marks_launch_change(self, tmp_path):
        shared = dict(
            dur_ns=10_000.0, sm_hz=1.8e9, cycles=18_000.0, stalls={"wait": 2.0}
        )
        rep_a = tmp_path / "a.ncu-rep"
        rep_b = tmp_path / "b.ncu-rep"
        rep_a.write_bytes(b"")
        rep_b.write_bytes(b"")
        sidecar_a = {
            "schema_version": 1,
            "collection": {
                "ncu_defaults_known": True,
                "cache_control": "all",
                "kernel_aliases": {"kernel_v1": "fused_gemm"},
            },
        }
        sidecar_b = {
            "schema_version": 1,
            "collection": {
                "ncu_defaults_known": True,
                "cache_control": "all",
                "kernel_aliases": {"kernel_v2": "fused_gemm"},
            },
        }
        Path(str(rep_a) + ".collection.json").write_text(json.dumps(sidecar_a))
        Path(str(rep_b) + ".collection.json").write_text(json.dumps(sidecar_b))
        module = _FakeNcuModule(
            {
                "a.ncu-rep": _FakeContext(
                    [_FakeRange([_FakeAction("kernel_v1", _kernel_metrics(**shared))])]
                ),
                "b.ncu-rep": _FakeContext(
                    [
                        _FakeRange(
                            [_FakeAction("kernel_v2", _kernel_metrics(**{**shared, "block": 256.0}))]
                        )
                    ]
                ),
            }
        )
        payload = report_diff.diff_ncu_reports(
            str(rep_a), str(rep_b), ncu_report_module=module
        )
        assert payload["matched_kernel_count"] == 1
        match = payload["kernels"][0]["match"]
        assert match["method"] == "logical_kernel_alias"
        assert match["launch_signature_changed"] is True


class TestCoverageAndProvenance:
    def test_missing_analysis_coverage_is_not_reported_as_disappearance(self):
        finding = {
            "category": "stall_long_scoreboard",
            "title": "scoreboard",
            "severity": "high",
            "source": "heuristic",
        }
        result = report_diff._diff_findings(
            [finding],
            [],
            diag_a={"coverage": {"ran": ["stalls"]}},
            diag_b={"coverage": {"ran": []}},
        )
        assert not result["disappeared"]
        assert result["not_evaluated_in_b"][0]["category"] == "stall_long_scoreboard"

    def test_workload_mismatch_blocks_duration_claim(self, tmp_path):
        metrics = _kernel_metrics(
            dur_ns=100_000.0,
            sm_hz=1.80e9,
            cycles=180_000.0,
            stalls={"wait": 2.0},
        )
        rep_a, rep_b, module = _write_pair(tmp_path, metrics, metrics)
        for path, workload in ((rep_a, "shape_a"), (rep_b, "shape_b")):
            Path(path + ".collection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "collection": {
                            "ncu_defaults_known": True,
                            "cache_control": "all",
                            "workload_id": workload,
                        },
                    }
                ),
                encoding="utf-8",
            )
        payload = report_diff.diff_ncu_reports(rep_a, rep_b, ncu_report_module=module)
        assert payload["kernels"][0]["result_status"] == "NOT_COMPARABLE"
        assert any(
            "workload id differs" in blocker
            for blocker in payload["kernels"][0]["clock_comparison"]["blockers"]
        )

    def test_work_normalisation_prevents_raw_count_only_verdict(self):
        metrics_a = {
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": 100.0,
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": 100.0,
        }
        metrics_b = {
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": 80.0,
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum": 40.0,
        }
        rows = _rows_by_metric(
            report_diff._ratio_rows(
                report_diff.MetricView(metrics_a), report_diff.MetricView(metrics_b)
            )
        )
        row = rows["shared_bank_conflicts_st_per_wavefront"]
        assert row["a"] == 1.0 and row["b"] == 2.0
        assert row["status"] == "regressed"

    def test_pm_and_pc_diffs_are_validity_gated(self):
        source_a = {
            "pm_sampling_validity": {"usable": True},
            "pm_sampling": {
                "available": True,
                "series": [
                    {
                        "metric": "sm__throughput",
                        "pass_group": "0",
                        "duty_cycle": 0.5,
                        "mean_in_active_window": 10.0,
                        "peak": 20.0,
                        "peak_to_mean": 2.0,
                    }
                ],
            },
            "sampling_validity": {"usable": True},
            "stall_attribution": {
                "source_lines": [
                    {"file_name": "kernel.cu", "line": 12, "share_of_samples": 0.6}
                ]
            },
        }
        source_b = {
            **source_a,
            "pm_sampling": {
                "available": True,
                "series": [
                    {
                        "metric": "sm__throughput",
                        "pass_group": "0",
                        "duty_cycle": 0.8,
                        "mean_in_active_window": 12.0,
                        "peak": 22.0,
                        "peak_to_mean": 1.8,
                    }
                ],
            },
            "stall_attribution": {
                "source_lines": [
                    {"file_name": "kernel.cu", "line": 12, "share_of_samples": 0.2}
                ]
            },
        }
        pm = report_diff._pm_sampling_diff(source_a, source_b)
        pc = report_diff._pc_hotspot_diff(source_a, source_b)
        assert pm["available"] and len(pm["features"]) == 4
        assert pc["available"] and abs(pc["hotspots"][0]["delta"] + 0.4) < 1e-12
        source_b["pm_sampling_validity"] = {"usable": False}
        assert report_diff._pm_sampling_diff(source_a, source_b)["available"] is False

    def test_repeat_reports_upgrade_only_stable_locked_speedup(self, tmp_path):
        paths = [tmp_path / name for name in ("a0.ncu-rep", "a1.ncu-rep", "b0.ncu-rep", "b1.ncu-rep")]
        contexts = {}
        durations = (100_000.0, 101_000.0, 80_000.0, 81_000.0)
        for path, duration in zip(paths, durations):
            path.write_bytes(b"")
            Path(str(path) + ".collection.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "collection": {
                            "ncu_defaults_known": True,
                            "cache_control": "all",
                            "clocks_locked": True,
                            "workload_id": "gemm_case",
                            "problem_shape": {"m": 128, "n": 128, "k": 64},
                            "dtype": "fp16",
                            "input_hash": "same-input",
                        },
                    }
                ),
                encoding="utf-8",
            )
            contexts[path.name] = _FakeContext(
                [
                    _FakeRange(
                        [
                            _FakeAction(
                                "kern",
                                _kernel_metrics(
                                    dur_ns=duration,
                                    sm_hz=1.8e9,
                                    cycles=duration * 1.8,
                                    stalls={"wait": 2.0},
                                ),
                            )
                        ]
                    )
                ]
            )
        payload = report_diff.diff_ncu_reports(
            str(paths[0]),
            str(paths[2]),
            repeat_reports_a=[str(paths[1])],
            repeat_reports_b=[str(paths[3])],
            ncu_report_module=_FakeNcuModule(contexts),
        )
        assert payload["repeat_statistics"]["available"] is True
        assert payload["repeat_statistics"]["kernels"][0]["outcome"] == "stable_improvement"
        assert payload["kernels"][0]["result_status"] == "VALID_SPEEDUP"


# ---------------------------------------------------------------------------
# The delta-row severity coding itself
# ---------------------------------------------------------------------------


class TestDeltaRow:
    def test_sub_noise_rate_move_is_unchanged(self):
        row = report_diff._delta_row(
            "hit rate", 90.0, 91.0, direction="higher_better", abs_floor=0.5
        )
        # 1.1% relative: below the 2% noise floor for rates.
        assert row["status"] == "unchanged"

    def test_direction_decides_improved_vs_regressed(self):
        up = report_diff._delta_row("ipc", 1.0, 1.5, direction="higher_better")
        down = report_diff._delta_row("spills", 1000.0, 200.0, direction="lower_better")
        worse = report_diff._delta_row(
            "spills", 200.0, 1000.0, direction="lower_better"
        )
        assert up["status"] == "improved"
        assert down["status"] == "improved"
        assert worse["status"] == "regressed"

    def test_neutral_metrics_report_changed_not_a_verdict(self):
        row = report_diff._delta_row("SOL", 30.0, 40.0, direction=None)
        assert row["status"] == "changed"

    def test_one_sided_metric_forms_no_delta(self):
        row = report_diff._delta_row("x", None, 5.0)
        assert row["status"] == "b_only"
        assert row["delta"] is None

    def test_baseline_zero_has_no_ratio_but_is_flagged(self):
        row = report_diff._delta_row("conflicts", 0.0, 5000.0, direction="lower_better")
        assert row["status"] == "regressed"
        assert row["rel_change"] is None

    def test_absent_on_both_sides_yields_no_row(self):
        assert report_diff._delta_row("x", None, None) is None


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCliSurface:
    def test_ncu_diff_subcommand_is_registered_with_expected_flags(self):
        import my_utils.profiling.cli as profiling_cli

        parser = profiling_cli.build_parser()
        args = parser.parse_args(
            ["ncu-diff", "--report-a", "a.ncu-rep", "--report-b", "b.ncu-rep"]
        )
        assert args.func is profiling_cli.cmd_ncu_diff
        assert args.report_a == "a.ncu-rep"
        assert args.report_b == "b.ncu-rep"
        assert args.kernel == "%"
        assert args.format == "md"

    def test_cmd_ncu_diff_renders_selected_format(self, monkeypatch, capsys, tmp_path):
        import my_utils.profiling.cli as profiling_cli

        captured = {}

        def _fake_diff(
            report_a,
            report_b,
            *,
            kernel_like="%",
            findings_per_kernel=24,
            collection_manifest_a="",
            collection_manifest_b="",
            repeat_reports_a=(),
            repeat_reports_b=(),
        ):
            captured["args"] = (
                report_a,
                report_b,
                kernel_like,
                findings_per_kernel,
                collection_manifest_a,
                collection_manifest_b,
                repeat_reports_a,
                repeat_reports_b,
            )
            return {"kernels": [], "clock_guard": {"all_comparable": True}}

        monkeypatch.setattr(profiling_cli, "diff_ncu_reports", _fake_diff)
        rc = profiling_cli.main(
            [
                "ncu-diff",
                "--report-a",
                "x.ncu-rep",
                "--report-b",
                "y.ncu-rep",
                "--kernel",
                "gemm%",
                "--format",
                "json",
            ]
        )
        assert rc == 0
        assert captured["args"] == (
            "x.ncu-rep", "y.ncu-rep", "gemm%", 24, "", "", (), ()
        )
        out = capsys.readouterr().out
        assert '"clock_guard"' in out

    def test_entry_ncu_diff_forwards_subcommand(self, monkeypatch):
        import sys as _sys

        import my_utils.profiling.cli as profiling_cli

        captured = {}

        def _fake_main(argv=None):
            captured["argv"] = list(argv or [])
            return 0

        monkeypatch.setattr(profiling_cli, "main", _fake_main)
        monkeypatch.setattr(
            _sys,
            "argv",
            ["myutils-ncu-diff", "--report-a", "a.ncu-rep", "--report-b", "b.ncu-rep"],
        )
        assert profiling_cli.entry_ncu_diff() == 0
        assert captured["argv"][0] == "ncu-diff"
        assert "--report-a" in captured["argv"]
