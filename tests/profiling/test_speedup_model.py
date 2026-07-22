# SPDX-License-Identifier: Apache-2.0
"""GPA-style speedup upper bounds, and the honesty constraints around them.

The model (my_utils/profiling/ncu/speedup_model.py) turns the closed stall
stack into a per-finding *upper bound* on the win from removing that stall
class. What these tests pin down is less the arithmetic than the honesty
contract:

* bounds come only from stall-backed findings,
* a stall stack that failed its closure check yields no bound -- with the
  reason stated, not a silent absence,
* the bound respects ceilings the engine already knows (speed-of-light,
  DRAM roofline, occupancy headroom), and
* the rendered markdown labels every number as an upper bound.
"""

import pytest

from _synthetic_loader import (
    gpu_specs,
    ncu_diagnostics,
    ncu_report_tools,
    speedup_model,
)


def _stall_metrics(reasons, total=20.0, extra=None):
    """A closed (or deliberately unclosed) stall stack as raw ncu metrics."""
    metrics = {
        "smsp__average_warp_latency_per_inst_issued.ratio": total,
        "smsp__issue_active.avg.per_cycle_active": 0.1,
    }
    for name, value in reasons.items():
        metrics[f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio"] = (
            value
        )
    metrics.update(extra or {})
    return metrics


class TestBoundEmission:
    def test_closed_stack_yields_a_bound_from_the_share(self):
        # 12 + 6 + 2 = 20: closure holds exactly. long_scoreboard share 0.6.
        out = ncu_diagnostics.diagnose_kernel(
            _stall_metrics({"long_scoreboard": 12.0, "barrier": 6.0, "wait": 2.0})
        )
        ls = next(
            f for f in out["findings"] if f["category"] == "stall_long_scoreboard"
        )
        assert ls["estimated_speedup_upper_bound"] == pytest.approx(2.5, abs=0.01)
        assert "1/(1-0.60)" in ls["speedup_basis"]
        assert "assumes" in ls["speedup_basis"]

    def test_non_stall_findings_never_get_a_bound(self):
        out = ncu_diagnostics.diagnose_kernel(
            _stall_metrics(
                {"long_scoreboard": 12.0, "barrier": 6.0, "wait": 2.0},
                extra={
                    # Register spilling: a real finding, but not stall-backed.
                    "smsp__inst_executed_op_local_ld.sum": 70000.0,
                    "smsp__inst_executed_op_local_st.sum": 20000.0,
                    "smsp__inst_executed.sum": 500000.0,
                    "launch__registers_per_thread": 40.0,
                    "launch__block_size": 256.0,
                },
            )
        )
        for finding in out["findings"]:
            if not finding["category"].startswith("stall_"):
                assert finding["estimated_speedup_upper_bound"] is None
                assert finding["speedup_basis"] is None

    def test_bound_survives_the_to_dict_roundtrip(self):
        finding = ncu_diagnostics.Finding(
            category="stall_barrier",
            title="t",
            summary="s",
            estimated_speedup_upper_bound=1.4,
            speedup_basis="model",
        )
        as_dict = finding.to_dict()
        assert as_dict["estimated_speedup_upper_bound"] == 1.4
        assert as_dict["speedup_basis"] == "model"


class TestClosureHonesty:
    """No closure, no bound -- and the reason is stated, not implied."""

    def test_underclosed_stack_withholds_the_bound(self):
        # Only 8 of 20 cycles accounted: 40% closure, far below the 90% gate.
        out = ncu_diagnostics.diagnose_kernel(_stall_metrics({"long_scoreboard": 8.0}))
        ls = next(
            f for f in out["findings"] if f["category"] == "stall_long_scoreboard"
        )
        assert ls["estimated_speedup_upper_bound"] is None
        assert "closure check" in ls["speedup_basis"]
        assert "40%" in ls["speedup_basis"]

    def test_overclosed_stack_withholds_the_bound(self):
        # States sum to 22 against a reported total of 20: 110% closure.
        out = ncu_diagnostics.diagnose_kernel(
            _stall_metrics({"long_scoreboard": 14.0, "barrier": 8.0})
        )
        ls = next(
            f for f in out["findings"] if f["category"] == "stall_long_scoreboard"
        )
        assert ls["estimated_speedup_upper_bound"] is None
        assert "closure check" in ls["speedup_basis"]
        assert "110%" in ls["speedup_basis"]

    def test_absent_total_withholds_the_bound_with_its_own_reason(self):
        closed, why = speedup_model.stall_stack_closure({"explained_share": None})
        assert not closed
        assert "no denominator" in why

    def test_closure_gates_match_the_stall_analysis_gates(self):
        # analyze_stalls flags closure failure below 0.9 and above 1.02; the
        # bound model must withhold on exactly the same interval, or a report
        # could carry a closure warning and a bound side by side.
        assert speedup_model.CLOSURE_LOW == 0.90
        assert speedup_model.CLOSURE_HIGH == 1.02
        ok, _ = speedup_model.stall_stack_closure({"explained_share": 1.0})
        assert ok


class TestCeilings:
    """The share-removal figure is capped by ceilings we already compute."""

    def _view(self, extra=None):
        return ncu_diagnostics.MetricView(extra or {})

    def test_speed_of_light_caps_the_bound(self):
        # Raw model says 2.5x, but the memory system is at 80% of peak: no
        # speedup can exceed 100/80 = 1.25x.
        bound, basis = speedup_model.estimate_stall_speedup_bound(
            0.6,
            stall_key="long_scoreboard",
            view=self._view(
                {
                    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 80.0
                }
            ),
        )
        assert bound == pytest.approx(1.25, abs=0.01)
        assert "80% of peak" in basis
        assert "capping" in basis

    def test_dram_roofline_caps_a_device_memory_stall(self):
        spec = gpu_specs.lookup_gpu_spec("H100 SXM5")
        assert spec is not None and spec.hbm_bandwidth_gbps == 3350.0
        # 200 KB in 100 us = 2000 GB/s achieved; roofline caps at 3350/2000.
        view = self._view(
            {"dram__bytes.sum": 200_000.0, "gpu__time_duration.sum": 100.0}
        )
        bound, basis = speedup_model.estimate_stall_speedup_bound(
            0.6, stall_key="long_scoreboard", view=view, gpu_spec=spec
        )
        assert bound == pytest.approx(3350.0 / 2000.0, abs=0.01)
        assert "roofline" in basis

    def test_dram_roofline_does_not_cap_a_synchronization_stall(self):
        # A barrier stall is not a memory stall: eliminating it moves no bytes
        # faster, so the bandwidth ceiling must not shrink its bound.
        spec = gpu_specs.lookup_gpu_spec("H100 SXM5")
        view = self._view(
            {"dram__bytes.sum": 200_000.0, "gpu__time_duration.sum": 100.0}
        )
        bound, _basis = speedup_model.estimate_stall_speedup_bound(
            0.6, stall_key="barrier", view=view, gpu_spec=spec
        )
        assert bound == pytest.approx(2.5, abs=0.01)

    def test_occupancy_headroom_caps_a_latency_hiding_stall(self):
        # Long-scoreboard waits are hidden by more warps, and the limiter data
        # says only 60/50 = 1.2x more warps can exist.
        bound, basis = speedup_model.estimate_stall_speedup_bound(
            0.6,
            stall_key="long_scoreboard",
            view=self._view(),
            occupancy={
                "achieved_occupancy_pct": 50.0,
                "theoretical_occupancy_pct": 60.0,
            },
        )
        assert bound == pytest.approx(1.2, abs=0.01)
        assert "occupancy headroom" in basis

    def test_headroom_cap_skipped_when_occupancy_model_does_not_apply(self):
        # Warp-specialized kernels opt out of the occupancy model, so their
        # headroom figure is an artifact and must not cap anything.
        bound, _basis = speedup_model.estimate_stall_speedup_bound(
            0.6,
            stall_key="long_scoreboard",
            view=self._view(),
            occupancy={
                "achieved_occupancy_pct": 50.0,
                "theoretical_occupancy_pct": 60.0,
                "occupancy_model_applicable": False,
            },
        )
        assert bound == pytest.approx(2.5, abs=0.01)

    def test_headroom_does_not_cap_a_synchronization_stall(self):
        bound, _basis = speedup_model.estimate_stall_speedup_bound(
            0.6,
            stall_key="barrier",
            view=self._view(),
            occupancy={
                "achieved_occupancy_pct": 50.0,
                "theoretical_occupancy_pct": 60.0,
            },
        )
        assert bound == pytest.approx(2.5, abs=0.01)

    def test_degenerate_share_yields_no_bound(self):
        for share in (0.0, 1.0, -0.1):
            bound, basis = speedup_model.estimate_stall_speedup_bound(
                share, stall_key="barrier", view=self._view()
            )
            assert bound is None
            assert "cannot represent" in basis


class TestRankingAndRendering:
    def test_findings_are_ordered_by_bound_within_severity(self):
        # barrier 8/20 = 40% (high tier) and long_scoreboard 9/20 = 45% (high
        # tier): same severity, so the larger bound must come first.
        out = ncu_diagnostics.diagnose_kernel(
            _stall_metrics({"long_scoreboard": 9.0, "barrier": 8.0, "wait": 3.0})
        )
        stall_findings = [
            f for f in out["findings"] if f["category"].startswith("stall_")
        ]
        assert len(stall_findings) == 2
        bounds = [f["estimated_speedup_upper_bound"] for f in stall_findings]
        assert bounds == sorted(bounds, reverse=True)

    def _payload(self, finding):
        return {
            "report_path": "r.ncu-rep",
            "kernels_analyzed": 1,
            "kernels": [
                {
                    "kernel_name": "k",
                    "verdict": "latency_bound",
                    "findings": [finding],
                }
            ],
        }

    def test_markdown_labels_the_bound_as_an_upper_bound(self):
        text = ncu_report_tools.diagnose_result_to_markdown(
            self._payload(
                {
                    "severity": "high",
                    "title": "Warp stalls dominated by Barrier",
                    "summary": "s",
                    "estimated_speedup_upper_bound": 1.71,
                    "speedup_basis": "share-removal model",
                    "actions": [],
                }
            )
        )
        assert "at most 1.71x" in text
        assert "upper bound, not a prediction" in text
        assert "would buy at most 71%" in text
        assert "share-removal model" in text

    def test_markdown_states_why_a_bound_was_withheld(self):
        text = ncu_report_tools.diagnose_result_to_markdown(
            self._payload(
                {
                    "severity": "high",
                    "title": "Warp stalls dominated by Barrier",
                    "summary": "s",
                    "estimated_speedup_upper_bound": None,
                    "speedup_basis": "No speedup bound: the stall stack failed "
                    "its closure check",
                    "actions": [],
                }
            )
        )
        assert "speedup bound withheld" in text
        assert "closure check" in text
        assert "at most" not in text


class TestKernelProCalibration:
    """Severity tiers and triggers calibrated to KernelPro (arXiv:2606.26453)."""

    def _stall_finding(self, reasons, key):
        result = ncu_diagnostics.analyze_stalls(
            ncu_diagnostics.MetricView(_stall_metrics(reasons))
        )
        return next(f for f in result["findings"] if f.category == f"stall_{key}")

    def test_forty_percent_share_is_high_severity(self):
        # 9/20 = 45%: below the old 50% boundary, above KernelPro's 40% tier.
        f = self._stall_finding(
            {"long_scoreboard": 9.0, "barrier": 8.0, "wait": 3.0}, "long_scoreboard"
        )
        assert f.severity == "high"
        assert f.evidence["severity_tier"] == "high"
        assert "KernelPro" in f.evidence["severity_tier_source"]

    def test_sixty_percent_share_is_critical_tier(self):
        f = self._stall_finding(
            {"long_scoreboard": 13.0, "barrier": 5.0, "wait": 2.0}, "long_scoreboard"
        )
        assert f.severity == "high"  # the enum has no critical level
        assert f.evidence["severity_tier"] == "critical"

    def test_below_forty_percent_stays_medium(self):
        f = self._stall_finding(
            {"long_scoreboard": 7.0, "barrier": 10.0, "wait": 3.0}, "long_scoreboard"
        )
        assert f.severity == "medium"
        assert f.evidence["severity_tier"] == "moderate"

    def test_barrier_trigger_fires_just_above_thirty_percent(self):
        # KernelPro's dedicated barrier trigger: 30% of issue-slots.
        f = self._stall_finding(
            {"barrier": 6.4, "long_scoreboard": 11.0, "wait": 2.6}, "barrier"
        )
        assert f.evidence["share_of_warp_latency"] == pytest.approx(0.32, abs=0.01)

    def test_thresholds_are_pinned(self):
        t = ncu_diagnostics.SOL_THRESHOLDS
        assert t["stall_share_gate_barrier"] == 0.30
        assert t["stall_share_high"] == 0.40
        assert t["stall_share_critical"] == 0.60
        # The general gate keeps NVIDIA's documented CPIStall value, which is
        # stricter than KernelPro's 40% general trigger.
        assert t["stall_share_gate"] == 0.30


class TestOccupancyAdviceSuppression:
    """Volkov (GTC 2010): ILP substitutes for occupancy. A saturated pipe means
    more warps cannot help, so occupancy advice is suppressed -- audibly."""

    _LOW_OCCUPANCY = {
        # Low SOL on both axes so the latency gate stays open.
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 18.0,
        "smsp__issue_active.avg.per_cycle_active": 0.2,
        "sm__warps_active.avg.pct_of_peak_sustained_active": 25.0,
        "sm__maximum_warps_per_active_cycle_pct": 40.0,
        "launch__occupancy_limit_registers": 12,
        "launch__occupancy_limit_warps": 64,
        "launch__occupancy_limit_blocks": 64,
        "launch__occupancy_limit_shared_mem": 64,
    }

    def test_saturated_pipe_suppresses_the_advice_but_not_the_finding(self):
        view = ncu_diagnostics.MetricView(
            {
                **self._LOW_OCCUPANCY,
                "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": 86.0,
            }
        )
        result = ncu_diagnostics.analyze_occupancy(view)
        finding = next(
            f for f in result["findings"] if f.category == "occupancy_limited_registers"
        )
        assert "more warps would not increase throughput" in finding.summary
        assert "fma" in finding.summary
        assert "~86%" in finding.summary
        assert finding.evidence["occupancy_advice_suppressed"]["pipe"] == "fma"
        assert any("Do not raise occupancy" in a for a in finding.actions)
        assert not any("__launch_bounds__" in a for a in finding.actions)

    def test_unsaturated_pipes_leave_the_advice_alone(self):
        view = ncu_diagnostics.MetricView(
            {
                **self._LOW_OCCUPANCY,
                "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": 30.0,
            }
        )
        result = ncu_diagnostics.analyze_occupancy(view)
        finding = next(
            f for f in result["findings"] if f.category == "occupancy_limited_registers"
        )
        assert "occupancy_advice_suppressed" not in finding.evidence
        assert any("__launch_bounds__" in a for a in finding.actions)

    def test_high_issue_slot_utilization_also_suppresses(self):
        guard = ncu_diagnostics._occupancy_advice_suppressor(
            ncu_diagnostics.MetricView(
                {"smsp__issue_active.avg.pct_of_peak_sustained_active": 84.0}
            )
        )
        assert guard is not None
        assert guard["pipe"] == "issue"
        assert "more warps would not increase" in guard["reason"]

    def test_nothing_saturated_returns_none(self):
        assert (
            ncu_diagnostics._occupancy_advice_suppressor(
                ncu_diagnostics.MetricView(
                    {
                        "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": 40.0,
                        "smsp__issue_active.avg.per_cycle_active": 0.3,
                    }
                )
            )
            is None
        )

    def test_scheduler_advice_swaps_to_pipe_work_when_saturated(self):
        starved = {
            "smsp__warps_eligible.avg.per_cycle_active": 0.4,
            "smsp__warps_active.avg.per_cycle_active": 3.0,
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active": 88.0,
        }
        result = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView(starved)
        )
        finding = result["findings"][0]
        assert any("Do not increase occupancy" in a for a in finding.actions)
        assert not any(a.startswith("Increase occupancy") for a in finding.actions)
        assert "more warps would not increase throughput" in finding.summary

    def test_scheduler_advice_unchanged_when_nothing_saturated(self):
        starved = {
            "smsp__warps_eligible.avg.per_cycle_active": 0.4,
            "smsp__warps_active.avg.per_cycle_active": 3.0,
        }
        result = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView(starved)
        )
        finding = result["findings"][0]
        assert any(a.startswith("Increase occupancy") for a in finding.actions)
