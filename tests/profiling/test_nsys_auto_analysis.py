# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.sources.nsys_auto_analysis."""

from __future__ import annotations


from _synthetic_loader import nsys_auto


class TestCollectiveBandwidthHonesty:
    """Bus bandwidth is not derivable from kernel timing, and must not be faked."""

    def test_bandwidth_is_reported_as_unmeasurable(self):
        cov = nsys_auto._collective_bandwidth_coverage(
            [
                {
                    "kernel_name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL",
                    "total_ms": 12.0,
                }
            ]
        )
        assert cov["measurable"] is False
        assert cov["busbw_gbps"] is None and cov["algbw_gbps"] is None
        assert "message bytes" in cov["reason"]
        assert "flight recorder" in cov["how_to_measure"]

    def test_collective_kind_is_read_but_algorithm_is_caveated(self):
        cov = nsys_auto._collective_bandwidth_coverage(
            [
                {
                    "kernel_name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL",
                    "total_ms": 12.0,
                }
            ]
        )
        assert cov["collective_time_ms_by_kind"] == {"allreduce": 12.0}
        assert "NOT reliable" in cov["caveat"] or "NOT" in cov["caveat"]


class TestTraceQualityGating:
    def test_absent_checks_are_not_reported_as_passing(self):
        quality = nsys_auto._assess_quality(None, [])
        assert quality["checked"] is False
        assert quality["trustworthy"] is None, "unvalidated must not read as validated"

    def test_blocked_conclusions_leave_the_recommendation_list(self):
        recs = ["Increase batch size", "Reduce dataloader stalls", "Use CUDA graphs"]
        out = nsys_auto._strike_blocked_recommendations(recs, {"dataloader"})
        assert "Reduce dataloader stalls" not in out
        assert any("WITHHELD" in r for r in out)

    def test_nothing_struck_when_nothing_blocked(self):
        recs = ["Increase batch size"]
        assert nsys_auto._strike_blocked_recommendations(recs, set()) == recs
