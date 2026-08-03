# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.ncu.sampling_validity."""

from __future__ import annotations


import pytest


from _synthetic_loader import sampling_validity


class TestPcSamplingValidity:
    """Mirrors NVIDIA's PCSamplingData rule; biased samples must not be used."""

    def test_dropped_samples_block_attribution(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=5000, interval_cycles=1000, dropped_bytes=4096
        )
        assert out["usable"] is False
        assert "stall_attribution" in out["blocked_conclusions"]
        issue = next(i for i in out["issues"] if i["key"] == "pcsamp_dropped_samples")
        assert "--warp-sampling-interval" in issue["remedy"]

    def test_buffer_overflow_blocks_attribution(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=5000,
            interval_cycles=1000,
            buffer_overflow=1,
            buffer_size_bytes=1 << 20,
        )
        assert out["usable"] is False
        assert "--warp-sampling-buffer-size" in out["issues"][0]["remedy"]

    def test_zero_samples_explains_short_kernel(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=0, interval_cycles=100000, kernel_duration_cycles=5000
        )
        issue = next(i for i in out["issues"] if i["key"] == "pcsamp_no_samples")
        assert "shorter than" in issue["detail"]

    def test_few_samples_block_ranking_but_not_distribution(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=40, interval_cycles=1000
        )
        assert "hot_line_ranking" in out["blocked_conclusions"]
        assert "sampled_stall_distribution" not in out["blocked_conclusions"]

    def test_healthy_sampling_is_usable(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=50000, interval_cycles=1000, kernel_duration_cycles=10_000_000
        )
        assert out["usable"] is True and out["blocked_conclusions"] == []

    def test_absent_interval_is_not_reported_as_valid(self):
        out = sampling_validity.check_pc_sampling_validity(sample_count=100)
        assert out["checked"] is False and out["usable"] is None


class TestPmSamplingValidity:
    """Mirrors NVIDIA's PMSamplingData rule, including its architecture gate."""

    def test_unsupported_architecture_is_reported(self):
        out = sampling_validity.check_pm_sampling_validity(cc_major=7, cc_minor=0)
        assert out["supported"] is False
        assert "pm_sampling_timeline" in out["blocked_conclusions"]

    def test_interval_longer_than_workload_blocks_the_timeline(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=2_000_000, duration=1_000_000
        )
        assert out["usable"] is False
        assert out["interval_duration_ratio"] == pytest.approx(2.0)

    def test_interval_over_ten_percent_is_flagged(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=200_000, duration=1_000_000
        )
        assert out["usable"] is False
        assert "phase_detection" in out["blocked_conclusions"]

    def test_floor_interval_advises_longer_workload_not_smaller_interval(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=500, duration=600
        )
        remedy = out["issues"][0]["remedy"]
        assert "longer-running" in remedy

    def test_fine_interval_is_usable(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=1000, duration=10_000_000
        )
        assert out["usable"] is True
        assert out["estimated_sample_count"] == pytest.approx(10000)

    def test_dropped_samples_block_the_pm_timeline(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9,
            cc_minor=0,
            interval=1000,
            duration=10_000_000,
            dropped_samples=390,
        )
        assert out["usable"] is False
        assert "pm_sampling_timeline" in out["blocked_conclusions"]
        issue = next(i for i in out["issues"] if i["key"] == "pm_sampling_dropped_samples")
        assert "--pm-sampling-buffer-size" in issue["remedy"]

    def test_dropped_samples_fail_closed_without_interval_metadata(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, dropped_samples=1
        )
        assert out["usable"] is False
        assert "pm_sampling_timeline" in out["blocked_conclusions"]

    def test_merged_samples_keep_the_timeline_but_block_phase_claims(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9,
            cc_minor=0,
            interval=1000,
            duration=10_000_000,
            merged_samples=12,
        )
        assert out["usable"] is False
        assert "phase_detection" in out["blocked_conclusions"]
        assert "pm_sampling_timeline" not in out["blocked_conclusions"]

    def test_merged_samples_are_reported_without_interval_metadata(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, merged_samples=1
        )
        assert out["usable"] is False
        assert "phase_detection" in out["blocked_conclusions"]

    def test_missing_context_trace_blocks_device_wide_mps_samples(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9,
            cc_minor=0,
            interval=1_000,
            duration=100_000,
            context_switch_trace_available=False,
            mps_active=True,
        )
        assert out["usable"] is False
        assert "pm_sampling_timeline" in out["blocked_conclusions"]
        issue = next(
            item
            for item in out["issues"]
            if item["key"] == "pm_sampling_context_scope_unavailable"
        )
        assert "MPS" in issue["detail"]

    def test_missing_context_trace_is_not_a_false_mps_claim(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9,
            cc_minor=0,
            interval=1_000,
            duration=100_000,
            context_switch_trace_available=False,
            mps_active=False,
            mig_active=False,
        )
        assert out["usable"] is True


class TestReplayClockDrift:
    """Thermal sag between replay passes: one collection, several clock states.

    ncu multiplexes the metric set across many replay passes; on a high-TDP
    part the clock can sag between the early and late ones, and nothing in the
    report says so. The check compares per-pass effective-clock estimates and
    only trusts a gap whose direction the estimator kinds can prove: an
    estimate ABOVE a measured clock proves drift, a lower bound BELOW one
    proves nothing (the unit may have idled).
    """

    def test_two_measured_clocks_apart_trigger(self):
        out = sampling_validity.check_replay_clock_drift(
            {
                "pass 0 (sm__cycles_elapsed.avg)": {
                    "clock_hz": 1.80e9,
                    "kind": "measured",
                },
                "pass 5 (sm__cycles_elapsed.avg)": {
                    "clock_hz": 1.70e9,
                    "kind": "measured",
                },
            }
        )
        assert out["checked"] is True and out["drifted"] is True
        assert out["supported_drift"] == pytest.approx(1.80 / 1.70 - 1)
        issue = out["issues"][0]
        assert issue["key"] == "replay_clock_drift"
        # A caveat, not a finding: it must not block conclusions.
        assert issue["blocks"] is False
        assert "nvidia-smi -lgc" in issue["remedy"]

    def test_lower_bound_above_a_measured_clock_proves_drift(self):
        """The real cta_pingpong case: the PM pass's cycles-per-bucket bound
        (1825 MHz) sits above the collection's measured clock (1745 MHz)."""
        out = sampling_validity.check_replay_clock_drift(
            {
                "pass 4 (sm__cycles_active.avg)": {
                    "clock_hz": 1.825e9,
                    "kind": "lower_bound",
                },
                "collection (sm__cycles_elapsed.avg.per_second)": {
                    "clock_hz": 1.7445e9,
                    "kind": "measured",
                },
            }
        )
        assert out["drifted"] is True
        assert "at least" in out["issues"][0]["detail"]

    def test_lower_bound_below_a_measured_clock_is_inconclusive(self):
        """An activity-derived bound below a measured clock can be a partly
        idle unit rather than a slower clock; claiming drift there would
        manufacture a caveat from estimator undercount."""
        out = sampling_validity.check_replay_clock_drift(
            {
                "a": {"clock_hz": 1.80e9, "kind": "measured"},
                "b": {"clock_hz": 1.70e9, "kind": "lower_bound"},
            }
        )
        assert out["drifted"] is False and out["issues"] == []
        assert out["raw_spread"] == pytest.approx(1.80 / 1.70 - 1)
        assert "Inconclusive" in out["note"]

    def test_two_lower_bounds_prove_nothing(self):
        out = sampling_validity.check_replay_clock_drift(
            {
                "a": {"clock_hz": 1.85e9, "kind": "lower_bound"},
                "b": {"clock_hz": 1.70e9, "kind": "lower_bound"},
            }
        )
        assert out["drifted"] is False and out["issues"] == []

    def test_agreeing_clocks_are_clean(self):
        out = sampling_validity.check_replay_clock_drift(
            {
                "a": {"clock_hz": 1.750e9, "kind": "measured"},
                "b": {"clock_hz": 1.760e9, "kind": "measured"},
            }
        )
        assert out["drifted"] is False and out["issues"] == []
        assert "agree" in out["note"]

    def test_fewer_than_two_estimates_is_unchecked_not_clean(self):
        out = sampling_validity.check_replay_clock_drift(
            {"only": {"clock_hz": 1.75e9, "kind": "measured"}}
        )
        assert out["checked"] is False and out["drifted"] is None
        assert "not a clean result" in out["note"]
        assert sampling_validity.check_replay_clock_drift({})["checked"] is False
        assert sampling_validity.check_replay_clock_drift(None)["checked"] is False

    def test_caveat_does_not_claim_the_data_is_wrong(self):
        out = sampling_validity.check_replay_clock_drift(
            {
                "a": {"clock_hz": 1.80e9, "kind": "measured"},
                "b": {"clock_hz": 1.70e9, "kind": "measured"},
            }
        )
        detail = out["issues"][0]["detail"]
        assert "faithful reading" in detail
        assert "mix different clock states" in detail
