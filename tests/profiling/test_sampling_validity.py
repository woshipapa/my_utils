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
