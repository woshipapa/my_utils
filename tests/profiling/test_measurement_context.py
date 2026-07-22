# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.analyzers.measurement_context."""

from __future__ import annotations


import pytest


from _synthetic_loader import measurement_context


class TestMeasurementContext:
    """ncu is cold-cache by default; comparing it to wall-clock is invalid."""

    def test_ncu_default_is_cold_cache(self):
        ctx = measurement_context.describe_collection_mode(source="ncu")
        assert ctx.cache_state == measurement_context.CacheState.COLD
        assert any("--cache-control" in n for n in ctx.notes)

    def test_ncu_cannot_answer_overlap(self):
        ctx = measurement_context.describe_collection_mode(source="ncu")
        assert any("overlap" in c for c in ctx.cannot_answer)

    def test_cache_control_none_is_warm(self):
        ctx = measurement_context.describe_collection_mode(
            source="ncu", cache_control="none"
        )
        assert ctx.cache_state == measurement_context.CacheState.WARM

    def test_cold_vs_warm_comparison_is_refused(self):
        cold = measurement_context.describe_collection_mode(source="ncu")
        warm = measurement_context.describe_collection_mode(source="wallclock")
        out = measurement_context.compare_measurements(
            cold, warm, baseline_value=2.0, candidate_value=1.0
        )
        assert out["comparable"] is False
        assert out["ratio"] is None, (
            "an invalid ratio must not be presented as a result"
        )
        assert out["uncomparable_raw_ratio"] == pytest.approx(0.5)
        assert any("cache state" in b for b in out["blockers"])

    def test_like_for_like_comparison_is_allowed(self):
        # Clocks must be supplied: an unrecorded clock no longer passes silently.
        a = measurement_context.describe_collection_mode(
            source="ncu", clocks_locked=True, sm_clock_hz=1.7e9, gpc_clock_hz=1.7e9
        )
        b = measurement_context.describe_collection_mode(
            source="ncu", clocks_locked=True, sm_clock_hz=1.7e9, gpc_clock_hz=1.7e9
        )
        out = measurement_context.compare_measurements(
            a, b, baseline_value=2.0, candidate_value=1.5
        )
        assert out["comparable"] is True
        assert out["ratio"] == pytest.approx(0.75)

    def test_long_unlocked_loop_warns_about_thermals(self):
        ctx = measurement_context.describe_collection_mode(
            source="wallclock", iterations=5000, clocks_locked=False
        )
        assert any("clock" in n.lower() for n in ctx.notes)

    def test_synthetic_inputs_are_recorded_as_a_limit(self):
        ctx = measurement_context.describe_collection_mode(
            source="wallclock", input_distribution="random"
        )
        assert any("real data" in c for c in ctx.cannot_answer)


class TestClockConfound:
    """A duration measured at one clock is not comparable with one at another.

    On a real pair of reports of the same kernel, the wall-clock gap was 12.4%
    and the cycle-normalised gap 5.2% -- half the apparent speedup was the SM
    clock (1674 vs 1789 MHz), and the tool reported the 12.4% as if it were the
    schedule change. It had both clocks and mentioned neither.
    """

    def _ctx(self, sm, gpc=None):
        return measurement_context.describe_collection_mode(
            source="ncu", sm_clock_hz=sm, gpc_clock_hz=gpc or sm
        )

    def test_different_clocks_block_a_duration_comparison(self):
        out = measurement_context.compare_measurements(
            self._ctx(1.674e9),
            self._ctx(1.789e9),
            baseline_value=82464.0,
            candidate_value=73344.0,
        )
        assert out["comparable"] is False
        assert any("different SM clocks" in b for b in out["blockers"])

    def test_clock_normalised_ratio_is_reported(self):
        out = measurement_context.compare_measurements(
            self._ctx(1.674e9),
            self._ctx(1.789e9),
            baseline_value=82464.0,
            candidate_value=73344.0,
        )
        # 0.8894 observed x 1.0691 clock = 0.9508 -- 5% not 11%
        assert out["clock_normalised_ratio"] == pytest.approx(0.951, abs=0.005)
        assert "Normalising for the clock" in out["verdict"]

    def test_same_clock_does_not_block(self):
        out = measurement_context.compare_measurements(
            self._ctx(1.789e9),
            self._ctx(1.789e9),
            baseline_value=100.0,
            candidate_value=90.0,
        )
        assert out["comparable"] is True

    def test_gpc_sm_disagreement_invalidates_cycle_figures(self):
        """5.3% between domains that should agree means they were not measured
        over the same window."""
        ctx = self._ctx(1.674e9, gpc=1.762e9)
        assert ctx.clock_disagreement == pytest.approx(1.053, abs=0.005)
        assert any("derived from cycles" in c for c in ctx.cannot_answer)
        assert any("re-collect" in n for n in ctx.notes)

    def test_agreeing_clocks_raise_nothing(self):
        ctx = self._ctx(1.789e9, gpc=1.789e9)
        assert not [c for c in ctx.cannot_answer if "derived from cycles" in c]


class TestClockControlBias:
    """`--clock-control base` cannot lower the HBM clock on H100/B200-class
    parts, so the compute:memory balance reads memory-rich. A caveat about the
    balance, never a claim that any counter is wrong."""

    # The real cta_pingpong H100 report: SM at 88% of rated, DRAM at 99.97%.
    _REAL = dict(
        sm_clock_hz=1.7445e9,
        dram_clock_hz=2.618e9,
        rated_sm_clock_hz=1.98e9,
        rated_dram_clock_hz=2.619e9,
    )

    def test_declared_base_clock_control_is_biased(self):
        out = measurement_context.assess_clock_control_bias(clock_control="base")
        assert out["checked"] is True and out["biased"] is True
        assert out["method"] == "declared"
        assert "memory-rich" in out["caveat"]

    def test_symptom_detection_on_real_report_values(self):
        out = measurement_context.assess_clock_control_bias(**self._REAL)
        assert out["biased"] is True and out["method"] == "symptom"
        assert out["sm_share_of_rated"] == pytest.approx(0.881, abs=0.001)
        assert out["dram_share_of_rated"] == pytest.approx(0.9996, abs=0.001)
        assert "nvidia-smi -lgc" in out["caveat"]
        assert "--clock-control none" in out["caveat"]

    def test_symptom_names_its_own_limits(self):
        """The report does not record --clock-control, and the same clock
        signature is produced by power or thermal capping. The caveat must say
        so -- and say why the bias direction holds either way."""
        out = measurement_context.assess_clock_control_bias(**self._REAL)
        assert "thermal" in out["limits"]
        assert "bias direction is the same" in out["limits"]

    def test_full_clock_sm_is_not_biased(self):
        out = measurement_context.assess_clock_control_bias(
            **{**self._REAL, "sm_clock_hz": 1.96e9}
        )
        assert out["checked"] is True and out["biased"] is False
        assert out["caveat"] == ""

    def test_missing_clock_is_unassessed_not_unbiased(self):
        out = measurement_context.assess_clock_control_bias(
            sm_clock_hz=1.7445e9, rated_sm_clock_hz=1.98e9
        )
        assert out["checked"] is False and out["biased"] is None
        assert "Unassessed is not the same as unbiased" in out["limits"]

    def test_caveat_states_bias_not_wrongness(self):
        out = measurement_context.assess_clock_control_bias(**self._REAL)
        assert "not wrong" in out["caveat"]
        assert "biased toward looking memory-rich" in out["caveat"]

    def test_describe_collection_mode_carries_the_caveat(self):
        ctx = measurement_context.describe_collection_mode(source="ncu", **self._REAL)
        assert ctx.clock_control_bias and ctx.clock_control_bias["biased"] is True
        assert any("memory-rich" in n for n in ctx.notes)
        assert ctx.to_dict()["clock_control_bias"]["method"] == "symptom"

    def test_describe_collection_mode_without_clock_data_is_unchanged(self):
        ctx = measurement_context.describe_collection_mode(source="ncu")
        assert ctx.clock_control_bias is None
        assert not any("memory-rich" in n for n in ctx.notes)
