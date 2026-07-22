"""Tests for my_utils.profiling.analyzers.trace_quality."""

from __future__ import annotations


from _synthetic_loader import trace_quality


class TestDerivedMetricInvariants:
    """A violated identity means a wrong denominator, not a finding."""

    def test_above_peak_blocks(self):
        issues = trace_quality.check_derived_metric_invariants(mfu=520.0, dtype="bf16")
        assert any(i.key == "mfu_above_peak" and i.blocks for i in issues)

    def test_hfu_below_mfu_blocks(self):
        issues = trace_quality.check_derived_metric_invariants(
            mfu=45.0, hfu=38.0, dtype="bf16"
        )
        assert any(i.key == "hfu_below_mfu" and i.blocks for i in issues)

    def test_unknown_dtype_blocks(self):
        issues = trace_quality.check_derived_metric_invariants(mfu=41.0)
        assert any(i.key == "unknown_dtype_denominator" and i.blocks for i in issues)

    def test_healthy_is_clean(self):
        assert not trace_quality.check_derived_metric_invariants(
            mfu=45.0, hfu=52.0, dtype="bf16"
        )


class TestShapeKeyedGrouping:
    """One kernel name covers genuinely different work."""

    def test_shapes_are_not_merged(self):
        launches = [
            {"kernel_name": "k", "grid_size": 128, "duration_ns": 10_000}
        ] * 8 + [{"kernel_name": "k", "grid_size": 512, "duration_ns": 40_000}] * 8
        g = trace_quality.group_kernels_by_shape(launches)
        assert g["group_count"] == 2
        assert g["distinct_names"] == 1
        assert "would have merged them" in g["note"]

    def test_dispersed_group_is_flagged(self):
        launches = [
            {"kernel_name": "k", "grid_size": 512, "duration_ns": 40_000}
        ] * 8 + [{"kernel_name": "k", "grid_size": 512, "duration_ns": 900_000}]
        g = trace_quality.group_kernels_by_shape(launches)
        assert g["non_stationary"]
        assert "single" in g["warning"]
