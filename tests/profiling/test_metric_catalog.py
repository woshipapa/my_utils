"""Tests for my_utils.profiling.ncu.metric_catalog."""

from __future__ import annotations


import pytest


from _synthetic_loader import metric_catalog


def test_stall_taxonomy_is_complete():
    """Nsight Compute ships exactly 19 warp-stall reasons."""
    assert len(metric_catalog.STALL_REASONS) == 19
    for key, reason in metric_catalog.STALL_REASONS.items():
        assert reason.metric_name.startswith("smsp__average_warps_issue_stalled_")
        assert reason.metric_name.endswith("_per_issue_active.ratio")
        assert reason.bucket
        # Every actionable reason must carry advice; the benign ones need not.
        if key not in metric_catalog.BENIGN_STALL_KEYS:
            assert reason.fixes, f"{key} has no suggested fix"


def test_pc_sampling_spellings_differ_where_nvidia_renamed_them():
    gmma = metric_catalog.STALL_REASONS["gmma"]
    assert gmma.pcsamp_metric_name.endswith("warpgroup_arrive")


def test_metric_catalog_entries_are_well_formed():
    assert len(metric_catalog.METRIC_CATALOG) > 80
    for key, spec in metric_catalog.METRIC_CATALOG.items():
        assert spec.names, f"{key} has no metric names"
        assert spec.section and spec.category


@pytest.mark.parametrize(
    "cc_major,cc_minor,family",
    [(7, 0, "volta"), (7, 5, "turing"), (8, 0, "ampere"),
     (8, 9, "ada"), (9, 0, "hopper"), (10, 0, "blackwell"), (12, 0, "blackwell")],
)
def test_architecture_detection_spans_volta_to_blackwell(cc_major, cc_minor, family):
    assert metric_catalog.describe_arch(cc_major, cc_minor)["family"] == family


def test_architecture_falls_back_to_sm_count():
    result = metric_catalog.describe_arch(None, None, 108)
    assert result["family"] == "ampere"
    assert result["source"] == "sm_count_heuristic"
    assert metric_catalog.describe_arch()["family"] == "unknown"


def test_explain_metric_works_without_an_ncu_installation():
    """Falls back to decoding the name rather than fabricating an explanation."""
    result = metric_catalog.explain_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert result["metric"]
    # Either the index resolved it or the name was decoded; never silently empty.
    assert result.get("description") or result.get("interpretation") or result.get("unit")


def test_explain_metric_gives_a_verdict_only_where_a_rule_exists():
    bad = metric_catalog.explain_metric(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 28.0)
    assert bad["has_rule"] is True
    assert bad["verdict"] == "bad"
    assert bad["distance_from_ideal"] == pytest.approx(24.0)

    good = metric_catalog.explain_metric(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 4.0)
    assert good["verdict"] == "ok"

    # Without a value there is no verdict to give.
    assert "verdict" not in metric_catalog.explain_metric(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")
