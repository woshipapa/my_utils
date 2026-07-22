# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.ncu.section_index."""

from __future__ import annotations


import pytest


from _synthetic_loader import metric_catalog, section_index


def test_metric_name_decoding():
    """The name grammar carries unit / quantity / rollup / submetric."""
    decoded = section_index.decode_metric_name(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    )
    assert decoded["unit"] == "sm"
    assert decoded["quantity"] == "throughput"
    assert decoded["rollup"] == "avg"
    assert decoded["submetric"] == "pct_of_peak_sustained_elapsed"

    # No rollup: the submetric follows the name directly.
    plain = section_index.decode_metric_name("l1tex__t_sector_hit_rate.pct")
    assert plain["unit"] == "l1tex"
    assert plain["rollup"] == ""
    assert plain["submetric"] == "pct"

    # Collection prefixes are stripped, not mistaken for the unit.
    prefixed = section_index.decode_metric_name("pmsampling:dram__bytes.sum")
    assert prefixed["prefix"] == "pmsampling"
    assert prefixed["unit"] == "dram"

    # A launch property has no unit separator at all.
    assert section_index.decode_metric_name("launch__grid_size")["unit"] == "launch"


def test_active_vs_elapsed_distinction_is_documented():
    """Choosing the wrong denominator silently changes the conclusion."""
    active = section_index.SUBMETRIC_MEANINGS["pct_of_peak_sustained_active"]
    elapsed = section_index.SUBMETRIC_MEANINGS["pct_of_peak_sustained_elapsed"]
    assert "ACTIVE" in active
    assert "WHOLE" in elapsed


@pytest.mark.skipif(
    section_index.find_sections_dir() is None,
    reason="no Nsight Compute installation on this machine",
)
def test_section_index_covers_the_full_set():
    """Against a real install, every --set full metric must be indexed."""
    index = section_index.build_section_index()
    assert index is not None
    assert len(index.sections) >= 20
    # The index exists precisely so coverage is complete rather than curated.
    assert len(index.in_set("full")) > 500
    # NVIDIA's own Labels come through, which is what makes an unknown metric
    # explainable at all.
    entry = index.explain("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert entry is not None and entry.label
    # The units a user asks about are all represented.
    for unit in ("l1tex", "lts", "dram", "sm", "smsp", "launch"):
        assert index.by_unit(unit), f"no metrics indexed for {unit}"


class TestMetricGrammar:
    def test_every_catalog_metric_decodes_to_a_unit(self):
        """A catalog name that decodes to no unit would vanish from the inventory."""
        undecodable = []
        for spec in metric_catalog.METRIC_CATALOG.values():
            for name in spec.names:
                parts = section_index.decode_metric_name(name)
                if not parts.get("unit") and not section_index._legacy_unit(name):
                    undecodable.append(name)
        assert not undecodable, f"catalog names with no decodable unit: {undecodable}"

    def test_every_catalog_metric_lands_on_an_axis(self):
        orphans = []
        for spec in metric_catalog.METRIC_CATALOG.values():
            for name in spec.names:
                unit = section_index.decode_metric_name(name).get(
                    "unit"
                ) or section_index._legacy_unit(name)
                if unit and unit not in section_index.UNIT_AXIS:
                    orphans.append((name, unit))
        assert not orphans, f"metric units with no axis: {sorted(set(orphans))}"

    def test_active_and_elapsed_denominators_are_distinguished(self):
        """Mixing the two is how an idle unit gets ranked as the bottleneck."""
        assert (
            section_index.denominator_of(
                "sm__throughput.avg.pct_of_peak_sustained_active"
            )
            == "active"
        )
        assert (
            section_index.denominator_of(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed"
            )
            == "elapsed"
        )
        assert section_index.denominator_of("dram__bytes.sum") == ""

    def test_collection_prefixes_are_stripped(self):
        parts = section_index.decode_metric_name(
            "pmsampling:smsp__warps_issue_stalled_barrier.avg"
        )
        assert parts["prefix"] == "pmsampling"
        assert parts["unit"] == "smsp" and parts["rollup"] == "avg"

    def test_unknown_unit_is_reported_not_guessed(self):
        grouped = section_index.group_report_metrics(["weird__counter.avg"])
        assert grouped["unknown_units"] == {"weird": 1}
        assert section_index.axis_for_metric_name("weird__counter.avg") == ""

    def test_uncatalogued_metrics_are_counted_not_dropped(self):
        grouped = section_index.group_report_metrics(
            [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "lts__t_sectors_srcunit_tex_op_read.sum",
            ],
            catalog=metric_catalog.METRIC_CATALOG,
        )
        assert grouped["total"] == 2
        assert "lts__t_sectors_srcunit_tex_op_read.sum" in grouped["uncatalogued"]
        assert grouped["by_axis"]["memory_bandwidth"]

    def test_display_names_survive_as_undecodable(self):
        grouped = section_index.group_report_metrics(["Duration"])
        assert grouped["undecodable"] == ["Duration"]


class TestCatalogAgainstShippedSections:
    """Ground-truth check when a local Nsight Compute install is present."""

    def test_catalog_names_resolve_or_are_explained(self):
        audit = section_index.audit_catalog_against_sections(
            metric_catalog.METRIC_CATALOG
        )
        if not audit.get("available"):
            pytest.skip("no local Nsight Compute install")
        # Section-backed names must dominate; a large unknown set means drift.
        assert len(audit["section_backed"]) > len(audit["unknown"])
        assert audit["shipped_metric_count"] > 500
