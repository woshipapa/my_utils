"""Tests for my_utils.profiling.analyzers.axes."""

from __future__ import annotations


from pathlib import Path


from _synthetic_loader import axes, metric_catalog, ncu_diagnostics


class TestCoverageKeysAreReal:
    """Coverage tables must reference catalog keys that exist.

    `MetricView.get` falls through to a raw ncu-metric lookup for an unknown
    key and returns None, so a typo does not raise - it silently reports the
    analysis as "skipped, metrics absent" on every report forever. Seven keys in
    `_ANALYSIS_REQUIREMENTS` were wrong this way, which made stalls, coalescing,
    divergence and spilling permanently look uncollected even on --set full.
    """

    def test_analysis_requirements_keys_exist(self):
        catalog = set(metric_catalog.METRIC_CATALOG)
        bad = [
            (analysis, key)
            for analysis, (
                keys,
                _section,
            ) in ncu_diagnostics._ANALYSIS_REQUIREMENTS.items()
            for key in keys
            if key not in catalog
        ]
        assert not bad, f"_ANALYSIS_REQUIREMENTS references non-catalog keys: {bad}"

    def test_axis_metric_groups_keys_exist(self):
        catalog = set(metric_catalog.METRIC_CATALOG)
        bad = [
            (axis.axis_id, key)
            for axis in axes.AXES
            for group in axis.metric_groups
            for key in group
            if key not in catalog
        ]
        assert not bad, f"axes.AXES references non-catalog keys: {bad}"

    def test_every_emitted_category_maps_to_an_axis(self):
        """A finding whose category maps to no axis vanishes from coverage.

        Literal `category="..."` only. See the f-string test below for why that
        is not sufficient on its own.
        """
        import re

        source = (Path(ncu_diagnostics.__file__)).read_text()
        emitted = set(re.findall(r'category="([a-z0-9_]+)"', source))
        unmapped = sorted(c for c in emitted if not axes.axis_for_category(c))
        assert not unmapped, f"finding categories with no axis: {unmapped}"

    def test_fstring_categories_map_to_an_axis(self):
        """Categories built with an f-string are invisible to a literal grep.

        This is how 20 of 33 categories -- including 16 of the 19 stall reasons,
        which is most of what a latency-bound kernel reports -- were unmapped
        while the literal-string test above passed. An unmapped category is
        dropped from `by_axis`, so a kernel whose headline finding is
        `stall_long_scoreboard` reported the stall axis with finding_count=0 and
        a "collect WarpStateStats" remedy for an axis that had just fired.

        Every f-string site in ncu_diagnostics.py is expanded here by hand,
        because the interpolated values come from runtime data a static reader
        cannot see. `test_fstring_category_sites_are_all_covered` fails if a new
        site appears.
        """
        emitted = (
            [f"stall_{key}" for key in metric_catalog.STALL_REASONS]
            + [
                f"occupancy_limited_{b}"
                for b in ("registers", "shared_mem", "blocks", "warps", "barriers")
            ]
            # The interpolated labels are "load"/"store" here, not "ld"/"st";
            # the first version of this test asserted values the code never
            # emits, so it passed while proving nothing about the real ones.
            + [f"uncoalesced_global_{op}" for op in ("load", "store")]
            + [f"sparse_global_{op}" for op in ("load", "store")]
            + [f"shared_bank_conflicts_{op}" for op in ("ld", "st")]
            + [f"{unit}_load_imbalance" for unit in ("sm", "l2", "dram")]
        )
        unmapped = sorted(c for c in emitted if not axes.axis_for_category(c))
        assert not unmapped, f"f-string categories with no axis: {unmapped}"

    def test_fstring_category_sites_are_all_covered(self):
        """Fail when a new f-string category site is added upstream."""
        import re

        source = (Path(ncu_diagnostics.__file__)).read_text()
        sites = set(re.findall(r'category=f"([^"]+)"', source))
        known = {
            "stall_{row['key']}",
            "occupancy_limited_{binding}",
            "uncoalesced_global_{label}",
            "sparse_global_{label}",
            "shared_bank_conflicts_{op}",
            "{unit}_load_imbalance",
        }
        new = sorted(sites - known)
        assert not new, (
            "new f-string category site(s) not expanded in "
            f"test_fstring_categories_map_to_an_axis: {new}"
        )

    def test_axis_lookup_does_not_match_by_accident(self):
        """A wrong axis is worse than none: it makes a gap look covered.

        The substring fallback used to accept a match in either direction, so
        `stall_selected` resolved via `stalls` purely because "selected" starts
        with an s -- and `stall_long_scoreboard`, which does not, resolved to
        nothing at all.
        """
        assert axes.axis_for_category("stall_long_scoreboard") == "stall"
        assert axes.axis_for_category("stall_selected") == "stall"
        # A category that genuinely belongs nowhere must still return "".
        assert axes.axis_for_category("zzz_unrelated_thing") == ""
        assert axes.axis_for_category("") == ""


class TestAxisCoverage:
    def test_unexamined_axis_is_not_reported_as_clean(self):
        result = axes.axis_coverage([], metric_present=lambda key: False)
        stall = next(a for a in result["axes"] if a["axis"] == "stall")
        assert stall["examined"] is False
        assert stall["reason_not_examined"]
        assert stall["remedy"], "an unexamined axis must say how to examine it"

    def test_findings_mark_their_axis_examined(self):
        result = axes.axis_coverage(
            [{"category": "uncoalesced_global_access"}],
            metric_present=lambda key: False,
        )
        mem = next(a for a in result["axes"] if a["axis"] == "memory_bandwidth")
        assert mem["examined"] is True and mem["finding_count"] == 1

    def test_present_metrics_mark_axis_examined_without_findings(self):
        """A clean axis and an unchecked axis must be distinguishable."""
        present = {"achieved_occupancy", "theoretical_occupancy"}
        result = axes.axis_coverage([], metric_present=present)
        sched = next(a for a in result["axes"] if a["axis"] == "scheduler")
        assert sched["examined"] is True and sched["finding_count"] == 0

    def test_unmapped_categories_are_surfaced_not_dropped(self):
        result = axes.axis_coverage([{"category": "zzz_not_a_real_category"}])
        assert "zzz_not_a_real_category" in result["unmapped_categories"]
