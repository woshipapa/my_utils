"""Tests for my_utils.profiling.analyzers.nccl_bandwidth."""

from __future__ import annotations


from _synthetic_loader import nccl_bandwidth


class TestStragglerNeedsUsableClocks:
    """Naming a straggler across hosts requires clocks that can support it."""

    def _entries(self, late_ns):
        return [
            {
                "rank": r,
                "collective_seq_id": 7,
                "time_created_ns": 1_000_000_000 + (late_ns if r == 5 else 0),
            }
            for r in range(8)
        ]

    def test_large_spread_names_the_rank(self):
        r = nccl_bandwidth.detect_straggler_from_traces(
            self._entries(40_000_000), collective_seq_id=7, clock_alignment="UTC"
        )
        assert r["conclusive"] is True
        assert r["worst_rank"] == 5

    def test_small_spread_refused_under_utc(self):
        r = nccl_bandwidth.detect_straggler_from_traces(
            self._entries(2_000_000), collective_seq_id=7, clock_alignment="UTC"
        )
        assert r["conclusive"] is False
        assert "clock skew" in r["reason"]

    def test_small_spread_allowed_on_same_host(self):
        r = nccl_bandwidth.detect_straggler_from_traces(
            self._entries(2_000_000), collective_seq_id=7, same_host=True
        )
        assert r["conclusive"] is True
        assert r["worst_rank"] == 5
