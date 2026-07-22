# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.analyzers.triage."""

from __future__ import annotations


import pytest


from _synthetic_loader import triage


def test_merge_intervals_coalesces_and_sorts():
    assert triage.merge_intervals([(30, 40), (0, 10), (5, 20)]) == [
        (0.0, 20.0),
        (30.0, 40.0),
    ]
    assert triage.merge_intervals([]) == []
    # Zero-length and inverted intervals are dropped rather than corrupting the union.
    assert triage.merge_intervals([(5, 5), (10, 3)]) == []


def test_union_counts_overlapping_time_once():
    assert triage.interval_union_ns([(0, 10), (5, 20)]) == 20.0


def test_overlap_is_the_intersection_of_two_unions():
    assert triage.interval_overlap_ns([(0, 10)], [(5, 20)]) == 5.0
    assert triage.interval_overlap_ns([(0, 10)], [(20, 30)]) == 0.0
    assert triage.interval_overlap_ns([(0, 10), (20, 30)], [(5, 25)]) == 10.0
    assert triage.interval_overlap_ns([], [(0, 10)]) == 0.0


MS = 1_000_000


def test_idle_gpu_with_launch_overhead_is_host_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(70 * MS, 200 * MS)],
        launch_api_ns=25 * MS,
        kernel_durations_ns=[2 * MS] * 60,
    )
    assert verdict.verdict == "host_bound"
    assert "gpu_idle_ratio" in [s.key for s in verdict.signals if s.crossed]


def test_one_signal_alone_does_not_declare_host_bound():
    """NVIDIA's rule is that two or more host signals must cross."""
    verdict = triage.triage_step(
        wall_ns=100 * MS,
        compute_intervals=[(35 * MS, 100 * MS)],  # 35% idle, but launch time is tiny
        launch_api_ns=1 * MS,
        kernel_durations_ns=[10 * MS] * 6,
    )
    assert verdict.verdict != "host_bound"


def test_exposed_communication_is_comm_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(0, 120 * MS)],
        comm_intervals=[(118 * MS, 195 * MS)],
        launch_api_ns=5 * MS,
        kernel_durations_ns=[3 * MS] * 40,
    )
    assert verdict.verdict == "communication_bound"
    assert verdict.breakdown["overlap_pct_of_comm"] < 10


def test_same_comm_volume_hidden_under_compute_is_not_comm_bound():
    """The discriminator is exposure, not volume."""
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(0, 190 * MS)],
        comm_intervals=[(20 * MS, 95 * MS)],
        launch_api_ns=5 * MS,
        kernel_durations_ns=[3 * MS] * 40,
    )
    assert verdict.verdict == "kernel_bound"
    assert verdict.breakdown["overlap_pct_of_comm"] == pytest.approx(100.0)
    assert verdict.breakdown["exposed_comm_ns"] == 0


def test_transfers_on_the_critical_path_are_transfer_bound():
    verdict = triage.triage_step(
        wall_ns=100 * MS,
        compute_intervals=[(30 * MS, 95 * MS)],
        memcpy_intervals=[(0, 28 * MS)],
        launch_api_ns=3 * MS,
        kernel_durations_ns=[4 * MS] * 16,
    )
    assert verdict.verdict == "transfer_bound"


def test_busy_gpu_with_large_kernels_is_kernel_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(2 * MS, 198 * MS)],
        launch_api_ns=4 * MS,
        kernel_durations_ns=[8 * MS] * 24,
    )
    assert verdict.verdict == "kernel_bound"
    assert verdict.next_steps


def test_cuda_graph_mode_tightens_the_idle_gate():
    kwargs = dict(
        wall_ns=100 * MS,
        compute_intervals=[(18 * MS, 100 * MS)],
        launch_api_ns=1 * MS,
        kernel_durations_ns=[5 * MS] * 16,
    )
    default = triage.triage_step(**kwargs)
    graphs = triage.triage_step(
        **kwargs, thresholds=triage.TriageThresholds(cuda_graphs=True)
    )

    def idle(v):
        return next(s for s in v.signals if s.key == "gpu_idle_ratio")

    assert idle(default).crossed is False
    assert idle(graphs).crossed is True


def test_full_launch_queue_is_reported_as_gpu_bound_not_host_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(2 * MS, 198 * MS)],
        launch_api_ns=4 * MS,
        kernel_durations_ns=[8 * MS] * 24,
        max_queue_length=1024,
    )
    assert any("queue" in note.lower() for note in verdict.secondary)


def test_steady_state_allocations_surface_as_a_secondary_note():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(2 * MS, 195 * MS)],
        launch_api_ns=4 * MS,
        kernel_durations_ns=[8 * MS] * 24,
        steady_state_allocs=37,
    )
    assert any("cudaMalloc" in note for note in verdict.secondary)


def test_verdict_serialises_for_reports():
    verdict = triage.triage_step(wall_ns=10 * MS, compute_intervals=[(0, 9 * MS)])
    payload = verdict.to_dict()
    assert set(payload) >= {
        "verdict",
        "confidence",
        "summary",
        "signals",
        "breakdown",
        "next_steps",
    }
    assert isinstance(payload["signals"], list)


def test_empty_trace_degrades_to_low_confidence_rather_than_crashing():
    verdict = triage.triage_step(wall_ns=0)
    assert verdict.verdict
    assert verdict.confidence == "low"


class TestTriageRefusesMissingData:
    """Absent GPU intervals and an idle GPU are indistinguishable; say so."""

    def test_no_intervals_yields_no_verdict(self):
        tri = triage
        verdict = tri.triage_step(
            wall_ns=1e9,
            launch_api_ns=8e8,
            sync_api_ns=5e8,
            kernel_durations_ns=[4e3] * 50,
        )
        assert verdict.verdict == "undetermined"
        assert verdict.confidence == "low"
        assert "could not be measured" in verdict.summary

    def test_real_intervals_still_reach_host_bound(self):
        tri = triage
        verdict = tri.triage_step(
            wall_ns=1e9,
            compute_intervals=[(0, 5e7)],
            launch_api_ns=8e8,
            sync_api_ns=5e8,
            kernel_durations_ns=[4e3] * 50,
        )
        assert verdict.verdict == "host_bound"
