from __future__ import annotations

import time

from my_utils.profiling import MetricEvent, align_stage_latency, analyze_rank_skew


def test_rank_skew_detection() -> None:
    ts = time.time()
    events = [
        MetricEvent(
            ts,
            "latency.stage",
            10.0,
            "ms",
            provider_id="ut",
            tags={"step": "0", "rank": "0", "stage": "fwd"},
        ),
        MetricEvent(
            ts,
            "latency.stage",
            20.0,
            "ms",
            provider_id="ut",
            tags={"step": "0", "rank": "1", "stage": "fwd"},
        ),
        MetricEvent(
            ts,
            "latency.stage",
            9.0,
            "ms",
            provider_id="ut",
            tags={"step": "1", "rank": "0", "stage": "fwd"},
        ),
        MetricEvent(
            ts,
            "latency.stage",
            22.0,
            "ms",
            provider_id="ut",
            tags={"step": "1", "rank": "1", "stage": "fwd"},
        ),
    ]

    cube = align_stage_latency(events)
    assert 0 in cube and 1 in cube
    assert "0" in cube[0] and "1" in cube[0]

    skew = analyze_rank_skew(events, skew_ratio_threshold=1.3)
    assert skew["has_skew"] is True
    assert skew["worst_skew_ratio"] > 1.3
