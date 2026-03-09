from __future__ import annotations

import time

from my_utils.profiling import MetricEvent, MetricsAnalyzer


def _evt(name: str, value: float, unit: str, **tags: str) -> MetricEvent:
    return MetricEvent(
        timestamp=time.time(),
        name=name,
        value=value,
        unit=unit,
        provider_id="ut",
        tags=tags,
    )


def test_analyzer_pretrain_profile_findings() -> None:
    events = []

    # Rank-skewed stage latency
    for step in range(3):
        events.append(_evt("latency.stage", 20 + step, "ms", step=str(step), rank="0", stage="forward"))
        events.append(_evt("latency.stage", 40 + step, "ms", step=str(step), rank="1", stage="forward"))

    # Memory growth trend
    for step in range(3):
        events.append(
            _evt(
                "memory.gpu.allocated",
                float(1_000_000_000 + step * 50_000_000),
                "bytes",
                step=str(step),
                rank="0",
            )
        )

    # Communication events
    for step in range(3):
        events.append(_evt("comm.nccl.all_reduce", 8 + step, "ms", step=str(step), rank="0"))
        events.append(_evt("comm.nccl.all_reduce", 18 + step, "ms", step=str(step), rank="1"))

    analyzer = MetricsAnalyzer(workload_profile="pretrain", enable_advanced_rules=True)
    report = analyzer.analyze(events)

    assert report.summary["total_events"] == len(events)
    finding_types = {item.finding_type for item in report.findings}
    assert "bottleneck" in finding_types
    assert "distributed" in finding_types or "communication" in finding_types
    assert report.overall_score is not None

