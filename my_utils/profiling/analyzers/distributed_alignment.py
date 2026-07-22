# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from ..metrics.metrics_types import MetricEvent


def _to_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _to_ms(value: float, unit: str) -> Optional[float]:
    unit_l = str(unit or "").lower()
    if unit_l in ("ms", "millisecond", "milliseconds"):
        return value
    if unit_l in ("us", "microsecond", "microseconds"):
        return value / 1000.0
    if unit_l in ("ns", "nanosecond", "nanoseconds"):
        return value / 1_000_000.0
    if unit_l in ("s", "sec", "second", "seconds"):
        return value * 1000.0
    return None


def _event_step_rank(event: MetricEvent) -> Tuple[Optional[int], Optional[str]]:
    step = None
    rank = None
    if "step" in event.tags:
        try:
            step = int(event.tags["step"])
        except Exception:
            step = None
    if "rank" in event.tags:
        rank = str(event.tags["rank"])
    return step, rank


def align_stage_latency(
    events: Iterable[MetricEvent],
    *,
    metric_name_prefix: str = "latency.stage",
) -> Dict[int, Dict[str, Dict[str, List[float]]]]:
    """
    Return aligned latency cube:
      step -> rank -> stage -> [latency_ms...]
    """
    cube: Dict[int, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for event in events:
        if not event.name.startswith(metric_name_prefix):
            continue
        value = _to_float(event.value)
        if value is None:
            continue
        value_ms = _to_ms(value, event.unit)
        if value_ms is None:
            continue
        step, rank = _event_step_rank(event)
        if step is None or rank is None:
            continue
        stage = str(event.tags.get("stage", "unknown"))
        cube[step][rank][stage].append(value_ms)
    return cube


def analyze_rank_skew(
    events: Iterable[MetricEvent],
    *,
    metric_name_prefix: str = "latency.stage",
    min_ranks: int = 2,
    skew_ratio_threshold: float = 1.20,
) -> Dict[str, object]:
    """
    Analyze per-step rank skew from aligned stage latency.
    """
    cube = align_stage_latency(events, metric_name_prefix=metric_name_prefix)
    items: List[Dict[str, object]] = []
    for step, rank_map in cube.items():
        if len(rank_map) < min_ranks:
            continue
        stages = set()
        for stage_map in rank_map.values():
            stages.update(stage_map.keys())
        for stage in sorted(stages):
            by_rank: Dict[str, float] = {}
            for rank, stage_map in rank_map.items():
                values = stage_map.get(stage, [])
                if not values:
                    continue
                by_rank[rank] = statistics.mean(values)
            if len(by_rank) < min_ranks:
                continue
            vals = list(by_rank.values())
            min_v = min(vals)
            max_v = max(vals)
            if min_v <= 0:
                continue
            ratio = max_v / min_v
            if ratio < skew_ratio_threshold:
                continue
            items.append(
                {
                    "step": step,
                    "stage": stage,
                    "rank_count": len(by_rank),
                    "min_ms": min_v,
                    "max_ms": max_v,
                    "mean_ms": statistics.mean(vals),
                    "skew_ratio": ratio,
                    "rank_to_mean_ms": by_rank,
                }
            )
    items.sort(key=lambda x: float(x["skew_ratio"]), reverse=True)
    return {
        "total_aligned_steps": len(cube),
        "skew_items": items,
        "has_skew": bool(items),
        "worst_skew_ratio": float(items[0]["skew_ratio"]) if items else 0.0,
    }
