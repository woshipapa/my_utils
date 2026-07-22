# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional

from .distributed_alignment import analyze_rank_skew
from ..metrics.metrics_types import Bottleneck, Finding, MetricEvent


def _to_float(value) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _to_ms(value: float, unit: str) -> Optional[float]:
    u = str(unit or "").lower()
    if u in ("ms", "millisecond", "milliseconds"):
        return value
    if u in ("us", "microsecond", "microseconds"):
        return value / 1000.0
    if u in ("ns", "nanosecond", "nanoseconds"):
        return value / 1_000_000.0
    if u in ("s", "sec", "second", "seconds"):
        return value * 1000.0
    return None


def _group_key(event: MetricEvent) -> str:
    for key in ("stage", "module", "op", "kernel", "func", "name"):
        if key in event.tags and event.tags[key]:
            return f"{event.name}:{event.tags[key]}"
    return event.name


def _to_ratio(value) -> Optional[float]:
    numeric = _to_float(value)
    if numeric is None:
        return None
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, float(numeric)))


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    items = sorted(float(v) for v in values)
    if len(items) == 1:
        return items[0]
    qq = max(0.0, min(1.0, float(q)))
    pos = qq * float(len(items) - 1)
    lo = int(pos)
    hi = min(len(items) - 1, lo + 1)
    frac = pos - float(lo)
    return float(items[lo] * (1.0 - frac) + items[hi] * frac)


class AnalysisRule(ABC):
    rule_id = "rule"

    @abstractmethod
    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        raise NotImplementedError

    def recommendations(self, finding: Finding) -> List[str]:
        return []


class LatencyBottleneckRule(AnalysisRule):
    rule_id = "latency_bottleneck"

    def __init__(self, share_threshold: float = 0.10) -> None:
        self.share_threshold = float(share_threshold)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        grouped: Dict[str, List[float]] = defaultdict(list)
        total_ms = 0.0
        for event in events:
            if not event.name.startswith("latency."):
                continue
            numeric = _to_float(event.value)
            if numeric is None:
                continue
            value_ms = _to_ms(numeric, event.unit)
            if value_ms is None:
                continue
            key = _group_key(event)
            grouped[key].append(value_ms)
            total_ms += value_ms
        if total_ms <= 0:
            return None
        bottlenecks: List[Bottleneck] = []
        for key, values in grouped.items():
            sum_ms = sum(values)
            share = sum_ms / total_ms
            if share < self.share_threshold:
                continue
            bottlenecks.append(
                Bottleneck(
                    name=key,
                    share_percent=share * 100.0,
                    avg_value=sum_ms / len(values),
                    unit="ms",
                    sample_count=len(values),
                )
            )
        if not bottlenecks:
            return None
        bottlenecks.sort(key=lambda x: x.share_percent, reverse=True)
        return Finding(
            finding_type="bottleneck",
            severity="high",
            title="Latency Bottlenecks Detected",
            description=f"{len(bottlenecks)} groups exceed share threshold {self.share_threshold:.0%}.",
            data={
                "threshold": self.share_threshold,
                "total_latency_ms": total_ms,
                "bottlenecks": [b.__dict__ for b in bottlenecks[:30]],
            },
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Drill down top bottleneck groups with torch.profiler/Nsight kernel traces.",
            "Align bottleneck stages with rank skew and communication overlap before optimization.",
        ]


class MemoryGrowthRule(AnalysisRule):
    rule_id = "memory_growth"

    def __init__(self, growth_bytes_per_step: float = 10 * 1024 * 1024) -> None:
        self.growth_bytes_per_step = float(growth_bytes_per_step)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        step_series: Dict[str, List[tuple[int, float]]] = defaultdict(list)
        peaks: Dict[str, float] = {}
        for event in events:
            if not (
                event.name.startswith("memory.") or str(event.unit).lower() == "bytes"
            ):
                continue
            value = _to_float(event.value)
            if value is None:
                continue
            key = _group_key(event)
            peaks[key] = max(peaks.get(key, 0.0), value)
            step_text = event.tags.get("step")
            if step_text is None:
                continue
            try:
                step_series[key].append((int(step_text), value))
            except Exception:
                continue
        if not peaks:
            return None
        growth_items = []
        for key, series in step_series.items():
            if len(series) < 2:
                continue
            series.sort(key=lambda item: item[0])
            start_step, start_val = series[0]
            end_step, end_val = series[-1]
            delta_step = max(1, end_step - start_step)
            slope = (end_val - start_val) / delta_step
            if slope > self.growth_bytes_per_step:
                growth_items.append(
                    {
                        "name": key,
                        "start_step": start_step,
                        "end_step": end_step,
                        "growth_bytes_per_step": slope,
                    }
                )
        severity = "warning" if growth_items else "info"
        return Finding(
            finding_type="memory",
            severity=severity,
            title="Memory Growth Analysis",
            description="Detected memory growth trends across steps."
            if growth_items
            else "Memory metrics collected.",
            data={
                "growth_threshold_bytes_per_step": self.growth_bytes_per_step,
                "growth_items": growth_items,
                "peak_bytes": peaks,
            },
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        if not finding.data.get("growth_items"):
            return []
        return [
            "Check tensor/loss/history caches and stale references causing monotonic memory growth.",
            "Correlate memory growth steps with dataloader or optimizer state transitions.",
        ]


class LatencyVarianceRule(AnalysisRule):
    rule_id = "latency_variance"

    def __init__(self, cv_threshold: float = 0.50, min_samples: int = 5) -> None:
        self.cv_threshold = float(cv_threshold)
        self.min_samples = int(min_samples)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for event in events:
            if not event.name.startswith("latency."):
                continue
            value = _to_float(event.value)
            if value is None:
                continue
            value_ms = _to_ms(value, event.unit)
            if value_ms is None:
                continue
            grouped[_group_key(event)].append(value_ms)
        items = []
        for key, values in grouped.items():
            if len(values) < self.min_samples:
                continue
            mean_v = statistics.mean(values)
            if mean_v <= 0:
                continue
            cv = statistics.pstdev(values) / mean_v
            if cv >= self.cv_threshold:
                items.append(
                    {
                        "name": key,
                        "sample_count": len(values),
                        "mean_ms": mean_v,
                        "cv": cv,
                    }
                )
        if not items:
            return None
        items.sort(key=lambda x: x["cv"], reverse=True)
        return Finding(
            finding_type="variance",
            severity="warning",
            title="Latency Variability",
            description=f"{len(items)} groups exceed CV threshold {self.cv_threshold:.2f}.",
            data={"cv_threshold": self.cv_threshold, "items": items[:30]},
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Investigate data-dependent control flow and host synchronization hotspots.",
            "Compare high-variance steps with network/communication telemetry windows.",
        ]


class LatencyOutlierRule(AnalysisRule):
    rule_id = "latency_outlier"

    def __init__(self, z_threshold: float = 3.0, min_samples: int = 8) -> None:
        self.z_threshold = float(z_threshold)
        self.min_samples = int(min_samples)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for event in events:
            if not event.name.startswith("latency."):
                continue
            value = _to_float(event.value)
            if value is None:
                continue
            value_ms = _to_ms(value, event.unit)
            if value_ms is None:
                continue
            grouped[_group_key(event)].append(value_ms)
        outliers = []
        for key, values in grouped.items():
            if len(values) < self.min_samples:
                continue
            mean_v = statistics.mean(values)
            std_v = statistics.pstdev(values)
            if std_v <= 0:
                continue
            cutoff = mean_v + self.z_threshold * std_v
            bad = [v for v in values if v > cutoff]
            if bad:
                outliers.append(
                    {
                        "name": key,
                        "outlier_count": len(bad),
                        "max_ms": max(bad),
                        "threshold_ms": cutoff,
                    }
                )
        if not outliers:
            return None
        outliers.sort(key=lambda x: x["max_ms"], reverse=True)
        return Finding(
            finding_type="anomaly",
            severity="warning",
            title="Latency Outliers",
            description=f"Detected outliers in {len(outliers)} latency groups.",
            data={"z_threshold": self.z_threshold, "items": outliers[:30]},
            finding_id=self.rule_id,
        )


class CommunicationImbalanceRule(AnalysisRule):
    rule_id = "comm_imbalance"

    def __init__(self, imbalance_ratio_threshold: float = 1.25) -> None:
        self.imbalance_ratio_threshold = float(imbalance_ratio_threshold)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        by_rank: Dict[str, List[float]] = defaultdict(list)
        for event in events:
            if not (event.name.startswith("comm.") or "nccl" in event.name.lower()):
                continue
            value = _to_float(event.value)
            if value is None:
                continue
            value_ms = _to_ms(value, event.unit)
            if value_ms is None:
                continue
            rank = event.tags.get("rank")
            if rank is None:
                continue
            by_rank[str(rank)].append(value_ms)
        if len(by_rank) < 2:
            return None
        rank_means = {
            rank: statistics.mean(values) for rank, values in by_rank.items() if values
        }
        if len(rank_means) < 2:
            return None
        vals = list(rank_means.values())
        min_v = min(vals)
        max_v = max(vals)
        if min_v <= 0:
            return None
        ratio = max_v / min_v
        if ratio < self.imbalance_ratio_threshold:
            return None
        return Finding(
            finding_type="communication",
            severity="warning",
            title="Communication Imbalance",
            description=f"Per-rank communication mean latency ratio is {ratio:.2f}.",
            data={
                "imbalance_ratio_threshold": self.imbalance_ratio_threshold,
                "observed_ratio": ratio,
                "rank_mean_latency_ms": rank_means,
            },
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Check tensor partitioning and collective payload balance across ranks.",
            "Inspect straggler ranks with NSYS communication traces and NIC metrics.",
        ]


class PipelineBubbleRule(AnalysisRule):
    rule_id = "pipeline_bubble"

    def __init__(self, bubble_ratio_threshold: float = 0.15) -> None:
        self.bubble_ratio_threshold = float(bubble_ratio_threshold)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        bubble_ms = 0.0
        total_ms = 0.0
        for event in events:
            if not event.name.startswith("latency.stage"):
                continue
            value = _to_float(event.value)
            if value is None:
                continue
            value_ms = _to_ms(value, event.unit)
            if value_ms is None:
                continue
            total_ms += value_ms
            stage = str(event.tags.get("stage", "")).lower()
            if any(token in stage for token in ("idle", "bubble", "wait", "stall")):
                bubble_ms += value_ms
        if total_ms <= 0:
            return None
        ratio = bubble_ms / total_ms
        if ratio < self.bubble_ratio_threshold:
            return None
        return Finding(
            finding_type="pipeline",
            severity="warning",
            title="Pipeline Bubble Detected",
            description=f"Bubble/wait stages consume {ratio:.1%} of stage latency.",
            data={"bubble_ms": bubble_ms, "total_ms": total_ms, "bubble_ratio": ratio},
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Tune microbatch count / pipeline depth and rebalance stage workloads.",
            "Validate overlap between communication and computation in bottleneck windows.",
        ]


class DataloaderStallRule(AnalysisRule):
    rule_id = "dataloader_stall"

    def __init__(self, stall_ratio_threshold: float = 0.20) -> None:
        self.stall_ratio_threshold = float(stall_ratio_threshold)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        data_ms = 0.0
        step_ms = 0.0
        for event in events:
            value = _to_float(event.value)
            if value is None:
                continue
            value_ms = _to_ms(value, event.unit)
            if value_ms is None:
                continue
            name_l = event.name.lower()
            stage_l = str(event.tags.get("stage", "")).lower()
            if "dataloader" in name_l or "dataloader" in stage_l:
                data_ms += value_ms
            if event.name.startswith("latency.step") or stage_l in (
                "step",
                "iteration",
            ):
                step_ms += value_ms
        if data_ms <= 0 or step_ms <= 0:
            return None
        ratio = data_ms / step_ms
        if ratio < self.stall_ratio_threshold:
            return None
        return Finding(
            finding_type="data_pipeline",
            severity="warning",
            title="Dataloader Stall Risk",
            description=f"Dataloader latency contributes {ratio:.1%} of step latency.",
            data={"dataloader_ms": data_ms, "step_ms": step_ms, "ratio": ratio},
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Increase dataloader workers/prefetch and validate CPU decode throughput.",
            "Cache or shard expensive preprocessing for high-latency input stages.",
        ]


class GpuUtilizationThroughputRule(AnalysisRule):
    rule_id = "gpu_utilization"

    def __init__(self, utilization_threshold: float = 0.40) -> None:
        self.utilization_threshold = float(utilization_threshold)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        utils: List[float] = []
        throughput: List[float] = []
        for event in events:
            if event.name in (
                "compute.gpu.sm.active",
                "compute.gpu.sm.utilization",
                "compute.gpu.utilization",
            ):
                value = _to_float(event.value)
                if value is None:
                    continue
                if value > 1.0:
                    value = value / 100.0
                utils.append(max(0.0, min(1.0, value)))
            if event.name in (
                "compute.throughput.samples_per_sec",
                "compute.throughput.tokens_per_sec",
            ):
                value = _to_float(event.value)
                if value is not None:
                    throughput.append(value)
        if not utils:
            return None
        mean_u = statistics.mean(utils)
        if mean_u >= self.utilization_threshold:
            return None
        return Finding(
            finding_type="gpu_efficiency",
            severity="info",
            title="Low GPU Utilization",
            description=f"Average GPU utilization is {mean_u:.1%}.",
            data={
                "utilization_threshold": self.utilization_threshold,
                "mean_utilization": mean_u,
                "throughput_mean": statistics.mean(throughput) if throughput else None,
            },
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Correlate low utilization with dataloader stalls, sync overhead, and launch gaps.",
            "Increase batch size or fuse small kernels when memory headroom allows.",
        ]


class DistributedSkewRule(AnalysisRule):
    rule_id = "distributed_skew"

    def __init__(self, skew_ratio_threshold: float = 1.20) -> None:
        self.skew_ratio_threshold = float(skew_ratio_threshold)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        result = analyze_rank_skew(
            events, skew_ratio_threshold=self.skew_ratio_threshold
        )
        if not result.get("has_skew"):
            return None
        return Finding(
            finding_type="distributed",
            severity="warning",
            title="Cross-rank Stage Skew",
            description=(
                "Detected rank skew in aligned stage latency; "
                f"worst ratio {float(result.get('worst_skew_ratio', 0.0)):.2f}."
            ),
            data=result,
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Investigate straggler ranks and rebalance uneven stage workloads.",
            "Align NCCL timeline with skewed stages to identify communication head-of-line blocking.",
        ]


class CommunicationHealthRule(AnalysisRule):
    rule_id = "comm_health"

    def __init__(
        self,
        *,
        p95_ratio_threshold: float = 2.0,
        rank_imbalance_threshold: float = 1.30,
        min_good_busbw_gbps: float = 20.0,
    ) -> None:
        self.p95_ratio_threshold = float(p95_ratio_threshold)
        self.rank_imbalance_threshold = float(rank_imbalance_threshold)
        self.min_good_busbw_gbps = float(min_good_busbw_gbps)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        latencies_ms: List[float] = []
        busbw_values: List[float] = []
        by_rank_latency: Dict[str, List[float]] = defaultdict(list)
        issue_count = 0

        for event in events:
            name_l = str(event.name or "").lower()
            level = str(event.tags.get("level", "")).lower()
            severity = str(event.tags.get("severity", "")).lower()
            if (
                "nccl" in name_l
                or name_l.startswith("comm.")
                or name_l.startswith("ras.")
            ):
                if any(
                    token in name_l for token in ("error", "fault", "timeout", "issue")
                ):
                    numeric = _to_float(event.value)
                    if numeric is not None and numeric > 0:
                        issue_count += int(max(1.0, numeric))
                if level in ("warn", "warning", "error", "fatal") or severity in (
                    "warn",
                    "warning",
                    "error",
                    "critical",
                    "fatal",
                ):
                    issue_count += 1

            value = _to_float(event.value)
            if value is None:
                continue

            if "busbw" in name_l or "algbw" in name_l:
                busbw_values.append(float(value))

            is_comm_latency = (
                name_l.startswith("comm.")
                or "nccl" in name_l
                or name_l.startswith("latency.comm")
            )
            if not is_comm_latency:
                continue

            value_ms = _to_ms(value, event.unit)
            if value_ms is None:
                continue
            latencies_ms.append(value_ms)
            rank = event.tags.get("rank")
            if rank is not None:
                by_rank_latency[str(rank)].append(value_ms)

        if not latencies_ms and not busbw_values and issue_count <= 0:
            return None

        p50 = _percentile(latencies_ms, 0.50) or 0.0
        p95 = _percentile(latencies_ms, 0.95) or 0.0
        jitter_ratio = (p95 / p50) if p50 > 1e-12 else 0.0

        rank_ratio = 1.0
        rank_mean_latency = {
            rank: statistics.mean(values)
            for rank, values in by_rank_latency.items()
            if values
        }
        if len(rank_mean_latency) >= 2:
            vals = list(rank_mean_latency.values())
            low = min(vals)
            high = max(vals)
            if low > 1e-12:
                rank_ratio = high / low

        median_busbw = _percentile(busbw_values, 0.50)
        health_score = 100.0
        issues: List[str] = []
        if jitter_ratio >= self.p95_ratio_threshold:
            health_score -= 20.0
            issues.append(
                f"Communication tail latency jitter is high (p95/p50={jitter_ratio:.2f}, threshold={self.p95_ratio_threshold:.2f})."
            )
        if rank_ratio >= self.rank_imbalance_threshold:
            health_score -= 20.0
            issues.append(
                f"Cross-rank comm latency imbalance is high (max/min={rank_ratio:.2f}, threshold={self.rank_imbalance_threshold:.2f})."
            )
        if median_busbw is not None and median_busbw < self.min_good_busbw_gbps:
            health_score -= 15.0
            issues.append(
                f"NCCL bandwidth is low (median busbw={median_busbw:.2f} GB/s, expected >= {self.min_good_busbw_gbps:.2f} GB/s)."
            )
        if issue_count > 0:
            health_score -= min(40.0, float(issue_count) * 5.0)
            issues.append(f"Detected {issue_count} NCCL/RAS warning-or-error signals.")

        if not issues:
            return None

        severity = "warning"
        if issue_count > 0 or health_score < 50:
            severity = "high"
        elif health_score < 70:
            severity = "warning"
        else:
            severity = "info"

        return Finding(
            finding_type="communication_health",
            severity=severity,
            title="Communication Health Degradation",
            description=f"Communication health score is {max(0.0, health_score):.1f}/100.",
            data={
                "health_score": max(0.0, health_score),
                "issue_count": int(issue_count),
                "latency_p50_ms": p50,
                "latency_p95_ms": p95,
                "latency_jitter_ratio": jitter_ratio,
                "rank_latency_ratio": rank_ratio,
                "rank_mean_latency_ms": rank_mean_latency,
                "median_busbw_gbps": median_busbw,
                "issues": issues,
            },
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Enable NCCL debug timeline and align p95 windows with rank-level communication traces.",
            "Check topology affinity (NUMA/NVLink/PCIe) and rebalance payload sizes across ranks.",
            "Track RAS/error counters alongside NCCL busbw to separate transport issues from scheduling issues.",
        ]


class RooflineGapRule(AnalysisRule):
    rule_id = "roofline_gap"

    def __init__(self, *, gap_threshold: float = 0.20) -> None:
        self.gap_threshold = float(gap_threshold)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        compute_utils: List[float] = []
        memory_utils: List[float] = []
        total_flops = 0.0
        total_bytes = 0.0

        for event in events:
            name_l = str(event.name or "").lower()
            ratio = _to_ratio(event.value)
            if ratio is not None:
                if any(
                    token in name_l
                    for token in (
                        "sm.active",
                        "sm.util",
                        "tensor.active",
                        "compute.gpu",
                        "warps in flight",
                    )
                ):
                    compute_utils.append(ratio)
                if any(
                    token in name_l
                    for token in (
                        "dram",
                        "hbm",
                        "memory_throughput",
                        "mem_bw",
                        "bandwidth",
                    )
                ):
                    memory_utils.append(ratio)

            value = _to_float(event.value)
            if value is None:
                continue
            if "flops" in name_l:
                total_flops += float(value)
            if str(event.unit).lower() == "bytes" or "bytes" in name_l:
                total_bytes += float(value)

        if not compute_utils and not memory_utils:
            return None

        compute_avg = statistics.mean(compute_utils) if compute_utils else 0.0
        memory_avg = statistics.mean(memory_utils) if memory_utils else 0.0
        gap = max(0.0, memory_avg - compute_avg)
        bottleneck_hint = (
            "memory_bound" if memory_avg >= compute_avg else "compute_bound"
        )

        if total_flops > 0 and total_bytes > 0:
            arithmetic_intensity = total_flops / max(total_bytes, 1.0)
        else:
            arithmetic_intensity = None

        if gap < self.gap_threshold:
            return None

        severity = "warning" if gap < 0.35 else "high"
        return Finding(
            finding_type="roofline",
            severity=severity,
            title="Roofline Headroom Gap",
            description=(
                f"Observed roofline gap is {gap:.1%} "
                f"(compute={compute_avg:.1%}, memory={memory_avg:.1%}, hint={bottleneck_hint})."
            ),
            data={
                "gap_threshold": self.gap_threshold,
                "roofline_gap": gap,
                "compute_utilization": compute_avg,
                "memory_utilization": memory_avg,
                "bottleneck_hint": bottleneck_hint,
                "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
            },
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Correlate compute-utilization and DRAM-throughput time windows to identify memory-bound kernels.",
            "Use operator fusion / data-layout optimization to move points upward in the roofline plane.",
            "For low arithmetic intensity, prioritize memory traffic reduction before tuning raw FLOPS.",
        ]


class CrossLayerConsistencyRule(AnalysisRule):
    rule_id = "cross_layer_consistency"

    def __init__(
        self,
        *,
        min_step_coverage: float = 0.40,
        min_tag_alignment_ratio: float = 0.60,
    ) -> None:
        self.min_step_coverage = float(min_step_coverage)
        self.min_tag_alignment_ratio = float(min_tag_alignment_ratio)

    def apply(
        self, events: List[MetricEvent], context: Dict[str, object]
    ) -> Optional[Finding]:
        step_latency_ms: Dict[str, float] = defaultdict(float)
        component_latency_ms: Dict[str, float] = defaultdict(float)
        step_tagged = 0
        layer_tagged = 0

        has_rank_comm = False
        has_rank_compute = False

        for event in events:
            name_l = str(event.name or "").lower()
            value = _to_float(event.value)
            if value is None:
                continue
            value_ms = _to_ms(value, event.unit)
            step = event.tags.get("step")

            if step is not None:
                step_tagged += 1
            if step and value_ms is not None and event.name.startswith("latency.step"):
                step_latency_ms[str(step)] += value_ms

            is_component = False
            if value_ms is not None and (
                name_l.startswith("latency.kernel")
                or name_l.startswith("comm.")
                or "nccl" in name_l
                or "sync" in name_l
                or "dataloader" in name_l
                or "python" in name_l
            ):
                is_component = True
                if step:
                    component_latency_ms[str(step)] += value_ms

            if is_component and step is not None:
                layer_tagged += 1

            if ("comm." in name_l or "nccl" in name_l) and "rank" in event.tags:
                has_rank_comm = True
            if ("kernel" in name_l or "compute." in name_l) and "rank" in event.tags:
                has_rank_compute = True

        coverage_items: List[Dict[str, float]] = []
        for step, step_ms in step_latency_ms.items():
            comp_ms = component_latency_ms.get(step, 0.0)
            coverage = comp_ms / step_ms if step_ms > 1e-12 else 0.0
            coverage_items.append(
                {
                    "step": float(step) if str(step).isdigit() else step,
                    "step_ms": step_ms,
                    "component_ms": comp_ms,
                    "coverage": coverage,
                }
            )

        issues: List[str] = []
        if coverage_items:
            mean_cov = statistics.mean(
                float(item["coverage"]) for item in coverage_items
            )
            if mean_cov < self.min_step_coverage:
                issues.append(
                    f"Cross-layer step coverage is low ({mean_cov:.1%}); step latency is insufficiently explained by component layers."
                )
        else:
            mean_cov = None

        tag_alignment_ratio = (layer_tagged / step_tagged) if step_tagged > 0 else 1.0
        if step_tagged > 0 and tag_alignment_ratio < self.min_tag_alignment_ratio:
            issues.append(
                f"Only {tag_alignment_ratio:.1%} of step-tagged events are aligned to component layers; missing step tags reduce correlation quality."
            )
        if has_rank_comm and not has_rank_compute:
            issues.append(
                "Communication events have rank tags, but compute events do not; cross-rank attribution is incomplete."
            )

        if not issues:
            return None

        severity = "warning" if len(issues) > 1 else "info"
        return Finding(
            finding_type="consistency",
            severity=severity,
            title="Cross-Layer Consistency Gaps",
            description=f"Detected {len(issues)} observability-consistency gap(s) across step/comm/compute/cpu layers.",
            data={
                "min_step_coverage": self.min_step_coverage,
                "min_tag_alignment_ratio": self.min_tag_alignment_ratio,
                "mean_step_coverage": mean_cov,
                "tag_alignment_ratio": tag_alignment_ratio,
                "coverage_by_step": coverage_items[:40],
                "issues": issues,
            },
            finding_id=self.rule_id,
        )

    def recommendations(self, finding: Finding) -> List[str]:
        return [
            "Unify step/rank tags across GPU kernels, NCCL communication, and CPU synchronization events.",
            "Ensure each layer emits comparable latency metrics in the same time units for cross-layer joins.",
            "Add missing observability at layer boundaries (dispatch, comm, kernel, and runtime stalls).",
        ]
