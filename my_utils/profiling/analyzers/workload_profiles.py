from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .analysis_rules import (
    AnalysisRule,
    CommunicationHealthRule,
    CommunicationImbalanceRule,
    CrossLayerConsistencyRule,
    DataloaderStallRule,
    DistributedSkewRule,
    GpuUtilizationThroughputRule,
    LatencyBottleneckRule,
    LatencyOutlierRule,
    LatencyVarianceRule,
    MemoryGrowthRule,
    PipelineBubbleRule,
    RooflineGapRule,
)


@dataclass
class WorkloadProfile:
    name: str
    description: str
    rule_ids: List[str]
    kpi_metrics: List[str] = field(default_factory=list)
    threshold_overrides: Dict[str, float] = field(default_factory=dict)


WORKLOAD_PROFILES: Dict[str, WorkloadProfile] = {
    "default": WorkloadProfile(
        name="default",
        description="Generic workload profile for mixed training/inference pipelines.",
        rule_ids=[
            "latency_bottleneck",
            "memory_growth",
            "latency_variance",
            "latency_outlier",
        ],
        kpi_metrics=["latency.stage", "memory.gpu.allocated"],
    ),
    "pretrain": WorkloadProfile(
        name="pretrain",
        description="Large-scale distributed pretraining profile.",
        rule_ids=[
            "latency_bottleneck",
            "memory_growth",
            "latency_variance",
            "latency_outlier",
            "comm_imbalance",
            "comm_health",
            "pipeline_bubble",
            "distributed_skew",
            "gpu_utilization",
            "roofline_gap",
            "cross_layer_consistency",
        ],
        kpi_metrics=[
            "latency.step",
            "latency.stage",
            "comm.nic",
            "compute.gpu.sm.active",
            "memory.gpu.allocated",
        ],
    ),
    "finetune": WorkloadProfile(
        name="finetune",
        description="Finetuning profile with emphasis on stability and throughput balance.",
        rule_ids=[
            "latency_bottleneck",
            "memory_growth",
            "latency_variance",
            "dataloader_stall",
            "gpu_utilization",
        ],
        kpi_metrics=[
            "latency.step",
            "latency.stage",
            "memory.gpu.allocated",
            "compute.throughput.samples_per_sec",
        ],
    ),
    "inference": WorkloadProfile(
        name="inference",
        description="Inference/serving profile focused on tail latency and utilization.",
        rule_ids=[
            "latency_bottleneck",
            "latency_variance",
            "latency_outlier",
            "gpu_utilization",
        ],
        kpi_metrics=["latency.request", "latency.stage", "compute.gpu.sm.utilization"],
    ),
    "data_pipeline": WorkloadProfile(
        name="data_pipeline",
        description="Data ingestion and preprocessing profile.",
        rule_ids=["dataloader_stall", "latency_variance", "memory_growth"],
        kpi_metrics=["latency.dataloader", "latency.stage", "io.read.bytes"],
    ),
}


def _build_rule(rule_id: str, thresholds: Mapping[str, float]) -> AnalysisRule:
    if rule_id == "latency_bottleneck":
        return LatencyBottleneckRule(
            share_threshold=float(thresholds.get("bottleneck_threshold", 0.10))
        )
    if rule_id == "memory_growth":
        return MemoryGrowthRule(
            growth_bytes_per_step=float(
                thresholds.get("memory_growth_bytes_per_step", 10 * 1024 * 1024)
            )
        )
    if rule_id == "latency_variance":
        return LatencyVarianceRule(
            cv_threshold=float(thresholds.get("cv_threshold", 0.50))
        )
    if rule_id == "latency_outlier":
        return LatencyOutlierRule(
            z_threshold=float(thresholds.get("outlier_z_threshold", 3.0))
        )
    if rule_id == "comm_imbalance":
        return CommunicationImbalanceRule(
            imbalance_ratio_threshold=float(
                thresholds.get("comm_imbalance_ratio_threshold", 1.25)
            )
        )
    if rule_id == "pipeline_bubble":
        return PipelineBubbleRule(
            bubble_ratio_threshold=float(
                thresholds.get("pipeline_bubble_ratio_threshold", 0.15)
            )
        )
    if rule_id == "dataloader_stall":
        return DataloaderStallRule(
            stall_ratio_threshold=float(
                thresholds.get("dataloader_stall_ratio_threshold", 0.20)
            )
        )
    if rule_id == "gpu_utilization":
        return GpuUtilizationThroughputRule(
            utilization_threshold=float(
                thresholds.get("gpu_utilization_threshold", 0.40)
            )
        )
    if rule_id == "distributed_skew":
        return DistributedSkewRule(
            skew_ratio_threshold=float(
                thresholds.get("distributed_skew_ratio_threshold", 1.20)
            )
        )
    if rule_id == "comm_health":
        return CommunicationHealthRule(
            p95_ratio_threshold=float(
                thresholds.get("comm_health_p95_ratio_threshold", 2.0)
            ),
            rank_imbalance_threshold=float(
                thresholds.get("comm_health_rank_imbalance_threshold", 1.30)
            ),
            min_good_busbw_gbps=float(
                thresholds.get("comm_health_min_busbw_gbps", 20.0)
            ),
        )
    if rule_id == "roofline_gap":
        return RooflineGapRule(
            gap_threshold=float(thresholds.get("roofline_gap_threshold", 0.20))
        )
    if rule_id == "cross_layer_consistency":
        return CrossLayerConsistencyRule(
            min_step_coverage=float(
                thresholds.get("cross_layer_min_step_coverage", 0.40)
            ),
            min_tag_alignment_ratio=float(
                thresholds.get("cross_layer_min_tag_alignment_ratio", 0.60)
            ),
        )
    raise KeyError(f"Unknown rule id: {rule_id}")


def list_workload_profiles() -> List[str]:
    return sorted(WORKLOAD_PROFILES.keys())


def resolve_workload_profile(name: str) -> WorkloadProfile:
    key = str(name or "default").strip().lower()
    if key not in WORKLOAD_PROFILES:
        return WORKLOAD_PROFILES["default"]
    return WORKLOAD_PROFILES[key]


def build_rules_for_workload(
    workload_profile: str,
    *,
    thresholds: Optional[Mapping[str, float]] = None,
    enable_advanced_rules: bool = True,
) -> List[AnalysisRule]:
    profile = resolve_workload_profile(workload_profile)
    merged_thresholds: Dict[str, float] = dict(profile.threshold_overrides)
    merged_thresholds.update(dict(thresholds or {}))
    rules: List[AnalysisRule] = []
    for rule_id in profile.rule_ids:
        if not enable_advanced_rules and rule_id in (
            "comm_imbalance",
            "pipeline_bubble",
            "dataloader_stall",
            "gpu_utilization",
            "distributed_skew",
            "comm_health",
            "roofline_gap",
            "cross_layer_consistency",
        ):
            continue
        rules.append(_build_rule(rule_id, merged_thresholds))
    return rules
