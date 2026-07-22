# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

from .analysis_rules import AnalysisRule
from ..metrics.metrics_types import AnalysisReport, Finding, MetricEvent
from .workload_profiles import build_rules_for_workload, resolve_workload_profile


class MetricsAnalyzer:
    """
    Rule-based analyzer for unified metric events.

    Backward compatibility:
    - keeps old constructor thresholds
    - default profile still runs latency+memory basic rules
    """

    def __init__(
        self,
        *,
        bottleneck_share_threshold: float = 0.10,
        cv_threshold: float = 0.50,
        memory_growth_bytes_per_step: float = 10 * 1024 * 1024,
        workload_profile: str = "default",
        enable_advanced_rules: bool = True,
        rules: Optional[List[AnalysisRule]] = None,
    ) -> None:
        self.workload_profile = resolve_workload_profile(workload_profile).name
        self.enable_advanced_rules = bool(enable_advanced_rules)
        self.thresholds: Dict[str, float] = {
            "bottleneck_threshold": float(bottleneck_share_threshold),
            "cv_threshold": float(cv_threshold),
            "memory_growth_bytes_per_step": float(memory_growth_bytes_per_step),
        }
        self._rules = rules or build_rules_for_workload(
            self.workload_profile,
            thresholds=self.thresholds,
            enable_advanced_rules=self.enable_advanced_rules,
        )

    def list_rules(self) -> List[str]:
        return [rule.rule_id for rule in self._rules]

    def analyze(self, events: Iterable[MetricEvent]) -> AnalysisReport:
        event_list = list(events)
        report = AnalysisReport()
        report.metadata["workload_profile"] = self.workload_profile
        report.metadata["rules"] = self.list_rules()
        report.summary = self._build_summary(event_list)

        if not event_list:
            report.add_finding(
                Finding(
                    finding_type="empty",
                    severity="info",
                    title="No Metrics",
                    description="No metric events were collected.",
                )
            )
            report.overall_score = 100.0
            return report

        context: Dict[str, object] = {
            "summary": report.summary,
            "metadata": report.metadata,
            "workload_profile": self.workload_profile,
        }

        findings: List[Finding] = []
        recommendations: List[str] = []
        for rule in self._rules:
            finding = rule.apply(event_list, context)
            if finding is None:
                continue
            findings.append(finding)
            report.add_finding(finding)
            recommendations.extend(rule.recommendations(finding))

        report.add_recommendations(recommendations)
        report.overall_score = self._score(findings)
        report.metadata["severity_counts"] = dict(
            Counter(item.severity.lower() for item in findings)
        )
        return report

    def _build_summary(self, events: List[MetricEvent]) -> Dict[str, object]:
        providers = sorted({event.provider_id for event in events if event.provider_id})
        metric_names = sorted({event.name for event in events if event.name})
        steps: List[int] = []
        ranks: List[str] = []
        for event in events:
            if "step" in event.tags:
                try:
                    steps.append(int(event.tags["step"]))
                except Exception:
                    pass
            if "rank" in event.tags:
                ranks.append(str(event.tags["rank"]))
        summary: Dict[str, object] = {
            "total_events": len(events),
            "providers": providers,
            "metric_count": len(metric_names),
            "workload_profile": self.workload_profile,
            "rank_count": len(set(ranks)),
        }
        if steps:
            summary["step_min"] = min(steps)
            summary["step_max"] = max(steps)
            summary["step_count"] = len(set(steps))
        return summary

    @staticmethod
    def _score(findings: List[Finding]) -> float:
        # Conservative score model: starts at 100 and deducts by severity.
        penalty_by_severity = {
            "critical": 35.0,
            "high": 20.0,
            "warning": 10.0,
            "info": 3.0,
            "low": 5.0,
        }
        score = 100.0
        for item in findings:
            score -= penalty_by_severity.get(str(item.severity).lower(), 5.0)
        return max(0.0, round(score, 2))
