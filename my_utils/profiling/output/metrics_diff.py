from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..metrics.metrics_types import AnalysisReport


@dataclass
class ReportDiff:
    base_path: str
    target_path: str
    summary_delta: Dict[str, Any] = field(default_factory=dict)
    finding_delta: Dict[str, Any] = field(default_factory=dict)
    recommendation_delta: Dict[str, Any] = field(default_factory=dict)
    score_delta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_path": self.base_path,
            "target_path": self.target_path,
            "summary_delta": self.summary_delta,
            "finding_delta": self.finding_delta,
            "recommendation_delta": self.recommendation_delta,
            "score_delta": self.score_delta,
        }


def _load_report(report_path: str) -> AnalysisReport:
    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AnalysisReport.from_dict(payload)


def compare_reports(base_report_path: str, target_report_path: str) -> ReportDiff:
    base = _load_report(base_report_path)
    target = _load_report(target_report_path)

    summary_delta: Dict[str, Any] = {}
    keys = set(base.summary.keys()) | set(target.summary.keys())
    for key in sorted(keys):
        left = base.summary.get(key)
        right = target.summary.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            summary_delta[key] = {"base": left, "target": right, "delta": right - left}
        elif left != right:
            summary_delta[key] = {"base": left, "target": right}

    base_types = Counter(item.finding_type for item in base.findings)
    target_types = Counter(item.finding_type for item in target.findings)
    finding_delta = {
        "total": {
            "base": len(base.findings),
            "target": len(target.findings),
            "delta": len(target.findings) - len(base.findings),
        },
        "by_type": {
            key: {
                "base": base_types.get(key, 0),
                "target": target_types.get(key, 0),
                "delta": target_types.get(key, 0) - base_types.get(key, 0),
            }
            for key in sorted(set(base_types.keys()) | set(target_types.keys()))
        },
    }

    base_rec = set(base.recommendations)
    target_rec = set(target.recommendations)
    recommendation_delta = {
        "added": sorted(target_rec - base_rec),
        "removed": sorted(base_rec - target_rec),
        "unchanged_count": len(base_rec & target_rec),
    }

    score_delta = None
    if base.overall_score is not None and target.overall_score is not None:
        score_delta = float(target.overall_score) - float(base.overall_score)

    return ReportDiff(
        base_path=base_report_path,
        target_path=target_report_path,
        summary_delta=summary_delta,
        finding_delta=finding_delta,
        recommendation_delta=recommendation_delta,
        score_delta=score_delta,
    )


def write_diff(diff: ReportDiff, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(diff.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(path)
