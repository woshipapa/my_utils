from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


PROFILE_SCHEMA_VERSION = "1.0"
MetricValue = Union[float, int, str]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_str_dict(payload: Any) -> Dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    result: Dict[str, str] = {}
    for key, value in payload.items():
        if key is None:
            continue
        result[str(key)] = "" if value is None else str(value)
    return result


def _to_any_dict(payload: Any) -> Dict[str, Any]:
    return dict(payload or {}) if isinstance(payload, dict) else {}


@dataclass
class MetricEvent:
    """
    Canonical metric event across all profiling tools.

    Schema v1 adds:
    - `schema_version`
    - `event_id`
    """

    timestamp: float
    name: str
    value: MetricValue
    unit: str = ""
    provider_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    node_id: Optional[str] = None
    event_id: Optional[str] = None
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.name = str(self.name or "")
        self.unit = str(self.unit or "")
        self.provider_id = str(self.provider_id or "")
        self.tags = _to_str_dict(self.tags)
        self.attributes = _to_any_dict(self.attributes)
        if self.parent_id is not None:
            self.parent_id = str(self.parent_id)
        if self.node_id is not None:
            self.node_id = str(self.node_id)
        if self.event_id is not None:
            self.event_id = str(self.event_id)
        self.schema_version = str(self.schema_version or PROFILE_SCHEMA_VERSION)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MetricEvent":
        return cls(
            timestamp=float(payload.get("timestamp", 0.0)),
            name=str(payload.get("name", "")),
            value=payload.get("value", 0.0),
            unit=str(payload.get("unit", "")),
            provider_id=str(payload.get("provider_id", "")),
            tags=_to_str_dict(payload.get("tags", {})),
            attributes=_to_any_dict(payload.get("attributes", {})),
            parent_id=payload.get("parent_id"),
            node_id=payload.get("node_id"),
            event_id=payload.get("event_id"),
            schema_version=str(payload.get("schema_version", PROFILE_SCHEMA_VERSION)),
        )


@dataclass
class Bottleneck:
    name: str
    share_percent: float
    avg_value: float
    unit: str
    sample_count: int


@dataclass
class Finding:
    finding_type: str
    severity: str
    title: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    finding_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Finding":
        return cls(
            finding_type=str(payload.get("finding_type", "")),
            severity=str(payload.get("severity", "info")),
            title=str(payload.get("title", "")),
            description=str(payload.get("description", "")),
            data=_to_any_dict(payload.get("data", {})),
            finding_id=payload.get("finding_id"),
        )


@dataclass
class AnalysisReport:
    generated_at: str = field(default_factory=utc_now_iso)
    summary: Dict[str, Any] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = PROFILE_SCHEMA_VERSION
    overall_score: Optional[float] = None

    def add_finding(self, finding: Optional[Finding]) -> None:
        if finding is None:
            return
        self.findings.append(finding)

    def add_recommendations(self, suggestions: List[str]) -> None:
        for suggestion in suggestions:
            if suggestion and suggestion not in self.recommendations:
                self.recommendations.append(suggestion)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": dict(self.summary),
            "findings": [item.to_dict() for item in self.findings],
            "recommendations": list(self.recommendations),
            "metadata": dict(self.metadata),
            "schema_version": self.schema_version,
            "overall_score": self.overall_score,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AnalysisReport":
        findings_raw = payload.get("findings", []) or []
        findings: List[Finding] = []
        for item in findings_raw:
            if isinstance(item, dict):
                findings.append(Finding.from_dict(item))
        return cls(
            generated_at=str(payload.get("generated_at", utc_now_iso())),
            summary=_to_any_dict(payload.get("summary", {})),
            findings=findings,
            recommendations=[
                str(x) for x in (payload.get("recommendations", []) or [])
            ],
            metadata=_to_any_dict(payload.get("metadata", {})),
            schema_version=str(payload.get("schema_version", PROFILE_SCHEMA_VERSION)),
            overall_score=payload.get("overall_score"),
        )
