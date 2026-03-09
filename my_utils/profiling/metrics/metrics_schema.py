from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .metrics_types import MetricEvent, PROFILE_SCHEMA_VERSION


# Canonical unit recommendations by metric prefix.
CANONICAL_UNITS: Dict[str, Tuple[str, ...]] = {
    "latency": ("ns", "us", "ms", "s"),
    "memory": ("bytes", "kb", "mb", "gb"),
    "io": ("bytes", "kb", "mb", "gb", "b/s", "kb/s", "mb/s", "gb/s", "gbps", "gbit/s"),
    "compute": ("flops", "percent", "%", "", "ops/s"),
    "comm": ("bytes", "kb", "mb", "gb", "b/s", "kb/s", "mb/s", "gb/s", "count", ""),
    "calls": ("count", ""),
    "perf": ("", "count"),
    "external": ("", "count", "bytes", "us", "ms"),
}


def _prefix(name: str) -> str:
    parts = (name or "").split(".")
    return parts[0] if parts else ""


def _normalize_unit(unit: str) -> str:
    return str(unit or "").strip().lower()


@dataclass
class EventValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class MetricSchemaValidator:
    """
    Schema validator for MetricEvent schema v1.

    Validation is intentionally non-blocking by default.
    """

    def __init__(
        self,
        *,
        strict: bool = False,
        enforce_known_prefix: bool = False,
        enforce_recommended_units: bool = False,
    ) -> None:
        self.strict = bool(strict)
        self.enforce_known_prefix = bool(enforce_known_prefix)
        self.enforce_recommended_units = bool(enforce_recommended_units)

    def validate(self, event: MetricEvent) -> EventValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if not event.name:
            errors.append("event.name is empty")

        if not isinstance(event.timestamp, float):
            errors.append("event.timestamp must be float")

        metric_prefix = _prefix(event.name)
        if not metric_prefix:
            errors.append("event.name must contain metric prefix")
        elif metric_prefix not in CANONICAL_UNITS:
            message = f"unknown metric prefix: {metric_prefix}"
            if self.enforce_known_prefix:
                errors.append(message)
            else:
                warnings.append(message)

        if event.schema_version != PROFILE_SCHEMA_VERSION:
            warnings.append(
                f"schema_version mismatch: got={event.schema_version} expected={PROFILE_SCHEMA_VERSION}"
            )

        for key, value in event.tags.items():
            if not isinstance(key, str):
                errors.append("tag key must be str")
                break
            if not isinstance(value, str):
                errors.append(f"tag value must be str: {key}")
                break

        if metric_prefix in CANONICAL_UNITS:
            unit = _normalize_unit(event.unit)
            allowed = {_normalize_unit(x) for x in CANONICAL_UNITS[metric_prefix]}
            if unit not in allowed:
                message = f"unit '{event.unit}' not in recommended set for prefix '{metric_prefix}'"
                if self.enforce_recommended_units:
                    errors.append(message)
                else:
                    warnings.append(message)

        return EventValidationResult(is_valid=not errors, errors=errors, warnings=warnings)

    def normalize(self, event: MetricEvent) -> MetricEvent:
        event.unit = _normalize_unit(event.unit)
        event.tags = {str(k): "" if v is None else str(v) for k, v in (event.tags or {}).items()}
        event.provider_id = str(event.provider_id or "")
        event.name = str(event.name or "").strip()
        return event

    def validate_many(self, events: Iterable[MetricEvent]) -> List[EventValidationResult]:
        return [self.validate(event) for event in events]


def normalize_event(event: MetricEvent) -> MetricEvent:
    return MetricSchemaValidator().normalize(event)


def validate_event(
    event: MetricEvent,
    *,
    strict: bool = False,
    enforce_known_prefix: bool = False,
    enforce_recommended_units: bool = False,
) -> EventValidationResult:
    validator = MetricSchemaValidator(
        strict=strict,
        enforce_known_prefix=enforce_known_prefix,
        enforce_recommended_units=enforce_recommended_units,
    )
    return validator.validate(event)

