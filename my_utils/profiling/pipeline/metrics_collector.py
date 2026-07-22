# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..analyzers.metrics_analyzer import MetricsAnalyzer
from ..metrics.metrics_provider import MetricsProvider
from ..output.metrics_report import MetricsReportRenderer
from ..metrics.metrics_schema import MetricSchemaValidator
from ..metrics.metrics_store import MetricsStore
from ..output.metrics_trace import ChromeTraceExportConfig, write_chrome_trace
from ..metrics.metrics_types import AnalysisReport, MetricEvent
from ..metrics.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    MetricsProviderRegistry,
    ProviderSpec,
)


class MetricsCollector:
    """Unified orchestrator for providers, storage, analysis, and reports."""

    def __init__(
        self,
        output_dir: str = "metrics",
        *,
        store: Optional[MetricsStore] = None,
        analyzer: Optional[MetricsAnalyzer] = None,
        renderer: Optional[MetricsReportRenderer] = None,
        validator: Optional[MetricSchemaValidator] = None,
        validate_events: bool = False,
        drop_invalid_events: bool = False,
        provider_registry: Optional[MetricsProviderRegistry] = None,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = bool(enabled)
        self._providers: Dict[str, MetricsProvider] = {}
        self._store = store or MetricsStore(output_dir=str(self.output_dir))
        self._analyzer = analyzer or MetricsAnalyzer()
        self._renderer = renderer or MetricsReportRenderer()
        self._validator = validator or MetricSchemaValidator()
        self._validate_events = bool(validate_events)
        self._drop_invalid_events = bool(drop_invalid_events)
        self._provider_registry = provider_registry or DEFAULT_PROVIDER_REGISTRY
        self._bootstrap_warnings: List[str] = []
        self._validation_stats: Dict[str, int] = {
            "validated_events": 0,
            "invalid_events": 0,
            "dropped_invalid_events": 0,
            "validation_warnings": 0,
        }

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def set_validation(
        self,
        *,
        validate_events: Optional[bool] = None,
        drop_invalid_events: Optional[bool] = None,
    ) -> None:
        if validate_events is not None:
            self._validate_events = bool(validate_events)
        if drop_invalid_events is not None:
            self._drop_invalid_events = bool(drop_invalid_events)

    def register_provider(self, provider: MetricsProvider) -> None:
        self._providers[provider.provider_id] = provider

    def unregister_provider(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)

    def list_providers(self) -> List[str]:
        return sorted(self._providers.keys())

    def provider_capabilities(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for provider_id, provider in self._providers.items():
            try:
                caps = provider.capabilities()
                result[provider_id] = {
                    "provider_type": caps.provider_type,
                    "source_mode": caps.source_mode,
                    "metric_prefixes": list(caps.metric_prefixes),
                    "dimensions": list(caps.dimensions),
                    "supports_incremental": bool(caps.supports_incremental),
                    "supports_step_scope": bool(caps.supports_step_scope),
                    "supports_rank_scope": bool(caps.supports_rank_scope),
                    "notes": caps.notes,
                }
            except Exception:
                result[provider_id] = {"provider_type": "unknown"}
        return result

    def start(self) -> None:
        for provider in self._providers.values():
            try:
                provider.start_collection()
            except Exception:
                continue

    def stop(self) -> None:
        for provider in self._providers.values():
            try:
                provider.stop_collection()
            except Exception:
                continue

    def collect(
        self, *, step: Optional[int] = None, tags: Optional[Dict[str, str]] = None
    ) -> int:
        if not self._enabled:
            return 0

        all_events: List[MetricEvent] = []
        extra_tags = dict(tags or {})
        if step is not None:
            extra_tags["step"] = str(step)

        for provider in self._providers.values():
            try:
                if not provider.is_enabled():
                    continue
                events = provider.get_metrics()
            except Exception:
                continue

            for event in events:
                try:
                    if extra_tags:
                        event.tags.update(extra_tags)
                    if not event.provider_id:
                        event.provider_id = provider.provider_id
                    self._validator.normalize(event)
                    if self._validate_events:
                        verdict = self._validator.validate(event)
                        self._validation_stats["validated_events"] += 1
                        self._validation_stats["validation_warnings"] += len(
                            verdict.warnings
                        )
                        if not verdict.is_valid:
                            self._validation_stats["invalid_events"] += 1
                            if self._drop_invalid_events:
                                self._validation_stats["dropped_invalid_events"] += 1
                                continue
                    all_events.append(event)
                except Exception:
                    continue

        return self._store.write_events(all_events)

    def get_events(self) -> List[MetricEvent]:
        in_memory = self._store.read_all_events(prefer_disk=False)
        if in_memory and not self._store.has_buffer_overflowed():
            return in_memory
        return self._store.read_all_events(prefer_disk=True)

    def analyze(self, events: Optional[Iterable[MetricEvent]] = None) -> AnalysisReport:
        report = self._analyzer.analyze(
            events if events is not None else self.get_events()
        )
        report.metadata.setdefault("collector", {})
        report.metadata["collector"].update(
            {
                "providers": self.list_providers(),
                "provider_capabilities": self.provider_capabilities(),
                "validation_stats": dict(self._validation_stats),
                "bootstrap_warnings": list(self._bootstrap_warnings),
            }
        )
        return report

    def export_report(
        self,
        *,
        fmt: str = "json",
        output_path: Optional[str] = None,
        report: Optional[AnalysisReport] = None,
    ) -> str:
        report_obj = report or self.analyze()
        if output_path is None:
            suffix = "md" if fmt in ("md", "markdown") else fmt
            output_path = str(self.output_dir / f"analysis_report.{suffix}")
        return self._renderer.write(report_obj, output_path=output_path, fmt=fmt)

    def export_chrome_trace(
        self,
        *,
        output_path: Optional[str] = None,
        events: Optional[Iterable[MetricEvent]] = None,
        config: Optional[ChromeTraceExportConfig] = None,
    ) -> str:
        target_events = list(events) if events is not None else self.get_events()
        if output_path is None:
            output_path = str(self.output_dir / "metrics_trace.json")
        return write_chrome_trace(target_events, output_path=output_path, config=config)

    def load_events_file(self, events_path: str) -> int:
        events = MetricsStore.read_events_file(events_path)
        self._store.clear(clear_disk=False)
        return self._store.write_events(events)

    def get_bootstrap_warnings(self) -> List[str]:
        return list(self._bootstrap_warnings)

    def register_providers_from_specs(
        self,
        provider_specs: Iterable[Mapping[str, Any]],
        *,
        provider_context: Optional[Mapping[str, Any]] = None,
        ignore_errors: bool = False,
    ) -> None:
        for item in provider_specs:
            spec = ProviderSpec.from_dict(item)
            try:
                provider = self._provider_registry.create(
                    spec, context=provider_context or {}
                )
                self.register_provider(provider)
            except Exception as exc:
                message = f"Failed to register provider '{spec.provider_id}' ({spec.provider_type}): {exc}"
                self._bootstrap_warnings.append(message)
                if not ignore_errors:
                    raise RuntimeError(message) from exc

    @classmethod
    def from_config(
        cls,
        config_path: str,
        *,
        provider_context: Optional[Mapping[str, Any]] = None,
        provider_registry: Optional[MetricsProviderRegistry] = None,
    ) -> "MetricsCollector":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("PyYAML is required for YAML config files.") from exc
            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text or "{}")

        collector_cfg = payload.get("collector", {}) or {}
        output_dir = collector_cfg.get("output_dir", "metrics")
        enabled = bool(collector_cfg.get("enabled", True))
        validate_events = bool(collector_cfg.get("validate_events", False))
        drop_invalid_events = bool(collector_cfg.get("drop_invalid_events", False))
        analyzer_cfg = payload.get("analysis", {}) or {}
        schema_cfg = payload.get("schema", {}) or {}

        analyzer = MetricsAnalyzer(
            bottleneck_share_threshold=float(
                analyzer_cfg.get("bottleneck_threshold", 0.10)
            ),
            cv_threshold=float(analyzer_cfg.get("cv_threshold", 0.50)),
            memory_growth_bytes_per_step=float(
                analyzer_cfg.get("memory_growth_bytes_per_step", 10 * 1024 * 1024)
            ),
            workload_profile=str(analyzer_cfg.get("workload_profile", "default")),
            enable_advanced_rules=bool(analyzer_cfg.get("enable_advanced_rules", True)),
        )
        validator = MetricSchemaValidator(
            strict=bool(schema_cfg.get("strict", False)),
            enforce_known_prefix=bool(schema_cfg.get("enforce_known_prefix", False)),
            enforce_recommended_units=bool(
                schema_cfg.get("enforce_recommended_units", False)
            ),
        )

        collector = cls(
            output_dir=output_dir,
            analyzer=analyzer,
            validator=validator,
            validate_events=validate_events,
            drop_invalid_events=drop_invalid_events,
            provider_registry=provider_registry or DEFAULT_PROVIDER_REGISTRY,
            enabled=enabled,
        )

        provider_items = payload.get("providers", []) or []
        collector.register_providers_from_specs(
            provider_items,
            provider_context=provider_context or {},
            ignore_errors=bool(collector_cfg.get("ignore_provider_errors", False)),
        )
        return collector
