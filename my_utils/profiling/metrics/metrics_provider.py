from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable

from .metrics_types import MetricEvent


@dataclass
class ProviderCapabilities:
    provider_type: str
    source_mode: str = "online"  # online | offline | hybrid
    metric_prefixes: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    supports_incremental: bool = True
    supports_step_scope: bool = False
    supports_rank_scope: bool = False
    notes: str = ""


@runtime_checkable
class MetricsProvider(Protocol):
    """Provider interface for all profiling data sources."""

    provider_id: str

    def get_metrics(self) -> List[MetricEvent]: ...

    def start_collection(self) -> None: ...

    def stop_collection(self) -> None: ...

    def is_enabled(self) -> bool: ...

    def capabilities(self) -> ProviderCapabilities: ...


class BaseMetricsProvider:
    """Default helper for providers with simple on/off behavior."""

    provider_id: str = "base"
    _capabilities = ProviderCapabilities(provider_type="base")

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)

    def start_collection(self) -> None:
        return

    def stop_collection(self) -> None:
        return

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities
