from .metrics_provider import BaseMetricsProvider, MetricsProvider, ProviderCapabilities
from .metrics_providers import (
    CProfileStatsProvider,
    DcgmCsvMetricsProvider,
    ModuleProfilerMetricsProvider,
    MyTimerMetricsProvider,
    NcclLogMetricsProvider,
    NcuCsvMetricsProvider,
    PerfStatTextProvider,
    RasJsonMetricsProvider,
    TableCsvMetricsProvider,
    TorchProfilerMetricsProvider,
)
from .metrics_schema import (
    CANONICAL_UNITS,
    EventValidationResult,
    MetricSchemaValidator,
    normalize_event,
    validate_event,
)
from .metrics_store import MetricsStore
from .metrics_taxonomy import CANONICAL_METRIC_PREFIXES, TOOL_METRIC_ALIASES, normalize_external_metric
from .metrics_types import AnalysisReport, Bottleneck, Finding, MetricEvent, PROFILE_SCHEMA_VERSION
from .provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    MetricsProviderRegistry,
    ProviderSpec,
    register_builtin_providers,
)

__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "MetricEvent",
    "Finding",
    "Bottleneck",
    "AnalysisReport",
    "MetricsProvider",
    "ProviderCapabilities",
    "BaseMetricsProvider",
    "MetricSchemaValidator",
    "EventValidationResult",
    "CANONICAL_UNITS",
    "normalize_event",
    "validate_event",
    "MetricsStore",
    "CANONICAL_METRIC_PREFIXES",
    "TOOL_METRIC_ALIASES",
    "normalize_external_metric",
    "MyTimerMetricsProvider",
    "TorchProfilerMetricsProvider",
    "ModuleProfilerMetricsProvider",
    "TableCsvMetricsProvider",
    "NcuCsvMetricsProvider",
    "NsysSqliteMetricsProvider",
    "NsysSqliteGlobMetricsProvider",
    "CProfileStatsProvider",
    "PerfStatTextProvider",
    "DcgmCsvMetricsProvider",
    "NcclLogMetricsProvider",
    "RasJsonMetricsProvider",
    "ProviderSpec",
    "MetricsProviderRegistry",
    "register_builtin_providers",
    "DEFAULT_PROVIDER_REGISTRY",
]


# The two nsys SQLite providers live in ..sources, which imports back into this
# package. Importing them eagerly here closes that cycle and makes
# `import my_utils.profiling.sources` fail in a fresh interpreter. Resolved on
# first attribute access instead (PEP 562), so both import orders work.
_LAZY_PROVIDERS = {"NsysSqliteMetricsProvider", "NsysSqliteGlobMetricsProvider"}


def __getattr__(name: str):
    if name in _LAZY_PROVIDERS:
        from ..sources import nsys_sqlite_provider

        return getattr(nsys_sqlite_provider, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_PROVIDERS)
