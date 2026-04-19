from importlib import import_module
import sys

from .adapters import (
    DEFAULT_ADAPTER_REGISTRY,
    DeepSpeedAdapter,
    FrameworkAdapter,
    FrameworkAdapterRegistry,
    HuggingFaceAdapter,
    MegatronAdapter,
    PyTorchAdapter,
    RollAdapter,
    SGLangAdapter,
    SlimeAdapter,
    TorchTitanAdapter,
    VerlAdapter,
    VLLMAdapter,
    build_default_adapter_registry,
)
from .analyzers.analysis_rules import AnalysisRule
from .analyzers.distributed_alignment import align_stage_latency, analyze_rank_skew
from .analyzers.metrics_analyzer import MetricsAnalyzer
from .analyzers.workload_profiles import (
    WORKLOAD_PROFILES,
    WorkloadProfile,
    build_rules_for_workload,
    list_workload_profiles,
    resolve_workload_profile,
)
from .metrics.metrics_provider import BaseMetricsProvider, MetricsProvider, ProviderCapabilities
from .metrics.metrics_providers import (
    CProfileStatsProvider,
    DcgmCsvMetricsProvider,
    ModuleProfilerMetricsProvider,
    MyTimerMetricsProvider,
    NcclLogMetricsProvider,
    NcuCsvMetricsProvider,
    NsysSqliteGlobMetricsProvider,
    NsysSqliteMetricsProvider,
    PerfStatTextProvider,
    RasJsonMetricsProvider,
    TableCsvMetricsProvider,
    TorchProfilerMetricsProvider,
)
from .metrics.metrics_schema import (
    CANONICAL_UNITS,
    EventValidationResult,
    MetricSchemaValidator,
    normalize_event,
    validate_event,
)
from .metrics.metrics_store import MetricsStore
from .metrics.metrics_taxonomy import CANONICAL_METRIC_PREFIXES, TOOL_METRIC_ALIASES, normalize_external_metric
from .metrics.metrics_types import AnalysisReport, Bottleneck, Finding, MetricEvent, PROFILE_SCHEMA_VERSION
from .metrics.provider_registry import (
    DEFAULT_PROVIDER_REGISTRY,
    MetricsProviderRegistry,
    ProviderSpec,
    register_builtin_providers,
)
from .output.metrics_diff import ReportDiff, compare_reports, write_diff
from .output.metrics_report import MetricsReportRenderer
from .output.metrics_trace import (
    ChromeTraceExportConfig,
    estimate_rank_time_offsets,
    export_events_file_to_chrome_trace,
    metric_events_to_chrome_trace,
    write_chrome_trace,
)
from .pipeline.metrics_collector import MetricsCollector
from .runtime.backends import CaptureBackend, CudaProfilerBackend, NoOpBackend
from .runtime.capture_controller import CaptureController, HookEvent
from .runtime.config import NsysLaunchConfig, NsysProfilerConfig, ProfilingEnvConfig, TorchProfilerConfig
from .runtime.frameworkless import (
    apply_profiling_environment,
    build_nsys_launch_prefix,
    create_nsys_capture_backend,
)
from .runtime.meta_adapters import extract_meta_from_call
from .runtime.ProfileManager import ProfileManager
from .runtime.template_utils import get_profiling_template_path, get_profiling_templates_dir
from .sources.nsys_analyze import analyze_nsys_sqlite, analyze_to_markdown
from .sources.nsys_diff import diff_nsys_sqlite, diff_to_markdown
from .sources.nsys_flat_export import collect_kernel_rows, export_kernels_flat
from .sources.nsys_iterations import detect_iterations
from .sources.nsys_mfu import compute_mfu_compare, compute_mfu_single, infer_peak_tflops
from .sources.nsys_schema_adapter import NsightSchema, NsysVersionInfo, detect_nsys_version
from .sources.nsys_sql_skills import NsysSqlSkillEngine, SqlSkill, SqlSkillParam
from .sources.nsys_timeline_html import export_timeline_html

VISUALIZATION_AVAILABLE = False
try:
    from .visualization import (
        AnalysisReport as VizAnalysisReport,
        ChartConfig,
        ChartJsRenderer,
        ChartRenderer,
        DataTransformer,
        HTMLReportGenerator,
        LayoutBuilder,
        PlotlyRenderer,
        QuickReportGenerator,
        create_chart_renderer,
    )

    VISUALIZATION_AVAILABLE = True
except Exception:
    pass

__all__ = [
    "CaptureBackend",
    "NoOpBackend",
    "CudaProfilerBackend",
    "CaptureController",
    "HookEvent",
    "extract_meta_from_call",
    "ProfileManager",
    "TorchProfilerConfig",
    "NsysProfilerConfig",
    "ProfilingEnvConfig",
    "NsysLaunchConfig",
    "create_nsys_capture_backend",
    "apply_profiling_environment",
    "build_nsys_launch_prefix",
    "get_profiling_templates_dir",
    "get_profiling_template_path",
    "PROFILE_SCHEMA_VERSION",
    "MetricEvent",
    "Finding",
    "Bottleneck",
    "AnalysisReport",
    "MetricsProvider",
    "ProviderCapabilities",
    "BaseMetricsProvider",
    "MetricsStore",
    "ChromeTraceExportConfig",
    "estimate_rank_time_offsets",
    "metric_events_to_chrome_trace",
    "write_chrome_trace",
    "export_events_file_to_chrome_trace",
    "MetricSchemaValidator",
    "EventValidationResult",
    "CANONICAL_UNITS",
    "normalize_event",
    "validate_event",
    "MetricsAnalyzer",
    "MetricsReportRenderer",
    "MetricsCollector",
    "AnalysisRule",
    "FrameworkAdapter",
    "FrameworkAdapterRegistry",
    "DEFAULT_ADAPTER_REGISTRY",
    "build_default_adapter_registry",
    "PyTorchAdapter",
    "HuggingFaceAdapter",
    "DeepSpeedAdapter",
    "MegatronAdapter",
    "TorchTitanAdapter",
    "VerlAdapter",
    "SlimeAdapter",
    "RollAdapter",
    "SGLangAdapter",
    "VLLMAdapter",
    "MyTimerMetricsProvider",
    "TorchProfilerMetricsProvider",
    "ModuleProfilerMetricsProvider",
    "TableCsvMetricsProvider",
    "NcuCsvMetricsProvider",
    "NsysSqliteGlobMetricsProvider",
    "NsysSqliteMetricsProvider",
    "CProfileStatsProvider",
    "PerfStatTextProvider",
    "DcgmCsvMetricsProvider",
    "NcclLogMetricsProvider",
    "RasJsonMetricsProvider",
    "ReportDiff",
    "compare_reports",
    "write_diff",
    "CANONICAL_METRIC_PREFIXES",
    "TOOL_METRIC_ALIASES",
    "normalize_external_metric",
    "align_stage_latency",
    "analyze_rank_skew",
    "ProviderSpec",
    "MetricsProviderRegistry",
    "register_builtin_providers",
    "DEFAULT_PROVIDER_REGISTRY",
    "WORKLOAD_PROFILES",
    "WorkloadProfile",
    "resolve_workload_profile",
    "list_workload_profiles",
    "build_rules_for_workload",
    "NsysVersionInfo",
    "NsightSchema",
    "detect_nsys_version",
    "SqlSkillParam",
    "SqlSkill",
    "NsysSqlSkillEngine",
    "detect_iterations",
    "compute_mfu_single",
    "compute_mfu_compare",
    "infer_peak_tflops",
    "collect_kernel_rows",
    "export_kernels_flat",
    "analyze_nsys_sqlite",
    "analyze_to_markdown",
    "diff_nsys_sqlite",
    "diff_to_markdown",
    "export_timeline_html",
    "VISUALIZATION_AVAILABLE",
]

if VISUALIZATION_AVAILABLE:
    __all__.extend(
        [
            "ChartConfig",
            "ChartRenderer",
            "ChartJsRenderer",
            "PlotlyRenderer",
            "create_chart_renderer",
            "DataTransformer",
            "LayoutBuilder",
            "HTMLReportGenerator",
            "QuickReportGenerator",
            "VizAnalysisReport",
        ]
    )


# Compatibility aliases for legacy flat module paths under my_utils.profiling.
_LEGACY_MODULE_ALIASES = {
    "my_utils.profiling.analysis_rules": "my_utils.profiling.analyzers.analysis_rules",
    "my_utils.profiling.distributed_alignment": "my_utils.profiling.analyzers.distributed_alignment",
    "my_utils.profiling.metrics_analyzer": "my_utils.profiling.analyzers.metrics_analyzer",
    "my_utils.profiling.workload_profiles": "my_utils.profiling.analyzers.workload_profiles",
    "my_utils.profiling.metrics_types": "my_utils.profiling.metrics.metrics_types",
    "my_utils.profiling.metrics_provider": "my_utils.profiling.metrics.metrics_provider",
    "my_utils.profiling.metrics_schema": "my_utils.profiling.metrics.metrics_schema",
    "my_utils.profiling.metrics_store": "my_utils.profiling.metrics.metrics_store",
    "my_utils.profiling.metrics_taxonomy": "my_utils.profiling.metrics.metrics_taxonomy",
    "my_utils.profiling.metrics_providers": "my_utils.profiling.metrics.metrics_providers",
    "my_utils.profiling.provider_registry": "my_utils.profiling.metrics.provider_registry",
    "my_utils.profiling.metrics_diff": "my_utils.profiling.output.metrics_diff",
    "my_utils.profiling.metrics_report": "my_utils.profiling.output.metrics_report",
    "my_utils.profiling.metrics_trace": "my_utils.profiling.output.metrics_trace",
    "my_utils.profiling.metrics_collector": "my_utils.profiling.pipeline.metrics_collector",
    "my_utils.profiling.backends": "my_utils.profiling.runtime.backends",
    "my_utils.profiling.capture_controller": "my_utils.profiling.runtime.capture_controller",
    "my_utils.profiling.config": "my_utils.profiling.runtime.config",
    "my_utils.profiling.frameworkless": "my_utils.profiling.runtime.frameworkless",
    "my_utils.profiling.meta_adapters": "my_utils.profiling.runtime.meta_adapters",
    "my_utils.profiling.ProfileManager": "my_utils.profiling.runtime.ProfileManager",
    "my_utils.profiling.template_utils": "my_utils.profiling.runtime.template_utils",
    "my_utils.profiling.nsys_schema_adapter": "my_utils.profiling.sources.nsys_schema_adapter",
    "my_utils.profiling.nsys_sql_skills": "my_utils.profiling.sources.nsys_sql_skills",
    "my_utils.profiling.nsys_sqlite_provider": "my_utils.profiling.sources.nsys_sqlite_provider",
    "my_utils.profiling.nsys_iterations": "my_utils.profiling.sources.nsys_iterations",
    "my_utils.profiling.nsys_mfu": "my_utils.profiling.sources.nsys_mfu",
    "my_utils.profiling.nsys_flat_export": "my_utils.profiling.sources.nsys_flat_export",
    "my_utils.profiling.nsys_analyze": "my_utils.profiling.sources.nsys_analyze",
    "my_utils.profiling.nsys_diff": "my_utils.profiling.sources.nsys_diff",
    "my_utils.profiling.nsys_timeline_html": "my_utils.profiling.sources.nsys_timeline_html",
}


def _register_legacy_module_aliases() -> None:
    for legacy_name, target_name in _LEGACY_MODULE_ALIASES.items():
        if legacy_name in sys.modules:
            continue
        try:
            sys.modules[legacy_name] = import_module(target_name)
        except Exception:
            continue


_register_legacy_module_aliases()
