from importlib import import_module
import sys

from .core.utils import (
    register_hooks,
    print_model_params,
    tensor_md5,
    DebugLayer,
    filename,
    MyTimer,
    NoOpMyTimer,
    global_timer,
    setup_logging_and_timer,
    print_cuda_memory_gb,
    DebuggingEvent,
    print_tensor_info,
    record_oom_threshold,
    ChecksumUtils,
    get_global_timer,
)
from .hooks.ForwardProfileHook import ForwardProfilerHook
from .core.annotations import parametrize_shapes
from .core.logger import GlobalLogger, get_global_logger
from .legacy_profilers.DITProfiler import create_profiler_context
from .distributed.clockSyncUtils import ClockSynchronizer
from .memory.oom_restore import set_oom_flag, check_oom_flag
from .artifacts.dump_utils import DumpTensorIO, DumpConfig, UniversalDumper, get_dumper
from .hooks.module_hook import ForwardTraceRecorder
from .profiling import (
    CaptureBackend,
    NoOpBackend,
    CudaProfilerBackend,
    CaptureController,
    HookEvent,
    extract_meta_from_call,
    ProfileManager,
    TorchProfilerConfig,
    NsysProfilerConfig,
    ProfilingEnvConfig,
    NsysLaunchConfig,
    create_nsys_capture_backend,
    apply_profiling_environment,
    build_nsys_launch_prefix,
    get_profiling_templates_dir,
    get_profiling_template_path,
    MetricEvent,
    Finding,
    Bottleneck,
    AnalysisReport,
    MetricsProvider,
    BaseMetricsProvider,
    MetricsStore,
    ChromeTraceExportConfig,
    estimate_rank_time_offsets,
    metric_events_to_chrome_trace,
    write_chrome_trace,
    export_events_file_to_chrome_trace,
    MetricsAnalyzer,
    MetricsReportRenderer,
    MetricsCollector,
    MyTimerMetricsProvider,
    TorchProfilerMetricsProvider,
    ModuleProfilerMetricsProvider,
    TableCsvMetricsProvider,
    NcuCsvMetricsProvider,
    NsysSqliteMetricsProvider,
    CProfileStatsProvider,
    PerfStatTextProvider,
    ProviderCapabilities,
    MetricSchemaValidator,
    EventValidationResult,
    CANONICAL_UNITS,
    normalize_event,
    validate_event,
    ReportDiff,
    compare_reports,
    write_diff,
    align_stage_latency,
    analyze_rank_skew,
    ProviderSpec,
    MetricsProviderRegistry,
    register_builtin_providers,
    DEFAULT_PROVIDER_REGISTRY,
    WORKLOAD_PROFILES,
    WorkloadProfile,
    resolve_workload_profile,
    list_workload_profiles,
    build_rules_for_workload,
    NsysVersionInfo,
    detect_nsys_version,
    VISUALIZATION_AVAILABLE,
    CANONICAL_METRIC_PREFIXES,
    TOOL_METRIC_ALIASES,
    normalize_external_metric,
)
from .tracing.nvtx_utils import (
    LabelerProtocol,
    NoOpLabeler,
    NvtxLabeler,
    NVTX_AVAILABLE,
    TorchNvtxLabeler,
    TORCH_NVTX_AVAILABLE,
    create_labeler,
)
from .core.method_patch import MethodPatchHandle, MethodPatcher

__all__ = [
    "register_hooks",
    "print_model_params",
    "tensor_md5",
    "DebugLayer",
    "filename",
    "MyTimer",
    "NoOpMyTimer",
    "global_timer",
    "setup_logging_and_timer",
    "print_cuda_memory_gb",
    "DebuggingEvent",
    "print_tensor_info",
    "record_oom_threshold",
    "ChecksumUtils",
    "get_global_timer",
    "ForwardProfilerHook",
    "parametrize_shapes",
    "GlobalLogger",
    "get_global_logger",
    "create_profiler_context",
    "ClockSynchronizer",
    "set_oom_flag",
    "check_oom_flag",
    "DumpTensorIO",
    "DumpConfig",
    "UniversalDumper",
    "get_dumper",
    "ForwardTraceRecorder",
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
    "MetricEvent",
    "Finding",
    "Bottleneck",
    "AnalysisReport",
    "MetricsProvider",
    "BaseMetricsProvider",
    "MetricsStore",
    "ChromeTraceExportConfig",
    "estimate_rank_time_offsets",
    "metric_events_to_chrome_trace",
    "write_chrome_trace",
    "export_events_file_to_chrome_trace",
    "MetricsAnalyzer",
    "MetricsReportRenderer",
    "MetricsCollector",
    "MyTimerMetricsProvider",
    "TorchProfilerMetricsProvider",
    "ModuleProfilerMetricsProvider",
    "TableCsvMetricsProvider",
    "NcuCsvMetricsProvider",
    "NsysSqliteMetricsProvider",
    "CProfileStatsProvider",
    "PerfStatTextProvider",
    "ProviderCapabilities",
    "MetricSchemaValidator",
    "EventValidationResult",
    "CANONICAL_UNITS",
    "normalize_event",
    "validate_event",
    "ReportDiff",
    "compare_reports",
    "write_diff",
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
    "detect_nsys_version",
    "VISUALIZATION_AVAILABLE",
    "CANONICAL_METRIC_PREFIXES",
    "TOOL_METRIC_ALIASES",
    "normalize_external_metric",
    "LabelerProtocol",
    "NoOpLabeler",
    "NvtxLabeler",
    "NVTX_AVAILABLE",
    "TorchNvtxLabeler",
    "TORCH_NVTX_AVAILABLE",
    "create_labeler",
    "MethodPatchHandle",
    "MethodPatcher",
]

# Optional exports: keep import-time dependencies minimal for base install.
try:
    from .legacy_profilers.profilerwrapper import ProfilerWrapper

    __all__.append("ProfilerWrapper")
except Exception:
    pass

try:
    from .hooks.moduleProfiler import ModuleProfiler

    __all__.append("ModuleProfiler")
except Exception:
    pass

try:
    from .memory.gpu_mem_tracker import GPU_Performance_Tracker

    __all__.append("GPU_Performance_Tracker")
except Exception:
    pass

try:
    from .distributed.etcd_utils import etcd_barrier

    __all__.append("etcd_barrier")
except Exception:
    pass

try:
    from .artifacts.ncu_analyze_from_csv import (
        analyze_sm_throughput_from_csv,
        compare_kernel_metrics,
    )

    __all__.extend(["analyze_sm_throughput_from_csv", "compare_kernel_metrics"])
except Exception:
    pass

try:
    from .distributed.pad import pad_for_sequence_parallel, remove_pad_by_value

    __all__.extend(["pad_for_sequence_parallel", "remove_pad_by_value"])
except Exception:
    pass

try:
    from .profiling.adapters import (
        DEFAULT_ADAPTER_REGISTRY,
        FrameworkAdapter,
        FrameworkAdapterRegistry,
        HuggingFaceAdapter,
        MegatronAdapter,
        PyTorchAdapter,
        DeepSpeedAdapter,
    )

    __all__.extend(
        [
            "FrameworkAdapter",
            "FrameworkAdapterRegistry",
            "DEFAULT_ADAPTER_REGISTRY",
            "PyTorchAdapter",
            "HuggingFaceAdapter",
            "DeepSpeedAdapter",
            "MegatronAdapter",
        ]
    )
except Exception:
    pass


# Compatibility aliases for legacy flat-module import paths, e.g. `from my_utils.utils import MyTimer`.
_LEGACY_MODULE_ALIASES = {
    "my_utils.annotations": "my_utils.core.annotations",
    "my_utils.clockSyncUtils": "my_utils.distributed.clockSyncUtils",
    "my_utils.DITProfiler": "my_utils.legacy_profilers.DITProfiler",
    "my_utils.dump_utils": "my_utils.artifacts.dump_utils",
    "my_utils.etcd_utils": "my_utils.distributed.etcd_utils",
    "my_utils.ForwardProfileHook": "my_utils.hooks.ForwardProfileHook",
    "my_utils.gpu_mem_tracker": "my_utils.memory.gpu_mem_tracker",
    "my_utils.logger": "my_utils.core.logger",
    "my_utils.memory_snapshot": "my_utils.memory.memory_snapshot",
    "my_utils.method_patch": "my_utils.core.method_patch",
    "my_utils.moduleProfiler": "my_utils.hooks.moduleProfiler",
    "my_utils.module_hook": "my_utils.hooks.module_hook",
    "my_utils.ncu_analyze_from_csv": "my_utils.artifacts.ncu_analyze_from_csv",
    "my_utils.nvtx_utils": "my_utils.tracing.nvtx_utils",
    "my_utils.oom_restore": "my_utils.memory.oom_restore",
    "my_utils.pad": "my_utils.distributed.pad",
    "my_utils.profilerwrapper": "my_utils.legacy_profilers.profilerwrapper",
    "my_utils.utils": "my_utils.core.utils",
}


def _register_legacy_module_aliases() -> None:
    for legacy_name, target_name in _LEGACY_MODULE_ALIASES.items():
        if legacy_name in sys.modules:
            continue
        try:
            sys.modules[legacy_name] = import_module(target_name)
        except Exception:
            # Keep package import robust even when optional dependencies are absent.
            continue


_register_legacy_module_aliases()