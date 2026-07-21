from importlib import abc as _importlib_abc
from importlib import import_module
from importlib import machinery as _importlib_machinery
import sys

# Torch-free eager re-exports.  `my_utils.profiling` and `my_utils.tracing`
# never import torch at module level, so importing them keeps a plain
# `import my_utils` working on machines without the GPU stack installed.
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

# Lazy (PEP 562) re-exports.  The core/hooks/distributed/memory/artifacts/
# legacy_profilers subpackages import torch (or other heavy optional deps) at
# module level, so their names are only imported on first attribute access to
# keep `import my_utils` torch-free.
_LAZY_ATTRS = {
    # .core.utils
    "register_hooks": "my_utils.core.utils",
    "print_model_params": "my_utils.core.utils",
    "tensor_md5": "my_utils.core.utils",
    "DebugLayer": "my_utils.core.utils",
    "filename": "my_utils.core.utils",
    "MyTimer": "my_utils.core.utils",
    "NoOpMyTimer": "my_utils.core.utils",
    "global_timer": "my_utils.core.utils",
    "setup_logging_and_timer": "my_utils.core.utils",
    "print_cuda_memory_gb": "my_utils.core.utils",
    "DebuggingEvent": "my_utils.core.utils",
    "print_tensor_info": "my_utils.core.utils",
    "record_oom_threshold": "my_utils.core.utils",
    "ChecksumUtils": "my_utils.core.utils",
    "get_global_timer": "my_utils.core.utils",
    # .core.annotations / .core.logger / .core.method_patch
    "parametrize_shapes": "my_utils.core.annotations",
    "GlobalLogger": "my_utils.core.logger",
    "get_global_logger": "my_utils.core.logger",
    "MethodPatchHandle": "my_utils.core.method_patch",
    "MethodPatcher": "my_utils.core.method_patch",
    # .hooks
    "ForwardProfilerHook": "my_utils.hooks.ForwardProfileHook",
    "ForwardTraceRecorder": "my_utils.hooks.module_hook",
    "ModuleProfiler": "my_utils.hooks.moduleProfiler",
    # .legacy_profilers
    "create_profiler_context": "my_utils.legacy_profilers.DITProfiler",
    "ProfilerWrapper": "my_utils.legacy_profilers.profilerwrapper",
    # .distributed
    "ClockSynchronizer": "my_utils.distributed.clockSyncUtils",
    "etcd_barrier": "my_utils.distributed.etcd_utils",
    "pad_for_sequence_parallel": "my_utils.distributed.pad",
    "remove_pad_by_value": "my_utils.distributed.pad",
    # .memory
    "set_oom_flag": "my_utils.memory.oom_restore",
    "check_oom_flag": "my_utils.memory.oom_restore",
    "GPU_Performance_Tracker": "my_utils.memory.gpu_mem_tracker",
    # .artifacts
    "DumpTensorIO": "my_utils.artifacts.dump_utils",
    "DumpConfig": "my_utils.artifacts.dump_utils",
    "UniversalDumper": "my_utils.artifacts.dump_utils",
    "get_dumper": "my_utils.artifacts.dump_utils",
    "analyze_sm_throughput_from_csv": "my_utils.artifacts.ncu_analyze_from_csv",
    "compare_kernel_metrics": "my_utils.artifacts.ncu_analyze_from_csv",
}


def __getattr__(name: str):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_path)
    except ImportError as err:
        raise ImportError(
            f"my_utils.{name} requires {module_path}, which failed to import: {err}. "
            "If torch is missing, install it with `pip install torch`."
        ) from err
    value = getattr(module, name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))


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
    # Optional exports, resolved lazily; accessing them without their heavy
    # dependencies installed raises ImportError (previously they were probed
    # eagerly and silently dropped from __all__ when unavailable).
    "ProfilerWrapper",
    "ModuleProfiler",
    "GPU_Performance_Tracker",
    "etcd_barrier",
    "analyze_sm_throughput_from_csv",
    "compare_kernel_metrics",
    "pad_for_sequence_parallel",
    "remove_pad_by_value",
]

# Optional exports: keep import-time dependencies minimal for base install.
try:
    from .profiling.adapters import (
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
            "TorchTitanAdapter",
            "VerlAdapter",
            "SlimeAdapter",
            "RollAdapter",
            "SGLangAdapter",
            "VLLMAdapter",
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


class _LegacyAliasLoader(_importlib_abc.Loader):
    """Loader that resolves a legacy alias by importing its target module.

    ``exec_module`` swaps the freshly created placeholder for the real target
    module in ``sys.modules``; the import machinery re-reads ``sys.modules``
    after execution, so the legacy name ends up bound to the target module —
    exactly what the previous eager registration produced, minus the eager
    (torch-importing) part.
    """

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(self, spec):
        return None  # use default module creation

    def exec_module(self, module) -> None:
        sys.modules[module.__name__] = import_module(self._target_name)


class _LegacyAliasFinder(_importlib_abc.MetaPathFinder):
    """Meta-path finder serving `my_utils.<flat_name>` legacy aliases lazily."""

    def find_spec(self, fullname, path=None, target=None):
        target_name = _LEGACY_MODULE_ALIASES.get(fullname)
        if target_name is None:
            return None
        return _importlib_machinery.ModuleSpec(fullname, _LegacyAliasLoader(target_name))


def _register_legacy_module_aliases() -> None:
    if not any(isinstance(finder, _LegacyAliasFinder) for finder in sys.meta_path):
        sys.meta_path.append(_LegacyAliasFinder())


_register_legacy_module_aliases()
