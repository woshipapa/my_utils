from .utils import (
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
from .ForwardProfileHook import ForwardProfilerHook
from .annotations import parametrize_shapes
from .logger import GlobalLogger, get_global_logger
from .DITProfiler import create_profiler_context
from .clockSyncUtils import ClockSynchronizer
from .oom_restore import set_oom_flag, check_oom_flag
from .dump_utils import DumpTensorIO, DumpConfig, UniversalDumper, get_dumper
from .module_hook import ForwardTraceRecorder
from .profiling import (
    CaptureBackend,
    NoOpBackend,
    CudaProfilerBackend,
    CaptureController,
    HookEvent,
    extract_meta_from_call,
    ProfileManager,
)

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
]

# Optional exports: keep import-time dependencies minimal for base install.
try:
    from .profilerwrapper import ProfilerWrapper
    __all__.append("ProfilerWrapper")
except Exception:
    pass

try:
    from .moduleProfiler import ModuleProfiler
    __all__.append("ModuleProfiler")
except Exception:
    pass

try:
    from .gpu_mem_tracker import GPU_Performance_Tracker
    __all__.append("GPU_Performance_Tracker")
except Exception:
    pass

try:
    from .etcd_utils import etcd_barrier
    __all__.append("etcd_barrier")
except Exception:
    pass

try:
    from .ncu_analyze_from_csv import (
        analyze_sm_throughput_from_csv,
        compare_kernel_metrics,
    )
    __all__.extend(["analyze_sm_throughput_from_csv", "compare_kernel_metrics"])
except Exception:
    pass

try:
    from .pad import pad_for_sequence_parallel, remove_pad_by_value
    __all__.extend(["pad_for_sequence_parallel", "remove_pad_by_value"])
except Exception:
    pass

