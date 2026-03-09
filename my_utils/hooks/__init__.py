from .ForwardProfileHook import ForwardProfilerHook
from .module_hook import ForwardTraceRecorder

__all__ = ["ForwardProfilerHook", "ForwardTraceRecorder"]

try:
    from .moduleProfiler import ModuleProfiler

    __all__.append("ModuleProfiler")
except Exception:
    pass