from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TorchProfilerConfig:
    """Controls torch.profiler capture window and trace export."""

    enabled: bool = False
    start_step: int = 0
    stop_step: int = -1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    trace_dir: str = ""


@dataclass
class NsysProfilerConfig:
    """Controls CUDA profiler API window for Nsight Systems capture."""

    enabled: bool = False
    start_step: int = 0
    stop_step: int = -1
    enable_nvtx_labels: bool = False
    nvtx_domain: str = "light_bagel"


@dataclass
class ProfilingEnvConfig:
    """Environment variables derived from config for profiling/debugging."""

    enable_nvtx: Optional[bool] = None
    fsdp_param_debug: bool = False
    fsdp_param_debug_rank: int = 0
    fsdp_param_debug_max_params: int = 30


@dataclass
class NsysLaunchConfig:
    """Launcher-side nsys profile command config (framework-agnostic)."""

    enabled: bool = False
    output: str = ""
    force_overwrite: bool = True
    export_format: str = "none"
    trace: str = "cuda,nvtx,osrt,cublas,cudnn"
    capture_range: str = "cudaProfilerApi"
    capture_range_end: str = "stop"
    gpu_metrics_devices: str = ""
    sample: str = ""
    cudabacktrace: bool = False
    nic_metrics: bool = False
    nvtx_capture: str = ""
    nvtx_domain_include: str = ""
    nvtx_domain_exclude: str = ""
