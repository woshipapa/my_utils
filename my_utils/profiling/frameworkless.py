from __future__ import annotations

import os
from typing import Any

import torch

from .backends import CudaProfilerBackend


class TorchCudaProfilerBackend:
    """Fallback backend using torch.cuda.cudart().cudaProfilerStart/Stop."""

    def __init__(self, synchronize: bool = True) -> None:
        self.synchronize = bool(synchronize)

    def _sync(self) -> None:
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self) -> None:
        self._sync()
        torch.cuda.cudart().cudaProfilerStart()

    def stop(self) -> None:
        self._sync()
        torch.cuda.cudart().cudaProfilerStop()


def create_nsys_capture_backend(synchronize: bool = True) -> tuple[Any, str]:
    """
    Prefer my_utils profiling backend.
    Fall back to torch cudart backend when needed.
    """
    try:
        return CudaProfilerBackend(synchronize=synchronize), "my_utils.CudaProfilerBackend"
    except Exception as err:
        return TorchCudaProfilerBackend(synchronize=synchronize), (
            f"torch.cuda.cudart (fallback: {type(err).__name__})"
        )


def apply_profiling_environment(config: Any) -> dict[str, str]:
    """
    Materialize selected profiling env vars from config.
    Supports either:
    - config.profiling_env
    - a direct ProfilingEnvConfig-like object
    """
    env_updates: dict[str, str] = {}
    profiling_env = getattr(config, "profiling_env", config)
    if profiling_env is None:
        return env_updates

    enable_nvtx = getattr(profiling_env, "enable_nvtx", None)
    if enable_nvtx is not None:
        env_updates["ENABLE_NVTX"] = "1" if bool(enable_nvtx) else "0"

    if bool(getattr(profiling_env, "fsdp_param_debug", False)):
        env_updates["FSDP_PARAM_DEBUG"] = "1"
        env_updates["FSDP_PARAM_DEBUG_RANK"] = str(
            int(getattr(profiling_env, "fsdp_param_debug_rank", 0))
        )
        env_updates["FSDP_PARAM_DEBUG_MAX_PARAMS"] = str(
            int(getattr(profiling_env, "fsdp_param_debug_max_params", 30))
        )

    for key, value in env_updates.items():
        os.environ[key] = value

    return env_updates


def _bool_to_nsys(value: bool) -> str:
    return "true" if bool(value) else "false"


def build_nsys_launch_prefix(nsys_launch_cfg: Any) -> list[str]:
    """Build a framework-agnostic `nsys profile ...` command prefix from config."""
    if not bool(getattr(nsys_launch_cfg, "enabled", False)):
        return []

    cmd = [
        "nsys",
        "profile",
        f"--output={str(getattr(nsys_launch_cfg, 'output', ''))}",
        f"--force-overwrite={_bool_to_nsys(getattr(nsys_launch_cfg, 'force_overwrite', True))}",
        f"--export={str(getattr(nsys_launch_cfg, 'export_format', 'none'))}",
        f"--trace={str(getattr(nsys_launch_cfg, 'trace', 'cuda,nvtx,osrt,cublas,cudnn'))}",
        f"--capture-range={str(getattr(nsys_launch_cfg, 'capture_range', 'cudaProfilerApi'))}",
        f"--capture-range-end={str(getattr(nsys_launch_cfg, 'capture_range_end', 'stop'))}",
    ]

    gpu_metrics_devices = str(getattr(nsys_launch_cfg, "gpu_metrics_devices", "")).strip()
    if gpu_metrics_devices:
        cmd.append(f"--gpu-metrics-devices={gpu_metrics_devices}")

    sample = str(getattr(nsys_launch_cfg, "sample", "")).strip()
    if sample:
        cmd.append(f"--sample={sample}")

    if bool(getattr(nsys_launch_cfg, "cudabacktrace", False)):
        cmd.append("--cudabacktrace=true")
    if bool(getattr(nsys_launch_cfg, "nic_metrics", False)):
        cmd.append("--nic-metrics=true")

    capture_range = str(getattr(nsys_launch_cfg, "capture_range", ""))
    nvtx_capture = str(getattr(nsys_launch_cfg, "nvtx_capture", "")).strip()
    if capture_range == "nvtx" and nvtx_capture:
        cmd.append(f"--nvtx-capture={nvtx_capture}")

    nvtx_domain_include = str(getattr(nsys_launch_cfg, "nvtx_domain_include", "")).strip()
    if nvtx_domain_include:
        cmd.append(f"--nvtx-domain-include={nvtx_domain_include}")

    nvtx_domain_exclude = str(getattr(nsys_launch_cfg, "nvtx_domain_exclude", "")).strip()
    if nvtx_domain_exclude:
        cmd.append(f"--nvtx-domain-exclude={nvtx_domain_exclude}")

    return cmd
