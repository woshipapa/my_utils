from __future__ import annotations

import os
import re
import subprocess
from functools import lru_cache
from typing import Any

from .backends import CudaProfilerBackend

_VERSION_RE = re.compile(r"(\d{4})\.(\d+)")


def _torch():
    """Import torch on demand; only the capture backend needs it."""
    try:
        import torch
    except ImportError as err:
        raise ImportError(
            "torch is required for the capture backend; pip install torch"
        ) from err
    return torch


class TorchCudaProfilerBackend:
    """Fallback backend using torch.cuda.cudart().cudaProfilerStart/Stop."""

    def __init__(self, synchronize: bool = True) -> None:
        self.synchronize = bool(synchronize)

    def _sync(self) -> None:
        torch = _torch()
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self) -> None:
        self._sync()
        _torch().cuda.cudart().cudaProfilerStart()

    def stop(self) -> None:
        self._sync()
        _torch().cuda.cudart().cudaProfilerStop()


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


def _parse_version_hint(text: str) -> tuple[int, int]:
    match = _VERSION_RE.search(text or "")
    if not match:
        return (0, 0)
    try:
        return (int(match.group(1)), int(match.group(2)))
    except ValueError:
        return (0, 0)


@lru_cache(maxsize=1)
def _detect_nsys_version() -> tuple[int, int]:
    """Best-effort nsys version probe; returns (0, 0) when unavailable."""
    try:
        out = subprocess.check_output(
            ["nsys", "--version"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2,
        )
    except Exception:
        return (0, 0)
    return _parse_version_hint(out)


def _resolve_nsys_version(nsys_launch_cfg: Any) -> tuple[int, int]:
    hint = str(getattr(nsys_launch_cfg, "version_hint", "")).strip()
    parsed = _parse_version_hint(hint)
    if parsed != (0, 0):
        return parsed
    return _detect_nsys_version()


def _version_gte(version: tuple[int, int], major: int, minor: int = 0) -> bool:
    return version[0] > major or (version[0] == major and version[1] >= minor)


def _split_trace(trace: str) -> list[str]:
    return [token.strip() for token in trace.split(",") if token.strip()]


def _resolve_nic_metrics_arg(nsys_launch_cfg: Any, version: tuple[int, int]) -> str:
    mode = str(getattr(nsys_launch_cfg, "nic_metrics_mode", "")).strip().lower()
    if not mode and bool(getattr(nsys_launch_cfg, "nic_metrics", False)):
        mode = "lf" if _version_gte(version, 2026, 0) else "true"
    if not mode:
        return ""

    if _version_gte(version, 2026, 0):
        if mode in {"1", "true", "yes", "on"}:
            mode = "lf"
        elif mode in {"0", "false", "no", "off"}:
            mode = "none"
        return f"--nic-metrics={mode}"

    if mode in {"lf", "hf", "1", "true", "yes", "on"}:
        return "--nic-metrics=true"
    if mode in {"none", "0", "false", "no", "off"}:
        return "--nic-metrics=false"
    return f"--nic-metrics={mode}"


def build_nsys_launch_prefix(nsys_launch_cfg: Any) -> list[str]:
    """Build a framework-agnostic `nsys profile ...` command prefix from config."""
    if not bool(getattr(nsys_launch_cfg, "enabled", False)):
        return []

    version = _resolve_nsys_version(nsys_launch_cfg)
    default_trace = "cuda,nvtx,osrt,cublas,cudnn"
    trace_tokens = _split_trace(str(getattr(nsys_launch_cfg, "trace", default_trace)))
    syscall_mode = str(getattr(nsys_launch_cfg, "syscall", "")).strip().lower()

    if _version_gte(version, 2026, 0):
        if "syscall" in trace_tokens:
            trace_tokens = [token for token in trace_tokens if token != "syscall"]
            if not syscall_mode:
                syscall_mode = "process-tree"
    elif syscall_mode and syscall_mode != "none" and "syscall" not in trace_tokens:
        trace_tokens.append("syscall")

    trace_value = ",".join(trace_tokens) if trace_tokens else default_trace

    cmd = [
        "nsys",
        "profile",
        f"--output={str(getattr(nsys_launch_cfg, 'output', ''))}",
        f"--force-overwrite={_bool_to_nsys(getattr(nsys_launch_cfg, 'force_overwrite', True))}",
        f"--export={str(getattr(nsys_launch_cfg, 'export_format', 'none'))}",
        f"--trace={trace_value}",
        f"--capture-range={str(getattr(nsys_launch_cfg, 'capture_range', 'cudaProfilerApi'))}",
        f"--capture-range-end={str(getattr(nsys_launch_cfg, 'capture_range_end', 'stop'))}",
    ]

    gpu_metrics_devices = str(getattr(nsys_launch_cfg, "gpu_metrics_devices", "")).strip()
    if not gpu_metrics_devices:
        gpu_metrics_devices = str(getattr(nsys_launch_cfg, "gpu_metrics_device", "")).strip()
    if gpu_metrics_devices:
        cmd.append(f"--gpu-metrics-devices={gpu_metrics_devices}")

    sample = str(getattr(nsys_launch_cfg, "sample", "")).strip()
    if sample:
        cmd.append(f"--sample={sample}")

    if bool(getattr(nsys_launch_cfg, "cudabacktrace", False)):
        cmd.append("--cudabacktrace=true")

    nic_arg = _resolve_nic_metrics_arg(nsys_launch_cfg, version)
    if nic_arg:
        cmd.append(nic_arg)

    if _version_gte(version, 2026, 0) and syscall_mode:
        cmd.append(f"--syscall={syscall_mode}")

    capture_range = str(getattr(nsys_launch_cfg, "capture_range", "")).lower()
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
