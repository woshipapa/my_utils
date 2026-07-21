from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path


def _load_frameworkless():
    root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling" / "runtime"

    # Build a lightweight package chain so relative imports in frameworkless.py work
    # without importing the entire my_utils package tree.
    for name, path in [
        ("my_utils", str(root.parents[2])),
        ("my_utils.profiling", str(root.parents[1])),
        ("my_utils.profiling.runtime", str(root)),
    ]:
        if name not in sys.modules:
            mod = type(sys)(name)
            mod.__path__ = [path]  # type: ignore[attr-defined]
            sys.modules[name] = mod

    backends_name = "my_utils.profiling.runtime.backends"
    backends_spec = importlib.util.spec_from_file_location(backends_name, root / "backends.py")
    assert backends_spec is not None and backends_spec.loader is not None
    backends_mod = importlib.util.module_from_spec(backends_spec)
    sys.modules[backends_name] = backends_mod
    backends_spec.loader.exec_module(backends_mod)

    fw_name = "my_utils.profiling.runtime.frameworkless"
    fw_spec = importlib.util.spec_from_file_location(fw_name, root / "frameworkless.py")
    assert fw_spec is not None and fw_spec.loader is not None
    fw_mod = importlib.util.module_from_spec(fw_spec)
    sys.modules[fw_name] = fw_mod
    fw_spec.loader.exec_module(fw_mod)
    return fw_mod


_FW = _load_frameworkless()
build_nsys_launch_prefix = _FW.build_nsys_launch_prefix


def _cfg(**kwargs):
    base = {
        "enabled": True,
        "output": "./out/report",
        "force_overwrite": True,
        "export_format": "none",
        "trace": "cuda,nvtx,osrt,cublas,cudnn",
        "capture_range": "cudaProfilerApi",
        "capture_range_end": "stop",
        "gpu_metrics_devices": "",
        "gpu_metrics_device": "",
        "sample": "",
        "cudabacktrace": False,
        "nic_metrics": False,
        "nic_metrics_mode": "",
        "syscall": "",
        "version_hint": "",
        "nvtx_capture": "",
        "nvtx_domain_include": "",
        "nvtx_domain_exclude": "",
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_nic_metrics_bool_uses_lf_for_2026() -> None:
    cmd = build_nsys_launch_prefix(_cfg(nic_metrics=True, version_hint="2026.2"))
    assert "--nic-metrics=lf" in cmd
    assert "--nic-metrics=true" not in cmd


def test_nic_metrics_bool_keeps_true_for_2024() -> None:
    cmd = build_nsys_launch_prefix(_cfg(nic_metrics=True, version_hint="2024.7"))
    assert "--nic-metrics=true" in cmd


def test_nic_metrics_mode_hf_falls_back_on_2024() -> None:
    cmd = build_nsys_launch_prefix(_cfg(nic_metrics_mode="hf", version_hint="2024.7"))
    assert "--nic-metrics=true" in cmd


def test_nic_metrics_mode_hf_on_2026() -> None:
    cmd = build_nsys_launch_prefix(_cfg(nic_metrics_mode="hf", version_hint="2026.2"))
    assert "--nic-metrics=hf" in cmd


def test_trace_syscall_is_mapped_to_switch_on_2026() -> None:
    cmd = build_nsys_launch_prefix(_cfg(trace="cuda,nvtx,syscall", version_hint="2026.2"))
    assert "--trace=cuda,nvtx" in cmd
    assert "--syscall=process-tree" in cmd


def test_trace_syscall_kept_for_2024() -> None:
    cmd = build_nsys_launch_prefix(_cfg(trace="cuda,nvtx,syscall", version_hint="2024.7"))
    assert "--trace=cuda,nvtx,syscall" in cmd
    assert all(not item.startswith("--syscall=") for item in cmd)


def test_explicit_syscall_switch_for_2026() -> None:
    cmd = build_nsys_launch_prefix(_cfg(syscall="pid-namespace", version_hint="2026.2"))
    assert "--syscall=pid-namespace" in cmd


def test_legacy_gpu_metrics_device_alias_is_supported() -> None:
    cmd = build_nsys_launch_prefix(_cfg(gpu_metrics_device="all", version_hint="2026.2"))
    assert "--gpu-metrics-devices=all" in cmd
