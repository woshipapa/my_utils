from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_yaml_launcher_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "my_utils"
        / "profiling"
        / "templates"
        / "run_nsys_quick_yaml.py"
    )
    spec = importlib.util.spec_from_file_location("run_nsys_quick_yaml", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOD = _load_yaml_launcher_module()
build_command_from_payload = _MOD.build_command_from_payload


def test_yaml_launcher_2026_syscall_and_extra_args() -> None:
    payload = {
        "output_dir": "./logs/nsys",
        "output_prefix": "unit_%p",
        "env": {"ENABLE_NVTX": "1"},
        "nsys_launch": {
            "enabled": True,
            "version_hint": "2026.2",
            "trace": "cuda,nvtx,syscall",
            "capture_range": "cudaProfilerApi",
            "capture_range_end": "stop",
            "extra_profile_args": ["--duration=30"],
        },
        "command": ["python", "train.py", "--config", "cfg.yaml"],
    }

    cmd, env_updates = build_command_from_payload(payload)
    joined = " ".join(cmd)
    assert cmd[:2] == ["nsys", "profile"]
    assert "--trace=cuda,nvtx" in cmd
    assert "--syscall=process-tree" in cmd
    assert "--duration=30" in cmd
    assert joined.endswith("python train.py --config cfg.yaml")
    assert env_updates["ENABLE_NVTX"] == "1"


def test_yaml_launcher_2024_fallback_keeps_trace_syscall() -> None:
    payload = {
        "nsys_launch": {
            "enabled": True,
            "version_hint": "2024.7",
            "trace": "cuda,nvtx,syscall",
            "extra_profile_args": [],
        },
        "command": "python train.py",
    }
    cmd, _ = build_command_from_payload(payload)
    assert "--trace=cuda,nvtx,syscall" in cmd
    assert all(not item.startswith("--syscall=") for item in cmd)


def test_yaml_launcher_command_override() -> None:
    payload = {
        "nsys_launch": {"enabled": False},
        "command": ["python", "will_not_run.py"],
    }
    override = ["deepspeed", "--num_gpus=8", "train.py"]
    cmd, _ = build_command_from_payload(payload, override_command=override)
    assert cmd == override


def test_yaml_launcher_profile_switches_override_core_fields() -> None:
    payload = {
        "nsys_launch": {
            "enabled": True,
            "version_hint": "2026.2",
            "capture_range_end": "stop",
            "profile_switches": {
                "capture-range-end": "none",
                "session-new": "my_train_sess",
                "flush_on_cudaprofilerstop": False,
            },
        },
        "command": ["python", "train.py"],
    }
    cmd, _ = build_command_from_payload(payload)
    assert "--capture-range-end=none" in cmd
    assert "--capture-range-end=stop" not in cmd
    assert "--session-new=my_train_sess" in cmd
    assert "--flush-on-cudaprofilerstop=false" in cmd


def test_yaml_launcher_profile_switches_support_list_and_flag() -> None:
    payload = {
        "nsys_launch": {
            "enabled": False,
            "profile_switches": {
                "env-var": ["A=1", "B=2"],
                "help": "__flag__",
            },
        },
        "command": ["python", "train.py"],
    }
    cmd, _ = build_command_from_payload(payload)
    assert cmd[:2] == ["nsys", "profile"]
    assert "--env-var=A=1" in cmd
    assert "--env-var=B=2" in cmd
    assert "--help" in cmd
