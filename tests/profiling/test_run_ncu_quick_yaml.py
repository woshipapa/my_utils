from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_ncu_launcher_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "my_utils"
        / "profiling"
        / "ncu"
        / "run_ncu_quick_yaml.py"
    )
    spec = importlib.util.spec_from_file_location("run_ncu_quick_yaml", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOD = _load_ncu_launcher_module()
build_command_from_payload = _MOD.build_command_from_payload


def test_ncu_basic_command_build() -> None:
    payload = {
        "output_dir": "./logs/ncu",
        "output_prefix": "case_a",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "ncu": {
            "enabled": True,
            "set": "full",
            "section": ["LaunchStats"],
            "launch_count": 1,
            "launch_skip": 2,
            "profile_from_start": "off",
        },
        "command": ["python", "train.py"],
    }
    cmd, env_updates = build_command_from_payload(payload)
    assert cmd[0] == "ncu"
    assert "--set=full" in cmd
    assert "--section=LaunchStats" in cmd
    assert "--launch-count=1" in cmd
    assert "--launch-skip=2" in cmd
    assert "--profile-from-start=off" in cmd
    assert "--export=logs/ncu/case_a.ncu-rep" in cmd
    assert cmd[-2:] == ["python", "train.py"]
    assert env_updates["CUDA_VISIBLE_DEVICES"] == "0"


def test_ncu_profile_switches_override_core() -> None:
    payload = {
        "ncu": {
            "enabled": True,
            "set": "full",
            "profile_switches": {"set": "speed-of-light"},
        },
        "command": ["python", "train.py"],
    }
    cmd, _ = build_command_from_payload(payload)
    assert "--set=speed-of-light" in cmd
    assert "--set=full" not in cmd


def test_ncu_profile_switches_list_and_flag() -> None:
    payload = {
        "ncu": {
            "enabled": True,
            "profile_switches": {
                "section-folder": ["./sec_a", "./sec_b"],
                "help": "__flag__",
            },
        },
        "command": ["python", "train.py"],
    }
    cmd, _ = build_command_from_payload(payload)
    assert "--section-folder=./sec_a" in cmd
    assert "--section-folder=./sec_b" in cmd
    assert "--help" in cmd


def test_ncu_import_without_command_is_allowed() -> None:
    payload = {
        "ncu": {
            "enabled": True,
            "import_report": "./a.ncu-rep",
            "profile_switches": {"page": "details"},
        },
    }
    cmd, _ = build_command_from_payload(payload)
    assert cmd[0] == "ncu"
    assert "--import=./a.ncu-rep" in cmd
    assert "--page=details" in cmd


def test_ncu_command_override() -> None:
    payload = {
        "ncu": {"enabled": False},
        "command": ["python", "will_not_run.py"],
    }
    override = ["deepspeed", "--num_gpus=8", "train.py"]
    cmd, _ = build_command_from_payload(payload, override_command=override)
    assert cmd == override
