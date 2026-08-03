# SPDX-License-Identifier: Apache-2.0
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
build_collection_manifest = _MOD.build_collection_manifest


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


def test_collection_manifest_records_effective_profile_switches() -> None:
    payload = {
        "output_dir": "./logs/ncu",
        "output_prefix": "case_a",
        "ncu": {
            "enabled": True,
            "replay_mode": "kernel",
            "profile_switches": {
                "replay-mode": "application",
                "app-replay-match": "all",
                "app-replay-mode": "relaxed",
                "range-replay-options": "enable-greedy-sync",
                "graph-profiling": "graph",
                "cache-control": "none",
                "clock-control": "boost",
            },
        },
        "command": ["python", "train.py"],
    }
    command, _ = build_command_from_payload(payload)
    result = build_collection_manifest(command)
    assert result is not None
    path, manifest = result
    assert str(path) == "logs/ncu/case_a.ncu-rep.collection.json"
    assert manifest["collection"] == {
        "ncu_defaults_known": True,
        "replay_mode": "application",
        "app_replay_match": "all",
        "app_replay_mode": "relaxed",
        "range_replay_options": "enable-greedy-sync",
        "graph_profiling": "graph",
        "cache_control": "none",
        "clock_control": "boost",
        "clocks_locked": True,
    }


def test_collection_manifest_normalizes_an_extensionless_export_stem() -> None:
    result = build_collection_manifest(["ncu", "--export=logs/ncu/case_c"])

    assert result is not None
    path, manifest = result
    assert str(path) == "logs/ncu/case_c.ncu-rep.collection.json"
    assert manifest["report_path"] == "logs/ncu/case_c.ncu-rep"


def test_collection_manifest_records_selection_and_sampling_provenance() -> None:
    payload = {
        "output_dir": "./logs/ncu",
        "output_prefix": "case_b",
        "ncu": {
            "enabled": True,
            "mode": "attach",
            "devices": "6,7",
            "kernel_name": "regex:attention.*",
            "kernel_id": "::7",
            "launch_count": 2,
            "launch_skip": 3,
            "target_processes": "all",
            "nvtx": "on",
            "nvtx_include": "decode/",
            "pipeline_boost_state": "stable",
            "process_id": 31415,
            "pm_sampling_interval": 1000,
            "pm_sampling_buffer_size": 67108864,
            "pm_sampling_max_passes": 1,
            "warp_sampling_interval": 4,
            "communicator": "shmem",
            "communicator_num_peers": 2,
            "lockstep_kernel_launch": "on",
            "profile_switches": {
                "nvtx-exclude": ["warmup/", "optimizer/"],
                "disable-pm-warp-sampling": "__flag__",
            },
        },
        "command": ["python", "train.py"],
    }
    command, _ = build_command_from_payload(payload)
    result = build_collection_manifest(command)
    assert result is not None
    _, manifest = result
    collection = manifest["collection"]
    assert manifest["schema_version"] == 2
    assert collection["mode"] == "attach"
    assert collection["devices"] == "6,7"
    assert collection["kernel_name"] == "regex:attention.*"
    assert collection["kernel_id"] == "::7"
    assert collection["launch_count"] == "2"
    assert collection["launch_skip"] == "3"
    assert collection["target_processes"] == "all"
    assert collection["nvtx_include"] == ["decode/"]
    assert collection["nvtx_exclude"] == ["warmup/", "optimizer/"]
    assert collection["pipeline_boost_state"] == "stable"
    assert collection["process_id"] == "31415"
    assert collection["pm_sampling_interval"] == "1000"
    assert collection["pm_sampling_buffer_size"] == "67108864"
    assert collection["pm_sampling_max_passes"] == "1"
    assert collection["warp_sampling_interval"] == "4"
    assert collection["disable_pm_warp_sampling"] is True
    assert collection["communicator"] == "shmem"
    assert collection["communicator_num_peers"] == "2"
    assert collection["lockstep_kernel_launch"] == "on"
