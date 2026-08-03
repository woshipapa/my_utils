#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


# Schema 1 recorded only the controls that were already consumed by the
# measurement-context model.  Schema 2 additionally preserves the selection,
# sampling and device provenance needed to decide whether two reports refer to
# the same experiment.  Readers remain backward compatible with schema 1.
_COLLECTION_MANIFEST_SCHEMA_VERSION = 2


def _load_yaml_payload(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from exc

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return payload


def _coerce_command(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = shlex.split(value.strip())
        if not parts:
            return []
        return parts
    if isinstance(value, list) and all(
        isinstance(item, (str, int, float)) for item in value
    ):
        return [str(item) for item in value]
    raise ValueError("command must be a string or list.")


def _coerce_env(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("env must be a mapping.")
    return {str(key): str(val) for key, val in value.items()}


def _normalize_switch_name(name: str) -> str:
    normalized = str(name).strip()
    if normalized.startswith("--"):
        normalized = normalized[2:]
    return normalized.replace("_", "-")


def _coerce_profile_switches(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("ncu.profile_switches must be a mapping.")
    out: dict[str, Any] = {}
    for key, item in value.items():
        normalized = _normalize_switch_name(str(key))
        if not normalized:
            continue
        out[normalized] = item
    return out


def _coerce_extra_args(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("ncu.extra_args must be a list.")
    return [str(item) for item in value if str(item).strip()]


def _serialize_switches(switches: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for name, value in switches.items():
        opt = f"--{name}"
        if value is None:
            continue
        if isinstance(value, bool):
            args.append(f"{opt}={'on' if value else 'off'}")
            continue
        if isinstance(value, (int, float)):
            args.append(f"{opt}={value}")
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            if text == "__flag__":
                args.append(opt)
            else:
                args.append(f"{opt}={text}")
            continue
        if isinstance(value, list):
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if not text:
                    continue
                if text == "__flag__":
                    args.append(opt)
                else:
                    args.append(f"{opt}={text}")
            continue
        raise ValueError(
            f"Unsupported value type for switch '{name}': {type(value).__name__}"
        )
    return args


def _option_name(arg: str) -> str:
    if not arg.startswith("--"):
        return ""
    return arg.split("=", 1)[0][2:]


def _option_value(args: list[str], name: str) -> str:
    prefix = f"--{name}="
    for item in args:
        if item.startswith(prefix):
            return item[len(prefix) :]
    return ""


def _option_values(args: list[str], name: str) -> list[str]:
    """Return every value given to a repeatable long NCU option."""
    prefix = f"--{name}="
    return [item[len(prefix) :] for item in args if item.startswith(prefix)]


def _option_state(args: list[str], name: str) -> str | bool:
    """Return a long option's value, or ``True`` for a bare flag."""
    value = _option_value(args, name)
    if value:
        return value
    return True if f"--{name}" in args else ""


def _report_path_from_export(export_path: str) -> Path:
    """Normalise an NCU ``--export`` value to its resulting report path."""
    report_path = Path(export_path)
    if not str(report_path).endswith(".ncu-rep"):
        return Path(str(report_path) + ".ncu-rep")
    return report_path


def collection_manifest_path(export_path: str) -> Path:
    """Return the sidecar path for NCU's resulting ``.ncu-rep`` report.

    The YAML convention supplies the suffix itself, but NCU also accepts an
    extensionless ``--export`` stem and writes ``<stem>.ncu-rep``.  Normalise
    both forms so automatic sidecar discovery always addresses the same path.
    """
    return Path(str(_report_path_from_export(export_path)) + ".collection.json")


_COLLECTION_METADATA_FIELDS = frozenset(
    {
        "workload_id",
        "problem_shape",
        "dtype",
        "input_hash",
        "output_hash",
        "logical_kernel_id",
        "kernel_aliases",
        "kernel_config",
        "build_id",
        "git_commit",
        "ncu_version",
        "driver_version",
        "host_name",
        "cuda_visible_devices",
        "gpu_identities",
        "mig_instance_id",
        "mps_active",
        "iterations",
        "warmup_iterations",
        "input_distribution",
    }
)


def collection_metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Read explicit workload provenance from ``ncu.collection_metadata``.

    Metadata stays opt-in: the runner records exact command semantics itself,
    while callers provide identities that only their workload understands.
    """
    raw = (payload.get("ncu") or {}).get("collection_metadata")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("ncu.collection_metadata must be a mapping.")
    return {key: value for key, value in raw.items() if key in _COLLECTION_METADATA_FIELDS}


def build_collection_manifest(
    command: list[str], *, metadata: dict[str, Any] | None = None
) -> tuple[Path, dict[str, Any]] | None:
    """Extract report-interpretation settings from the exact NCU command."""
    if not command or command[0] != "ncu":
        return None
    export_path = _option_value(command, "export")
    if not export_path:
        return None
    replay_mode = _option_value(command, "replay-mode")
    app_replay_match = _option_value(command, "app-replay-match")
    app_replay_mode = _option_value(command, "app-replay-mode")
    range_replay_options = _option_value(command, "range-replay-options")
    graph_profiling = _option_value(command, "graph-profiling")
    cache_control = _option_value(command, "cache-control")
    clock_control = _option_value(command, "clock-control")
    pipeline_boost_state = _option_value(command, "pipeline-boost-state")
    # This runner constructed the command, so an omitted cache-control option
    # has a known NCU default. Imported reports never get this assertion.
    collection: dict[str, Any] = {"ncu_defaults_known": True}
    if replay_mode:
        collection["replay_mode"] = replay_mode
    if app_replay_match:
        collection["app_replay_match"] = app_replay_match
    if app_replay_mode:
        collection["app_replay_mode"] = app_replay_mode
    if range_replay_options:
        collection["range_replay_options"] = range_replay_options
    if graph_profiling:
        collection["graph_profiling"] = graph_profiling
    if cache_control:
        collection["cache_control"] = cache_control
    if clock_control:
        collection["clock_control"] = clock_control
        if clock_control in {"base", "boost", "force-boost"}:
            collection["clocks_locked"] = True
        elif clock_control == "none":
            collection["clocks_locked"] = False
    if pipeline_boost_state:
        collection["pipeline_boost_state"] = pipeline_boost_state

    # These do not all alter the counter values, but they do determine which
    # launch/range was measured and whether the sampling data can be attributed
    # to the target workload.  Keep exact values rather than reconstructing
    # them from a rendered command line later.
    scalar_options = {
        "mode": "mode",
        "devices": "devices",
        "chips": "chips",
        "kernel_name": "kernel-name",
        "kernel_id": "kernel-id",
        "kernel_name_base": "kernel-name-base",
        "launch_count": "launch-count",
        "launch_skip": "launch-skip",
        "target_processes": "target-processes",
        "target_processes_filter": "target-processes-filter",
        "profile_from_start": "profile-from-start",
        "nvtx": "nvtx",
        "range_filter": "range-filter",
        "pm_sampling_interval": "pm-sampling-interval",
        "pm_sampling_buffer_size": "pm-sampling-buffer-size",
        "pm_sampling_max_passes": "pm-sampling-max-passes",
        "warp_sampling_interval": "warp-sampling-interval",
        "communicator": "communicator",
        "communicator_num_peers": "communicator-num-peers",
        "lockstep_kernel_launch": "lockstep-kernel-launch",
        "process_id": "process-id",
    }
    for field, option in scalar_options.items():
        value = _option_state(command, option)
        if value != "":
            collection[field] = value
    for field, option in (
        ("nvtx_include", "nvtx-include"),
        ("nvtx_exclude", "nvtx-exclude"),
        ("lockstep_nvtx_include", "lockstep-nvtx-include"),
        ("lockstep_nvtx_exclude", "lockstep-nvtx-exclude"),
    ):
        values = _option_values(command, option)
        if values:
            collection[field] = values
    for field, option in (
        ("disable_pm_warp_sampling", "disable-pm-warp-sampling"),
        ("import_source", "import-source"),
    ):
        value = _option_state(command, option)
        if value != "":
            collection[field] = value
    for key, value in dict(metadata or {}).items():
        if key in _COLLECTION_METADATA_FIELDS:
            collection[key] = value
    report_path = _report_path_from_export(export_path)
    return (
        collection_manifest_path(export_path),
        {
            "schema_version": _COLLECTION_MANIFEST_SCHEMA_VERSION,
            "tool": "ncu",
            "report_path": str(report_path),
            "command": list(command),
            "collection": collection,
        },
    )


def write_collection_manifest(
    command: list[str], *, metadata: dict[str, Any] | None = None
) -> Path | None:
    """Persist a report sidecar after a successful NCU collection."""
    result = build_collection_manifest(command, metadata=metadata)
    if result is None:
        return None
    path, payload = result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _command_output(args: list[str], *, env: dict[str, str]) -> str:
    """Best-effort command output for provenance; collection never depends on it."""
    try:
        completed = subprocess.run(
            args,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (completed.stdout or completed.stderr or "").strip()


def collect_runtime_metadata(command: list[str], env: dict[str, str]) -> dict[str, Any]:
    """Collect immutable tool/device provenance after a successful NCU run.

    Failure to query either executable is deliberately non-fatal: the profile
    is still valid, but the resulting sidecar makes the missing provenance
    visible instead of inventing it.  ``CUDA_VISIBLE_DEVICES`` is recorded
    verbatim, as ordinal remapping is process-local and cannot be recovered
    safely from a report.
    """
    metadata: dict[str, Any] = {"host_name": socket.gethostname()}
    visible = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if visible:
        metadata["cuda_visible_devices"] = visible

    if command and command[0] == "ncu":
        version = _command_output([command[0], "--version"], env=env)
        if version:
            metadata["ncu_version"] = version.splitlines()[-1].strip()

    rows = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,pci.bus_id",
            "--format=csv,noheader",
        ],
        env=env,
    )
    if rows:
        identities = [line.strip() for line in rows.splitlines() if line.strip()]
        if identities:
            metadata["gpu_identities"] = identities
            driver = identities[0].split(",")
            if len(driver) >= 4:
                metadata["driver_version"] = driver[3].strip()
    return metadata


def _drop_duplicate_options(base_args: list[str], overriding: set[str]) -> list[str]:
    out: list[str] = []
    for item in base_args:
        name = _option_name(item)
        if name and name in overriding:
            continue
        out.append(item)
    return out


def _repeatable_option(name: str, value: Any) -> list[str]:
    opt = f"--{name}"
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                out.append(f"{opt}={text}")
        return out
    text = str(value).strip()
    if not text:
        return []
    return [f"{opt}={text}"]


def _resolve_default_export(payload: dict[str, Any], ncu_cfg: dict[str, Any]) -> str:
    if str(ncu_cfg.get("export", "")).strip():
        return str(ncu_cfg["export"]).strip()
    output_dir = str(payload.get("output_dir", "./logs/ncu")).strip() or "./logs/ncu"
    output_prefix = (
        str(payload.get("output_prefix", "ncu_profile")).strip() or "ncu_profile"
    )
    if output_prefix.endswith(".ncu-rep"):
        file_name = output_prefix
    else:
        file_name = f"{output_prefix}.ncu-rep"
    return str(Path(output_dir) / file_name)


def _build_core_ncu_args(payload: dict[str, Any], ncu_cfg: dict[str, Any]) -> list[str]:
    args: list[str] = []

    scalar_mapped = {
        "mode": "mode",
        "set": "set",
        "kernel_name": "kernel-name",
        "kernel_id": "kernel-id",
        "kernel_name_base": "kernel-name-base",
        "launch_count": "launch-count",
        "launch_skip": "launch-skip",
        "replay_mode": "replay-mode",
        "pipeline_boost_state": "pipeline-boost-state",
        "target_processes": "target-processes",
        "target_processes_filter": "target-processes-filter",
        "profile_from_start": "profile-from-start",
        "nvtx": "nvtx",
        "nvtx_include": "nvtx-include",
        "nvtx_exclude": "nvtx-exclude",
        "range_filter": "range-filter",
        "call_stack": "call-stack",
        "call_stack_type": "call-stack-type",
        "native_include": "native-include",
        "native_exclude": "native-exclude",
        "python_include": "python-include",
        "python_exclude": "python-exclude",
        "devices": "devices",
        "chips": "chips",
        "process_id": "process-id",
        "pm_sampling_interval": "pm-sampling-interval",
        "pm_sampling_buffer_size": "pm-sampling-buffer-size",
        "pm_sampling_max_passes": "pm-sampling-max-passes",
        "warp_sampling_interval": "warp-sampling-interval",
        "communicator": "communicator",
        "communicator_num_peers": "communicator-num-peers",
        "lockstep_kernel_launch": "lockstep-kernel-launch",
        "open_in_ui": "open-in-ui",
        "import_report": "import",
        "export": "export",
        "csv": "csv",
        "log_file": "log-file",
        "page": "page",
        "apply_rules": "apply-rules",
        "print_source": "print-source",
        "print_summary": "print-summary",
        "print_details": "print-details",
        "print_metric_name": "print-metric-name",
    }
    for key, switch_name in scalar_mapped.items():
        value = ncu_cfg.get(key, None)
        if value is None:
            continue
        if isinstance(value, bool):
            text = "on" if value else "off"
        else:
            text = str(value).strip()
        if not text:
            continue
        args.append(f"--{switch_name}={text}")

    for key, switch_name in [
        ("section", "section"),
        ("metrics", "metrics"),
        ("rule", "rule"),
        ("section_folder", "section-folder"),
        ("source_folders", "source-folders"),
    ]:
        args.extend(_repeatable_option(switch_name, ncu_cfg.get(key)))

    return args


def build_command_from_payload(
    payload: dict[str, Any],
    override_command: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    ncu_cfg_raw = payload.get("ncu", payload.get("ncu_launch", {})) or {}
    if not isinstance(ncu_cfg_raw, dict):
        raise ValueError("ncu (or ncu_launch) must be a mapping.")

    command = (
        override_command
        if override_command
        else _coerce_command(payload.get("command"))
    )
    env_updates = _coerce_env(payload.get("env"))

    ncu_cfg = dict(ncu_cfg_raw)
    enabled = bool(ncu_cfg.get("enabled", True))
    ncu_cfg["export"] = _resolve_default_export(payload, ncu_cfg)

    core_args = _build_core_ncu_args(payload, ncu_cfg)
    profile_switches = _coerce_profile_switches(ncu_cfg.get("profile_switches"))
    profile_switch_args = _serialize_switches(profile_switches)
    extra_args = _coerce_extra_args(ncu_cfg.get("extra_args"))

    overriding_names = {
        name for name in (_option_name(item) for item in profile_switch_args) if name
    }
    if overriding_names:
        core_args = _drop_duplicate_options(core_args, overriding_names)

    if not enabled and not profile_switch_args and not extra_args:
        return (command, env_updates)

    ncu_cmd = ["ncu"] + core_args + profile_switch_args + extra_args

    has_import = any(_option_name(item) == "import" for item in ncu_cmd)
    if not command and not has_import:
        raise ValueError(
            "command is required unless an import-based ncu command is configured."
        )

    return (ncu_cmd + command, env_updates)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run command with ncu wrapper from YAML config."
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print final command and exit."
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional command override after '--'.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    override = args.command
    if override and override[0] == "--":
        override = override[1:]
    if not override:
        override = None

    payload = _load_yaml_payload(args.config)
    cmd, env_updates = build_command_from_payload(payload, override_command=override)

    runtime_env = os.environ.copy()
    runtime_env.update(env_updates)

    print("[run_ncu_quick_yaml] Launching:")
    print(" " + " ".join(shlex.quote(item) for item in cmd))
    if env_updates:
        print("[run_ncu_quick_yaml] Env overrides:")
        for key in sorted(env_updates.keys()):
            print(f"  {key}={env_updates[key]}")

    if args.dry_run:
        return 0

    for item in cmd:
        if item.startswith("--export="):
            export_path = item.split("=", 1)[1]
            export_parent = Path(export_path).parent
            if str(export_parent):
                export_parent.mkdir(parents=True, exist_ok=True)
            break

    completed = subprocess.run(cmd, env=runtime_env, check=False)
    if completed.returncode == 0:
        # The exact invocation supplies collection controls; the successful
        # runtime supplies the tool and device identities that an imported NCU
        # report otherwise loses.  Runtime observations win over user-provided
        # hints for fields such as driver/Nsight version.
        metadata = collection_metadata_from_payload(payload)
        metadata.update(collect_runtime_metadata(cmd, runtime_env))
        manifest_path = write_collection_manifest(
            cmd, metadata=metadata
        )
        if manifest_path is not None:
            print(f"[run_ncu_quick_yaml] wrote collection manifest: {manifest_path}")
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
