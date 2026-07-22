#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


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
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
