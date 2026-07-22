#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import shlex
import subprocess
import sys
import types
from pathlib import Path
from typing import Any


def _load_runtime_symbols() -> tuple[type[Any], Any]:
    try:
        from my_utils.profiling.runtime.config import NsysLaunchConfig as _Cfg
        from my_utils.profiling.runtime.frameworkless import (
            build_nsys_launch_prefix as _Builder,
        )

        return _Cfg, _Builder
    except Exception:
        root = Path(__file__).resolve().parents[1] / "runtime"
        for name, path in [
            ("my_utils", str(root.parents[2])),
            ("my_utils.profiling", str(root.parents[1])),
            ("my_utils.profiling.runtime", str(root)),
        ]:
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = [path]  # type: ignore[attr-defined]
                sys.modules[name] = mod

        for mod_name, file_name in [
            ("my_utils.profiling.runtime.backends", "backends.py"),
            ("my_utils.profiling.runtime.config", "config.py"),
            ("my_utils.profiling.runtime.frameworkless", "frameworkless.py"),
        ]:
            if mod_name in sys.modules:
                continue
            spec = importlib.util.spec_from_file_location(mod_name, root / file_name)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to load runtime module: {mod_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)

        cfg_mod = sys.modules["my_utils.profiling.runtime.config"]
        fw_mod = sys.modules["my_utils.profiling.runtime.frameworkless"]
        return cfg_mod.NsysLaunchConfig, fw_mod.build_nsys_launch_prefix


NsysLaunchConfig, build_nsys_launch_prefix = _load_runtime_symbols()


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
    if isinstance(value, str):
        parts = shlex.split(value.strip())
        if not parts:
            raise ValueError("command string is empty.")
        return parts
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, (str, int, float)) for item in value)
    ):
        return [str(item) for item in value]
    raise ValueError("command must be a non-empty string or list.")


def _coerce_env(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("env must be a mapping.")
    return {str(key): str(val) for key, val in value.items()}


def _coerce_extra_profile_args(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("nsys_launch.extra_profile_args must be a list.")
    return [str(item) for item in value if str(item).strip()]


def _normalize_switch_name(name: str) -> str:
    normalized = str(name).strip()
    if normalized.startswith("--"):
        normalized = normalized[2:]
    return normalized.replace("_", "-")


def _coerce_profile_switches(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("nsys_launch.profile_switches must be a mapping.")
    out: dict[str, Any] = {}
    for key, item in value.items():
        name = _normalize_switch_name(str(key))
        if not name:
            continue
        out[name] = item
    return out


def _serialize_profile_switches(profile_switches: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for name, value in profile_switches.items():
        opt = f"--{name}"
        if value is None:
            continue
        if isinstance(value, bool):
            args.append(f"{opt}={'true' if value else 'false'}")
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
            f"Unsupported value type for nsys switch '{name}': {type(value).__name__}"
        )
    return args


def _option_name(arg: str) -> str:
    if not arg.startswith("--"):
        return ""
    return arg.split("=", 1)[0][2:]


def _drop_prefix_options(prefix: list[str], option_names: set[str]) -> list[str]:
    if len(prefix) < 2:
        return prefix
    out = prefix[:2]
    for item in prefix[2:]:
        name = _option_name(item)
        if name and name in option_names:
            continue
        out.append(item)
    return out


def _resolve_output(
    default_output_dir: str, default_output_prefix: str, cfg_dict: dict[str, Any]
) -> str:
    output = str(cfg_dict.get("output", "")).strip()
    if output:
        return output
    output_dir = (
        str(cfg_dict.get("output_dir", default_output_dir)).strip()
        or default_output_dir
    )
    output_prefix = (
        str(cfg_dict.get("output_prefix", default_output_prefix)).strip()
        or default_output_prefix
    )
    return str(Path(output_dir) / output_prefix)


def build_command_from_payload(
    payload: dict[str, Any],
    override_command: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    nsys_raw = payload.get("nsys_launch", {}) or {}
    if not isinstance(nsys_raw, dict):
        raise ValueError("nsys_launch must be a mapping.")

    command = (
        override_command
        if override_command
        else _coerce_command(payload.get("command"))
    )
    env_updates = _coerce_env(payload.get("env"))
    extra_profile_args = _coerce_extra_profile_args(nsys_raw.get("extra_profile_args"))
    profile_switches = _coerce_profile_switches(nsys_raw.get("profile_switches"))
    profile_switch_args = _serialize_profile_switches(profile_switches)
    explicit_option_names = {
        name for name in (_option_name(item) for item in profile_switch_args) if name
    }

    launch_fields = {item.name for item in dataclasses.fields(NsysLaunchConfig)}
    launch_kwargs: dict[str, Any] = {}
    for key, value in nsys_raw.items():
        if key in launch_fields:
            launch_kwargs[key] = value

    launch_kwargs.setdefault("enabled", True)
    launch_kwargs["output"] = _resolve_output(
        default_output_dir=str(payload.get("output_dir", "./logs/nsys")),
        default_output_prefix=str(payload.get("output_prefix", "nsys_profile_%p")),
        cfg_dict=launch_kwargs,
    )

    nsys_cfg = NsysLaunchConfig(**launch_kwargs)
    prefix = build_nsys_launch_prefix(nsys_cfg)
    if prefix and explicit_option_names:
        prefix = _drop_prefix_options(prefix, explicit_option_names)
    if prefix:
        return (
            prefix + profile_switch_args + extra_profile_args + command,
            env_updates,
        )
    if profile_switch_args or extra_profile_args:
        return (
            ["nsys", "profile"] + profile_switch_args + extra_profile_args + command,
            env_updates,
        )
    return (command, env_updates)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run command with nsys profile wrapper from YAML config.",
    )
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print final command and exit without running.",
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

    override_command = args.command
    if override_command and override_command[0] == "--":
        override_command = override_command[1:]
    if not override_command:
        override_command = None

    payload = _load_yaml_payload(args.config)
    cmd, env_updates = build_command_from_payload(
        payload, override_command=override_command
    )

    runtime_env = os.environ.copy()
    runtime_env.update(env_updates)

    print("[run_nsys_quick_yaml] Launching:")
    print(" " + " ".join(shlex.quote(item) for item in cmd))
    if env_updates:
        print("[run_nsys_quick_yaml] Env overrides:")
        for key in sorted(env_updates.keys()):
            print(f"  {key}={env_updates[key]}")

    if args.dry_run:
        return 0

    for item in cmd:
        if item.startswith("--output="):
            output_path = item.split("=", 1)[1]
            output_parent = Path(output_path).parent
            if str(output_parent):
                output_parent.mkdir(parents=True, exist_ok=True)
            break

    completed = subprocess.run(cmd, env=runtime_env, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
