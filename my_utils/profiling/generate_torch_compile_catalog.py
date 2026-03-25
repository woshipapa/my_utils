#!/usr/bin/env python
from __future__ import annotations

import ast
import argparse
import inspect
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(torch.__file__).resolve().parent
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR / "torch_compile_catalog.snapshot.yaml"
VERSION_INDEX_OUTPUT = THIS_DIR / "torch_compile_catalog_versions.yaml"


def _sanitize_version(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", value)


def _load_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_env_vars(path: Path) -> list[str]:
    pattern = re.compile(r'os\.environ(?:\.get)?\(["\']([^"\']+)["\']')
    source = _load_source(path)
    values = sorted({m.group(1) for m in pattern.finditer(source)})
    return values


def _extract_env_vars_from_source(source: str) -> list[str]:
    pattern = re.compile(r'os\.environ(?:\.get)?\(["\']([^"\']+)["\']')
    return sorted({m.group(1) for m in pattern.finditer(source)})


def _extract_top_level_assignments(path: Path) -> list[str]:
    module = ast.parse(_load_source(path), filename=str(path))
    names: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                names.append(target.id)
    return names


def _extract_top_level_assignments_from_source(source: str, filename: str) -> list[str]:
    module = ast.parse(source, filename=filename)
    names: list[str] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                names.append(target.id)
    return names


def _split_backends(all_backends: list[str], stable_backends: list[str]) -> dict[str, list[str]]:
    stable = sorted(stable_backends)
    stable_set = set(stable)
    experimental = sorted([name for name in all_backends if name not in stable_set])
    return {
        "stable": stable,
        "experimental_or_debug": experimental,
    }


def _build_payload() -> dict[str, Any]:
    dynamo_config = ROOT / "_dynamo" / "config.py"
    inductor_config = ROOT / "_inductor" / "config.py"

    stable_backends = list(torch._dynamo.list_backends())
    all_backends = list(torch._dynamo.list_backends(None))
    mode_options = torch._inductor.list_mode_options()
    inductor_option_names = sorted(torch._inductor.list_options())

    payload: dict[str, Any] = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "my_utils.profiling.generate_torch_compile_catalog",
            "catalog_kind": "runtime_generated",
            "torch_version": torch.__version__,
            "python_compile_signature": str(inspect.signature(torch.compile)),
            "references": [
                "https://docs.pytorch.org/docs/stable/generated/torch.compile.html",
                "https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html",
                "https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html",
            ],
        },
        "runtime_discovery": {
            "backends": _split_backends(all_backends, stable_backends),
            "mode_options": mode_options,
            "inductor_option_names": inductor_option_names,
            "inductor_option_count": len(inductor_option_names),
        },
        "source_discovery": {
            "dynamo_config_flags": sorted(set(_extract_top_level_assignments(dynamo_config))),
            "dynamo_env_vars": _extract_env_vars(dynamo_config),
            "inductor_env_vars": _extract_env_vars(inductor_config),
        },
        "documented_debug_env_vars": [
            {
                "name": "TORCH_LOGS",
                "scope": "official troubleshooting docs",
                "purpose": "Enable compiler logging artifacts such as graph_breaks, guards, recompiles, dynamic, perf_hints.",
            },
            {
                "name": "TORCH_TRACE",
                "scope": "official troubleshooting docs",
                "purpose": "Collect structured compiler traces for tlparse.",
            },
            {
                "name": "TORCH_COMPILE_DEBUG",
                "scope": "official docs + inductor source",
                "purpose": "Enable richer compiler debug dumps.",
            },
        ],
    }
    return payload


def _http_get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_latest_pypi_version() -> str:
    payload = json.loads(_http_get_text("https://pypi.org/pypi/torch/json"))
    return str(payload["info"]["version"])


def _extract_mode_options_from_source(source: str) -> dict[str, Any]:
    module = ast.parse(source, filename="<inductor.__init__>")
    dict_assignments: dict[str, Any] = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        dict_assignments[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name):
                try:
                    dict_assignments[target.id] = ast.literal_eval(node.value)
                except Exception:
                    pass
    for node in ast.walk(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "mode_options":
                    if isinstance(node.value, ast.Dict):
                        result: dict[str, Any] = {}
                        for key_node, value_node in zip(node.value.keys, node.value.values):
                            key = ast.literal_eval(key_node)
                            if isinstance(value_node, ast.Name) and value_node.id in dict_assignments:
                                result[key] = dict_assignments[value_node.id]
                            else:
                                result[key] = ast.literal_eval(value_node)
                        return result
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "mode_options":
                if isinstance(node.value, ast.Dict):
                    result: dict[str, Any] = {}
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        key = ast.literal_eval(key_node)
                        if isinstance(value_node, ast.Name) and value_node.id in dict_assignments:
                            result[key] = dict_assignments[value_node.id]
                        else:
                            result[key] = ast.literal_eval(value_node)
                    return result
    return {}


def _build_upstream_payload(version: str) -> dict[str, Any]:
    tag = f"v{version}"
    dynamo_config_url = f"https://raw.githubusercontent.com/pytorch/pytorch/{tag}/torch/_dynamo/config.py"
    inductor_config_url = f"https://raw.githubusercontent.com/pytorch/pytorch/{tag}/torch/_inductor/config.py"
    inductor_init_url = f"https://raw.githubusercontent.com/pytorch/pytorch/{tag}/torch/_inductor/__init__.py"
    compile_doc_url = "https://docs.pytorch.org/docs/stable/generated/torch.compile.html"
    dynamic_doc_url = "https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html"
    troubleshooting_doc_url = "https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html"

    dynamo_source = _http_get_text(dynamo_config_url)
    inductor_source = _http_get_text(inductor_config_url)
    inductor_init_source = _http_get_text(inductor_init_url)

    mode_options = _extract_mode_options_from_source(inductor_init_source)
    inductor_option_names = sorted(set(_extract_top_level_assignments_from_source(inductor_source, inductor_config_url)))
    dynamo_flag_names = sorted(set(_extract_top_level_assignments_from_source(dynamo_source, dynamo_config_url)))

    payload: dict[str, Any] = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator": "my_utils.profiling.generate_torch_compile_catalog",
            "catalog_kind": "upstream_source_generated",
            "torch_version": version,
            "source_tag": tag,
            "references": [
                compile_doc_url,
                dynamic_doc_url,
                troubleshooting_doc_url,
                dynamo_config_url,
                inductor_config_url,
                inductor_init_url,
            ],
            "notes": [
                "This catalog is generated from upstream source and official docs, not from a local runtime import of that version.",
                "Backend availability and some runtime-only discoveries may differ from a real installed build.",
            ],
        },
        "runtime_discovery": {
            "mode_options": mode_options,
            "inductor_option_names": inductor_option_names,
            "inductor_option_count": len(inductor_option_names),
        },
        "source_discovery": {
            "dynamo_config_flags": dynamo_flag_names,
            "dynamo_env_vars": _extract_env_vars_from_source(dynamo_source),
            "inductor_env_vars": _extract_env_vars_from_source(inductor_source),
        },
        "documented_debug_env_vars": [
            {
                "name": "TORCH_LOGS",
                "scope": "official troubleshooting docs",
                "purpose": "Enable compiler logging artifacts such as graph_breaks, guards, recompiles, dynamic, perf_hints.",
            },
            {
                "name": "TORCH_TRACE",
                "scope": "official troubleshooting docs",
                "purpose": "Collect structured compiler traces for tlparse.",
            },
            {
                "name": "TORCH_COMPILE_DEBUG",
                "scope": "official docs + inductor source",
                "purpose": "Enable richer compiler debug dumps.",
            },
        ],
    }
    return payload


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "{", "}", "[", "]", ",", "\n", '"']) or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text


def _to_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.extend(_to_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines if lines else [f"{prefix}{{}}"]
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.extend(_to_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines if lines else [f"{prefix}[]"]
    return [f"{prefix}{_yaml_scalar(value)}"]


def _dump_yaml(payload: dict[str, Any]) -> str:
    return "\n".join(_to_yaml_lines(payload)) + "\n"


def _load_existing_yaml_like(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _update_version_index(entry: dict[str, Any]) -> None:
    payload = _load_existing_yaml_like(VERSION_INDEX_OUTPUT)
    payload.setdefault("catalogs", [])
    catalogs = [item for item in payload["catalogs"] if not (item.get("catalog_kind") == entry["catalog_kind"] and item.get("torch_version") == entry["torch_version"])]
    catalogs.append(entry)
    catalogs.sort(key=lambda item: (str(item.get("catalog_kind", "")), str(item.get("torch_version", ""))))
    payload["catalogs"] = catalogs
    VERSION_INDEX_OUTPUT.write_text(_dump_yaml(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate torch.compile config catalogs.")
    parser.add_argument(
        "--catalog-kind",
        choices=("runtime", "upstream", "latest-upstream"),
        default="runtime",
    )
    parser.add_argument("--torch-version", default=None, help="Used with --catalog-kind=upstream.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.catalog_kind == "runtime":
        payload = _build_payload()
        version = payload["metadata"]["torch_version"]
        output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
        versioned_output = THIS_DIR / f"torch_compile_catalog.torch-{_sanitize_version(version)}.snapshot.yaml"
    else:
        version = args.torch_version or _fetch_latest_pypi_version()
        payload = _build_upstream_payload(version)
        if args.catalog_kind == "latest-upstream":
            output_path = Path(args.output) if args.output else THIS_DIR / "torch_compile_catalog.latest-upstream.snapshot.yaml"
        else:
            output_path = Path(args.output) if args.output else THIS_DIR / f"torch_compile_catalog.upstream-{_sanitize_version(version)}.snapshot.yaml"
        versioned_output = THIS_DIR / f"torch_compile_catalog.upstream-{_sanitize_version(version)}.snapshot.yaml"

    text = _dump_yaml(payload)
    output_path.write_text(text, encoding="utf-8")
    if versioned_output != output_path:
        versioned_output.write_text(text, encoding="utf-8")
    _update_version_index(
        {
            "catalog_kind": payload["metadata"].get("catalog_kind", "runtime_generated"),
            "torch_version": version,
            "primary_file": str(output_path.name),
            "versioned_file": str(versioned_output.name),
            "generated_at_utc": payload["metadata"]["generated_at_utc"],
        }
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {versioned_output}")


if __name__ == "__main__":
    main()
