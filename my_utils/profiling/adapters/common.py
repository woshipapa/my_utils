from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

from ..metrics.provider_registry import ProviderSpec


def normalize_framework_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.replace("-", "_")


def is_framework_mismatch(context: Mapping[str, Any], aliases: Sequence[str]) -> bool:
    declared = normalize_framework_name(context.get("framework"))
    if not declared:
        return False
    normalized = {normalize_framework_name(item) for item in aliases}
    return declared not in normalized


def context_has_any_key(context: Mapping[str, Any], keys: Iterable[str]) -> bool:
    return any((key in context and context.get(key) is not None) for key in keys)


def context_command_text(context: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for key in ("command", "cmd", "argv", "launch_command", "launcher_command"):
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                parts.append(value.strip())
            continue
        if isinstance(value, (list, tuple)):
            tokens = [str(item).strip() for item in value if str(item).strip()]
            if tokens:
                parts.append(" ".join(tokens))
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return " ".join(parts).lower()


def build_standard_training_specs(
    context: Mapping[str, Any],
    *,
    include_module_profiler: bool = False,
) -> List[ProviderSpec]:
    specs: List[ProviderSpec] = []
    if "my_timer" in context or "timer" in context:
        specs.append(ProviderSpec(provider_type="my_timer", provider_id="my_timer", enabled=True, params={}))
    if "torch_profiler" in context or "profiler" in context:
        specs.append(
            ProviderSpec(
                provider_type="torch_profiler",
                provider_id="torch_profiler",
                enabled=True,
                params={"include_memory": True, "include_flops": True},
            )
        )
    if include_module_profiler and "module_profiler" in context:
        specs.append(
            ProviderSpec(
                provider_type="module_profiler",
                provider_id="module_profiler",
                enabled=True,
                params={},
            )
        )
    return specs
