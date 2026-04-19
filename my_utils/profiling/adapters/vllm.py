from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..metrics.provider_registry import ProviderSpec
from .base import FrameworkAdapter
from .common import (
    build_standard_training_specs,
    context_command_text,
    context_has_any_key,
    is_framework_mismatch,
    normalize_framework_name,
)


class VLLMAdapter(FrameworkAdapter):
    name = "vllm"
    priority = 29

    def detect(self, context: Mapping[str, Any]) -> bool:
        framework = normalize_framework_name(context.get("framework"))
        if is_framework_mismatch(context, ("vllm",)):
            return False
        if framework == "vllm":
            return True
        if context_has_any_key(context, ("vllm_engine", "vllm_llm", "vllm_config")):
            return True

        command = context_command_text(context)
        if not command:
            return False
        return (
            "vllm serve" in command
            or "python -m vllm" in command
            or "python3 -m vllm" in command
            or "vllm.entrypoints" in command
        )

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
        return build_standard_training_specs(context)

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "VLLMAdapter"}
