from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..metrics.provider_registry import ProviderSpec
from .base import FrameworkAdapter
from .common import (
    build_standard_training_specs,
    is_framework_mismatch,
    normalize_framework_name,
)


class MegatronAdapter(FrameworkAdapter):
    name = "megatron"
    priority = 40

    def detect(self, context: Mapping[str, Any]) -> bool:
        framework = normalize_framework_name(context.get("framework"))
        if is_framework_mismatch(context, ("megatron",)):
            return False
        if framework == "megatron":
            return True
        if "megatron_args" in context:
            return True
        return bool(
            context.get("model_provider_func") is not None
            and context.get("forward_step_func") is not None
        )

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
        return build_standard_training_specs(context)

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "MegatronAdapter"}
