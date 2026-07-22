from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..metrics.provider_registry import ProviderSpec
from .base import FrameworkAdapter
from .common import (
    build_standard_training_specs,
    is_framework_mismatch,
    normalize_framework_name,
)


class DeepSpeedAdapter(FrameworkAdapter):
    name = "deepspeed"
    priority = 30

    def detect(self, context: Mapping[str, Any]) -> bool:
        framework = normalize_framework_name(context.get("framework"))
        if is_framework_mismatch(context, ("deepspeed",)):
            return False
        if framework == "deepspeed":
            return True
        if "deepspeed_engine" in context:
            return True
        try:
            import deepspeed  # noqa: F401

            return bool(context.get("engine") is not None)
        except Exception:
            return False

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
        return build_standard_training_specs(context)

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "DeepSpeedAdapter"}
