from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..metrics.provider_registry import ProviderSpec
from .base import FrameworkAdapter
from .common import (
    build_standard_training_specs,
    is_framework_mismatch,
    normalize_framework_name,
)


class PyTorchAdapter(FrameworkAdapter):
    name = "pytorch"
    priority = 90

    def detect(self, context: Mapping[str, Any]) -> bool:
        framework = normalize_framework_name(context.get("framework"))
        if is_framework_mismatch(context, ("pytorch", "torch")):
            return False
        if framework in ("pytorch", "torch"):
            return True
        if "torch_profiler" in context or "profiler" in context:
            return True
        try:
            import torch  # noqa: F401

            return "model" in context or "module" in context
        except Exception:
            return False

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
        return build_standard_training_specs(context, include_module_profiler=True)

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "PyTorchAdapter"}
