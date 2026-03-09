from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..metrics.provider_registry import ProviderSpec
from .base import FrameworkAdapter


class PyTorchAdapter(FrameworkAdapter):
    name = "pytorch"
    priority = 10

    def detect(self, context: Mapping[str, Any]) -> bool:
        if context.get("framework") == "pytorch":
            return True
        if "torch_profiler" in context or "profiler" in context:
            return True
        try:
            import torch  # noqa: F401

            return "model" in context or "module" in context
        except Exception:
            return False

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
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
        if "module_profiler" in context:
            specs.append(
                ProviderSpec(
                    provider_type="module_profiler",
                    provider_id="module_profiler",
                    enabled=True,
                    params={},
                )
            )
        return specs

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "PyTorchAdapter"}

