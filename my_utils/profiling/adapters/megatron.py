from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..metrics.provider_registry import ProviderSpec
from .base import FrameworkAdapter


class MegatronAdapter(FrameworkAdapter):
    name = "megatron"
    priority = 40

    def detect(self, context: Mapping[str, Any]) -> bool:
        if context.get("framework") == "megatron":
            return True
        if "megatron_args" in context:
            return True
        return bool(context.get("model_provider_func") is not None and context.get("forward_step_func") is not None)

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
        return specs

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "MegatronAdapter"}

