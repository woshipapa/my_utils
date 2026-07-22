# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..pipeline.metrics_collector import MetricsCollector
from .base import FrameworkAdapter
from .deepspeed import DeepSpeedAdapter
from .huggingface import HuggingFaceAdapter
from .megatron import MegatronAdapter
from .pytorch import PyTorchAdapter
from .roll import RollAdapter
from .sglang import SGLangAdapter
from .slime import SlimeAdapter
from .torchtitan import TorchTitanAdapter
from .verl import VerlAdapter
from .vllm import VLLMAdapter


class FrameworkAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: List[FrameworkAdapter] = []

    def register(self, adapter: FrameworkAdapter) -> None:
        self._adapters.append(adapter)
        self._adapters.sort(key=lambda item: item.priority)

    def list_adapters(self) -> List[str]:
        return [adapter.name for adapter in self._adapters]

    def detect(self, context: Mapping[str, Any]) -> List[FrameworkAdapter]:
        return [adapter for adapter in self._adapters if adapter.detect(context)]

    def auto_setup_collector(
        self,
        collector: MetricsCollector,
        *,
        context: Optional[Mapping[str, Any]] = None,
        prefer: str = "",
    ) -> Dict[str, object]:
        payload = dict(context or {})
        matched = self.detect(payload)
        if prefer:
            matched = [adapter for adapter in matched if adapter.name == prefer] + [
                adapter for adapter in matched if adapter.name != prefer
            ]
        if not matched:
            return {"matched_adapters": [], "registered_providers": []}

        chosen = matched[0]
        specs = chosen.build_provider_specs(payload)
        collector.register_providers_from_specs(
            [
                {
                    "type": spec.provider_type,
                    "id": spec.provider_id,
                    "enabled": spec.enabled,
                    "params": spec.params,
                }
                for spec in specs
            ],
            provider_context=payload,
            ignore_errors=True,
        )
        return {
            "matched_adapters": [item.name for item in matched],
            "selected_adapter": chosen.name,
            "registered_providers": collector.list_providers(),
            "runtime_tags": chosen.build_runtime_tags(payload),
        }


def build_default_adapter_registry() -> FrameworkAdapterRegistry:
    registry = FrameworkAdapterRegistry()
    registry.register(TorchTitanAdapter())
    registry.register(HuggingFaceAdapter())
    registry.register(VerlAdapter())
    registry.register(SlimeAdapter())
    registry.register(RollAdapter())
    registry.register(SGLangAdapter())
    registry.register(VLLMAdapter())
    registry.register(DeepSpeedAdapter())
    registry.register(MegatronAdapter())
    registry.register(PyTorchAdapter())
    return registry


DEFAULT_ADAPTER_REGISTRY = build_default_adapter_registry()
