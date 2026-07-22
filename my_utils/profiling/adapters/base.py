from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from ..metrics.provider_registry import ProviderSpec


@dataclass
class AdapterContext:
    objects: Dict[str, Any] = field(default_factory=dict)
    runtime_tags: Dict[str, str] = field(default_factory=dict)


class FrameworkAdapter:
    name = "framework"
    priority = 100

    def detect(self, context: Mapping[str, Any]) -> bool:
        return False

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
        return []

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name}
