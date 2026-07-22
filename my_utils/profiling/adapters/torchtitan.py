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


class TorchTitanAdapter(FrameworkAdapter):
    name = "torchtitan"
    priority = 11

    def detect(self, context: Mapping[str, Any]) -> bool:
        framework = normalize_framework_name(context.get("framework"))
        if is_framework_mismatch(context, ("torchtitan", "torch_titan", "titan")):
            return False
        if framework in ("torchtitan", "torch_titan", "titan"):
            return True
        if context_has_any_key(
            context, ("torchtitan_config", "torchtitan_job_config", "job_config")
        ):
            return True

        command = context_command_text(context)
        if not command:
            return False
        if "torchtitan" in command:
            return True
        if "run_train.sh" in command and (
            "module=" in command or "config=" in command or "--job.config" in command
        ):
            return True
        return False

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
        return build_standard_training_specs(context)

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "TorchTitanAdapter"}
