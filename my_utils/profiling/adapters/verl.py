# SPDX-License-Identifier: Apache-2.0
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


class VerlAdapter(FrameworkAdapter):
    name = "verl"
    priority = 25

    def detect(self, context: Mapping[str, Any]) -> bool:
        framework = normalize_framework_name(context.get("framework"))
        if is_framework_mismatch(context, ("verl",)):
            return False
        if framework == "verl":
            return True
        if context_has_any_key(
            context, ("verl_trainer", "verl_config", "verl_engine", "verl_args")
        ):
            return True

        command = context_command_text(context)
        if not command:
            return False
        return (
            "verl.trainer.main_ppo" in command
            or "-m verl.trainer" in command
            or "python -m verl" in command
            or "python3 -m verl" in command
        )

    def build_provider_specs(self, context: Mapping[str, Any]) -> List[ProviderSpec]:
        return build_standard_training_specs(context)

    def build_runtime_tags(self, context: Mapping[str, Any]) -> Dict[str, str]:
        return {"framework": self.name, "adapter": "VerlAdapter"}
