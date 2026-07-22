# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from my_utils.profiling import MetricsCollector
from my_utils.profiling.adapters import build_default_adapter_registry


def test_default_registry_contains_extended_framework_adapters() -> None:
    registry = build_default_adapter_registry()
    names = registry.list_adapters()
    expected = {
        "pytorch",
        "huggingface",
        "deepspeed",
        "megatron",
        "torchtitan",
        "verl",
        "slime",
        "roll",
        "sglang",
        "vllm",
    }
    assert expected.issubset(set(names))


@pytest.mark.parametrize(
    ("framework", "expected"),
    [
        ("torchtitan", "torchtitan"),
        ("verl", "verl"),
        ("slime", "slime"),
        ("roll", "roll"),
        ("sglang", "sglang"),
        ("vllm", "vllm"),
    ],
)
def test_auto_setup_selects_explicit_framework_adapter(
    tmp_path: Path, framework: str, expected: str
) -> None:
    registry = build_default_adapter_registry()
    collector = MetricsCollector(output_dir=str(tmp_path / framework))
    result = registry.auto_setup_collector(
        collector,
        context={
            "framework": framework,
            "my_timer": object(),
            "torch_profiler": object(),
        },
    )
    assert result["selected_adapter"] == expected
    assert set(collector.list_providers()) == {"my_timer", "torch_profiler"}


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["env", "MODULE=llama3", "CONFIG=llama3_8b", "./run_train.sh"], "torchtitan"),
        (["python3", "-m", "verl.trainer.main_ppo"], "verl"),
        (["python3", "-m", "sglang.launch_server", "--port", "30000"], "sglang"),
        (["vllm", "serve", "Qwen/Qwen2.5-1.5B-Instruct"], "vllm"),
    ],
)
def test_auto_setup_detects_framework_from_command(
    tmp_path: Path, command: list[str], expected: str
) -> None:
    registry = build_default_adapter_registry()
    collector = MetricsCollector(output_dir=str(tmp_path / "cmd"))
    result = registry.auto_setup_collector(
        collector,
        context={
            "command": command,
            "my_timer": object(),
        },
    )
    assert result["selected_adapter"] == expected
    assert collector.list_providers() == ["my_timer"]


def test_pytorch_adapter_does_not_override_explicit_framework(tmp_path: Path) -> None:
    registry = build_default_adapter_registry()
    collector = MetricsCollector(output_dir=str(tmp_path / "override"))
    result = registry.auto_setup_collector(
        collector,
        context={
            "framework": "torchtitan",
            "model": object(),
            "torch_profiler": object(),
        },
    )
    assert result["selected_adapter"] == "torchtitan"
