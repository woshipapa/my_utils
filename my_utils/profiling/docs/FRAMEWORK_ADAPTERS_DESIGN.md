# Framework Adapters Design

> Status: shipped ("P4 Framework adapter SDK + registry" in [ROADMAP.md](./ROADMAP.md)).
> This document originally predated the implementation; it has been revised to describe
> the code that actually shipped in `my_utils/profiling/adapters/`. Where the shipped
> design deliberately diverges from the original proposal, the rationale is kept.
>
> Related docs: [UNIFIED_METRICS_DESIGN.md](./UNIFIED_METRICS_DESIGN.md) (metrics/provider
> layer), [FRAMEWORK_INTEGRATION_PLAYBOOK.md](./FRAMEWORK_INTEGRATION_PLAYBOOK.md)
> (per-framework integration recipes).

## Design Goals

1. **Zero-intrusion integration** - usable without modifying framework source code
2. **Auto-detection** - identify the framework in use from the caller-supplied context
3. **Unified interface** - every adapter implements the same small interface
4. **On-demand providers** - only register the metrics providers that the current
   context can actually feed

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 FrameworkAdapterRegistry                     │
│  Holds all adapters, sorted by priority                      │
│  - detect(context)        -> matching adapters               │
│  - auto_setup_collector() -> register providers on a         │
│                              MetricsCollector                │
└─────────────────────────────────────────────────────────────┘
                              │
   ┌──────────┬──────────┬────┴─────┬──────────┬──────────┐
   ▼          ▼          ▼          ▼          ▼          ▼
torchtitan huggingface  verl      slime      roll      sglang
   ▼          ▼          ▼          ▼
 vllm     deepspeed   megatron   pytorch      (10 built-in adapters)
   │          │          │          │
   └──────────┴────┬─────┴──────────┘
                   ▼
        ┌────────────────────┐
        │  FrameworkAdapter   │  base class (adapters/base.py)
        │  - name, priority   │
        │  - detect()         │
        │  - build_provider_  │
        │    specs()          │
        │  - build_runtime_   │
        │    tags()           │
        └────────────────────┘
                   │  emits ProviderSpec list
                   ▼
        ┌────────────────────┐      ┌────────────────────┐
        │ MetricsProvider     │      │  MetricsCollector   │
        │ Registry            │─────▶│  (pipeline/         │
        │ (metrics/provider_  │      │   metrics_collector │
        │  registry.py)       │      │   .py)              │
        └────────────────────┘      └────────────────────┘
```

Compared with the original proposal, the shipped design makes two deliberate
simplifications:

- **Detection is context-driven, not import-driven.** The original proposal
  detected frameworks by attempting `__import__` on candidate module paths.
  That approach produces false positives (e.g. `transformers` being installed
  does not mean the current job is a HuggingFace Trainer run) and imports heavy
  packages as a side effect. The shipped `detect()` inspects an explicit
  context mapping supplied by the caller: a declared `framework` name,
  characteristic context objects (e.g. `trainer`, `deepspeed_engine`,
  `megatron_args`), or the launch command line.
- **Adapters are declarative, not hook-based.** The original proposal gave each
  adapter a large lifecycle-hook surface (`get_training_loop_hooks`,
  forward/backward hooks, Trainer callbacks). In the shipped design an adapter
  only *declares* which metrics providers make sense for the detected
  framework, as a list of `ProviderSpec` entries; the actual instrumentation
  lives in the providers themselves (timer, torch profiler, module profiler,
  file-based sources), which pull their inputs from the same context mapping.
  This keeps adapters tiny (~40 lines each) and keeps all measurement logic in
  one layer.

## 1. Adapter Base Class

Real implementation in `my_utils/profiling/adapters/base.py`:

```python
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
```

The interface is intentionally minimal:

- `name` - stable lowercase identifier (`"megatron"`, `"vllm"`, ...). Used for
  the `prefer=` override in the registry and as the `framework` runtime tag.
- `priority` - lower numbers are tried first. Specific frameworks get low
  numbers (torchtitan=11, huggingface=20, verl=25, slime=26, roll=27,
  sglang=28, vllm=29, deepspeed=30, megatron=40); the generic PyTorch adapter
  is the fallback at 90.
- `detect(context)` - returns `True` if the context looks like this framework.
  Must be cheap and side-effect free.
- `build_provider_specs(context)` - returns the `ProviderSpec` list to register
  on the collector. Specs are only emitted for providers whose required context
  objects are present, so a missing profiler simply means no
  `torch_profiler` provider rather than an error.
- `build_runtime_tags(context)` - tags describing the run (at minimum
  `{"framework": name}`); returned to the caller by
  `auto_setup_collector()` so they can be attached to collected events.

`AdapterContext` is a small convenience container for callers that want to
bundle context objects and runtime tags together; the adapter API itself
accepts any `Mapping[str, Any]`.

### Shared detection helpers

`my_utils/profiling/adapters/common.py` provides the helpers every built-in
adapter uses:

- `normalize_framework_name(value)` - lowercase, `-` -> `_`.
- `is_framework_mismatch(context, aliases)` - if the caller *declared* a
  `framework` in the context and it is not one of this adapter's aliases,
  detection must fail. An explicit declaration always wins over heuristics.
- `context_has_any_key(context, keys)` - true if any key is present and
  non-`None`.
- `context_command_text(context)` - joins `command` / `cmd` / `argv` /
  `launch_command` / `launcher_command` into one lowercase string, so adapters
  can match launch-command patterns (e.g. `"vllm serve"`).
- `build_standard_training_specs(context, include_module_profiler=False)` -
  the shared spec builder. It emits:
  - a `my_timer` spec when `my_timer`/`timer` is in the context,
  - a `torch_profiler` spec (with `include_memory` and `include_flops` params)
    when `torch_profiler`/`profiler` is in the context,
  - optionally a `module_profiler` spec when `module_profiler` is in the
    context.

## 2. Built-in Adapters

Ten adapters ship in `my_utils/profiling/adapters/`:

| Adapter class | `name` | Priority | Detection signals besides `framework=` |
|---|---|---|---|
| `TorchTitanAdapter` | `torchtitan` | 11 | `torchtitan_config`/`torchtitan_job_config`/`job_config` keys; `torchtitan` or `run_train.sh` in the launch command |
| `HuggingFaceAdapter` | `huggingface` | 20 | `hf_trainer`/`transformers_trainer` keys; a `trainer` object whose class name contains `Trainer` |
| `VerlAdapter` | `verl` | 25 | `verl_trainer`/`verl_config`/`verl_engine`/`verl_args` keys; `verl.trainer.main_ppo` / `python -m verl` in the command |
| `SlimeAdapter` | `slime` | 26 | `slime_runner`/`slime_config`/`slime_trainer` keys; slime paths in the command |
| `RollAdapter` | `roll` | 27 | `roll_config`/`roll_pipeline`/`roll_runner` keys; ROLL pipeline scripts in the command |
| `SGLangAdapter` | `sglang` | 28 | `sglang_server`/`sglang_runtime` keys; `sglang.launch_server` / `sglang serve` in the command |
| `VLLMAdapter` | `vllm` | 29 | `vllm_engine`/`vllm_llm`/`vllm_config` keys; `vllm serve` / `vllm.entrypoints` in the command |
| `DeepSpeedAdapter` | `deepspeed` | 30 | `deepspeed_engine` key; importable `deepspeed` plus an `engine` object |
| `MegatronAdapter` | `megatron` | 40 | `megatron_args` key; both `model_provider_func` and `forward_step_func` present |
| `PyTorchAdapter` | `pytorch` | 90 | `torch_profiler`/`profiler` keys; importable `torch` plus a `model`/`module` object (generic fallback) |

All ten currently delegate to `build_standard_training_specs()`; only
`PyTorchAdapter` also opts into the `module_profiler` spec. This is by design:
framework-specific *measurement* belongs in providers, and an adapter can grow
framework-specific specs later without changing the interface.

Example - the real Megatron adapter (`adapters/megatron.py`), in full:

```python
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
```

The original proposal's per-framework lifecycle hooks (Megatron
`forward_step_start`/`forward_step_end` hooks, DeepSpeed `set_engine()` and
gradient-accumulation hooks, a HuggingFace `TrainerCallback`) were not built.
Step-scoped collection is instead done by calling
`MetricsCollector.collect(step=...)` from the training loop, and per-framework
wiring recipes live in
[FRAMEWORK_INTEGRATION_PLAYBOOK.md](./FRAMEWORK_INTEGRATION_PLAYBOOK.md).

## 3. Adapter Registry

Real implementation in `my_utils/profiling/adapters/registry.py`:

```python
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
        ...
```

Key behaviors of `auto_setup_collector()`:

1. Runs `detect()` over all registered adapters (priority order).
2. If `prefer="<name>"` is given, that adapter is moved to the front of the
   matched list - useful when several adapters plausibly match.
3. The **first** matched adapter is selected; its provider specs are registered
   on the collector via
   `MetricsCollector.register_providers_from_specs(..., provider_context=context,
   ignore_errors=True)`, so a provider whose context object is missing degrades
   to a bootstrap warning instead of an exception.
4. Returns a summary dict:
   `{"matched_adapters": [...], "selected_adapter": "...",
   "registered_providers": [...], "runtime_tags": {...}}`.
   With no match it returns
   `{"matched_adapters": [], "registered_providers": []}`.

Unlike the original proposal, the registry is instance-based rather than a
class-level singleton with global state, and it selects a single adapter per
setup rather than setting up every detected framework (one job is profiled as
one framework; mixed stacks pick the most specific adapter by priority or via
`prefer=`).

A module-level default is provided:

```python
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
```

## 4. Provider Layer

The original proposal sketched framework-specific provider modules
(`profiling/providers/megatron.py` with timer/memory/communication/parallel
providers, `providers/deepspeed.py`, `providers/huggingface.py`). Those modules
were not built. The shipped provider layer is framework-neutral and lives in
`my_utils/profiling/metrics/`:

- `metrics/metrics_providers.py` - concrete `MetricsProvider` implementations
  (`MyTimerMetricsProvider`, `TorchProfilerMetricsProvider`,
  `ModuleProfilerMetricsProvider`, plus file-based providers for table CSV,
  ncu CSV, cProfile, perf stat, DCGM CSV, NCCL logs, RAS JSON).
- `metrics/provider_registry.py` - `ProviderSpec`, `MetricsProviderRegistry`,
  and `register_builtin_providers()`, which registers factories for the
  built-in provider types: `my_timer`, `torch_profiler`, `module_profiler`,
  `table_csv`, `ncu_csv`, `nsys_sqlite`, `nsys_sqlite_glob`, `cprofile`,
  `perf_stat`, `dcgm_csv`, `nccl_log`, `ras_json`.
  A module-level `DEFAULT_PROVIDER_REGISTRY` is pre-populated with all of them.

`ProviderSpec` is the contract between adapters and the provider registry:

```python
@dataclass
class ProviderSpec:
    provider_type: str
    provider_id: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
```

Factories receive `(provider_id, params, context)`; live providers such as
`my_timer` and `torch_profiler` fetch their backing object from the context
(`_require_context_obj(context, "my_timer", "timer")`), which is why the same
context mapping is passed to both `detect()` and provider construction.

Framework-specific signals (Megatron parallel state, DeepSpeed ZeRO memory,
NCCL communication) are covered by the neutral sources instead - e.g. the
`nccl_log`, `dcgm_csv`, and `nsys_sqlite` providers - see
[UNIFIED_METRICS_DESIGN.md](./UNIFIED_METRICS_DESIGN.md) and
[CROSS_FRAMEWORK_PROFILE_REFERENCE.md](./CROSS_FRAMEWORK_PROFILE_REFERENCE.md).

> The proposed `my_utils/profiling/callbacks.py` (a `transformers`
> `TrainerCallback` that pushed NVTX ranges and called `collector.collect()`)
> was not built; equivalent wiring is documented per framework in the
> integration playbook.

## 5. Usage Examples

### 5.1 Auto-detection and setup

```python
from my_utils.profiling import DEFAULT_ADAPTER_REGISTRY, MetricsCollector

collector = MetricsCollector(output_dir="./metrics_logs")

# The context declares what this job is and carries the objects
# that providers need.
result = DEFAULT_ADAPTER_REGISTRY.auto_setup_collector(
    collector,
    context={
        "framework": "megatron",   # optional explicit declaration
        "my_timer": my_timer,      # -> my_timer provider
        "torch_profiler": prof,    # -> torch_profiler provider
    },
)

print(result["selected_adapter"])      # "megatron"
print(result["registered_providers"])  # ["my_timer", "torch_profiler"]
print(result["runtime_tags"])          # {"framework": "megatron", "adapter": "MegatronAdapter"}
```

Without an explicit `framework` key, detection falls back to characteristic
context objects or the launch command:

```python
result = DEFAULT_ADAPTER_REGISTRY.auto_setup_collector(
    collector,
    context={"command": "vllm serve meta-llama/Llama-3-8B", "torch_profiler": prof},
)
# result["selected_adapter"] == "vllm"
```

If several adapters match, pass `prefer="<name>"` to pin the choice.

### 5.2 Manual adapter use

```python
from my_utils.profiling import MegatronAdapter, MetricsCollector

adapter = MegatronAdapter()
context = {"megatron_args": args, "my_timer": my_timer}

assert adapter.detect(context)
specs = adapter.build_provider_specs(context)

collector = MetricsCollector()
collector.register_providers_from_specs(
    [
        {"type": s.provider_type, "id": s.provider_id,
         "enabled": s.enabled, "params": s.params}
        for s in specs
    ],
    provider_context=context,
)
```

### 5.3 Collecting in a training loop

Adapters only wire up providers; collection is driven from the training loop:

```python
collector.start()
for step, batch in enumerate(data_loader):
    loss = train_step(batch)
    collector.collect(step=step)   # drain provider events, tag with step
collector.stop()

report = collector.analyze()
collector.export_report(fmt="md")
```

### 5.4 Custom adapters

Third-party frameworks can plug in without touching this package:

```python
from my_utils.profiling import FrameworkAdapter, build_default_adapter_registry
from my_utils.profiling.adapters.common import build_standard_training_specs


class MyFrameworkAdapter(FrameworkAdapter):
    name = "myframework"
    priority = 15  # more specific than huggingface/pytorch fallbacks

    def detect(self, context):
        return context.get("framework") == "myframework" or "myfw_trainer" in context

    def build_provider_specs(self, context):
        return build_standard_training_specs(context)


registry = build_default_adapter_registry()
registry.register(MyFrameworkAdapter())
```

## 6. Config File Support

The proposed adapter-specific YAML (`framework_adapters_config.yaml` with
per-adapter `enabled`/`auto_detect`/provider toggles) was not built. Config
support shipped one level down, on the collector:
`MetricsCollector.from_config()` (in
`my_utils/profiling/pipeline/metrics_collector.py`) accepts a YAML or JSON
file with `collector`, `providers`, `analysis`, and `schema` sections:

```yaml
collector:
  output_dir: ./metrics_logs
  enabled: true
  validate_events: false
  ignore_provider_errors: true

providers:
  - type: my_timer
    id: my_timer
  - type: torch_profiler
    id: torch_profiler
    params:
      include_memory: true
      include_flops: true
  - type: nccl_log
    id: nccl
    params:
      log_path: /path/to/nccl_debug.log

analysis:
  bottleneck_threshold: 0.10
  cv_threshold: 0.50
```

```python
from my_utils.profiling import MetricsCollector

collector = MetricsCollector.from_config(
    "metrics_config.yaml",
    provider_context={"my_timer": my_timer, "torch_profiler": prof},
)
```

Config-declared providers and adapter-declared providers use the same
`ProviderSpec` mechanism, so the two approaches compose: load a base config,
then let `auto_setup_collector()` add framework-appropriate providers on top.

## Summary

The framework adapter system achieves framework-agnostic profiling through:

1. **Unified interface** - every adapter implements
   `FrameworkAdapter.detect() / build_provider_specs() / build_runtime_tags()`
2. **Auto-detection** - `FrameworkAdapterRegistry.detect()` matches adapters
   against an explicit context (declared name, context objects, launch
   command), in priority order, with the generic PyTorch adapter as fallback
3. **Declarative providers** - adapters emit `ProviderSpec`s; the
   `MetricsProviderRegistry` constructs providers from the shared context, and
   missing context objects degrade to warnings
4. **Zero intrusion** - no framework source is modified; instrumentation
   attaches through objects the training script already has

Shipped adapter roster (10): TorchTitan, HuggingFace, veRL, slime, ROLL,
SGLang, vLLM, DeepSpeed, Megatron-LM, and generic PyTorch. Per-framework
wiring recipes are in
[FRAMEWORK_INTEGRATION_PLAYBOOK.md](./FRAMEWORK_INTEGRATION_PLAYBOOK.md).
