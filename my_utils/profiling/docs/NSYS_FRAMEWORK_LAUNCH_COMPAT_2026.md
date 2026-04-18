# NSYS Framework Launch Compatibility (2026)

Updated: 2026-04-18

This note focuses on launch-time compatibility:

1. one-click nsys wrapping by environment/tool version
2. YAML-based argument control

## 1) Framework launcher matrix

| Framework | Official launcher style | Supported by `run_nsys_quick` | Supported by `run_nsys_quick_yaml.py` | Notes |
|---|---|---|---|---|
| Megatron-LM | `torchrun ... pretrain_gpt.py ...` | Yes | Yes | Keep `torchrun --no_python` when wrapping `python ...` payload. |
| DeepSpeed | `deepspeed --num_gpus=... train.py ...` | Yes | Yes | Works as plain command wrapping. |
| Hugging Face Trainer | `accelerate launch ...` or `deepspeed ...` | Yes | Yes | Both launcher styles are command-compatible. |
| VERL | `python -m verl.trainer.main_ppo ...` | Yes | Yes | Works as Python module command wrapping. |
| SLIME | `ray job submit ... -- python3 train.py ...` | Yes (submit side) | Yes (submit side) | For per-worker traces, inject wrapper in worker command template. |

## 2) 2026 vs 2024 launch-side argument diff

| Area | 2024 behavior | 2026 behavior | Current my_utils behavior |
|---|---|---|---|
| syscall tracing | commonly via `trace=syscall` | prefers `--syscall=<mode>` | Auto switch by detected/hinted version with fallback. |
| nic metrics mode | usually bool-like `true/false` | supports mode values `lf/hf/none` | Auto normalize both directions by version. |
| gpu metrics field name | legacy code may use `gpu_metrics_device` | canonical `gpu_metrics_devices` | Both accepted; legacy alias retained. |
| forced version override | often unavailable in wrappers | needed in container/multi-host edge cases | `version_hint` / `NSYS_VERSION_HINT` supported. |

## 3) Quick recipes

Megatron-LM:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  torchrun --nproc_per_node=8 --no_python \
  python pretrain_gpt.py --tensor-model-parallel-size 4
```

DeepSpeed:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

Hugging Face Trainer (accelerate):

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  accelerate launch --num_processes 8 train.py --config_name cfg.yaml
```

VERL:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python -m verl.trainer.main_ppo --config-name=ppo_trainer
```

SLIME (ray submit):

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  ray job submit --working-dir . -- python3 train.py
```

YAML equivalent (`run_nsys_quick_yaml.py`) can keep the same launch command in `command` and tune `nsys_launch` fields without changing shell scripts.

## 4) Official references used

- Megatron-Core quickstart: https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html
- DeepSpeed getting started: https://www.deepspeed.ai/getting-started/
- Hugging Face Accelerate launch: https://huggingface.co/docs/accelerate/main/en/basic_tutorials/launch
- Hugging Face Transformers + DeepSpeed: https://huggingface.co/docs/transformers/en/deepspeed
- VERL quickstart: https://verl.readthedocs.io/en/latest/start/quickstart.html
- SLIME quick start: https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md
