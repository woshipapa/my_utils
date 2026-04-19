# NSYS Framework Launch Compatibility (2026)

Updated: 2026-04-19

This note focuses on launch-time compatibility:

1. one-click nsys wrapping by environment/tool version
2. YAML-based argument control
3. launcher order for distributed workers

## 1) Framework launcher matrix

| Framework | Official launcher style | Supported by `run_nsys_quick` | Supported by `run_nsys_quick_yaml.py` | Notes |
|---|---|---|---|---|
| TorchTitan | `MODULE=... CONFIG=... ./run_train.sh` | Yes | Yes | Wrap outer script directly; command is shell-compatible. |
| Megatron-LM | `torchrun ... pretrain_gpt.py ...` | Yes | Yes | Keep `torchrun --no_python` when wrapping `python ...` payload. |
| DeepSpeed | `deepspeed --num_gpus=... train.py ...` | Yes | Yes | Works as plain command wrapping. |
| Hugging Face Trainer | `torchrun ...` or `accelerate launch ...` | Yes | Yes | Both launcher styles are command-compatible. |
| VERL | `python -m verl.trainer.main_ppo ...` | Yes | Yes | Works as Python module command wrapping. |
| SLIME | `bash scripts/run-*.sh` / `ray job submit ... -- python ...` | Yes | Yes | For per-worker traces in Ray, wrapper should be in worker command template. |
| ROLL | `bash examples/...single_node_demo.sh` | Yes | Yes | Shell script launch is wrapper-compatible. |
| SGLang | `python -m sglang.launch_server ...` | Yes | Yes | Service launch can be wrapped directly. |
| vLLM | `vllm serve ...` | Yes | Yes | CLI entrypoint can be wrapped directly. |

## 2) 2026 vs 2024 launch-side argument diff

| Area | 2024 behavior | 2026 behavior | Current my_utils behavior |
|---|---|---|---|
| syscall tracing | commonly via `trace=syscall` | prefers `--syscall=<mode>` | Auto switch by detected/hinted version with fallback. |
| nic metrics mode | usually bool-like `true/false` | supports mode values `lf/hf/none` | Auto normalize both directions by version. |
| gpu metrics field name | legacy code may use `gpu_metrics_device` | canonical `gpu_metrics_devices` | Both accepted; legacy alias retained. |
| forced version override | often unavailable in wrappers | needed in container/multi-host edge cases | `version_hint` / `NSYS_VERSION_HINT` supported. |

## 3) Quick recipes

TorchTitan:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  env MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

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

Hugging Face Trainer (`torchrun`):

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  torchrun --nproc_per_node 8 examples/pytorch/summarization/run_summarization.py --fp16
```

VERL:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python3 -m verl.trainer.main_ppo trainer.n_gpus_per_node=8
```

SLIME:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  bash scripts/run-glm4-9B.sh
```

ROLL:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```

SGLang:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --port 30000
```

vLLM:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

YAML equivalent (`run_nsys_quick_yaml.py`) can keep the same launch command in `command` and tune `nsys_launch` fields without changing shell scripts.

## 4) torchrun + capture-range checklist

When `capture_range=cudaProfilerApi` is used:

1. Keep `nsys profile` at the outermost layer.
2. Keep `target_processes=all` so worker processes are traced.
3. Call `cudaProfilerStart/Stop` in the process that launches CUDA kernels.

If start/stop is only called in a parent process but kernels run in child workers, capture windows can appear ineffective.

## 5) Official references used

- TorchTitan README: https://github.com/pytorch/torchtitan
- Megatron-Core quickstart: https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html
- DeepSpeed getting started: https://www.deepspeed.ai/getting-started/
- Hugging Face training scripts: https://huggingface.co/docs/transformers/main/run_scripts
- Hugging Face Accelerate launch: https://huggingface.co/docs/accelerate/main/en/basic_tutorials/launch
- VERL quickstart: https://verl.readthedocs.io/en/latest/start/quickstart.html
- SLIME quick start: https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md
- ROLL single-node quick start: https://alibaba.github.io/ROLL/docs/Getting%20Started/Quick%20Start/single_node_quick_start
- SGLang basic usage: https://docs.sglang.io/basic_usage/send_request.html
- vLLM quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
