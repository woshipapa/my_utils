# Cross-Framework Profiling Integration Playbook (2026)

Last verified: 2026-04-19. Chinese original: [`docs/zh/profiling/docs/FRAMEWORK_INTEGRATION_PLAYBOOK.md`](../../../docs/zh/profiling/docs/FRAMEWORK_INTEGRATION_PLAYBOOK.md) (repository root).

This guide solves exactly one problem:
**hook the NSYS/NCU/Timer/NVTX capabilities of `my_utils` into mainstream frameworks with zero or minimal changes to your existing training/inference scripts, while keeping everything one-command runnable.**

---

## 0) Three integration modes (pick one first)

### Mode A: zero-intrusion (recommended starting point)

Do not touch the training code — only wrap the launch command:

- NSYS: `run_nsys_quick.sh` / `run_nsys_quick_yaml.py`
- NCU: `run_ncu_quick_yaml.py`

Applies to: TorchTitan, Megatron, DeepSpeed, HF Trainer, VERL, SLIME, ROLL, SGLang, vLLM.

### Mode B: light-intrusion (add timer/logger/NVTX)

Add to the training/inference loop:

- `GlobalLogger`
- `MyTimer`
- `create_labeler`
- `create_nsys_capture_backend` (`cudaProfilerStart/Stop`)

Applies when: you can modify the business loop and want stage-granularity trends.

### Mode C: unified metrics pipeline (collector)

`MetricsCollector` aggregates multiple providers (`my_timer`, `torch_profiler`, `nsys_sqlite`, `ncu_csv`, ...) and produces a unified report.

Applies when: you want long-term regression tracking, automated comparison, and unified reporting.

---

## 1) Framework support matrix (current code + launch compatibility)

### 1.1 Built-in framework adapters (auto-detection/registration)

- `pytorch`
- `huggingface`
- `deepspeed`
- `megatron`
- `torchtitan`
- `verl`
- `slime`
- `roll`
- `sglang`
- `vllm`

Code location: `my_utils/profiling/adapters/*`

### 1.2 Launcher-level compatibility (launcher-agnostic; Mode A works directly)

- TorchTitan (official `run_train.sh` / `torchrun`)
- SLIME (`bash scripts/run-*.sh`, `ray job submit ... -- python3 train.py ...`)
- VERL (`python -m verl.trainer.main_ppo ...`)
- ROLL (`bash examples/...single_node_demo.sh` and similar pipeline scripts)
- SGLang (`python -m sglang.launch_server ...`)
- vLLM (`vllm serve ...`)

Note: all of the frameworks above are covered by adapter auto-detection, and all of them also support transparent NSYS/NCU wrapper launching.

---

## 2) Minimal one-command launch per framework (Mode A)

### 2.1 TorchTitan (Meta)

Common official launch (from its README):

```bash
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

Wrapped with `my_utils`:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  env MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

### 2.2 Megatron-LM

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  torchrun --nproc_per_node=8 --no_python \
  python pretrain_gpt.py --tensor-model-parallel-size 4 ...
```

### 2.3 DeepSpeed

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  deepspeed --num_gpus=8 train.py --deepspeed ds_config.json ...
```

### 2.4 Hugging Face Trainer

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  torchrun --nproc_per_node 8 examples/pytorch/summarization/run_summarization.py --fp16 ...
```

or:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  accelerate launch run_summarization_no_trainer.py ...
```

### 2.5 VERL

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python3 -m verl.trainer.main_ppo \
  data.train_files=... data.val_files=... trainer.n_gpus_per_node=8 ...
```

### 2.6 SLIME

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  bash scripts/run-glm4-9B.sh
```

Ray-submit style:

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  ray job submit --working-dir . -- python3 train.py ...
```

### 2.7 ROLL

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```

### 2.8 SGLang (inference server)

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

### 2.9 vLLM (inference server)

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

---

## 3) NCU deep-dive (single entry point)

Identical for every framework:

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_full_collection.yaml -- \
  <your framework launch command>
```

For example, TorchTitan:

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_full_collection.yaml -- \
  env MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

Report analysis:

```bash
myutils-profile ncu-report-analyze --report ./logs/ncu/ncu_full_collection.ncu-rep --top-k 20 --pretty
```

---

## 4) Why one YAML can wrap many frameworks (the mechanism)

The core of `run_nsys_quick_yaml.py` / `run_ncu_quick_yaml.py`:

1. `command` is the original command as an array, passed through verbatim (the semantics of your framework arguments are never altered).
2. All profiling parameters live under `nsys_launch` / `ncu` in the YAML.
3. The wrapper ultimately assembles:
   - NSYS: `nsys profile <args> <command...>`
   - NCU: `ncu <args> <command...>`
4. Framework differences appear only in `command`; the profiling knobs are reused from the same YAML.

That is why "one YAML + a different command" covers both training and inference frameworks.

---

## 5) torchrun / capture-range ordering advice (avoid the common trap)

For distributed training, prefer:

1. `nsys profile` wrapped at the outermost level (the wrapper does this by default).
2. `target_processes=all` (so worker subprocesses are captured too).
3. With `capture_range=cudaProfilerApi`, make sure `cudaProfilerStart/Stop` is actually called inside the workers.

If you only call start/stop in the parent process while the computation runs in child processes, `capture_range` may appear to "do nothing".

---

## 6) Ready-made sample scripts

- `my_utils/profiling/examples/framework_playbook_samples/nsys_*.sh`
- `my_utils/profiling/examples/framework_playbook_samples/ncu_*.sh`
- `my_utils/profiling/examples/framework_playbook_samples/nsys_framework_template.yaml` / `ncu_framework_template.yaml`

These scripts are copy-paste templates for TorchTitan / HF / VERL / SLIME / ROLL / SGLang / vLLM.

---

## 7) Official references (verified for this revision)

- TorchTitan README: https://github.com/pytorch/torchtitan
- SLIME quick start: https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md
- VERL quickstart: https://verl.readthedocs.io/en/latest/start/quickstart.html
- ROLL single-node quick start: https://alibaba.github.io/ROLL/docs/Getting%20Started/Quick%20Start/single_node_quick_start
- Transformers training scripts: https://huggingface.co/docs/transformers/main/run_scripts
- SGLang send request / launch server: https://docs.sglang.io/basic_usage/send_request.html
- vLLM quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
