# 跨框架 Profiling 接入实战指南（2026 版）

更新时间：2026-04-19

这份指南只解决一件事：
**不改或少改现有训练/推理脚本，把 `my_utils` 的 NSYS/NCU/Timer/NVTX 能力接到主流框架，并保持“一键可跑”。**

---

## 0) 三种接入模式（先选一个）

### 模式 A：零侵入（推荐先做）

不改训练代码，只包裹启动命令：

- NSYS：`run_nsys_quick.sh` / `run_nsys_quick_yaml.py`
- NCU：`run_ncu_quick_yaml.py`

适用：TorchTitan、Megatron、DeepSpeed、HF Trainer、VERL、SLIME、ROLL、SGLang、vLLM。

### 模式 B：轻侵入（加 timer/logger/NVTX）

在训练/推理循环里增加：

- `GlobalLogger`
- `MyTimer`
- `create_labeler`
- `create_nsys_capture_backend`（`cudaProfilerStart/Stop`）

适用：你能改业务循环，并希望拿到 stage 粒度趋势。

### 模式 C：统一指标管线（collector）

`MetricsCollector` 聚合多个 provider（`my_timer`、`torch_profiler`、`nsys_sqlite`、`ncu_csv` 等），产出统一报告。

适用：要做长期回归、自动化比对、统一报表。

---

## 1) 框架支持矩阵（按当前代码 + 启动兼容）

### 1.1 内置 framework adapters（自动识别/注册）

- `pytorch`
- `huggingface`
- `deepspeed`
- `megatron`

代码位置：`my_utils/profiling/adapters/*`

### 1.2 启动层兼容（launcher-agnostic，模式 A 直接可用）

- TorchTitan（官方 `run_train.sh` / `torchrun`）
- SLIME（`bash scripts/run-*.sh`、`ray job submit ... -- python3 train.py ...`）
- VERL（`python -m verl.trainer.main_ppo ...`）
- ROLL（`bash examples/...single_node_demo.sh` 等 pipeline 脚本）
- SGLang（`python -m sglang.launch_server ...`）
- vLLM（`vllm serve ...`）

说明：这些框架目前不一定都有专用 adapter，但**都可以被 NSYS/NCU wrapper 透明包裹**。

---

## 2) 各框架最小一键命令（模式 A）

### 2.1 TorchTitan（Meta）

官方常见启动（README）：

```bash
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

用 `my_utils` 包裹：

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

或：

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

Ray 提交式：

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  ray job submit --working-dir . -- python3 train.py ...
```

### 2.7 ROLL

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  bash examples/agentic_demo/run_agentic_pipeline_frozen_lake_single_node_demo.sh
```

### 2.8 SGLang（推理服务）

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

### 2.9 vLLM（推理服务）

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

---

## 3) NCU 深挖（统一入口）

所有框架统一：

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_full_collection.yaml -- \
  <你的框架启动命令>
```

例如 TorchTitan：

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_full_collection.yaml -- \
  env MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

报告分析：

```bash
myutils-profile ncu-report-analyze --report ./logs/ncu/ncu_full_collection.ncu-rep --top-k 20 --pretty
```

---

## 4) 一个 YAML 为什么能 wrap 住多框架（原理）

`run_nsys_quick_yaml.py` / `run_ncu_quick_yaml.py` 的核心是：

1. `command` 是原始命令数组，原样透传（不改你的框架参数语义）。
2. profiling 参数全部在 `nsys_launch` / `ncu` 下配置。
3. wrapper 最终拼接成：
   - NSYS: `nsys profile <args> <command...>`
   - NCU: `ncu <args> <command...>`
4. 框架差异只体现在 `command`，profiling 控制项在同一 YAML 里复用。

这也是“同一套 YAML + 不同命令”就能覆盖训练和推理框架的原因。

---

## 5) torchrun / capture-range 顺序建议（避免踩坑）

对于分布式训练，优先使用：

1. `nsys profile` 包在最外层（wrapper 默认如此）。
2. `target_processes=all`（让 worker 子进程也被采集）。
3. `capture_range=cudaProfilerApi` 时，确保 `cudaProfilerStart/Stop` 真在 worker 里被调用。

如果你只在父进程调用 start/stop，而计算在子进程，`capture_range` 可能看起来“不生效”。

---

## 6) 现成样例目录（新增）

- `my_utils/profiling/examples/framework_playbook_samples/README.md`
- `my_utils/profiling/examples/framework_playbook_samples/nsys_*.sh`
- `my_utils/profiling/examples/framework_playbook_samples/ncu_*.sh`

这些脚本是“可复制模板”，专门给 TorchTitan / HF / VERL / SLIME / ROLL / SGLang / vLLM。

---

## 7) 官方参考（本次核对）

- TorchTitan README: https://github.com/pytorch/torchtitan
- SLIME quick start: https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md
- VERL quickstart: https://verl.readthedocs.io/en/latest/start/quickstart.html
- ROLL single-node quick start: https://alibaba.github.io/ROLL/docs/Getting%20Started/Quick%20Start/single_node_quick_start
- Transformers training scripts: https://huggingface.co/docs/transformers/main/run_scripts
- SGLang send request / launch server: https://docs.sglang.io/basic_usage/send_request.html
- vLLM quickstart: https://docs.vllm.ai/en/latest/getting_started/quickstart/
