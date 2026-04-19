# 跨框架 Profiling 接入实战指南（Megatron / DeepSpeed / SGLang / vLLM）

更新时间：2026-04-19

这份指南专门回答一个问题：  
**你要把 `my_utils` 的 profile/timer/logger/NVTX 能力接到不同训练/推理框架里，最稳妥的做法是什么。**

---

## 0) 三种接入模式（先选一个）

### 模式 A：零侵入（推荐先做）

不改训练代码，只包裹启动命令：

- NSYS：`run_nsys_quick.sh` / `run_nsys_quick_yaml.py`
- NCU：`run_ncu_quick_yaml.py`

适用：Megatron、DeepSpeed、SGLang、vLLM 都可直接用。

### 模式 B：轻侵入（加 timer/logger/NVTX）

在训练/推理循环里增加：

- `GlobalLogger`（统一日志）
- `MyTimer`（stage 计时）
- `create_labeler`（NVTX range）
- `create_nsys_capture_backend`（`cudaProfilerStart/Stop` 捕获窗口）

适用：你能改业务循环代码，且希望拿到 stage 粒度时间趋势。

### 模式 C：统一指标管线（collector）

使用 `MetricsCollector` 统一收集多个 provider（如 `my_timer` + `torch_profiler` + `nsys_sqlite` + `ncu_csv`），生成同一份报告。

适用：需要长期规范化报表、自动化回归对比。

---

## 1) 框架支持现状（按代码实际）

### 内置 framework adapters（自动识别/注册）

- `pytorch`
- `huggingface`
- `deepspeed`
- `megatron`

代码位置：`my_utils/profiling/adapters/*`

### 当前未内置专用 adapter

- `sglang`
- `vllm`

这两个框架建议：

1. 启动层先用模式 A（CLI 包裹）  
2. 代码可改时再用模式 B（timer/logger/NVTX）  
3. 报表统一用模式 C（collector + offline providers）

---

## 2) 各框架最小接入命令（模式 A）

### 2.1 Megatron-LM（训练）

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  torchrun --nproc_per_node=8 --no_python \
  python pretrain_gpt.py --tensor-model-parallel-size 4 ...
```

### 2.2 DeepSpeed（训练）

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  deepspeed --num_gpus=8 train.py --deepspeed ds_config.json ...
```

### 2.3 SGLang（推理服务）

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python -m sglang.launch_server --model-path /path/to/model ...
```

### 2.4 vLLM（推理服务）

```bash
bash my_utils/profiling/templates/run_nsys_quick.sh -- \
  python -m vllm.entrypoints.openai.api_server --model /path/to/model ...
```

说明：`run_nsys_quick.sh` 是 launcher-agnostic，命令前加包裹即可。

---

## 3) NCU 深挖（单 kernel 瓶颈）

所有框架统一用法：

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_full_collection.yaml -- \
  <你的框架启动命令>
```

例如 DeepSpeed：

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_full_collection.yaml -- \
  deepspeed --num_gpus=8 train.py --deepspeed ds_config.json ...
```

分析报告：

```bash
myutils-profile ncu-report-analyze --report ./logs/ncu/ncu_full_collection.ncu-rep --top-k 20 --pretty
```

---

## 4) 轻侵入接入模板（模式 B）

下面这段是**框架无关模板**，训练循环/推理循环都可套：

```python
from my_utils.core import GlobalLogger, MyTimer
from my_utils.tracing import create_labeler
from my_utils.profiling.runtime.frameworkless import create_nsys_capture_backend

# 1) logger
logger_mgr = GlobalLogger()
if not logger_mgr.is_configured:
    logger_mgr.setup(log_dir="./logs/train", rank=0, world_size=1, extra_log_label="worker")
logger = logger_mgr.get_logger()

# 2) timer + NVTX
timer = MyTimer(
    use_cuda=True,
    tag="train",
    labeler=create_labeler(enabled=True),  # 无 NVTX 环境会自动降级 NoOp
)
timer.set_logger(logger)

# 3) NSYS capture backend (cudaProfilerStart/Stop)
capture_backend, _ = create_nsys_capture_backend(synchronize=True)

for step in range(num_steps):
    if step == warmup_start:
        capture_backend.start()

    timer.start("step")
    # ... forward/backward/update or inference step ...
    timer.stop("step")
    timer.step()  # 完成同步与日志输出

    if step == warmup_end:
        capture_backend.stop()
```

配套 NSYS 启动建议（YAML）：

```yaml
nsys_launch:
  enabled: true
  capture_range: cudaProfilerApi
  capture_range_end: stop
```

---

## 5) 统一 collector 接入（模式 C）

### 5.1 Python 方式（在线 + 离线混合）

```python
from my_utils.profiling import MetricsCollector
from my_utils.profiling.metrics.provider_registry import ProviderSpec

collector = MetricsCollector(output_dir="./metrics_out")

collector.register_providers_from_specs(
    [
        {"type": "my_timer", "id": "my_timer", "enabled": True, "params": {}},
        {"type": "torch_profiler", "id": "torch_profiler", "enabled": True, "params": {"include_memory": True}},
    ],
    provider_context={
        "my_timer": timer,              # 你的 MyTimer 实例
        "torch_profiler": profiler,     # 你的 torch.profiler 实例
    },
    ignore_errors=True,
)

collector.collect(step=step)
report = collector.analyze()
collector.export_report(fmt="html", report=report)
```

### 5.2 CLI 方式（纯离线）

```bash
myutils-profile ingest \
  --config ./my_utils/profiling/examples/collector_config_nsys_sqlite_full.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html
```

---

## 6) SGLang / vLLM 专项建议

### 6.1 先拿全局行为，再看 kernel

1. 先用 NSYS 包服务启动命令（看 request pipeline、NCCL/通信、CUDA kernels）  
2. 再用 NCU 针对热点 kernel 深挖（`ncu_full_collection.yaml`）  

### 6.2 Kernel 分类面板（NSYS timeline）

`nsys-timeline-html` 支持 kernel category 规则：

- 内置 `sglang` 规则（`--kernel-category-engine sglang`）
- `vllm` 建议提供自定义 JSON 映射：

```bash
myutils-profile nsys-timeline-html \
  --sqlite ./trace.sqlite \
  --output ./timeline.html \
  --kernel-category-map-json ./vllm_kernel_map.json \
  --kernel-category-engine vllm \
  --kernel-category-model llama
```

JSON 支持格式：

- 扁平：`{"regex_pattern": "category"}`
- 分层：`{"engine":{"model":{"regex_pattern":"category"}}}`

---

## 7) 排障清单（高频）

1. `torchrun` 包裹后没采到窗口  
检查 `capture_range` 与 `cudaProfilerStart/Stop` 触发位置是否一致。

2. NSYS/NCU 命令能跑但输出为空  
确认 `CUDA_VISIBLE_DEVICES`、`target_processes`、`launch_count/skip` 与真实执行匹配。

3. collector 报 provider 缺 context  
`my_timer` 需要传 `provider_context={"my_timer": timer}`；  
`torch_profiler` 需要传 `{"torch_profiler": profiler}`。

4. SGLang/vLLM 不能自动识别 adapter  
属预期（当前无专用 adapter），用模式 A/B/C 组合即可。

---

## 8) 推荐落地顺序

1. 模式 A：先把 NSYS/NCU 命令跑通  
2. 模式 B：在关键循环加 `MyTimer + NVTX + capture backend`  
3. 模式 C：把在线/离线指标统一进 `MetricsCollector`  
4. 固化成 YAML + CI 回归脚本（`nsys-diff` / 报告对比）  
