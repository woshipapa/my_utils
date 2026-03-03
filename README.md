# my_utils

PyTorch训练/推理工作流的性能分析、日志记录、追踪和调试工具集。

本README按照代码实际行为组织，提供可直接复制的使用模式。

## 安装

从仓库根目录：

```bash
cd my_utils
pip install -e .
```

可选依赖：

```bash
pip install -e .[profiling,tensordict,etcd,nvml,nvtx,system,megatron]
```

安装所有依赖：

```bash
pip install -e .[all]
```

## 快速导航

- [核心工作流](#核心工作流-globallogger--mytimer)
- [工具类参考](#工具类参考)
  - [日志系统](#1-日志系统-loggerpygloballogger)
  - [计时器](#2-计时器-utilspymytimer)
  - [NVTX标记](#3-nvtx标记-nvtx_utilspynvtxlabeler)
  - [模块性能分析](#4-模块性能分析-moduleprofilerpymoduleprofiler)
  - [PyTorch Profiler封装](#5-pytorch-profiler封装-profilerwrapperpyprofilerwrapper)
  - [内存快照](#6-内存快照-memory_snapshotpymemorysnapshotterr)
  - [GPU性能追踪](#7-gpu性能追踪-gpu_mem_trackerpygpu_performance_tracker)
  - [前向传播Hook](#8-前向传播profiling-hook-forwardprofilehookpyforwardprofilerhook)
  - [灵活的Profiler](#9-灵活的profiler-ditprofilerpyflexibleprofiler)
  - [时钟同步](#10-时钟同步-clocksyncutilspyclocksynchronizer)
  - [高级Profiling系统](#11-高级profiling系统-profilingprofilemanager--capturecontroller)
- [输出文件说明](#输出文件说明)
- [环境变量](#环境变量)

## 核心工作流: GlobalLogger + MyTimer

这是最主要的工作流程，用于分阶段计时和生成机器可读的profiling日志。

### 基本使用

```python
import time
import torch
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer

# 1) 每个进程设置一次logger
logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs/train", rank=0, world_size=1, extra_log_label="Trainer")
logger = logger_mgr.get_logger()

# 2) 创建timer并注入logger
timer = MyTimer(use_cuda=torch.cuda.is_available(), tag="train", log_dir="logs/train")
timer.set_logger(logger)

for step in range(3):
    timer.set_step(step)

    timer.start("iter")
    timer.start("forward")
    time.sleep(0.01)
    timer.stop("forward")

    timer.start("backward")
    time.sleep(0.02)
    timer.stop("backward")
    timer.stop("iter")

    # 刷新CUDA事件计时并写入机器日志
    timer.step()  # 等同于 timer.synchronize_and_log()
```

### 工作原理

**MyTimer记录的内容：**
- **CPU时间**: 使用 `time.perf_counter()` 测量
- **CUDA时间**: 使用CUDA事件 (`cuda_start.elapsed_time(cuda_end)`)，如果启用CUDA
- **层级结构**: 嵌套的 `start/stop` 被追踪为树形结构，包含 `node_id/parent_id`
- **机器日志**: `GlobalLogger.log_profile_event(...)` 将 `START/END` 行写入 `profile_rank_<rank>.csv`

**CSV列格式：**
```
timestamp_unix, readable_time, machine_id, step, event_name, event_type, duration_ms, metadata
```

这个profile event可以设计成自定义的Chrome Trace要记录的信息。

### 将CSV转换为Chrome Trace

`MyTimer` 已经写入了可解析的 `START/END` 行。你可以将CSV转换为Chrome/Perfetto trace：

```python
import csv
import json
import re

def csv_to_trace(csv_path: str, out_json: str) -> None:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    rows.sort(key=lambda r: float(r["timestamp_unix"]))

    trace_events = []
    for r in rows:
        ts_us = int(float(r["timestamp_unix"]) * 1_000_000)
        step = int(r["step"])
        metadata = r.get("metadata", "")
        node_m = re.search(r"node_id=(\d+)", metadata or "")
        node_id = int(node_m.group(1)) if node_m else None

        event = {
            "name": r["event_name"],
            "cat": "MyTimer",
            "ph": "B" if r["event_type"] == "START" else "E",
            "ts": ts_us,
            "pid": r["machine_id"],      # process lane
            "tid": f"step_{step}",       # thread lane
            "args": {
                "step": step,
                "duration_ms": float(r["duration_ms"] or 0.0),
                "metadata": metadata,
            },
        }
        if node_id is not None:
            event["args"]["node_id"] = node_id
        trace_events.append(event)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": trace_events}, f)

# 使用示例:
# csv_to_trace("logs/train/profile_rank_0.csv", "logs/train/mytimer_trace.json")
```

然后在 `chrome://tracing` 或 Perfetto UI 中打开JSON文件。

---

## 工具类参考

### 1. 日志系统 (logger.py::GlobalLogger)

**功能**: 单例模式的全局日志管理器，支持控制台和文件输出，以及机器可读的profile CSV。

**核心方法**:
- `setup(log_dir, rank, world_size, extra_log_label)`: 配置日志系统（幂等，只首次生效）
- `get_logger()`: 获取logger实例
- `log_profile_event(step, event_name, event_type, duration_ms, metadata)`: 写入profile CSV
- `set_time_offset(offset)`: 设置时间偏移（用于多机时钟同步）

**使用示例**:
```python
from my_utils.logger import GlobalLogger, get_global_logger

# 方式1: 单例模式
logger_mgr = GlobalLogger()
logger_mgr.setup(
    log_dir="logs/experiment",
    rank=0,
    world_size=4,
    extra_log_label="Worker",
    data_parallel_rank=0,  # 可选: Megatron并行信息
    data_parallel_world_size=2
)
logger = logger_mgr.get_logger()

# 方式2: 便捷函数
logger = get_global_logger()

# 使用
logger.info("Training started")
logger.warning("Learning rate adjusted")

# 记录profile事件
logger_mgr.log_profile_event(
    step=10,
    event_name="forward_pass",
    event_type="START",
    duration_ms=0.0,
    metadata="layer=transformer_0"
)
```

**输出文件**:
- `rank_<rank>.log`: 人类可读的日志
- `profile_rank_<rank>.csv`: 机器可读的性能事件

**工作流程**:
1. 首次调用 `setup()` 时初始化文件handlers和CSV文件
2. 所有日志通过标准logging模块输出
3. Profile事件直接写入CSV（行缓冲，防止崩溃丢失数据）
4. 支持时间偏移，用于多机时间对齐

---

### 2. 计时器 (utils.py::MyTimer)

**功能**: 层级化的CPU/CUDA计时器，支持嵌套计时和自动日志记录。

**核心方法**:
- `start(name)`: 开始计时一个事件
- `stop(name)`: 停止计时（必须匹配最近的start）
- `set_step(step)`: 设置当前迭代步数
- `step()` / `synchronize_and_log()`: 同步CUDA并写入所有事件到日志
- `set_logger(logger)`: 注入GlobalLogger实例

**使用示例**:
```python
from my_utils import create_labeler
from my_utils.utils import MyTimer
import torch

timer = MyTimer(
    use_cuda=torch.cuda.is_available(),
    tag="training",
    log_dir="logs/perf",
    labeler=create_labeler(enabled=True, default_domain="training"),
)
timer.set_logger(logger)

for epoch in range(10):
    timer.set_step(epoch)

    timer.start("epoch")

    # 嵌套计时
    timer.start("data_loading")
    data = load_batch()
    timer.stop("data_loading")

    timer.start("forward")
    output = model(data)
    timer.stop("forward")

    timer.start("backward")
    loss.backward()
    timer.stop("backward")

    timer.stop("epoch")

    # 刷新并记录所有事件
    timer.step()
```

**工作原理**:
1. **CPU计时**: 使用 `time.perf_counter()` 记录墙钟时间
2. **CUDA计时**: 创建CUDA事件对，在start/stop时记录
3. **层级追踪**: 维护栈结构，自动分配 `node_id` 和 `parent_id`
4. **延迟写入**: 调用 `step()` 时才同步CUDA并写入日志
5. **错误处理**: 检测不匹配的start/stop调用

**注意事项**:
- 必须在每次迭代结束时调用 `timer.step()` 以获取准确的CUDA时间
- `stop()` 必须与最近的 `start()` 匹配（LIFO顺序）
- CUDA计时需要显式同步才能获取准确结果

---

### 3. NVTX标记 (nvtx_utils.py::NvtxLabeler)

**功能**: 独立的NVTX范围标记工具，用于Nsight Systems分析。

**核心方法**:
- `register_label(label, color, domain_name, category)`: 预注册标签（性能优化）
- `start(label, color, domain_name, category)`: 开始NVTX范围
- `stop(token)`: 结束NVTX范围
- `range(label, ...)`: 上下文管理器

**使用示例**:
```python
from my_utils import create_labeler

# 推荐：框架代码统一通过 create_labeler 获取后端
labeler = create_labeler(enabled=True, default_domain="my_app")

# 预注册常用标签（可选，提升性能）
labeler.register_label("model.forward", color="blue")
labeler.register_label("model.layer_00.attention", color="green")
labeler.register_label("model.layer_00.ffn", color="red")

# 方式1: 上下文管理器（推荐）
with labeler.range("model.forward"):
    output = model(input)

    with labeler.range("model.layer_00.attention"):
        attn_out = attention(x)

# 方式2: 手动start/stop
token = labeler.start("custom_operation", color="yellow")
# ... 你的代码 ...
labeler.stop(token)

# 方式3: 标记单点事件
labeler.mark("checkpoint_saved")
```

**当前最佳实践**:
- 优先使用 `create_labeler(...)`，它会在 `NvtxLabeler`、`TorchNvtxLabeler` 和 `NoOpLabeler` 之间自动选择。
- 框架层和业务层都只依赖统一协议：`register_label()`、`start()`、`stop()`、`mark()`、`range()`。
- 只有在你明确要固定到 `torch.cuda.nvtx` 后端时，才直接使用 `TorchNvtxLabeler`。

**TorchNvtxLabeler** (torch.cuda.nvtx后端):
```python
from my_utils import TorchNvtxLabeler

nvtx = TorchNvtxLabeler(enabled=True, default_domain="torch_app")

with nvtx.range("training_step"):
    loss = train_step()

# 支持push/pop栈式API
nvtx.push("data_processing")
process_data()
nvtx.pop()
```

**工作原理**:
1. 检测 `nvtx` 或 `torch.cuda.nvtx` 包是否可用
2. 如果不可用或disabled，所有API变为no-op（零开销）
3. 预注册的标签缓存domain和attributes，避免重复创建
4. 维护活动栈，支持嵌套和错误恢复

**使用场景**:
- 配合 `nsys profile` 使用，在Nsight Systems中可视化
- 不依赖MyTimer，可独立使用
- 适合标记粗粒度的代码区域

---

### 4. 模块性能分析 (moduleProfiler.py::ModuleProfiler)

**功能**: 基于Hook的PyTorch模块级性能分析器，自动测量每个叶子模块的执行时间。

**核心方法**:
- `start()`: 开始一次profiling（在forward前调用）
- `stop()`: 结束profiling并记录数据（在同步后调用）
- `summary(output_path)`: 生成统计报告DataFrame
- `cleanup()`: 移除所有hooks

**使用示例**:
```python
from my_utils.moduleProfiler import ModuleProfiler
import torch

model = MyModel().cuda()

# 方式1: 上下文管理器（推荐，自动cleanup）
with ModuleProfiler(model) as profiler:
    for i in range(100):
        profiler.start()

        output = model(input_data)
        torch.cuda.synchronize()  # 必须！

        profiler.stop()

    # 生成报告
    df = profiler.summary(output_path="module_profile.csv")
    print(df)

# 方式2: 手动管理
profiler = ModuleProfiler(model)
# ... 运行循环 ...
df = profiler.summary()
profiler.cleanup()
```

**输出示例**:
```
                    module_name  mean_ms  median_ms  std_ms  total_ms  run_count  percentage
0   model.transformer.layer_0.attn    12.5      12.3     0.8    1250.0        100        45.2
1   model.transformer.layer_0.ffn      8.3       8.1     0.5     830.0        100        30.0
2   model.embedding                    2.1       2.0     0.2     210.0        100         7.6
```

**工作原理**:
1. 遍历模型，为所有叶子模块注册 `forward_pre_hook` 和 `forward_hook`
2. Pre-hook记录CUDA事件start，post-hook记录end
3. 调用 `stop()` 时同步并计算所有事件的elapsed time
4. 累积多次运行的统计数据
5. `summary()` 计算均值、中位数、标准差和占比

**注意事项**:
- 仅支持CUDA（需要CUDA事件计时）
- 必须在 `stop()` 前调用 `torch.cuda.synchronize()`
- 只在叶子模块上注册hook（避免重复计数）
- 适合找出模型中的性能瓶颈

---

### 5. PyTorch Profiler封装 (profilerwrapper.py::ProfilerWrapper)

**功能**: 封装 `torch.profiler`，自动生成trace、图表和内存快照。

**核心方法**:
- `__enter__` / `__exit__`: 上下文管理器，自动启动/停止profiler
- `record_cuda_average_time()`: 记录CUDA时间统计
- `record_cuda_mm()`: 记录显存使用
- `plot_cuda_time_line()`: 绘制CUDA时间折线图
- `plot_memory_usage_line_chart()`: 绘制显存使用图
- `leave()`: 保存所有输出并退出

**使用示例**:
```python
from my_utils.profilerwrapper import ProfilerWrapper
import torch.distributed as dist

profiler = ProfilerWrapper(
    is_st=False,  # 项目标识
    profile_memory=True,
    log_dir="logs/profiler",
    mem_dir="logs/memory",
    enabled_record_cuda_average_time=True,
    enable_record_cuda_mm=True,
    enable_print_summary=True
)

for step in range(10):
    with profiler:
        output = model(input)
        loss = criterion(output, target)
        loss.backward()

# 生成所有图表和报告
profiler.leave()
```

**输出文件**:
- `<prefix>_trace.json`: Chrome trace文件
- `<prefix>_cuda_time_plot.png`: CUDA时间折线图
- `<prefix>_memory_usage_line_chart.png`: 显存使用图
- `<prefix>_rank_<rank>.log`: 性能摘要表格
- `<prefix>_mem_<timestamp>.pickle`: 内存快照（可选）

**工作原理**:
1. 每次进入上下文时启动 `torch.profiler.profile`
2. 退出时自动记录统计数据（CUDA时间、显存）
3. 累积多次运行的数据用于绘图
4. `leave()` 时生成所有可视化和导出文件

**适用场景**:
- 需要详细的算子级性能分析
- 想要可视化CUDA时间和显存趋势
- 调试OOM问题

---

### 6. 内存快照 (memory_snapshot.py::MemorySnapshotter)

**功能**: 层级化的PyTorch内存快照工具，支持嵌套区域的内存追踪。

**核心方法**:
- `start(name, max_entries)`: 开始记录内存历史
- `stop(name)`: 停止并保存快照到 `.pt` 文件
- `set_logger(logger)`: 注入logger

**使用示例**:
```python
from my_utils.memory_snapshot import global_snapshotter

# 环境变量控制: export ENABLE_MEMORY_SNAPSHOT=1

global_snapshotter.set_logger(logger)

# 嵌套快照
global_snapshotter.start("training_loop")

for epoch in range(10):
    global_snapshotter.start("epoch")

    global_snapshotter.start("forward")
    output = model(input)
    global_snapshotter.stop("forward")  # 保存 forward__<time>__rank0.pt

    global_snapshotter.start("backward")
    loss.backward()
    global_snapshotter.stop("backward")  # 保存 backward__<time>__rank0.pt

    global_snapshotter.stop("epoch")  # 保存 epoch__<time>__rank0.pt

global_snapshotter.stop("training_loop")  # 保存 training_loop__<time>__rank0.pt
```

**工作原理**:
1. 维护一个栈结构，类似MyTimer
2. 首次 `start()` 时启动 `torch.cuda.memory._record_memory_history()`
3. 每次 `stop()` 时调用 `_dump_snapshot()` 保存当前所有历史
4. 最后一个 `stop()` 时重置历史记录器
5. 每个快照包含从根start到当前stop的所有内存操作

**分析快照**:
```python
import torch

# 加载快照
snapshot = torch.load("forward__20260303_120000__rank0.pt")

# 使用PyTorch的内存分析工具
from torch.cuda._memory_viz import profile_plot
profile_plot(snapshot, "memory_viz.html")
```

**注意事项**:
- 快照文件可能很大（数百MB）
- 嵌套的快照包含父级的所有历史（需要diff分析）
- 仅在CUDA可用时工作

---

### 7. GPU性能追踪 (gpu_mem_tracker.py::GPU_Performance_Tracker)

**功能**: 后台线程持续监控GPU显存、利用率和功耗。

**核心方法**:
- `start_monitoring()`: 启动后台监控线程
- `stop_monitoring()`: 停止监控并生成图表
- `plot_usage_graphs()`: 绘制性能图表

**使用示例**:
```python
from my_utils.gpu_mem_tracker import GPU_Performance_Tracker

tracker = GPU_Performance_Tracker(
    prefix="experiment_1",
    interval=1,  # 采样间隔（秒）
    use_distributed=True,
    track_hardware_metrics=True  # 需要pynvml
)

tracker.start_monitoring()

# 你的训练代码
for epoch in range(100):
    train_one_epoch()

tracker.stop_monitoring()  # 自动生成图表
```

**输出文件**:
- `<prefix>_rank<rank>_performance_profile.png`: 包含两个子图
  - 子图1: PyTorch显存（allocated/reserved）
  - 子图2: GPU利用率、显存带宽利用率、功耗

**工作原理**:
1. 启动daemon线程，按固定间隔采样
2. 使用 `torch.cuda.memory_allocated/reserved` 获取显存
3. 使用 `pynvml` 获取硬件指标（利用率、功耗）
4. 停止时使用matplotlib绘制时间序列图

**适用场景**:
- 长时间训练的资源监控
- 诊断显存泄漏
- 优化GPU利用率

---

### 8. 前向传播Profiling Hook (ForwardProfileHook.py::ForwardProfilerHook)

**功能**: 在指定迭代范围自动启动/停止CUDA profiler和NVTX标记。

**核心方法**:
- `attach(module)`: 将hook附加到模块
- 自动在 `start_iter` 启动profiler
- 自动在 `stop_iter` 停止profiler

**使用示例**:
```python
from my_utils import ForwardProfilerHook, create_labeler

model = MyModel()

# 创建hook：在第5-10次迭代时profile
hook = ForwardProfilerHook(
    start_iter=5,
    stop_iter=10,
    rank_only=[0, 1],  # 只在rank 0和1上profile
    nvtx_range="model_forward",
    labeler=create_labeler(enabled=True, default_domain="training"),
)

hook.attach(model)

# 正常训练，hook会自动触发
for i in range(100):
    output = model(input)
    # 在iter 5时自动启动profiler
    # 在iter 10时自动停止profiler
```

**工作原理**:
1. 注册 `forward_hook` 到模块
2. 每次forward时检查当前迭代数
3. 在 `start_iter` 时调用 `torch.cuda.cudart().cudaProfilerStart()`
4. 在 `stop_iter` 时调用 `cudaProfilerStop()` 并移除hook
5. 可选地推送NVTX范围标记

**适用场景**:
- 配合 `nsys profile --capture-range=cudaProfilerApi` 使用
- 只profile特定迭代，减少开销和文件大小
- 分布式训练中选择性profile某些rank

---

### 9. 灵活的Profiler (DITProfiler.py::FlexibleProfiler)

**功能**: 任务类型感知的profiler，支持正则表达式匹配算子并生成详细分析。

**核心方法**:
- `create_profiler_context(current_task_type, logger, ops_to_analyze)`: 工厂函数

**使用示例**:
```python
from my_utils.DITProfiler import create_profiler_context
import os

# 设置环境变量: export PROFILE_TASK_TYPES=DIT,VAE

logger = get_global_logger()

# 定义要分析的算子（支持正则表达式）
ops_to_analyze = {
    r"aten::linear": "self_cuda_time_total",
    r"aten::matmul": "self_cuda_time_total",
    r"tile_encoder_\d+": "cuda_time_total",  # 匹配 tile_encoder_0, tile_encoder_1, ...
}

# DIT任务会被profile（因为在PROFILE_TASK_TYPES中）
with create_profiler_context(
    current_task_type="DIT",
    logger=logger,
    ops_to_analyze=ops_to_analyze
):
    dit_output = dit_model(input)

# VAE任务也会被profile
with create_profiler_context(
    current_task_type="VAE",
    logger=logger
):
    vae_output = vae_model(input)

# Transformer任务不会被profile（不在环境变量中）
with create_profiler_context(
    current_task_type="Transformer",
    logger=logger
):
    transformer_output = transformer_model(input)  # 零开销
```

**输出文件**:
- `<base_output_dir>/<task_type>/torch_prof_rank<rank>.json`: Chrome trace（VAE任务跳过）
- `<base_output_dir>/<task_type>/prof_summary_rank<rank>.txt`: 性能摘要
- `<base_output_dir>/<task_type>/<op_name>_rank<rank>_analysis.json`: 每个匹配算子的详细统计

**算子分析JSON格式**:
```json
{
  "tile_encoder_0": {
    "metric": "cuda_time_total",
    "count": 150,
    "total_us": 125000.5,
    "mean_us": 833.34,
    "median_us": 820.0,
    "std_us": 45.2,
    "min_us": 750.0,
    "max_us": 950.0
  }
}
```

**工作原理**:
1. 读取环境变量 `PROFILE_TASK_TYPES`（逗号分隔）
2. 只有当前任务类型在列表中时才启动profiler
3. 使用 `torch.profiler.profile` 收集事件
4. 退出时遍历所有事件，用正则表达式匹配
5. 按完整事件名分组，计算统计量并保存JSON

**适用场景**:
- 多阶段pipeline（DIT、VAE、Transformer等）
- 只想profile特定阶段
- 需要特定算子的详细统计

---

### 10. 时钟同步 (clockSyncUtils.py::ClockSynchronizer)

**功能**: 基于NTP协议的分布式时钟同步，用于多机profiling时间对齐。

**核心方法**:
- `sync(is_server, peer_rank, group)`: 执行同步并返回时间偏移

**使用示例**:
```python
from my_utils.clockSyncUtils import ClockSynchronizer
import torch.distributed as dist

# 初始化分布式
dist.init_process_group(backend="nccl")
rank = dist.get_rank()

# Producer (rank 0) 作为时间基准
if rank == 0:
    offset = ClockSynchronizer.sync(is_server=True, peer_rank=1)
    # offset = 0.0
else:
    # Consumer (rank 1) 计算相对于Producer的偏移
    offset = ClockSynchronizer.sync(is_server=False, peer_rank=0)
    # offset = 0.002345 (例如，Consumer比Producer快2.3ms)

# 将偏移应用到logger
logger_mgr = GlobalLogger()
logger_mgr.set_time_offset(offset)

# 现在两个机器的profile CSV时间戳已对齐
```

**SocketClockSynchronizer** (不依赖torch.distributed):
```python
from my_utils.clockSyncUtils import SocketClockSynchronizer

sync = SocketClockSynchronizer(port=12345)

# Machine A (Producer)
if is_producer:
    offset = sync.sync(is_server=True)

# Machine B (Consumer)
else:
    offset = sync.sync(is_server=False, peer_ip="192.168.1.100")
```

**工作原理** (NTP算法):
1. Client发送ping，记录 T1
2. Server接收，记录 T2
3. Server立即回复，记录 T3
4. Client接收，记录 T4
5. 计算：
   - RTT = (T4 - T1) - (T3 - T2)
   - Latency = RTT / 2
   - Offset = (T2 - Latency) - T1

**适用场景**:
- 多机分布式训练的profiling
- 需要合并多机的trace文件
- 分析跨机器的通信延迟

---

### 11. 高级Profiling系统 (profiling::ProfileManager + CaptureController)

**功能**: 基于配置的窗口化profiling系统，支持精确控制start/stop条件。

**核心组件**:
- `ProfileManager`: 读取YAML配置，管理profiling窗口
- `CaptureController`: 执行具体的capture逻辑，响应hook事件

**配置示例** (YAML):
```yaml
profile:
  enabled: true
  features:
    capture:
      enabled: true
      backend: "cudaProfilerApi"
      nvtx_window: true
      default_stop_policy: "ON_TARGET_FUNC_EXIT"

  capture:
    debug_watch: false
    schedule:
      steps: [5, 10]  # 只在这些step允许arm
      max_windows_per_step: 2
      roles: ["generator", "critic"]

    windows:
      - name: "forward_window"
        role: "generator"
        start_iter: 5
        start_mb_selector: "first"  # first, last, 或数字
        start_profile_names: ["model.forward"]

        stop_iter: 5
        stop_mb_selector: "last"
        stop_profile_names: ["model.forward"]
        stop_policy: "ON_STOP_PROFILE_NAME"  # 或 ON_TARGET_FUNC_EXIT
        stop_edge: "EXIT"  # 或 ENTER

        ranks_filter: [0, 1]
        enable_nvtx_window: true
```

**使用示例**:
```python
from my_utils import (
    CaptureController,
    CudaProfilerBackend,
    HookEvent,
    ProfileManager,
    create_labeler,
)

# 1. 创建ProfileManager
profile_cfg = load_yaml("profile_config.yaml")
pm = ProfileManager(profile_cfg, logger=logger)

# 2. 为每个角色创建backend和controller
backend = CudaProfilerBackend(synchronize=True)
controller = CaptureController(
    backend=backend,
    logger=logger,
    window_labeler=create_labeler(enabled=True, default_domain="capture"),
)

# 3. 在训练循环中arm
for it in range(100):
    if pm.should_capture_this_iter(it):
        specs = pm.build_specs_to_arm_at_iter(it, num_microbatches=8)
        for spec in specs:
            controller.arm(spec)

    # 4. 在代码中插入hook点
    for mb in range(num_microbatches):
        # 触发start条件
        controller.on_enter(HookEvent(
            profile_name="model.forward",
            meta={"iter": it, "mb": mb},
            role="generator",
            rank=0
        ))

        output = model(input)

        # 触发stop条件
        controller.on_exit(HookEvent(
            profile_name="model.forward",
            meta={"iter": it, "mb": mb},
            role="generator",
            rank=0
        ))
```

**工作原理**:
1. **ProfileManager** 解析配置，确定哪些iter需要arm
2. 在目标iter，构建capture spec（包含start/stop条件）
3. **CaptureController** 接收spec并进入armed状态
4. 代码中的hook点触发 `on_enter/on_exit` 事件
5. Controller匹配条件（iter、mb、profile_name、role、rank）
6. 满足start条件时调用 `backend.start()`（如启动nsys）
7. 满足stop条件时调用 `backend.stop()`
8. 自动disarm，等待下一个窗口

**Stop Policy**:
- `ON_TARGET_FUNC_EXIT`: 历史别名，内部会归一化为 `ON_TRIGGER_FUNC_EXIT`。
- `ON_TRIGGER_FUNC_EXIT`: 在触发start的同一个profile_name的exit时stop
- `ON_STOP_PROFILE_NAME`: 在配置的stop_profile_names匹配时stop

**适用场景**:
- 复杂的多阶段pipeline
- 需要精确控制profiling窗口（特定iter的特定microbatch）
- 多角色系统（generator、critic、discriminator等）
- 与Nsight Systems集成

---

## 输出文件说明

根据使用的工具，你会看到以下输出文件：

### Logger相关
- `rank_<rank>.log`: 人类可读的日志文件
- `profile_rank_<rank>.csv`: 机器可读的性能事件CSV

### ProfilerWrapper
- `<prefix>_trace.json`: Chrome trace文件
- `<prefix>_cuda_time_plot.png`: CUDA时间折线图
- `<prefix>_memory_usage_line_chart.png`: 显存使用折线图
- `<prefix>_mem_<timestamp>.pickle`: 内存快照（可选）

### ModuleProfiler
- 通过 `summary(output_path=...)` 生成的CSV文件

### MemorySnapshotter
- `<name>__<timestamp>__rank<rank>.pt`: PyTorch内存快照文件

### GPU_Performance_Tracker
- `<prefix>_rank<rank>_performance_profile.png`: GPU性能图表

### FlexibleProfiler
- `<base_output_dir>/<task_type>/torch_prof_rank<rank>.json`: Chrome trace
- `<base_output_dir>/<task_type>/prof_summary_rank<rank>.txt`: 性能摘要
- `<base_output_dir>/<task_type>/<op_name>_rank<rank>_analysis.json`: 算子分析

---

## 环境变量

以下环境变量控制工具的行为：

- `ENABLE_TIMER=1`: 启用全局 `MyTimer` 实例 (`global_timer`)
- `ENABLE_NVTX=1`: 作为 `create_labeler(enabled=None)` 和 `ForwardProfilerHook(nvtx_range=...)` 的默认开关；如果显式传了 `enabled=`，以显式配置为准
- `ENABLE_MEMORY_SNAPSHOT=1`: 启用 `global_snapshotter`
- `PROFILE_TASK_TYPES=DIT,VAE`: 为 `create_profiler_context` 启用指定任务类型的profiling
- `DEBUG_DATA_CONSISTENCY=1`: 启用 `ChecksumUtils` 签名/验证
- `WAN_DPO_PREVAE_TENSOR_DIR`: `DumpTensorIO` 张量文件的根目录
- `WAN_DPO_PREVAE_COMPARE_FILE`: `DumpTensorIO` 的可选比较输出路径模板

---

## 常见工作流组合

### 1. 基础训练监控
```python
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer

logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs", rank=0, world_size=1)
logger = logger_mgr.get_logger()

timer = MyTimer(use_cuda=True, tag="train")
timer.set_logger(logger)

for step in range(100):
    timer.set_step(step)
    timer.start("iteration")
    # 训练代码
    timer.stop("iteration")
    timer.step()
```

### 2. 详细性能分析
```python
from my_utils.profilerwrapper import ProfilerWrapper
from my_utils.gpu_mem_tracker import GPU_Performance_Tracker

# GPU监控
tracker = GPU_Performance_Tracker(prefix="exp1", interval=1)
tracker.start_monitoring()

# Profiler
profiler = ProfilerWrapper(
    is_st=False,
    profile_memory=True,
    enabled_record_cuda_average_time=True
)

for step in range(10):
    with profiler:
        train_step()

profiler.leave()
tracker.stop_monitoring()
```

### 3. 模块级性能分析
```python
from my_utils.moduleProfiler import ModuleProfiler

with ModuleProfiler(model) as profiler:
    for i in range(100):
        profiler.start()
        output = model(input)
        torch.cuda.synchronize()
        profiler.stop()

    df = profiler.summary("module_timings.csv")
    print(df.head(10))  # 查看最慢的10个模块
```

### 4. Nsight Systems集成
```python
from my_utils import ForwardProfilerHook, create_labeler

# NVTX标记
labeler = create_labeler(enabled=True, default_domain="training")
labeler.register_label("forward", color="blue")
labeler.register_label("backward", color="red")

# 自动profiler控制
hook = ForwardProfilerHook(
    start_iter=10,
    stop_iter=20,
    rank_only=[0],
    nvtx_range="training_window",
    labeler=labeler,
)
hook.attach(model)

for i in range(100):
    with labeler.range("forward"):
        output = model(input)
    with labeler.range("backward"):
        loss.backward()

# 运行: nsys profile --capture-range=cudaProfilerApi python train.py
```

### 5. 多机时间同步
```python
from my_utils.clockSyncUtils import ClockSynchronizer
from my_utils.logger import GlobalLogger

# 同步时钟
offset = ClockSynchronizer.sync(
    is_server=(rank == 0),
    peer_rank=1 if rank == 0 else 0
)

# 应用偏移
logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs", rank=rank, world_size=world_size)
logger_mgr.set_time_offset(offset)

# 现在所有机器的时间戳已对齐
```

### 6. 内存调试
```python
from my_utils.memory_snapshot import global_snapshotter

# export ENABLE_MEMORY_SNAPSHOT=1

global_snapshotter.set_logger(logger)

global_snapshotter.start("training")

for epoch in range(10):
    global_snapshotter.start(f"epoch_{epoch}")
    train_epoch()
    global_snapshotter.stop(f"epoch_{epoch}")

global_snapshotter.stop("training")

# 分析快照: 使用PyTorch的memory_viz工具
```

---

## 注意事项

### 性能开销
- **MyTimer**: 极低开销（仅事件记录）
- **NVTX**: 零开销（如果disabled）
- **ModuleProfiler**: 中等开销（每个模块的hook）
- **ProfilerWrapper**: 高开销（完整的算子追踪）
- **MemorySnapshotter**: 高开销（记录所有内存操作）

### 最佳实践
1. **分阶段profiling**: 不要同时启用所有工具
2. **选择性采样**: 只profile几个迭代，不是全部
3. **使用环境变量**: 方便在不修改代码的情况下开关profiling
4. **先粗后细**: 先用MyTimer找瓶颈，再用ProfilerWrapper深入分析
5. **CUDA同步**: 使用CUDA计时时必须显式同步
6. **文件大小**: ProfilerWrapper和MemorySnapshotter会生成大文件，注意磁盘空间

### 常见问题

**Q: MyTimer的CUDA时间不准确？**
A: 确保在每次迭代结束时调用 `timer.step()` 来同步CUDA事件。

**Q: ModuleProfiler报错"requires CUDA"？**
A: 该工具仅支持CUDA。如果在CPU上运行，使用ProfilerWrapper代替。

**Q: NVTX标记在Nsight Systems中看不到？**
A: 先检查 `ENABLE_NVTX=1` 或显式 `create_labeler(enabled=True)` 是否生效，再确认 `nvtx` 包已安装；如果走的是 fallback backend，则只能看到 `torch.cuda.nvtx` 能表达的能力。

**Q: 内存快照文件太大？**
A: 减少 `max_entries` 参数，或只在关键区域使用。

**Q: 多机trace时间不对齐？**
A: 使用 `ClockSynchronizer` 同步时钟，并调用 `set_time_offset()`。

---

## 进阶主题

### 自定义Backend
```python
from my_utils import CaptureBackend, CaptureController, create_labeler

class MyCustomBackend(CaptureBackend):
    def start(self):
        # 启动你的profiler
        pass

    def stop(self):
        # 停止并保存结果
        pass

# 与CaptureController配合使用
controller = CaptureController(
    backend=MyCustomBackend(),
    logger=logger,
    window_labeler=create_labeler(enabled=True, default_domain="custom_capture"),
)
```

### 扩展MyTimer
```python
from my_utils.utils import MyTimer

class MyCustomTimer(MyTimer):
    def step(self):
        super().step()
        # 添加自定义逻辑，如上传到监控系统
        self.upload_metrics()
```

---

## 相关工具

- **Nsight Systems**: NVIDIA的系统级profiler，配合NVTX使用
- **PyTorch Profiler**: 内置的profiler，被ProfilerWrapper封装
- **chrome://tracing**: Chrome浏览器的trace查看器
- **Perfetto UI**: 更现代的trace查看器 (https://ui.perfetto.dev)

---

## 贡献与反馈

如有问题或建议，请提交issue或PR。

## 许可证

[根据你的项目添加许可证信息]
