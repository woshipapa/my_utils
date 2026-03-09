# my_utils - 新特性预览

> **🚧 正在开发中** - 以下特性正在积极开发中，查看 [profiling/docs/ROADMAP.md](my_utils/profiling/docs/ROADMAP.md) 了解完整计划。

## 即将推出

### 1. 统一指标收集系统

自动整合所有现有工具（MyTimer、ProfilerWrapper、ModuleProfiler等）的数据：

```python
from my_utils.profiling import MetricsCollector

collector = MetricsCollector(output_dir="./metrics_logs")
collector.register_provider(MyTimerMetricsProvider(timer))
collector.register_provider(TorchProfilerMetricsProvider(profiler))

# 训练过程中自动收集
for step in range(100):
    # ... 训练代码 ...
    collector.collect(step=step)
```

### 2. 智能性能分析器

自动识别问题并给出优化建议：

```python
from my_utils.profiling import AnalyzerPipeline

pipeline = AnalyzerPipeline()
report = pipeline.analyze(collector.get_events())

# 输出:
# 性能得分: 72/100
# 发现 5 个问题 | 其中 1 个严重问题
# 主要瓶颈: attention (38.5%)

for finding in report.findings:
    print(f"[{finding.severity}] {finding.title}")
    print(f"  {finding.description}")

for rec in report.recommendations:
    print(f"[P{rec.priority}] {rec.title}")
    print(f"  预期影响: {rec.estimated_impact}")
    for action in rec.actions:
        print(f"  - {action}")
```

### 3. 增强可视化

#### 交互式HTML报告
```python
from my_utils.profiling.visualization import HTMLReportGenerator

generator = HTMLReportGenerator()
html = generator.generate(report, events)

with open("report.html", "w") as f:
    f.write(html)
```

#### TensorBoard插件
```bash
# 自动集成到TensorBoard
tensorboard --logdir=./metrics_logs --load_plugins=my_utils.profiling.tensorboard_plugin
```

### 4. 框架适配器

开箱即用的框架集成：

```python
from my_utils.profiling import FrameworkRegistry

# 自动检测并设置所有可用框架
collector = MetricsCollector()
adapters = FrameworkRegistry.auto_setup(collector)

# 检测到: Megatron v3.0.0
# 检测到: DeepSpeed v0.9.0

# 正常训练，适配器自动收集指标
for iteration in range(...):
    forward_step(model, input)
    # 框架特定的hooks自动触发
```

## 设计文档

详细设计请查看：

- [**统一指标收集系统**](my_utils/profiling/docs/UNIFIED_METRICS_DESIGN.md) - 核心架构设计
- [**自动化分析器**](my_utils/profiling/docs/AUTO_ANALYZER_DESIGN.md) - 智能分析算法
- [**可视化增强**](my_utils/profiling/docs/VISUALIZATION_DESIGN.md) - 多种可视化方案
- [**框架适配器**](my_utils/profiling/docs/FRAMEWORK_ADAPTERS_DESIGN.md) - 框架集成设计

## 兼容性

新功能完全向后兼容：

- ✅ 现有API保持不变
- ✅ 可选启用，零开销禁用
- ✅ 渐进迁移路径

```python
# 现有代码继续工作
from my_utils import MyTimer, get_global_logger

timer = MyTimer(use_cuda=True)
timer.set_logger(get_global_logger())
# ... 原有代码不变 ...

# 可选：逐步采用新功能
from my_utils.profiling import MetricsCollector
collector = MetricsCollector(enabled=False)  # 默认禁用
```

## 实施进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 📋 设计 | 架构设计文档 | ✅ 完成 |
| 🔨 核心 | MetricsCollector、Provider协议 | ⏳ 待开始 |
| 🧠 分析 | 自动分析器、推荐引擎 | ⏳ 待开始 |
| 📊 可视化 | HTML报告、TensorBoard插件 | ⏳ 待开始 |
| 🔌 适配器 | 框架适配器 | ⏳ 待开始 |

## 参与贡献

欢迎参与！查看 [ROADMAP.md](my_utils/profiling/docs/ROADMAP.md) 了解如何贡献。

---

## 当前可用功能

在等待新功能的同时，现有工具依然强大：

### 核心工作流

```python
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer

logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs/train", rank=0, world_size=1)
logger = logger_mgr.get_logger()

timer = MyTimer(use_cuda=True, tag="train")
timer.set_logger(logger)

for step in range(100):
    timer.set_step(step)
    timer.start("forward")
    output = model(input)
    timer.stop("forward")
    timer.step()
```

### NVTX标记

```python
from my_utils import create_labeler

labeler = create_labeler(enabled=True, default_domain="training")

with labeler.range("model.forward"):
    output = model(input)
```

### 更多工具

- `ProfilerWrapper` - PyTorch Profiler封装
- `ModuleProfiler` - 模块级性能分析
- `GPU_Performance_Tracker` - GPU监控
- `MemorySnapshotter` - 内存快照
- `ForwardProfilerHook` - 自动profiling控制
- `create_profiler_context` - 任务类型感知profiling

查看完整文档: [README.md](README.md)
