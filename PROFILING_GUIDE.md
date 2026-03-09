# my_utils Profiling 工具 - 完整指南

> 一个强大的、框架无关的 PyTorch 性能分析工具集

## 🎯 核心特性

- ✅ **框架无关** - 适用于 PyTorch、Megatron、DeepSpeed、HuggingFace 等
- ✅ **多数据源支持** - 统一收集 MyTimer、torch.profiler、NCU、nsys 等数据
- ✅ **智能分析** - 自动检测瓶颈、内存泄漏、性能异常
- ✅ **多种报告格式** - HTML、Markdown、JSON
- ✅ **可视化增强** - 美观的图表和交互式报告
- ✅ **易于集成** - 最小化代码改动，开箱即用

## 📖 目录

1. [快速开始](#快速开始)
2. [核心工作流](#核心工作流)
3. [统一指标系统](#统一指标系统)
4. [数据源提供商](#数据源提供商)
5. [可视化报告](#可视化报告)
6. [高级功能](#高级功能)
7. [配置选项](#配置选项)
8. [常见问题](#常见问题)
9. [API 参考](#api-参考)

---

## 🚀 快速开始

### 方式一：从 MyTimer 快速开始

```python
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer
from my_utils.profiling import MyTimerMetricsProvider, MetricsCollector

# 1. 设置 logger 和 timer
logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs", rank=0, world_size=1)
logger = logger_mgr.get_logger()

timer = MyTimer(use_cuda=True, tag="train")
timer.set_logger(logger)

# 2. 创建 MetricsCollector
collector = MetricsCollector(output_dir="./profiling_output")
collector.register_provider(MyTimerMetricsProvider(timer))

# 3. 正常训练
for step in range(100):
    timer.set_step(step)

    timer.start("forward")
    output = model(input)
    timer.stop("forward")

    timer.start("backward")
    loss.backward()
    timer.stop("backward")

    timer.step()

    # 每10步收集一次指标
    if step % 10 == 0:
        collector.collect(step=step)

# 4. 生成报告
report = collector.analyze()
html_path = collector.export_report(fmt="html", report=report)
print(f"报告已生成: {html_path}")
```

### 方式二：从 CSV 文件分析

```python
from my_utils.profiling import MetricsCollector, TableCsvMetricsProvider

collector = MetricsCollector(output_dir="./analysis_output")
collector.register_provider(
    TableCsvMetricsProvider(
        csv_path="logs/profile_rank_0.csv",
        value_column="duration_ms",
        name_column="event_name",
        tag_columns=["step"],
        unit="ms",
    )
)

# 收集数据
collector.start()
collector.collect()
collector.stop()

# 生成报告
report = collector.analyze()
collector.export_report(fmt="html", report=report)
```

### 方式三：运行 Demo

```bash
python -m my_utils.profiling.examples.unified_metrics_demo --steps 30 --output-dir ./demo_output
```

---

## 💾 核心工作流

### 1. GlobalLogger + MyTimer (基础计时)

```python
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer

# 设置
logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs", rank=0, world_size=1)
logger = logger_mgr.get_logger()

timer = MyTimer(use_cuda=True)
timer.set_logger(logger)

# 使用
for step in range(100):
    timer.set_step(step)

    timer.start("forward")
    output = model(input)
    timer.stop("forward")

    timer.step()  # 重要：同步并记录
```

**输出**: `logs/profile_rank_0.csv`

### 2. NVTX 标记 (Nsight Systems 兼容)

```python
from my_utils import create_labeler

labeler = create_labeler(enabled=True, default_domain="training")

# 预注册标签（性能优化）
labeler.register_label("forward", color="blue")
labeler.register_label("backward", color="red")

# 使用
with labeler.range("model.forward"):
    output = model(input)

# 运行: nsys profile --trace=cuda,nvtx python train.py
```

### 3. ModuleProfiler (模块级分析)

```python
from my_utils.moduleProfiler import ModuleProfiler

with ModuleProfiler(model) as profiler:
    for i in range(100):
        profiler.start()
        output = model(input)
        torch.cuda.synchronize()
        profiler.stop()

    # 生成统计
    df = profiler.summary("module_timings.csv")
    print(df.head(10))
```

---

## 📊 统一指标系统

### 核心概念

```
┌─────────────────────────────────────────────┐
│          MetricsCollector                   │
│  - 注册多个 MetricsProvider                  │
│  - 统一数据格式: MetricEvent                │
│  - 自动存储和分析                           │
└─────────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ MyTimer │   │  Torch  │   │   CSV   │
│Provider │   │Profiler │   │Provider │
└─────────┘   └─────────┘   └─────────┘
```

### 使用 MetricsCollector

```python
from my_utils.profiling import (
    MetricsCollector,
    MyTimerMetricsProvider,
    TorchProfilerMetricsProvider,
)

# 创建 collector
collector = MetricsCollector(output_dir="./metrics_output")

# 注册 providers
collector.register_provider(MyTimerMetricsProvider(timer))
collector.register_provider(TorchProfilerMetricsProvider(profiler))

# 收集数据
collector.start()
for step in range(100):
    # ... 训练代码 ...
    collector.collect(step=step)
collector.stop()

# 分析和报告
report = collector.analyze()
print(f"发现 {len(report.findings)} 个问题")

# 导出报告
collector.export_report(fmt="html", report=report)
```

### 配置文件支持

```yaml
# metrics_config.yaml
collector:
  output_dir: "./metrics_output"
  enabled: true

analysis:
  bottleneck_threshold: 0.10    # 10%
  cv_threshold: 0.50              # 变异系数阈值
  memory_growth_bytes_per_step: 10485760  # 10MB/step
```

```python
collector = MetricsCollector.from_config("metrics_config.yaml")
```

---

## 🔌 数据源提供商

### 内置 Providers

#### 1. MyTimerMetricsProvider

```python
from my_utils.profiling import MyTimerMetricsProvider

provider = MyTimerMetricsProvider(
    timer=timer,
    include_cpu=True,
    include_cuda=True,
)
collector.register_provider(provider)
```

**输出指标**:
- `latency.stage` - 各阶段延迟

#### 2. TorchProfilerMetricsProvider

```python
from my_utils.profiling import TorchProfilerMetricsProvider

provider = TorchProfilerMetricsProvider(
    profiler=torch_profiler,
    include_kernel_events=True,
)
collector.register_provider(provider)
```

**输出指标**:
- `latency.kernel` - Kernel 延迟
- `memory.*` - 内存相关

#### 3. TableCsvMetricsProvider (通用 CSV)

```python
from my_utils.profiling import TableCsvMetricsProvider

provider = TableCsvMetricsProvider(
    csv_path="custom_metrics.csv",
    value_column="value",
    name_column="metric_name",
    tag_columns=["step", "rank"],
    unit="ms",
    event_name_prefix="custom.",
)
collector.register_provider(provider)
```

#### 4. NcuCsvMetricsProvider (NCU 导出)

```python
from my_utils.profiling import NcuCsvMetricsProvider

provider = NcuCsvMetricsProvider(
    csv_path="ncu_report.csv",
)
collector.register_provider(provider)
```

#### 5. NsysSqliteMetricsProvider (nsys sqlite 数据库)

```python
from my_utils.profiling import NsysSqliteMetricsProvider

provider = NsysSqliteMetricsProvider(
    sqlite_path="train_rank0.sqlite",
    include_gpu_metrics=True,
    include_network_metrics=True,
    parse_step_from_nvtx=True,
)
collector.register_provider(provider)
```

#### 6. CProfileStatsProvider (Python cProfile)

```python
from my_utils.profiling import CProfileStatsProvider

provider = CProfileStatsProvider(
    stats_path="profile.stats",
)
collector.register_provider(provider)
```

### 自定义 Provider

```python
from my_utils.profiling import BaseMetricsProvider, MetricEvent
import time

class MyCustomProvider(BaseMetricsProvider):
    provider_id = "custom"

    def __init__(self, enabled=True):
        super().__init__(enabled=enabled)
        self._metrics = {}

    def record(self, name: str, value: float):
        self._metrics[name] = value

    def get_metrics(self):
        now = time.time()
        events = []
        for name, value in self._metrics.items():
            events.append(MetricEvent(
                timestamp=now,
                name=f"custom.{name}",
                value=value,
                unit="",
                provider_id=self.provider_id,
                tags={},
            ))
        return events

# 使用
custom_provider = MyCustomProvider()
collector.register_provider(custom_provider)

custom_provider.record("throughput", 128.5)
collector.collect()
```

---

## 📈 可视化报告

### 报告类型

#### 1. HTML 报告 (推荐)

```python
html_path = collector.export_report(fmt="html")
# 在浏览器中打开查看
```

**特性**:
- 美观的界面设计
- 交互式图表
- 响应式布局
- 性能得分展示

#### 2. Markdown 报告

```python
md_path = collector.export_report(fmt="markdown")
```

#### 3. JSON 报告

```python
json_path = collector.export_report(fmt="json")
```

### 可视化增强 (独立使用)

```python
from my_utils.profiling.visualization import (
    HTMLReportGenerator,
    QuickReportGenerator,
)

# 从 MetricsCollector 的数据生成精美报告
from my_utils.profiling.visualization import MetricEvent, Finding, AnalysisReport, Severity

events = [...]  # 你的 MetricEvent 列表
report = AnalysisReport(...)
report.summary = {"total_events": len(events)}
report.findings = [
    Finding(
        finding_type="bottleneck",
        severity="high",
        title="Forward pass is slow",
        description="Forward pass takes 60% of total time",
        data={"share": 0.6},
    )
]

generator = HTMLReportGenerator()
html = generator.generate(report, events, "report.html")
```

---

## 🛠️ 高级功能

### 1. 瓶颈检测

自动检测性能瓶颈：

```python
report = collector.analyze()

for finding in report.findings:
    if finding.finding_type == "bottleneck":
        print(f"瓶颈: {finding.title}")
        print(f"数据: {finding.data}")
```

### 2. 内存分析

检测内存泄漏和增长模式：

```python
for finding in report.findings:
    if finding.finding_type == "memory":
        if finding.data.get("growth_warnings"):
            print("检测到内存增长！")
```

### 3. 异常检测

自动检测异常值：

```python
for finding in report.findings:
    if finding.finding_type == "anomaly":
        print(f"检测到异常: {finding.title}")
```

### 4. 变异性分析

检测性能波动：

```python
for finding in report.findings:
    if finding.finding_type == "variance":
        print(f"高变异性: {finding.title}")
```

---

## ⚙️ 配置选项

### MetricsCollector 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `output_dir` | str | "metrics" | 输出目录 |
| `store` | MetricsStore | None | 自定义存储 |
| `analyzer` | MetricsAnalyzer | None | 自定义分析器 |
| `renderer` | MetricsReportRenderer | None | 自定义渲染器 |
| `enabled` | bool | True | 是否启用 |

### MetricsAnalyzer 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `bottleneck_share_threshold` | float | 0.10 | 瓶颈阈值 (10%) |
| `cv_threshold` | float | 0.50 | 变异系数阈值 |
| `memory_growth_bytes_per_step` | float | 10MB | 内存增长阈值 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `ENABLE_NVTX` | 启用 NVTX 标记 (0/1) |
| `ENABLE_MEMORY_SNAPSHOT` | 启用内存快照 |
| `PROFILE_TASK_TYPES` | 指定要 profile 的任务类型 |

---

## ❓ 常见问题

### Q1: 如何与现有代码集成？

A: 只需 3 行代码：

```python
from my_utils.profiling import MetricsCollector, MyTimerMetricsProvider

collector = MetricsCollector()
collector.register_provider(MyTimerMetricsProvider(timer))

# 在训练循环中
collector.collect(step=step)
```

### Q2: 性能开销有多大？

A: 禁用时零开销，启用时 <1%

```python
# 禁用
collector.set_enabled(False)

# 或条件启用
enabled = os.getenv("ENABLE_PROFILING", "0") == "1"
collector = MetricsCollector(enabled=enabled)
```

### Q3: 支持哪些 PyTorch 版本？

A: PyTorch 1.10+ (推荐 2.0+)

### Q4: 如何分析分布式训练？

A: 使用 tag 参数区分 rank：

```python
collector.collect(step=step, tags={"rank": str(dist.get_rank())})
```

### Q5: 报告太大怎么办？

A: 限制数据收集频率：

```python
# 只收集某些 step
if step % 100 == 0:
    collector.collect(step=step)

# 或使用窗口聚合
from my_utils.profiling.visualization import DataTransformer
transformer = DataTransformer()
aggregated = transformer.aggregate_by_time_window(
    events, window_size=1.0
)
```

---

## 📚 API 参考

### MetricsCollector

```python
class MetricsCollector:
    def __init__(
        self,
        output_dir: str = "metrics",
        *,
        store: Optional[MetricsStore] = None,
        analyzer: Optional[MetricsAnalyzer] = None,
        renderer: Optional[MetricsReportRenderer] = None,
        enabled: bool = True,
    )

    def register_provider(self, provider: MetricsProvider) -> None: ...
    def collect(self, *, step: Optional[int] = None, tags: Optional[Dict] = None) -> int: ...
    def analyze(self, events: Optional[Iterable[MetricEvent]] = None) -> AnalysisReport: ...
    def export_report(self, *, fmt: str = "json", output_path: Optional[str] = None, report: Optional[AnalysisReport] = None) -> str: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

### MetricEvent

```python
@dataclass
class MetricEvent:
    timestamp: float          # Unix 时间戳
    name: str                 # 指标名称
    value: float              # 指标值
    unit: str = ""            # 单位
    provider_id: str = ""     # 提供者 ID
    tags: Dict[str, str] = None  # 标签
    node_id: str = None       # 节点 ID
    parent_id: str = None     # 父节点 ID
```

### AnalysisReport

```python
class AnalysisReport:
    generated_at: str
    summary: Dict[str, Any]
    findings: List[Finding]
    recommendations: List[str]
```

---

## 🎓 完整示例

### 示例 1: 端到端训练分析

```python
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer
from my_utils.profiling import MetricsCollector, MyTimerMetricsProvider

# 初始化
logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs", rank=0, world_size=4)
logger = logger_mgr.get_logger()

timer = MyTimer(use_cuda=True, tag="training")
timer.set_logger(logger)

# 设置 profiling
collector = MetricsCollector(output_dir="./profiling_results")
collector.register_provider(MyTimerMetricsProvider(timer))

collector.start()

# 训练循环
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

for step in range(1000):
    timer.set_step(step)

    # Forward
    timer.start("forward")
    output = model(input)
    loss = criterion(output, target)
    timer.stop("forward")

    # Backward
    timer.start("backward")
    loss.backward()
    optimizer.step()
    timer.stop("backward")

    timer.step()

    # 每 100 步收集一次
    if step % 100 == 0:
        collector.collect(step=step)

collector.stop()

# 生成报告
report = collector.analyze()
print(f"得分: {report.summary.get('overall_score', 'N/A')}")
print(f"发现: {len(report.findings)} 个问题")

html_path = collector.export_report(fmt="html", report=report)
print(f"报告: {html_path}")
```

### 示例 2: 分析现有 CSV

```python
from my_utils.profiling import MetricsCollector, TableCsvMetricsProvider

collector = MetricsCollector(output_dir="./csv_analysis")

collector.register_provider(
    TableCsvMetricsProvider(
        csv_path="logs/profile_rank_0.csv",
        value_column="duration_ms",
        name_column="event_name",
        tag_columns=["step", "type"],
        unit="ms",
    )
)

collector.start()
collector.collect()
collector.stop()

report = collector.analyze()
collector.export_report(fmt="html", report=report)
```

### 示例 3: 多数据源联合分析

```python
from my_utils.profiling import (
    MetricsCollector,
    MyTimerMetricsProvider,
    TorchProfilerMetricsProvider,
    TableCsvMetricsProvider,
)

collector = MetricsCollector(output_dir="./multi_source")

# 1. MyTimer
collector.register_provider(MyTimerMetricsProvider(timer))

# 2. Torch Profiler
collector.register_provider(TorchProfilerMetricsProvider(profiler))

# 3. 外部工具 (NCU)
collector.register_provider(NcuCsvMetricsProvider("ncu.csv"))

# 收集并分析
collector.start()
collector.collect()
collector.stop()

# 统一分析
report = collector.analyze()
collector.export_report(fmt="html", report=report)
```

---

## 🔗 相关文档

- [可视化增强使用指南](./my_utils/profiling/visualization/README.md)
- [设计文档](./my_utils/profiling/docs/ROADMAP.md)
- [快速开始指南](./my_utils/profiling/docs/UNIFIED_PROFILING_QUICKSTART.md)

---

## 🤝 贡献

欢迎贡献！请查看 [ROADMAP.md](./my_utils/profiling/docs/ROADMAP.md) 了解贡献方式。

## 📄 许可证

[根据项目添加许可证信息]
