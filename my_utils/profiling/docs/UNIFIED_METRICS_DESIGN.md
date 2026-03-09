# 统一性能指标系统设计方案

## 设计目标

1. **框架无感知** - 核心系统不依赖任何特定训练框架
2. **模块化** - 各组件可独立使用或组合使用
3. **可扩展** - 易于添加新的数据源和分析器
4. **轻量级** - 禁用时零开销

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    MetricsCollector                          │
│  - 注册多个 MetricsProvider                                  │
│  - 统一数据格式: MetricEvent(timestamp, name, value, tags)   │
│  - 支持实时流式处理和批量写入                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   MetricsStore                               │
│  - 内存缓冲 + 持久化存储                                      │
│  - 支持多种后端: 文件、Redis、Prometheus                      │
│  - 自动聚合和降采样                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  MetricsAnalyzer                             │
│  - 瓶颈检测算法                                              │
│  - 趋势分析                                                  │
│  - 异常检测                                                  │
│  - 优化建议生成                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Visualization                              │
│  - HTML Report (静态)                                        │
│  - TensorBoard Plugin                                        │
│  - Web Dashboard (可选)                                      │
└─────────────────────────────────────────────────────────────┘
```

## 1. MetricsProvider 协议

```python
@runtime_checkable
class MetricsProvider(Protocol):
    """所有指标提供者的统一接口"""

    provider_id: str

    def get_metrics(self) -> list[MetricEvent]:
        """返回当前收集的指标事件"""
        ...

    def start_collection(self) -> None:
        """开始收集指标"""
        ...

    def stop_collection(self) -> None:
        """停止收集指标"""
        ...

    def is_enabled(self) -> bool:
        """检查是否启用"""
        ...
```

## 2. MetricEvent 统一数据格式

```python
@dataclass
class MetricEvent:
    timestamp: float          # Unix时间戳(秒)
    name: str                 # 指标名称 (如 "cuda.time", "memory.allocated")
    value: float | int | str  # 指标值
    unit: str = ""            # 单位 (ms, GB, count等)
    tags: dict[str, str] = field(default_factory=dict)
    # tags示例: {"rank": "0", "step": "100", "module": "transformer.layer.0"}

    # 层级关系(可选)
    parent_id: str | None = None
    node_id: str | None = None
```

## 3. 现有工具的适配器

### 3.1 MyTimerMetricsProvider
```python
class MyTimerMetricsProvider(MetricsProvider):
    def __init__(self, timer: MyTimer):
        self.timer = timer
        self.provider_id = "my_timer"

    def get_metrics(self) -> list[MetricEvent]:
        # 从MyTimer的_event_stack转换为MetricEvent
        events = []
        for event in self.timer._events:
            events.append(MetricEvent(
                timestamp=event.timestamp,
                name=f"timer.{event.name}",
                value=event.duration_ms,
                unit="ms",
                tags={
                    "step": str(event.step),
                    "type": event.type,  # "START" or "END"
                },
                node_id=event.node_id,
                parent_id=event.parent_id,
            ))
        return events
```

### 3.2 TorchProfilerMetricsProvider
```python
class TorchProfilerMetricsProvider(MetricsProvider):
    def __init__(self, profiler: torch.profiler.profile):
        self.profiler = profiler
        self.provider_id = "torch_profiler"

    def get_metrics(self) -> list[MetricEvent]:
        events = []
        for evt in self.profiler.events():
            events.append(MetricEvent(
                timestamp=evt.cpu_time / 1_000_000,
                name=evt.name,
                value=evt.cuda_time_total if evt.is_cuda else evt.cpu_time_total,
                unit="us",
                tags={
                    "device": "cuda" if evt.is_cuda else "cpu",
                    "is_async": str(evt.is_async),
                }
            ))
        return events
```

### 3.3 ModuleProfilerMetricsProvider
```python
class ModuleProfilerMetricsProvider(MetricsProvider):
    def __init__(self, profiler: ModuleProfiler):
        self.profiler = profiler
        self.provider_id = "module_profiler"

    def get_metrics(self) -> list[MetricEvent]:
        df = self.profiler.summary()
        events = []
        for _, row in df.iterrows():
            events.append(MetricEvent(
                name=f"module.{row['module_name']}",
                value=row['mean_ms'],
                unit="ms",
                tags={
                    "count": str(row['run_count']),
                    "percentage": f"{row['percentage']:.1f}",
                }
            ))
        return events
```

## 4. MetricsCollector 实现

```python
class MetricsCollector:
    def __init__(self, output_dir: str = "metrics"):
        self._providers: dict[str, MetricsProvider] = {}
        self._store = MetricsStore(output_dir)
        self._analyzer = MetricsAnalyzer()
        self._enabled = True

    def register_provider(self, provider: MetricsProvider) -> None:
        """注册指标提供者"""
        self._providers[provider.provider_id] = provider

    def collect(self, step: int | None = None) -> None:
        """收集所有provider的指标"""
        if not self._enabled:
            return

        all_events = []
        for provider in self._providers.values():
            if provider.is_enabled():
                events = provider.get_metrics()
                if step is not None:
                    for evt in events:
                        evt.tags["step"] = str(step)
                all_events.extend(events)

        self._store.write_events(all_events)

    def analyze_and_report(self) -> AnalysisReport:
        """生成分析报告"""
        events = self._store.read_all_events()
        return self._analyzer.analyze(events)

    def export_report(self, format: str = "html") -> str:
        """导出报告"""
        report = self.analyze_and_report()
        if format == "html":
            return HTMLReportGenerator().generate(report)
        elif format == "json":
            return report.to_json()
        # ...
```

## 5. MetricsAnalyzer 自动分析

```python
class MetricsAnalyzer:
    def analyze(self, events: list[MetricEvent]) -> AnalysisReport:
        report = AnalysisReport()

        # 1. 瓶颈检测
        report.add_finding(self._detect_bottlenecks(events))

        # 2. 内存分析
        report.add_finding(self._analyze_memory(events))

        # 3. 趋势分析
        report.add_finding(self._analyze_trends(events))

        # 4. 异常检测
        report.add_finding(self._detect_anomalies(events))

        # 5. 优化建议
        report.add_recommendations(self._generate_recommendations(report))

        return report

    def _detect_bottlenecks(self, events: list[MetricEvent]) -> Finding:
        """检测性能瓶颈"""
        # 按name分组，计算总时间占比
        timer_events = [e for e in events if e.name.startswith("timer.") or e.name.startswith("kernel.")]
        total_time = sum(e.value for e in timer_events)

        bottlenecks = []
        for name in set(e.name for e in timer_events):
            name_events = [e for e in timer_events if e.name == name]
            name_time = sum(e.value for e in name_events)
            if name_time / total_time > 0.1:  # 超过10%
                bottlenecks.append(Bottleneck(
                    name=name,
                    percentage=name_time / total_time * 100,
                    avg_value=mean(e.value for e in name_events),
                ))

        return Finding(
            type="bottleneck",
            severity="high" if bottlenecks else "info",
            description=f"发现 {len(bottlenecks)} 个主要瓶颈",
            data=bottlenecks,
        )
```

## 6. HTML 报告生成

```python
class HTMLReportGenerator:
    def generate(self, report: AnalysisReport) -> str:
        """生成交互式HTML报告"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                /* 内联样式 */
            </style>
        </head>
        <body>
            <h1>性能分析报告</h1>
            <div id="summary">{summary}</div>
            <div id="bottlenecks">
                <h2>性能瓶颈</h2>
                <canvas id="bottleneckChart"></canvas>
            </div>
            <div id="recommendations">
                <h2>优化建议</h2>
                {recommendations}
            </div>
            <script>
                // Chart.js 图表
                const bottleneckData = {bottleneck_data};
                new Chart(document.getElementById('bottleneckChart'), {{
                    type: 'pie',
                    data: {{
                        labels: {labels},
                        datasets: [{{
                            data: {data},
                        }}]
                    }}
                }});
            </script>
        </body>
        </html>
        """
        # 填充模板
        return template.format(
            summary=self._render_summary(report),
            recommendations=self._render_recommendations(report),
            bottleneck_data=self._get_bottleneck_data(report),
            # ...
        )
```

## 7. TensorBoard 插件

```python
# my_utils/profiling/tb_plugin/metrics_plugin.py

from tensorboard.plugins.hparams import api as hp
from tensorboard.plugins import base_plugin

class MetricsPlugin(base_plugin.TBLoader):
    """TensorBoard插件用于可视化MetricsCollector数据"""

    def __init__(self, context):
        self._context = context

    def get_metadata(self) -> dict:
        return {"name": "my_utils_metrics"}

    def frontend_source(self) -> str:
        return """..."""  # WebSocket实时更新逻辑

# my_utils/profiling/tb_plugin/__init__.py
def register_with_tensorboard(logdir: str):
    """注册插件到TensorBoard"""
    from tensorboard import program
    # ...
```

## 8. 使用示例

```python
from my_utils.profiling import (
    MetricsCollector,
    MyTimerMetricsProvider,
    TorchProfilerMetricsProvider,
    ModuleProfilerMetricsProvider,
)
from my_utils import MyTimer, ModuleProfiler, get_global_logger

# 1. 初始化
logger = get_global_logger()
timer = MyTimer(use_cuda=True, tag="train")
timer.set_logger(logger)

collector = MetricsCollector(output_dir="./metrics_logs")

# 2. 注册providers
collector.register_provider(MyTimerMetricsProvider(timer))

# 3. 训练循环
model = MyModel().cuda()
with ModuleProfiler(model) as module_prof:
    collector.register_provider(ModuleProfilerMetricsProvider(module_prof))

    for step in range(1000):
        timer.set_step(step)

        timer.start("forward")
        output = model(input)
        timer.stop("forward")

        timer.start("backward")
        loss.backward()
        timer.stop("backward")

        timer.step()

        # 每100步收集一次
        if step % 100 == 0:
            collector.collect(step=step)

# 4. 生成报告
report = collector.analyze_and_report()
html_path = collector.export_report(format="html")
print(f"报告已生成: {html_path}")
```

## 9. 框架适配器设计

框架适配器在核心系统之外，作为可选扩展：

```
my_utils/
├── profiling/
│   ├── core/              # 核心系统(框架无关)
│   │   ├── collector.py
│   │   ├── analyzer.py
│   │   └── store.py
│   ├── providers/         # Provider实现
│   │   ├── my_timer.py
│   │   ├── torch_profiler.py
│   │   └── module_profiler.py
│   └── adapters/          # 框架适配器(可选)
│       ├── megatron.py
│       ├── deepspeed.py
│       ├── huggingface.py
│       └── base.py        # 适配器基类
```

### 适配器基类

```python
class FrameworkAdapter(ABC):
    """框架适配器基类"""

    @classmethod
    @abstractmethod
    def detect(cls) -> bool:
        """检测当前环境是否使用该框架"""
        ...

    @classmethod
    @abstractmethod
    def auto_setup(cls, collector: MetricsCollector) -> None:
        """自动设置和注册相关providers"""
        ...

    @classmethod
    @abstractmethod
    def get_train_loop_hooks(cls) -> dict[str, Callable]:
        """返回训练循环的hook点"""
        ...
```

### Megatron适配器示例

```python
class MegatronAdapter(FrameworkAdapter):
    @classmethod
    def detect(cls) -> bool:
        try:
            import megatron
            return True
        except ImportError:
            return False

    @classmethod
    def auto_setup(cls, collector: MetricsCollector) -> None:
        from megatron.core import parallel_state
        from my_utils.profiling.providers.megatron import (
            MegatronTimerProvider,
            MegatronMemoryProvider,
        )

        rank = parallel_state.get_tensor_model_parallel_rank()
        collector.register_provider(MegatronTimerProvider(rank=rank))
        collector.register_provider(MegatronMemoryProvider())

    @classmethod
    def get_train_loop_hooks(cls) -> dict[str, Callable]:
        def on_forward_step_end(model, loss, step):
            # 自动触发指标收集
            collector.collect(step=step)

        return {
            "forward_step_end": on_forward_step_end,
        }
```

### 自动检测和设置

```python
def auto_setup_framework_adapters(collector: MetricsCollector) -> None:
    """自动检测并设置框架适配器"""
    adapters = [
        MegatronAdapter,
        DeepSpeedAdapter,
        HuggingFaceAdapter,
    ]

    for adapter in adapters:
        if adapter.detect():
            adapter.auto_setup(collector)
            logger.info(f"已自动配置 {adapter.__name__}")
```

## 10. 配置驱动

支持YAML配置文件：

```yaml
# metrics_config.yaml
collector:
  output_dir: "./metrics"
  collect_interval_steps: 100
  auto_detect_frameworks: true

providers:
  my_timer:
    enabled: true
    cuda: true

  torch_profiler:
    enabled: true
    profile_memory: true
    activities: ["CPU", "CUDA"]

  module_profiler:
    enabled: false

analysis:
  bottleneck_threshold: 0.1  # 10%
  anomaly_detection: true
  trend_analysis:
    window_size: 100

output:
  formats: ["html", "json"]
  tensorboard: true
  realtime_dashboard: false
```

```python
collector = MetricsCollector.from_config("metrics_config.yaml")
```

## 11. 环境变量支持

```bash
# 全局开关
export ENABLE_METRICS_COLLECTOR=1

# 配置覆盖
export METRICS_OUTPUT_DIR=./custom_metrics
export METRICS_TENSORBOARD=true
export METRICS_COLLECT_INTERVAL=50
```

## 12. 与现有系统集成

```python
# 与ProfileManager集成
class ProfileManager:
    def __init__(self, profile_cfg: dict, logger=None):
        # ... 现有代码 ...

        # 新增：自动设置MetricsCollector
        if os.environ.get("ENABLE_METRICS_COLLECTOR") == "1":
            self.metrics_collector = MetricsCollector()
            auto_setup_framework_adapters(self.metrics_collector)
        else:
            self.metrics_collector = None

    def capture_arm(self, spec: dict) -> None:
        # ... 现有代码 ...

        # 新增：在capture窗口开始时收集指标
        if self.metrics_collector:
            self.metrics_collector.collect()
```

## 13. 性能考虑

1. **零开销禁用** - 所有组件在禁用时返回noop对象
2. **异步写入** - MetricsStore使用后台线程写入
3. **内存限制** - 环形缓冲区，自动清理旧数据
4. **降采样** - 高频指标自动聚合

```python
class MetricsStore:
    def __init__(self, max_memory_mb: int = 100):
        self._max_events = max_memory_mb * 1024 * 1024 // 200  # 假设每事件200字节
        self._buffer = collections.deque(maxlen=self._max_events)
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    def _write_loop(self):
        while True:
            events = self._write_queue.get()
            # 异步写入文件/数据库
```

## 14. 测试策略

```python
# tests/profiling/test_metrics_collector.py
def test_collector_with_multiple_providers():
    collector = MetricsCollector()
    timer = MyTimer(use_cuda=False)

    collector.register_provider(MyTimerMetricsProvider(timer))

    # 模拟训练
    timer.set_step(0)
    timer.start("test")
    timer.stop("test")
    timer.step()

    collector.collect(step=0)

    # 验证
    events = collector._store.read_all_events()
    assert len(events) > 0
    assert events[0].name == "timer.test"
```

## 总结

这个设计方案通过以下方式实现目标：

1. **框架无感知** - 核心系统基于协议和抽象类
2. **模块化** - Provider、Store、Analyzer独立实现
3. **可扩展** - 新增数据源只需实现MetricsProvider
4. **轻量级** - 禁用时所有组件都是noop

下一步实现顺序：
1. 核心协议和数据结构
2. MetricsCollector基础实现
3. 现有工具的Provider适配器
4. MetricsAnalyzer和报告生成
5. TensorBoard插件
6. 框架适配器(可选)
