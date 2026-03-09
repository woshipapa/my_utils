# 可视化增强 - 使用指南

## 概述

可视化增强模块为 `my_utils` 性能分析工具提供了强大的报告生成能力，支持多种图表类型和输出格式。

## 主要特性

- ✅ **多种图表类型** - 折线图、柱状图、饼图、散点图等
- ✅ **多渲染器支持** - Chart.js、Plotly、ECharts
- ✅ **灵活的布局系统** - 自定义报告布局
- ✅ **自动数据转换** - 将原始数据转换为图表友好格式
- ✅ **美观的HTML报告** - 开箱即用的报告样式
- ✅ **框架无关** - 可独立使用或集成到任何项目

## 快速开始

### 安装

可视化组件是 `my_utils` 的一部分，无需额外安装：

```bash
cd /path/to/my_utils
pip install -e .
```

### 基础使用

#### 1. 创建简单图表

```python
from my_utils.profiling.visualization import ChartConfig, create_chart_renderer

# 创建渲染器
renderer = create_chart_renderer()

# 创建图表配置
config = ChartConfig(
    chart_type="line",
    title="Performance Over Time",
    data={
        "labels": [1, 2, 3, 4, 5],
        "datasets": [{
            "label": "Loss",
            "data": [2.5, 2.1, 1.8, 1.5, 1.3],
            "borderColor": "rgb(75, 192, 192)",
            "backgroundColor": "rgba(75, 192, 192, 0.2)",
            "fill": True,
        }]
    }
)

# 渲染为HTML
html = renderer.render(config)
```

#### 2. 使用数据转换器

```python
from my_utils.profiling.visualization import DataTransformer, MetricEvent

# 创建事件数据
events = [
    MetricEvent(
        timestamp=time.time(),
        name="loss",
        value=0.5,
        unit="",
        tags={"step": "1"}
    ),
    # ... 更多事件
]

# 转换数据
transformer = DataTransformer()

# 时间序列
time_series = transformer.to_time_series(events, metric_name="loss")

# 统计信息
stats = transformer.compute_statistics(events)
```

#### 3. 构建报告

```python
from my_utils.profiling.visualization import (
    LayoutBuilder,
    Finding,
    Recommendation,
    AnalysisReport,
    HTMLReportGenerator,
    Severity,
)

# 创建布局
builder = LayoutBuilder()
builder.add_header(title="My Report")
builder.add_summary(summary="Training completed", score=85)
builder.add_metrics_grid({"Loss": "0.25", "Accuracy": "0.92"})

# 生成HTML
html = builder.build()
```

#### 4. 完整报告生成

```python
# 创建报告对象
report = AnalysisReport(
    metadata={"event_count": 1000},
    findings=[
        Finding(
            id="f1",
            title="High Memory Usage",
            description="Memory usage is above threshold",
            severity=Severity.HIGH,
            category="memory",
            evidence={"usage_gb": 8.5},
            affected_components=["model"],
            metrics={}
        )
    ],
    recommendations=[
        Recommendation(
            id="r1",
            title="Reduce Batch Size",
            description="Consider reducing batch size",
            priority=8,
            estimated_impact="20% less memory",
            effort="low",
            actions=["Reduce batch size by 2x"],
            references=[]
        )
    ],
    summary="Training shows good performance",
    overall_score=75.0
)

# 生成报告
generator = HTMLReportGenerator()
html = generator.generate(report, events, output_path="report.html")
```

## 从现有工具生成报告

### 从 MyTimer 生成

```python
from my_utils.utils import MyTimer
from my_utils.profiling.visualization import QuickReportGenerator

# 使用MyTimer
timer = MyTimer(use_cuda=True)
timer.set_logger(logger)

for step in range(100):
    timer.set_step(step)
    timer.start("forward")
    # ... 训练代码 ...
    timer.stop("forward")
    timer.step()

# 生成报告
quick_gen = QuickReportGenerator()
html = quick_gen.generate_from_timer(timer, "timer_report.html")
```

### 从 CSV 文件生成

```python
from my_utils.profiling.visualization import QuickReportGenerator

quick_gen = QuickReportGenerator()
html = quick_gen.generate_from_csv(
    csv_path="profile_rank_0.csv",
    output_path="csv_report.html"
)
```

## 高级功能

### 切换图表渲染器

```python
from my_utils.profiling.visualization import PlotlyRenderer, EChartsRenderer

# 使用Plotly（更丰富的交互）
plotly_renderer = PlotlyRenderer()
html = plotly_renderer.render(config)

# 使用ECharts（更多图表类型）
echarts_renderer = EChartsRenderer()
html = echarts_renderer.render(config)
```

### 自定义样式

```python
builder = LayoutBuilder()

# 添加自定义CSS
builder.add_style("""
    .custom-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
""")

# 添加自定义JavaScript
builder.add_script("""
    console.log("Report loaded");
""")
```

### 创建复杂布局

```python
builder = LayoutBuilder()

# 两列图表
from my_utils.profiling.visualization import create_chart_renderer

renderer = create_chart_renderer()
left_chart = renderer.render(left_config)
right_chart = renderer.render(right_config)

builder.add_two_column_charts(
    left_chart_html=left_chart,
    right_chart_html=right_chart,
    left_title="Training Loss",
    right_title="Validation Accuracy"
)
```

## API 参考

### ChartConfig

图表配置类

```python
@dataclass
class ChartConfig:
    chart_type: str  # line, bar, pie, doughnut, scatter
    title: str
    data: dict
    options: dict = field(default_factory=dict)
    width: Optional[str] = None
    height: Optional[str] = None
```

### MetricEvent

统一的指标事件格式

```python
@dataclass
class MetricEvent:
    timestamp: float
    name: str
    value: float
    unit: str = ""
    tags: dict[str, Any] = None
```

### DataTransformer

数据转换器类

**方法:**
- `to_time_series(events, metric_name)` - 转换为时间序列
- `to_multiple_time_series(events, metric_names)` - 多条时间序列
- `to_comparison(events, metric_name, group_by)` - 对比图表
- `to_pie_chart(events, metric_name)` - 饼图数据
- `compute_statistics(events, metric_name)` - 统计信息

### LayoutBuilder

HTML布局构建器

**方法:**
- `add_header(title, subtitle)` - 添加标题
- `add_summary(summary, score, details)` - 添加摘要
- `add_metrics_grid(metrics)` - 添加指标网格
- `add_chart(chart_html, title)` - 添加图表
- `add_findings(findings)` - 添加发现列表
- `add_recommendations(recommendations)` - 添加建议列表
- `add_table(data, title)` - 添加表格
- `build()` - 构建完整HTML

### HTMLReportGenerator

报告生成器

```python
generator = HTMLReportGenerator(
    renderer=None,  # 自动选择
    transformer=None,  # 使用默认
)

html = generator.generate(
    report=AnalysisReport,
    events=list[MetricEvent],
    output_path=str  # 可选，自动保存
)
```

## 图表类型

### 折线图 (line)

用于显示趋势：

```python
config = ChartConfig(
    chart_type="line",
    title="Loss Over Time",
    data={
        "labels": [1, 2, 3, 4, 5],
        "datasets": [{
            "label": "Loss",
            "data": [2.5, 2.1, 1.8, 1.5, 1.3],
            "borderColor": "rgb(75, 192, 192)",
        }]
    }
)
```

### 柱状图 (bar)

用于对比：

```python
config = ChartConfig(
    chart_type="bar",
    title="Layer-wise Time",
    data={
        "labels": ["Layer 1", "Layer 2", "Layer 3"],
        "datasets": [{
            "data": [100, 150, 120],
        }]
    }
)
```

### 饼图 (pie)

用于占比分析：

```python
config = ChartConfig(
    chart_type="pie",
    title="Time Distribution",
    data={
        "labels": ["Forward", "Backward", "Optimizer"],
        "datasets": [{
            "data": [40, 50, 10],
        }]
    }
)
```

### 环形图 (doughnut)

类似饼图，中心有空洞：

```python
config = ChartConfig(
    chart_type="doughnut",
    title="Memory Usage",
    data={
        "labels": ["Model", "Optimizer", "Gradients"],
        "datasets": [{
            "data": [60, 20, 20],
        }]
    }
)
```

## 样式定制

### 颜色主题

默认使用的颜色：

```python
COLORS = [
    ("rgb(75, 192, 192)", "rgba(75, 192, 192, 0.2)"),  # 青色
    ("rgb(255, 99, 132)", "rgba(255, 99, 132, 0.2)"),  # 红色
    ("rgb(54, 162, 235)", "rgba(54, 162, 235, 0.2)"),  # 蓝色
    ("rgb(255, 206, 86)", "rgba(255, 206, 86, 0.2)"),  # 黄色
    ("rgb(153, 102, 255)", "rgba(153, 102, 255, 0.2)"),  # 紫色
]
```

### 严重程度颜色

- **Critical**: #ef4444 (红色)
- **High**: #f97316 (橙色)
- **Medium**: #f59e0b (黄色)
- **Low**: #3b82f6 (蓝色)
- **Info**: #6b7280 (灰色)

### 得分颜色

- **80-100**: #10b981 (绿色) - 良好
- **60-79**: #f59e0b (黄色) - 警告
- **0-59**: #ef4444 (红色) - 严重

## 最佳实践

### 1. 数据聚合

对于大量数据点，先聚合再绘图：

```python
transformer = DataTransformer()

# 按时间窗口聚合
aggregated_events = transformer.aggregate_by_time_window(
    events,
    window_size=1.0,  # 1秒
    aggregation="mean"
)
```

### 2. 数据过滤

使用标签过滤事件：

```python
# 只看特定rank的数据
rank0_events = transformer.filter_by_tags(events, {"rank": "0"})

# 只看特定step
step10_events = transformer.filter_by_tags(events, {"step": "10"})
```

### 3. 多图表组合

使用两列布局展示相关图表：

```python
builder.add_two_column_charts(
    left_chart_html=loss_chart,
    right_chart_html=accuracy_chart,
    left_title="Training Loss",
    right_title="Validation Accuracy"
)
```

### 4. 响应式设计

图表默认响应式，在不同屏幕上自动调整：

```python
config = ChartConfig(
    chart_type="line",
    title="Responsive Chart",
    data={...},
    options={
        "responsive": True,
        "maintainAspectRatio": False,
    }
)
```

## 示例代码

完整示例请参考：

- `my_utils/profiling/visualization/examples.py` - 详细示例
- `test_visualization.py` - 快速测试脚本

运行测试：

```bash
cd /path/to/my_utils
python test_visualization.py
```

生成的文件：
- `test_chart.html` - 基本图表
- `test_layout.html` - 布局示例
- `test_full_report.html` - 完整报告

## 常见问题

### Q: 如何切换到 Plotly？

A: 创建渲染器时指定：

```python
from my_utils.profiling.visualization import PlotlyRenderer

renderer = PlotlyRenderer()
generator = HTMLReportGenerator(renderer=renderer)
```

### Q: 如何自定义报告样式？

A: 使用 `add_style` 方法：

```python
builder.add_style("""
    body { background: #f0f0f0; }
    .metric-card { border: 2px solid blue; }
""")
```

### Q: 如何导出为PDF？

A: 使用浏览器的打印功能，或：

```python
# 安装 weasyprint
pip install weasyprint

from weasyprint import HTML

HTML(html).write_pdf("report.pdf")
```

### Q: 图表太大怎么办？

A: 设置固定高度：

```python
config = ChartConfig(
    ...
    height="300px",
)
```

### Q: 支持实时更新吗？

A: 静态HTML不支持实时更新，但可以：
1. 定期重新生成报告
2. 使用 WebSocket 实现实时更新（需要额外开发）
3. 集成到 TensorBoard（规划中）

## 未来计划

- [ ] TensorBoard 插件
- [ ] Web Dashboard 服务器
- [ ] 实时数据更新
- [ ] 更多图表类型（热力图、桑基图等）
- [ ] PDF 导出
- [ ] 交互式钻取功能

## 贡献

欢迎贡献代码和提出建议！

查看 [ROADMAP.md](../docs/ROADMAP.md) 了解完整开发计划。
