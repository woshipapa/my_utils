# 可视化增强设计方案

## 设计目标

1. **多种输出格式** - HTML、TensorBoard、Web Dashboard
2. **实时监控** - 支持训练过程中的实时指标展示
3. **交互式分析** - 支持钻取、过滤、对比
4. **框架无关** - 可视化层不依赖特定训练框架

## 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                   VisualizationLayer                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ HTMLReport   │  │ TensorBoard  │  │   Web        │      │
│  │  Generator   │  │   Plugin     │  │ Dashboard    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
│                            ▼                                 │
│              ┌──────────────────────────┐                    │
│              │   Shared Components      │                    │
│              │  - Chart Templates       │                    │
│              │  - Layout Builders       │                    │
│              │  - Data Transform        │                    │
│              └──────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │ MetricsCollector │
                  │ & Analyzer      │
                  └─────────────────┘
```

## 1. 共享组件库

### 1.1 图表模板

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import json

@dataclass
class ChartConfig:
    """图表配置"""
    chart_type: str  # "line", "bar", "pie", "heatmap", "scatter", "gantt"
    title: str
    data: dict[str, Any]
    options: dict[str, Any] = None

    def to_dict(self) -> dict:
        return {
            "type": self.chart_type,
            "title": self.title,
            "data": self.data,
            "options": self.options or {},
        }

class ChartRenderer(ABC):
    """图表渲染器基类"""

    @abstractmethod
    def render(self, config: ChartConfig) -> str:
        """返回渲染后的HTML/SVG/JSON"""
        pass

class ChartJsRenderer(ChartRenderer):
    """Chart.js渲染器"""

    def render(self, config: ChartConfig) -> str:
        template = '''
        <div class="chart-container">
            <canvas id="{chart_id}"></canvas>
        </div>
        <script>
            new Chart(document.getElementById("{chart_id}"), {{
                type: "{chart_type}",
                data: {data},
                options: {options}
            }});
        </script>
        '''
        return template.format(
            chart_id=f"chart_{id(config)}",
            chart_type=config.chart_type,
            data=json.dumps(config.data),
            options=json.dumps(config.options or self._default_options(config)),
        )

    def _default_options(self, config: ChartConfig) -> dict:
        """默认图表选项"""
        return {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"position": "top"},
                "tooltip": {"mode": "index", "intersect": False},
            },
            "interaction": {"mode": "nearest", "axis": "x", "intersect": False},
        }

class PlotlyRenderer(ChartRenderer):
    """Plotly渲染器（支持交互式图表）"""

    def render(self, config: ChartConfig) -> str:
        import plotly.graph_objects as go
        from plotly.offline import plot

        fig = self._create_figure(config)
        return plot(fig, output_type="div", include_plotlyjs=False)

    def _create_figure(self, config: ChartConfig) -> go.Figure:
        """根据配置创建Plotly图表"""
        if config.chart_type == "line":
            return go.Figure(data=go.Scatter(**config.data))
        elif config.chart_type == "bar":
            return go.Figure(data=go.Bar(**config.data))
        elif config.chart_type == "pie":
            return go.Figure(data=go.Pie(**config.data))
        # ...
```

### 1.2 数据转换器

```python
class DataTransformer:
    """将MetricEvent转换为图表友好的格式"""

    @staticmethod
    def to_time_series(events: list[MetricEvent], metric_name: str) -> dict:
        """转换为时间序列数据"""
        filtered = [e for e in events if e.name == metric_name]
        filtered.sort(key=lambda e: e.timestamp)

        return {
            "labels": [e.timestamp for e in filtered],
            "datasets": [{
                "label": metric_name,
                "data": [e.value for e in filtered],
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
            }]
        }

    @staticmethod
    def to_comparison(events: list[MetricEvent], group_by: str) -> dict:
        """转换为对比图表数据"""
        groups: dict[str, list[float]] = {}
        for evt in events:
            key = evt.tags.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(evt.value)

        return {
            "labels": list(groups.keys()),
            "datasets": [{
                "label": f"by {group_by}",
                "data": [np.mean(groups[k]) for k in groups.keys()],
                "backgroundColor": [
                    "rgba(255, 99, 132, 0.7)",
                    "rgba(54, 162, 235, 0.7)",
                    "rgba(255, 206, 86, 0.7)",
                    "rgba(75, 192, 192, 0.7)",
                ][:len(groups)],
            }]
        }

    @staticmethod
    def to_trace(events: list[MetricEvent]) -> dict:
        """转换为Chrome Trace格式"""
        trace_events = []
        for evt in events:
            if evt.tags.get("type") == "START":
                trace_events.append({
                    "name": evt.name,
                    "cat": evt.tags.get("category", "default"),
                    "ph": "B",
                    "ts": int(evt.timestamp * 1_000_000),
                    "pid": evt.tags.get("rank", 0),
                    "tid": evt.tags.get("step", 0),
                    "args": {"value": evt.value, "unit": evt.unit},
                })
            elif evt.tags.get("type") == "END":
                trace_events.append({
                    "name": evt.name,
                    "cat": evt.tags.get("category", "default"),
                    "ph": "E",
                    "ts": int(evt.timestamp * 1_000_000),
                    "pid": evt.tags.get("rank", 0),
                    "tid": evt.tags.get("step", 0),
                })

        return {"traceEvents": trace_events}
```

### 1.3 布局构建器

```python
class LayoutBuilder:
    """HTML布局构建器"""

    def __init__(self):
        self.sections = []
        self.scripts = []
        self.styles = []

    def add_header(self, title: str, subtitle: str = "") -> "LayoutBuilder":
        """添加标题"""
        self.sections.append({
            "type": "header",
            "content": f"<h1>{title}</h1><p class='subtitle'>{subtitle}</p>"
        })
        return self

    def add_summary(self, summary: str, score: float) -> "LayoutBuilder":
        """添加摘要"""
        score_class = "good" if score >= 80 else "warning" if score >= 60 else "critical"

        self.sections.append({
            "type": "summary",
            "content": f"""
            <div class="summary-card score-{score_class}">
                <div class="score-circle">{score:.0f}</div>
                <div class="summary-text">{summary}</div>
            </div>
            """
        })
        return self

    def add_chart(self, config: ChartConfig, renderer: ChartRenderer) -> "LayoutBuilder":
        """添加图表"""
        self.sections.append({
            "type": "chart",
            "content": renderer.render(config)
        })
        return self

    def add_findings(self, findings: list[Finding]) -> "LayoutBuilder":
        """添加发现列表"""
        items = []
        for finding in findings:
            items.append(f"""
            <div class="finding-item severity-{finding.severity.value}">
                <div class="finding-header">
                    <span class="finding-title">{finding.title}</span>
                    <span class="finding-badge">{finding.severity.value}</span>
                </div>
                <div class="finding-description">{finding.description}</div>
                <div class="finding-evidence">
                    {self._format_evidence(finding.evidence)}
                </div>
            </div>
            """)

        self.sections.append({
            "type": "findings",
            "content": f'<div class="findings-list">{"".join(items)}</div>'
        })
        return self

    def add_recommendations(self, recommendations: list[Recommendation]) -> "LayoutBuilder":
        """添加建议列表"""
        items = []
        for rec in recommendations:
            items.append(f"""
            <div class="recommendation-item priority-{rec.priority}">
                <div class="rec-header">
                    <span class="rec-title">{rec.title}</span>
                    <span class="rec-priority">P{rec.priority}</span>
                </div>
                <div class="rec-description">{rec.description}</div>
                <div class="rec-meta">
                    <span class="rec-impact">预期影响: {rec.estimated_impact}</span>
                    <span class="rec-effort">工作量: {rec.effort}</span>
                </div>
                <div class="rec-actions">
                    <strong>建议操作:</strong>
                    <ul>{"".join(f"<li>{a}</li>" for a in rec.actions)}</ul>
                </div>
            </div>
            """)

        self.sections.append({
            "type": "recommendations",
            "content": f'<div class="recommendations-list">{"".join(items)}</div>'
        })
        return self

    def add_table(self, data: list[dict], title: str = "") -> "LayoutBuilder":
        """添加表格"""
        if not data:
            return self

        headers = list(data[0].keys())
        rows = []
        for row in data:
            rows.append("<tr>" + "".join(f"<td>{row[h]}</td>" for h in headers) + "</tr>")

        table_html = f"""
        <div class="table-container">
            {f"<h3>{title}</h3>" if title else ""}
            <table class="data-table">
                <thead>
                    <tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr>
                </thead>
                <tbody>{"".join(rows)}</tbody>
            </table>
        </div>
        """

        self.sections.append({"type": "table", "content": table_html})
        return self

    def _format_evidence(self, evidence: dict) -> str:
        """格式化证据"""
        items = []
        for key, value in evidence.items():
            if key not in ["components", "metrics"]:
                if isinstance(value, float):
                    items.append(f"<strong>{key}:</strong> {value:.4f}")
                else:
                    items.append(f"<strong>{key}:</strong> {value}")

        return "<br>".join(items)

    def build(self) -> str:
        """构建完整HTML"""
        template = '''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Performance Analysis Report</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
            <style>{css}</style>
        </head>
        <body>
            <div class="container">{sections}</div>
            <script>{scripts}</script>
        </body>
        </html>
        '''

        css = self._get_css()
        sections = "".join(s["content"] for s in self.sections)
        scripts = "\n".join(self.scripts)

        return template.format(css=css, sections=sections, scripts=scripts)

    def _get_css(self) -> str:
        """获取CSS样式"""
        return '''
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f7fa;
            color: #2c3e50;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .summary-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 24px;
        }
        .score-good { border-left: 4px solid #10b981; }
        .score-warning { border-left: 4px solid #f59e0b; }
        .score-critical { border-left: 4px solid #ef4444; }
        .score-circle {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            font-weight: bold;
            color: white;
        }
        .score-good .score-circle { background: #10b981; }
        .score-warning .score-circle { background: #f59e0b; }
        .score-critical .score-circle { background: #ef4444; }
        .finding-item, .recommendation-item {
            background: white;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border-left: 4px solid #cbd5e1;
        }
        .severity-critical { border-left-color: #ef4444; }
        .severity-high { border-left-color: #f97316; }
        .severity-medium { border-left-color: #f59e0b; }
        .finding-header, .rec-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .finding-title, .rec-title { font-weight: 600; }
        .finding-badge, .rec-priority {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            height: 400px;
        }
        .table-container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            overflow-x: auto;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
        }
        .data-table th, .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        .data-table th {
            font-weight: 600;
            background: #f8fafc;
        }
        '''
```

## 2. HTML报告生成器

```python
class HTMLReportGenerator:
    """HTML报告生成器"""

    def __init__(self, renderer: ChartRenderer = None):
        self.renderer = renderer or ChartJsRenderer()
        self.transformer = DataTransformer()

    def generate(self, report: AnalysisReport, events: list[MetricEvent]) -> str:
        """生成完整HTML报告"""
        builder = LayoutBuilder()

        # 1. 标题
        builder.add_header(
            "Performance Analysis Report",
            f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # 2. 摘要
        builder.add_summary(report.summary, report.overall_score)

        # 3. 性能趋势图
        self._add_trend_charts(builder, events)

        # 4. 瓶颈分析图
        self._add_bottleneck_charts(builder, report.findings)

        # 5. 内存分析图
        self._add_memory_charts(builder, events)

        # 6. 发现列表
        builder.add_findings(report.findings)

        # 7. 优化建议
        builder.add_recommendations(report.recommendations)

        # 8. 详细数据表
        self._add_detail_tables(builder, report, events)

        return builder.build()

    def _add_trend_charts(self, builder: LayoutBuilder, events: list[MetricEvent]) -> None:
        """添加趋势图表"""
        # 按step分组的时间趋势
        time_events = [e for e in events if "timer" in e.name.lower()]

        if time_events:
            by_step: dict[int, list[MetricEvent]] = {}
            for evt in time_events:
                step = int(evt.tags.get("step", 0))
                if step not in by_step:
                    by_step[step] = []
                by_step[step].append(evt)

            # 每个timer的总时间
            timer_totals: dict[str, dict[int, float]] = {}
            for step, step_events in by_step.items():
                for evt in step_events:
                    name = evt.name
                    if name not in timer_totals:
                        timer_totals[name] = {}
                    timer_totals[name][step] = timer_totals[name].get(step, 0) + evt.value

            # 创建折线图
            for timer_name, step_data in timer_totals.items():
                sorted_steps = sorted(step_data.keys())
                config = ChartConfig(
                    chart_type="line",
                    title=f"{timer_name} Trend",
                    data={
                        "labels": sorted_steps,
                        "datasets": [{
                            "label": f"{timer_name} (ms)",
                            "data": [step_data[s] for s in sorted_steps],
                            "borderColor": "rgb(75, 192, 192)",
                            "backgroundColor": "rgba(75, 192, 192, 0.2)",
                            "fill": True,
                        }]
                    },
                    options={
                        "responsive": True,
                        "maintainAspectRatio": False,
                        "scales": {
                            "y": {"beginAtZero": True},
                            "x": {"title": {"display": True, "text": "Step"}}
                        }
                    }
                )
                builder.add_chart(config, self.renderer)

    def _add_bottleneck_charts(self, builder: LayoutBuilder, findings: list[Finding]) -> None:
        """添加瓶颈分析图表"""
        bottlenecks = [f for f in findings if "瓶颈" in f.title]

        if bottlenecks:
            # 饼图
            config = ChartConfig(
                chart_type="pie",
                title="Performance Bottlenecks",
                data={
                    "labels": [b.evidence.get("component", b.title) for b in bottlenecks],
                    "datasets": [{
                        "data": [b.evidence.get("ratio", 0) * 100 for b in bottlenecks],
                        "backgroundColor": [
                            "rgba(239, 68, 68, 0.8)",
                            "rgba(249, 115, 22, 0.8)",
                            "rgba(245, 158, 11, 0.8)",
                            "rgba(34, 197, 94, 0.8)",
                        ][:len(bottlenecks)]
                    }]
                }
            )
            builder.add_chart(config, self.renderer)

    def _add_memory_charts(self, builder: LayoutBuilder, events: list[MetricEvent]) -> None:
        """添加内存分析图表"""
        memory_events = [e for e in events if "memory" in e.name.lower()]

        if memory_events:
            # 按类型分组
            by_type: dict[str, list[tuple[float, float]]] = {}  # (timestamp, value)
            for evt in memory_events:
                mem_type = evt.name.split(".")[-1]  # allocated, reserved等
                if mem_type not in by_type:
                    by_type[mem_type] = []
                by_type[mem_type].append((evt.timestamp, evt.value))

            # 折线图
            datasets = []
            colors = {"allocated": "rgb(75, 192, 192)", "reserved": "rgb(255, 99, 132)"}
            for mem_type, data in by_type.items():
                data.sort(key=lambda x: x[0])
                datasets.append({
                    "label": f"Memory {mem_type}",
                    "data": [d[1] for d in data],
                    "borderColor": colors.get(mem_type, "rgb(153, 102, 255)"),
                    "backgroundColor": colors.get(mem_type, "rgb(153, 102, 255)").replace("rgb", "rgba").replace(")", ", 0.2)"),
                })

            config = ChartConfig(
                chart_type="line",
                title="Memory Usage Over Time",
                data={
                    "labels": list(set([e.timestamp for e in memory_events])),
                    "datasets": datasets
                },
                options={
                    "responsive": True,
                    "scales": {
                        "y": {"beginAtZero": True, "title": {"display": True, "text": "Memory (GB)"}},
                        "x": {"title": {"display": True, "text": "Time"}}
                    }
                }
            )
            builder.add_chart(config, self.renderer)

    def _add_detail_tables(self, builder: LayoutBuilder, report: AnalysisReport, events: list[MetricEvent]) -> None:
        """添加详细数据表格"""
        # 指标统计表
        stats = self._compute_statistics(events)
        builder.add_table(stats, title="Metrics Statistics")

    def _compute_statistics(self, events: list[MetricEvent]) -> list[dict]:
        """计算统计数据"""
        by_name: dict[str, list[float]] = {}
        for evt in events:
            name = evt.name
            if name not in by_name:
                by_name[name] = []
            by_name[name].append(evt.value)

        stats = []
        for name, values in by_name.items():
            stats.append({
                "Metric": name,
                "Count": len(values),
                "Mean": f"{np.mean(values):.4f}",
                "Std": f"{np.std(values):.4f}",
                "Min": f"{np.min(values):.4f}",
                "Max": f"{np.max(values):.4f}",
                "Unit": events[[e.name for e in events].index(name)].unit,
            })

        return sorted(stats, key=lambda x: float(x["Mean"]), reverse=True)
```

## 3. TensorBoard插件

```python
# my_utils/profiling/tensorboard_plugin/__init__.py

"""
TensorBoard插件用于可视化my_utils收集的指标

安装:
pip install tensorboard

使用:
tensorboard --logdir=./metrics_logs --load_plugins=my_utils.profiling.tensorboard_plugin
"""

from tensorboard.plugins import base_plugin
from tensorboard.plugins.hparams import api as hp
from tensorboard.data import provider
from tensorboard.types import plugin_data_pb2 as plugin_data_pb2
import json

class MetricsPluginLoader(base_plugin.TBLoader):
    """插件加载器"""

    def load(self, context: base_plugin.TBContext) -> "MetricsPlugin":
        return MetricsPlugin(context)

class MetricsPlugin(base_plugin.TBPlugin):
    """TensorBoard Metrics插件"""

    plugin_name = "my_utils_metrics"

    def __init__(self, context: base_plugin.TBContext):
        self._context = context
        self._data_provider = MetricsDataProvider(context.logdir)

    def get_plugin_apps(self):
        """返回Flask应用"""
        from flask import Blueprint
        bp = Blueprint(self.plugin_name, __name__)

        @bp.route("/index")
        def index():
            return self._render_index()

        @bp.route("/data")
        def data():
            from flask import jsonify
            return jsonify(self._get_data())

        @bp.route("/ws")
        def websocket():
            """WebSocket端点，用于实时更新"""
            # 实现WebSocket逻辑
            pass

        return [bp]

    def _render_index(self) -> str:
        """渲染插件主页"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metrics Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                .dashboard { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; padding: 20px; }
                .card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .full-width { grid-column: 1 / -1; }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="card full-width">
                    <h2>Real-time Metrics</h2>
                    <canvas id="realtimeChart"></canvas>
                </div>
                <div class="card">
                    <h2>Memory Usage</h2>
                    <canvas id="memoryChart"></canvas>
                </div>
                <div class="card">
                    <h2>Top Bottlenecks</h2>
                    <div id="bottlenecks"></div>
                </div>
            </div>
            <script>
                // WebSocket连接
                const ws = new WebSocket(`ws://${location.host}/data/plugins/my_utils_metrics/ws`);

                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    updateCharts(data);
                };

                function updateCharts(data) {
                    // 更新图表
                }
            </script>
        </body>
        </html>
        '''

    def _get_data(self) -> dict:
        """获取当前数据"""
        return self._data_provider.get_latest_data()

class MetricsDataProvider:
    """数据提供者，读取MetricsCollector的输出"""

    def __init__(self, logdir: str):
        self.logdir = logdir
        self._cache = {}
        self._last_update = 0

    def get_latest_data(self) -> dict:
        """获取最新数据"""
        import time
        import os

        # 检查是否有新数据
        metrics_file = os.path.join(self.logdir, "metrics.json")
        if os.path.exists(metrics_file):
            mtime = os.path.getmtime(metrics_file)
            if mtime > self._last_update:
                with open(metrics_file) as f:
                    self._cache = json.load(f)
                self._last_update = mtime

        return self._cache
```

## 4. Web Dashboard (可选)

```python
# my_utils/profiling/dashboard/app.py

"""
独立的Web Dashboard服务器

使用:
python -m my_utils.profiling.dashboard --logdir=./metrics_logs --port=8080
"""

from flask import Flask, render_template, jsonify, Response
from flask_sock import Sock
import json
import os
import time
from pathlib import Path

app = Flask(__name__)
sock = Sock(app)

class DashboardServer:
    """Dashboard服务器"""

    def __init__(self, logdir: str):
        self.logdir = Path(logdir)
        self._clients = set()

    def watch_files(self):
        """监控文件变化"""
        """后台线程，监控metrics文件变化并通过WebSocket推送"""

    def broadcast_update(self, data: dict):
        """向所有客户端广播更新"""
        import json
        message = json.dumps(data)
        for client in self._clients:
            try:
                client.send(message)
            except Exception:
                self._clients.remove(client)

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/metrics")
def get_metrics():
    """获取当前指标"""
    # 读取并返回metrics数据
    pass

@sock.route("/ws")
def websocket_connection(ws):
    """WebSocket连接处理"""
    server = app.config["dashboard_server"]
    server._clients.add(ws)

    try:
        while True:
            ws.receive()  # 保持连接
    except Exception:
        pass
    finally:
        server._clients.remove(ws)

# templates/dashboard.html
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Training Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue@3/dist/vue.global.js"></script>
    <style>
        body { margin: 0; font-family: system-ui; background: #f5f7fa; }
        .app { display: grid; grid-template-rows: auto 1fr; height: 100vh; }
        .header { background: #1e293b; color: white; padding: 1rem 2rem; }
        .content { padding: 2rem; overflow-y: auto; }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
        }
        .card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e293b;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .chart-container {
            position: relative;
            height: 250px;
        }
    </style>
</head>
<body>
    <div id="app" class="app">
        <div class="header">
            <h1>Training Metrics Dashboard</h1>
            <span>Last update: {{ lastUpdate }}</span>
        </div>
        <div class="content">
            <div class="metrics-grid">
                <div class="card">
                    <div class="metric-label">Current Step</div>
                    <div class="metric-value">{{ metrics.step || '-' }}</div>
                </div>
                <div class="card">
                    <div class="metric-label">Throughput (samples/s)</div>
                    <div class="metric-value">{{ metrics.throughput?.toFixed(1) || '-' }}</div>
                </div>
                <div class="card">
                    <div class="metric-label">Loss</div>
                    <div class="metric-value">{{ metrics.loss?.toFixed(4) || '-' }}</div>
                </div>
                <div class="card">
                    <div class="metric-label">Memory (GB)</div>
                    <div class="metric-value">{{ metrics.memory?.toFixed(2) || '-' }}</div>
                </div>
            </div>

            <div class="metrics-grid" style="margin-top: 2rem;">
                <div class="card" style="grid-column: span 2;">
                    <h3>Loss Curve</h3>
                    <div class="chart-container">
                        <canvas id="lossChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <h3>Performance Score</h3>
                    <div class="chart-container">
                        <canvas id="scoreChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const { createApp } = Vue;

        createApp({
            data() {
                return {
                    metrics: {},
                    lastUpdate: 'Never',
                    ws: null,
                    charts: {}
                };
            },
            mounted() {
                this.initCharts();
                this.connectWebSocket();
            },
            methods: {
                initCharts() {
                    this.charts.loss = new Chart(document.getElementById('lossChart'), {
                        type: 'line',
                        data: { labels: [], datasets: [{ label: 'Loss', data: [] }] },
                        options: { responsive: true, maintainAspectRatio: false }
                    });

                    this.charts.score = new Chart(document.getElementById('scoreChart'), {
                        type: 'doughnut',
                        data: {
                            labels: ['Compute', 'Memory', 'Communication', 'IO'],
                            datasets: [{ data: [40, 30, 20, 10] }]
                        },
                        options: { responsive: true, maintainAspectRatio: false }
                    });
                },
                connectWebSocket() {
                    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                    this.ws = new WebSocket(`${protocol}//${location.host}/ws`);

                    this.ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        this.metrics = { ...this.metrics, ...data };
                        this.lastUpdate = new Date().toLocaleTimeString();
                        this.updateCharts(data);
                    };
                },
                updateCharts(data) {
                    if (data.step !== undefined && data.loss !== undefined) {
                        this.charts.loss.data.labels.push(data.step);
                        this.charts.loss.data.datasets[0].data.push(data.loss);
                        this.charts.loss.update('none');
                    }
                }
            }
        }).mount('#app');
    </script>
</body>
</html>
'''
```

## 5. 使用示例

```python
from my_utils.profiling import (
    MetricsCollector,
    AnalyzerPipeline,
    HTMLReportGenerator,
)

# 收集指标
collector = MetricsCollector(output_dir="./metrics_logs")
# ... 注册providers ...
# ... 训练 ...

# 生成分析报告
pipeline = AnalyzerPipeline()
events = collector._store.read_all_events()
report = pipeline.analyze(events)

# 生成HTML报告
generator = HTMLReportGenerator()
html_path = "./metrics_logs/report.html"
with open(html_path, "w") as f:
    f.write(generator.generate(report, events))

print(f"HTML报告已生成: {html_path}")

# 启动TensorBoard
# tensorboard --logdir=./metrics_logs --load_plugins=my_utils.profiling.tensorboard_plugin

# 或启动独立Dashboard
# python -m my_utils.profiling.dashboard --logdir=./metrics_logs
```

## 总结

可视化增强通过多层次方案满足不同需求：

1. **HTML报告** - 静态、可分享、完整分析
2. **TensorBoard插件** - 集成到现有工作流
3. **Web Dashboard** - 实时监控、多用户

下一步：
1. 实现HTMLReportGenerator
2. 开发TensorBoard插件
3. 添加更多图表类型（热力图、桑基图等）
4. 支持自定义主题和布局
