# SPDX-License-Identifier: Apache-2.0
"""
图表渲染组件

提供统一的图表配置和渲染接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class ChartConfig:
    """图表配置"""

    chart_type: str  # "line", "bar", "pie", "heatmap", "scatter", "doughnut"
    title: str
    data: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    width: Optional[str] = None
    height: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "type": self.chart_type,
            "title": self.title,
            "data": self.data,
            "options": self.options,
            "width": self.width,
            "height": self.height,
        }


class ChartRenderer(ABC):
    """图表渲染器基类"""

    @abstractmethod
    def render(self, config: ChartConfig) -> str:
        """
        渲染图表为HTML

        Args:
            config: 图表配置

        Returns:
            HTML字符串
        """
        pass

    @abstractmethod
    def render_to_json(self, config: ChartConfig) -> str:
        """
        渲染图表为JSON格式（用于API返回）

        Args:
            config: 图表配置

        Returns:
            JSON字符串
        """
        pass

    def _get_chart_id(self, config: ChartConfig) -> str:
        """生成唯一的图表ID"""
        import hashlib

        content = f"{config.chart_type}_{config.title}_{config.data}"
        hash_hex = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chart_{hash_hex}"

    def _get_default_options(self, config: ChartConfig) -> dict:
        """获取默认选项"""
        return {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"position": "top"},
                "tooltip": {"mode": "index", "intersect": False},
            },
        }


class ChartJsRenderer(ChartRenderer):
    """
    Chart.js 渲染器

    使用 Chart.js 库渲染图表
    """

    def __init__(self, version: str = "4.4.0"):
        self.version = version
        self.cdn_url = f"https://cdn.jsdelivr.net/npm/chart.js@{self.version}/dist/chart.umd.min.js"

    def render(self, config: ChartConfig) -> str:
        """渲染为HTML"""
        chart_id = self._get_chart_id(config)
        options = self._merge_options(config)

        template = """
        <div class="chart-container" {style}>
            <canvas id="{chart_id}"></canvas>
        </div>
        <script>
            (function() {{
                const ctx = document.getElementById('{chart_id}');
                if (ctx) {{
                    new Chart(ctx, {{
                        type: '{chart_type}',
                        data: {data},
                        options: {options}
                    }});
                }}
            }})();
        </script>
        """

        style = ""
        if config.width:
            style += f"width: {config.width};"
        if config.height:
            style += f"height: {config.height};"

        return template.format(
            chart_id=chart_id,
            chart_type=config.chart_type,
            data=json.dumps(config.data, ensure_ascii=False),
            options=json.dumps(options, ensure_ascii=False),
            style=f'style="{style}"' if style else "",
        )

    def render_to_json(self, config: ChartConfig) -> str:
        """渲染为JSON"""
        return json.dumps(
            {
                "library": "chart.js",
                "version": self.version,
                "config": config.to_dict(),
            },
            ensure_ascii=False,
        )

    def _merge_options(self, config: ChartConfig) -> dict:
        """合并默认选项和自定义选项"""
        default_opts = self._get_default_options(config)

        # 根据图表类型添加特定选项
        type_specific = self._get_type_specific_options(config.chart_type)

        # 深度合并选项
        merged = self._deep_merge(default_opts, type_specific)
        merged = self._deep_merge(merged, config.options)

        return merged

    def _get_type_specific_options(self, chart_type: str) -> dict:
        """获取图表类型特定的选项"""
        options_map = {
            "line": {
                "scales": {
                    "x": {"display": True, "title": {"display": True}},
                    "y": {"display": True, "beginAtZero": True},
                },
                "elements": {"line": {"tension": 0.4}},  # 平滑曲线
            },
            "bar": {
                "scales": {
                    "x": {"display": True, "title": {"display": True}},
                    "y": {"display": True, "beginAtZero": True},
                }
            },
            "pie": {"plugins": {"legend": {"position": "right"}}},
            "doughnut": {"plugins": {"legend": {"position": "right"}}},
        }
        return options_map.get(chart_type, {})

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """深度合并两个字典"""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


class PlotlyRenderer(ChartRenderer):
    """
    Plotly 渲染器

    使用 Plotly 库渲染交互式图表
    """

    def __init__(self, offline: bool = True):
        self.offline = offline

    def render(self, config: ChartConfig) -> str:
        """渲染为HTML"""
        try:
            import plotly.graph_objects as go  # noqa: F401 -- availability probe
            from plotly.offline import plot
        except ImportError:
            # Plotly不可用，回退到Chart.js
            return ChartJsRenderer().render(config)

        fig = self._create_figure(config)

        # 使用offline plot生成HTML
        plot_div = plot(
            fig,
            output_type="div",
            include_plotlyjs="inline" if self.offline else False,
            config={"displayModeBar": True, "displaylogo": False},
        )

        # 包装在容器中
        return f"""
        <div class="chart-container plotly-chart" {'style="width:' + config.width + ';"' if config.width else ""}>
            {plot_div}
        </div>
        """

    def render_to_json(self, config: ChartConfig) -> str:
        """渲染为JSON"""
        return json.dumps(
            {
                "library": "plotly",
                "config": config.to_dict(),
            },
            ensure_ascii=False,
        )

    def _create_figure(self, config: ChartConfig):
        """根据配置创建Plotly图表"""
        import plotly.graph_objects as go

        fig = go.Figure()

        if config.chart_type == "line":
            for dataset in config.data.get("datasets", []):
                fig.add_trace(
                    go.Scatter(
                        x=config.data.get("labels", []),
                        y=dataset.get("data", []),
                        name=dataset.get("label", ""),
                        mode="lines+markers",
                        line={"color": dataset.get("borderColor")},
                    )
                )

        elif config.chart_type == "bar":
            for dataset in config.data.get("datasets", []):
                fig.add_trace(
                    go.Bar(
                        x=config.data.get("labels", []),
                        y=dataset.get("data", []),
                        name=dataset.get("label", ""),
                        marker={"color": dataset.get("backgroundColor")},
                    )
                )

        elif config.chart_type == "pie":
            fig.add_trace(
                go.Pie(
                    labels=config.data.get("labels", []),
                    values=config.data.get("datasets", [{}])[0].get("data", []),
                )
            )

        elif config.chart_type == "doughnut":
            fig.add_trace(
                go.Pie(
                    labels=config.data.get("labels", []),
                    values=config.data.get("datasets", [{}])[0].get("data", []),
                    hole=0.4,
                )
            )

        # 添加标题
        if config.title:
            fig.update_layout(title=config.title)

        # 应用自定义选项
        if config.options:
            fig.update_layout(**self._convert_options(config.options))

        return fig

    def _convert_options(self, options: dict) -> dict:
        """转换Chart.js风格选项到Plotly风格"""
        # 简化处理，只转换常用的
        plotly_options = {}

        if "scales" in options:
            scales = options["scales"]
            if "y" in scales and scales["y"].get("beginAtZero"):
                plotly_options["yaxis"] = {"rangemode": "tozero"}

        return plotly_options


class EChartsRenderer(ChartRenderer):
    """
    Apache ECharts 渲染器

    使用 ECharts 库渲染图表（功能更丰富）
    """

    def __init__(self, version: str = "5.4.3"):
        self.version = version
        self.cdn_url = (
            f"https://cdn.jsdelivr.net/npm/echarts@{self.version}/dist/echarts.min.js"
        )

    def render(self, config: ChartConfig) -> str:
        """渲染为HTML"""
        chart_id = self._get_chart_id(config)
        option = self._convert_to_echarts_option(config)

        template = """
        <div class="chart-container echarts-chart" {style}>
            <div id="{chart_id}" style="width:100%;height:100%;"></div>
        </div>
        <script>
            (function() {{
                const chart = echarts.init(document.getElementById('{chart_id}'));
                const option = {option};
                chart.setOption(option);
                window.addEventListener('resize', function() {{
                    chart.resize();
                }});
            }})();
        </script>
        """

        style = ""
        if config.width:
            style += f"width:{config.width};"
        if config.height:
            style += f"height:{config.height};"

        return template.format(
            chart_id=chart_id,
            option=json.dumps(option, ensure_ascii=False),
            style=f'style="{style}"' if style else "",
        )

    def render_to_json(self, config: ChartConfig) -> str:
        """渲染为JSON"""
        return json.dumps(
            {
                "library": "echarts",
                "version": self.version,
                "option": self._convert_to_echarts_option(config),
            },
            ensure_ascii=False,
        )

    def _convert_to_echarts_option(self, config: ChartConfig) -> dict:
        """转换为ECharts option格式"""
        option = {
            "title": {"text": config.title},
            "tooltip": {
                "trigger": "axis" if config.chart_type in ["line", "bar"] else "item"
            },
        }

        if config.chart_type == "line":
            option.update(
                {
                    "xAxis": {
                        "type": "category",
                        "data": config.data.get("labels", []),
                    },
                    "yAxis": {"type": "value"},
                    "series": [
                        {
                            "name": ds.get("label", ""),
                            "type": "line",
                            "data": ds.get("data", []),
                            "smooth": True,
                        }
                        for ds in config.data.get("datasets", [])
                    ],
                }
            )

        elif config.chart_type == "bar":
            option.update(
                {
                    "xAxis": {
                        "type": "category",
                        "data": config.data.get("labels", []),
                    },
                    "yAxis": {"type": "value"},
                    "series": [
                        {
                            "name": ds.get("label", ""),
                            "type": "bar",
                            "data": ds.get("data", []),
                        }
                        for ds in config.data.get("datasets", [])
                    ],
                }
            )

        elif config.chart_type == "pie":
            option.update(
                {
                    "series": [
                        {
                            "type": "pie",
                            "data": [
                                {"name": name, "value": value}
                                for name, value in zip(
                                    config.data.get("labels", []),
                                    config.data.get("datasets", [{}])[0].get(
                                        "data", []
                                    ),
                                )
                            ],
                        }
                    ]
                }
            )

        elif config.chart_type == "doughnut":
            option.update(
                {
                    "series": [
                        {
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "data": [
                                {"name": name, "value": value}
                                for name, value in zip(
                                    config.data.get("labels", []),
                                    config.data.get("datasets", [{}])[0].get(
                                        "data", []
                                    ),
                                )
                            ],
                        }
                    ]
                }
            )

        return option


def create_chart_renderer(
    backend: str = "auto",
) -> ChartRenderer:
    """
    创建图表渲染器

    Args:
        backend: 渲染器类型
            - "auto": 自动选择（优先Plotly，否则Chart.js）
            - "chartjs": Chart.js渲染器
            - "plotly": Plotly渲染器
            - "echarts": ECharts渲染器

    Returns:
        图表渲染器实例
    """
    if backend == "auto":
        # 优先尝试Plotly
        try:
            import plotly  # noqa: F401 -- availability probe

            return PlotlyRenderer()
        except ImportError:
            return ChartJsRenderer()

    if backend == "chartjs":
        return ChartJsRenderer()

    if backend == "plotly":
        return PlotlyRenderer()

    if backend == "echarts":
        return EChartsRenderer()

    raise ValueError(f"Unknown renderer backend: {backend}")
