# SPDX-License-Identifier: Apache-2.0
"""
HTML报告生成器

整合所有组件，生成完整的性能分析HTML报告
"""

from typing import Optional
import time
from pathlib import Path

from .charts import ChartConfig, ChartRenderer, create_chart_renderer
from .transformers import DataTransformer, MetricEvent
from .layouts import (
    LayoutBuilder,
    AnalysisReport,
)


class HTMLReportGenerator:
    """
    HTML报告生成器

    整合图表渲染、数据转换和布局构建，生成完整的HTML报告
    """

    def __init__(
        self,
        renderer: Optional[ChartRenderer] = None,
        transformer: Optional[DataTransformer] = None,
    ):
        """
        初始化报告生成器

        Args:
            renderer: 图表渲染器（None则自动选择）
            transformer: 数据转换器（None则使用默认）
        """
        self.renderer = renderer or create_chart_renderer()
        self.transformer = transformer or DataTransformer()

    def generate(
        self,
        report: AnalysisReport,
        events: list[MetricEvent],
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成完整HTML报告

        Args:
            report: 分析报告
            events: 指标事件列表
            output_path: 输出文件路径（None则不保存）

        Returns:
            HTML字符串
        """
        builder = LayoutBuilder()

        # 1. 标题
        builder.add_header(
            title="Performance Analysis Report",
            subtitle=f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )

        # 2. 摘要
        self._add_summary_section(builder, report)

        # 3. 关键指标
        self._add_key_metrics(builder, events)

        # 4. 性能趋势图表
        self._add_trend_charts(builder, events)

        # 5. 瓶颈分析图表
        self._add_bottleneck_charts(builder, report)

        # 6. 内存分析图表
        self._add_memory_charts(builder, events)

        # 7. 发现列表
        if report.findings:
            builder.add_findings(report.findings, title="Key Findings")

        # 8. 优化建议
        if report.recommendations:
            builder.add_recommendations(report.recommendations, title="Recommendations")

        # 9. 详细数据表
        self._add_detail_tables(builder, events)

        # 10. 生成HTML
        html = builder.build()

        # 11. 保存（如果需要）
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    def _add_summary_section(
        self, builder: LayoutBuilder, report: AnalysisReport
    ) -> None:
        """添加摘要部分"""
        details = {
            "Total Events": report.metadata.get("event_count", 0),
            "Findings": report.metadata.get("finding_count", 0),
            "Recommendations": report.metadata.get("recommendation_count", 0),
        }

        builder.add_summary(
            summary=report.summary,
            score=report.overall_score,
            details=details,
        )

    def _add_key_metrics(
        self, builder: LayoutBuilder, events: list[MetricEvent]
    ) -> None:
        """添加关键指标卡片"""
        # 提取关键指标
        metrics = {}

        # 最新step
        steps = [int(e.tags.get("step", 0)) for e in events if "step" in e.tags]
        if steps:
            metrics["Current Step"] = max(steps)

        # 计算统计数据
        stats = self.transformer.compute_statistics(events)
        for stat in stats[:5]:  # 只显示前5个
            key = stat["metric"].split(".")[-1]  # 取最后一部分作为名称
            metrics[f"{key} (avg)"] = f"{stat['mean']:.2f} {stat.get('unit', '')}"

        if metrics:
            builder.add_metrics_grid(metrics)

    def _add_trend_charts(
        self, builder: LayoutBuilder, events: list[MetricEvent]
    ) -> None:
        """添加趋势图表"""
        # 获取所有唯一的指标名称
        metric_names = list(set(e.name for e in events))

        # 优先显示的指标
        priority_metrics = [
            "timer.iter",
            "timer.forward",
            "timer.backward",
            "memory.allocated",
            "memory.reserved",
            "loss",
        ]

        # 选择要显示的指标
        selected_metrics = [m for m in priority_metrics if m in metric_names]
        if not selected_metrics and metric_names:
            selected_metrics = metric_names[:3]  # 如果没有优先指标，显示前3个

        if selected_metrics:
            # 多条时间序列
            chart_data = self.transformer.to_multiple_time_series(
                events, selected_metrics
            )

            config = ChartConfig(
                chart_type="line",
                title="Performance Trends",
                data=chart_data,
                options={
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "interaction": {"mode": "index", "intersect": False},
                    "plugins": {
                        "title": {"display": True, "text": "Metrics Over Time"},
                        "tooltip": {"mode": "index", "intersect": False},
                    },
                },
            )

            chart_html = self.renderer.render(config)
            builder.add_chart(chart_html, title="Performance Trends Over Steps")

    def _add_bottleneck_charts(
        self, builder: LayoutBuilder, report: AnalysisReport
    ) -> None:
        """添加瓶颈分析图表"""
        # 从findings中提取瓶颈信息
        bottlenecks = [
            f
            for f in report.findings
            if "瓶颈" in f.title or "bottleneck" in f.title.lower()
        ]

        if bottlenecks:
            # 饼图显示瓶颈分布
            labels = []
            values = []

            for finding in bottlenecks:
                component = finding.evidence.get("component", finding.title)
                ratio = finding.evidence.get(
                    "ratio", finding.evidence.get("percentage", 0)
                )

                if isinstance(ratio, str):
                    ratio = float(ratio.rstrip("%")) / 100
                else:
                    ratio = float(ratio)

                labels.append(component)
                values.append(ratio * 100)

            if labels and values:
                chart_data = {
                    "labels": labels,
                    "datasets": [
                        {
                            "data": values,
                            "backgroundColor": [
                                "rgba(239, 68, 68, 0.8)",
                                "rgba(249, 115, 22, 0.8)",
                                "rgba(245, 158, 11, 0.8)",
                                "rgba(34, 197, 94, 0.8)",
                                "rgba(59, 130, 246, 0.8)",
                            ][: len(labels)],
                        }
                    ],
                }

                config = ChartConfig(
                    chart_type="pie",
                    title="Performance Bottlenecks Distribution",
                    data=chart_data,
                )

                chart_html = self.renderer.render(config)

                # 准备统计表格
                table_data = [
                    {
                        "Component": label,
                        "Time (%)": f"{value:.1f}%",
                    }
                    for label, value in zip(labels, values)
                ]

                # 使用两列布局
                table_html = self._create_table_html(table_data)

                from .layouts import LayoutBuilder as LB

                temp_builder = LB()
                temp_builder.add_table(table_data, title="")

                builder.add_two_column_charts(
                    left_chart_html=chart_html,
                    right_chart_html=temp_builder.sections[0]["content"],
                    left_title="Bottlenecks Distribution",
                    right_title="Detailed Breakdown",
                )

    def _add_memory_charts(
        self, builder: LayoutBuilder, events: list[MetricEvent]
    ) -> None:
        """添加内存分析图表"""
        memory_events = [e for e in events if "memory" in e.name.lower()]

        if not memory_events:
            return

        # 按内存类型分组
        by_type: dict[str, list[MetricEvent]] = {}
        for evt in memory_events:
            mem_type = evt.name.split(".")[-1] if "." in evt.name else evt.name
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(evt)

        # 创建折线图
        datasets = []
        all_labels = set()

        colors = {
            "allocated": ("rgb(75, 192, 192)", "rgba(75, 192, 192, 0.2)"),
            "reserved": ("rgb(255, 99, 132)", "rgba(255, 99, 132, 0.2)"),
        }

        for mem_type, type_events in by_type.items():
            # 按step或时间聚合
            by_step: dict[int, list[float]] = {}
            for evt in type_events:
                step = int(evt.tags.get("step", evt.timestamp))
                if step not in by_step:
                    by_step[step] = []
                by_step[step].append(evt.value)
                all_labels.add(step)

            steps = sorted(by_step.keys())
            values = [by_step.get(s, [0])[0] for s in sorted(all_labels)]

            color = colors.get(
                mem_type, ("rgb(153, 102, 255)", "rgba(153, 102, 255, 0.2)")
            )

            datasets.append(
                {
                    "label": f"Memory {mem_type}",
                    "data": values,
                    "borderColor": color[0],
                    "backgroundColor": color[1],
                    "fill": False,
                }
            )

        chart_data = {
            "labels": sorted(all_labels),
            "datasets": datasets,
        }

        config = ChartConfig(
            chart_type="line",
            title="Memory Usage Over Time",
            data=chart_data,
            options={
                "responsive": True,
                "maintainAspectRatio": False,
                "scales": {
                    "y": {
                        "beginAtZero": True,
                        "title": {"display": True, "text": "Memory (GB)"},
                    },
                    "x": {"title": {"display": True, "text": "Step"}},
                },
            },
        )

        chart_html = self.renderer.render(config)
        builder.add_chart(chart_html, title="Memory Usage")

    def _add_detail_tables(
        self, builder: LayoutBuilder, events: list[MetricEvent]
    ) -> None:
        """添加详细数据表格"""
        # 计算统计信息
        stats = self.transformer.compute_statistics(events)

        if stats:
            # 格式化表格数据
            table_data = []
            for stat in stats[:20]:  # 只显示前20个
                table_data.append(
                    {
                        "Metric": stat["metric"],
                        "Count": stat["count"],
                        "Mean": f"{stat['mean']:.4f}",
                        "Std": f"{stat['std']:.4f}",
                        "Min": f"{stat['min']:.4f}",
                        "Max": f"{stat['max']:.4f}",
                        "Median": f"{stat['median']:.4f}",
                    }
                )

            builder.add_table(table_data, title="Detailed Statistics")

    def _create_table_html(self, data: list[dict]) -> str:
        """创建表格HTML"""
        if not data:
            return ""

        headers = list(data[0].keys())
        rows_html = ""

        for row in data:
            cells = "".join(f"<td>{str(row[h])}</td>" for h in headers)
            rows_html += f"<tr>{cells}</tr>"

        headers_html = "".join(f"<th>{h}</th>" for h in headers)

        return f"""
        <table class="data-table">
            <thead>
                <tr>{headers_html}</tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """

    def generate_from_events(
        self,
        events: list[MetricEvent],
        output_path: Optional[str] = None,
    ) -> str:
        """
        仅从事件生成简单报告（不需要AnalysisReport）

        Args:
            events: 指标事件列表
            output_path: 输出文件路径

        Returns:
            HTML字符串
        """
        # 创建简单的报告
        stats = self.transformer.compute_statistics(events)

        # 计算得分（简化版）
        score = 75.0  # 默认分数

        # 创建一个空的报告
        report = AnalysisReport(
            metadata={
                "event_count": len(events),
                "finding_count": 0,
                "recommendation_count": 0,
            },
            findings=[],
            recommendations=[],
            summary=f"Collected {len(events)} events across {len(set(e.name for e in events))} metrics",
            overall_score=score,
        )

        return self.generate(report, events, output_path)


class QuickReportGenerator:
    """
    快速报告生成器

    用于从现有工具（如MyTimer）快速生成报告
    """

    def __init__(self):
        self.generator = HTMLReportGenerator()

    def generate_from_timer(
        self,
        timer,
        output_path: str = "timer_report.html",
    ) -> str:
        """
        从MyTimer生成报告

        Args:
            timer: MyTimer实例
            output_path: 输出路径

        Returns:
            HTML字符串
        """
        # 从MyTimer提取事件
        events = self._extract_events_from_timer(timer)

        return self.generator.generate_from_events(events, output_path)

    def generate_from_csv(
        self,
        csv_path: str,
        output_path: str = "csv_report.html",
    ) -> str:
        """
        从CSV日志文件生成报告

        Args:
            csv_path: CSV文件路径
            output_path: 输出路径

        Returns:
            HTML字符串
        """
        import csv

        events = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    events.append(
                        MetricEvent(
                            timestamp=float(row.get("timestamp_unix", 0)),
                            name=row.get("event_name", ""),
                            value=float(row.get("duration_ms", 0))
                            if row.get("duration_ms")
                            else 0,
                            unit="ms",
                            tags={
                                "step": row.get("step", ""),
                                "type": row.get("event_type", ""),
                            },
                        )
                    )
                except (ValueError, KeyError):
                    continue

        return self.generator.generate_from_events(events, output_path)

    def _extract_events_from_timer(self, timer) -> list[MetricEvent]:
        """从MyTimer提取事件"""
        events = []

        # 访问timer的事件列表
        if hasattr(timer, "_events"):
            for event in timer._events:
                events.append(
                    MetricEvent(
                        timestamp=event.timestamp
                        if hasattr(event, "timestamp")
                        else time.time(),
                        name=event.name if hasattr(event, "name") else "timer",
                        value=event.duration_ms if hasattr(event, "duration_ms") else 0,
                        unit="ms",
                        tags={
                            "step": str(event.step) if hasattr(event, "step") else "0",
                            "type": event.type if hasattr(event, "type") else "unknown",
                        },
                    )
                )

        return events
