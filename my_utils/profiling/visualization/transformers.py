"""
数据转换器

将原始数据转换为图表友好的格式
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from collections import defaultdict


@dataclass
class MetricEvent:
    """统一的指标事件格式"""

    timestamp: float  # Unix时间戳(秒)
    name: str  # 指标名称
    value: float  # 指标值
    unit: str = ""  # 单位
    tags: dict[str, Any] = None  # 额外标签

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class DataTransformer:
    """
    数据转换器

    将MetricEvent列表转换为各种图表所需的数据格式
    """

    @staticmethod
    def to_time_series(
        events: list[MetricEvent],
        metric_name: str | None = None,
        group_by_step: bool = True,
    ) -> dict:
        """
        转换为时间序列数据

        Args:
            events: 指标事件列表
            metric_name: 指标名称过滤（None表示所有）
            group_by_step: 是否按step分组

        Returns:
            适合折线图的数据格式
        """
        # 过滤
        if metric_name:
            events = [e for e in events if e.name == metric_name]

        if not events:
            return {"labels": [], "datasets": []}

        # 按step或时间分组
        if group_by_step and "step" in events[0].tags:
            # 按step分组
            by_step: dict[int, list[float]] = defaultdict(list)
            for evt in events:
                step = int(evt.tags.get("step", 0))
                by_step[step].append(evt.value)

            # 对每个step求平均
            steps = sorted(by_step.keys())
            values = [np.mean(by_step[s]) for s in steps]

            return {
                "labels": steps,
                "datasets": [
                    {
                        "label": metric_name or "value",
                        "data": values,
                        "borderColor": "rgb(75, 192, 192)",
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "fill": True,
                    }
                ],
            }
        else:
            # 按时间排序
            events_sorted = sorted(events, key=lambda e: e.timestamp)
            timestamps = [e.timestamp for e in events_sorted]
            values = [e.value for e in events_sorted]

            return {
                "labels": timestamps,
                "datasets": [
                    {
                        "label": metric_name or "value",
                        "data": values,
                        "borderColor": "rgb(75, 192, 192)",
                        "backgroundColor": "rgba(75, 192, 192, 0.2)",
                        "fill": True,
                    }
                ],
            }

    @staticmethod
    def to_multiple_time_series(
        events: list[MetricEvent],
        metric_names: list[str],
    ) -> dict:
        """
        转换多条时间序列

        Args:
            events: 指标事件列表
            metric_names: 指标名称列表

        Returns:
            适合多条折线图的数据格式
        """
        colors = [
            ("rgb(75, 192, 192)", "rgba(75, 192, 192, 0.2)"),
            ("rgb(255, 99, 132)", "rgba(255, 99, 132, 0.2)"),
            ("rgb(54, 162, 235)", "rgba(54, 162, 235, 0.2)"),
            ("rgb(255, 206, 86)", "rgba(255, 206, 86, 0.2)"),
            ("rgb(153, 102, 255)", "rgba(153, 102, 255, 0.2)"),
        ]

        datasets = []
        all_steps = set()

        for idx, metric_name in enumerate(metric_names):
            metric_events = [e for e in events if e.name == metric_name]
            if not metric_events:
                continue

            # 按step分组
            by_step: dict[int, list[float]] = defaultdict(list)
            for evt in metric_events:
                step = int(evt.tags.get("step", 0))
                by_step[step].append(evt.value)
                all_steps.add(step)

            steps = sorted(by_step.keys())
            values = [np.mean(by_step[s]) if s in by_step else None for s in steps]

            color = colors[idx % len(colors)]
            datasets.append(
                {
                    "label": metric_name,
                    "data": values,
                    "borderColor": color[0],
                    "backgroundColor": color[1],
                    "fill": False,
                    "steplabels": steps,  # 保存每个值的对应step
                }
            )

        # 合并所有steps作为x轴
        sorted_steps = sorted(all_steps)

        return {
            "labels": sorted_steps,
            "datasets": datasets,
        }

    @staticmethod
    def to_comparison(
        events: list[MetricEvent],
        metric_name: str,
        group_by: str = "rank",
    ) -> dict:
        """
        转换为对比图表数据

        Args:
            events: 指标事件列表
            metric_name: 指标名称
            group_by: 分组依据（rank, name等）

        Returns:
            适合柱状图的数据格式
        """
        # 过滤
        filtered = [e for e in events if e.name == metric_name]
        if not filtered:
            return {"labels": [], "datasets": []}

        # 按group_by字段分组
        groups: dict[str, list[float]] = defaultdict(list)
        for evt in filtered:
            key = evt.tags.get(group_by, "unknown")
            groups[key].append(evt.value)

        # 计算统计
        labels = sorted(groups.keys())
        values = [np.mean(groups[k]) for k in labels]
        errors = [np.std(groups[k]) if len(groups[k]) > 1 else 0 for k in labels]

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": metric_name,
                    "data": values,
                    "backgroundColor": [
                        "rgba(255, 99, 132, 0.7)",
                        "rgba(54, 162, 235, 0.7)",
                        "rgba(255, 206, 86, 0.7)",
                        "rgba(75, 192, 192, 0.7)",
                    ][: len(labels)],
                }
            ],
            "errors": errors,  # 误差条
        }

    @staticmethod
    def to_distribution(
        events: list[MetricEvent],
        metric_name: str,
        bins: int = 20,
    ) -> dict:
        """
        转换为分布直方图数据

        Args:
            events: 指标事件列表
            metric_name: 指标名称
            bins: 直方图bin数量

        Returns:
            适合直方图的数据格式
        """
        filtered = [e.value for e in events if e.name == metric_name]
        if not filtered:
            return {"labels": [], "datasets": []}

        # 计算直方图
        hist, bin_edges = np.histogram(filtered, bins=bins)

        # bin中心作为标签
        labels = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(hist))]

        return {
            "labels": labels,
            "datasets": [
                {
                    "label": f"{metric_name} distribution",
                    "data": hist.tolist(),
                    "backgroundColor": "rgba(75, 192, 192, 0.7)",
                    "borderColor": "rgb(75, 192, 192)",
                }
            ],
        }

    @staticmethod
    def to_pie_chart(
        events: list[MetricEvent],
        metric_name: str,
    ) -> dict:
        """
        转换为饼图数据

        用于显示各部分占比（如各kernel的时间占比）

        Args:
            events: 指标事件列表
            metric_name: 指标名称（通常是基于此聚合）

        Returns:
            适合饼图的数据格式
        """
        # 按名称分组聚合
        groups: dict[str, float] = defaultdict(float)
        for evt in events:
            if metric_name in evt.name:
                # 提取具体操作名
                parts = evt.name.split(".")
                if len(parts) > 1:
                    op_name = parts[-1]
                else:
                    op_name = evt.name
                groups[op_name] += evt.value

        if not groups:
            return {"labels": [], "datasets": []}

        # 排序
        sorted_items = sorted(groups.items(), key=lambda x: x[1], reverse=True)

        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]

        # 颜色
        colors = [
            "rgba(255, 99, 132, 0.8)",
            "rgba(54, 162, 235, 0.8)",
            "rgba(255, 206, 86, 0.8)",
            "rgba(75, 192, 192, 0.8)",
            "rgba(153, 102, 255, 0.8)",
            "rgba(255, 159, 64, 0.8)",
        ]

        return {
            "labels": labels,
            "datasets": [
                {
                    "data": values,
                    "backgroundColor": colors[: len(labels)],
                }
            ],
        }

    @staticmethod
    def to_scatter_plot(
        events: list[MetricEvent],
        x_metric: str,
        y_metric: str,
    ) -> dict:
        """
        转换为散点图数据

        用于分析两个指标的关系

        Args:
            events: 指标事件列表
            x_metric: X轴指标名称
            y_metric: Y轴指标名称

        Returns:
            适合散点图的数据格式
        """
        # 按step配对
        x_by_step: dict[int, float] = {}
        y_by_step: dict[int, float] = {}

        for evt in events:
            step = int(evt.tags.get("step", 0))
            if evt.name == x_metric:
                x_by_step[step] = evt.value
            elif evt.name == y_metric:
                y_by_step[step] = evt.value

        # 找出共同step
        common_steps = set(x_by_step.keys()) & set(y_by_step.keys())

        if not common_steps:
            return {"datasets": []}

        x_values = [x_by_step[s] for s in sorted(common_steps)]
        y_values = [y_by_step[s] for s in sorted(common_steps)]

        return {
            "datasets": [
                {
                    "label": f"{y_metric} vs {x_metric}",
                    "data": [{"x": x, "y": y} for x, y in zip(x_values, y_values)],
                    "backgroundColor": "rgba(75, 192, 192, 0.7)",
                    "borderColor": "rgb(75, 192, 192)",
                }
            ]
        }

    @staticmethod
    def to_heatmap(
        events: list[MetricEvent],
        row_metric: str,
        col_metric: str,
        value_metric: str,
    ) -> dict:
        """
        转换为热力图数据

        Args:
            events: 指标事件列表
            row_metric: 行维度指标
            col_metric: 列维度指标
            value_metric: 值指标

        Returns:
            适合热力图的数据格式
        """
        # 三维聚合
        data: dict[tuple[str, str], list[float]] = defaultdict(list)

        for evt in events:
            row_val = evt.tags.get(row_metric, "")
            col_val = evt.tags.get(col_metric, "")

            if evt.name == value_metric:
                data[(row_val, col_val)].append(evt.value)

        # 转换为矩阵
        rows = sorted(set(k[0] for k in data.keys()))
        cols = sorted(set(k[1] for k in data.keys()))

        matrix = []
        for row in rows:
            row_data = []
            for col in cols:
                values = data.get((row, col), [])
                row_data.append(np.mean(values) if values else 0)
            matrix.append(row_data)

        return {
            "rows": rows,
            "cols": cols,
            "matrix": matrix,
        }

    @staticmethod
    def compute_statistics(
        events: list[MetricEvent],
        metric_name: str | None = None,
    ) -> list[dict]:
        """
        计算统计信息

        Args:
            events: 指标事件列表
            metric_name: 指标名称过滤

        Returns:
            统计信息列表
        """
        if metric_name:
            events = [e for e in events if e.name == metric_name]

        # 按名称分组
        by_name: dict[str, list[float]] = defaultdict(list)
        for evt in events:
            by_name[evt.name].append(evt.value)

        stats = []
        for name, values in by_name.items():
            if values:
                stats.append(
                    {
                        "metric": name,
                        "count": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "p25": float(np.percentile(values, 25)),
                        "p75": float(np.percentile(values, 75)),
                    }
                )

        # 按均值排序
        stats.sort(key=lambda x: x["mean"], reverse=True)
        return stats

    @staticmethod
    def filter_by_tags(
        events: list[MetricEvent],
        tags: dict[str, Any],
    ) -> list[MetricEvent]:
        """
        按标签过滤事件

        Args:
            events: 指标事件列表
            tags: 标签过滤条件

        Returns:
            过滤后的事件列表
        """
        filtered = events
        for key, value in tags.items():
            filtered = [e for e in filtered if e.tags.get(key) == value]
        return filtered

    @staticmethod
    def aggregate_by_time_window(
        events: list[MetricEvent],
        window_size: float = 1.0,  # 秒
        aggregation: str = "mean",
    ) -> list[MetricEvent]:
        """
        按时间窗口聚合

        Args:
            events: 指标事件列表
            window_size: 窗口大小（秒）
            aggregation: 聚合方式 (mean, sum, min, max, count)

        Returns:
            聚合后的事件列表
        """
        if not events:
            return []

        # 按名称分组
        by_name: dict[str, list[MetricEvent]] = defaultdict(list)
        for evt in events:
            by_name[evt.name].append(evt)

        aggregated = []

        for name, name_events in by_name.items():
            # 按时间排序
            name_events.sort(key=lambda e: e.timestamp)

            # 确定窗口边界
            start_time = name_events[0].timestamp
            end_time = name_events[-1].timestamp

            # 按窗口聚合
            window_start = start_time
            while window_start < end_time:
                window_end = window_start + window_size

                # 找出窗口内的事件
                window_events = [
                    e for e in name_events if window_start <= e.timestamp < window_end
                ]

                if window_events:
                    values = [e.value for e in window_events]

                    if aggregation == "mean":
                        agg_value = np.mean(values)
                    elif aggregation == "sum":
                        agg_value = np.sum(values)
                    elif aggregation == "min":
                        agg_value = np.min(values)
                    elif aggregation == "max":
                        agg_value = np.max(values)
                    elif aggregation == "count":
                        agg_value = len(values)
                    else:
                        agg_value = np.mean(values)

                    aggregated.append(
                        MetricEvent(
                            timestamp=window_start + window_size / 2,
                            name=name,
                            value=agg_value,
                            unit=window_events[0].unit,
                            tags={"window": f"{window_start:.1f}-{window_end:.1f}"},
                        )
                    )

                window_start = window_end

        return aggregated
