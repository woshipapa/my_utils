"""
可视化增强 - 使用示例

展示如何使用新的可视化组件生成性能分析报告
"""

import sys
import os

# 添加父目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
profiling_dir = os.path.dirname(current_dir)
my_utils_dir = os.path.dirname(profiling_dir)
if my_utils_dir not in sys.path:
    sys.path.insert(0, my_utils_dir)

from my_utils.profiling.visualization import (
    ChartConfig,
    ChartJsRenderer,
    PlotlyRenderer,
    create_chart_renderer,
    DataTransformer,
    LayoutBuilder,
    HTMLReportGenerator,
    QuickReportGenerator,
    MetricEvent,
)


def example_1_basic_chart():
    """示例1: 创建基本图表"""
    print("=" * 60)
    print("示例1: 创建基本图表")
    print("=" * 60)

    # 创建渲染器
    renderer = create_chart_renderer(backend="chartjs")

    # 创建折线图配置
    config = ChartConfig(
        chart_type="line",
        title="Performance Over Time",
        data={
            "labels": [0, 1, 2, 3, 4, 5],
            "datasets": [{
                "label": "Loss",
                "data": [2.5, 2.1, 1.8, 1.5, 1.3, 1.1],
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "fill": True,
            }]
        },
        options={
            "responsive": True,
            "maintainAspectRatio": False,
        }
    )

    # 渲染为HTML
    html = renderer.render(config)
    print("图表HTML已生成")
    print(f"HTML长度: {len(html)} 字符")

    # 保存到文件
    output_path = "example_chart.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chart Example</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        </head>
        <body>
            <h1>Performance Chart</h1>
            {html}
        </body>
        </html>
        """)
    print(f"已保存到: {output_path}")


def example_2_data_transformer():
    """示例2: 使用数据转换器"""
    print("\n" + "=" * 60)
    print("示例2: 数据转换器")
    print("=" * 60)

    # 创建模拟数据
    import time
    events = []

    for step in range(10):
        # 模拟训练损失
        events.append(MetricEvent(
            timestamp=time.time() + step,
            name="loss",
            value=2.5 - step * 0.15,
            unit="",
            tags={"step": str(step)}
        ))

        # 模拟forward时间
        events.append(MetricEvent(
            timestamp=time.time() + step,
            name="timer.forward",
            value=100 + step * 2,
            unit="ms",
            tags={"step": str(step)}
        ))

        # 模拟backward时间
        events.append(MetricEvent(
            timestamp=time.time() + step,
            name="timer.backward",
            value=150 + step * 3,
            unit="ms",
            tags={"step": str(step)}
        ))

    print(f"创建了 {len(events)} 个模拟事件")

    # 使用DataTransformer转换
    transformer = DataTransformer()

    # 转换为时间序列
    time_series = transformer.to_time_series(events, metric_name="loss")
    print(f"\n时间序列数据:")
    print(f"  Labels: {time_series['labels']}")
    print(f"  Data points: {len(time_series['datasets'][0]['data'])}")

    # 多条时间序列
    multi_series = transformer.to_multiple_time_series(
        events,
        metric_names=["timer.forward", "timer.backward"]
    )
    print(f"\n多条时间序列:")
    print(f"  Datasets: {len(multi_series['datasets'])}")
    for ds in multi_series['datasets']:
        print(f"    - {ds['label']}")

    # 计算统计
    stats = transformer.compute_statistics(events)
    print(f"\n统计信息:")
    for stat in stats[:3]:
        print(f"  {stat['metric']}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")


def example_3_layout_builder():
    """示例3: 使用布局构建器"""
    print("\n" + "=" * 60)
    print("示例3: 布局构建器")
    print("=" * 60)

    builder = LayoutBuilder()

    # 添加标题
    builder.add_header(
        title="My Training Report",
        subtitle="Generated on 2024-01-15"
    )

    # 添加摘要
    builder.add_summary(
        summary="Training completed successfully with 75% performance score",
        score=75,
        details={
            "Total Steps": "1000",
            "Final Loss": "0.25",
            "Training Time": "2.5 hours",
        }
    )

    # 添加指标网格
    builder.add_metrics_grid({
        "Throughput": "128 samples/s",
        "GPU Memory": "8.5 GB",
        "Utilization": "92%",
    })

    # 添加表格
    table_data = [
        {"Metric": "Loss", "Value": "0.25", "Change": "-15%"},
        {"Metric": "Accuracy", "Value": "0.92", "Change": "+8%"},
        {"Metric": "Latency", "Value": "45ms", "Change": "-12%"},
    ]
    builder.add_table(table_data, title="Performance Metrics")

    # 生成HTML
    html = builder.build()
    print(f"HTML已生成，长度: {len(html)} 字符")

    # 保存
    output_path = "example_layout.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"已保存到: {output_path}")


def example_4_full_report():
    """示例4: 生成完整报告"""
    print("\n" + "=" * 60)
    print("示例4: 完整性能报告")
    print("=" * 60)

    # 创建模拟数据
    import time
    from my_utils.profiling.visualization.layouts import (
        Finding, Recommendation, AnalysisReport, Severity
    )

    events = []

    for step in range(20):
        # 损失
        events.append(MetricEvent(
            timestamp=time.time() + step,
            name="loss",
            value=2.0 * (0.95 ** step),
            unit="",
            tags={"step": str(step)}
        ))

        # Forward时间
        events.append(MetricEvent(
            timestamp=time.time() + step,
            name="timer.forward",
            value=100 + step * 0.5,
            unit="ms",
            tags={"step": str(step)}
        ))

        # Backward时间
        events.append(MetricEvent(
            timestamp=time.time() + step,
            name="timer.backward",
            value=150 + step * 0.8,
            unit="ms",
            tags={"step": str(step)}
        ))

        # 内存
        events.append(MetricEvent(
            timestamp=time.time() + step,
            name="memory.allocated",
            value=8.0 + step * 0.01,
            unit="GB",
            tags={"step": str(step)}
        ))

    # 创建模拟分析报告
    report = AnalysisReport(
        metadata={
            "event_count": len(events),
            "finding_count": 2,
            "recommendation_count": 3,
        },
        findings=[
            Finding(
                id="f1",
                title="Backward pass is a bottleneck",
                description="The backward pass takes 50% more time than forward pass",
                severity=Severity.HIGH,
                category="compute",
                evidence={
                    "forward_time_ms": 105,
                    "backward_time_ms": 158,
                    "ratio": 1.5,
                },
                affected_components=["model.backward"],
                metrics={"bottleneck_ratio": 0.6}
            ),
            Finding(
                id="f2",
                title="Memory usage is stable",
                description="Memory usage remains consistent across training steps",
                severity=Severity.INFO,
                category="memory",
                evidence={
                    "trend": "stable",
                    "growth_rate_mb_per_step": 0.01,
                },
                affected_components=["memory_allocator"],
                metrics={"memory_stability": 0.95}
            ),
        ],
        recommendations=[
            Recommendation(
                id="r1",
                title="Optimize backward pass",
                description="Consider using gradient checkpointing to reduce memory usage",
                priority=8,
                estimated_impact="20-30% faster",
                effort="medium",
                actions=[
                    "Enable gradient checkpointing for large layers",
                    "Consider using mixed precision training",
                    "Profile individual layer backward times",
                ],
                references=[]
            ),
            Recommendation(
                id="r2",
                title="Increase batch size",
                description="GPU memory has headroom for larger batch size",
                priority=6,
                estimated_impact="10-15% faster",
                effort="low",
                actions=[
                    "Try increasing batch size by 2x",
                    "Monitor memory usage",
                    "Adjust learning rate accordingly",
                ],
                references=[]
            ),
        ],
        summary="Training shows good convergence with minor optimization opportunities",
        overall_score=78.0,
    )

    # 生成报告
    generator = HTMLReportGenerator()
    html = generator.generate(
        report=report,
        events=events,
        output_path="example_full_report.html"
    )

    print(f"完整报告已生成")
    print(f"  总事件数: {len(events)}")
    print(f"  发现数: {len(report.findings)}")
    print(f"  建议数: {len(report.recommendations)}")
    print(f"  得分: {report.overall_score:.0f}/100")
    print(f"\n已保存到: example_full_report.html")


def example_5_quick_report_from_timer():
    """示例5: 从MyTimer快速生成报告"""
    print("\n" + "=" * 60)
    print("示例5: 快速报告生成")
    print("=" * 60)

    # 如果有MyTimer，展示如何使用
    try:
        from my_utils.utils import MyTimer
        from my_utils.logger import GlobalLogger

        # 初始化
        logger_mgr = GlobalLogger()
        logger_mgr.setup(log_dir="logs", rank=0, world_size=1)
        logger = logger_mgr.get_logger()

        timer = MyTimer(use_cuda=False, tag="demo")
        timer.set_logger(logger)

        # 模拟训练
        import time as time_module
        for step in range(5):
            timer.set_step(step)
            timer.start("forward")
            time_module.sleep(0.01)
            timer.stop("forward")
            timer.start("backward")
            time_module.sleep(0.02)
            timer.stop("backward")
            timer.step()

        # 生成报告
        quick_gen = QuickReportGenerator()
        html = quick_gen.generate_from_timer(
            timer=timer,
            output_path="example_timer_report.html"
        )

        print("已从MyTimer生成报告")
        print("已保存到: example_timer_report.html")

    except ImportError:
        print("MyTimer不可用，跳过此示例")


def example_6_from_csv():
    """示例6: 从CSV文件生成报告"""
    print("\n" + "=" * 60)
    print("示例6: 从CSV生成报告")
    print("=" * 60)

    # 创建示例CSV
    csv_path = "example_profile.csv"
    with open(csv_path, "w") as f:
        f.write("timestamp_unix,readable_time,step,event_name,event_type,duration_ms\n")
        import time as time_module
        for step in range(5):
            ts = time_module.time()
            f.write(f"{ts},2024-01-15 12:00:0{step},{step},forward,START,0\n")
            f.write(f"{ts+0.1},2024-01-15 12:00:0{step},{step},forward,END,100\n")
            f.write(f"{ts+0.1},2024-01-15 12:00:0{step},{step},backward,START,0\n")
            f.write(f"{ts+0.3},2024-01-15 12:00:0{step},{step},backward,END,200\n")

    print(f"已创建示例CSV: {csv_path}")

    # 从CSV生成报告
    try:
        quick_gen = QuickReportGenerator()
        html = quick_gen.generate_from_csv(
            csv_path=csv_path,
            output_path="example_csv_report.html"
        )
        print("已从CSV生成报告")
        print("已保存到: example_csv_report.html")
    except Exception as e:
        print(f"生成报告失败: {e}")


def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "可视化增强使用示例" + " " * 30 + "║")
    print("╚" + "═" * 58 + "╝")

    # 运行各个示例
    example_1_basic_chart()
    example_2_data_transformer()
    example_3_layout_builder()
    example_4_full_report()
    example_5_quick_report_from_timer()
    example_6_from_csv()

    print("\n" + "=" * 60)
    print("所有示例已完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - example_chart.html        # 基本图表")
    print("  - example_layout.html       # 布局示例")
    print("  - example_full_report.html  # 完整报告")
    print("  - example_timer_report.html # Timer报告")
    print("  - example_csv_report.html   # CSV报告")
    print("\n在浏览器中打开这些HTML文件查看结果！")


if __name__ == "__main__":
    main()
