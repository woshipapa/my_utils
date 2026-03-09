# 统一 Profiling 快速开始指南

本指南帮助你在 5 分钟内上手 my_utils 的统一 profiling 功能。

## 🎯 选择你的场景

### 场景 1: 我正在训练，想实时监控性能

→ 使用 [MyTimer + MetricsCollector](#场景-1-实时监控训练性能)

### 场景 2: 我已经有训练日志 (CSV)

→ 使用 [CSV 分析](#场景-2-分析现有-csv-日志)

### 场景 3: 我想要详细的性能分析

→ 使用 [torch.profiler + MetricsCollector](#场景-3-详细性能分析)

### 场景 4: 我想分析 NCU/nsys 输出

→ 使用 [NCU/nsys 数据导入](#场景-4-分析-ncunsys-数据)

---

## 场景 1: 实时监控训练性能

### 步骤 1: 添加 Profiling (3 行代码)

```python
# 在你的训练脚本顶部添加
from my_utils.profiling import MetricsCollector, MyTimerMetricsProvider

# 创建 collector
profiler = MetricsCollector(output_dir="./profiling_results")
profiler.register_provider(MyTimerMetricsProvider(your_timer))
```

### 步骤 2: 在训练循环中收集数据

```python
# 在你的训练循环中
profiler.start()

for step in range(num_steps):
    # ... 你的训练代码 ...

    # 每 N 步收集一次（推荐 10-100）
    if step % 100 == 0:
        profiler.collect(step=step)

profiler.stop()
```

### 步骤 3: 生成报告

```python
# 训练结束后
report = profiler.analyze()
html_path = profiler.export_report(fmt="html", report=report)
print(f"报告已生成: {html_path}")
```

### 完整示例

```python
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer
from my_utils.profiling import MetricsCollector, MyTimerMetricsProvider

# 1. 设置 logger 和 timer
logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs", rank=0, world_size=1)
timer = MyTimer(use_cuda=True)
timer.set_logger(logger_mgr.get_logger())

# 2. 设置 profiler
profiler = MetricsCollector(output_dir="./profiling_output")
profiler.register_provider(MyTimerMetricsProvider(timer))

# 3. 训练
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())

profiler.start()

for step in range(100):
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

    # 每 20 步收集一次
    if step % 20 == 0:
        profiler.collect(step=step)

profiler.stop()

# 4. 生成报告
report = profiler.analyze()
print(f"\n=== 性能分析报告 ===")
print(f"总事件数: {report.summary.get('total_events', 0)}")
print(f"发现问题: {len(report.findings)}")

for finding in report.findings:
    print(f"\n[{finding.severity.upper()}] {finding.title}")

html_path = profiler.export_report(fmt="html", report=report)
print(f"\n详细报告: {html_path}")
```

**预期输出**:
```
=== 性能分析报告 ===
总事件数: 240
发现问题: 2

[HIGH] Latency Bottlenecks Detected
[INFO] Memory Profile

详细报告: ./profiling_output/report.html
```

---

## 场景 2: 分析现有 CSV 日志

### 步骤 1: 准备 CSV 文件

确保你的 CSV 包含以下列：
- 时间相关列 (如 `duration_ms`, `latency`)
- 名称列 (如 `event_name`, `op_name`)
- 可选: 标签列 (如 `step`, `rank`)

### 步骤 2: 创建分析脚本

```python
from my_utils.profiling import MetricsCollector, TableCsvMetricsProvider

# 创建 collector
collector = MetricsCollector(output_dir="./csv_analysis")

# 注册 CSV provider
collector.register_provider(
    TableCsvMetricsProvider(
        csv_path="logs/profile_rank_0.csv",  # 你的 CSV 文件
        value_column="duration_ms",            # 数值列
        name_column="event_name",               # 名称列
        tag_columns=["step"],                   # 标签列
        unit="ms",                              # 单位
    )
)

# 收集数据
collector.start()
collector.collect()
collector.stop()

# 生成报告
report = collector.analyze()
html_path = collector.export_report(fmt="html", report=report)
print(f"报告已生成: {html_path}")
```

### 步骤 3: 在浏览器中查看报告

打开生成的 HTML 文件查看：
- 性能摘要
- 瓶颈分析
- 内存使用情况
- 优化建议

---

## 场景 3: 详细性能分析

### 步骤 1: 启用 torch.profiler

```python
from my_utils.profiling import MetricsCollector, TorchProfilerMetricsProvider

# 创建 profiler
profiler = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
)

# 创建 collector
collector = MetricsCollector(output_dir="./detailed_profiling")
collector.register_provider(TorchProfilerMetricsProvider(profiler))

# 使用 profiler
collector.start()

with profiler:
    for step in range(10):
        # 训练代码
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 每步收集
        collector.collect(step=step)

collector.stop()
```

### 步骤 2: 分析报告

```python
# 生成报告
report = collector.analyze()

# 查看 kernel 级别的统计
from my_utils.profiling.metrics_taxonomy import normalize_external_metric

for event in collector.get_events():
    if event.name.startswith("latency.kernel"):
        print(f"{event.tags.get('kernel', 'unknown')}: {event.value} ms")
```

---

## 场景 4: 分析 NCU/nsys 数据

### 从 NCU CSV 导入

```python
from my_utils.profiling import MetricsCollector, NcuCsvMetricsProvider

collector = MetricsCollector(output_dir="./ncu_analysis")

collector.register_provider(
    NcuCsvMetricsProvider(
        csv_path="ncu_report.csv",  # NCU 导出的 CSV
    )
)

collector.start()
collector.collect()
collector.stop()

report = collector.analyze()
collector.export_report(fmt="html", report=report)
```

### 从 nsys SQLite 数据库导入

```python
from my_utils.profiling import MetricsCollector, NsysSqliteMetricsProvider

collector = MetricsCollector(output_dir="./nsys_analysis")

collector.register_provider(
    NsysSqliteMetricsProvider(
        sqlite_path="train_rank0.sqlite",  # nsys export sqlite output
        include_gpu_metrics=True,
        include_network_metrics=True,
        parse_step_from_nvtx=True,
    )
)

collector.start()
collector.collect()
collector.stop()

report = collector.analyze()
collector.export_report(fmt="html", report=report)
```

---

## 🔧 进阶技巧

### 技巧 1: 条件性 Profiling

只对特定 step profile：

```python
for step in range(1000):
    # 训练代码...

    # 只在第 100-110 步收集
    if 100 <= step <= 110:
        profiler.collect(step=step)
```

### 技巧 2: 多数据源联合分析

```python
from my_utils.profiling import (
    MetricsCollector,
    MyTimerMetricsProvider,
    TableCsvMetricsProvider,
    NcuCsvMetricsProvider,
)

collector = MetricsCollector(output_dir="./multi_source")

# 注册多个数据源
collector.register_provider(MyTimerMetricsProvider(timer))
collector.register_provider(TableCsvMetricsProvider("custom_metrics.csv"))
collector.register_provider(NcuCsvMetricsProvider("ncu.csv"))

# 统一收集和分析
collector.start()
collector.collect()
collector.stop()

report = collector.analyze()
collector.export_report(fmt="html", report=report)
```

### 技巧 3: 自定义分析阈值

```python
from my_utils.profiling import MetricsCollector, MetricsAnalyzer

# 创建自定义分析器
analyzer = MetricsAnalyzer(
    bottleneck_share_threshold=0.15,  # 15%
    cv_threshold=0.30,                # 30%
    memory_growth_bytes_per_step=5*1024*1024,  # 5MB
)

# 使用自定义分析器
collector = MetricsCollector(
    output_dir="./custom_analysis",
    analyzer=analyzer,
)
```

### 技巧 4: 导出多种格式

```python
# 同时生成多种格式
json_path = collector.export_report(fmt="json", output_path="report.json")
md_path = collector.export_report(fmt="markdown", output_path="report.md")
html_path = collector.export_report(fmt="html", output_path="report.html")

print(f"JSON: {json_path}")
print(f"Markdown: {md_path}")
print(f"HTML: {html_path}")
```

### 技巧 5: 使用配置文件

创建 `profiling_config.yaml`:

```yaml
collector:
  output_dir: "./profiling_output"
  enabled: true

analysis:
  bottleneck_threshold: 0.15
  cv_threshold: 0.40
  memory_growth_bytes_per_step: 10485760
```

使用配置：

```python
collector = MetricsCollector.from_config("profiling_config.yaml")
```

---

## 📊 报告解读

### HTML 报告包含

1. **摘要部分**
   - 总事件数
   - 数据源列表
   - 关键统计

2. **Findings 部分**
   - 瓶颈分析 (红色 = 高优先级)
   - 内存分析 (黄色 = 警告)
   - 异常检测

3. **Recommendations 部分**
   - 具体的优化建议
   - 按优先级排序

### 如何阅读报告

#### 查看瓶颈

```
[HIGH] Latency Bottlenecks Detected
  - backward: 56.9% (31.4ms)
  - forward: 35.7% (19.7ms)
```

**含义**:
- backward 占总时间的 56.9%
- 平均延迟 31.4ms

**建议**: 优化 backward pass

#### 查看内存

```
[INFO] Memory Profile
  peak_bytes: 8.66 GB
  growth_warnings: []
```

**含义**:
- 峰值内存 8.66 GB
- 无明显内存泄漏

---

## 🐛 故障排查

### 问题 1: "No metrics collected"

**原因**: Provider 没有产生数据

**解决**:
```python
# 检查 provider 是否启用
print(profiler.list_providers())  # 应该显示你的 provider

# 检查是否有数据
events = profiler.get_events()
print(f"Collected {len(events)} events")
```

### 问题 2: "Report is empty"

**原因**: 数据格式不匹配

**解决**:
```python
# 检查数据格式
for event in profiler.get_events()[:5]:
    print(f"{event.name}: {event.value} {event.unit}")
```

### 问题 3: "File not found"

**原因**: CSV 路径错误

**解决**:
```python
from pathlib import Path
csv_path = Path("logs/profile.csv")
print(f"File exists: {csv_path.exists()}")
print(f"Absolute path: {csv_path.absolute()}")
```

---

## 📚 更多资源

- [完整指南](./PROFILING_GUIDE.md)
- [可视化增强](./my_utils/profiling/visualization/README.md)
- [设计文档](./my_utils/profiling/docs/ROADMAP.md)

---

## 💡 最佳实践

1. **定期收集** - 每 100-1000 步收集一次
2. **多数据源** - 结合 MyTimer 和外部工具
3. **保存报告** - 定期导出 HTML 报告用于对比
4. **关注趋势** - 比较不同训练的性能变化
5. **及时分析** - 发现问题立即深入分析

---

开始使用吧！如果遇到问题，查看 [完整指南](./PROFILING_GUIDE.md) 或运行 demo：
```bash
python -m my_utils.profiling.examples.unified_metrics_demo --steps 30
```
