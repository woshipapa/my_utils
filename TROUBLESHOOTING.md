# Profiling 工具故障排查指南

本文档帮助你解决使用 my_utils profiling 工具时遇到的常见问题。

## 📋 目录

1. [安装问题](#安装问题)
2. [配置问题](#配置问题)
3. [数据收集问题](#数据收集问题)
4. [报告生成问题](#报告生成问题)
5. [性能问题](#性能问题)
6. [可视化问题](#可视化问题)

---

## 🔧 安装问题

### 问题: ImportError: No module named 'my_utils'

**症状**:
```python
ImportError: No module named 'my_utils'
```

**原因**: 未安装 my_utils 包

**解决方案**:

```bash
cd /path/to/my_utils
pip install -e .
```

**验证**:
```python
python -c "from my_utils.profiling import MetricsCollector; print('OK')"
```

---

### 问题: 缺少可选依赖

**症状**:
```python
ImportError: cannot import name 'nvtx' from 'my_utils'
```

**解决方案**:

```bash
# 安装所有依赖
pip install -e .[all]

# 或只安装需要的
pip install -e .[nvtx,profiling]
```

---

### 问题: Plotly 不可用

**症状**: 可视化报告回退到简单版本

**解决方案**:

```bash
pip install plotly
```

---

## ⚙️ 配置问题

### 问题: MetricsCollector 没有收集数据

**症状**:
```python
report = collector.analyze()
print(report.summary.get('total_events', 0))  # 输出 0
```

**诊断步骤**:

```python
# 1. 检查是否启用
print(f"Enabled: {collector.is_enabled()}")

# 2. 检查注册的 providers
print(f"Providers: {collector.list_providers()}")

# 3. 检查每个 provider
for provider_id in collector.list_providers():
    # 获取 provider 实例
    provider = collector._providers.get(provider_id)
    print(f"{provider_id}: enabled={provider.is_enabled()}")

# 4. 手动测试 provider
from my_utils.profiling import BaseMetricsProvider, MetricEvent
import time

class TestProvider(BaseMetricsProvider):
    provider_id = "test"

    def get_metrics(self):
        return [
            MetricEvent(
                timestamp=time.time(),
                name="test.metric",
                value=1.0,
                unit="",
                provider_id=self.provider_id,
                tags={},
            )
        ]

test_provider = TestProvider()
collector.register_provider(test_provider)
collector.collect()
print(f"Events: {len(collector.get_events())}")
```

**解决方案**:

确保 provider 正确初始化：

```python
# 错误示例
provider = MyTimerMetricsProvider()  # ❌ 缺少 timer 参数

# 正确示例
provider = MyTimerMetricsProvider(timer=timer)  # ✅
```

---

### 问题: CSV Provider 没有读取数据

**症状**:
```python
TableCsvMetricsProvider 返回空列表
```

**诊断步骤**:

```python
from pathlib import Path
import csv

csv_path = "your_file.csv"

# 1. 检查文件存在
print(f"Exists: {Path(csv_path).exists()}")

# 2. 检查格式
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    print(f"Columns: {reader.fieldnames}")
    print(f"Rows: {sum(1 for _ in reader)}")

# 3. 检查列名是否匹配
provider = TableCsvMetricsProvider(
    csv_path=csv_path,
    value_column="duration_ms",    # 确保这个列存在
    name_column="event_name",      # 确保这个列存在
    tag_columns=["step"],          # 确保这些列存在
)
```

**常见错误**:

| 错误 | 解决方案 |
|------|----------|
| `KeyError: 'duration_ms'` | 检查 CSV 列名，使用 `value_column="正确的列名"` |
| `空列表` | 检查 CSV 是否有数据行 |
| `编码错误` | 确保使用 UTF-8 编码 |

---

### 问题: Torch Profiler 没有输出

**症状**: TorchProfilerMetricsProvider 返回空列表

**解决方案**:

```python
# 确保 profiler 已运行
profiler = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
)

with profiler:
    # ... 训练代码 ...
    pass

# 在 profiler 退出后注册 provider
provider = TorchProfilerMetricsProvider(profiler)
collector.register_provider(provider)
```

---

## 📊 数据收集问题

### 问题: 收集到的数据不完整

**症状**:
```
总事件数: 50 (预期 100)
```

**原因**: 收集频率太高或 provider 产生数据慢

**解决方案**:

```python
# 方案 1: 增加 collect 间隔
for step in range(1000):
    # ... 训练代码 ...
    if step % 100 == 0:  # 每 100 步收集一次
        collector.collect(step=step)

# 方案 2: 同步收集
for step in range(100):
    # ... 训练代码 ...
    torch.cuda.synchronize()  # 确保 CUDA 完成
    collector.collect(step=step)
```

---

### 问题: MyTimer 数据不正确

**症状**:
```
CUDA 时间为 0 或 NA
```

**解决方案**:

```python
# 1. 确保启用了 CUDA
timer = MyTimer(use_cuda=True)

# 2. 确保正确同步
timer.start("operation")
# ... CUDA 操作 ...
timer.stop("operation")
timer.step()  # 重要！调用 step 同步

# 3. 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

### 问题: NVTX 标记在 nsys 中看不到

**症状**: 运行 nsys profile 但看不到自定义标记

**解决方案**:

```python
import os
os.environ["ENABLE_NVTX"] = "1"  # 设置环境变量

from my_utils import create_labeler

labeler = create_labeler(enabled=True)

with labeler.range("my_custom_range"):
    # ... 代码 ...

# 运行 nsys
# nsys profile --trace=cuda,nvtx python script.py
```

---

## 📄 报告生成问题

### 问题: HTML 报告显示不正常

**症状**: 图表不显示或样式错误

**诊断步骤**:

```python
# 1. 检查报告是否生成
from pathlib import Path
report_path = Path("report.html")
print(f"Exists: {report_path.exists()}")
print(f"Size: {report_path.stat().st_size} bytes")

# 2. 检查浏览器控制台
# 打开 report.html，按 F12 查看是否有 JS 错误

# 3. 检查 CDN 连接
# 确保能访问 https://cdn.jsdelivr.net
```

**解决方案**:

```python
# 使用本地 Chart.js (离线模式)
from my_utils.profiling.visualization import ChartJsRenderer

renderer = ChartJsRenderer()
# 下载 chart.js 并放在同目录
# 修改 HTML 使用本地版本
```

---

### 问题: 报告为空或缺少预期内容

**症状**:
```python
report.findings == []  # 空列表
```

**诊断步骤**:

```python
# 1. 检查原始数据
events = collector.get_events()
print(f"Total events: {len(events)}")
for event in events[:5]:
    print(f"  {event.name}: {event.value}")

# 2. 检查数据格式
# 瓶颈检测需要特定格式
latency_events = [e for e in events if e.name.startswith("latency.")]
print(f"Latency events: {len(latency_events)}")

# 3. 调整分析阈值
from my_utils.profiling import MetricsAnalyzer

analyzer = MetricsAnalyzer(
    bottleneck_share_threshold=0.05,  # 降低阈值
    cv_threshold=0.30,
)
collector = MetricsCollector(analyzer=analyzer)
```

---

### 问题: JSON 报告乱码

**症状**: JSON 文件中中文显示为 `\uXXXX`

**解决方案**: 这是正常的 JSON 编码，使用支持 UTF-8 的查看器：

```python
import json

with open("report.json", "r", encoding="utf-8") as f:
    report = json.load(f)  # 自动解码

print(report)
```

---

## ⚡ 性能问题

### 问题: Profiling 严重影响训练速度

**症状**:
- 训练速度变慢 50%+
- 内存占用显著增加

**诊断**:

```python
import time

start = time.time()
collector.collect(step=step)
overhead = time.time() - start

print(f"Collect overhead: {overhead*1000:.2f} ms")
```

**解决方案**:

1. **减少收集频率**
```python
if step % 1000 == 0:  # 每 1000 步收集一次
    collector.collect(step=step)
```

2. **条件启用**
```python
import os
if os.getenv("ENABLE_PROFILING") == "1":
    collector.collect(step=step)
```

3. **使用更少的 provider**
```python
# 只注册必要的 provider
collector.register_provider(MyTimerMetricsProvider(timer))
# 不注册其他
```

4. **异步写入**
```python
from my_utils.profiling import MetricsStore

store = MetricsStore(
    async_write=True,  # 异步写入
    buffer_size=10000,  # 缓冲更多
)
collector = MetricsCollector(store=store)
```

---

### 问题: 内存占用过高

**症状**:
- `metrics_events.jsonl` 文件很大
- 内存占用持续增长

**解决方案**:

1. **限制数据量**
```python
# 只收集关键 step
if step in [10, 50, 100, 500, 1000]:
    collector.collect(step=step)
```

2. **定期清理**
```python
from my_utils.profiling import MetricsStore

store = MetricsStore(
    max_events=100000,  # 最多保存 10 万个事件
    rotate=True,  # 超过限制时轮转
)
```

3. **使用压缩**
```python
store = MetricsStore(
    compress=True,  # 压缩存储
)
```

---

## 🎨 可视化问题

### 问题: 图表不显示

**症状**: HTML 报告中图表区域为空

**诊断**:

```python
# 检查数据格式
events = collector.get_events()
print(f"Events: {len(events)}")

# 检查是否有数值数据
for event in events[:5]:
    print(f"{event.name}: {event.value} ({type(event.value)})")
```

**解决方案**:

确保数据是数值类型：

```python
# 错误
MetricEvent(..., value="123", ...)  # 字符串

# 正确
MetricEvent(..., value=float("123"), ...)  # 浮点数
```

---

### 问题: 图表颜色难看

**症状**: 颜色搭配不协调

**解决方案**:

```python
from my_utils.profiling.visualization import ChartConfig

# 自定义颜色
config = ChartConfig(
    chart_type="line",
    data={
        "datasets": [{
            "label": "Loss",
            "data": [...],
            "borderColor": "#FF6B6B",  # 自定义颜色
            "backgroundColor": "rgba(255, 107, 107, 0.2)",
        }]
    }
)
```

---

### 问题: 报告在移动端显示异常

**症状**: 手机上查看报告时布局错乱

**解决方案**:

报告已经是响应式的，如果仍有问题，检查：

1. 浏览器版本（建议 Chrome 90+）
2. 横屏/竖屏切换
3. 使用桌面模式

---

## 🔍 调试技巧

### 启用调试日志

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# 现在所有操作都会输出详细日志
collector.collect(step=step)
```

### 检查数据流

```python
# 步骤 1: 检查 provider
provider_events = test_provider.get_metrics()
print(f"Provider events: {len(provider_events)}")

# 步骤 2: 检查 collect
count = collector.collect()
print(f"Collected: {count} events")

# 步骤 3: 检查存储
events = collector.get_events()
print(f"Stored: {len(events)} events")

# 步骤 4: 检查分析
report = collector.analyze(events)
print(f"Findings: {len(report.findings)}")
```

### 使用 Demo 验证环境

```bash
# 运行内置 demo 验证环境是否正常
python -m my_utils.profiling.examples.unified_metrics_demo --steps 10

# 检查输出
ls test_metrics_output/
cat test_metrics_output/report.md
```

---

## 🆘 仍然无法解决？

### 收集诊断信息

创建诊断脚本：

```python
# diagnose.py
import sys
import torch
from my_utils import get_profiling_templates_dir
from my_utils.profiling import VISUALIZATION_AVAILABLE

print("=== Environment Info ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

print("\n=== my_utils Info ===")
print(f"Templates: {get_profiling_templates_dir()}")
print(f"Visualization: {VISUALIZATION_AVAILABLE}")

print("\n=== Test MetricsCollector ===")
from my_utils.profiling import MetricsCollector, BaseMetricsProvider, MetricEvent
import time

class TestProvider(BaseMetricsProvider):
    provider_id = "test"

    def get_metrics(self):
        return [MetricEvent(
            timestamp=time.time(),
            name="test",
            value=1.0,
            unit="",
            provider_id=self.provider_id,
            tags={},
        )]

try:
    collector = MetricsCollector(output_dir="./test_diagnose")
    collector.register_provider(TestProvider())
    collector.start()
    collector.collect()
    collector.stop()

    events = collector.get_events()
    print(f"✓ MetricsCollector OK ({len(events)} events)")

    report = collector.analyze()
    print(f"✓ Analyzer OK ({len(report.findings)} findings)")

    path = collector.export_report(fmt="json")
    print(f"✓ Export OK ({path})")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
```

运行诊断：
```bash
python diagnose.py
```

### 获取帮助

1. 查看文档
   - [PROFILING_GUIDE.md](./PROFILING_GUIDE.md)
   - [QUICKSTART_PROFILING.md](./QUICKSTART_PROFILING.md)

2. 查看 Demo
   - [examples/unified_metrics_demo.py](./my_utils/profiling/examples/unified_metrics_demo.py)

3. 提交 Issue
   - 包含诊断脚本输出
   - 包含最小复现代码
   - 包含环境信息

---

## 📚 常用错误代码速查

| 错误代码 | 含义 | 解决方案 |
|---------|------|----------|
| `ModuleNotFoundError` | 未安装 | `pip install -e .` |
| `KeyError` | CSV 列名错误 | 检查列名配置 |
| `ValueError` | 数值转换失败 | 检查数据格式 |
| `EmptyReport` | 没有收集到数据 | 检查 provider 配置 |
| `ImportError: No module named 'yaml'` | 缺少 PyYAML | `pip install pyyaml` |
| `ImportError: No module named 'nvtx'` | 缺少 nvtx | `pip install nvtx` |

---

## 💡 最佳实践

1. **从小规模开始**
   - 先用 10 步测试
   - 验证数据正确
   - 再扩大规模

2. **增量式添加**
   - 先添加一个 provider
   - 验证工作后
   - 再添加其他

3. **定期验证**
   - 每次训练后查看报告
   - 验证数据完整性
   - 及时发现问题

4. **保存配置**
   - 使用 YAML 配置文件
   - 版本控制配置
   - 团队共享配置

---

**需要更多帮助？**
- 查看 [完整指南](./PROFILING_GUIDE.md)
- 运行 [demo](./my_utils/profiling/examples/unified_metrics_demo.py)
- 提交 [Issue](https://github.com/your-repo/issues)
