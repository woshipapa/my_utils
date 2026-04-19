# visualization（报告可视化）

这个目录负责把 profiling 数据渲染成更直观的图表和 HTML 报告。

## 30秒定位

1. 我已经有 report/events，想快速出 HTML  
用 `HTMLReportGenerator`

2. 我只想画一个图（折线/柱状）  
用 `charts.py` 的 renderer + `ChartConfig`

3. 我想把原始事件转成可画图的数据  
用 `transformers.py` 的 `DataTransformer`

4. 我想自定义报告布局  
用 `layouts.py` 的 `LayoutBuilder`

## 最小示例

```python
from my_utils.profiling.visualization import (
    HTMLReportGenerator,
    QuickReportGenerator,
)

# 1) 从 CSV 快速生成报告
quick = QuickReportGenerator()
quick.generate_from_csv("profile_rank_0.csv", output_path="csv_report.html")

# 2) 从统一分析报告对象生成 HTML
gen = HTMLReportGenerator()
html = gen.generate(report_obj, events, output_path="analysis_report.html")
```

## 关键文件

- `charts.py`  
  图表配置与渲染器工厂（Chart.js / Plotly / ECharts）。

- `transformers.py`  
  指标事件 -> 图表输入数据（时间序列、统计聚合）。

- `layouts.py`  
  报告布局构建（header/summary/cards/sections）。

- `html_generator.py`  
  最终 HTML 报告生成器（整合 findings/recommendations/charts）。

- `examples.py`  
  可视化模块内置示例。

## 实战建议

1. 先用 `QuickReportGenerator` 出一版默认报告。  
2. 如果要定制，先改 `layouts.py`，再补 `charts.py` 风格。  
3. 复杂图表先在 `transformers.py` 做数据清洗，避免模板层写逻辑。  
