# output

输出层：把分析结果渲染成你能直接消费的文件。

## 主要能力

- 报告渲染：JSON / Markdown / HTML
- 报告对比：before vs after diff
- Trace 导出：Chrome Trace（含 rank 时钟对齐）

## 关键文件

- `metrics_report.py`: 报告写出与格式渲染。
- `metrics_diff.py`: 报告差异对比。
- `metrics_trace.py`: 事件转 Chrome Trace。  
