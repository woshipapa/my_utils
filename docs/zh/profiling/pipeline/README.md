# pipeline

编排层：把 provider、store、analyzer、report 串成一条统一流程。

## 核心入口

- `metrics_collector.py` 中的 `MetricsCollector`

## 典型流程

1. 注册 providers  
2. `collect()` 写入事件  
3. `analyze()` 生成分析结果  
4. `export_report()` / `export_chrome_trace()` 导出结果  

## 你什么时候会改这里

- 你要改“统一采集/分析流程本身”时（不是改单个 provider）。
- 你要新增统一导出能力（例如新的 report format）时。  
