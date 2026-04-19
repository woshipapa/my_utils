# metrics

统一指标数据层：定义“指标是什么、从哪来、怎么存、怎么校验”。

## 你什么时候会改这里

- 接入新数据源（新的 provider）。
- 增加新的指标字段或 schema 约束。
- 调整指标分类、命名、归一化策略。

## 关键文件

- `metrics_types.py`: 核心类型（`MetricEvent` / `AnalysisReport` 等）。
- `metrics_schema.py`: schema 校验与规范化。
- `metrics_provider.py`: provider 抽象接口。
- `metrics_providers.py`: 内置 provider 实现。
- `provider_registry.py`: provider 注册与按配置实例化。
- `metrics_store.py`: 事件落盘与读取。
- `metrics_taxonomy.py`: 指标分类体系。

## 最常见改动

1. 新 provider：实现 `MetricsProvider` 接口。  
2. 在 `provider_registry.py` 注册。  
3. 在 `examples/collector_config_example.json` 写配置并验证。  
