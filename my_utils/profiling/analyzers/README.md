# analyzers

分析层，负责把“原始指标事件”变成“可执行结论”。

## 你什么时候会改这里

- 新增瓶颈规则（比如 memory bound / load imbalance / comm skew）。
- 调整不同 workload（pretrain / inference / rl）的判断逻辑。
- 增加多机多卡对齐分析维度。

## 关键文件

- `metrics_analyzer.py`: 主分析入口（统一输出 findings/recommendations）。
- `analysis_rules.py`: 规则定义与命中逻辑。
- `workload_profiles.py`: 不同业务场景的分析配置。
- `distributed_alignment.py`: rank/stage 对齐分析。

## 建议改动顺序

1. 先在 `analysis_rules.py` 定义规则。  
2. 在 `metrics_analyzer.py` 挂载规则。  
3. 用 `examples/` 的 demo 跑一遍确认输出变化。  
