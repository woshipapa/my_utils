# Profiling 文档导航

这份索引只做一件事：帮你快速找到“下一步该看哪篇文档”。

## 30秒定位

0. **我要系统学会用 my_utils 做性能分析（采集全部指标 + 得到优化建议）**  
看 **`PERFORMANCE_ANALYSIS_HANDBOOK.md`** —— 端到端权威手册，涵盖采集命令、分析 API、
判读阈值（附出处）、硬件天花板、以及会让数据说谎的陷阱。人和 agent 都从这里开始。

0b. **我想知道这套工具都有哪些能力、每个能力是怎么来的、为什么要加**  
看 **`CAPABILITY_EVOLUTION.md`** —— 从最早版本到现在的完整演进记录：每个能力的来源
（本地验证 / 官方源码 / 我们自己推导）和它解决的具体问题。凡是因为“出过错”才加的
能力，都写明了错在哪。

1. 我只想马上跑起来  
看 `UNIFIED_PROFILING_QUICKSTART.md`

2. 我想看 NSYS 的版本差异/参数参考  
看 `nsys_2026_2_cli_quick_reference.md` 和 `NSYS_FRAMEWORK_LAUNCH_COMPAT_2026.md`

3. 我想看 NCU 参数与完整性审计  
看 `../ncu/ncu_2026_1_1_cli_quick_reference.md` 和 `../ncu/NCU_ANALYSIS_COMPLETENESS_AUDIT_2026_04_19.md`

4. 我想理解统一架构设计  
看 `UNIFIED_METRICS_DESIGN.md`、`AUTO_ANALYZER_DESIGN.md`、`VISUALIZATION_DESIGN.md`

5. 我想看迁移/发布规范  
看 `MIGRATION_CHECKLIST.md`、`RELEASE_GOVERNANCE.md`

## 常用文档（按用途）

### 快速上手

- [PERFORMANCE_ANALYSIS_HANDBOOK.md](./PERFORMANCE_ANALYSIS_HANDBOOK.md) —— 端到端手册
- [CAPABILITY_EVOLUTION.md](./CAPABILITY_EVOLUTION.md) —— 能力演进与出处
- [UNIFIED_PROFILING_QUICKSTART.md](./UNIFIED_PROFILING_QUICKSTART.md)
- [CROSS_FRAMEWORK_PROFILE_REFERENCE.md](./CROSS_FRAMEWORK_PROFILE_REFERENCE.md)
- [FRAMEWORK_INTEGRATION_PLAYBOOK_ZH.md](./FRAMEWORK_INTEGRATION_PLAYBOOK_ZH.md)
- [../examples/framework_playbook_samples/README.md](../examples/framework_playbook_samples/README.md)

### NSYS 相关

- [NSYS_SQLITE_PARSING.md](./NSYS_SQLITE_PARSING.md)
- [NSYS_AI_ALIGNMENT_MATRIX.md](./NSYS_AI_ALIGNMENT_MATRIX.md)
- [NSYS_FRAMEWORK_LAUNCH_COMPAT_2026.md](./NSYS_FRAMEWORK_LAUNCH_COMPAT_2026.md)
- [nsys_2026_2_cli_quick_reference.md](./nsys_2026_2_cli_quick_reference.md)
- [nsys_2024_7_1_cli_quick_reference.md](./nsys_2024_7_1_cli_quick_reference.md)

### NCU 相关

- [../ncu/README.md](../ncu/README.md)
- [../ncu/ncu_2026_1_1_cli_quick_reference.md](../ncu/ncu_2026_1_1_cli_quick_reference.md)
- [../ncu/NCU_ANALYSIS_COMPLETENESS_AUDIT_2026_04_19.md](../ncu/NCU_ANALYSIS_COMPLETENESS_AUDIT_2026_04_19.md)

### 架构与实现

- [UNIFIED_METRICS_DESIGN.md](./UNIFIED_METRICS_DESIGN.md)
- [AUTO_ANALYZER_DESIGN.md](./AUTO_ANALYZER_DESIGN.md)
- [VISUALIZATION_DESIGN.md](./VISUALIZATION_DESIGN.md)
- [FRAMEWORK_ADAPTERS_DESIGN.md](./FRAMEWORK_ADAPTERS_DESIGN.md)
- [TORCH_COMPILE_REFERENCE.md](./TORCH_COMPILE_REFERENCE.md)

### 规划与治理

- [ROADMAP.md](./ROADMAP.md)
- [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md)
- [MIGRATION_CHECKLIST.md](./MIGRATION_CHECKLIST.md)
- [RELEASE_GOVERNANCE.md](./RELEASE_GOVERNANCE.md)
