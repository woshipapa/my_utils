# nsys-ai 全量扫描对齐矩阵

更新时间：2026-03-09  
扫描范围：
- 参考实现：`_tmp_nsys_ai/src/nsys_ai`（63 个 Python 文件）+ `tests`（22 个 Python 测试）
- 当前实现：`my_utils/profiling`（48 个 Python 文件）+ `third_party/my_utils/tests/profiling`（5 个 Python 测试）

状态定义：
- `已对齐`：能力已具备，接口可直接使用
- `部分对齐`：有核心能力但缺关键特性/易用性
- `未对齐`：当前没有对应能力

## 1. 核心能力对齐矩阵

| 能力域 | nsys-ai 参考能力 | my_utils 当前状态 | 代码位置（my_utils） | 差距说明 |
|---|---|---|---|---|
| Profile 输入 | 支持 `.sqlite` 与 `.nsys-rep`（自动 `nsys export`） | 部分对齐 | `sources/nsys_sqlite_provider.py` | 当前仅直接消费 SQLite；未内建 `.nsys-rep -> .sqlite` 自动转换 |
| Schema 自适应 | `NsightSchema` 自动识别版本/表 | 已对齐 | `sources/nsys_schema_adapter.py` | 已支持版本信息与 canonical table 选择 |
| StringIds 名称解析 | `StringIds` JOIN 恢复可读名 | 已对齐 | `sources/nsys_sqlite_provider.py` | runtime/kernel/nvtx 等都做了解析 |
| Kernel 细粒度指标 | kernel duration + launch config + resource 字段 | 已对齐 | `sources/nsys_sqlite_provider.py` | 已解析 `grid/block/registers/shared/localMemory*` 等 |
| Runtime 关联 | `correlationId` 对齐 runtime 与 kernel | 已对齐 | `sources/nsys_sqlite_provider.py` | 已注入 `runtime_api` 标签 |
| NVTX 解析 | range/mark + 自定义文本 | 已对齐 | `sources/nsys_sqlite_provider.py` | 已输出 `latency.nvtx.range`，含 `nvtx_text`/`name` |
| GPU Metrics 解析 | `GPU_METRICS` + `GENERIC_EVENTS` | 已对齐 | `sources/nsys_sqlite_provider.py` | 已含显式映射和动态 fallback |
| Network/PMU/UM | NIC/IB/PMU/UM 页面故障指标 | 已对齐 | `sources/nsys_sqlite_provider.py` | 已实现独立解析路径 |
| 未来表兼容 | 新 schema 表 fallback 解析 | 已对齐 | `sources/nsys_sqlite_provider.py` | 已支持 auto duration table |
| 多进程/多 rank 归因 | pid/tid/rank 维度切片 | 已对齐 | `sources/nsys_sqlite_provider.py` | 支持 pid/tid 解码与 rank 标签 |
| 多文件聚合 | 多 rank 文件 glob 聚合 | 已对齐 | `sources/nsys_sqlite_provider.py` | `NsysSqliteGlobMetricsProvider` 已支持 |
| SQL Skill 框架 | skill 参数化 SQL 执行模型 | 已对齐 | `sources/nsys_sql_skills.py` | 已有 `SqlSkill/SqlSkillParam/NsysSqlSkillEngine` |
| Skill 元信息 | 列出技能/参数 schema | 已对齐 | `sources/nsys_sql_skills.py`, `sources/nsys_sqlite_provider.py` | 已支持 `describe_sql_skills()` |
| Skill: top_kernels | 热点 kernel 聚合 | 已对齐 | `sources/nsys_sql_skills.py` | 已实现 |
| Skill: gpu_idle_gaps | 流级 idle gap 检测 | 已对齐 | `sources/nsys_sql_skills.py` | 已实现 |
| Skill: kernel_launch_overhead | CPU->GPU launch overhead | 已对齐 | `sources/nsys_sql_skills.py` | 已实现 |
| Skill: memory_transfers | copy kind 聚合 | 部分对齐 | `sources/nsys_sql_skills.py` | 已有 `memcpy_in_window`，但无统一方向名与总览格式化 |
| Skill: nvtx_kernel_map | NVTX -> kernel 关联映射 | 已对齐 | `sources/nsys_sql_skills.py` | 已新增 `nvtx_kernel_map` skill |
| Skill: nccl_breakdown | NCCL collective 分类统计 | 已对齐 | `sources/nsys_sql_skills.py` | 已新增 `nccl_breakdown` skill |
| Skill: schema_inspect | schema 结构探查 | 已对齐 | `sources/nsys_sql_skills.py` | 已新增 `schema_inspect` skill |
| Skill: thread_utilization | COMPOSITE_EVENTS 线程利用率 | 已对齐 | `sources/nsys_sql_skills.py` | 已新增 `thread_utilization` skill |
| 计算通信重叠分析 | compute-vs-NCCL overlap | 已对齐 | `sources/nsys_sql_skills.py`, `sources/nsys_sqlite_provider.py` | 已有 `analyze_compute_comm_overlap()` |
| GPU summary | span/busy/idle/top kernels | 已对齐 | `sources/nsys_sql_skills.py`, `sources/nsys_sqlite_provider.py` | 已有 `summarize_gpu_kernels()` |
| iteration 检测 | NVTX marker 驱动迭代边界 | 已对齐 | `sources/nsys_iterations.py`, `sources/nsys_sqlite_provider.py` | 已支持 `detect_iterations(marker=...)` |
| MFU 计算 | `model_flops_per_step` + `peak_tflops` | 已对齐 | `sources/nsys_mfu.py`, `sources/nsys_sqlite_provider.py` | 已支持 MFU 单次/对比计算 |
| Auto analyze 报告 | summary + overlap + nccl + iter + nvtx 汇总 | 部分对齐 | `analyzers/*`, `pipeline/metrics_collector.py` | 有统一规则分析器，但缺 nsys 专项“单命令全景报告” |
| Perfetto/Chrome trace 导出 | 时间轴导出 | 已对齐 | `output/metrics_trace.py` | 支持统一事件导出 + rank 对齐 |
| 平面导出 CSV/JSON | kernel flat export | 已对齐 | `sources/nsys_flat_export.py`, `cli.py` | 已提供 `nsys-export`（json/csv） |
| Web 时间线 UI | 浏览器交互式 timeline | 部分对齐 | `sources/nsys_timeline_html.py`, `cli.py` | 已支持静态 html 导出，尚未服务化交互式 viewer |
| TUI 时间线/树视图 | 终端时间线 + NVTX 树 | 未对齐 | - | 当前无 TUI 模块 |
| Profile diff（before/after） | kernel/nvtx 细粒度 diff + narrative | 部分对齐 | `output/metrics_diff.py` | 有 report diff，但没有 nsys 细粒度 diff 工具链 |
| AI Agent/Chat | DB tool + chat/ask/agent loop | 未对齐 | - | 当前 profiling 侧无 LLM agent 子系统 |
| CLI 专项命令覆盖 | info/summary/overlap/nccl/skill/agent/diff/export | 已对齐 | `cli.py` | 已新增 `nsys-analyze`/`nsys-export`/`nsys-diff`/`nsys-timeline-html` |
| 测试覆盖密度 | 参考实现测试较全 | 部分对齐 | `tests/profiling/*` | 现有 smoke/规则/trace 测试可用，但 nsys 专项覆盖不足 |

## 2. 总结视图

- `已对齐`：26 项  
- `部分对齐`：6 项  
- `未对齐`：2 项

结论：
- 你当前系统在“SQLite 指标提取 + 统一 metrics 管线 + 可视化报告 + trace 导出”上，已经显著强于普通脚本式解析。  
- 与 `nsys-ai` 的主要差距集中在“交互产品层（Web/TUI/Agent）”与“nsys 专项分析闭环（iteration/MFU/nccl_breakdown/nvtx_kernel_map/diff）”。

## 3. 建议对齐优先级（按 ROI）

### P0（已完成）
1. 补 `nccl_breakdown`、`nvtx_kernel_map`、`schema_inspect`、`thread_utilization` skills
2. 增加 `detect_iterations(marker=...)` 与 `mfu.py`（纯计算模块）

### P1（已完成，基础版）
1. 增加 nsys 专项 flat export（`nsys-export`，支持 json/csv）
2. 增加 nsys 专项 analyze 子命令（`nsys-analyze`，汇总 summary+overlap+nccl+iters+mfu）
3. 增加 nsys 专项 diff（`nsys-diff`，kernel/nvtx 双维基础版）

### P2（产品化增强）
1. Web timeline viewer（已完成静态 html 导出：`nsys-timeline-html`）
2. TUI timeline/tree
3. AI agent/chat 与 profile-db tool

## 4. 本矩阵依据的关键文件

参考实现（nsys-ai）：
- `src/nsys_ai/profile.py`
- `src/nsys_ai/summary.py`
- `src/nsys_ai/overlap.py`
- `src/nsys_ai/mfu.py`
- `src/nsys_ai/report.py`
- `src/nsys_ai/skills/base.py`
- `src/nsys_ai/skills/registry.py`
- `src/nsys_ai/skills/builtins/*.py`
- `src/nsys_ai/cli/app.py`

当前实现（my_utils）：
- `my_utils/profiling/sources/nsys_schema_adapter.py`
- `my_utils/profiling/sources/nsys_sqlite_provider.py`
- `my_utils/profiling/sources/nsys_sql_skills.py`
- `my_utils/profiling/cli.py`
- `my_utils/profiling/analyzers/*`
- `my_utils/profiling/output/metrics_trace.py`
- `my_utils/profiling/output/metrics_diff.py`
