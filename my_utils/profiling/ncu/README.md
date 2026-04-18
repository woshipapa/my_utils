# NCU Profiling Templates

这个目录给你一套和 `nsys` 类似的 `ncu` 启动方式：改 YAML -> 一键跑。

## 文件说明

- `run_ncu_quick_yaml.py`: YAML 启动器，自动拼接 `ncu` 命令。
- `ncu_quick_launch.yaml`: 最小模板（先跑通）。
- `ncu_2026_1_1_full_args.yaml`: 全量参数模板（按官方分类 + 每项注释）。
- `ncu_2026_1_1_cli_quick_reference.md`: 参数扫描记录与分类索引。
- `ncu_csv_tools.py`: ncu CSV 解析与分析技能引擎。
- `ncu_report_tools.py`: `.ncu-rep` 直读解析与分析技能引擎（基于 `ncu_report`）。
- `run_ncu_csv_skill.sh`: shell 包装（运行 `ncu-csv-skill`）。
- `run_ncu_csv_analyze.sh`: shell 包装（运行 `ncu-csv-analyze`）。
- `run_ncu_report_skill.sh`: shell 包装（运行 `ncu-report-skill`）。
- `run_ncu_report_analyze.sh`: shell 包装（运行 `ncu-report-analyze`）。

## 最短流程

1. 复制并修改 YAML（训练命令在 `command`）。
2. 把你要的参数从 `null` 改成值。
3. 运行：

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_quick_launch.yaml
```

全量参数模板：

```bash
python my_utils/profiling/ncu/run_ncu_quick_yaml.py \
  --config my_utils/profiling/ncu/ncu_2026_1_1_full_args.yaml
```

## `profile_switches` 规则

- `null`: 不传参数
- 标量: 变成 `--key=value`
- 列表: 变成多次 `--key=item`
- `"__flag__"`: 变成裸参数 `--key`

## 备注

- 启动器支持 `ncu` 与 `ncu_launch` 两个 YAML 顶层键名（兼容旧写法）。
- 同名参数冲突时，`profile_switches` 优先级高于 core 字段。
- `myutils-profile nsys-panel` 现在同时可选择 `nsys-*` 和 `ncu-*` 子命令（统一交互面板）。

## CSV 解析工具（类似 nsys-sql skills）

先从 `.ncu-rep` 导出 CSV（建议 `--page=raw --csv`），然后运行：

```bash
myutils-profile ncu-csv-skill --csv ./ncu_raw.csv --list-skills --pretty
myutils-profile ncu-csv-skill --csv ./ncu_raw.csv --skill summary --pretty
myutils-profile ncu-csv-skill --csv ./ncu_raw.csv --skill top_kernels --param metric_like=%time% --param top_k=20 --pretty
myutils-profile ncu-csv-analyze --csv ./ncu_raw.csv --top-k 20 --pretty
```

内置 skills：

- `summary`: 行数、指标数、kernel 数、统计分位。
- `top_kernels`: 按 metric 聚合 kernel 排名（sum/avg/max/min）。
- `top_metrics`: 指标聚合排名。
- `metric_percentiles`: 指标分位（p50/p90/p99）。
- `schema_inspect`: CSV 列结构与样例行。

## .ncu-rep 直读（你要求的“全指标”路径）

依赖官方 Python Report Interface（`import ncu_report`）：

```bash
myutils-profile ncu-report-skill --report ./run.ncu-rep --list-skills --pretty
myutils-profile ncu-report-skill --report ./run.ncu-rep --skill summary --pretty
myutils-profile ncu-report-skill --report ./run.ncu-rep --skill per_metric_stats --pretty
myutils-profile ncu-report-skill --report ./run.ncu-rep --skill rule_results --pretty
myutils-profile ncu-report-skill --report ./run.ncu-rep --skill bottleneck_report --param top_k=10 --pretty
myutils-profile ncu-report-analyze --report ./run.ncu-rep --top-k 20 --pretty
```

`ncu-report-analyze` 默认会输出：

- `summary`
- `per_metric_stats`（每个 metric 的 samples/min/max/avg/p50/p90/p99）
- `top_kernels`
- `all_metrics`（全指标明细行，默认上限 20000，可调）
- `rule_results`（官方 NCU rules 解析）
- `bottleneck_report`（`rules + fallback heuristics + coverage`）

## 完整性与瓶颈定位

`bottleneck_report` 里新增了三层信息：

- `coverage`: 关键分析维度覆盖检查（SOL compute/memory、occupancy、scheduler、stall、memory hierarchy、launch）。
- `rule_findings`: 直接来自 NCU `rule_results_as_dicts()` 的诊断结论（优先级最高）。
- `heuristic_findings`: 当 rule 不完整时的 fallback 判定（SM/DRAM/occupancy/issue/stall 信号）。

建议流程：

1. 先看 `coverage_score`，低于 100 先补采指标再下结论。
2. 再看 `top_bottlenecks`，优先处理 `source=ncu_rule`。
3. 最后结合 `all_metrics`/源码做二次确认。
