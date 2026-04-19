# examples（可运行示例）

这个目录用于两类事情：

1. 快速验证统一 metrics 管线是否工作。  
2. 给你可复制的配置模板（尤其是 NSYS sqlite 配置）。  

## 30秒定位

1. 我想跑最小 demo（provider + analyze + report）  
运行 `unified_metrics_demo.py`

2. 我想做端到端验收（含 diff）  
运行 `p0_p13_end_to_end_demo.py`

3. 我想直接用 CLI 配置跑离线分析  
用 `collector_config_*.json` + `myutils-profile ingest`

## 最常用命令

### A) 最小统一 demo

```bash
python -m my_utils.profiling.examples.unified_metrics_demo \
  --output-dir ./demo_metrics_output \
  --steps 30
```

产物：`metrics_events.jsonl`、`report.json`、`report.md`、`report.html` 等。

### B) 端到端验收 demo

```bash
python -m my_utils.profiling.examples.p0_p13_end_to_end_demo \
  --output-dir ./p0_p13_demo_output \
  --steps 20
```

可选加 sqlite 探测：

```bash
python -m my_utils.profiling.examples.p0_p13_end_to_end_demo \
  --output-dir ./p0_p13_demo_output \
  --nsys-sqlite ./train_rank0.sqlite
```

### C) 用 JSON 配置跑 CLI（离线）

单 sqlite：

```bash
myutils-profile ingest \
  --config ./my_utils/profiling/examples/collector_config_nsys_sqlite_full.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html
```

多 rank glob：

```bash
myutils-profile ingest \
  --config ./my_utils/profiling/examples/collector_config_nsys_multi_rank_full.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html
```

## 配置文件说明

- `collector_config_example.json`  
  全量离线 provider 示例（table_csv / ncu_csv / nsys_sqlite / cprofile / perf_stat）。

- `collector_config_nsys_sqlite_full.json`  
  单个 sqlite 的完整配置模板。

- `collector_config_nsys_multi_rank_full.json`  
  多 rank sqlite_glob 的完整配置模板。

## 常见坑

1. `nsys_sqlite` 用的是 `sqlite_path`，不是 `db_path`。  
2. `nsys_sqlite_glob` 用的是 `sqlite_glob`。  
3. `sqlite_glob` 命中文件扩展名可以不是 `.sqlite`，但文件内容必须是真正 SQLite。  
