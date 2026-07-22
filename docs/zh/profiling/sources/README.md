# sources（NSYS SQLite 离线解析）

这个目录负责：读取 `nsys export` 产生的 SQLite，然后做可复用分析。

## 30秒定位

1. 我只想看训练整体分析  
用 CLI：`myutils-profile nsys-analyze`

2. 我想跑某个 SQL skill  
用 CLI：`myutils-profile nsys-sql-skill`

3. 我想导出 kernel 明细  
用 CLI：`myutils-profile nsys-export`

4. 我想对比两次 profile  
用 CLI：`myutils-profile nsys-diff`

5. 我想出 timeline HTML  
用 CLI：`myutils-profile nsys-timeline-html`

## 最常用命令

统一分析：

```bash
myutils-profile nsys-analyze --sqlite ./train_rank0.sqlite --output ./nsys_analyze.json
```

列出 SQL skills：

```bash
myutils-profile nsys-sql-skill --sqlite ./train_rank0.sqlite --list-skills --pretty
```

运行一个 skill：

```bash
myutils-profile nsys-sql-skill \
  --sqlite ./train_rank0.sqlite \
  --skill top_kernels \
  --param device_id=0 \
  --param limit=20 \
  --pretty
```

导出 kernel 明细：

```bash
myutils-profile nsys-export --sqlite ./train_rank0.sqlite --format csv --output ./kernels.csv
```

两次对比：

```bash
myutils-profile nsys-diff --before-sqlite ./a.sqlite --after-sqlite ./b.sqlite --output ./diff.json
```

timeline html：

```bash
myutils-profile nsys-timeline-html --sqlite ./train_rank0.sqlite --output ./timeline.html
```

## 关键文件（按职责）

- `nsys_schema_adapter.py`  
  跨版本 schema 识别（不同 nsys 导出的表名/列名差异适配）。

- `nsys_sql_skills.py`  
  内置 SQL skill 引擎（top kernels、overlap、nvtx、memcpy、occupancy 等）。

- `nsys_sqlite_provider.py`  
  上层 provider 封装，给统一 metrics 管线调用。

- `nsys_analyze.py`  
  一站式分析聚合（summary/overlap/nccl/iterations/mfu）。

- `nsys_diff.py`  
  before/after 差异分析。

- `nsys_flat_export.py`  
  扁平化导出 kernel timeline 到 json/csv。

- `nsys_timeline_html.py`  
  静态 HTML 时间线导出。

- `nsys_iterations.py`  
  基于 NVTX marker 的 iteration 切分。

- `nsys_mfu.py`  
  MFU 相关辅助计算。

- `nsys_module_kernel_compare.py`  
  模块级 kernel 对比（更细粒度）。

## 调试建议

1. 先用 `schema_inspect` skill 看 sqlite 是否识别正确。  
2. 再跑 `nsys-analyze` 看总览。  
3. 如果出现回退，优先跑 `nsys-diff`。  
4. 需要可视化定位时再导出 `timeline.html`。  
