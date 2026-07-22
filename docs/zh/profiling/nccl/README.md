# NCCL Inspector 快速使用手册

这部分用于解析 NCCL 最新 profiler plugin 里的 Inspector 输出：

- JSON/JSONL dump：`NCCL_INSPECTOR_PROM_DUMP=0`，每条 collective/P2P 写一个 JSON 对象。
- Prometheus textfile：`NCCL_INSPECTOR_PROM_DUMP=1`，写 `nccl_*` 指标供 node exporter 抓取。

## 采集

```bash
export NCCL_PROFILER_PLUGIN=/path/to/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_DIR=./nccl-inspector-logs
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500

torchrun --nproc_per_node=8 train.py
```

也可以用 wrapper：

```bash
bash my_utils/profiling/nccl/run_nccl_inspector.sh \
  --plugin /path/to/libnccl-profiler-inspector.so \
  --dump-dir ./nccl-inspector-logs \
  --interval-us 500 \
  -- torchrun --nproc_per_node=8 train.py
```

Prometheus 模式：

```bash
export NCCL_PROFILER_PLUGIN=/path/to/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_PROM_DUMP=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000
export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector
```

## 解析

列出技能：

```bash
myutils-profile nccl-inspector-skill --input ./nccl-inspector-logs --list-skills --pretty
```

直接分析 JSON dump：

```bash
myutils-profile nccl-inspector-analyze \
  --input ./nccl-inspector-logs \
  --top-k 20 \
  --pretty
```

同时带上 Prometheus textfile：

```bash
myutils-profile nccl-inspector-analyze \
  --input ./nccl-inspector-logs \
  --prometheus-path /var/lib/node_exporter/nccl_inspector \
  --format markdown \
  --output nccl_inspector.md
```

只看大消息的 AllReduce：

```bash
myutils-profile nccl-inspector-skill \
  --input ./nccl-inspector-logs \
  --skill top_collectives \
  --param op_like=AllReduce \
  --param min_msg_size_bytes=1048576 \
  --pretty
```

## 输出重点

1. `summary.timing_sources`：确认是否主要来自 `kernel_gpu`。
2. `top_collectives` / `top_p2p`：按 op、消息大小 bucket、comm 聚合耗时和带宽。
3. `rank_skew`：同一 NCCL sequence 在不同 rank 上的耗时差异。
4. `prometheus_summary`：Prometheus 模式下各 `nccl_*` 指标的 label 和最大值。

NCCL Inspector 最新文档中的关键环境变量也要留意：

- `NCCL_INSPECTOR_ENABLE_P2P`
- `NCCL_INSPECTOR_DUMP_VERBOSE`
- `NCCL_INSPECTOR_PROM_DUMP`
- `NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES`
- `NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING`
