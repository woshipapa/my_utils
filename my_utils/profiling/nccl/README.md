# NCCL Inspector quick guide

Parses the Inspector output of NCCL's profiler plugin:

- JSON/JSONL dumps (`NCCL_INSPECTOR_PROM_DUMP=0`): one JSON object per
  collective/P2P operation.
- Prometheus textfile (`NCCL_INSPECTOR_PROM_DUMP=1`): `nccl_*` metrics for
  the node exporter to scrape.

Parsing is offline and pure Python; only the capture step needs a real
NCCL/GPU environment.

## Capture

```bash
export NCCL_PROFILER_PLUGIN=/path/to/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_DIR=./nccl-inspector-logs
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500

torchrun --nproc_per_node=8 train.py
```

Or use the wrapper:

```bash
bash my_utils/profiling/nccl/run_nccl_inspector.sh \
  --plugin /path/to/libnccl-profiler-inspector.so \
  --dump-dir ./nccl-inspector-logs \
  --interval-us 500 \
  -- torchrun --nproc_per_node=8 train.py
```

Prometheus mode:

```bash
export NCCL_PROFILER_PLUGIN=/path/to/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_PROM_DUMP=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000
export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector
```

## Parse

List skills:

```bash
myutils-profile nccl-inspector-skill --input ./nccl-inspector-logs --list-skills --pretty
```

Analyze JSON dumps directly:

```bash
myutils-profile nccl-inspector-analyze \
  --input ./nccl-inspector-logs \
  --top-k 20 \
  --pretty
```

Include a Prometheus textfile as well:

```bash
myutils-profile nccl-inspector-analyze \
  --input ./nccl-inspector-logs \
  --prometheus-path /var/lib/node_exporter/nccl_inspector \
  --format markdown \
  --output nccl_inspector.md
```

Only large-message AllReduce:

```bash
myutils-profile nccl-inspector-skill \
  --input ./nccl-inspector-logs \
  --skill top_collectives \
  --param op_like=AllReduce \
  --param min_msg_size_bytes=1048576 \
  --pretty
```

## What to look at in the output

1. `summary.timing_sources` — confirm timings mostly come from `kernel_gpu`.
2. `top_collectives` / `top_p2p` — time and bandwidth aggregated by op,
   message-size bucket, and communicator.
3. `rank_skew` — per-rank duration spread for the same NCCL sequence.
4. `prometheus_summary` — labels and max values of each `nccl_*` metric in
   Prometheus mode.

Other Inspector environment variables worth knowing (see the upstream NCCL
Inspector docs):

- `NCCL_INSPECTOR_ENABLE_P2P`
- `NCCL_INSPECTOR_DUMP_VERBOSE`
- `NCCL_INSPECTOR_PROM_DUMP`
- `NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES`
- `NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING`

---

Chinese original: [docs/zh/profiling/nccl/README.md](../../../docs/zh/profiling/nccl/README.md)
