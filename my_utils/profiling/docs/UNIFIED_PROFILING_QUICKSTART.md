# Unified Profiling Quickstart

This quickstart shows the new unified flow:

1. collect metrics from multiple providers into one schema,
2. run workload-aware analysis rules,
3. export JSON/Markdown/HTML reports,
4. compare two runs with report diff.

## 1. Install

```bash
pip install -e .
```

CLI entry:

```bash
myutils-profile --help
```

## 2. Canonical Metric Schema

All providers emit `MetricEvent`:

- `name`: canonical namespace (`latency.*`, `memory.*`, `compute.*`, `comm.*`, `io.*`, `calls.*`, `perf.*`)
- `value`, `unit`
- `tags`: dimensions (`step`, `rank`, `stage`, `op`, `kernel`, ...)
- `provider_id`
- `schema_version` (currently `1.0`)

## 3. Config-Driven Collector

`collector.json` example:

```json
{
  "collector": {
    "output_dir": "./metrics_out",
    "enabled": true,
    "validate_events": true,
    "drop_invalid_events": false
  },
  "schema": {
    "strict": false,
    "enforce_known_prefix": false,
    "enforce_recommended_units": false
  },
  "analysis": {
    "workload_profile": "pretrain",
    "bottleneck_threshold": 0.1,
    "cv_threshold": 0.5,
    "memory_growth_bytes_per_step": 10485760
  },
  "providers": [
    {
      "type": "table_csv",
      "id": "ext_csv",
      "enabled": true,
      "params": {
        "csv_path": "./external_metrics.csv",
        "value_column": "latency_ms",
        "name_column": "op_name",
        "tag_columns": ["step", "rank"],
        "unit": "ms",
        "event_name_prefix": "latency.external"
      }
    },
    {
      "type": "nsys_sqlite",
      "id": "nsys_rank0",
      "enabled": true,
      "params": {
        "sqlite_path": "./train_rank0.sqlite",
        "include_gpu_metrics": true,
        "include_network_metrics": true
      }
    }
  ]
}
```

## 4. CLI Workflow

### 4.1 Ingest + Analyze

```bash
myutils-profile ingest \
  --config ./collector.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html
```

### 4.2 Analyze Existing Events JSONL

```bash
myutils-profile analyze \
  --events ./metrics_out/metrics_events.jsonl \
  --workload pretrain \
  --output-dir ./analysis_out \
  --report-formats json,markdown,html
```

### 4.3 Diff Two Reports

```bash
myutils-profile diff \
  --base-report ./run_a/analysis_report.json \
  --target-report ./run_b/analysis_report.json \
  --output ./diff.json \
  --markdown ./diff.md
```

### 4.4 Export Chrome Trace from Unified Events

```bash
myutils-profile trace \
  --events ./metrics_out/metrics_events.jsonl \
  --output ./metrics_out/metrics_trace.json \
  --auto-align-ranks \
  --reference-rank 0 \
  --include-metric-prefixes latency
```

Then open `metrics_trace.json` in `chrome://tracing`.

## 5. Python API Workflow

```python
from my_utils.profiling import MetricsCollector, MetricsAnalyzer

collector = MetricsCollector.from_config("./collector.json")
collector.collect(step=100, tags={"run": "exp_a"})

report = collector.analyze()
collector.export_report(fmt="json", report=report)
collector.export_report(fmt="markdown", report=report)
collector.export_report(fmt="html", report=report)
collector.export_chrome_trace(output_path="./metrics_out/metrics_trace.json")
```

## 6. Workload Profiles

Built-in workload profiles:

- `default`
- `pretrain`
- `finetune`
- `inference`
- `data_pipeline`

Each profile selects a rule set and KPI focus. You can extend by adding new rules and profile definitions.

## 7. Distributed Alignment

Use cross-rank alignment helpers:

```python
from my_utils.profiling import align_stage_latency, analyze_rank_skew

cube = align_stage_latency(events)
skew = analyze_rank_skew(events, skew_ratio_threshold=1.2)
```

## 8. Framework Adapters

Adapters available:

- `PyTorchAdapter`
- `HuggingFaceAdapter`
- `DeepSpeedAdapter`
- `MegatronAdapter`
- `TorchTitanAdapter`
- `VerlAdapter`
- `SlimeAdapter`
- `RollAdapter`
- `SGLangAdapter`
- `VLLMAdapter`

They can auto-generate provider specs from runtime context and register providers into `MetricsCollector`.

## 9. Migration Checklist

If you are migrating from existing scripts, follow:

- `my_utils/profiling/docs/MIGRATION_CHECKLIST.md`
