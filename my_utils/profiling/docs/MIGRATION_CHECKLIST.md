# Profiling Migration Checklist (Legacy -> Unified)

This checklist is for users already running:

- `MyTimer`
- optional `torch.profiler`
- optional NSYS/NCU offline files

and migrating to the new unified profiling stack.

## 0. Quick Compatibility Summary

- Existing pattern still works:
  - `collector = MetricsCollector(...)`
  - `collector.register_provider(...)`
  - `collector.collect(step=...)`
  - `collector.analyze()`
- New fields are additive (`schema_version`, `overall_score`, etc.).
- No forced migration to CLI/config mode.

## 1. Keep Old Behavior First (Safe Mode)

Use this to minimize surprises:

```python
from my_utils.profiling import MetricsCollector, MetricsAnalyzer

collector = MetricsCollector(
    output_dir="./metrics",
    validate_events=False,
    drop_invalid_events=False,
    analyzer=MetricsAnalyzer(
        workload_profile="default",
        enable_advanced_rules=False,
    ),
)
```

This keeps analysis close to old behavior while you verify outputs.

## 2. If You Use Custom JSON Parsers

If downstream scripts validate report/event JSON fields strictly:

1. add allow-list entries for new keys:
   - event: `schema_version`, `event_id`
   - report: `schema_version`, `overall_score`
2. avoid exact-key equality checks
3. prefer tolerant parsing with defaults

## 3. MyTimer + torch.profiler Alignment (Recommended)

To correlate stage-level and op-level data, ensure both carry:

- `step`
- `rank`
- stable stage/op names (`stage`, `op`)

Recommended runtime tags:

```python
collector.collect(step=step, tags={"run": run_id, "rank": str(rank)})
```

## 4. Add Offline NSYS/NCU Incrementally

Start with online providers, then add offline ones:

```python
collector.register_provider(NsysSqliteMetricsProvider("./train_rank0.sqlite"))
collector.register_provider(NcuCsvMetricsProvider("./ncu.csv"))
```

For NSYS schema checks:

```python
p = NsysSqliteMetricsProvider("./train_rank0.sqlite")
print(p.describe_schema()["version_info"])
```

## 5. Switch to Config-Driven Bootstrap (Optional)

When ready, move provider wiring into config:

```python
collector = MetricsCollector.from_config("./collector.json")
```

If some providers are optional in your environment:

- set `collector.ignore_provider_errors = true` in config.

## 6. Switch to Workload Profiles

Use profile by workload:

- `default`: conservative baseline
- `pretrain`: distributed/comm-heavy checks
- `finetune`: stability + dataloader checks
- `inference`: latency-tail focus
- `data_pipeline`: ingestion/preprocess focus

## 7. Add Regression Diff to CI

Generate report JSON per run, then compare:

```bash
myutils-profile diff \
  --base-report ./baseline/analysis_report.json \
  --target-report ./current/analysis_report.json \
  --output ./diff.json \
  --markdown ./diff.md
```

## 8. Export Chrome Trace with Cross-Rank Alignment

```bash
myutils-profile trace \
  --events ./metrics_out/metrics_events.jsonl \
  --output ./metrics_out/metrics_trace.json \
  --auto-align-ranks \
  --reference-rank 0 \
  --rank-clock-offset 1=-0.0023
```

Notes:

- `--auto-align-ranks` estimates offsets from shared step anchors.
- `--rank-clock-offset rank=sec` applies manual correction (can repeat).
- You can combine both; final offset is `estimated + manual`.
- Open output JSON in `chrome://tracing`.

## 9. Final Validation Before Full Cutover

1. run one baseline experiment with old flow
2. run same experiment with unified flow in safe mode
3. compare:
   - key stage latency
   - memory trend
   - top bottlenecks
4. enable advanced rules and workload profile
5. freeze config and CI thresholds

## 10. Rollback Plan

If outputs diverge unexpectedly:

1. disable advanced rules (`enable_advanced_rules=False`)
2. disable schema validation (`validate_events=False`)
3. keep only MyTimer provider
4. add providers back one-by-one
