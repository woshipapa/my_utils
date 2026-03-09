# Unified Profiling Roadmap (P0-P13)

This file tracks the implementation status of the profiling unification plan.

## Scope

- One event schema for all tools (`MetricEvent`, schema v1).
- Pluggable providers for online and offline profilers.
- Workload-aware analyzers and distributed alignment.
- Single CLI workflow for ingest/analyze/report/diff.
- Framework migration SDK (PyTorch/HF/DeepSpeed/Megatron).
- CI/test/doc/release governance baseline.

## Status

| Phase | Goal | Status |
| --- | --- | --- |
| P0 | Stability fixes, import/export chain, smoke readiness | Done |
| P1 | Unified metric schema v1 + validation/normalization | Done |
| P2 | Provider registry + config auto-wiring (`from_config`) | Done |
| P3 | NSYS schema-version adapter metadata + schema introspection | Done |
| P4 | Framework adapter SDK + registry | Done |
| P5 | Workload packs (`default/pretrain/finetune/inference/data_pipeline`) | Done |
| P6 | Distributed rank/step/stage alignment + skew diagnostics | Done |
| P7 | Analyzer v2 rule engine + advanced rules | Done |
| P8 | CLI workflow (`ingest/analyze/report/diff/list-providers`) | Done |
| P9 | Report diff model + markdown rendering | Done |
| P10 | Profiling tests + CI workflow | Done |
| P11 | Documentation refresh (quickstart + NSYS parsing + roadmap) | Done |
| P12 | Release governance policy and compatibility strategy | Done |
| P13 | End-to-end acceptance demo and runbook | Done |

## Deliverables by Phase

### P0-P2
- `metrics_types.py`: schema v1 fields (`schema_version`, `event_id`).
- `metrics_schema.py`: validator + normalizer.
- `provider_registry.py`: provider factories and auto registration.
- `metrics_collector.py`: config-driven provider bootstrap and validation stats.

### P3
- `nsys_schema_adapter.py`: version/family detection.
- `nsys_sqlite_provider.py`: schema tags include adapter family and exporter version.
- `nsys_sqlite_provider.py::describe_schema()`: table/column introspection.

### P4-P7
- `profiling/adapters/*`: framework adapter base + registry + built-ins.
- `workload_profiles.py`: workload packs and KPI focus.
- `analysis_rules.py`: modular rules (bottleneck, memory growth, variance, outlier, comm imbalance, pipeline bubble, dataloader stall, gpu utilization, distributed skew).
- `distributed_alignment.py`: rank-stage skew analysis.
- `metrics_analyzer.py`: workload-aware analyzer v2.

### P8-P10
- `cli.py`: single command surface.
- `metrics_diff.py`: run-to-run diff.
- `tests/profiling/*`: collector/analyzer/alignment/CLI tests.
- `.github/workflows/profiling-ci.yml`: test matrix workflow.

### P11-P13
- `UNIFIED_PROFILING_QUICKSTART.md`: CLI + config-driven quickstart.
- `NSYS_SQLITE_PARSING.md`: version adaptation and schema probing usage.
- `RELEASE_GOVERNANCE.md`: semver/compatibility/deprecation policy.
- `examples/p0_p13_end_to_end_demo.py`: acceptance demo.

