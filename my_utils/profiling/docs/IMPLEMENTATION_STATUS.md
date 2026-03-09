# Implementation Status

This file maps the P0-P13 plan to concrete code.

## P0-P2 Core

- `metrics_types.py`: canonical schema v1.
- `metrics_schema.py`: validator + normalizer.
- `metrics_provider.py`: provider capabilities contract.
- `provider_registry.py`: provider factory registry + built-ins.
- `metrics_collector.py`: config-driven provider bootstrap and collection.

## P3 NSYS

- `nsys_sqlite_provider.py`: schema tags and parser.
- `nsys_schema_adapter.py`: exporter/version family detection.
- `NsysSqliteMetricsProvider.describe_schema()`: introspection helper.

## P4-P7 Analysis and Extensibility

- `adapters/*`: framework adapter SDK.
- `distributed_alignment.py`: rank/step/stage alignment and skew checks.
- `analysis_rules.py`: modular rule engine.
- `workload_profiles.py`: workload packs.
- `metrics_analyzer.py`: analyzer v2.

## P8-P9 Workflow and Diff

- `cli.py`: ingest/analyze/report/diff/list-providers.
- `metrics_diff.py`: run-to-run report diff.
- `metrics_report.py`: report rendering and diff markdown.
- `metrics_trace.py`: metrics events -> Chrome Trace (`traceEvents`) with rank alignment.

## P10 Quality

- `tests/profiling/*`: unit and smoke tests.
- `.github/workflows/profiling-ci.yml`: CI matrix.

## P11-P13 Docs and Acceptance

- `UNIFIED_PROFILING_QUICKSTART.md`
- `NSYS_SQLITE_PARSING.md`
- `ROADMAP.md`
- `RELEASE_GOVERNANCE.md`
- `examples/p0_p13_end_to_end_demo.py`
