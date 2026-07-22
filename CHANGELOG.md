# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The supported public API surface is declared in
[docs/API_STABILITY.md](docs/API_STABILITY.md).

## [0.1.0] - 2026-07-21

Initial public release.

### Added

- **Nsight Compute (ncu) and Nsight Systems (nsys) diagnostic engine**:
  schema-adaptive readers for nsys SQLite exports (`NsysSqliteMetricsProvider`,
  `NsysSqlSkillEngine`), ncu CSV and `.ncu-rep` analysis, one-shot analysis
  (`analyze_nsys_sqlite`) and capture-to-capture diffing (`diff_nsys_sqlite`),
  iteration detection, MFU computation, flat kernel export, and standalone HTML
  timeline export.
- **PC/PM sampling analysis**: PC-sampling and GPU performance-metrics
  (PM sampling) ingestion and analysis rules for kernel- and pipeline-level
  bottleneck diagnosis.
- **Source correlation**: kernel-to-source attribution linking sampled hotspots
  back to originating source lines and modules.
- **Clock guards**: distributed clock synchronization and rank-skew analysis
  (`ClockSynchronizer`, `align_stage_latency`, `analyze_rank_skew`,
  `estimate_rank_time_offsets`) to keep multi-rank timelines comparable.
- **Metrics pipeline**: canonical `MetricEvent` schema
  (`PROFILE_SCHEMA_VERSION`), schema validation and unit canonicalization,
  metric taxonomy with tool-alias normalization, `MetricsStore`,
  `MetricsCollector`, `MetricsAnalyzer` with workload profiles and pluggable
  `AnalysisRule`s, report rendering, report diffing, and Chrome-trace export.
- **Provider registry**: 12 built-in metrics providers (timers, torch profiler,
  module profiler, table/ncu CSV, nsys SQLite single and glob, cProfile,
  perf stat, DCGM CSV, NCCL logs, RAS JSON) behind a uniform
  `MetricsProviderRegistry`.
- **Framework adapters** for PyTorch, HuggingFace, DeepSpeed, Megatron,
  TorchTitan, verl, slime, ROLL, SGLang, and vLLM.
- **NCCL Inspector tooling**: event/Prometheus ingestion, skill engine, and
  markdown analysis reports.
- **16 console-script CLIs**: `myutils-profile`, `nsys-panel`,
  `nsys-sql-skill`, `nsys-export`, `nsys-analyze`, `nsys-diff`,
  `nsys-module-kernel-compare`, `nsys-timeline-html`, `nsys-iter-overlap`,
  `nsys-iter-outliers`, `ncu-csv-skill`, `ncu-csv-analyze`,
  `ncu-report-skill`, `ncu-report-analyze`, `nccl-inspector-skill`,
  `nccl-inspector-analyze`.
- **Torch-optional design**: `import my_utils` and the whole
  `my_utils.profiling` surface work without torch installed; torch-dependent
  helpers are resolved lazily on first access.

### Deprecated

The following legacy shims emit `DeprecationWarning` on use in 0.1.x and will
be **removed in 0.3.0** (see docs/API_STABILITY.md for the policy):

- Flat legacy module aliases under `my_utils.profiling`
  (e.g. `my_utils.profiling.metrics_types`,
  `my_utils.profiling.capture_controller`, `my_utils.profiling.nsys_mfu`);
  use the names re-exported by `my_utils.profiling`, or the relocated
  submodules, instead. Full mapping: `_LEGACY_MODULE_ALIASES` in
  `my_utils/profiling/__init__.py`.
- `_LegacyNsysSqliteMetricsProvider`
  (`my_utils.profiling.metrics.metrics_providers`); use
  `my_utils.profiling.NsysSqliteMetricsProvider`.
- `NsysLaunchConfig.gpu_metrics_device` (use `gpu_metrics_devices`) and
  `NsysLaunchConfig.nic_metrics` (use `nic_metrics_mode`).
- Flat legacy module aliases at the top level of `my_utils`
  (e.g. `my_utils.utils`, `my_utils.logger`, `my_utils.pad`); use the
  relocated submodules (`my_utils.core.utils`, `my_utils.core.logger`,
  `my_utils.distributed.pad`, ...). Full mapping: `_LEGACY_MODULE_ALIASES`
  in `my_utils/__init__.py`.

[0.1.0]: https://github.com/woshipapa/my_utils/releases/tag/v0.1.0
