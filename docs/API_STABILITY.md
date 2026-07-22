# API Stability

This document declares the public API surface of `my_utils` as of version
0.1.0, and the deprecation policy that governs changes to it.

## Public API surface

The public, supported API consists of exactly two things:

1. **The names re-exported by `my_utils.profiling`** (i.e. everything in
   `my_utils/profiling/__init__.py`'s `__all__`), listed below.
2. **The 16 console-script CLIs** installed by the package, listed below.

Everything else — submodule paths such as `my_utils.profiling.sources.*`,
`my_utils.profiling.metrics.*`, `my_utils.profiling.runtime.*`, and the
`my_utils.core`, `my_utils.hooks`, `my_utils.memory`, `my_utils.artifacts`,
`my_utils.distributed`, `my_utils.tracing`, and `my_utils.legacy_profilers`
subpackages — is **internal/incidental** and may change or disappear in any
release without notice. Import public names from `my_utils.profiling` (or
`my_utils`), not from the submodules that happen to define them.

### Public names (`my_utils.profiling`)

Runtime capture and configuration:

- `CaptureBackend`, `NoOpBackend`, `CudaProfilerBackend`
- `CaptureController`, `HookEvent`, `extract_meta_from_call`
- `ProfileManager`
- `TorchProfilerConfig`, `NsysProfilerConfig`, `ProfilingEnvConfig`, `NsysLaunchConfig`
- `create_nsys_capture_backend`, `apply_profiling_environment`, `build_nsys_launch_prefix`
- `get_profiling_templates_dir`, `get_profiling_template_path`

Metrics model, schema, and store:

- `PROFILE_SCHEMA_VERSION`, `MetricEvent`, `Finding`, `Bottleneck`, `AnalysisReport`
- `MetricsProvider`, `ProviderCapabilities`, `BaseMetricsProvider`
- `MetricsStore`
- `MetricSchemaValidator`, `EventValidationResult`, `CANONICAL_UNITS`,
  `normalize_event`, `validate_event`
- `CANONICAL_METRIC_PREFIXES`, `TOOL_METRIC_ALIASES`, `normalize_external_metric`

Metrics providers and registry:

- `MyTimerMetricsProvider`, `TorchProfilerMetricsProvider`,
  `ModuleProfilerMetricsProvider`, `TableCsvMetricsProvider`,
  `NcuCsvMetricsProvider`, `NsysSqliteGlobMetricsProvider`,
  `NsysSqliteMetricsProvider`, `CProfileStatsProvider`, `PerfStatTextProvider`,
  `DcgmCsvMetricsProvider`, `NcclLogMetricsProvider`, `RasJsonMetricsProvider`
- `ProviderSpec`, `MetricsProviderRegistry`, `register_builtin_providers`,
  `DEFAULT_PROVIDER_REGISTRY`

Analysis, reporting, and diffing:

- `MetricsAnalyzer`, `MetricsCollector`, `MetricsReportRenderer`, `AnalysisRule`
- `align_stage_latency`, `analyze_rank_skew`
- `WORKLOAD_PROFILES`, `WorkloadProfile`, `resolve_workload_profile`,
  `list_workload_profiles`, `build_rules_for_workload`
- `ReportDiff`, `compare_reports`, `write_diff`
- `ChromeTraceExportConfig`, `estimate_rank_time_offsets`,
  `metric_events_to_chrome_trace`, `write_chrome_trace`,
  `export_events_file_to_chrome_trace`

Framework adapters:

- `FrameworkAdapter`, `FrameworkAdapterRegistry`, `DEFAULT_ADAPTER_REGISTRY`,
  `build_default_adapter_registry`
- `PyTorchAdapter`, `HuggingFaceAdapter`, `DeepSpeedAdapter`, `MegatronAdapter`,
  `TorchTitanAdapter`, `VerlAdapter`, `SlimeAdapter`, `RollAdapter`,
  `SGLangAdapter`, `VLLMAdapter`

Nsight Systems (nsys) analysis:

- `NsysVersionInfo`, `NsightSchema`, `detect_nsys_version`
- `SqlSkillParam`, `SqlSkill`, `NsysSqlSkillEngine`
- `detect_iterations`
- `compute_mfu_single`, `compute_mfu_compare`, `infer_peak_tflops`
- `collect_kernel_rows`, `export_kernels_flat`
- `analyze_nsys_sqlite`, `analyze_to_markdown`
- `diff_nsys_sqlite`, `diff_to_markdown`
- `export_timeline_html`

NCCL Inspector analysis:

- `NcclInspectorSkillEngine`, `analyze_nccl_inspector`,
  `analyze_nccl_inspector_to_markdown`, `load_nccl_inspector_events`,
  `load_nccl_prometheus_metrics`

Visualization (optional extra; only exported when its dependencies are
installed):

- `VISUALIZATION_AVAILABLE` (always exported)
- `ChartConfig`, `ChartRenderer`, `ChartJsRenderer`, `PlotlyRenderer`,
  `create_chart_renderer`, `DataTransformer`, `LayoutBuilder`,
  `HTMLReportGenerator`, `QuickReportGenerator`, `VizAnalysisReport`

### Console-script CLIs (16)

Declared in `pyproject.toml` `[project.scripts]`:

| Script | Purpose |
| --- | --- |
| `myutils-profile` | Main profiling CLI |
| `nsys-panel` | Nsight Systems report panel |
| `nsys-sql-skill` | Run SQL skills against nsys SQLite exports |
| `nsys-export` | Export nsys data |
| `nsys-analyze` | One-shot nsys diagnostic analysis |
| `nsys-diff` | Diff two nsys captures |
| `nsys-module-kernel-compare` | Module/kernel-level comparison |
| `nsys-timeline-html` | Standalone HTML timeline export |
| `nsys-iter-overlap` | Per-iteration overlap analysis |
| `nsys-iter-outliers` | Per-iteration outlier detection |
| `ncu-csv-skill` | Run skills against Nsight Compute CSV exports |
| `ncu-csv-analyze` | Nsight Compute CSV analysis |
| `ncu-report-skill` | Run skills against `.ncu-rep` reports |
| `ncu-report-analyze` | `.ncu-rep` report analysis |
| `nccl-inspector-skill` | Run skills against NCCL Inspector output |
| `nccl-inspector-analyze` | NCCL Inspector analysis |

CLI command names and their documented flags are stable; their human-readable
text output is not a stable interface (use the machine-readable export formats
for automation).

## Stability guarantees

For the public surface above, within the 0.x series:

- Names are not removed and signatures are not broken in patch releases
  (0.1.x).
- Removals or breaking changes happen only at a minor version bump, and only
  after at least one minor release in which the old form emits a
  `DeprecationWarning`.

No guarantees are made for internal modules, private names (leading
underscore), or anything not listed above.

## Deprecation policy

- **Legacy shims warn in 0.1.x and are removed in 0.3.0.**
- Deprecated entry points emit `DeprecationWarning` on *use* (import of a
  legacy module path, access of a legacy attribute, instantiation of a legacy
  class, or setting of a legacy config field) — never on plain
  `import my_utils.profiling` or on use of the modern public API, which are
  guaranteed warning-free (enforced by `tests/profiling/test_deprecations.py`).
- Each warning names the replacement and the removal version.

### Currently deprecated (removal in 0.3.0)

1. **Flat legacy module aliases under `my_utils.profiling`** — e.g.
   `my_utils.profiling.metrics_types` →
   `my_utils.profiling.metrics.metrics_types`,
   `my_utils.profiling.nsys_mfu` → `my_utils.profiling.sources.nsys_mfu`.
   The full mapping is `_LEGACY_MODULE_ALIASES` in
   `my_utils/profiling/__init__.py`. Import the names you need from
   `my_utils.profiling` directly instead.
2. **`_LegacyNsysSqliteMetricsProvider`**
   (`my_utils.profiling.metrics.metrics_providers`) — use the public
   `NsysSqliteMetricsProvider` re-exported by `my_utils.profiling`.
3. **`NsysLaunchConfig` legacy fields** — `gpu_metrics_device` (use
   `gpu_metrics_devices`) and `nic_metrics` (use `nic_metrics_mode`).
4. **Flat legacy module aliases at the top level of `my_utils`** — e.g.
   `my_utils.utils` → `my_utils.core.utils`,
   `my_utils.pad` → `my_utils.distributed.pad`. The full mapping is
   `_LEGACY_MODULE_ALIASES` in `my_utils/__init__.py`.
