# Unified Performance Metrics System Design

## Design Goals

1. **Framework-agnostic** - the core system does not depend on any specific training framework
2. **Modular** - each component can be used independently or composed
3. **Extensible** - new data sources and analyzers are easy to add
4. **Lightweight** - near-zero overhead when disabled

## Core Architecture

```
+--------------------------------------------------------------+
|                    MetricsCollector                          |
|  - registers multiple MetricsProvider instances              |
|  - unified data format: MetricEvent (schema v1)              |
|  - optional per-event schema validation / normalization      |
|  - config-driven bootstrap via from_config()                 |
+--------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------+
|                   MetricsStore                               |
|  - in-memory ring buffer for fast reads                      |
|  - append-only JSONL file for persistence                    |
+--------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------+
|                  MetricsAnalyzer                             |
|  - rule-based bottleneck / memory / stability analysis       |
|  - workload profiles select the rule set                     |
|  - recommendation generation and overall scoring             |
+--------------------------------------------------------------+
                              |
                              v
+--------------------------------------------------------------+
|                   Output & Visualization                     |
|  - JSON / Markdown / HTML reports (MetricsReportRenderer)    |
|  - interactive HTML report (HTMLReportGenerator)             |
|  - Chrome trace export (chrome://tracing / Perfetto)         |
+--------------------------------------------------------------+
```

## 1. The MetricsProvider Protocol

Every data source implements the same protocol
(`my_utils/profiling/metrics/metrics_provider.py`):

```python
@runtime_checkable
class MetricsProvider(Protocol):
    """Provider interface for all profiling data sources."""

    provider_id: str

    def get_metrics(self) -> List[MetricEvent]:
        ...

    def start_collection(self) -> None:
        ...

    def stop_collection(self) -> None:
        ...

    def is_enabled(self) -> bool:
        ...

    def capabilities(self) -> ProviderCapabilities:
        ...
```

`capabilities()` lets the collector and analyzers reason about what a provider
can deliver without hard-coding provider types:

```python
@dataclass
class ProviderCapabilities:
    provider_type: str
    source_mode: str = "online"  # online | offline | hybrid
    metric_prefixes: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    supports_incremental: bool = True
    supports_step_scope: bool = False
    supports_rank_scope: bool = False
    notes: str = ""
```

A concrete helper base class, `BaseMetricsProvider`, supplies the boilerplate
(enable/disable flag, no-op `start_collection`/`stop_collection`, capability
accessor) so that most providers only implement `get_metrics()`.

Design rationale: a `Protocol` (instead of a mandatory ABC) keeps third-party
providers structurally compatible without importing anything from this package;
`BaseMetricsProvider` is offered purely as a convenience.

## 2. MetricEvent: The Unified Data Format

`my_utils/profiling/metrics/metrics_types.py` defines the canonical event.
Schema v1 added `schema_version` and `event_id` on top of the original sketch:

```python
PROFILE_SCHEMA_VERSION = "1.0"

@dataclass
class MetricEvent:
    timestamp: float                     # Unix timestamp (seconds)
    name: str                            # metric name, e.g. "latency.stage"
    value: float | int | str             # metric value
    unit: str = ""                       # ms, bytes, count, ...
    provider_id: str = ""                # which provider emitted the event
    tags: Dict[str, str] = field(default_factory=dict)
    # tag examples: {"rank": "0", "step": "100", "module": "transformer.layer.0"}
    attributes: Dict[str, Any] = field(default_factory=dict)  # free-form payload
    parent_id: Optional[str] = None      # optional hierarchy
    node_id: Optional[str] = None
    event_id: Optional[str] = None
    schema_version: str = PROFILE_SCHEMA_VERSION
```

`__post_init__` coerces every field to its canonical type (string tags, float
timestamp), and `to_dict()` / `from_dict()` give a stable JSON round-trip that
the JSONL store relies on.

### 2.1 Metric taxonomy

Metric names use a dotted hierarchy whose first segment is a canonical prefix
(`my_utils/profiling/metrics/metrics_taxonomy.py`):

| Prefix    | Meaning                                  |
|-----------|------------------------------------------|
| `latency` | execution time                           |
| `memory`  | memory usage / allocation                |
| `compute` | FLOPs, occupancy, throughput             |
| `comm`    | communication / NCCL / network           |
| `io`      | memcpy, storage/network bytes            |
| `calls`   | call counts                              |
| `perf`    | perf stat counters                       |

The taxonomy module also maintains `TOOL_METRIC_ALIASES`, which maps
tool-native metric names (torch profiler attributes, nsys table columns, ...)
to canonical `(name, unit)` pairs, so events from different tools stay
comparable.

### 2.2 Schema validation

`my_utils/profiling/metrics/metrics_schema.py` provides
`MetricSchemaValidator` with `validate(event)` and `normalize(event)`.
Validation is intentionally non-blocking by default: unknown prefixes and
non-recommended units are warnings unless `enforce_known_prefix` /
`enforce_recommended_units` are set. The collector calls `normalize()` on
every event and can optionally validate and drop invalid events (see below).

## 3. Providers for the Existing Tools

All built-in providers live in
`my_utils/profiling/metrics/metrics_providers.py` and extend
`BaseMetricsProvider`. Each keeps an internal cursor so repeated `collect()`
calls only emit new events (incremental reads).

### 3.1 MyTimerMetricsProvider

Wraps a `MyTimer` instance and converts its `records` list into events:

```python
class MyTimerMetricsProvider(BaseMetricsProvider):
    provider_id = "my_timer"

    def __init__(self, timer, *, include_cpu=True, include_cuda=True,
                 provider_id=None, enabled=True): ...
```

- Emits `latency.stage` events (unit `ms`) with tags
  `{"stage": ..., "rank": ..., "step": ..., "device": "cpu" | "cuda"}`.
- CPU and CUDA durations become separate events, distinguished by the
  `device` tag.
- Preserves the timer's `node_id` / `parent_id` hierarchy and records
  absolute start/end timestamps in `attributes`.

### 3.2 TorchProfilerMetricsProvider

Wraps a `torch.profiler.profile` object:

```python
class TorchProfilerMetricsProvider(BaseMetricsProvider):
    provider_id = "torch_profiler"

    def __init__(self, profiler, *, include_memory=True, include_flops=True,
                 provider_id=None, enabled=True): ...
```

Instead of one event per profiler event, it fans out into canonical metrics
per op (tags `{"op": ..., "device": ..., "step": ...}`):

- `latency.op.self_cpu` / `latency.op.total_cpu` /
  `latency.op.self_cuda` / `latency.op.total_cuda` (unit `us`)
- `memory.op.cpu` / `memory.op.cuda` / `memory.op.self_cpu` /
  `memory.op.self_cuda` (unit `bytes`, when `include_memory`)
- `compute.op.flops` (unit `flops`, when `include_flops`)

### 3.3 ModuleProfilerMetricsProvider

Wraps a `ModuleProfiler` (per-`nn.Module` hook profiler) and converts its
`summary()` DataFrame:

```python
class ModuleProfilerMetricsProvider(BaseMetricsProvider):
    provider_id = "module_profiler"

    def __init__(self, module_profiler, *, provider_id=None, enabled=True): ...
```

Emits `latency.module.mean` / `.median` / `.std` / `.total` (unit `ms`) and
`latency.module.share_percent` (unit `percent`) with tags
`{"module": ..., "run_count": ...}`. Because `summary()` is cumulative, the
provider deduplicates via a summary signature rather than a cursor.

### 3.4 Offline providers

Beyond the three online wrappers, the same protocol covers offline artifacts
produced by external tools:

| Class                       | Registry type      | Source                                |
|-----------------------------|--------------------|---------------------------------------|
| `TableCsvMetricsProvider`   | `table_csv`        | generic CSV tables                    |
| `NcuCsvMetricsProvider`     | `ncu_csv`          | Nsight Compute CSV export             |
| `NsysSqliteMetricsProvider` | `nsys_sqlite`      | Nsight Systems SQLite export          |
| `CProfileStatsProvider`     | `cprofile`         | cProfile stats dumps                  |
| `PerfStatTextProvider`      | `perf_stat`        | `perf stat` text output               |
| `DcgmCsvMetricsProvider`    | `dcgm_csv`         | DCGM field-value CSV                  |
| `NcclLogMetricsProvider`    | `nccl_log`         | NCCL debug logs                       |
| `RasJsonMetricsProvider`    | `ras_json`         | RAS / health-event JSON(L)            |

### 3.5 Provider registry

`my_utils/profiling/metrics/provider_registry.py` supplies
`MetricsProviderRegistry` plus `ProviderSpec` (`type` / `id` / `enabled` /
`params`). `register_builtin_providers()` wires up all the types above into
`DEFAULT_PROVIDER_REGISTRY`. Providers that need a live object (`my_timer`,
`torch_profiler`, `module_profiler`) receive it through a `provider_context`
mapping at creation time; file-based providers are constructed from `params`
alone. This is what makes config-driven bootstrap (section 8) possible.

## 4. MetricsCollector

`my_utils/profiling/pipeline/metrics_collector.py` is the orchestrator. The
real API is richer than the original sketch:

```python
class MetricsCollector:
    def __init__(self, output_dir="metrics", *, store=None, analyzer=None,
                 renderer=None, validator=None, validate_events=False,
                 drop_invalid_events=False, provider_registry=None,
                 enabled=True): ...

    # provider management
    def register_provider(self, provider: MetricsProvider) -> None: ...
    def unregister_provider(self, provider_id: str) -> None: ...
    def list_providers(self) -> List[str]: ...
    def provider_capabilities(self) -> Dict[str, Dict[str, Any]]: ...
    def register_providers_from_specs(self, provider_specs, *,
                                      provider_context=None,
                                      ignore_errors=False) -> None: ...

    # lifecycle
    def start(self) -> None: ...   # start_collection() on every provider
    def stop(self) -> None: ...

    # collection
    def collect(self, *, step: Optional[int] = None,
                tags: Optional[Dict[str, str]] = None) -> int: ...

    # analysis and export
    def get_events(self) -> List[MetricEvent]: ...
    def analyze(self, events=None) -> AnalysisReport: ...
    def export_report(self, *, fmt="json", output_path=None,
                      report=None) -> str: ...
    def export_chrome_trace(self, *, output_path=None, events=None,
                            config=None) -> str: ...
    def load_events_file(self, events_path: str) -> int: ...

    @classmethod
    def from_config(cls, config_path, *, provider_context=None,
                    provider_registry=None) -> "MetricsCollector": ...
```

Key behaviors:

- `collect()` returns the number of events written. When the collector is
  disabled it returns `0` immediately.
- The optional `step` and `tags` arguments are merged into every event's tags
  for that collection cycle.
- Every event is normalized (`MetricSchemaValidator.normalize`); when
  `validate_events=True` invalid events are counted and, with
  `drop_invalid_events=True`, discarded. Validation statistics and any
  provider bootstrap warnings are attached to
  `AnalysisReport.metadata["collector"]`.
- Provider failures are isolated: an exception in one provider (or one event)
  never aborts the collection cycle.
- `get_events()` prefers the in-memory buffer and transparently falls back to
  the on-disk JSONL file once the ring buffer has overflowed.

## 5. MetricsAnalyzer

`my_utils/profiling/analyzers/metrics_analyzer.py` replaced the original
hard-coded bottleneck routine with a rule engine:

```python
class MetricsAnalyzer:
    def __init__(self, *,
                 bottleneck_share_threshold: float = 0.10,
                 cv_threshold: float = 0.50,
                 memory_growth_bytes_per_step: float = 10 * 1024 * 1024,
                 workload_profile: str = "default",
                 enable_advanced_rules: bool = True,
                 rules: Optional[List[AnalysisRule]] = None) -> None: ...

    def list_rules(self) -> List[str]: ...
    def analyze(self, events: Iterable[MetricEvent]) -> AnalysisReport: ...
```

- Each `AnalysisRule` inspects the event list and may return a `Finding`
  (`finding_type`, `severity`, `title`, `description`, `data`) plus
  recommendations.
- `workload_profile` selects which rule set to build (e.g. training vs.
  inference oriented profiles); `resolve_workload_profile` /
  `build_rules_for_workload` live in `analyzers/workload_profiles.py`.
- The report gets a conservative `overall_score` starting at 100 with
  per-severity deductions, a summary (event/provider/step/rank counts), and
  severity counts in metadata.

The detailed rule catalog and scoring design are documented in
[AUTO_ANALYZER_DESIGN.md](./AUTO_ANALYZER_DESIGN.md).

## 6. Report Generation and Visualization

Two complementary paths exist:

1. **`MetricsCollector.export_report(fmt=...)`** delegates to
   `MetricsReportRenderer` (`my_utils/profiling/output/metrics_report.py`),
   which renders an `AnalysisReport` as `json`, `md`/`markdown`, or `html`.
   The default output path is `<output_dir>/analysis_report.<ext>`.
2. **`HTMLReportGenerator`**
   (`my_utils/profiling/visualization/html_generator.py`) builds a full
   interactive report from a report *and* the raw events:

   ```python
   from my_utils.profiling.visualization import HTMLReportGenerator

   html = HTMLReportGenerator().generate(report, events, output_path="report.html")
   ```

   It composes `ChartRenderer` (Chart.js or Plotly backend, selected by
   `create_chart_renderer()`), `DataTransformer`, and `LayoutBuilder` into
   sections: summary, key metrics, trend charts, bottleneck charts, memory
   charts, findings, recommendations, and detail tables.

Additionally, `MetricsCollector.export_chrome_trace()` writes the event
stream as a Chrome trace JSON (`my_utils/profiling/output/metrics_trace.py`)
viewable in `chrome://tracing` or Perfetto.

The chart/layout architecture is covered in
[VISUALIZATION_DESIGN.md](./VISUALIZATION_DESIGN.md).

> Note: an earlier revision of this document proposed a TensorBoard plugin as
> an additional visualization backend. It was never built and is not planned;
> the HTML report and Chrome trace export cover those use cases.

## 7. Usage Example

```python
from my_utils import MyTimer, ModuleProfiler
from my_utils.profiling import (
    MetricsCollector,
    MyTimerMetricsProvider,
    ModuleProfilerMetricsProvider,
)

# 1. Initialize
timer = MyTimer(use_cuda=True, tag="train")
collector = MetricsCollector(output_dir="./metrics_logs")

# 2. Register providers
collector.register_provider(MyTimerMetricsProvider(timer))

# 3. Training loop
model = MyModel().cuda()
with ModuleProfiler(model) as module_prof:
    collector.register_provider(ModuleProfilerMetricsProvider(module_prof))

    for step in range(1000):
        timer.set_step(step)

        timer.start("forward")
        output = model(batch)
        timer.stop("forward")

        timer.start("backward")
        loss.backward()
        timer.stop("backward")

        timer.next_iteration()

        # Collect every 100 steps
        if step % 100 == 0:
            collector.collect(step=step)

# 4. Generate reports
report = collector.analyze()
html_path = collector.export_report(fmt="html")
trace_path = collector.export_chrome_trace()
print(f"Report written to: {html_path}")
```

## 8. Config-Driven Setup

`MetricsCollector.from_config()` accepts a YAML (requires PyYAML) or JSON
file with four top-level keys — `collector`, `analysis`, `schema`, and
`providers`:

```yaml
# metrics_config.yaml
collector:
  output_dir: "./metrics"
  enabled: true
  validate_events: true
  drop_invalid_events: false
  ignore_provider_errors: false

analysis:
  workload_profile: "default"
  bottleneck_threshold: 0.10
  cv_threshold: 0.50
  memory_growth_bytes_per_step: 10485760
  enable_advanced_rules: true

schema:
  strict: false
  enforce_known_prefix: false
  enforce_recommended_units: false

providers:
  - type: table_csv
    id: ext_csv
    enabled: true
    params:
      csv_path: /path/to/external.csv
      value_column: value
      name_column: name
      tag_columns: [step]
      unit: ms
      event_name_prefix: latency.external
  - type: my_timer
    id: train_timer
    enabled: true
```

```python
collector = MetricsCollector.from_config(
    "metrics_config.yaml",
    provider_context={"my_timer": timer},  # live objects for online providers
)
```

Each entry under `providers` is a `ProviderSpec` resolved through the
provider registry (section 3.5). Online provider types (`my_timer`,
`torch_profiler`, `module_profiler`) pull their wrapped object from
`provider_context`; offline types are constructed from `params` alone. With
`collector.ignore_provider_errors: true`, a provider that fails to construct
is recorded as a bootstrap warning instead of raising.

> There is no environment-variable configuration layer. Enablement is always
> explicit: the `enabled` flag in the config/constructor, or the
> `profile.metrics.enabled` block when driven by `ProfileManager` (below).

## 9. Integration with ProfileManager

`ProfileManager` (`my_utils/profiling/runtime/ProfileManager.py`) owns an
optional collector, enabled from its profile config rather than environment
variables:

```yaml
profile:
  metrics:
    enabled: true
    output_dir: "metrics"
```

- When enabled, `ProfileManager` constructs a `MetricsCollector` at init
  (failures are logged as warnings, never fatal).
- At each capture-arm point it calls
  `collector.collect(step=it, tags={"phase": "capture_arm"})`, so metrics
  snapshots line up with profiling capture windows.
- `attach_metrics_provider(provider)` registers extra providers after init.
- `export_metrics_report(fmt="html", output_path=None)` proxies to
  `collector.export_report()`.

## 10. Framework Adapters

Framework adapters shipped as a separate, optional layer in
`my_utils/profiling/adapters/` (Megatron, DeepSpeed, HuggingFace, PyTorch,
TorchTitan, veRL, slime, ROLL, SGLang, vLLM). The design evolved away from
this document's original ABC sketch (`auto_setup()` / train-loop hooks) to a
declarative model: each `FrameworkAdapter` implements `detect(context)`,
`build_provider_specs(context)`, and `build_runtime_tags(context)`, and
`FrameworkAdapterRegistry.auto_setup_collector(collector, context=...)`
registers the resulting provider specs on a collector. See
[FRAMEWORK_ADAPTERS_DESIGN.md](./FRAMEWORK_ADAPTERS_DESIGN.md) for the full
design and per-framework details.

## 11. Performance Considerations

1. **Cheap when disabled** - `collect()` short-circuits to `return 0`; a
   disabled provider is skipped via `is_enabled()`.
2. **Incremental reads** - online providers keep cursors and offline
   providers track file positions/mtimes, so each `collect()` only processes
   new data.
3. **Bounded memory** - `MetricsStore` keeps events in a fixed-size ring
   buffer (`deque(maxlen=max_memory_events)`, default 500k events) and flags
   overflow so readers know to fall back to disk.
4. **Simple, durable persistence** - writes are synchronous appends to a
   JSONL file under a lock. There is no background writer thread and no
   Redis/Prometheus backend; JSONL keeps the store dependency-free and
   crash-tolerant (partial lines are skipped on read).
5. **Failure isolation** - provider and per-event exceptions are swallowed by
   the collector so profiling can never take down training.

## 12. Test Strategy

The system is covered by the suite under `tests/profiling/`, notably:

- `test_collector_from_config.py` - config-driven bootstrap, collection,
  validation stats, report metadata
- `test_analysis_engine.py` / `test_analyzer_rules.py` - analyzer rules and
  scoring
- `test_metrics_trace.py` - Chrome trace export
- `test_framework_adapters.py` - adapter detection and provider spec wiring

A representative end-to-end test:

```python
def test_collector_with_csv_provider(tmp_path):
    csv_path = tmp_path / "external.csv"
    csv_path.write_text("name,value,step\nfoo,1.0,0\nbar,2.0,1\n")

    collector = MetricsCollector(output_dir=str(tmp_path / "out"))
    collector.register_provider(
        TableCsvMetricsProvider(
            str(csv_path),
            value_column="value",
            name_column="name",
            tag_columns=["step"],
            unit="ms",
            event_name_prefix="latency.external",
        )
    )

    assert collector.collect(step=3) == 2
    events = collector.get_events()
    assert all(e.provider_id == "csv_table" for e in events)

    report = collector.analyze(events)
    assert report.summary["total_events"] == 2
```

## Summary

The shipped system meets the original goals:

1. **Framework-agnostic** - the core is built on a `Protocol` and a provider
   registry; framework knowledge lives only in the optional adapters layer
2. **Modular** - providers, store, analyzer, renderers, and trace export are
   independent components composed by the collector
3. **Extensible** - a new data source is one `MetricsProvider`
   implementation plus (optionally) a registry factory for config-driven use
4. **Lightweight** - disabled paths short-circuit, reads are incremental,
   and memory is bounded

All phases of the original implementation plan (core schema, collector,
providers, analyzer, reporting, framework adapters) have shipped; see
[ROADMAP.md](./ROADMAP.md) and
[IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) for status history.
