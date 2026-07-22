# Automated Performance Analyzer Design

This document describes the design of the automated analysis layer that ships in
`my_utils/profiling/analyzers/`. It consumes the unified metric events defined in
[./UNIFIED_METRICS_DESIGN.md](./UNIFIED_METRICS_DESIGN.md) and produces findings,
recommendations, and an overall score.

> An earlier draft of this document proposed a set of per-domain analyzer classes
> (`TimeAnalyzer`, `MemoryAnalyzer`, `CommunicationAnalyzer`) coordinated by an
> `AnalyzerPipeline`, plus a standalone `RecommendationGenerator`. That design was
> superseded by a single `MetricsAnalyzer` driving a modular rule engine
> (`analyzers/analysis_rules.py`); the sections below describe the system as built.

## Design Goals

1. **Automatic bottleneck identification** — find hotspots and problems without a
   human reading raw traces.
2. **Explainable recommendations** — every finding carries the data that supports
   it, and every recommendation is tied to a finding.
3. **Multi-dimensional analysis** — latency, memory, communication, GPU
   utilization, pipeline structure, and the data pipeline are covered by
   independent rules that share one event schema.
4. **Trustworthy conclusions** — data-quality checks, measurement-context checks,
   and evidence provenance guard against confidently wrong output. A refusal is
   preferred over a misleading number.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ MetricsCollector (pipeline/metrics_collector.py)               │
│   providers → MetricEvent stream → MetricsStore                │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ MetricsAnalyzer (analyzers/metrics_analyzer.py)                │
│   workload profile → ordered list of AnalysisRule instances    │
│   each rule: apply(events, context) -> Finding | None          │
│              recommendations(finding) -> list[str]             │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ AnalysisReport (metrics/metrics_types.py)                      │
│   summary + findings + recommendations + overall_score         │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ MetricsReportRenderer (output/metrics_report.py)               │
│   json / markdown / html                                       │
└────────────────────────────────────────────────────────────────┘
```

The rule engine is supported by a set of standalone analysis modules that rules
and report generation draw on:

| Module | Responsibility |
| --- | --- |
| `analyzers/analysis_rules.py` | Rule definitions: one class per detection concern |
| `analyzers/workload_profiles.py` | Per-workload rule selection and threshold overrides |
| `analyzers/distributed_alignment.py` | Cross-rank stage alignment and skew analysis |
| `analyzers/triage.py` | Top-down decision tree: where does a training step's time go |
| `analyzers/trace_quality.py` | Trace-validity checks that can block unsafe conclusions |
| `analyzers/evidence.py` | Evidence fusion with provenance ranking for kernel attribution |
| `analyzers/measurement_context.py` | Collection-mode compatibility (cold vs warm cache, etc.) |
| `analyzers/axes.py` | Canonical performance axes and coverage accounting |
| `analyzers/nccl_bandwidth.py` | Collective bus-bandwidth analysis and straggler detection |

## 1. Core Data Structures

All analyzer inputs and outputs are plain dataclasses defined in
`my_utils/profiling/metrics/metrics_types.py`.

```python
@dataclass
class MetricEvent:
    """Canonical metric event across all profiling tools."""
    timestamp: float
    name: str                 # namespaced, e.g. "latency.stage", "memory.gpu.allocated"
    value: Union[float, int, str]
    unit: str = ""            # "ms", "bytes", ...
    provider_id: str = ""
    tags: Dict[str, str] = field(default_factory=dict)       # "step", "rank", "stage", ...
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    node_id: Optional[str] = None
    event_id: Optional[str] = None
    schema_version: str = PROFILE_SCHEMA_VERSION


@dataclass
class Bottleneck:
    name: str
    share_percent: float
    avg_value: float
    unit: str
    sample_count: int


@dataclass
class Finding:
    finding_type: str          # "bottleneck", "memory", "communication", ...
    severity: str              # "critical" | "high" | "warning" | "info" | "low"
    title: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)   # supporting evidence
    finding_id: Optional[str] = None                     # usually the rule_id


@dataclass
class AnalysisReport:
    generated_at: str
    summary: Dict[str, Any]
    findings: List[Finding]
    recommendations: List[str]
    metadata: Dict[str, Any]
    schema_version: str = PROFILE_SCHEMA_VERSION
    overall_score: Optional[float] = None    # 0-100
```

Design notes:

- **Severity is a lowercase string**, not an enum, so findings survive JSON
  round-trips (`Finding.from_dict` / `AnalysisReport.from_dict`) without import
  coupling. The scoring model (section 8) is the single place that interprets it.
- **Evidence lives in `Finding.data`.** Each rule stores the exact numbers it
  based its decision on (thresholds, observed ratios, per-rank breakdowns), so a
  report reader can audit the conclusion.
- **Recommendations are plain strings**, produced per-rule and deduplicated by
  `AnalysisReport.add_recommendations`. The earlier draft's `Recommendation`
  dataclass (priority / estimated impact / effort fields) was not built; ordering
  comes from rule order and finding severity instead.

## 2. Rule Interface

Every detection concern is one class implementing `AnalysisRule`
(`analyzers/analysis_rules.py`):

```python
class AnalysisRule(ABC):
    rule_id = "rule"

    @abstractmethod
    def apply(self, events: List[MetricEvent], context: Dict[str, object]) -> Optional[Finding]:
        ...

    def recommendations(self, finding: Finding) -> List[str]:
        return []
```

Contract:

- A rule returns **at most one `Finding`** per run; `None` means "nothing to
  report". Rules that also want to report a healthy state (e.g. memory) may emit
  an `info` finding.
- `context` carries the report summary, metadata, and the active workload
  profile, so rules can adapt without re-scanning events.
- Thresholds are constructor arguments with defaults, overridable per workload
  profile (section 7) or per `MetricsAnalyzer` instance.

Shared helpers keep rules consistent:

- `_to_ms(value, unit)` normalizes `ns/us/ms/s` to milliseconds; events with
  unknown time units are skipped rather than misread.
- `_group_key(event)` groups events by the first available structural tag
  (`stage`, `module`, `op`, `kernel`, `func`, `name`), falling back to the metric
  name. This is what turns a flat event stream into per-component statistics.
- `_to_ratio(value)` accepts both fractional (0.73) and percent (73) utilization
  encodings.

## 3. Latency Rules

### `LatencyBottleneckRule` (`rule_id: latency_bottleneck`)

Algorithm:

1. Select events whose name starts with `latency.` and normalize values to ms.
2. Group by `_group_key` and sum per group.
3. Any group whose share of total latency exceeds `share_threshold`
   (default **0.10**) becomes a `Bottleneck` record.
4. Emit one `high` finding listing the top 30 bottleneck groups sorted by share,
   with per-group average and sample count in `Finding.data`.

### `LatencyVarianceRule` (`rule_id: latency_variance`)

Detects unstable stages via the coefficient of variation (CV = population
stddev / mean) per latency group. A group is reported when it has at least
**5 samples** and CV ≥ `cv_threshold` (default **0.50**). High variance usually
points at data-dependent control flow, host synchronization, or interference —
the recommendations steer toward those.

### `LatencyOutlierRule` (`rule_id: latency_outlier`)

Z-score based spike detection per latency group: with at least **8 samples**,
values above `mean + z_threshold * stddev` (default z = **3.0**) are outliers.
The finding reports outlier count, worst value, and the cutoff per group.

The earlier draft's per-kernel efficiency analysis (FLOPs-per-cycle heuristics)
was superseded by the roofline rule (section 6) and the evidence-fusion module
(section 9), which use hardware counters instead of kernel-name heuristics.

## 4. Memory Rule

### `MemoryGrowthRule` (`rule_id: memory_growth`)

Detects monotonic memory growth across training steps — the pattern behind
leaks and unbounded caches.

Algorithm:

1. Select events with a `memory.` name prefix or `unit == "bytes"`.
2. Track the peak value per group, and build a `(step, value)` series for
   events carrying a `step` tag.
3. For each series with ≥ 2 points, compute the endpoint slope
   `(last - first) / (last_step - first_step)`.
4. Groups whose slope exceeds `growth_bytes_per_step` (default
   **10 MiB/step**) are reported as growth items.

The rule always emits a finding: `warning` with growth items when growth is
detected, otherwise an `info` finding carrying the per-group peaks — so "memory
was examined and looks flat" is distinguishable from "memory was never measured".

Implementation note: the endpoint slope deliberately replaces the earlier
draft's linear regression + R² fit. Step-tagged memory samples are sparse and
allocator caching makes intermediate points noisy; the endpoint slope over the
full window answers the question that matters (is the trend large and
sustained?) without pretending to more statistical precision than the data has.

Never built from the earlier draft (no current plans — see
[./ROADMAP.md](./ROADMAP.md)):

- Allocator fragmentation detection (allocated vs reserved ratio).
- OOM-risk prediction against `torch.cuda` device capacity. Peak usage per group
  is available in the `memory_growth` finding data for manual headroom checks.

## 5. Communication Rules

### `CommunicationImbalanceRule` (`rule_id: comm_imbalance`)

Selects events with a `comm.` prefix or `nccl` in the name that carry a `rank`
tag, computes the mean latency per rank, and reports when
`max(rank_mean) / min(rank_mean)` ≥ `imbalance_ratio_threshold` (default
**1.25**). The per-rank means are included in the finding for straggler
identification.

### `CommunicationHealthRule` (`rule_id: comm_health`)

A composite health score over communication telemetry (NCCL latency events,
`busbw`/`algbw` bandwidth samples, and NCCL/RAS log signals surfaced as events).
Starting from 100, it deducts:

| Signal | Threshold (default) | Deduction |
| --- | --- | --- |
| Tail-latency jitter `p95/p50` | ≥ 2.0 | −20 |
| Cross-rank mean-latency ratio | ≥ 1.30 | −20 |
| Median bus bandwidth | < 20.0 GB/s | −15 |
| NCCL/RAS warning-or-error signals | any | −5 each, capped at −40 |

Severity: `high` when any error signals are present or the score drops below
50; `warning` below 70; `info` otherwise. The finding data carries p50/p95,
jitter ratio, per-rank means, median busbw, and the individual issue strings.

### Bus-bandwidth analysis (`analyzers/nccl_bandwidth.py`)

Collective duration alone cannot distinguish a slow link from a late rank.
`analyze_collective` converts measured algorithmic bandwidth
(`algbw = message_bytes / time`) to bus bandwidth
(`busbw = algbw * factor(n)`, nccl-tests convention) so it can be compared
against interconnect ceilings, and `detect_straggler` /
`detect_straggler_from_traces` attribute stretched collectives to late-arriving
ranks (including from flight-recorder dumps via
`arrivals_from_flight_recorder`). The `comm_health` busbw threshold builds on
this normalization.

The earlier draft's compute/communication overlap analysis is covered by the
triage decision tree (section 9), which classifies exposed communication time
directly.

## 6. Pipeline, Data, GPU, Roofline, and Consistency Rules

### `PipelineBubbleRule` (`rule_id: pipeline_bubble`)

Over `latency.stage*` events, sums time in stages whose `stage` tag contains
`idle` / `bubble` / `wait` / `stall`, and reports when the bubble share of total
stage latency ≥ `bubble_ratio_threshold` (default **0.15**).

### `DataloaderStallRule` (`rule_id: dataloader_stall`)

Compares dataloader latency (name or `stage` tag containing `dataloader`)
against step latency (`latency.step*` or `stage` in `step`/`iteration`); reports
when the ratio ≥ `stall_ratio_threshold` (default **0.20**).

### `GpuUtilizationThroughputRule` (`rule_id: gpu_utilization`)

Averages `compute.gpu.sm.active` / `compute.gpu.sm.utilization` /
`compute.gpu.utilization` samples (accepting fraction or percent encodings) and
reports an `info` finding when the mean is below `utilization_threshold`
(default **0.40**), attaching mean throughput
(`compute.throughput.samples_per_sec` / `tokens_per_sec`) when available.

### `DistributedSkewRule` (`rule_id: distributed_skew`)

Delegates to `analyze_rank_skew` (`analyzers/distributed_alignment.py`), which
aligns per-stage latency across ranks and flags stages whose worst-to-best rank
ratio exceeds `skew_ratio_threshold` (default **1.20**). Unlike
`comm_imbalance`, this covers all stage latency, not just communication.

### `RooflineGapRule` (`rule_id: roofline_gap`)

Compares average compute-side utilization (SM active / tensor active) against
memory-side utilization (DRAM / HBM / bandwidth ratios). A gap
`memory_util − compute_util` ≥ `gap_threshold` (default **0.20**) yields a
`warning` (or `high` at ≥ 0.35) with a `memory_bound` / `compute_bound` hint.
When FLOPs and byte counters are both present it also reports arithmetic
intensity (FLOPs/byte).

### `CrossLayerConsistencyRule` (`rule_id: cross_layer_consistency`)

An observability meta-rule: it checks whether the collected data is internally
joinable rather than whether the workload is fast.

- **Step coverage**: the share of `latency.step` time explained by
  component-layer events (kernel / comm / sync / dataloader / python) must
  average ≥ `min_step_coverage` (default **0.40**).
- **Tag alignment**: at least `min_tag_alignment_ratio` (default **0.60**) of
  step-tagged events must be attributable to a component layer.
- **Rank symmetry**: communication events carrying `rank` tags while compute
  events do not means cross-rank attribution is incomplete.

## 7. Workload Profiles

Different workloads warrant different rule sets and KPIs.
`analyzers/workload_profiles.py` defines them declaratively:

```python
@dataclass
class WorkloadProfile:
    name: str
    description: str
    rule_ids: List[str]
    kpi_metrics: List[str] = field(default_factory=list)
    threshold_overrides: Dict[str, float] = field(default_factory=dict)
```

| Profile | Rules | Focus |
| --- | --- | --- |
| `default` | latency_bottleneck, memory_growth, latency_variance, latency_outlier | Generic mixed pipelines |
| `pretrain` | all of the above + comm_imbalance, comm_health, pipeline_bubble, distributed_skew, gpu_utilization, roofline_gap, cross_layer_consistency | Large-scale distributed training |
| `finetune` | latency_bottleneck, memory_growth, latency_variance, dataloader_stall, gpu_utilization | Stability and throughput balance |
| `inference` | latency_bottleneck, latency_variance, latency_outlier, gpu_utilization | Tail latency and utilization |
| `data_pipeline` | dataloader_stall, latency_variance, memory_growth | Ingestion and preprocessing |

`build_rules_for_workload(name, thresholds=..., enable_advanced_rules=...)`
instantiates the rules, merging profile-level `threshold_overrides` with
caller-supplied thresholds (caller wins). Threshold keys map one-to-one to rule
constructor arguments, e.g. `bottleneck_threshold`, `cv_threshold`,
`memory_growth_bytes_per_step`, `outlier_z_threshold`,
`comm_imbalance_ratio_threshold`, `pipeline_bubble_ratio_threshold`,
`dataloader_stall_ratio_threshold`, `gpu_utilization_threshold`,
`distributed_skew_ratio_threshold`, `comm_health_p95_ratio_threshold`,
`comm_health_rank_imbalance_threshold`, `comm_health_min_busbw_gbps`,
`roofline_gap_threshold`, `cross_layer_min_step_coverage`,
`cross_layer_min_tag_alignment_ratio`.

Setting `enable_advanced_rules=False` restricts any profile to the four basic
rules (bottleneck, memory, variance, outlier), which keeps analysis cheap and
dependency-free for smoke runs.

## 8. Recommendations and Scoring

**Recommendations** come from the rules themselves: after a rule produces a
finding, `MetricsAnalyzer` calls `rule.recommendations(finding)` and appends the
returned strings to the report, deduplicated by exact text. Because each rule
owns its advice, the guidance stays next to the detection logic it explains.
(The earlier draft's standalone `RecommendationGenerator` with priority /
impact / effort scoring was superseded by this per-rule mechanism.)

**Overall score** (`MetricsAnalyzer._score`) starts at 100 and deducts per
finding by severity:

| Severity | Penalty |
| --- | --- |
| critical | 35 |
| high | 20 |
| warning | 10 |
| info | 3 |
| low | 5 |
| (unknown) | 5 |

The score floors at 0. An empty event stream yields a single `info` finding
("No Metrics") and a score of 100. The report metadata also records the active
workload profile, the executed rule ids, and a severity histogram
(`severity_counts`) so consumers can weight scores themselves.

## 9. Trust Layer: Guarding the Conclusions

Rule output is only as good as the data underneath it. Four modules exist to
keep the analyzer from being confidently wrong; report generation and advanced
workflows consume them alongside the rule engine.

- **Top-down triage** (`triage.py`) — `triage_step` runs the ordered decision
  tree an expert follows (GPU idle? communication exposed? transfers on the
  critical path? otherwise kernel-bound) and returns a single
  `TriageVerdict`, so a report can lead with one attribution instead of a flat
  findings list. Thresholds are calibrated defaults and overridable via
  `TriageThresholds`; the verdict requires multiple signals to cross before
  declaring a run host-bound.
- **Trace-validity checks** (`trace_quality.py`) — `assess_trace_quality` and
  the individual `check_*` functions detect silent data corruptions before
  analysis: autotuning burned into the first iterations, CUDA-graph launches
  destroying per-kernel attribution, non-unique compiled-kernel names, GPU
  metric sampler gaps, incomplete rank file sets, profiler overhead, clock
  misalignment, and more. Each returns a `QualityIssue` with a `blocks` flag —
  `True` means the affected conclusion should be refused, not caveated.
- **Evidence fusion** (`evidence.py`) — kernel symbols routinely misrepresent
  what ran (NCCL launch stubs hard-code algorithm tokens; DSL-generated names
  carry no shape info; CUTLASS symbols may be plain `torch.matmul` via
  cuBLASLt). Every attribution is a `Claim` with a `Provenance` rank (hardware
  counters > NVTX/source > launch config > symbol name); `fuse_claims` resolves
  conflicts by provenance and surfaces contradictions as findings in their own
  right.
- **Measurement context** (`measurement_context.py`) — records how a number was
  collected (cold vs warm cache, serialized replay, clock/thermal state) and
  `compare_measurements` refuses cross-mode comparisons (e.g. an Nsight Compute
  cold-cache duration vs a warm wall-clock duration) instead of reporting the
  delta as a regression.
- **Axis coverage** (`axes.py`) — maps every finding category (ours and the
  external tools') onto a canonical set of performance axes, so a report can
  state which axes were *examined* — not just which had problems — and which
  collection flags would close the gaps.

## 10. Pipeline Integration

`MetricsCollector` (`pipeline/metrics_collector.py`) owns the end-to-end flow:
provider registration, event collection and validation, storage, analysis, and
report export.

```python
collector = MetricsCollector(output_dir="metrics", analyzer=MetricsAnalyzer(...))
collector.register_provider(provider)          # any MetricsProvider
collector.start()
collector.collect(step=i)                      # tags events with the step
collector.stop()
report = collector.analyze()                   # AnalysisReport
collector.export_report(fmt="markdown", report=report)   # json | markdown | html
```

- `analyze()` runs the configured `MetricsAnalyzer` over the stored events and
  attaches collector metadata (registered providers, their capabilities,
  validation statistics, bootstrap warnings) to `report.metadata["collector"]`.
- `MetricsCollector.from_config(path)` builds the whole stack from a JSON/YAML
  file; the `analysis` section maps directly to `MetricsAnalyzer` arguments
  (`bottleneck_threshold`, `cv_threshold`, `memory_growth_bytes_per_step`,
  `workload_profile`, `enable_advanced_rules`).
- `load_events_file(path)` supports offline analysis of previously captured
  event files, and `export_chrome_trace()` emits a Chrome-trace view of the same
  events.

## 11. Usage Example

```python
from my_utils.core.utils import MyTimer
from my_utils.profiling import (
    MetricsAnalyzer,
    MetricsCollector,
    MyTimerMetricsProvider,
)

# Analyzer with a workload profile and custom thresholds.
analyzer = MetricsAnalyzer(
    workload_profile="pretrain",
    bottleneck_share_threshold=0.15,
    cv_threshold=0.25,
    memory_growth_bytes_per_step=512 * 1024,
)

collector = MetricsCollector(output_dir="metrics", analyzer=analyzer)

timer = MyTimer(use_cuda=True)
collector.register_provider(MyTimerMetricsProvider(timer))

collector.start()
for step in range(num_steps):
    # ... training step instrumented with `timer` ...
    collector.collect(step=step)
collector.stop()

report = collector.analyze()

print(f"Overall score: {report.overall_score:.0f}/100")
print(report.summary)

print("\n=== Findings ===")
for finding in report.findings:
    print(f"[{finding.severity.upper()}] {finding.title}")
    print(f"  {finding.description}")

print("\n=== Recommendations ===")
for rec in report.recommendations:
    print(f"- {rec}")

collector.export_report(fmt="markdown", report=report)
```

For CLI-driven and config-driven equivalents, see
[./UNIFIED_PROFILING_QUICKSTART.md](./UNIFIED_PROFILING_QUICKSTART.md).

## Summary

The analyzer achieves automated analysis through:

1. **One engine, many rules** — a single `MetricsAnalyzer` runs independent
   `AnalysisRule` instances selected per workload profile.
2. **Rule-driven, threshold-configurable** — every detection threshold is a
   constructor argument with a documented default and a config-file mapping.
3. **Evidence-oriented** — each finding carries the numbers it was decided on,
   and the trust layer blocks conclusions the data cannot support.
4. **Actionable output** — recommendations are attached to the finding that
   motivated them, and reports render to JSON, Markdown, and HTML.

Implementation status: all phases of this design are complete; see
[./ROADMAP.md](./ROADMAP.md) and
[./IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md).

---

Chinese original: [docs/zh/profiling/docs/AUTO_ANALYZER_DESIGN.md](../../../docs/zh/profiling/docs/AUTO_ANALYZER_DESIGN.md)
