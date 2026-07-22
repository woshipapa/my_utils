# Visualization Enhancement Design

## Design Goals

1. **Shareable output** - self-contained static HTML reports that can be archived and sent around
2. **Timeline interoperability** - export collected metrics as Chrome `traceEvents` JSON for Perfetto / `chrome://tracing`
3. **Interactive analysis** - drill-down, filtering, and comparison inside the generated reports
4. **Framework agnostic** - the visualization layer does not depend on any specific training framework

> **Historical note:** earlier revisions of this design also planned a TensorBoard
> plugin and a standalone real-time Web Dashboard. Neither was built and neither is
> planned; static HTML reports plus the Chrome-trace exporter
> (`my_utils/profiling/output/metrics_trace.py`) replaced both. See
> [Retired plans](#3-retired-plans-tensorboard-plugin-and-web-dashboard).

## Overall Architecture

```
┌──────────────────────────────────────────────────────────────┐
│              Visualization Layer                             │
│              my_utils/profiling/visualization/               │
│                                                              │
│   ┌────────────────────┐    ┌────────────────────┐           │
│   │ HTMLReport         │    │ QuickReport        │           │
│   │ Generator          │    │ Generator          │           │
│   └─────────┬──────────┘    └─────────┬──────────┘           │
│             └─────────────┬───────────┘                      │
│                           ▼                                  │
│             ┌──────────────────────────┐                     │
│             │   Shared Components      │                     │
│             │  - charts.py (renderers) │                     │
│             │  - transformers.py       │                     │
│             │  - layouts.py            │                     │
│             └──────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
              ▲                            ▲
              │ MetricEvent lists          │ MetricEvent lists
┌─────────────┴──────────┐    ┌────────────┴────────────────────┐
│ MetricsCollector       │───▶│ output/metrics_trace.py         │
│ + MetricsAnalyzer      │    │ Chrome traceEvents JSON export  │
└────────────────────────┘    └─────────────────────────────────┘
```

All components below exist in the codebase; code excerpts show the real
signatures (docstrings and CSS are abridged where noted).

## 1. Shared Component Library

### 1.1 Chart Templates (`visualization/charts.py`)

A chart is described by a renderer-independent `ChartConfig`; concrete
renderers turn it into embeddable HTML (or JSON for API-style consumers).
This keeps the report layout code decoupled from the charting library.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import json


@dataclass
class ChartConfig:
    """Chart configuration."""
    chart_type: str  # "line", "bar", "pie", "heatmap", "scatter", "doughnut"
    title: str
    data: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    width: Optional[str] = None
    height: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.chart_type,
            "title": self.title,
            "data": self.data,
            "options": self.options,
            "width": self.width,
            "height": self.height,
        }


class ChartRenderer(ABC):
    """Base class for chart renderers."""

    @abstractmethod
    def render(self, config: ChartConfig) -> str:
        """Render the chart as an HTML fragment."""

    @abstractmethod
    def render_to_json(self, config: ChartConfig) -> str:
        """Render the chart as JSON (for API-style responses)."""

    def _get_chart_id(self, config: ChartConfig) -> str:
        """Derive a stable, unique chart id from the config (md5 of type/title/data)."""

    def _get_default_options(self, config: ChartConfig) -> dict:
        return {
            "responsive": True,
            "maintainAspectRatio": False,
            "plugins": {
                "legend": {"position": "top"},
                "tooltip": {"mode": "index", "intersect": False},
            },
        }
```

Three concrete renderers are implemented:

```python
class ChartJsRenderer(ChartRenderer):
    """
    Chart.js renderer (default fallback; loads Chart.js from a CDN).
    """

    def __init__(self, version: str = "4.4.0"):
        self.version = version
        self.cdn_url = f"https://cdn.jsdelivr.net/npm/chart.js@{self.version}/dist/chart.umd.min.js"

    def render(self, config: ChartConfig) -> str:
        chart_id = self._get_chart_id(config)
        options = self._merge_options(config)

        template = '''
        <div class="chart-container" {style}>
            <canvas id="{chart_id}"></canvas>
        </div>
        <script>
            (function() {{
                const ctx = document.getElementById('{chart_id}');
                if (ctx) {{
                    new Chart(ctx, {{
                        type: '{chart_type}',
                        data: {data},
                        options: {options}
                    }});
                }}
            }})();
        </script>
        '''
        # ... fills in chart_id, chart_type, JSON-encoded data/options,
        # and an optional inline width/height style.

    def _merge_options(self, config: ChartConfig) -> dict:
        """Deep-merge defaults, chart-type-specific options, and config.options."""

    def _get_type_specific_options(self, chart_type: str) -> dict:
        """Per-type defaults: axis scaffolding for line/bar (with smoothed
        lines via tension 0.4), right-hand legends for pie/doughnut."""


class PlotlyRenderer(ChartRenderer):
    """
    Plotly renderer for interactive charts.

    If Plotly is not importable, render() transparently falls back to
    ChartJsRenderer, so callers never need to guard the import themselves.
    """

    def __init__(self, offline: bool = True):
        self.offline = offline  # inline plotly.js for fully offline reports

    def render(self, config: ChartConfig) -> str:
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
        except ImportError:
            return ChartJsRenderer().render(config)

        fig = self._create_figure(config)
        plot_div = plot(
            fig,
            output_type="div",
            include_plotlyjs="inline" if self.offline else False,
            config={"displayModeBar": True, "displaylogo": False},
        )
        # ... wraps plot_div in a .chart-container div

    def _create_figure(self, config: ChartConfig):
        """Map ChartConfig to a plotly Figure.

        Supports "line" (Scatter, lines+markers), "bar", "pie", and
        "doughnut" (Pie with hole=0.4); applies title and converts a small
        subset of Chart.js-style options (e.g. beginAtZero -> rangemode)."""


class EChartsRenderer(ChartRenderer):
    """
    Apache ECharts renderer (richer feature set, e.g. built-in resize handling).
    """

    def __init__(self, version: str = "5.4.3"):
        self.version = version
        self.cdn_url = f"https://cdn.jsdelivr.net/npm/echarts@{self.version}/dist/echarts.min.js"

    def _convert_to_echarts_option(self, config: ChartConfig) -> dict:
        """Translate ChartConfig into an ECharts `option` object for
        line / bar / pie / doughnut charts."""
```

A factory selects the backend so the rest of the pipeline never hard-codes
a charting library:

```python
def create_chart_renderer(backend: str = "auto") -> ChartRenderer:
    """
    Create a chart renderer.

    Args:
        backend:
            - "auto":    prefer Plotly if importable, else Chart.js
            - "chartjs": Chart.js renderer
            - "plotly":  Plotly renderer
            - "echarts": ECharts renderer
    """
```

### 1.2 Data Transformers (`visualization/transformers.py`)

The transformers convert lists of metric events into the `labels` /
`datasets` shapes the renderers expect.

`transformers.py` defines its **own lightweight `MetricEvent` dataclass**.
It is intentionally decoupled from the richer
`my_utils.profiling.metrics.metrics_types.MetricEvent` used by the
collector pipeline, so the visualization layer can be fed from any source
(timers, CSV logs, ad-hoc scripts) without importing the metrics stack:

```python
@dataclass
class MetricEvent:
    """Unified metric event format (visualization-local)."""
    timestamp: float             # Unix timestamp in seconds
    name: str                    # metric name
    value: float                 # metric value
    unit: str = ""               # unit
    tags: dict[str, Any] = None  # extra tags; defaults to {} in __post_init__
```

`DataTransformer` is a namespace of static methods. Representative
implementations:

```python
class DataTransformer:
    """Convert MetricEvent lists into chart-friendly formats."""

    @staticmethod
    def to_time_series(
        events: list[MetricEvent],
        metric_name: str | None = None,
        group_by_step: bool = True,
    ) -> dict:
        """
        Convert to time-series data for a line chart.

        If group_by_step is True and events carry a "step" tag, values are
        grouped by step and averaged per step; otherwise events are sorted
        by timestamp and plotted directly.

        Returns {"labels": [...], "datasets": [{...}]}.
        """

    @staticmethod
    def to_comparison(
        events: list[MetricEvent],
        metric_name: str,
        group_by: str = "rank",
    ) -> dict:
        """
        Convert to comparison (bar-chart) data.

        Groups the named metric by a tag (e.g. "rank"), returning per-group
        means plus an "errors" list of standard deviations for error bars.
        """
```

The full set of transformations:

| Method | Purpose |
|---|---|
| `to_time_series(events, metric_name=None, group_by_step=True)` | Single line series, grouped by step or by timestamp |
| `to_multiple_time_series(events, metric_names)` | Multiple series on a shared step axis, with a color cycle |
| `to_comparison(events, metric_name, group_by="rank")` | Per-group means + std-dev error bars for bar charts |
| `to_distribution(events, metric_name, bins=20)` | Histogram via `np.histogram`, bin centers as labels |
| `to_pie_chart(events, metric_name)` | Share-of-total breakdown (e.g. per-kernel time), sorted descending |
| `to_scatter_plot(events, x_metric, y_metric)` | Pairs two metrics by common step for correlation analysis |
| `to_heatmap(events, row_metric, col_metric, value_metric)` | `{"rows", "cols", "matrix"}` aggregated by two tag dimensions |
| `compute_statistics(events, metric_name=None)` | Per-metric count/mean/std/min/max/median/p25/p75, sorted by mean |
| `filter_by_tags(events, tags)` | Keep only events matching all tag key/value pairs |
| `aggregate_by_time_window(events, window_size=1.0, aggregation="mean")` | Downsample into fixed time windows (mean/sum/min/max/count) |

**Chrome trace conversion is not part of `DataTransformer`.** Trace export
operates on the collector-side `MetricEvent` (which carries `attributes`
such as precise start/end timestamps) and lives in
`my_utils/profiling/output/metrics_trace.py`:

```python
from my_utils.profiling import (
    ChromeTraceExportConfig,
    estimate_rank_time_offsets,
    metric_events_to_chrome_trace,   # events -> {"traceEvents": [...], ...}
    write_chrome_trace,              # events -> trace JSON file
    export_events_file_to_chrome_trace,  # events.jsonl file -> trace JSON file
)
```

The resulting `traceEvents` JSON loads directly into Perfetto or
`chrome://tracing`. `MetricsCollector.export_chrome_trace()` wraps
`write_chrome_trace` for the common case (see the usage example below).

### 1.3 Layout Builders (`visualization/layouts.py`)

`layouts.py` owns the report-domain dataclasses and the HTML assembly. The
report model is deliberately renderer-free so analyzers can produce it
without importing any charting code:

```python
class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    """A single analysis finding."""
    id: str
    title: str
    description: str
    severity: Severity
    category: str
    evidence: dict[str, Any]
    affected_components: list[str]
    metrics: dict[str, float]


@dataclass
class Recommendation:
    """An optimization recommendation."""
    id: str
    title: str
    description: str
    priority: int            # rendered as P<priority>; >=8 high, >=5 medium
    estimated_impact: str
    effort: str
    actions: list[str]
    references: list[str]


@dataclass
class AnalysisReport:
    """A full analysis report (visualization-side model)."""
    metadata: dict[str, Any]
    findings: list[Finding]
    recommendations: list[Recommendation]
    summary: str
    overall_score: float     # 0-100
```

Timeline-style reports additionally use an extensible panel system, so new
panels can be added (or reordered) without touching the builder:

```python
@dataclass(frozen=True)
class PanelSpec:
    """Extensible panel definition."""
    panel_id: str
    title: str
    description: str = ""
    order: int = 0
    enabled: bool = True
    min_height: str = "260px"


def default_timeline_panel_specs(include_metrics: bool = True) -> list[PanelSpec]:
    """Default timeline panel order: gpu_metrics, rank_heatmap,
    roofline_proxy, gil_lane, kernel_category, nvtx_stability,
    all_streams, nvtx_scopes, kernel_timeline."""
```

`LayoutBuilder` assembles a report section by section with a fluent
(chainable) API. Note that `add_chart` takes **already-rendered chart
HTML** (from any `ChartRenderer`), keeping layout and rendering decoupled:

```python
class LayoutBuilder:
    """
    HTML layout builder.

    Builds the sections of an HTML report step by step.
    """

    def __init__(self):
        self.sections = []
        self.scripts = []
        self.styles = []
        self.title = "Performance Analysis Report"
        self.subtitle = ""

    def set_title(self, title: str, subtitle: str = "") -> "LayoutBuilder": ...

    def add_header(self, title: str = None, subtitle: str = None) -> "LayoutBuilder":
        """Title header; defaults to self.title and a 'Generated at ...' subtitle."""

    def add_summary(
        self,
        summary: str,
        score: float,
        details: dict[str, Any] = None,
    ) -> "LayoutBuilder":
        """Score card. score >= 80 renders green ('good'), >= 60 amber
        ('warning'), otherwise red ('critical'); details render as
        label/value pairs under the summary text."""

    def add_metrics_grid(self, metrics: dict[str, Any]) -> "LayoutBuilder":
        """Grid of headline metric cards ({"name": value})."""

    def add_chart(
        self,
        chart_html: str,
        title: str = "",
        width: str = None,
        height: str = "400px",
    ) -> "LayoutBuilder":
        """Add a chart section wrapping pre-rendered chart HTML."""

    def add_two_column_charts(
        self,
        left_chart_html: str,
        right_chart_html: str,
        left_title: str = "",
        right_title: str = "",
    ) -> "LayoutBuilder":
        """Side-by-side layout, e.g. a pie chart next to its data table."""

    def add_panel_grid(
        self,
        panel_html_map: dict[str, str],
        *,
        panel_specs: list[PanelSpec] | None = None,
        title: str = "Panels",
    ) -> "LayoutBuilder":
        """Extensible panel grid ordered by PanelSpec; defaults to
        default_timeline_panel_specs(). Panels missing from the map are
        skipped silently."""

    def add_findings(self, findings: list[Finding], title: str = "Findings") -> "LayoutBuilder":
        """Findings list with severity badge, description, affected
        components, and formatted evidence key/value pairs."""

    def add_recommendations(
        self,
        recommendations: list[Recommendation],
        title: str = "Recommendations",
    ) -> "LayoutBuilder":
        """Recommendations with priority badge, expected impact, effort,
        an action checklist, and optional reference links."""

    def add_table(
        self,
        data: list[dict[str, Any]],
        title: str = "",
        sortable: bool = True,
    ) -> "LayoutBuilder":
        """Table from a list of row dicts; headers come from the first row."""

    def add_code_block(self, code: str, language: str = "python") -> "LayoutBuilder": ...
    def add_section(self, html: str) -> "LayoutBuilder":
        """Escape hatch: append arbitrary custom HTML."""
    def add_divider(self) -> "LayoutBuilder": ...
    def add_script(self, script: str) -> "LayoutBuilder": ...
    def add_style(self, style: str) -> "LayoutBuilder": ...

    def build(self, include_styles: bool = True) -> str:
        """
        Assemble the final self-contained HTML document: <head> pulls in
        Chart.js 4.4.0 from the jsDelivr CDN plus the built-in CSS (and any
        add_style additions), <body> concatenates all sections inside a
        .container div, followed by any add_script scripts.
        """
```

The built-in stylesheet (`_get_builtin_css`, ~400 lines) defines the visual
language of every generated report:

- card-based layout on a gradient page background, max content width 1400px;
- score cards and severity accents share one palette: green `#10b981`
  (good / low), amber `#f59e0b` (warning / medium), orange `#f97316`
  (high), red `#ef4444` (critical), blue `#3b82f6` (low/info badges);
- dedicated styles for the metrics grid, chart sections, panel grid,
  findings/recommendations lists, evidence blocks, data tables (hover
  highlighting), dark code blocks, and section dividers;
- a responsive breakpoint at 768px collapses two-column layouts and the
  metrics grid for small screens.

## 2. HTML Report Generator (`visualization/html_generator.py`)

`HTMLReportGenerator` ties the three shared components together: it
transforms events, renders charts, and drives a `LayoutBuilder` to produce
the final document.

```python
class HTMLReportGenerator:
    """
    HTML report generator.

    Combines chart rendering, data transformation, and layout building
    into a complete performance-analysis HTML report.
    """

    def __init__(
        self,
        renderer: Optional[ChartRenderer] = None,
        transformer: Optional[DataTransformer] = None,
    ):
        # renderer defaults to create_chart_renderer() ("auto" backend);
        # transformer defaults to DataTransformer()
        self.renderer = renderer or create_chart_renderer()
        self.transformer = transformer or DataTransformer()

    def generate(
        self,
        report: AnalysisReport,
        events: list[MetricEvent],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate the full HTML report. If output_path is given, the file
        (and its parent directories) are written as a side effect; the HTML
        string is always returned.
        """
        builder = LayoutBuilder()

        # 1. Header
        builder.add_header(
            title="Performance Analysis Report",
            subtitle=f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )
        # 2. Summary card (score + event/finding/recommendation counts)
        self._add_summary_section(builder, report)
        # 3. Key metric cards
        self._add_key_metrics(builder, events)
        # 4. Performance trend charts
        self._add_trend_charts(builder, events)
        # 5. Bottleneck analysis charts
        self._add_bottleneck_charts(builder, report)
        # 6. Memory analysis charts
        self._add_memory_charts(builder, events)
        # 7. Findings
        if report.findings:
            builder.add_findings(report.findings, title="Key Findings")
        # 8. Recommendations
        if report.recommendations:
            builder.add_recommendations(report.recommendations, title="Recommendations")
        # 9. Detailed statistics table
        self._add_detail_tables(builder, events)
        # 10. Build (and optionally save)
        html = builder.build()
        ...
        return html
```

Section builders (all private methods on the generator):

- `_add_summary_section` - feeds `report.summary` / `report.overall_score`
  into `add_summary`, with `event_count` / `finding_count` /
  `recommendation_count` pulled from `report.metadata` as details.
- `_add_key_metrics` - shows the latest `step` seen in event tags plus the
  top-5 metrics by mean (from `DataTransformer.compute_statistics`) as a
  metrics grid.
- `_add_trend_charts` - plots a multi-series line chart via
  `to_multiple_time_series`. Preferred metrics, in order: `timer.iter`,
  `timer.forward`, `timer.backward`, `memory.allocated`,
  `memory.reserved`, `loss`; if none of these are present it falls back to
  the first three metric names found.
- `_add_bottleneck_charts` - selects findings whose title contains
  "bottleneck" (matched case-insensitively, in English or Chinese), reads
  `component` and `ratio` / `percentage` from their evidence, and renders a
  pie chart next to a percentage breakdown table using
  `add_two_column_charts`.
- `_add_memory_charts` - groups `memory.*` events by suffix (`allocated`,
  `reserved`, ...) and plots them as line series over steps.
- `_add_detail_tables` - top-20 rows of `compute_statistics` (count, mean,
  std, min, max, median) as a sortable table.

For the common case where no analyzer has run, a report can be produced
from raw events alone:

```python
    def generate_from_events(
        self,
        events: list[MetricEvent],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a simple report from events only (no AnalysisReport
        required). Wraps the events in a stub report (default score 75,
        no findings/recommendations) and delegates to generate().
        """
```

### 2.1 Quick Report Generator

`QuickReportGenerator` adapts existing lightweight tooling to the report
pipeline without requiring the metrics stack:

```python
class QuickReportGenerator:
    """
    Quick report generator.

    Produces reports directly from existing tools (e.g. MyTimer) or logs.
    """

    def generate_from_timer(self, timer, output_path: str = "timer_report.html") -> str:
        """Extract events from a my_utils.utils.MyTimer instance
        (its _events list) and generate a report."""

    def generate_from_csv(self, csv_path: str, output_path: str = "csv_report.html") -> str:
        """Parse a timer CSV log (columns: timestamp_unix, step, event_name,
        event_type, duration_ms) into MetricEvents and generate a report.
        Malformed rows are skipped."""
```

## 3. Retired Plans: TensorBoard Plugin and Web Dashboard

Earlier drafts of this document specified a TensorBoard plugin
(`my_utils.profiling.tensorboard_plugin`) and a standalone WebSocket-based
Web Dashboard (`my_utils.profiling.dashboard`) for real-time monitoring.
Neither was implemented, and the roadmap (see [ROADMAP](./ROADMAP.md))
completed without them: static HTML reports (section 2) cover shareable
post-hoc analysis, and the Chrome-trace exporter in
`my_utils/profiling/output/metrics_trace.py` covers interactive timeline
exploration through Perfetto / `chrome://tracing`. The original Chinese
specifications for both retired components are preserved in the archived
source document.

## 4. Usage Example

Generate a report directly from the visualization package (fully
self-contained; also re-exported from `my_utils.profiling` when the
optional visualization dependencies import cleanly — check
`my_utils.profiling.VISUALIZATION_AVAILABLE`):

```python
import time

from my_utils.profiling.visualization import (
    HTMLReportGenerator,
    MetricEvent,
)

# Build (or adapt) events - here, simulated timer/loss data
events = []
for step in range(20):
    events.append(MetricEvent(
        timestamp=time.time() + step,
        name="timer.forward",
        value=100 + step * 0.5,
        unit="ms",
        tags={"step": str(step)},
    ))
    events.append(MetricEvent(
        timestamp=time.time() + step,
        name="loss",
        value=2.0 * (0.95 ** step),
        unit="",
        tags={"step": str(step)},
    ))

# Generate an HTML report (no analyzer needed)
generator = HTMLReportGenerator()
generator.generate_from_events(events, output_path="./metrics_logs/report.html")
print("HTML report written to ./metrics_logs/report.html")
```

With the full metrics pipeline, the collector drives analysis, report
export, and Chrome-trace export:

```python
from my_utils.profiling import MetricsCollector

collector = MetricsCollector(output_dir="./metrics_logs")
# ... register providers (collector.register_provider(...)), then during
# training call collector.collect(step=step) ...

events = collector.get_events()
report = collector.analyze(events)                      # AnalysisReport
collector.export_report(fmt="md", report=report)        # markdown/json report
trace_path = collector.export_chrome_trace()            # metrics_trace.json
print(f"Open {trace_path} in Perfetto or chrome://tracing")
```

For a walkthrough of every component (renderers, transformers, layout
builder, full reports, `QuickReportGenerator` from `MyTimer` and from
CSV), run the executable examples in
`my_utils/profiling/visualization/examples.py`.

## Summary

The visualization layer serves two complementary needs:

1. **HTML reports** - static, shareable, complete analysis
   (`HTMLReportGenerator` / `QuickReportGenerator`)
2. **Chrome trace export** - interactive timeline exploration in
   Perfetto / `chrome://tracing` (`output/metrics_trace.py`)

Possible future work:

1. More chart types (heatmap rendering in all backends, Sankey diagrams)
2. Custom themes and layouts for generated reports
