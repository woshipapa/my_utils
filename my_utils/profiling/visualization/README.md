# visualization

Renders profiling data into charts and HTML reports.

## Quick orientation

1. You already have a report/events and want HTML fast:
   `HTMLReportGenerator`.
2. You just want one chart (line/bar): the renderers in `charts.py` plus
   `ChartConfig`.
3. You want to turn raw events into chartable data:
   `DataTransformer` in `transformers.py`.
4. You want a custom report layout: `LayoutBuilder` in `layouts.py`.

## Minimal example

```python
from my_utils.profiling.visualization import (
    HTMLReportGenerator,
    QuickReportGenerator,
)

# 1) quick report from a CSV
quick = QuickReportGenerator()
quick.generate_from_csv("profile_rank_0.csv", output_path="csv_report.html")

# 2) HTML from a unified analysis report object
gen = HTMLReportGenerator()
html = gen.generate(report_obj, events, output_path="analysis_report.html")
```

## Key files

- `charts.py` — chart config and renderer factory
  (`ChartJsRenderer`, `PlotlyRenderer`, `EChartsRenderer`,
  `create_chart_renderer`).
- `transformers.py` — metric events to chart inputs (time series,
  statistical aggregation).
- `layouts.py` — report layout building (header/summary/cards/sections).
- `html_generator.py` — final HTML report generation (findings,
  recommendations, charts).
- `examples.py` — built-in examples for this module.

## Practical advice

1. Start with `QuickReportGenerator` for a default report.
2. To customize, change `layouts.py` first, then chart styling in `charts.py`.
3. For complex charts, do the data shaping in `transformers.py`; keep logic
   out of the templates.

---

Chinese original: [docs/zh/profiling/visualization/README.md](../../../docs/zh/profiling/visualization/README.md)
