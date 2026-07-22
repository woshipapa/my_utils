# pipeline

Orchestration layer: wires providers, store, analyzer, and report output into
one unified flow.

## Core entry point

- `MetricsCollector` in `metrics_collector.py`.

## Typical flow

1. Register providers.
2. `collect()` writes events.
3. `analyze()` produces the analysis result.
4. `export_report()` / `export_chrome_trace()` export the results.

## When you touch this package

- Changing the unified collect/analyze flow itself (not a single provider).
- Adding a unified export capability (e.g. a new report format).

---

Chinese original: [docs/zh/profiling/pipeline/README.md](../../../docs/zh/profiling/pipeline/README.md)
