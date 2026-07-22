# metrics

Unified metrics data layer: defines what a metric is, where it comes from, how
it is stored, and how it is validated.

## When you touch this package

- Integrating a new data source (a new provider).
- Adding metric fields or schema constraints.
- Changing metric taxonomy, naming, or normalization.

## Key files

- `metrics_types.py` — core types (`MetricEvent`, `AnalysisReport`, ...).
- `metrics_schema.py` — schema validation and normalization.
- `metrics_provider.py` — the provider abstraction.
- `metrics_providers.py` — built-in provider implementations.
- `provider_registry.py` — provider registration and config-driven
  instantiation.
- `metrics_store.py` — event persistence and reading.
- `metrics_taxonomy.py` — metric taxonomy.

## Most common change

1. New provider: implement the `MetricsProvider` interface.
2. Register it in `provider_registry.py`.
3. Add a config entry in `examples/collector_config_example.json` and verify.

---

Chinese original: [docs/zh/profiling/metrics/README.md](../../../docs/zh/profiling/metrics/README.md)
