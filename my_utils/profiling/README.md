# profiling

The `profiling` package is now split by responsibility. The root keeps entry points (API/CLI), docs, and templates.

## Layout

```text
profiling/
  analyzers/   # rules, workload profiles, rank alignment
  metrics/     # metric schema/types/providers/store/registry/taxonomy
  output/      # report/diff/chrome-trace export
  runtime/     # capture runtime integration and ProfileManager
  sources/     # nsys sqlite/schema parsers
  pipeline/    # MetricsCollector orchestration
  docs/        # design/spec/roadmap docs
  adapters/
  visualization/
  templates/
  examples/
  cli.py
  __init__.py
```

## Flow

1. `metrics/*` providers convert tool outputs to `MetricEvent`.
2. `pipeline/MetricsCollector` does collect -> normalize/validate -> store.
3. `analyzers/*` runs rule-based analysis in `analyze()`.
4. `output/*` exports reports and chrome trace.
5. `runtime/*` integrates capture windows and runtime controls.

## CLI Quick Commands

```bash
# list built-in providers
myutils-profile list-providers

# run built-in Nsight SQLite SQL skills
myutils-profile nsys-sql-skill --sqlite ./train_rank0.sqlite --list-skills --pretty
myutils-profile nsys-sql-skill --sqlite ./train_rank0.sqlite --skill top_kernels --param device_id=0 --param limit=20 --pretty

# nsys-oriented offline workflow
myutils-profile nsys-analyze --sqlite ./train_rank0.sqlite --device-id 0 --top-k 20 --output ./nsys_analyze.json
myutils-profile nsys-export --sqlite ./train_rank0.sqlite --device-id 0 --format csv --output ./kernels_flat.csv
myutils-profile nsys-diff --before-sqlite ./run_a.sqlite --after-sqlite ./run_b.sqlite --device-id 0 --output ./nsys_diff.json
myutils-profile nsys-timeline-html --sqlite ./train_rank0.sqlite --device-id 0 --output ./timeline.html
```

## Submodule Docs

- [docs/README.md](./docs/README.md)
- [analyzers/README.md](./analyzers/README.md)
- [metrics/README.md](./metrics/README.md)
- [output/README.md](./output/README.md)
- [runtime/README.md](./runtime/README.md)
- [sources/README.md](./sources/README.md)
- [pipeline/README.md](./pipeline/README.md)

## Compatibility

Legacy module paths are still supported via aliases in `profiling.__init__`, e.g.
- `my_utils.profiling.metrics_collector`
- `my_utils.profiling.metrics_analyzer`
- `my_utils.profiling.metrics_trace`

New code should use layered paths, e.g.
- `my_utils.profiling.pipeline.metrics_collector`
- `my_utils.profiling.analyzers.metrics_analyzer`
- `my_utils.profiling.output.metrics_trace`
