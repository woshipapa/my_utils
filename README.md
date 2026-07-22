# my_utils

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A GPU profiling toolkit for NVIDIA GPUs. It wraps the collection side (Nsight
Systems, Nsight Compute, the NCCL Inspector profiler plugin) in ready-to-run
presets, and turns the resulting reports into diagnoses: an evidence-based
kernel diagnosis engine that reads an `.ncu-rep` and produces ranked findings
with per-metric evidence and PC/PM-sampling source attribution, top-down nsys
triage, NCCL collective analysis, cross-framework capture adapters, and HTML
timeline visualization.

**Start here for performance work:**
[`my_utils/profiling/docs/PERFORMANCE_ANALYSIS_HANDBOOK.md`](my_utils/profiling/docs/PERFORMANCE_ANALYSIS_HANDBOOK.md)
— the end-to-end guide from collection to diagnosis.

## What runs where

- The **analysis core is pure Python** — no torch, no CUDA, no GPU. It runs
  anywhere Python 3.10+ runs, including a laptop analysing reports collected
  on a cluster.
- The **capture paths** need the NVIDIA tools themselves (`nsys`, `ncu`, or
  the NCCL Inspector plugin) on the machine doing the collection.
- Reading `.ncu-rep` files requires the `ncu_report` Python module, which
  **ships with Nsight Compute** — it is not on PyPI, so the fix for
  `import ncu_report` failures is a `PYTHONPATH` entry, not a pip install
  (the CLI prints the exact command when it cannot find it).

## Install

```bash
pip install git+https://github.com/woshipapa/my_utils.git
```

`torch` is intentionally **not** a dependency — it is only needed for the
in-process capture backend, and pulling it in would disturb environments with
a tuned torch/cuDNN stack. Install it separately or use the `[torch]` extra
if you need that path.

## Quickstart

```bash
# You have an .ncu-rep: per-kernel diagnosis (bottleneck class, stalls, roofline, fixes)
myutils-profile ncu-diagnose --report run.ncu-rep --gpu "H100 SXM5" --format md

# Or the raw summarized view with per-metric stats
ncu-report-analyze --report run.ncu-rep --top-k 20 --pretty

# You have an nsys capture: export sqlite, then summary/overlap/NCCL/iteration analysis
nsys export --type sqlite run.nsys-rep
nsys-analyze --sqlite run.sqlite --output nsys_analyze.json
```

## CLI commands

Every command below is also available as a `myutils-profile <name>` subcommand;
`myutils-profile` additionally exposes the unified metrics pipeline
(`ingest` / `analyze` / `report` / `diff` / `trace`) and the
`ncu-metrics` / `ncu-diagnose` helpers.

| Command | What it does |
|---|---|
| `myutils-profile` | Umbrella CLI: all subcommands plus the unified metrics pipeline |
| `nsys-panel` | Interactive panel to pick an nsys subcommand, fill args, run or print it |
| `nsys-sql-skill` | Run built-in SQL skills against an nsys-exported SQLite |
| `nsys-export` | Export SQLite kernel timeline rows to JSON/CSV |
| `nsys-analyze` | Summary / overlap / NCCL / iteration / MFU analysis of an nsys SQLite |
| `nsys-diff` | Diff two nsys SQLite profiles by kernel/NVTX aggregates |
| `nsys-module-kernel-compare` | Compare two profiles for one module/NVTX scope (stream timeline + resource deltas) |
| `nsys-timeline-html` | Export an interactive HTML timeline from an nsys SQLite |
| `nsys-iter-overlap` | Per-iteration compute/comm/overlap breakdown from NVTX markers |
| `nsys-iter-outliers` | Flag statistically anomalous training iterations by step duration |
| `ncu-csv-skill` | Run built-in parsing skills for ncu CSV exports |
| `ncu-csv-analyze` | Summarized analysis of an ncu CSV export |
| `ncu-report-skill` | Run built-in parsing skills for `.ncu-rep` reports |
| `ncu-report-analyze` | Summarized `.ncu-rep` analysis with per-metric stats and top bottlenecks |
| `nccl-inspector-skill` | Run built-in parsing skills for NCCL Inspector JSON/Prometheus output |
| `nccl-inspector-analyze` | Summarize NCCL Inspector dumps: collectives/P2P, bandwidth, rank skew |

## Documentation

- [`PERFORMANCE_ANALYSIS_HANDBOOK.md`](my_utils/profiling/docs/PERFORMANCE_ANALYSIS_HANDBOOK.md)
  — the flagship document: collect every metric that matters, then turn it
  into a diagnosis.
- [`CAPABILITY_EVOLUTION.md`](my_utils/profiling/docs/CAPABILITY_EVOLUTION.md)
  — how the analysis engine's capabilities are organised and grown.
- [`UNIFIED_PROFILING_QUICKSTART.md`](my_utils/profiling/docs/UNIFIED_PROFILING_QUICKSTART.md)
  — the config-driven metrics collector: providers, workload-aware rules,
  report diff, Chrome trace export.
- [`my_utils/profiling/README.md`](my_utils/profiling/README.md) — the
  profiling package landing page and subpackage map.

Some in-depth design docs are currently in Chinese; English translations are
in progress.

The non-profiling subpackages (`core/`, `hooks/`, `memory/`, `distributed/`,
`artifacts/`, `legacy_profilers/`, `tracing/`) are auxiliary utilities from
the same training workflows, shipped as-is without support commitments.

## Troubleshooting

- **`import ncu_report` fails** — the module ships inside Nsight Compute
  (`<install>/extras/python`), not on PyPI. The CLI auto-detects common
  install locations and prints the exact `PYTHONPATH` export to use when it
  cannot.
- **NVTX ranges do not appear in the nsys timeline** — make sure NVTX is in
  the trace set (`nsys profile -t cuda,nvtx ...`); labelers created by
  `my_utils.tracing` degrade to no-ops when NVTX is unavailable or disabled.
- **HTML report charts render blank** — generated reports load Chart.js /
  ECharts from `cdn.jsdelivr.net`; on an offline machine, open the report
  where that CDN is reachable or vendor the script locally.

### Locating Nsight Compute

The ncu tools auto-detect common install locations (`/opt/nvidia/nsight-compute`,
the CUDA-bundled copy, the macOS app bundle under `/Applications`). If yours
lives elsewhere, point them at it:

```bash
export NCU_PATH=/path/to/ncu                          # the ncu binary or the install dir
# or
export NSIGHT_COMPUTE_HOME=/opt/nvidia/nsight-compute/2026.1.1
```

`NCU_PATH` wins over `NSIGHT_COMPUTE_HOME`, and both win over the built-in
defaults; unset, behaviour is unchanged. `NCU_PYTHON_DIR` is still honoured for
pointing directly at the directory containing `ncu_report.py`, and outranks both.

## License

Apache-2.0. See [LICENSE](LICENSE).
