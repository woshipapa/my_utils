# analyzers

Analysis layer: turns raw metric events into actionable conclusions.

## When you touch this package

- Adding a bottleneck rule (e.g. memory bound / load imbalance / comm skew).
- Adjusting per-workload judgment logic (pretrain / inference / rl).
- Extending multi-node, multi-GPU alignment analysis.

## Key files

- `metrics_analyzer.py` — main entry point; produces unified
  findings/recommendations.
- `analysis_rules.py` — rule definitions and matching logic.
- `workload_profiles.py` — per-workload analysis configuration.
- `distributed_alignment.py` — rank/stage alignment analysis.
- `triage.py`, `axes.py`, `evidence.py` — triage flow, analysis axes, and
  evidence tracking used by the report output.
- `nccl_bandwidth.py` — NCCL bandwidth estimation helpers.
- `trace_quality.py`, `measurement_context.py` — data-quality and
  measurement-context checks that guard against misleading numbers.

## Suggested change order

1. Define the rule in `analysis_rules.py`.
2. Wire it up in `metrics_analyzer.py`.
3. Run a demo from `examples/` to confirm the output changed as intended.

---

Chinese original: [docs/zh/profiling/analyzers/README.md](../../../docs/zh/profiling/analyzers/README.md)
