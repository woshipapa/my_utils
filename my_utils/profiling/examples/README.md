# examples (runnable)

This directory serves two purposes:

1. Quickly verify that the unified metrics pipeline works.
2. Provide copy-paste config templates (especially for NSYS sqlite input).

## Quick orientation

1. Minimal demo (provider + analyze + report): `unified_metrics_demo.py`.
2. End-to-end acceptance run (including diff): `p0_p13_end_to_end_demo.py`.
3. Offline analysis straight from a CLI config:
   `collector_config_*.json` + `myutils-profile ingest`.
4. Per-framework ready-made commands (TorchTitan/SLIME/VERL/ROLL/HF/SGLang/vLLM):
   `framework_playbook_samples/README.md`.

## Most-used commands

### A) Minimal unified demo

```bash
python -m my_utils.profiling.examples.unified_metrics_demo \
  --output-dir ./demo_metrics_output \
  --steps 30
```

Outputs `metrics_events.jsonl`, `report.json`, `report.md`, `report.html`, etc.

### B) End-to-end acceptance demo

```bash
python -m my_utils.profiling.examples.p0_p13_end_to_end_demo \
  --output-dir ./p0_p13_demo_output \
  --steps 20
```

Optional sqlite probing:

```bash
python -m my_utils.profiling.examples.p0_p13_end_to_end_demo \
  --output-dir ./p0_p13_demo_output \
  --nsys-sqlite ./train_rank0.sqlite
```

### C) CLI with a JSON config (offline)

Single sqlite:

```bash
myutils-profile ingest \
  --config ./my_utils/profiling/examples/collector_config_nsys_sqlite_full.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html
```

Multi-rank glob:

```bash
myutils-profile ingest \
  --config ./my_utils/profiling/examples/collector_config_nsys_multi_rank_full.json \
  --collect-times 1 \
  --analyze \
  --report-formats json,markdown,html
```

### D) One-command framework templates (NSYS/NCU)

```bash
bash my_utils/profiling/examples/framework_playbook_samples/nsys_torchtitan.sh
```

```bash
bash my_utils/profiling/examples/framework_playbook_samples/ncu_generic_wrap.sh -- \
  torchrun --nproc_per_node=8 train.py --config cfg.yaml
```

## Config files

- `collector_config_example.json` — full offline provider example
  (table_csv / ncu_csv / nsys_sqlite / cprofile / perf_stat).
- `collector_config_nsys_sqlite_full.json` — complete template for a single
  sqlite.
- `collector_config_nsys_multi_rank_full.json` — complete template for a
  multi-rank `sqlite_glob`.
- `framework_playbook_samples/*` — copy-paste launch templates for
  TorchTitan, Megatron, DeepSpeed, HF Trainer, VERL, SLIME, ROLL, SGLang,
  and vLLM.

## Common pitfalls

1. `nsys_sqlite` takes `sqlite_path`, not `db_path`.
2. `nsys_sqlite_glob` takes `sqlite_glob`.
3. Files matched by `sqlite_glob` may have any extension, but their content
   must be a real SQLite database.

The demos and offline CLI runs are pure Python — no torch or GPU needed;
only the framework capture templates require a real training environment.

---

Chinese original: [docs/zh/profiling/examples/README.md](../../../docs/zh/profiling/examples/README.md)
