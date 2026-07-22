# artifacts

Offline artifact layer: dumping and reloading intermediate data, plus NCU CSV
analysis helpers.

## Quick orientation

1. Dump tensors / intermediate results to disk:
   `UniversalDumper` / `DumpConfig` (or the process-wide `get_dumper()`).
2. Analyze and compare NCU CSV metrics:
   `analyze_sm_throughput_from_csv` / `compare_kernel_metrics`.

## Minimal example

```python
from my_utils.artifacts import DumpConfig, UniversalDumper

cfg = DumpConfig(output_dir="./dump_out")
dumper = UniversalDumper(cfg)
dumper.dump_tensor("x", x_tensor)
```

## Key files

- `dump_utils.py` — `DumpTensorIO`, `DumpConfig`, `UniversalDumper`,
  `get_dumper` (singleton access); `UniversalLoader` for reading dumps back
  (import it from `my_utils.artifacts.dump_utils`).
- `ncu_analyze_from_csv.py` — NCU CSV metric analysis and comparison.

torch is optional: it is only needed when actually dumping/loading torch
tensors. The CSV analysis helpers are pure Python.

---

Chinese original: [docs/zh/artifacts/README.md](../../docs/zh/artifacts/README.md)
