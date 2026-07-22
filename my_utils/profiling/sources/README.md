# sources (NSYS SQLite offline parsing)

Reads the SQLite files produced by `nsys export` and runs reusable analyses
over them. Everything here is offline — no torch or GPU required.

## Quick orientation

1. Overall training analysis: `myutils-profile nsys-analyze`.
2. Run a single SQL skill: `myutils-profile nsys-sql-skill`.
3. Export kernel details: `myutils-profile nsys-export`.
4. Compare two profiles: `myutils-profile nsys-diff`.
5. Produce a timeline HTML page: `myutils-profile nsys-timeline-html`.

## Most-used commands

```bash
# unified analysis
myutils-profile nsys-analyze --sqlite ./train_rank0.sqlite --output ./nsys_analyze.json

# list SQL skills
myutils-profile nsys-sql-skill --sqlite ./train_rank0.sqlite --list-skills --pretty

# run one skill
myutils-profile nsys-sql-skill \
  --sqlite ./train_rank0.sqlite \
  --skill top_kernels \
  --param device_id=0 \
  --param limit=20 \
  --pretty

# export kernel details
myutils-profile nsys-export --sqlite ./train_rank0.sqlite --format csv --output ./kernels.csv

# before/after comparison
myutils-profile nsys-diff --before-sqlite ./a.sqlite --after-sqlite ./b.sqlite --output ./diff.json

# timeline html
myutils-profile nsys-timeline-html --sqlite ./train_rank0.sqlite --output ./timeline.html
```

## Key files (by responsibility)

- `nsys_schema_adapter.py` — cross-version schema detection (adapts to
  table/column differences between nsys exports).
- `nsys_sql_skills.py` — built-in SQL skill engine (top kernels, overlap,
  nvtx, memcpy, occupancy, ...).
- `nsys_sqlite_provider.py` — provider wrapper for the unified metrics
  pipeline.
- `nsys_analyze.py` — one-stop aggregation (summary/overlap/nccl/
  iterations/mfu).
- `nsys_auto_analysis.py` — automated analysis flow on top of the above.
- `nsys_diff.py` — before/after diff analysis.
- `nsys_flat_export.py` — flat kernel-timeline export to json/csv.
- `nsys_timeline_html.py` — static HTML timeline export.
- `nsys_iterations.py` — iteration splitting from NVTX markers.
- `nsys_mfu.py` — MFU-related helper computation.
- `nsys_module_kernel_compare.py` — module-level kernel comparison.
- `kernel_taxonomy.py` — kernel-name classification.

## Debugging tips

1. Run the `schema_inspect` skill first to confirm the sqlite is recognized.
2. Then `nsys-analyze` for the overview.
3. On a regression, reach for `nsys-diff` first.
4. Export `timeline.html` only when you need visual localization.

---

Chinese original: [docs/zh/profiling/sources/README.md](../../../docs/zh/profiling/sources/README.md)
