# torch.compile Reference

This package now keeps `torch.compile` knowledge in two layers:

1. [`../torch_compile_reference.yaml`](../torch_compile_reference.yaml)
   Stable, human-maintained reference for public API, recommended defaults, notable backend options, and common environment variables.
2. [`../torch_compile_catalog.snapshot.yaml`](../torch_compile_catalog.snapshot.yaml)
   Runtime-generated exhaustive catalog derived from the currently installed PyTorch build.
3. [`../torch_compile_catalog_versions.yaml`](../torch_compile_catalog_versions.yaml)
   Version index that distinguishes local runtime-generated catalogs from upstream-source-generated catalogs.

## Why split it this way

`torch.compile` has two very different configuration surfaces:

- Stable public API:
  `fullgraph`, `dynamic`, `backend`, `mode`, `options`, `disable`.
- Version-sensitive internals:
  `torch._dynamo.config`, `torch._inductor.config`, hundreds of backend options, and many env vars.

Putting both into one hand-written file does not age well. The generated snapshot avoids drift when PyTorch is upgraded.

## What is covered

- Official `torch.compile(...)` arguments
- Stable backends from `torch._dynamo.list_backends()`
- Experimental/debug backends from `torch._dynamo.list_backends(None)`
- Mode presets from `torch._inductor.list_mode_options()`
- Full inductor option key list from `torch._inductor.list_options()`
- Source-derived Dynamo config flags
- Source-derived Dynamo/Inductor environment variables
- Documented troubleshooting env vars such as `TORCH_LOGS`, `TORCH_TRACE`, and `TORCH_COMPILE_DEBUG`

## Refreshing the snapshot

From the repo root:

```bash
python third_party/my_utils/my_utils/profiling/generate_torch_compile_catalog.py
python third_party/my_utils/my_utils/profiling/generate_torch_compile_catalog.py --catalog-kind latest-upstream
```

This regenerates:

- [`../torch_compile_catalog.snapshot.yaml`](../torch_compile_catalog.snapshot.yaml)
- [`../torch_compile_catalog.torch-<local-version>.snapshot.yaml`](../torch_compile_catalog.snapshot.yaml)
- [`../torch_compile_catalog.latest-upstream.snapshot.yaml`](../torch_compile_catalog.latest-upstream.snapshot.yaml)
- [`../torch_compile_catalog.upstream-<latest-version>.snapshot.yaml`](../torch_compile_catalog.latest-upstream.snapshot.yaml)
- [`../torch_compile_catalog_versions.yaml`](../torch_compile_catalog_versions.yaml)

## Recommended usage in framework configs

- Expose only stable public arguments in experiment configs by default.
- Treat `options` as an advanced escape hatch.
- Prefer `dynamic: null` as the framework default for training.
- Use the generated snapshot when you need to confirm exact option or env-var spellings for a specific PyTorch version.

## Primary references

- PyTorch API: <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- Dynamic Shapes Guide: <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html>
- Troubleshooting Guide: <https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html>
