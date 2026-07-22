# runtime

Runtime integration layer: how training code hooks into profiling.

Note: this subpackage drives live capture, so it expects `torch` (and for NSYS
paths, an `nsys`-capable environment). torch is an optional dependency of
`my_utils` overall — only the capture/runtime paths need it.

## When you touch this package

- Adjusting the framework-agnostic integration helpers (`frameworkless.py`).
- Adding or changing a capture backend.
- Changing runtime configuration structures (`NsysLaunchConfig` and friends).

## Key files

- `frameworkless.py` — the most commonly used runtime helpers, callable
  directly from a training script: `create_nsys_capture_backend`,
  `apply_profiling_environment`, `build_nsys_launch_prefix` (with NSYS
  version detection).
- `config.py` — runtime config dataclasses: `TorchProfilerConfig`,
  `NsysProfilerConfig`, `ProfilingEnvConfig`, `NsysLaunchConfig`.
- `backends.py` — capture backend abstraction (`CaptureBackend`,
  `NoOpBackend`, `CudaProfilerBackend`).
- `capture_controller.py` — capture lifecycle control (`CaptureController`,
  `HookEvent`).
- `ProfileManager.py` — runtime manager (`ProfileManager`).
- `meta_adapters.py` — extract metadata from framework call sites.
- `template_utils.py` — locate the shipped launch templates
  (`get_profiling_templates_dir`, `get_profiling_template_path`).

## Practical advice

Integrate at the `frameworkless.py` level first, so profiling logic stays in
one place instead of being scattered across the training codebase.

---

Chinese original: [docs/zh/profiling/runtime/README.md](../../../docs/zh/profiling/runtime/README.md)
