# Profiling Shell Templates

This folder provides framework-agnostic launcher templates for profiling.

## Files

- `profile_cli_common.sh`: shared shell helpers to load env presets, build training CLI args, and optionally wrap execution with `nsys profile`.
- `preset_nsys_default.env`: common Nsight Systems capture preset.
- `preset_torch_profiler.env`: common torch profiler preset.
- `preset_disabled.env`: disable all profiling by default.

## Typical usage

```bash
source /path/to/my_utils/profiling/templates/profile_cli_common.sh

PROFILE_PRESET=/path/to/my_utils/profiling/templates/preset_nsys_default.env
profile_prepare "$PROFILE_PRESET"

EXEC_CMD=(python path/to/train.py)
profile_wrap_exec_with_nsys EXEC_CMD

CMD=(
  torchrun ... --no_python
  "${EXEC_CMD[@]}"
  "${PROFILE_SETTINGS[@]}"
  --your_framework_args ...
)
"${CMD[@]}"
```

## NSYS output naming

When `NSYS_OUTPUT` is empty, `profile_cli_common.sh` auto-composes output path by:

- base path: `NSYS_OUTPUT_DIR` + `NSYS_OUTPUT_PREFIX`
- step window: `step_<NSYS_START_STEP>_<NSYS_STOP_STEP>`
- capture range: `cap_<NSYS_CAPTURE_RANGE>` (except `none`)
- metrics tag: `with_metrics_<NSYS_GPU_METRICS_DEVICES>` or `no_metrics`
- optional custom suffix: `NSYS_OUTPUT_SUFFIX`

Example output:

`./logs/nsys/nsys_profile_rank_%q{RANK}_step_1_3_cap_cudaProfilerApi_with_metrics_0`

## What the framework must parse

The framework config parser (e.g. tyro, argparse, hydra) must support these argument groups:

- `--torch_profiler.*`
- `--nsys_profiler.*`
- `--profiling_env.*`
- `--nsys_launch.*`

Recommended config objects are in `my_utils.profiling.config`:

- `TorchProfilerConfig`
- `NsysProfilerConfig`
- `ProfilingEnvConfig`
- `NsysLaunchConfig`

At runtime, use helper functions in `my_utils.profiling.frameworkless`:

- `apply_profiling_environment`
- `create_nsys_capture_backend`
- `build_nsys_launch_prefix`
