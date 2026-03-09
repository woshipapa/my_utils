# Profiling Shell Templates

This folder provides shell templates for both capture-time profiling and offline Nsight SQLite post-processing.

## Files

### Capture templates
- `profile_cli_common.sh`: shared shell helpers to load env presets, build training CLI args, and optionally wrap execution with `nsys profile`.
- `preset_nsys_default.env`: common Nsight Systems capture preset.
- `preset_torch_profiler.env`: common torch profiler preset.
- `preset_disabled.env`: disable all profiling by default.

### Offline NSYS templates
- `nsys_offline_common.sh`: shared helpers for offline scripts (`PYTHONPATH`, file checks, CLI runner).
- `run_nsys_sql_skill.sh`: run one SQL skill or list built-in skills.
- `run_nsys_analyze.sh`: run unified nsys analysis (`summary + overlap + nccl + iterations + mfu`).
- `run_nsys_export.sh`: flat-export kernels timeline to JSON/CSV.
- `run_nsys_diff.sh`: compare two sqlite profiles by kernel/nvtx aggregates.
- `run_nsys_timeline_html.sh`: export static timeline HTML.
- `run_nsys_full_postprocess.sh`: one-shot pipeline to generate all major offline artifacts.

## Quick usage

### A) Capture-time (training launch)

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

### B) Offline sqlite processing

```bash
# Analyze
SQLITE_PATH=/abs/path/to/trace.sqlite \
OUTPUT=./nsys_metrics_out/analyze.json \
/path/to/my_utils/profiling/templates/run_nsys_analyze.sh

# Export kernels
SQLITE_PATH=/abs/path/to/trace.sqlite \
FORMAT=csv \
OUTPUT=./nsys_metrics_out/kernels.csv \
/path/to/my_utils/profiling/templates/run_nsys_export.sh

# SQL skill
SQLITE_PATH=/abs/path/to/trace.sqlite \
SKILL_NAME=top_kernels \
SKILL_PARAMS="device_id=0 limit=20" \
/path/to/my_utils/profiling/templates/run_nsys_sql_skill.sh

# Timeline HTML
SQLITE_PATH=/abs/path/to/trace.sqlite \
OUTPUT=./nsys_metrics_out/timeline.html \
/path/to/my_utils/profiling/templates/run_nsys_timeline_html.sh

# Diff two runs
BEFORE_SQLITE=/abs/path/to/before.sqlite \
AFTER_SQLITE=/abs/path/to/after.sqlite \
OUTPUT=./nsys_metrics_out/diff.json \
/path/to/my_utils/profiling/templates/run_nsys_diff.sh

# One-shot full pipeline
SQLITE_PATH=/abs/path/to/trace.sqlite \
OUT_DIR=./nsys_metrics_out \
OUTPUT_PREFIX=run_a \
/path/to/my_utils/profiling/templates/run_nsys_full_postprocess.sh
```

## Common environment variables (offline)

- Required:
  - `SQLITE_PATH` for single-run scripts.
  - `BEFORE_SQLITE` and `AFTER_SQLITE` for `run_nsys_diff.sh`.
- Basic filters:
  - `DEVICE_ID` (default `-1`, all devices)
  - `START_NS` / `END_NS` (default `-1`, no trim)
  - `LIMIT` (default `500000` for analyze/export)
- Analysis tuning:
  - `TOP_K`, `ITERATION_MARKER`
  - `MODEL_FLOPS_PER_STEP`, `PEAK_TFLOPS`, `PEAK_PRECISION`
- Output:
  - `OUTPUT`, `OUT_DIR`, `OUTPUT_PREFIX`

## NSYS output naming (capture)

When `NSYS_OUTPUT` is empty, `profile_cli_common.sh` auto-composes output path by:

- base path: `NSYS_OUTPUT_DIR` + `NSYS_OUTPUT_PREFIX`
- step window: `step_<NSYS_START_STEP>_<NSYS_STOP_STEP>`
- capture range: `cap_<NSYS_CAPTURE_RANGE>` (except `none`)
- metrics tag: `with_metrics_<NSYS_GPU_METRICS_DEVICES>` or `no_metrics`
- optional custom suffix: `NSYS_OUTPUT_SUFFIX`

Example output:

`./logs/nsys/nsys_profile_rank_%q{RANK}_step_1_3_cap_cudaProfilerApi_with_metrics_0`

## What the framework must parse

The framework config parser (for example tyro/argparse/hydra) should support:

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
