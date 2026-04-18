# Profiling Shell Templates

This folder provides shell templates for both capture-time profiling and offline Nsight SQLite post-processing.

## Files

### Capture templates
- `profile_cli_common.sh`: shared shell helpers to load env presets, build training CLI args, and optionally wrap execution with `nsys profile`.
- `preset_nsys_default.env`: common Nsight Systems capture preset.
- `preset_torch_profiler.env`: common torch profiler preset.
- `preset_disabled.env`: disable all profiling by default.
- `run_nsys_quick.sh`: quickest wrapper to run any command with version-compatible `nsys profile`.
- `run_nsys_quick_yaml.py`: YAML-driven one-click wrapper for `nsys profile`.
- `nsys_quick_launch.yaml`: minimal YAML template for `run_nsys_quick_yaml.py`.
- `nsys_2026_2_full_args.yaml`: full v2026.2 profile switches template with per-arg comments.

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

Fastest path:

```bash
/path/to/my_utils/profiling/templates/run_nsys_quick.sh -- python train.py --config cfg.yaml
```

Or with explicit compatibility overrides:

```bash
NSYS_NIC_METRICS_MODE=lf \
NSYS_SYSCALL=process-tree \
/path/to/my_utils/profiling/templates/run_nsys_quick.sh -- python train.py
```

Framework-integrated path:

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

YAML one-click path:

```bash
python /path/to/my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config /path/to/my_utils/profiling/templates/nsys_quick_launch.yaml
```

YAML with command override:

```bash
python /path/to/my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config /path/to/profile.yaml \
  -- torchrun --nproc_per_node=8 --no_python python train.py --config cfg.yaml
```

The YAML launcher supports:

- `env`: env var overrides (for example `CUDA_VISIBLE_DEVICES`, `NCCL_DEBUG`).
- `command`: string or argv list.
- `nsys_launch`: all `NsysLaunchConfig` fields.
- `nsys_launch.profile_switches`: official switch map (`key -> value`) for full-arg configuration.
- `nsys_launch.extra_profile_args`: raw extra nsys switches for quick experiments.

Full-args template:

```bash
python /path/to/my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config /path/to/my_utils/profiling/templates/nsys_2026_2_full_args.yaml
```

`profile_switches` value conventions:

- `null`: do not pass this switch.
- scalar (`str/int/float/bool`): converted to `--key=value`.
- list: converted to repeated `--key=item`.
- `"__flag__"`: converted to bare `--key` (for flag-style switches such as `--help`).

Manual stop mode (recommended for some distributed capture-range cases):

```bash
# 1) Start with session + capture-range-end=none
python /path/to/my_utils/profiling/templates/run_nsys_quick_yaml.py \
  --config /path/to/profile.yaml \
  -- torchrun --nproc_per_node=8 --no_python python train.py

# profile.yaml snippet:
# nsys_launch:
#   enabled: true
#   capture_range: cudaProfilerApi
#   capture_range_end: none
#   extra_profile_args:
#     - --session-new=my_train_sess
#     - --flush-on-cudaprofilerstop=false
#
# 2) Stop from another shell when your target training range is done
# nsys stop --session=my_train_sess
```

Notes:

- `capture_range` still controls start (for example by `cudaProfilerStart()`).
- `capture_range_end=none` means capture does not end on `cudaProfilerStop()` automatically.
- `nsys stop --session=<name>` ends the session and flushes report output.

### A2) Framework launcher compatibility

| Framework | Official launch style (summary) | Direct wrapping command |
|---|---|---|
| Megatron-LM | `torchrun ... pretrain_gpt.py ...` | `run_nsys_quick.sh -- torchrun ... --no_python python pretrain_gpt.py ...` |
| DeepSpeed | `deepspeed --num_gpus=... train.py ...` | `run_nsys_quick.sh -- deepspeed --num_gpus=... train.py ...` |
| Hugging Face Trainer | `accelerate launch ... train.py ...` or `deepspeed ...` | `run_nsys_quick.sh -- accelerate launch ... train.py ...` |
| VERL | `python -m verl.trainer.main_ppo ...` | `run_nsys_quick.sh -- python -m verl.trainer.main_ppo ...` |
| SLIME | `ray job submit ... -- python3 train.py ...` | `run_nsys_quick.sh -- ray job submit ... -- python3 train.py ...` |

Compatibility note:

- The wrapper is launcher-agnostic. It prepends `nsys profile ...` to any final executable command.
- For distributed `torchrun`, keep `--no_python` when the wrapped executable starts with `python ...`.
- For Ray/SLIME, wrap the `ray job submit` CLI itself on the submit node. For per-worker traces in a cluster, inject the wrapper in the worker startup command.

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

## Nsys version-aware fallback

`build_nsys_launch_prefix` now performs best-effort version-aware switch selection:

- For `nsys >= 2026`: `nic_metrics=true` maps to `--nic-metrics=lf`.
- For older `nsys`: `nic_metrics_mode=lf|hf` falls back to `--nic-metrics=true`.
- For `nsys >= 2026`: `trace` containing `syscall` is translated to `--syscall=process-tree` (or explicit `nsys_launch.syscall`).
- Legacy field `gpu_metrics_device` is still accepted and mapped to `--gpu-metrics-devices`.
- Optional override `nsys_launch.version_hint` (example: `2026.2`) can force behavior when runtime detection is unavailable.

Shell template variables:

- `NSYS_NIC_METRICS_MODE` (preferred: `lf|hf|none`)
- `NSYS_SYSCALL` (preferred for `nsys>=2026`)
- `NSYS_VERSION_HINT` (optional override for version detection)
