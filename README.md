# my_utils

Utilities for profiling, logging, tracing, and debugging PyTorch training or inference workflows.

This README is organized by real class behavior from code, with usage patterns you can copy.

## Installation

From repo root:

```bash
cd my_utils
pip install -e .
```

Optional extras:

```bash
pip install -e .[profiling,tensordict,etcd,nvml,nvtx,system,megatron]
```

Install all extras:

```bash
pip install -e .[all]
```

## Core Workflow: GlobalLogger + MyTimer

This is the main workflow if you want per-stage timing and machine-readable profiling logs.

```python
import time
import torch
from my_utils.logger import GlobalLogger
from my_utils.utils import MyTimer

# 1) Setup logger once per process
logger_mgr = GlobalLogger()
logger_mgr.setup(log_dir="logs/train", rank=0, world_size=1, extra_log_label="Trainer")
logger = logger_mgr.get_logger()

# 2) Build timer and inject logger
timer = MyTimer(use_cuda=torch.cuda.is_available(), tag="train", log_dir="logs/train")
timer.set_logger(logger)

for step in range(3):
    timer.set_step(step)

    timer.start("iter")
    timer.start("forward")
    time.sleep(0.01)
    timer.stop("forward")

    timer.start("backward")
    time.sleep(0.02)
    timer.stop("backward")
    timer.stop("iter")

    # flush CUDA event timings and write machine logs
    timer.step()  # same as timer.synchronize_and_log()
```

### What gets recorded

- CPU time: from `time.perf_counter()`.
- CUDA time: from CUDA events (`cuda_start.elapsed_time(cuda_end)`), if CUDA is enabled.
- Hierarchy: nested `start/stop` are tracked as a tree with `node_id/parent_id`.
- Machine log rows: `GlobalLogger.log_profile_event(...)` writes `START/END` rows into `profile_rank_<rank>.csv`.


这个profile event 可以设计成自定义的chrome trace要记录的信息

CSV columns:

`timestamp_unix, readable_time, machine_id, step, event_name, event_type, duration_ms, metadata`

## MyTimer `start/stop` to trace pipeline

`MyTimer` already writes parseable `START/END` rows. You can convert that CSV to Chrome/Perfetto trace:

```python
import csv
import json
import re

def csv_to_trace(csv_path: str, out_json: str) -> None:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    rows.sort(key=lambda r: float(r["timestamp_unix"]))

    trace_events = []
    for r in rows:
        ts_us = int(float(r["timestamp_unix"]) * 1_000_000)
        step = int(r["step"])
        metadata = r.get("metadata", "")
        node_m = re.search(r"node_id=(\d+)", metadata or "")
        node_id = int(node_m.group(1)) if node_m else None

        event = {
            "name": r["event_name"],
            "cat": "MyTimer",
            "ph": "B" if r["event_type"] == "START" else "E",
            "ts": ts_us,
            "pid": r["machine_id"],      # process lane
            "tid": f"step_{step}",       # thread lane
            "args": {
                "step": step,
                "duration_ms": float(r["duration_ms"] or 0.0),
                "metadata": metadata,
            },
        }
        if node_id is not None:
            event["args"]["node_id"] = node_id
        trace_events.append(event)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": trace_events}, f)

# Example:
# csv_to_trace("logs/train/profile_rank_0.csv", "logs/train/mytimer_trace.json")
```

Then open the JSON in `chrome://tracing` or Perfetto UI.

## Class Reference (by file)

## `my_utils/logger.py`

- `SingletonMeta`
  - Thread-safe singleton metaclass used by `GlobalLogger`.
- `FlushingFileHandler`
  - File handler that flushes on every `emit`, useful for crash-safe logs.
- `GlobalLogger`
  - Central logger manager.
  - `setup(...)`: sets console log, file log (`rank_<rank>.log`), and profile CSV (`profile_rank_<rank>.csv`).
  - `set_time_offset(offset)`: shift profile timestamps (for cross-machine clock sync).
  - `log_profile_event(...)`: append machine-readable profiling event.
  - `get_logger()`: returns configured logger (or fallback logger).

Minimal usage:

```python
from my_utils.logger import GlobalLogger

g = GlobalLogger()
g.setup(log_dir="logs", rank=0, world_size=1)
logger = g.get_logger()
g.log_profile_event(timestamp=0.0, step=0, event_name="warmup", event_type="START")
```

## `my_utils/utils.py`

- `DebugLayer`
  - Identity layer for debugging model graph placement.
- `MyTimer`
  - Hierarchical timer with CPU/CUDA timing, optional NVTX ranges, and profile CSV logging.
  - Key methods:
    - `set_logger(logger)`
    - `start(stage_name, color="blue", domain_name=None)`
    - `stop(stage_name)`
    - `time_stage(stage_name)` context manager
    - `set_step(iteration)`
    - `step()` / `synchronize_and_log()`
    - `register_stage(...)` for NVTX pre-registration
    - `generate_report(stage_pattern, output_filename, iteration_filter=None)`
    - `generate_csv(report_data, csv_filename=...)`
- `NoOpMyTimer`
  - API-compatible no-op timer.
- `DebuggingEvent`
  - `threading.Event` variant that logs stack trace when `.set()` is called.
- `ChecksumUtils`
  - `sign(payload)`: appends per-sample checksum tensors to outgoing payload.
  - `verify(batch, logger)`: validates checksums on receiver side.

## `my_utils/profilerwrapper.py`

- `ProfilerWrapper`
  - Wrapper over `torch.profiler.profile` with optional:
    - summary printing,
    - CUDA average-time recording,
    - memory curve recording,
    - chrome trace export,
    - memory snapshot pickle dump.
  - Use `with ProfilerWrapper(...) as pw:` around workload, then call `pw.leave()` at the end of a run series.

Minimal usage:

```python
from my_utils.profilerwrapper import ProfilerWrapper

pw = ProfilerWrapper(
    is_st=False,
    profile_memory=True,
    enable_print_summary=True,
    enabled_record_cuda_average_time=True,
)

with pw:
    # run one profiled step
    pass

pw.leave()  # export trace/plots if enabled
```

## `my_utils/DITProfiler.py`

- `FlexibleProfiler`
  - Internal context class that runs `torch.profiler.profile`.
  - Exports summary and trace, and can run per-op detailed analysis to JSON.
- `create_profiler_context(...)`
  - Factory context manager controlled by env var `PROFILE_TASK_TYPES`.
  - Enables profiling only when `current_task_type` is listed.

Usage:

```python
import os
from my_utils.DITProfiler import create_profiler_context
from my_utils.logger import get_global_logger

os.environ["PROFILE_TASK_TYPES"] = "DIT,VAE"
logger = get_global_logger()

with create_profiler_context(
    current_task_type="DIT",
    logger=logger,
    ops_to_analyze={"aten::matmul": "self_cuda_time_total"},
):
    pass
```

## `my_utils/ForwardProfileHook.py`

- `ForwardProfilerHook`
  - Attaches to a module forward hook and toggles CUDA profiler between `start_iter` and `stop_iter`.
  - Can limit to specific ranks and optionally push an NVTX range.

Usage:

```python
from my_utils.ForwardProfileHook import ForwardProfilerHook

hook = ForwardProfilerHook(start_iter=10, stop_iter=20, rank_only=[0], nvtx_range="capture_window")
hook.attach(model)
```

## `my_utils/moduleProfiler.py`

- `ModuleProfiler`
  - Registers leaf-module pre/post forward hooks.
  - Measures per-module CUDA time and summarizes into a pandas DataFrame.
  - Works as context manager; call `start()` before forward and `stop()` after synchronize.

Usage:

```python
import torch
from my_utils.moduleProfiler import ModuleProfiler

model = torch.nn.Linear(10, 10).cuda()
x = torch.randn(8, 10, device="cuda")

with ModuleProfiler(model) as mp:
    mp.start()
    _ = model(x)
    torch.cuda.synchronize()
    mp.stop()
    df = mp.summary(output_path="module_summary.csv")
```

## `my_utils/module_hook.py`

- `TraceTensorMeta` (dataclass)
  - Metadata schema for traced tensors (shape/dtype/device/hash/stats).
- `ForwardTraceRecorder`
  - Forward hook recorder for module call events in execution order.
  - Supports:
    - input/output capture,
    - include/exclude filters by module name/type,
    - tensor sampling (`none/head/rand`),
    - per-call param/buffer dtype snapshots,
    - per-call param/buffer MD5 snapshots,
    - save to `.pt`.

Usage:

```python
from my_utils.module_hook import ForwardTraceRecorder

rec = ForwardTraceRecorder(
    model,
    name="forward_trace",
    record_inputs=False,
    record_outputs=True,
    include_name_regex="attn|mlp",
    sample_mode="head",
    sample_max_elems=10000,
)
rec.install()
_ = model(x)
rec.remove()
rec.save("forward_trace.pt", extra={"note": "single step"})
```

## `my_utils/dump_utils.py`

- `DumpTensorIO`
  - Simple tensor dump/load/compare helper keyed by `(dump_id, tag, rank)`.
  - Supports JSONL compare report writing.
- `DumpConfig` (dataclass)
  - Config for dump root directory, truncation limits, tensor save behavior, image format.
- `UniversalDumper`
  - Recursive serializer for python objects.
  - Handles primitives, tensors, numpy arrays, PIL images, dict/list/tuple, and pickle fallback.
- `DumperSingleton`
  - Process-level singleton for `UniversalDumper`.
- `UniversalLoader`
  - Reloads dumps created by `UniversalDumper`.

Usage:

```python
from my_utils.dump_utils import DumpConfig, UniversalDumper, UniversalLoader

cfg = DumpConfig(root_dir="./dump_dataset", enable=True, save_tensor_cpu=True)
dumper = UniversalDumper(cfg)
dump_dir = dumper.dump(stage="after_tokenize", obj={"x": 1}, data_id=42, branch="train")

loader = UniversalLoader(root_dir="./dump_dataset")
obj = loader.load(dump_dir)
```

## `my_utils/memory_snapshot.py`

- `MemorySnapshotter`
  - Nested `start(name)` / `stop(name)` API for CUDA memory history snapshots.
  - Saves snapshot `.pt` files on each stop.
  - Controlled by env var `ENABLE_MEMORY_SNAPSHOT=1`.
- `NoOpMemorySnapshotter`
  - No-op replacement when disabled.
- `global_snapshotter`
  - Global instance created at import time based on env var.

Usage:

```python
import os
os.environ["ENABLE_MEMORY_SNAPSHOT"] = "1"

from my_utils.memory_snapshot import global_snapshotter

global_snapshotter.start("outer")
global_snapshotter.start("inner")
global_snapshotter.stop("inner")
global_snapshotter.stop("outer")
```

## `my_utils/gpu_mem_tracker.py`

- `GPU_Performance_Tracker`
  - Background thread sampler for:
    - torch allocated/reserved memory,
    - optional NVML metrics (GPU util, mem util, power).
  - Generates usage plots and reports peak memory.

Usage:

```python
from my_utils.gpu_mem_tracker import GPU_Performance_Tracker

tracker = GPU_Performance_Tracker(prefix="run", interval=1, track_hardware_metrics=True)
tracker.start_monitoring()
# workload
tracker.stop_monitoring()
```

## `my_utils/clockSyncUtils.py`

- `ClockSynchronizer`
  - NTP-style two-way timestamp sync over `torch.distributed` send/recv.
  - Returns offset that can be applied via `GlobalLogger.set_time_offset(...)`.
- `SocketClockSynchronizer`
  - Same idea over raw TCP socket, independent of torch.distributed.

Usage:

```python
from my_utils.clockSyncUtils import ClockSynchronizer
from my_utils.logger import GlobalLogger

offset = ClockSynchronizer.sync(is_server=False, peer_rank=0, group=None)
GlobalLogger().set_time_offset(offset)
```

## `my_utils/CtrlRandom/control_random.py`

- `ControlRandom`
  - Class-level helper for saving/loading random state payloads into `.pt` files.
  - Main APIs:
    - `save_random(tag, random_data)`
    - `read_database(path=None)`
    - `load_random(tag)`
    - `deal_with_random(save, tag, random_data=None)`

## `my_utils/profiling/backends.py`

- `CaptureBackend`
  - Backend interface (`start`, `stop`).
- `NoOpBackend`
  - Safe no-op backend.
- `CudaProfilerBackend`
  - Calls `torch.cuda.cudart().cudaProfilerStart/Stop` with optional synchronize.

## `my_utils/profiling/capture_controller.py`

- `HookEvent` (dataclass)
  - Event envelope: `profile_name`, `meta`, `role`, `rank`.
- `CaptureController`
  - Window controller that arms capture spec and starts/stops backend on hook enter/exit.
  - Supports:
    - start/stop profile names,
    - iter and microbatch gates,
    - role/rank filters,
    - stop policies (`MANUAL`, `ON_TRIGGER_FUNC_EXIT`, `ON_STOP_PROFILE_NAME`),
    - optional NVTX window markers.

Minimal flow:

```python
from my_utils.profiling.backends import CudaProfilerBackend
from my_utils.profiling.capture_controller import CaptureController, HookEvent

cc = CaptureController(CudaProfilerBackend(), logger=my_logger)
cc.arm({
    "window_id": "w1",
    "start_profile_names": ["generator.forward"],
    "start_iter": 10,
    "start_mb": 0,
    "stop_profile_names": ["generator.forward"],
    "stop_iter": 10,
    "stop_mb": 0,
    "stop_policy": "ON_TRIGGER_FUNC_EXIT",
    "stop_edge": "EXIT",
})

cc.on_enter(HookEvent(profile_name="generator.forward", meta={"iter": 10, "microbatch_index": 0}, role="generator", rank=0))
cc.on_exit(HookEvent(profile_name="generator.forward", meta={"iter": 10, "microbatch_index": 0}, role="generator", rank=0))
```

## `my_utils/profiling/ProfileManager.py`

- `ProfileManager`
  - Driver-side helper to convert YAML/dict capture config into concrete arm specs.
  - Key methods:
    - `enabled()`
    - `should_capture_this_iter(it)`
    - `build_specs_to_arm_at_iter(it, num_microbatches)`
    - `arm_iter(it, num_microbatches, wg_by_role)`

## Utility functions (non-class)

- `my_utils.etcd_utils.etcd_barrier(...)`: etcd-based multi-process barrier.
- `my_utils.ncu_analyze_from_csv.analyze_sm_throughput_from_csv(...)`: summarize NCU CSV by kernel.
- `my_utils.ncu_analyze_from_csv.compare_kernel_metrics(...)`: compare two NCU CSV files.
- `my_utils.pad.pad_for_sequence_parallel(...)` and `remove_pad_by_value(...)`: sequence-parallel padding helpers.
- `my_utils.oom_restore.set_oom_flag()` and `check_oom_flag()`: distributed OOM signal via default store.
- `my_utils.annotations.parametrize_shapes(...)`: decorator to sweep shape and batch-size combinations.

## Output artifacts you should expect

- Logger:
  - `rank_<rank>.log`
  - `profile_rank_<rank>.csv`
- `ProfilerWrapper`:
  - `<prefix>_trace.json`
  - `<prefix>_cuda_time_plot.png`
  - `<prefix>_memory_usage_line_chart.png`
  - optional `<prefix>_mem_<timestamp>.pickle`
- `ModuleProfiler`:
  - optional module summary CSV from `summary(output_path=...)`
- `MemorySnapshotter`:
  - `<name>__<timestamp>__rank<rank>.pt`
- `ForwardTraceRecorder`:
  - trace payload `.pt`
- `GPU_Performance_Tracker`:
  - `<prefix>_rank<rank>_performance_profile.png`
- `UniversalDumper`:
  - `dump_dataset/sample_<id>/<timestamp>__<stage>__<branch>/...`

## Environment variables

- `ENABLE_TIMER=1`: enables global `MyTimer` instance (`global_timer`) in `utils.py`.
- `ENABLE_MEMORY_SNAPSHOT=1`: enables `global_snapshotter`.
- `PROFILE_TASK_TYPES=DIT,VAE`: enables `create_profiler_context` for listed task types.
- `DEBUG_DATA_CONSISTENCY=1`: enables `ChecksumUtils` sign/verify.
- `WAN_DPO_PREVAE_TENSOR_DIR`: root for `DumpTensorIO` tensor files.
- `WAN_DPO_PREVAE_COMPARE_FILE`: optional compare output path template for `DumpTensorIO`.

## Notes

- Several tools assume CUDA and, in some paths, initialized `torch.distributed`.
- For accurate CUDA timings in manual flows, call `timer.step()` after finishing one iteration.
- If you use `GlobalLogger.set_time_offset(...)` with `ClockSynchronizer`, multi-machine CSV timelines can be aligned before trace conversion.

