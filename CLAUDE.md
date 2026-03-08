# CLAUDE.md — AI Assistant Guide for `my_utils`

This file provides essential context for AI assistants (Claude Code and similar tools) working with this repository.

---

## Project Overview

**`my_utils`** is a PyTorch training/inference performance analysis toolkit. It provides profiling, timing, tracing, GPU memory monitoring, and debugging utilities designed for production-scale distributed ML training.

**Language**: Python 3.8+
**License**: Apache 2.0
**Version**: 0.1 (active development)

---

## Repository Structure

```
my_utils/
├── setup.py                          # Package installation
├── pyproject.toml                    # Build system config
├── README.md                         # Comprehensive usage docs (1100+ lines)
├── README_IMPROVEMENTS.md            # Docs improvement notes
│
├── CtrlRandom/                       # Random state control for reproducibility
│   ├── __init__.py
│   ├── control_random.py
│   └── set_seeds.py
│
└── my_utils/                         # Main package
    ├── __init__.py                   # Public API exports (source of truth for public API)
    ├── utils.py                      # Core: MyTimer, NoOpMyTimer (~1425 lines)
    ├── logger.py                     # GlobalLogger singleton with CSV profiling
    ├── nvtx_utils.py                 # NVTX labeling (NvtxLabeler, TorchNvtxLabeler)
    ├── moduleProfiler.py             # Per-module CUDA timing hooks
    ├── profilerwrapper.py            # PyTorch profiler wrapper with visualization
    ├── gpu_mem_tracker.py            # Background GPU metrics monitor
    ├── memory_snapshot.py            # PyTorch CUDA memory snapshots
    ├── clockSyncUtils.py             # NTP/socket-based distributed clock sync
    ├── DITProfiler.py                # FlexibleProfiler: regex-based op analysis
    ├── ForwardProfileHook.py         # Auto profiler control via forward hooks
    ├── dump_utils.py                 # Tensor dump/load with checksums
    ├── module_hook.py                # ForwardTraceRecorder for module tracing
    ├── method_patch.py               # Runtime method patching utilities
    ├── step_timer.py                 # NEW: Per-step wall-clock timer + anomaly detection
    ├── annotations.py                # Shape parametrization
    ├── etcd_utils.py                 # Distributed synchronization via etcd
    ├── ncu_analyze_from_csv.py       # NVIDIA Nsight Compute CSV analysis
    ├── oom_restore.py                # OOM recovery utilities
    ├── pad.py                        # Sequence parallel padding
    │
    └── profiling/                    # Advanced profiling subsystem
        ├── __init__.py               # Subpackage exports
        ├── ProfileManager.py         # YAML-driven profiling config manager
        ├── capture_controller.py     # Window-based capture coordination
        ├── backends.py               # Profiler backends (Nsys, NoOp)
        ├── config.py                 # Config dataclasses
        ├── frameworkless.py          # Framework-agnostic profiling helpers
        ├── meta_adapters.py          # Metadata extraction
        ├── template_utils.py         # Utility functions
        ├── trace_aggregator.py       # NEW: Merge per-rank CSVs → Chrome Trace JSON
        ├── profile.yaml              # Main profiling config example
        ├── nsys_profile.yaml         # Nsight Systems config example
        ├── profile_wrapper.sh        # Universal Nsight Systems wrapper script
        ├── nsys_torchrun_template.sh # Torchrun + Nsight integration template
        └── templates/
            ├── README.md
            ├── profile_cli_common.sh      # Reusable shell profiling functions
            ├── preset_nsys_default.env    # Nsight Systems preset
            ├── preset_torch_profiler.env  # PyTorch profiler preset
            └── preset_disabled.env        # No-op preset
```

---

## Core Abstractions

### 1. `MyTimer` (`utils.py`)
Hierarchical CPU/CUDA timer with nested event tracking. Central to most profiling workflows.
- Use `start(name)` / `stop(name)` or as context manager
- Supports `set_time_offset()` for multi-machine synchronization
- `NoOpMyTimer` is the zero-overhead drop-in for disabled profiling

### 2. `GlobalLogger` (`logger.py`)
Singleton logging system (via `SingletonMeta`) that writes both human-readable logs and machine-readable CSV profile events.
- Thread-safe singleton — always access via `GlobalLogger()`
- Outputs: `rank_<rank>.log` and `profile_rank_<rank>.csv`
- CSV format: `timestamp_unix,readable_time,machine_id,step,event_name,event_type,duration_ms,metadata`

### 3. NVTX Labeling (`nvtx_utils.py`)
- `NvtxLabeler`: Standalone NVTX range marking
- `TorchNvtxLabeler`: PyTorch-specific backend
- `create_labeler()`: Factory — auto-selects backend based on availability
- `LabelerProtocol`: Abstract interface — implement this for custom backends

### 4. `StepTimer` (`step_timer.py`) — NEW
Framework-agnostic per-step wall-clock timer. Zero external dependencies.
- Context manager API: `with timer.step(step_idx): ...`
- Rolling statistics: mean, std, p50/p95/p99
- Automatic anomaly detection (flags steps N× slower than the rolling mean)
- `timer.summary()` — formatted summary table
- `timer.throughput(batch_size)` — samples/second

### 5. `ProfileManager` (`profiling/ProfileManager.py`)
YAML-driven profiling config manager. Reads `profile.yaml` to configure profiling windows, step ranges, and feature flags. Integrates with `CaptureController`.

### 6. `CaptureController` (`profiling/capture_controller.py`)
Window-based capture coordination. Controls when profiling capture is active (by step count, time window, etc.). Works with pluggable `CaptureBackend` implementations.

### 7. `TraceAggregator` (`profiling/trace_aggregator.py`) — NEW
Post-training trace merging tool. Converts per-rank `profile_rank_*.csv` files (from `GlobalLogger`) into a single Chrome Trace JSON for Perfetto / chrome://tracing.
- `agg.load_directory(log_dir)` — scan and load all rank CSVs
- `agg.set_clock_offset(rank, offset_seconds)` — apply `ClockSynchronizer` offsets
- `agg.export_chrome_trace(output_path)` — emit merged JSON
- `agg.rank_summary()` — per-rank event count and time span

---

## Key Design Patterns

| Pattern | Where Used |
|--------|-----------|
| Singleton | `GlobalLogger` with `SingletonMeta` |
| Factory | `create_labeler()`, `create_profiler_context()`, `create_nsys_capture_backend()` |
| Context Manager | `ProfilerWrapper`, `FlexibleProfiler`, `ModuleProfiler`, `labeler.range()`, `StepTimer.step()` |
| Protocol/Interface | `LabelerProtocol`, `CaptureBackend` — use `@runtime_checkable` |
| Hook Pattern | `ModuleProfiler._register_hooks()`, `ForwardProfilerHook`, `ForwardTraceRecorder` |
| Dataclass Config | `TorchProfilerConfig`, `NsysProfilerConfig`, etc. in `profiling/config.py` |
| Lazy Loading | Optional dependencies wrapped in `try/except` at import time |

---

## Public API

The canonical public API is defined in `my_utils/__init__.py`. **Always check this file before assuming a class/function is accessible** — not everything in submodules is re-exported.

Key exports include:
- `MyTimer`, `NoOpMyTimer`
- `GlobalLogger`
- `StepTimer`, `StepStats`, `AnomalyRecord`
- `NvtxLabeler`, `TorchNvtxLabeler`, `create_labeler`, `LabelerProtocol`
- `ModuleProfiler`, `ProfilerWrapper`, `FlexibleProfiler`
- `MemorySnapshotter`, `GPU_Performance_Tracker`
- `ClockSynchronizer`, `SocketClockSynchronizer`
- `DumpTensorIO`, `UniversalDumper`, `ChecksumUtils`
- `ForwardTraceRecorder`, `MethodPatcher`
- `ProfileManager`, `CaptureController` (from `my_utils.profiling`)
- `TraceAggregator`, `RankStat` (from `my_utils.profiling`)

---

## Typical Workflows

### Single-machine training profiling
```python
from my_utils import StepTimer, GlobalLogger, MyTimer

logger = GlobalLogger()
logger.setup(log_dir="./logs", rank=rank, world_size=world_size)

timer = StepTimer(window_size=50, rank=rank)
for step in range(max_steps):
    with timer.step(step):
        train_step(batch)

if rank == 0:
    print(timer.summary())
```

### Multi-machine trace correlation
```python
# After training: merge per-rank CSVs with clock correction
from my_utils.profiling import TraceAggregator

agg = TraceAggregator()
agg.load_directory("./logs")

# offsets from ClockSynchronizer (rank 1 was 3 ms ahead of rank 0)
agg.set_clock_offsets({1: -0.003, 2: -0.001})

agg.export_chrome_trace("./logs/merged_trace.json")
print(agg.rank_summary())
# Open merged_trace.json at https://ui.perfetto.dev
```

### NVTX + Nsight Systems
```python
from my_utils import create_labeler, CaptureController

labeler = create_labeler()
with labeler.range("forward"):
    output = model(input)
```

---

## Dependencies

### Required
```
torch
numpy
```

### Optional (install with extras)
```
pandas, matplotlib    → pip install my_utils[profiling]
tensordict            → pip install my_utils[tensordict]
etcd3                 → pip install my_utils[etcd]
pynvml                → pip install my_utils[nvml]
nvtx                  → pip install my_utils[nvtx]
psutil                → pip install my_utils[system]
megatron-core         → pip install my_utils[megatron]
```

All optional dependencies are loaded lazily — missing ones cause graceful degradation, not import errors.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ENABLE_TIMER` | Activate global `MyTimer` instance |
| `ENABLE_NVTX` | Enable NVTX labeling |
| `ENABLE_MEMORY_SNAPSHOT` | Enable `torch.cuda` memory snapshots |
| `PROFILE_TASK_TYPES` | Comma-separated task types (e.g., `"DIT,VAE"`) |
| `DEBUG_DATA_CONSISTENCY` | Enable `ChecksumUtils` validation |
| `WAN_DPO_PREVAE_TENSOR_DIR` | Root directory for tensor dumps |
| `WAN_DPO_PREVAE_COMPARE_FILE` | Comparison output path template |
| `NSYS_START_STEP` | Nsight Systems capture start step |
| `NSYS_STOP_STEP` | Nsight Systems capture stop step |
| `NSYS_OUTPUT_DIR` | Nsight output directory (default: `./logs/nsys`) |
| `NSYS_OUTPUT_PREFIX` | Nsight output file prefix |

---

## Coding Conventions

### Style
- **Classes**: CamelCase (e.g., `MyTimer`, `GlobalLogger`, `ProfilerWrapper`)
- **Functions/variables**: snake_case
- **Constants/env vars**: ALL_CAPS
- **Type hints**: Use throughout; required for new public API additions
- **Comments**: Mix of English and Chinese is acceptable in existing code; prefer English for new code

### Patterns to Follow
- Wrap optional imports in `try/except ImportError` — never make optional deps required
- Use `@dataclass` for configuration objects
- New profiling backends should implement `CaptureBackend` protocol
- New labelers should implement `LabelerProtocol`
- Add new public exports to `my_utils/__init__.py`
- New modules with no external deps (like `step_timer.py`) go into the top-level `my_utils/` package
- New modules that depend on the profiling subsystem (like `trace_aggregator.py`) go in `my_utils/profiling/`

### Patterns to Avoid
- Do not import optional dependencies at module level without a try/except guard
- Do not use `torch.distributed` in code paths that should work on single-GPU setups
- Do not add hard synchronization (`torch.cuda.synchronize()`) outside of timer boundaries — it has performance implications
- Do not use `sys.exit()` inside library code — raise exceptions instead

---

## Output Artifacts

| Artifact | Format | Source |
|----------|--------|--------|
| `rank_<rank>.log` | Text | `GlobalLogger` |
| `profile_rank_<rank>.csv` | CSV (START/END rows) | `GlobalLogger` |
| `merged_trace.json` | Chrome Trace JSON | `TraceAggregator` |
| `<prefix>_trace.json` | Chrome trace | `ProfilerWrapper` |
| `<prefix>_cuda_time_plot.png` | PNG | `ProfilerWrapper` |
| `<prefix>_memory_usage_line_chart.png` | PNG | `ProfilerWrapper` |
| `<name>__<timestamp>__rank<rank>.pt` | PyTorch | `MemorySnapshotter` |
| Module timing CSV | CSV | `ModuleProfiler.summary()` |

---

## Testing

There is currently **no test suite**. When writing new code:
- Add docstring examples for simple utilities
- For context managers, verify `__enter__`/`__exit__` behavior manually
- Validate optional-dependency guards by testing with and without optional packages installed

If adding tests in the future, use `pytest` as the framework.

---

## Build & Installation

```bash
# Standard install
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install specific extras
pip install -e ".[profiling,nvtx,nvml]"
```

No Makefile, CI/CD, or pre-commit hooks are currently configured.

---

## Git Workflow

- **Main branch**: `master`
- **Feature branches**: `claude/<description>` or descriptive names
- Commit messages are short and imperative (e.g., `"add standalone NVTX labeler"`)
- Documentation updates (README.md) should accompany any new tool or API addition

---

## Known Issues / Improvement Areas

### Bugs Fixed
- `CtrlRandom/control_random.py`: `endwith` typo fixed to `endswith`; `load_random` key mismatch also fixed (was looking up `"tag.pt"` but dict stored under `"tag"`)

### Remaining Known Issues
- `profilerwrapper.py` line 29: `dist.get_rank()` called unconditionally at init — will crash if `torch.distributed` is not initialized; consider making rank optional
- `profilerwrapper.py` line 135: `sys.exit(1)` inside `record_function()` — kills the training process on OOM; should raise instead
- `moduleProfiler.py` line 73: `torch.cuda.synchronize()` is a hard serialization point; use CUDA events for async timing
- `pad.py`: Megatron dependency not guarded with `try/except`; crashes on import if megatron is not installed
- `etcd_utils.py`: etcd host/port is hardcoded with no env var override
- `DITProfiler.py` line 110: `re.match()` only matches at string start; should use `re.search()` for substring matching

### Missing Features (Future Work)
| Area | Gap | Suggested approach |
|------|-----|--------------------|
| Distributed tracing | No automatic collective communication tracing | Hook `dist.all_reduce` etc. via `MethodPatcher` |
| Bottleneck detection | No compute/communication imbalance analysis | Analyze `MyTimer` CSV: compare compute vs comm event durations |
| Real-time monitoring | No live dashboard | Integrate with TensorBoard `SummaryWriter` |
| Log rotation | `GlobalLogger` CSV can grow unboundedly | Add `max_size_mb` / `rotate_every_n_steps` option |
| Async profiling | All profiling hooks are synchronous | Offload write I/O to a background thread |
| Distributed aggregation | `StepTimer` and `MyTimer` stats are per-process | Add `aggregate_from_ranks()` using `dist.all_gather` |

---

## Module Dependency Map

```
my_utils (public API)
├── step_timer.py          ← zero deps (stdlib only)
├── logger.py              ← stdlib only
├── utils.py               ← torch, numpy
├── nvtx_utils.py          ← optional: nvtx, torch.cuda.nvtx
├── clockSyncUtils.py      ← torch.distributed OR socket (no dist)
├── dump_utils.py          ← torch, optional: tensordict
├── module_hook.py         ← torch
├── moduleProfiler.py      ← torch, optional: pandas
├── profilerwrapper.py     ← torch, matplotlib, torch.distributed
├── gpu_mem_tracker.py     ← torch, optional: pynvml, matplotlib
├── memory_snapshot.py     ← torch (private API)
├── DITProfiler.py         ← torch (regex-based)
└── profiling/
    ├── trace_aggregator.py ← stdlib only (json, glob, re)
    ├── ProfileManager.py  ← yaml, torch
    ├── capture_controller.py ← torch, nvtx_utils
    └── backends.py        ← torch.cuda
```
