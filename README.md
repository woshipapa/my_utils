# my_utils

Profiling, logging, and debugging helpers used in PyTorch training loops. This
repository groups small utilities that are handy when working with distributed
training, CUDA profiling, and performance analysis.

## Features

- GlobalLogger: rank-aware logging to stdout, per-rank log files, and CSV
  profile events.
- MyTimer: nested stage timing with optional CUDA events and NVTX ranges.
- ProfilerWrapper: torch.profiler wrapper with summary/trace export and optional
  memory plots.
- DITProfiler: task-gated profiler context with optional per-op analysis.
- ModuleProfiler: per-module forward timing via hooks with pandas summary output.
- GPU_Performance_Tracker: GPU memory tracking and optional NVML hardware
  metrics, with plots.
- MemorySnapshotter: start/stop CUDA memory history snapshots to .pt files.
- ForwardProfilerHook: enable/disable CUDA profiler around target iterations.
- ClockSynchronizer: NTP-style time offset sync (torch.distributed or raw TCP).
- etcd_barrier: etcd-based barrier for multi-process sync.
- CtrlRandom: save/load RNG data for reproducibility.
- ChecksumUtils: checksum helpers for TensorDict batches.

## Installation

From the repo root:

```bash
cd my_utils
pip install -e .
```

Optional extras:

```bash
pip install -e .[profiling,tensordict,etcd,nvml,nvtx,system,megatron]
```

All extras:

```bash
pip install -e .[all]
```

Note: Some tools require a CUDA-capable environment and an initialized
`torch.distributed` process group.

## Dependencies

Core:

- torch
- numpy

Optional:

- pandas (ModuleProfiler, NCU CSV analysis)
- matplotlib (ProfilerWrapper plots, GPU_Performance_Tracker plots)
- tensordict (ChecksumUtils)
- etcd3 (etcd_barrier)
- pynvml (GPU hardware metrics)
- nvtx (MyTimer NVTX ranges)
- psutil (MyTimer memory profiling)
- megatron-core (pad_for_sequence_parallel)

## Quick start

### Global logger + timer

```python
from my_utils import GlobalLogger, MyTimer

global_logger = GlobalLogger()
global_logger.setup("logs", rank=0, world_size=1)

timer = MyTimer(tag="train", use_cuda=True)
timer.set_logger(global_logger.get_logger())

with timer.time_stage("step"):
    # your code
    pass
```

### GPU performance tracker

```python
from my_utils import GPU_Performance_Tracker

tracker = GPU_Performance_Tracker(prefix="run", interval=1)
tracker.start_monitoring()
# your workload
tracker.stop_monitoring()
```

### Torch profiler (task gated)

```python
import os
from my_utils import create_profiler_context, get_global_logger

os.environ["PROFILE_TASK_TYPES"] = "DIT"

logger = get_global_logger()
with create_profiler_context(
    current_task_type="DIT",
    logger=logger,
    ops_to_analyze={"aten::matmul": "self_cuda_time_total"},
):
    # your workload
    pass
```

### Memory snapshots

```python
import os
os.environ["ENABLE_MEMORY_SNAPSHOT"] = "1"

from my_utils.memory_snapshot import global_snapshotter

global_snapshotter.start("phase1")
# your workload
global_snapshotter.stop("phase1")
```

Note: set `ENABLE_MEMORY_SNAPSHOT=1` before importing `global_snapshotter`.

### Module profiler

```python
import torch
from my_utils import ModuleProfiler

model = torch.nn.Linear(10, 10).cuda()
inputs = torch.randn(8, 10, device="cuda")

with ModuleProfiler(model) as profiler:
    profiler.start()
    _ = model(inputs)
    torch.cuda.synchronize()
    profiler.stop()
    df = profiler.summary()
```

### etcd barrier

```python
from my_utils import etcd_barrier

etcd_barrier("job-123", world_size=8, etcd_host="127.0.0.1", etcd_port=2379)
```

## Package layout

- `my_utils/`: core profiling and logging helpers.
- `CtrlRandom/`: RNG save/restore helpers.

## License

MIT (update if you change the license).
