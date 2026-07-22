# memory

Memory diagnostics: snapshots, OOM signaling, and GPU memory/utilization
tracking.

## Quick orientation

1. Take memory snapshots at key phases: `global_snapshotter`.
2. Propagate an OOM flag across ranks: `set_oom_flag` / `check_oom_flag`.
3. Monitor GPU memory and utilization: `GPU_Performance_Tracker`
   (optional dependency: `pynvml`).

## Minimal example

```python
from my_utils.memory import global_snapshotter, set_oom_flag, check_oom_flag

global_snapshotter.snapshot("before_step")
# ... train step ...
if check_oom_flag():
    print("OOM detected")
```

## Key files

- `memory_snapshot.py` — `MemorySnapshotter`, `NoOpMemorySnapshotter`,
  `global_snapshotter` (falls back to the no-op variant when torch is
  unavailable).
- `oom_restore.py` — OOM flag set/check.
- `gpu_mem_tracker.py` — `GPU_Performance_Tracker`.

## Notes

- Enable monitoring and snapshots on demand; avoid the extra overhead in
  long-running training jobs.
- torch is optional: without it the snapshotter degrades to a no-op.

---

Chinese original: [docs/zh/memory/README.md](../../docs/zh/memory/README.md)
