# distributed

Helpers for distributed training.

## Quick orientation

1. Cross-rank clock alignment: `ClockSynchronizer`.
2. An etcd-based barrier: `etcd_barrier` (optional dependency: `etcd3`).
3. Sequence-parallel padding: `pad_for_sequence_parallel` /
   `remove_pad_by_value` (optional dependency: `megatron.core`).

## Minimal example

```python
from my_utils.distributed import ClockSynchronizer

sync = ClockSynchronizer()
offset = sync.sync_once()
print("clock offset us:", offset)
```

## Key files

- `clockSyncUtils.py` — `ClockSynchronizer`, `SocketClockSynchronizer`.
- `etcd_utils.py` — `etcd_barrier`.
- `pad.py` — sequence-parallel padding helpers.

## Notes

- The etcd and megatron capabilities are optional imports; missing
  dependencies do not affect the rest of the package.
- torch is optional and only needed by the torch-based helpers.

---

Chinese original: [docs/zh/distributed/README.md](../../docs/zh/distributed/README.md)
