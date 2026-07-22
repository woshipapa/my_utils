# tracing

Trace annotation layer; currently centered on an NVTX labeler abstraction with
automatic fallback.

## Quick orientation

1. NVTX available and you want range annotations:
   `create_labeler(preferred="nvtx")`.
2. Code that should run unchanged without NVTX:
   `create_labeler(preferred="auto")` (falls back to a no-op labeler).

## Minimal example

```python
from my_utils.tracing import create_labeler

labeler = create_labeler(preferred="auto")
with labeler.range("forward"):
    # ... forward ...
    pass
```

## Key files

- `nvtx_utils.py` — `LabelerProtocol`, `NoOpLabeler`, `NvtxLabeler`,
  `TorchNvtxLabeler`, `create_labeler`, plus the availability flags
  `NVTX_AVAILABLE` / `TORCH_NVTX_AVAILABLE`.

Neither `nvtx` nor `torch` is required: both backends are optional and the
labeler degrades to a no-op when they are missing.

---

Chinese original: [docs/zh/tracing/README.md](../../docs/zh/tracing/README.md)
