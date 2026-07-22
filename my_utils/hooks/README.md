# hooks

Observation and scoped profiling control built on PyTorch hooks.

torch is required to actually use these hooks (it is an optional dependency of
`my_utils` overall — only capture/runtime paths like this one need it).

## Quick orientation

1. Trigger profiler start/stop on a training window: `ForwardProfilerHook`.
2. Record module-level forward input/output info: `ForwardTraceRecorder`.
3. Module-level timing statistics (legacy approach): `ModuleProfiler`
   (optional export).

## Minimal example

```python
from my_utils.hooks import ForwardTraceRecorder

recorder = ForwardTraceRecorder()
recorder.register(model)
# after forward, read the recorded results from `recorder`
```

## Key files

- `ForwardProfileHook.py` — `ForwardProfilerHook`.
- `module_hook.py` — `ForwardTraceRecorder`.
- `moduleProfiler.py` — `ModuleProfiler`.

## Compatibility

- The legacy import paths `my_utils.ForwardProfileHook` and
  `my_utils.module_hook` still work.

---

Chinese original: [docs/zh/hooks/README.md](../../docs/zh/hooks/README.md)
