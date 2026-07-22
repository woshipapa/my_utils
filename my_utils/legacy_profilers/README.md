# legacy_profilers

Compatibility layer for historical profilers, kept so older projects can
migrate smoothly.

## Quick orientation

1. Old projects still using the DITProfiler semantics:
   `create_profiler_context`.
2. Old projects using the raw torch.profiler wrapper: `ProfilerWrapper`
   (optional export; requires torch).
3. New projects: prefer the unified flow in `my_utils.profiling`.

## Key files

- `DITProfiler.py` — `create_profiler_context`.
- `profilerwrapper.py` — `ProfilerWrapper`.

## Migration advice

Leave `legacy_profilers` calls in place at first; migrate the main workflow to
`my_utils.profiling`, then remove the historical calls incrementally.

torch is optional and only needed by the torch.profiler-based paths.

---

Chinese original: [docs/zh/legacy_profilers/README.md](../../docs/zh/legacy_profilers/README.md)
