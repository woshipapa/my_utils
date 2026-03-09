# Cross-Framework Profiling Reference

This document maps common profiling tools to one canonical metrics model so all
tool outputs can be analyzed in one pipeline.

## Tool Matrix

| Stack | Primary Tool | Collection Entry | Typical Raw Metrics | Canonical Mapping |
|---|---|---|---|---|
| PyTorch | `torch.profiler` | `torch.profiler.profile(...)` | `self_cpu_time_total`, `cuda_time_total`, `*_memory_usage`, `flops` | `latency.op.*`, `memory.op.*`, `compute.op.flops` |
| TensorFlow | TF Profiler + TensorBoard | `tf.profiler.experimental.start/stop` | host/op time, accelerator time, input pipeline stats | `latency.op.*`, `compute.*`, `io.*` |
| JAX | `jax.profiler` + TensorBoard | `jax.profiler.start_trace/stop_trace` | trace events, host/device timeline slices | `latency.stage`, `latency.op.*`, `comm.*` |
| NVIDIA System | Nsight Systems (`nsys`) | `nsys profile ...` + capture window | CUDA kernel duration, memcpy duration, NVTX ranges | `latency.kernel.cuda`, `latency.memcpy.cuda`, `io.memcpy.bytes` |
| NVIDIA Kernel | Nsight Compute (`ncu`) | `ncu --csv ...` | SM throughput, occupancy, memory throughput | `compute.sm.*`, `memory.dram.*` |
| Python runtime | `cProfile` | `cProfile.Profile()` | `tottime`, `cumtime`, call counts | `latency.python.*`, `calls.python` |
| Linux CPU | `perf stat` | `perf stat <cmd>` | cycles, instructions, cache misses, branch misses | `perf.*` |

## Recommended Analysis Flow

1. Keep always-on low-overhead stage metrics:
   - Use `MyTimer` for iteration/stage trend tracking.
2. Use narrow windows for heavy profilers:
   - Trigger `torch.profiler`, `nsys`, `ncu` in short windows around suspicious steps.
3. Normalize all outputs:
   - Convert to `MetricEvent` and canonical names.
4. Run one analyzer:
   - bottleneck share
   - memory peak/growth
   - variance/outliers
5. Export one report format per run:
   - JSON for automation
   - Markdown/HTML for manual review

## Official Docs

- PyTorch Profiler: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- TensorFlow Profiler: https://www.tensorflow.org/guide/profiler
- JAX Profiler: https://docs.jax.dev/en/latest/profiling.html
- Nsight Systems user guide: https://docs.nvidia.com/nsight-systems/UserGuide/
- Nsight Compute profiling guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- Python cProfile: https://docs.python.org/3/library/profile.html
- Linux perf stat: https://man7.org/linux/man-pages/man1/perf-stat.1.html
