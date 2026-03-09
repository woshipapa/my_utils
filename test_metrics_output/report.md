# Profiling Analysis Report

- Generated at: `2026-03-08T15:12:53.373939+00:00`
- Total findings: `2`

## Summary
- `total_events`: `50`
- `providers`: `['external_csv', 'ncu_csv', 'synthetic_train']`

## Findings
- **HIGH** `bottleneck`: Latency Bottlenecks Detected
  - Detected 2 latency bottlenecks above 10%.
  - data keys: bottlenecks, total_latency_ms
- **INFO** `memory`: Memory Profile
  - Memory metrics collected.
  - data keys: peak_bytes, growth_warnings, growth_threshold_bytes_per_step

## Recommendations
- Use operator-level tracing (torch.profiler or Nsight) on top bottleneck groups for root-cause isolation.
- Correlate memory peaks with stage/op tags and verify whether growth is monotonic across steps.
