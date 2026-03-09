# Profiling Analysis Report

- Generated at: `2026-03-08T16:48:23.395512+00:00`
- Schema version: `1.0`
- Overall score: `77.0`
- Total findings: `2`

## Summary
- `total_events`: `25`
- `providers`: `['external_csv', 'ncu_csv', 'synthetic_train']`
- `metric_count`: `4`
- `workload_profile`: `default`
- `rank_count`: `0`
- `step_min`: `0`
- `step_max`: `4`
- `step_count`: `5`

## Findings
- **HIGH** `bottleneck`: Latency Bottlenecks Detected
  - 2 groups exceed share threshold 10%.
  - data keys: threshold, total_latency_ms, bottlenecks
- **INFO** `memory`: Memory Growth Analysis
  - Memory metrics collected.
  - data keys: growth_threshold_bytes_per_step, growth_items, peak_bytes

## Recommendations
- Drill down top bottleneck groups with torch.profiler/Nsight kernel traces.
- Align bottleneck stages with rank skew and communication overlap before optimization.
