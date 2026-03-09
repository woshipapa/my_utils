# Profiling Analysis Report

- Generated at: `2026-03-08T17:10:26.606834+00:00`
- Schema version: `1.0`
- Overall score: `50.0`
- Total findings: `4`

## Summary
- `total_events`: `40`
- `providers`: `['external_csv', 'synthetic_dist']`
- `metric_count`: `4`
- `workload_profile`: `pretrain`
- `rank_count`: `2`
- `step_min`: `0`
- `step_max`: `3`
- `step_count`: `4`

## Findings
- **HIGH** `bottleneck`: Latency Bottlenecks Detected
  - 2 groups exceed share threshold 10%.
  - data keys: threshold, total_latency_ms, bottlenecks
- **WARNING** `memory`: Memory Growth Analysis
  - Detected memory growth trends across steps.
  - data keys: growth_threshold_bytes_per_step, growth_items, peak_bytes
- **WARNING** `communication`: Communication Imbalance
  - Per-rank communication mean latency ratio is 1.61.
  - data keys: imbalance_ratio_threshold, observed_ratio, rank_mean_latency_ms
- **WARNING** `distributed`: Cross-rank Stage Skew
  - Detected rank skew in aligned stage latency; worst ratio 1.29.
  - data keys: total_aligned_steps, skew_items, has_skew, worst_skew_ratio

## Recommendations
- Drill down top bottleneck groups with torch.profiler/Nsight kernel traces.
- Align bottleneck stages with rank skew and communication overlap before optimization.
- Check tensor/loss/history caches and stale references causing monotonic memory growth.
- Correlate memory growth steps with dataloader or optimizer state transitions.
- Check tensor partitioning and collective payload balance across ranks.
- Inspect straggler ranks with NSYS communication traces and NIC metrics.
- Investigate straggler ranks and rebalance uneven stage workloads.
- Align NCCL timeline with skewed stages to identify communication head-of-line blocking.
