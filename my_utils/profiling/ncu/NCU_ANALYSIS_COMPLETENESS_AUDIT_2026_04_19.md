# NCU Analysis Completeness Audit (2026-04-19)

## Scope

- Audit target: `my_utils/profiling/ncu` argument templates + `.ncu-rep` analysis logic.
- Goal: verify CLI args coverage and bottleneck-analysis completeness against latest official docs and NVIDIA optimization blogs.

## Sources

- Nsight Compute CLI docs (`v2026.1.1`, page says `Last updated on Mar 13, 2026`):  
  <https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>
- Nsight Compute Profiling Guide:  
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- Nsight Compute Python Report Interface (`rule_results_as_dicts`):  
  <https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html>
- NVIDIA Analysis-Driven Optimization blog (Part 2):  
  <https://developer.nvidia.com/blog/analysis-driven-optimization-analyzing-and-improving-performance-with-nvidia-nsight-compute-part-2/>

## 1) NCU args audit result

- Official `Command Line Options` tables: **111** long options.
- Local full template (`ncu_2026_1_1_full_args.yaml`) official set: **111** options.
- Diff result: **0 missing** official options.

Additional compatibility/prose/deprecated keys (not counted in official 111) were added:

- `communicator-num-peers` (prose/example compatibility alias).
- `communicator-shmem-num-peers` (prose for shmem communicator).
- `details-all` (deprecated; replaced by `print-details=all`).
- `kernel-regex-base` (legacy name referenced in print-kernel-base text).

Doc typo note:

- NVTX examples include `--nvtx-inlcude` typo in one snippet; correct flag is `--nvtx-include`.

## 2) Analysis completeness audit result

Current `ncu_report_tools.py` now combines:

- Official rules from `rule_results_as_dicts` / `rule_results`.
- Fallback heuristics when rule outputs are missing.
- Coverage score for major analysis dimensions.

Coverage dimensions now include:

- Speed-of-light compute/memory.
- Occupancy.
- Scheduler/eligible warps.
- Warp stalls.
- Memory hierarchy.
- Launch stats.
- Source/coalescing signals.
- Control-flow divergence signals.
- Shared-memory bank-conflict signals.
- Tensor/roofline readiness signals.

Added heuristic diagnostics from docs/blog guidance:

- `actual vs ideal global transactions` ratio (`global_memory_coalescing` finding).
- Control-flow divergence detection.
- Shared-memory bank-conflict detection.

## 3) Practical completeness conclusion

- If report collection includes the required sections/metrics, current analyzer can provide:
  - full metric row dump,
  - per-metric statistics,
  - rule-driven bottleneck findings,
  - heuristic fallback bottleneck findings,
  - coverage/missing-category guidance.
- Therefore, the analysis pipeline is now complete for common kernel bottleneck triage, with explicit guardrails when input metrics are insufficient.

## 4) Remaining caveat

- “Analysis complete” still depends on collection quality.
- If `coverage.missing_categories` is non-empty, collect a richer report first (`--set full`, relevant sections, and source/lineinfo when investigating coalescing/divergence).
