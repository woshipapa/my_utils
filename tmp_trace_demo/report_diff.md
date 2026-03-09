# Profiling Diff Report

- Base: `tmp_trace_demo\run_pretrain\analysis_report.json`
- Target: `tmp_trace_demo\run_inference\analysis_report.json`
- Score delta: `+30.00`

## Finding Delta
- base=4 target=1 delta=-3
- `bottleneck`: 1 -> 1 (delta 0)
- `communication`: 1 -> 0 (delta -1)
- `distributed`: 1 -> 0 (delta -1)
- `memory`: 1 -> 0 (delta -1)
