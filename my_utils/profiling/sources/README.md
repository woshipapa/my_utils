# sources

Offline source parsers:
- nsys schema/version/table autodetection (`NsightSchema`)
- nsys SQL skill engine (`NsysSqlSkillEngine`)
- nsys sqlite metrics provider (`NsysSqliteMetricsProvider`)
- nsys iteration detection (`detect_iterations`)
- nsys MFU helpers (`compute_mfu_single`, `infer_peak_tflops`)
- nsys flat export (`export_kernels_flat`)
- nsys analyze/diff helpers (`analyze_nsys_sqlite`, `diff_nsys_sqlite`)
- nsys static timeline html export (`export_timeline_html`)

Provider helper APIs:
- `describe_schema()`
- `list_sql_skills()`
- `describe_sql_skills()`
- `run_sql_skill(name, **params)`
- `analyze_compute_comm_overlap(...)`
- `summarize_gpu_kernels(...)`
- `detect_iterations(...)`
- `compute_mfu(...)`
