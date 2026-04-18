# Nsight Compute CLI v2026.1.1 参数扫描与分类

更新时间：2026-04-18  
官方来源：<https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html>  
文档版本标识：`v2026.1.1`（页面 `Last updated on Mar 13, 2026`）

## 1) 扫描范围与方法

- 扫描对象：`Command Line Options` 章节的官方表格。
- 处理方式：提取所有长参数名（去除单字符短别名），得到 111 个参数。
- 参数详细用途：已逐项写入 [`ncu_2026_1_1_full_args.yaml`](./ncu_2026_1_1_full_args.yaml) 的注释中。

## 2) 分类索引（111 个）

### A. General / Session / Connection

`help`, `version`, `mode`, `port`, `hostname`, `max-connections`, `config-file`, `config-file-path`, `verbose`, `quiet`, `log-file`, `null-stdin`, `check-exit-code`, `support-32bit`, `injection-path-64`, `preload-library`, `forward-signals`, `kill`

### B. Target Process / Multi-process

`target-processes`, `target-processes-filter`, `devices`, `chips`

### C. Launch Window / Kernel Selection

`profile-from-start`, `disable-profiler-start-stop`, `launch-count`, `launch-skip`, `launch-skip-before-match`, `kernel-name`, `kernel-id`, `kernel-name-base`, `filter-mode`, `range-filter`, `graph-profiling`, `disable-extra-suffixes`, `rename-kernels`, `rename-kernels-path`, `rename-kernels-export`

### D. Replay / Cache / Clock

`replay-mode`, `app-replay-mode`, `app-replay-match`, `app-replay-buffer`, `range-replay-options`, `cache-control`, `clock-control`, `pipeline-boost-state`

### E. Sections / Sets / Metrics / Rules

`set`, `section`, `section-folder`, `section-folder-recursive`, `section-folder-restore`, `list-sets`, `list-sections`, `list-rules`, `list-metrics`, `list-chips`, `metrics`, `metric-distribution-groups`, `query-metrics`, `query-metrics-mode`, `query-metrics-collection`, `rule`, `apply-rules`

### F. NVTX / Lockstep / Callstack Filtering

`nvtx`, `nvtx-include`, `nvtx-exclude`, `nvtx-push-pop-scope`, `lockstep-kernel-launch`, `lockstep-nvtx-include`, `lockstep-nvtx-exclude`, `call-stack`, `call-stack-type`, `native-include`, `native-exclude`, `python-include`, `python-exclude`

### G. PM / Warp Sampling

`pm-sampling-interval`, `pm-sampling-max-passes`, `pm-sampling-buffer-size`, `disable-pm-warp-sampling`, `warp-sampling-interval`, `warp-sampling-max-passes`, `warp-sampling-buffer-size`, `warp-samples-per-interval`

### H. MPS / Communicator

`mps`, `mps-num-clients`, `mps-timeout`, `communicator`, `communicator-tcp-hostname`, `communicator-tcp-port`, `communicator-tcp-num-peers`

### I. File / Import-Export / Source

`export`, `import`, `open-in-ui`, `force-overwrite`, `import-sass`, `import-source`, `resolve-source-file`, `source-folders`

### J. Console / Report Printing

`page`, `csv`, `print-summary`, `print-details`, `print-source`, `print-fp`, `print-units`, `print-kernel-base`, `print-metric-name`, `print-metric-instances`, `print-metric-attribution`, `print-rule-details`, `print-nvtx-rename`

## 3) 建议用法

- 先用 `ncu_quick_launch.yaml` 跑通链路。
- 需要精细控制时切到 `ncu_2026_1_1_full_args.yaml`，按分类改参数。
- 对于不常见参数，直接在 `profile_switches` 中改，避免改脚本代码。
