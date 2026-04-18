# Nsight Systems 2026.2 CLI Profile 快速参考

本文用于替换仓库内以 `2024.7.1` 为基准的参数说明，统一对齐到 NVIDIA 官方 `v2026.2` 文档。

## 1) 版本与来源

- 当前对齐版本: **Nsight Systems v2026.2**
- 对比基线: **Nsight Systems 2024.7.1**
- 官方来源: `https://docs.nvidia.com/nsight-systems/UserGuide/index.html`
- 本文更新时间: `2026-04-18`

## 2) 项目代码扫描结果（nsys profile 实际使用参数）

下列参数来自仓库内实际组装 `nsys profile` 命令的入口:

- `my_utils/profiling/nsys_torchrun_template.sh`
- `my_utils/profiling/profile_wrapper.sh`
- `my_utils/profiling/templates/profile_cli_common.sh`
- `my_utils/profiling/runtime/frameworkless.py`

当前代码路径实际使用到的 profile 参数:

```text
--capture-range
--capture-range-end
--cuda-memory-usage
--cudabacktrace
--export
--force-overwrite
--gpu-metrics-devices
--gpu-metrics-frequency
--nic-metrics
--nvtx-capture
--nvtx-domain-exclude
--nvtx-domain-include
--osrt-threshold
--output
--python-sampling
--python-sampling-frequency
--sample
--show-output
--stats
--trace
```

说明:

- 参数集合与 `v2026.2` 兼容。
- `--nic-metrics=true|false` 在 `v2026.2` 中仍可兼容，但官方标记 `true` 为 deprecated（建议迁移到 `lf|hf|none`）。
- 仓库历史文案中的 `--gpu-metrics-device`（单数）已在本次更新统一为 `--gpu-metrics-devices`。

## 3) 训练任务推荐起步参数（2026.2）

```bash
nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --sample=none \
  --export=none \
  --output=./logs/nsys/train_%p \
  python train.py
```

需要扩展时再按需加:

- 硬件指标: `--gpu-metrics-devices=all`
- 网络指标: `--nic-metrics=lf`
- NVTX 触发: `--capture-range=nvtx --nvtx-capture=<range[@domain]>`

## 4) 2024.7.1 -> 2026.2 参数 Diff

### 4.1 新增参数（2026.2 相对 2024.7.1）

```text
--after-collection-start
--after-report-ready
--cpu-metrics
--cpu-socket-metrics
--cuda-event-trace
--cuda-trace-all-apis
--cuda-trace-scope
--dask
--debug-symbols
--discard-environment
--dx-force-declare
--event-sampling
--event-sampling-interval
--event-sampling-multiplex-interval
--flush-on
--gds-libs-path
--gds-metrics
--ib-switch-congestion
--ib-switch-congestion-devices
--ib-switch-metrics
--ib-switch-metrics-devices
--ib-switch-metrics-nic
--ib-switch-metrics-nic-device
--osrt-backtrace
--osrt-file-access
--pytorch
--qnx-kernel
--syscall
--wddm-memory-trace
```

### 4.2 移除参数（2024.7.1 有，2026.2 参数表中不再作为独立项）

```text
--event-sampling-frequency
--ib-switch-congestion-device
--ib-switch-metrics-device
```

### 4.3 重要语义变化

- `--nic-metrics`: `2024.7.1` 文档是 `true|false`；`2026.2` 推荐 `lf|hf|none`，且 `true` 仅为兼容别名（映射到 `lf`，未来版本计划移除）。
- 事件采样配置:
  - `2024.7.1`: `--event-sampling-frequency`
  - `2026.2`: 使用 `--event-sampling-interval` 与 `--event-sampling-multiplex-interval`
- IB switch 参数命名:
  - `--ib-switch-congestion-device` -> `--ib-switch-congestion-devices`
  - `--ib-switch-metrics-device` -> `--ib-switch-metrics-devices`
  - 同时新增聚合入口 `--ib-switch-congestion` / `--ib-switch-metrics`
- `--flush-on-cudaprofilerstop` 仍可用，同时 `2026.2` 文档增加 `--flush-on` 别名入口。
- `syscall` 采集方式在新文档中明确建议使用 `--syscall=...`；`--trace=syscall` 被标注为 deprecated/忽略。

## 5) 2026.2 Profile 参数名全量清单（官方表）

> 以下列表由 `v2026.2 User Guide` 的 `CLI Profile Command Switch Options` 段自动提取并去重。

```text
--accelerator-trace
--after-collection-start
--after-report-ready
--auto-report-name
--backtrace
--capture-range
--capture-range-end
--clock-frequency-changes
--command-file
--cpu-cluster-events
--cpu-core-events
--cpu-core-metrics
--cpu-metrics
--cpu-socket-events
--cpu-socket-metrics
--cpuctxsw
--cuda-event-trace
--cuda-flush-interval
--cuda-graph-trace
--cuda-memory-usage
--cuda-trace-all-apis
--cuda-trace-scope
--cuda-um-cpu-page-faults
--cuda-um-gpu-page-faults
--cudabacktrace
--dask
--debug-symbols
--delay
--discard-environment
--duration
--duration-frames
--dx-force-declare
--dx-force-declare-adapter-removal-support
--dx12-gpu-workload
--dx12-wait-calls
--enable
--env-var
--etw-provider
--event-sample
--event-sampling
--event-sampling-interval
--event-sampling-multiplex-interval
--export
--flush-on
--flush-on-cudaprofilerstop
--force-overwrite
--ftrace
--ftrace-keep-user-config
--gds-libs-path
--gds-metrics
--gpu-metrics-devices
--gpu-metrics-frequency
--gpu-metrics-set
--gpu-video-device
--gpuctxsw
--help
--hotkey-capture
--ib-net-info-devices
--ib-net-info-files
--ib-net-info-output
--ib-switch-congestion
--ib-switch-congestion-devices
--ib-switch-congestion-nic-device
--ib-switch-congestion-percent
--ib-switch-congestion-threshold-high
--ib-switch-metrics
--ib-switch-metrics-devices
--ib-switch-metrics-nic
--ib-switch-metrics-nic-device
--inherit-environment
--injection-use-detours
--isr
--kill
--mpi-impl
--nic-metrics
--nvtx-capture
--nvtx-domain-exclude
--nvtx-domain-include
--opengl-gpu-workload
--os-events
--osrt-backtrace
--osrt-backtrace-depth
--osrt-backtrace-stack-size
--osrt-backtrace-threshold
--osrt-file-access
--osrt-threshold
--output
--process-scope
--python-backtrace
--python-functions-trace
--python-sampling
--python-sampling-frequency
--pytorch
--qnx-kernel
--qnx-kernel-events
--qnx-kernel-events-mode
--resolve-symbols
--retain-etw-files
--run-as
--sample
--samples-per-backtrace
--sampling-frequency
--sampling-period
--sampling-trigger
--session-new
--show-output
--soc-metrics
--soc-metrics-frequency
--soc-metrics-set
--start-frame-index
--start-later
--stats
--stop-on-exit
--syscall
--trace
--trace-fork-before-exec
--vsync
--vulkan-gpu-workload
--wait
--wddm-additional-events
--wddm-backtraces
--wddm-memory-trace
--xhv-trace
--xhv-trace-events
--xhv-vm-symbols
```

## 6) 官方链接

- 最新 User Guide (`v2026.2`): <https://docs.nvidia.com/nsight-systems/UserGuide/index.html>
- `CLI Profile Command Switch Options`: <https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profile-command-switch-options>
- 2024.7.1 User Guide（对比基线）: <https://docs.nvidia.com/nsight-systems/2024.7/UserGuide/index.html>
- GPU counter 权限说明: <https://developer.nvidia.com/ERR_NVGPUCTRPERM>
