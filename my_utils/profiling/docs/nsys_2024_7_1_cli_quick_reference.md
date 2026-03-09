# Nsight Systems 2024.7.1 CLI Profile 快速参考

基于 NVIDIA 官方 2024.7 User Guide，面向训练任务（PyTorch/torchrun、单机多卡、多机）整理。

## 1) 命令骨架

```bash
nsys [global-options] profile [options] [application] [application-arguments]
```

官方说明：CLI 参数区分大小写；长参数推荐 `--key=value` 形式。

## 2) 训练任务高频参数（你当前脚本同类）

- `--trace=cuda,nvtx,osrt,cublas,cudnn`: 深度学习常用追踪组合。
- `--capture-range=cudaProfilerApi`: 由 `cudaProfilerStart/Stop` 控制采集窗口。
- `--capture-range=nvtx`: 由 NVTX range 控制采集窗口（配合 `--nvtx-capture`）。
- `--capture-range-end=stop`: 到达采集结束信号后停止采集但不中止应用。
- `--output=...`: 报告文件前缀，支持 `%q{ENV_VAR}`、`%h`、`%p`、`%%`。
- `--force-overwrite=true`: 同名文件覆盖。
- `--export=none|sqlite|json|...`: 额外导出格式；`none` 可减少收尾耗时。
- `--sample=process-tree|system-wide|none`: CPU 采样范围；`none` 可降开销。
- `--cudabacktrace=all|none|kernel|memory|sync|other`: CUDA API 回溯，开销较大。
- `--gpu-metrics-devices=all|none|<id>|help`: GPU metrics 开关与设备选择。
- `--nic-metrics=true|false`: NIC/HCA 指标采集（系统范围）。
- `--show-output=true|false`: 是否打印目标程序 stdout/stderr。

## 3) 其他常用 profile 参数（官方但你当前未必用到）

### 3.1 采集时序与生命周期

- `--delay=<seconds>`: 启动后延迟采集。
- `--duration=<seconds>`: 固定时长采集。
- `--kill=none|sigterm|sigkill|<signal>`: 结束会话时对目标进程组发信号。
- `--wait=primary|all`: CLI 等待主进程或全部派生进程结束。
- `--stop-on-exit=true|false`: 目标退出时是否停止会话。
- `--start-later=true|false`: 先建会话后开始采集。
- `--session-new=<name>`: 指定会话名。

### 3.2 NVTX 过滤与触发

- `--nvtx-capture=range[@domain]`: 指定触发采集的 NVTX range（仅 `capture-range=nvtx`）。
- `--nvtx-domain-include=a,b`: 仅保留指定 domain。
- `--nvtx-domain-exclude=a,b`: 排除指定 domain。

### 3.3 CPU 采样与回溯

- `--cpuctxsw=process-tree|system-wide|none`: 线程调度事件。
- `--backtrace=auto|fp|lbr|dwarf|none`: CPU 回溯方法。
- `--samples-per-backtrace=<N>`: 每 N 个 IP 样本采一次回溯。
- `--sampling-frequency=<Hz>`: 采样频率（平台相关）。
- `--sampling-period=<count>`: 按事件计数触发采样（平台相关）。
- `--event-sample=system-wide|none`: 事件采样开关（非 Embedded）。
- `--event-sampling-frequency=1..20`: 事件采样频率。
- `--cpu-core-events=...`: 采集 CPU Core PMU 事件。
- `--cpu-core-metrics=...`: 采集 CPU Core 指标（Grace 支持）。

### 3.4 CUDA 相关进阶

- `--cuda-flush-interval=<ms>`: CUDA trace buffer 刷盘策略。
- `--cuda-graph-trace=graph|node`: CUDA Graph 粒度与开销权衡。
- `--cuda-memory-usage=true|false`: 追踪 CUDA 显存使用（开销高）。
- `--cuda-um-cpu-page-faults=true|false`: 采集 UM CPU page faults。
- `--cuda-um-gpu-page-faults=true|false`: 采集 UM GPU page faults。
- `--flush-on-cudaprofilerstop=true|false`: `cudaProfilerStop()` 时刷新 CUDA buffer。

### 3.5 Python 相关

- `--python-sampling=true|false`: Python 栈采样。
- `--python-sampling-frequency=1..2000`: Python 采样频率。
- `--python-backtrace=cuda|none`: 由 CUDA 触发 Python backtrace（需配合 CUDA 回溯与 CPU sampling）。
- `--python-functions-trace=<json>`: 基于配置文件函数级 Python trace（无需改源码可用于部分场景）。

### 3.6 输出、统计、符号与配置文件

- `--stats=true|false`: 采集结束后自动生成统计（会触发 SQLite 生成，收尾更慢）。
- `--resolve-symbols=true|false`: 解析符号信息。
- `--command-file=<file>`: 从文件加载 profile 参数。
- `--env-var=A=B`: 为被测进程设置环境变量。
- `--inherit-environment=true|false`: 是否继承当前环境变量。
- `--run-as=<username>`: 指定用户运行目标（Linux，需要更高权限）。

### 3.7 网络 / InfiniBand 相关

- `--nic-metrics=true|false`: NIC/HCA 指标。
- `--ib-net-info-devices=...`: 基于 NIC 执行网络发现。
- `--ib-net-info-files=...`: 读取已有 ibdiagnet 文件。
- `--ib-switch-metrics-device=...`: 指定 IB 交换机采样。
- `--ib-switch-congestion-device=...`: 采集 IB 拥塞事件。
- `--ib-switch-congestion-percent=<1..100>`: 拥塞事件采样比例。
- `--ib-switch-congestion-threshold-high=<...>`: 拥塞阈值。

## 4) 平台专属参数（按官方说明）

- Windows 图形链路: `--trace=dx11/dx12/wddm` + `--dx12-gpu-workload` + `--wddm-backtraces` + `--etw-provider` + `--isr`。
- OpenGL/Vulkan 图形链路: `--opengl-gpu-workload`、`--vulkan-gpu-workload`。
- Embedded/QNX/Tegra: `--soc-metrics`、`--xhv-trace`、`--qnx-kernel-events`、`--accelerator-trace`、`--ftrace`。
- Linux 专属常见: `--trace-fork-before-exec`、`--run-as`。

## 5) 多进程/多机命名建议

避免 rank 互相覆盖文件：

- OpenMPI: `%q{OMPI_COMM_WORLD_RANK}`
- MPICH: `%q{PMI_RANK}`
- Slurm: `%q{SLURM_PROCID}`
- 通用兜底: `%p`

示例：

```bash
--output=./logs/nsys/train_rank_%q{RANK}
```

## 6) 训练任务推荐组合

低开销起步（先定位大头）：

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

需要硬件指标时再加：

```bash
--gpu-metrics-devices=all --nic-metrics=true
```

需要 NVTX 触发时：

```bash
--capture-range=nvtx --nvtx-capture=fusion_loop --capture-range-end=stop
```

## 7) Profile 参数名总览（2024.7 profile 表）

```text
--accelerator-trace
--auto-report-name
--backtrace
--capture-range
--capture-range-end
--clock-frequency-changes
--command-file
--cpu-cluster-events
--cpu-core-events
--cpu-core-metrics
--cpuctxsw
--cpu-socket-events
--cudabacktrace
--cuda-flush-interval
--cuda-graph-trace
--cuda-memory-usage
--cuda-um-cpu-page-faults
--cuda-um-gpu-page-faults
--delay
--duration
--duration-frames
--dx12-gpu-workload
--dx12-wait-calls
--dx-force-declare-adapter-removal-support
--enable
--env-var
--etw-provider
--event-sample
--event-sampling-frequency
--export
--flush-on-cudaprofilerstop
--force-overwrite
--ftrace
--ftrace-keep-user-config
--gpuctxsw
--gpu-metrics-devices
--gpu-metrics-frequency
--gpu-metrics-set
--gpu-video-device
--help
--hotkey-capture
--ib-net-info-devices
--ib-net-info-files
--ib-net-info-output
--ib-switch-congestion-device
--ib-switch-congestion-nic-device
--ib-switch-congestion-percent
--ib-switch-congestion-threshold-high
--ib-switch-metrics-device
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
--osrt-backtrace-depth
--osrt-backtrace-stack-size
--osrt-backtrace-threshold
--osrt-threshold
--output
--process-scope
--python-backtrace
--python-functions-trace
--python-sampling
--python-sampling-frequency
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
--trace
--trace-fork-before-exec
--vsync
--vulkan-gpu-workload
--wait
--wddm-additional-events
--wddm-backtraces
--xhv-trace
--xhv-trace-events
--xhv-vm-symbols
```

## 8) 参考链接

- Nsight Systems 2024.7 User Guide:  
  <https://docs.nvidia.com/nsight-systems/2024.7/UserGuide/index.html>
- CLI Profile Command Switch Options:  
  <https://docs.nvidia.com/nsight-systems/2024.7/UserGuide/index.html#cli-profile-command-switch-options>
- Profiling from the CLI:  
  <https://docs.nvidia.com/nsight-systems/2024.7/UserGuide/index.html#cli-profiling>
- GPU counter 权限说明:  
  <https://developer.nvidia.com/ERR_NVGPUCTRPERM>

