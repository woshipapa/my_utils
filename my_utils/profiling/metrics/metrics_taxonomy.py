# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Dict, Tuple


# Canonical dimensions used across providers.
CANONICAL_METRIC_PREFIXES = (
    "latency",  # execution time
    "memory",  # memory usage/allocation
    "compute",  # FLOPs, occupancy, throughput
    "comm",  # communication/NCCL/network
    "io",  # memcpy, storage/network bytes
    "calls",  # call counts
    "perf",  # perf stat counters
)


# Tool-specific metric aliases to canonical metric names.
TOOL_METRIC_ALIASES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "torch_profiler": {
        "self_cpu_time_total": ("latency.op.self_cpu", "us"),
        "cpu_time_total": ("latency.op.total_cpu", "us"),
        "self_cuda_time_total": ("latency.op.self_cuda", "us"),
        "cuda_time_total": ("latency.op.total_cuda", "us"),
        "self_cuda_memory_usage": ("memory.op.self_cuda", "bytes"),
        "cuda_memory_usage": ("memory.op.cuda", "bytes"),
        "flops": ("compute.op.flops", "flops"),
    },
    "tensorflow_profiler": {
        "self_time_us": ("latency.op.self_cpu", "us"),
        "accelerator_time_us": ("latency.op.total_cuda", "us"),
        "flops": ("compute.op.flops", "flops"),
    },
    "nsys": {
        "runtime_api_duration": ("latency.cuda.runtime_api", "us"),
        "kernel_duration": ("latency.kernel.cuda", "us"),
        "memcpy_duration": ("latency.memcpy.cuda", "us"),
        "memset_duration": ("latency.memset.cuda", "us"),
        "sync_duration": ("latency.cuda.sync", "us"),
        "nvtx_range_duration": ("latency.nvtx.range", "us"),
        "memcpy_bytes": ("io.memcpy.bytes", "bytes"),
        "memcpy_bandwidth": ("io.memcpy.bandwidth_gbps", "GB/s"),
        "cuda_alloc_bytes": ("memory.cuda.alloc.bytes", "bytes"),
        "cuda_free_bytes": ("memory.cuda.free.bytes", "bytes"),
        "memory_pool_event": ("calls.cuda.memory_pool_event", "count"),
        "pool_min_bytes_to_keep": ("memory.cuda.pool.min_bytes_to_keep", "bytes"),
        "pool_local_release_threshold": (
            "memory.cuda.pool.local_release_threshold_bytes",
            "bytes",
        ),
        "pool_local_size": ("memory.cuda.pool.local_size_bytes", "bytes"),
        "pool_local_utilized": ("memory.cuda.pool.local_utilized_bytes", "bytes"),
        "gpu_sm_active": ("compute.gpu.sm.active", ""),
        "gpu_sm_utilization": ("compute.gpu.sm.utilization", ""),
        "gpu_sm_throughput": ("compute.gpu.sm.throughput", ""),
        "gpu_dram_throughput": ("memory.gpu.dram.throughput", ""),
        "gpu_pcie_tx": ("io.gpu.pcie.tx_throughput", ""),
        "gpu_pcie_rx": ("io.gpu.pcie.rx_throughput", ""),
        "gpu_nvlink_tx": ("io.gpu.nvlink.tx_throughput", ""),
        "gpu_nvlink_rx": ("io.gpu.nvlink.rx_throughput", ""),
        "nic_metric": ("comm.nic", ""),
        "ib_switch_metric": ("comm.ib.switch", ""),
        "pmu_metric": ("perf.pmu", ""),
    },
    "ncu": {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": (
            "compute.sm.throughput_pct",
            "percent",
        ),
        "sm__warps_active.avg.pct_of_peak_sustained_active": (
            "compute.sm.occupancy_pct",
            "percent",
        ),
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": (
            "memory.dram.throughput_pct",
            "percent",
        ),
    },
    "cprofile": {
        "tottime": ("latency.python.self", "ms"),
        "cumtime": ("latency.python.total", "ms"),
        "ncalls": ("calls.python", "count"),
    },
    "perf_stat": {
        "task-clock": ("perf.task_clock", ""),
        "cycles": ("perf.cycles", ""),
        "instructions": ("perf.instructions", ""),
        "cache-misses": ("perf.cache_misses", ""),
        "branches": ("perf.branches", ""),
        "branch-misses": ("perf.branch_misses", ""),
    },
}


def normalize_external_metric(tool: str, raw_metric: str) -> Tuple[str, str]:
    tool_map = TOOL_METRIC_ALIASES.get(tool, {})
    if raw_metric in tool_map:
        return tool_map[raw_metric]
    return f"external.{tool}.{raw_metric}", ""
