# NCU Report Analyze

## Summary

- metric_records: 2160
- filtered_records: 2160
- unique_metrics: 2160
- unique_kernels: 1
- numeric_values: 2139
- non_numeric_values: 21

## Bottleneck

- coverage: 100% (11/11)
- 1. [ncu_rule] Issue Slot Utilization (SchedulerStats)
  - Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 7.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 16 warps per scheduler, this workload allocates an average of 2.99 active warps per scheduler, but only an average of 0.15 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the @section:WarpStateStats:Warp State Statistics@ and @section:SourceCounters:Source Counters@ sections can help, too.
- 2. [ncu_rule] Theoretical Occupancy (Occupancy)
  - The 3.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 16. This kernel's theoretical occupancy (18.8%) is limited by the number of required registers, and the required amount of shared memory.
- 3. [ncu_rule] Long Scoreboard Stalls (WarpStateStats)
  - On average, each warp of this workload spends 10.0 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality (coalescing), or by changing the cache configuration. Consider moving frequently used data to shared memory. This stall type represents about 42.8% of the total average of 23.2 cycles between issuing two instructions.
- 4. [ncu_rule] Barrier Stalls (WarpStateStats)
  - On average, each warp of this workload spends 8.4 cycles being stalled waiting for sibling warps at a CTA barrier. A high number of warps waiting at a barrier is commonly caused by diverging code paths before a barrier. This causes some warps to wait a long time until other warps reach the synchronization point. Whenever possible, try to divide up the work into blocks of uniform workloads. If the block size is 512 threads or greater, consider splitting it into smaller groups. This can increase eligible warps without affecting occupancy, unless shared memory becomes a new occupancy limiter. Also, try to identify which barrier instruction causes the most stalls, and optimize the code executed before that synchronization point first. This stall type represents about 36.1% of the total average of 23.2 cycles between issuing two instructions.
- 5. [ncu_rule] L1TEX Local Load Access Pattern (MemoryWorkloadAnalysis_Tables)
  - The memory access pattern for local loads from L1TEX might not be optimal. On average, only 2.1 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the @section:SourceCounters:Source Counters@ section for uncoalesced local loads.
- 6. [ncu_rule] L1TEX Local Store Access Pattern (MemoryWorkloadAnalysis_Tables)
  - The memory access pattern for local stores to L1TEX might not be optimal. On average, only 2.1 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the @section:SourceCounters:Source Counters@ section for uncoalesced local stores.
- 7. [ncu_rule] Low Compression Rate (MemoryWorkloadAnalysis_Chart)
  - Out of the 34430752.0 bytes sent to the L2 Compression unit only 0.00% were successfully compressed. To increase this success rate, consider marking only those memory regions as compressible that contain the most zero values and/or expose the most homogeneous values.
- 8. [ncu_rule] Shared Store Bank Conflicts (MemoryWorkloadAnalysis_Tables)
  - The memory access pattern for shared stores might not be optimal and causes on average a 5.0 - way bank conflict across all 24972 shared store requests.This results in 20103 bank conflicts,  which represent 16.23% of the overall 123856 wavefronts for shared stores. Check the @section:SourceCounters:Source Counters@ section for uncoalesced shared stores.
- 9. [ncu_rule] FP32 Non-Fused Instructions (InstructionStats)
  - This kernel executes 397312 fused and 989848 non-fused FP32 instructions. By converting pairs of non-fused instructions to their @url:fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP32 performance could be increased by up to 36% (relative to its current performance). Check the Source page to identify where this kernel executes FP32 instructions.
- 10. [ncu_rule] Latency Issue (SpeedOfLight)
  - This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at @section:SchedulerStats:Scheduler Statistics@ and @section:WarpStateStats:Warp State Statistics@ for potential reasons.
- dimensions_need_attention: occupancy_launch_geometry, stall_breakdown, memory_access_cache

### Six Dimensions

- occupancy_launch_geometry: needs_attention (12 signals, 1 findings)
- thread_block_balance_tail_effect: covered (4 signals, 0 findings)
- stall_breakdown: needs_attention (2 signals, 1 findings)
- tensor_core_compute: covered (3 signals, 0 findings)
- pm_sampling_timeline: covered (1 signals, 0 findings)
- memory_access_cache: needs_attention (12 signals, 2 findings)

## Top Kernels

- metric_like: `breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`

| kernel | samples | score | avg | max |
|---|---:|---:|---:|---:|

## Per Metric Stats

| metric | samples | numeric | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| breakdown:gpu__compute_memory_throughput.max.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| breakdown:gpu__compute_memory_throughput.min.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| breakdown:gpu__compute_memory_throughput.sum.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| breakdown:sm__throughput.max.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| breakdown:sm__throughput.min.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| breakdown:sm__throughput.sum.pct_of_peak_sustained_elapsed | 1 | 0 | None | None | None |
| c2clink__enabled_mask | 1 | 1 | 0.0 | 0.0 | 0.0 |
| c2clink__present | 1 | 1 | 0.0 | 0.0 | 0.0 |
| derived__avg_thread_executed | 1 | 1 | 49200.0 | 49200.0 | 49200.0 |
| derived__avg_thread_executed_true | 1 | 1 | 46654.0 | 46654.0 | 46654.0 |
| derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.peak_sustained | 1 | 1 | 16896.0 | 16896.0 | 16896.0 |
| derived__l1tex__lsu_writeback_bytes_mem_lgds.sum.per_second | 1 | 1 | 999964899451.554 | 999964899451.554 | 999964899451.554 |
| derived__lts__lts2xbar_bytes.sum.peak_sustained | 1 | 1 | 5120.0 | 5120.0 | 5120.0 |
| derived__lts__lts2xbar_bytes.sum.per_second | 1 | 1 | 2828259597806.216 | 2828259597806.216 | 2828259597806.216 |
| derived__memory_l1_conflicts_shared_nway | 1 | 1 | 264.0 | 264.0 | 264.0 |
| derived__memory_l1_wavefronts_shared_excessive | 1 | 1 | 0.0 | 0.0 | 0.0 |
| derived__memory_l2_theoretical_sectors_global_excessive | 1 | 1 | 0.0 | 0.0 | 0.0 |
| derived__pct_occupancy_per_barrier_count | 1 | 1 | 1170.0 | 1170.0 | 1170.0 |
| derived__pct_occupancy_per_block_size | 1 | 1 | 233.0 | 233.0 | 233.0 |
| derived__pct_occupancy_per_register_count | 1 | 1 | 3042.0 | 3042.0 | 3042.0 |
| derived__pct_occupancy_per_shared_mem_size | 1 | 1 | 8190.0 | 8190.0 | 8190.0 |
| derived__sm__sass_thread_inst_executed_op_dfma_pred_on_x2 | 1 | 1 | 16896.0 | 16896.0 | 16896.0 |
| derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2 | 1 | 1 | 33792.0 | 33792.0 | 33792.0 |
| derived__sm__sass_thread_inst_executed_op_hfma_pred_on_x4 | 1 | 1 | 67584.0 | 67584.0 | 67584.0 |
| derived__smsp__inst_executed_op_branch_pct | 1 | 1 | 0.07378617013899504 | 0.07378617013899504 | 0.07378617013899504 |
| derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2 | 1 | 1 | 0.0 | 0.0 | 0.0 |
| derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2 | 1 | 1 | 152.09060379185468 | 152.09060379185468 | 152.09060379185468 |
| derived__smsp__sass_thread_inst_executed_op_hadd_pred_on_x2 | 1 | 1 | 0.0 | 0.0 | 0.0 |
| derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x4 | 1 | 1 | 0.0 | 0.0 | 0.0 |
| derived__smsp__sass_thread_inst_executed_op_hmul_pred_on_x2 | 1 | 1 | 0.0 | 0.0 | 0.0 |
| derived_tempMetric0 | 1 | 1 | 78000.0 | 78000.0 | 78000.0 |
| device__attribute_architecture | 1 | 1 | 384.0 | 384.0 | 384.0 |
| device__attribute_async_engine_count | 1 | 1 | 3.0 | 3.0 | 3.0 |
| device__attribute_can_flush_remote_writes | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_can_map_host_memory | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_can_tex2d_gather | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_can_use_64_bit_stream_mem_ops | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_can_use_64_bit_stream_mem_ops_v1 | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_can_use_host_pointer_for_registered_mem | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_can_use_stream_mem_ops_v1 | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_can_use_stream_wait_value_nor | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_can_use_stream_wait_value_nor_v1 | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_chip | 1 | 1 | 384.0 | 384.0 | 384.0 |
| device__attribute_clock_rate | 1 | 1 | 1980000.0 | 1980000.0 | 1980000.0 |
| device__attribute_cluster_launch | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_compute_capability_major | 1 | 1 | 9.0 | 9.0 | 9.0 |
| device__attribute_compute_capability_minor | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_compute_mode | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_compute_preemption_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_concurrent_kernels | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_concurrent_managed_access | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_confidential_computing_mode | 1 | 0 | None | None | None |
| device__attribute_cooperative_launch | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_cooperative_multi_device_launch | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_deferred_mapping_cuda_array_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_device_index | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_direct_managed_mem_access_from_host | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_display_name | 1 | 0 | None | None | None |
| device__attribute_dma_buf_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_ecc_enabled | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_fb_bus_width | 1 | 1 | 5120.0 | 5120.0 | 5120.0 |
| device__attribute_fbp_count | 1 | 1 | 10.0 | 10.0 | 10.0 |
| device__attribute_generic_compression_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_global_l1_cache_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_global_memory_bus_width | 1 | 1 | 5120.0 | 5120.0 | 5120.0 |
| device__attribute_gpu_direct_rdma_flush_writes_options | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_gpu_direct_rdma_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_gpu_direct_rdma_with_cuda_vmm_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_gpu_direct_rdma_writes_ordering | 1 | 1 | 100.0 | 100.0 | 100.0 |
| device__attribute_gpu_overlap | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_gpu_pci_device_id | 1 | 1 | 590352606.0 | 590352606.0 | 590352606.0 |
| device__attribute_gpu_pci_ext_device_id | 1 | 1 | 9008.0 | 9008.0 | 9008.0 |
| device__attribute_gpu_pci_ext_downstream_link_rate | 1 | 1 | 32000.0 | 32000.0 | 32000.0 |
| device__attribute_gpu_pci_ext_downstream_link_width | 1 | 1 | 16.0 | 16.0 | 16.0 |
| device__attribute_gpu_pci_ext_gen | 1 | 1 | 4.0 | 4.0 | 4.0 |
| device__attribute_gpu_pci_ext_gpu_gen | 1 | 1 | 4.0 | 4.0 | 4.0 |
| device__attribute_gpu_pci_ext_gpu_link_rate | 1 | 1 | 32000.0 | 32000.0 | 32000.0 |
| device__attribute_gpu_pci_ext_gpu_link_width | 1 | 1 | 16.0 | 16.0 | 16.0 |
| device__attribute_gpu_pci_revision_id | 1 | 1 | 161.0 | 161.0 | 161.0 |
| device__attribute_gpu_pci_sub_system_id | 1 | 1 | 381751518.0 | 381751518.0 | 381751518.0 |
| device__attribute_handle_type_fabric_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_handle_type_posix_file_descriptor_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_handle_type_win32_handle_supported | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_handle_type_win32_kmt_handle_supported | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_host_native_atomic_supported | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_host_numa_id | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_host_register_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_implementation | 1 | 1 | 384.0 | 384.0 | 384.0 |
| device__attribute_integrated | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_ipc_event_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_kernel_exec_timeout | 1 | 1 | 0.0 | 0.0 | 0.0 |
| device__attribute_l2_cache_size | 1 | 1 | 52428800.0 | 52428800.0 | 52428800.0 |
| device__attribute_l2s_count | 1 | 1 | 80.0 | 80.0 | 80.0 |
| device__attribute_limits_max_cta_per_sm | 1 | 1 | 32.0 | 32.0 | 32.0 |
| device__attribute_limits_num_tpcs | 1 | 1 | 66.0 | 66.0 | 66.0 |
| device__attribute_local_l1_cache_supported | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_managed_memory | 1 | 1 | 1.0 | 1.0 | 1.0 |
| device__attribute_max_access_policy_window_size | 1 | 1 | 134217728.0 | 134217728.0 | 134217728.0 |
