# NCU Report Diff (A -> B)

- A (baseline): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode0_register_baseline.ncu-rep`
- B (candidate): `/Users/papa/Desktop/kernel_traces/ncu_kernels/cta_pingpong_128x128x64_auto_no_barrier_rms.ncu-rep`
- GPU: H100 SXM5
- matched kernels: 1

## Clock guard and comparability

**WARNING: 1 of 1 matched kernel(s) fail one or more compatibility guards; raw-duration deltas are not speedups. 1x at least one measurement has an unrecorded cache state, and cache state moves a duration by more than most optimisations do; 1x the baseline report's GPC and SM clocks disagree by 6.6%, indicating replay passes were not measured over a stable clock window; 1x the two measurements ran at different SM clocks (1891 vs 1745 MHz, -7.8%). A duration ratio therefore mixes the change in the code with the change in the clock; compare cycles, or lock the clock and re-measure**

## Repeatability

at least two reports per side are required for a repeatability judgement

## `pod_fused_device_kernel` (grid 132 -> 132, block 384 -> 384)

- result: **NOT_COMPARABLE** - collection/device/workload guards block a speed claim
- match: demangled_name (confidence high)

### Duration

| quantity | A | B | B/A |
|---|---|---|---|
| raw duration (ns) -- confounded by clock, NOT a speedup | 87,520 | 72,672 | 0.830 |
| SM clock (MHz) | 1,891 | 1,745 | 0.922 |
| clock-normalised duration ratio | - | - | 0.766 |
| elapsed GPC cycles (clock-independent) | 156,558 | 127,867 | 0.817 |

raw durations are NOT comparable as a speedup (SM clocks 1891 vs 1745 MHz); clock-normalised, B runs at 0.766x of A (raw 0.830x contains the clock change); the clock-independent elapsed-cycle ratio is 0.817x. NOTE: the clock-normalised duration ratio (0.766x) and the elapsed-cycle ratio (0.817x) disagree by 6.2%, which means the clock varied between replay passes; trust neither figure to better than that.

- guard: at least one measurement has an unrecorded cache state, and cache state moves a duration by more than most optimisations do
- guard: the baseline report's GPC and SM clocks disagree by 6.6%, indicating replay passes were not measured over a stable clock window
- guard: the two measurements ran at different SM clocks (1891 vs 1745 MHz, -7.8%). A duration ratio therefore mixes the change in the code with the change in the clock; compare cycles, or lock the clock and re-measure

- caveat: Clocks were not confirmed locked on both sides, so small differences may be clock rather than code.
- caveat: workload id was not recorded on both sides; equal workload is unproven
- caveat: problem shape was not recorded on both sides; equal workload is unproven
- caveat: dtype was not recorded on both sides; equal workload is unproven
- caveat: input hash was not recorded on both sides; equal workload is unproven
- caveat: output hash was not recorded on both sides; equal workload is unproven

### What changed the verdict

- bottleneck verdict unchanged: `latency_bound`
- disappeared (medium): **Warp stalls dominated by Long Scoreboard** [stall_long_scoreboard]
- disappeared (medium): **Shared-memory st bank conflicts on 16% of wavefronts** [shared_bank_conflicts_st]
- escalated: Warp stalls dominated by Barrier [stall_barrier] medium -> high
- escalated: Register spilling to local memory [register_spilling] low -> medium
- 6 finding(s) unchanged on both sides

### Stall composition (cycles per issue-slot)

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Long Scoreboard (cycles/issue-slot) | 9.95 | 4.97 | -4.98 | -50.1% | improved |
| No Instruction (cycles/issue-slot) | 0.398 | 1.46 | 1.07 | +267.4% | REGRESSED |
| Barrier (cycles/issue-slot) | 8.39 | 7.98 | -0.412 | -4.9% | improved |
| MIO Throttle (cycles/issue-slot) | 0.0645 | 0.345 | 0.28 | +434.6% | REGRESSED |
| Short Scoreboard (cycles/issue-slot) | 0.574 | 0.316 | -0.259 | -45.0% | improved |
| Sleeping (cycles/issue-slot) | 0.383 | 0.148 | -0.234 | -61.2% | improved |
| Wait (cycles/issue-slot) | 1.65 | 1.47 | -0.182 | -11.0% | improved |
| Not Selected (cycles/issue-slot) | 0.163 | 0.341 | 0.178 | +108.8% | REGRESSED |
| Branch Resolving (cycles/issue-slot) | 0.437 | 0.271 | -0.167 | -38.2% | improved |
| Warpgroup Arrive (cycles/issue-slot) | 0.0178 | 0.155 | 0.137 | +770.8% | REGRESSED |
| Math Pipe Throttle (cycles/issue-slot) | 0.0665 | 0.141 | 0.0748 | +112.5% | REGRESSED |

_8 further metric(s) unchanged within noise._

### Speed of light

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| SM compute throughput (% of peak) | 28.80 | 37.60 | 8.8 | +30.6% | changed |
| Compute-memory throughput (% of peak) | 32.37 | 37.16 | 4.78 | +14.8% | changed |
| DRAM throughput (% of peak) | 30.43 | 37.16 | 6.73 | +22.1% | changed |
| L1/TEX throughput (% of peak) | 28.06 | 34.66 | 6.6 | +23.5% | changed |
| L2 throughput (% of peak) | 41.87 | 44.36 | 2.49 | +5.9% | changed |

### Occupancy

All 4 tracked metrics unchanged within noise.

### Memory hierarchy (traffic-weighted)

| metric | A | B | delta | rel | traffic A -> B | status |
|---|---|---|---|---|---|---|
| L1/TEX sector hit rate (%) | 50.31 | 28.90 | -21.42 | -42.6% | - -> - | changed (interpret with miss and traffic rows; hit rate alone is not a performance verdict) |
| L2 sector hit rate (%) | 69.61 | 63.99 | -5.63 | -8.1% | 11,082,969 -> 9,727,569 | changed (interpret with miss and traffic rows; hit rate alone is not a performance verdict) |
| L2 read hit rate (%) | 60.68 | 55.88 | -4.8 | -7.9% | - -> - | changed (interpret with miss and traffic rows; hit rate alone is not a performance verdict) |
| Local-load L1TEX hit rate (%) | 96.53 | 51.97 | -44.56 | -46.2% | 260,532 -> 22,512 | changed (interpret with miss and traffic rows; hit rate alone is not a performance verdict) |
| Local-store L1TEX hit rate (%) | 35.16 | 0.0502 | -35.11 | -99.9% | 91,616 -> 127,460 | changed (interpret with miss and traffic rows; hit rate alone is not a performance verdict) |
| Local-load sectors missing L1TEX (sent to L2) (sectors) | 9,028 | 10,812 | 1,784 | +19.8% | - | REGRESSED |
| Local-store sectors missing L1TEX (sent to L2) (sectors) | 59,404 | 127,396 | 67,992 | +114.5% | - | REGRESSED |
| L2 sector traffic (sectors) | 11,082,969 | 9,727,569 | -1,355,400 | -12.2% | - | changed |

_2 further metric(s) unchanged within noise._

### Instruction mix

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Instructions executed (warp) (inst) | 10,729,016 | 9,810,285 | -918,731 | -8.6% | changed |
| Executed IPC (inst/cycle) | 0.511 | 0.633 | 0.121 | +23.7% | improved |
| Issue-active fraction (of cycles) | 0.129 | 0.157 | 0.0283 | +21.9% | improved |
| Avg threads active per inst (threads) | 49,200 | 44,628 | -4,572 | -9.3% | changed |
| FMA pipe utilisation (% of peak) | 2.91 | 4.4 | 1.49 | +51.2% | changed |
| ALU pipe utilisation (% of peak) | 8.29 | 10.25 | 1.95 | +23.6% | changed |
| Tensor (HMMA) pipe utilisation (% of peak) | 29.99 | 40.57 | 10.58 | +35.3% | changed |

_2 further metric(s) unchanged within noise._

### Spills

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Local-load instructions (inst) | 83,986 | 5,632 | -78,354 | -93.3% | improved |
| Local-store instructions (inst) | 23,060 | 19,560 | -3,500 | -15.2% | improved |

### Shared memory

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Shared-mem store bank conflicts (conflicts) | 20,103 | 0 | -20,103 | -100.0% | improved |

_2 further metric(s) unchanged within noise._

### Work-normalised counters

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Shared-load bank conflicts per wavefront (shared_bank_conflicts_ld/shared_wavefronts_ld) | 0 | 2.49e-05 | 2.49e-05 | from 0 | REGRESSED (normalised by shared_wavefronts_ld; raw numerator is not a workload-independent verdict (baseline is zero; no ratio)) |
| Shared-store bank conflicts per wavefront (shared_bank_conflicts_st/shared_wavefronts_st) | 0.162 | 0 | -0.162 | -100.0% | improved (normalised by shared_wavefronts_st; raw numerator is not a workload-independent verdict) |
| Local-load instructions per executed instruction (local_ld_inst/inst_executed) | 0.00783 | 0.000574 | -0.00725 | -92.7% | improved (normalised by inst_executed; raw numerator is not a workload-independent verdict) |
| Local-store instructions per executed instruction (local_st_inst/inst_executed) | 0.00215 | 0.00199 | -0.000155 | -7.2% | improved (normalised by inst_executed; raw numerator is not a workload-independent verdict) |

### PM sampling aggregates

### Same-pass PM aggregate features

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| dramc__read_throughput.avg.pct_of_peak_sustained_elapsed [4] duty cycle | 1 | 0.979 | -0.0208 | -2.1% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__read_throughput.avg.pct_of_peak_sustained_elapsed [4] mean in active window | 11.66 | 14.17 | 2.51 | +21.6% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__read_throughput.avg.pct_of_peak_sustained_elapsed [4] peak | 16.47 | 41.17 | 24.70 | +150.0% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__read_throughput.avg.pct_of_peak_sustained_elapsed [4] peak-to-mean | 1.41 | 2.9 | 1.49 | +105.7% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__throughput.avg.pct_of_peak_sustained_elapsed [4] mean in active window | 15.37 | 19.09 | 3.72 | +24.2% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__throughput.avg.pct_of_peak_sustained_elapsed [4] peak | 22.34 | 46.14 | 23.81 | +106.6% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__throughput.avg.pct_of_peak_sustained_elapsed [4] peak-to-mean | 1.45 | 2.42 | 0.964 | +66.4% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__write_throughput.avg.pct_of_peak_sustained_elapsed [4] duty cycle | 0.897 | 0.812 | -0.0841 | -9.4% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__write_throughput.avg.pct_of_peak_sustained_elapsed [4] mean in active window | 3.71 | 4.92 | 1.2 | +32.4% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__write_throughput.avg.pct_of_peak_sustained_elapsed [4] peak | 9.53 | 17.20 | 7.67 | +80.5% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| dramc__write_throughput.avg.pct_of_peak_sustained_elapsed [4] peak-to-mean | 2.57 | 3.5 | 0.933 | +36.4% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__ctas_launched.sum [4] duty cycle | 0.0172 | 0.0208 | 0.00359 | +20.8% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__ctas_launched.sum [4] mean in active window | 2.28 | 2.75 | 0.474 | +20.8% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__ctas_launched.sum [4] peak-to-mean | 58 | 48 | -10 | -17.2% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__cycles_active.avg [4] mean in active window | 2,574 | 2,475 | -99.06 | -3.8% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__cycles_active.avg [4] peak-to-mean | 1.07 | 1.11 | 0.0466 | +4.4% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_pipe_alu_realtime.avg.pct_of_peak_sustained_elapsed [4] mean in active window | 8.13 | 9.51 | 1.38 | +17.0% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_pipe_alu_realtime.avg.pct_of_peak_sustained_elapsed [4] peak | 17.50 | 27.27 | 9.77 | +55.8% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_pipe_alu_realtime.avg.pct_of_peak_sustained_elapsed [4] peak-to-mean | 2.15 | 2.87 | 0.715 | +33.2% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed [4] mean in active window | 12.64 | 14.80 | 2.16 | +17.1% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed [4] peak | 20.24 | 33.01 | 12.77 | +63.1% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed [4] peak-to-mean | 1.6 | 2.23 | 0.628 | +39.2% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_realtime.avg.per_cycle_active [4] mean in active window | 0.515 | 0.606 | 0.0911 | +17.7% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_realtime.avg.per_cycle_active [4] peak | 0.81 | 1.32 | 0.51 | +63.0% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__inst_executed_realtime.avg.per_cycle_active [4] peak-to-mean | 1.57 | 2.18 | 0.606 | +38.5% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed [4] duty cycle | 0.845 | 0.667 | -0.178 | -21.1% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed [4] mean in active window | 30.90 | 39.31 | 8.41 | +27.2% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed [4] peak | 60.54 | 100 | 39.46 | +65.2% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed [4] peak-to-mean | 1.96 | 2.54 | 0.584 | +29.8% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_no_instruction.avg [3] mean in active window | 116 | 519 | 403 | +346.6% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_no_instruction.avg [3] peak | 1,535 | 7,187 | 5,652 | +368.2% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_no_instruction.avg [3] peak-to-mean | 13.20 | 13.84 | 0.639 | +4.8% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_sleeping.avg [3] duty cycle | 0.845 | 0.646 | -0.199 | -23.6% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_sleeping.avg [3] mean in active window | 151 | 60.49 | -90.27 | -59.9% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_sleeping.avg [3] peak | 329 | 214 | -116 | -35.1% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_sleeping.avg [3] peak-to-mean | 2.19 | 3.53 | 1.35 | +61.7% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_wait.avg [3] mean in active window | 556 | 573 | 16.27 | +2.9% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_wait.avg [3] peak | 1,100 | 1,522 | 422 | +38.4% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |
| smsp__warps_issue_stalled_wait.avg [3] peak-to-mean | 1.98 | 2.66 | 0.681 | +34.5% | changed (same PM pass group only; timeline buckets are never diffed across replay passes) |

_12 further metric(s) unchanged within noise._

- unmatched PM series were not compared because pass-group identity changed.

### PC sampling hotspots

### Source-line sample-share changes

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| /workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:533 (share of PC samples) | 0.182 | 0.0358 | -0.147 | -80.4% | changed |
| /workspace/cutlass/include/cutlass/bfloat16.h:165 (share of PC samples) | 0.121 | 0.0184 | -0.103 | -84.8% | changed |
| /workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:547 (share of PC samples) | 0.00513 | 0.0603 | 0.0552 | +1076.3% | changed |
| /workspace/cutlass/include/cutlass/arch/barrier.h:426 (share of PC samples) | 0.0711 | 0.0208 | -0.0503 | -70.8% | changed |
| /workspace/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:208 (share of PC samples) | 0.00753 | 0.0346 | 0.0271 | +359.2% | changed |
| /workspace/cutlass/include/cute/algorithm/gemm.hpp:414 (share of PC samples) | 0.0219 | 0.0413 | 0.0195 | +89.1% | changed |
| /workspace/cutlass/include/cute/arch/cluster_sm90.hpp:194 (share of PC samples) | 0.0235 | 0.00583 | -0.0177 | -75.2% | changed |
| /workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:487 (share of PC samples) | 0.00157 | 0.0144 | 0.0129 | +820.5% | changed |
| /workspace/horizonal_fuse_kernel/rmsnorm_warpgroup.hpp:385 (share of PC samples) | 0.0105 | 0.00252 | -0.00794 | -75.9% | changed |
| /workspace/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:1323 (share of PC samples) | 0.00157 | 0.00596 | 0.00439 | +280.0% | changed |
| /workspace/cutlass/include/cute/arch/copy_sm90_tma.hpp:185 (share of PC samples) | 0.00209 | 0.00371 | 0.00162 | +77.3% | changed |
| /workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:372 (share of PC samples) | 0.00178 | 0.00172 | -5.57e-05 | -3.1% | changed |
| /usr/local/cuda/include/sm_20_intrinsics.hpp:151 (share of PC samples) | 0.00178 | - | - | - | A only (present in only one report; no delta can be formed) |
| /workspace/cutlass/include/cute/arch/cluster_sm90.hpp:130 (share of PC samples) | - | 0.00318 | - | - | B only (present in only one report; no delta can be formed) |
| /workspace/cutlass/include/cutlass/arch/barrier.h:474 (share of PC samples) | 0.00272 | - | - | - | A only (present in only one report; no delta can be formed) |
| /workspace/cutlass/include/cutlass/arch/barrier.h:497 (share of PC samples) | 0.00251 | - | - | - | A only (present in only one report; no delta can be formed) |
| /workspace/cutlass/include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp:697 (share of PC samples) | - | 0.00265 | - | - | B only (present in only one report; no delta can be formed) |
| /workspace/cutlass/include/cutlass/fast_math.h:575 (share of PC samples) | - | 0.00265 | - | - | B only (present in only one report; no delta can be formed) |
| /workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:529 (share of PC samples) | - | 0.00212 | - | - | B only (present in only one report; no delta can be formed) |
| /workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:532 (share of PC samples) | - | 0.00172 | - | - | B only (present in only one report; no delta can be formed) |

_38 additional hotspot rows in JSON output._

### Metric catalog coverage

146 of 182 catalog keys are present on both sides; A-only 0, B-only 0, absent on both 36.

### Changed or one-sided catalog metrics

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| shared_wavefronts_ld (wavefront) | 38,580 | 401,218 | 362,638 | +940.0% | changed |
| occupancy_limit_barriers (warp) | 4 | 32 | 28 | +700.0% | changed |
| Useful bytes per 32-byte sector. Wasted bandwidth fraction is 1 - value/32. (byte/sector) | 32.09 | 96 | 63.91 | +199.2% | improved |
| shared_bank_conflicts_st | 20,103 | 0 | -20,103 | -100.0% | improved |
| l1_local_st_hit_rate (%) | 35.16 | 0.0502 | -35.11 | -99.9% | REGRESSED |
| LDL instructions - local memory traffic, almost always register spill. (inst) | 83,986 | 5,632 | -78,354 | -93.3% | improved |
| l1_local_ld_sectors (sector) | 260,532 | 22,512 | -238,020 | -91.4% | changed |
| shared_mem_per_block (byte) | 110,079 | 207,903 | 97,824 | +88.9% | changed |
| barrier_count | 16 | 2 | -14 | -87.5% | changed |
| Transcendental/special-function unit - the exp in softmax lands here. (%) | 0.105 | 0.197 | 0.0917 | +87.3% | changed |
| branch_targets_divergent | 17.83 | 3.44 | -14.38 | -80.7% | improved |
| shared_mem_config_size (byte) | 135,168 | 233,472 | 98,304 | +72.7% | changed |
| shared_wavefronts_st (wavefront) | 123,856 | 51,816 | -72,040 | -58.2% | changed |
| l2_sectors_local (sector) | 348,580 | 150,016 | -198,564 | -57.0% | changed |
| inst_pipe_fma (%) | 2.8 | 4.29 | 1.5 | +53.5% | changed |
| pipe_fma_util (%) | 2.91 | 4.4 | 1.49 | +51.2% | changed |
| Spills that miss L1 reach L2/DRAM and get dramatically more expensive. (%) | 96.53 | 51.97 | -44.56 | -46.2% | REGRESSED |
| L1/TEX sector hit rate. No bare threshold, because TMA bypasses L1 entirely: a correctly TMA-driven Hopper GEMM measures 0% here while running at 85% of peak. Judge it only when the TMA pipe was idle. (%) | 50.31 | 28.90 | -21.42 | -42.6% | REGRESSED |
| warps_eligible_per_scheduler (warp) | 0.15 | 0.211 | 0.061 | +40.8% | improved |
| l1_local_st_sectors (sector) | 91,616 | 127,460 | 35,844 | +39.1% | changed |
| pipe_tma_util (%) | 0.375 | 0.509 | 0.134 | +35.6% | changed |
| inst_pipe_gmma (%) | 1.87 | 2.54 | 0.661 | +35.3% | changed |
| pipe_tensor_hmma_util (%) | 29.99 | 40.57 | 10.58 | +35.3% | changed |
| pipe_tensor_util (%) | 29.99 | 40.57 | 10.58 | +35.3% | improved |
| Shared/composite pipe utilisation. NOT an independent shared-memory signal on Hopper: measured on a real sm_90 GEMM it returns values bit-identical to the tensor pipe and to sm__throughput across all four rollups, because the wgmma operand path and the tensor pipe are the same underlying counter on GH100. For actual shared-memory pressure use l1tex__data_bank_reads / _bank_writes. (%) | 29.99 | 40.57 | 10.58 | +35.3% | improved |
| l2_cycles_active (cycles) | 153,059 | 100,803 | -52,256 | -34.1% | changed |
| l2_cycles_active_avg (cycles) | 153,059 | 100,803 | -52,256 | -34.1% | changed |
| global_ld_requests (request) | 147,456 | 98,304 | -49,152 | -33.3% | changed |
| global_ld_sectors (sector) | 2,359,296 | 1,572,864 | -786,432 | -33.3% | changed |
| l2_cycles_active_max (cycles) | 154,248 | 105,276 | -48,972 | -31.7% | changed |
| SM throughput vs peak. NVIDIA's own grading: >80 excellent, 60-80 good, 40-60 fair, <40 poor. (%) | 28.80 | 37.60 | 8.8 | +30.6% | improved |
| sm_cycles_active_min (cycles) | 155,536 | 108,969 | -46,567 | -29.9% | changed |
| L1/TEX efficiency over ACTIVE cycles only - how hard L1 worked when it was working, independent of how often it was idle. Answers a different question from l1_sol and must never be compared against an _elapsed value. (%) | 29.22 | 37.40 | 8.18 | +28.0% | improved |
| l1_throughput (%) | 29.22 | 37.40 | 8.18 | +28.0% | improved |
| inst_branch (inst) | 791,653 | 571,305 | -220,348 | -27.8% | changed |
| l1_global_ld_hit_rate (%) | 62.51 | 45.47 | -17.04 | -27.3% | REGRESSED |
| l1_cycles_active (cycles) | 158,933 | 117,485 | -41,448 | -26.1% | changed |
| sm_cycles_active (cycles) | 158,933 | 117,485 | -41,448 | -26.1% | changed |
| sm_cycles_active_max (cycles) | 163,034 | 121,937 | -41,097 | -25.2% | changed |
| Per-source-line global sectors actually required. (sector) | 3,150,220 | 2,360,838 | -789,382 | -25.1% | changed |
| l2_sectors_global_ideal (sector) | 3,150,220 | 2,360,838 | -789,382 | -25.1% | changed |
| smsp_cycles_active (cycles) | 159,338 | 119,467 | -39,872 | -25.0% | changed |
| executed_ipc (inst/cycle) | 0.511 | 0.633 | 0.121 | +23.7% | improved |
| issued_ipc (inst/cycle) | 0.516 | 0.638 | 0.122 | +23.7% | improved |
| inst_pipe_alu (%) | 8.29 | 10.25 | 1.95 | +23.6% | changed |
| pipe_alu_util (%) | 8.29 | 10.25 | 1.95 | +23.6% | changed |
| L1/TEX throughput vs peak, over elapsed cycles so it is comparable with the other SOL values. The ncu SOL header shows the _active rollup for this one metric only; do not rank cache levels using the header numbers as displayed. (%) | 28.06 | 34.66 | 6.6 | +23.5% | improved |
| l1_cycles_elapsed (cycles) | 165,520 | 126,777 | -38,742 | -23.4% | changed |
| sm_cycles_elapsed (cycles) | 165,520 | 126,777 | -38,742 | -23.4% | changed |
| inst_pipe_cbu (%) | 0.301 | 0.371 | 0.0698 | +23.2% | changed |
| dram_bytes_per_sec (byte/s) | 1,019,366,727,605 | 1,245,121,972,699 | 225,755,245,094 | +22.1% | changed |
| DRAM bandwidth vs peak. A healthy streaming kernel reaches 80-95%. (%) | 30.43 | 37.16 | 6.73 | +22.1% | improved |
| Instructions issued per scheduler per active cycle. NVIDIA's IssueSlotUtilization rule fires below 0.6, but a warp-specialized kernel measured 0.14 while hitting 85% of peak - its producer warps park in wait loops by design. Used as a gate, not as a standalone finding. (inst/cycle) | 0.129 | 0.157 | 0.0283 | +21.9% | improved |
| issue_slot_util (%) | 12.87 | 15.70 | 2.83 | +21.9% | improved |
| branch_pct (%) | 0.0738 | 0.0582 | -0.0156 | -21.1% | changed |
| PC samples collected. Zero means sampling ran but caught nothing, commonly because the kernel is shorter than one sampling interval. (sample) | 9,559 | 7,546 | -2,013 | -21.1% | changed |
| gpc_cycles_elapsed (cycles) | 156,558 | 127,867 | -28,691 | -18.3% | changed |
| Denominator for every stall ratio: how many warp-cycles each issued instruction cost. (cycle) | 23.25 | 19.20 | -4.05 | -17.4% | changed |
| l2_cycles_elapsed (cycles) | 149,346 | 123,628 | -25,718 | -17.2% | changed |
| duration_ns (ns) | 87,520 | 72,672 | -14,848 | -17.0% | changed |
| dram_cycles_elapsed (cycles) | 229,056 | 190,259 | -38,797 | -16.9% | changed |
| inst_pipe_adu (%) | 5.58 | 4.65 | -0.924 | -16.6% | changed |
| local_st_inst (inst) | 23,060 | 19,560 | -3,500 | -15.2% | improved |
| 'Mem Pipes Busy' - request-side saturation, distinct from bytes moved. (%) | 32.37 | 37.16 | 4.78 | +14.8% | improved |
| Aggregate memory-pipeline throughput vs peak. (%) | 32.37 | 37.16 | 4.78 | +14.8% | improved |
| shared_conflicts_nway (way) | 264 | 297 | 33 | +12.5% | REGRESSED |
| l2_sectors_total (sector) | 11,082,969 | 9,727,569 | -1,355,400 | -12.2% | changed |
| inst_pipe_lsu (%) | 2.58 | 2.89 | 0.308 | +11.9% | changed |
| pipe_lsu_util (%) | 2.58 | 2.89 | 0.308 | +11.9% | changed |
| inst_pipe_uniform (%) | 3.1 | 3.42 | 0.32 | +10.3% | changed |
| 'Mem Busy' - how much of the time the memory system was moving data. (%) | 31.68 | 34.70 | 3.02 | +9.5% | improved |
| avg_thread_executed (thread) | 49,200 | 44,628 | -4,572 | -9.3% | changed |
| inst_executed_per_opcode | 10,681,131 | 9,702,811 | -978,320 | -9.2% | changed |
| inst_executed (inst) | 10,729,016 | 9,810,285 | -918,731 | -8.6% | changed |
| l2_hit_rate (%) | 69.61 | 63.99 | -5.63 | -8.1% | REGRESSED |
| l2_read_hit_rate (%) | 60.68 | 55.88 | -4.8 | -7.9% | changed |
| Measured SM clock. Compare across reports before comparing their durations. (Hz) | 1,891,223,305 | 1,744,515,832 | -146,707,473 | -7.8% | changed |
| l2_sectors_from_l1 (sector) | 8,826,400 | 8,194,970 | -631,430 | -7.2% | changed |
| l2_sol (%) | 41.87 | 44.36 | 2.49 | +5.9% | improved |
| l2_throughput (%) | 41.87 | 44.36 | 2.49 | +5.9% | improved |
| dram_bytes_write (byte) | 21,317,888 | 22,078,464 | 760,576 | +3.6% | changed |
| l2_miss_from_l1 (sector) | 2,445,496 | 2,361,818 | -83,678 | -3.4% | changed |
| l2_compression_input_sectors (sector) | 1,075,961 | 1,111,727 | 35,766 | +3.3% | changed |
| Share of cycles with no eligible warp - the direct measure of latency exposure. (%) | 87.13 | 84.30 | -2.83 | -3.2% | improved |
| shared_bank_conflicts_ld | 0 | 10 | 10 | from 0 | REGRESSED ((baseline is zero; no ratio)) |

### Raw metric audit appendix

numeric metrics A/B/common: 2139/2139/2139; changed common: 863; A-only/B-only: 0/0. JSON retains every changed raw metric; the Markdown appendix shows the largest 25 relative changes.

### Largest raw metric changes

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| smsp__sass_average_data_bytes_per_sector_mem_local_op_ld.ratio | 2.08 | 2,235 | 2,233 | +107140.0% | changed |
| l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed | 0.177 | 2.4 | 2.22 | +1257.8% | changed |
| lts__t_sectors_srcunit_tex_aperture_device_op_atom_dot_alu_lookup_miss.sum | 2 | 26 | 24 | +1200.0% | changed |
| l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | 38,580 | 401,218 | 362,638 | +940.0% | changed |
| smsp__average_warps_issue_stalled_gmma_per_issue_active.ratio | 0.0178 | 0.155 | 0.137 | +770.8% | changed |
| launch__occupancy_limit_barriers | 4 | 32 | 28 | +700.0% | changed |
| smsp__pcsamp_warps_issue_stalled_warpgroup_arrive_not_issued | 9 | 72 | 63 | +700.0% | changed |
| smsp__pcsamp_warps_issue_stalled_warpgroup_arrive | 10 | 72 | 62 | +620.0% | changed |
| pmsampling:smsp__warps_issue_stalled_mio_throttle.avg | 1,067 | 6,645 | 5,578 | +522.8% | changed |
| smsp__pcsamp_warps_issue_stalled_mio_throttle | 33 | 177 | 144 | +436.4% | changed |
| smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio | 0.0645 | 0.345 | 0.28 | +434.6% | changed |
| smsp__pcsamp_warps_issue_stalled_mio_throttle_not_issued | 28 | 147 | 119 | +425.0% | changed |
| pmsampling:smsp__warps_issue_stalled_no_instruction.avg | 6,743 | 24,925 | 18,181 | +269.6% | changed |
| smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio | 0.398 | 1.46 | 1.07 | +267.4% | changed |
| smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio | 32.09 | 96 | 63.91 | +199.2% | changed |
| smsp__pcsamp_warps_issue_stalled_no_instructions_not_issued | 166 | 479 | 313 | +188.6% | changed |
| smsp__pcsamp_warps_issue_stalled_math_pipe_throttle | 22 | 61 | 39 | +177.3% | changed |
| sass__inst_executed_shared_loads | 38,580 | 106,296 | 67,716 | +175.5% | changed |
| smsp__sass_inst_executed_op_shared_ld.sum | 38,580 | 106,296 | 67,716 | +175.5% | changed |
| smsp__pcsamp_warps_issue_stalled_no_instructions | 186 | 504 | 318 | +171.0% | changed |
| pmsampling:smsp__warps_issue_stalled_math_pipe_throttle.avg | 972 | 2,629 | 1,657 | +170.5% | changed |
| l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss.sum | 59,404 | 127,396 | 67,992 | +114.5% | changed |
| smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio | 0.0665 | 0.141 | 0.0748 | +112.5% | changed |
| smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio | 0.163 | 0.341 | 0.178 | +108.8% | changed |
| sm__inst_executed_pipe_xu.min.pct_of_peak_sustained_active | 0.0654 | 0.136 | 0.0708 | +108.1% | changed |

## Honesty notes

- Deltas in this report share one cause -- the code change between A and B -- but the diff does not establish causality between any two deltas. A stall that fell and a hit rate that rose moved together; whether one produced the other is a question for the source, not for this table.
