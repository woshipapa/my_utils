# NCU Kernel Diagnosis

- report: `/Users/papa/Desktop/kernel_traces/ncu_kernels/cta_pingpong_128x128x64_auto_no_barrier_rms.ncu-rep`
- kernels analyzed: 1
- gpu: H100 SXM5

## Bottleneck classes

| verdict | kernels |
|---|---|
| latency_bound | 1 |

## Most frequent findings

| finding | kernels |
|---|---|
| scheduler | 2 |
| stall_barrier | 1 |
| below_roofline | 1 |
| register_spilling | 1 |
| poor_cache_locality | 1 |
| ncu_shipped_rule | 1 |
| unit_hit_rate | 1 |

## 1. pod_fused_device_kernel - 72.7 us

- category: `other` | framework: `-` | arch: `h100/sm_90`
- **verdict: latency_bound**
- Compute 37.6% and memory 37.2% are both below 60% of peak: the kernel is waiting, not working.
- read next: `WarpStateStats`
- roofline: AI = 284.8 FLOP/byte, achieved 354.6 TFLOP/s, ceiling 954.1 TFLOP/s, memory_bound
  - graded against 989 TFLOP/s: bf16 dense peak, no sparsity (tensor_op_counters)
- AI hierarchy: L2 AI=82.79 FLOP/byte | DRAM AI=284.79 FLOP/byte | not collected: L1
  - intensity ratios: L2->DRAM 3.4x
- instruction roofline: 135.0 G warp-inst/s; 15% of issue ceiling; warp-inst per 32B transaction: L2 1.01, DRAM 3.47
  - Warp-instruction throughput is 15% of the issue ceiling: instruction issue is not the limiter, and the FLOP roofline verdict stands.
- SFU check: XU/SFU 0.2% of peak, tensor pipe 40.6% - no special-function pressure
- dominant stall bucket: `synchronization`
- axes: 9 of 14 axes examined. Not examined: communication, latency_launch, host_pipeline, power_clock, multi_gpu. Those axes produced no findings because they were never checked, which is not the same as being clean.
- **measurement caveat**: The SM ran at 88% of its rated clock (1745 of 1980 MHz) while DRAM ran at 100% of its rated clock (2618 of 2619 MHz). Nsight Compute's --clock-control base lowers the SM clock but cannot lower the HBM clock on H100/B200-class parts, so the measured compute:memory balance is biased toward looking memory-rich (compute-poor) relative to full-clock operation. The counters are not wrong; verdicts that weigh compute against memory bandwidth (Speed-of-Light compute vs memory, roofline side) lean memory-rich. For an unbiased balance, pin the SM clock externally (nvidia-smi -lgc <clock>,<clock>) and profile with --clock-control none.

### Where it stalls

| source | samples | dominant stall | line |
|---|---|---|---|
| `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:1976` | 2153 | BARRIER | `__syncthreads();` |
| `rmsnorm_warpgroup.hpp:538` | 794 | LONG_SCOREBOARD | `float gf = static_cast<float>(gv.elt[i]);` |
| `rmsnorm_warpgroup.hpp:462` | 588 | LONG_SCOREBOARD | `__syncwarp();` |
| `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:1852` | 479 | BARRIER | `int const op = selected_op;` |
| `sm90_mma_tma_gmma_ss_warpspecialized.hpp:547` | 455 | BARRIER | `warpgroup_wait<K_PIPE_MMAS>();` |
| `gemm.hpp:414` | 312 | MIO_THROTTLE | `gemm(mma, D, A(_,_,k), B(_,_,k), C);` |
| `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` | 270 | LONG_SCOREBOARD | `pipeline.consumer_wait(smem_pipe_read, barrier_token);` |
| `sm90_pipeline.hpp:208` | 261 | NO_INSTRUCTIONS | `if (index_ == Stages) {` |

_7546 sampled stall cycles attributed across 164 source lines. PC sampling is statistical: a difference of a few samples between lines is noise._

### PC sampling (usable)

- 7,546 samples at a 2,048-cycle interval

#### Stalling instructions

One source line compiles to many instructions and they do not stall for the same reason, so this is a level finer than the table above.

| samples | stall | SASS | source |
|---|---|---|---|
| 2152 (28.5%) | BARRIER | `BRA 0x7fd7317bda60` | `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:1976` |
| 789 (10.5%) | LONG_SCOREBOARD | `IMAD.U32 R15, R4, 0x10000, RZ` | `rmsnorm_warpgroup.hpp:538` |
| 587 (7.8%) | LONG_SCOREBOARD | `WARPSYNC.ALL` | `rmsnorm_warpgroup.hpp:462` |
| 462 (6.1%) | BARRIER | `UMOV UR16, 0x400` | `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:1852` |
| 455 (6.0%) | BARRIER | `WARPGROUP.DEPBAR.LE gsb0, 0x1` | `sm90_mma_tma_gmma_ss_warpspecialized.hpp:547` |
| 270 (3.6%) | LONG_SCOREBOARD | `@P0   BRA 0x7fd7317efb10` | `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` |
| 108 (1.4%) | LONG_SCOREBOARD | `@P0   BRA 0x7fd7317ef690` | `sm90_mma_tma_gmma_ss_warpspecialized.hpp:487` |
| 77 (1.0%) | WARPGROUP_ARRIVE | `HGMMA.64x128x16.F32.BF16 R88, gdesc[UR16].tn` | `sm90_pipeline.hpp:208` |

_496 instructions carried a non-zero stall sample. One source line compiles to many instructions and they do not stall for the same reason, which is why this is a level finer than the line view._

### PM sampling (utilisation over time)

- 715 time buckets at 1504 ns, spanning 300.8 us
- kernel active window: 48 buckets = 70.7 us
- each bucket value is the counter **accumulated over that window**, not an instantaneous reading; `.avg` is the average across SM instances, not across time
- _The sampled span covers the whole profiling session. Under kernel replay that is several executions of this kernel, so a span larger than the kernel duration is expected._

| pass | metric | peak (1 bucket) | mean over active window | non-zero share | mean over whole session |
|---|---|---|---|---|---|
| 4 | `sm__pipe_tensor_cycles_active_realtime.avg.pct` | 100.0% | 39.3% | 67% | 9.4% |
| 0 | `l1tex__t_sector_hit_rate.pct` | 100.0% | 20.3% | 54% | 4.6% |
| 4 | `dramc__throughput.avg.pct_of_peak_sustained_el` | 46.1% | 19.1% | 100% | 4.6% |
| 4 | `dramc__read_throughput.avg.pct_of_peak_sustain` | 41.2% | 14.2% | 98% | 3.4% |
| 4 | `sm__inst_executed_realtime.avg.pct_of_peak_sus` | 33.0% | 14.8% | 98% | 3.5% |
| 4 | `sm__inst_executed_pipe_alu_realtime.avg.pct_of` | 27.3% | 9.5% | 98% | 2.3% |
| 4 | `dramc__write_throughput.avg.pct_of_peak_sustai` | 17.2% | 4.9% | 81% | 1.2% |
| 5 | `smsp__warps_issue_stalled_long_scoreboard.avg` | 7336.0 | 2091.1 | 100% | 525.5 |

_`mean_in_active_window` divides by the buckets in which the kernel was running, bounded per pass group (48 of 715 buckets). `mean_all_buckets` divides by the sampler's whole session, most of which the kernel was not running in -- it is reported for transparency, not for comparison._

**These 26 series come from 7 replay passes, each a separate execution with its own capture window and bucket count. Series within one pass share a clock and can be compared bucket for bucket; series from different passes cannot. Comparing them by bucket index compares different moments of different runs. SM clock drifted at least 4.6% across the replay passes of this collection -- metrics multiplexed across passes mix different clock states, and PM-sampling series from different passes are additionally skewed relative to each other.**

**sm__pipe_tensor_cycles_active_realtime peaks at 100% and averages 39% across the kernel's active window (non-zero in 67% of that window); l1tex__t_sector_hit_rate peaks at 100% and averages 20% across the kernel's active window (non-zero in 54% of that window)**

_A unit that peaks high and averages low is not inefficient; it is idle most of the time. The fix is to keep it busy, not to make it faster._

_Series marked `unit: count` are raw per-bucket counts, not percentages; they have no ceiling to be measured against._

### Signal to source

**Warp stalls dominated by Barrier**
- correlated via `BARRIER` (concentrated, 99% of those samples)
  - `/workspace/horizonal_fuse_kernel/fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:1976` 2152 samples (69%) `__syncthreads();`
  - `/workspace/horizonal_fuse_kernel/fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:1852` 459 samples (15%) `int const op = selected_op;`
  - `/workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:547` 452 samples (15%) `warpgroup_wait<K_PIPE_MMAS>();`

**Register spilling to local memory**
- correlated via `LONG_SCOREBOARD` (`LG_THROTTLE` carried no samples in this kernel) (moderate, 79% of those samples)
  - `/workspace/horizonal_fuse_kernel/rmsnorm_warpgroup.hpp:538` 784 samples (38%) `float gf = static_cast<float>(gv.elt[i]);`
  - `/workspace/horizonal_fuse_kernel/rmsnorm_warpgroup.hpp:462` 587 samples (28%) `__syncwarp();`
  - `/workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` 269 samples (13%) `pipeline.consumer_wait(smem_pipe_read, barrier_token`

_Correlation by stall reason, not proof of cause. A line can stall for several reasons at once._

### Findings

- **[high]** Warp stalls dominated by Barrier _(at most 1.71x -- upper bound, not a prediction)_
  - Barrier accounts for 42% of warp cycles per issued instruction (7.98 cycles). Waiting at a CTA barrier for sibling warps.
  - speedup bound: removing this stall entirely would buy at most 71% -- an upper bound, not a prediction. Share-removal model on the closed stall stack: removing this stall's 42% share of issue-active stall cycles rescales predicted kernel cycles to at best 1/(1-0.42) = 1.71x; assumes the stall is entirely removable, the stack's fractional shares are exact, and no new limiter appears.
  - fix: Balance the work done on divergent paths before __syncthreads().
  - fix: Split blocks of >=512 threads into smaller cooperating groups.
  - fix: Replace __syncthreads() with __syncwarp() where only warp scope is needed.
- **[medium]** Kernel sits well below its roofline ceiling
  - Achieved 354.6 TFLOP/s against an attainable 954.1 TFLOP/s at arithmetic intensity 284.8 FLOP/byte (37% of ceiling). Being far below *both* roofs points at latency or occupancy rather than bandwidth or math.
  - fix: Check warp stalls and occupancy before optimising math or memory layout.
  - fix: Confirm the kernel is large enough to fill the GPU (waves per SM >= 1).
- **[medium]** Register spilling to local memory
  - 5,632 local loads and 19,560 local stores with 168 registers/thread. Worse, only 52% of local loads hit L1, so spills are reaching L2 or DRAM. 168 registers x 384 threads = 64,512 of the 65,536 available, and 65,536/384 = 170.7 is the per-thread ceiling. This is the largest allocation that fits, so it was chosen, not overrun. The spilling is the cost of that choice; whether it is worth paying is a design question, not a defect. Note the register figure is kernel-level: on a warp-specialized kernel the warpgroup actually spilling is usually the producer, whose budget after warpgroup_reg_dealloc is far smaller than the number above, and only the source shows it.
  - fix: Shorten live ranges or recompute values instead of keeping them alive.
  - fix: Reduce the per-thread tile so the working set fits in registers.
  - fix: If __launch_bounds__ is capping registers, weigh the spills against the occupancy gained.
  - fix: Check for dynamically indexed local arrays - those always spill.
- **[medium]** L2 read hit rate is far below the write hit rate
  - L2 hits 56% of reads and 100% of writes. The aggregate hit rate hides this: the read path is missing to DRAM while the other is served from cache.
  - fix: Look at the read access pattern specifically -- the aggregate hit rate will not show the improvement or the regression.
- **[medium]** IssueSlotUtilization
  - Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 6.4 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 16 warps per scheduler, this workload allocates an average of 3.01 active warps per scheduler, but only an average of 0.21 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the @section:WarpStateStats:Warp State Statistics@ and @section:SourceCounters:Source Counters@ sections can help, too.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
- **[medium]** FP32 Non-Fused Instructions
  - This kernel executes 397312 fused and 908288 non-fused FP32 instructions. By converting pairs of non-fused instructions to their @url:fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP32 performance could be increased by up to 35% (relative to its current performance). Check the Source page to identify where this kernel executes FP32 instructions.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
- **[medium]** TheoreticalOccupancy
  - The 3.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 16. This kernel's theoretical occupancy (18.8%) is limited by the number of required registers, and the required amount of shared memory.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
- **[medium]** L1 data cache / texture unit, per SM hit rate is 0%
  - `l1tex__t_sector_pipe_lsu_mem_global_op_atom_hit_rate.pct` reads 0.0%, so 100% of the 1,602 requests on this path miss and go further out.
  - fix: Compare this against the traffic volume: a low hit rate on a small number of requests costs little.
