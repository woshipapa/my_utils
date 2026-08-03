# NCU Kernel Diagnosis

- report: `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode1_single_buffer.ncu-rep`
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
| stall_long_scoreboard | 1 |
| stall_barrier | 1 |
| shared_bank_conflicts_st | 1 |
| below_roofline | 1 |
| poor_cache_locality | 1 |
| ncu_shipped_rule | 1 |

## 1. pod_fused_device_kernel - 82.0 us

- category: `other` | framework: `-` | arch: `h100/sm_90`
- **verdict: latency_bound**
- Compute 28.5% and memory 36.7% are both below 60% of peak: the kernel is waiting, not working.
- read next: `WarpStateStats`
- roofline: AI = 281.1 FLOP/byte, achieved 314.4 TFLOP/s, ceiling 941.8 TFLOP/s, memory_bound
  - graded against 989 TFLOP/s: bf16 dense peak, no sparsity (tensor_op_counters)
- AI hierarchy: L2 AI=73.21 FLOP/byte | DRAM AI=281.15 FLOP/byte | not collected: L1
  - intensity ratios: L2->DRAM 3.8x
- instruction roofline: 130.9 G warp-inst/s; 12% of issue ceiling; warp-inst per 32B transaction: L2 0.98, DRAM 3.75
  - Warp-instruction throughput is 12% of the issue ceiling: instruction issue is not the limiter, and the FLOP roofline verdict stands.
- SFU check: XU/SFU 0.1% of peak, tensor pipe 29.7% - no special-function pressure
- dominant stall bucket: `device_memory`
- axes: 9 of 14 axes examined. Not examined: communication, latency_launch, host_pipeline, power_clock, multi_gpu. Those axes produced no findings because they were never checked, which is not the same as being clean.

### Where it stalls

| source | samples | dominant stall | line |
|---|---|---|---|
| `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2592` | 2480 | BARRIER | `if (tail_group < TailRmsGroups) {` |
| `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` | 1734 | LONG_SCOREBOARD | `pipeline.consumer_wait(smem_pipe_read, barrier_token);` |
| `bfloat16.h:165` | 753 | LONG_SCOREBOARD | `unsigned bits = (unsigned(storage) << 16);` |
| `barrier.h:426` | 713 | LONG_SCOREBOARD | `: "r"(smem_addr), "r"(phase), "r"(ticks)` |
| `rmsnorm_warpgroup.hpp:2652` | 347 | LONG_SCOREBOARD | `float xf = static_cast<float>(xv.elt[i]);` |
| `cluster_sm90.hpp:194` | 232 | SLEEPING | `: "r"(0xFFFFFFFF));` |
| `gemm.hpp:414` | 223 | WAIT | `gemm(mma, D, A(_,_,k), B(_,_,k), C);` |
| `rmsnorm_warpgroup.hpp:2933` | 187 | BARRIER | `if (first_row >= args.rows) {` |

_8842 sampled stall cycles attributed across 166 source lines. PC sampling is statistical: a difference of a few samples between lines is noise._

### PC sampling (usable)

- 8,842 samples at a 2,048-cycle interval

#### Stalling instructions

One source line compiles to many instructions and they do not stall for the same reason, so this is a level finer than the table above.

| samples | stall | SASS | source |
|---|---|---|---|
| 2480 (28.0%) | BARRIER | `@P0   BRA 0x7f98b77f6190` | `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2592` |
| 1734 (19.6%) | LONG_SCOREBOARD | `@P0   BRA 0x7f98b77bef90` | `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` |
| 542 (6.1%) | LONG_SCOREBOARD | `@!P0  NANOSLEEP.SYNCS 0x989680` | `barrier.h:426` |
| 343 (3.9%) | LONG_SCOREBOARD | `PRMT R14, R8, 0x7732, RZ` | `rmsnorm_warpgroup.hpp:2652` |
| 326 (3.7%) | LONG_SCOREBOARD | `PRMT R2, R8, 0x7732, RZ` | `bfloat16.h:165` |
| 317 (3.6%) | LONG_SCOREBOARD | `IMAD.U32 R13, R8, 0x10000, RZ` | `bfloat16.h:165` |
| 170 (1.9%) | BARRIER | `LDC R7, c[0x0][0x758]` | `rmsnorm_warpgroup.hpp:2933` |
| 120 (1.4%) | SLEEPING | `WARPSYNC.COLLECTIVE R10, 0x7f98b77f77d0` | `cluster_sm90.hpp:194` |

_590 instructions carried a non-zero stall sample. One source line compiles to many instructions and they do not stall for the same reason, which is why this is a level finer than the line view._

- PM sampling unavailable: PM timeline withheld: the sampled data cannot support it. See pm_sampling_validity.

### Signal to source

**Warp stalls dominated by Long Scoreboard**
- correlated via `LONG_SCOREBOARD` (moderate, 80% of those samples)
  - `/workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` 1727 samples (46%) `pipeline.consumer_wait(smem_pipe_read, barrier_token`
  - `/workspace/cutlass/include/cutlass/bfloat16.h:165` 649 samples (17%) `unsigned bits = (unsigned(storage) << 16);`
  - `/workspace/cutlass/include/cutlass/arch/barrier.h:426` 629 samples (17%) `: "r"(smem_addr), "r"(phase), "r"(ticks)`

**Warp stalls dominated by Barrier**
- correlated via `BARRIER` (concentrated, 89% of those samples)
  - `/workspace/horizonal_fuse_kernel/fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2592` 2480 samples (83%) `if (tail_group < TailRmsGroups) {`
  - `/workspace/horizonal_fuse_kernel/rmsnorm_warpgroup.hpp:2933` 170 samples (6%) `if (first_row >= args.rows) {`

**Shared-memory st bank conflicts on 17% of wavefronts**
- correlated via `MIO_THROTTLE` (spread, 9% of those samples)
  - `/workspace/cutlass/include/cute/algorithm/gemm.hpp:414` 23 samples (9%) `gemm(mma, D, A(_,_,k), B(_,_,k), C);`

_Correlation by stall reason, not proof of cause. A line can stall for several reasons at once._

### Findings

- **[medium]** Warp stalls dominated by Long Scoreboard _(at most 1.74x -- upper bound, not a prediction)_
  - Long Scoreboard accounts for 43% of warp cycles per issued instruction (9.99 cycles). Waiting on an L1TEX (global/local/surface/texture) data return - i.e. memory latency. NOTE: this kernel uses warpgroup MMA or TMA, and on such kernels Long Scoreboard absorbs warpgroup synchronisation as well as global-memory latency -- NVIDIA staff document a case where GMMA and Barrier stalls summed to within 3% of it. Read it together with the gmma and barrier stalls below before concluding memory.
  - speedup bound: removing this stall entirely would buy at most 74% -- an upper bound, not a prediction. Share-removal model on the closed stall stack: removing this stall's 43% share of issue-active stall cycles rescales predicted kernel cycles to at best 1/(1-0.43) = 1.74x; assumes the stall is entirely removable, the stack's fractional shares are exact, and no new limiter appears.
  - fix: Stage reused data in shared memory.
  - fix: Increase instruction-level parallelism so more loads are in flight.
  - fix: Unroll loops to overlap independent loads.
  - fix: Check the gmma and barrier stalls first: on a warpgroup-MMA kernel those are part of what Long Scoreboard is counting.
- **[medium]** Warp stalls dominated by Barrier _(at most 1.57x -- upper bound, not a prediction)_
  - Barrier accounts for 36% of warp cycles per issued instruction (8.46 cycles). Waiting at a CTA barrier for sibling warps.
  - speedup bound: removing this stall entirely would buy at most 57% -- an upper bound, not a prediction. Share-removal model on the closed stall stack: removing this stall's 36% share of issue-active stall cycles rescales predicted kernel cycles to at best 1/(1-0.36) = 1.57x; assumes the stall is entirely removable, the stack's fractional shares are exact, and no new limiter appears.
  - fix: Balance the work done on divergent paths before __syncthreads().
  - fix: Split blocks of >=512 threads into smaller cooperating groups.
  - fix: Replace __syncthreads() with __syncwarp() where only warp scope is needed.
- **[medium]** Shared-memory st bank conflicts on 17% of wavefronts _(up to 1.20x)_
  - 21,221 conflicts across 125,976 st wavefronts. Conflicting lanes serialise, so the shared-memory pipe delivers a fraction of its bandwidth.
  - fix: Pad the shared array's leading dimension (classic +1 element padding).
  - fix: Swizzle the shared layout so a warp's lanes hit distinct banks.
  - fix: For tensor-core tiles, use the CUTLASS-style XOR swizzle instead of padding.
- **[medium]** Kernel sits well below its roofline ceiling
  - Achieved 314.4 TFLOP/s against an attainable 941.8 TFLOP/s at arithmetic intensity 281.1 FLOP/byte (33% of ceiling). Being far below *both* roofs points at latency or occupancy rather than bandwidth or math.
  - fix: Check warp stalls and occupancy before optimising math or memory layout.
  - fix: Confirm the kernel is large enough to fill the GPU (waves per SM >= 1).
- **[medium]** L2 read hit rate is far below the write hit rate
  - L2 hits 62% of reads and 100% of writes. The aggregate hit rate hides this: the read path is missing to DRAM while the other is served from cache.
  - fix: Look at the read access pattern specifically -- the aggregate hit rate will not show the improvement or the regression.
- **[medium]** IssueSlotUtilization
  - Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 7.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 16 warps per scheduler, this workload allocates an average of 2.99 active warps per scheduler, but only an average of 0.15 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the @section:WarpStateStats:Warp State Statistics@ and @section:SourceCounters:Source Counters@ sections can help, too.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
- **[medium]** FP32 Non-Fused Instructions
  - This kernel executes 397312 fused and 989848 non-fused FP32 instructions. By converting pairs of non-fused instructions to their @url:fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP32 performance could be increased by up to 36% (relative to its current performance). Check the Source page to identify where this kernel executes FP32 instructions.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
- **[medium]** TheoreticalOccupancy
  - The 3.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 16. This kernel's theoretical occupancy (18.8%) is limited by the number of required registers, and the required amount of shared memory.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
