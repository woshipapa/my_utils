# NCU Kernel Diagnosis

- report: `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode3_chunk_ring.ncu-rep`
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

## 1. pod_fused_device_kernel - 86.4 us

- category: `other` | framework: `-` | arch: `h100/sm_90`
- **verdict: latency_bound**
- Compute 28.9% and memory 34.7% are both below 60% of peak: the kernel is waiting, not working.
- read next: `WarpStateStats`
- roofline: AI = 287.7 FLOP/byte, achieved 298.2 TFLOP/s, ceiling 963.6 TFLOP/s, memory_bound
  - graded against 989 TFLOP/s: bf16 dense peak, no sparsity (tensor_op_counters)
- AI hierarchy: L2 AI=74.27 FLOP/byte | DRAM AI=287.66 FLOP/byte | not collected: L1
  - intensity ratios: L2->DRAM 3.9x
- instruction roofline: 124.2 G warp-inst/s; 12% of issue ceiling; warp-inst per 32B transaction: L2 0.99, DRAM 3.83
  - Warp-instruction throughput is 12% of the issue ceiling: instruction issue is not the limiter, and the FLOP roofline verdict stands.
- SFU check: XU/SFU 0.1% of peak, tensor pipe 29.9% - no special-function pressure
- dominant stall bucket: `device_memory`
- axes: 9 of 14 axes examined. Not examined: communication, latency_launch, host_pipeline, power_clock, multi_gpu. Those axes produced no findings because they were never checked, which is not the same as being clean.

### Where it stalls

| source | samples | dominant stall | line |
|---|---|---|---|
| `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2592` | 2677 | BARRIER | `if (tail_group < TailRmsGroups) {` |
| `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` | 1786 | LONG_SCOREBOARD | `pipeline.consumer_wait(smem_pipe_read, barrier_token);` |
| `bfloat16.h:165` | 973 | LONG_SCOREBOARD | `unsigned bits = (unsigned(storage) << 16);` |
| `barrier.h:426` | 701 | LONG_SCOREBOARD | `: "r"(smem_addr), "r"(phase), "r"(ticks)` |
| `rmsnorm_warpgroup.hpp:2652` | 375 | LONG_SCOREBOARD | `float xf = static_cast<float>(xv.elt[i]);` |
| `cluster_sm90.hpp:194` | 229 | SLEEPING | `: "r"(0xFFFFFFFF));` |
| `rmsnorm_warpgroup.hpp:2933` | 215 | BARRIER | `if (first_row >= args.rows) {` |
| `gemm.hpp:414` | 172 | WAIT | `gemm(mma, D, A(_,_,k), B(_,_,k), C);` |

_9481 sampled stall cycles attributed across 175 source lines. PC sampling is statistical: a difference of a few samples between lines is noise._

### PC sampling (usable)

- 9,481 samples at a 2,048-cycle interval

#### Stalling instructions

One source line compiles to many instructions and they do not stall for the same reason, so this is a level finer than the table above.

| samples | stall | SASS | source |
|---|---|---|---|
| 2675 (28.2%) | BARRIER | `@P0   BRA 0x7fac837f6190` | `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2592` |
| 1786 (18.8%) | LONG_SCOREBOARD | `@P0   BRA 0x7fac837bef90` | `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` |
| 530 (5.6%) | LONG_SCOREBOARD | `@!P0  NANOSLEEP.SYNCS 0x989680` | `barrier.h:426` |
| 403 (4.3%) | LONG_SCOREBOARD | `IMAD.U32 R13, R8, 0x10000, RZ` | `bfloat16.h:165` |
| 384 (4.1%) | LONG_SCOREBOARD | `PRMT R2, R8, 0x7732, RZ` | `bfloat16.h:165` |
| 375 (4.0%) | LONG_SCOREBOARD | `PRMT R14, R8, 0x7732, RZ` | `rmsnorm_warpgroup.hpp:2652` |
| 186 (2.0%) | BARRIER | `LDC R7, c[0x0][0x758]` | `rmsnorm_warpgroup.hpp:2933` |
| 117 (1.2%) | SLEEPING | `WARPSYNC.COLLECTIVE R10, 0x7fac837f77d0` | `cluster_sm90.hpp:194` |

_721 instructions carried a non-zero stall sample. One source line compiles to many instructions and they do not stall for the same reason, which is why this is a level finer than the line view._

- PM sampling unavailable: PM timeline withheld: the sampled data cannot support it. See pm_sampling_validity.

### Signal to source

**Warp stalls dominated by Long Scoreboard**
- correlated via `LONG_SCOREBOARD` (moderate, 82% of those samples)
  - `/workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` 1777 samples (44%) `pipeline.consumer_wait(smem_pipe_read, barrier_token`
  - `/workspace/cutlass/include/cutlass/bfloat16.h:165` 888 samples (22%) `unsigned bits = (unsigned(storage) << 16);`
  - `/workspace/cutlass/include/cutlass/arch/barrier.h:426` 625 samples (16%) `: "r"(smem_addr), "r"(phase), "r"(ticks)`

**Warp stalls dominated by Barrier**
- correlated via `BARRIER` (concentrated, 88% of those samples)
  - `/workspace/horizonal_fuse_kernel/fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2592` 2675 samples (82%) `if (tail_group < TailRmsGroups) {`
  - `/workspace/horizonal_fuse_kernel/rmsnorm_warpgroup.hpp:2933` 184 samples (6%) `if (first_row >= args.rows) {`

**Shared-memory st bank conflicts on 17% of wavefronts**
- correlated via `MIO_THROTTLE, SHORT_SCOREBOARD` (spread, 9% of those samples)
  - `/workspace/cutlass/include/cute/algorithm/gemm.hpp:414` 18 samples (8%) `gemm(mma, D, A(_,_,k), B(_,_,k), C);`
  - `/workspace/horizonal_fuse_kernel/fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2592` 2 samples (1%) `if (tail_group < TailRmsGroups) {`

_Correlation by stall reason, not proof of cause. A line can stall for several reasons at once._

### Findings

- **[medium]** Warp stalls dominated by Long Scoreboard _(at most 1.75x -- upper bound, not a prediction)_
  - Long Scoreboard accounts for 43% of warp cycles per issued instruction (10.01 cycles). Waiting on an L1TEX (global/local/surface/texture) data return - i.e. memory latency. NOTE: this kernel uses warpgroup MMA or TMA, and on such kernels Long Scoreboard absorbs warpgroup synchronisation as well as global-memory latency -- NVIDIA staff document a case where GMMA and Barrier stalls summed to within 3% of it. Read it together with the gmma and barrier stalls below before concluding memory.
  - speedup bound: removing this stall entirely would buy at most 75% -- an upper bound, not a prediction. Share-removal model on the closed stall stack: removing this stall's 43% share of issue-active stall cycles rescales predicted kernel cycles to at best 1/(1-0.43) = 1.75x; assumes the stall is entirely removable, the stack's fractional shares are exact, and no new limiter appears.
  - fix: Stage reused data in shared memory.
  - fix: Increase instruction-level parallelism so more loads are in flight.
  - fix: Unroll loops to overlap independent loads.
  - fix: Check the gmma and barrier stalls first: on a warpgroup-MMA kernel those are part of what Long Scoreboard is counting.
- **[medium]** Warp stalls dominated by Barrier _(at most 1.57x -- upper bound, not a prediction)_
  - Barrier accounts for 36% of warp cycles per issued instruction (8.49 cycles). Waiting at a CTA barrier for sibling warps.
  - speedup bound: removing this stall entirely would buy at most 57% -- an upper bound, not a prediction. Share-removal model on the closed stall stack: removing this stall's 36% share of issue-active stall cycles rescales predicted kernel cycles to at best 1/(1-0.36) = 1.57x; assumes the stall is entirely removable, the stack's fractional shares are exact, and no new limiter appears.
  - fix: Balance the work done on divergent paths before __syncthreads().
  - fix: Split blocks of >=512 threads into smaller cooperating groups.
  - fix: Replace __syncthreads() with __syncwarp() where only warp scope is needed.
- **[medium]** Shared-memory st bank conflicts on 17% of wavefronts _(up to 1.20x)_
  - 19,773 conflicts across 117,301 st wavefronts. Conflicting lanes serialise, so the shared-memory pipe delivers a fraction of its bandwidth.
  - fix: Pad the shared array's leading dimension (classic +1 element padding).
  - fix: Swizzle the shared layout so a warp's lanes hit distinct banks.
  - fix: For tensor-core tiles, use the CUTLASS-style XOR swizzle instead of padding.
- **[medium]** Kernel sits well below its roofline ceiling
  - Achieved 298.2 TFLOP/s against an attainable 963.6 TFLOP/s at arithmetic intensity 287.7 FLOP/byte (31% of ceiling). Being far below *both* roofs points at latency or occupancy rather than bandwidth or math.
  - fix: Check warp stalls and occupancy before optimising math or memory layout.
  - fix: Confirm the kernel is large enough to fill the GPU (waves per SM >= 1).
- **[medium]** L2 read hit rate is far below the write hit rate
  - L2 hits 60% of reads and 100% of writes. The aggregate hit rate hides this: the read path is missing to DRAM while the other is served from cache.
  - fix: Look at the read access pattern specifically -- the aggregate hit rate will not show the improvement or the regression.
- **[medium]** IssueSlotUtilization
  - Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 7.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 16 warps per scheduler, this workload allocates an average of 3.01 active warps per scheduler, but only an average of 0.15 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the @section:WarpStateStats:Warp State Statistics@ and @section:SourceCounters:Source Counters@ sections can help, too.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
- **[medium]** FP32 Non-Fused Instructions
  - This kernel executes 397312 fused and 989848 non-fused FP32 instructions. By converting pairs of non-fused instructions to their @url:fused:https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point@, higher-throughput equivalent, the achieved FP32 performance could be increased by up to 36% (relative to its current performance). Check the Source page to identify where this kernel executes FP32 instructions.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
- **[medium]** TheoreticalOccupancy
  - The 3.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 16. This kernel's theoretical occupancy (18.8%) is limited by the number of required registers, and the required amount of shared memory.
  - fix: Nsight Compute reported this speedup as LOCAL to its section, not for the whole kernel. Weight it by that section's share of runtime before treating it as a kernel-level win.
