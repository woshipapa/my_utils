# NCU Kernel Diagnosis

- report: `/Users/papa/Desktop/kernel_traces/ncu_kernels/producer_warp_128x128x64_auto_w3.ncu-rep`
- kernels analyzed: 1
- gpu: H100 SXM5

## Bottleneck classes

| verdict | kernels |
|---|---|
| latency_bound | 1 |

## Most frequent findings

| finding | kernels |
|---|---|
| poor_cache_locality | 2 |
| stall_long_scoreboard | 1 |
| stall_barrier | 1 |
| shared_bank_conflicts_st | 1 |
| measurement_above_physical_limit | 1 |
| below_roofline | 1 |
| register_spilling | 1 |

## 1. pod_fused_device_kernel - 85.4 us

- category: `other` | framework: `-` | arch: `h100/sm_90`
- **verdict: latency_bound**
- Compute 32.1% and memory 43.0% are both below 60% of peak: the kernel is waiting, not working.
- read next: `WarpStateStats`
- roofline: AI = 309.4 FLOP/byte, achieved 301.6 TFLOP/s, ceiling 989.4 TFLOP/s, compute_bound
  - graded against 989 TFLOP/s: bf16 dense peak, no sparsity (tensor_op_counters)
- AI hierarchy: L2 AI=61.47 FLOP/byte | DRAM AI=309.35 FLOP/byte | not collected: L1
  - intensity ratios: L2->DRAM 5.0x
  - L2-to-DRAM intensity ratio is 5.0x: the L2 cache is absorbing most of the traffic that reaches it.
- instruction roofline: 122.5 G warp-inst/s; 13% of issue ceiling; warp-inst per 32B transaction: L2 0.80, DRAM 4.02
  - Warp-instruction throughput is 13% of the issue ceiling: instruction issue is not the limiter, and the FLOP roofline verdict stands.
- SFU check: XU/SFU 0.1% of peak, tensor pipe 33.7% - no special-function pressure
- dominant stall bucket: `device_memory`
- axes: 9 of 14 axes examined. Not examined: communication, latency_launch, host_pipeline, power_clock, multi_gpu. Those axes produced no findings because they were never checked, which is not the same as being clean.
- **measurement caveat**: The SM ran at 88% of its rated clock (1737 of 1980 MHz) while DRAM ran at 100% of its rated clock (2618 of 2619 MHz). Nsight Compute's --clock-control base lowers the SM clock but cannot lower the HBM clock on H100/B200-class parts, so the measured compute:memory balance is biased toward looking memory-rich (compute-poor) relative to full-clock operation. The counters are not wrong; verdicts that weigh compute against memory bandwidth (Speed-of-Light compute vs memory, roofline side) lean memory-rich. For an unbiased balance, pin the SM clock externally (nvidia-smi -lgc <clock>,<clock>) and profile with --clock-control none.

### Where it stalls

| source | samples | dominant stall | line |
|---|---|---|---|
| `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2528` | 1760 | BARRIER | `if (tail_group < TailRmsGroups) {` |
| `bfloat16.h:165` | 1158 | LONG_SCOREBOARD | `unsigned bits = (unsigned(storage) << 16);` |
| `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` | 789 | LONG_SCOREBOARD | `pipeline.consumer_wait(smem_pipe_read, barrier_token);` |
| `rmsnorm_warpgroup.hpp:1389` | 542 | LONG_SCOREBOARD | `float xf = static_cast<float>(xv.elt[i]);` |
| `rmsnorm_warpgroup.hpp:2426` | 527 | LONG_SCOREBOARD | `float xf = static_cast<float>(xv.elt[i]);` |
| `barrier.h:426` | 504 | LONG_SCOREBOARD | `: "r"(smem_addr), "r"(phase), "r"(ticks)` |
| `rmsnorm_warpgroup.hpp:2390` | 474 | LONG_SCOREBOARD | `float xf = static_cast<float>(xv.elt[i]);` |
| `gemm.hpp:414` | 258 | WAIT | `gemm(mma, D, A(_,_,k), B(_,_,k), C);` |

_9407 sampled stall cycles attributed across 184 source lines. PC sampling is statistical: a difference of a few samples between lines is noise._

### PC sampling (usable)

- 9,407 samples at a 2,048-cycle interval

#### Stalling instructions

One source line compiles to many instructions and they do not stall for the same reason, so this is a level finer than the table above.

| samples | stall | SASS | source |
|---|---|---|---|
| 1760 (18.7%) | BARRIER | `@P0   BRA 0x7fe2737e5e40` | `fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2528` |
| 789 (8.4%) | LONG_SCOREBOARD | `@P0   BRA 0x7fe2737bf320` | `sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` |
| 540 (5.7%) | LONG_SCOREBOARD | `PRMT R19, R8, 0x7732, RZ` | `rmsnorm_warpgroup.hpp:1389` |
| 526 (5.6%) | LONG_SCOREBOARD | `IMAD.U32 R2, R12, 0x10000, RZ` | `rmsnorm_warpgroup.hpp:2426` |
| 469 (5.0%) | LONG_SCOREBOARD | `PRMT R14, R8, 0x7732, RZ` | `rmsnorm_warpgroup.hpp:2390` |
| 444 (4.7%) | LONG_SCOREBOARD | `PRMT R2, R8, 0x7732, RZ` | `bfloat16.h:165` |
| 411 (4.4%) | LONG_SCOREBOARD | `IMAD.U32 R13, R8, 0x10000, RZ` | `bfloat16.h:165` |
| 390 (4.1%) | LONG_SCOREBOARD | `@!P0  NANOSLEEP.SYNCS 0x989680` | `barrier.h:426` |

_606 instructions carried a non-zero stall sample. One source line compiles to many instructions and they do not stall for the same reason, which is why this is a level finer than the line view._

### PM sampling (utilisation over time)

- 200 time buckets at 1504 ns, spanning 290.3 us
- kernel active window: 57 buckets = 84.2 us
- each bucket value is the counter **accumulated over that window**, not an instantaneous reading; `.avg` is the average across SM instances, not across time
- _The sampled span covers the whole profiling session. Under kernel replay that is several executions of this kernel, so a span larger than the kernel duration is expected._

| pass | metric | peak (1 bucket) | mean over active window | non-zero share | mean over whole session |
|---|---|---|---|---|---|
| 6 | `sm__pipe_tensor_cycles_active_realtime.avg.pct` | 92.6% | 31.9% | 70% | 9.4% |
| 0 | `l1tex__t_sector_hit_rate.pct` | 87.7% | 22.9% | 98% | 6.8% |
| 6 | `dramc__throughput.avg.pct_of_peak_sustained_el` | 27.1% | 14.6% | 100% | 4.3% |
| 6 | `dramc__read_throughput.avg.pct_of_peak_sustain` | 26.4% | 11.2% | 100% | 3.3% |
| 6 | `sm__inst_executed_realtime.avg.pct_of_peak_sus` | 18.8% | 13.0% | 98% | 3.8% |
| 6 | `sm__inst_executed_pipe_alu_realtime.avg.pct_of` | 18.1% | 9.1% | 98% | 2.7% |
| 6 | `dramc__write_throughput.avg.pct_of_peak_sustai` | 8.1% | 3.4% | 93% | 1.0% |
| 3 | `smsp__warps_issue_stalled_long_scoreboard.avg` | 5304.0 | 4138.0 | 100% | 1204.0 |

_`mean_in_active_window` divides by the buckets in which the kernel was running, bounded per pass group (57 of 200 buckets). `mean_all_buckets` divides by the sampler's whole session, most of which the kernel was not running in -- it is reported for transparency, not for comparison._

**These 26 series come from 7 replay passes, each a separate execution with its own capture window and bucket count. Series within one pass share a clock and can be compared bucket for bucket; series from different passes cannot. Comparing them by bucket index compares different moments of different runs. SM clock drifted at least 4.9% across the replay passes of this collection -- metrics multiplexed across passes mix different clock states, and PM-sampling series from different passes are additionally skewed relative to each other.**

**sm__pipe_tensor_cycles_active_realtime peaks at 93% and averages 32% across the kernel's active window (non-zero in 70% of that window); l1tex__t_sector_hit_rate peaks at 88% and averages 23% across the kernel's active window (non-zero in 98% of that window)**

_A unit that peaks high and averages low is not inefficient; it is idle most of the time. The fix is to keep it busy, not to make it faster._

_Series marked `unit: count` are raw per-bucket counts, not percentages; they have no ceiling to be measured against._

### Signal to source

**Warp stalls dominated by Long Scoreboard**
- correlated via `LONG_SCOREBOARD` (moderate, 50% of those samples)
  - `/workspace/cutlass/include/cutlass/bfloat16.h:165` 1064 samples (22%) `unsigned bits = (unsigned(storage) << 16);`
  - `/workspace/cutlass/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp:533` 781 samples (16%) `pipeline.consumer_wait(smem_pipe_read, barrier_token`
  - `/workspace/horizonal_fuse_kernel/rmsnorm_warpgroup.hpp:1389` 539 samples (11%) `float xf = static_cast<float>(xv.elt[i]);`

**Warp stalls dominated by Barrier**
- correlated via `BARRIER` (concentrated, 70% of those samples)
  - `/workspace/horizonal_fuse_kernel/fused_gemm_rmsnorm_sm_aware_pingpong_sm90.hpp:2528` 1760 samples (70%) `if (tail_group < TailRmsGroups) {`

**Shared-memory st bank conflicts on 11% of wavefronts**
- correlated via `MIO_THROTTLE` (spread, 24% of those samples)
  - `/workspace/cutlass/include/cute/algorithm/gemm.hpp:414` 79 samples (24%) `gemm(mma, D, A(_,_,k), B(_,_,k), C);`

_Correlation by stall reason, not proof of cause. A line can stall for several reasons at once._

### Findings

- **[medium]** Warp stalls dominated by Long Scoreboard _(up to 2.01x)_
  - Long Scoreboard accounts for 50% of warp cycles per issued instruction (10.30 cycles). Waiting on an L1TEX (global/local/surface/texture) data return - i.e. memory latency. NOTE: this kernel uses warpgroup MMA or TMA, and on such kernels Long Scoreboard absorbs warpgroup synchronisation as well as global-memory latency -- NVIDIA staff document a case where GMMA and Barrier stalls summed to within 3% of it. Read it together with the gmma and barrier stalls below before concluding memory.
  - speedup bound withheld: No speedup bound: the stall stack failed its closure check -- the disjoint stall states sum to 105% of the reported warp latency (two replay passes disagree), so every share is computed against a total its own components exceed.
  - fix: Stage reused data in shared memory.
  - fix: Increase instruction-level parallelism so more loads are in flight.
  - fix: Unroll loops to overlap independent loads.
  - fix: Check the gmma and barrier stalls first: on a warpgroup-MMA kernel those are part of what Long Scoreboard is counting.
- **[medium]** Warp stalls dominated by Barrier _(up to 1.45x)_
  - Barrier accounts for 31% of warp cycles per issued instruction (6.35 cycles). Waiting at a CTA barrier for sibling warps.
  - speedup bound withheld: No speedup bound: the stall stack failed its closure check -- the disjoint stall states sum to 105% of the reported warp latency (two replay passes disagree), so every share is computed against a total its own components exceed.
  - fix: Balance the work done on divergent paths before __syncthreads().
  - fix: Split blocks of >=512 threads into smaller cooperating groups.
  - fix: Replace __syncthreads() with __syncwarp() where only warp scope is needed.
- **[medium]** Shared-memory st bank conflicts on 11% of wavefronts _(up to 1.12x)_
  - 15,080 conflicts across 140,389 st wavefronts. Conflicting lanes serialise, so the shared-memory pipe delivers a fraction of its bandwidth.
  - fix: Pad the shared array's leading dimension (classic +1 element padding).
  - fix: Swizzle the shared layout so a warp's lanes hit distinct banks.
  - fix: For tensor-core tiles, use the CUTLASS-style XOR swizzle instead of padding.
- **[medium]** Stall states sum to more than the total they partition
  - The stall reasons account for 104.9% of warp latency (21.53 against a reported 20.52 cycles per issued instruction). These states are disjoint, so they cannot exceed their own total: the two figures come from different replay passes and disagree. Every share below is computed against the smaller of them and is correspondingly inflated.
  - fix: Re-collect with locked clocks; a cross-pass disagreement of this size makes severity boundaries unreliable.
- **[medium]** Kernel sits well below its roofline ceiling
  - Achieved 301.6 TFLOP/s against an attainable 989.4 TFLOP/s at arithmetic intensity 309.4 FLOP/byte (30% of ceiling). Being far below *both* roofs points at latency or occupancy rather than bandwidth or math.
  - fix: Check warp stalls and occupancy before optimising math or memory layout.
  - fix: Confirm the kernel is large enough to fill the GPU (waves per SM >= 1).
- **[medium]** Register spilling to local memory
  - 74,386 local loads and 22,897 local stores with 168 registers/thread. Worse, only 37% of local loads hit L1, so spills are reaching L2 or DRAM. 168 registers x 384 threads = 64,512 of the 65,536 available, and 65,536/384 = 170.7 is the per-thread ceiling. This is the largest allocation that fits, so it was chosen, not overrun. The spilling is the cost of that choice; whether it is worth paying is a design question, not a defect. Note the register figure is kernel-level: on a warp-specialized kernel the warpgroup actually spilling is usually the producer, whose budget after warpgroup_reg_dealloc is far smaller than the number above, and only the source shows it.
  - fix: Shorten live ranges or recompute values instead of keeping them alive.
  - fix: Reduce the per-thread tile so the working set fits in registers.
  - fix: If __launch_bounds__ is capping registers, weigh the spills against the occupancy gained.
  - fix: Check for dynamically indexed local arrays - those always spill.
- **[medium]** L2 read hit rate is far below the write hit rate
  - L2 hits 69% of reads and 100% of writes. The aggregate hit rate hides this: the read path is missing to DRAM while the other is served from cache.
  - fix: Look at the read access pattern specifically -- the aggregate hit rate will not show the improvement or the regression.
- **[medium]** Compute-bound at DRAM but L2-bandwidth-bound: cache-blocking opportunity
  - Against DRAM the kernel clears the ridge (AI 309.4 > 295.3 FLOP/byte), but at L2 its intensity is 61.5 against an L2 ridge of 82.5: it asks more L2 bandwidth per FLOP than the machine has. Blocking for L1/shared memory raises the L2-level intensity directly; more DRAM bandwidth changes nothing.
  - fix: Block for L1/shared memory so each L2 line is reused before eviction; the L2-level AI must rise above 82 FLOP/byte before the math peak is reachable.
