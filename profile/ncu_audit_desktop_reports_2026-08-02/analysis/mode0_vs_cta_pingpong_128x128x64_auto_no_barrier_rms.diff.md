# NCU Report Diff (A -> B)

- A (baseline): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode0_register_baseline.ncu-rep`
- B (candidate): `/Users/papa/Desktop/kernel_traces/ncu_kernels/cta_pingpong_128x128x64_auto_no_barrier_rms.ncu-rep`
- GPU: H100 SXM5
- matched kernels: 1

## Clock guard

**WARNING: SM clocks differ by more than 1% on 1 of 1 matched kernel(s). Raw-time deltas for those kernels are NOT speedups; read the clock-normalised and elapsed-cycle figures instead.**

## `pod_fused_device_kernel` (grid 132, block 384)

### Duration

| quantity | A | B | B/A |
|---|---|---|---|
| raw duration (ns) -- confounded by clock, NOT a speedup | 87,520 | 72,672 | 0.830 |
| SM clock (MHz) | 1,891 | 1,745 | 0.922 |
| clock-normalised duration ratio | - | - | 0.766 |
| elapsed GPC cycles (clock-independent) | 156,558 | 127,867 | 0.817 |

raw durations are NOT comparable as a speedup (SM clocks 1891 vs 1745 MHz); clock-normalised, B runs at 0.766x of A (raw 0.830x contains the clock change); the clock-independent elapsed-cycle ratio is 0.817x. NOTE: the clock-normalised duration ratio (0.766x) and the elapsed-cycle ratio (0.817x) disagree by 6.2%, which means the clock varied between replay passes; trust neither figure to better than that.

- guard: the two measurements ran at different SM clocks (1891 vs 1745 MHz, -7.8%). A duration ratio therefore mixes the change in the code with the change in the clock; compare cycles, or lock the clock and re-measure

- caveat: Clocks were not confirmed locked on both sides, so small differences may be clock rather than code.

### What changed the verdict

- bottleneck verdict unchanged: `latency_bound`
- disappeared (medium): **Warp stalls dominated by Long Scoreboard** [stall_long_scoreboard]
- disappeared (medium): **Shared-memory st bank conflicts on 16% of wavefronts** [shared_bank_conflicts_st]
- escalated: Warp stalls dominated by Barrier [stall_barrier] medium -> high
- escalated: Register spilling to local memory [register_spilling] low -> medium
- 5 finding(s) unchanged on both sides

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
| L1/TEX sector hit rate (%) | 50.31 | 28.90 | -21.42 | -42.6% | - -> - | REGRESSED |
| L2 sector hit rate (%) | 69.61 | 63.99 | -5.63 | -8.1% | 11,082,969 -> 9,727,569 | REGRESSED |
| L2 read hit rate (%) | 60.68 | 55.88 | -4.8 | -7.9% | - -> - | REGRESSED |
| Local-load L1TEX hit rate (%) | 96.53 | 51.97 | -44.56 | -46.2% | 260,532 -> 22,512 | REGRESSED |
| Local-store L1TEX hit rate (%) | 35.16 | 0.0502 | -35.11 | -99.9% | 91,616 -> 127,460 | REGRESSED |
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

## Honesty notes

- Deltas in this report share one cause -- the code change between A and B -- but the diff does not establish causality between any two deltas. A stall that fell and a hit rate that rose moved together; whether one produced the other is a question for the source, not for this table.
