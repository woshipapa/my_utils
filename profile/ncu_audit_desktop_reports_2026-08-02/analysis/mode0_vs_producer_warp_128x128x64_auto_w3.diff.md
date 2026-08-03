# NCU Report Diff (A -> B)

- A (baseline): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode0_register_baseline.ncu-rep`
- B (candidate): `/Users/papa/Desktop/kernel_traces/ncu_kernels/producer_warp_128x128x64_auto_w3.ncu-rep`
- GPU: H100 SXM5
- matched kernels: 1

## Clock guard

**WARNING: SM clocks differ by more than 1% on 1 of 1 matched kernel(s). Raw-time deltas for those kernels are NOT speedups; read the clock-normalised and elapsed-cycle figures instead.**

## `pod_fused_device_kernel` (grid 132, block 384)

### Duration

| quantity | A | B | B/A |
|---|---|---|---|
| raw duration (ns) -- confounded by clock, NOT a speedup | 87,520 | 85,440 | 0.976 |
| SM clock (MHz) | 1,891 | 1,737 | 0.918 |
| clock-normalised duration ratio | - | - | 0.897 |
| elapsed GPC cycles (clock-independent) | 156,558 | 149,794 | 0.957 |

raw durations are NOT comparable as a speedup (SM clocks 1891 vs 1737 MHz); clock-normalised, B runs at 0.897x of A (raw 0.976x contains the clock change); the clock-independent elapsed-cycle ratio is 0.957x. NOTE: the clock-normalised duration ratio (0.897x) and the elapsed-cycle ratio (0.957x) disagree by 6.3%, which means the clock varied between replay passes; trust neither figure to better than that.

- guard: the two measurements ran at different SM clocks (1891 vs 1737 MHz, -8.2%). A duration ratio therefore mixes the change in the code with the change in the clock; compare cycles, or lock the clock and re-measure

- caveat: Clocks were not confirmed locked on both sides, so small differences may be clock rather than code.

### What changed the verdict

- bottleneck verdict unchanged: `latency_bound`
- appeared (medium): **Stall states sum to more than the total they partition** [measurement_above_physical_limit]
- escalated: Register spilling to local memory [register_spilling] low -> medium
- 8 finding(s) unchanged on both sides

**Stall deltas unreliable: stall accounting failed closure in report B: disjoint stall states sum to 104.9% of the total they partition, so the replay passes disagree.**

### Stall composition (cycles per issue-slot)

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Barrier (cycles/issue-slot) | 8.39 | 6.35 | -2.04 | -24.3% | improved |
| Long Scoreboard (cycles/issue-slot) | 9.95 | 10.30 | 0.347 | +3.5% | REGRESSED |
| Sleeping (cycles/issue-slot) | 0.383 | 0.145 | -0.238 | -62.1% | improved |
| MIO Throttle (cycles/issue-slot) | 0.0645 | 0.171 | 0.106 | +164.7% | REGRESSED |
| Branch Resolving (cycles/issue-slot) | 0.437 | 0.368 | -0.0699 | -16.0% | improved |
| Wait (cycles/issue-slot) | 1.65 | 1.71 | 0.0593 | +3.6% | REGRESSED |
| Warpgroup Arrive (cycles/issue-slot) | 0.0178 | 0.0711 | 0.0533 | +299.8% | REGRESSED |

_12 further metric(s) unchanged within noise._

### Speed of light

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| SM compute throughput (% of peak) | 28.80 | 32.11 | 3.32 | +11.5% | changed |
| Compute-memory throughput (% of peak) | 32.37 | 42.97 | 10.60 | +32.7% | changed |
| DRAM throughput (% of peak) | 30.43 | 29.09 | -1.34 | -4.4% | changed |
| L1/TEX throughput (% of peak) | 28.06 | 30.78 | 2.72 | +9.7% | changed |

_1 further metric(s) unchanged within noise._

### Occupancy

All 4 tracked metrics unchanged within noise.

### Memory hierarchy (traffic-weighted)

| metric | A | B | delta | rel | traffic A -> B | status |
|---|---|---|---|---|---|---|
| L1/TEX sector hit rate (%) | 50.31 | 22.13 | -28.18 | -56.0% | - -> - | REGRESSED |
| L2 read hit rate (%) | 60.68 | 69.28 | 8.6 | +14.2% | - -> - | improved |
| Local-load L1TEX hit rate (%) | 96.53 | 37.48 | -59.05 | -61.2% | 260,532 -> 237,054 | REGRESSED |
| Local-store L1TEX hit rate (%) | 35.16 | 28.62 | -6.54 | -18.6% | 91,616 -> 90,571 | REGRESSED |
| L2 miss sectors (to DRAM/sysmem) (sectors) | 3,416,794 | 3,173,382 | -243,412 | -7.1% | - | improved |
| Local-load sectors missing L1TEX (sent to L2) (sectors) | 9,028 | 148,200 | 139,172 | +1541.6% | - | REGRESSED |
| Local-store sectors missing L1TEX (sent to L2) (sectors) | 59,404 | 64,648 | 5,244 | +8.8% | - | REGRESSED |
| L2 sector traffic (sectors) | 11,082,969 | 13,101,563 | 2,018,594 | +18.2% | - | changed |

_2 further metric(s) unchanged within noise._

### Instruction mix

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Instructions executed (warp) (inst) | 10,729,016 | 10,466,896 | -262,120 | -2.4% | changed |
| Executed IPC (inst/cycle) | 0.511 | 0.56 | 0.0487 | +9.5% | improved |
| Issue-active fraction (of cycles) | 0.129 | 0.137 | 0.00804 | +6.2% | improved |
| Avg threads active per inst (threads) | 49,200 | 55,510 | 6,310 | +12.8% | changed |
| ALU pipe utilisation (% of peak) | 8.29 | 9.74 | 1.44 | +17.4% | changed |
| Tensor (HMMA) pipe utilisation (% of peak) | 29.99 | 33.67 | 3.68 | +12.3% | changed |

_3 further metric(s) unchanged within noise._

### Spills

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Local-load instructions (inst) | 83,986 | 74,386 | -9,600 | -11.4% | improved |

_1 further metric(s) unchanged within noise._

### Shared memory

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Shared-mem load bank conflicts (conflicts) | 0 | 641 | 641 | from 0 | REGRESSED ((baseline is zero; no ratio)) |
| Shared-mem store bank conflicts (conflicts) | 20,103 | 15,080 | -5,023 | -25.0% | improved |

_1 further metric(s) unchanged within noise._

## Honesty notes

- Deltas in this report share one cause -- the code change between A and B -- but the diff does not establish causality between any two deltas. A stall that fell and a hit rate that rose moved together; whether one produced the other is a question for the source, not for this table.
- `pod_fused_device_kernel`: stall accounting failed closure in report B: disjoint stall states sum to 104.9% of the total they partition, so the replay passes disagree; the stall-composition deltas for this kernel are unreliable.
