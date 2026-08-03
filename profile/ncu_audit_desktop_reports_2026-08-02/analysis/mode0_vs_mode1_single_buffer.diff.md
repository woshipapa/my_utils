# NCU Report Diff (A -> B)

- A (baseline): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode0_register_baseline.ncu-rep`
- B (candidate): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode1_single_buffer.ncu-rep`
- GPU: H100 SXM5
- matched kernels: 1

## Clock guard

**WARNING: SM clocks differ by more than 1% on 1 of 1 matched kernel(s). Raw-time deltas for those kernels are NOT speedups; read the clock-normalised and elapsed-cycle figures instead.**

## `pod_fused_device_kernel` (grid 132, block 384)

### Duration

| quantity | A | B | B/A |
|---|---|---|---|
| raw duration (ns) -- confounded by clock, NOT a speedup | 87,520 | 81,952 | 0.936 |
| SM clock (MHz) | 1,891 | 2,039 | 1.078 |
| clock-normalised duration ratio | - | - | 1.010 |
| elapsed GPC cycles (clock-independent) | 156,558 | 146,452 | 0.935 |

raw durations are NOT comparable as a speedup (SM clocks 1891 vs 2039 MHz); clock-normalised, B runs at 1.010x of A (raw 0.936x contains the clock change); the clock-independent elapsed-cycle ratio is 0.935x. NOTE: the clock-normalised duration ratio (1.010x) and the elapsed-cycle ratio (0.935x) disagree by 7.9%, which means the clock varied between replay passes; trust neither figure to better than that.

- guard: the two measurements ran at different SM clocks (1891 vs 2039 MHz, +7.8%). A duration ratio therefore mixes the change in the code with the change in the clock; compare cycles, or lock the clock and re-measure

- caveat: Clocks were not confirmed locked on both sides, so small differences may be clock rather than code.

### What changed the verdict

- bottleneck verdict unchanged: `latency_bound`
- no findings appeared or disappeared (9 present on both sides)

### Stall composition (cycles per issue-slot)

All 19 tracked metrics unchanged within noise.

### Speed of light

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Compute-memory throughput (% of peak) | 32.37 | 36.75 | 4.37 | +13.5% | changed |
| DRAM throughput (% of peak) | 30.43 | 33.38 | 2.95 | +9.7% | changed |

_3 further metric(s) unchanged within noise._

### Occupancy

All 4 tracked metrics unchanged within noise.

### Memory hierarchy (traffic-weighted)

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| L2 miss sectors (to DRAM/sysmem) (sectors) | 3,416,794 | 3,280,507 | -136,287 | -4.0% | improved |

_9 further metric(s) unchanged within noise._

### Instruction mix

All 9 tracked metrics unchanged within noise.

### Spills

All 2 tracked metrics unchanged within noise.

### Shared memory

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Shared-mem store bank conflicts (conflicts) | 20,103 | 21,221 | 1,118 | +5.6% | REGRESSED |

_2 further metric(s) unchanged within noise._

## Honesty notes

- Deltas in this report share one cause -- the code change between A and B -- but the diff does not establish causality between any two deltas. A stall that fell and a hit rate that rose moved together; whether one produced the other is a question for the source, not for this table.
