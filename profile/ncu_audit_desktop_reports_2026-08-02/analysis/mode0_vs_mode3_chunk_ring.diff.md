# NCU Report Diff (A -> B)

- A (baseline): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode0_register_baseline.ncu-rep`
- B (candidate): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode3_chunk_ring.ncu-rep`
- GPU: H100 SXM5
- matched kernels: 1

## Clock guard

SM clocks agree within 1% on every matched kernel; raw-duration ratios are presented with elapsed-cycle ratios as cross-check.

## `pod_fused_device_kernel` (grid 132, block 384)

### Duration

| quantity | A | B | B/A |
|---|---|---|---|
| raw duration (ns) | 87,520 | 86,432 | 0.988 |
| SM clock (MHz) | 1,891 | 1,908 | 1.009 |
| clock-normalised duration ratio | - | - | 0.996 |
| elapsed GPC cycles (clock-independent) | 156,558 | 155,137 | 0.991 |

SM clocks agree (1891 vs 1908 MHz), so the raw-time ratio stands: B runs at 0.988x of A; the elapsed-cycle ratio 0.991x cross-checks it.

- caveat: Clocks were not confirmed locked on both sides, so small differences may be clock rather than code.

### What changed the verdict

- bottleneck verdict unchanged: `latency_bound`
- no findings appeared or disappeared (9 present on both sides)

### Stall composition (cycles per issue-slot)

All 19 tracked metrics unchanged within noise.

### Speed of light

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Compute-memory throughput (% of peak) | 32.37 | 34.72 | 2.35 | +7.3% | changed |
| L2 throughput (% of peak) | 41.87 | 40.05 | -1.83 | -4.4% | changed |

_3 further metric(s) unchanged within noise._

### Occupancy

All 4 tracked metrics unchanged within noise.

### Memory hierarchy (traffic-weighted)

| metric | A | B | delta | rel | traffic A -> B | status |
|---|---|---|---|---|---|---|
| L2 sector hit rate (%) | 69.61 | 74.64 | 5.02 | +7.2% | 11,082,969 -> 10,842,373 | improved |
| L2 sector traffic (sectors) | 11,082,969 | 10,842,373 | -240,596 | -2.2% | - | changed |

_8 further metric(s) unchanged within noise._

### Instruction mix

All 9 tracked metrics unchanged within noise.

### Spills

All 2 tracked metrics unchanged within noise.

### Shared memory

All 3 tracked metrics unchanged within noise.

## Honesty notes

- Deltas in this report share one cause -- the code change between A and B -- but the diff does not establish causality between any two deltas. A stall that fell and a hit rate that rose moved together; whether one produced the other is a question for the source, not for this table.
