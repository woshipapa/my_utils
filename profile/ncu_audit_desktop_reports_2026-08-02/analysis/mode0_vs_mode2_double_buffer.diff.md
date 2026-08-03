# NCU Report Diff (A -> B)

- A (baseline): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode0_register_baseline.ncu-rep`
- B (candidate): `/Users/papa/Desktop/kernel_traces/ncu_kernels/mode2_double_buffer.ncu-rep`
- GPU: H100 SXM5
- matched kernels: 1

## Clock guard

**WARNING: SM clocks differ by more than 1% on 1 of 1 matched kernel(s). Raw-time deltas for those kernels are NOT speedups; read the clock-normalised and elapsed-cycle figures instead.**

## `pod_fused_device_kernel` (grid 132, block 384)

### Duration

| quantity | A | B | B/A |
|---|---|---|---|
| raw duration (ns) -- confounded by clock, NOT a speedup | 87,520 | 82,080 | 0.938 |
| SM clock (MHz) | 1,891 | 2,019 | 1.067 |
| clock-normalised duration ratio | - | - | 1.001 |
| elapsed GPC cycles (clock-independent) | 156,558 | 146,907 | 0.938 |

raw durations are NOT comparable as a speedup (SM clocks 1891 vs 2019 MHz); clock-normalised, B runs at 1.001x of A (raw 0.938x contains the clock change); the clock-independent elapsed-cycle ratio is 0.938x. NOTE: the clock-normalised duration ratio (1.001x) and the elapsed-cycle ratio (0.938x) disagree by 6.7%, which means the clock varied between replay passes; trust neither figure to better than that.

- guard: the two measurements ran at different SM clocks (1891 vs 2019 MHz, +6.7%). A duration ratio therefore mixes the change in the code with the change in the clock; compare cycles, or lock the clock and re-measure

- caveat: Clocks were not confirmed locked on both sides, so small differences may be clock rather than code.

### What changed the verdict

- bottleneck verdict unchanged: `latency_bound`
- no findings appeared or disappeared (9 present on both sides)

### Stall composition (cycles per issue-slot)

All 19 tracked metrics unchanged within noise.

### Speed of light

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Compute-memory throughput (% of peak) | 32.37 | 35.88 | 3.51 | +10.8% | changed |
| DRAM throughput (% of peak) | 30.43 | 33.30 | 2.87 | +9.4% | changed |
| L2 throughput (% of peak) | 41.87 | 44.42 | 2.55 | +6.1% | changed |

_2 further metric(s) unchanged within noise._

### Occupancy

All 4 tracked metrics unchanged within noise.

### Memory hierarchy (traffic-weighted)

| metric | A | B | delta | rel | traffic A -> B | status |
|---|---|---|---|---|---|---|
| L2 sector hit rate (%) | 69.61 | 73.53 | 3.92 | +5.6% | 11,082,969 -> 10,537,760 | improved |
| L2 read hit rate (%) | 60.68 | 63.50 | 2.82 | +4.7% | - -> - | improved |
| Local-store L1TEX hit rate (%) | 35.16 | 34.34 | -0.819 | -2.3% | 91,616 -> 93,836 | REGRESSED |
| Local-load sectors missing L1TEX (sent to L2) (sectors) | 9,028 | 8,416 | -612 | -6.8% | - | improved |
| Local-store sectors missing L1TEX (sent to L2) (sectors) | 59,404 | 61,612 | 2,208 | +3.7% | - | REGRESSED |
| L2 sector traffic (sectors) | 11,082,969 | 10,537,760 | -545,209 | -4.9% | - | changed |

_4 further metric(s) unchanged within noise._

### Instruction mix

All 9 tracked metrics unchanged within noise.

### Spills

All 2 tracked metrics unchanged within noise.

### Shared memory

| metric | A | B | delta | rel | status |
|---|---|---|---|---|---|
| Shared-mem store bank conflicts (conflicts) | 20,103 | 21,268 | 1,165 | +5.8% | REGRESSED |

_2 further metric(s) unchanged within noise._

## Honesty notes

- Deltas in this report share one cause -- the code change between A and B -- but the diff does not establish causality between any two deltas. A stall that fell and a hit rate that rose moved together; whether one produced the other is a question for the source, not for this table.
