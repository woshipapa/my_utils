"""Clock throttling: was the GPU actually allowed to run at the speed you assumed?

A throttled run makes every derived number wrong in the same direction - lower
achieved FLOPS, lower bandwidth, worse MFU - while looking exactly like a code
regression. This module exists so that "the kernel got slower" can be separated
from "the clock got lower", which have nothing to do with each other.

Bitmask values are VERIFIED against
https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksEventReasons.html and
field IDs against NVIDIA/DCGM ``dcgmlib/dcgm_fields.h`` on main.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "CLOCK_EVENT_REASONS",
    "THROTTLING_MASK",
    "DCGM_FIELDS",
    "ThrottleReading",
    "decode_clock_event_mask",
    "analyze_throttling",
]


# The full mask. Note that two of these are *not* throttling, which is the trap:
# a bare ``mask != 0`` test fires on every idle GPU in the fleet.
CLOCK_EVENT_REASONS: Dict[int, Dict[str, Any]] = {
    0x1: {
        "name": "GpuIdle",
        "throttling": False,
        "meaning": "Nothing was running. Clocks drop because there is no work, not because of a limit.",
    },
    0x2: {
        "name": "ApplicationsClocksSetting",
        "throttling": False,
        "meaning": (
            "Clocks were pinned by an operator via 'nvidia-smi -ac' or '--lock-gpu-clocks'. "
            "This is usually deliberate and is what a benchmark *wants*."
        ),
    },
    0x4: {
        "name": "SwPowerCap",
        "throttling": True,
        "meaning": "The driver lowered clocks to stay under the board power limit.",
        "remedy": "Raise the limit with 'nvidia-smi -pl', or accept it and report the achieved clock.",
    },
    0x8: {
        "name": "HwSlowdown",
        "throttling": True,
        "meaning": (
            "Hardware slowdown engaged - thermal limit, power brake, or a failing supply. "
            "This is a coarse hammer and usually halves the clock."
        ),
        "remedy": "Check cooling and PSU. Inspect HwThermalSlowdown / HwPowerBrakeSlowdown for which.",
    },
    0x10: {
        "name": "SyncBoost",
        "throttling": True,
        "meaning": (
            "The GPU is in a sync-boost group and was held to the group's common clock, so a "
            "slow peer sets this GPU's speed."
        ),
        "remedy": "Expected in some multi-GPU setups; compare against a single-GPU baseline.",
    },
    0x20: {
        "name": "SwThermalSlowdown",
        "throttling": True,
        "meaning": "Driver-side thermal throttling - GPU or memory temperature above its limit.",
        "remedy": "Improve airflow. Re-run once temperatures are at steady state, not from cold.",
    },
    0x40: {
        "name": "HwThermalSlowdown",
        "throttling": True,
        "meaning": "Hardware thermal slowdown - temperature hit the hard limit.",
        "remedy": "Cooling problem. Any benchmark taken here is invalid.",
    },
    0x80: {
        "name": "HwPowerBrakeSlowdown",
        "throttling": True,
        "meaning": "External power-brake signal asserted - the power supply or PDU pulled it.",
        "remedy": "Facility/PSU issue, not a code issue.",
    },
    0x100: {
        "name": "DisplayClockSetting",
        "throttling": True,
        "meaning": "Clocks constrained by a display configuration. Rare on datacentre parts.",
    },
}

# The mask to actually test against. GpuIdle (0x1) and ApplicationsClocksSetting
# (0x2) are excluded deliberately: both appear in the same field and neither is a
# limit being hit, so including them makes every idle or deliberately-pinned GPU
# report as throttled.
THROTTLING_MASK = 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100

# DCGM field IDs. Match on the numeric ID rather than the name: DCGM renamed
# CLOCK_THROTTLE_REASONS to CLOCKS_EVENT_REASONS while keeping the same id 112,
# and the old spelling survives only as a deprecated alias.
DCGM_FIELDS: Dict[str, int] = {
    "SM_CLOCK": 100,
    "MEM_CLOCK": 101,
    "CLOCKS_EVENT_REASONS": 112,       # was CLOCK_THROTTLE_REASONS, same id
    "MEMORY_TEMP_CELSIUS": 140,
    "GPU_TEMP_CELSIUS": 150,
    "BOARD_POWER_WATTS": 155,          # NOT "POWER_USAGE" - that name does not exist
    "GPU_TEMP_SLOWDOWN_CELSIUS": 158,
    "BOARD_POWER_LIMIT_ENFORCED_WATTS": 164,
    "POWER_VIOLATION": 240,            # accumulating duration, not a sample
    "THERMAL_VIOLATION": 241,
}


@dataclass
class ThrottleReading:
    """One decoded clock-event sample."""

    reasons: List[str]
    throttling: bool
    detail: str
    remedies: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasons": list(self.reasons),
            "throttling": self.throttling,
            "detail": self.detail,
            "remedies": list(self.remedies),
        }


def decode_clock_event_mask(mask: Optional[int]) -> ThrottleReading:
    """Decode an NVML clock-event bitmask into named reasons.

    Reports ``throttling`` only for reasons that represent a limit being hit, so
    an idle GPU and an operator-pinned clock do not read as problems.
    """
    if mask is None:
        return ThrottleReading([], False, "No clock-event data was collected.", [])
    try:
        bits = int(mask)
    except (TypeError, ValueError):
        return ThrottleReading([], False, f"Unparseable clock-event mask {mask!r}.", [])

    names: List[str] = []
    remedies: List[str] = []
    details: List[str] = []
    for bit, info in sorted(CLOCK_EVENT_REASONS.items()):
        if not bits & bit:
            continue
        names.append(info["name"])
        details.append(f"{info['name']}: {info['meaning']}")
        remedy = info.get("remedy")
        if remedy and info["throttling"]:
            remedies.append(remedy)

    throttling = bool(bits & THROTTLING_MASK)
    if not names:
        detail = "No clock-event reasons set; the GPU ran unconstrained."
    else:
        detail = " ".join(details)
    return ThrottleReading(names, throttling, detail, remedies)


def analyze_throttling(
    *,
    clock_event_mask: Optional[int] = None,
    sm_clock_mhz: Optional[float] = None,
    boost_clock_mhz: Optional[float] = None,
    power_violation_ns: Optional[float] = None,
    thermal_violation_ns: Optional[float] = None,
    window_ns: Optional[float] = None,
) -> Dict[str, Any]:
    """Decide whether measured performance can be attributed to the code at all.

    Prefers the accumulating violation counters (DCGM 240/241) over the
    instantaneous bitmask where both are available: throttling is bursty, and a
    1 Hz sample of a bitmask misses most of it.
    """
    reading = decode_clock_event_mask(clock_event_mask)
    result: Dict[str, Any] = {
        "clock_events": reading.to_dict(),
        "invalidates": [],
        "notes": [],
    }

    # Violation counters are durations, so they catch throttling that a sampled
    # mask would step over entirely.
    for label, value in (("power", power_violation_ns), ("thermal", thermal_violation_ns)):
        if value is None or window_ns is None or window_ns <= 0:
            continue
        share = float(value) / float(window_ns)
        result[f"{label}_violation_share"] = share
        if share > 0.01:
            result["notes"].append(
                f"{label.capitalize()} violation accounted for {share * 100:.1f}% of the window. "
                "This is measured duration, not a sample, so it is the more reliable signal."
            )
            reading.throttling = True

    if sm_clock_mhz and boost_clock_mhz and boost_clock_mhz > 0:
        ratio = float(sm_clock_mhz) / float(boost_clock_mhz)
        result["clock_ratio"] = ratio
        if ratio < 0.95:
            result["notes"].append(
                f"SM clock ran at {ratio * 100:.0f}% of boost ({sm_clock_mhz:.0f} of "
                f"{boost_clock_mhz:.0f} MHz). Peak-relative numbers computed against the boost "
                "clock are overstated by roughly the same factor."
            )

    result["throttling"] = reading.throttling
    if reading.throttling:
        # A throttled run cannot be compared against an unthrottled baseline, and
        # every percent-of-peak is measured against a peak that was unavailable.
        result["invalidates"] = [
            "mfu", "achieved_tflops", "achieved_bandwidth",
            "pct_of_peak", "regression_comparison",
        ]
        result["notes"].append(
            "The GPU was clock-limited during this run. Percent-of-peak figures are computed "
            "against a peak the hardware was not permitted to reach, and a comparison against "
            "an unthrottled baseline will read as a code regression."
        )
    return result
