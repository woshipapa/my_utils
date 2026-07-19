"""NCCL bus-bandwidth analysis: is the network slow, or is a rank late?

Collective duration on its own says almost nothing. A slow AllReduce is equally
consistent with a degraded link, a badly chosen protocol, a message too small to
amortise latency, or - most often - one rank arriving late and stretching
everyone else's collective. This module separates those.

Two conversions do the work:

* **Algorithmic bandwidth** ``algbw = message_bytes / time`` is what you measure.
  It is *not* comparable to a link speed, and it changes with rank count for
  identical hardware.
* **Bus bandwidth** ``busbw = algbw * factor(n)`` normalises for how many times
  each byte crosses the fabric, and *is* comparable to the link ceiling. The
  factors are the nccl-tests convention.

Comparing algbw against a link speed is the standard mistake: for AllReduce the
factor approaches 2, so it understates efficiency by nearly half and sends
people hunting a bottleneck that is not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "InterconnectCeiling",
    "CollectiveMeasurement",
    "CollectiveAnalysis",
    "analyze_collective",
    "analyze_collectives",
    "detect_straggler",
    "INTERCONNECT_CEILINGS",
    "PROTOCOL_NOTES",
]


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InterconnectCeiling:
    """Practically achievable bus bandwidth for one fabric, in GB/s.

    These are measured nccl-tests figures, not link specs. Note the NVLS row:
    NVLink SHARP reduces inside the switch, so each byte crosses the fabric
    fewer times and the *normalised* bus bandwidth legitimately exceeds the
    per-link spec. An analyzer with a hard-coded ring cost model will flag a
    correct NVLS run as impossible.
    """

    name: str
    ring_gbps: float
    nvls_gbps: float = 0.0
    notes: str = ""


INTERCONNECT_CEILINGS: Dict[str, InterconnectCeiling] = {
    "h100_nvlink4_8gpu": InterconnectCeiling(
        "8x H100 NVLink4", ring_gbps=360.0, nvls_gbps=480.0,
        notes="NVLS busbw exceeds the 450 GB/s link spec by design - it is a normalised figure."),
    "h800_nvlink_8gpu": InterconnectCeiling(
        "8x H800 NVLink (export)", ring_gbps=160.0, nvls_gbps=0.0,
        notes="NVLink cut to 400 GB/s per direction on the export SKU."),
    "a100_nvlink3_8gpu": InterconnectCeiling(
        "8x A100 NVLink3", ring_gbps=235.0, nvls_gbps=0.0,
        notes="No NVLink SHARP on Ampere."),
    "a800_nvlink_8gpu": InterconnectCeiling(
        "8x A800 NVLink (export)", ring_gbps=155.0, nvls_gbps=0.0),
    "ib_ndr_per_rail": InterconnectCeiling(
        "InfiniBand NDR (per rail)", ring_gbps=50.0),
    "ib_hdr_per_rail": InterconnectCeiling(
        "InfiniBand HDR (per rail)", ring_gbps=25.0),
    "pcie5": InterconnectCeiling("PCIe Gen5 x16", ring_gbps=55.0),
}

# Protocol behaviour, from NCCL's own tuning model.
PROTOCOL_NOTES: Dict[str, str] = {
    "ll": "LL: ~2 us latency but roughly half the bandwidth (8-byte flag per 4 bytes of data). "
          "Correct for small messages, wrong for large ones.",
    "ll128": "LL128: ~2 us latency at ~95% of bandwidth (120 useful bytes per 128). "
             "The usual sweet spot on NVLink.",
    "simple": "Simple: full bandwidth at ~6 us latency. Correct for large messages.",
}

ALGORITHM_NOTES: Dict[str, str] = {
    "ring": "Ring: saturates bandwidth at large sizes, latency grows with rank count.",
    "tree": "Tree: latency scales logarithmically, preferred for small messages at scale.",
    "nvls": "NVLS: reduction offloaded into the NVSwitch, so busbw can exceed the link spec.",
    "pat": "PAT: AllGather/ReduceScatter only, one GPU per node, added in NCCL 2.23.4.",
    "collnet": "CollNet: in-network reduction via SHARP on the IB fabric.",
}

# Below this many bytes a collective is latency-bound and bandwidth is the wrong
# yardstick; NCCL's own tuner switches protocols around here.
LATENCY_REGIME_BYTES = 1 << 20          # 1 MiB
BANDWIDTH_REGIME_BYTES = 128 << 20      # 128 MiB
EFFICIENCY_WARN = 0.70                  # below 70% of ceiling is worth a finding


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class CollectiveMeasurement:
    """One observed collective instance."""

    collective: str
    message_bytes: float
    duration_ns: float
    num_ranks: int
    algorithm: str = ""
    protocol: str = ""
    dtype: str = ""
    kernel_name: str = ""


@dataclass
class CollectiveAnalysis:
    """Bandwidth verdict for one collective."""

    collective: str
    num_ranks: int
    message_bytes: float
    duration_ns: float
    algbw_gbps: Optional[float] = None
    busbw_gbps: Optional[float] = None
    busbw_factor: Optional[float] = None
    ceiling_gbps: Optional[float] = None
    efficiency: Optional[float] = None
    regime: str = ""                     # latency | transition | bandwidth
    findings: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collective": self.collective,
            "num_ranks": self.num_ranks,
            "message_bytes": self.message_bytes,
            "duration_ns": self.duration_ns,
            "algbw_gbps": self.algbw_gbps,
            "busbw_gbps": self.busbw_gbps,
            "busbw_factor": self.busbw_factor,
            "ceiling_gbps": self.ceiling_gbps,
            "efficiency": self.efficiency,
            "regime": self.regime,
            "findings": self.findings,
            "notes": self.notes,
        }


def _busbw_factor(collective: str, num_ranks: int) -> Optional[float]:
    """nccl-tests bus-bandwidth correction factor."""
    if not num_ranks or num_ranks < 2:
        return None
    key = collective.lower().replace("_", "")
    n = float(num_ranks)
    if key == "allreduce":
        return 2.0 * (n - 1.0) / n
    if key in ("allgather", "reducescatter", "alltoall"):
        return (n - 1.0) / n
    if key in ("broadcast", "reduce", "sendrecv", "scatter", "gather"):
        return 1.0
    return None


def analyze_collective(
    measurement: CollectiveMeasurement,
    *,
    ceiling: Optional[InterconnectCeiling] = None,
) -> CollectiveAnalysis:
    """Convert one collective into bus bandwidth and grade it."""
    result = CollectiveAnalysis(
        collective=measurement.collective,
        num_ranks=measurement.num_ranks,
        message_bytes=measurement.message_bytes,
        duration_ns=measurement.duration_ns,
    )

    if measurement.duration_ns > 0 and measurement.message_bytes > 0:
        result.algbw_gbps = measurement.message_bytes / (measurement.duration_ns * 1e-9) / 1e9

    factor = _busbw_factor(measurement.collective, measurement.num_ranks)
    result.busbw_factor = factor
    if result.algbw_gbps is not None and factor is not None:
        result.busbw_gbps = result.algbw_gbps * factor

    size = measurement.message_bytes
    result.regime = (
        "latency" if size < LATENCY_REGIME_BYTES
        else "bandwidth" if size >= BANDWIDTH_REGIME_BYTES
        else "transition"
    )

    if measurement.algorithm and measurement.algorithm.lower() in ALGORITHM_NOTES:
        result.notes.append(ALGORITHM_NOTES[measurement.algorithm.lower()])
    if measurement.protocol and measurement.protocol.lower() in PROTOCOL_NOTES:
        result.notes.append(PROTOCOL_NOTES[measurement.protocol.lower()])

    if ceiling is not None and result.busbw_gbps is not None:
        uses_nvls = measurement.algorithm.lower() == "nvls"
        limit = ceiling.nvls_gbps if (uses_nvls and ceiling.nvls_gbps) else ceiling.ring_gbps
        result.ceiling_gbps = limit
        if limit > 0:
            result.efficiency = result.busbw_gbps / limit
        if ceiling.notes:
            result.notes.append(ceiling.notes)

    # -- findings ------------------------------------------------------
    if result.regime == "bandwidth" and result.efficiency is not None and result.efficiency < EFFICIENCY_WARN:
        result.findings.append({
            "category": "low_bus_bandwidth",
            "severity": "high" if result.efficiency < 0.5 else "medium",
            "title": f"{measurement.collective} reaches {result.efficiency * 100:.0f}% of the fabric ceiling",
            "summary": (
                f"{result.busbw_gbps:.0f} GB/s bus bandwidth against a practical ceiling of "
                f"{result.ceiling_gbps:.0f} GB/s on a {size / (1 << 20):.0f} MiB message. At this "
                "size the collective should be bandwidth bound, so the gap is real."
            ),
            "actions": [
                "Check the topology: an NVLink group crossing a host bridge collapses to PCIe.",
                "Confirm the protocol and algorithm NCCL chose (NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=TUNING).",
                "Rule out a straggler first - a late rank stretches every rank's collective.",
            ],
        })

    if result.regime == "latency" and measurement.protocol.lower() == "simple":
        result.findings.append({
            "category": "protocol_mismatch_small_message",
            "severity": "low",
            "title": "Simple protocol on a latency-regime message",
            "summary": (
                f"A {size / 1024:.0f} KiB message ran the Simple protocol (~6 us latency). "
                "LL or LL128 would trade bandwidth you cannot use for latency you can."
            ),
            "actions": ["Usually NCCL's tuner is right; only override with NCCL_PROTO if measured."],
        })

    if result.regime == "bandwidth" and measurement.protocol.lower() == "ll":
        result.findings.append({
            "category": "protocol_mismatch_large_message",
            "severity": "medium",
            "title": "LL protocol on a bandwidth-regime message",
            "summary": (
                f"A {size / (1 << 20):.0f} MiB message ran the LL protocol, which spends roughly "
                "half the wire on flags. LL128 or Simple would move the same data faster."
            ),
            "actions": ["Check why the tuner chose LL; NCCL_PROTO=LL128 to test the hypothesis."],
        })

    return result


def analyze_collectives(
    measurements: Sequence[CollectiveMeasurement],
    *,
    ceiling: Optional[InterconnectCeiling] = None,
) -> Dict[str, Any]:
    """Analyse many collectives and summarise the communication picture."""
    analyses = [analyze_collective(m, ceiling=ceiling) for m in measurements]

    by_collective: Dict[str, List[CollectiveAnalysis]] = {}
    for analysis in analyses:
        by_collective.setdefault(analysis.collective, []).append(analysis)

    summary: Dict[str, Any] = {}
    for name, group in by_collective.items():
        busbws = [a.busbw_gbps for a in group if a.busbw_gbps is not None]
        effs = [a.efficiency for a in group if a.efficiency is not None]
        summary[name] = {
            "count": len(group),
            "total_bytes": sum(a.message_bytes for a in group),
            "total_ns": sum(a.duration_ns for a in group),
            "peak_busbw_gbps": max(busbws) if busbws else None,
            "median_busbw_gbps": sorted(busbws)[len(busbws) // 2] if busbws else None,
            "worst_efficiency": min(effs) if effs else None,
        }

    findings = [f for a in analyses for f in a.findings]
    return {
        "ceiling": ceiling.name if ceiling else "",
        "collectives_analyzed": len(analyses),
        "per_collective": summary,
        "findings": findings,
        "details": [a.to_dict() for a in analyses],
    }


# ---------------------------------------------------------------------------
# Straggler vs bandwidth
# ---------------------------------------------------------------------------

def detect_straggler(
    per_rank_arrival_ns: Mapping[int, float],
    *,
    collective_duration_ns: Optional[float] = None,
    skew_tolerance: float = 0.10,
) -> Dict[str, Any]:
    """Decide whether a slow collective is the network or a late rank.

    Every rank leaves a synchronising collective at roughly the same instant, so
    the spread in *arrival* times is the straggler signal. If that spread is a
    large fraction of the collective's duration, the fabric is not the problem -
    the collective was waiting.

    ``per_rank_arrival_ns`` maps rank to the time it entered the collective, in a
    common time base. **Cross-rank timestamps are only meaningful when the hosts
    share a synchronised clock**; on NTP the offset can exceed the effect being
    measured, so treat a marginal result as no result.
    """
    if len(per_rank_arrival_ns) < 2:
        return {"conclusive": False, "reason": "need at least two ranks"}

    arrivals = dict(per_rank_arrival_ns)
    earliest = min(arrivals.values())
    latest = max(arrivals.values())
    spread = latest - earliest
    late_rank = max(arrivals, key=arrivals.get)

    delays = {rank: t - earliest for rank, t in arrivals.items()}
    ordered = sorted(delays.values())
    median_delay = ordered[len(ordered) // 2]

    result: Dict[str, Any] = {
        "conclusive": True,
        "arrival_spread_ns": spread,
        "latest_rank": late_rank,
        "latest_rank_delay_ns": delays[late_rank],
        "median_delay_ns": median_delay,
        "per_rank_delay_ns": delays,
    }

    if collective_duration_ns and collective_duration_ns > 0:
        share = spread / collective_duration_ns
        result["spread_share_of_duration"] = share
        if share > skew_tolerance:
            result["verdict"] = "straggler"
            result["summary"] = (
                f"Rank {late_rank} entered the collective {delays[late_rank] / 1e6:.1f} ms after "
                f"the first rank, which is {share * 100:.0f}% of the collective's duration. The "
                "collective was waiting, not transferring - do not read this as a slow network."
            )
        else:
            result["verdict"] = "bandwidth_or_topology"
            result["summary"] = (
                f"Ranks arrived within {spread / 1e6:.2f} ms of each other "
                f"({share * 100:.0f}% of the collective duration), so arrival skew does not "
                "explain the time. Compare bus bandwidth against the fabric ceiling."
            )
    else:
        result["verdict"] = "unknown"
        result["summary"] = "Collective duration not supplied; cannot weigh the skew."

    result["caveat"] = (
        "Cross-rank timestamps carry host clock offset. Under NTP that offset (100 us - 10 ms) "
        "can exceed the skew being measured; only trust this on a PTP-synchronised cluster."
    )
    return result
