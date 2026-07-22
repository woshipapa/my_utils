"""nsys_auto_analysis.py

Comprehensive automatic analysis pipeline for Nsight Systems SQLite profiles.

Analysis dimensions:
  1. Hardware spec detection  – GPU model → peak HBM / PCIe / NVLink bandwidth
  2. Memory IO efficiency     – Actual vs theoretical peak PCIe / HBM utilisation
  3. Stream parallelism       – Concurrent stream count, copy-compute overlap ratio
  4. Communication efficiency – NCCL bus bandwidth, pipeline bubble fraction
  5. GPU metrics distribution – sm_active / tensor_active / dram_throughput percentiles
  6. Kernel composition       – Workload breakdown (matmul / attention / comm / …)
  7. CPU-GPU pipeline         – Kernel dispatch latency, sync overhead
  8. Performance scoring      – 0-100 score per dimension + overall weighted score
  9. Prioritised recommendations with references

References
----------
[1] NVIDIA Nsight Systems User Guide
    https://docs.nvidia.com/nsight-systems/
[2] NVIDIA Deep Learning Performance Guide
    https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/
[3] NCCL User Guide
    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
[4] CUDA Best Practices Guide
    https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
[5] Roofline: An Insightful Visual Performance Model (Williams et al., CACM 2009)
[6] GTC 2023 S51080 – Profiling Deep Learning at Scale with Nsight Systems
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Hardware specification database
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GpuSpecs:
    """Known theoretical peak bandwidths / capacities for a GPU model."""

    name: str
    hbm_peak_tbps: float  # HBM read bandwidth  (TB/s)
    pcie_peak_gbps: float  # PCIe unidirectional bandwidth (GB/s)
    nvlink_peak_gbps: float  # NVLink per-direction bandwidth (GB/s); 0 if absent
    sm_count: int  # Streaming Multiprocessors
    fp16_tflops: float  # Peak FP16 TFLOPS (Tensor Core)


# List of (substring, GpuSpecs) ordered from most-specific to least-specific.
# Matching is case-insensitive substring of the GPU name reported by nsys.
_GPU_DB: List[Tuple[str, GpuSpecs]] = [
    # NVIDIA H-series (Hopper)
    ("h100 nvl", GpuSpecs("H100 NVL", 3.93, 128.0, 450.0, 132, 989.0)),
    ("h100 sxm5", GpuSpecs("H100 SXM5", 3.35, 128.0, 450.0, 132, 989.0)),
    ("h100 sxm", GpuSpecs("H100 SXM", 3.35, 128.0, 450.0, 132, 989.0)),
    ("h100", GpuSpecs("H100", 3.35, 128.0, 450.0, 132, 989.0)),
    # NVIDIA A-series (Ampere)
    ("a100 80gb sxm", GpuSpecs("A100 80GB SXM", 2.0, 64.0, 300.0, 108, 312.0)),
    ("a100 80gb", GpuSpecs("A100 80GB", 2.0, 64.0, 300.0, 108, 312.0)),
    ("a100 40gb", GpuSpecs("A100 40GB", 1.555, 64.0, 300.0, 108, 312.0)),
    ("a100", GpuSpecs("A100", 2.0, 64.0, 300.0, 108, 312.0)),
    ("a30", GpuSpecs("A30", 0.933, 64.0, 200.0, 56, 165.0)),
    ("a10g", GpuSpecs("A10G", 0.6, 64.0, 0.0, 80, 31.2)),
    ("a10", GpuSpecs("A10", 0.6, 64.0, 0.0, 72, 31.2)),
    ("a6000", GpuSpecs("RTX A6000", 0.768, 64.0, 0.0, 84, 38.7)),
    ("a5000", GpuSpecs("RTX A5000", 0.576, 64.0, 0.0, 64, 27.8)),
    ("a4000", GpuSpecs("RTX A4000", 0.448, 32.0, 0.0, 48, 19.2)),
    # NVIDIA V-series (Volta)
    ("v100 sxm2", GpuSpecs("V100 SXM2", 0.9, 32.0, 150.0, 80, 112.0)),
    ("v100", GpuSpecs("V100", 0.9, 32.0, 150.0, 80, 112.0)),
    # Ada Lovelace (L-series / RTX 40xx)
    ("l40s", GpuSpecs("L40S", 0.864, 64.0, 0.0, 142, 91.6)),
    ("l40", GpuSpecs("L40", 0.864, 64.0, 0.0, 142, 90.5)),
    ("l4", GpuSpecs("L4", 0.300, 64.0, 0.0, 58, 30.3)),
    ("rtx 4090", GpuSpecs("RTX 4090", 1.008, 64.0, 0.0, 128, 165.0)),
    ("rtx 4080", GpuSpecs("RTX 4080", 0.717, 64.0, 0.0, 76, 97.5)),
    ("rtx 4070 ti", GpuSpecs("RTX 4070 Ti", 0.504, 64.0, 0.0, 60, 40.0)),
    ("rtx 4070", GpuSpecs("RTX 4070", 0.432, 32.0, 0.0, 46, 29.0)),
    # Turing (RTX 20xx / T-series)
    ("t4", GpuSpecs("T4", 0.320, 32.0, 0.0, 40, 65.0)),
    ("rtx 3090 ti", GpuSpecs("RTX 3090 Ti", 1.008, 32.0, 0.0, 84, 40.0)),
    ("rtx 3090", GpuSpecs("RTX 3090", 0.936, 32.0, 0.0, 82, 35.6)),
    ("rtx 3080 ti", GpuSpecs("RTX 3080 Ti", 0.912, 32.0, 0.0, 80, 34.1)),
    ("rtx 3080", GpuSpecs("RTX 3080", 0.760, 32.0, 0.0, 68, 29.8)),
    # P-series (Pascal) – no Tensor Cores
    ("p100", GpuSpecs("P100", 0.732, 16.0, 80.0, 56, 0.0)),
    ("p40", GpuSpecs("P40", 0.346, 16.0, 0.0, 30, 0.0)),
    ("p4", GpuSpecs("P4", 0.192, 8.0, 0.0, 20, 0.0)),
]


def lookup_gpu_specs(gpu_name: str) -> Optional[GpuSpecs]:
    """Return GPU hardware specs for the best-matching GPU model name, or None.

    Delegates to :mod:`my_utils.profiling.hardware.gpu_specs`, which is the
    authoritative table, and adapts the result to this module's ``GpuSpecs``
    shape so callers do not change. The local ``_GPU_DB`` below is retained only
    as a fallback for the (unlikely) case where the shared table cannot be
    imported.

    The shared table matters here for two reasons the local one got wrong:
    it knows the export SKUs this project actually runs on (H800/A800, whose
    NVLink bandwidth differs from H100/A100 even though the FLOPs match), and
    all of its tensor peaks are **dense** - the local table mixed dense and
    with-sparsity figures, which silently halved utilisation for the affected
    parts.
    """
    name_lower = str(gpu_name or "").lower()
    if not name_lower:
        return None

    try:
        from ..hardware.gpu_specs import lookup_gpu_spec as _lookup_shared
    except Exception:  # pragma: no cover - only if the package layout changes
        _lookup_shared = None

    if _lookup_shared is not None:
        shared = _lookup_shared(gpu_name)
        if shared is not None:
            # bf16 is the training dtype of interest; fall back to fp16 for
            # pre-Ampere parts that have no bf16 path.
            peak = shared.peak_tflops("bf16") or shared.peak_tflops("fp16") or 0.0
            return GpuSpecs(
                name=shared.name,
                hbm_peak_tbps=shared.hbm_bandwidth_gbps / 1000.0,
                pcie_peak_gbps=shared.pcie_gbps,
                nvlink_peak_gbps=shared.nvlink_gbps,
                sm_count=shared.sm_count,
                fp16_tflops=peak,
            )

    for key, specs in _GPU_DB:
        if key in name_lower:
            return specs
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Kernel categorisation by name patterns
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of (category, [regex patterns]) – first match wins.
_KERNEL_CATEGORIES: List[Tuple[str, List[str]]] = [
    (
        "communication",
        [
            r"nccl",
            r"all_?reduce",
            r"all_?gather",
            r"reduce_?scatter",
            r"send_?recv",
            r"broadcast",
            r"ncclKernel",
        ],
    ),
    (
        "matmul",
        [
            r"gemm",
            r"cublas",
            r"cutlass",
            r"matmul",
            r"sgemm",
            r"dgemm",
            r"hgemm",
            r"wgrad",
            r"dgrad",
            r"fprop",
            r"mm_",
        ],
    ),
    (
        "attention",
        [
            r"attention",
            r"flash_?attn",
            r"fmha",
            r"scaled_dot",
            r"multihead",
            r"xformers",
        ],
    ),
    ("softmax", [r"softmax", r"log_?softmax"]),
    (
        "normalization",
        [
            r"layer_?norm",
            r"rmsnorm",
            r"batch_?norm",
            r"instance_?norm",
            r"group_?norm",
            r"rms_norm",
        ],
    ),
    (
        "activation",
        [r"gelu", r"silu", r"relu", r"sigmoid", r"tanh", r"swiglu", r"activation"],
    ),
    (
        "elementwise",
        [r"elementwise", r"pointwise", r"fused_bias", r"add_", r"mul_", r"eltwise"],
    ),
    ("embedding", [r"embedding", r"lookup", r"vocab", r"token"]),
    (
        "optimizer",
        [
            r"adam",
            r"lamb",
            r"sgd",
            r"momentum",
            r"clip_grad",
            r"optim",
            r"weight_update",
        ],
    ),
    (
        "memory_ops",
        [r"copy_", r"fill", r"memset", r"scatter", r"gather", r"index_select"],
    ),
    (
        "reduction",
        [
            r"\breduce\b",
            r"\bsum\b",
            r"\bmean\b",
            r"\bvar\b",
            r"\bstd\b",
            r"\bmax\b",
            r"\bmin\b",
        ],
    ),
    ("dropout", [r"dropout"]),
    ("loss", [r"cross_entropy", r"nll_loss", r"loss_kernel"]),
]


def categorize_kernel(name: str) -> str:
    """Return the workload category for a kernel name."""
    name_lower = str(name or "").lower()
    for category, patterns in _KERNEL_CATEGORIES:
        if any(re.search(p, name_lower) for p in patterns):
            return category
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Metric name pattern matching
# ─────────────────────────────────────────────────────────────────────────────


def _match_metric(name: str, *patterns: str) -> bool:
    n = name.lower()
    return any(re.search(p, n) for p in patterns)


def _classify_metric(name: str) -> Optional[str]:
    """Map a GPU metric name to a short key (sm_active / tensor_active / dram_throughput / None)."""
    if _match_metric(name, r"sm.*active", r"smsp.*active", r"sm__active"):
        return "sm_active"
    if _match_metric(
        name, r"tensor.*active", r"pipe.*tensor", r"hmma.*active", r"imma.*active"
    ):
        return "tensor_active"
    if _match_metric(
        name,
        r"dram.*throughput",
        r"dram__throughput",
        r"mem_bw",
        r"dram_bw",
        r"hbm.*throughput",
    ):
        return "dram_throughput"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except Exception:
        return default


def _pct(part: float, total: float) -> float:
    return round(100.0 * part / total, 1) if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Memory IO analysis
# ─────────────────────────────────────────────────────────────────────────────

# copyKind values from CUPTI:
_COPY_KIND_LABELS = {1: "HtoD", 2: "DtoH", 3: "DtoD", 8: "HtoD_Pinned", 10: "Peer2Peer"}

# DtoD is internal GPU memory; H↔D goes over PCIe (or NVLink for P2P).
_PCIE_KINDS = {"HtoD", "DtoH", "HtoD_Pinned"}
_NVLINK_KINDS = {"Peer2Peer"}
_HBM_KINDS = {"DtoD"}


def analyze_memory_io(
    bandwidth_rows: List[Dict],  # from memcpy_bandwidth_analysis
    transfer_rows: List[Dict],  # from memcpy_transfers_detail
    specs: Optional[GpuSpecs],
    total_span_ms: float,
) -> Dict:
    """
    Analyse memory transfer efficiency.

    Metrics:
    - Per-direction: volume (GB), duration (ms), bandwidth (GB/s)
    - PCIe utilisation %  = (HtoD + DtoH BW) / PCIe_peak
    - HBM utilisation %   = DtoD BW / HBM_peak  (if GPU metrics unavailable)
    - Small transfer fraction (< 1 MB) — high fixed launch overhead
    - Transfer time as % of total profiling span
    """
    # Aggregate per direction from bandwidth_rows (copy_kind is an int or label)
    by_dir: Dict[str, Dict] = {}
    for row in bandwidth_rows:
        ck = row.get("copy_kind")
        label = (
            _COPY_KIND_LABELS.get(int(ck), f"Kind_{ck}")
            if ck is not None
            else str(row.get("direction", ""))
        )
        by_dir[label] = {
            "count": int(row.get("count") or 0),
            "total_gb": _f(row.get("total_gb")),
            "total_ms": _f(row.get("total_ms")),
            "avg_gbps": _f(row.get("avg_gbps")),
        }

    h2d = by_dir.get("HtoD", {})
    d2h = by_dir.get("DtoH", {})
    d2d = by_dir.get("DtoD", {})
    pin = by_dir.get("HtoD_Pinned", {})
    p2p = by_dir.get("Peer2Peer", {})

    pcie_gb = h2d.get("total_gb", 0) + d2h.get("total_gb", 0) + pin.get("total_gb", 0)
    pcie_ms = h2d.get("total_ms", 0) + d2h.get("total_ms", 0) + pin.get("total_ms", 0)
    d2d_gb = d2d.get("total_gb", 0)
    d2d_ms = d2d.get("total_ms", 0)
    p2p_gb = p2p.get("total_gb", 0)
    p2p_ms = p2p.get("total_ms", 0)
    total_memcpy_ms = sum(d.get("total_ms", 0) for d in by_dir.values())

    avg_pcie_gbps = (pcie_gb / pcie_ms * 1000.0) if pcie_ms > 0 else 0.0
    avg_d2d_gbps = (d2d_gb / d2d_ms * 1000.0) if d2d_ms > 0 else 0.0
    avg_p2p_gbps = (p2p_gb / p2p_ms * 1000.0) if p2p_ms > 0 else 0.0

    pcie_util: Optional[float] = None
    hbm_util_from_d2d: Optional[float] = None
    p2p_util: Optional[float] = None
    if specs:
        if specs.pcie_peak_gbps > 0 and avg_pcie_gbps > 0:
            pcie_util = round(avg_pcie_gbps / specs.pcie_peak_gbps * 100.0, 1)
        if specs.hbm_peak_tbps > 0 and avg_d2d_gbps > 0:
            hbm_util_from_d2d = round(
                avg_d2d_gbps / (specs.hbm_peak_tbps * 1000.0) * 100.0, 1
            )
        if specs.nvlink_peak_gbps > 0 and avg_p2p_gbps > 0:
            p2p_util = round(avg_p2p_gbps / specs.nvlink_peak_gbps * 100.0, 1)

    # Per-transfer size analysis
    small_count = sum(1 for r in transfer_rows if _f(r.get("bytes")) < 1e6)
    total_count = len(transfer_rows)
    small_pct = _pct(small_count, total_count)

    issues: List[str] = []
    if pcie_util is not None and pcie_util < 30.0 and pcie_ms > 10.0:
        issues.append(
            f"PCIe underutilised: {pcie_util:.1f}% of peak "
            f"({avg_pcie_gbps:.1f} / {specs.pcie_peak_gbps:.0f} GB/s). "
            "Many small or serialised transfers waste bandwidth. "
            "Ref: CUDA Best Practices §Pinned Memory."
        )
    if small_pct > 40.0:
        issues.append(
            f"{small_pct:.0f}% of transfers are <1 MB ({small_count} ops). "
            "High fixed-overhead per transfer kills effective PCIe throughput. "
            "Batch small host↔device copies or use CUDA managed memory."
        )
    if total_memcpy_ms > 0 and total_span_ms > 0:
        memcpy_span_pct = _pct(total_memcpy_ms, total_span_ms)
        if memcpy_span_pct > 20.0:
            issues.append(
                f"Memory transfers consume {memcpy_span_pct:.1f}% of the profiling window. "
                "Overlap H↔D copies with compute using cudaMemcpyAsync + separate stream. "
                "Ref: NVIDIA DL Perf Guide §Data Ingest."
            )

    return {
        "by_direction": by_dir,
        "pcie_total_gb": round(pcie_gb, 3),
        "pcie_total_ms": round(pcie_ms, 3),
        "pcie_avg_gbps": round(avg_pcie_gbps, 2),
        "pcie_utilisation_pct": pcie_util,
        "d2d_total_gb": round(d2d_gb, 3),
        "d2d_total_ms": round(d2d_ms, 3),
        "d2d_avg_gbps": round(avg_d2d_gbps, 2),
        "hbm_utilisation_from_d2d_pct": hbm_util_from_d2d,
        "p2p_total_gb": round(p2p_gb, 3),
        "p2p_avg_gbps": round(avg_p2p_gbps, 2),
        "p2p_utilisation_pct": p2p_util,
        "total_memcpy_ms": round(total_memcpy_ms, 3),
        "memcpy_span_pct": _pct(total_memcpy_ms, total_span_ms),
        "small_transfers_count": small_count,
        "small_transfers_total": total_count,
        "small_transfers_pct": round(small_pct, 1),
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stream parallelism analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_stream_parallelism(
    per_stream_rows: List[Dict],
    total_span_ms: float,
) -> Dict:
    """
    Derive stream concurrency from per-stream busy/span data.

    Parallelism ratio  = Σ(stream_busy_ms) / max(stream_span_ms)
      > 1.0 means streams genuinely overlap (parallelism exists).
      = 1.0 means entirely sequential.

    Copy-compute overlap is detected by checking if a dedicated copy
    stream (low kernel count, distinct from the primary compute stream)
    runs in parallel with high-utilisation compute streams.

    Reference: NVIDIA CUDA Programming Guide §Streams and Events.
    """
    rows = [r for r in per_stream_rows if _f(r.get("stream_span_ms")) > 0]
    if not rows:
        return {
            "stream_count": 0,
            "parallelism_ratio": 0.0,
            "copy_stream_detected": False,
            "copy_compute_overlap_score": 0.0,
            "issues": ["No stream utilisation data available."],
        }

    stream_count = len(rows)
    sum_busy = sum(_f(r.get("kernel_busy_ms")) for r in rows)
    max_span = max(_f(r.get("stream_span_ms")) for r in rows)
    parallelism_ratio = round(sum_busy / max(max_span, 0.001), 3)

    # Detect copy streams: low kernel count AND span ≈ total span (long-lived)
    # vs compute streams: high kernel count.
    sorted_by_count = sorted(
        rows, key=lambda r: _f(r.get("kernel_count")), reverse=True
    )
    copy_streams = [
        r
        for r in rows
        if _f(r.get("kernel_count")) < 20
        and _f(r.get("stream_span_ms")) > max_span * 0.5
    ]
    copy_stream_detected = len(copy_streams) > 0

    # Overlap score between copy and compute streams
    # If copy_stream exists and is busier, it's overlapping with compute.
    copy_busy_ms = sum(_f(r.get("kernel_busy_ms")) for r in copy_streams)
    compute_busy_ms = sum_busy - copy_busy_ms
    copy_compute_overlap_score = (
        _pct(copy_busy_ms, max_span) if copy_stream_detected else 0.0
    )

    # Per-stream summary
    per_stream = [
        {
            "stream_id": r.get("stream_id"),
            "kernel_count": int(r.get("kernel_count") or 0),
            "kernel_busy_ms": _f(r.get("kernel_busy_ms")),
            "stream_span_ms": _f(r.get("stream_span_ms")),
            "utilisation_pct": _f(r.get("utilization_pct")),
            "role": "copy" if r in copy_streams else "compute",
        }
        for r in rows
    ]

    issues: List[str] = []
    if stream_count == 1:
        issues.append(
            "Only 1 CUDA stream detected. No stream parallelism possible. "
            "Consider launching H↔D copies on a dedicated stream to overlap "
            "data movement with compute. "
            "Ref: NVIDIA DL Perf Guide §Overlapping Data Transfers."
        )
    elif parallelism_ratio < 1.05 and stream_count > 1:
        issues.append(
            f"{stream_count} streams present but parallelism ratio is only "
            f"{parallelism_ratio:.2f}x (nearly sequential execution). "
            "Streams may be blocking on cudaStreamSynchronize or sharing "
            "the same HW queue. Use different stream priorities or check "
            "for implicit synchronisation barriers."
        )
    if not copy_stream_detected and stream_count > 1:
        issues.append(
            "No dedicated copy stream found. Async host↔device transfers "
            "overlapped with compute could save significant wall time."
        )

    return {
        "stream_count": stream_count,
        "parallelism_ratio": parallelism_ratio,
        "copy_stream_detected": copy_stream_detected,
        "copy_compute_overlap_ms": round(copy_busy_ms, 3),
        "copy_compute_overlap_score": round(copy_compute_overlap_score, 1),
        "per_stream": per_stream,
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Communication (NCCL) efficiency
# ─────────────────────────────────────────────────────────────────────────────


def analyze_communication(
    nccl_rows: List[Dict],
    overlap: Dict,
    total_span_ms: float,
) -> Dict:
    """
    Analyse NCCL collective communication efficiency.

    Key metrics:
    - NCCL time as % of total span (communication intensity)
    - Compute / communication time ratio  (target: >5:1 for hidden comm)
    - Pipeline bubble %  ≈ max(0, comm_only_ms) / span_ms
    - Top collective operations by cumulative time

    Reference: NCCL User Guide; NVIDIA Reducing Communication Overhead blog.
    """
    nccl_total_ms = sum(_f(r.get("total_ms")) for r in nccl_rows)
    nccl_span_pct = _pct(nccl_total_ms, total_span_ms)

    comm_total_ms = _f(overlap.get("comm_total_ms"))
    compute_total_ms = _f(overlap.get("compute_total_ms"))
    overlap_ms = _f(overlap.get("overlap_ms"))
    comm_only_ms = _f(overlap.get("comm_only_ms"))
    overlap_pct_comm = _f(overlap.get("overlap_pct_of_comm"))
    overlap_pct_compute = _f(overlap.get("overlap_pct_of_compute"))

    compute_comm_ratio = round(compute_total_ms / max(comm_total_ms, 0.001), 2)
    pipeline_bubble_pct = _pct(comm_only_ms, total_span_ms)

    # Top collectives
    top_ops = sorted(nccl_rows, key=lambda r: _f(r.get("total_ms")), reverse=True)[:10]

    issues: List[str] = []
    if nccl_total_ms > 0:
        if overlap_pct_comm < 30.0 and comm_total_ms > 5.0:
            issues.append(
                f"Only {overlap_pct_comm:.0f}% of NCCL communication overlaps with compute "
                f"(comm={comm_total_ms:.1f} ms, overlap={overlap_ms:.1f} ms). "
                "Enable async_op=True in PyTorch all-reduce, or use ZeRO-3 with "
                "overlap_comm=True. Ref: DeepSpeed ZeRO; Megatron-LM pipeline schedule."
            )
        if pipeline_bubble_pct > 10.0:
            issues.append(
                f"Communication-only stall contributes {pipeline_bubble_pct:.1f}% of the "
                "profiling window as pipeline bubble. Consider gradient compression, "
                "FP16/BF16 all-reduce, or increasing micro-batch size to amortise comm. "
                "Ref: NVIDIA DL Perf Guide §Communication."
            )
        if compute_comm_ratio < 3.0:
            issues.append(
                f"Compute/comm ratio is {compute_comm_ratio:.1f}x (target ≥5x). "
                "Model is communication-heavy relative to compute. "
                "Options: larger batch size, model parallelism, gradient checkpoint. "
                "Ref: GTC 2023 S51080."
            )

    return {
        "nccl_total_ms": round(nccl_total_ms, 3),
        "nccl_span_pct": round(nccl_span_pct, 1),
        "compute_total_ms": round(compute_total_ms, 3),
        "comm_total_ms": round(comm_total_ms, 3),
        "overlap_ms": round(overlap_ms, 3),
        "overlap_pct_of_comm": round(overlap_pct_comm, 1),
        "overlap_pct_of_compute": round(overlap_pct_compute, 1),
        "compute_comm_ratio": compute_comm_ratio,
        "pipeline_bubble_pct": round(pipeline_bubble_pct, 1),
        "top_collectives": [
            {
                "kernel_name": r.get("kernel_name"),
                "total_ms": _f(r.get("total_ms")),
                "count": int(r.get("invocations") or r.get("count") or 0),
                "avg_ms": _f(r.get("avg_ms")),
            }
            for r in top_ops
        ],
        "issues": issues,
        "bandwidth_efficiency": _collective_bandwidth_coverage(nccl_rows),
    }


def _collective_bandwidth_coverage(nccl_rows: List[Dict]) -> Dict:
    """State plainly that bus bandwidth is not measurable from this data.

    The only honest answer here is a negative one. Grading a collective needs
    algorithmic bandwidth (message bytes / duration) converted to bus bandwidth
    by a factor that depends on the collective and the rank count -- for
    all-reduce, ``2(n-1)/n``. The nccl_breakdown table carries kernel names and
    durations and nothing else: no message size, no rank count.

    Reporting "communication looks fine" from timing alone would be inventing a
    verdict. A collective moving a tenth of the bytes it should, at a tenth of
    achievable bandwidth, takes exactly as long as a healthy one -- duration
    cannot distinguish them. So this names what is missing and how to get it.

    :func:`~my_utils.profiling.analyzers.nccl_bandwidth.analyze_collective` does
    the grading as soon as sizes are available from any of the sources below.
    """
    collectives: Dict[str, float] = {}
    for row in nccl_rows or ():
        name = str(row.get("kernel_name") or "")
        low = name.lower()
        for kind in (
            "allreduce",
            "allgather",
            "reducescatter",
            "broadcast",
            "reduce",
            "alltoall",
            "sendrecv",
        ):
            if kind in low.replace("_", ""):
                collectives[kind] = collectives.get(kind, 0.0) + _f(row.get("total_ms"))
                break

    return {
        "measurable": False,
        "algbw_gbps": None,
        "busbw_gbps": None,
        "efficiency": None,
        "collective_time_ms_by_kind": {
            k: round(v, 3) for k, v in sorted(collectives.items())
        },
        "reason": (
            "Bus bandwidth needs message bytes and rank count; the NCCL kernel table "
            "records neither. Kernel duration alone cannot separate a collective that "
            "is slow from one that is simply large, so no efficiency verdict is given "
            "here rather than a guessed one."
        ),
        "how_to_measure": (
            "Any one of: (1) NCCL flight recorder "
            "(TORCH_NCCL_TRACE_BUFFER_SIZE=<n>), which records size and entry time per "
            "collective; (2) NVTX ranges carrying the message size around each "
            "collective; (3) a matching nccl-tests run for the same shapes and rank "
            "count. Feed any of these to analyzers.nccl_bandwidth.analyze_collective, "
            "which converts algbw to busbw with the per-collective factor and grades "
            "it against the interconnect ceiling."
        ),
        "caveat": (
            "Collective kinds above are read from kernel symbols. NCCL's launch stubs "
            "collapse many device functions onto few kernel symbols, so the symbol is "
            "reliable for the collective kind but NOT for the algorithm (ring vs tree) "
            "or protocol (LL/LL128/Simple)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. GPU metrics distribution
# ─────────────────────────────────────────────────────────────────────────────


def analyze_gpu_metrics(
    aggregate_rows: List[Dict],
    percentile_rows: List[Dict],
    specs: Optional[GpuSpecs],
) -> Dict:
    """
    Analyse GPU hardware sampling metrics (requires --gpu-metrics-devices).

    Classifies each metric row into sm_active / tensor_active / dram_throughput.
    Reports average and p95 to reveal sustained vs bursty utilisation.

    High p95, low avg → bursty compute; pipeline stalls between bursts.
    Low p95, low avg  → sustained underutilisation; check occupancy / sync overhead.

    Reference: Nsight Systems Metrics Guide; Roofline model.
    """
    if not aggregate_rows and not percentile_rows:
        return {"available": False, "issues": []}

    # Build per-key dicts from percentile rows (richer) or aggregate fallback
    keyed_pct: Dict[str, Dict] = {}
    for row in percentile_rows:
        key = _classify_metric(str(row.get("metric_name") or ""))
        if key and key not in keyed_pct:
            keyed_pct[key] = {
                "metric_name": row.get("metric_name"),
                "sample_count": int(row.get("sample_count") or 0),
                "avg": _f(row.get("avg_val")),
                "p50": _f(row.get("p50")),
                "p75": _f(row.get("p75")),
                "p95": _f(row.get("p95")),
                "max": _f(row.get("max_val")),
                "stddev": _f(row.get("stddev_val")),
            }

    keyed_agg: Dict[str, Dict] = {}
    for row in aggregate_rows:
        key = _classify_metric(str(row.get("metric_name") or ""))
        if key and key not in keyed_pct and key not in keyed_agg:
            keyed_agg[key] = {
                "metric_name": row.get("metric_name"),
                "sample_count": int(row.get("sample_count") or 0),
                "avg": _f(row.get("avg_value")),
                "p50": None,
                "p75": None,
                "p95": None,
                "max": _f(row.get("max_value")),
                "stddev": None,
            }

    merged: Dict[str, Dict] = {**keyed_agg, **keyed_pct}

    sm = merged.get("sm_active", {})
    tc = merged.get("tensor_active", {})
    dr = merged.get("dram_throughput", {})

    sm_avg = _f(sm.get("avg"))
    tc_avg = _f(tc.get("avg"))
    dr_avg = _f(dr.get("avg"))
    sm_p95 = sm.get("p95")
    tc_p95 = tc.get("p95")
    dr_p95 = dr.get("p95")

    # HBM utilisation via dram_throughput (metric is already a % in most nsys exports)
    hbm_util: Optional[float] = None
    if dr_avg > 0:
        # If metric is expressed as % of peak: use directly; else compute from specs
        if dr_avg <= 100.0:
            hbm_util = round(dr_avg, 1)
        elif specs and specs.hbm_peak_tbps > 0:
            hbm_util = round(dr_avg / (specs.hbm_peak_tbps * 1e6) * 100.0, 1)

    # Bottleneck classification (consistent with classify_gpu_bottleneck())
    if not sm and not tc and not dr:
        bottleneck = "no_metric_data"
    elif tc_avg >= 60.0:
        bottleneck = "tensor_bound"
    elif sm_avg >= 70.0 and dr_avg < 60.0:
        bottleneck = "compute_bound"
    elif dr_avg >= 60.0:
        bottleneck = "memory_bound"
    elif sm_avg < 30.0 and tc_avg < 30.0 and dr_avg < 30.0:
        bottleneck = "latency_bound"
    else:
        bottleneck = "mixed"

    issues: List[str] = []
    if sm_avg > 0 and sm_avg < 50.0:
        issues.append(
            f"SM active cycles avg is only {sm_avg:.1f}% — substantial GPU idle time. "
            "Root causes: insufficient parallelism (low occupancy), excessive sync, "
            "or CPU-side bottleneck. "
            "Ref: Nsight Systems §SM Activity; CUDA Occupancy Calculator."
        )
    if sm_p95 is not None and sm_avg > 0 and sm_p95 / max(sm_avg, 0.1) > 2.5:
        issues.append(
            f"SM utilisation is highly variable (avg={sm_avg:.1f}%, p95={sm_p95:.1f}%). "
            "Long idle periods between compute bursts suggest serialised CPU→GPU pipeline. "
            "Use CUDA Graphs or persistent kernels to eliminate repeated launch overhead."
        )
    if tc_avg > 0 and tc_avg < 30.0 and sm_avg > 60.0:
        issues.append(
            f"Tensor Cores are underutilised ({tc_avg:.1f}% active) despite high SM activity. "
            "Ensure operands are in FP16/BF16/INT8 and shapes are multiples of 16. "
            "Ref: NVIDIA Tensor Core Programming §Data Layout."
        )
    if hbm_util is not None and hbm_util > 85.0:
        issues.append(
            f"HBM bandwidth utilisation is {hbm_util:.1f}% — approaching saturation. "
            "Fuse memory-bound operations, increase tile sizes, or use Flash-Attention "
            "to reduce global memory traffic. "
            "Ref: Roofline Model; FlashAttention paper (Dao et al., 2022)."
        )

    return {
        "available": True,
        "bottleneck_type": bottleneck,
        "sm_active": sm or {},
        "tensor_active": tc or {},
        "dram_throughput": dr or {},
        "hbm_utilisation_pct": hbm_util,
        "all_metrics": merged,
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Kernel composition analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_kernel_composition(
    aggregate_rows: List[Dict],
    total_span_ms: float,
) -> Dict:
    """
    Categorise GPU kernels by workload type and compute time share.

    Useful to understand what fraction of GPU time is spent on
    matmul vs attention vs communication vs elementwise etc.
    """
    categories: Dict[str, Dict] = {}
    for row in aggregate_rows:
        name = str(row.get("kernel_name") or "")
        cat = categorize_kernel(name)
        ms = _f(row.get("total_ms"))
        count = int(row.get("invocations") or 0)
        if cat not in categories:
            categories[cat] = {"ms": 0.0, "count": 0, "kernels": []}
        categories[cat]["ms"] += ms
        categories[cat]["count"] += count
        categories[cat]["kernels"].append(name)

    total_ms = sum(d["ms"] for d in categories.values())
    result: Dict[str, Dict] = {}
    for cat, d in sorted(categories.items(), key=lambda x: x[1]["ms"], reverse=True):
        result[cat] = {
            "total_ms": round(d["ms"], 3),
            "pct_of_gpu_time": _pct(d["ms"], total_ms),
            "kernel_count": d["count"],
            "top_kernels": d["kernels"][:3],
        }

    top_cat = (
        max(categories, key=lambda c: categories[c]["ms"]) if categories else "other"
    )
    issues: List[str] = []
    comm_cat = categories.get("communication", {})
    if comm_cat.get("ms", 0) > 0:
        comm_pct = _pct(comm_cat["ms"], total_ms)
        if comm_pct > 30.0:
            issues.append(
                f"Communication kernels (NCCL) use {comm_pct:.1f}% of GPU compute time. "
                "Consider gradient compression, fewer collective operations, or "
                "pipelining communication behind compute."
            )

    return {
        "categories": result,
        "total_kernel_ms": round(total_ms, 3),
        "top_category": top_cat,
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. CPU-GPU pipeline analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_cpu_pipeline(
    launch_gap_rows: List[Dict],
    sync_rows: List[Dict],
    total_span_ms: float,
) -> Dict:
    """
    Analyse CPU-side overhead in the GPU dispatch pipeline.

    Kernel launch gap: time from CPU API return to GPU kernel start.
    Sync overhead:     time spent in cudaStreamSynchronize / cudaDeviceSynchronize.

    Reference: CUDA Best Practices Guide §Asynchronous Transfers and Streams.
    """
    total_gap_ms = sum(_f(r.get("total_gap_ms")) for r in launch_gap_rows)
    worst_gap = max(launch_gap_rows, key=lambda r: _f(r.get("avg_gap_us")), default={})

    total_sync_ms = sum(_f(r.get("total_ms")) for r in sync_rows)
    sync_pct = _pct(total_sync_ms, total_span_ms)
    gap_pct = _pct(total_gap_ms, total_span_ms)

    issues: List[str] = []
    if gap_pct > 5.0:
        issues.append(
            f"CPU→GPU launch gap totals {total_gap_ms:.1f} ms ({gap_pct:.1f}% of window). "
            f"Worst kernel: '{worst_gap.get('kernel_name')}' "
            f"avg gap = {_f(worst_gap.get('avg_gap_us')):.1f} µs. "
            "Use CUDA Graphs to amortise repeated launch overhead or pre-populate "
            "kernel argument buffers. "
            "Ref: CUDA Graphs Programming Guide."
        )
    if sync_pct > 10.0:
        issues.append(
            f"Synchronisation overhead is {total_sync_ms:.1f} ms ({sync_pct:.1f}% of window). "
            "Excessive cudaStreamSynchronize / cudaDeviceSynchronize stalls the CPU "
            "dispatch thread. Replace with CUDA events or stream-ordered memory allocator. "
            "Ref: CUDA Best Practices §Asynchronous Transfers."
        )

    return {
        "total_launch_gap_ms": round(total_gap_ms, 3),
        "launch_gap_pct": round(gap_pct, 1),
        "total_sync_ms": round(total_sync_ms, 3),
        "sync_pct": round(sync_pct, 1),
        "worst_launch_gap": {
            "kernel_name": worst_gap.get("kernel_name"),
            "avg_gap_us": _f(worst_gap.get("avg_gap_us")),
            "total_gap_ms": _f(worst_gap.get("total_gap_ms")),
        }
        if worst_gap
        else {},
        "top_sync_types": [
            {
                "sync_type": r.get("sync_type"),
                "count": int(r.get("count") or 0),
                "total_ms": _f(r.get("total_ms")),
                "avg_ms": _f(r.get("avg_ms")),
            }
            for r in sorted(
                sync_rows, key=lambda r: _f(r.get("total_ms")), reverse=True
            )[:5]
        ],
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Communication health / roofline / cross-layer consistency
# ─────────────────────────────────────────────────────────────────────────────


def analyze_communication_health(
    communication: Dict,
    nccl_rows: List[Dict],
) -> Dict:
    overlap_pct_comm = _f(communication.get("overlap_pct_of_comm"))
    bubble_pct = _f(communication.get("pipeline_bubble_pct"))
    compute_comm_ratio = _f(communication.get("compute_comm_ratio"))

    per_op_ms = [_f(r.get("total_ms")) for r in nccl_rows if _f(r.get("total_ms")) > 0]
    mean_op_ms = sum(per_op_ms) / len(per_op_ms) if per_op_ms else 0.0
    max_op_ms = max(per_op_ms) if per_op_ms else 0.0
    op_skew = (max_op_ms / max(mean_op_ms, 1e-6)) if per_op_ms else 0.0

    score = 100.0
    score -= max(0.0, 30.0 - overlap_pct_comm) * 1.1
    score -= bubble_pct * 1.4
    if compute_comm_ratio < 5.0:
        score -= (5.0 - compute_comm_ratio) * 7.0
    if op_skew > 2.0:
        score -= min(20.0, (op_skew - 2.0) * 8.0)
    score = max(0.0, min(100.0, score))

    issues: List[str] = []
    if overlap_pct_comm < 30.0 and _f(communication.get("comm_total_ms")) > 5.0:
        issues.append(
            f"Communication overlap is low ({overlap_pct_comm:.1f}% of comm hidden by compute)."
        )
    if bubble_pct > 10.0:
        issues.append(
            f"Communication-only bubble is high ({bubble_pct:.1f}% of total span)."
        )
    if compute_comm_ratio < 3.0:
        issues.append(
            f"Compute/comm ratio is low ({compute_comm_ratio:.2f}x, recommended >=5x)."
        )
    if op_skew > 2.0:
        issues.append(f"NCCL op-time skew is high (max/mean={op_skew:.2f}).")

    status = "healthy"
    if score < 50:
        status = "critical"
    elif score < 70:
        status = "degraded"
    elif score < 85:
        status = "watch"

    return {
        "health_score": round(score, 1),
        "status": status,
        "overlap_pct_of_comm": round(overlap_pct_comm, 1),
        "pipeline_bubble_pct": round(bubble_pct, 1),
        "compute_comm_ratio": round(compute_comm_ratio, 2),
        "op_skew_ratio_max_over_mean": round(op_skew, 2) if op_skew else 0.0,
        "issues": issues,
    }


def analyze_roofline_gap(
    gpu_metrics: Dict,
    specs: Optional[GpuSpecs],
) -> Dict:
    sm_avg = _f((gpu_metrics.get("sm_active") or {}).get("avg"))
    tc_avg = _f((gpu_metrics.get("tensor_active") or {}).get("avg"))
    dram_avg = _f((gpu_metrics.get("dram_throughput") or {}).get("avg"))

    # sm_active is the fraction of time an SM had at least one resident warp. It
    # says nothing about how many FLOPs were issued: a purely memory-bound kernel
    # can sit at 90% sm_active while doing almost no math. tensor_active is the
    # only one of these that tracks actual math throughput, so it - not
    # max(sm, tensor) - is what a compute estimate may be built on.
    compute_avg = tc_avg if tc_avg > 0 else 0.0
    residency_avg = max(sm_avg, tc_avg)
    memory_avg = dram_avg
    gap_pct = max(0.0, memory_avg - residency_avg)

    # Proxy arithmetic intensity:
    # High DRAM util with low compute util -> low arithmetic intensity / memory bound.
    if memory_avg <= 1e-6:
        ai_proxy = 0.0
    else:
        ai_proxy = compute_avg / memory_avg

    bottleneck = "balanced"
    if memory_avg > compute_avg + 10.0:
        bottleneck = "memory_bound"
    elif compute_avg > memory_avg + 10.0:
        bottleneck = "compute_bound"

    issues: List[str] = []
    if gap_pct > 20.0:
        issues.append(
            f"Roofline gap is large ({gap_pct:.1f} points): memory utilisation exceeds compute utilisation."
        )
    if bottleneck == "memory_bound" and memory_avg > 70.0:
        issues.append(
            f"Workload appears memory-bound (dram={memory_avg:.1f}%, compute={compute_avg:.1f}%)."
        )

    achieved_compute_tflops = None
    memory_bw_gbps = None
    if specs:
        # Only derive a FLOP rate from tensor-pipe activity. Deriving it from
        # sm_active would report ~890 TFLOP/s on an H100 for a kernel that
        # issued no math at all, which is the dimensional error this guards.
        achieved_compute_tflops = (
            round(specs.fp16_tflops * compute_avg / 100.0, 2)
            if compute_avg > 0
            else None
        )
        memory_bw_gbps = round(specs.hbm_peak_tbps * 1000.0 * memory_avg / 100.0, 2)

    return {
        "compute_util_pct": round(compute_avg, 1),
        "memory_util_pct": round(memory_avg, 1),
        "roofline_gap_pct": round(gap_pct, 1),
        "arithmetic_intensity_proxy": round(ai_proxy, 3),
        "bottleneck_hint": bottleneck,
        "achieved_compute_tflops_est": achieved_compute_tflops,
        "achieved_compute_basis": (
            "tensor_pipe_active"
            if compute_avg > 0
            else "unavailable: no tensor-pipe activity was sampled"
        ),
        "sm_residency_pct": round(residency_avg, 1),
        "achieved_memory_gbps_est": memory_bw_gbps,
        "issues": issues,
    }


def analyze_cross_layer_consistency(
    summary: Dict,
    stream_parallelism: Dict,
    communication: Dict,
    cpu_pipeline: Dict,
    kernel_composition: Dict,
) -> Dict:
    timing = (summary or {}).get("timing") or {}
    span_ms = _f(timing.get("span_ms"))
    busy_ms = _f(timing.get("busy_ms"))

    compute_ms = 0.0
    categories = (kernel_composition or {}).get("categories") or {}
    for name, item in categories.items():
        if str(name).lower() == "communication":
            continue
        compute_ms += _f(item.get("total_ms"))
    comm_ms = _f((communication or {}).get("comm_total_ms")) or _f(
        (communication or {}).get("nccl_total_ms")
    )
    cpu_sync_ms = _f((cpu_pipeline or {}).get("total_sync_ms"))
    launch_gap_ms = _f((cpu_pipeline or {}).get("total_launch_gap_ms"))
    stream_parallelism_ratio = _f((stream_parallelism or {}).get("parallelism_ratio"))

    modeled_ms = compute_ms + comm_ms + cpu_sync_ms
    coverage_ratio = modeled_ms / span_ms if span_ms > 0 else 0.0
    busy_alignment_ratio = (
        busy_ms / max(compute_ms + comm_ms, 1e-6) if (compute_ms + comm_ms) > 0 else 0.0
    )

    issues: List[str] = []
    if span_ms > 0 and coverage_ratio < 0.45:
        issues.append(
            f"Cross-layer modeled time covers only {coverage_ratio * 100.0:.1f}% of wall span; missing layer metrics likely."
        )
    if busy_alignment_ratio > 1.5 or busy_alignment_ratio < 0.5:
        issues.append(
            f"GPU busy/model alignment is inconsistent (ratio={busy_alignment_ratio:.2f}); verify time-window and layer attribution."
        )
    if (
        stream_parallelism_ratio > 1.05
        and launch_gap_ms > 0
        and (launch_gap_ms / max(span_ms, 1e-6)) > 0.05
    ):
        issues.append(
            "Kernel launch gap is still high despite stream parallelism; CPU dispatch remains a bottleneck."
        )

    consistency_score = 100.0
    consistency_score -= max(0.0, 45.0 - coverage_ratio * 100.0) * 0.8
    consistency_score -= abs(1.0 - busy_alignment_ratio) * 25.0
    consistency_score -= (launch_gap_ms / max(span_ms, 1e-6)) * 120.0
    consistency_score = max(0.0, min(100.0, consistency_score))

    return {
        "consistency_score": round(consistency_score, 1),
        "coverage_ratio": round(coverage_ratio, 3),
        "busy_alignment_ratio": round(busy_alignment_ratio, 3),
        "span_ms": round(span_ms, 3),
        "modeled_ms": round(modeled_ms, 3),
        "component_breakdown_ms": {
            "compute_ms": round(compute_ms, 3),
            "comm_ms": round(comm_ms, 3),
            "cpu_sync_ms": round(cpu_sync_ms, 3),
            "cpu_launch_gap_ms": round(launch_gap_ms, 3),
        },
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Multi-dimensional performance scoring (0-100)
# ─────────────────────────────────────────────────────────────────────────────


def compute_performance_scores(
    util_pct: Optional[float],
    memory_io: Dict,
    stream_parallelism: Dict,
    communication: Dict,
    gpu_metrics: Dict,
    cpu_pipeline: Dict,
) -> Dict:
    """
    Compute a 0-100 score for each performance dimension.

    Scoring rubric:
      90-100  Excellent – near-optimal
      70-89   Good      – minor headroom
      50-69   Fair      – significant improvement possible
      20-49   Poor      – major bottleneck
      0-19    Critical  – severely underperforming

    Dimension weights for overall score:
      GPU Utilisation  25%  – overall compute pipeline health
      Memory Efficiency 20% – data movement quality
      Stream Parallelism 20%– pipeline concurrency
      Comm Efficiency  20%  – multi-GPU scaling
      CPU Pipeline     15%  – dispatch and sync overhead
    """

    def _clamp(v: float) -> float:
        return max(0.0, min(100.0, v))

    # 1. GPU Utilisation
    if util_pct is not None:
        gpu_util_score = _clamp(util_pct * 1.1)  # 91% util → ~100 score
    elif gpu_metrics.get("available"):
        sm = _f(gpu_metrics.get("sm_active", {}).get("avg"))
        gpu_util_score = _clamp(sm * 1.1)
    else:
        gpu_util_score = 50.0  # unknown

    # 2. Memory Efficiency  (PCIe + D2D utilisation)
    pcie_u = gpu_metrics.get("hbm_utilisation_pct") or memory_io.get(
        "hbm_utilisation_from_d2d_pct"
    )
    if pcie_u is not None:
        # HBM near-saturation is BAD for latency-bound workloads but expected
        # for memory-bound ones. Score peaks at 70-80% utilisation.
        mem_score = _clamp(100.0 - abs(pcie_u - 75.0) * 1.3)
    elif memory_io.get("pcie_utilisation_pct") is not None:
        pu = _f(memory_io.get("pcie_utilisation_pct"))
        mem_score = _clamp(pu * 1.2)  # higher PCIe util = better data-fed GPU
    else:
        mem_score = 50.0

    # Penalise many small transfers
    small_pct = _f(memory_io.get("small_transfers_pct"))
    mem_score = _clamp(mem_score - small_pct * 0.5)

    # 3. Stream Parallelism
    ratio = _f(stream_parallelism.get("parallelism_ratio"))
    sc = _f(stream_parallelism.get("stream_count"))
    if sc == 0:
        sp_score = 0.0
    elif sc == 1:
        # A single stream is not evidence of a problem. CUDA-graph capture,
        # torch.compile and ncu's own replay all serialise onto one stream, and
        # plenty of healthy training loops use one. Scoring it 15/100 made it the
        # "primary bottleneck" for those runs by construction, so treat it as
        # not-assessable instead of bad.
        sp_score = None
    else:
        sp_score = _clamp(min(ratio - 1.0, 2.0) / 2.0 * 80.0 + 20.0)
    if sp_score is not None and stream_parallelism.get("copy_stream_detected"):
        sp_score = _clamp(sp_score + 10.0)

    # 4. Communication Efficiency
    if communication.get("nccl_total_ms", 0) == 0:
        comm_score = 100.0  # no NCCL → perfect comm score
    else:
        ov_pct = _f(communication.get("overlap_pct_of_comm"))
        bubble = _f(communication.get("pipeline_bubble_pct"))
        comm_score = _clamp(ov_pct * 0.8 + max(0.0, 20.0 - bubble))

    # 5. CPU Pipeline
    gap_pct = _f(cpu_pipeline.get("launch_gap_pct"))
    sync_pct = _f(cpu_pipeline.get("sync_pct"))
    cpu_score = _clamp(100.0 - gap_pct * 4.0 - sync_pct * 3.0)

    weights = [
        ("gpu_utilisation", gpu_util_score, 0.25),
        ("memory_efficiency", mem_score, 0.20),
        ("stream_parallelism", sp_score, 0.20),
        ("communication", comm_score, 0.20),
        ("cpu_pipeline", cpu_score, 0.15),
    ]

    # A dimension scored None was not measurable on this trace. Including it
    # would let an absent measurement win the "lowest score" ranking and be
    # reported as the primary bottleneck, which is how an unmeasured axis
    # becomes a confident finding. Drop it from both the average and the
    # ranking, and renormalise the weights over what remains.
    measured = [(k, v, w) for k, v, w in weights if v is not None]
    unmeasured = [k for k, v, _ in weights if v is None]

    total_weight = sum(w for _, _, w in measured)
    overall = (
        round(sum(v * w for _, v, w in measured) / total_weight, 1)
        if total_weight
        else 0.0
    )

    scores = {k: round(v, 1) for k, v, _ in measured}
    for key in unmeasured:
        scores[key] = None
    scores["overall"] = overall
    scores["unmeasured_dimensions"] = unmeasured

    # Rank only what was actually measured.
    ranked = sorted(measured, key=lambda x: x[1])
    scores["primary_bottleneck"] = ranked[0][0] if ranked else "unknown"
    scores["secondary_bottleneck"] = ranked[1][0] if len(ranked) > 1 else "none"

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 8. Recommendation generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_recommendations(
    scores: Dict,
    memory_io: Dict,
    stream_parallelism: Dict,
    communication: Dict,
    communication_health: Dict,
    gpu_metrics: Dict,
    roofline_gap: Dict,
    kernel_composition: Dict,
    cpu_pipeline: Dict,
    cross_layer_consistency: Dict,
    util_pct: Optional[float],
    short_kernels: List[Dict],
    kernel_jitter: List[Dict],
) -> List[Dict]:
    """
    Generate a priority-ordered list of actionable recommendations.

    Priority: critical (score <30) > high (30-50) > medium (50-70) > low (>70).
    """
    recs: List[Dict] = []

    def _add(
        score: float, category: str, title: str, detail: str, refs: List[str]
    ) -> None:
        if score < 30:
            priority = "critical"
        elif score < 50:
            priority = "high"
        elif score < 70:
            priority = "medium"
        else:
            priority = "low"
        recs.append(
            {
                "priority": priority,
                "score": round(score, 1),
                "category": category,
                "title": title,
                "detail": detail,
                "references": refs,
            }
        )

    # --- GPU Utilisation ---
    gu = _f(scores.get("gpu_utilisation"))
    if gu < 70:
        _add(
            gu,
            "gpu_utilisation",
            "Increase GPU utilisation",
            f"GPU utilisation is {gu:.0f}%. The primary driver is likely one of: "
            "(a) excessive CPU↔GPU synchronisation (check sync_breakdown), "
            "(b) CPU-bound data loading or preprocessing stalling the GPU, "
            "(c) small batch size limiting parallelism, or "
            "(d) many short <10 µs kernels causing launch overhead. "
            "Prioritise resolving the CUDA stream synchronisation pattern and "
            "increase batch size if memory allows.",
            [
                "https://docs.nvidia.com/deeplearning/performance/",
                "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/",
            ],
        )

    # --- Stream Parallelism ---
    sp = _f(scores.get("stream_parallelism"))
    if sp < 50:
        detail = (
            "No async copy-compute pipeline detected. "
            if not stream_parallelism.get("copy_stream_detected")
            else ""
        )
        _add(
            sp,
            "stream_parallelism",
            "Enable copy-compute pipeline with async streams",
            detail
            + "Launch H↔D memory transfers on a dedicated non-default stream using "
            "cudaMemcpyAsync. This overlaps data movement with GPU compute and can "
            "eliminate idle time between mini-batches. In PyTorch: use pin_memory=True "
            "and num_workers>0 in DataLoader, or use torch.cuda.Stream() explicitly.",
            [
                "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-transfers",
                "https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/",
            ],
        )

    # --- Communication Efficiency ---
    ce = _f(scores.get("communication"))
    if ce < 60 and _f(communication.get("nccl_total_ms")) > 0:
        overlap_pct = _f(communication.get("overlap_pct_of_comm"))
        _add(
            ce,
            "communication",
            "Improve compute-communication overlap",
            f"NCCL all-reduce overlaps with compute only {overlap_pct:.0f}% of the time. "
            "Use non-blocking collectives (async_op=True in PyTorch DDP), or switch to "
            "ZeRO-2/ZeRO-3 with overlap_comm=True. Gradient bucketing size matters: "
            "larger buckets reduce NCCL call frequency but increase latency. "
            "For pipeline parallelism: 1F1B schedule hides inter-stage all-reduce.",
            [
                "https://docs.nvidia.com/deeplearning/nccl/user-guide/",
                "https://deepspeed.readthedocs.io/en/latest/zero3.html",
            ],
        )

    # --- Communication Health ---
    ch_score = _f(communication_health.get("health_score"))
    if ch_score < 70:
        _add(
            ch_score,
            "communication_health",
            "Stabilize communication health and tail latency",
            f"Communication health score is {ch_score:.0f}. "
            f"overlap={communication_health.get('overlap_pct_of_comm', 0):.1f}% "
            f"bubble={communication_health.get('pipeline_bubble_pct', 0):.1f}% "
            f"op_skew={communication_health.get('op_skew_ratio_max_over_mean', 0):.2f}. "
            "Inspect rank-level stragglers, NCCL algorithm/protocol choice, and transport-level RAS counters.",
            ["https://docs.nvidia.com/deeplearning/nccl/user-guide/"],
        )

    # --- Memory Efficiency ---
    me = _f(scores.get("memory_efficiency"))
    small_pct = _f(memory_io.get("small_transfers_pct"))
    pcie_util = memory_io.get("pcie_utilisation_pct")
    if me < 60:
        detail_parts = []
        if small_pct > 30:
            detail_parts.append(
                f"{small_pct:.0f}% of memory transfers are <1 MB (high per-transfer overhead). "
                "Batch small transfers into one large cudaMemcpyAsync call."
            )
        if pcie_util is not None and _f(pcie_util) < 30:
            detail_parts.append(
                f"PCIe utilisation is only {pcie_util:.1f}%. Use pinned (page-locked) host "
                "memory to maximise PCIe throughput."
            )
        hbm = gpu_metrics.get("hbm_utilisation_pct") or memory_io.get(
            "hbm_utilisation_from_d2d_pct"
        )
        if hbm and _f(hbm) > 80:
            detail_parts.append(
                f"HBM bandwidth at {hbm:.0f}% — near saturation. Increase operator fusion to "
                "reduce repeated global memory reads. Flash-Attention, Triton fused kernels, "
                "or torch.compile with fusion can help."
            )
        _add(
            me,
            "memory_efficiency",
            "Improve memory transfer and HBM efficiency",
            " ".join(detail_parts)
            or "Review memory access patterns and transfer sizes.",
            [
                "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations",
                "https://arxiv.org/abs/2205.14135",
            ],
        )  # FlashAttention

    # --- CPU Pipeline ---
    cp = _f(scores.get("cpu_pipeline"))
    if cp < 60:
        detail_parts = []
        gap_pct = _f(cpu_pipeline.get("launch_gap_pct"))
        sync_pct = _f(cpu_pipeline.get("sync_pct"))
        if gap_pct > 5:
            detail_parts.append(
                f"Kernel launch gaps total {cpu_pipeline.get('total_launch_gap_ms', 0):.1f} ms "
                f"({gap_pct:.1f}% of window). Use CUDA Graphs (torch.cuda.CUDAGraph) to "
                "replay a recorded kernel sequence with zero re-launch overhead."
            )
        if sync_pct > 8:
            detail_parts.append(
                f"Synchronisation overhead {cpu_pipeline.get('total_sync_ms', 0):.1f} ms "
                f"({sync_pct:.1f}%). Replace blocking syncs with CUDA events; batch "
                "CPU-side checks to avoid serialising the dispatch thread."
            )
        _add(
            cp,
            "cpu_pipeline",
            "Reduce CPU-side dispatch and synchronisation overhead",
            " ".join(detail_parts) or "Profile CPU-side dispatch bottlenecks.",
            [
                "https://developer.nvidia.com/blog/cuda-graphs/",
                "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-transfers",
            ],
        )

    # --- Roofline Gap ---
    rg_gap = _f(roofline_gap.get("roofline_gap_pct"))
    if rg_gap > 15:
        _add(
            max(0.0, 100.0 - rg_gap),
            "roofline",
            "Close roofline gap between memory and compute",
            f"Roofline gap is {rg_gap:.1f} points "
            f"(compute={roofline_gap.get('compute_util_pct', 0):.1f}%, "
            f"memory={roofline_gap.get('memory_util_pct', 0):.1f}%, "
            f"hint={roofline_gap.get('bottleneck_hint', 'balanced')}). "
            "For memory-bound windows, prioritize memory traffic reduction and fusion; "
            "for compute-bound windows, improve occupancy and tensor-core utilization.",
            [
                "https://docs.nvidia.com/nsight-compute/",
                "https://crd.lbl.gov/departments/computer-science/PAR/research/roofline/",
            ],
        )

    # --- Cross-layer consistency ---
    cl_score = _f(cross_layer_consistency.get("consistency_score"))
    if cl_score < 75:
        _add(
            cl_score,
            "cross_layer_consistency",
            "Improve cross-layer observability consistency",
            f"Consistency score is {cl_score:.0f} "
            f"(coverage={_f(cross_layer_consistency.get('coverage_ratio')) * 100.0:.1f}%, "
            f"busy_alignment={cross_layer_consistency.get('busy_alignment_ratio', 0):.2f}). "
            "Align step/rank tags and units across CPU, communication, and GPU events to reduce attribution ambiguity.",
            ["https://perfetto.dev/docs/", "https://opentelemetry.io/docs/"],
        )

    # --- Short kernels / launch overhead ---
    short_pct = 0.0
    for r in short_kernels:
        if str(r.get("duration_bracket", "")).startswith(("a_", "b_")):
            short_pct += _f(r.get("pct_count"))
    if short_pct > 25:
        _add(
            max(100 - short_pct * 2, 0),
            "kernel_efficiency",
            "Fuse short kernels or use CUDA Graphs",
            f"{short_pct:.0f}% of kernel invocations are <10 µs. "
            "Each kernel launch carries ~2-5 µs of CPU overhead + ~5-10 µs GPU scheduling. "
            "Fuse elementwise/activation ops with torch.compile, or record the full "
            "forward-backward pass as a CUDAGraph for overhead-free replay.",
            [
                "https://pytorch.org/docs/stable/torch.compiler.html",
                "https://developer.nvidia.com/blog/cuda-graphs/",
            ],
        )

    # --- Kernel jitter ---
    high_jitter = [r for r in kernel_jitter if _f(r.get("cv_pct")) > 50]
    if high_jitter:
        worst = high_jitter[0]
        _add(
            50.0,
            "kernel_efficiency",
            "Fix high-variance kernels (execution jitter)",
            f"'{worst.get('kernel_name')}' CV={worst.get('cv_pct')}% "
            f"(avg={worst.get('avg_ms')} ms, stddev={worst.get('stddev_ms')} ms). "
            "High CV indicates wavefront imbalance, dynamic branching, or "
            "irregular memory access patterns. Profile with Nsight Compute for "
            "warp stall reasons and shared memory bank conflicts.",
            [
                "https://docs.nvidia.com/nsight-compute/",
                "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#control-flow",
            ],
        )

    # Sort: critical first, then by score ascending (worst first)
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    recs.sort(key=lambda r: (priority_order.get(r["priority"], 4), r["score"]))
    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def analyze_kernel_attribution(aggregate_kernels: List[Dict]) -> Dict:
    """Fuse per-kernel evidence and surface the places a name cannot be trusted.

    This exists because a kernel symbol is the weakest evidence a profiler has,
    and in named cases it is wrong rather than merely vague: NCCL launch stubs
    hard-code the algorithm, CuTe DSL emits ``kernel_kernel_<args>_0`` with no
    shape or dtype, and a ``cutlass3x_*`` symbol may have come from cuBLASLt
    rather than from user code. The useful output here is ``caveats`` - the
    kernels whose names would mislead a reader who took them at face value.
    """
    from ..analyzers.evidence import attribute_kernel

    caveats: List[Dict] = []
    unlabeled = 0
    stub_named = 0

    for row in aggregate_kernels or ():
        if not isinstance(row, Mapping):
            continue
        name = str(row.get("name") or row.get("kernel_name") or "").strip()
        if not name:
            continue
        fused, warnings = attribute_kernel(name)

        informative = fused.get("name_is_informative")
        if informative is not None and informative.value is False:
            unlabeled += 1
        algo = fused.get("nccl_algorithm")
        if algo is not None and algo.confidence == "low":
            stub_named += 1

        if warnings:
            caveats.append(
                {
                    "kernel": name[:120],
                    "total_ms": _f(row.get("total_ms") or row.get("total_time_ms")),
                    "warnings": warnings,
                }
            )

    caveats.sort(key=lambda c: c["total_ms"], reverse=True)
    notes: List[str] = []
    if unlabeled:
        notes.append(
            f"{unlabeled} kernel name(s) carry no shape or dtype information "
            "(CuTe DSL default naming). Call set_name_prefix() at the launch site "
            "to make these traces attributable."
        )
    if stub_named:
        notes.append(
            f"{stub_named} NCCL kernel(s) are launch stubs whose algorithm and protocol "
            "tokens are hard-coded. Read the real algorithm from NCCL_DEBUG=INFO, not "
            "from the kernel name."
        )
    return {"caveats": caveats[:20], "caveat_count": len(caveats), "notes": notes}


def _triage_from_tables(
    summary: Dict,
    aggregate_kernels: List[Dict],
    nccl_breakdown: List[Dict],
    memcpy_bandwidth: List[Dict],
    sync_breakdown: List[Dict],
    cpu_launch_gap: List[Dict],
    kernel_duration_stats: List[Dict],
) -> Optional[Dict]:
    """Lead with a single verdict rather than a wall of per-dimension scores.

    Falls back to ``None`` when the trace lacks the timing needed to attribute a
    cause, which is the honest outcome - a verdict assembled from missing
    evidence reads exactly like one built from real evidence.
    """
    from ..analyzers.triage import triage_step

    timing = (summary or {}).get("timing") or {}
    wall_ns = _f(timing.get("span_ms")) * 1e6
    if wall_ns <= 0:
        return None

    def _total_ns(rows: Iterable[Mapping], *keys: str) -> float:
        out = 0.0
        for row in rows or ():
            if not isinstance(row, Mapping):
                continue
            for key in keys:
                if row.get(key) is not None:
                    out += _f(row[key])
                    break
        return out * 1e6

    # The nsys summary tables give totals, not intervals, so the union-based
    # overlap accounting in triage_step cannot run. Feed it the durations it can
    # use and let it report reduced confidence rather than inventing intervals.
    durations = [
        _f(r.get("avg_ms") or r.get("mean_ms")) * 1e6
        for r in (kernel_duration_stats or ())
        if isinstance(r, Mapping)
    ]
    delays = [
        _f(r.get("gap_us") or r.get("avg_gap_us")) * 1e3
        for r in (cpu_launch_gap or ())
        if isinstance(r, Mapping)
    ]

    verdict = triage_step(
        wall_ns=wall_ns,
        launch_api_ns=_total_ns(cpu_launch_gap, "launch_ms", "total_ms") or None,
        sync_api_ns=_total_ns(sync_breakdown, "total_ms", "duration_ms") or None,
        kernel_durations_ns=[d for d in durations if d > 0],
        launch_delays_ns=[d for d in delays if d > 0],
    )
    return verdict.to_dict() if hasattr(verdict, "to_dict") else None


def _assess_quality(inputs: Optional[Dict], aggregate_kernels: List[Dict]) -> Dict:
    """Run the trace-validity checks, or say clearly that they did not run."""
    if inputs is None:
        return {
            "checked": False,
            "trustworthy": None,
            "blocked_conclusions": [],
            "issues": [],
            "note": (
                "Trace-validity checks did not run: no trace_quality_inputs were "
                "supplied. Nothing below has been gated on warmup, autotuning, CUDA "
                "graph opacity, rank completeness, GPU-metric gaps or profiler "
                "overhead. An unvalidated trace is not a validated one."
            ),
        }

    from ..analyzers.trace_quality import assess_trace_quality  # local: avoid cycle

    payload = dict(inputs)
    payload.setdefault(
        "kernel_names",
        [str(r.get("kernel_name") or "") for r in (aggregate_kernels or ())],
    )
    result = assess_trace_quality(**payload)
    result["checked"] = True
    return result


def _strike_blocked_recommendations(recs: Any, blocked: set) -> Any:
    """Remove recommendations the trace cannot support, and say why.

    Annotating a blocked conclusion is not enough. A reader scanning a
    recommendation list acts on the entries, not on the caveats attached to
    them, so an unsupportable recommendation has to leave the list.
    """
    if not isinstance(recs, list):
        return recs
    kept: List[Any] = []
    struck: List[Any] = []
    for rec in recs:
        text = (rec if isinstance(rec, str) else str(rec.get("text", rec))).lower()
        if any(term.replace("_", " ") in text or term in text for term in blocked):
            struck.append(rec)
        else:
            kept.append(rec)
    if struck:
        kept.append(
            "WITHHELD: " + str(len(struck)) + " recommendation(s) were removed because "
            "the trace cannot support them (" + ", ".join(sorted(blocked)) + "). "
            "They are not absent findings - they are unanswerable questions."
        )
    return kept


def build_comprehensive_analysis(
    gpu_name: str,
    summary: Dict,
    overlap: Dict,
    nccl_breakdown: List[Dict],
    aggregate_kernels: List[Dict],
    per_stream_utilization: List[Dict],
    memcpy_bandwidth: List[Dict],
    sync_breakdown: List[Dict],
    gpu_metrics_aggregate: List[Dict],
    gpu_metrics_percentiles: List[Dict],
    memcpy_transfers_detail: List[Dict],
    cpu_launch_gap: List[Dict],
    short_kernels: List[Dict],
    kernel_duration_stats: List[Dict],
    *,
    trace_quality_inputs: Optional[Dict] = None,
) -> Dict:
    """
    Orchestrate the full automatic analysis and return a rich result dict.

    ``trace_quality_inputs`` is forwarded to
    :func:`~my_utils.profiling.analyzers.trace_quality.assess_trace_quality`
    (iteration counts, report paths, expected world size, profiler overhead).
    Whatever it reports as a blocked conclusion is stripped from the
    recommendations below rather than merely annotated: a trace that cannot
    support a claim should not produce that claim at all. Omitting it means the
    validity checks did not run, which the result records explicitly -- an
    unvalidated trace is not a validated one.
    """
    specs = lookup_gpu_specs(gpu_name)

    timing = (summary or {}).get("timing") or {}
    span_ms = _f(timing.get("span_ms")) or 1.0
    util_pct_raw = timing.get("utilization_pct")
    util_pct = _f(util_pct_raw) if util_pct_raw is not None else None

    mem_io = analyze_memory_io(
        memcpy_bandwidth, memcpy_transfers_detail, specs, span_ms
    )
    stream_para = analyze_stream_parallelism(per_stream_utilization, span_ms)
    comm = analyze_communication(nccl_breakdown, overlap, span_ms)
    gm = analyze_gpu_metrics(gpu_metrics_aggregate, gpu_metrics_percentiles, specs)
    kernel_comp = analyze_kernel_composition(aggregate_kernels, span_ms)
    cpu_pipe = analyze_cpu_pipeline(cpu_launch_gap, sync_breakdown, span_ms)
    comm_health = analyze_communication_health(comm, nccl_breakdown)
    roofline_gap = analyze_roofline_gap(gm, specs)
    cross_layer_consistency = analyze_cross_layer_consistency(
        summary,
        stream_para,
        comm,
        cpu_pipe,
        kernel_comp,
    )

    scores = compute_performance_scores(
        util_pct,
        mem_io,
        stream_para,
        comm,
        gm,
        cpu_pipe,
    )
    recs = generate_recommendations(
        scores,
        mem_io,
        stream_para,
        comm,
        comm_health,
        gm,
        roofline_gap,
        kernel_comp,
        cpu_pipe,
        cross_layer_consistency,
        util_pct,
        short_kernels,
        kernel_duration_stats,
    )

    # The verdict leads. A reader who stops after one line should still get the
    # dominant cause rather than having to rank nine dimension scores themselves.
    triage = _triage_from_tables(
        summary,
        aggregate_kernels,
        nccl_breakdown,
        memcpy_bandwidth,
        sync_breakdown,
        cpu_launch_gap,
        kernel_duration_stats,
    )
    attribution = analyze_kernel_attribution(aggregate_kernels)

    # Validity gating. This runs last so it can strike conclusions the other
    # analyses produced, and its verdict travels with the result: "we checked
    # and it is sound" and "we never checked" must not read the same.
    quality = _assess_quality(trace_quality_inputs, aggregate_kernels)
    blocked = set(quality.get("blocked_conclusions") or ())
    if blocked:
        recs = _strike_blocked_recommendations(recs, blocked)

    return {
        "trace_quality": quality,
        "triage": triage,
        "kernel_attribution": attribution,
        "hardware": {
            "gpu_name": gpu_name,
            "specs": {
                "hbm_peak_tbps": specs.hbm_peak_tbps if specs else None,
                "pcie_peak_gbps": specs.pcie_peak_gbps if specs else None,
                "nvlink_peak_gbps": specs.nvlink_peak_gbps if specs else None,
                "sm_count": specs.sm_count if specs else None,
                "fp16_tflops": specs.fp16_tflops if specs else None,
            }
            if specs
            else {},
        },
        "memory_io": mem_io,
        "stream_parallelism": stream_para,
        "communication": comm,
        "communication_health": comm_health,
        "gpu_metrics": gm,
        "roofline_gap": roofline_gap,
        "kernel_composition": kernel_comp,
        "cpu_pipeline": cpu_pipe,
        "cross_layer_consistency": cross_layer_consistency,
        "performance_scores": scores,
        "recommendations": recs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report generation
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_EMOJI = {
    (90, 101): "✅",
    (70, 90): "🟡",
    (50, 70): "🟠",
    (30, 50): "🔴",
    (0, 30): "💀",
}


def _score_emoji(score: float) -> str:
    for (lo, hi), sym in _SCORE_EMOJI.items():
        if lo <= score < hi:
            return sym
    return "❓"


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def comprehensive_to_markdown(analysis: Dict) -> str:
    """Render the comprehensive analysis dict as a human-readable Markdown section."""
    lines: List[str] = []
    a = analysis

    def h2(t: str) -> None:
        lines.extend(["", f"## {t}", ""])

    def h3(t: str) -> None:
        lines.extend(["", f"### {t}", ""])

    # Verdict first. Everything below it is supporting detail.
    triage = a.get("triage") or {}
    if triage.get("verdict"):
        h2("Verdict")
        conf = triage.get("confidence") or "unknown"
        lines.append(f"**{triage['verdict']}**  (confidence: {conf})")
        if triage.get("summary"):
            lines.extend(["", str(triage["summary"])])
        evidence = triage.get("evidence") or {}
        if evidence:
            lines.append("")
            for key, value in list(evidence.items())[:8]:
                if value is None:
                    continue
                shown = (
                    f"{value:.3g}" if isinstance(value, (int, float)) else str(value)
                )
                lines.append(f"- {key}: {shown}")
        if conf in ("low", "unknown"):
            lines.extend(
                [
                    "",
                    "> Confidence is limited by what this trace carries. The nsys summary "
                    "tables give totals rather than intervals, so overlap between "
                    "communication and compute could not be measured directly.",
                ]
            )

    attribution = a.get("kernel_attribution") or {}
    if attribution.get("notes") or attribution.get("caveats"):
        h2("Kernel name caveats")
        lines.append(
            "Kernel symbols are the weakest evidence in a trace. These names would "
            "mislead a reader who took them at face value."
        )
        for note in attribution.get("notes") or ():
            lines.extend(["", f"- {note}"])
        caveats = attribution.get("caveats") or []
        if caveats:
            lines.append("")
            for item in caveats[:8]:
                lines.append(f"- `{item['kernel']}` ({item['total_ms']:.1f} ms)")
                for warning in item.get("warnings", [])[:2]:
                    lines.append(f"    - {warning}")
        total = attribution.get("caveat_count") or 0
        if total > len(caveats):
            lines.extend(["", f"_{total - len(caveats)} further kernel(s) not shown._"])

    def tbl_row(*cells: str) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    # ── Performance Dashboard ────────────────────────────────────────────────
    h2("🚀 Performance Dashboard")
    scores = a.get("performance_scores", {})
    overall = _f(scores.get("overall"))
    primary = str(scores.get("primary_bottleneck", "unknown")).replace("_", " ").title()
    secondary = (
        str(scores.get("secondary_bottleneck", "none")).replace("_", " ").title()
    )
    lines.append(f"**Overall Score: {overall:.0f} / 100** {_score_emoji(overall)}")
    lines.append(f"Primary bottleneck: **{primary}** · Secondary: **{secondary}**")
    lines.append("")
    lines.append(tbl_row("Dimension", "Score", "Visual", "Status"))
    lines.append(tbl_row("---", "---", "---", "---"))
    dim_labels = {
        # Deliberately not "GPU Utilisation". This is the share of wall time
        # during which *some* kernel was resident - the same quantity nvidia-smi
        # reports - and a single-block kernel occupying one SM scores 100%. It
        # says whether the GPU was occupied, never how hard it worked.
        "gpu_utilisation": "GPU Busy-Time",
        "memory_efficiency": "Memory Efficiency",
        "stream_parallelism": "Stream Parallelism",
        "communication": "Comm Efficiency",
        "cpu_pipeline": "CPU Pipeline",
    }
    for key, label in dim_labels.items():
        s = _f(scores.get(key))
        lines.append(tbl_row(label, f"{s:.0f}", _bar(s), _score_emoji(s)))

    lines.append("")
    lines.append(
        "> **GPU Busy-Time** is the share of wall time during which some kernel was "
        "resident, which is what `nvidia-smi` reports. A kernel occupying a single SM "
        "scores 100%. It answers *was the GPU occupied*, never *how hard did it work* - "
        "for that read SM Active, tensor-pipe activity and DRAM throughput."
    )

    # ── Hardware ─────────────────────────────────────────────────────────────
    hw = a.get("hardware", {})
    gpu_name = hw.get("gpu_name", "")
    specs = hw.get("specs", {})
    if gpu_name:
        h2("🖥  Hardware")
        lines.append(f"**GPU:** {gpu_name}")
        if specs:
            lines.append(
                f"- HBM peak: {specs.get('hbm_peak_tbps', '?')} TB/s  "
                f"· PCIe peak: {specs.get('pcie_peak_gbps', '?')} GB/s  "
                f"· NVLink: {specs.get('nvlink_peak_gbps') or 'N/A'} GB/s  "
                f"· SMs: {specs.get('sm_count', '?')}  "
                f"· FP16 Tensor: {specs.get('fp16_tflops', '?')} TFLOPS"
            )

    # ── Memory IO ────────────────────────────────────────────────────────────
    h2("💾 Memory IO Analysis")
    mi = a.get("memory_io", {})
    lines.append(
        tbl_row(
            "Direction", "Volume (GB)", "Duration (ms)", "Avg BW (GB/s)", "Peak Util %"
        )
    )
    lines.append(tbl_row("---", "---", "---", "---", "---"))
    by_dir = mi.get("by_direction", {})
    for label, d in sorted(
        by_dir.items(), key=lambda x: x[1].get("total_ms", 0), reverse=True
    ):
        lines.append(
            tbl_row(
                label,
                f"{d.get('total_gb', 0):.3f}",
                f"{d.get('total_ms', 0):.1f}",
                f"{d.get('avg_gbps', 0):.1f}",
                "—",
            )
        )
    pcie_u = mi.get("pcie_utilisation_pct")
    hbm_u = mi.get("hbm_utilisation_from_d2d_pct")
    if pcie_u is not None:
        lines.append(f"\n**PCIe utilisation:** {pcie_u:.1f}%  {_bar(pcie_u, 15)}")
    if hbm_u is not None:
        lines.append(f"**HBM utilisation (from D2D):** {hbm_u:.1f}%  {_bar(hbm_u, 15)}")
    sp = mi.get("small_transfers_pct", 0)
    if mi.get("small_transfers_total", 0) > 0:
        lines.append(
            f"\nSmall transfers (<1 MB): {mi.get('small_transfers_count', 0)}"
            f" / {mi.get('small_transfers_total', 0)} ({sp:.0f}%)"
        )
    for issue in mi.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── Stream Parallelism ───────────────────────────────────────────────────
    h2("🔀 Stream Parallelism")
    sp_data = a.get("stream_parallelism", {})
    lines.append(f"**Streams detected:** {sp_data.get('stream_count', 0)}")
    lines.append(
        f"**Parallelism ratio:** {sp_data.get('parallelism_ratio', 0):.2f}x "
        f"(>1.0 = concurrent execution)"
    )
    copy_det = sp_data.get("copy_stream_detected", False)
    lines.append(f"**Dedicated copy stream:** {'Yes ✅' if copy_det else 'No ⚠️'}")
    per_stream = sp_data.get("per_stream", [])
    if per_stream:
        lines.append("")
        lines.append(
            tbl_row(
                "Stream ID", "Role", "Kernel Count", "Busy (ms)", "Span (ms)", "Util %"
            )
        )
        lines.append(tbl_row("---", "---", "---", "---", "---", "---"))
        for r in per_stream[:10]:
            lines.append(
                tbl_row(
                    r.get("stream_id", "?"),
                    r.get("role", "compute"),
                    r.get("kernel_count", 0),
                    f"{r.get('kernel_busy_ms', 0):.1f}",
                    f"{r.get('stream_span_ms', 0):.1f}",
                    f"{r.get('utilisation_pct', 0):.1f}",
                )
            )
    for issue in sp_data.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── Communication ────────────────────────────────────────────────────────
    h2("📡 Communication (NCCL) Analysis")
    comm = a.get("communication", {})
    nccl_ms = comm.get("nccl_total_ms", 0)
    if nccl_ms == 0:
        lines.append("*No NCCL communication kernels detected in this window.*")
    else:
        lines.append(
            f"NCCL total: **{nccl_ms:.1f} ms** ({comm.get('nccl_span_pct', 0):.1f}% of window)"
        )
        lines.append(
            f"Compute / comm ratio: **{comm.get('compute_comm_ratio', 0):.1f}x** "
            f"(target ≥5x for effective overlap)"
        )
        ov = comm.get("overlap_pct_of_comm", 0)
        lines.append(f"Compute-comm overlap: **{ov:.0f}%** of NCCL time")
        bubble = comm.get("pipeline_bubble_pct", 0)
        lines.append(f"Pipeline bubble: **{bubble:.1f}%** of window (comm stall)")
        top_ops = comm.get("top_collectives", [])
        if top_ops:
            lines.append("")
            lines.append(tbl_row("Collective", "Total (ms)", "Count", "Avg (ms)"))
            lines.append(tbl_row("---", "---", "---", "---"))
            for op in top_ops[:6]:
                name = str(op.get("kernel_name") or "")[:60]
                lines.append(
                    tbl_row(
                        name,
                        f"{op.get('total_ms', 0):.1f}",
                        op.get("count", 0),
                        f"{op.get('avg_ms', 0):.2f}",
                    )
                )
    for issue in comm.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── Communication Health ────────────────────────────────────────────────
    h2("🩺 Communication Health")
    ch = a.get("communication_health", {})
    lines.append(
        f"Health score: **{_f(ch.get('health_score')):.1f}/100** · "
        f"Status: **{str(ch.get('status') or 'unknown')}**"
    )
    lines.append(
        f"Overlap: **{_f(ch.get('overlap_pct_of_comm')):.1f}%** · "
        f"Bubble: **{_f(ch.get('pipeline_bubble_pct')):.1f}%** · "
        f"Compute/Comm: **{_f(ch.get('compute_comm_ratio')):.2f}x** · "
        f"Op skew(max/mean): **{_f(ch.get('op_skew_ratio_max_over_mean')):.2f}**"
    )
    for issue in ch.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── GPU Metrics ──────────────────────────────────────────────────────────
    h2("📊 GPU Hardware Metrics")
    gm = a.get("gpu_metrics", {})
    if not gm.get("available"):
        lines.append(
            "*GPU hardware metrics not available. Re-profile with "
            "`--gpu-metrics-devices all` to enable SM active, tensor active, "
            "and DRAM throughput analysis.*"
        )
    else:
        bt = str(gm.get("bottleneck_type", "unknown")).replace("_", " ").title()
        lines.append(f"**Detected bottleneck type:** {bt}")
        hbm_u = gm.get("hbm_utilisation_pct")
        if hbm_u is not None:
            lines.append(
                f"**HBM bandwidth utilisation:** {hbm_u:.1f}%  {_bar(hbm_u, 15)}"
            )
        lines.append("")
        lines.append(tbl_row("Metric", "Avg", "P50", "P75", "P95", "Max", "StdDev"))
        lines.append(tbl_row("---", "---", "---", "---", "---", "---", "---"))

        def _mrow(key: str, label: str) -> None:
            d = gm.get(key, {})
            if not d:
                return
            fmt = lambda v: f"{v:.1f}" if v is not None else "—"
            lines.append(
                tbl_row(
                    label,
                    fmt(d.get("avg")),
                    fmt(d.get("p50")),
                    fmt(d.get("p75")),
                    fmt(d.get("p95")),
                    fmt(d.get("max")),
                    fmt(d.get("stddev")),
                )
            )

        _mrow("sm_active", "SM Active (%)")
        _mrow("tensor_active", "Tensor Active (%)")
        _mrow("dram_throughput", "DRAM Throughput (%)")
    for issue in gm.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── Roofline Gap ───────────────────────────────────────────────────────
    h2("📐 Roofline Gap")
    rg = a.get("roofline_gap", {})
    lines.append(
        f"Compute util: **{_f(rg.get('compute_util_pct')):.1f}%** · "
        f"Memory util: **{_f(rg.get('memory_util_pct')):.1f}%** · "
        f"Gap: **{_f(rg.get('roofline_gap_pct')):.1f}** points · "
        f"Hint: **{str(rg.get('bottleneck_hint') or 'balanced')}**"
    )
    ai_proxy = rg.get("arithmetic_intensity_proxy")
    if ai_proxy is not None:
        lines.append(f"Arithmetic intensity proxy: **{_f(ai_proxy):.3f}**")
    if rg.get("achieved_compute_tflops_est") is not None:
        lines.append(
            f"Estimated achieved compute: **{_f(rg.get('achieved_compute_tflops_est')):.2f} TFLOPS**"
        )
    if rg.get("achieved_memory_gbps_est") is not None:
        lines.append(
            f"Estimated achieved memory BW: **{_f(rg.get('achieved_memory_gbps_est')):.2f} GB/s**"
        )
    for issue in rg.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── Kernel Composition ───────────────────────────────────────────────────
    h2("🧩 Kernel Workload Composition")
    kc = a.get("kernel_composition", {})
    cats = kc.get("categories", {})
    if cats:
        lines.append(
            tbl_row(
                "Category",
                "GPU Time (ms)",
                "% of Kernel Time",
                "Kernel Count",
                "Visual",
            )
        )
        lines.append(tbl_row("---", "---", "---", "---", "---"))
        for cat, d in cats.items():
            pct_v = _f(d.get("pct_of_gpu_time"))
            lines.append(
                tbl_row(
                    cat.replace("_", " ").title(),
                    f"{d.get('total_ms', 0):.1f}",
                    f"{pct_v:.1f}",
                    d.get("kernel_count", 0),
                    _bar(pct_v, 15),
                )
            )
    for issue in kc.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── CPU Pipeline ─────────────────────────────────────────────────────────
    h2("⚙️  CPU-GPU Pipeline")
    cp = a.get("cpu_pipeline", {})
    lines.append(
        f"Launch gap total: **{cp.get('total_launch_gap_ms', 0):.1f} ms** "
        f"({cp.get('launch_gap_pct', 0):.1f}% of window)"
    )
    lines.append(
        f"Sync overhead: **{cp.get('total_sync_ms', 0):.1f} ms** "
        f"({cp.get('sync_pct', 0):.1f}% of window)"
    )
    worst = cp.get("worst_launch_gap", {})
    if worst.get("kernel_name"):
        lines.append(
            f"Worst launch gap kernel: `{worst['kernel_name']}` "
            f"avg {worst.get('avg_gap_us', 0):.1f} µs"
        )
    top_sync = cp.get("top_sync_types", [])
    if top_sync:
        lines.append("")
        lines.append(tbl_row("Sync Type", "Count", "Total (ms)", "Avg (ms)"))
        lines.append(tbl_row("---", "---", "---", "---"))
        for r in top_sync:
            lines.append(
                tbl_row(
                    r.get("sync_type", "?"),
                    r.get("count", 0),
                    f"{r.get('total_ms', 0):.1f}",
                    f"{r.get('avg_ms', 0):.2f}",
                )
            )
    for issue in cp.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── Cross-layer Consistency ─────────────────────────────────────────────
    h2("🧬 Cross-Layer Consistency")
    cl = a.get("cross_layer_consistency", {})
    lines.append(
        f"Consistency score: **{_f(cl.get('consistency_score')):.1f}/100** · "
        f"Coverage: **{_f(cl.get('coverage_ratio')) * 100.0:.1f}%** · "
        f"Busy alignment ratio: **{_f(cl.get('busy_alignment_ratio')):.2f}**"
    )
    comp = cl.get("component_breakdown_ms") or {}
    if comp:
        lines.append(
            f"Modeled components: compute={_f(comp.get('compute_ms')):.1f} ms, "
            f"comm={_f(comp.get('comm_ms')):.1f} ms, "
            f"cpu_sync={_f(comp.get('cpu_sync_ms')):.1f} ms, "
            f"launch_gap={_f(comp.get('cpu_launch_gap_ms')):.1f} ms."
        )
    for issue in cl.get("issues", []):
        lines.append(f"\n> ⚠️  {issue}")

    # ── Recommendations ──────────────────────────────────────────────────────
    h2("🎯 Prioritised Recommendations")
    recs = a.get("recommendations", [])
    if not recs:
        lines.append("*No significant performance issues detected.*")
    else:
        priority_icon = {"critical": "💀", "high": "🔴", "medium": "🟠", "low": "🟡"}
        for i, rec in enumerate(recs, 1):
            icon = priority_icon.get(rec.get("priority", "low"), "ℹ️")
            lines.append(
                f"\n**{i}. [{rec.get('priority', '?').upper()}] {icon} {rec.get('title')}**"
                f"  *(score: {rec.get('score', 0):.0f})*"
            )
            lines.append(f"\n{rec.get('detail', '')}")
            refs = rec.get("references", [])
            if refs:
                lines.append(
                    "\nReferences: " + " · ".join(f"[link]({r})" for r in refs)
                )

    return "\n".join(lines)
