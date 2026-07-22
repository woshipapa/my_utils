"""Authoritative GPU hardware specifications for roofline / speed-of-light analysis.

Every peak in this module is a **dense** figure. NVIDIA datasheets usually quote
tensor-core numbers "with sparsity", which is 2x the dense rate; comparing an
achieved FLOP/s against a sparsity-inflated peak silently halves every
utilisation number, so the sparse variants are derived on demand instead
(:meth:`GpuSpec.peak_tflops` with ``sparse=True``).

Sources for the constants:

- NVIDIA datasheets (H100/H200/L40S/B200) for advertised peaks.
- Luo et al., "Benchmarking and Dissecting the Nvidia Hopper GPU Architecture"
  (arXiv:2402.13499) for measured L2/SMEM/global bandwidth and latencies.
- Jia et al., "Dissecting the NVIDIA Volta GPU Architecture via
  Microbenchmarking" (arXiv:1804.06826) for the Volta-era measured numbers.

Measured bandwidths are recorded separately from theoretical ones because
achieved DRAM bandwidth tops out around 80-90% of spec; grading a kernel
against the spec number makes a perfectly good kernel look broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "GpuSpec",
    "lookup_gpu_spec",
    "list_known_gpus",
    "DTYPE_ALIASES",
    "normalize_dtype",
]


# Canonical dtype keys used by :meth:`GpuSpec.peak_tflops`.
DTYPE_ALIASES: Dict[str, str] = {
    "fp64": "fp64",
    "float64": "fp64",
    "double": "fp64",
    "fp32": "fp32",
    "float32": "fp32",
    "float": "fp32",
    "tf32": "tf32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "float8": "fp8",
    "e4m3": "fp8",
    "e5m2": "fp8",
    "fp4": "fp4",
    "int8": "int8",
    "i8": "int8",
}


def normalize_dtype(dtype: str) -> str:
    """Map a user-supplied dtype spelling onto a canonical key."""
    return DTYPE_ALIASES.get(
        str(dtype or "").strip().lower(), str(dtype or "").strip().lower()
    )


@dataclass(frozen=True)
class GpuSpec:
    """Peak capabilities of one GPU SKU.

    All TFLOPS fields are dense tensor-core rates except ``fp32_tflops`` and
    ``fp64_tflops``, which are the non-tensor vector rates (what a plain
    ``FFMA``/``DFMA`` loop can reach).  ``int8_tops`` is TOPS, not TFLOPS.
    """

    name: str
    arch: str  # volta | turing | ampere | ada | hopper | blackwell
    compute_capability: Tuple[int, int]
    sm_count: int

    # Memory subsystem
    hbm_bandwidth_gbps: float  # theoretical peak, GB/s
    memory_gb: float
    l2_size_mb: float = 0.0
    # Measured ceilings (GB/s). 0.0 means "not microbenchmarked here"; callers
    # should fall back to a fraction of the theoretical peak.
    hbm_measured_gbps: float = 0.0
    l2_bandwidth_gbps: float = 0.0
    smem_bandwidth_gbps: float = 0.0

    # Interconnect, per-direction GB/s
    nvlink_gbps: float = 0.0
    pcie_gbps: float = 0.0

    # Dense compute peaks (TFLOPS; int8 in TOPS). 0.0 means "not supported".
    fp64_tflops: float = 0.0
    fp32_tflops: float = 0.0
    tf32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    fp8_tflops: float = 0.0
    fp4_tflops: float = 0.0
    int8_tops: float = 0.0

    supports_sparsity: bool = True
    aliases: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    # -- derived ---------------------------------------------------------

    def peak_tflops(
        self, dtype: str = "bf16", *, sparse: bool = False
    ) -> Optional[float]:
        """Dense peak for ``dtype``; ``sparse=True`` applies the 2x structured-sparsity rate."""
        key = normalize_dtype(dtype)
        value = {
            "fp64": self.fp64_tflops,
            "fp32": self.fp32_tflops,
            "tf32": self.tf32_tflops,
            "fp16": self.fp16_tflops,
            "bf16": self.bf16_tflops,
            "fp8": self.fp8_tflops,
            "fp4": self.fp4_tflops,
            "int8": self.int8_tops,
        }.get(key)
        if not value:
            return None
        if sparse:
            if not self.supports_sparsity or key in ("fp32", "fp64"):
                return None
            return value * 2.0
        return float(value)

    def effective_hbm_gbps(self, *, achievable_fraction: float = 0.85) -> float:
        """Bandwidth a well-written streaming kernel can actually reach.

        Prefers the microbenchmarked number when we have one, otherwise applies
        ``achievable_fraction`` to the spec figure (Citadel measured 83.3% on
        V100, and the Hopper study lands in the same band).
        """
        if self.hbm_measured_gbps > 0:
            return self.hbm_measured_gbps
        return self.hbm_bandwidth_gbps * float(achievable_fraction)

    def ridge_point(
        self, dtype: str = "bf16", *, sparse: bool = False
    ) -> Optional[float]:
        """Arithmetic intensity (FLOP/byte of DRAM traffic) where the roofline knee sits.

        ``AI < ridge`` => memory bound, ceiling is ``AI * hbm_bandwidth``.
        ``AI > ridge`` => compute bound, ceiling is the dtype peak.
        """
        peak = self.peak_tflops(dtype, sparse=sparse)
        if peak is None or self.hbm_bandwidth_gbps <= 0:
            return None
        # TFLOP/s / (GB/s) == 1000 * FLOP/byte
        return (peak * 1e12) / (self.hbm_bandwidth_gbps * 1e9)

    def attainable_tflops(
        self, arithmetic_intensity: float, dtype: str = "bf16"
    ) -> Optional[float]:
        """Roofline ceiling at a given arithmetic intensity (FLOP/byte)."""
        peak = self.peak_tflops(dtype)
        if peak is None:
            return None
        memory_bound = (
            float(arithmetic_intensity) * self.hbm_bandwidth_gbps * 1e9
        ) / 1e12
        return min(peak, memory_bound)

    def matmul_vs_vector_ratio(self) -> Optional[float]:
        """How much cheaper a tensor-core FLOP is than a vector FP32 FLOP.

        Horace He's central cost-model constant: ~16x on A100.  When this is
        large, non-matmul work that looks negligible in FLOP terms can still
        dominate runtime.
        """
        if not self.bf16_tflops or not self.fp32_tflops:
            return None
        return self.bf16_tflops / self.fp32_tflops


# Ordered most-specific first: matching walks this list and takes the first
# substring hit, so "h100 sxm" must precede the bare "h100".
_GPU_SPECS: List[GpuSpec] = [
    # ---- Hopper -------------------------------------------------------
    GpuSpec(
        name="H200 SXM",
        arch="hopper",
        compute_capability=(9, 0),
        sm_count=132,
        hbm_bandwidth_gbps=4800.0,
        memory_gb=141.0,
        l2_size_mb=50.0,
        l2_bandwidth_gbps=12000.0,
        smem_bandwidth_gbps=31000.0,
        nvlink_gbps=900.0,
        pcie_gbps=64.0,
        fp64_tflops=34.0,
        fp32_tflops=67.0,
        tf32_tflops=494.7,
        fp16_tflops=989.4,
        bf16_tflops=989.4,
        fp8_tflops=1979.0,
        int8_tops=1979.0,
        aliases=("h200",),
        notes="Lowest BF16 ridge in the Hopper family (~206) thanks to HBM3e.",
    ),
    GpuSpec(
        name="H100 SXM5",
        arch="hopper",
        compute_capability=(9, 0),
        sm_count=132,
        hbm_bandwidth_gbps=3350.0,
        memory_gb=80.0,
        l2_size_mb=50.0,
        l2_bandwidth_gbps=12000.0,
        smem_bandwidth_gbps=31000.0,
        nvlink_gbps=900.0,
        pcie_gbps=64.0,
        fp64_tflops=34.0,
        fp32_tflops=67.0,
        tf32_tflops=494.7,
        fp16_tflops=989.4,
        bf16_tflops=989.4,
        fp8_tflops=1979.0,
        int8_tops=1979.0,
        aliases=("h100 sxm", "h100 80gb hbm3", "h100-sxm"),
    ),
    GpuSpec(
        name="H100 NVL",
        arch="hopper",
        compute_capability=(9, 0),
        sm_count=132,
        hbm_bandwidth_gbps=3900.0,
        memory_gb=94.0,
        l2_size_mb=50.0,
        l2_bandwidth_gbps=12000.0,
        smem_bandwidth_gbps=31000.0,
        nvlink_gbps=600.0,
        pcie_gbps=64.0,
        fp64_tflops=30.0,
        fp32_tflops=60.0,
        tf32_tflops=494.7,
        fp16_tflops=989.4,
        bf16_tflops=989.4,
        fp8_tflops=1979.0,
        int8_tops=1979.0,
        aliases=("h100nvl",),
    ),
    GpuSpec(
        name="H100 PCIe",
        arch="hopper",
        compute_capability=(9, 0),
        sm_count=114,
        hbm_bandwidth_gbps=2000.0,
        memory_gb=80.0,
        l2_size_mb=50.0,
        l2_bandwidth_gbps=10000.0,
        smem_bandwidth_gbps=26000.0,
        nvlink_gbps=600.0,
        pcie_gbps=64.0,
        fp64_tflops=26.0,
        fp32_tflops=51.0,
        tf32_tflops=378.0,
        fp16_tflops=756.0,
        bf16_tflops=756.0,
        fp8_tflops=1513.0,
        int8_tops=1513.0,
        aliases=("h100-pcie",),
    ),
    GpuSpec(
        name="H800 SXM",
        arch="hopper",
        compute_capability=(9, 0),
        sm_count=132,
        hbm_bandwidth_gbps=3350.0,
        memory_gb=80.0,
        l2_size_mb=50.0,
        hbm_measured_gbps=1861.5,
        l2_bandwidth_gbps=7800.0,
        smem_bandwidth_gbps=30000.0,
        nvlink_gbps=400.0,
        pcie_gbps=64.0,
        fp64_tflops=34.0,
        fp32_tflops=67.0,
        tf32_tflops=494.7,
        fp16_tflops=989.4,
        bf16_tflops=989.4,
        fp8_tflops=1979.0,
        int8_tops=1979.0,
        aliases=("h800",),
        notes=(
            "Export SKU: NVLink cut to 400 GB/s. Measured single-stream HBM 1861 GB/s "
            "(Luo et al. on the PCIe part). DeepSeek-V3 burned 20/132 SMs on comm kernels."
        ),
    ),
    GpuSpec(
        name="H20",
        arch="hopper",
        compute_capability=(9, 0),
        sm_count=78,
        hbm_bandwidth_gbps=4000.0,
        memory_gb=96.0,
        l2_size_mb=60.0,
        nvlink_gbps=900.0,
        pcie_gbps=64.0,
        fp64_tflops=1.0,
        fp32_tflops=44.0,
        tf32_tflops=74.0,
        fp16_tflops=148.0,
        bf16_tflops=148.0,
        fp8_tflops=296.0,
        int8_tops=296.0,
        aliases=("h20",),
        notes="Export SKU: huge HBM bandwidth relative to a small compute peak (ridge ~37).",
    ),
    GpuSpec(
        name="GH200",
        arch="hopper",
        compute_capability=(9, 0),
        sm_count=132,
        hbm_bandwidth_gbps=4900.0,
        memory_gb=96.0,
        l2_size_mb=50.0,
        l2_bandwidth_gbps=12000.0,
        smem_bandwidth_gbps=31000.0,
        nvlink_gbps=900.0,
        pcie_gbps=64.0,
        fp64_tflops=34.0,
        fp32_tflops=67.0,
        tf32_tflops=494.7,
        fp16_tflops=989.4,
        bf16_tflops=989.4,
        fp8_tflops=1979.0,
        int8_tops=1979.0,
        aliases=("gh200", "grace hopper"),
    ),
    # ---- Blackwell ----------------------------------------------------
    GpuSpec(
        name="B200",
        arch="blackwell",
        compute_capability=(10, 0),
        sm_count=148,
        hbm_bandwidth_gbps=8000.0,
        memory_gb=180.0,
        l2_size_mb=126.0,
        nvlink_gbps=1800.0,
        pcie_gbps=128.0,
        fp64_tflops=45.0,
        fp32_tflops=90.0,
        tf32_tflops=1250.0,
        fp16_tflops=2250.0,
        bf16_tflops=2250.0,
        fp8_tflops=4500.0,
        fp4_tflops=9000.0,
        int8_tops=4500.0,
        aliases=("b200", "hgx b200"),
        notes="HGX B200 figures; GB200 at 1200 W is quoted ~2500 TFLOPS dense BF16.",
    ),
    GpuSpec(
        name="B100",
        arch="blackwell",
        compute_capability=(10, 0),
        sm_count=148,
        hbm_bandwidth_gbps=8000.0,
        memory_gb=180.0,
        l2_size_mb=126.0,
        nvlink_gbps=1800.0,
        pcie_gbps=128.0,
        fp64_tflops=30.0,
        fp32_tflops=60.0,
        tf32_tflops=875.0,
        fp16_tflops=1750.0,
        bf16_tflops=1750.0,
        fp8_tflops=3500.0,
        fp4_tflops=7000.0,
        int8_tops=3500.0,
        aliases=("b100",),
    ),
    # ---- Ampere -------------------------------------------------------
    GpuSpec(
        name="A100 SXM 80GB",
        arch="ampere",
        compute_capability=(8, 0),
        sm_count=108,
        hbm_bandwidth_gbps=2039.0,
        memory_gb=80.0,
        l2_size_mb=40.0,
        hbm_measured_gbps=1407.2,
        l2_bandwidth_gbps=2600.0,
        smem_bandwidth_gbps=19500.0,
        nvlink_gbps=600.0,
        pcie_gbps=64.0,
        fp64_tflops=9.7,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        int8_tops=624.0,
        aliases=("a100 80gb sxm", "a100-sxm4-80gb", "a100 sxm4 80gb"),
        notes="Horace He's cost model anchor: 312 TFLOPS matmul vs 19.5 vector => 16x.",
    ),
    GpuSpec(
        name="A100 PCIe 80GB",
        arch="ampere",
        compute_capability=(8, 0),
        sm_count=108,
        hbm_bandwidth_gbps=1935.0,
        memory_gb=80.0,
        l2_size_mb=40.0,
        l2_bandwidth_gbps=2600.0,
        smem_bandwidth_gbps=19500.0,
        nvlink_gbps=600.0,
        pcie_gbps=64.0,
        fp64_tflops=9.7,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        int8_tops=624.0,
        aliases=("a100 80gb pcie",),
    ),
    GpuSpec(
        name="A100 40GB",
        arch="ampere",
        compute_capability=(8, 0),
        sm_count=108,
        hbm_bandwidth_gbps=1555.0,
        memory_gb=40.0,
        l2_size_mb=40.0,
        l2_bandwidth_gbps=2600.0,
        smem_bandwidth_gbps=19500.0,
        nvlink_gbps=600.0,
        pcie_gbps=64.0,
        fp64_tflops=9.7,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        int8_tops=624.0,
        aliases=("a100 40gb", "a100-sxm4-40gb", "a100-pcie-40gb"),
    ),
    GpuSpec(
        name="A100",
        arch="ampere",
        compute_capability=(8, 0),
        sm_count=108,
        hbm_bandwidth_gbps=2039.0,
        memory_gb=80.0,
        l2_size_mb=40.0,
        l2_bandwidth_gbps=2600.0,
        smem_bandwidth_gbps=19500.0,
        nvlink_gbps=600.0,
        pcie_gbps=64.0,
        fp64_tflops=9.7,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        int8_tops=624.0,
        aliases=("a100",),
    ),
    GpuSpec(
        name="A800",
        arch="ampere",
        compute_capability=(8, 0),
        sm_count=108,
        hbm_bandwidth_gbps=2039.0,
        memory_gb=80.0,
        l2_size_mb=40.0,
        l2_bandwidth_gbps=2600.0,
        smem_bandwidth_gbps=19500.0,
        nvlink_gbps=400.0,
        pcie_gbps=64.0,
        fp64_tflops=9.7,
        fp32_tflops=19.5,
        tf32_tflops=156.0,
        fp16_tflops=312.0,
        bf16_tflops=312.0,
        int8_tops=624.0,
        aliases=("a800",),
        notes="Export SKU: A100 compute with NVLink cut to 400 GB/s.",
    ),
    GpuSpec(
        name="A30",
        arch="ampere",
        compute_capability=(8, 0),
        sm_count=56,
        hbm_bandwidth_gbps=933.0,
        memory_gb=24.0,
        l2_size_mb=24.0,
        nvlink_gbps=200.0,
        pcie_gbps=64.0,
        fp64_tflops=5.2,
        fp32_tflops=10.3,
        tf32_tflops=82.0,
        fp16_tflops=165.0,
        bf16_tflops=165.0,
        int8_tops=330.0,
        aliases=("a30",),
    ),
    GpuSpec(
        name="A10G",
        arch="ampere",
        compute_capability=(8, 6),
        sm_count=80,
        hbm_bandwidth_gbps=600.0,
        memory_gb=24.0,
        l2_size_mb=6.0,
        pcie_gbps=64.0,
        fp32_tflops=31.5,
        tf32_tflops=35.0,
        fp16_tflops=70.0,
        bf16_tflops=70.0,
        int8_tops=140.0,
        aliases=("a10g",),
    ),
    GpuSpec(
        name="A10",
        arch="ampere",
        compute_capability=(8, 6),
        sm_count=72,
        hbm_bandwidth_gbps=600.0,
        memory_gb=24.0,
        l2_size_mb=6.0,
        pcie_gbps=64.0,
        fp32_tflops=31.2,
        tf32_tflops=31.2,
        fp16_tflops=62.5,
        bf16_tflops=62.5,
        int8_tops=125.0,
        aliases=("a10",),
    ),
    GpuSpec(
        name="RTX A6000",
        arch="ampere",
        compute_capability=(8, 6),
        sm_count=84,
        hbm_bandwidth_gbps=768.0,
        memory_gb=48.0,
        l2_size_mb=6.0,
        pcie_gbps=64.0,
        fp32_tflops=38.7,
        tf32_tflops=38.7,
        fp16_tflops=77.4,
        bf16_tflops=77.4,
        int8_tops=155.0,
        aliases=("a6000", "rtx a6000"),
        notes="Simon Boehm's SGEMM worklog device (30 TFLOPS FP32 quoted, 768 GB/s).",
    ),
    # ---- Ada ----------------------------------------------------------
    GpuSpec(
        name="L40S",
        arch="ada",
        compute_capability=(8, 9),
        sm_count=142,
        hbm_bandwidth_gbps=864.0,
        memory_gb=48.0,
        l2_size_mb=96.0,
        pcie_gbps=64.0,
        fp32_tflops=91.6,
        tf32_tflops=91.6,
        fp16_tflops=181.0,
        bf16_tflops=181.0,
        fp8_tflops=366.0,
        int8_tops=366.0,
        aliases=("l40s",),
        notes=(
            "Datasheet quotes 362 TFLOPS BF16 / 733 TFLOPS FP8 *with sparsity*; the dense "
            "halves are stored here. Highest BF16 ridge of the common SKUs (~209), so on "
            "L40S nearly every transformer kernel is memory bound."
        ),
    ),
    GpuSpec(
        name="L40",
        arch="ada",
        compute_capability=(8, 9),
        sm_count=142,
        hbm_bandwidth_gbps=864.0,
        memory_gb=48.0,
        l2_size_mb=96.0,
        pcie_gbps=64.0,
        fp32_tflops=90.5,
        tf32_tflops=90.5,
        fp16_tflops=181.0,
        bf16_tflops=181.0,
        fp8_tflops=362.0,
        int8_tops=362.0,
        aliases=("l40",),
    ),
    GpuSpec(
        name="L20",
        arch="ada",
        compute_capability=(8, 9),
        sm_count=92,
        hbm_bandwidth_gbps=864.0,
        memory_gb=48.0,
        l2_size_mb=96.0,
        pcie_gbps=64.0,
        fp32_tflops=59.8,
        tf32_tflops=59.8,
        fp16_tflops=119.5,
        bf16_tflops=119.5,
        fp8_tflops=239.0,
        int8_tops=239.0,
        aliases=("l20",),
    ),
    GpuSpec(
        name="L4",
        arch="ada",
        compute_capability=(8, 9),
        sm_count=58,
        hbm_bandwidth_gbps=300.0,
        memory_gb=24.0,
        l2_size_mb=48.0,
        pcie_gbps=64.0,
        fp32_tflops=30.3,
        tf32_tflops=30.3,
        fp16_tflops=60.5,
        bf16_tflops=60.5,
        fp8_tflops=121.0,
        int8_tops=121.0,
        aliases=("l4",),
    ),
    GpuSpec(
        name="RTX 4090",
        arch="ada",
        compute_capability=(8, 9),
        sm_count=128,
        hbm_bandwidth_gbps=1008.0,
        memory_gb=24.0,
        l2_size_mb=72.0,
        hbm_measured_gbps=929.8,
        l2_bandwidth_gbps=2800.0,
        pcie_gbps=64.0,
        fp32_tflops=82.6,
        tf32_tflops=82.6,
        fp16_tflops=165.2,
        bf16_tflops=165.2,
        fp8_tflops=330.3,
        int8_tops=330.3,
        aliases=("rtx 4090", "geforce rtx 4090", "4090"),
    ),
    # ---- Volta / Turing ------------------------------------------------
    GpuSpec(
        name="V100 SXM2",
        arch="volta",
        compute_capability=(7, 0),
        sm_count=80,
        hbm_bandwidth_gbps=900.0,
        memory_gb=32.0,
        l2_size_mb=6.0,
        hbm_measured_gbps=750.0,
        l2_bandwidth_gbps=2155.0,
        smem_bandwidth_gbps=12080.0,
        nvlink_gbps=300.0,
        pcie_gbps=32.0,
        fp64_tflops=7.8,
        fp32_tflops=15.7,
        fp16_tflops=125.0,
        bf16_tflops=0.0,
        supports_sparsity=False,
        aliases=("v100 sxm2", "v100 sxm", "v100"),
        notes="Citadel measured 750 GB/s of the 900 GB/s spec (83.3%). No BF16.",
    ),
    GpuSpec(
        name="T4",
        arch="turing",
        compute_capability=(7, 5),
        sm_count=40,
        hbm_bandwidth_gbps=320.0,
        memory_gb=16.0,
        l2_size_mb=4.0,
        pcie_gbps=32.0,
        fp32_tflops=8.1,
        fp16_tflops=65.0,
        bf16_tflops=0.0,
        int8_tops=130.0,
        supports_sparsity=False,
        aliases=("t4",),
    ),
]

_ALIAS_INDEX: List[Tuple[str, GpuSpec]] = []
for _spec in _GPU_SPECS:
    for _key in (_spec.name.lower(),) + tuple(a.lower() for a in _spec.aliases):
        _ALIAS_INDEX.append((_key, _spec))
# Longest key first so "h100 sxm5" beats "h100".
_ALIAS_INDEX.sort(key=lambda item: len(item[0]), reverse=True)


def lookup_gpu_spec(gpu_name: str) -> Optional[GpuSpec]:
    """Best-effort match of an nsys/ncu-reported GPU name onto a known SKU."""
    text = str(gpu_name or "").lower()
    if not text:
        return None
    for key, spec in _ALIAS_INDEX:
        if key in text:
            return spec
    return None


def list_known_gpus() -> List[str]:
    """Canonical names of every SKU in the table."""
    return [spec.name for spec in _GPU_SPECS]
