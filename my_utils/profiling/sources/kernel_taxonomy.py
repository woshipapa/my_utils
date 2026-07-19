"""Kernel-name classification: category, tensor-core usage, GEMM tiles, NCCL collectives.

Kernel names are the only workload signal present in every nsys report, so a lot
of diagnosis hangs off parsing them well.  Three things are extracted here:

1. :func:`classify_kernel` - a coarse category (matmul / attention / elementwise
   / communication / ...) plus the framework that emitted it.  Categories drive
   the "what should this kernel's speed-of-light look like" expectations.
2. :func:`parse_gemm_kernel` - tile shape and dtype out of cuBLAS/CUTLASS names,
   which is what tile- and wave-quantisation checks need.
3. :func:`parse_nccl_kernel` - collective, dtype, algorithm and protocol, which
   is what bus-bandwidth accounting needs.

Tensor-core detection uses the substring allowlist that PyTorch's TensorBoard
profiler plugin shipped (``TC_Allowlist``), extended with Hopper/Blackwell
warpgroup-MMA and FP8 spellings that postdate it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "KernelInfo",
    "GemmShape",
    "NcclCollective",
    "classify_kernel",
    "is_megakernel",
    "parse_gemm_kernel",
    "parse_nccl_kernel",
    "uses_tensor_cores",
    "is_communication_kernel",
    "uses_nvls_multicast",
    "KERNEL_CATEGORIES",
    "CATEGORY_EXPECTATIONS",
]


# ---------------------------------------------------------------------------
# Tensor-core detection
# ---------------------------------------------------------------------------

# Substrings that identify a tensor-core (MMA) kernel. The first block is the
# allowlist from torch-tb-profiler (kineto tb_plugin tensor_core.py); the rest
# covers Hopper WGMMA, Blackwell tcgen05, CUTLASS 3.x and FP8 paths that the
# original list predates.
_TENSOR_CORE_TOKENS: Tuple[str, ...] = (
    # torch-tb-profiler TC_Allowlist
    "h884", "s884", "h1688", "s1688", "hmma", "i8816", "16816",
    "dgrad_1x1_stride_2x2", "first_layer_wgrad_kernel", "conv1x1", "conv2d_c1_k1",
    "direct_group", "xmma_implicit_gemm", "xmma_sparse_conv",
    "xmma_warp_specialized_implicit_gemm", "xmma_gemm", "xmma_sparse_gemm", "c1688",
    # Hopper / Blackwell / modern CUTLASS
    "wgmma", "gmma", "qgmma", "igmma", "bgmma", "tcgen05", "utcmma", "utchmma",
    "sm90_", "sm100_", "cutlass3x", "nvjet",
    # dtype-tagged tensor paths
    "imma", "bmma", "dmma", "e4m3", "e5m2", "fp8_gemm", "scaled_mm",
)

# Kernels that are matmul-shaped but explicitly NOT on tensor cores.
_NON_TENSOR_CORE_TOKENS: Tuple[str, ...] = ("simt", "sgemm_", "cgemm", "zgemm")


def uses_tensor_cores(kernel_name: str) -> Optional[bool]:
    """Whether a kernel runs on tensor cores.

    Returns ``None`` when the name carries no evidence either way - important to
    distinguish from ``False``, because "no tensor cores" is only actionable
    advice for kernels that *could* have used them.
    """
    low = str(kernel_name or "").lower()
    if not low:
        return None
    if any(token in low for token in _TENSOR_CORE_TOKENS):
        return True
    if any(token in low for token in _NON_TENSOR_CORE_TOKENS):
        return False
    return None


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

# Ordered (category, patterns) - first match wins, so specific categories such as
# attention must precede the generic matmul patterns they may also contain.
KERNEL_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("communication", (
        r"^nccl", r"ncclkernel", r"nccldevkernel", r"^rccl", r"^hccl",
        r"all_?reduce", r"all_?gather", r"reduce_?scatter", r"broadcast_?kernel",
        r"send_?recv", r"alltoall", r"all_?to_?all", r"_a2a\b", r"\ba2a_",
        r"nvshmem", r"\bshmem_", r"put_?block", r"pull_?kernel", r"push_?kernel",
    )),
    ("attention", (
        r"flash_?fwd", r"flash_?bwd", r"flash::", r"flash_?attn", r"flashattn",
        r"fmha", r"fused_?multihead", r"cudnn.*sdpa", r"sdpa",
        r"attention_?kernel", r"scaled_?dot", r"multihead", r"xformers",
        r"paged_?attention", r"ring_?attention", r"splitkv",
    )),
    ("matmul", (
        r"gemm", r"cublas", r"cutlass", r"nvjet", r"matmul", r"^mm_", r"_mm_kernel",
        r"s1688", r"s16816", r"h1688", r"h16816", r"wgmma", r"xmma",
        r"wgrad", r"dgrad", r"fprop", r"implicit_?gemm", r"scaled_?mm",
        r"grouped_?gemm", r"splitk", r"split_?k",
    )),
    ("convolution", (
        r"conv2d", r"conv3d", r"conv1d", r"cudnn.*conv", r"winograd",
        r"implicit_?conv", r"depthwise",
    )),
    ("normalization", (
        r"layer_?norm", r"layernorm", r"rms_?norm", r"rmsnorm", r"batch_?norm",
        r"instance_?norm", r"group_?norm", r"groupnorm", r"adanorm", r"ada_?ln",
    )),
    ("softmax", (r"softmax", r"log_?softmax", r"online_?softmax")),
    ("activation", (
        r"gelu", r"silu", r"relu", r"sigmoid", r"\btanh\b", r"swiglu", r"geglu",
        r"activation", r"hardswish", r"mish",
    )),
    ("moe", (
        r"\bmoe\b", r"moe_", r"expert", r"top_?k_?gating", r"permute_?tokens",
        r"unpermute", r"sinkhorn", r"router",
    )),
    ("embedding", (r"embedding", r"^lookup", r"vocab_?parallel", r"rotary", r"\brope\b")),
    ("optimizer", (
        r"adam", r"lamb", r"\bsgd\b", r"momentum", r"clip_?grad", r"optim",
        r"weight_?update", r"multi_?tensor_?apply", r"multi_?tensor_",
        r"foreach_", r"_fused_adam",
    )),
    ("quantization", (
        r"quantize", r"dequant", r"cast_?transpose", r"fp8_?cast", r"scaled_?quant",
        r"amax", r"scale_?inv",
    )),
    ("reduction", (
        r"reduce_?kernel", r"\breduce\b", r"\bsum\b", r"\bmean\b", r"\bvar\b",
        r"\bstd\b", r"\bmax\b", r"\bmin\b", r"cub::", r"device_?reduce",
    )),
    ("memory_ops", (
        r"catarraybatchedcopy", r"direct_?copy", r"copy_?kernel", r"copy_?device",
        r"\bfill\b", r"memset", r"\bscatter\b", r"\bgather\b", r"index_?select",
        r"index_?put", r"\bpad\b", r"transpose", r"permute_?kernel", r"contiguous",
        # Layout conversions: pure overhead, and a signal that a channels-last /
        # NHWC boundary is being crossed mid-model.
        r"nhwctonchw", r"nchwtonhwc", r"nhwc_?to_?nchw", r"nchw_?to_?nhwc",
        r"layout_?transform", r"\bmemcpy\b",
    )),
    ("elementwise", (
        r"elementwise", r"vectorized_?elementwise", r"unrolled_?elementwise",
        r"pointwise", r"fused_?bias", r"eltwise", r"\badd_\b", r"\bmul_\b",
        r"triton_poi", r"triton_per",
    )),
    ("dropout", (r"dropout",)),
    ("loss", (r"cross_?entropy", r"nll_?loss", r"loss_?kernel", r"kl_?div")),
    ("sampling", (r"top_?p", r"top_?k_?sampling", r"multinomial", r"argmax_?sample")),
    ("vae_upsample", (r"upsample", r"interpolate", r"pixel_?shuffle", r"resize_?nearest")),
)

_COMPILED_CATEGORIES: Tuple[Tuple[str, Tuple[re.Pattern, ...]], ...] = tuple(
    (name, tuple(re.compile(p) for p in patterns)) for name, patterns in KERNEL_CATEGORIES
)

# Framework fingerprints, checked independently of category. Order matters:
# the most specific library wins, because e.g. a ThunderKittens attention kernel
# also matches the generic attention patterns.
_FRAMEWORK_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    # ThunderKittens is header-only, so the __global__ symbol is user-named. The
    # reliable fingerprints are the namespace in parameter types, the prototype
    # kernels, and the mangled forms. Note rt_bf/st_bf are `using` aliases and
    # never survive mangling - searching for them finds nothing.
    ("thunderkittens", re.compile(
        r"\bkittens::"
        r"|_ZN?7kittens"
        r"|N7kittens[0-9]*(?:2gl|2st|2rt|2sv|2rv|2tt)I"
        r"|\b(?:fwd_attend_ker|bwd_attend_ker|bwd_attend_prep_ker"
        r"|based_linear_attention|hedgehog_linear_attention_smd"
        r"|layernorm_tk|fftconv_tk)\b"
    )),
    # Triton-distributed emits plain Python function names; NVSHMEM device calls
    # are LLVM-linked into the kernel, so no nvshmem* symbols ever appear.
    ("triton_distributed", re.compile(
        r"\bkernel_(?:dispatch|combine|skipped)_token"
        r"|\bmega_kernel_(?:dispatch|moe)"
        r"|\bkernel_(?:consumer|producer|gemm_rs_producer|fused_ag_gemm|fused_gemm_allreduce)"
        r"|\bkernel_(?:pre|post)_attn_(?:qkv_pack_)?a2a"
        r"|\b(?:all_to_all_kernel|allreduce_(?:one|two)_shot|allreduce_double_tree"
        r"|reduce_scatter_ring_push|consumer_(?:ring_)?all_reduce)"
        r"|\b_forward_p(?:ull|ush)"
        r"|\bbarrier_all_intra_node"
        r"|\bmoe_(?:grouped|gather_(?:rs|ar)_grouped)_gemm"
        r"|\bkernel_(?:intra_rank_|inter_rank_)?gqa_fwd_batch_decode"
    )),
    ("triton", re.compile(r"^triton_|triton_poi_|triton_red_|triton_per_|triton_tem_")),
    ("cutlass", re.compile(r"cutlass|sm90_|sm100_|sm103_|nvjet|cute::|tcgen05")),
    ("cublas", re.compile(r"cublas|ampere_|turing_|volta_|^sm\d+_xmma")),
    ("cudnn", re.compile(r"cudnn")),
    ("flash_attention", re.compile(r"flash_?fwd|flash_?bwd|flash::|flash_?attn")),
    ("transformer_engine", re.compile(r"^te_|transformer_engine|cast_transpose|layernorm_geglu")),
    ("apex", re.compile(r"multi_?tensor_apply|multi_?tensor_adam")),
    ("nccl", re.compile(r"^nccl|ncclkernel|ncclDevKernel")),
    ("nvshmem", re.compile(r"nvshmem|rocshmem|mori_shmem")),
    ("aten", re.compile(r"at::native|aten::")),
)

# Kernels that fuse a whole model step into one persistent launch. When one of
# these dominates a trace, per-kernel attribution stops being meaningful and the
# analyzer has to say so rather than reporting "one kernel took 100% of time".
_MEGAKERNEL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"kittens::prototype::interpreter::kernel"),
    re.compile(r"_ZN7kittens9prototype11interpreter6kernel"),
    re.compile(r"^mega_kernel_"),
    re.compile(r"megakernel|mega_triton"),
)


def is_megakernel(kernel_name: str) -> bool:
    """Whether a kernel fuses many logical ops into one persistent launch."""
    low = str(kernel_name or "").lower()
    raw = str(kernel_name or "")
    return any(p.search(raw) or p.search(low) for p in _MEGAKERNEL_PATTERNS)

# torch.compile / Inductor kernel prefixes -> what they fused.
_TRITON_KIND = {
    "poi": "pointwise",
    "red": "reduction",
    "per": "persistent_reduction",
    "tem": "template",  # usually a matmul template
    "for": "foreach",
}


@dataclass(frozen=True)
class KernelInfo:
    """Structured view of a kernel name."""

    name: str
    category: str
    framework: str = ""
    tensor_cores: Optional[bool] = None
    triton_kind: str = ""
    fused_ops: Tuple[str, ...] = ()
    megakernel: bool = False

    @property
    def is_compiled(self) -> bool:
        """Whether this kernel came out of torch.compile / Inductor."""
        return self.framework == "triton"


def classify_kernel(kernel_name: str) -> KernelInfo:
    """Classify a kernel name into a category plus framework/tensor-core metadata."""
    raw = str(kernel_name or "")
    low = raw.lower()
    if not low:
        return KernelInfo(name=raw, category="unknown")

    framework = ""
    for fw, pattern in _FRAMEWORK_PATTERNS:
        # Mangled symbols (_ZN7kittens...) are case-sensitive, so test the raw
        # name as well as the lowercased one.
        if pattern.search(low) or pattern.search(raw):
            framework = fw
            break

    triton_kind = ""
    fused_ops: Tuple[str, ...] = ()
    if framework == "triton":
        match = re.match(r"triton_(poi|red|per|tem|for)_fused_(.*?)(?:_\d+)?$", low)
        if match:
            triton_kind = _TRITON_KIND.get(match.group(1), match.group(1))
            # TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 embeds the fused aten ops.
            fused_ops = tuple(tok for tok in match.group(2).split("_") if tok)
        else:
            kind = re.match(r"triton_(poi|red|per|tem|for)_", low)
            if kind:
                triton_kind = _TRITON_KIND.get(kind.group(1), kind.group(1))

    category = "other"
    for name, patterns in _COMPILED_CATEGORIES:
        if any(pattern.search(low) for pattern in patterns):
            category = name
            break

    # A Triton template kernel is a matmul even though the name says "triton".
    if category == "other" and triton_kind:
        category = "matmul" if triton_kind == "template" else "elementwise"

    return KernelInfo(
        name=raw,
        category=category,
        framework=framework,
        tensor_cores=uses_tensor_cores(raw),
        triton_kind=triton_kind,
        fused_ops=fused_ops,
        megakernel=is_megakernel(raw),
    )


def is_communication_kernel(kernel_name: str) -> bool:
    """Whether a kernel is an NCCL/RCCL collective.

    Uses the same ``nccl`` prefix test that NVIDIA's own nsys recipes apply
    (``nsys_recipe/lib/nccl.py``), widened to the ROCm spellings.
    """
    low = str(kernel_name or "").lower()
    return low.startswith(("nccl", "rccl", "hccl")) or "ncclkernel" in low or "ncclDevKernel".lower() in low


# ---------------------------------------------------------------------------
# GEMM name parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GemmShape:
    """Tile geometry and dtype recovered from a GEMM kernel name."""

    tile_m: Optional[int] = None
    tile_n: Optional[int] = None
    tile_k: Optional[int] = None
    dtype: str = ""
    layout: str = ""       # nn / nt / tn / tt
    stages: Optional[int] = None
    split_k: bool = False

    @property
    def tile_area(self) -> Optional[int]:
        if self.tile_m and self.tile_n:
            return self.tile_m * self.tile_n
        return None


# e.g. ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn
#      sm90_xmma_gemm_..._tilesize128x128x64_warpgroupsize1x1x1
# The leading guard rejects digits glued to a preceding number (so the "16816"
# in "s16816gemm" is not read as a tile) but must still admit the common
# "tilesize"/"tile"/"cta" word prefixes.
_TILE_RE = re.compile(
    r"(?:(?<=[^0-9a-z])|(?<=tilesize)|(?<=tile)|(?<=cta)|^)(\d{2,4})x(\d{2,4})(?:x(\d{2,4}))?(?![0-9a-z])"
)
_LAYOUT_RE = re.compile(r"_(nn|nt|tn|tt)(?:_|$)")
_STAGES_RE = re.compile(r"stages?_(?:\d+x)?(\d+)")
_DTYPE_TOKENS: Tuple[Tuple[str, str], ...] = (
    ("bf16", "bf16"), ("bfloat16", "bf16"),
    ("fp16", "fp16"), ("f16", "fp16"), ("half", "fp16"),
    ("tf32", "tf32"),
    ("fp8", "fp8"), ("e4m3", "fp8"), ("e5m2", "fp8"),
    ("int8", "int8"), ("i8", "int8"),
    ("fp64", "fp64"), ("double", "fp64"),
    ("fp32", "fp32"), ("f32", "fp32"), ("float", "fp32"),
)


def parse_gemm_kernel(kernel_name: str) -> Optional[GemmShape]:
    """Recover tile shape / dtype / layout from a cuBLAS or CUTLASS kernel name.

    Returns ``None`` for names that carry no tile information at all.  Tile
    dimensions are what wave- and tile-quantisation checks need; without them a
    grid size alone cannot tell you how much of the last wave is wasted.
    """
    low = str(kernel_name or "").lower()
    if not low:
        return None

    dtype = ""
    for token, canonical in _DTYPE_TOKENS:
        if token in low:
            dtype = canonical
            break

    tile_m = tile_n = tile_k = None
    # Prefer the largest plausible NxM pair: names often contain several
    # (e.g. "s16816gemm" instruction shape and "256x128" tile).
    best_area = -1
    for match in _TILE_RE.finditer(low):
        m_val, n_val = int(match.group(1)), int(match.group(2))
        k_val = int(match.group(3)) if match.group(3) else None
        # Instruction shapes (16x8x16) are not threadblock tiles.
        if m_val < 16 or n_val < 16:
            continue
        area = m_val * n_val
        if area > best_area:
            best_area, tile_m, tile_n, tile_k = area, m_val, n_val, k_val

    layout_match = _LAYOUT_RE.search(low)
    stages_match = _STAGES_RE.search(low)

    if tile_m is None and not dtype and not layout_match:
        return None

    return GemmShape(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        dtype=dtype,
        layout=layout_match.group(1) if layout_match else "",
        stages=int(stages_match.group(1)) if stages_match else None,
        split_k=("splitk" in low or "split_k" in low),
    )


# ---------------------------------------------------------------------------
# NCCL name parsing
# ---------------------------------------------------------------------------

# Bus-bandwidth correction factors (nccl-tests doc/PERFORMANCE.md).
# busbw = algbw * factor(n_ranks); algbw = message_bytes / time.
_BUSBW_FACTORS: Dict[str, str] = {
    "allreduce": "2*(n-1)/n",
    "allgather": "(n-1)/n",
    "reducescatter": "(n-1)/n",
    "alltoall": "(n-1)/n",
    "broadcast": "1",
    "reduce": "1",
    "sendrecv": "1",
}

_NCCL_COLLECTIVES: Tuple[str, ...] = (
    "allreduce", "allgather", "reducescatter", "alltoall",
    "broadcast", "reduce", "sendrecv", "scatter", "gather",
)
_NCCL_ALGOS: Tuple[str, ...] = ("ring", "tree", "nvls", "collnetdirect", "collnetchain", "collnet", "pat")

# NCCL's symmetric-memory kernels (ncclSymkDevKernel_*) encode the transport in
# the name: LDMC/STMC are multimem.ld_reduce / multimem.st, i.e. the reduction
# happened inside the NVSwitch. Seeing them is positive evidence that NVLink
# SHARP is engaged, not merely that NVLS was requested.
_NCCL_SYMMETRIC_RE = re.compile(r"ncclsymk", re.IGNORECASE)
# Underscore is a word character, so \b does not fire inside "LDMC_STMC".
_NCCL_MULTICAST_RE = re.compile(r"(?:^|[^a-z0-9])(?:ld|st)mc(?:[^a-z0-9]|$)|multimem", re.IGNORECASE)


def uses_nvls_multicast(kernel_name: str) -> bool:
    """Whether a collective kernel actually performs in-switch (NVLS) reduction.

    Distinct from ``algorithm == 'nvls'`` in the kernel name, which only says
    NCCL selected the algorithm. LDMC/STMC or a ``multimem`` token means the
    multicast instructions are really being issued.
    """
    return bool(_NCCL_MULTICAST_RE.search(str(kernel_name or "")))
_NCCL_PROTOS: Tuple[str, ...] = ("ll128", "ll", "simple")


@dataclass(frozen=True)
class NcclCollective:
    """Collective / dtype / algorithm / protocol recovered from an NCCL kernel name."""

    collective: str = ""
    reduction_op: str = ""
    dtype: str = ""
    algorithm: str = ""
    protocol: str = ""

    def busbw_factor(self, num_ranks: int) -> Optional[float]:
        """Multiplier converting algorithmic bandwidth into bus bandwidth.

        AllReduce moves each byte ``2(n-1)/n`` times around the ring, AllGather
        and ReduceScatter ``(n-1)/n``, point-to-point collectives once.
        """
        if num_ranks is None or num_ranks < 2:
            return None
        key = self.collective.lower()
        n = float(num_ranks)
        if key == "allreduce":
            return 2.0 * (n - 1.0) / n
        if key in ("allgather", "reducescatter", "alltoall"):
            return (n - 1.0) / n
        if key in ("broadcast", "reduce", "sendrecv", "scatter", "gather"):
            return 1.0
        return None

    def busbw_formula(self) -> str:
        """Human-readable form of the correction factor, for report footnotes."""
        return _BUSBW_FACTORS.get(self.collective.lower(), "")


def parse_nccl_kernel(kernel_name: str) -> Optional[NcclCollective]:
    """Parse ``ncclDevKernel_AllReduce_Sum_bf16_RING_LL128``-style names.

    Algorithm and protocol matter because they set what "good" looks like:
    LL trades ~50% of bandwidth for latency, LL128 keeps ~95%, and Simple is
    full bandwidth at ~6us of latency.
    """
    raw = str(kernel_name or "")
    if not raw or not is_communication_kernel(raw):
        return None
    # Drop the C++ argument list: "..._RING_LL128(ncclDevKernelArgsStorage<4096ul>)"
    # would otherwise hide the protocol behind a '(' and leak arg tokens into the
    # dtype/algorithm scans.
    low = raw.lower().split("(")[0].strip()

    collective = ""
    for candidate in _NCCL_COLLECTIVES:
        if candidate in low.replace("_", ""):
            collective = candidate
            break

    reduction_op = ""
    for candidate in ("prodmax", "premulsum", "sumpostdiv", "sum", "prod", "max", "min", "avg"):
        if candidate in low.replace("_", ""):
            reduction_op = candidate
            break

    dtype = ""
    for token, canonical in _DTYPE_TOKENS + (("f32", "fp32"), ("f16", "fp16"), ("u8", "uint8")):
        if token in low:
            dtype = canonical
            break

    algorithm = ""
    for candidate in _NCCL_ALGOS:
        if candidate in low.replace("_", ""):
            algorithm = candidate
            break

    protocol = ""
    for candidate in _NCCL_PROTOS:
        # Check as a delimited token so "ll" does not match inside "collnet".
        if re.search(rf"(?:^|_){candidate}(?:_|$)", low):
            protocol = candidate
            break

    if not any((collective, algorithm, protocol)):
        return NcclCollective()
    return NcclCollective(
        collective=collective,
        reduction_op=reduction_op,
        dtype=dtype,
        algorithm=algorithm,
        protocol=protocol,
    )


# ---------------------------------------------------------------------------
# Per-category speed-of-light expectations
# ---------------------------------------------------------------------------

# What a healthy kernel of each category should look like. ``dram_sol_pct`` and
# ``compute_sol_pct`` are the floors below which the kernel deserves a finding;
# ``bound`` records which roofline side it is *supposed* to sit on, so that a
# memory-bound-by-nature kernel is not flagged for low tensor-core usage.
CATEGORY_EXPECTATIONS: Dict[str, Dict[str, object]] = {
    "matmul": {
        "bound": "compute",
        "compute_sol_pct": 60.0,
        "expect_tensor_cores": True,
        "advice": "Large aligned GEMMs should reach 85-95% of cuBLAS; check tile/wave quantisation and dtype alignment.",
    },
    "attention": {
        "bound": "compute",
        "compute_sol_pct": 50.0,
        "expect_tensor_cores": True,
        "advice": "Prefill should be compute bound; decode (q_len=1) is memory bound by definition - grade it on KV-cache bytes/s, not TFLOPs.",
    },
    "convolution": {
        "bound": "compute",
        "compute_sol_pct": 50.0,
        "expect_tensor_cores": True,
        "advice": "Check NHWC/channels-last layout and channel counts divisible by 8 (fp16) / 4 (tf32).",
    },
    "elementwise": {
        "bound": "memory",
        "dram_sol_pct": 70.0,
        "expect_tensor_cores": False,
        "advice": "Elementwise kernels should saturate DRAM (80-95% SOL). Low SOL means too small, uncoalesced, or latency bound.",
    },
    "normalization": {
        "bound": "memory",
        "dram_sol_pct": 60.0,
        "expect_tensor_cores": False,
        "advice": "Normalisation is bandwidth bound; fuse with neighbouring pointwise ops to halve the traffic.",
    },
    "softmax": {
        "bound": "memory",
        "dram_sol_pct": 55.0,
        "expect_tensor_cores": False,
        "advice": "Softmax leans on the SFU (exp): on H100 that pipe is 3.9 TFLOPS vs 989 for matmul, so it rarely reaches DRAM peak.",
    },
    "activation": {
        "bound": "memory",
        "dram_sol_pct": 70.0,
        "expect_tensor_cores": False,
        "advice": "Should be fused into the producing GEMM or the neighbouring pointwise chain.",
    },
    "memory_ops": {
        "bound": "memory",
        "dram_sol_pct": 70.0,
        "expect_tensor_cores": False,
        "advice": "Copy/cat/transpose traffic is pure overhead - usually removable by fixing layouts or preallocating.",
    },
    "reduction": {
        "bound": "memory",
        "dram_sol_pct": 55.0,
        "expect_tensor_cores": False,
        "advice": "Reductions are bandwidth bound; check for enough parallelism across the reduced axis.",
    },
    "communication": {
        "bound": "network",
        "expect_tensor_cores": False,
        "advice": "Grade against bus bandwidth for the collective and interconnect, not against SM/DRAM speed-of-light.",
    },
    "optimizer": {
        "bound": "memory",
        "dram_sol_pct": 60.0,
        "expect_tensor_cores": False,
        "advice": "Use fused/foreach optimizers; per-parameter kernels turn the step into a launch storm.",
    },
    "quantization": {
        "bound": "memory",
        "dram_sol_pct": 55.0,
        "expect_tensor_cores": False,
        "advice": "Cast/transpose kernels should be fused into the producer or consumer GEMM.",
    },
    "embedding": {"bound": "memory", "dram_sol_pct": 40.0, "expect_tensor_cores": False,
                  "advice": "Gather-heavy and typically uncoalesced; judge on achieved bandwidth, not SOL."},
    "moe": {"bound": "mixed", "expect_tensor_cores": None,
            "advice": "Expert dispatch/combine is all-to-all bound; grouped GEMMs should still hit tensor cores."},
    "dropout": {"bound": "memory", "dram_sol_pct": 60.0, "expect_tensor_cores": False, "advice": ""},
    "loss": {"bound": "memory", "dram_sol_pct": 40.0, "expect_tensor_cores": False, "advice": ""},
    "vae_upsample": {"bound": "memory", "dram_sol_pct": 55.0, "expect_tensor_cores": False,
                     "advice": "VAE decode is the usual OOM cliff in video pipelines - tile spatially and slice temporally."},
    "sampling": {"bound": "latency", "expect_tensor_cores": False, "advice": ""},
    "other": {"bound": "unknown", "expect_tensor_cores": None, "advice": ""},
    "unknown": {"bound": "unknown", "expect_tensor_cores": None, "advice": ""},
}


def summarize_categories(kernel_names: Sequence[str]) -> Dict[str, int]:
    """Count kernels per category - a quick shape-of-workload readout."""
    counts: Dict[str, int] = {}
    for name in kernel_names:
        info = classify_kernel(name)
        counts[info.category] = counts.get(info.category, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
