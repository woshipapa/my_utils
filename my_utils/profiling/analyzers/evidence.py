# SPDX-License-Identifier: Apache-2.0
"""Evidence fusion: never conclude from a kernel name alone.

A kernel symbol is the weakest evidence a profiler has, and it does not merely
under-inform - it actively misleads. Three verified cases drove this module:

* **NCCL names lie about the algorithm.** ``generate.py::best_kernel()`` collapses
  ~670 device functions onto ~40 launch stubs with the algo/proto tokens
  hard-coded to ``RING``/``TREE`` + ``LL``; the real algorithm is dispatched at
  runtime via ``ncclDevFuncTable``. A ReduceScatter running NVLS+Simple appears
  as ``ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL``.
* **CuTe DSL names carry nothing.** The default is ``kernel_kernel_<args>_0``,
  and ``mangle_name`` skips every IR value, so tensor shapes and dtypes are
  absent. FlashAttention-4 ships this way.
* **A CUTLASS symbol does not mean the user wrote CUTLASS.** cuBLASLt embeds
  CUTLASS-generated kernels, so ``cutlass3x_sm90_*`` may be plain
  ``torch.matmul``. Attribution needs the call stack, not the symbol.

So every claim carries where it came from, and conflicting claims are resolved
by provenance rather than by whichever check ran last. The most valuable output
is not the winning claim but :attr:`FusedAttribute.contradictions` - a
disagreement between the name and a hardware counter is a finding in itself,
because it means the kernel is not doing what it is named for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Provenance",
    "PROVENANCE_RANK",
    "Claim",
    "FusedAttribute",
    "fuse_claims",
    "claims_from_kernel_name",
    "claims_from_launch_config",
    "claims_from_counters",
    "claims_from_nvtx",
    "attribute_kernel",
]


class Provenance:
    """Where a claim came from, ordered by how much it can be trusted.

    Hardware counters win because they record what the silicon actually did.
    NVTX payloads and source analysis come next: they are authored by the
    framework that issued the work, so they describe intent accurately, but they
    can be stale or absent. The CUDA API call and launch config are structural
    facts about the launch but say little about the kernel body. The symbol name
    is last, for the reasons in the module docstring.
    """

    HW_COUNTER = "hw_counter"
    NVTX = "nvtx"
    SOURCE = "source"
    CUDA_API = "cuda_api"
    LAUNCH_CONFIG = "launch_config"
    KERNEL_NAME = "kernel_name"


PROVENANCE_RANK: Dict[str, int] = {
    Provenance.HW_COUNTER: 100,
    Provenance.NVTX: 80,
    Provenance.SOURCE: 75,
    Provenance.CUDA_API: 60,
    Provenance.LAUNCH_CONFIG: 50,
    Provenance.KERNEL_NAME: 20,
}


@dataclass(frozen=True)
class Claim:
    """One assertion about a kernel, tagged with its evidence."""

    attribute: str
    value: Any
    provenance: str
    detail: str = ""
    # Set when the evidence is structurally incapable of being authoritative -
    # an NCCL launch stub, a truncated symbol, a generic DSL name. Advisory
    # claims never win a fusion and never raise a contradiction on their own.
    advisory: bool = False

    @property
    def rank(self) -> int:
        base = PROVENANCE_RANK.get(self.provenance, 0)
        return base // 4 if self.advisory else base


@dataclass
class FusedAttribute:
    """The resolved value for one attribute, plus the evidence behind it."""

    attribute: str
    value: Any = None
    provenance: str = ""
    confidence: str = "low"  # low | medium | high
    supporting: List[Claim] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attribute": self.attribute,
            "value": self.value,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "evidence": [
                {"value": c.value, "provenance": c.provenance, "detail": c.detail}
                for c in self.supporting
            ],
            "contradictions": list(self.contradictions),
        }


def _comparable(value: Any) -> Any:
    """Normalise a claim value so equal-but-differently-spelled claims agree."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 6)
    return value


def fuse_claims(claims: Iterable[Claim]) -> Dict[str, FusedAttribute]:
    """Resolve claims per attribute, highest-provenance first.

    Two claims that disagree produce a contradiction *only* when the loser is
    non-advisory: an NCCL launch stub asserting ``RING`` is not evidence against
    a counter-derived ``NVLS``, it is simply silent. A real disagreement - a name
    saying ``gemm`` while the tensor pipe reads zero - is recorded verbatim,
    because that gap is usually the finding.
    """
    grouped: Dict[str, List[Claim]] = {}
    for claim in claims:
        if claim is None:
            continue
        grouped.setdefault(claim.attribute, []).append(claim)

    fused: Dict[str, FusedAttribute] = {}
    for attribute, group in grouped.items():
        ordered = sorted(group, key=lambda c: c.rank, reverse=True)
        winner = ordered[0]
        result = FusedAttribute(
            attribute=attribute,
            value=winner.value,
            provenance=winner.provenance,
            supporting=ordered,
        )

        agreeing = [
            c for c in ordered if _comparable(c.value) == _comparable(winner.value)
        ]
        dissenting = [
            c
            for c in ordered
            if _comparable(c.value) != _comparable(winner.value) and not c.advisory
        ]

        for loser in dissenting:
            result.contradictions.append(
                f"{loser.provenance} says {loser.value!r}"
                f"{' (' + loser.detail + ')' if loser.detail else ''}, "
                f"but {winner.provenance} says {winner.value!r}"
            )

        # Confidence reflects both the strength of the winning evidence and
        # whether anything independent corroborates it. A lone kernel-name claim
        # never rises above low.
        distinct_sources = {c.provenance for c in agreeing}
        if winner.advisory:
            result.confidence = "low"
        elif (
            winner.rank >= PROVENANCE_RANK[Provenance.NVTX]
            and len(distinct_sources) > 1
        ):
            result.confidence = "high"
        elif winner.rank >= PROVENANCE_RANK[Provenance.NVTX]:
            result.confidence = "medium" if result.contradictions else "high"
        elif len(distinct_sources) > 1 and not result.contradictions:
            result.confidence = "medium"
        else:
            result.confidence = "low"

        fused[attribute] = result
    return fused


# ---------------------------------------------------------------------------
# Claim extractors, weakest source first
# ---------------------------------------------------------------------------


def claims_from_kernel_name(kernel_name: str) -> List[Claim]:
    """Claims recoverable from the symbol, each marked with how far to trust it."""
    from ..sources.kernel_taxonomy import (
        classify_kernel,
        parse_gemm_kernel,
        parse_nccl_kernel,
    )

    raw = str(kernel_name or "")
    if not raw:
        return []
    out: List[Claim] = []
    info = classify_kernel(raw)

    # Truncation is common: nsys clips long CUTLASS symbols, and CuTe DSL caps
    # generated names at 180 chars. A clipped name may have lost the very tokens
    # a classifier keyed on, so everything from it drops to advisory.
    truncated = raw.endswith(("...", "…")) or (
        len(raw) >= 180 and "<" in raw and ">" not in raw[-40:]
    )

    out.append(
        Claim(
            "category",
            info.category,
            Provenance.KERNEL_NAME,
            detail=f"symbol {raw[:60]!r}",
            advisory=truncated,
        )
    )
    if info.framework:
        # A cutlass3x_* symbol may have been emitted by cuBLASLt rather than
        # written by the user, so the framework claim is never authoritative
        # about who owns the kernel - only about what generated the code.
        out.append(
            Claim(
                "framework",
                info.framework,
                Provenance.KERNEL_NAME,
                detail="symbol pattern; cuBLASLt embeds CUTLASS kernels",
                advisory=True,
            )
        )

    gemm = parse_gemm_kernel(raw)
    if gemm is not None and getattr(gemm, "tile_m", None):
        out.append(
            Claim(
                "gemm_tile",
                (gemm.tile_m, gemm.tile_n, gemm.tile_k),
                Provenance.KERNEL_NAME,
                detail="tile tokens in symbol",
                advisory=truncated,
            )
        )

    nccl = parse_nccl_kernel(raw)
    if nccl is not None and nccl.collective:
        out.append(
            Claim(
                "collective",
                nccl.collective,
                Provenance.KERNEL_NAME,
                detail="NCCL symbol",
                advisory=truncated,
            )
        )
        if nccl.algorithm:
            out.append(
                Claim(
                    "nccl_algorithm",
                    nccl.algorithm,
                    Provenance.KERNEL_NAME,
                    detail=nccl.algorithm_caveat() or "NCCL symbol",
                    advisory=not nccl.algorithm_is_authoritative,
                )
            )
        if nccl.protocol:
            out.append(
                Claim(
                    "nccl_protocol",
                    nccl.protocol,
                    Provenance.KERNEL_NAME,
                    detail=nccl.algorithm_caveat() or "NCCL symbol",
                    advisory=not nccl.algorithm_is_authoritative,
                )
            )

    # An unlabelled CuTe DSL kernel carries no shape or dtype at all: mangle_name
    # skips every IR value. Say so explicitly rather than reporting "unknown".
    low = raw.lower()
    if low.startswith("kernel_kernel_") or (
        info.framework == "cute_dsl" and "set_name_prefix" not in low
    ):
        out.append(
            Claim(
                "name_is_informative",
                False,
                Provenance.KERNEL_NAME,
                detail=(
                    "CuTe DSL default naming: tensor args are IR values and are "
                    "skipped by mangle_name, so no shape/dtype is in the symbol. "
                    "Call set_name_prefix() to make traces attributable."
                ),
            )
        )
    return out


def claims_from_launch_config(
    *,
    grid_size: Optional[int] = None,
    block_size: Optional[int] = None,
    shared_mem_bytes: Optional[int] = None,
    registers_per_thread: Optional[int] = None,
    launch_api: str = "",
) -> List[Claim]:
    """Structural facts about the launch itself.

    ``launch_api`` discriminates a JIT-loaded module from a statically linked
    one: CuTe DSL and Triton go through the driver API (``cuLaunchKernel``)
    because they load a cubin at runtime, while C++ CUTLASS compiled into the
    binary launches via the runtime API (``cudaLaunchKernel``).
    """
    out: List[Claim] = []
    if launch_api:
        api = launch_api.strip()
        if api.startswith("cuLaunch"):
            out.append(
                Claim(
                    "jit_compiled",
                    True,
                    Provenance.CUDA_API,
                    detail=f"{api}: kernel came from a runtime-loaded module",
                )
            )
        elif api.startswith("cudaLaunch"):
            out.append(
                Claim(
                    "jit_compiled",
                    False,
                    Provenance.CUDA_API,
                    detail=f"{api}: statically linked launch",
                )
            )
    if block_size:
        out.append(Claim("block_size", int(block_size), Provenance.LAUNCH_CONFIG))
    if grid_size:
        out.append(Claim("grid_size", int(grid_size), Provenance.LAUNCH_CONFIG))
    if shared_mem_bytes is not None and shared_mem_bytes > 0:
        # Past ~100 KB/block the kernel must have opted into the Hopper/Blackwell
        # dynamic shared-memory carve-out, which effectively means a TMA or
        # warp-specialized pipeline rather than a naive tiled loop.
        out.append(
            Claim("shared_mem_bytes", int(shared_mem_bytes), Provenance.LAUNCH_CONFIG)
        )
        if shared_mem_bytes > 100 * 1024:
            out.append(
                Claim(
                    "uses_smem_carveout",
                    True,
                    Provenance.LAUNCH_CONFIG,
                    detail=f"{shared_mem_bytes / 1024:.0f} KB/block exceeds the "
                    "default 48 KB limit, so opt-in carve-out was requested",
                )
            )
    if registers_per_thread:
        out.append(
            Claim(
                "registers_per_thread",
                int(registers_per_thread),
                Provenance.LAUNCH_CONFIG,
                detail="static allocation; does not reflect setmaxnreg redistribution",
            )
        )
    return out


def claims_from_counters(metrics: Any) -> List[Claim]:
    """Claims from hardware counters - what the kernel actually executed.

    ``metrics`` is a mapping of ncu metric name to value, or anything exposing a
    catalog-key ``get()`` (a :class:`~..ncu.ncu_diagnostics.MetricView`).
    """
    if metrics is None:
        return []
    if hasattr(metrics, "get") and hasattr(metrics, "raw"):
        view = metrics
    else:
        from ..ncu.ncu_diagnostics import MetricView

        view = MetricView(metrics)

    out: List[Claim] = []

    tensor = view.get("pipe_tensor_util")
    if tensor is not None:
        used = tensor > 1.0
        out.append(
            Claim(
                "uses_tensor_cores",
                used,
                Provenance.HW_COUNTER,
                detail=f"tensor pipe at {tensor:.1f}% of peak",
            )
        )

    tma = view.get("pipe_tma_util")
    if tma is not None:
        out.append(
            Claim(
                "uses_tma",
                tma > 0.0,
                Provenance.HW_COUNTER,
                detail=f"TMA pipe at {tma:.2f}% of peak",
            )
        )

    fp64 = view.get("pipe_fp64_util")
    if fp64 is not None and fp64 > 1.0:
        out.append(
            Claim(
                "uses_fp64",
                True,
                Provenance.HW_COUNTER,
                detail=f"FP64 pipe at {fp64:.1f}% of peak",
            )
        )

    # Cross-checking the launch config against the counters catches a green
    # context: the grid looks small against the full device but fills the
    # partition the kernel was actually given.
    green = view.raw("launch__green_context_id")
    if green is not None:
        out.append(
            Claim(
                "green_context",
                True,
                Provenance.HW_COUNTER,
                detail="kernel ran in a green context; SOL peaks and wave counts "
                "are relative to the partition, not the whole device",
            )
        )

    passes = view.raw("profiler__replayer_passes")
    if passes is not None and passes > 1:
        out.append(
            Claim(
                "replay_passes",
                int(passes),
                Provenance.HW_COUNTER,
                detail=f"kernel was replayed {int(passes)} times; metrics from "
                "different passes are not simultaneous",
            )
        )
    return out


def claims_from_nvtx(
    ranges: Sequence[Any] = (), payloads: Optional[Dict[str, Any]] = None
) -> List[Claim]:
    """Claims from framework annotations.

    NCCL's own NVTX payloads carry the fields its kernel names lack - the
    communicator hash, per-message size, and reduction op - so where a name is a
    launch stub this is the authoritative source. Requires NCCL >= 2.25 for the
    communicator hash, and NCCL_NVTX_DISABLE unset.
    """
    out: List[Claim] = []
    for rng in ranges or ():
        name = str(getattr(rng, "name", rng) or "").strip()
        if not name:
            continue
        if name.startswith("nccl"):
            out.append(
                Claim(
                    "collective",
                    name[4:].lower(),
                    Provenance.NVTX,
                    detail=f"NCCL NVTX range {name!r}",
                )
            )
    for key, value in (payloads or {}).items():
        label = str(key).strip()
        if not label:
            continue
        if label == "Collective name":
            out.append(
                Claim(
                    "collective",
                    str(value).lower(),
                    Provenance.NVTX,
                    detail="PyTorch/Kineto collective metadata",
                )
            )
        elif label in ("Message size [bytes]", "In msg nelems"):
            out.append(
                Claim(
                    "message_bytes",
                    value,
                    Provenance.NVTX,
                    detail="NVTX payload: size of ONE message, not aggregate bytes",
                )
            )
        elif label in ("Group size", "Process Group Ranks"):
            out.append(
                Claim("group_size", value, Provenance.NVTX, detail="NVTX payload")
            )
        elif label == "Reduction operation":
            out.append(
                Claim(
                    "reduction_op",
                    str(value).lower(),
                    Provenance.NVTX,
                    detail="NVTX payload",
                )
            )
    return out


def attribute_kernel(
    kernel_name: str,
    *,
    metrics: Any = None,
    nvtx_ranges: Sequence[Any] = (),
    nvtx_payloads: Optional[Dict[str, Any]] = None,
    **launch: Any,
) -> Tuple[Dict[str, FusedAttribute], List[str]]:
    """Fuse every available source for one kernel.

    Returns the resolved attributes and a list of human-readable warnings, which
    is where the interesting output usually is: a name/counter disagreement means
    the kernel is not doing what its name advertises.
    """
    claims: List[Claim] = []
    claims.extend(claims_from_kernel_name(kernel_name))
    claims.extend(claims_from_launch_config(**launch))
    claims.extend(claims_from_counters(metrics))
    claims.extend(claims_from_nvtx(nvtx_ranges, nvtx_payloads))

    fused = fuse_claims(claims)
    warnings: List[str] = []
    for attribute in sorted(fused):
        warnings.extend(fused[attribute].contradictions)

    # The specific mismatch worth calling out by name: a kernel that advertises
    # matmul but never lit the tensor pipe.
    category = fused.get("category")
    tensor = fused.get("uses_tensor_cores")
    if (
        category is not None
        and str(category.value) == "matmul"
        and tensor is not None
        and tensor.value is False
    ):
        warnings.append(
            "Kernel name says matmul but the tensor pipe never activated - either it is "
            "an FP32/SIMT fallback, the dtype is unsupported on this arch, or the shapes "
            "are too small to tile. This is a finding, not a naming quirk."
        )

    informative = fused.get("name_is_informative")
    if informative is not None and informative.value is False:
        warnings.append(
            "Kernel name carries no shape or dtype information; every attribute below "
            "must come from counters, NVTX, or source analysis."
        )
    return fused, warnings
