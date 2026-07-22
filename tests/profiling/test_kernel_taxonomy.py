"""Tests for my_utils.profiling.sources.kernel_taxonomy."""

from __future__ import annotations


import pytest


from _synthetic_loader import kernel_taxonomy


@pytest.mark.parametrize(
    "name,category",
    [
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL128", "communication"),
        ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", "matmul"),
        ("void flash_fwd_kernel<Flash_fwd_kernel_traits<128,128,128,4>>", "attention"),
        (
            "void at::native::vectorized_elementwise_kernel<4, CUDAFunctor_add<float>>",
            "elementwise",
        ),
        (
            "void at::native::(anonymous namespace)::CatArrayBatchedCopy<float>",
            "memory_ops",
        ),
        (
            "void multi_tensor_apply_kernel<TensorListMetadata<4>, AdamFunctor<float>>",
            "optimizer",
        ),
        ("triton_red_fused_native_layer_norm_7", "normalization"),
        ("void moe_permute_topk_kernel<float>", "moe"),
        ("void te_cast_transpose_kernel<float>", "quantization"),
        ("void upsample_nearest3d_out_frame<float>", "vae_upsample"),
        ("void cudnn::ops::nhwcToNchwKernel<float>", "memory_ops"),
    ],
)
def test_kernel_categories(name, category):
    assert kernel_taxonomy.classify_kernel(name).category == category


@pytest.mark.parametrize(
    "name,framework",
    [
        ("void fwd_attend_ker<128, true>(fwd_globals<128>)", "thunderkittens"),
        # Mangled symbols are case-sensitive and must still be recognised.
        (
            "_ZN7kittens9prototype3lcf6kernelI15matmul_templateILi2EEEEv",
            "thunderkittens",
        ),
        ("void kernel_dispatch_token(int)", "triton_distributed"),
        ("void moe_grouped_gemm_persistent_tma_kernel(int)", "triton_distributed"),
        ("triton_poi_fused_add_mul_silu_23", "triton"),
        ("cutlass3x_sm90_tensorop_s64x128x16gemm_bf16", "cutlass"),
        ("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn", "cublas"),
        ("ncclDevKernel_AllGather_NVLS_Simple", "nccl"),
    ],
)
def test_framework_detection(name, framework):
    assert kernel_taxonomy.classify_kernel(name).framework == framework


def test_megakernel_detection():
    assert kernel_taxonomy.is_megakernel(
        "void kittens::prototype::interpreter::kernel<config>(g)"
    )
    assert kernel_taxonomy.is_megakernel("mega_kernel_dispatch_token_moe_grouped_gemm")
    assert not kernel_taxonomy.is_megakernel("ampere_bf16_s16816gemm_bf16_256x128_nn")


def test_tensor_core_detection_distinguishes_unknown_from_absent():
    # Positive evidence.
    assert (
        kernel_taxonomy.uses_tensor_cores("ampere_bf16_s16816gemm_bf16_256x128_nn")
        is True
    )
    # Negative evidence: an explicit SIMT/sgemm path.
    assert kernel_taxonomy.uses_tensor_cores("volta_sgemm_128x64_nn") is False
    # No evidence either way must stay None, not False - "no tensor cores" is
    # only actionable when the name actually says so.
    assert kernel_taxonomy.uses_tensor_cores("void my_custom_kernel(float*)") is None


@pytest.mark.parametrize(
    "name,tile_m,tile_n",
    [
        ("ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn", 256, 128),
        ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", 128, 128),
        (
            "cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_128x128x64_2x1x1",
            128,
            128,
        ),
        ("volta_sgemm_128x64_nn", 128, 64),
        ("ampere_h16816gemm_256x64_ldg8_stages_32x6_tn", 256, 64),
    ],
)
def test_gemm_tile_parsing(name, tile_m, tile_n):
    """The instruction shape (s16816) must not be mistaken for the tile shape."""
    shape = kernel_taxonomy.parse_gemm_kernel(name)
    assert shape is not None
    assert (shape.tile_m, shape.tile_n) == (tile_m, tile_n)


def test_gemm_parsing_extracts_dtype_and_layout():
    shape = kernel_taxonomy.parse_gemm_kernel(
        "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn"
    )
    assert shape.dtype == "bf16"
    assert shape.layout == "nn"
    assert shape.stages == 3


@pytest.mark.parametrize(
    "name,collective,algorithm,protocol",
    [
        # The C++ argument list must not hide the protocol behind a '('.
        (
            "ncclDevKernel_AllReduce_Sum_bf16_RING_LL128(ncclDevKernelArgsStorage<4096ul>)",
            "allreduce",
            "ring",
            "ll128",
        ),
        (
            "ncclDevKernel_ReduceScatter_Sum_f32_TREE_SIMPLE",
            "reducescatter",
            "tree",
            "simple",
        ),
        ("ncclDevKernel_AllGather_NVLS_Simple", "allgather", "nvls", "simple"),
        # PAT: AllGather/ReduceScatter only, added in NCCL 2.23.4.
        ("ncclDevKernel_AllGather_PAT_SIMPLE", "allgather", "pat", "simple"),
    ],
)
def test_nccl_parsing(name, collective, algorithm, protocol):
    parsed = kernel_taxonomy.parse_nccl_kernel(name)
    assert parsed is not None
    assert parsed.collective == collective
    assert parsed.algorithm == algorithm
    assert parsed.protocol == protocol


@pytest.mark.parametrize(
    "collective,ranks,factor",
    [
        # nccl-tests bus-bandwidth correction factors.
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL128", 8, 2 * 7 / 8),
        ("ncclDevKernel_AllGather_NVLS_Simple", 8, 7 / 8),
        ("ncclDevKernel_ReduceScatter_Sum_f32_RING_SIMPLE", 4, 3 / 4),
        ("ncclDevKernel_Broadcast_RING_LL", 8, 1.0),
    ],
)
def test_nccl_busbw_factors(collective, ranks, factor):
    parsed = kernel_taxonomy.parse_nccl_kernel(collective)
    assert parsed.busbw_factor(ranks) == pytest.approx(factor)


def test_nccl_busbw_factor_needs_at_least_two_ranks():
    parsed = kernel_taxonomy.parse_nccl_kernel(
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL128"
    )
    assert parsed.busbw_factor(1) is None


def test_non_nccl_kernel_is_not_parsed_as_a_collective():
    assert kernel_taxonomy.parse_nccl_kernel("ampere_sgemm_128x64_nn") is None


@pytest.mark.parametrize(
    "name,expected",
    [
        # LDMC/STMC are multimem.ld_reduce / multimem.st: the NVSwitch really did
        # the reduction. Underscore is a word character, so \b does not fire here.
        ("ncclSymkDevKernel_AllReduce_LDMC_STMC_bf16", True),
        ("ncclSymkDevKernel_AllGather_STMC", True),
        ("allreduce_two_shot_multimem_intra_node_kernel", True),
        # Naming the NVLS *algorithm* is not evidence the multicast path ran.
        ("ncclDevKernel_AllReduce_Sum_bf16_NVLS_SIMPLE", False),
        ("allreduce_one_shot_push_intra_node_kernel", False),
        ("ampere_sgemm_128x64_nn", False),
    ],
)
def test_nvls_multicast_detection(name, expected):
    assert kernel_taxonomy.uses_nvls_multicast(name) is expected


class TestCutlassSymbolParsing:
    """The template arguments ARE the configuration, not a label someone chose."""

    REAL = (
        "void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<"
        "cute::tuple<int,int,int>, cutlass::gemm::collective::CollectiveMma<"
        "cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<9, "
        "cute::tuple<cute::C<1>,cute::C<1>,cute::C<1> >, "
        "cutlass::gemm::KernelTmaWarpSpecializedPingpong>, "
        "cute::tuple<cute::C<128>,cute::C<64>,cute::C<64> >, cutlass::bfloat16_t, "
        "cute::TiledMMA<cute::MMA_Atom<cute::SM90::GMMA::MMA_64x64x16_F32BF16BF16_SS> >, "
        "cute::SM90_TMA_LOAD, cute::ComposedLayout<cute::Swizzle<3,4,3> > > > >"
    )

    def test_recovers_configuration(self):
        cfg = kernel_taxonomy.parse_cutlass_symbol(self.REAL)
        assert cfg.arch == "sm90"
        assert cfg.stages == 9
        assert cfg.cluster == (1, 1, 1)
        assert cfg.tile == (128, 64, 64)
        assert cfg.mma_shape == (64, 64, 16)
        assert cfg.operand_source == "SS"
        assert cfg.schedule == "pingpong"
        assert cfg.copy_atom == "SM90_TMA_LOAD"
        assert cfg.uses_tma() is True
        assert cfg.swizzle == (3, 4, 3)

    def test_ignores_non_cutlass(self):
        assert kernel_taxonomy.parse_cutlass_symbol("ampere_sgemm_128x64_nn") is None

    def test_truncation_is_reported_not_hidden(self):
        cfg = kernel_taxonomy.parse_cutlass_symbol(
            "void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int"
        )
        assert cfg.truncated is True
        assert any("not visible" in o for o in cfg.observations())

    def test_unswizzled_layout_is_flagged(self):
        cfg = kernel_taxonomy.parse_cutlass_symbol(
            "void cutlass::device_kernel<X<cute::Swizzle<0,4,3> > >"
        )
        assert any("unswizzled" in o for o in cfg.observations())
