"""Tests for the GPU-spec tables, kernel taxonomy, ncu rule engine and triage tree.

These modules are pure analysis code with no CUDA or torch dependency, so the
tests load them directly from their file paths.  Going through the ``my_utils``
package would drag in ``my_utils.core.utils``, which imports torch, and that
would make these tests unrunnable exactly where they are most useful: a CI box
with no GPU stack installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
_PKG = "_prof"


def _package(name: str) -> types.ModuleType:
    """Create (once) a synthetic package so relative imports resolve."""
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__path__ = []          # marks it as a package
    sys.modules[name] = module
    if "." in name:
        parent, _, child = name.rpartition(".")
        setattr(_package(parent), child, module)
    return module


def _load(dotted: str, relative_path: str):
    """Import one profiling module under a synthetic package root.

    The modules use relative imports (``from .metric_catalog import ...``), so
    they cannot be loaded as standalone files - they need a parent package. We
    build that package tree by hand rather than importing ``my_utils``, whose
    ``__init__`` pulls in torch.
    """
    full = f"{_PKG}.{dotted}"
    if full in sys.modules:
        return sys.modules[full]
    parent, _, leaf = full.rpartition(".")
    _package(parent)
    spec = importlib.util.spec_from_file_location(full, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    setattr(sys.modules[parent], leaf, module)
    return module


gpu_specs = _load("hardware.gpu_specs", "hardware/gpu_specs.py")
kernel_taxonomy = _load("sources.kernel_taxonomy", "sources/kernel_taxonomy.py")
metric_catalog = _load("ncu.metric_catalog", "ncu/metric_catalog.py")
triage = _load("analyzers.triage", "analyzers/triage.py")
ncu_diagnostics = _load("ncu.ncu_diagnostics", "ncu/ncu_diagnostics.py")


# ---------------------------------------------------------------------------
# GPU specs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "probe,expected_name,expected_sms",
    [
        ("NVIDIA H100 80GB HBM3", "H100 SXM5", 132),
        ("NVIDIA A100-SXM4-80GB", "A100 SXM 80GB", 108),
        ("NVIDIA H800", "H800 SXM", 132),
        ("NVIDIA H200", "H200 SXM", 132),
        ("NVIDIA L40S", "L40S", 142),
        ("NVIDIA A800-SXM4-80GB", "A800", 108),
        ("Tesla V100-SXM2-32GB", "V100 SXM2", 80),
    ],
)
def test_gpu_lookup_matches_most_specific_sku(probe, expected_name, expected_sms):
    spec = gpu_specs.lookup_gpu_spec(probe)
    assert spec is not None, f"no spec matched {probe!r}"
    assert spec.name == expected_name
    assert spec.sm_count == expected_sms


def test_unknown_gpu_returns_none():
    assert gpu_specs.lookup_gpu_spec("Some Future GPU 9000") is None
    assert gpu_specs.lookup_gpu_spec("") is None


@pytest.mark.parametrize(
    "gpu,dtype,expected_ridge",
    [
        # Published ridge points (dense peak / HBM bandwidth), rounded.
        ("H100 SXM5", "bf16", 295),
        ("H100 SXM5", "fp8", 591),
        ("A100 SXM 80GB", "bf16", 153),
        ("H200 SXM", "bf16", 206),
        ("B200", "bf16", 281),
    ],
)
def test_ridge_points_match_published_values(gpu, dtype, expected_ridge):
    spec = gpu_specs.lookup_gpu_spec(gpu)
    assert round(spec.ridge_point(dtype)) == expected_ridge


def test_sparsity_doubles_tensor_peaks_but_not_vector_peaks():
    h100 = gpu_specs.lookup_gpu_spec("H100 SXM5")
    assert h100.peak_tflops("bf16", sparse=True) == pytest.approx(2 * h100.peak_tflops("bf16"))
    # Structured sparsity does not apply to the FP32/FP64 vector pipes.
    assert h100.peak_tflops("fp32", sparse=True) is None


def test_unsupported_dtype_reports_none_rather_than_zero():
    v100 = gpu_specs.lookup_gpu_spec("V100 SXM2")
    assert v100.peak_tflops("bf16") is None  # Volta has no BF16
    assert v100.peak_tflops("fp16") == pytest.approx(125.0)


def test_effective_bandwidth_prefers_measured_over_spec():
    v100 = gpu_specs.lookup_gpu_spec("V100 SXM2")
    # Citadel measured 750 GB/s of the 900 GB/s spec.
    assert v100.effective_hbm_gbps() == pytest.approx(750.0)
    l4 = gpu_specs.lookup_gpu_spec("L4")
    assert l4.effective_hbm_gbps() == pytest.approx(l4.hbm_bandwidth_gbps * 0.85)


def test_attainable_tflops_follows_the_roofline():
    h100 = gpu_specs.lookup_gpu_spec("H100 SXM5")
    # Well left of the ridge: bandwidth limited.
    assert h100.attainable_tflops(50, "bf16") == pytest.approx(50 * 3350e9 / 1e12)
    # Well right of the ridge: clipped to the compute peak.
    assert h100.attainable_tflops(5000, "bf16") == pytest.approx(989.4)


def test_matmul_vs_vector_ratio_reproduces_the_a100_cost_model():
    a100 = gpu_specs.lookup_gpu_spec("A100 SXM 80GB")
    assert a100.matmul_vs_vector_ratio() == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# Kernel taxonomy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,category",
    [
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL128", "communication"),
        ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", "matmul"),
        ("void flash_fwd_kernel<Flash_fwd_kernel_traits<128,128,128,4>>", "attention"),
        ("void at::native::vectorized_elementwise_kernel<4, CUDAFunctor_add<float>>", "elementwise"),
        ("void at::native::(anonymous namespace)::CatArrayBatchedCopy<float>", "memory_ops"),
        ("void multi_tensor_apply_kernel<TensorListMetadata<4>, AdamFunctor<float>>", "optimizer"),
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
        ("_ZN7kittens9prototype3lcf6kernelI15matmul_templateILi2EEEEv", "thunderkittens"),
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
    assert kernel_taxonomy.is_megakernel("void kittens::prototype::interpreter::kernel<config>(g)")
    assert kernel_taxonomy.is_megakernel("mega_kernel_dispatch_token_moe_grouped_gemm")
    assert not kernel_taxonomy.is_megakernel("ampere_bf16_s16816gemm_bf16_256x128_nn")


def test_tensor_core_detection_distinguishes_unknown_from_absent():
    # Positive evidence.
    assert kernel_taxonomy.uses_tensor_cores("ampere_bf16_s16816gemm_bf16_256x128_nn") is True
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
        ("cutlass3x_sm90_tensorop_s64x128x16gemm_bf16_bf16_f32_void_f32_128x128x64_2x1x1", 128, 128),
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
        "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn")
    assert shape.dtype == "bf16"
    assert shape.layout == "nn"
    assert shape.stages == 3


@pytest.mark.parametrize(
    "name,collective,algorithm,protocol",
    [
        # The C++ argument list must not hide the protocol behind a '('.
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL128(ncclDevKernelArgsStorage<4096ul>)",
         "allreduce", "ring", "ll128"),
        ("ncclDevKernel_ReduceScatter_Sum_f32_TREE_SIMPLE", "reducescatter", "tree", "simple"),
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
    parsed = kernel_taxonomy.parse_nccl_kernel("ncclDevKernel_AllReduce_Sum_bf16_RING_LL128")
    assert parsed.busbw_factor(1) is None


def test_non_nccl_kernel_is_not_parsed_as_a_collective():
    assert kernel_taxonomy.parse_nccl_kernel("ampere_sgemm_128x64_nn") is None


# ---------------------------------------------------------------------------
# Metric catalog
# ---------------------------------------------------------------------------

def test_stall_taxonomy_is_complete():
    """Nsight Compute ships exactly 19 warp-stall reasons."""
    assert len(metric_catalog.STALL_REASONS) == 19
    for key, reason in metric_catalog.STALL_REASONS.items():
        assert reason.metric_name.startswith("smsp__average_warps_issue_stalled_")
        assert reason.metric_name.endswith("_per_issue_active.ratio")
        assert reason.bucket
        # Every actionable reason must carry advice; the benign ones need not.
        if key not in metric_catalog.BENIGN_STALL_KEYS:
            assert reason.fixes, f"{key} has no suggested fix"


def test_pc_sampling_spellings_differ_where_nvidia_renamed_them():
    gmma = metric_catalog.STALL_REASONS["gmma"]
    assert gmma.pcsamp_metric_name.endswith("warpgroup_arrive")


def test_metric_catalog_entries_are_well_formed():
    assert len(metric_catalog.METRIC_CATALOG) > 80
    for key, spec in metric_catalog.METRIC_CATALOG.items():
        assert spec.names, f"{key} has no metric names"
        assert spec.section and spec.category


@pytest.mark.parametrize(
    "cc_major,cc_minor,family",
    [(7, 0, "volta"), (7, 5, "turing"), (8, 0, "ampere"),
     (8, 9, "ada"), (9, 0, "hopper"), (10, 0, "blackwell"), (12, 0, "blackwell")],
)
def test_architecture_detection_spans_volta_to_blackwell(cc_major, cc_minor, family):
    assert metric_catalog.describe_arch(cc_major, cc_minor)["family"] == family


def test_architecture_falls_back_to_sm_count():
    result = metric_catalog.describe_arch(None, None, 108)
    assert result["family"] == "ampere"
    assert result["source"] == "sm_count_heuristic"
    assert metric_catalog.describe_arch()["family"] == "unknown"


# ---------------------------------------------------------------------------
# Interval algebra
# ---------------------------------------------------------------------------

def test_merge_intervals_coalesces_and_sorts():
    assert triage.merge_intervals([(30, 40), (0, 10), (5, 20)]) == [(0.0, 20.0), (30.0, 40.0)]
    assert triage.merge_intervals([]) == []
    # Zero-length and inverted intervals are dropped rather than corrupting the union.
    assert triage.merge_intervals([(5, 5), (10, 3)]) == []


def test_union_counts_overlapping_time_once():
    assert triage.interval_union_ns([(0, 10), (5, 20)]) == 20.0


def test_overlap_is_the_intersection_of_two_unions():
    assert triage.interval_overlap_ns([(0, 10)], [(5, 20)]) == 5.0
    assert triage.interval_overlap_ns([(0, 10)], [(20, 30)]) == 0.0
    assert triage.interval_overlap_ns([(0, 10), (20, 30)], [(5, 25)]) == 10.0
    assert triage.interval_overlap_ns([], [(0, 10)]) == 0.0


# ---------------------------------------------------------------------------
# Triage tree
# ---------------------------------------------------------------------------

MS = 1_000_000


def test_idle_gpu_with_launch_overhead_is_host_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(70 * MS, 200 * MS)],
        launch_api_ns=25 * MS,
        kernel_durations_ns=[2 * MS] * 60,
    )
    assert verdict.verdict == "host_bound"
    assert "gpu_idle_ratio" in [s.key for s in verdict.signals if s.crossed]


def test_one_signal_alone_does_not_declare_host_bound():
    """NVIDIA's rule is that two or more host signals must cross."""
    verdict = triage.triage_step(
        wall_ns=100 * MS,
        compute_intervals=[(35 * MS, 100 * MS)],  # 35% idle, but launch time is tiny
        launch_api_ns=1 * MS,
        kernel_durations_ns=[10 * MS] * 6,
    )
    assert verdict.verdict != "host_bound"


def test_exposed_communication_is_comm_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(0, 120 * MS)],
        comm_intervals=[(118 * MS, 195 * MS)],
        launch_api_ns=5 * MS,
        kernel_durations_ns=[3 * MS] * 40,
    )
    assert verdict.verdict == "communication_bound"
    assert verdict.breakdown["overlap_pct_of_comm"] < 10


def test_same_comm_volume_hidden_under_compute_is_not_comm_bound():
    """The discriminator is exposure, not volume."""
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(0, 190 * MS)],
        comm_intervals=[(20 * MS, 95 * MS)],
        launch_api_ns=5 * MS,
        kernel_durations_ns=[3 * MS] * 40,
    )
    assert verdict.verdict == "kernel_bound"
    assert verdict.breakdown["overlap_pct_of_comm"] == pytest.approx(100.0)
    assert verdict.breakdown["exposed_comm_ns"] == 0


def test_transfers_on_the_critical_path_are_transfer_bound():
    verdict = triage.triage_step(
        wall_ns=100 * MS,
        compute_intervals=[(30 * MS, 95 * MS)],
        memcpy_intervals=[(0, 28 * MS)],
        launch_api_ns=3 * MS,
        kernel_durations_ns=[4 * MS] * 16,
    )
    assert verdict.verdict == "transfer_bound"


def test_busy_gpu_with_large_kernels_is_kernel_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(2 * MS, 198 * MS)],
        launch_api_ns=4 * MS,
        kernel_durations_ns=[8 * MS] * 24,
    )
    assert verdict.verdict == "kernel_bound"
    assert verdict.next_steps


def test_cuda_graph_mode_tightens_the_idle_gate():
    kwargs = dict(
        wall_ns=100 * MS,
        compute_intervals=[(18 * MS, 100 * MS)],
        launch_api_ns=1 * MS,
        kernel_durations_ns=[5 * MS] * 16,
    )
    default = triage.triage_step(**kwargs)
    graphs = triage.triage_step(**kwargs, thresholds=triage.TriageThresholds(cuda_graphs=True))

    def idle(v):
        return next(s for s in v.signals if s.key == "gpu_idle_ratio")

    assert idle(default).crossed is False
    assert idle(graphs).crossed is True


def test_full_launch_queue_is_reported_as_gpu_bound_not_host_bound():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(2 * MS, 198 * MS)],
        launch_api_ns=4 * MS,
        kernel_durations_ns=[8 * MS] * 24,
        max_queue_length=1024,
    )
    assert any("queue" in note.lower() for note in verdict.secondary)


def test_steady_state_allocations_surface_as_a_secondary_note():
    verdict = triage.triage_step(
        wall_ns=200 * MS,
        compute_intervals=[(2 * MS, 195 * MS)],
        launch_api_ns=4 * MS,
        kernel_durations_ns=[8 * MS] * 24,
        steady_state_allocs=37,
    )
    assert any("cudaMalloc" in note for note in verdict.secondary)


def test_verdict_serialises_for_reports():
    verdict = triage.triage_step(wall_ns=10 * MS, compute_intervals=[(0, 9 * MS)])
    payload = verdict.to_dict()
    assert set(payload) >= {"verdict", "confidence", "summary", "signals", "breakdown", "next_steps"}
    assert isinstance(payload["signals"], list)


def test_empty_trace_degrades_to_low_confidence_rather_than_crashing():
    verdict = triage.triage_step(wall_ns=0)
    assert verdict.verdict
    assert verdict.confidence == "low"


# ---------------------------------------------------------------------------
# Correctness hardening: refusing to draw wrong conclusions
# ---------------------------------------------------------------------------


def _h100():
    return gpu_specs.lookup_gpu_spec("H100 SXM5")


def test_measurement_above_hardware_peak_is_rejected():
    """A number above the physical ceiling is a measurement bug, not a result.

    Publicised "100x speedup" claims have failed exactly here - the figure
    exceeded what the hardware can do and nobody checked the arithmetic.
    """
    result = ncu_diagnostics.diagnose_kernel(
        {
            "gpu__time_duration.sum": 100_000,
            "dram__bytes.sum": 1_000_000,
            # 500e9 FMA instructions in 100 us => ~10 PFLOP/s, ~10x above peak.
            "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum": 500_000_000_000,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 50.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        },
        kernel_name="void suspicious_kernel()",
        gpu_spec=_h100(),
    )
    assert result["sections"]["roofline"]["sanity_violations"]
    assert result["findings"][0]["category"] == "measurement_above_physical_limit"
    # And it must not go on to report a roofline percentage built on that number.
    assert not any(f["category"] == "below_roofline" for f in result["findings"])


SATURATED_LOW_OCCUPANCY = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": 87.0,
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
    "sm__maximum_warps_per_active_cycle_pct": 25.0,
    "sm__warps_active.avg.pct_of_peak_sustained_active": 24.0,
    "launch__occupancy_limit_registers": 12,
    "launch__occupancy_limit_warps": 64,
    "launch__occupancy_limit_blocks": 64,
    "launch__occupancy_limit_shared_mem": 64,
    "smsp__issue_active.avg.per_cycle_active": 0.85,
}


def test_low_occupancy_on_a_saturated_kernel_is_not_a_finding():
    """CUTLASS-class GEMMs run at low occupancy by design and still hit peak."""
    result = ncu_diagnostics.diagnose_kernel(
        SATURATED_LOW_OCCUPANCY, kernel_name="cutlass3x_sm90_gemm", gpu_spec=_h100())
    assert not [f for f in result["findings"] if "occupancy" in f["category"]]


def test_low_occupancy_is_a_finding_when_schedulers_are_starving():
    starved = dict(SATURATED_LOW_OCCUPANCY, **{
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 18.0,
        "smsp__issue_active.avg.per_cycle_active": 0.15,
    })
    result = ncu_diagnostics.diagnose_kernel(
        starved, kernel_name="void my_kernel()", gpu_spec=_h100())
    assert [f for f in result["findings"] if "occupancy" in f["category"]]


def test_warp_specialized_kernels_are_excluded_from_the_occupancy_model():
    """setmaxnreg makes registers-per-thread a weighted artifact."""
    metrics = dict(SATURATED_LOW_OCCUPANCY, **{
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 18.0,
        "smsp__issue_active.avg.per_cycle_active": 0.15,
        "sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active": 60.0,
        "launch__registers_per_thread": 168,
    })
    section = ncu_diagnostics.diagnose_kernel(
        metrics, kernel_name="void flash_fwd_ws()", gpu_spec=_h100())["sections"]["occupancy"]
    assert section["warp_specialized"] is True
    assert section["occupancy_model_applicable"] is False
    assert not section["findings"]


def test_green_context_grid_is_judged_against_the_partition():
    """A green context owns a subset of SMs; the device total is the wrong denominator."""
    result = ncu_diagnostics.diagnose_kernel(
        {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 25.0,
            "launch__grid_size": 20,
            "launch__block_size": 256,
            "launch__sm_count": 16,                              # the partition
            "device__attribute_multiprocessor_count": 132,       # the whole GPU
            "launch__uses_green_context": 1,
        },
        kernel_name="void partitioned()",
        gpu_spec=_h100(),
    )
    # 20 blocks genuinely fills a 16-SM partition, so this must not be flagged.
    assert not [f for f in result["findings"] if f["category"] == "small_grid"]


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
