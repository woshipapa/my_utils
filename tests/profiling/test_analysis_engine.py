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
throttling = _load("hardware.throttling", "hardware/throttling.py")
kernel_taxonomy = _load("sources.kernel_taxonomy", "sources/kernel_taxonomy.py")
metric_catalog = _load("ncu.metric_catalog", "ncu/metric_catalog.py")
triage = _load("analyzers.triage", "analyzers/triage.py")
evidence = _load("analyzers.evidence", "analyzers/evidence.py")
nccl_bandwidth = _load("analyzers.nccl_bandwidth", "analyzers/nccl_bandwidth.py")
trace_quality = _load("analyzers.trace_quality", "analyzers/trace_quality.py")
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


# ---------------------------------------------------------------------------
# Complete metric index (generated from the installed Nsight Compute)
# ---------------------------------------------------------------------------

section_index = _load("ncu.section_index", "ncu/section_index.py")


def test_metric_name_decoding():
    """The name grammar carries unit / quantity / rollup / submetric."""
    decoded = section_index.decode_metric_name(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert decoded["unit"] == "sm"
    assert decoded["quantity"] == "throughput"
    assert decoded["rollup"] == "avg"
    assert decoded["submetric"] == "pct_of_peak_sustained_elapsed"

    # No rollup: the submetric follows the name directly.
    plain = section_index.decode_metric_name("l1tex__t_sector_hit_rate.pct")
    assert plain["unit"] == "l1tex"
    assert plain["rollup"] == ""
    assert plain["submetric"] == "pct"

    # Collection prefixes are stripped, not mistaken for the unit.
    prefixed = section_index.decode_metric_name("pmsampling:dram__bytes.sum")
    assert prefixed["prefix"] == "pmsampling"
    assert prefixed["unit"] == "dram"

    # A launch property has no unit separator at all.
    assert section_index.decode_metric_name("launch__grid_size")["unit"] == "launch"


def test_active_vs_elapsed_distinction_is_documented():
    """Choosing the wrong denominator silently changes the conclusion."""
    active = section_index.SUBMETRIC_MEANINGS["pct_of_peak_sustained_active"]
    elapsed = section_index.SUBMETRIC_MEANINGS["pct_of_peak_sustained_elapsed"]
    assert "ACTIVE" in active
    assert "WHOLE" in elapsed


def test_explain_metric_works_without_an_ncu_installation():
    """Falls back to decoding the name rather than fabricating an explanation."""
    result = metric_catalog.explain_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert result["metric"]
    # Either the index resolved it or the name was decoded; never silently empty.
    assert result.get("description") or result.get("interpretation") or result.get("unit")


def test_explain_metric_gives_a_verdict_only_where_a_rule_exists():
    bad = metric_catalog.explain_metric(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 28.0)
    assert bad["has_rule"] is True
    assert bad["verdict"] == "bad"
    assert bad["distance_from_ideal"] == pytest.approx(24.0)

    good = metric_catalog.explain_metric(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", 4.0)
    assert good["verdict"] == "ok"

    # Without a value there is no verdict to give.
    assert "verdict" not in metric_catalog.explain_metric(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")


@pytest.mark.skipif(
    section_index.find_sections_dir() is None,
    reason="no Nsight Compute installation on this machine",
)
def test_section_index_covers_the_full_set():
    """Against a real install, every --set full metric must be indexed."""
    index = section_index.build_section_index()
    assert index is not None
    assert len(index.sections) >= 20
    # The index exists precisely so coverage is complete rather than curated.
    assert len(index.in_set("full")) > 500
    # NVIDIA's own Labels come through, which is what makes an unknown metric
    # explainable at all.
    entry = index.explain("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    assert entry is not None and entry.label
    # The units a user asks about are all represented.
    for unit in ("l1tex", "lts", "dram", "sm", "smsp", "launch"):
        assert index.by_unit(unit), f"no metrics indexed for {unit}"


# ---------------------------------------------------------------------------
# Evidence fusion
# ---------------------------------------------------------------------------

class TestEvidenceFusion:
    """A conclusion must never outrank the evidence it rests on."""

    def test_counter_beats_name(self):
        """A name saying matmul loses to a tensor pipe that never activated."""
        ev = evidence
        fused, warnings = ev.attribute_kernel(
            "ampere_sgemm_128x64_nn",
            metrics={"sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 0.0},
        )
        assert fused["uses_tensor_cores"].value is False
        assert fused["uses_tensor_cores"].provenance == ev.Provenance.HW_COUNTER
        assert any("tensor pipe never activated" in w for w in warnings)

    def test_nccl_stub_algorithm_is_never_authoritative(self):
        """The RING/LL tokens are hard-coded, so they must not read as measured."""
        ev = evidence
        fused, _ = ev.attribute_kernel("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL")
        algo = fused["nccl_algorithm"]
        assert algo.confidence == "low"
        assert algo.supporting[0].advisory

    def test_nvtx_outranks_a_launch_stub_name(self):
        """Where the name is a stub, the framework's own annotation wins."""
        ev = evidence
        fused, _ = ev.attribute_kernel(
            "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
            nvtx_payloads={"Collective name": "reducescatter"},
        )
        assert fused["collective"].value == "reducescatter"
        assert fused["collective"].provenance == ev.Provenance.NVTX

    def test_unlabeled_cute_dsl_kernel_is_flagged(self):
        """FlashAttention-4 ships with exactly this naming."""
        ev = evidence
        fused, warnings = ev.attribute_kernel("kernel_kernel_128_64_0")
        assert fused["name_is_informative"].value is False
        assert any("no shape or dtype" in w for w in warnings)


class TestTriageRefusesMissingData:
    """Absent GPU intervals and an idle GPU are indistinguishable; say so."""

    def test_no_intervals_yields_no_verdict(self):
        tri = triage
        verdict = tri.triage_step(
            wall_ns=1e9, launch_api_ns=8e8, sync_api_ns=5e8,
            kernel_durations_ns=[4e3] * 50,
        )
        assert verdict.verdict == "undetermined"
        assert verdict.confidence == "low"
        assert "could not be measured" in verdict.summary

    def test_real_intervals_still_reach_host_bound(self):
        tri = triage
        verdict = tri.triage_step(
            wall_ns=1e9, compute_intervals=[(0, 5e7)], launch_api_ns=8e8,
            sync_api_ns=5e8, kernel_durations_ns=[4e3] * 50,
        )
        assert verdict.verdict == "host_bound"


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


class TestAnalysisCoverage:
    """A skipped analysis and a clean analysis both produce zero findings."""

    def test_missing_sections_are_reported(self):
        # A SpeedOfLight-only report, which is what real ncu runs often carry.
        view = ncu_diagnostics.MetricView({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
        })
        cov = ncu_diagnostics.analysis_coverage(view)
        assert "bottleneck" in cov["ran"]
        skipped = {s["analysis"] for s in cov["skipped"]}
        assert "stalls" in skipped
        assert "shared_memory" in skipped
        assert cov["coverage_pct"] < 100.0
        assert "not a clean result" in cov["summary"]

    def test_named_section_is_actionable(self):
        view = ncu_diagnostics.MetricView({})
        cov = ncu_diagnostics.analysis_coverage(view)
        for entry in cov["skipped"]:
            assert entry["needs_section"], f"{entry['analysis']} has no section to collect"
        assert cov["remedy"]


class TestEveryFindingCarriesEvidence:
    """A conclusion without its evidence is an assertion, not an analysis."""

    def test_all_findings_have_evidence(self):
        metrics = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 25.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            "sm__warps_active.avg.pct_of_peak_sustained_active": 12.0,
            "sm__maximum_warps_per_active_cycle_pct": 50.0,
            "smsp__issue_active.avg.per_cycle_active": 0.15,
            "launch__grid_size": 8.0,
            "launch__block_size": 100.0,
            "launch__waves_per_multiprocessor": 0.3,
        }
        result = ncu_diagnostics.diagnose_kernel(metrics, kernel_name="my_fused_kernel")
        findings = [f if isinstance(f, dict) else f.to_dict() for f in result["findings"]]
        assert findings, "expected findings from a deliberately unhealthy kernel"
        for f in findings:
            assert f.get("evidence"), f"finding {f['title']!r} carries no evidence"
            assert f.get("summary"), f"finding {f['title']!r} carries no explanation"


class TestPackedFp32Correction:
    """Blackwell packs two FP32 ops per instruction; missing it halves the answer."""

    BASE = {
        "gpu__time_duration.sum": 1e6,
        "dram__bytes.sum": 1e8,
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9,
    }

    def test_applied_on_cc10(self):
        m = dict(self.BASE, **{
            "device__attribute_compute_capability_major": 10,
            "device__attribute_compute_capability_minor": 0,
            "smsp__sass_thread_inst_executed_op_ffma2_pred_on.sum": 1e9,
        })
        r = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(m))
        assert r["packed_fp32_applied"] is True
        base = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(self.BASE))
        assert r["achieved_tflops"] > base["achieved_tflops"]

    def test_not_applied_off_cc10(self):
        r = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(self.BASE))
        assert r["packed_fp32_applied"] is False
        assert not r.get("caveats")

    def test_absent_counters_on_cc10_are_flagged(self):
        m = dict(self.BASE, **{
            "device__attribute_compute_capability_major": 10,
            "device__attribute_compute_capability_minor": 0,
        })
        r = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(m))
        assert r["packed_fp32_applied"] is False
        assert any("undercounted by up to 2x" in c for c in r["caveats"])


class TestStragglerNeedsUsableClocks:
    """Naming a straggler across hosts requires clocks that can support it."""

    def _entries(self, late_ns):
        return [
            {"rank": r, "collective_seq_id": 7,
             "time_created_ns": 1_000_000_000 + (late_ns if r == 5 else 0)}
            for r in range(8)
        ]

    def test_large_spread_names_the_rank(self):
        r = nccl_bandwidth.detect_straggler_from_traces(
            self._entries(40_000_000), collective_seq_id=7, clock_alignment="UTC")
        assert r["conclusive"] is True
        assert r["worst_rank"] == 5

    def test_small_spread_refused_under_utc(self):
        r = nccl_bandwidth.detect_straggler_from_traces(
            self._entries(2_000_000), collective_seq_id=7, clock_alignment="UTC")
        assert r["conclusive"] is False
        assert "clock skew" in r["reason"]

    def test_small_spread_allowed_on_same_host(self):
        r = nccl_bandwidth.detect_straggler_from_traces(
            self._entries(2_000_000), collective_seq_id=7, same_host=True)
        assert r["conclusive"] is True
        assert r["worst_rank"] == 5


class TestDerivedMetricInvariants:
    """A violated identity means a wrong denominator, not a finding."""

    def test_above_peak_blocks(self):
        issues = trace_quality.check_derived_metric_invariants(mfu=520.0, dtype="bf16")
        assert any(i.key == "mfu_above_peak" and i.blocks for i in issues)

    def test_hfu_below_mfu_blocks(self):
        issues = trace_quality.check_derived_metric_invariants(mfu=45.0, hfu=38.0, dtype="bf16")
        assert any(i.key == "hfu_below_mfu" and i.blocks for i in issues)

    def test_unknown_dtype_blocks(self):
        issues = trace_quality.check_derived_metric_invariants(mfu=41.0)
        assert any(i.key == "unknown_dtype_denominator" and i.blocks for i in issues)

    def test_healthy_is_clean(self):
        assert not trace_quality.check_derived_metric_invariants(
            mfu=45.0, hfu=52.0, dtype="bf16")


class TestShapeKeyedGrouping:
    """One kernel name covers genuinely different work."""

    def test_shapes_are_not_merged(self):
        launches = (
            [{"kernel_name": "k", "grid_size": 128, "duration_ns": 10_000}] * 8
            + [{"kernel_name": "k", "grid_size": 512, "duration_ns": 40_000}] * 8
        )
        g = trace_quality.group_kernels_by_shape(launches)
        assert g["group_count"] == 2
        assert g["distinct_names"] == 1
        assert "would have merged them" in g["note"]

    def test_dispersed_group_is_flagged(self):
        launches = (
            [{"kernel_name": "k", "grid_size": 512, "duration_ns": 40_000}] * 8
            + [{"kernel_name": "k", "grid_size": 512, "duration_ns": 900_000}]
        )
        g = trace_quality.group_kernels_by_shape(launches)
        assert g["non_stationary"]
        assert "single" in g["warning"]


class TestCaveatsReachTheFindingsList:
    """A caveat that stays inside its section is invisible to every caller."""

    def test_incomplete_measurement_surfaces(self):
        # CC 10.0 without the packed-FP32 counters: the FP32 figure is halved,
        # and a caller reading only `findings` must still learn that.
        metrics = {
            "gpu__time_duration.sum": 1e6,
            "dram__bytes.sum": 1e8,
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9,
            "device__attribute_compute_capability_major": 10,
            "device__attribute_compute_capability_minor": 0,
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 70.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        }
        result = ncu_diagnostics.diagnose_kernel(metrics, kernel_name="fused")
        assert any(f["category"] == "measurement_caveat" for f in result["findings"]), \
            "roofline caveat never reached the findings list"
        assert result["sections"]["roofline"]["packed_fp32_applied"] is False

    def test_no_spurious_caveat_when_complete(self):
        metrics = {
            "gpu__time_duration.sum": 1e6,
            "dram__bytes.sum": 1e8,
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9,
        }
        result = ncu_diagnostics.diagnose_kernel(metrics, kernel_name="fused")
        assert not any(f["category"] == "measurement_caveat" for f in result["findings"])


class TestPackageExportsResolve:
    """A documented import that does not exist is worse than no documentation."""

    @staticmethod
    def _check(init_rel, pkg_dir):
        import ast
        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        tree = ast.parse((root.parent.parent / init_rel).read_text())
        imported, exported = {}, []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                for al in node.names:
                    imported[al.asname or al.name] = node.module
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "__all__":
                        exported = [e.value for e in node.value.elts]

        cache = {}

        def defined_in(mod):
            if mod not in cache:
                names = set()
                src = (root.parent.parent / pkg_dir / f"{mod}.py").read_text()
                for n in ast.walk(ast.parse(src)):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.add(n.name)
                    elif isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                names.add(t.id)
                    elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                        names.add(n.target.id)
                cache[mod] = names
            return cache[mod]

        broken = [
            name for name in exported
            if name not in imported or name not in defined_in(imported[name])
        ]
        assert not broken, f"{init_rel} exports names that do not resolve: {broken}"
        assert exported, f"{init_rel} exports nothing"

    def test_analyzers_exports(self):
        self._check("my_utils/profiling/analyzers/__init__.py", "my_utils/profiling/analyzers")

    def test_hardware_exports(self):
        self._check("my_utils/profiling/hardware/__init__.py", "my_utils/profiling/hardware")


class TestHandbookExamplesAreReal:
    """Every symbol the handbook tells a reader to import must exist."""

    def test_documented_imports_resolve(self):
        import ast, re
        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        handbook = root / "docs" / "PERFORMANCE_ANALYSIS_HANDBOOK.md"
        if not handbook.exists():
            pytest.skip("handbook not present")
        text = handbook.read_text()

        missing = []
        for module, names in re.findall(r"from (my_utils\.profiling[\w.]*) import ([\w, ]+)", text):
            rel = module.replace("my_utils.profiling", "").lstrip(".").replace(".", "/")
            candidates = [root / f"{rel}.py", root / rel / "__init__.py"]
            path = next((c for c in candidates if c.exists()), None)
            if path is None:
                missing.append(f"{module} (module not found)")
                continue
            defined = set()
            for n in ast.walk(ast.parse(path.read_text())):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(n.name)
                elif isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            defined.add(t.id)
                elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                    defined.add(n.target.id)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        defined.add(a.asname or a.name.split(".")[0])
            for name in (x.strip() for x in names.split(",") if x.strip()):
                if name not in defined:
                    missing.append(f"{module}.{name}")
        assert not missing, f"handbook documents imports that do not exist: {missing}"

    def test_python_blocks_parse(self):
        import ast, re
        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        handbook = root / "docs" / "PERFORMANCE_ANALYSIS_HANDBOOK.md"
        if not handbook.exists():
            pytest.skip("handbook not present")
        bad = []
        for i, block in enumerate(re.findall(r"```python\n(.*?)```", handbook.read_text(), re.S)):
            try:
                ast.parse(block)
            except SyntaxError as exc:
                bad.append(f"block {i}: {exc.msg}")
        assert not bad, f"handbook python blocks do not parse: {bad}"


# ---------------------------------------------------------------------------
# Axis vocabulary and coverage
# ---------------------------------------------------------------------------

axes = _load("analyzers.axes", "analyzers/axes.py")
shipped_rules = _load("ncu.shipped_rules", "ncu/shipped_rules.py")


class TestCoverageKeysAreReal:
    """Coverage tables must reference catalog keys that exist.

    `MetricView.get` falls through to a raw ncu-metric lookup for an unknown
    key and returns None, so a typo does not raise - it silently reports the
    analysis as "skipped, metrics absent" on every report forever. Seven keys in
    `_ANALYSIS_REQUIREMENTS` were wrong this way, which made stalls, coalescing,
    divergence and spilling permanently look uncollected even on --set full.
    """

    def test_analysis_requirements_keys_exist(self):
        catalog = set(metric_catalog.METRIC_CATALOG)
        bad = [
            (analysis, key)
            for analysis, (keys, _section) in ncu_diagnostics._ANALYSIS_REQUIREMENTS.items()
            for key in keys
            if key not in catalog
        ]
        assert not bad, f"_ANALYSIS_REQUIREMENTS references non-catalog keys: {bad}"

    def test_axis_metric_groups_keys_exist(self):
        catalog = set(metric_catalog.METRIC_CATALOG)
        bad = [
            (axis.axis_id, key)
            for axis in axes.AXES
            for group in axis.metric_groups
            for key in group
            if key not in catalog
        ]
        assert not bad, f"axes.AXES references non-catalog keys: {bad}"

    def test_every_emitted_category_maps_to_an_axis(self):
        """A finding whose category maps to no axis vanishes from coverage."""
        import re
        source = (Path(ncu_diagnostics.__file__)).read_text()
        emitted = set(re.findall(r'category="([a-z0-9_]+)"', source))
        unmapped = sorted(c for c in emitted if not axes.axis_for_category(c))
        assert not unmapped, f"finding categories with no axis: {unmapped}"


class TestAxisCoverage:
    def test_unexamined_axis_is_not_reported_as_clean(self):
        result = axes.axis_coverage([], metric_present=lambda key: False)
        stall = next(a for a in result["axes"] if a["axis"] == "stall")
        assert stall["examined"] is False
        assert stall["reason_not_examined"]
        assert stall["remedy"], "an unexamined axis must say how to examine it"

    def test_findings_mark_their_axis_examined(self):
        result = axes.axis_coverage(
            [{"category": "uncoalesced_global_access"}], metric_present=lambda key: False,
        )
        mem = next(a for a in result["axes"] if a["axis"] == "memory_bandwidth")
        assert mem["examined"] is True and mem["finding_count"] == 1

    def test_present_metrics_mark_axis_examined_without_findings(self):
        """A clean axis and an unchecked axis must be distinguishable."""
        present = {"achieved_occupancy", "theoretical_occupancy"}
        result = axes.axis_coverage([], metric_present=present)
        sched = next(a for a in result["axes"] if a["axis"] == "scheduler")
        assert sched["examined"] is True and sched["finding_count"] == 0

    def test_unmapped_categories_are_surfaced_not_dropped(self):
        result = axes.axis_coverage([{"category": "zzz_not_a_real_category"}])
        assert "zzz_not_a_real_category" in result["unmapped_categories"]


class TestShippedRuleReconciliation:
    def _rule(self, ident, msg, mtype="warning", stype="", speedup=None):
        return {
            "rule_identifier": ident,
            "rule_message": {"message": msg, "title": msg, "message_type": mtype},
            "speedup_estimation": {"type": stype, "speedup": speedup},
        }

    def test_local_speedup_is_not_promoted_to_kernel_level(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("UncoalescedGlobalAccess", "excess sectors", stype="LOCAL", speedup=45.0)]
        )
        assert rules[0].speedup_ceiling is None
        findings = shipped_rules.shipped_rules_to_findings(rules)
        assert any("LOCAL" in a for a in findings[0].actions)

    def test_global_speedup_converts_to_a_ceiling(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("SOLBottleneck", "memory bound", stype="GLOBAL", speedup=50.0)]
        )
        assert rules[0].speedup_ceiling == pytest.approx(2.0)

    def test_ok_messages_are_not_raised_as_problems(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("Occupancy", "no issues", mtype="ok")]
        )
        assert rules[0].is_actionable is False
        assert shipped_rules.shipped_rules_to_findings(rules) == []

    def test_bottleneck_disagreement_is_reported(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("SOLBottleneck", "This kernel is memory bound.")]
        )
        out = shipped_rules.reconcile_with_shipped_rules([], rules, our_verdict="compute_bound")
        assert out["conflicts"], "compute-vs-memory disagreement must be raised"
        assert any(f.category == "evidence_conflict" for f in out["findings"])

    def test_agreement_promotes_confidence_and_names_the_rule(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("UncoalescedGlobalAccess", "excess sectors")]
        )
        ours = [ncu_diagnostics.Finding(
            category="uncoalesced_global_access", title="t", summary="s", confidence="medium",
        )]
        out = shipped_rules.reconcile_with_shipped_rules(ours, rules)
        promoted = next(f for f in out["findings"] if f.category == "uncoalesced_global_access")
        assert promoted.confidence == "high"
        assert "corroborated_by_ncu_rule" in promoted.evidence

    def test_absent_shipped_rules_do_not_weaken_findings(self):
        ours = [ncu_diagnostics.Finding(category="uncoalesced_global_access", title="t", summary="s")]
        out = shipped_rules.reconcile_with_shipped_rules(ours, [])
        assert out["shipped_rules_available"] is False
        assert out["findings"] == ours

    def test_ncu_only_rules_are_not_dropped(self):
        rules = shipped_rules.normalize_shipped_rules(
            [self._rule("SharedMemoryConflicts", "bank conflicts detected")]
        )
        out = shipped_rules.reconcile_with_shipped_rules([], rules)
        assert any(f.source == "ncu_rule" for f in out["findings"])

    def test_malformed_rule_blocks_do_not_raise(self):
        assert shipped_rules.normalize_shipped_rules([None, 42, "x", {}]) != []


class TestDiagnoseKernelReportsAxes:
    def test_axes_present_and_gaps_named(self):
        result = ncu_diagnostics.diagnose_kernel(
            {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
             "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 20.0},
            kernel_name="fused_kernel",
        )
        assert result["axes"]["axis_count"] == len(axes.AXES)
        assert "power_clock" in result["axes"]["not_examined"]
        assert result["corroboration"]["shipped_rules_available"] is False


# ---------------------------------------------------------------------------
# Metric-name grammar and full-report accounting
# ---------------------------------------------------------------------------

section_index = _load("ncu.section_index", "ncu/section_index.py")


class TestMetricGrammar:
    def test_every_catalog_metric_decodes_to_a_unit(self):
        """A catalog name that decodes to no unit would vanish from the inventory."""
        undecodable = []
        for spec in metric_catalog.METRIC_CATALOG.values():
            for name in spec.names:
                parts = section_index.decode_metric_name(name)
                if not parts.get("unit") and not section_index._legacy_unit(name):
                    undecodable.append(name)
        assert not undecodable, f"catalog names with no decodable unit: {undecodable}"

    def test_every_catalog_metric_lands_on_an_axis(self):
        orphans = []
        for spec in metric_catalog.METRIC_CATALOG.values():
            for name in spec.names:
                unit = (section_index.decode_metric_name(name).get("unit")
                        or section_index._legacy_unit(name))
                if unit and unit not in section_index.UNIT_AXIS:
                    orphans.append((name, unit))
        assert not orphans, f"metric units with no axis: {sorted(set(orphans))}"

    def test_active_and_elapsed_denominators_are_distinguished(self):
        """Mixing the two is how an idle unit gets ranked as the bottleneck."""
        assert section_index.denominator_of(
            "sm__throughput.avg.pct_of_peak_sustained_active") == "active"
        assert section_index.denominator_of(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed") == "elapsed"
        assert section_index.denominator_of("dram__bytes.sum") == ""

    def test_collection_prefixes_are_stripped(self):
        parts = section_index.decode_metric_name(
            "pmsampling:smsp__warps_issue_stalled_barrier.avg")
        assert parts["prefix"] == "pmsampling"
        assert parts["unit"] == "smsp" and parts["rollup"] == "avg"

    def test_unknown_unit_is_reported_not_guessed(self):
        grouped = section_index.group_report_metrics(["weird__counter.avg"])
        assert grouped["unknown_units"] == {"weird": 1}
        assert section_index.axis_for_metric_name("weird__counter.avg") == ""

    def test_uncatalogued_metrics_are_counted_not_dropped(self):
        grouped = section_index.group_report_metrics(
            ["sm__throughput.avg.pct_of_peak_sustained_elapsed",
             "lts__t_sectors_srcunit_tex_op_read.sum"],
            catalog=metric_catalog.METRIC_CATALOG,
        )
        assert grouped["total"] == 2
        assert "lts__t_sectors_srcunit_tex_op_read.sum" in grouped["uncatalogued"]
        assert grouped["by_axis"]["memory_bandwidth"]

    def test_display_names_survive_as_undecodable(self):
        grouped = section_index.group_report_metrics(["Duration"])
        assert grouped["undecodable"] == ["Duration"]


class TestDiagnoseKernelAccountsForEveryMetric:
    def test_uncatalogued_metrics_are_reported(self):
        result = ncu_diagnostics.diagnose_kernel({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 70.0,
            "lts__t_sectors_srcunit_tex_op_read.sum": 12345.0,
        })
        inventory = result["metric_inventory"]
        assert inventory["total"] == 2
        assert inventory["uncatalogued_count"] == 1
        assert "memory_bandwidth" in inventory["axis_counts"]


class TestThrottlingIsWiredIn:
    """A throttled run measures the clock, not the kernel."""

    def test_power_clock_axis_is_a_gap_without_telemetry(self):
        result = ncu_diagnostics.diagnose_kernel({"sm__cycles_elapsed.avg": 1e6})
        assert "power_clock" in result["axes"]["not_examined"]

    def test_throttling_is_reported_and_demotes_confidence(self):
        result = ncu_diagnostics.diagnose_kernel(
            {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
             "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 20.0},
            kernel_name="gemm",
            throttling={"clock_event_mask": 0x4},   # SwPowerCap
        )
        assert "power_clock" not in result["axes"]["not_examined"]
        throttle_findings = [f for f in result["findings"] if f["category"] == "throttling"]
        assert throttle_findings, "a throttled run must say so"
        assert throttle_findings[0]["severity"] == "high"
        for finding in result["findings"]:
            if finding["category"] in ("bottleneck", "below_roofline"):
                assert finding["confidence"] == "low"

    def test_idle_gpu_is_not_reported_as_throttled(self):
        """GpuIdle (0x1) and ApplicationsClocksSetting (0x2) are not throttling."""
        result = ncu_diagnostics.diagnose_kernel(
            {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0},
            throttling={"clock_event_mask": 0x1 | 0x2},
        )
        assert not [f for f in result["findings"] if f["category"] == "throttling"]


class TestTileQuantization:
    """The symbol carries the tile shape and never the problem shape."""

    _KERNEL = "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nn"

    def test_missing_problem_shape_is_reported_as_unasked(self):
        result = ncu_diagnostics.diagnose_kernel(
            {"launch__grid_size": 512, "launch__block_size": 256},
            kernel_name=self._KERNEL,
        )
        unasked = [f for f in result["findings"]
                   if f["category"] == "tile_quantization"
                   and "could not be checked" in f["title"]]
        assert unasked, "an unaskable question must not read as a clean result"
        assert unasked[0]["severity"] == "info"

    def test_ragged_problem_shape_is_flagged_with_a_ceiling(self):
        """M=257 against a 128 tile needs 3 tiles covering 384: 33% padding."""
        result = ncu_diagnostics.diagnose_kernel(
            {"launch__grid_size": 512, "launch__block_size": 256},
            kernel_name=self._KERNEL,
            problem_shape={"m": 257, "n": 4096},
        )
        flagged = [f for f in result["findings"]
                   if f["category"] == "tile_quantization" and "M" in f["title"]]
        assert flagged, "257 against a 128 tile computes a third padding"
        assert flagged[0]["evidence"]["waste_fraction"] == pytest.approx(0.331, abs=0.01)
        assert flagged[0]["speedup_ceiling"] == pytest.approx(1.494, abs=0.01)

    def test_large_ragged_dimension_is_not_flagged(self):
        """M=4097 wastes only 3%: real, but not worth an action."""
        result = ncu_diagnostics.diagnose_kernel(
            {"launch__grid_size": 512, "launch__block_size": 256},
            kernel_name=self._KERNEL,
            problem_shape={"m": 4097, "n": 4096},
        )
        assert not [f for f in result["findings"]
                    if f["category"] == "tile_quantization" and "M" in f["title"]]

    def test_evenly_divided_shape_is_not_flagged(self):
        result = ncu_diagnostics.diagnose_kernel(
            {"launch__grid_size": 512, "launch__block_size": 256},
            kernel_name=self._KERNEL,
            problem_shape={"m": 4096, "n": 4096},
        )
        assert not [f for f in result["findings"]
                    if f["category"] == "tile_quantization"]


class TestMemoryHierarchy:
    def test_saturated_l2_is_distinguished_from_dram(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(ncu_diagnostics.MetricView({
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": 88.0,
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 25.0,
        }))
        assert result["tightest_level"] == "L2"
        finding = next(f for f in result["findings"] if "saturated level" in f.title)
        assert "not DRAM" in finding.title

    def test_sysmem_aperture_access_is_high_severity(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(ncu_diagnostics.MetricView({
            "lts__t_sectors_srcunit_tex_aperture_sysmem_lookup_miss.sum": 50000.0,
        }))
        finding = next(f for f in result["findings"] if "system memory" in f.title)
        assert finding.severity == "high"

    def test_asymmetric_l2_hit_rate_is_flagged(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(ncu_diagnostics.MetricView({
            "lts__t_sector_op_read_hit_rate.pct": 15.0,
            "lts__t_sector_op_write_hit_rate.pct": 90.0,
        }))
        assert any(f.category == "poor_cache_locality" for f in result["findings"])

    def test_write_dominated_traffic_is_reported(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(ncu_diagnostics.MetricView({
            "dram__bytes_read.sum": 1e6, "dram__bytes_write.sum": 9e6,
        }))
        assert result["dram_write_share"] == pytest.approx(0.9)
        assert any("write-dominated" in f.title for f in result["findings"])

    def test_silent_when_no_memory_counters(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(ncu_diagnostics.MetricView({}))
        assert result["findings"] == []


class TestIssueEfficiency:
    def test_stalled_resident_warps_are_not_an_occupancy_problem(self):
        """High occupancy + no eligible warps means latency, not occupancy."""
        result = ncu_diagnostics.analyze_issue_efficiency(ncu_diagnostics.MetricView({
            "smsp__warps_eligible.avg.per_cycle_active": 0.3,
            "smsp__warps_active.avg.per_cycle_active": 12.0,
        }))
        finding = result["findings"][0]
        assert "not occupancy" in finding.summary
        assert "stall breakdown" in finding.actions[0]

    def test_low_occupancy_gets_the_occupancy_advice(self):
        result = ncu_diagnostics.analyze_issue_efficiency(ncu_diagnostics.MetricView({
            "smsp__warps_eligible.avg.per_cycle_active": 0.4,
            "smsp__warps_active.avg.per_cycle_active": 2.0,
        }))
        assert "Increase occupancy" in result["findings"][0].actions[0]

    def test_healthy_scheduler_is_silent(self):
        result = ncu_diagnostics.analyze_issue_efficiency(ncu_diagnostics.MetricView({
            "smsp__warps_eligible.avg.per_cycle_active": 3.5,
        }))
        assert result["findings"] == []


# ---------------------------------------------------------------------------
# nsys entry-point gating
# ---------------------------------------------------------------------------

nsys_auto = _load("sources.nsys_auto_analysis", "sources/nsys_auto_analysis.py")


class TestCollectiveBandwidthHonesty:
    """Bus bandwidth is not derivable from kernel timing, and must not be faked."""

    def test_bandwidth_is_reported_as_unmeasurable(self):
        cov = nsys_auto._collective_bandwidth_coverage(
            [{"kernel_name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL", "total_ms": 12.0}]
        )
        assert cov["measurable"] is False
        assert cov["busbw_gbps"] is None and cov["algbw_gbps"] is None
        assert "message bytes" in cov["reason"]
        assert "flight recorder" in cov["how_to_measure"]

    def test_collective_kind_is_read_but_algorithm_is_caveated(self):
        cov = nsys_auto._collective_bandwidth_coverage(
            [{"kernel_name": "ncclDevKernel_AllReduce_Sum_f32_RING_LL", "total_ms": 12.0}]
        )
        assert cov["collective_time_ms_by_kind"] == {"allreduce": 12.0}
        assert "NOT reliable" in cov["caveat"] or "NOT" in cov["caveat"]


class TestTraceQualityGating:
    def test_absent_checks_are_not_reported_as_passing(self):
        quality = nsys_auto._assess_quality(None, [])
        assert quality["checked"] is False
        assert quality["trustworthy"] is None, "unvalidated must not read as validated"

    def test_blocked_conclusions_leave_the_recommendation_list(self):
        recs = ["Increase batch size", "Reduce dataloader stalls", "Use CUDA graphs"]
        out = nsys_auto._strike_blocked_recommendations(recs, {"dataloader"})
        assert "Reduce dataloader stalls" not in out
        assert any("WITHHELD" in r for r in out)

    def test_nothing_struck_when_nothing_blocked(self):
        recs = ["Increase batch size"]
        assert nsys_auto._strike_blocked_recommendations(recs, set()) == recs


class TestInstructionMix:
    """SpeedOfLight compute is a max over pipes, so it hides the busy one."""

    def test_transcendental_bound_kernel_is_named(self):
        result = ncu_diagnostics.analyze_instruction_mix(ncu_diagnostics.MetricView({
            "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active": 85.0,
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": 20.0,
        }))
        assert "XU" in result["busiest_pipe"]
        finding = result["findings"][0]
        assert finding.severity == "medium"
        assert "__expf" in finding.actions[0] or "transcendental" in finding.actions[0]

    def test_lsu_bound_is_not_advised_as_a_bandwidth_problem(self):
        result = ncu_diagnostics.analyze_instruction_mix(ncu_diagnostics.MetricView({
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": 90.0,
        }))
        assert "vectorise" in result["findings"][0].actions[0].lower()

    def test_active_denominator_is_declared(self):
        """These are _active percentages and must not be ranked against SOL."""
        result = ncu_diagnostics.analyze_instruction_mix(ncu_diagnostics.MetricView({
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": 75.0,
        }))
        assert "active" in result["findings"][0].evidence["denominator"]

    def test_incidental_fp64_is_flagged_separately(self):
        result = ncu_diagnostics.analyze_instruction_mix(ncu_diagnostics.MetricView({
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": 70.0,
            "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": 4.0,
        }))
        assert any(f.category == "unexpected_fp64" for f in result["findings"])

    def test_silent_without_pipe_counters(self):
        assert ncu_diagnostics.analyze_instruction_mix(
            ncu_diagnostics.MetricView({}))["findings"] == []
