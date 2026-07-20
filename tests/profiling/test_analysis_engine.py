"""Tests for the GPU-spec tables, kernel taxonomy, ncu rule engine and triage tree.

These modules are pure analysis code with no CUDA or torch dependency, so the
tests load them directly from their file paths.  Going through the ``my_utils``
package would drag in ``my_utils.core.utils``, which imports torch, and that
would make these tests unrunnable exactly where they are most useful: a CI box
with no GPU stack installed.
"""

from __future__ import annotations

import importlib.util
import re
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
ncu_report_tools = _load("ncu.ncu_report_tools", "ncu/ncu_report_tools.py")


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
        """A finding whose category maps to no axis vanishes from coverage.

        Literal `category="..."` only. See the f-string test below for why that
        is not sufficient on its own.
        """
        import re
        source = (Path(ncu_diagnostics.__file__)).read_text()
        emitted = set(re.findall(r'category="([a-z0-9_]+)"', source))
        unmapped = sorted(c for c in emitted if not axes.axis_for_category(c))
        assert not unmapped, f"finding categories with no axis: {unmapped}"

    def test_fstring_categories_map_to_an_axis(self):
        """Categories built with an f-string are invisible to a literal grep.

        This is how 20 of 33 categories -- including 16 of the 19 stall reasons,
        which is most of what a latency-bound kernel reports -- were unmapped
        while the literal-string test above passed. An unmapped category is
        dropped from `by_axis`, so a kernel whose headline finding is
        `stall_long_scoreboard` reported the stall axis with finding_count=0 and
        a "collect WarpStateStats" remedy for an axis that had just fired.

        Every f-string site in ncu_diagnostics.py is expanded here by hand,
        because the interpolated values come from runtime data a static reader
        cannot see. `test_fstring_category_sites_are_all_covered` fails if a new
        site appears.
        """
        emitted = (
            [f"stall_{key}" for key in metric_catalog.STALL_REASONS]
            + [f"occupancy_limited_{b}" for b in
               ("registers", "shared_mem", "blocks", "warps", "barriers")]
            # The interpolated labels are "load"/"store" here, not "ld"/"st";
            # the first version of this test asserted values the code never
            # emits, so it passed while proving nothing about the real ones.
            + [f"uncoalesced_global_{op}" for op in ("load", "store")]
            + [f"sparse_global_{op}" for op in ("load", "store")]
            + [f"shared_bank_conflicts_{op}" for op in ("ld", "st")]
            + [f"{unit}_load_imbalance" for unit in ("sm", "l2", "dram")]
        )
        unmapped = sorted(c for c in emitted if not axes.axis_for_category(c))
        assert not unmapped, f"f-string categories with no axis: {unmapped}"

    def test_fstring_category_sites_are_all_covered(self):
        """Fail when a new f-string category site is added upstream."""
        import re
        source = (Path(ncu_diagnostics.__file__)).read_text()
        sites = set(re.findall(r'category=f"([^"]+)"', source))
        known = {
            'stall_{row[\'key\']}',
            "occupancy_limited_{binding}",
            "uncoalesced_global_{label}",
            "sparse_global_{label}",
            "shared_bank_conflicts_{op}",
            "{unit}_load_imbalance",
        }
        new = sorted(sites - known)
        assert not new, (
            "new f-string category site(s) not expanded in "
            f"test_fstring_categories_map_to_an_axis: {new}"
        )

    def test_axis_lookup_does_not_match_by_accident(self):
        """A wrong axis is worse than none: it makes a gap look covered.

        The substring fallback used to accept a match in either direction, so
        `stall_selected` resolved via `stalls` purely because "selected" starts
        with an s -- and `stall_long_scoreboard`, which does not, resolved to
        nothing at all.
        """
        assert axes.axis_for_category("stall_long_scoreboard") == "stall"
        assert axes.axis_for_category("stall_selected") == "stall"
        # A category that genuinely belongs nowhere must still return "".
        assert axes.axis_for_category("zzz_unrelated_thing") == ""
        assert axes.axis_for_category("") == ''


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


class TestRooflineDtypeBasis:
    """Grading against the wrong precision's peak scales efficiency directly."""

    def _spec(self):
        return gpu_specs.lookup_gpu_spec("H100 SXM") or gpu_specs.lookup_gpu_spec("H100")

    def test_fp8_counters_pick_the_fp8_peak(self):
        spec = self._spec()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView({
            "sm__ops_path_tensor_src_fp8_dst_fp32_sparsity_off.sum": 1e12,
            "dram__bytes.sum": 1e9, "gpu__time_duration.sum": 1e6,
        }), spec)
        assert result["dtype_basis"] == "fp8"
        assert result["dtype_basis_source"] == "tensor_op_counters"

    def test_unknown_precision_reports_the_spread_not_a_guess(self):
        spec = self._spec()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView({
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": 70.0,
            "dram__bytes.sum": 1e9, "gpu__time_duration.sum": 1e6,
        }), spec)
        assert result.get("dtype_ambiguous") is True
        assert result["dtype_basis_source"] == "tensor_pipe_active_precision_unknown"
        assert any("could not be determined" in c for c in result["caveats"])

    def test_plain_fp32_kernel_is_unambiguous(self):
        spec = self._spec()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView({
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9,
            "dram__bytes.sum": 1e9, "gpu__time_duration.sum": 1e6,
        }), spec)
        assert result["dtype_basis"] == "fp32"
        assert not result.get("dtype_ambiguous")


# ---------------------------------------------------------------------------
# Measurement context
# ---------------------------------------------------------------------------

measurement_context = _load("analyzers.measurement_context", "analyzers/measurement_context.py")


class TestMeasurementContext:
    """ncu is cold-cache by default; comparing it to wall-clock is invalid."""

    def test_ncu_default_is_cold_cache(self):
        ctx = measurement_context.describe_collection_mode(source="ncu")
        assert ctx.cache_state == measurement_context.CacheState.COLD
        assert any("--cache-control" in n for n in ctx.notes)

    def test_ncu_cannot_answer_overlap(self):
        ctx = measurement_context.describe_collection_mode(source="ncu")
        assert any("overlap" in c for c in ctx.cannot_answer)

    def test_cache_control_none_is_warm(self):
        ctx = measurement_context.describe_collection_mode(
            source="ncu", cache_control="none")
        assert ctx.cache_state == measurement_context.CacheState.WARM

    def test_cold_vs_warm_comparison_is_refused(self):
        cold = measurement_context.describe_collection_mode(source="ncu")
        warm = measurement_context.describe_collection_mode(source="wallclock")
        out = measurement_context.compare_measurements(
            cold, warm, baseline_value=2.0, candidate_value=1.0)
        assert out["comparable"] is False
        assert out["ratio"] is None, "an invalid ratio must not be presented as a result"
        assert out["uncomparable_raw_ratio"] == pytest.approx(0.5)
        assert any("cache state" in b for b in out["blockers"])

    def test_like_for_like_comparison_is_allowed(self):
        a = measurement_context.describe_collection_mode(source="ncu", clocks_locked=True)
        b = measurement_context.describe_collection_mode(source="ncu", clocks_locked=True)
        out = measurement_context.compare_measurements(
            a, b, baseline_value=2.0, candidate_value=1.5)
        assert out["comparable"] is True
        assert out["ratio"] == pytest.approx(0.75)

    def test_long_unlocked_loop_warns_about_thermals(self):
        ctx = measurement_context.describe_collection_mode(
            source="wallclock", iterations=5000, clocks_locked=False)
        assert any("clock" in n.lower() for n in ctx.notes)

    def test_synthetic_inputs_are_recorded_as_a_limit(self):
        ctx = measurement_context.describe_collection_mode(
            source="wallclock", input_distribution="random")
        assert any("real data" in c for c in ctx.cannot_answer)


class TestCatalogAgainstShippedSections:
    """Ground-truth check when a local Nsight Compute install is present."""

    def test_catalog_names_resolve_or_are_explained(self):
        audit = section_index.audit_catalog_against_sections(metric_catalog.METRIC_CATALOG)
        if not audit.get("available"):
            pytest.skip("no local Nsight Compute install")
        # Section-backed names must dominate; a large unknown set means drift.
        assert len(audit["section_backed"]) > len(audit["unknown"])
        assert audit["shipped_metric_count"] > 500


# ---------------------------------------------------------------------------
# Sampling validity
# ---------------------------------------------------------------------------

sampling_validity = _load("ncu.sampling_validity", "ncu/sampling_validity.py")


class TestPcSamplingValidity:
    """Mirrors NVIDIA's PCSamplingData rule; biased samples must not be used."""

    def test_dropped_samples_block_attribution(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=5000, interval_cycles=1000, dropped_bytes=4096)
        assert out["usable"] is False
        assert "stall_attribution" in out["blocked_conclusions"]
        issue = next(i for i in out["issues"] if i["key"] == "pcsamp_dropped_samples")
        assert "--warp-sampling-interval" in issue["remedy"]

    def test_buffer_overflow_blocks_attribution(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=5000, interval_cycles=1000, buffer_overflow=1,
            buffer_size_bytes=1 << 20)
        assert out["usable"] is False
        assert "--warp-sampling-buffer-size" in out["issues"][0]["remedy"]

    def test_zero_samples_explains_short_kernel(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=0, interval_cycles=100000, kernel_duration_cycles=5000)
        issue = next(i for i in out["issues"] if i["key"] == "pcsamp_no_samples")
        assert "shorter than" in issue["detail"]

    def test_few_samples_block_ranking_but_not_distribution(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=40, interval_cycles=1000)
        assert "hot_line_ranking" in out["blocked_conclusions"]
        assert "sampled_stall_distribution" not in out["blocked_conclusions"]

    def test_healthy_sampling_is_usable(self):
        out = sampling_validity.check_pc_sampling_validity(
            sample_count=50000, interval_cycles=1000, kernel_duration_cycles=10_000_000)
        assert out["usable"] is True and out["blocked_conclusions"] == []

    def test_absent_interval_is_not_reported_as_valid(self):
        out = sampling_validity.check_pc_sampling_validity(sample_count=100)
        assert out["checked"] is False and out["usable"] is None


class TestPmSamplingValidity:
    """Mirrors NVIDIA's PMSamplingData rule, including its architecture gate."""

    def test_unsupported_architecture_is_reported(self):
        out = sampling_validity.check_pm_sampling_validity(cc_major=7, cc_minor=0)
        assert out["supported"] is False
        assert "pm_sampling_timeline" in out["blocked_conclusions"]

    def test_interval_longer_than_workload_blocks_the_timeline(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=2_000_000, duration=1_000_000)
        assert out["usable"] is False
        assert out["interval_duration_ratio"] == pytest.approx(2.0)

    def test_interval_over_ten_percent_is_flagged(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=200_000, duration=1_000_000)
        assert out["usable"] is False
        assert "phase_detection" in out["blocked_conclusions"]

    def test_floor_interval_advises_longer_workload_not_smaller_interval(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=500, duration=600)
        remedy = out["issues"][0]["remedy"]
        assert "longer-running" in remedy

    def test_fine_interval_is_usable(self):
        out = sampling_validity.check_pm_sampling_validity(
            cc_major=9, cc_minor=0, interval=1000, duration=10_000_000)
        assert out["usable"] is True
        assert out["estimated_sample_count"] == pytest.approx(10000)


# ---------------------------------------------------------------------------
# Source correlation
# ---------------------------------------------------------------------------

source_correlation = _load("ncu.source_correlation", "ncu/source_correlation.py")


class _FakeStall:
    def __init__(self, name):
        self.name = name


class _FakeMetric:
    """Mirrors the IMetric surface verified against ncu_report 2026.1.1."""

    def __init__(self, values, correlation=None):
        self._values = values
        self._correlation = correlation

    def num_instances(self):
        return len(self._values)

    def as_double(self, index):
        return float(self._values[index])

    def has_correlation_ids(self):
        return self._correlation is not None

    def correlation_ids(self):
        return _FakeMetric(self._correlation) if self._correlation else None


class _FakeAction:
    """Mirrors the IAction surface: source_info/sass_by_pc/source_files/..."""

    def __init__(self, *, metrics=None, sources=None, lines=None,
                 sass=None, samples=None):
        self._metrics = metrics or {}
        self._sources = sources or {}
        self._lines = lines or {}       # address -> (file, line)
        self._sass = sass or {}
        self._samples = samples or []

    def metric_names(self):
        return list(self._metrics)

    def metric_by_name(self, name):
        return self._metrics.get(name)

    def source_files(self):
        return dict(self._sources)

    def source_info(self, address):
        entry = self._lines.get(address)
        if entry is None:
            return None
        file_name, line = entry

        class _Info:
            def file_name(self_inner):
                return file_name

            def line(self_inner):
                return line

        return _Info()

    def sass_by_pc(self, address):
        return self._sass.get(address, "")

    def ptx_by_pc(self, address):
        return ""

    def timed_warp_samples(self):
        return list(self._samples)


class TestSourceAvailability:
    def test_basic_set_is_diagnosed_as_the_cause(self):
        """The most common cause: a bare ncu run collects no source metrics."""
        report = source_correlation.source_availability(_FakeAction())
        assert report["source_correlation_possible"] is False
        assert any("basic" in r and "SourceCounters" in r
                   for r in report["reasons_unavailable"])

    def test_missing_lineinfo_is_distinguished_from_missing_section(self):
        action = _FakeAction(metrics={
            "sass__inst_executed": _FakeMetric([1.0], correlation=[0x10]),
        })
        report = source_correlation.source_availability(action)
        assert report["source_correlation_possible"] is True
        assert any("-lineinfo" in r for r in report["reasons_unavailable"])

    def test_empty_file_content_suggests_import_source(self):
        action = _FakeAction(
            metrics={"sass__inst_executed": _FakeMetric([1.0], correlation=[0x10])},
            sources={"kernel.cu": ""},
        )
        report = source_correlation.source_availability(action)
        assert any("--import-source" in r for r in report["reasons_unavailable"])


class TestMetricToSourceCorrelation:
    def _action(self):
        return _FakeAction(
            metrics={"sass__inst_executed": _FakeMetric(
                [10.0, 90.0, 5.0], correlation=[0x10, 0x20, 0x30])},
            sources={"kernel.cu": "line one\nline two\nline three\n"},
            lines={0x10: ("kernel.cu", 1), 0x20: ("kernel.cu", 2)},
            sass={0x10: "LDG.E R0", 0x20: "FFMA R2, R0, R1", 0x30: "EXIT"},
        )

    def test_hot_line_is_ranked_first_with_its_source_text(self):
        out = source_correlation.correlate_metric_to_source(
            self._action(), "sass__inst_executed")
        assert out["available"] is True
        top = out["source_lines"][0]
        assert top["line"] == 2 and top["source_text"] == "line two"
        assert "FFMA" in top["sass_samples"][0]

    def test_unlocated_instructions_are_counted_not_hidden(self):
        out = source_correlation.correlate_metric_to_source(
            self._action(), "sass__inst_executed")
        assert out["unlocated_value"] == pytest.approx(5.0)
        assert "could not be tied to a source line" in out["note"]

    def test_whole_kernel_metric_says_so_rather_than_returning_empty(self):
        action = _FakeAction(metrics={"sm__throughput": _FakeMetric([50.0])})
        out = source_correlation.correlate_metric_to_source(action, "sm__throughput")
        assert out["available"] is False
        assert "whole-kernel total" in out["reason"]

    def test_absent_metric_is_reported(self):
        out = source_correlation.correlate_metric_to_source(_FakeAction(), "nope")
        assert out["available"] is False and "not in this report" in out["reason"]


class TestStallAttribution:
    def _action(self):
        samples = (
            [{"timestamp": 1000 + i, "pc": 0x20,
              "stall_reason": _FakeStall("LONG_SCOREBOARD"), "not_issued": True}
             for i in range(30)]
            + [{"timestamp": 2000 + i, "pc": 0x10,
                "stall_reason": _FakeStall("WAIT"), "not_issued": False}
               for i in range(5)]
        )
        return _FakeAction(
            sources={"kernel.cu": "load here\ncompute here\n"},
            lines={0x10: ("kernel.cu", 1), 0x20: ("kernel.cu", 2)},
            sass={0x20: "LDG.E.128 R4"},
            samples=samples,
        )

    def test_stalls_are_attributed_to_the_hottest_line(self):
        out = source_correlation.attribute_stalls_to_source(self._action())
        assert out["available"] is True
        top = out["source_lines"][0]
        assert top["line"] == 2
        assert top["dominant_stall_reason"] == "LONG_SCOREBOARD"
        assert top["source_text"] == "compute here"

    def test_sample_count_confidence_is_stated(self):
        out = source_correlation.attribute_stalls_to_source(self._action())
        assert "statistical" in out["confidence_note"]
        assert out["total_samples"] == 35

    def test_missing_samples_are_explained(self):
        out = source_correlation.attribute_stalls_to_source(_FakeAction())
        assert out["available"] is False
        assert "SourceCounters" in out["reason"] or "set full" in out["reason"]


class TestPcSamplingTimeline:
    def test_phase_change_is_detected_not_averaged_away(self):
        """A kernel memory-bound then compute-bound must not average to 'mediocre'."""
        samples = (
            [{"timestamp": i * 1000, "pc": 0x10,
              "stall_reason": _FakeStall("LONG_SCOREBOARD"), "not_issued": True}
             for i in range(100)]
            + [{"timestamp": 200_000 + i * 1000, "pc": 0x20,
                "stall_reason": _FakeStall("MATH_PIPE_THROTTLE"), "not_issued": False}
               for i in range(100)]
        )
        out = source_correlation.pc_sampling_timeline(
            _FakeAction(samples=samples), bucket_ns=50_000)
        assert out["available"] is True
        assert out["phase_change_count"] >= 1
        assert "LONG_SCOREBOARD" in out["phase_sequence"]
        assert "MATH_PIPE_THROTTLE" in out["phase_sequence"]
        assert "wrong fix for each phase" in out["note"]

    def test_uniform_kernel_says_the_average_is_representative(self):
        samples = [{"timestamp": i * 1000, "pc": 0x10,
                    "stall_reason": _FakeStall("WAIT"), "not_issued": False}
                   for i in range(200)]
        out = source_correlation.pc_sampling_timeline(
            _FakeAction(samples=samples), bucket_ns=50_000)
        assert out["phase_change_count"] == 0
        assert "representative" in out["note"]

    def test_summary_names_its_relationship_to_warpstatestats(self):
        samples = [{"timestamp": 1, "pc": 0x10,
                    "stall_reason": _FakeStall("BARRIER"), "not_issued": True}]
        out = source_correlation.summarize_warp_samples(_FakeAction(samples=samples))
        assert "WarpStateStats" in out["comparison_note"]


class TestHierarchicalRoofline:
    """Three arithmetic intensities sharing a numerator; the spread is the signal."""

    _FLOPS = {"smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9}

    def test_leaking_l1_is_reported(self):
        """L1 and L2 intensities close together means L1 catches no reuse."""
        result = ncu_diagnostics.hierarchical_roofline(ncu_diagnostics.MetricView({
            **self._FLOPS,
            "l1tex__t_sectors.sum": 1e6,
            "lts__t_sectors.sum": 0.9e6,
            "dram__bytes.sum": 1e9,
        }))
        assert result["available"] is True
        assert result["l1_to_l2_intensity_ratio"] < 1.5
        assert any("L1 is not capturing reuse" in f.title for f in result["findings"])

    def test_healthy_hierarchy_advises_against_tiling_work(self):
        result = ncu_diagnostics.hierarchical_roofline(ncu_diagnostics.MetricView({
            **self._FLOPS,
            "l1tex__t_sectors.sum": 1e6,
            "lts__t_sectors.sum": 1e5,
            "dram__bytes.sum": 1e6,
        }))
        assert result["locality_verdict"] == "healthy"
        finding = next(f for f in result["findings"] if "already capturing" in f.title)
        assert "Do not spend effort" in finding.actions[0]

    def test_bytes_derived_from_sectors_when_byte_counters_absent(self):
        result = ncu_diagnostics.hierarchical_roofline(ncu_diagnostics.MetricView({
            **self._FLOPS, "l1tex__t_sectors.sum": 1000.0,
        }))
        assert result["levels"]["l1"]["bytes"] == pytest.approx(32000.0)
        assert "x 32" in result["levels"]["l1"]["byte_source"]

    def test_direct_byte_counter_is_preferred(self):
        result = ncu_diagnostics.hierarchical_roofline(ncu_diagnostics.MetricView({
            **self._FLOPS, "l1tex__t_bytes.sum": 4096.0, "l1tex__t_sectors.sum": 1000.0,
        }))
        assert result["levels"]["l1"]["bytes"] == pytest.approx(4096.0)
        assert result["levels"]["l1"]["byte_source"] == "byte counter"

    def test_shared_memory_caveat_fires_for_tiled_kernels(self):
        """l1tex__t_bytes excludes shared traffic -- critical for attention/GEMM."""
        result = ncu_diagnostics.hierarchical_roofline(ncu_diagnostics.MetricView({
            **self._FLOPS,
            "l1tex__t_sectors.sum": 1e6,
            "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": 5e5,
        }))
        assert any("shared-memory traffic" in c for c in result["caveats"])
        assert any("overestimate" in c for c in result["caveats"])

    def test_missing_flops_reports_why_not_empty(self):
        result = ncu_diagnostics.hierarchical_roofline(ncu_diagnostics.MetricView({
            "l1tex__t_sectors.sum": 1e6,
        }))
        assert result["available"] is False
        assert "FLOP counters" in result["reason"]


class TestSolThresholdsMatchShippedRule:
    """Pinned to NVIDIA's SpeedOfLight.py so drift is caught, not debated."""

    def test_four_thresholds_match_nvidia(self):
        t = ncu_diagnostics.SOL_THRESHOLDS
        assert t["balanced_delta"] == 10.0     # balanced_threshold
        assert t["latency_bound"] == 60.0      # latency_bound_threshold
        assert t["saturated"] == 80.0          # no_bound_threshold
        assert t["waves_small_grid"] == 1.0    # waves_threshold

    def test_unverified_two_axis_table_is_gone(self):
        t = ncu_diagnostics.SOL_THRESHOLDS
        assert "compute_bound_compute" not in t
        assert "compute_bound_memory" not in t


class TestDocsQuoteRealCounts:
    """Counts cited in the docs drift silently as code is added.

    Two had already drifted when this was written: the section-backed catalog
    split, and `trace_quality.py (12 checks)` when there were 13 -- the handbook
    table was also missing a row. A number in prose is a claim like any other.
    """

    def _docs(self):
        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling" / "docs"
        out = {}
        for name in ("PERFORMANCE_ANALYSIS_HANDBOOK.md", "CAPABILITY_EVOLUTION.md"):
            path = root / name
            if path.exists():
                out[name] = path.read_text()
        return out

    def test_cited_counts_match_the_code(self):
        import re
        counts = {
            f"{len(metric_catalog.METRIC_CATALOG)} metrics": True,
            f"{len(axes.AXES)} axes": True,
            f"{len(metric_catalog.STALL_REASONS)} stall reasons": True,
            f"of {len(ncu_diagnostics._ANALYSIS_REQUIREMENTS)} analyses": True,
        }
        checks = len([n for n in dir(trace_quality) if n.startswith("check_")])
        counts[f"({checks} checks)"] = True

        wrong = []
        for name, text in self._docs().items():
            # Any "<n> metrics"/"<n> axes"/... that is NOT the real number is stale.
            # Anchored to the phrasings that state a fact about this codebase.
            # A bare "<n> metrics" also appears in example output and in
            # unrelated thresholds, so matching that loosely produces noise.
            for pattern, real in (
                (r"catalog interprets (\d+) metrics", len(metric_catalog.METRIC_CATALOG)),
                (r"metric_catalog\.py` [^\n]*?(\d+) metrics", len(metric_catalog.METRIC_CATALOG)),
                (r"`METRIC_CATALOG` \((\d+) metrics\)", len(metric_catalog.METRIC_CATALOG)),
                (r"(\d+) axes\b", len(axes.AXES)),
                (r"(\d+) stall reasons\b", len(metric_catalog.STALL_REASONS)),
                (r"of (\d+) analyses\b", len(ncu_diagnostics._ANALYSIS_REQUIREMENTS)),
                (r"\((\d+) checks\)", checks),
            ):
                for found in re.findall(pattern, text):
                    if int(found) != real:
                        wrong.append(f"{name}: '{pattern}' cites {found}, code says {real}")
        assert not wrong, "docs cite stale counts: " + "; ".join(wrong)

    def test_trace_quality_table_lists_every_check(self):
        """The 9c table must not silently omit a check."""
        text = self._docs().get("PERFORMANCE_ANALYSIS_HANDBOOK.md", "")
        if not text:
            pytest.skip("handbook not present")
        documented = set(re.findall(r"\| `(check_\w+)`", text))
        actual = {n for n in dir(trace_quality) if n.startswith("check_")}
        missing = sorted(actual - documented)
        assert not missing, f"checks implemented but absent from the 9c table: {missing}"


class TestNoOrphanedAnalysisModules:
    """An analysis module with no caller is dead weight that looks like coverage.

    This has happened repeatedly: throttling, nccl_bandwidth, trace_quality and
    distributed_alignment were each complete, exported, tested -- and invoked by
    nothing, so the axes they cover silently reported as gaps. measurement_context
    was added in the same session that fixed the others and immediately repeated
    the mistake. Exported and tested is not the same as reachable.
    """

    _ENTRY_POINTS = (
        "ncu/ncu_diagnostics.py",
        "ncu/ncu_report_tools.py",
        "sources/nsys_auto_analysis.py",
        "analyzers/metrics_analyzer.py",
    )

    def test_analysis_modules_are_reachable_from_an_entry_point(self):
        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        entry_text = "\n".join(
            (root / rel).read_text() for rel in self._ENTRY_POINTS if (root / rel).exists()
        )
        # Modules that must be invoked, not merely importable.
        required = [
            "analyzers/axes.py",
            "analyzers/measurement_context.py",
            "analyzers/trace_quality.py",
            "hardware/throttling.py",
            "ncu/shipped_rules.py",
            "ncu/source_correlation.py",
            "ncu/sampling_validity.py",
            "ncu/section_index.py",
        ]
        orphans = []
        for rel in required:
            stem = Path(rel).stem
            if stem not in entry_text:
                orphans.append(rel)
        assert not orphans, (
            "analysis modules with no caller in any entry point "
            f"(exported and tested is not reachable): {orphans}"
        )


class TestSubpackagesImportInAnyOrder:
    """`import my_utils.profiling.sources` used to fail in a fresh interpreter.

    sources.nsys_sqlite_provider imported ..metrics, whose __init__ imported
    metrics_providers, which imported back into sources while it was still
    initialising. Normal use never hit it because profiling/__init__ happens to
    import .metrics before .sources -- which made that ordering load-bearing and
    undocumented. The re-exports are now resolved lazily.
    """

    def test_every_subpackage_imports_first(self):
        import importlib
        import importlib.util as iu

        root = Path(__file__).resolve().parents[2] / "my_utils" / "profiling"
        failures = []
        for first in ("sources", "metrics", "analyzers", "ncu", "hardware"):
            for mod in [m for m in sys.modules if m.startswith("my_utils")]:
                del sys.modules[mod]
            pkg = types.ModuleType("my_utils")
            pkg.__path__ = [str(root.parent)]
            sys.modules["my_utils"] = pkg
            spec = iu.spec_from_file_location(
                "my_utils.profiling", root / "__init__.py",
                submodule_search_locations=[str(root)])
            prof = iu.module_from_spec(spec)
            sys.modules["my_utils.profiling"] = prof
            pkg.profiling = prof
            try:
                importlib.import_module(f"my_utils.profiling.{first}")
                metrics = importlib.import_module("my_utils.profiling.metrics")
                assert metrics.NsysSqliteMetricsProvider is not None
            except Exception as exc:
                failures.append(f"{first} first: {type(exc).__name__}: {exc}")
            finally:
                for mod in [m for m in sys.modules if m.startswith("my_utils")]:
                    del sys.modules[mod]
        assert not failures, "subpackage import order is load-bearing: " + "; ".join(failures)


class TestReportDiagnosisUsesShippedRules:
    """The CLI path must cross-check against NVIDIA's rules, not just the API.

    `diagnose_kernel` accepted `shipped_rules=` from the start, but
    `diagnose_ncu_report` -- what `mp ncu-diagnose` actually runs -- never
    passed them, so corroboration reported "no shipped rules" on every real
    report. A feature reachable only from a hand-written call is not reachable.
    """

    def _fake_module(self, with_rules=True):
        class M:
            def __init__(self, v): self.v = v
            def value(self): return self.v
            def as_double(self): return self.v
            def as_uint64(self): return int(self.v)
            def unit(self): return ""
            def has_correlation_ids(self): return False

        values = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 30.0,
        }

        class Action:
            def name(self): return "gemm_kernel"
            def metric_names(self): return list(values)
            def metric_by_name(self, k): return M(values.get(k, 0.0))
            def rule_results_as_dicts(self):
                if not with_rules:
                    return []
                return [{
                    "rule_identifier": "SOLBottleneck",
                    "section_identifier": "SpeedOfLight",
                    "rule_message": {"title": "Memory more utilized",
                                     "message": "This kernel is memory bound.",
                                     "message_type": "warning"},
                    "speedup_estimation": {"type": "GLOBAL", "speedup": 25.0},
                }]

        class Rng:
            num_actions = 1
            def action_by_idx(self, i): return Action()

        class Ctx:
            num_ranges = 1
            def range_by_idx(self, i): return Rng()

        return types.SimpleNamespace(load_report=lambda p: Ctx())

    def _first(self, out):
        return (out.get("kernels") or out.get("diagnoses"))[0]

    def test_shipped_rules_reach_the_diagnosis(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._fake_module())
        assert self._first(out)["corroboration"]["shipped_rules_available"] is True

    def test_disagreement_with_nvidia_surfaces_through_the_report_path(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._fake_module())
        assert self._first(out)["corroboration"]["conflicts"]

    def test_report_without_rules_is_reported_honestly(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._fake_module(with_rules=False))
        assert self._first(out)["corroboration"]["shipped_rules_available"] is False


class TestDiagnoseIsSelfContained:
    """`ncu-diagnose` must answer both what and where, in one command.

    Source attribution used to be reachable only through a separate SkillEngine
    call, so the question a fused kernel most needs answered -- which line
    stalls -- was absent from the command people actually run.
    """

    def _module(self, with_samples=True):
        class Stall:
            def __init__(self, n): self.name = n

        class M:
            def __init__(self, v): self.v = v
            def value(self): return self.v
            def as_double(self): return self.v
            def as_uint64(self): return int(self.v)
            def unit(self): return ""
            def has_correlation_ids(self): return False

        vals = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 32.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 28.0,
            "smsp__pcsamp_sample_count": 5000.0,
            "smsp__pcsamp_interval_cycles": 1000.0,
        }

        class Action:
            def name(self): return "fused_attn_fwd"
            def metric_names(self): return list(vals)
            def metric_by_name(self, k): return M(vals[k]) if k in vals else None
            def rule_results_as_dicts(self): return []
            def source_files(self): return {"attn.cu": "load\nsoftmax\nmm\n"}
            def source_info(self, a):
                table = {0x10: ("attn.cu", 1), 0x20: ("attn.cu", 2)}
                if a not in table:
                    return None
                fname, ln = table[a]

                class I:
                    def file_name(self): return fname
                    def line(self): return ln
                return I()
            def sass_by_pc(self, a): return ""
            def ptx_by_pc(self, a): return ""
            def timed_warp_samples(self):
                if not with_samples:
                    return []
                return [{"timestamp": i * 100, "pc": 0x20,
                         "stall_reason": Stall("MIO_THROTTLE"), "not_issued": True}
                        for i in range(600)]

        class Rng:
            num_actions = 1
            def action_by_idx(self, i): return Action()

        class Ctx:
            num_ranges = 1
            def range_by_idx(self, i): return Rng()

        return types.SimpleNamespace(load_report=lambda p: Ctx())

    def test_source_attribution_is_in_the_diagnosis(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._module())
        kernel = out["kernels"][0]
        assert "source_attribution" in kernel
        rows = kernel["source_attribution"]["stall_attribution"]["source_lines"]
        assert rows and rows[0]["line"] == 2

    def test_markdown_renders_where_it_stalls(self):
        text = ncu_report_tools.diagnose_result_to_markdown(
            ncu_report_tools.diagnose_ncu_report(
                "/dev/null", ncu_report_module=self._module()))
        assert "### Where it stalls" in text
        assert "MIO_THROTTLE" in text

    def test_no_contradiction_when_attribution_succeeds(self):
        """Do not print 'no source data' directly beneath the source data."""
        text = ncu_report_tools.diagnose_result_to_markdown(
            ncu_report_tools.diagnose_ncu_report(
                "/dev/null", ncu_report_module=self._module()))
        assert "### Where it stalls" in text
        assert "No source-correlated metrics" not in text

    def test_include_source_false_skips_it(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", include_source=False, ncu_report_module=self._module())
        assert "source_attribution" not in out["kernels"][0]

    def test_absent_samples_do_not_break_the_diagnosis(self):
        out = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=self._module(with_samples=False))
        kernel = out["kernels"][0]
        assert kernel["verdict"]
        assert kernel["source_attribution"]["stall_attribution"]["available"] is False


signal_scan = _load("ncu.signal_scan", "ncu/signal_scan.py")
source_correlation_mod = _load("ncu.source_correlation", "ncu/source_correlation.py")


class TestSignalScanOverAllMetrics:
    """Reason about metrics no curated rule covers, without inventing noise."""

    def test_saturated_uncatalogued_unit_is_found(self):
        out = signal_scan.scan_all_signals({
            "idc__request_cycles_active.avg.pct_of_peak_sustained_elapsed": 91.0,
        })
        assert any(f.category == "unit_saturated" for f in out["findings"])

    def test_units_with_curated_rules_are_not_duplicated(self):
        out = signal_scan.scan_all_signals({
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 95.0,
        })
        assert not [f for f in out["findings"] if f.category == "unit_saturated"]

    def test_bursty_unit_is_explained_rather_than_called_idle(self):
        out = signal_scan.scan_all_signals({
            "tpc__warps_active.avg.pct_of_peak_sustained_active": 96.0,
            "tpc__warps_active.avg.pct_of_peak_sustained_elapsed": 8.0,
        })
        finding = next(f for f in out["findings"] if f.category == "unit_duty_cycle")
        assert "Both" in finding.summary and "correct" in finding.summary

    def test_percentage_above_100_is_a_measurement_fault(self):
        out = signal_scan.scan_all_signals({
            "fbpa__x.avg.pct_of_peak_sustained_elapsed": 118.0,
        })
        finding = next(f for f in out["findings"]
                       if f.category == "measurement_above_physical_limit")
        assert finding.severity == "high"

    def test_hit_rate_is_not_read_as_utilisation(self):
        """A 95% hit rate must not be reported as a saturated unit."""
        out = signal_scan.scan_all_signals({
            "lts__t_sector_hit_rate.pct": 95.0,
        })
        assert not [f for f in out["findings"] if f.category == "unit_saturated"]

    def test_quiet_report_produces_nothing(self):
        out = signal_scan.scan_all_signals({
            "idc__request_cycles_active.avg.pct_of_peak_sustained_elapsed": 12.0,
        })
        assert out["findings"] == []


class TestSignalToSourceLinkage:
    """Join what-is-wrong to where-it-happens, and refuse invented joins."""

    def _attribution(self):
        return {
            "available": True,
            "stall_reasons": {"LONG_SCOREBOARD": 700, "MATH_PIPE_THROTTLE": 100},
            "source_lines": [
                {"file_name": "attn.cu", "line": 1, "source_text": "load_qkv",
                 "samples": 700, "stall_reasons": {"LONG_SCOREBOARD": 700},
                 "sass_samples": ["LDG.E.128"]},
                {"file_name": "attn.cu", "line": 3, "source_text": "mm(p,v)",
                 "samples": 100, "stall_reasons": {"MATH_PIPE_THROTTLE": 100},
                 "sass_samples": []},
            ],
        }

    def _link(self, category):
        return source_correlation_mod.link_findings_to_source(
            [{"category": category, "title": "t"}], None,
            attribution=self._attribution())

    def test_memory_finding_lands_on_the_loading_line(self):
        out = self._link("uncoalesced_global_load")
        assert out["linked"], "f-string category must match by prefix"
        row = out["linked"][0]["source_lines"][0]
        assert row["line"] == 1 and row["share_of_reason"] == pytest.approx(1.0)

    def test_stall_category_localises_itself(self):
        out = self._link("stall_long_scoreboard")
        assert out["linked"][0]["source_lines"][0]["line"] == 1

    def test_generic_scan_findings_are_not_linked(self):
        """A saturated constant cache has no known relation to any stall reason."""
        assert self._link("unit_saturated")["linked"] == []
        assert self._link("unit_duty_cycle")["linked"] == []

    def test_launch_geometry_findings_are_not_linked(self):
        """Grid shape is a property of the launch, not of a line."""
        assert self._link("small_grid")["linked"] == []
        assert self._link("tile_quantization")["linked"] == []

    def test_measurement_faults_are_not_linked(self):
        assert self._link("measurement_above_physical_limit")["linked"] == []

    def test_link_carries_its_own_caveat(self):
        out = self._link("uncoalesced_global_load")
        assert "not proof of cause" in out["linked"][0]["caveat"]

    def test_absent_attribution_is_reported_not_faked(self):
        out = source_correlation_mod.link_findings_to_source(
            [{"category": "uncoalesced_global_load", "title": "t"}], None,
            attribution={"available": False, "reason": "no samples"})
        assert out["available"] is False and out["linked"] == []


class TestNcuReportModuleDiscovery:
    """`ncu_report` ships with Nsight Compute, not on PyPI.

    The first-run failure for anyone with a real report is an ImportError whose
    fix is a PYTHONPATH entry, not a pip install. Discovery removes the step
    where it can be found; the error message covers when it cannot.
    """

    def test_discovery_returns_a_dir_or_none_never_a_guess(self):
        found = ncu_report_tools.find_ncu_report_dir()
        assert found is None or (found / "ncu_report.py").exists()

    def test_error_message_is_actionable(self):
        import re
        source = Path(ncu_report_tools.__file__).read_text()
        block = source.split("The `ncu_report` module is required")[1][:1200]
        assert "PYTHONPATH" in block
        assert "not on PyPI" in block or "nothing to" in block
        assert "find /" in block, "must tell the user how to locate it"


class TestReportIsReadOnce:
    """Four full traversals of a --set full report was three too many.

    `diagnose_ncu_report` called four loaders -- metrics, shipped rules, source
    attribution, and one just to retain the action objects -- each of which
    opened the report and walked every range and action. `walk_report_once`
    visits each action a single time and gathers all four.
    """

    def _counting_module(self):
        opens = {"n": 0}

        class Stall:
            def __init__(self, n): self.name = n

        class M:
            def __init__(self, v): self.v = v
            def value(self): return self.v
            def as_double(self): return self.v
            def as_uint64(self): return int(self.v)
            def unit(self): return ""
            def has_correlation_ids(self): return False

        vals = {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 34.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 72.0,
            "smsp__pcsamp_sample_count": 8000.0,
            "smsp__pcsamp_interval_cycles": 1000.0,
            "gpu__time_duration.sum": 410000.0,
        }

        class Action:
            def name(self): return "k"
            def metric_names(self): return list(vals)
            def metric_by_name(self, k): return M(vals[k]) if k in vals else None
            def rule_results_as_dicts(self):
                return [{"rule_identifier": "SOLBottleneck",
                         "rule_message": {"title": "t", "message": "m",
                                          "message_type": "optimization"},
                         "speedup_estimation": {"type": "GLOBAL", "speedup": 20.0}}]
            def source_files(self): return {"k.cu": "a\nb\n"}
            def source_info(self, a):
                if a != 0x10:
                    return None

                class I:
                    def file_name(self): return "k.cu"
                    def line(self): return 1
                return I()
            def sass_by_pc(self, a): return ""
            def ptx_by_pc(self, a): return ""
            def timed_warp_samples(self):
                return [{"timestamp": i, "pc": 0x10,
                         "stall_reason": Stall("LONG_SCOREBOARD"), "not_issued": True}
                        for i in range(400)]

        class Rng:
            num_actions = 1
            def action_by_idx(self, i): return Action()

        class Ctx:
            num_ranges = 1
            def range_by_idx(self, i): return Rng()

        def loader(path):
            opens["n"] += 1
            return Ctx()

        return types.SimpleNamespace(load_report=loader), opens

    def test_report_is_opened_exactly_once(self):
        module, opens = self._counting_module()
        ncu_report_tools.diagnose_ncu_report("/dev/null", ncu_report_module=module)
        assert opens["n"] == 1, f"report opened {opens['n']} times; expected 1"

    def test_single_pass_still_produces_every_section(self):
        module, _ = self._counting_module()
        kernel = ncu_report_tools.diagnose_ncu_report(
            "/dev/null", ncu_report_module=module)["kernels"][0]
        for key in ("verdict", "coverage", "axes", "metric_inventory",
                    "corroboration", "signal_scan", "source_attribution",
                    "duration_ns"):
            assert key in kernel, f"single-pass rewrite dropped `{key}`"
        assert kernel["corroboration"]["shipped_rules_available"] is True
        assert kernel["source_attribution"]["stall_attribution"]["available"] is True

    def test_no_source_still_reads_once(self):
        module, opens = self._counting_module()
        ncu_report_tools.diagnose_ncu_report(
            "/dev/null", include_source=False, ncu_report_module=module)
        assert opens["n"] == 1


class TestRealReportRegressions:
    """Three bugs a real H100 report exposed that no fixture had.

    Each survived because my fixtures happened to use the shape my code
    assumed: raw warp samples where a real report has aggregated metrics, and a
    plain dict where ncu_report hands back a SWIG map.
    """

    class _SwigLikeMap:
        """Supports keys()/__getitem__ but is NOT a collections.abc.Mapping.

        This is what `IAction.source_files()` actually returns. An
        `isinstance(..., Mapping)` guard discarded all 64 source files on a real
        report and then reported the kernel as built without -lineinfo.
        """

        def __init__(self, data):
            self._data = dict(data)

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    def test_swig_map_is_not_discarded(self):
        files = source_correlation._as_dict(
            self._SwigLikeMap({"k.cu": "line one\nline two\n"}))
        assert files == {"k.cu": "line one\nline two\n"}
        assert not isinstance(self._SwigLikeMap({}), __import__("collections.abc",
                              fromlist=["Mapping"]).Mapping), "guard premise"

    class _CorrelatedAction:
        """A report with aggregated pcsamp metrics and no raw warp samples.

        This is the normal `--set full` shape: `timed_warp_samples()` empty,
        `smsp__pcsamp_warps_issue_stalled_*` present with correlation IDs.
        """

        _PREFIX = "smsp__pcsamp_warps_issue_stalled_"

        class _Metric:
            def __init__(self, values, correlation=None):
                self._values = values
                self._correlation = correlation

            def num_instances(self): return len(self._values)
            def as_double(self, i): return float(self._values[i])
            def as_uint64(self, i): return int(self._values[i])
            def has_correlation_ids(self): return self._correlation is not None

            def correlation_ids(self):
                if self._correlation is None:
                    return None
                return TestRealReportRegressions._CorrelatedAction._Metric(
                    self._correlation)

        def metric_names(self):
            return [self._PREFIX + "long_scoreboard",
                    self._PREFIX + "long_scoreboard_not_issued",
                    self._PREFIX + "barrier"]

        def metric_by_name(self, name):
            M = TestRealReportRegressions._CorrelatedAction._Metric
            if name.endswith("_not_issued"):
                return M([999.0], correlation=[0x10])   # must be ignored
            if name.endswith("long_scoreboard"):
                return M([100.0, 20.0], correlation=[0x10, 0x20])
            if name.endswith("barrier"):
                return M([50.0], correlation=[0x20])
            return None

        def source_files(self):
            return TestRealReportRegressions._SwigLikeMap(
                {"k.cu": "load line\ncompute line\n"})

        def source_info(self, address):
            table = {0x10: ("k.cu", 1), 0x20: ("k.cu", 2)}
            if address not in table:
                return None
            name, line = table[address]

            class _Info:
                def file_name(self): return name
                def line(self): return line
            return _Info()

        def sass_by_pc(self, address): return "LDG.E"
        def ptx_by_pc(self, address): return ""
        def timed_warp_samples(self): return []

    def test_attribution_works_without_raw_warp_samples(self):
        out = source_correlation.attribute_stalls_to_source(self._CorrelatedAction())
        assert out["available"] is True
        assert out["source"] == "correlated pcsamp metrics"
        top = out["source_lines"][0]
        assert top["line"] == 1 and top["dominant_stall_reason"] == "LONG_SCOREBOARD"
        assert top["source_text"] == "load line"

    def test_not_issued_variants_are_not_double_counted(self):
        out = source_correlation.attribute_stalls_to_source(self._CorrelatedAction())
        # 100 + 20 long_scoreboard + 50 barrier == 170; the 999 _not_issued
        # counts the same stalls again and must be excluded.
        assert out["total_samples"] == 170

    def test_hit_rate_with_no_traffic_is_suppressed(self):
        """0% hit rate over 0 requests is not a result."""
        out = signal_scan.scan_all_signals({
            "l1tex__t_sector_pipe_lsu_mem_global_op_red_hit_rate.pct": 0.0,
            "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum": 0.0,
        })
        assert not [f for f in out["findings"] if f.category == "unit_hit_rate"]
        assert out["hit_rates_skipped_no_traffic"] == 1

    def test_hit_rate_with_traffic_is_reported_with_its_volume(self):
        out = signal_scan.scan_all_signals({
            "l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate.pct": 0.0,
            "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": 49152.0,
        })
        finding = next(f for f in out["findings"] if f.category == "unit_hit_rate")
        assert finding.evidence["requests_behind_it"] == 49152.0
        assert "more than 100%" not in finding.summary, "100 - 0 rendered as prose"
        assert "100% of the 49,152 requests" in finding.summary

    def test_scan_hit_rate_findings_are_not_linked_to_source(self):
        """The scan knows a path missed, not which lines use that path."""
        out = source_correlation.link_findings_to_source(
            [{"category": "unit_hit_rate", "title": "t"}], None,
            attribution={"available": True, "stall_reasons": {"LONG_SCOREBOARD": 100},
                         "source_lines": [{"file_name": "k.cu", "line": 1,
                                           "samples": 100,
                                           "stall_reasons": {"LONG_SCOREBOARD": 100}}]})
        assert out["linked"] == []

    def test_identical_linkages_are_folded(self):
        attribution = {
            "available": True, "stall_reasons": {"LONG_SCOREBOARD": 100},
            "source_lines": [{"file_name": "k.cu", "line": 1, "samples": 100,
                              "stall_reasons": {"LONG_SCOREBOARD": 100}}],
        }
        # Both resolve to exactly ("LONG_SCOREBOARD",), so same reasons AND
        # same lines. Findings whose reasons differ are NOT folded, because the
        # same line stalling for a second reason is a second fact.
        out = source_correlation.link_findings_to_source(
            [{"category": "poor_cache_locality", "title": "first"},
             {"category": "l2_load_imbalance", "title": "second"}],
            None, attribution=attribution)
        assert len(out["linked"]) == 1
        assert out["duplicate_links"] and out["duplicate_note"]

    def test_different_reasons_on_the_same_lines_are_kept(self):
        attribution = {
            "available": True,
            "stall_reasons": {"LONG_SCOREBOARD": 100, "LG_THROTTLE": 10},
            "source_lines": [{"file_name": "k.cu", "line": 1, "samples": 110,
                              "stall_reasons": {"LONG_SCOREBOARD": 100,
                                                "LG_THROTTLE": 10}}],
        }
        out = source_correlation.link_findings_to_source(
            [{"category": "poor_cache_locality", "title": "locality"},
             {"category": "register_spilling", "title": "spilling"}],
            None, attribution=attribution)
        assert len(out["linked"]) == 2, (
            "same lines but different mechanisms is two findings, not one")


class TestStringValuedMetrics:
    """21 metrics on a real report have string values, and they are not noise.

    They were dropped as unparseable. Among them: the GPU model (which the
    caller was being asked to supply by hand), the constituent lists behind each
    Speed-of-Light rollup, and the launch scheduling policy.
    """

    class _Action:
        _NUM = {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 33.7,
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 33.7,
                "sm__issue_active.avg.pct_of_peak_sustained_elapsed": 14.1}
        _STR = {"device__attribute_display_name": "NVIDIA H100 80GB HBM3",
                "launch__cluster_scheduling_policy": "PolicySpread",
                "breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed":
                    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
                    "sm__issue_active.avg.pct_of_peak_sustained_elapsed"}

        class _M:
            def __init__(self, v): self.v = v
            def value(self): return self.v
            def as_double(self): return self.v if isinstance(self.v, float) else None
            def as_string(self): return self.v if isinstance(self.v, str) else None
            def unit(self): return ""

        def name(self): return "k"
        def metric_names(self): return list(self._NUM) + list(self._STR)
        def metric_by_name(self, n):
            if n in self._NUM: return self._M(self._NUM[n])
            if n in self._STR: return self._M(self._STR[n])
            return None

    def test_string_metrics_are_kept_not_dropped(self):
        numeric, text = ncu_report_tools._metrics_for_action(self._Action())
        assert len(numeric) == 3 and len(text) == 3
        assert text["device__attribute_display_name"] == "NVIDIA H100 80GB HBM3"

    def test_gpu_name_comes_from_the_report(self):
        _, text = ncu_report_tools._metrics_for_action(self._Action())
        assert ncu_report_tools.gpu_name_from_report(text) == "NVIDIA H100 80GB HBM3"

    def test_sol_breakdown_names_the_driving_constituent(self):
        """A SOL throughput is a max over constituents, not an average."""
        numeric, text = ncu_report_tools._metrics_for_action(self._Action())
        out = ncu_report_tools.resolve_sol_breakdown(text, numeric)
        entry = out["sm__throughput.avg.pct_of_peak_sustained_elapsed"]
        assert entry["rollup_value"] == pytest.approx(33.7)
        top = entry["top_constituents"][0]
        assert "pipe_tensor" in top["metric"], "the max constituent drives the rollup"
        assert "maximum over these" in entry["note"]

    def test_inventory_counts_string_metrics(self):
        result = ncu_diagnostics.diagnose_kernel(
            {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 33.7},
            string_metrics={"device__attribute_display_name": "NVIDIA H100"},
        )
        inventory = result["metric_inventory"]
        assert inventory["string_valued_count"] == 1
        assert inventory["total_including_string"] == inventory["total"] + 1
        assert "not lost" in inventory["summary"]

    def test_breakdown_with_unresolvable_constituents_is_skipped(self):
        out = ncu_report_tools.resolve_sol_breakdown(
            {"breakdown:x": "not_collected_a,not_collected_b"}, {})
        assert out == {}


class TestInstructionAndPmSampling:
    """Instruction-level attribution, and the PM-sampling timeline."""

    class _Action:
        _P = "smsp__pcsamp_warps_issue_stalled_"
        _PM = "pmsampling:"

        class _M:
            def __init__(self, values, correlation=None):
                self.values = values
                self._c = correlation
            def num_instances(self): return len(self.values)
            def as_double(self, i): return float(self.values[i])
            def as_uint64(self, i): return int(self.values[i])
            def has_correlation_ids(self): return self._c is not None
            def correlation_ids(self):
                if self._c is None: return None
                return TestInstructionAndPmSampling._Action._M(self._c)

        def metric_names(self):
            return [self._P + "long_scoreboard",
                    self._PM + "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                    self._PM + "sm__cycles_active.avg"]

        def metric_by_name(self, n):
            M = TestInstructionAndPmSampling._Action._M
            if n.endswith("long_scoreboard"):
                return M([600.0, 100.0], correlation=[0x10, 0x20])
            if n.endswith("pct_of_peak_sustained_elapsed"):
                # bursty: high peak, low average
                return M([0.0, 0.0, 94.0, 90.0, 0.0, 0.0],
                         correlation=[1000, 2500, 4000, 5500, 7000, 8500])
            if n.endswith("sm__cycles_active.avg"):
                return M([0.0, 2964.0, 100.0, 0.0, 0.0, 0.0],
                         correlation=[1000, 2500, 4000, 5500, 7000, 8500])
            return None

        def source_files(self): return {"k.cu": "convert line\nload line\n"}
        def source_info(self, a):
            t = {0x10: ("k.cu", 1), 0x20: ("k.cu", 2)}
            if a not in t: return None
            f, l = t[a]
            class I:
                def file_name(self): return f
                def line(self): return l
            return I()
        def sass_by_pc(self, a):
            return {0x10: "PRMT R19, R8, 0x7732, RZ", 0x20: "LDG.E.128 R4"}.get(a, "")
        def ptx_by_pc(self, a): return ""
        def timed_warp_samples(self): return []

    def test_top_instruction_carries_its_sass(self):
        out = source_correlation.top_stalling_instructions(self._Action())
        assert out["available"] is True
        top = out["instructions"][0]
        assert top["sass"] == "PRMT R19, R8, 0x7732, RZ"
        assert top["line"] == 1 and top["dominant_stall_reason"] == "LONG_SCOREBOARD"
        assert top["address_hex"] == "0x10"

    def test_instruction_view_is_finer_than_the_line_view(self):
        out = source_correlation.top_stalling_instructions(self._Action())
        assert out["distinct_instructions"] == 2
        assert "finer than the line view" in out["note"]

    def test_pm_sampling_reports_peak_against_the_active_window(self):
        """The denominator must be the window the kernel ran in.

        Averaging over the sampler's whole session divides by a lot of time the
        kernel was not running. On a real report that turned an 81us kernel into
        a 216us "active window" and reported the tensor pipe at 8.7% when the
        figure over the kernel's own window is 32.3% -- which matches the
        whole-kernel counter, as it should.
        """
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert out["available"] is True
        tensor = next(e for e in out["series"] if "tensor" in e["metric"])
        assert tensor["peak"] == pytest.approx(94.0)
        assert tensor["mean_in_active_window"] > tensor["mean_all_buckets"], (
            "the window average must exceed the whole-session average")
        assert out["window_source"] == "sm__cycles_active"

    def test_duty_cycle_is_counted_inside_the_window(self):
        """Counting non-zero buckets series-wide over a window denominator
        produced shares above 100% for DRAM, which is active either side of the
        launch."""
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert all(0.0 <= e["duty_cycle"] <= 1.0 for e in out["series"])

    def test_denominators_are_named_in_the_output(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert "mean_all_buckets" in out["denominator_note"]
        assert "not for comparison" in out["denominator_note"]

    def test_raw_counts_are_not_rendered_as_percentages(self):
        """`sm__cycles_active.avg` is a count; "2964%" is nonsense."""
        out = source_correlation.analyze_pm_sampling(self._Action())
        counts = [e for e in out["series"] if not e["is_percentage"]]
        assert counts and counts[0]["unit"] == "count"
        # and a count must never be called bursty, having no ceiling
        assert all(e["is_percentage"] for e in out["bursty"])

    def test_pm_sampling_absent_is_explained(self):
        class _Bare:
            def metric_names(self): return ["sm__throughput.avg"]
            def metric_by_name(self, n): return None
        out = source_correlation.analyze_pm_sampling(_Bare())
        assert out["available"] is False
        assert "pmsampling" in out["reason"]


class TestSamplingAppearsInTheReport:
    """PC and PM sampling must reach the rendered report, not just the payload."""

    def _module(self):
        A = TestInstructionAndPmSampling._Action

        class Action(A):
            def name(self): return "k"
            def rule_results_as_dicts(self): return []
            def metric_names(self):
                return A.metric_names(self) + [
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "smsp__pcsamp_sample_count", "smsp__pcsamp_interval_cycles"]

            def metric_by_name(self, n):
                simple = {"sm__throughput.avg.pct_of_peak_sustained_elapsed": 33.7,
                          "smsp__pcsamp_sample_count": 9244.0,
                          "smsp__pcsamp_interval_cycles": 2048.0}
                if n in simple:
                    class _M:
                        def __init__(self, v): self.v = v
                        def value(self): return self.v
                        def as_double(self): return self.v
                        def as_uint64(self): return int(self.v)
                        def unit(self): return ""
                        def has_correlation_ids(self): return False
                    return _M(simple[n])
                return A.metric_by_name(self, n)

        class Rng:
            num_actions = 1
            def action_by_idx(self, i): return Action()

        class Ctx:
            num_ranges = 1
            def range_by_idx(self, i): return Rng()

        return types.SimpleNamespace(load_report=lambda p: Ctx())

    def _markdown(self):
        return ncu_report_tools.diagnose_result_to_markdown(
            ncu_report_tools.diagnose_ncu_report(
                "/dev/null", ncu_report_module=self._module()))

    def test_pc_sampling_section_is_rendered(self):
        text = self._markdown()
        assert "### PC sampling" in text
        assert "9,244 samples" in text and "2,048-cycle" in text

    def test_stalling_instructions_are_rendered_with_sass(self):
        text = self._markdown()
        assert "#### Stalling instructions" in text
        assert "PRMT R19, R8, 0x7732, RZ" in text

    def test_pm_sampling_section_is_rendered(self):
        text = self._markdown()
        assert "### PM sampling" in text
        assert "time buckets" in text

    def test_pm_span_is_explained_as_covering_replays(self):
        """316us of samples for an 81us kernel is replay, not a long kernel."""
        text = self._markdown()
        assert "several executions" in text

    def test_raw_counts_carry_no_percent_sign_in_the_table(self):
        text = self._markdown()
        assert "2964.0%" not in text and "2965%" not in text


class TestPmSamplingPassGroups:
    """PM sampling is multiplexed across replay passes.

    Each pass re-runs the kernel with its own capture window, so series from
    different passes have different timestamp bases and bucket counts and share
    no timestamps. On a real report there were seven groups whose start times
    differed by milliseconds. Bucket index N is a different moment in each, so
    a window derived from one pass must not be applied to another -- which is
    exactly what the first version of this function did.
    """

    class _Action:
        _PM = "pmsampling:"

        class _M:
            def __init__(self, values, t0, step):
                self.values, self.t0, self.step = values, t0, step
            def num_instances(self): return len(self.values)
            def as_double(self, i): return float(self.values[i])
            def as_uint64(self, i): return self.t0 + i * self.step
            def has_correlation_ids(self): return True
            def correlation_ids(self): return self

        def metric_names(self):
            return [self._PM + "sm__cycles_active.avg",
                    self._PM + "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                    self._PM + "smsp__warps_issue_stalled_barrier.avg"]

        def metric_by_name(self, n):
            M = TestPmSamplingPassGroups._Action._M
            # pass A: 6 buckets from t=1000, step 1500
            if n.endswith("sm__cycles_active.avg"):
                return M([0.0, 100.0, 100.0, 100.0, 0.0, 0.0], 1_000, 1500)
            if n.endswith("pct_of_peak_sustained_elapsed"):
                return M([0.0, 90.0, 60.0, 30.0, 0.0, 0.0], 1_000, 1500)
            # pass B: a DIFFERENT execution -- different base, different length
            if n.endswith("barrier.avg"):
                return M([50.0, 400.0, 200.0, 0.0], 9_000_000, 1472)
            return None

    def test_passes_are_detected(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        assert out["pass_group_count"] == 2
        assert out["cross_pass_warning"]

    def test_each_series_records_its_pass(self):
        out = source_correlation.analyze_pm_sampling(self._Action())
        groups = {e["metric"]: e["pass_group"] for e in out["series"]}
        assert groups["sm__cycles_active.avg"] == groups[
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"]
        assert groups["smsp__warps_issue_stalled_barrier.avg"] != groups[
            "sm__cycles_active.avg"]

    def test_window_is_computed_per_pass_not_borrowed(self):
        """Pass A is active in buckets 1..3; pass B in 0..2. Applying A's
        window to B would average in a bucket B never had."""
        out = source_correlation.analyze_pm_sampling(self._Action())
        barrier = next(e for e in out["series"] if "barrier" in e["metric"])
        # B's own window is buckets 0..2, mean (50+400+200)/3 = 216.7
        assert barrier["mean_in_active_window"] == pytest.approx(216.7, abs=1.0)

    def test_single_pass_report_has_no_warning(self):
        class _One(TestPmSamplingPassGroups._Action):
            def metric_names(self):
                return [self._PM + "sm__cycles_active.avg"]
        out = source_correlation.analyze_pm_sampling(_One())
        assert out["pass_group_count"] == 1
        assert out["cross_pass_warning"] == ""
