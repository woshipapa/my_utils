"""Tests for my_utils.profiling.hardware.gpu_specs."""

from __future__ import annotations


import pytest


from _synthetic_loader import gpu_specs


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
