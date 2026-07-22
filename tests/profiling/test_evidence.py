"""Tests for my_utils.profiling.analyzers.evidence."""

from __future__ import annotations


from _synthetic_loader import evidence


class TestEvidenceFusion:
    """A conclusion must never outrank the evidence it rests on."""

    def test_counter_beats_name(self):
        """A name saying matmul loses to a tensor pipe that never activated."""
        ev = evidence
        fused, warnings = ev.attribute_kernel(
            "ampere_sgemm_128x64_nn",
            metrics={
                "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": 0.0
            },
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
