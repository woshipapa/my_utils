# SPDX-License-Identifier: Apache-2.0
"""Tests for my_utils.profiling.ncu.ncu_diagnostics."""

from __future__ import annotations


import pytest


from _synthetic_loader import (
    axes,
    gpu_specs,
    measurement_context,
    metric_catalog,
    ncu_diagnostics,
    source_correlation,
)


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
        SATURATED_LOW_OCCUPANCY, kernel_name="cutlass3x_sm90_gemm", gpu_spec=_h100()
    )
    assert not [f for f in result["findings"] if "occupancy" in f["category"]]


def test_low_occupancy_is_a_finding_when_schedulers_are_starving():
    starved = dict(
        SATURATED_LOW_OCCUPANCY,
        **{
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 18.0,
            "smsp__issue_active.avg.per_cycle_active": 0.15,
        },
    )
    result = ncu_diagnostics.diagnose_kernel(
        starved, kernel_name="void my_kernel()", gpu_spec=_h100()
    )
    assert [f for f in result["findings"] if "occupancy" in f["category"]]


def test_warp_specialized_kernels_are_excluded_from_the_occupancy_model():
    """setmaxnreg makes registers-per-thread a weighted artifact."""
    metrics = dict(
        SATURATED_LOW_OCCUPANCY,
        **{
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 18.0,
            "smsp__issue_active.avg.per_cycle_active": 0.15,
            "sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active": 60.0,
            "launch__registers_per_thread": 168,
        },
    )
    section = ncu_diagnostics.diagnose_kernel(
        metrics, kernel_name="void flash_fwd_ws()", gpu_spec=_h100()
    )["sections"]["occupancy"]
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
            "launch__sm_count": 16,  # the partition
            "device__attribute_multiprocessor_count": 132,  # the whole GPU
            "launch__uses_green_context": 1,
        },
        kernel_name="void partitioned()",
        gpu_spec=_h100(),
    )
    # 20 blocks genuinely fills a 16-SM partition, so this must not be flagged.
    assert not [f for f in result["findings"] if f["category"] == "small_grid"]


def test_mps_scope_is_a_measurement_caveat_not_a_kernel_bottleneck():
    result = ncu_diagnostics.diagnose_kernel(
        {
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 55.0,
            "launch__uses_mps": 1,
        },
        collection={"mps_active": True},
    )
    scope = result["sections"]["execution_scope"]
    assert scope["uses_mps"] is True
    finding = next(
        item
        for item in result["findings"]
        if item["category"] == "mps_shared_execution_scope"
    )
    assert finding["source"] == "measurement"


class TestAnalysisCoverage:
    """A skipped analysis and a clean analysis both produce zero findings."""

    def test_missing_sections_are_reported(self):
        # A SpeedOfLight-only report, which is what real ncu runs often carry.
        view = ncu_diagnostics.MetricView(
            {
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 60.0,
            }
        )
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
            assert entry["needs_section"], (
                f"{entry['analysis']} has no section to collect"
            )
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
        findings = [
            f if isinstance(f, dict) else f.to_dict() for f in result["findings"]
        ]
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
        m = dict(
            self.BASE,
            **{
                "device__attribute_compute_capability_major": 10,
                "device__attribute_compute_capability_minor": 0,
                "smsp__sass_thread_inst_executed_op_ffma2_pred_on.sum": 1e9,
            },
        )
        r = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(m))
        assert r["packed_fp32_applied"] is True
        base = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(self.BASE))
        assert r["achieved_tflops"] > base["achieved_tflops"]

    def test_not_applied_off_cc10(self):
        r = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(self.BASE))
        assert r["packed_fp32_applied"] is False
        assert not r.get("caveats")

    def test_absent_counters_on_cc10_are_flagged(self):
        m = dict(
            self.BASE,
            **{
                "device__attribute_compute_capability_major": 10,
                "device__attribute_compute_capability_minor": 0,
            },
        )
        r = ncu_diagnostics.compute_roofline(ncu_diagnostics.MetricView(m))
        assert r["packed_fp32_applied"] is False
        assert any("undercounted by up to 2x" in c for c in r["caveats"])


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
        assert any(f["category"] == "measurement_caveat" for f in result["findings"]), (
            "roofline caveat never reached the findings list"
        )
        assert result["sections"]["roofline"]["packed_fp32_applied"] is False

    def test_no_spurious_caveat_when_complete(self):
        metrics = {
            "gpu__time_duration.sum": 1e6,
            "dram__bytes.sum": 1e8,
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9,
        }
        result = ncu_diagnostics.diagnose_kernel(metrics, kernel_name="fused")
        assert not any(
            f["category"] == "measurement_caveat" for f in result["findings"]
        )


class TestDiagnoseKernelReportsAxes:
    def test_axes_present_and_gaps_named(self):
        result = ncu_diagnostics.diagnose_kernel(
            {
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 80.0,
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            },
            kernel_name="fused_kernel",
        )
        assert result["axes"]["axis_count"] == len(axes.AXES)
        assert "power_clock" in result["axes"]["not_examined"]
        assert result["corroboration"]["shipped_rules_available"] is False


class TestDiagnoseKernelAccountsForEveryMetric:
    def test_uncatalogued_metrics_are_reported(self):
        result = ncu_diagnostics.diagnose_kernel(
            {
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 70.0,
                "lts__t_sectors_srcunit_tex_op_read.sum": 12345.0,
            }
        )
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
            {
                "sm__throughput.avg.pct_of_peak_sustained_elapsed": 85.0,
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": 20.0,
            },
            kernel_name="gemm",
            throttling={"clock_event_mask": 0x4},  # SwPowerCap
        )
        assert "power_clock" not in result["axes"]["not_examined"]
        throttle_findings = [
            f for f in result["findings"] if f["category"] == "throttling"
        ]
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
        unasked = [
            f
            for f in result["findings"]
            if f["category"] == "tile_quantization"
            and "could not be checked" in f["title"]
        ]
        assert unasked, "an unaskable question must not read as a clean result"
        assert unasked[0]["severity"] == "info"

    def test_ragged_problem_shape_is_flagged_with_a_ceiling(self):
        """M=257 against a 128 tile needs 3 tiles covering 384: 33% padding."""
        result = ncu_diagnostics.diagnose_kernel(
            {"launch__grid_size": 512, "launch__block_size": 256},
            kernel_name=self._KERNEL,
            problem_shape={"m": 257, "n": 4096},
        )
        flagged = [
            f
            for f in result["findings"]
            if f["category"] == "tile_quantization" and "M" in f["title"]
        ]
        assert flagged, "257 against a 128 tile computes a third padding"
        assert flagged[0]["evidence"]["waste_fraction"] == pytest.approx(
            0.331, abs=0.01
        )
        assert flagged[0]["speedup_ceiling"] == pytest.approx(1.494, abs=0.01)

    def test_large_ragged_dimension_is_not_flagged(self):
        """M=4097 wastes only 3%: real, but not worth an action."""
        result = ncu_diagnostics.diagnose_kernel(
            {"launch__grid_size": 512, "launch__block_size": 256},
            kernel_name=self._KERNEL,
            problem_shape={"m": 4097, "n": 4096},
        )
        assert not [
            f
            for f in result["findings"]
            if f["category"] == "tile_quantization" and "M" in f["title"]
        ]

    def test_evenly_divided_shape_is_not_flagged(self):
        result = ncu_diagnostics.diagnose_kernel(
            {"launch__grid_size": 512, "launch__block_size": 256},
            kernel_name=self._KERNEL,
            problem_shape={"m": 4096, "n": 4096},
        )
        assert not [
            f for f in result["findings"] if f["category"] == "tile_quantization"
        ]


class TestMemoryHierarchy:
    def test_saturated_l2_is_distinguished_from_dram(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(
            ncu_diagnostics.MetricView(
                {
                    "lts__throughput.avg.pct_of_peak_sustained_elapsed": 88.0,
                    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 25.0,
                }
            )
        )
        assert result["tightest_level"] == "L2"
        finding = next(f for f in result["findings"] if "saturated level" in f.title)
        assert "not DRAM" in finding.title

    def test_sysmem_aperture_access_is_high_severity(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(
            ncu_diagnostics.MetricView(
                {
                    "lts__t_sectors_srcunit_tex_aperture_sysmem_lookup_miss.sum": 50000.0,
                }
            )
        )
        finding = next(f for f in result["findings"] if "system memory" in f.title)
        assert finding.severity == "high"

    def test_asymmetric_l2_hit_rate_is_flagged(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(
            ncu_diagnostics.MetricView(
                {
                    "lts__t_sector_op_read_hit_rate.pct": 15.0,
                    "lts__t_sector_op_write_hit_rate.pct": 90.0,
                }
            )
        )
        assert any(f.category == "poor_cache_locality" for f in result["findings"])

    def test_write_dominated_traffic_is_reported(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(
            ncu_diagnostics.MetricView(
                {
                    "dram__bytes_read.sum": 1e6,
                    "dram__bytes_write.sum": 9e6,
                }
            )
        )
        assert result["dram_write_share"] == pytest.approx(0.9)
        assert any("write-dominated" in f.title for f in result["findings"])

    def test_silent_when_no_memory_counters(self):
        result = ncu_diagnostics.analyze_memory_hierarchy(
            ncu_diagnostics.MetricView({})
        )
        assert result["findings"] == []


class TestIssueEfficiency:
    def test_stalled_resident_warps_are_not_an_occupancy_problem(self):
        """High occupancy + no eligible warps means latency, not occupancy."""
        result = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView(
                {
                    "smsp__warps_eligible.avg.per_cycle_active": 0.3,
                    "smsp__warps_active.avg.per_cycle_active": 12.0,
                }
            )
        )
        finding = result["findings"][0]
        assert "not occupancy" in finding.summary
        assert "stall breakdown" in finding.actions[0]

    def test_low_occupancy_gets_the_occupancy_advice(self):
        result = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView(
                {
                    "smsp__warps_eligible.avg.per_cycle_active": 0.4,
                    "smsp__warps_active.avg.per_cycle_active": 2.0,
                }
            )
        )
        assert "Increase occupancy" in result["findings"][0].actions[0]

    def test_healthy_scheduler_is_silent(self):
        result = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView(
                {
                    "smsp__warps_eligible.avg.per_cycle_active": 3.5,
                }
            )
        )
        assert result["findings"] == []


class TestInstructionMix:
    """SpeedOfLight compute is a max over pipes, so it hides the busy one."""

    def test_transcendental_bound_kernel_is_named(self):
        result = ncu_diagnostics.analyze_instruction_mix(
            ncu_diagnostics.MetricView(
                {
                    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active": 85.0,
                    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": 20.0,
                }
            )
        )
        assert "XU" in result["busiest_pipe"]
        finding = result["findings"][0]
        assert finding.severity == "medium"
        assert "__expf" in finding.actions[0] or "transcendental" in finding.actions[0]

    def test_lsu_bound_is_not_advised_as_a_bandwidth_problem(self):
        result = ncu_diagnostics.analyze_instruction_mix(
            ncu_diagnostics.MetricView(
                {
                    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active": 90.0,
                }
            )
        )
        assert "vectorise" in result["findings"][0].actions[0].lower()

    def test_active_denominator_is_declared(self):
        """These are _active percentages and must not be ranked against SOL."""
        result = ncu_diagnostics.analyze_instruction_mix(
            ncu_diagnostics.MetricView(
                {
                    "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active": 75.0,
                }
            )
        )
        assert "active" in result["findings"][0].evidence["denominator"]

    def test_incidental_fp64_is_flagged_separately(self):
        result = ncu_diagnostics.analyze_instruction_mix(
            ncu_diagnostics.MetricView(
                {
                    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active": 70.0,
                    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active": 4.0,
                }
            )
        )
        assert any(f.category == "unexpected_fp64" for f in result["findings"])

    def test_silent_without_pipe_counters(self):
        assert (
            ncu_diagnostics.analyze_instruction_mix(ncu_diagnostics.MetricView({}))[
                "findings"
            ]
            == []
        )


class TestRooflineDtypeBasis:
    """Grading against the wrong precision's peak scales efficiency directly."""

    def _spec(self):
        return gpu_specs.lookup_gpu_spec("H100 SXM") or gpu_specs.lookup_gpu_spec(
            "H100"
        )

    def test_fp8_counters_pick_the_fp8_peak(self):
        spec = self._spec()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.compute_roofline(
            ncu_diagnostics.MetricView(
                {
                    "sm__ops_path_tensor_src_fp8_dst_fp32_sparsity_off.sum": 1e12,
                    "dram__bytes.sum": 1e9,
                    "gpu__time_duration.sum": 1e6,
                }
            ),
            spec,
        )
        assert result["dtype_basis"] == "fp8"
        assert result["dtype_basis_source"] == "tensor_op_counters"

    def test_unknown_precision_reports_the_spread_not_a_guess(self):
        spec = self._spec()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.compute_roofline(
            ncu_diagnostics.MetricView(
                {
                    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": 70.0,
                    "dram__bytes.sum": 1e9,
                    "gpu__time_duration.sum": 1e6,
                }
            ),
            spec,
        )
        assert result.get("dtype_ambiguous") is True
        assert result["dtype_basis_source"] == "tensor_pipe_active_precision_unknown"
        assert any("could not be determined" in c for c in result["caveats"])

    def test_plain_fp32_kernel_is_unambiguous(self):
        spec = self._spec()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.compute_roofline(
            ncu_diagnostics.MetricView(
                {
                    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9,
                    "dram__bytes.sum": 1e9,
                    "gpu__time_duration.sum": 1e6,
                }
            ),
            spec,
        )
        assert result["dtype_basis"] == "fp32"
        assert not result.get("dtype_ambiguous")


class TestHierarchicalRoofline:
    """Three arithmetic intensities sharing a numerator; the spread is the signal."""

    _FLOPS = {"smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9}

    def test_leaking_l1_is_reported(self):
        """L1 and L2 intensities close together means L1 catches no reuse."""
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    **self._FLOPS,
                    "l1tex__t_sectors.sum": 1e6,
                    "lts__t_sectors.sum": 0.9e6,
                    "dram__bytes.sum": 1e9,
                }
            )
        )
        assert result["available"] is True
        assert result["l1_to_l2_intensity_ratio"] < 1.5
        assert any("L1 is not capturing reuse" in f.title for f in result["findings"])

    def test_healthy_hierarchy_advises_against_tiling_work(self):
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    **self._FLOPS,
                    "l1tex__t_sectors.sum": 1e6,
                    "lts__t_sectors.sum": 1e5,
                    "dram__bytes.sum": 1e6,
                }
            )
        )
        assert result["locality_verdict"] == "healthy"
        finding = next(f for f in result["findings"] if "already capturing" in f.title)
        assert "Do not spend effort" in finding.actions[0]

    def test_bytes_derived_from_sectors_when_byte_counters_absent(self):
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    **self._FLOPS,
                    "l1tex__t_sectors.sum": 1000.0,
                }
            )
        )
        assert result["levels"]["l1"]["bytes"] == pytest.approx(32000.0)
        assert "x 32" in result["levels"]["l1"]["byte_source"]

    def test_direct_byte_counter_is_preferred(self):
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    **self._FLOPS,
                    "l1tex__t_bytes.sum": 4096.0,
                    "l1tex__t_sectors.sum": 1000.0,
                }
            )
        )
        assert result["levels"]["l1"]["bytes"] == pytest.approx(4096.0)
        assert result["levels"]["l1"]["byte_source"] == "byte counter"

    def test_shared_memory_caveat_fires_for_tiled_kernels(self):
        """l1tex__t_bytes excludes shared traffic -- critical for attention/GEMM."""
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    **self._FLOPS,
                    "l1tex__t_sectors.sum": 1e6,
                    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum": 5e5,
                }
            )
        )
        assert any("shared-memory traffic" in c for c in result["caveats"])
        assert any("overestimate" in c for c in result["caveats"])

    def test_missing_flops_reports_why_not_empty(self):
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    "l1tex__t_sectors.sum": 1e6,
                }
            )
        )
        assert result["available"] is False
        assert "FLOP counters" in result["reason"]


class TestSolThresholdsMatchShippedRule:
    """Pinned to NVIDIA's SpeedOfLight.py so drift is caught, not debated."""

    def test_four_thresholds_match_nvidia(self):
        t = ncu_diagnostics.SOL_THRESHOLDS
        assert t["balanced_delta"] == 10.0  # balanced_threshold
        assert t["latency_bound"] == 60.0  # latency_bound_threshold
        assert t["saturated"] == 80.0  # no_bound_threshold
        assert t["waves_small_grid"] == 1.0  # waves_threshold

    def test_unverified_two_axis_table_is_gone(self):
        t = ncu_diagnostics.SOL_THRESHOLDS
        assert "compute_bound_compute" not in t
        assert "compute_bound_memory" not in t


class TestStallAccountingCloses:
    """After GCStack (ISCA 2025): a cycle stack that does not sum to the total
    cannot say how much of the runtime it explained.

    GCStack's own mechanism needs a cycle-level simulator -- it inspects every
    warp slot each cycle -- so it is not portable to hardware counters. Its
    critique of *priority-based* attribution also does not apply here: each
    `..._per_issue_active` metric is the average number of warps in that state,
    so a cycle with several warps stalled differently is already split between
    them, which is what GCStack achieves in simulation.

    What transfers is the closure check.
    """

    def _view(self, reasons):
        metrics = {
            "smsp__average_warp_latency_per_inst_issued.ratio": 20.0,
            "smsp__issue_active.avg.per_cycle_active": 0.1,
        }
        for name, value in reasons.items():
            metrics[
                f"smsp__average_warps_issue_stalled_{name}_per_issue_active.ratio"
            ] = value
        return ncu_diagnostics.MetricView(metrics)

    def test_complete_accounting_reports_full_explanation(self):
        result = ncu_diagnostics.analyze_stalls(
            self._view({"long_scoreboard": 12.0, "barrier": 6.0, "wait": 2.0})
        )
        assert result["explained_share"] == pytest.approx(1.0, abs=0.01)
        assert not [
            f
            for f in result["findings"]
            if f.title == "Stall accounting does not close"
        ]

    def test_missing_reasons_are_flagged_not_absorbed(self):
        """Half the reasons absent must not read as 'the rest is fine'."""
        result = ncu_diagnostics.analyze_stalls(self._view({"long_scoreboard": 8.0}))
        assert result["explained_share"] == pytest.approx(0.4, abs=0.01)
        finding = next(
            f
            for f in result["findings"]
            if f.title == "Stall accounting does not close"
        )
        assert "not unexplained stalling" in finding.summary
        assert "WarpStateStats" in finding.actions[0]

    def test_note_states_what_the_top_stalls_cover(self):
        result = ncu_diagnostics.analyze_stalls(
            self._view({"long_scoreboard": 12.0, "barrier": 6.0, "wait": 2.0}), top_k=1
        )
        assert "top 1 account for 60" in result["accounting_note"]

    def test_absent_total_says_the_shares_have_no_denominator(self):
        result = ncu_diagnostics.analyze_stalls(ncu_diagnostics.MetricView({}))
        assert result["explained_share"] is None
        assert "no denominator" in result["accounting_note"]


class TestWarpSpecializedKernelsAreNotJudgedAsCommodity:
    """A Hopper warp-specialized GEMM violates commodity-kernel rules by design.

    On a real CUTLASS kernel this tool emitted two high-severity findings --
    register spilling and low occupancy -- both of which told the author to undo
    the design. 168 registers x 384 threads is 64,512 of a 65,536-register file:
    the largest allocation that fits. Producer warps wait on TMA for nearly the
    whole kernel; that is the steady state, not a scheduling fault.
    """

    _WS = {
        "sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active": 30.0
    }

    def test_full_register_file_is_recognised_as_deliberate(self):
        # Near the cap AND warp-specialized: an ordinary kernel sitting near its
        # cap is still spilling, so the excuse is gated on the design.
        view = ncu_diagnostics.MetricView(
            {
                "launch__registers_per_thread": 168.0,
                "launch__block_size": 384.0,
                **self._WS,
            }
        )
        out = ncu_diagnostics._registers_are_deliberate(view)
        assert out is not None
        assert out["registers_used"] == 64512
        assert "chosen, not overrun" in out["reason"]

    def test_ordinary_register_count_is_not_excused(self):
        view = ncu_diagnostics.MetricView(
            {"launch__registers_per_thread": 40.0, "launch__block_size": 256.0}
        )
        assert ncu_diagnostics._registers_are_deliberate(view) is None

    def test_spilling_severity_drops_when_registers_are_deliberate(self):
        base = {
            "smsp__inst_executed_op_local_ld.sum": 70000.0,
            "smsp__inst_executed_op_local_st.sum": 20000.0,
            "smsp__inst_executed.sum": 500000.0,
            "l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct": 38.0,
        }
        careless = ncu_diagnostics.analyze_spilling(
            ncu_diagnostics.MetricView(
                {
                    **base,
                    "launch__registers_per_thread": 40.0,
                    "launch__block_size": 256.0,
                }
            )
        )
        deliberate = ncu_diagnostics.analyze_spilling(
            ncu_diagnostics.MetricView(
                {
                    **base,
                    "launch__registers_per_thread": 168.0,
                    "launch__block_size": 384.0,
                    **self._WS,
                }
            )
        )
        assert careless["findings"][0].severity == "high"
        assert deliberate["findings"][0].severity != "high"
        assert "design question, not a defect" in deliberate["findings"][0].summary

    def test_eligible_warp_rule_is_not_applied_to_warp_specialized(self):
        starved = {
            "smsp__warps_eligible.avg.per_cycle_active": 0.16,
            "smsp__warps_active.avg.per_cycle_active": 3.0,
        }
        ordinary = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView(starved)
        )
        specialized = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView({**starved, **self._WS})
        )
        assert ordinary["findings"], "the rule must still fire on a normal kernel"
        assert specialized["findings"] == []
        assert specialized["warp_specialized"] is True
        assert "by design" in specialized["note"]

    def test_long_scoreboard_confound_is_stated_on_wgmma_kernels(self):
        stalls = {
            "smsp__average_warp_latency_per_inst_issued.ratio": 20.0,
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": 12.0,
            "smsp__issue_active.avg.per_cycle_active": 0.1,
        }
        plain = ncu_diagnostics.analyze_stalls(ncu_diagnostics.MetricView(stalls))
        wgmma = ncu_diagnostics.analyze_stalls(
            ncu_diagnostics.MetricView({**stalls, **self._WS})
        )
        plain_ls = next(
            f for f in plain["findings"] if f.category == "stall_long_scoreboard"
        )
        wgmma_ls = next(
            f for f in wgmma["findings"] if f.category == "stall_long_scoreboard"
        )
        assert plain_ls.severity == "high"
        assert wgmma_ls.severity == "medium", "confounded, so not asserted at high"
        assert "warpgroup synchronisation" in wgmma_ls.summary
        assert not any("coalesc" in a.lower() for a in wgmma_ls.actions), (
            "coalescing advice sends the author after memory that is really sync"
        )

    def test_bf16_tensor_ops_resolve_without_the_sparsity_suffix(self):
        """The report carries `..._src_bf16_dst_fp32.sum`; the catalog only had
        the `_sparsity_off` spelling, so the dtype came back unknown."""
        view = ncu_diagnostics.MetricView(
            {
                "sm__ops_path_tensor_src_bf16_dst_fp32.sum": 25769803776.0,
                "gpu__time_duration.sum": 82464.0,
                "dram__bytes.sum": 1e9,
            }
        )
        spec = gpu_specs.lookup_gpu_spec("H100 SXM5")
        result = ncu_diagnostics.compute_roofline(view, spec)
        assert result["tensor_ops"] == pytest.approx(25769803776.0)
        if spec is not None:
            # dtype_basis is only derived once a spec supplies the peaks.
            assert result["dtype_basis"] == "bf16"
            assert not result.get("dtype_ambiguous")
            # 25.77e9 ops / 82.464us = 312.5 TOPS
            assert result["achieved_tflops"] == pytest.approx(312.5, abs=1.0)


class TestVerificationRegressions:
    """Cases an adversarial verification found the previous fixes got wrong.

    The suite that passed before this class exercised only the inputs that
    worked: a dedup fixture where the extra reason was exactly zero, and
    `compare_measurements` only ever on a duration.
    """

    # --- dedup must not fold a reason that is real but spread thin ----------
    _SPREAD = {
        "available": True,
        "stall_reasons": {"LONG_SCOREBOARD": 600, "LG_THROTTLE": 300},
        "source_lines": (
            [
                {
                    "file_name": "k.cu",
                    "line": i,
                    "samples": 200,
                    "stall_reasons": {"LONG_SCOREBOARD": 200},
                }
                for i in (1, 2, 3)
            ]
            + [
                {
                    "file_name": "k.cu",
                    "line": 10 + i,
                    "samples": 30,
                    "stall_reasons": {"LG_THROTTLE": 30},
                }
                for i in range(10)
            ]
        ),
    }

    def test_reason_below_the_display_cut_still_distinguishes(self):
        """LG_THROTTLE is a third of the samples, spread under the top-K cut."""
        out = source_correlation.link_findings_to_source(
            [
                {"category": "stall_long_scoreboard", "title": "LS"},
                {"category": "register_spilling", "title": "SPILL"},
            ],
            None,
            attribution=self._SPREAD,
            top_k=3,
        )
        assert len(out["linked"]) == 2, "distinct mechanisms must not be folded"
        assert out["duplicate_links"] == []

    def test_declared_but_absent_means_absent_from_the_kernel(self):
        out = source_correlation.link_findings_to_source(
            [{"category": "register_spilling", "title": "SPILL"}],
            None,
            attribution=self._SPREAD,
            top_k=3,
        )
        entry = out["linked"][0]
        assert entry["declared_but_absent"] == [], (
            "LG_THROTTLE carried 300 samples; calling it absent is a false claim"
        )
        assert entry["contributing_below_cut"] == ["LG_THROTTLE"]

    def test_dedup_is_independent_of_the_display_cut(self):
        keys = []
        for k in (2, 3, 5):
            out = source_correlation.link_findings_to_source(
                [
                    {"category": "stall_long_scoreboard", "title": "LS"},
                    {"category": "register_spilling", "title": "SPILL"},
                ],
                None,
                attribution=self._SPREAD,
                top_k=k,
            )
            keys.append(len(out["linked"]))
        assert len(set(keys)) == 1, f"dedup varied with top_k: {keys}"

    def test_truly_zero_reason_still_folds(self):
        zero = {
            "available": True,
            "stall_reasons": {"LONG_SCOREBOARD": 1000, "LG_THROTTLE": 0},
            "source_lines": [
                {
                    "file_name": "k.cu",
                    "line": 1,
                    "samples": 1000,
                    "stall_reasons": {"LONG_SCOREBOARD": 1000},
                }
            ],
        }
        out = source_correlation.link_findings_to_source(
            [
                {"category": "stall_long_scoreboard", "title": "LS"},
                {"category": "register_spilling", "title": "SPILL"},
            ],
            None,
            attribution=zero,
        )
        assert len(out["linked"]) == 1

    # --- register tolerance must be relative, and gated ---------------------
    _WS = {
        "sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active": 30.0
    }

    def test_large_block_does_not_get_a_looser_excuse(self):
        """56 regs at block=1024 is 12.5% below the cap, not one granule."""
        view = ncu_diagnostics.MetricView(
            {
                "launch__registers_per_thread": 56.0,
                "launch__block_size": 1024.0,
                **self._WS,
            }
        )
        assert ncu_diagnostics._registers_are_deliberate(view) is None

    def test_deliberate_requires_warp_specialization(self):
        near_cap = {"launch__registers_per_thread": 168.0, "launch__block_size": 384.0}
        assert (
            ncu_diagnostics._registers_are_deliberate(
                ncu_diagnostics.MetricView(near_cap)
            )
            is None
        )
        assert (
            ncu_diagnostics._registers_are_deliberate(
                ncu_diagnostics.MetricView({**near_cap, **self._WS})
            )
            is not None
        )

    # --- warp-specialization detection needs a threshold -------------------
    def test_trace_tma_does_not_suppress_a_real_finding(self):
        starved = {
            "smsp__warps_eligible.avg.per_cycle_active": 0.16,
            "smsp__warps_active.avg.per_cycle_active": 3.0,
        }
        trace = {"sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active": 1e-9}
        out = ncu_diagnostics.analyze_issue_efficiency(
            ncu_diagnostics.MetricView({**starved, **trace})
        )
        assert out["findings"], "one TMA instruction must not suppress the rule"

    def test_tma_alone_is_not_warp_specialization(self):
        tma_only = {"sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active": 5.0}
        assert (
            ncu_diagnostics._is_warp_specialized(ncu_diagnostics.MetricView(tma_only))
            is False
        )
        with_tensor = {
            **tma_only,
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": 30.0,
        }
        assert (
            ncu_diagnostics._is_warp_specialized(
                ncu_diagnostics.MetricView(with_tensor)
            )
            is True
        )

    # --- clock correction depends on what is being compared ----------------
    def _pair(self):
        return (
            measurement_context.describe_collection_mode(
                source="ncu", sm_clock_hz=1.674e9, gpc_clock_hz=1.674e9
            ),
            measurement_context.describe_collection_mode(
                source="ncu", sm_clock_hz=1.789e9, gpc_clock_hz=1.789e9
            ),
        )

    def test_byte_counts_are_not_clock_corrected(self):
        a, b = self._pair()
        out = measurement_context.compare_measurements(
            a, b, baseline_value=1e9, candidate_value=1e9, metric="dram__bytes.sum"
        )
        assert out["comparable"] is True, "a byte count does not depend on the clock"
        assert out.get("clock_normalised_ratio") is None

    def test_throughput_is_divided_not_multiplied(self):
        a, b = self._pair()
        out = measurement_context.compare_measurements(
            a, b, baseline_value=301.6, candidate_value=354.6, metric="achieved_tflops"
        )
        assert out["metric_kind"] == "rate"
        # 1.1757 / 1.0687 = 1.100, not 1.1757 * 1.0687 = 1.256
        assert out["clock_normalised_ratio"] == pytest.approx(1.100, abs=0.01)

    def test_duration_is_multiplied(self):
        a, b = self._pair()
        out = measurement_context.compare_measurements(
            a, b, baseline_value=82464.0, candidate_value=73344.0, metric="duration_ns"
        )
        assert out["clock_normalised_ratio"] == pytest.approx(0.951, abs=0.005)

    def test_missing_clock_does_not_fail_open(self):
        known = measurement_context.describe_collection_mode(
            source="ncu", sm_clock_hz=1.7e9
        )
        unknown = measurement_context.describe_collection_mode(source="ncu")
        out = measurement_context.compare_measurements(
            known,
            unknown,
            baseline_value=100.0,
            candidate_value=90.0,
            metric="duration_ns",
        )
        assert out["comparable"] is False
        assert any("unrecorded" in b for b in out["blockers"])

    # --- closure must fire in both directions ------------------------------
    def test_states_exceeding_their_total_are_flagged(self):
        view = ncu_diagnostics.MetricView(
            {
                "smsp__average_warp_latency_per_inst_issued.ratio": 20.0,
                "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": 15.0,
                "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": 6.0,
                "smsp__issue_active.avg.per_cycle_active": 0.1,
            }
        )
        out = ncu_diagnostics.analyze_stalls(view)
        assert out["explained_share"] > 1.02
        finding = next(f for f in out["findings"] if "sum to more than" in f.title)
        assert "cannot exceed their own total" in finding.summary
        assert out["residual_cycles_per_issued_inst"] < 0


class TestRegisterFigureIsKernelLevel:
    """`launch__registers_per_thread` cannot name the pressured warpgroup.

    Warp-specialized kernels reallocate registers at runtime, so no warpgroup
    holds the reported number. On the CUTLASS kernel that prompted this, the
    source carries a comment about one added register write "pushing the
    producer warp past its post-reg_dealloc budget" -- against a reported 168.
    The mechanism the tool describes is right; the number does not describe the
    warpgroup that is spilling.
    """

    _WS = {
        "sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active": 30.0
    }

    def test_result_states_what_it_cannot_identify(self):
        out = ncu_diagnostics._registers_are_deliberate(
            ncu_diagnostics.MetricView(
                {
                    "launch__registers_per_thread": 168.0,
                    "launch__block_size": 384.0,
                    **self._WS,
                }
            )
        )
        note = out["does_not_identify_the_pressured_warpgroup"]
        assert "warpgroup_reg_dealloc" in note
        assert "LoadRegisterRequirement" in note, "must say where the real budget lives"

    def test_spilling_summary_carries_the_caveat(self):
        out = ncu_diagnostics.analyze_spilling(
            ncu_diagnostics.MetricView(
                {
                    "smsp__inst_executed_op_local_ld.sum": 70000.0,
                    "smsp__inst_executed_op_local_st.sum": 20000.0,
                    "smsp__inst_executed.sum": 500000.0,
                    "launch__registers_per_thread": 168.0,
                    "launch__block_size": 384.0,
                    **self._WS,
                }
            )
        )
        summary = out["findings"][0].summary
        assert "kernel-level" in summary
        assert "post-dealloc" in summary or "warpgroup_reg_dealloc" in summary

    def test_catalog_entry_warns_before_the_metric_is_used(self):
        spec = metric_catalog.METRIC_CATALOG["registers_per_thread"]
        assert "KERNEL-LEVEL" in spec.description
        assert "warpgroup" in spec.description


class TestSfuPressure:
    """The FlashAttention-3 rule: SFU near peak while the tensor pipe is not
    the limiter means softmax/normalisation is the critical path."""

    _XU = "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active"
    _TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"

    def test_fires_on_softmax_bound_attention_shape(self):
        result = ncu_diagnostics.analyze_sfu_pressure(
            ncu_diagnostics.MetricView({self._XU: 72.0, self._TENSOR: 45.0})
        )
        assert result["available"] is True
        assert result["fired"] is True
        (finding,) = result["findings"]
        assert finding.category == "pipe_sfu_softmax_bound"
        assert "softmax" in finding.title or "normalisation" in finding.title
        assert any("pingpong" in a for a in finding.actions)
        # The H100 ratio is cited in the evidence.
        assert "989" in finding.evidence["hardware_ratio"]
        assert "3.9" in finding.evidence["hardware_ratio"]

    def test_h100_spec_quantifies_the_ratio(self):
        spec = _h100()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.analyze_sfu_pressure(
            ncu_diagnostics.MetricView({self._XU: 72.0, self._TENSOR: 45.0}), spec
        )
        (finding,) = result["findings"]
        assert finding.evidence["sfu_peak_tflops"] == pytest.approx(3.9)
        assert finding.evidence["tensor_peak_fp16_dense_tflops"] == pytest.approx(989.4)
        assert finding.evidence["tensor_to_sfu_ratio"] == pytest.approx(253.7, abs=0.1)

    def test_ordinary_gemm_without_sfu_pressure_never_fires(self):
        """A GEMM+RMSNorm whose rsqrt is one op per row has XU near zero."""
        result = ncu_diagnostics.analyze_sfu_pressure(
            ncu_diagnostics.MetricView({self._XU: 0.2, self._TENSOR: 40.6})
        )
        assert result["fired"] is False
        assert not result["findings"]
        assert "below" in result["reason"]

    def test_saturated_tensor_pipe_suppresses_the_finding(self):
        """When the MMA pipe is itself the limiter, relieving the SFU buys nothing."""
        result = ncu_diagnostics.analyze_sfu_pressure(
            ncu_diagnostics.MetricView({self._XU: 65.0, self._TENSOR: 85.0})
        )
        assert result["fired"] is False
        assert "limiter" in result["reason"]

    def test_no_tensor_activity_defers_to_instruction_mix(self):
        """Pure transcendental kernels have no GEMM to overlap with."""
        result = ncu_diagnostics.analyze_sfu_pressure(
            ncu_diagnostics.MetricView({self._XU: 90.0})
        )
        assert result["fired"] is False
        assert "instruction-mix" in result["reason"]

    def test_absent_counter_reports_not_assessed(self):
        result = ncu_diagnostics.analyze_sfu_pressure(
            ncu_diagnostics.MetricView({self._TENSOR: 50.0})
        )
        assert result["available"] is False
        assert "sm__inst_executed_pipe_xu" in result["reason"]
        assert "Not assessed" in result["reason"]


class TestInstructionRoofline:
    """Ding & Williams 2019: warp instructions/sec against inst-per-transaction
    catches the integer/LSU-bound kernels a FLOP roofline mislabels."""

    _BASE = {
        "smsp__inst_executed.sum": 300_000.0,
        "gpu__time_duration.sum": 1_000.0,  # ns -> 300 G warp-inst/s
        "device__attribute_multiprocessor_count": 100.0,
        "sm__cycles_elapsed.avg.per_second": 1e9,  # peak = 100*4*1 GHz = 400 G/s
    }

    def test_issue_bound_kernel_is_named(self):
        result = ncu_diagnostics.instruction_roofline(
            ncu_diagnostics.MetricView(self._BASE)
        )
        assert result["available"] is True
        assert result["achieved_warp_ginst_per_sec"] == pytest.approx(300.0)
        assert result["peak_warp_ginst_per_sec"] == pytest.approx(400.0)
        assert result["pct_of_peak"] == pytest.approx(75.0)
        (finding,) = result["findings"]
        assert finding.category == "pipe_instruction_issue_bound"
        assert "Ding" in finding.summary

    def test_low_issue_rate_defers_to_the_flop_roofline(self):
        metrics = dict(self._BASE)
        metrics["smsp__inst_executed.sum"] = 30_000.0  # 7.5% of peak
        result = ncu_diagnostics.instruction_roofline(
            ncu_diagnostics.MetricView(metrics)
        )
        assert not result["findings"]
        assert "not the limiter" in result["interpretation"]

    def test_intensity_is_per_32_byte_transaction(self):
        metrics = dict(self._BASE)
        metrics["lts__t_sectors.sum"] = 150_000.0
        metrics["dram__bytes.sum"] = 3_200_000.0  # 100k transactions
        result = ncu_diagnostics.instruction_roofline(
            ncu_diagnostics.MetricView(metrics)
        )
        per_txn = result["warp_inst_per_transaction"]
        assert per_txn["l2"] == pytest.approx(2.0)
        assert per_txn["dram"] == pytest.approx(3.0)
        assert "l1" in result["missing_levels"]
        assert "l1tex__t_sectors" in result["missing_levels"]["l1"]

    def test_missing_instruction_counter_states_absence(self):
        result = ncu_diagnostics.instruction_roofline(
            ncu_diagnostics.MetricView({"gpu__time_duration.sum": 1_000.0})
        )
        assert result["available"] is False
        assert "smsp__inst_executed.sum" in result["reason"]

    def test_missing_clock_reports_no_ceiling_not_an_invented_one(self):
        metrics = {
            "smsp__inst_executed.sum": 300_000.0,
            "gpu__time_duration.sum": 1_000.0,
        }
        result = ncu_diagnostics.instruction_roofline(
            ncu_diagnostics.MetricView(metrics)
        )
        assert result["available"] is True
        assert result["peak_warp_ginst_per_sec"] is None
        assert result["pct_of_peak"] is None
        assert "unknown" in result["interpretation"]
        assert not result["findings"]

    def test_peak_states_its_assumption(self):
        result = ncu_diagnostics.instruction_roofline(
            ncu_diagnostics.MetricView(self._BASE)
        )
        assert "4 schedulers/SM" in result["peak_basis"]


class TestHierarchicalRooflineExtensions:
    """Missing levels are named, and the L2 ridge exposes cache-blocking room."""

    _FLOPS = {"smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": 1e9}

    def test_absent_levels_are_named_not_omitted(self):
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {**self._FLOPS, "lts__t_sectors.sum": 1e6, "dram__bytes.sum": 1e7}
            )
        )
        assert "l1" in result["missing_levels"]
        assert "l1tex__t_bytes" in result["missing_levels"]["l1"]
        assert "not collected: L1" in result["summary"]

    def test_wide_l2_dram_gap_is_stated_as_absorption(self):
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    **self._FLOPS,
                    "lts__t_sectors.sum": 1e6,  # AI_l2 ~ 31
                    "dram__bytes.sum": 6.4e6,  # AI_dram ~ 156 -> ratio ~5
                }
            )
        )
        assert result["l2_to_dram_intensity_ratio"] > 4.0
        assert "absorbing" in result["l2_dram_locality_note"]

    def test_dram_compute_bound_but_l2_bandwidth_bound_flags_blocking(self):
        """AI clears the DRAM ridge (~295 bf16) but sits under the L2 ridge (~82)."""
        spec = _h100()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    # 1e12 tensor bf16 FLOPs over 2 ms = 500 TFLOP/s (< peak).
                    "sm__ops_path_tensor_src_bf16_dst_fp32.sum": 1e12,
                    "gpu__time_duration.sum": 2e6,
                    "dram__bytes.sum": 3.2e9,  # AI_dram = 312.5 > ridge 295.3
                    "lts__t_sectors.sum": 6.25e8,  # AI_l2 = 50 < 0.8 * 82.45
                }
            ),
            spec,
        )
        assert result["l2_ridge_point"] == pytest.approx(82.45, abs=0.1)
        assert result["l2_roofline_side"] == "memory_bound"
        finding = next(
            f for f in result["findings"] if "cache-blocking" in f.title.lower()
        )
        assert finding.category == "poor_cache_locality"
        assert finding.evidence["l2_ridge_point"] == pytest.approx(82.45, abs=0.1)
        assert "L2 bandwidth" in finding.summary or "L2 line" in finding.actions[0]

    def test_no_blocking_finding_when_l2_intensity_clears_its_ridge(self):
        """The real cta_pingpong shape: AI_l2 just above the L2 ridge."""
        spec = _h100()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.hierarchical_roofline(
            ncu_diagnostics.MetricView(
                {
                    "sm__ops_path_tensor_src_bf16_dst_fp32.sum": 1e12,
                    "gpu__time_duration.sum": 2e6,
                    "dram__bytes.sum": 3.2e9,
                    "lts__t_sectors.sum": 3.6e8,  # AI_l2 ~ 86.8 > ridge
                }
            ),
            spec,
        )
        assert result["l2_roofline_side"] == "compute_bound"
        assert not any("cache-blocking" in f.title.lower() for f in result["findings"])


class TestPrecisionAwareCeilings:
    """The roofline states which dense per-dtype peak graded it."""

    def test_bf16_kernel_is_graded_against_the_bf16_dense_peak(self):
        spec = _h100()
        if spec is None:
            pytest.skip("no H100 spec")
        result = ncu_diagnostics.compute_roofline(
            ncu_diagnostics.MetricView(
                {
                    "sm__ops_path_tensor_src_bf16_dst_fp32.sum": 1e12,
                    "gpu__time_duration.sum": 2e6,
                    "dram__bytes.sum": 3.2e9,
                }
            ),
            spec,
        )
        assert result["peak_tflops"] == pytest.approx(989.4)
        assert "dense" in result["peak_tflops_basis"]
        assert "no sparsity" in result["peak_tflops_basis"]

    def test_sparse_peaks_are_never_pre_doubled_in_the_table(self):
        spec = _h100()
        if spec is None:
            pytest.skip("no H100 spec")
        # Dense datasheet halves; sparse derived on demand, exactly 2x.
        assert spec.peak_tflops("fp16") == pytest.approx(989.4)
        assert spec.peak_tflops("fp8") == pytest.approx(1979.0)
        assert spec.peak_tflops("fp16", sparse=True) == pytest.approx(2 * 989.4)


class TestRatedClocksReachTheCollectionContext:
    """diagnose_kernel forwards measured + rated clocks so the clock-control
    bias symptom check can run end-to-end (kHz device attributes -> Hz)."""

    def test_reduced_sm_clock_with_full_dram_clock_is_flagged(self):
        result = ncu_diagnostics.diagnose_kernel(
            {
                "gpu__time_duration.sum": 72_672.0,
                "sm__cycles_elapsed.avg.per_second": 1.7445e9,  # 88% of rated
                "dram__cycles_elapsed.avg.per_second": 2.618e9,  # ~100% of rated
                "device__attribute_clock_rate": 1_980_000.0,  # kHz
                "device__attribute_memory_clock_rate": 2_619_000.0,  # kHz
            },
            kernel_name="fused",
        )
        bias = result["measurement_context"]["clock_control_bias"]
        assert bias["checked"] is True
        assert bias["biased"] is True
        assert bias["sm_share_of_rated"] == pytest.approx(0.881, abs=0.01)
        assert bias["dram_share_of_rated"] == pytest.approx(1.0, abs=0.01)

    def test_missing_rated_clocks_report_unassessed_not_unbiased(self):
        result = ncu_diagnostics.diagnose_kernel(
            {
                "gpu__time_duration.sum": 72_672.0,
                "sm__cycles_elapsed.avg.per_second": 1.7445e9,
            },
            kernel_name="fused",
        )
        # The context carries None for "unassessed" (assess_clock_control_bias
        # itself distinguishes unassessed from unbiased); what matters here is
        # that no bias claim was invented from missing rated clocks.
        assert result["measurement_context"]["clock_control_bias"] is None
