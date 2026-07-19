"""Complete Nsight Compute metric catalog: names, meanings, thresholds, collection presets.

This is the single place that knows *what a metric is called* and *what a good
value looks like*, so the analyzers never hard-code metric spellings.  The names
were taken from the ``.section`` and rule sources NVIDIA ships with Nsight
Compute 2026.1.1 (``<install>/sections/``), so they are exact rather than
paraphrased from prose docs.

Three things live here:

* :data:`METRIC_CATALOG` - every metric the analyzers can consume, keyed by a
  stable short name, with the ncu spelling(s), section, unit and interpretation.
* :data:`STALL_REASONS` - the full 19-reason warp-stall taxonomy with the cause
  and fix for each, which is the backbone of latency-bound diagnosis.
* Collection presets - the metric/section lists that guarantee a single ``ncu``
  run captures everything the diagnosis rules need.

Metric name grammar (for anyone extending this):
``unit__(subunit_)(pipestage_)quantity_(qualifiers).(rollup).(submetric)``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MetricSpec",
    "StallReason",
    "METRIC_CATALOG",
    "STALL_REASONS",
    "STALL_METRIC_TEMPLATE",
    "PCSAMP_STALL_TEMPLATE",
    "PMSAMPLING_STALL_TEMPLATE",
    "SECTION_SETS",
    "FULL_DIAGNOSIS_SECTIONS",
    "ONE_PASS_METRIC_GROUPS",
    "ARCH_BY_COMPUTE_CAPABILITY",
    "resolve_metric",
    "metrics_for_category",
    "describe_arch",
]


# ---------------------------------------------------------------------------
# Metric catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    """One logical quantity and the ncu metric name(s) that carry it."""

    key: str
    names: Tuple[str, ...]          # candidates, most-preferred first
    section: str
    category: str
    unit: str = ""
    description: str = ""
    higher_is_better: Optional[bool] = None
    ideal: Optional[float] = None   # the value a perfect kernel would show
    # Threshold at which the value becomes interesting, plus which side of it.
    warn_below: Optional[float] = None
    warn_above: Optional[float] = None
    notes: str = ""


def _m(key, names, section, category, **kw) -> MetricSpec:
    return MetricSpec(key=key, names=tuple(names), section=section, category=category, **kw)


_CATALOG_LIST: List[MetricSpec] = [
    # -- Speed of Light ---------------------------------------------------
    _m("compute_sol", ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
       "SpeedOfLight", "speed_of_light", unit="%", higher_is_better=True, ideal=100.0,
       description="SM throughput vs peak. NVIDIA's own grading: >80 excellent, 60-80 good, 40-60 fair, <40 poor."),
    _m("memory_sol", ["gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"],
       "SpeedOfLight", "speed_of_light", unit="%", higher_is_better=True, ideal=100.0,
       description="Aggregate memory-pipeline throughput vs peak."),
    _m("dram_sol", ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
       "SpeedOfLight", "speed_of_light", unit="%", higher_is_better=True,
       description="DRAM bandwidth vs peak. A healthy streaming kernel reaches 80-95%."),
    _m("l1_sol", ["l1tex__throughput.avg.pct_of_peak_sustained_active"],
       "SpeedOfLight", "speed_of_light", unit="%", higher_is_better=True),
    _m("l2_sol", ["lts__throughput.avg.pct_of_peak_sustained_elapsed"],
       "SpeedOfLight", "speed_of_light", unit="%", higher_is_better=True),
    _m("duration_ns", ["gpu__time_duration.sum"], "SpeedOfLight", "timing", unit="ns"),
    _m("sm_cycles_active", ["sm__cycles_active.avg"], "SpeedOfLight", "timing", unit="cycles"),
    _m("gpc_cycles_elapsed", ["gpc__cycles_elapsed.max"], "SpeedOfLight", "timing", unit="cycles"),
    _m("dram_clock_hz", ["dram__cycles_elapsed.avg.per_second"], "SpeedOfLight", "timing", unit="Hz"),

    # -- Memory traffic ---------------------------------------------------
    _m("dram_bytes", ["dram__bytes.sum"], "MemoryWorkloadAnalysis", "memory", unit="byte",
       description="Total DRAM traffic; the denominator of arithmetic intensity."),
    _m("dram_bytes_read", ["dram__bytes_read.sum", "dram__bytes_op_read.sum"],
       "MemoryWorkloadAnalysis", "memory", unit="byte"),
    _m("dram_bytes_write", ["dram__bytes_write.sum", "dram__bytes_op_write.sum"],
       "MemoryWorkloadAnalysis", "memory", unit="byte"),
    _m("dram_bytes_per_sec", ["dram__bytes.sum.per_second"], "MemoryWorkloadAnalysis", "memory", unit="byte/s"),
    _m("l1_hit_rate", ["l1tex__t_sector_hit_rate.pct"], "MemoryWorkloadAnalysis", "memory",
       unit="%", higher_is_better=True, warn_below=80.0,
       description="L1/TEX sector hit rate. NVIDIA's guidance treats >80% as good."),
    _m("l2_hit_rate", ["lts__t_sector_hit_rate.pct"], "MemoryWorkloadAnalysis", "memory",
       unit="%", higher_is_better=True, warn_below=80.0),
    _m("l2_read_hit_rate", ["lts__t_sector_op_read_hit_rate.pct"], "MemoryWorkloadAnalysis_Tables", "memory", unit="%"),
    _m("l2_write_hit_rate", ["lts__t_sector_op_write_hit_rate.pct"], "MemoryWorkloadAnalysis_Tables", "memory", unit="%"),
    _m("l2_miss_sectors", ["lts__t_sectors_lookup_miss.sum"], "MemoryWorkloadAnalysis", "memory", unit="sector"),

    # -- Coalescing / access pattern --------------------------------------
    _m("global_ld_sectors_per_request",
       ["l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
        "l1tex__t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"],
       "MemoryWorkloadAnalysis_Tables", "coalescing", unit="sector/request",
       higher_is_better=False, ideal=4.0, warn_above=5.0,
       description="Sectors touched per global load request. 4 is ideal for a fully coalesced fp32 warp load; 32 is worst case."),
    _m("global_st_sectors_per_request",
       ["l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"],
       "MemoryWorkloadAnalysis_Tables", "coalescing", unit="sector/request",
       higher_is_better=False, ideal=4.0, warn_above=5.0),
    _m("global_ld_bytes_per_sector",
       ["smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.ratio",
        "smsp__average_data_bytes_per_sector_mem_global_op_ld.ratio"],
       "MemoryWorkloadAnalysis_Tables", "coalescing", unit="byte/sector",
       higher_is_better=True, ideal=32.0, warn_below=16.0,
       description="Useful bytes per 32-byte sector. Wasted bandwidth fraction is 1 - value/32."),
    _m("global_st_bytes_per_sector",
       ["smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
        "smsp__average_data_bytes_per_sector_mem_global_op_st.ratio"],
       "MemoryWorkloadAnalysis_Tables", "coalescing", unit="byte/sector",
       higher_is_better=True, ideal=32.0, warn_below=16.0),
    _m("global_ld_sectors", ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"],
       "MemoryWorkloadAnalysis_Tables", "coalescing", unit="sector"),
    _m("global_ld_requests", ["l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"],
       "MemoryWorkloadAnalysis_Tables", "coalescing", unit="request"),
    _m("l2_sectors_global_actual", ["memory_l2_theoretical_sectors_global"],
       "SourceCounters", "coalescing", unit="sector",
       description="Per-source-line global sectors actually required."),
    _m("l2_sectors_global_ideal", ["memory_l2_theoretical_sectors_global_ideal"],
       "SourceCounters", "coalescing", unit="sector"),
    _m("l2_sectors_global_excessive", ["derived__memory_l2_theoretical_sectors_global_excessive"],
       "SourceCounters", "coalescing", unit="sector", higher_is_better=False, ideal=0.0, warn_above=0.0,
       description="Sectors above ideal; any non-zero value trips NVIDIA's UncoalescedGlobalAccess rule."),
    _m("l2_sectors_local", ["memory_l2_theoretical_sectors_local"], "SourceCounters", "spilling", unit="sector"),

    # -- Shared memory ----------------------------------------------------
    _m("shared_bank_conflicts_ld", ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"],
       "MemoryWorkloadAnalysis_Tables", "shared_memory", unit="", higher_is_better=False, ideal=0.0),
    _m("shared_bank_conflicts_st", ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"],
       "MemoryWorkloadAnalysis_Tables", "shared_memory", unit="", higher_is_better=False, ideal=0.0),
    _m("shared_wavefronts_ld", ["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"],
       "MemoryWorkloadAnalysis_Tables", "shared_memory", unit="wavefront"),
    _m("shared_wavefronts_st", ["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"],
       "MemoryWorkloadAnalysis_Tables", "shared_memory", unit="wavefront"),
    _m("shared_wavefronts_excessive", ["derived__memory_l1_wavefronts_shared_excessive"],
       "SourceCounters", "shared_memory", unit="wavefront", higher_is_better=False, ideal=0.0),
    _m("shared_conflicts_nway", ["derived__memory_l1_conflicts_shared_nway"],
       "SourceCounters", "shared_memory", unit="way", higher_is_better=False, ideal=1.0),
    _m("shared_pipe_util", ["sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "shared_memory", unit="%"),

    # -- Occupancy --------------------------------------------------------
    _m("achieved_occupancy", ["sm__warps_active.avg.pct_of_peak_sustained_active"],
       "Occupancy", "occupancy", unit="%", higher_is_better=True, warn_below=35.0,
       description="Actually resident warps vs hardware maximum."),
    _m("theoretical_occupancy", ["sm__maximum_warps_per_active_cycle_pct"],
       "Occupancy", "occupancy", unit="%", higher_is_better=True, warn_below=80.0,
       description="Ceiling implied by launch config alone. NVIDIA flags <80% of hardware max."),
    _m("occupancy_limit_blocks", ["launch__occupancy_limit_blocks"], "Occupancy", "occupancy", unit="warp"),
    _m("occupancy_limit_registers", ["launch__occupancy_limit_registers"], "Occupancy", "occupancy", unit="warp"),
    _m("occupancy_limit_shared_mem", ["launch__occupancy_limit_shared_mem"], "Occupancy", "occupancy", unit="warp"),
    _m("occupancy_limit_warps", ["launch__occupancy_limit_warps"], "Occupancy", "occupancy", unit="warp"),
    _m("occupancy_limit_barriers", ["launch__occupancy_limit_barriers"], "Occupancy", "occupancy", unit="warp",
       notes="Hopper+ only."),

    # -- Launch geometry --------------------------------------------------
    _m("grid_size", ["launch__grid_size"], "LaunchStats", "launch", unit="block"),
    _m("block_size", ["launch__block_size"], "LaunchStats", "launch", unit="thread"),
    _m("registers_per_thread", ["launch__registers_per_thread"], "LaunchStats", "launch", unit="register",
       warn_above=64.0, description="Above ~64 registers/thread occupancy usually becomes register limited."),
    _m("shared_mem_per_block", ["launch__shared_mem_per_block"], "LaunchStats", "launch", unit="byte"),
    _m("waves_per_sm", ["launch__waves_per_multiprocessor"], "LaunchStats", "launch", unit="wave",
       description="Grid size in units of a full GPU. Fractional tails below 1 wave leave SMs idle."),
    _m("sm_count", ["launch__sm_count", "device__attribute_multiprocessor_count"], "LaunchStats", "launch", unit="SM"),
    _m("cluster_size", ["launch__cluster_size"], "LaunchStats", "launch", unit="block", notes="Hopper+ thread-block clusters."),
    _m("uses_green_context", ["launch__uses_green_context"], "LaunchStats", "launch"),

    # -- Scheduler / issue ------------------------------------------------
    _m("issue_active", ["smsp__issue_active.avg.per_cycle_active"],
       "SchedulerStats", "scheduler", unit="inst/cycle", higher_is_better=True, ideal=1.0, warn_below=0.6,
       description="Instructions issued per scheduler per active cycle. NVIDIA's IssueSlotUtilization rule fires below 0.6."),
    _m("issue_slot_util", ["smsp__issue_active.avg.pct_of_peak_sustained_active"],
       "SchedulerStats", "scheduler", unit="%", higher_is_better=True),
    _m("no_eligible_pct", ["smsp__issue_inst0.avg.pct_of_peak_sustained_active"],
       "SchedulerStats", "scheduler", unit="%", higher_is_better=False,
       description="Share of cycles with no eligible warp - the direct measure of latency exposure."),
    _m("warps_active_per_scheduler", ["smsp__warps_active.avg.per_cycle_active"],
       "SchedulerStats", "scheduler", unit="warp", higher_is_better=True),
    _m("warps_eligible_per_scheduler", ["smsp__warps_eligible.avg.per_cycle_active"],
       "SchedulerStats", "scheduler", unit="warp", higher_is_better=True, warn_below=1.0),
    _m("max_warps_per_scheduler", ["smsp__maximum_warps_avg_per_active_cycle"],
       "SchedulerStats", "scheduler", unit="warp"),
    _m("warp_cycles_per_issued_inst", ["smsp__average_warp_latency_per_inst_issued.ratio"],
       "WarpStateStats", "scheduler", unit="cycle",
       description="Denominator for every stall ratio: how many warp-cycles each issued instruction cost."),

    # -- Instruction / compute --------------------------------------------
    _m("executed_ipc", ["sm__inst_executed.avg.per_cycle_active"],
       "ComputeWorkloadAnalysis", "compute", unit="inst/cycle", higher_is_better=True),
    _m("issued_ipc", ["sm__inst_issued.avg.per_cycle_active"],
       "ComputeWorkloadAnalysis", "compute", unit="inst/cycle", higher_is_better=True),
    _m("sm_busy", ["sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed"],
       "ComputeWorkloadAnalysis", "compute", unit="%", higher_is_better=True),
    _m("inst_executed", ["smsp__inst_executed.sum"], "InstructionStats", "compute", unit="inst"),
    _m("warp_exec_efficiency", ["smsp__thread_inst_executed_per_inst_executed.ratio"],
       "WarpStateStats", "divergence", unit="thread/inst", higher_is_better=True, ideal=32.0, warn_below=24.0,
       description="Average active threads per executed instruction. NVIDIA's ThreadDivergence rule fires below 24 of 32."),
    _m("warp_exec_efficiency_pred_on", ["smsp__thread_inst_executed_pred_on_per_inst_executed.ratio"],
       "WarpStateStats", "divergence", unit="thread/inst", higher_is_better=True, ideal=32.0, warn_below=24.0),
    _m("branch_efficiency", ["smsp__sass_average_branch_targets_threads_uniform.pct"],
       "SourceCounters", "divergence", unit="%", higher_is_better=True, ideal=100.0, warn_below=80.0),
    _m("branch_targets_divergent", ["smsp__sass_branch_targets_threads_divergent.avg",
                                    "smsp__branch_targets_threads_divergent"],
       "SourceCounters", "divergence", higher_is_better=False, ideal=0.0),

    # -- Pipe utilisation --------------------------------------------------
    _m("pipe_alu_util", ["sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%"),
    _m("pipe_fma_util", ["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%"),
    _m("pipe_fp64_util", ["sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%", warn_above=15.0,
       description="FP64 activity in a model that should be FP32/BF16 usually means stray double literals."),
    _m("pipe_tensor_util", ["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                            "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%", higher_is_better=True,
       notes="Ada (CC 8.9) uses the _v2 spelling."),
    _m("pipe_tensor_hmma_util", ["sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%"),
    _m("pipe_tensor_imma_util", ["sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%"),
    _m("pipe_tma_util", ["sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%", notes="Hopper+ Tensor Memory Accelerator."),
    _m("pipe_tc_util", ["sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%", notes="Blackwell tcgen05 pipe."),
    _m("pipe_lsu_util", ["sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%"),
    _m("inst_pipe_gmma", ["sm__inst_executed_pipe_tensor_op_gmma.avg.pct_of_peak_sustained_active"],
       "ComputeWorkloadAnalysis", "pipes", unit="%", notes="Hopper warpgroup MMA (wgmma)."),

    # -- FLOP counters (roofline inputs) -----------------------------------
    _m("flop_fadd", ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"],
       "InstructionStats", "flops", unit="inst", description="FP32 adds; weight 1 FLOP."),
    _m("flop_fmul", ["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"],
       "InstructionStats", "flops", unit="inst", description="FP32 multiplies; weight 1 FLOP."),
    _m("flop_ffma", ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"],
       "InstructionStats", "flops", unit="inst", description="FP32 fused multiply-adds; weight 2 FLOPs."),
    _m("flop_hadd", ["smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"], "InstructionStats", "flops", unit="inst"),
    _m("flop_hmul", ["smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"], "InstructionStats", "flops", unit="inst"),
    _m("flop_hfma", ["smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"], "InstructionStats", "flops", unit="inst"),
    _m("flop_dadd", ["smsp__sass_thread_inst_executed_op_dadd_pred_on.sum"], "InstructionStats", "flops", unit="inst"),
    _m("flop_dmul", ["smsp__sass_thread_inst_executed_op_dmul_pred_on.sum"], "InstructionStats", "flops", unit="inst"),
    _m("flop_dfma", ["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"], "InstructionStats", "flops", unit="inst"),
    _m("tensor_ops_fp16", ["sm__ops_path_tensor_src_fp16_dst_fp32_sparsity_off.sum",
                           "sm__ops_path_tensor_src_fp16.sum"],
       "ComputeWorkloadAnalysis", "flops", unit="op"),
    _m("tensor_ops_bf16", ["sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum"],
       "ComputeWorkloadAnalysis", "flops", unit="op"),
    _m("tensor_ops_fp8", ["sm__ops_path_tensor_src_fp8_dst_fp32_sparsity_off.sum",
                          "sm__ops_path_tensor_src_fp8_dst_fp16_sparsity_off.sum"],
       "ComputeWorkloadAnalysis", "flops", unit="op"),

    # -- Register spilling -------------------------------------------------
    _m("local_ld_inst", ["smsp__sass_inst_executed_op_local_ld.sum", "smsp__inst_executed_op_local_ld.sum"],
       "InstructionStats", "spilling", unit="inst", higher_is_better=False, ideal=0.0,
       description="LDL instructions - local memory traffic, almost always register spill."),
    _m("local_st_inst", ["smsp__sass_inst_executed_op_local_st.sum", "smsp__inst_executed_op_local_st.sum"],
       "InstructionStats", "spilling", unit="inst", higher_is_better=False, ideal=0.0),
    _m("spill_local_inst", ["sass__inst_executed_register_spilling_mem_local"],
       "InstructionStats", "spilling", unit="inst", notes="2025.1+; section-only, not requestable via --metrics."),
    _m("local_spill_requests_pct", ["derived__local_spilling_requests_pct"],
       "MemoryWorkloadAnalysis_Tables", "spilling", unit="%", higher_is_better=False, ideal=0.0),
    _m("local_ld_hit_rate", ["l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate.pct"],
       "MemoryWorkloadAnalysis_Tables", "spilling", unit="%", higher_is_better=True, warn_below=80.0,
       description="Spills that miss L1 reach L2/DRAM and get dramatically more expensive."),

    # -- Workload distribution --------------------------------------------
    _m("sm_cycles_active_max", ["sm__cycles_active.max"], "WorkloadDistribution", "imbalance", unit="cycles"),
    _m("sm_cycles_active_min", ["sm__cycles_active.min"], "WorkloadDistribution", "imbalance", unit="cycles"),
    _m("l2_cycles_active_avg", ["lts__cycles_active.avg"], "WorkloadDistribution", "imbalance", unit="cycles"),
    _m("l2_cycles_active_max", ["lts__cycles_active.max"], "WorkloadDistribution", "imbalance", unit="cycles"),
    _m("dram_cycles_active_avg", ["dram__cycles_active.avg"], "WorkloadDistribution", "imbalance", unit="cycles"),
    _m("dram_cycles_active_max", ["dram__cycles_active.max"], "WorkloadDistribution", "imbalance", unit="cycles"),

    # -- Device attributes -------------------------------------------------
    _m("cc_major", ["device__attribute_compute_capability_major"], "LaunchStats", "device"),
    _m("cc_minor", ["device__attribute_compute_capability_minor"], "LaunchStats", "device"),
    _m("device_sm_count", ["device__attribute_multiprocessor_count"], "LaunchStats", "device", unit="SM"),
    _m("max_warps_per_sm", ["device__attribute_max_warps_per_multiprocessor"], "LaunchStats", "device", unit="warp"),

    # -- PC sampling -------------------------------------------------------
    _m("pcsamp_sample_count", ["smsp__pcsamp_sample_count"], "SourceCounters", "sampling", unit="sample"),
]

METRIC_CATALOG: Dict[str, MetricSpec] = {spec.key: spec for spec in _CATALOG_LIST}


def resolve_metric(key: str) -> Optional[MetricSpec]:
    """Look up a catalog entry by its stable short key."""
    return METRIC_CATALOG.get(key)


def metrics_for_category(category: str) -> List[MetricSpec]:
    """Every catalog entry in one analysis category (e.g. ``coalescing``)."""
    return [spec for spec in _CATALOG_LIST if spec.category == category]


# ---------------------------------------------------------------------------
# Warp stall taxonomy
# ---------------------------------------------------------------------------

STALL_METRIC_TEMPLATE = "smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio"
PCSAMP_STALL_TEMPLATE = "smsp__pcsamp_warps_issue_stalled_{reason}"
PMSAMPLING_STALL_TEMPLATE = "pmsampling:smsp__warps_issue_stalled_{reason}.avg"


@dataclass(frozen=True)
class StallReason:
    """One warp-stall reason with its cause and the fix it implies."""

    key: str
    display: str
    meaning: str
    fixes: Tuple[str, ...]
    bucket: str          # device_memory | shared_memory | synchronization | instruction | other
    # PC-sampling spells a few reasons differently from the summary metrics.
    pcsamp_key: str = ""

    @property
    def metric_name(self) -> str:
        return STALL_METRIC_TEMPLATE.format(reason=self.key)

    @property
    def pcsamp_metric_name(self) -> str:
        return PCSAMP_STALL_TEMPLATE.format(reason=self.pcsamp_key or self.key)


# The complete 19-reason set. Buckets follow DrGPU's top-down decomposition
# (ICPE 2023), fixes follow the text NVIDIA's CPIStall rule emits.
_STALL_LIST: List[StallReason] = [
    StallReason(
        "long_scoreboard", "Long Scoreboard",
        "Waiting on an L1TEX (global/local/surface/texture) data return - i.e. memory latency.",
        ("Improve global-memory coalescing and locality.",
         "Stage reused data in shared memory.",
         "Increase instruction-level parallelism so more loads are in flight.",
         "Unroll loops to overlap independent loads."),
        "device_memory"),
    StallReason(
        "short_scoreboard", "Short Scoreboard",
        "MIO dependency, typically a shared-memory operation or a special-math (MUFU) result.",
        ("Remove shared-memory bank conflicts.",
         "Keep hot values in registers instead of shared memory.",
         "Restructure the access pattern to widen each shared access."),
        "shared_memory"),
    StallReason(
        "wait", "Wait",
        "Fixed-latency execution dependency - the next instruction needs the previous result.",
        ("Increase ILP: interleave independent work.",
         "Unroll loops.",
         "Consider fast-math variants for long-latency math.",
         "Often a sign of an already well-optimised kernel."),
        "instruction"),
    StallReason(
        "barrier", "Barrier",
        "Waiting at a CTA barrier for sibling warps.",
        ("Balance the work done on divergent paths before __syncthreads().",
         "Split blocks of >=512 threads into smaller cooperating groups.",
         "Replace __syncthreads() with __syncwarp() where only warp scope is needed."),
        "synchronization"),
    StallReason(
        "membar", "Membar",
        "Waiting on a memory barrier / fence to drain.",
        ("Remove fences that are not required.",
         "Narrow the fence scope (block instead of device where possible)."),
        "synchronization"),
    StallReason(
        "mio_throttle", "MIO Throttle",
        "The MIO instruction queue is full - too many shared-memory, special-math or dynamic-branch ops.",
        ("Issue fewer, wider shared-memory accesses (vectorise to 64/128-bit).",
         "Reduce special-function (exp/log/rsqrt) density or overlap it with matmul work."),
        "shared_memory"),
    StallReason(
        "lg_throttle", "LG Throttle",
        "The local/global instruction queue in L1 is full.",
        ("Avoid redundant global accesses.",
         "Eliminate local memory traffic (register spills, dynamically indexed local arrays).",
         "Vectorise loads/stores so the same bytes need fewer instructions."),
        "device_memory"),
    StallReason(
        "tex_throttle", "Tex Throttle",
        "The L1TEX texture/surface pipeline queue is full.",
        ("Issue fewer texture/surface operations.",
         "Consider plain global loads: the texture path accepts four threads' requests per cycle versus 32 for global."),
        "device_memory"),
    StallReason(
        "math_pipe_throttle", "Math Pipe Throttle",
        "The targeted math pipeline is oversubscribed.",
        ("Rebalance the instruction mix across pipes (e.g. move FP64 work to FP32/INT).",
         "Increase the warp count so other warps can use idle pipes."),
        "instruction"),
    StallReason(
        "not_selected", "Not Selected",
        "The warp was eligible but the scheduler picked another warp - a healthy sign of surplus parallelism.",
        ("No action needed; if anything, fewer warps may improve cache locality.",),
        "other"),
    StallReason(
        "selected", "Selected",
        "The warp was selected and issued this cycle - this is productive time, not a stall.",
        (), "other"),
    StallReason(
        "drain", "Drain",
        "After EXIT, waiting for outstanding memory stores to drain.",
        ("Reduce the volume of stores at the end of the kernel.",
         "Spread stores through the kernel instead of bursting them at exit."),
        "device_memory"),
    StallReason(
        "imc_miss", "IMC Miss",
        "Immediate-constant cache miss; divergent constant addressing serialises.",
        ("Make constant-bank accesses uniform across the warp.",
         "Move divergently indexed data out of constant memory."),
        "other"),
    StallReason(
        "no_instruction", "No Instruction",
        "Instruction-cache miss or a very short kernel - the SM ran out of fetched instructions.",
        ("Shorten very long unrolled bodies that thrash the instruction cache.",
         "Avoid large jumps in the instruction stream.",
         "For sub-one-wave kernels this is expected; fuse or batch instead."),
        "instruction"),
    StallReason(
        "branch_resolving", "Branch Resolving",
        "Waiting for a branch target to be resolved.",
        ("Reduce dynamic/indirect branching.",
         "Hoist loop-invariant conditions out of hot loops."),
        "instruction"),
    StallReason(
        "dispatch_stall", "Dispatch Stall",
        "The instruction was ready but the dispatch port was unavailable.",
        ("Usually a downstream-pipe symptom; look at whichever pipe is near its throughput limit.",),
        "instruction"),
    StallReason(
        "sleeping", "Sleeping",
        "The warp is sleeping (e.g. __nanosleep or a spin-wait backoff).",
        ("Check for spin-wait loops and hand-rolled locks.",),
        "other"),
    StallReason(
        "misc", "Misc",
        "Stalls the hardware could not attribute to any of the named reasons.",
        ("There is no specific fix for this bucket - it is a residual.",
         "If Misc dominates, collect PC sampling (SourceCounters) and read the stalls per SASS line "
         "instead of relying on the kernel-level summary.",
         "Check that the report is complete: a partially collected WarpStateStats section can push "
         "weight into this bucket."),
        "other"),
    StallReason(
        "gmma", "Warpgroup Arrive",
        "Hopper warpgroup-MMA arrive/wait (WARPGROUP.ARRIVES / WGMMA.WAIT).",
        ("Deepen the WGMMA software pipeline so math overlaps the operand fetch.",
         "Check producer/consumer warp specialisation balance."),
        "instruction",
        pcsamp_key="warpgroup_arrive"),
]

STALL_REASONS: Dict[str, StallReason] = {reason.key: reason for reason in _STALL_LIST}

# Reasons that represent productive or benign time and must never be reported
# as a bottleneck.
BENIGN_STALL_KEYS: Tuple[str, ...] = ("selected", "not_selected")


# ---------------------------------------------------------------------------
# Collection presets
# ---------------------------------------------------------------------------

SECTION_SETS: Dict[str, Tuple[str, ...]] = {
    "basic": ("SpeedOfLight", "LaunchStats", "Occupancy", "WorkloadDistribution"),
    "detailed": (
        "SpeedOfLight", "SpeedOfLight_RooflineChart", "LaunchStats", "Occupancy",
        "WorkloadDistribution", "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis",
        "MemoryWorkloadAnalysis_Chart", "SourceCounters",
    ),
    "full": (
        "SpeedOfLight", "SpeedOfLight_RooflineChart", "LaunchStats", "Occupancy",
        "WorkloadDistribution", "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis",
        "MemoryWorkloadAnalysis_Chart", "MemoryWorkloadAnalysis_Tables",
        "SchedulerStats", "WarpStateStats", "InstructionStats", "SourceCounters",
        "NumaAffinity", "PmSampling",
    ),
}

# The sections our diagnosis rules actually read. Anything the analyzer can
# conclude requires one of these to have been collected; `ncu --set full`
# is a superset and is what the collection YAMLs use.
FULL_DIAGNOSIS_SECTIONS: Tuple[str, ...] = (
    "SpeedOfLight",
    "SpeedOfLight_RooflineChart",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "MemoryWorkloadAnalysis_Chart",
    "MemoryWorkloadAnalysis_Tables",
    "SchedulerStats",
    "WarpStateStats",
    "Occupancy",
    "LaunchStats",
    "InstructionStats",
    "SourceCounters",
    "WorkloadDistribution",
)

# Metric *groups* worth requesting explicitly when driving ncu with --metrics
# rather than sections. Regex groups keep the command line short and survive
# per-architecture metric renames.
ONE_PASS_METRIC_GROUPS: Tuple[str, ...] = (
    "regex:smsp__average_warps_issue_stalled_.*_per_issue_active",
    "regex:sm__pipe_.*_cycles_active.avg.pct_of_peak_sustained_active",
    "regex:smsp__sass_thread_inst_executed_op_[dhf](add|mul|fma)_pred_on.sum",
    "regex:launch__occupancy_limit_.*",
    "group:memory__chart",
    "group:memory__first_level_cache_table",
    "group:memory__l2_cache_table",
    "group:memory__dram_table",
    "group:memory__shared_table",
    "breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
)


# ---------------------------------------------------------------------------
# Architecture identification
# ---------------------------------------------------------------------------

ARCH_BY_COMPUTE_CAPABILITY: Dict[Tuple[int, int], Dict[str, str]] = {
    (7, 0): {"family": "volta", "alias": "v100/sm_70"},
    (7, 2): {"family": "volta", "alias": "xavier/sm_72"},
    (7, 5): {"family": "turing", "alias": "t4-rtx20/sm_75"},
    (8, 0): {"family": "ampere", "alias": "a100/sm_80"},
    (8, 6): {"family": "ampere", "alias": "a10-rtx30/sm_86"},
    (8, 7): {"family": "ampere", "alias": "orin/sm_87"},
    (8, 9): {"family": "ada", "alias": "l40s-rtx40/sm_89"},
    (9, 0): {"family": "hopper", "alias": "h100/sm_90"},
    (10, 0): {"family": "blackwell", "alias": "b200/sm_100"},
    (10, 3): {"family": "blackwell", "alias": "b300/sm_103"},
    (11, 0): {"family": "blackwell", "alias": "thor/sm_110"},
    (12, 0): {"family": "blackwell", "alias": "rtx50/sm_120"},
    (12, 1): {"family": "blackwell", "alias": "gb10/sm_121"},
}

# When compute capability is missing, SM count is a decent fingerprint.
_SM_COUNT_HINTS: Dict[int, Tuple[str, str]] = {
    80: ("volta", "v100/sm_70"),
    108: ("ampere", "a100/sm_80"),
    114: ("hopper", "h100-pcie/sm_90"),
    132: ("hopper", "h100-sxm/sm_90"),
    142: ("ada", "l40s/sm_89"),
    148: ("blackwell", "b200/sm_100"),
}


def describe_arch(
    cc_major: Optional[float] = None,
    cc_minor: Optional[float] = None,
    sm_count: Optional[float] = None,
) -> Dict[str, object]:
    """Identify the GPU family from compute capability, falling back to SM count."""
    if cc_major is not None:
        key = (int(cc_major), int(cc_minor or 0))
        info = ARCH_BY_COMPUTE_CAPABILITY.get(key)
        if info is None:
            # Unknown minor revision: fall back to the major family.
            same_major = [v for k, v in ARCH_BY_COMPUTE_CAPABILITY.items() if k[0] == int(cc_major)]
            info = same_major[0] if same_major else {"family": "unknown", "alias": "unknown"}
        return {
            "family": info["family"],
            "alias": info["alias"],
            "compute_capability": f"{int(cc_major)}.{int(cc_minor or 0)}",
            "sm_count": int(sm_count) if sm_count else None,
            "source": "compute_capability",
        }
    if sm_count:
        family, alias = _SM_COUNT_HINTS.get(int(sm_count), ("unknown", "unknown"))
        return {
            "family": family,
            "alias": alias,
            "compute_capability": "",
            "sm_count": int(sm_count),
            "source": "sm_count_heuristic",
        }
    return {"family": "unknown", "alias": "unknown", "compute_capability": "",
            "sm_count": None, "source": "none"}
