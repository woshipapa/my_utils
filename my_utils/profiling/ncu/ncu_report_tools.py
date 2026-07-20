from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _like_match(text: str, pattern: str) -> bool:
    p = str(pattern or "").strip()
    if not p or p in {"%", "*"}:
        return True
    p = p.replace("*", "%")
    parts: List[str] = ["^"]
    for ch in p:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    parts.append("$")
    return re.match("".join(parts), str(text or ""), flags=re.IGNORECASE) is not None


def _to_number(value: object) -> Optional[float]:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    if text.lower() in {"nan", "na", "none", "null", "-"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 100:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def _load_ncu_report_module(ncu_report_module: Any = None) -> ModuleType:
    if ncu_report_module is not None:
        return ncu_report_module
    try:
        import ncu_report as mod  # type: ignore
    except Exception:
        pass
    else:
        return mod  # type: ignore[return-value]

    # Not on the path. It ships with Nsight Compute rather than on PyPI, so the
    # fix is a PYTHONPATH entry, not an install. Locating it here turns the most
    # common first-run failure into a copy-pasteable command.
    found = find_ncu_report_dir()
    if found:
        import sys as _sys

        _sys.path.insert(0, str(found))
        try:
            import ncu_report as mod  # type: ignore
        except Exception:
            pass
        else:
            return mod  # type: ignore[return-value]

    raise RuntimeError(
        "The `ncu_report` module is required to read .ncu-rep files, and it is not "
        "importable.\n\n"
        "It ships inside Nsight Compute; it is not on PyPI, so there is nothing to "
        "pip install. Point PYTHONPATH at the directory containing ncu_report.py:\n\n"
        "  # Linux (typical)\n"
        "  export PYTHONPATH=/opt/nvidia/nsight-compute/<version>/extras/python:$PYTHONPATH\n"
        "  # bundled with CUDA\n"
        "  export PYTHONPATH=/usr/local/cuda/nsight-compute-<version>/extras/python:$PYTHONPATH\n"
        "  # macOS\n"
        "  export PYTHONPATH='/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python':$PYTHONPATH\n\n"
        "To find it on this machine:\n"
        "  find / -name 'ncu_report.py' 2>/dev/null | head\n\n"
        "The module must also match the Python version you are running it under."
    )


def find_ncu_report_dir() -> Optional[Path]:
    """Locate the directory holding ``ncu_report.py``, or None.

    Searched because the module ships with Nsight Compute rather than on PyPI,
    and its location differs between a standalone install, a CUDA-bundled one,
    and macOS. Returns None rather than guessing when nothing is found.
    """
    import glob as _glob
    import os as _os

    candidates = [
        _os.environ.get("NCU_PYTHON_DIR", ""),
        "/opt/nvidia/nsight-compute/*/extras/python",
        "/usr/local/cuda*/nsight-compute-*/extras/python",
        "/usr/local/NVIDIA-Nsight-Compute*/extras/python",
        "/Applications/NVIDIA Nsight Compute.app/Contents/MacOS/python",
        str(Path.home() / "nsight-compute" / "*" / "extras" / "python"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        for path in sorted(_glob.glob(candidate), reverse=True) or [candidate]:
            if (Path(path) / "ncu_report.py").exists():
                return Path(path)
    return None


def _maybe_call(obj: object, name: str, default: Any = None) -> Any:
    if not hasattr(obj, name):
        return default
    value = getattr(obj, name)
    if callable(value):
        try:
            return value()
        except Exception:
            return default
    return value


def _iter_ranges(ctx: object) -> List[object]:
    ranges: List[object] = []
    try:
        for item in ctx:  # type: ignore[operator]
            ranges.append(item)
        if ranges:
            return ranges
    except Exception:
        pass
    num_ranges = _maybe_call(ctx, "num_ranges", 0) or 0
    for i in range(int(num_ranges)):
        try:
            item = ctx.range_by_idx(i)  # type: ignore[attr-defined]
        except Exception:
            item = None
        if item is not None:
            ranges.append(item)
    return ranges


def _iter_actions(range_obj: object) -> List[object]:
    actions: List[object] = []
    try:
        for item in range_obj:  # type: ignore[operator]
            actions.append(item)
        if actions:
            return actions
    except Exception:
        pass
    num_actions = _maybe_call(range_obj, "num_actions", 0) or 0
    for i in range(int(num_actions)):
        try:
            item = range_obj.action_by_idx(i)  # type: ignore[attr-defined]
        except Exception:
            item = None
        if item is not None:
            actions.append(item)
    return actions


def _metric_value(metric: object) -> Any:
    for name in ("value", "as_double", "as_uint64", "as_string"):
        if not hasattr(metric, name):
            continue
        attr = getattr(metric, name)
        try:
            if callable(attr):
                value = attr()
            else:
                value = attr
            if value is not None and str(value) != "":
                return value
        except Exception:
            continue
    return None


def _enum_to_text(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "name"):
        try:
            return str(getattr(value, "name"))
        except Exception:
            return str(value)
    return str(value)


def _normalize_metric_name(text: str) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"[^a-z0-9]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw


def _name_has_tokens(name: str, tokens: Sequence[str]) -> bool:
    low = _normalize_metric_name(name)
    return all(str(token or "").lower() in low for token in tokens)


def _name_has_any_tokens(name: str, tokens: Sequence[str]) -> bool:
    low = _normalize_metric_name(name)
    return any(str(token or "").lower() in low for token in tokens)


def _metric_stat_value(row: Dict[str, object], stat: str = "avg") -> Optional[float]:
    value = _to_number(row.get(stat))
    if value is not None:
        return value
    for fallback in ("p90", "p50", "max", "min"):
        value = _to_number(row.get(fallback))
        if value is not None:
            return value
    return None


@dataclass
class NcuReportMetricRecord:
    kernel_name: str
    metric_name: str
    raw_value: object
    numeric_value: Optional[float]
    unit: str
    range_index: int
    action_index: int


@dataclass
class SkillParam:
    name: str
    description: str
    type: str = "str"
    required: bool = False
    default: object = None


@dataclass
class ReportSkill:
    name: str
    title: str
    description: str
    category: str
    params: List[SkillParam] = field(default_factory=list)
    run_fn: Optional[Callable[..., object]] = None


def _focus_metrics_summary(focus_metrics: object, top_k: int = 5) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not isinstance(focus_metrics, list):
        return out
    for item in focus_metrics:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": str(item.get("name", "")),
                "value": item.get("value"),
                "severity": _enum_to_text(item.get("severity", "")),
                "info": str(item.get("info", "")),
            }
        )
    return out[: int(top_k)]


def _rule_row_sort_key(row: Dict[str, object]) -> Tuple[float, int]:
    speedup = _to_number(row.get("speedup")) or 0.0
    focus = row.get("focus_metrics", [])
    severity_boost = 0
    if isinstance(focus, list):
        sev = " ".join(str(x.get("severity", "")).lower() for x in focus if isinstance(x, dict))
        if "high" in sev:
            severity_boost = 2
        elif "medium" in sev:
            severity_boost = 1
    return (float(speedup), int(severity_boost))


def _build_rule_summary(rule_rows: List[Dict[str, object]], *, top_k: int) -> Dict[str, object]:
    sorted_rows = sorted(rule_rows, key=_rule_row_sort_key, reverse=True)
    by_rule: Dict[str, int] = {}
    for row in rule_rows:
        name = str(row.get("rule_identifier") or row.get("rule_name") or "unknown")
        by_rule[name] = by_rule.get(name, 0) + 1
    return {
        "total_rows": len(rule_rows),
        "top_rows": sorted_rows[: int(top_k)],
        "top_rules": sorted(by_rule.items(), key=lambda x: x[1], reverse=True)[: int(top_k)],
    }


def _find_signal(
    metric_stats: List[Dict[str, object]],
    token_groups: Sequence[Sequence[str]],
    *,
    stat: str = "avg",
    exclude_tokens: Sequence[str] = (),
) -> Optional[Dict[str, object]]:
    best: Optional[Dict[str, object]] = None
    for tokens in token_groups:
        for row in metric_stats:
            metric_name = str(row.get("metric_name", ""))
            if not metric_name:
                continue
            if not _name_has_tokens(metric_name, tokens):
                continue
            if exclude_tokens and _name_has_any_tokens(metric_name, exclude_tokens):
                continue
            value = _metric_stat_value(row, stat=stat)
            if value is None:
                continue
            candidate = {
                "metric_name": metric_name,
                "value": value,
                "samples": int(row.get("samples", 0) or 0),
                "stat": stat,
            }
            if best is None or int(candidate["samples"]) > int(best["samples"]):
                best = candidate
    return best


def _find_metric_value(
    metric_stats: List[Dict[str, object]],
    metric_names: Sequence[str],
    *,
    token_groups: Sequence[Sequence[str]] = (),
    stat: str = "avg",
    exclude_tokens: Sequence[str] = (),
) -> Optional[Dict[str, object]]:
    exact = {str(name).strip().lower() for name in metric_names if str(name).strip()}
    for row in metric_stats:
        metric_name = str(row.get("metric_name", ""))
        if metric_name.lower() not in exact:
            continue
        value = _metric_stat_value(row, stat=stat)
        if value is None:
            continue
        return {
            "metric_name": metric_name,
            "value": value,
            "samples": int(row.get("samples", 0) or 0),
            "stat": stat,
        }
    if token_groups:
        return _find_signal(
            metric_stats,
            token_groups,
            stat=stat,
            exclude_tokens=exclude_tokens,
        )
    return None


def _signal_value(signal: Optional[Dict[str, object]]) -> Optional[float]:
    if not isinstance(signal, dict):
        return None
    return _to_number(signal.get("value"))


def _ratio_signal(
    numerator: Optional[Dict[str, object]],
    denominator: Optional[Dict[str, object]],
    *,
    name: str,
    ideal: Optional[float] = None,
) -> Optional[Dict[str, object]]:
    num = _signal_value(numerator)
    den = _signal_value(denominator)
    if num is None or den is None or den == 0:
        return None
    out: Dict[str, object] = {
        "metric_name": name,
        "value": num / den,
        "numerator": numerator,
        "denominator": denominator,
    }
    if ideal is not None:
        out["ideal"] = ideal
        out["over_ideal"] = (num / den) / ideal if ideal else None
    return out


def _detect_architecture(metric_stats: List[Dict[str, object]]) -> Dict[str, object]:
    cc_major = _find_metric_value(metric_stats, ("device__attribute_compute_capability_major",))
    cc_minor = _find_metric_value(metric_stats, ("device__attribute_compute_capability_minor",))
    num_sms = _find_metric_value(metric_stats, ("device__attribute_multiprocessor_count",))
    max_warps = _find_metric_value(metric_stats, ("device__attribute_max_warps_per_multiprocessor",))
    major = _signal_value(cc_major)
    minor = _signal_value(cc_minor)
    sms = _signal_value(num_sms)

    family = "unknown"
    alias = "unknown"
    if major is not None:
        if int(major) == 9:
            family = "hopper"
            alias = "h100/sm_90"
        elif int(major) == 10:
            family = "blackwell"
            alias = "b200/sm_100"
    elif sms is not None:
        if int(sms) == 132:
            family = "hopper"
            alias = "h100/sm_90"
        elif int(sms) == 148:
            family = "blackwell"
            alias = "b200/sm_100"

    return {
        "family": family,
        "alias": alias,
        "compute_capability": f"{int(major)}.{int(minor or 0)}" if major is not None else "",
        "num_sms": sms,
        "signals": {
            "compute_capability_major": cc_major,
            "compute_capability_minor": cc_minor,
            "multiprocessor_count": num_sms,
            "max_warps_per_multiprocessor": max_warps,
        },
    }


def _extract_top_stall_metrics(
    metric_stats: List[Dict[str, object]],
    *,
    stat: str = "avg",
    top_k: int = 5,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in metric_stats:
        metric_name = str(row.get("metric_name", ""))
        if not metric_name:
            continue
        low = _normalize_metric_name(metric_name)
        if "stall" not in low:
            continue
        value = _metric_stat_value(row, stat=stat)
        if value is None:
            continue
        rows.append(
            {
                "metric_name": metric_name,
                "value": value,
                "samples": int(row.get("samples", 0) or 0),
                "stat": stat,
            }
        )
    rows.sort(key=lambda x: float(x.get("value", 0.0)), reverse=True)
    return rows[: int(top_k)]


def _classify_stall_reason(metric_name: str) -> str:
    low = _normalize_metric_name(metric_name)
    if "long_scoreboard" in low:
        return "long_scoreboard (often memory latency on global memory/L2)"
    if "short_scoreboard" in low:
        return "short_scoreboard (often MIO/shared memory dependency latency)"
    if "barrier" in low:
        return "barrier (synchronization overhead)"
    if "not_selected" in low:
        return "not_selected (not enough eligible warps / scheduling imbalance)"
    if "math_pipe_throttle" in low or "pipe_throttle" in low:
        return "math/pipe throttle (execution pipeline saturation)"
    return "general stall pressure"


def _dimension_entry(
    key: str,
    title: str,
    signals: Dict[str, object],
    findings: List[Dict[str, object]],
    actions: List[str],
) -> Dict[str, object]:
    present_signals = {k: v for k, v in signals.items() if v is not None}
    return {
        "key": key,
        "title": title,
        "signals": present_signals,
        "findings": findings,
        "actions": actions,
        "status": "needs_attention" if findings else ("covered" if present_signals else "missing_metrics"),
    }


def _build_dimension_report(metric_stats: List[Dict[str, object]], *, top_k: int = 10) -> Dict[str, object]:
    architecture = _detect_architecture(metric_stats)
    grid_size = _find_metric_value(metric_stats, ("launch__grid_size",), token_groups=(("launch", "grid_size"),))
    block_size = _find_metric_value(metric_stats, ("launch__block_size",), token_groups=(("launch", "block_size"),))
    waves_per_sm = _find_metric_value(
        metric_stats,
        ("launch__waves_per_multiprocessor",),
        token_groups=(("launch", "waves", "multiprocessor"),),
    )
    num_sms = _find_metric_value(
        metric_stats,
        ("device__attribute_multiprocessor_count",),
        token_groups=(("device", "multiprocessor", "count"),),
    )
    occ_limit_blocks = _find_metric_value(metric_stats, ("launch__occupancy_limit_blocks",))
    occ_limit_regs = _find_metric_value(metric_stats, ("launch__occupancy_limit_registers",))
    occ_limit_smem = _find_metric_value(metric_stats, ("launch__occupancy_limit_shared_mem",))
    occ_limit_warps = _find_metric_value(metric_stats, ("launch__occupancy_limit_warps",))
    regs_per_thread = _find_metric_value(metric_stats, ("launch__registers_per_thread",))
    smem_per_block = _find_metric_value(metric_stats, ("launch__shared_mem_per_block",))
    theoretical_occ = _find_metric_value(
        metric_stats,
        ("sm__maximum_warps_per_active_cycle_pct",),
        token_groups=(("maximum", "warps", "active", "cycle", "pct"),),
    )
    achieved_occ = _find_metric_value(
        metric_stats,
        ("sm__warps_active.avg.pct_of_peak_sustained_active",),
        token_groups=(("warps_active", "pct_of_peak_sustained_active"),),
    )

    occ_findings: List[Dict[str, object]] = []
    grid_val = _signal_value(grid_size)
    sm_val = _signal_value(num_sms)
    waves_val = _signal_value(waves_per_sm)
    theoretical_val = _signal_value(theoretical_occ)
    achieved_val = _signal_value(achieved_occ)
    if grid_val is not None and sm_val is not None and grid_val < sm_val:
        occ_findings.append(
            {
                "category": "small_grid",
                "title": "Grid has fewer CTAs than SMs",
                "summary": "Some SMs may be idle for the whole kernel.",
                "evidence": {"grid_size": grid_size, "num_sms": num_sms},
            }
        )
    if waves_val is not None and waves_val < 1.0:
        occ_findings.append(
            {
                "category": "small_grid",
                "title": "Less than one wave per SM",
                "summary": "Launch geometry is too small to fill the GPU.",
                "evidence": {"waves_per_sm": waves_per_sm},
            }
        )
    elif waves_val is not None and 1.0 <= waves_val < 2.0:
        occ_findings.append(
            {
                "category": "tail_wave",
                "title": "Partial tail wave likely",
                "summary": "A partial final wave can make achieved occupancy lower than theoretical occupancy.",
                "evidence": {"waves_per_sm": waves_per_sm},
            }
        )
    if theoretical_val is not None and achieved_val is not None and theoretical_val >= 50.0 and achieved_val < 0.6 * theoretical_val:
        occ_findings.append(
            {
                "category": "achieved_vs_theoretical_gap",
                "title": "Achieved occupancy is far below theoretical occupancy",
                "summary": "Look at stall reasons and tail effect; launch limits alone do not explain utilization.",
                "evidence": {"theoretical_occupancy": theoretical_occ, "achieved_occupancy": achieved_occ},
            }
        )

    sm_cycles_avg = _find_metric_value(metric_stats, ("sm__cycles_active.avg",), token_groups=(("sm", "cycles_active", "avg"),))
    sm_cycles_max = _find_metric_value(metric_stats, ("sm__cycles_active.max",), token_groups=(("sm", "cycles_active", "max"),))
    sm_cycles_min = _find_metric_value(metric_stats, ("sm__cycles_active.min",), token_groups=(("sm", "cycles_active", "min"),))
    tail_findings: List[Dict[str, object]] = []
    avg_cycles = _signal_value(sm_cycles_avg)
    max_cycles = _signal_value(sm_cycles_max)
    min_cycles = _signal_value(sm_cycles_min)
    if avg_cycles and max_cycles is not None and max_cycles > 1.5 * avg_cycles:
        tail_findings.append(
            {
                "category": "sm_active_cycle_imbalance",
                "title": "Some SMs run much longer than average",
                "summary": "This often indicates load imbalance or variable per-CTA work.",
                "evidence": {"sm_cycles_avg": sm_cycles_avg, "sm_cycles_max": sm_cycles_max},
            }
        )
    if avg_cycles and min_cycles is not None and min_cycles < 0.5 * avg_cycles:
        tail_findings.append(
            {
                "category": "idle_sm_tail",
                "title": "Some SMs have much lower active cycles than average",
                "summary": "The kernel may have a tail effect or uneven block scheduling.",
                "evidence": {"sm_cycles_avg": sm_cycles_avg, "sm_cycles_min": sm_cycles_min},
            }
        )

    top_stalls = _extract_top_stall_metrics(metric_stats, top_k=max(5, int(top_k)))
    pcsamp_samples = _find_metric_value(metric_stats, ("smsp__pcsamp_sample_count",))
    stall_findings: List[Dict[str, object]] = []
    sample_count = _signal_value(pcsamp_samples)
    stall_ratios: List[Dict[str, object]] = []
    for stall in top_stalls:
        value = _signal_value(stall)
        ratio = (value / sample_count) if value is not None and sample_count else None
        enriched = dict(stall)
        if ratio is not None:
            enriched["pct_of_pcsamp_samples"] = 100.0 * ratio
        enriched["meaning"] = _classify_stall_reason(str(stall.get("metric_name", "")))
        stall_ratios.append(enriched)
    if stall_ratios:
        top = stall_ratios[0]
        top_pct = _to_number(top.get("pct_of_pcsamp_samples"))
        top_value = _to_number(top.get("value"))
        if (top_pct is not None and top_pct >= 20.0) or (sample_count is None and top_value is not None and top_value > 0):
            stall_findings.append(
                {
                    "category": "dominant_stall",
                    "title": f"Dominant stall signal: {top.get('metric_name')}",
                    "summary": str(top.get("meaning", "")),
                    "evidence": {"top_stall": top, "top_stalls": stall_ratios[:5]},
                }
            )

    tensor_active = _find_metric_value(
        metric_stats,
        (
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        ),
        token_groups=(("pipe_tensor", "cycles_active"), ("tensor", "cycles_active")),
    )
    fma_active = _find_metric_value(
        metric_stats,
        (
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed",
        ),
        token_groups=(("inst_executed", "pipe_fma"),),
    )
    fp64_active = _find_metric_value(
        metric_stats,
        ("sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",),
        token_groups=(("pipe_fp64",),),
    )
    tensor_findings: List[Dict[str, object]] = []
    tensor_val = _signal_value(tensor_active)
    fma_val = _signal_value(fma_active)
    fp64_val = _signal_value(fp64_active)
    if tensor_val is not None and tensor_val == 0.0 and fma_val is not None and fma_val >= 30.0:
        tensor_findings.append(
            {
                "category": "scalar_fma_no_tensor_core",
                "title": "FMA pipe is active but tensor pipe is idle",
                "summary": "For GEMM/attention/conv-like kernels this can indicate a missed tensor-core optimization.",
                "evidence": {"tensor_active": tensor_active, "fma_active": fma_active},
            }
        )
    if fp64_val is not None and fp64_val > 0.0:
        tensor_findings.append(
            {
                "category": "fp64_activity",
                "title": "FP64 pipe activity detected",
                "summary": "If the kernel should be FP32/BF16, check accidental double literals or math functions.",
                "evidence": {"fp64_active": fp64_active},
            }
        )

    pm_metrics = []
    for row in metric_stats:
        name = str(row.get("metric_name", ""))
        if name.startswith("pmsampling:"):
            value = _metric_stat_value(row)
            if value is not None:
                pm_metrics.append(
                    {
                        "metric_name": name,
                        "value": value,
                        "samples": int(row.get("samples", 0) or 0),
                    }
                )
    pm_metrics.sort(key=lambda x: float(x["value"]), reverse=True)
    timeline_findings: List[Dict[str, object]] = []
    if not pm_metrics:
        timeline_findings.append(
            {
                "category": "missing_pm_sampling",
                "title": "PM sampling metrics were not collected",
                "summary": "Tail effects and pipeline bubbles are hard to confirm without pmsampling:* time-series metrics.",
                "evidence": {},
            }
        )

    dram_read_pct = _find_metric_value(
        metric_stats,
        (
            "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        ),
        token_groups=(("dram", "read", "pct_of_peak"), ("dram", "throughput", "pct_of_peak")),
    )
    dram_write_pct = _find_metric_value(
        metric_stats,
        ("dram__bytes_write.sum.pct_of_peak_sustained_elapsed",),
        token_groups=(("dram", "write", "pct_of_peak"),),
    )
    l1_hit = _find_metric_value(metric_stats, ("l1tex__t_sector_hit_rate.pct",), token_groups=(("l1tex", "hit_rate"),))
    l2_hit = _find_metric_value(metric_stats, ("lts__t_sector_hit_rate.pct",), token_groups=(("lts", "hit_rate"),))
    global_ld_sectors = _find_metric_value(
        metric_stats,
        ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",),
        token_groups=(("l1tex", "sectors", "global", "ld"),),
    )
    global_ld_requests = _find_metric_value(
        metric_stats,
        ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",),
        token_groups=(("l1tex", "requests", "global", "ld"),),
    )
    sectors_per_ld_request = _find_metric_value(
        metric_stats,
        (
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
            "l1tex__t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
        ),
        token_groups=(("sectors", "request", "global", "ld"),),
    )
    if sectors_per_ld_request is None:
        sectors_per_ld_request = _ratio_signal(
            global_ld_sectors,
            global_ld_requests,
            name="derived_global_load_sectors_per_request",
            ideal=4.0,
        )
    elif "ideal" not in sectors_per_ld_request:
        sectors_per_ld_request = dict(sectors_per_ld_request)
        sectors_per_ld_request["ideal"] = 4.0
        value = _signal_value(sectors_per_ld_request)
        sectors_per_ld_request["over_ideal"] = value / 4.0 if value is not None else None

    global_ld_inst = _find_metric_value(
        metric_stats,
        (
            "smsp__sass_inst_executed_op_global_ld.sum",
            "smsp__inst_executed_op_global_ld.sum",
        ),
        token_groups=(("global_ld",), ("op_global_ld",)),
    )
    global_st_inst = _find_metric_value(
        metric_stats,
        (
            "smsp__sass_inst_executed_op_global_st.sum",
            "smsp__inst_executed_op_global_st.sum",
        ),
        token_groups=(("global_st",), ("op_global_st",)),
    )
    store_bytes_per_sector = _find_metric_value(
        metric_stats,
        (
            "smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio",
            "smsp__average_data_bytes_per_sector_mem_global_op_st.ratio",
        ),
        token_groups=(("bytes", "sector", "global", "st"),),
    )
    local_ld = _find_metric_value(
        metric_stats,
        ("smsp__sass_inst_executed_op_local_ld.sum", "smsp__inst_executed_op_local_ld.sum"),
        token_groups=(("local_ld",), ("op_local_ld",)),
    )
    local_st = _find_metric_value(
        metric_stats,
        ("smsp__sass_inst_executed_op_local_st.sum", "smsp__inst_executed_op_local_st.sum"),
        token_groups=(("local_st",), ("op_local_st",)),
    )
    memory_findings: List[Dict[str, object]] = []
    sectors_val = _signal_value(sectors_per_ld_request)
    if sectors_val is not None and sectors_val > 5.0:
        memory_findings.append(
            {
                "category": "uncoalesced_global_loads",
                "title": "Global load sectors/request is above the ideal",
                "summary": "Likely uncoalesced global loads. Ideal is about 4 sectors/request for 128B aligned warp loads.",
                "evidence": {"sectors_per_ld_request": sectors_per_ld_request},
            }
        )
    store_eff = _signal_value(store_bytes_per_sector)
    if store_eff is not None and 0.0 < store_eff < 16.0:
        memory_findings.append(
            {
                "category": "sparse_global_stores",
                "title": "Low useful bytes per global store sector",
                "summary": "Sparse or predicated writes may waste store bandwidth.",
                "evidence": {"store_bytes_per_sector": store_bytes_per_sector},
            }
        )
    if (_signal_value(local_ld) or 0.0) > 0.0 or (_signal_value(local_st) or 0.0) > 0.0:
        memory_findings.append(
            {
                "category": "register_spill",
                "title": "Local memory load/store instructions detected",
                "summary": "Local memory operations often indicate register spill.",
                "evidence": {"local_ld": local_ld, "local_st": local_st, "registers_per_thread": regs_per_thread},
            }
        )

    dimensions = [
        _dimension_entry(
            "occupancy_launch_geometry",
            "SM occupancy & launch geometry",
            {
                "grid_size": grid_size,
                "block_size": block_size,
                "num_sms": num_sms,
                "waves_per_sm": waves_per_sm,
                "occupancy_limit_blocks": occ_limit_blocks,
                "occupancy_limit_registers": occ_limit_regs,
                "occupancy_limit_shared_mem": occ_limit_smem,
                "occupancy_limit_warps": occ_limit_warps,
                "registers_per_thread": regs_per_thread,
                "shared_mem_per_block": smem_per_block,
                "theoretical_occupancy": theoretical_occ,
                "achieved_occupancy": achieved_occ,
            },
            occ_findings,
            [
                "If grid/waves are small, increase parallel work or split reductions.",
                "If registers/shared memory limit occupancy, tune launch bounds, tile size, or live ranges.",
            ],
        ),
        _dimension_entry(
            "thread_block_balance_tail_effect",
            "Thread-block balance / tail effect",
            {
                "sm_cycles_active_avg": sm_cycles_avg,
                "sm_cycles_active_max": sm_cycles_max,
                "sm_cycles_active_min": sm_cycles_min,
                "waves_per_sm": waves_per_sm,
            },
            tail_findings,
            [
                "For variable-size inputs, inspect per-CTA work distribution.",
                "Consider sorting/packing by length, chunking long work, or work stealing.",
            ],
        ),
        _dimension_entry(
            "stall_breakdown",
            "Stall reason breakdown",
            {"pcsamp_sample_count": pcsamp_samples, "top_stalls": stall_ratios[: int(top_k)]},
            stall_findings,
            [
                "Use source counters with -lineinfo to map dominant stalls to source lines.",
                "For long_scoreboard, inspect global memory coalescing and add ILP/pipelining.",
            ],
        ),
        _dimension_entry(
            "tensor_core_compute",
            "Tensor core / compute pipeline",
            {"tensor_active": tensor_active, "fma_active": fma_active, "fp64_active": fp64_active},
            tensor_findings,
            [
                "For matmul-like work with no tensor activity, consider CUTLASS/cuBLAS or MMA kernels.",
                "For unexpected FP64 activity, audit floating-point literals and math functions.",
            ],
        ),
        _dimension_entry(
            "pm_sampling_timeline",
            "SM utilization timeline / PM sampling",
            {"top_pm_sampling_metrics": pm_metrics[: int(top_k)]},
            timeline_findings,
            [
                "Collect pmsampling:* metrics to distinguish flat-low, long-tail, and sawtooth timeline shapes.",
            ],
        ),
        _dimension_entry(
            "memory_access_cache",
            "Memory access pattern & cache efficiency",
            {
                "dram_read_pct_peak": dram_read_pct,
                "dram_write_pct_peak": dram_write_pct,
                "l1_hit_rate": l1_hit,
                "l2_hit_rate": l2_hit,
                "global_load_sectors": global_ld_sectors,
                "global_load_requests": global_ld_requests,
                "sectors_per_ld_request": sectors_per_ld_request,
                "global_ld_instructions": global_ld_inst,
                "global_st_instructions": global_st_inst,
                "store_bytes_per_sector": store_bytes_per_sector,
                "local_ld": local_ld,
                "local_st": local_st,
            },
            memory_findings,
            [
                "For sectors/request > 5, rework lane-to-address mapping or vectorize loads.",
                "For register spill, reduce live ranges or tune __launch_bounds__.",
            ],
        ),
    ]
    findings = [finding for dim in dimensions for finding in dim.get("findings", []) if isinstance(finding, dict)]
    return {
        "architecture": architecture,
        "dimensions": dimensions,
        "needs_attention": [dim["key"] for dim in dimensions if dim.get("status") == "needs_attention"],
        "missing_metric_dimensions": [dim["key"] for dim in dimensions if dim.get("status") == "missing_metrics"],
        "top_findings": findings[: int(top_k)],
    }


def _build_metric_coverage(metric_stats: List[Dict[str, object]]) -> Dict[str, object]:
    metric_names = [str(row.get("metric_name", "")) for row in metric_stats if isinstance(row, dict)]

    def has_any(groups: Sequence[Sequence[str]]) -> bool:
        for name in metric_names:
            for tokens in groups:
                if _name_has_tokens(name, tokens):
                    return True
        return False

    categories: List[Tuple[str, Sequence[Sequence[str]], str]] = [
        (
            "speed_of_light_compute",
            (("sm", "throughput", "pct_of_peak"), ("smsp", "throughput", "pct_of_peak")),
            "SM throughput vs peak (compute saturation)",
        ),
        (
            "speed_of_light_memory",
            (("dram", "throughput", "pct_of_peak"), ("memory", "throughput", "pct_of_peak")),
            "DRAM throughput vs peak (memory bandwidth saturation)",
        ),
        (
            "occupancy",
            (("occupancy",), ("warps_active", "pct_of_peak")),
            "Occupancy / active warps",
        ),
        (
            "scheduler",
            (("issue_active",), ("warps_eligible",)),
            "Scheduler efficiency / eligible warps",
        ),
        (
            "warp_stalls",
            (("stall",), ("pcsamp", "stall")),
            "Warp stall reasons",
        ),
        (
            "memory_hierarchy",
            (("l1tex",), ("lts",), ("dram",), ("shared",), ("local",), ("global",)),
            "L1/L2/DRAM/shared/local/global memory behavior",
        ),
        (
            "launch_stats",
            (("launch",), ("grid", "block")),
            "Kernel launch/block/grid stats",
        ),
        (
            "source_coalescing",
            (
                ("ideal", "transactions", "global"),
                ("transactions", "global"),
                ("sectors", "global"),
            ),
            "Source-counters style global transaction efficiency signals",
        ),
        (
            "control_flow_divergence",
            (("branch",), ("diverg",), ("predication",)),
            "Control-flow efficiency / divergence signals",
        ),
        (
            "shared_memory_conflicts",
            (("shared", "bank", "conflict"), ("bank", "conflict")),
            "Shared-memory bank conflict signals",
        ),
        (
            "tensor_core_roofline_readiness",
            (
                ("tensor",),
                ("pipe_tensor",),
                ("roofline",),
                ("arithmetic", "intensity"),
                ("flop", "count"),
            ),
            "Tensor/roofline readiness signals (compute intensity + tensor usage)",
        ),
    ]

    details: List[Dict[str, object]] = []
    covered = 0
    for key, groups, desc in categories:
        present = has_any(groups)
        if present:
            covered += 1
        details.append({"category": key, "present": present, "description": desc})
    total = len(categories)
    score = int(round(100.0 * covered / total)) if total else 0
    missing = [row["category"] for row in details if not bool(row["present"])]
    recommendation = ""
    if missing:
        recommendation = (
            "missing categories detected; consider collecting with "
            "`ncu --set full --section ComputeWorkloadAnalysis,MemoryWorkloadAnalysis,"
            "Occupancy,SchedulerStats,WarpStateStats,LaunchStats,SpeedOfLight` "
            "and add source-level counters/lineinfo when investigating coalescing/divergence."
        )
    return {
        "coverage_score": score,
        "covered_categories": covered,
        "total_categories": total,
        "missing_categories": missing,
        "details": details,
        "recommendation": recommendation,
    }


def _build_rule_findings(rule_rows: List[Dict[str, object]], *, top_k: int) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    sorted_rows = sorted(rule_rows, key=_rule_row_sort_key, reverse=True)
    for row in sorted_rows[: int(top_k)]:
        title = str(row.get("rule_message_title") or row.get("rule_name") or row.get("rule_identifier") or "")
        message = str(row.get("rule_message") or "")
        findings.append(
            {
                "source": "ncu_rule",
                "category": str(row.get("section_identifier") or "rule"),
                "title": title,
                "summary": message,
                "kernel_name": str(row.get("kernel_name") or ""),
                "speedup_estimate": _to_number(row.get("speedup")),
                "speedup_type": str(row.get("speedup_type") or ""),
                "focus_metrics": _focus_metrics_summary(row.get("focus_metrics", []), top_k=3),
                "confidence": "high",
            }
        )
    return findings


def _build_heuristic_findings(metric_stats: List[Dict[str, object]], *, top_k: int) -> Dict[str, object]:
    sm = _find_signal(
        metric_stats,
        (
            ("sm", "throughput", "pct_of_peak"),
            ("smsp", "throughput", "pct_of_peak"),
        ),
    )
    dram = _find_signal(
        metric_stats,
        (
            ("dram", "throughput", "pct_of_peak"),
            ("memory", "throughput", "pct_of_peak"),
        ),
    )
    occ = _find_signal(
        metric_stats,
        (
            ("occupancy",),
            ("warps_active", "pct_of_peak"),
        ),
    )
    issue = _find_signal(
        metric_stats,
        (
            ("issue_active",),
            ("inst_issued", "pct"),
        ),
    )
    eligible = _find_signal(metric_stats, (("warps_eligible",),))
    ideal_l2_txn_global = _find_signal(
        metric_stats,
        (
            ("ideal", "transactions", "global"),
            ("memory", "ideal", "l2", "transactions", "global"),
            ("l2", "transactions", "global", "ideal"),
        ),
    )
    actual_l2_txn_global = _find_signal(
        metric_stats,
        (
            ("memory", "l2", "transactions", "global"),
            ("l2", "transactions", "global"),
            ("transactions", "global"),
        ),
        exclude_tokens=("ideal",),
    )
    branch_divergence = _find_signal(
        metric_stats,
        (
            ("branch", "diverg"),
            ("diverg",),
            ("branch", "efficiency"),
        ),
    )
    shared_bank_conflict = _find_signal(
        metric_stats,
        (
            ("shared", "bank", "conflict"),
            ("bank", "conflict"),
        ),
    )
    roofline_intensity = _find_signal(
        metric_stats,
        (
            ("arithmetic", "intensity"),
            ("roofline",),
        ),
    )
    tensor_usage = _find_signal(
        metric_stats,
        (
            ("pipe_tensor",),
            ("tensor",),
            ("hmma",),
        ),
    )
    stalls = _extract_top_stall_metrics(metric_stats, top_k=5)

    findings: List[Dict[str, object]] = []
    sm_val = _to_number(sm.get("value")) if isinstance(sm, dict) else None
    dram_val = _to_number(dram.get("value")) if isinstance(dram, dict) else None
    occ_val = _to_number(occ.get("value")) if isinstance(occ, dict) else None
    issue_val = _to_number(issue.get("value")) if isinstance(issue, dict) else None
    ideal_l2_txn_global_val = (
        _to_number(ideal_l2_txn_global.get("value")) if isinstance(ideal_l2_txn_global, dict) else None
    )
    actual_l2_txn_global_val = (
        _to_number(actual_l2_txn_global.get("value")) if isinstance(actual_l2_txn_global, dict) else None
    )
    branch_divergence_val = (
        _to_number(branch_divergence.get("value")) if isinstance(branch_divergence, dict) else None
    )
    shared_bank_conflict_val = (
        _to_number(shared_bank_conflict.get("value")) if isinstance(shared_bank_conflict, dict) else None
    )
    roofline_intensity_val = _to_number(roofline_intensity.get("value")) if isinstance(roofline_intensity, dict) else None
    tensor_usage_val = _to_number(tensor_usage.get("value")) if isinstance(tensor_usage, dict) else None

    if sm_val is not None and dram_val is not None:
        if dram_val >= 70.0 and sm_val < 70.0:
            findings.append(
                {
                    "source": "heuristic",
                    "category": "memory_bandwidth_bound",
                    "title": "DRAM throughput close to peak while SM throughput is lower",
                    "summary": "Likely memory-bandwidth bound. Focus on data reuse/coalescing/compression.",
                    "evidence": {"sm_pct_peak": sm, "dram_pct_peak": dram},
                    "confidence": "medium",
                }
            )
        elif sm_val >= 70.0 and dram_val < 70.0:
            findings.append(
                {
                    "source": "heuristic",
                    "category": "compute_bound",
                    "title": "SM throughput close to peak while DRAM throughput is lower",
                    "summary": "Likely compute pipeline bound. Check instruction mix and tensor/ALU pipe utilization.",
                    "evidence": {"sm_pct_peak": sm, "dram_pct_peak": dram},
                    "confidence": "medium",
                }
            )
        elif sm_val < 40.0 and dram_val < 40.0:
            findings.append(
                {
                    "source": "heuristic",
                    "category": "under_utilized",
                    "title": "Both SM and DRAM throughput are low",
                    "summary": "Kernel may be under-utilized; inspect occupancy, launch config and latency stalls.",
                    "evidence": {"sm_pct_peak": sm, "dram_pct_peak": dram},
                    "confidence": "medium",
                }
            )

    if occ_val is not None and occ_val < 35.0:
        findings.append(
            {
                "source": "heuristic",
                "category": "occupancy_limited",
                "title": "Low occupancy / active warps",
                "summary": "Low active warps can reduce latency hiding. Review registers/shared memory/block size.",
                "evidence": {"occupancy_signal": occ},
                "confidence": "medium",
            }
        )

    if issue_val is not None and issue_val < 60.0:
        findings.append(
            {
                "source": "heuristic",
                "category": "scheduler_efficiency",
                "title": "Low scheduler issue activity",
                "summary": "Schedulers are not issuing enough instructions; likely latency or dependency pressure.",
                "evidence": {"issue_signal": issue, "eligible_warps_signal": eligible},
                "confidence": "medium",
            }
        )

    if stalls:
        top_stall = stalls[0]
        reason = _classify_stall_reason(str(top_stall.get("metric_name", "")))
        findings.append(
            {
                "source": "heuristic",
                "category": "warp_stall",
                "title": f"Dominant stall signal: {top_stall.get('metric_name')}",
                "summary": f"Top observed stall category: {reason}.",
                "evidence": {"top_stall": top_stall, "top_stalls": stalls},
                "confidence": "medium",
            }
        )

    if (
        ideal_l2_txn_global_val is not None
        and actual_l2_txn_global_val is not None
        and ideal_l2_txn_global_val > 0
    ):
        txn_ratio = actual_l2_txn_global_val / ideal_l2_txn_global_val
        if txn_ratio >= 1.3:
            findings.append(
                {
                    "source": "heuristic",
                    "category": "global_memory_coalescing",
                    "title": "Actual global transactions are much higher than ideal",
                    "summary": (
                        "Likely uncoalesced global accesses. Review memory access patterns, "
                        "data layout, and per-thread access stride."
                    ),
                    "evidence": {
                        "actual_transactions_global": actual_l2_txn_global,
                        "ideal_transactions_global": ideal_l2_txn_global,
                        "actual_over_ideal": txn_ratio,
                    },
                    "confidence": "medium",
                }
            )

    if branch_divergence_val is not None and branch_divergence_val >= 20.0:
        findings.append(
            {
                "source": "heuristic",
                "category": "control_flow_divergence",
                "title": "Branch divergence signal is high",
                "summary": "Warp control flow likely diverges. Consider branch simplification or data regrouping.",
                "evidence": {"branch_divergence_signal": branch_divergence},
                "confidence": "medium",
            }
        )

    if shared_bank_conflict_val is not None and shared_bank_conflict_val > 0.0:
        findings.append(
            {
                "source": "heuristic",
                "category": "shared_memory_bank_conflict",
                "title": "Shared-memory bank conflict signal detected",
                "summary": "Shared-memory layout/access may cause bank conflicts; check padding/index mapping.",
                "evidence": {"shared_bank_conflict_signal": shared_bank_conflict},
                "confidence": "medium",
            }
        )

    findings = findings[: int(top_k)]
    return {
        "signals": {
            "sm_pct_peak": sm,
            "dram_pct_peak": dram,
            "occupancy": occ,
            "issue_active": issue,
            "eligible_warps": eligible,
            "ideal_l2_transactions_global": ideal_l2_txn_global,
            "actual_l2_transactions_global": actual_l2_txn_global,
            "branch_divergence": branch_divergence,
            "shared_bank_conflict": shared_bank_conflict,
            "roofline_intensity": roofline_intensity,
            "tensor_usage": tensor_usage,
            "roofline_intensity_value": roofline_intensity_val,
            "tensor_usage_value": tensor_usage_val,
            "top_stalls": stalls,
        },
        "findings": findings,
    }


def _build_bottleneck_report(
    metric_stats: List[Dict[str, object]],
    rule_rows: List[Dict[str, object]],
    *,
    top_k: int,
) -> Dict[str, object]:
    coverage = _build_metric_coverage(metric_stats)
    dimension_report = _build_dimension_report(metric_stats, top_k=max(int(top_k), 5))
    rule_findings = _build_rule_findings(rule_rows, top_k=max(int(top_k), 5))
    heuristic_payload = _build_heuristic_findings(metric_stats, top_k=max(int(top_k), 5))
    heuristic_findings = list(heuristic_payload.get("findings", []))

    top_bottlenecks: List[Dict[str, object]] = []
    if rule_findings:
        top_bottlenecks.extend(rule_findings[: int(top_k)])
    if len(top_bottlenecks) < int(top_k):
        remain = int(top_k) - len(top_bottlenecks)
        dimension_findings = [
            {
                "source": "dimension_report",
                "category": str(item.get("category", "")),
                "title": str(item.get("title", "")),
                "summary": str(item.get("summary", "")),
                "evidence": item.get("evidence", {}),
                "confidence": "medium",
            }
            for item in dimension_report.get("top_findings", [])
            if isinstance(item, dict)
        ]
        top_bottlenecks.extend(dimension_findings[:remain])
    if len(top_bottlenecks) < int(top_k):
        remain = int(top_k) - len(top_bottlenecks)
        top_bottlenecks.extend(heuristic_findings[:remain])

    notes: List[str] = [
        "Prefer NCU built-in rule findings when available; they are section-aware and metric-context aware.",
        "Dimension findings follow the Codex NCU workflow: occupancy, balance, stalls, tensor core, timeline, memory.",
        "Heuristic findings are fallback signals and should be validated with kernel/source context.",
    ]
    if coverage.get("missing_categories"):
        notes.append("Coverage is incomplete for at least one core analysis category.")

    return {
        "coverage": coverage,
        "dimension_report": dimension_report,
        "signals": heuristic_payload.get("signals", {}),
        "rule_findings": rule_findings,
        "heuristic_findings": heuristic_findings,
        "top_bottlenecks": top_bottlenecks,
        "notes": notes,
    }


class NcuReportSkillEngine:
    def __init__(
        self,
        report_path: str,
        *,
        ncu_report_module: Any = None,
    ) -> None:
        self.report_path = str(report_path)
        self._ncu_report_module = ncu_report_module
        self._records = load_ncu_report_records(
            report_path,
            metric_like="%",
            kernel_like="%",
            ncu_report_module=ncu_report_module,
        )
        self._skills = self._build_skills()

    def list_skills(self) -> List[str]:
        return sorted(self._skills.keys())

    def describe_skills(self) -> Dict[str, object]:
        out = []
        for name in self.list_skills():
            item = self._skills[name]
            out.append(
                {
                    "name": item.name,
                    "title": item.title,
                    "description": item.description,
                    "category": item.category,
                    "params": [
                        {
                            "name": p.name,
                            "description": p.description,
                            "type": p.type,
                            "required": p.required,
                            "default": p.default,
                        }
                        for p in item.params
                    ],
                }
            )
        return {
            "report_path": self.report_path,
            "metric_records": len(self._records),
            "skills": out,
        }

    def run_skill(self, name: str, **kwargs: Any) -> object:
        skill = self._skills.get(str(name))
        if not skill or not skill.run_fn:
            raise ValueError(f"unknown ncu report skill: {name}")
        params = self._resolve_params(skill, kwargs)
        return skill.run_fn(**params)

    def _resolve_params(self, skill: ReportSkill, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in skill.params:
            if p.name in kwargs:
                value = kwargs[p.name]
            elif p.default is not None:
                value = p.default
            elif p.required:
                raise ValueError(f"skill '{skill.name}' missing required param '{p.name}'")
            else:
                continue
            t = str(p.type).lower()
            if t == "int":
                value = int(value)
            elif t == "float":
                value = float(value)
            elif t == "bool":
                value = bool(value)
            else:
                value = str(value)
            out[p.name] = value
        return out

    def _filtered_records(self, *, metric_like: str = "%", kernel_like: str = "%") -> List[NcuReportMetricRecord]:
        out: List[NcuReportMetricRecord] = []
        for rec in self._records:
            if not _like_match(rec.metric_name, metric_like):
                continue
            if not _like_match(rec.kernel_name, kernel_like):
                continue
            out.append(rec)
        return out

    def _skill_summary(self, *, metric_like: str = "%", kernel_like: str = "%", top_k: int = 20) -> Dict[str, object]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        by_metric: Dict[str, int] = {}
        by_kernel: Dict[str, int] = {}
        values: List[float] = []
        non_numeric = 0
        for rec in rows:
            by_metric[rec.metric_name] = by_metric.get(rec.metric_name, 0) + 1
            by_kernel[rec.kernel_name] = by_kernel.get(rec.kernel_name, 0) + 1
            if rec.numeric_value is None:
                non_numeric += 1
            else:
                values.append(float(rec.numeric_value))
        values_sorted = sorted(values)
        return {
            "report_path": self.report_path,
            "metric_records": len(self._records),
            "filtered_records": len(rows),
            "filters": {"metric_like": metric_like, "kernel_like": kernel_like},
            "unique_metrics": len(by_metric),
            "unique_kernels": len(by_kernel),
            "numeric_values": len(values),
            "non_numeric_values": non_numeric,
            "value_stats": {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "avg": (sum(values) / len(values)) if values else None,
                "p50": _percentile(values_sorted, 50) if values else None,
                "p90": _percentile(values_sorted, 90) if values else None,
                "p99": _percentile(values_sorted, 99) if values else None,
            },
            "top_metrics": sorted(by_metric.items(), key=lambda x: x[1], reverse=True)[: int(top_k)],
            "top_kernels": sorted(by_kernel.items(), key=lambda x: x[1], reverse=True)[: int(top_k)],
        }

    def _skill_per_metric_stats(self, *, metric_like: str = "%", kernel_like: str = "%") -> List[Dict[str, object]]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        grouped: Dict[str, List[NcuReportMetricRecord]] = {}
        for rec in rows:
            grouped.setdefault(rec.metric_name, []).append(rec)
        out: List[Dict[str, object]] = []
        for metric_name, recs in grouped.items():
            values = [float(r.numeric_value) for r in recs if r.numeric_value is not None]
            values_sorted = sorted(values)
            out.append(
                {
                    "metric_name": metric_name,
                    "samples": len(recs),
                    "numeric_samples": len(values),
                    "non_numeric_samples": len(recs) - len(values),
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "avg": (sum(values) / len(values)) if values else None,
                    "p50": _percentile(values_sorted, 50) if values else None,
                    "p90": _percentile(values_sorted, 90) if values else None,
                    "p99": _percentile(values_sorted, 99) if values else None,
                }
            )
        out.sort(key=lambda x: int(x.get("samples", 0)), reverse=True)
        return out

    def _skill_top_kernels(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        top_k: int = 20,
        score: str = "sum",
    ) -> List[Dict[str, object]]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        grouped: Dict[str, List[float]] = {}
        for rec in rows:
            if rec.numeric_value is None:
                continue
            grouped.setdefault(rec.kernel_name, []).append(float(rec.numeric_value))
        score_name = str(score or "sum").lower()
        out: List[Dict[str, object]] = []
        for kernel_name, vals in grouped.items():
            total = sum(vals)
            avg = total / len(vals)
            mx = max(vals)
            mn = min(vals)
            if score_name == "avg":
                score_val = avg
            elif score_name == "max":
                score_val = mx
            elif score_name == "min":
                score_val = mn
            else:
                score_val = total
            out.append(
                {
                    "kernel_name": kernel_name,
                    "samples": len(vals),
                    "sum": total,
                    "avg": avg,
                    "max": mx,
                    "min": mn,
                    "score": score_val,
                    "score_mode": score_name,
                }
            )
        out.sort(key=lambda x: float(x["score"]), reverse=True)
        return out[: int(top_k)]

    def _skill_all_metrics(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        limit: int = 20000,
    ) -> List[Dict[str, object]]:
        rows = self._filtered_records(metric_like=metric_like, kernel_like=kernel_like)
        out: List[Dict[str, object]] = []
        for rec in rows[: int(limit)]:
            out.append(
                {
                    "range_index": rec.range_index,
                    "action_index": rec.action_index,
                    "kernel_name": rec.kernel_name,
                    "metric_name": rec.metric_name,
                    "raw_value": rec.raw_value,
                    "numeric_value": rec.numeric_value,
                    "unit": rec.unit,
                }
            )
        return out

    def _rule_payload(self, *, kernel_like: str = "%", top_k: int = 200) -> Dict[str, object]:
        rows = load_ncu_report_rule_rows(
            self.report_path,
            kernel_like=kernel_like,
            ncu_report_module=self._ncu_report_module,
        )
        summary = _build_rule_summary(rows, top_k=top_k)
        return {
            "report_path": self.report_path,
            "kernel_like": kernel_like,
            "total_rule_rows": summary["total_rows"],
            "top_rows": summary["top_rows"],
            "top_rules": summary["top_rules"],
        }

    def _skill_rule_results(self, *, kernel_like: str = "%", top_k: int = 200) -> Dict[str, object]:
        return self._rule_payload(kernel_like=kernel_like, top_k=top_k)

    def _skill_source_attribution(
        self, *, kernel_like: str = "%", top_k: int = 15, metric_like: str = "",
    ) -> Dict[str, object]:
        """Attribute stalls and instruction counts to source lines.

        This is the only analysis in the package that answers *where* rather than
        *what*, which for a fused kernel is usually the whole question: knowing
        the kernel stalls on long-scoreboard does not say which of the fused
        stages is stalling, and no whole-kernel counter can.

        Everything here is gated on sampling validity first. Dropped samples and
        buffer overflows bias the distribution toward whatever ran early, so a
        ranking built on them is confidently wrong rather than merely noisy.
        """
        from .sampling_validity import (
            check_pc_sampling_validity,
            check_pm_sampling_validity,
        )
        from .source_correlation import (
            attribute_stalls_to_source,
            correlate_metric_to_source,
            pc_sampling_timeline,
            source_availability,
            summarize_warp_samples,
        )

        mod = _load_ncu_report_module(self._ncu_report_module)
        ctx = mod.load_report(str(self.report_path))

        out: List[Dict[str, object]] = []
        for range_obj in _iter_ranges(ctx):
            for action in _iter_actions(range_obj):
                name = str(_maybe_call(action, "name", "") or "")
                if not _like_match(name, kernel_like):
                    continue

                availability = source_availability(action)

                def _metric(key: str, _action: object = action) -> Optional[float]:
                    # _maybe_call takes no call arguments, and metric_by_name
                    # needs one, so this reads the attribute directly.
                    getter = getattr(_action, "metric_by_name", None)
                    if getter is None:
                        return None
                    try:
                        metric = getter(key)
                    except Exception:
                        return None
                    if metric is None:
                        return None
                    value = _maybe_call(metric, "as_double", None)
                    if value is None:
                        value = _maybe_call(metric, "as_uint64", None)
                    try:
                        return float(value) if value is not None else None
                    except (TypeError, ValueError):
                        return None

                validity = check_pc_sampling_validity(
                    sample_count=_metric("smsp__pcsamp_sample_count"),
                    interval_cycles=_metric("smsp__pcsamp_interval_cycles"),
                    kernel_duration_cycles=_metric("gpc__cycles_elapsed.max"),
                    dropped_bytes=_metric("smsp__pcsamp_dropped_bytes"),
                    buffer_overflow=_metric("smsp__pcsamp_buffer_overflow"),
                    buffer_size_bytes=_metric("smsp__pcsamp_buffer_size_bytes"),
                )
                blocked = set(validity.get("blocked_conclusions") or ())

                # PM sampling is a separate instrument with its own validity
                # rules (architecture floor, interval vs duration). Reporting
                # only PC-sampling validity would leave a PM timeline unchecked.
                pm_validity = check_pm_sampling_validity(
                    cc_major=_metric("device__attribute_compute_capability_major"),
                    cc_minor=_metric("device__attribute_compute_capability_minor"),
                    interval=(_metric("profiler__pmsampler_interval_time")
                              or _metric("profiler__pmsampler_interval_cycles")),
                    duration=(_metric("gpu__time_duration.sum")
                              or _metric("gpc__cycles_elapsed.max")),
                    pass_groups=_metric("profiler__pmsampler_pass_groups"),
                )

                entry: Dict[str, object] = {
                    "kernel_name": name,
                    "availability": availability,
                    "sampling_validity": validity,
                    "pm_sampling_validity": pm_validity,
                    "warp_sample_summary": summarize_warp_samples(action),
                }

                # Each analysis is withheld individually: too few samples blocks
                # ranking lines against each other but not the overall
                # distribution, and saying so is more useful than one verdict.
                if "hot_line_ranking" in blocked or "stall_attribution" in blocked:
                    entry["stall_attribution"] = {
                        "available": False,
                        "withheld_because": sorted(blocked),
                        "reason": (
                            "Source-line ranking was withheld: the PC samples in this "
                            "report cannot support it. See sampling_validity."
                        ),
                    }
                else:
                    entry["stall_attribution"] = attribute_stalls_to_source(
                        action, top_k=int(top_k))

                if "pc_sampling_timeline" in blocked:
                    entry["timeline"] = {
                        "available": False,
                        "withheld_because": sorted(blocked),
                    }
                else:
                    entry["timeline"] = pc_sampling_timeline(action)

                if metric_like:
                    entry["metric_attribution"] = correlate_metric_to_source(
                        action, metric_like, top_k=int(top_k))

                out.append(entry)

        return {
            "kernels": out,
            "kernel_count": len(out),
            "note": (
                "Empty results here mean source data was not collected, not that the "
                "kernel has no hot lines. A bare `ncu` run uses the `basic` set, which "
                "does not include SourceCounters; use --set full or "
                "--section SourceCounters, and build with -lineinfo."
                if not any(k["availability"]["source_correlation_possible"] for k in out)
                else ""
            ),
        }

    def _skill_bottleneck_report(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        top_k: int = 10,
    ) -> Dict[str, object]:
        metric_stats = self._skill_per_metric_stats(metric_like=metric_like, kernel_like=kernel_like)
        all_rule_rows = load_ncu_report_rule_rows(
            self.report_path,
            kernel_like=kernel_like,
            ncu_report_module=self._ncu_report_module,
        )
        summary = _build_rule_summary(all_rule_rows, top_k=max(200, int(top_k) * 10))
        report = _build_bottleneck_report(metric_stats, all_rule_rows, top_k=top_k)
        report["rule_results"] = {
            "total_rule_rows": int(summary.get("total_rows", 0) or 0),
            "top_rules": summary.get("top_rules", []),
            "top_rows": summary.get("top_rows", []),
        }
        report["filters"] = {"metric_like": metric_like, "kernel_like": kernel_like}
        return report

    def _skill_dimension_report(
        self,
        *,
        metric_like: str = "%",
        kernel_like: str = "%",
        top_k: int = 10,
    ) -> Dict[str, object]:
        metric_stats = self._skill_per_metric_stats(metric_like=metric_like, kernel_like=kernel_like)
        return {
            "report_path": self.report_path,
            "filters": {"metric_like": metric_like, "kernel_like": kernel_like},
            **_build_dimension_report(metric_stats, top_k=top_k),
        }

    def _build_skills(self) -> Dict[str, ReportSkill]:
        return {
            "summary": ReportSkill(
                name="summary",
                title="Report Summary",
                description="Summarize parsed metric records and value stats.",
                category="overview",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern (%/_/*)", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern (%/_/*)", "str", False, "%"),
                    SkillParam("top_k", "top rows limit", "int", False, 20),
                ],
                run_fn=self._skill_summary,
            ),
            "per_metric_stats": ReportSkill(
                name="per_metric_stats",
                title="Per Metric Stats",
                description="Compute count/min/max/avg/p50/p90/p99 for each metric.",
                category="metrics",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                ],
                run_fn=self._skill_per_metric_stats,
            ),
            "top_kernels": ReportSkill(
                name="top_kernels",
                title="Top Kernels",
                description="Aggregate selected numeric metric values by kernel.",
                category="kernels",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                    SkillParam("top_k", "top rows limit", "int", False, 20),
                    SkillParam("score", "score mode: sum|avg|max|min", "str", False, "sum"),
                ],
                run_fn=self._skill_top_kernels,
            ),
            "all_metrics": ReportSkill(
                name="all_metrics",
                title="All Metrics Rows",
                description="Dump all parsed metric rows (with optional limit).",
                category="inspect",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                    SkillParam("limit", "max output rows", "int", False, 20000),
                ],
                run_fn=self._skill_all_metrics,
            ),
            "rule_results": ReportSkill(
                name="rule_results",
                title="Rule Results",
                description="Read built-in NCU rule findings (rule_results_as_dicts).",
                category="diagnose",
                params=[
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                    SkillParam("top_k", "top rows limit", "int", False, 200),
                ],
                run_fn=self._skill_rule_results,
            ),
            "source_attribution": ReportSkill(
                name="source_attribution",
                title="Source Attribution",
                description=(
                    "Attribute stalls and metrics to source lines and SASS, gated on "
                    "PC-sampling validity."
                ),
                params=[
                    SkillParam("kernel_like", "kernel name filter", "str", False, "%"),
                    SkillParam("top_k", "max source lines", "int", False, 15),
                    SkillParam("metric_like", "extra metric to attribute", "str", False, ""),
                ],
                run_fn=self._skill_source_attribution,
            ),
            "bottleneck_report": ReportSkill(
                name="bottleneck_report",
                title="Bottleneck Report",
                description="Combine NCU rules + heuristic fallback + coverage check.",
                category="diagnose",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                    SkillParam("top_k", "top bottlenecks limit", "int", False, 10),
                ],
                run_fn=self._skill_bottleneck_report,
            ),
            "dimension_report": ReportSkill(
                name="dimension_report",
                title="Six-Dimension Diagnostic Report",
                description="Diagnose occupancy, balance, stalls, tensor core, timeline, and memory signals.",
                category="diagnose",
                params=[
                    SkillParam("metric_like", "metric LIKE pattern", "str", False, "%"),
                    SkillParam("kernel_like", "kernel LIKE pattern", "str", False, "%"),
                    SkillParam("top_k", "top findings/signals limit", "int", False, 10),
                ],
                run_fn=self._skill_dimension_report,
            ),
        }


def load_ncu_report_records(
    report_path: str,
    *,
    metric_like: str = "%",
    kernel_like: str = "%",
    limit_actions: int = -1,
    ncu_report_module: Any = None,
) -> List[NcuReportMetricRecord]:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")

    mod = _load_ncu_report_module(ncu_report_module)
    ctx = mod.load_report(str(path))

    out: List[NcuReportMetricRecord] = []
    action_counter = 0
    ranges = _iter_ranges(ctx)
    for range_idx, range_obj in enumerate(ranges):
        actions = _iter_actions(range_obj)
        for action_idx, action in enumerate(actions):
            action_counter += 1
            if int(limit_actions) > 0 and action_counter > int(limit_actions):
                return out
            kernel_name = str(_maybe_call(action, "name", "") or "")
            if not _like_match(kernel_name, kernel_like):
                continue
            try:
                metric_names = list(action.metric_names())  # type: ignore[attr-defined]
            except Exception:
                metric_names = []
            for metric_name in metric_names:
                name = str(metric_name)
                if not _like_match(name, metric_like):
                    continue
                try:
                    metric = action.metric_by_name(name)  # type: ignore[attr-defined]
                except Exception:
                    continue
                raw_value = _metric_value(metric)
                unit = str(_maybe_call(metric, "unit", "") or "")
                out.append(
                    NcuReportMetricRecord(
                        kernel_name=kernel_name,
                        metric_name=name,
                        raw_value=raw_value,
                        numeric_value=_to_number(raw_value),
                        unit=unit,
                        range_index=range_idx,
                        action_index=action_idx,
                    )
                )
    return out


def _metric_reader(action: Any):
    """Return a `(key) -> Optional[float]` reader for one action.

    `_maybe_call` takes no call arguments and `metric_by_name` needs one, so the
    attribute is read directly.
    """
    def read(key: str) -> Optional[float]:
        getter = getattr(action, "metric_by_name", None)
        if getter is None:
            return None
        try:
            metric = getter(key)
        except Exception:
            return None
        if metric is None:
            return None
        value = _maybe_call(metric, "as_double", None)
        if value is None:
            value = _maybe_call(metric, "as_uint64", None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return read


def _rule_rows_for_action(
    action: Any, *, range_idx: int, action_idx: int, kernel_name: str,
) -> List[Dict[str, object]]:
    """Normalise one action's shipped rule results into flat rows."""
    raw_rules = _maybe_call(action, "rule_results_as_dicts", None)
    if raw_rules is None:
        raw_rules = _maybe_call(action, "rule_results", None)
    if raw_rules is None:
        return []
    if not isinstance(raw_rules, list):
        try:
            raw_rules = list(raw_rules)
        except Exception:
            return []

    rows: List[Dict[str, object]] = []
    for rule_idx, rule_item in enumerate(raw_rules):
        item = dict(rule_item) if isinstance(rule_item, dict) else {}
        rule_message = item.get("rule_message", {})
        if not isinstance(rule_message, dict):
            rule_message = {}
        speedup = item.get("speedup_estimation", {})
        if not isinstance(speedup, dict):
            speedup = {}
        message_type = rule_message.get("message_type", rule_message.get("type", ""))
        rows.append({
            "range_index": range_idx,
            "action_index": action_idx,
            "kernel_name": kernel_name,
            "rule_index": rule_idx,
            "rule_identifier": str(item.get("rule_identifier", "") or ""),
            "rule_name": str(item.get("name", "") or ""),
            "section_identifier": str(item.get("section_identifier", "") or ""),
            "parent_weights": item.get("parent_weights", {}),
            "rule_message_title": str(rule_message.get("title", "") or ""),
            "rule_message_type": _enum_to_text(message_type),
            "rule_message": str(rule_message.get("message", "") or ""),
            "speedup_type": _enum_to_text(speedup.get("type", "")),
            "speedup": _to_number(speedup.get("speedup")),
            "focus_metrics": _focus_metrics_summary(item.get("focus_metrics", []), top_k=20),
        })
    return rows


def _metrics_for_action(
    action: Any, *, metric_like: str = "%",
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Every metric this action carries, split into numeric and string-valued.

    No curated filter: `metric_names()` is whatever ncu recorded, which for a
    --set full collection is thousands of entries.

    The string-valued ones were previously discarded as unparseable. On a real
    H100 report that silently dropped 21 metrics, and they are not noise:
    `device__attribute_display_name` is the GPU model (which the caller was
    being asked to supply by hand), `breakdown:<metric>` lists the constituents
    a Speed-of-Light rollup maxes over, and `launch__*` carries scheduling
    policy and cache config. Returned separately rather than dropped.
    """
    try:
        names = list(action.metric_names())
    except Exception:
        return {}, {}
    numeric: Dict[str, float] = {}
    text: Dict[str, str] = {}
    for raw in names:
        name = str(raw)
        if not _like_match(name, metric_like):
            continue
        try:
            metric = action.metric_by_name(name)
        except Exception:
            continue
        value = _metric_value(metric)
        number = _to_number(value)
        if number is not None:
            numeric[name] = number
        elif isinstance(value, str) and value:
            text[name] = value
    return numeric, text


# The report names the GPU it was collected on, so asking the caller for it is
# asking for something we already have. Only used when no name was supplied.
_GPU_NAME_METRIC = "device__attribute_display_name"


def gpu_name_from_report(string_metrics: Mapping[str, str]) -> str:
    """GPU model recorded in the report, or "" when absent."""
    return str(string_metrics.get(_GPU_NAME_METRIC, "") or "").strip()


def resolve_sol_breakdown(
    string_metrics: Mapping[str, str],
    numeric_metrics: Mapping[str, float],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Say which constituent drives each Speed-of-Light rollup.

    A SOL throughput is a **maximum** over its constituents, not an average, so
    "SM throughput 34%" can mean one sub-unit at 34% and everything else idle.
    The report carries the constituent list as a `breakdown:<metric>` string;
    resolving it against the numeric metrics turns a single opaque percentage
    into the sub-unit actually responsible.
    """
    out: Dict[str, Any] = {}
    for name, listing in (string_metrics or {}).items():
        if not name.startswith("breakdown:"):
            continue
        target = name[len("breakdown:"):]
        parts = [p.strip() for p in str(listing).split(",") if p.strip()]
        resolved = [(p, numeric_metrics[p]) for p in parts if p in numeric_metrics]
        if not resolved:
            continue
        resolved.sort(key=lambda kv: kv[1], reverse=True)
        out[target] = {
            "rollup_value": numeric_metrics.get(target),
            "constituent_count": len(parts),
            "resolved_count": len(resolved),
            "top_constituents": [
                {"metric": m, "value": v} for m, v in resolved[: int(top_k)]
            ],
            "note": (
                "A Speed-of-Light throughput is the maximum over these, not their "
                "average: the top entry is what the headline number is measuring."
            ),
        }
    return out


def _source_for_action(action: Any, *, top_k: int = 8) -> Dict[str, object]:
    """Source attribution for one action, gated on sampling validity."""
    from .sampling_validity import check_pc_sampling_validity, check_pm_sampling_validity
    from .source_correlation import (
        analyze_pm_sampling,
        attribute_stalls_to_source,
        pc_sampling_timeline,
        source_availability,
        top_stalling_instructions,
    )

    read = _metric_reader(action)
    validity = check_pc_sampling_validity(
        sample_count=read("smsp__pcsamp_sample_count"),
        interval_cycles=read("smsp__pcsamp_interval_cycles"),
        kernel_duration_cycles=read("gpc__cycles_elapsed.max"),
        dropped_bytes=read("smsp__pcsamp_dropped_bytes"),
        buffer_overflow=read("smsp__pcsamp_buffer_overflow"),
        buffer_size_bytes=read("smsp__pcsamp_buffer_size_bytes"),
    )
    blocked = set(validity.get("blocked_conclusions") or ())

    entry: Dict[str, object] = {
        "availability": source_availability(action),
        "sampling_validity": validity,
        "pm_sampling_validity": check_pm_sampling_validity(
            cc_major=read("device__attribute_compute_capability_major"),
            cc_minor=read("device__attribute_compute_capability_minor"),
            interval=(read("profiler__pmsampler_interval_time")
                      or read("profiler__pmsampler_interval_cycles")),
            duration=(read("gpu__time_duration.sum")
                      or read("gpc__cycles_elapsed.max")),
            pass_groups=read("profiler__pmsampler_pass_groups"),
        ),
    }
    if blocked & {"hot_line_ranking", "stall_attribution"}:
        entry["stall_attribution"] = {
            "available": False,
            "withheld_because": sorted(blocked),
            "reason": (
                "Source-line ranking withheld: the PC samples in this report "
                "cannot support it. See sampling_validity."
            ),
        }
    else:
        entry["stall_attribution"] = attribute_stalls_to_source(action, top_k=top_k)
    if "pc_sampling_timeline" not in blocked:
        entry["timeline"] = pc_sampling_timeline(action)
    # One level finer than the line view: the exact instruction.
    if not (blocked & {"hot_line_ranking", "stall_attribution"}):
        entry["top_instructions"] = top_stalling_instructions(action, top_k=top_k)
    # PM sampling is a separate instrument: PC sampling says where, this says when.
    entry["pm_sampling"] = analyze_pm_sampling(action)
    return entry


@dataclass
class _LaunchBundle:
    """Everything one kernel launch contributes, gathered in a single visit."""

    kernel_name: str
    action: Any
    metrics: Dict[str, float]
    rules: List[Dict[str, object]]
    string_metrics: Dict[str, str] = field(default_factory=dict)
    source: Optional[Dict[str, object]] = None


def walk_report_once(
    report_path: str,
    *,
    kernel_like: str = "%",
    include_source: bool = True,
    source_top_k: int = 8,
    ncu_report_module: Any = None,
) -> Dict[Tuple[int, int], _LaunchBundle]:
    """Read the report once and return everything the diagnosis needs.

    `diagnose_ncu_report` previously called four separate loaders, each of which
    opened the report and walked every range and action: one for metrics, one
    for shipped rules, one for source attribution, one to keep the action
    objects. Four full traversals of a --set full report, three of them
    redundant. This visits each action exactly once and gathers all four.
    """
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")

    mod = _load_ncu_report_module(ncu_report_module)
    ctx = mod.load_report(str(path))

    bundles: Dict[Tuple[int, int], _LaunchBundle] = {}
    for range_idx, range_obj in enumerate(_iter_ranges(ctx)):
        for action_idx, action in enumerate(_iter_actions(range_obj)):
            kernel_name = str(_maybe_call(action, "name", "") or "")
            if not _like_match(kernel_name, kernel_like):
                continue
            source: Optional[Dict[str, object]] = None
            if include_source:
                try:
                    source = _source_for_action(action, top_k=source_top_k)
                except Exception:
                    # Source data is optional and often absent. Its failure must
                    # not cost us the metrics and rules already gathered here.
                    source = None
            numeric, text = _metrics_for_action(action)
            bundles[(range_idx, action_idx)] = _LaunchBundle(
                kernel_name=kernel_name,
                action=action,
                metrics=numeric,
                string_metrics=text,
                rules=_rule_rows_for_action(
                    action, range_idx=range_idx, action_idx=action_idx,
                    kernel_name=kernel_name),
                source=source,
            )
    return bundles


def _collect_source_attribution(
    report_path: str,
    *,
    kernel_like: str = "%",
    top_k: int = 8,
    ncu_report_module: Any = None,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Per-launch source attribution, gated on PC-sampling validity.

    Kept for callers that want only this. `diagnose_ncu_report` gets the same
    data from :func:`walk_report_once` without a second traversal.
    """
    bundles = walk_report_once(
        report_path, kernel_like=kernel_like, include_source=True,
        source_top_k=top_k, ncu_report_module=ncu_report_module,
    )
    return {key: b.source for key, b in bundles.items() if b.source is not None}

def diagnose_ncu_report(
    report_path: str,
    *,
    kernel_like: str = "%",
    top_kernels: int = 10,
    findings_per_kernel: int = 8,
    gpu_name: str = "",
    include_source: bool = True,
    ncu_report_module: Any = None,
) -> Dict[str, object]:
    """Run the full rule engine over every profiled kernel in a report.

    Unlike :func:`analyze_ncu_report`, which summarises metrics, this walks each
    kernel launch and produces a ranked, evidence-carrying diagnosis: bottleneck
    class, stall attribution, roofline placement, occupancy limiter, coalescing,
    bank conflicts, divergence, spilling and launch geometry.

    ``gpu_name`` unlocks the absolute roofline. When empty, the GPU is inferred
    from the report's device attributes where possible; failing that the
    analysis still runs but reports arithmetic intensity without a ceiling.
    """
    from ..hardware.gpu_specs import lookup_gpu_spec
    from .ncu_diagnostics import diagnose_kernel

    # One traversal. This used to be four separate loaders, each opening the
    # report and walking every action: metrics, shipped rules, source
    # attribution, and the action objects themselves.
    bundles = walk_report_once(
        report_path,
        kernel_like=kernel_like,
        include_source=include_source,
        source_top_k=int(findings_per_kernel),
        ncu_report_module=ncu_report_module,
    )

    # The report records the GPU it ran on. Asking the caller to supply a name
    # we already have is how a roofline ends up with no ceiling for no reason.
    detected_gpu = ""
    for bundle in bundles.values():
        detected_gpu = gpu_name_from_report(bundle.string_metrics)
        if detected_gpu:
            break
    effective_gpu = gpu_name or detected_gpu
    gpu_spec = lookup_gpu_spec(effective_gpu) if effective_gpu else None

    diagnoses: List[Dict[str, object]] = []
    for (range_idx, action_idx), bundle in sorted(bundles.items()):
        metrics = bundle.metrics
        if not metrics:
            continue

        diagnosis = diagnose_kernel(
            metrics,
            kernel_name=bundle.kernel_name,
            gpu_spec=gpu_spec,
            top_k=int(findings_per_kernel),
            # NVIDIA's own rule output for this same launch.
            shipped_rules=bundle.rules,
            string_metrics=bundle.string_metrics,
        )
        diagnosis["range_index"] = range_idx
        diagnosis["action_index"] = action_idx
        diagnosis["duration_ns"] = metrics.get("gpu__time_duration.sum")

        # Reason over every metric the report carried, not only the curated
        # ones. The curated rules know fixes; this finds anomalies in the
        # counters no rule was written for.
        from .signal_scan import scan_all_signals

        scan = scan_all_signals(metrics)
        diagnosis["signal_scan"] = {k: v for k, v in scan.items() if k != "findings"}

        # String-valued metrics: previously dropped as unparseable. They carry
        # the GPU model, the launch scheduling policy, and the constituent lists
        # a Speed-of-Light rollup maxes over.
        diagnosis["string_metrics"] = dict(bundle.string_metrics)
        diagnosis["sol_breakdown"] = resolve_sol_breakdown(
            bundle.string_metrics, metrics)
        if scan["findings"]:
            merged = list(diagnosis.get("findings") or [])
            merged.extend(f.to_dict() for f in scan["findings"])
            merged.sort(key=lambda f: (
                {"high": 0, "medium": 1, "low": 2, "info": 3}.get(f.get("severity"), 9),
                -(f.get("speedup_ceiling") or 1.0),
            ))
            diagnosis["findings"] = merged[: int(findings_per_kernel)]

        if bundle.source is not None:
            diagnosis["source_attribution"] = bundle.source
            # The join: each finding against the lines whose sampled stalls
            # explain it. This is what makes the pipeline end-to-end rather
            # than two analyses printed next to each other.
            attribution = bundle.source.get("stall_attribution")
            if isinstance(attribution, dict) and attribution.get("available"):
                from .source_correlation import link_findings_to_source

                diagnosis["signal_to_source"] = link_findings_to_source(
                    diagnosis.get("findings") or [],
                    bundle.action,
                    attribution=attribution,
                )
        diagnoses.append(diagnosis)

    diagnoses.sort(key=lambda d: -(float(d.get("duration_ns") or 0.0)))

    verdict_counts: Dict[str, int] = {}
    finding_counts: Dict[str, int] = {}
    for diagnosis in diagnoses:
        verdict = str(diagnosis.get("verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        for finding in diagnosis.get("findings", []) or []:
            category = str(finding.get("category") or "")
            finding_counts[category] = finding_counts.get(category, 0) + 1

    return {
        "report_path": str(report_path),
        "gpu": gpu_spec.name if gpu_spec else "",
        "gpu_detected_from_report": detected_gpu,
        "gpu_name_source": (
            "caller" if gpu_name else ("report" if detected_gpu else "unknown")
        ),
        "kernels_analyzed": len(diagnoses),
        "verdict_counts": dict(sorted(verdict_counts.items(), key=lambda kv: -kv[1])),
        "finding_counts": dict(sorted(finding_counts.items(), key=lambda kv: -kv[1])),
        "kernels": diagnoses[: int(top_kernels)],
    }


def diagnose_result_to_markdown(payload: Dict[str, object]) -> str:
    """Render :func:`diagnose_ncu_report` output as a readable report."""
    if not isinstance(payload, dict):
        return "# NCU Diagnosis\n\n(no payload)\n"

    lines: List[str] = ["# NCU Kernel Diagnosis", ""]
    gpu = str(payload.get("gpu") or "")
    lines.append(f"- report: `{payload.get('report_path', '')}`")
    lines.append(f"- kernels analyzed: {payload.get('kernels_analyzed', 0)}")
    lines.append(f"- gpu: {gpu or '(not supplied - roofline ceilings unavailable)'}")
    lines.append("")

    verdicts = payload.get("verdict_counts", {})
    if isinstance(verdicts, dict) and verdicts:
        lines.append("## Bottleneck classes")
        lines.append("")
        lines.append("| verdict | kernels |")
        lines.append("|---|---|")
        for name, count in verdicts.items():
            lines.append(f"| {name} | {count} |")
        lines.append("")

    finding_counts = payload.get("finding_counts", {})
    if isinstance(finding_counts, dict) and finding_counts:
        lines.append("## Most frequent findings")
        lines.append("")
        lines.append("| finding | kernels |")
        lines.append("|---|---|")
        for name, count in list(finding_counts.items())[:15]:
            lines.append(f"| {name} | {count} |")
        lines.append("")

    kernels = payload.get("kernels", [])
    if isinstance(kernels, list):
        for index, kernel in enumerate(kernels, start=1):
            if not isinstance(kernel, dict):
                continue
            name = str(kernel.get("kernel_name") or "(unnamed)")
            duration_ns = kernel.get("duration_ns")
            duration = f" - {float(duration_ns) / 1000.0:.1f} us" if duration_ns else ""
            lines.append(f"## {index}. {name}{duration}")
            lines.append("")
            category = kernel.get("kernel_category") or "?"
            framework = kernel.get("kernel_framework") or "-"
            arch = kernel.get("architecture", {})
            arch_alias = arch.get("alias", "?") if isinstance(arch, dict) else "?"
            lines.append(f"- category: `{category}` | framework: `{framework}` | arch: `{arch_alias}`")
            lines.append(f"- **verdict: {kernel.get('verdict', 'unknown')}**")

            sections = kernel.get("sections", {})
            bottleneck = sections.get("bottleneck", {}) if isinstance(sections, dict) else {}
            if isinstance(bottleneck, dict) and bottleneck.get("explanation"):
                lines.append(f"- {bottleneck['explanation']}")
                if bottleneck.get("next_section"):
                    lines.append(f"- read next: `{bottleneck['next_section']}`")

            roofline = sections.get("roofline", {}) if isinstance(sections, dict) else {}
            if isinstance(roofline, dict) and roofline.get("arithmetic_intensity"):
                pieces = [f"AI = {roofline['arithmetic_intensity']:.1f} FLOP/byte"]
                if roofline.get("achieved_tflops"):
                    pieces.append(f"achieved {roofline['achieved_tflops']:.1f} TFLOP/s")
                if roofline.get("attainable_tflops"):
                    pieces.append(f"ceiling {roofline['attainable_tflops']:.1f} TFLOP/s")
                if roofline.get("roofline_side") not in (None, "unknown"):
                    pieces.append(str(roofline["roofline_side"]))
                lines.append(f"- roofline: {', '.join(pieces)}")
                if roofline.get("flops_undercounted"):
                    lines.append(f"  - NOTE: {roofline.get('flops_undercount_reason', '')}")

            stalls = sections.get("stalls", {}) if isinstance(sections, dict) else {}
            if isinstance(stalls, dict) and stalls.get("dominant_bucket"):
                lines.append(f"- dominant stall bucket: `{stalls['dominant_bucket']}`")

            axes_block = kernel.get("axes", {})
            if isinstance(axes_block, dict) and axes_block.get("summary"):
                lines.append(f"- axes: {axes_block['summary']}")
            corrob = kernel.get("corroboration", {})
            if isinstance(corrob, dict) and corrob.get("conflicts"):
                lines.append(
                    f"- **{len(corrob['conflicts'])} disagreement(s) with Nsight Compute's "
                    "own rules - resolve before acting**"
                )
            lines.append("")

            # Where, not just what. For a fused kernel this is usually the only
            # actionable part, so it is rendered inline rather than left to a
            # separate command.
            source = kernel.get("source_attribution", {})
            if isinstance(source, dict):
                attribution = source.get("stall_attribution", {})
                if isinstance(attribution, dict) and attribution.get("available"):
                    rows = attribution.get("source_lines") or []
                    if rows:
                        lines.append("### Where it stalls")
                        lines.append("")
                        lines.append("| source | samples | dominant stall | line |")
                        lines.append("|---|---|---|---|")
                        for row in rows[:8]:
                            short_path = str(row.get("file_name") or "?").split("/")[-1]
                            lines.append(
                                f"| `{short_path}:{row.get('line','?')}` "
                                f"| {row.get('samples', 0)} "
                                f"| {row.get('dominant_stall_reason','')} "
                                f"| `{(row.get('source_text') or '')[:60]}` |"
                            )
                        lines.append("")
                        note = attribution.get("confidence_note")
                        if note:
                            lines.append(f"_{note}_")
                            lines.append("")
                elif isinstance(attribution, dict) and attribution.get("reason"):
                    lines.append(f"- source attribution unavailable: {attribution['reason']}")
                    lines.append("")
                    # Only when attribution actually failed. `source_availability`
                    # reports on source-correlated *metrics*; stall attribution
                    # can succeed without them. Printing its reasons after a
                    # successful table said "no source data" directly beneath the
                    # source data.
                    availability = source.get("availability", {})
                    if isinstance(availability, dict):
                        for reason in (availability.get("reasons_unavailable") or [])[:2]:
                            lines.append(f"  - {reason}")
                        if availability.get("reasons_unavailable"):
                            lines.append("")

                # --- PC sampling: which instruction, and is the data sound ----
                validity = source.get("sampling_validity", {})
                if isinstance(validity, dict) and validity.get("checked"):
                    state = ("usable" if validity.get("usable") else
                             "NOT usable -- see blocked conclusions")
                    lines.append(f"### PC sampling ({state})")
                    lines.append("")
                    lines.append(
                        f"- {validity.get('sample_count') or 0:,.0f} samples at a "
                        f"{validity.get('interval_cycles') or 0:,.0f}-cycle interval"
                    )
                    for issue in (validity.get("issues") or [])[:3]:
                        lines.append(
                            f"- **{issue.get('title','')}** -- {issue.get('remedy','')}")
                    if validity.get("blocked_conclusions"):
                        lines.append(
                            "- blocked: `"
                            + "`, `".join(validity["blocked_conclusions"]) + "`")
                    lines.append("")

                instructions = source.get("top_instructions", {})
                if isinstance(instructions, dict) and instructions.get("available"):
                    rows = instructions.get("instructions") or []
                    if rows:
                        lines.append("#### Stalling instructions")
                        lines.append("")
                        lines.append(
                            "One source line compiles to many instructions and they do "
                            "not stall for the same reason, so this is a level finer "
                            "than the table above."
                        )
                        lines.append("")
                        lines.append("| samples | stall | SASS | source |")
                        lines.append("|---|---|---|---|")
                        for row in rows[:10]:
                            where = (
                                f"{str(row.get('file_name') or '?').split('/')[-1]}"
                                f":{row.get('line', '?')}"
                            )
                            lines.append(
                                f"| {row.get('samples', 0)} "
                                f"({float(row.get('share') or 0) * 100:.1f}%) "
                                f"| {row.get('dominant_stall_reason','')} "
                                f"| `{(row.get('sass') or '')[:44]}` "
                                f"| `{where}` |"
                            )
                        lines.append("")
                        lines.append(f"_{instructions.get('note','')}_")
                        lines.append("")

                # --- PM sampling: when, not where -----------------------------
                pm = source.get("pm_sampling", {})
                if isinstance(pm, dict) and pm.get("available"):
                    interval = pm.get("bucket_interval_ns")
                    lines.append("### PM sampling (utilisation over time)")
                    lines.append("")
                    lines.append(
                        f"- {pm.get('bucket_count', 0)} time buckets"
                        + (f" at {interval:.0f} ns" if interval else "")
                        + (f", spanning {pm.get('sampled_span_ns', 0) / 1000.0:.1f} us"
                           if pm.get("sampled_span_ns") else "")
                    )
                    window = pm.get("active_window_ns")
                    if window:
                        lines.append(
                            f"- kernel active window: {pm.get('active_window_length')} "
                            f"buckets = {window / 1000.0:.1f} us"
                        )
                    lines.append(
                        "- each bucket value is the counter **accumulated over that "
                        "window**, not an instantaneous reading; `.avg` is the average "
                        "across SM instances, not across time"
                    )
                    if pm.get("span_note"):
                        lines.append(f"- _{pm['span_note']}_")
                    lines.append("")
                    lines.append(
                        "| metric | peak (1 bucket) | mean over active window "
                        "| non-zero share | mean over whole session |")
                    lines.append("|---|---|---|---|---|")
                    for entry in (pm.get("series") or [])[:8]:
                        unit = "%" if entry.get("is_percentage") else ""
                        lines.append(
                            f"| `{entry.get('metric','')[:48]}` "
                            f"| {entry.get('peak', 0):.1f}{unit} "
                            f"| {entry.get('mean_in_active_window', 0):.1f}{unit} "
                            f"| {float(entry.get('duty_cycle') or 0) * 100:.0f}% "
                            f"| {entry.get('mean_all_buckets', 0):.1f}{unit} |"
                        )
                    lines.append("")
                    if pm.get("denominator_note"):
                        lines.append(f"_{pm['denominator_note']}_")
                        lines.append("")
                    if pm.get("bursty"):
                        lines.append(f"**{pm.get('note','')}**")
                        lines.append("")
                        lines.append(
                            "_A unit that peaks high and averages low is not "
                            "inefficient; it is idle most of the time. The fix is to "
                            "keep it busy, not to make it faster._"
                        )
                        lines.append("")
                    if not all(e.get("is_percentage") for e in (pm.get("series") or [])):
                        lines.append(f"_{pm.get('counts_note','')}_")
                        lines.append("")
                elif isinstance(pm, dict) and pm.get("reason"):
                    lines.append(f"- PM sampling unavailable: {pm['reason']}")
                    lines.append("")

            link = kernel.get("signal_to_source", {})
            linked = link.get("linked", []) if isinstance(link, dict) else []
            if linked:
                lines.append("### Signal to source")
                lines.append("")
                for item in linked[:6]:
                    lines.append(f"**{item.get('finding_title','')}**")
                    reasons = ", ".join(item.get("matched_on_stall_reasons") or [])
                    lines.append(
                        f"- correlated via `{reasons}` "
                        f"({item.get('concentration','')}, "
                        f"{float(item.get('share_explained') or 0.0) * 100:.0f}% of those samples)"
                    )
                    for row in item.get("source_lines", [])[:3]:
                        lines.append(
                            f"  - `{row.get('file_name','?')}:{row.get('line','?')}` "
                            f"{row.get('samples', 0)} samples "
                            f"({float(row.get('share_of_reason') or 0.0) * 100:.0f}%) "
                            f"`{(row.get('source_text') or '')[:52]}`"
                        )
                    lines.append("")
                lines.append(
                    "_Correlation by stall reason, not proof of cause. A line can stall "
                    "for several reasons at once._"
                )
                lines.append("")

            findings = kernel.get("findings", [])
            if isinstance(findings, list) and findings:
                lines.append("### Findings")
                lines.append("")
                for finding in findings:
                    if not isinstance(finding, dict):
                        continue
                    ceiling = finding.get("speedup_ceiling")
                    ceiling_text = f" _(up to {float(ceiling):.2f}x)_" if ceiling else ""
                    lines.append(
                        f"- **[{finding.get('severity', 'info')}]** {finding.get('title', '')}{ceiling_text}"
                    )
                    lines.append(f"  - {finding.get('summary', '')}")
                    for action in finding.get("actions", []) or []:
                        lines.append(f"  - fix: {action}")
                lines.append("")
            else:
                lines.append("_No findings: this kernel looks healthy, or the report lacks the "
                             "metrics needed to judge it._")
                lines.append("")

    return "\n".join(lines)


def load_ncu_report_rule_rows(
    report_path: str,
    *,
    kernel_like: str = "%",
    limit_actions: int = -1,
    ncu_report_module: Any = None,
) -> List[Dict[str, object]]:
    path = Path(report_path)
    if not path.exists():
        raise FileNotFoundError(f"report not found: {report_path}")

    mod = _load_ncu_report_module(ncu_report_module)
    ctx = mod.load_report(str(path))
    out: List[Dict[str, object]] = []
    action_counter = 0
    ranges = _iter_ranges(ctx)
    for range_idx, range_obj in enumerate(ranges):
        actions = _iter_actions(range_obj)
        for action_idx, action in enumerate(actions):
            action_counter += 1
            if int(limit_actions) > 0 and action_counter > int(limit_actions):
                return out
            kernel_name = str(_maybe_call(action, "name", "") or "")
            if not _like_match(kernel_name, kernel_like):
                continue

            raw_rules = _maybe_call(action, "rule_results_as_dicts", None)
            if raw_rules is None:
                raw_rules = _maybe_call(action, "rule_results", None)
            if raw_rules is None:
                continue
            if not isinstance(raw_rules, list):
                try:
                    raw_rules = list(raw_rules)
                except Exception:
                    raw_rules = []

            for rule_idx, rule_item in enumerate(raw_rules):
                if isinstance(rule_item, dict):
                    item = dict(rule_item)
                else:
                    item = {}
                rule_message = item.get("rule_message", {})
                if not isinstance(rule_message, dict):
                    rule_message = {}
                speedup = item.get("speedup_estimation", {})
                if not isinstance(speedup, dict):
                    speedup = {}
                focus_metrics = item.get("focus_metrics", [])
                focus_summary = _focus_metrics_summary(focus_metrics, top_k=20)
                message_type = rule_message.get("message_type", rule_message.get("type", ""))

                out.append(
                    {
                        "range_index": range_idx,
                        "action_index": action_idx,
                        "kernel_name": kernel_name,
                        "rule_index": rule_idx,
                        "rule_identifier": str(item.get("rule_identifier", "") or ""),
                        "rule_name": str(item.get("name", "") or ""),
                        "section_identifier": str(item.get("section_identifier", "") or ""),
                        "parent_weights": item.get("parent_weights", {}),
                        "rule_message_title": str(rule_message.get("title", "") or ""),
                        "rule_message_type": _enum_to_text(message_type),
                        "rule_message": str(rule_message.get("message", "") or ""),
                        "speedup_type": _enum_to_text(speedup.get("type", "")),
                        "speedup": _to_number(speedup.get("speedup")),
                        "focus_metrics": focus_summary,
                    }
                )
    return out


def analyze_ncu_report(
    report_path: str,
    *,
    top_k: int = 20,
    metric_like: str = "",
    kernel_like: str = "%",
    include_all_metrics: bool = True,
    all_metrics_limit: int = 20000,
    ncu_report_module: Any = None,
) -> Dict[str, object]:
    engine = NcuReportSkillEngine(report_path, ncu_report_module=ncu_report_module)
    metric_pattern = str(metric_like or "").strip() or "%"
    summary = engine.run_skill("summary", metric_like=metric_pattern, kernel_like=kernel_like, top_k=top_k)
    per_metric_stats = engine.run_skill("per_metric_stats", metric_like=metric_pattern, kernel_like=kernel_like)

    selected_metric = metric_pattern
    if selected_metric in {"%", "*"} and isinstance(per_metric_stats, list) and per_metric_stats:
        preferred_keywords = ("duration", "elapsed", "time", "cycles", "throughput")
        for row in per_metric_stats:
            name = str(row.get("metric_name", "")).lower() if isinstance(row, dict) else ""
            if any(k in name for k in preferred_keywords):
                selected_metric = str(row.get("metric_name"))
                break
        if selected_metric in {"%", "*"}:
            first = per_metric_stats[0]
            if isinstance(first, dict):
                selected_metric = str(first.get("metric_name", selected_metric))

    top_kernels = engine.run_skill(
        "top_kernels",
        metric_like=selected_metric,
        kernel_like=kernel_like,
        top_k=top_k,
        score="sum",
    )
    bottleneck_report = engine.run_skill(
        "bottleneck_report",
        metric_like="%",
        kernel_like=kernel_like,
        top_k=max(5, int(top_k)),
    )
    payload: Dict[str, object] = {
        "summary": summary,
        "selected_metric_like": selected_metric,
        "per_metric_stats": per_metric_stats,
        "top_kernels": top_kernels,
        "bottleneck_report": bottleneck_report,
        "rule_results": (
            bottleneck_report.get("rule_results", {})
            if isinstance(bottleneck_report, dict)
            else {}
        ),
        "available_skills": engine.list_skills(),
    }
    if bool(include_all_metrics):
        payload["all_metrics"] = engine.run_skill(
            "all_metrics",
            metric_like=metric_pattern,
            kernel_like=kernel_like,
            limit=int(all_metrics_limit),
        )
    return payload


def analyze_ncu_report_to_markdown(payload: Dict[str, object]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    selected_metric_like = str(payload.get("selected_metric_like", "")) if isinstance(payload, dict) else ""
    top_kernels = payload.get("top_kernels", []) if isinstance(payload, dict) else []
    metric_stats = payload.get("per_metric_stats", []) if isinstance(payload, dict) else []
    bottleneck = payload.get("bottleneck_report", {}) if isinstance(payload, dict) else {}
    lines: List[str] = []
    lines.append("# NCU Report Analyze")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if isinstance(summary, dict):
        lines.append(f"- metric_records: {summary.get('metric_records')}")
        lines.append(f"- filtered_records: {summary.get('filtered_records')}")
        lines.append(f"- unique_metrics: {summary.get('unique_metrics')}")
        lines.append(f"- unique_kernels: {summary.get('unique_kernels')}")
        lines.append(f"- numeric_values: {summary.get('numeric_values')}")
        lines.append(f"- non_numeric_values: {summary.get('non_numeric_values')}")
    lines.append("")
    lines.append("## Bottleneck")
    lines.append("")
    if isinstance(bottleneck, dict):
        coverage = bottleneck.get("coverage", {})
        if isinstance(coverage, dict):
            lines.append(
                "- coverage: {score}% ({covered}/{total})".format(
                    score=coverage.get("coverage_score", 0),
                    covered=coverage.get("covered_categories", 0),
                    total=coverage.get("total_categories", 0),
                )
            )
            missing = coverage.get("missing_categories", [])
            if isinstance(missing, list) and missing:
                lines.append(f"- missing_categories: {', '.join(str(x) for x in missing)}")
        top_bottlenecks = bottleneck.get("top_bottlenecks", [])
        if isinstance(top_bottlenecks, list):
            for idx, item in enumerate(top_bottlenecks[:10], 1):
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"- {idx}. [{item.get('source')}] {item.get('title')} ({item.get('category')})"
                )
                summary_text = str(item.get("summary", "")).strip()
                if summary_text:
                    lines.append(f"  - {summary_text}")
        dimension_report = bottleneck.get("dimension_report", {})
        if isinstance(dimension_report, dict):
            needs_attention = dimension_report.get("needs_attention", [])
            if isinstance(needs_attention, list) and needs_attention:
                lines.append(f"- dimensions_need_attention: {', '.join(str(x) for x in needs_attention)}")
            dimensions = dimension_report.get("dimensions", [])
            if isinstance(dimensions, list):
                lines.append("")
                lines.append("### Six Dimensions")
                lines.append("")
                for dim in dimensions:
                    if not isinstance(dim, dict):
                        continue
                    lines.append(
                        f"- {dim.get('key')}: {dim.get('status')} "
                        f"({len(dim.get('signals', {}) or {})} signals, {len(dim.get('findings', []) or [])} findings)"
                    )
    lines.append("")
    lines.append("## Top Kernels")
    lines.append("")
    lines.append(f"- metric_like: `{selected_metric_like}`")
    lines.append("")
    lines.append("| kernel | samples | score | avg | max |")
    lines.append("|---|---:|---:|---:|---:|")
    if isinstance(top_kernels, list):
        for row in top_kernels:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {kernel} | {samples} | {score} | {avg} | {mx} |".format(
                    kernel=str(row.get("kernel_name", "")),
                    samples=row.get("samples", 0),
                    score=round(float(row.get("score", 0.0)), 6),
                    avg=round(float(row.get("avg", 0.0)), 6),
                    mx=round(float(row.get("max", 0.0)), 6),
                )
            )
    lines.append("")
    lines.append("## Per Metric Stats")
    lines.append("")
    lines.append("| metric | samples | numeric | p50 | p90 | p99 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    if isinstance(metric_stats, list):
        for row in metric_stats[:100]:
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {metric} | {samples} | {num} | {p50} | {p90} | {p99} |".format(
                    metric=str(row.get("metric_name", "")),
                    samples=row.get("samples", 0),
                    num=row.get("numeric_samples", 0),
                    p50=row.get("p50", ""),
                    p90=row.get("p90", ""),
                    p99=row.get("p99", ""),
                )
            )
    lines.append("")
    return "\n".join(lines)


def report_result_to_json(payload: object, *, pretty: bool = False) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2 if pretty else None)
