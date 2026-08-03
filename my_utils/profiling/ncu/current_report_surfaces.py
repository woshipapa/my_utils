# SPDX-License-Identifier: Apache-2.0
"""Discover and summarize NCU report surfaces added after a local install.

Nsight Compute evolves metric names and report-interface objects independently
of the stable Python API.  Hard-coding an unverified 2026.2 metric spelling
would be worse than not supporting it: a future report would quietly look
healthy while the new data was simply never read.  These helpers therefore
discover candidate surfaces from the report itself and keep their raw metric
names in the result.

The module is deliberately conservative.  It summarizes a surface only after
the report explicitly exposes it; no threshold or optimisation recommendation
is fabricated for a new NVIDIA metric until controlled reports establish its
units and semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "discover_current_report_surfaces",
    "summarize_instruction_breakdowns",
    "summarize_current_report_surfaces",
]


def _metric_names(action: Any) -> List[str]:
    try:
        return [str(name) for name in action.metric_names()]
    except Exception:
        return []


def _metric(action: Any, name: str) -> Any:
    try:
        return action.metric_by_name(name)
    except Exception:
        return None


def _value(metric: Any, index: Optional[int] = None) -> Any:
    if metric is None:
        return None
    args = () if index is None else (index,)
    method = getattr(metric, "value", None)
    if callable(method):
        try:
            return method(*args)
        except Exception:
            pass
    for name in ("as_double", "as_uint64", "as_string"):
        method = getattr(metric, name, None)
        if not callable(method):
            continue
        try:
            value = method(*args)
        except Exception:
            continue
        if value is not None:
            return value
    return None


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _instance_label(metric: Any, index: int) -> str:
    correlation = None
    method = getattr(metric, "correlation_ids", None)
    if callable(method):
        try:
            correlation = method()
        except Exception:
            correlation = None
    label = _value(correlation, index)
    return str(label) if label not in (None, "") else f"instance {index}"


def _metric_breakdown(action: Any, name: str, *, top_k: int) -> Dict[str, Any]:
    metric = _metric(action, name)
    if metric is None:
        return {"metric": name, "available": False, "entries": []}
    try:
        count = int(metric.num_instances())
    except Exception:
        count = 0
    entries = []
    if count:
        for index in range(count):
            value = _number(_value(metric, index))
            if value is None:
                continue
            entries.append({"label": _instance_label(metric, index), "value": value})
    else:
        value = _number(_value(metric))
        if value is not None:
            entries.append({"label": "aggregate", "value": value})
    entries.sort(key=lambda entry: float(entry["value"]), reverse=True)
    total = sum(float(entry["value"]) for entry in entries)
    for entry in entries:
        entry["share"] = float(entry["value"]) / total if total > 0 else None
    return {
        "metric": name,
        "available": bool(entries),
        "entry_count": len(entries),
        "total": total if entries else None,
        "entries": entries[: int(top_k)],
        "all_entries_available": len(entries),
    }


def discover_current_report_surfaces(action: Any) -> Dict[str, Any]:
    """Discover current-version report features without assuming metric names."""
    names = _metric_names(action)
    lower_names = {name: name.lower() for name in names}

    def matching(*tokens: str) -> List[str]:
        return [
            name
            for name, lowered in lower_names.items()
            if all(token in lowered for token in tokens)
        ]

    api_names = {name for name in dir(action) if not name.startswith("_")}
    function_api = sorted(
        name
        for name in api_names
        if "function" in name.lower() and ("stat" in name.lower() or "time" in name.lower())
    )
    return {
        "sass_instruction_size": {
            "observed": bool(matching("sass", "size")),
            "metrics": matching("sass", "size"),
            "status": "observed" if matching("sass", "size") else "not_observed",
        },
        "instruction_stats_hw_warp_id": {
            "observed": bool(matching("warp", "stall", "id")),
            "metrics": matching("warp", "stall", "id"),
            "status": "observed" if matching("warp", "stall", "id") else "not_observed",
        },
        "function_statistics_line_time_range": {
            "observed": bool(function_api),
            "api_members": function_api,
            "status": "observed" if function_api else "not_exposed_by_report_api",
        },
    }


def summarize_instruction_breakdowns(action: Any, *, top_k: int = 24) -> Dict[str, Any]:
    """Read SASS opcode/category/pipeline breakdowns when a report contains them."""
    candidates = [
        name
        for name in _metric_names(action)
        if name.startswith("sass__") and "per_opcode" in name
    ]
    breakdowns = [_metric_breakdown(action, name, top_k=top_k) for name in candidates]
    return {
        "available": bool(breakdowns),
        "metric_count": len(breakdowns),
        "breakdowns": breakdowns,
        "note": (
            "Breakdown metrics are software-counter distributions. Values are reported "
            "by their report-provided instance labels; no throughput conclusion is made "
            "without the corresponding elapsed-time and pipeline counters."
            if breakdowns
            else "No SASS per-opcode breakdown metric was collected in this report."
        ),
    }


def summarize_current_report_surfaces(action: Any, *, top_k: int = 24) -> Dict[str, Any]:
    """Summarize newly discovered report surfaces and stable opcode breakdowns."""
    discovered = discover_current_report_surfaces(action)
    summaries = {}
    for key in ("sass_instruction_size", "instruction_stats_hw_warp_id"):
        metrics = list(discovered[key].get("metrics") or ())
        summaries[key] = {
            "available": bool(metrics),
            "metrics": [_metric_breakdown(action, name, top_k=top_k) for name in metrics],
        }
    return {
        "discovery": discovered,
        "instruction_breakdowns": summarize_instruction_breakdowns(action, top_k=top_k),
        "surfaces": summaries,
    }
