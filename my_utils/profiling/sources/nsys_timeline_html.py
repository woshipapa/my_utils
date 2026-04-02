from __future__ import annotations
import csv
import hashlib
import html
import json
import math
import re
import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Pattern, Sequence, Tuple

from .nsys_flat_export import collect_kernel_rows
from .nsys_schema_adapter import NsightSchema
from .nsys_sql_skills import calculate_h100_occupancy
from .nsys_sqlite_provider import NsysSqliteMetricsProvider

_RANK_RE = re.compile(r"\brank(?:\s*|[:=_-])(\d+)\b", re.IGNORECASE)
_GPU_IDX_RE = re.compile(r"(?:gpu|device)\s*[_:#=\- ]?\s*(\d+)", re.IGNORECASE)
_DEFAULT_FOCUS_METRIC_TOKENS: Tuple[str, ...] = (
    "compute warps in flight",
    "unallocated warps in active sms",
    "unallocated warps in active sm s",
    "tensor active",
    "tensor__active",
    "dram throughput",
    "dram read bandwidth",
    "dram write bandwidth",
    "dram bytes read",
    "dram bytes write",
    "dram__throughput",
    "dram__bytes_read",
    "dram__bytes_write",
    "nvlink",
)
_DEFAULT_KERNEL_CATEGORY_MAPS: Dict[str, Dict[str, Dict[str, str]]] = {
    "sglang": {
        "llama": {
            "gemm|nvjet": "gemm",
            "fused_moe_kernel|GroupProblemShape|group_gemm_starts|bmm_|GemmUniversal": "moe_gemm",
            "moe|sigmoid": "moe",
            "CatArrayBatched|prepare_inputs": "prepare_next",
            "ncclDevKernel|cross_device_reduce": "nccl_and_custom_ar",
            "_norm_|Norm": "norm",
            "topk": "topk",
            "act_and_mul_": "activation",
            "Rotary": "rope",
            "SoftMax": "softmax",
            "flash|fmha": "attn",
            "elementwise": "elementwise",
            "fp8_quant|cvt_|quantize": "quantize",
            "reduce_kernel": "reduce",
            "triton": "triton_kernel",
            "CUDA mem": "non-gpu-H_D_memops",
            ".*": "misc",
        },
        "ds": {
            "block_fp8_matmul": "block_fp8_gemm",
            "gemm|matmul|nvjet": "gemm",
            "fused_moe_kernel": "moe_gemm",
            "moe|expert|sigmoid": "moe",
            "CatArrayBatched|write_req_to": "prepare_next",
            "ncclDevKernel|cross_device_reduce|all_gather": "nccl_and_custom_ar",
            "Norm": "norm",
            "topk": "topk",
            "activation|act_and_mul": "activation",
            "compute_position_kernel": "rope",
            "elementwise": "elementwise",
            "fp8_quant|quant_fp8|quantize": "quantize",
            "SoftMax": "softmax",
            "reduce": "reduce",
            "_fwd_|create_flash|::mla::|KVCache": "attn",
            "CUDA mem": "non-gpu-H_D_memops",
            ".*": "misc",
        },
        "gpt-oss": {
            "gemm|nvjet": "gemm",
            "fused_moe_kernel|_group_gemm|GroupProblemShape|GemmUniversal|bmm_|matmul_ogs_|_topk_forward|_combined_routing|_sum_bitmatrix_rows|_compute_writeback_idx": "moe_gemm",
            "moe|sigmoid": "moe",
            "CatArrayBatched|prepare_inputs": "prepare_next",
            "_norm_|Norm": "norm",
            "ncclDevKernel|cross_device_reduce|allreduce": "nccl_and_custom_ar",
            "topk|TopK": "topk",
            "act_and_mul_": "activation",
            "Rotary": "rope",
            "SoftMax": "softmax",
            "flash|fmha": "attn",
            "elementwise": "elementwise",
            "fp8_quant|cvt_|quantize": "quantize",
            "reduce_kernel": "reduce",
            "triton": "triton_kernel",
            "CUDA mem": "non-gpu-H_D_memops",
            ".*": "misc",
        },
    }
}


def _ident(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        raise ValueError("empty SQL identifier")
    for ch in text:
        if not (ch.isalnum() or ch == "_"):
            raise ValueError(f"unsafe SQL identifier: {name}")
    return text


def _color_for_name(name: str) -> str:
    digest = int(hashlib.md5((name or "").encode(), usedforsecurity=False).hexdigest(), 16)
    r = 70 + (digest >> 16 & 0xFF) * 130 // 255
    g = 70 + (digest >> 8 & 0xFF) * 130 // 255
    b = 70 + (digest & 0xFF) * 130 // 255
    return f"rgb({r},{g},{b})"


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _resolve_occupancy_pct_estimate(
    occupancy_pct_estimate: object,
    *,
    threads_per_block: object,
    registers_per_thread: object,
    total_shared_bytes: object,
) -> Optional[float]:
    try:
        if occupancy_pct_estimate is not None and str(occupancy_pct_estimate).strip() != "":
            occ = float(occupancy_pct_estimate)
            if math.isfinite(occ):
                return float(occ)
    except Exception:
        pass
    occ_h100 = calculate_h100_occupancy(
        threads_per_block,
        registers_per_thread,
        total_shared_bytes,
    )
    if occ_h100 is None:
        return None
    try:
        if math.isfinite(float(occ_h100)):
            return float(occ_h100)
    except Exception:
        return None
    return None


def _parse_rank_from_text(text: object) -> Optional[int]:
    s = str(text or "")
    m = _RANK_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _normalize_nvtx_like_pattern(text: object) -> str:
    """
    Normalize user NVTX query into a SQL LIKE pattern.

    Behavior:
    - empty -> "%"
    - shell-style wildcard "*" is converted to "%"
    - if resulting text contains "%", keep as-is
    - otherwise default to substring match: "%text%"
    """
    raw = str(text or "").strip()
    if not raw:
        return "%"
    normalized = raw.replace("*", "%")
    if "%" in normalized:
        return normalized
    return f"%{normalized}%"


def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return int(end_a) > int(start_b) and int(start_a) < int(end_b)


def _merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    items: List[Tuple[int, int]] = []
    for s, e in intervals:
        ss = _to_int(s, -1)
        ee = _to_int(e, -1)
        if ss < 0 or ee <= ss:
            continue
        items.append((ss, ee))
    if not items:
        return merged
    items.sort(key=lambda x: (x[0], x[1]))
    cur_s, cur_e = items[0]
    for s, e in items[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
            continue
        merged.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _percentile_linear(values: Sequence[float], q: float) -> float:
    arr = sorted(float(v) for v in values)
    if not arr:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * float(len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    frac = float(pos - lo)
    return float(arr[lo]) * (1.0 - frac) + float(arr[hi]) * frac


def _iqr_clip(values: Sequence[float], k: float = 1.5) -> Tuple[List[float], float, float]:
    vals = [float(v) for v in values]
    if len(vals) < 4:
        return list(vals), float("-inf"), float("inf")
    q1 = _percentile_linear(vals, 0.25)
    q3 = _percentile_linear(vals, 0.75)
    iqr = max(0.0, float(q3 - q1))
    if iqr <= 1e-12:
        return list(vals), float(q1), float(q3)
    low = float(q1 - float(k) * iqr)
    high = float(q3 + float(k) * iqr)
    kept = [v for v in vals if v >= low and v <= high]
    if not kept:
        kept = list(vals)
    return kept, low, high


def _series_stats_with_iqr(values: Sequence[float], k: float = 1.5) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "avg": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "avg_raw": 0.0,
            "std_raw": 0.0,
            "min_raw": 0.0,
            "max_raw": 0.0,
            "clip_low": float("-inf"),
            "clip_high": float("inf"),
            "kept_count": 0.0,
            "removed_count": 0.0,
            "count": 0.0,
        }
    mean_raw = sum(vals) / float(len(vals))
    variance_raw = sum((v - mean_raw) * (v - mean_raw) for v in vals) / float(len(vals))
    std_raw = math.sqrt(max(0.0, variance_raw))
    kept_vals, clip_low, clip_high = _iqr_clip(vals, k=k)
    mean_clipped = sum(kept_vals) / float(len(kept_vals))
    variance_clipped = sum((v - mean_clipped) * (v - mean_clipped) for v in kept_vals) / float(len(kept_vals))
    std_clipped = math.sqrt(max(0.0, variance_clipped))
    removed_count = max(0, int(len(vals) - len(kept_vals)))
    return {
        "avg": float(mean_clipped),
        "std": float(std_clipped),
        "min": float(min(kept_vals)),
        "max": float(max(kept_vals)),
        "avg_raw": float(mean_raw),
        "std_raw": float(std_raw),
        "min_raw": float(min(vals)),
        "max_raw": float(max(vals)),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "kept_count": float(len(kept_vals)),
        "removed_count": float(removed_count),
        "count": float(len(vals)),
    }


def _noop_debug(_: str) -> None:
    return


def _build_debug_logger(
    *,
    enabled: bool,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Callable[[str], None]:
    if not bool(enabled):
        return _noop_debug
    sink = log_fn or print

    def _emit(message: str) -> None:
        try:
            sink(f"[nsys-timeline-html][debug] {message}")
        except Exception:
            pass

    return _emit


def _preview_dict_rows(
    rows: Sequence[Dict[str, object]],
    *,
    keys: Sequence[str],
    limit: int = 3,
) -> str:
    try:
        max_rows = int(limit)
    except Exception:
        max_rows = 3
    out: List[Dict[str, object]] = []
    selected = list(rows) if max_rows <= 0 else list(rows)[:max_rows]
    for row in selected:
        item: Dict[str, object] = {}
        for key in keys:
            item[key] = row.get(key)
        out.append(item)
    return json.dumps(out, ensure_ascii=False)


def _safe_table_count(conn: sqlite3.Connection, table_name: str) -> Optional[int]:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {_ident(table_name)}").fetchone()
    except Exception:
        return None
    if not row:
        return None
    try:
        return int(row[0] or 0)
    except Exception:
        return None


def _resolve_kernel_category_rules(
    *,
    map_json_path: str,
    engine: str,
    model: str,
    debug_log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[Pattern[str], str]], str]:
    debug = debug_log or _noop_debug
    raw_map_obj: object = _DEFAULT_KERNEL_CATEGORY_MAPS
    map_path = str(map_json_path or "").strip()
    if map_path:
        try:
            raw_map_obj = json.loads(Path(map_path).read_text(encoding="utf-8"))
            debug(f"kernel-category map loaded from {map_path}")
        except Exception as exc:
            debug(f"kernel-category map load failed path={map_path} err={exc}; fallback to built-in defaults")
            raw_map_obj = _DEFAULT_KERNEL_CATEGORY_MAPS

    mapping: Dict[str, str] = {}
    profile_name = ""
    engine_v = str(engine or "").strip()
    model_v = str(model or "").strip()
    try:
        if isinstance(raw_map_obj, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in raw_map_obj.items()
        ):
            mapping = dict(raw_map_obj)
            profile_name = "custom:flat"
        elif isinstance(raw_map_obj, dict):
            selected_engine = None
            if engine_v and engine_v in raw_map_obj:
                selected_engine = engine_v
            elif "sglang" in raw_map_obj:
                selected_engine = "sglang"
            else:
                selected_engine = next(iter(raw_map_obj.keys()), None)
            selected_models = raw_map_obj.get(selected_engine) if selected_engine else None
            if isinstance(selected_models, dict):
                selected_model = None
                if model_v and model_v in selected_models:
                    selected_model = model_v
                elif "llama" in selected_models:
                    selected_model = "llama"
                else:
                    selected_model = next(iter(selected_models.keys()), None)
                if selected_model and isinstance(selected_models.get(selected_model), dict):
                    mapping = dict(selected_models[selected_model])
                    profile_name = f"{selected_engine}:{selected_model}"
    except Exception as exc:
        debug(f"kernel-category map resolve failed err={exc}; fallback to misc")
        mapping = {}
        profile_name = ""

    if ".*" not in mapping:
        mapping[".*"] = "misc"
    compiled: List[Tuple[Pattern[str], str]] = []
    for pattern, category in mapping.items():
        try:
            compiled.append((re.compile(str(pattern), re.IGNORECASE), str(category)))
        except Exception as exc:
            debug(f"kernel-category regex compile failed pattern={pattern} err={exc}")
    if not compiled:
        compiled = [(re.compile(".*", re.IGNORECASE), "misc")]
        profile_name = profile_name or "fallback:misc"
    if not profile_name:
        profile_name = "fallback:misc"
    debug(
        "kernel-category rules profile={} count={}".format(
            profile_name,
            len(compiled),
        )
    )
    return compiled, profile_name


def _classify_kernel_name(
    kernel_name: object,
    *,
    rules: Sequence[Tuple[Pattern[str], str]],
    cache: Optional[Dict[str, str]] = None,
) -> str:
    name = str(kernel_name or "")
    if cache is not None:
        cached = cache.get(name)
        if cached is not None:
            return cached
    category = "misc"
    for pattern, value in rules:
        if pattern.search(name):
            category = str(value or "misc")
            break
    if cache is not None:
        cache[name] = category
    return category


def _build_kernel_category_breakdown(
    kernels: Sequence[Dict[str, object]],
    *,
    rules: Sequence[Tuple[Pattern[str], str]],
    wall_start_ns: int = -1,
    wall_end_ns: int = -1,
) -> Dict[str, object]:
    events: List[Tuple[int, int, str]] = []
    intervals_by_cat: Dict[str, List[Tuple[int, int]]] = {}
    raw_ns_by_cat: Dict[str, int] = {}
    instances_by_cat: Dict[str, int] = {}
    streams_by_cat: Dict[str, set] = {}
    kinds_by_cat: Dict[str, Dict[str, int]] = {}
    cat_cache: Dict[str, str] = {}
    raw_total_ns = 0
    min_kernel_start_ns: Optional[int] = None
    max_kernel_end_ns: Optional[int] = None

    for row in kernels:
        s = _to_int(row.get("start_ns"), -1)
        e = _to_int(row.get("end_ns"), -1)
        if s < 0 or e <= s:
            continue
        if min_kernel_start_ns is None or s < min_kernel_start_ns:
            min_kernel_start_ns = int(s)
        if max_kernel_end_ns is None or e > max_kernel_end_ns:
            max_kernel_end_ns = int(e)
        name = str(row.get("kernel_name") or "")
        category = _classify_kernel_name(name, rules=rules, cache=cat_cache)
        kind = str(row.get("kind") or "compute")

        events.append((s, 1, category))
        events.append((e, -1, category))
        intervals_by_cat.setdefault(category, []).append((s, e))
        raw_ns_by_cat[category] = int(raw_ns_by_cat.get(category, 0)) + int(e - s)
        instances_by_cat[category] = int(instances_by_cat.get(category, 0)) + 1
        streams_by_cat.setdefault(category, set()).add(
            (_to_int(row.get("device_id"), -1), _to_int(row.get("stream_id"), 0))
        )
        kind_counter = kinds_by_cat.setdefault(category, {})
        kind_counter[kind] = int(kind_counter.get(kind, 0)) + 1
        raw_total_ns += int(e - s)

    if not events:
        return {
            "wall_ms": 0.0,
            "raw_total_ms": 0.0,
            "non_overlap_ms": 0.0,
            "busy_union_ms": 0.0,
            "idle_ms": 0.0,
            "busy_pct_of_wall": 0.0,
            "cross_category_overlap_ms": 0.0,
            "overlap_saved_ms": 0.0,
            "rows": [],
        }

    ws = _to_int(wall_start_ns, -1)
    we = _to_int(wall_end_ns, -1)
    if ws < 0 or we <= ws:
        ws = int(min_kernel_start_ns if min_kernel_start_ns is not None else 0)
        we = int(max_kernel_end_ns if max_kernel_end_ns is not None else (ws + 1))
    wall_ns = max(1.0, float(we - ws))

    events.sort(key=lambda x: (int(x[0]), 0 if int(x[1]) < 0 else 1))
    active_by_cat: Dict[str, int] = {}
    weighted_ns_by_cat: Dict[str, float] = {}
    exclusive_ns_by_cat: Dict[str, float] = {}
    overlap_ns_by_cat: Dict[str, float] = {}
    non_overlap_ns = 0.0
    cross_category_overlap_ns = 0.0
    prev_t: Optional[int] = None
    for t, delta_kind, category in events:
        tt = int(t)
        if prev_t is not None and tt > prev_t and active_by_cat:
            span = float(tt - prev_t)
            non_overlap_ns += span
            active_cats = [cat for cat, count in active_by_cat.items() if int(count) > 0]
            if active_cats:
                if len(active_cats) == 1:
                    only_cat = str(active_cats[0])
                    exclusive_ns_by_cat[only_cat] = float(exclusive_ns_by_cat.get(only_cat, 0.0)) + span
                else:
                    cross_category_overlap_ns += span
                    for cat in active_cats:
                        overlap_ns_by_cat[cat] = float(overlap_ns_by_cat.get(cat, 0.0)) + span
                share = span / float(len(active_cats))
                for cat in active_cats:
                    weighted_ns_by_cat[cat] = float(weighted_ns_by_cat.get(cat, 0.0)) + share
        prev_t = tt
        if int(delta_kind) > 0:
            active_by_cat[category] = int(active_by_cat.get(category, 0)) + 1
        else:
            new_v = int(active_by_cat.get(category, 0)) - 1
            if new_v <= 0:
                active_by_cat.pop(category, None)
            else:
                active_by_cat[category] = new_v

    rows: List[Dict[str, object]] = []
    for category in intervals_by_cat.keys():
        merged = _merge_intervals(intervals_by_cat.get(category, []))
        union_ns = sum(int(e - s) for s, e in merged)
        weighted_ns = float(weighted_ns_by_cat.get(category, 0.0))
        exclusive_ns = float(exclusive_ns_by_cat.get(category, 0.0))
        overlap_ns = float(overlap_ns_by_cat.get(category, 0.0))
        non_overlap = max(1e-9, float(non_overlap_ns))
        union_ns = float(union_ns)
        if overlap_ns > union_ns:
            overlap_ns = union_ns
        if exclusive_ns > union_ns:
            exclusive_ns = union_ns
        if abs((exclusive_ns + overlap_ns) - union_ns) > 1e-6:
            if union_ns > 0:
                overlap_ns = max(0.0, union_ns - exclusive_ns)
            else:
                exclusive_ns = 0.0
                overlap_ns = 0.0
        kind_counter = kinds_by_cat.get(category, {})
        raw_ns = float(raw_ns_by_cat.get(category, 0))
        rows.append(
            {
                "category": category,
                "instances": int(instances_by_cat.get(category, 0)),
                "stream_count": len(streams_by_cat.get(category, set())),
                "raw_total_ms": raw_ns / 1e6,
                "raw_pct_of_wall": (raw_ns / wall_ns) * 100.0,
                "union_elapsed_ms": union_ns / 1e6,
                "weighted_elapsed_ms": weighted_ns / 1e6,
                "exclusive_elapsed_ms": exclusive_ns / 1e6,
                "overlap_elapsed_ms": overlap_ns / 1e6,
                "shared_elapsed_ms": overlap_ns / 1e6,
                "weighted_pct_of_wall": (weighted_ns / wall_ns) * 100.0,
                "weighted_pct_of_nonoverlap": (weighted_ns / non_overlap) * 100.0,
                "weighted_pct_of_busy": (weighted_ns / non_overlap) * 100.0,
                "union_pct_of_nonoverlap": (union_ns / non_overlap) * 100.0,
                "union_pct_of_wall": (union_ns / wall_ns) * 100.0,
                "exclusive_pct_of_wall": (exclusive_ns / wall_ns) * 100.0,
                "overlap_pct_of_wall": (overlap_ns / wall_ns) * 100.0,
                "shared_pct_of_wall": (overlap_ns / wall_ns) * 100.0,
                "compute_instances": int(kind_counter.get("compute", 0)),
                "comm_instances": int(kind_counter.get("comm", 0)),
            }
        )
    rows.sort(
        key=lambda x: (
            -_safe_float(x.get("weighted_elapsed_ms")),
            -_safe_float(x.get("union_elapsed_ms")),
            str(x.get("category") or ""),
        )
    )
    raw_total_ms = float(raw_total_ns) / 1e6
    non_overlap_ms = float(non_overlap_ns) / 1e6
    wall_ms = float(wall_ns) / 1e6
    idle_ms = max(0.0, wall_ms - non_overlap_ms)
    return {
        "wall_ms": wall_ms,
        "raw_total_ms": raw_total_ms,
        "non_overlap_ms": non_overlap_ms,
        "busy_union_ms": non_overlap_ms,
        "busy_pct_of_wall": (float(non_overlap_ns) / wall_ns) * 100.0,
        "idle_ms": idle_ms,
        "idle_pct_of_wall": (idle_ms / wall_ms) * 100.0 if wall_ms > 0 else 0.0,
        "cross_category_overlap_ms": float(cross_category_overlap_ns) / 1e6,
        "cross_category_overlap_pct_of_wall": (float(cross_category_overlap_ns) / wall_ns) * 100.0,
        "overlap_saved_ms": max(0.0, raw_total_ms - non_overlap_ms),
        "rows": rows,
    }


def _build_kernel_category_kernel_table(
    kernels: Sequence[Dict[str, object]],
    *,
    rules: Sequence[Tuple[Pattern[str], str]],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    cache: Dict[str, str] = {}
    for row in kernels:
        s = _to_int(row.get("start_ns"), -1)
        e = _to_int(row.get("end_ns"), -1)
        if s < 0 or e <= s:
            continue
        kernel_name = str(row.get("kernel_name") or "")
        category = _classify_kernel_name(kernel_name, rules=rules, cache=cache)
        dur_ms = _safe_float(row.get("duration_ms"))
        if dur_ms <= 0:
            dur_ms = float(e - s) / 1e6
        key = (category, kernel_name)
        entry = grouped.get(key)
        if entry is None:
            entry = {
                "category": category,
                "kernel_name": kernel_name,
                "instances": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "stream_keys": set(),
                "compute_instances": 0,
                "comm_instances": 0,
            }
            grouped[key] = entry
        entry["instances"] = int(entry.get("instances") or 0) + 1
        entry["total_ms"] = _safe_float(entry.get("total_ms")) + float(dur_ms)
        entry["max_ms"] = max(_safe_float(entry.get("max_ms")), float(dur_ms))
        entry["stream_keys"].add((_to_int(row.get("device_id"), -1), _to_int(row.get("stream_id"), 0)))
        if str(row.get("kind") or "compute").lower() == "comm":
            entry["comm_instances"] = int(entry.get("comm_instances") or 0) + 1
        else:
            entry["compute_instances"] = int(entry.get("compute_instances") or 0) + 1

    rows: List[Dict[str, object]] = []
    for entry in grouped.values():
        instances = int(entry.get("instances") or 0)
        total_ms = _safe_float(entry.get("total_ms"))
        rows.append(
            {
                "category": str(entry.get("category") or "misc"),
                "kernel_name": str(entry.get("kernel_name") or ""),
                "instances": instances,
                "total_ms": total_ms,
                "avg_ms": (total_ms / float(max(1, instances))),
                "max_ms": _safe_float(entry.get("max_ms")),
                "stream_count": len(entry.get("stream_keys") or set()),
                "compute_instances": int(entry.get("compute_instances") or 0),
                "comm_instances": int(entry.get("comm_instances") or 0),
            }
        )
    rows.sort(
        key=lambda x: (
            str(x.get("category") or ""),
            -_safe_float(x.get("total_ms")),
            -int(x.get("instances") or 0),
            str(x.get("kernel_name") or ""),
        )
    )
    return rows


def _write_rows_table(path: str, rows: Sequence[Dict[str, object]]) -> str:
    out = Path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(r) for r in rows]
    if out.suffix.lower() == ".csv":
        preferred = [
            "sqlite_index",
            "sqlite_label",
            "sqlite_path",
            "category",
            "kernel_name",
            "instances",
            "total_ms",
            "avg_ms",
            "max_ms",
            "stream_count",
            "compute_instances",
            "comm_instances",
        ]
        keys = set()
        for row in rows_list:
            keys.update(str(k) for k in row.keys())
        fieldnames: List[str] = []
        for k in preferred:
            if k in keys:
                fieldnames.append(k)
                keys.discard(k)
        fieldnames.extend(sorted(keys))
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_list:
                writer.writerow({k: row.get(k) for k in fieldnames})
    else:
        out.write_text(json.dumps(rows_list, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out)


def _build_nvtx_window_category_stats(
    *,
    nvtx_windows: Sequence[Dict[str, object]],
    kernels: Sequence[Dict[str, object]],
    rules: Sequence[Tuple[Pattern[str], str]],
) -> Dict[str, object]:
    windows = []
    for idx, scope in enumerate(nvtx_windows):
        s = _to_int(scope.get("start_ns"), -1)
        e = _to_int(scope.get("end_ns"), -1)
        if s < 0 or e <= s:
            continue
        windows.append(
            {
                "index": int(idx),
                "nvtx_text": str(scope.get("nvtx_text") or ""),
                "start_ns": int(s),
                "end_ns": int(e),
                "duration_ms": float(e - s) / 1e6,
            }
        )
    if not windows:
        return {
            "window_count": 0,
            "windows": [],
            "category_summary_rows": [],
            "outlier_windows": [],
            "window_duration_stats": {},
        }

    pair_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for i, w in enumerate(windows):
        pair_to_indices.setdefault((int(w["start_ns"]), int(w["end_ns"])), []).append(i)

    per_window_kernels: List[List[Dict[str, object]]] = [[] for _ in windows]
    for row in kernels:
        ks = _to_int(row.get("start_ns"), -1)
        ke = _to_int(row.get("end_ns"), -1)
        if ks < 0 or ke <= ks:
            continue

        assigned: set[int] = set()
        ns = _to_int(row.get("nvtx_start_ns"), -1)
        ne = _to_int(row.get("nvtx_end_ns"), -1)
        if ns >= 0 and ne > ns:
            for wi in pair_to_indices.get((ns, ne), []):
                assigned.add(int(wi))

        if not assigned:
            for wi, w in enumerate(windows):
                if _intervals_overlap(ks, ke, int(w["start_ns"]), int(w["end_ns"])):
                    assigned.add(int(wi))

        for wi in assigned:
            per_window_kernels[wi].append(dict(row))

    per_window_rows: List[Dict[str, object]] = []
    category_union: set[str] = set()
    for wi, w in enumerate(windows):
        wk = per_window_kernels[wi]
        if wk:
            gpu_start_ns = min(_to_int(item.get("start_ns"), 0) for item in wk)
            gpu_end_ns = max(_to_int(item.get("end_ns"), 0) for item in wk)
            breakdown = _build_kernel_category_breakdown(
                wk,
                rules=list(rules or []),
                wall_start_ns=int(gpu_start_ns),
                wall_end_ns=int(gpu_end_ns),
            )
            cat_rows = list(breakdown.get("rows") or [])
            cat_pct_map = {
                str(item.get("category") or "misc"): _safe_float(item.get("weighted_pct_of_nonoverlap"))
                for item in cat_rows
            }
            cat_weighted_ms_map = {
                str(item.get("category") or "misc"): _safe_float(item.get("weighted_elapsed_ms"))
                for item in cat_rows
            }
            cat_raw_total_ms_map = {
                str(item.get("category") or "misc"): _safe_float(item.get("raw_total_ms"))
                for item in cat_rows
            }
            for cat_name in cat_pct_map.keys():
                category_union.add(str(cat_name))
            top_cat = ""
            top_cat_pct = 0.0
            if cat_pct_map:
                top_cat, top_cat_pct = max(cat_pct_map.items(), key=lambda kv: float(kv[1]))
        else:
            gpu_start_ns = -1
            gpu_end_ns = -1
            breakdown = {
                "rows": [],
                "wall_ms": 0.0,
                "non_overlap_ms": 0.0,
                "raw_total_ms": 0.0,
                "busy_union_ms": 0.0,
                "idle_ms": 0.0,
                "overlap_saved_ms": 0.0,
            }
            cat_pct_map = {}
            cat_weighted_ms_map = {}
            cat_raw_total_ms_map = {}
            top_cat = ""
            top_cat_pct = 0.0
        per_window_rows.append(
            {
                "window_index": int(w["index"]),
                "nvtx_text": str(w["nvtx_text"]),
                "cpu_start_ns": int(w["start_ns"]),
                "cpu_end_ns": int(w["end_ns"]),
                "cpu_duration_ms": _safe_float(w.get("duration_ms")),
                "gpu_start_ns": int(gpu_start_ns) if gpu_start_ns >= 0 else None,
                "gpu_end_ns": int(gpu_end_ns) if gpu_end_ns > gpu_start_ns >= 0 else None,
                "gpu_duration_ms": (float(gpu_end_ns - gpu_start_ns) / 1e6) if gpu_end_ns > gpu_start_ns >= 0 else 0.0,
                "kernel_count": len(wk),
                "non_overlap_ms": _safe_float(breakdown.get("non_overlap_ms")),
                "gpu_idle_ms": _safe_float(breakdown.get("idle_ms")),
                "raw_total_ms": _safe_float(breakdown.get("raw_total_ms")),
                "top_category": str(top_cat or ""),
                "top_category_pct": float(top_cat_pct),
                "category_weighted_pct": cat_pct_map,
                "category_weighted_ms": cat_weighted_ms_map,
                "category_raw_total_ms": cat_raw_total_ms_map,
            }
        )

    category_summary_rows: List[Dict[str, object]] = []
    for cat in sorted(category_union):
        pct_values = [
            _safe_float(row.get("category_weighted_pct", {}).get(cat))
            for row in per_window_rows
        ]
        weighted_ms_values = [
            _safe_float(row.get("category_weighted_ms", {}).get(cat))
            for row in per_window_rows
        ]
        raw_total_ms_values = [
            _safe_float(row.get("category_raw_total_ms", {}).get(cat))
            for row in per_window_rows
        ]
        if not pct_values:
            continue
        pct_stats = _series_stats_with_iqr(pct_values, k=1.5)
        weighted_ms_stats = _series_stats_with_iqr(weighted_ms_values, k=1.5)
        raw_total_ms_stats = _series_stats_with_iqr(raw_total_ms_values, k=1.5)
        category_summary_rows.append(
            {
                "category": str(cat),
                "avg_pct": _safe_float(pct_stats.get("avg")),
                "std_pct": _safe_float(pct_stats.get("std")),
                "min_pct": _safe_float(pct_stats.get("min")),
                "max_pct": _safe_float(pct_stats.get("max")),
                "avg_pct_raw": _safe_float(pct_stats.get("avg_raw")),
                "std_pct_raw": _safe_float(pct_stats.get("std_raw")),
                "min_pct_raw": _safe_float(pct_stats.get("min_raw")),
                "max_pct_raw": _safe_float(pct_stats.get("max_raw")),
                "avg_pct_excl_outliers": _safe_float(pct_stats.get("avg")),
                "std_pct_excl_outliers": _safe_float(pct_stats.get("std")),
                "avg_weighted_ms": _safe_float(weighted_ms_stats.get("avg")),
                "std_weighted_ms": _safe_float(weighted_ms_stats.get("std")),
                "min_weighted_ms": _safe_float(weighted_ms_stats.get("min")),
                "max_weighted_ms": _safe_float(weighted_ms_stats.get("max")),
                "avg_weighted_ms_raw": _safe_float(weighted_ms_stats.get("avg_raw")),
                "std_weighted_ms_raw": _safe_float(weighted_ms_stats.get("std_raw")),
                "avg_raw_total_ms": _safe_float(raw_total_ms_stats.get("avg")),
                "std_raw_total_ms": _safe_float(raw_total_ms_stats.get("std")),
                "min_raw_total_ms": _safe_float(raw_total_ms_stats.get("min")),
                "max_raw_total_ms": _safe_float(raw_total_ms_stats.get("max")),
                "avg_raw_total_ms_raw": _safe_float(raw_total_ms_stats.get("avg_raw")),
                "std_raw_total_ms_raw": _safe_float(raw_total_ms_stats.get("std_raw")),
                "kept_windows": int(_safe_float(pct_stats.get("kept_count"))),
                "removed_windows": int(_safe_float(pct_stats.get("removed_count"))),
                "clip_low_pct": _safe_float(pct_stats.get("clip_low")),
                "clip_high_pct": _safe_float(pct_stats.get("clip_high")),
                "clip_method": "iqr_1.5x",
                "removed_windows_weighted_ms": int(_safe_float(weighted_ms_stats.get("removed_count"))),
                "removed_windows_raw_total_ms": int(_safe_float(raw_total_ms_stats.get("removed_count"))),
                "nonzero_windows": int(sum(1 for v in pct_values if v > 1e-9)),
                "window_count": len(pct_values),
            }
        )
    category_summary_rows.sort(
        key=lambda x: (
            -_safe_float(x.get("avg_pct")),
            str(x.get("category") or ""),
        )
    )
    cat_stats = {str(item["category"]): item for item in category_summary_rows}

    outlier_rows: List[Dict[str, object]] = []
    warmup_head = max(2, min(10, int(math.ceil(len(per_window_rows) * 0.1))))
    for row in per_window_rows:
        cat_map = dict(row.get("category_weighted_pct") or {})
        top_cat = ""
        top_z = 0.0
        for cat in category_union:
            stat = cat_stats.get(str(cat))
            if not stat:
                continue
            std_v = _safe_float(stat.get("std_pct"))
            if std_v <= 1e-12:
                continue
            v = _safe_float(cat_map.get(cat))
            z = (v - _safe_float(stat.get("avg_pct"))) / std_v
            if abs(z) > abs(top_z):
                top_z = z
                top_cat = str(cat)
        row["max_abs_z"] = abs(float(top_z))
        row["outlier_category"] = top_cat
        row["outlier_z"] = float(top_z)
        if abs(float(top_z)) >= 2.0:
            outlier_rows.append(
                {
                    "window_index": int(row.get("window_index") or 0),
                    "nvtx_text": str(row.get("nvtx_text") or ""),
                    "kernel_count": int(row.get("kernel_count") or 0),
                    "gpu_duration_ms": _safe_float(row.get("gpu_duration_ms")),
                    "outlier_category": str(top_cat),
                    "outlier_z": float(top_z),
                    "max_abs_z": abs(float(top_z)),
                    "warmup_head_window": bool(int(row.get("window_index") or 0) < int(warmup_head)),
                }
            )
    outlier_rows.sort(
        key=lambda x: (
            -_safe_float(x.get("max_abs_z")),
            int(x.get("window_index") or 0),
        )
    )

    cpu_durations_ms = [_safe_float(row.get("cpu_duration_ms")) for row in per_window_rows]
    gpu_durations_ms = [_safe_float(row.get("gpu_duration_ms")) for row in per_window_rows]
    non_overlap_ms = [_safe_float(row.get("non_overlap_ms")) for row in per_window_rows]
    gpu_idle_ms = [_safe_float(row.get("gpu_idle_ms")) for row in per_window_rows]
    raw_total_ms = [_safe_float(row.get("raw_total_ms")) for row in per_window_rows]
    cpu_stats = _series_stats_with_iqr(cpu_durations_ms, k=1.5)
    gpu_stats = _series_stats_with_iqr(gpu_durations_ms, k=1.5)
    non_overlap_stats = _series_stats_with_iqr(non_overlap_ms, k=1.5)
    gpu_idle_stats = _series_stats_with_iqr(gpu_idle_ms, k=1.5)
    raw_total_stats = _series_stats_with_iqr(raw_total_ms, k=1.5)

    return {
        "window_count": len(per_window_rows),
        "category_count": len(category_union),
        "warmup_head_window_count": int(warmup_head),
        "avg_filter": {
            "method": "iqr_1.5x",
            "description": "category avg/std exclude outliers outside [Q1-1.5*IQR, Q3+1.5*IQR]",
            "min_points_for_clip": 4,
        },
        "window_duration_stats": {
            "cpu_duration_ms": {
                "avg_clipped": _safe_float(cpu_stats.get("avg")),
                "avg_raw": _safe_float(cpu_stats.get("avg_raw")),
                "std_clipped": _safe_float(cpu_stats.get("std")),
                "std_raw": _safe_float(cpu_stats.get("std_raw")),
                "removed_windows": int(_safe_float(cpu_stats.get("removed_count"))),
                "window_count": int(_safe_float(cpu_stats.get("count"))),
            },
            "gpu_duration_ms": {
                "avg_clipped": _safe_float(gpu_stats.get("avg")),
                "avg_raw": _safe_float(gpu_stats.get("avg_raw")),
                "std_clipped": _safe_float(gpu_stats.get("std")),
                "std_raw": _safe_float(gpu_stats.get("std_raw")),
                "removed_windows": int(_safe_float(gpu_stats.get("removed_count"))),
                "window_count": int(_safe_float(gpu_stats.get("count"))),
            },
            "non_overlap_ms": {
                "avg_clipped": _safe_float(non_overlap_stats.get("avg")),
                "avg_raw": _safe_float(non_overlap_stats.get("avg_raw")),
                "std_clipped": _safe_float(non_overlap_stats.get("std")),
                "std_raw": _safe_float(non_overlap_stats.get("std_raw")),
                "removed_windows": int(_safe_float(non_overlap_stats.get("removed_count"))),
                "window_count": int(_safe_float(non_overlap_stats.get("count"))),
            },
            "gpu_idle_ms": {
                "avg_clipped": _safe_float(gpu_idle_stats.get("avg")),
                "avg_raw": _safe_float(gpu_idle_stats.get("avg_raw")),
                "std_clipped": _safe_float(gpu_idle_stats.get("std")),
                "std_raw": _safe_float(gpu_idle_stats.get("std_raw")),
                "removed_windows": int(_safe_float(gpu_idle_stats.get("removed_count"))),
                "window_count": int(_safe_float(gpu_idle_stats.get("count"))),
            },
            "raw_total_ms": {
                "avg_clipped": _safe_float(raw_total_stats.get("avg")),
                "avg_raw": _safe_float(raw_total_stats.get("avg_raw")),
                "std_clipped": _safe_float(raw_total_stats.get("std")),
                "std_raw": _safe_float(raw_total_stats.get("std_raw")),
                "removed_windows": int(_safe_float(raw_total_stats.get("removed_count"))),
                "window_count": int(_safe_float(raw_total_stats.get("count"))),
            },
        },
        "windows": per_window_rows,
        "category_summary_rows": category_summary_rows,
        "outlier_windows": outlier_rows,
    }


def _select_nvtx_windows(
    provider: NsysSqliteMetricsProvider,
    *,
    nvtx_text: str,
) -> List[Dict[str, object]]:
    rows = provider.run_sql_skill(
        "nvtx_ranges_hierarchy",
        nvtx_text=str(nvtx_text or "%"),
        top_level_only=0,
        limit=500000,
    )
    if not rows:
        return []
    items = []
    for row in rows:
        s = _to_int(row.get("start_ns"), -1)
        e = _to_int(row.get("end_ns"), -1)
        if s < 0 or e <= s:
            continue
        items.append(
            {
                "nvtx_text": str(row.get("nvtx_text") or ""),
                "start_ns": s,
                "end_ns": e,
                "duration_ms": round((e - s) / 1e6, 3),
            }
        )
    if not items:
        return []
    items.sort(key=lambda x: (int(x["start_ns"]), int(x["end_ns"])))
    return items


def _pick_nvtx_windows(
    windows: Sequence[Dict[str, object]],
    *,
    nvtx_index: int,
) -> List[Dict[str, object]]:
    if not windows:
        return []
    # -1 means all matched scopes (default behavior).
    if int(nvtx_index) < 0:
        return list(windows)
    idx = max(0, min(int(nvtx_index), len(windows) - 1))
    return [dict(windows[idx])]


def _collect_kernels_in_window(
    provider: NsysSqliteMetricsProvider,
    *,
    start_ns: int,
    end_ns: int,
    nvtx_text: str,
    nvtx_windows: Optional[Sequence[Dict[str, object]]] = None,
    device_id: int,
    limit: int,
    debug_log: Optional[Callable[[str], None]] = None,
    debug_rows: int = 3,
) -> List[Dict[str, object]]:
    debug = debug_log or _noop_debug
    try:
        debug_rows_i = int(debug_rows)
    except Exception:
        debug_rows_i = 3
    rows: List[Dict[str, object]] = []
    attributed_kernel_keys = set()
    selected_windows: List[Tuple[int, int]] = []
    selected_pairs = set()
    if nvtx_windows:
        for item in nvtx_windows:
            ws = _to_int(item.get("start_ns"), -1)
            we = _to_int(item.get("end_ns"), -1)
            if we <= ws:
                continue
            selected_pairs.add((ws, we))
            selected_windows.append((ws, we))
    debug(
        "kernel query window start_ns={} end_ns={} device_id={} selected_nvtx_windows={}".format(
            int(start_ns),
            int(end_ns),
            int(device_id),
            len(selected_windows),
        )
    )

    # Prefer launch-attribution view (NVTX -> Runtime -> correlationId -> Kernel).
    skills = set(provider.list_sql_skills())
    debug("available skills={} has_nvtx_kernel_sm_detail={}".format(len(skills), "nvtx_kernel_sm_detail" in skills))
    detailed_kept = 0
    if "nvtx_kernel_sm_detail" in skills:
        detailed = provider.run_sql_skill(
            "nvtx_kernel_sm_detail",
            nvtx_text=str(nvtx_text or "%"),
            device_id=int(device_id),
            limit=int(limit),
        )
        debug("skill nvtx_kernel_sm_detail returned {} rows before filtering".format(len(detailed)))
        for row in detailed:
            ns = _to_int(row.get("nvtx_start_ns"), -1)
            ne = _to_int(row.get("nvtx_end_ns"), -1)
            if selected_pairs:
                if (ns, ne) not in selected_pairs:
                    continue
            # Robust fallback: older payload variants may expose generic start/end names.
            ks = _to_int(row.get("kernel_start_ns", row.get("start_ns")), -1)
            ke = _to_int(row.get("kernel_end_ns", row.get("end_ns")), -1)
            if ks < 0 or ke <= ks:
                continue
            if not selected_pairs and int(start_ns) >= 0 and int(end_ns) > int(start_ns):
                # Timeline window filtering must follow GPU execution timestamps.
                # Using NVTX CPU-side timestamps here can drop valid kernels due to
                # async CPU launch vs GPU execution lag.
                if not _intervals_overlap(ks, ke, int(start_ns), int(end_ns)):
                    continue
            uniq = (ks, ke, _to_int(row.get("stream_id"), 0), str(row.get("kernel_name") or ""))
            attributed_kernel_keys.add(uniq)
            detailed_kept += 1
            threads_per_block = row.get("threads_per_block")
            registers_per_thread = row.get("registersPerThread")
            total_shared_bytes = row.get("total_shared_bytes")
            rows.append(
                {
                    "stream_id": _to_int(row.get("stream_id"), 0),
                    "device_id": _to_int(row.get("device_id"), int(device_id)),
                    "kernel_name": str(row.get("kernel_name") or ""),
                    "start_ns": ks,
                    "end_ns": ke,
                    "duration_ms": float(row.get("duration_ms") or round((ke - ks) / 1e6, 6)),
                    "kind": str(row.get("kind") or "compute"),
                    "registers_per_thread": row.get("registersPerThread"),
                    "threads_per_block": row.get("threads_per_block"),
                    "static_shared_bytes": row.get("static_shared_bytes"),
                    "dynamic_shared_bytes": row.get("dynamic_shared_bytes"),
                    "total_shared_bytes": row.get("total_shared_bytes"),
                    "occupancy_pct_estimate": _resolve_occupancy_pct_estimate(
                        row.get("occupancy_pct_estimate"),
                        threads_per_block=threads_per_block,
                        registers_per_thread=registers_per_thread,
                        total_shared_bytes=total_shared_bytes,
                    ),
                    "nvtx_text": str(row.get("nvtx_text") or ""),
                    "nvtx_start_ns": ns,
                    "nvtx_end_ns": ne,
                    "rank": _parse_rank_from_text(row.get("nvtx_text")),
                }
            )
        debug("skill nvtx_kernel_sm_detail kept {} rows after filtering".format(detailed_kept))
    else:
        debug("skill nvtx_kernel_sm_detail unavailable, fallback to kernel_map only")

    # Fallback/补齐: plain kernel rows in time window.
    # This keeps timeline complete when launch-attribution misses some kernels.
    fallback = collect_kernel_rows(
        provider,
        device_id=int(device_id),
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        limit=int(limit),
        attach_iteration=False,
    )
    fallback_kept = 0
    debug("skill kernel_map returned {} rows before fallback filtering".format(len(fallback)))
    for row in fallback:
        ks = _to_int(row.get("start_ns"), -1)
        ke = _to_int(row.get("end_ns"), -1)
        if ks < 0 or ke <= ks:
            continue
        if selected_windows:
            if not any(_intervals_overlap(ks, ke, ws, we) for ws, we in selected_windows):
                continue
        elif int(start_ns) >= 0 and int(end_ns) > int(start_ns):
            if not _intervals_overlap(ks, ke, int(start_ns), int(end_ns)):
                continue
        uniq = (ks, ke, _to_int(row.get("stream_id"), 0), str(row.get("kernel_name") or ""))
        if uniq in attributed_kernel_keys:
            continue
        fallback_kept += 1
        threads_per_block = row.get("threads_per_block")
        registers_per_thread = row.get("registersPerThread")
        total_shared_bytes = row.get("total_shared_bytes")
        rows.append(
            {
                "stream_id": _to_int(row.get("stream_id"), 0),
                "device_id": _to_int(row.get("device_id"), int(device_id)),
                "kernel_name": str(row.get("kernel_name") or ""),
                "start_ns": ks,
                "end_ns": ke,
                "duration_ms": float(row.get("duration_ms") or 0.0),
                "kind": "comm" if bool(row.get("is_nccl")) else "compute",
                "registers_per_thread": None,
                "threads_per_block": None,
                "static_shared_bytes": None,
                "dynamic_shared_bytes": None,
                "total_shared_bytes": None,
                "occupancy_pct_estimate": _resolve_occupancy_pct_estimate(
                    row.get("occupancy_pct_estimate"),
                    threads_per_block=threads_per_block,
                    registers_per_thread=registers_per_thread,
                    total_shared_bytes=total_shared_bytes,
                ),
                "nvtx_text": None,
                "nvtx_start_ns": None,
                "nvtx_end_ns": None,
                "rank": None,
            }
        )
    rows.sort(key=lambda x: (_to_int(x.get("start_ns"), 0), _to_int(x.get("end_ns"), 0), _to_int(x.get("stream_id"), 0)))
    debug("kernel rows final={} (from detail={} + fallback_added={})".format(len(rows), detailed_kept, fallback_kept))
    stream_count = len(
        {
            (_to_int(item.get("device_id"), -1), _to_int(item.get("stream_id"), 0))
            for item in rows
        }
    )
    unique_gpu_kernel_count = len(
        {
            (
                _to_int(item.get("start_ns"), -1),
                _to_int(item.get("end_ns"), -1),
                _to_int(item.get("device_id"), -1),
                _to_int(item.get("stream_id"), 0),
                str(item.get("kernel_name") or ""),
            )
            for item in rows
        }
    )
    debug(
        "matched kernels total={} unique_gpu_kernels={} stream_count={} selected_nvtx_windows={}".format(
            len(rows),
            unique_gpu_kernel_count,
            stream_count,
            len(selected_windows),
        )
    )
    if rows:
        debug(
            "kernel sample={}".format(
                _preview_dict_rows(
                    rows,
                    keys=("kernel_name", "stream_id", "device_id", "start_ns", "end_ns", "kind"),
                    limit=debug_rows_i,
                )
            )
        )
    return rows


def _downsample_points(points: Sequence[Tuple[int, float]], max_points: int = 2000) -> List[Tuple[int, float]]:
    if int(max_points) <= 0:
        return [(int(t), float(v)) for t, v in points]
    if len(points) <= max_points:
        return [(int(t), float(v)) for t, v in points]
    step = max(1, int(math.ceil(len(points) / float(max_points))))
    sampled = [(int(points[i][0]), float(points[i][1])) for i in range(0, len(points), step)]
    last_t, last_v = points[-1]
    if sampled[-1][0] != int(last_t):
        sampled.append((int(last_t), float(last_v)))
    return sampled


def _metric_device_tag(device_id: object, source_name: object) -> Tuple[str, str]:
    dev = _to_int(device_id, -1)
    if dev >= 0:
        return "gpu", str(dev)
    src = str(source_name or "").strip()
    if src:
        m = _GPU_IDX_RE.search(src)
        if m:
            return "gpu", str(m.group(1))
        return "source", src
    return "unknown", "unknown"


def _metric_series_name(metric_name: str, tag_kind: str, tag_value: str) -> str:
    if tag_kind == "gpu":
        return f"{metric_name} [gpu {tag_value}]"
    if tag_kind == "source":
        return f"{metric_name} [src {tag_value}]"
    return f"{metric_name} [unknown]"


def _normalize_metric_text(text: object) -> str:
    norm = re.sub(r"\s+", " ", str(text or "").replace("_", " ").strip().lower())
    # Some exports split "SMs" into "SM s". Normalize both forms.
    return norm.replace("sm s", "sms")


def _is_percent_throughput_metric_name(metric_name: object) -> bool:
    raw = str(metric_name or "").lower()
    norm = _normalize_metric_text(metric_name)
    if "%" in raw:
        return True
    if "throughput" in norm:
        return True
    if "pct" in norm:
        return True
    if "percent" in norm:
        return True
    return False


def _is_default_focus_metric_name(metric_name: object) -> bool:
    norm = _normalize_metric_text(metric_name)
    if not norm:
        return False
    if "compute warps in flight" in norm or "unallocated warps in active sms" in norm:
        # For the two warps families, keep all mainstream variants:
        # Avg / Throughput / Avg Warps Per Cycle.
        if "avg warps per cycle" in norm:
            return True
        if "throughput" in norm or "pct" in norm or "%" in str(metric_name or ""):
            return True
        if "avg" in norm:
            return True
        return False
    if "tensor active" in norm or "tensor__active" in str(metric_name or "").lower():
        return True
    for token in _DEFAULT_FOCUS_METRIC_TOKENS:
        if token in norm:
            return True
    return False


def _collect_metric_samples(
    sqlite_path: str,
    *,
    start_ns: int,
    end_ns: int,
    metric_name_like: str = "%",
    include_all_sources: bool = False,
    device_id: int = -1,
    limit: int = -1,
    max_points_per_series: int = -1,
    apply_default_focus_filter: bool = False,
    restrict_to_intervals: Optional[Sequence[Tuple[int, int]]] = None,
    debug_log: Optional[Callable[[str], None]] = None,
    debug_rows: int = 3,
) -> List[Dict[str, object]]:
    debug = debug_log or _noop_debug
    try:
        debug_rows_i = int(debug_rows)
    except Exception:
        debug_rows_i = 3
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        schema = NsightSchema(conn)
        if not schema.metrics_table:
            debug("metrics table not detected in schema")
            return []
        metrics_table = _ident(schema.metrics_table)
        ts_col = schema.resolve_column(metrics_table, ("timestamp",))
        id_col = schema.metrics_id_col or schema.resolve_column(metrics_table, ("metricId", "nameId", "eventId"))
        val_col = schema.metrics_value_col or schema.resolve_column(metrics_table, ("value", "metricValue", "val"))
        if not ts_col:
            debug(
                "metrics timestamp unresolved table={} required_col=timestamp (rawTimestamp fallback disabled)".format(
                    metrics_table
                )
            )
            return []
        if not id_col or not val_col:
            debug(
                "metrics columns unresolved table={} ts_col={} id_col={} val_col={}".format(
                    metrics_table, ts_col, id_col, val_col
                )
            )
            return []
        debug(
            "metrics source table={} ts_col={} id_col={} val_col={} device_id={} include_all_sources={} metric_like={}".format(
                metrics_table,
                ts_col,
                id_col,
                val_col,
                int(device_id),
                int(bool(include_all_sources)),
                str(metric_name_like or "%"),
            )
        )

        name_expr = ""
        name_not_null = ""
        joins: List[str] = []

        if schema.table_exists("TARGET_INFO_GPU_METRICS"):
            gm_tbl = _ident("TARGET_INFO_GPU_METRICS")
            gm_id_col = schema.resolve_column(gm_tbl, ("metricId", "id", "metric_id"))
            gm_name_col = schema.resolve_column(gm_tbl, ("name", "metricName", "metric_name", "value"))
            g_type_col = schema.resolve_column(metrics_table, ("typeId", "type_id", "eventType", "event_type"))
            gm_type_col = schema.resolve_column(gm_tbl, ("typeId", "type_id", "eventType", "event_type"))
            if gm_id_col and gm_name_col:
                if g_type_col and gm_type_col:
                    joins.append(
                        f"JOIN {gm_tbl} gm "
                        f"ON g.{_ident(id_col)} = gm.{_ident(gm_id_col)} "
                        f"AND g.{_ident(g_type_col)} = gm.{_ident(gm_type_col)}"
                    )
                else:
                    joins.append(
                        f"JOIN {gm_tbl} gm ON g.{_ident(id_col)} = gm.{_ident(gm_id_col)}"
                    )
                name_expr = f"gm.{_ident(gm_name_col)}"
                name_not_null = f"AND gm.{_ident(gm_name_col)} IS NOT NULL "
        has_gpu_info_mapping = bool(name_expr.startswith("gm."))
        if has_gpu_info_mapping:
            debug("metrics name mapping=TARGET_INFO_GPU_METRICS")

        if not name_expr and schema.string_table:
            string_table = _ident(schema.string_table)
            joins.append(f"JOIN {string_table} s ON g.{_ident(id_col)} = s.id")
            name_expr = "s.value"
            name_not_null = "AND s.value IS NOT NULL "
            debug("metrics name mapping=StringIds")

        if not name_expr:
            name_expr = f"CAST(g.{_ident(id_col)} AS TEXT)"
            name_not_null = ""
            debug("metrics name mapping=CAST(metricId AS TEXT)")

        source_join = ""
        source_where = ""
        source_name_expr = "NULL"
        source_key_expr = ""
        source_col = schema.resolve_column(metrics_table, ("sourceId", "source_id"))
        if source_col:
            source_key_expr = f"g.{_ident(source_col)}"
        elif has_gpu_info_mapping:
            gm_source_col = schema.resolve_column("TARGET_INFO_GPU_METRICS", ("sourceId", "source_id"))
            if gm_source_col:
                source_key_expr = f"gm.{_ident(gm_source_col)}"

        if source_key_expr and schema.table_exists("GENERIC_EVENT_SOURCES"):
            ges_tbl = _ident("GENERIC_EVENT_SOURCES")
            ges_id_col = schema.resolve_column(ges_tbl, ("sourceId", "id", "source_id"))
            ges_name_col = schema.resolve_column(ges_tbl, ("name", "source", "sourceName"))
            ges_name_id_col = schema.resolve_column(ges_tbl, ("nameId", "name_id"))
            if ges_id_col:
                source_join = (
                    f"LEFT JOIN {ges_tbl} gs ON {source_key_expr} = gs.{_ident(ges_id_col)}"
                )
                if ges_name_col:
                    source_name_expr = f"gs.{_ident(ges_name_col)}"
                elif ges_name_id_col and schema.string_table:
                    source_join += (
                        f" LEFT JOIN {_ident(schema.string_table)} sgs "
                        f"ON gs.{_ident(ges_name_id_col)} = sgs.id"
                    )
                    source_name_expr = "sgs.value"
                if source_name_expr != "NULL":
                    source_where = (
                        "AND (? = 1 "
                        f"OR {source_name_expr} IS NULL "
                        f"OR LOWER({source_name_expr}) LIKE '%gpu%metric%' "
                        f"OR LOWER({source_name_expr}) = 'gpumetrics') "
                    )
        debug(
            "metrics source mapping key_expr={} source_name_expr={}".format(
                source_key_expr or "NULL",
                source_name_expr,
            )
        )

        device_where = ""
        device_expr = "NULL"
        filter_params: List[object] = [
            int(start_ns),
            int(end_ns),
            str(metric_name_like or "%"),
            str(metric_name_like or "%"),
        ]
        device_col = schema.resolve_column(metrics_table, ("deviceId", "gpuId", "device", "gpu"))
        if device_col:
            device_expr = f"CAST(g.{_ident(device_col)} AS INTEGER)"
            device_where = f"AND (? < 0 OR g.{_ident(device_col)} = ?) "
            filter_params.extend([int(device_id), int(device_id)])

        if source_where:
            filter_params.append(1 if bool(include_all_sources) else 0)

        from_join_expr = (
            f"FROM {metrics_table} g "
            + " ".join(joins)
            + " "
            + source_join
            + " "
        )

        limit_i = int(limit)
        tmp_tbl = "_myu_metrics_base"
        def _materialize_rows() -> Tuple[int, List[sqlite3.Row]]:
            ts_ident = _ident(str(ts_col))
            base_where = (
                "WHERE g.{ts} >= ? AND g.{ts} <= ? ".format(ts=ts_ident)
                + "AND (? = '%' OR {name_expr} LIKE ?) ".format(name_expr=name_expr)
                + name_not_null
                + device_where
                + source_where
                + f"AND g.{_ident(val_col)} IS NOT NULL "
            )
            select_expr = (
                f"SELECT g.{ts_ident} AS ts_ns, "
                f"{name_expr} AS metric_name, "
                f"{device_expr} AS metric_device_id, "
                f"{source_name_expr} AS metric_source_name, "
                f"CAST(g.{_ident(val_col)} AS REAL) AS metric_value "
            )
            conn.execute(f"DROP TABLE IF EXISTS {tmp_tbl}")
            create_tmp_sql = (
                f"CREATE TEMP TABLE {tmp_tbl} AS "
                + select_expr
                + from_join_expr
                + base_where
            )
            conn.execute(create_tmp_sql, filter_params)
            total = int(conn.execute(f"SELECT COUNT(*) FROM {tmp_tbl}").fetchone()[0] or 0)
            debug("metrics temp rows={} ts_col={} sampling_limit={}".format(total, ts_col, int(limit_i)))
            if total > 0:
                sample_sql = (
                    f"SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value "
                    f"FROM {tmp_tbl} ORDER BY ts_ns ASC"
                )
                if debug_rows_i > 0:
                    sample_sql += f" LIMIT {int(debug_rows_i)}"
                sample_rows = conn.execute(sample_sql).fetchall()
                debug("metrics temp sample ts_col={} data={}".format(ts_col, json.dumps([dict(x) for x in sample_rows], ensure_ascii=False)))

            if total <= 0:
                return total, []
            if limit_i <= 0 or total <= limit_i:
                data_rows = conn.execute(
                    f"SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value "
                    f"FROM {tmp_tbl} ORDER BY ts_ns ASC"
                ).fetchall()
                return total, data_rows

            # Distribute sampling budget by metric/device series, so each series keeps timeline coverage.
            series_rows = conn.execute(
                f"SELECT metric_name, metric_device_id, metric_source_name, COUNT(*) AS cnt "
                f"FROM {tmp_tbl} "
                f"GROUP BY metric_name, metric_device_id, metric_source_name "
                f"ORDER BY metric_name, metric_device_id, metric_source_name"
            ).fetchall()
            series_count = max(1, len(series_rows))
            target_per_series = max(2, int(math.floor(float(limit_i) / float(series_count))))
            collected: List[sqlite3.Row] = []
            for sr in series_rows:
                metric_name = sr["metric_name"]
                metric_device_id = sr["metric_device_id"]
                metric_source_name = sr["metric_source_name"]
                cnt = int(sr["cnt"] or 0)
                if cnt <= target_per_series:
                    stride = 1
                else:
                    stride = max(1, int(math.ceil(float(cnt) / float(target_per_series))))
                sampled_sql = (
                    "WITH numbered AS ("
                    f"SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value, "
                    "ROW_NUMBER() OVER (ORDER BY ts_ns ASC) AS rn, "
                    "COUNT(*) OVER () AS total_rows "
                    f"FROM {tmp_tbl} "
                    "WHERE metric_name = ? "
                    "AND metric_device_id IS ? "
                    "AND metric_source_name IS ?"
                    ") "
                    "SELECT ts_ns, metric_name, metric_device_id, metric_source_name, metric_value "
                    "FROM numbered "
                    "WHERE rn = 1 OR rn = total_rows OR ((rn - 1) % ?) = 0 "
                    "ORDER BY ts_ns ASC"
                )
                sampled_rows = conn.execute(
                    sampled_sql,
                    [metric_name, metric_device_id, metric_source_name, int(stride)],
                ).fetchall()
                collected.extend(sampled_rows)
            data_rows = sorted(collected, key=lambda r: (_to_int(r["ts_ns"], 0), str(r["metric_name"] or "")))
            return total, data_rows

        total_rows, rows = _materialize_rows()
        if restrict_to_intervals:
            merged_intervals = _merge_intervals(restrict_to_intervals)
            before = len(rows)
            if not merged_intervals:
                rows = []
            else:
                filtered_rows: List[sqlite3.Row] = []
                idx = 0
                for row in rows:
                    ts = _to_int(row["ts_ns"], -1)
                    if ts < 0:
                        continue
                    while idx < len(merged_intervals) and ts > merged_intervals[idx][1]:
                        idx += 1
                    if idx >= len(merged_intervals):
                        break
                    lo, hi = merged_intervals[idx]
                    if ts >= lo and ts <= hi:
                        filtered_rows.append(row)
                rows = filtered_rows
            debug(
                "metrics interval filter applied intervals={} rows_before={} rows_after={}".format(
                    len(merged_intervals) if restrict_to_intervals else 0,
                    int(before),
                    int(len(rows)),
                )
            )
        debug("metrics rows after sampling={} using_ts_col={}".format(len(rows), ts_col))
    finally:
        conn.close()

    grouped: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}
    for row in rows:
        name = str(row["metric_name"] or "")
        if not name:
            continue
        tag_kind, tag_value = _metric_device_tag(row["metric_device_id"], row["metric_source_name"])
        ts_ns = _to_int(row["ts_ns"], -1)
        if ts_ns < 0:
            continue
        try:
            value = float(row["metric_value"])
        except Exception:
            continue
        grouped.setdefault((name, tag_kind, tag_value), []).append((ts_ns, value))

    if bool(apply_default_focus_filter):
        before_count = len(grouped)
        grouped = {k: v for k, v in grouped.items() if _is_default_focus_metric_name(k[0])}
        debug(
            "metrics default focus filter enabled tokens={} before_series={} after_series={}".format(
                len(_DEFAULT_FOCUS_METRIC_TOKENS),
                int(before_count),
                int(len(grouped)),
            )
        )

    series: List[Dict[str, object]] = []
    for name, tag_kind, tag_value in sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2])):
        points = _downsample_points(
            grouped[(name, tag_kind, tag_value)],
            max_points=int(max_points_per_series),
        )
        display_name = _metric_series_name(str(name), str(tag_kind), str(tag_value))
        series.append(
            {
                "name": display_name,
                "color": _color_for_name(display_name),
                "points": [[int(t), float(v)] for t, v in points],
            }
        )
    total_points = sum(len(item.get("points", [])) for item in series)
    debug(
        "metrics series={} total_points={} max_points_per_series={}".format(
            len(series),
            int(total_points),
            int(max_points_per_series),
        )
    )
    if series:
        debug(
            "metrics series sample={}".format(
                _preview_dict_rows(series, keys=("name", "color"), limit=debug_rows_i)
            )
        )
    return series


def _build_rank_heatmap_rows(kernels: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[Optional[int], int], Dict[str, object]] = {}
    for row in kernels:
        rank_raw = row.get("rank")
        rank = int(rank_raw) if isinstance(rank_raw, int) else None
        dev = _to_int(row.get("device_id"), -1)
        key = (rank, dev)
        item = grouped.setdefault(
            key,
            {
                "rank": rank,
                "device_id": dev,
                "total_ms": 0.0,
                "compute_ms": 0.0,
                "comm_ms": 0.0,
                "kernel_count": 0,
                "streams": set(),
                "occ_num": 0.0,
                "occ_den": 0.0,
            },
        )
        dur_ms = _safe_float(row.get("duration_ms"))
        item["total_ms"] = _safe_float(item.get("total_ms")) + dur_ms
        item["kernel_count"] = int(item.get("kernel_count") or 0) + 1
        item["streams"].add(_to_int(row.get("stream_id"), 0))
        if str(row.get("kind") or "") == "comm":
            item["comm_ms"] = _safe_float(item.get("comm_ms")) + dur_ms
        else:
            item["compute_ms"] = _safe_float(item.get("compute_ms")) + dur_ms
        occ = row.get("occupancy_pct_estimate")
        if occ is not None:
            item["occ_num"] = _safe_float(item.get("occ_num")) + dur_ms * _safe_float(occ)
            item["occ_den"] = _safe_float(item.get("occ_den")) + dur_ms

    rows: List[Dict[str, object]] = []
    for (rank, dev), item in grouped.items():
        occ_den = _safe_float(item.get("occ_den"))
        avg_occ = (_safe_float(item.get("occ_num")) / occ_den) if occ_den > 0 else None
        rows.append(
            {
                "rank": rank,
                "device_id": int(dev),
                "total_ms": round(_safe_float(item.get("total_ms")), 6),
                "compute_ms": round(_safe_float(item.get("compute_ms")), 6),
                "comm_ms": round(_safe_float(item.get("comm_ms")), 6),
                "kernel_count": int(item.get("kernel_count") or 0),
                "stream_count": len(set(item.get("streams") or set())),
                "avg_occupancy_pct": round(avg_occ, 3) if avg_occ is not None else None,
            }
        )
    rows.sort(
        key=lambda x: (
            x.get("rank") is None,
            int(x.get("rank")) if x.get("rank") is not None else 10**9,
            int(x.get("device_id") or -1),
        )
    )
    return rows


def _series_points_numeric(series_item: Dict[str, object]) -> List[Tuple[int, float]]:
    points_raw = list(series_item.get("points") or [])
    out: List[Tuple[int, float]] = []
    for item in points_raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        t = _to_int(item[0], -1)
        v = _safe_float(item[1], float("nan"))
        if t < 0 or not math.isfinite(v):
            continue
        out.append((t, v))
    out.sort(key=lambda x: x[0])
    return out


def _resample_points(points: Sequence[Tuple[int, float]], target_len: int) -> List[Tuple[int, float]]:
    items = list(points)
    if target_len <= 0 or len(items) <= target_len:
        return items
    if target_len == 1:
        return [items[-1]]
    result: List[Tuple[int, float]] = []
    span = float(len(items) - 1)
    for idx in range(target_len):
        pos = int(round((idx / float(target_len - 1)) * span))
        pos = max(0, min(len(items) - 1, pos))
        result.append(items[pos])
    return result


def _series_device_id(name: object) -> int:
    m = re.search(r"\[gpu\s+(\d+)\]", str(name or ""), flags=re.IGNORECASE)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _series_role(name: object) -> str:
    n = _normalize_metric_text(name)
    if "dram throughput" in n or "dram bytes" in n or "hbm" in n or "mem bw" in n:
        return "memory"
    if "tensor active" in n or "sm active" in n or "compute warps in flight" in n:
        return "compute"
    return ""


def _build_roofline_proxy_data(
    metric_series: Sequence[Dict[str, object]],
    *,
    max_points: int = 240,
) -> Dict[str, object]:
    by_dev_role: Dict[int, Dict[str, Dict[str, object]]] = {}
    for item in metric_series:
        name = str(item.get("name") or "")
        role = _series_role(name)
        if not role:
            continue
        dev = _series_device_id(name)
        points = _series_points_numeric(item)
        if len(points) < 2:
            continue
        min_v = min(v for _, v in points)
        max_v = max(v for _, v in points)
        variation = max_v - min_v
        role_bucket = by_dev_role.setdefault(dev, {})
        current = role_bucket.get(role)
        if current is None or _safe_float(current.get("variation")) < variation:
            role_bucket[role] = {
                "name": name,
                "points": points,
                "variation": variation,
            }

    paired_points: List[Dict[str, object]] = []
    stats_rows: List[Dict[str, object]] = []
    for dev, roles in sorted(by_dev_role.items(), key=lambda x: int(x[0])):
        comp = roles.get("compute")
        mem = roles.get("memory")
        if not comp or not mem:
            continue
        comp_points = list(comp.get("points") or [])
        mem_points = list(mem.get("points") or [])
        sample_len = min(len(comp_points), len(mem_points), max(2, int(max_points)))
        comp_resampled = _resample_points(comp_points, sample_len)
        mem_resampled = _resample_points(mem_points, sample_len)

        dev_points: List[Tuple[float, float]] = []
        for (tc, cv), (tm, mv) in zip(comp_resampled, mem_resampled):
            ts = int((tc + tm) // 2)
            mem_pct = max(0.0, min(100.0, float(mv)))
            comp_pct = max(0.0, min(100.0, float(cv)))
            paired_points.append(
                {
                    "device_id": int(dev),
                    "ts_ns": ts,
                    "x_mem_pct": round(mem_pct, 4),
                    "y_compute_pct": round(comp_pct, 4),
                }
            )
            dev_points.append((mem_pct, comp_pct))
        if dev_points:
            avg_mem = sum(p[0] for p in dev_points) / float(len(dev_points))
            avg_comp = sum(p[1] for p in dev_points) / float(len(dev_points))
            gap = max(0.0, avg_mem - avg_comp)
            stats_rows.append(
                {
                    "device_id": int(dev),
                    "point_count": len(dev_points),
                    "avg_mem_pct": round(avg_mem, 3),
                    "avg_compute_pct": round(avg_comp, 3),
                    "gap_pct": round(gap, 3),
                    "compute_series": str(comp.get("name") or ""),
                    "memory_series": str(mem.get("name") or ""),
                }
            )
    return {
        "points": paired_points[: int(max_points) * max(1, len(stats_rows))],
        "stats": stats_rows,
    }


def _build_gil_lane_series(
    metric_series: Sequence[Dict[str, object]],
    *,
    max_points: int = 1500,
) -> List[Dict[str, object]]:
    tokens = (
        "gil",
        "python thread",
        "thread util",
        "interpreter",
        "cpython",
    )
    rows: List[Dict[str, object]] = []
    for item in metric_series:
        name = str(item.get("name") or "")
        norm = _normalize_metric_text(name)
        if not any(token in norm for token in tokens):
            continue
        points = _series_points_numeric(item)
        if not points:
            continue
        sampled = _downsample_points(points, max_points=max_points)
        values = [float(v) for _, v in sampled]
        rows.append(
            {
                "name": name,
                "color": str(item.get("color") or _color_for_name(name)),
                "points": [[int(t), float(v)] for t, v in sampled],
                "avg": sum(values) / float(max(1, len(values))),
                "min": min(values),
                "max": max(values),
                "samples": len(values),
            }
        )
    return rows


def _render_html(
    *,
    sqlite_path: str,
    kernels: List[Dict[str, object]],
    metric_series: List[Dict[str, object]],
    kernel_category_summary: Optional[Dict[str, object]],
    kernel_category_profile: str,
    nvtx_window_category_stats: Optional[Dict[str, object]],
    window_start_ns: int,
    window_end_ns: int,
    display_span_ns: int,
    nvtx_windows: Optional[Sequence[Dict[str, object]]],
    width_px: int,
    include_metrics: bool,
    overlay_metrics_per_track: int,
) -> str:
    data_span = max(1, int(window_end_ns) - int(window_start_ns))
    span = int(data_span)
    if int(display_span_ns) > 0:
        span = max(int(data_span), int(display_span_ns))
    display_end_ns = int(window_start_ns) + int(span)
    grouped: Dict[Tuple[Optional[int], int], Dict[int, List[Dict[str, object]]]] = {}
    for row in kernels:
        rv = row.get("rank")
        rank = int(rv) if isinstance(rv, int) else None
        dev = _to_int(row.get("device_id"), -1)
        sid = _to_int(row.get("stream_id"), 0)
        key = (rank, dev)
        grouped.setdefault(key, {}).setdefault(sid, []).append(row)
    group_keys = sorted(
        grouped.keys(),
        key=lambda k: (k[0] is None, int(k[0]) if k[0] is not None else 10**9, int(k[1])),
    )
    stream_count = sum(len(streams) for streams in grouped.values())
    known_ranks = sorted({int(k[0]) for k in group_keys if k[0] is not None})
    has_unknown_rank = any(k[0] is None for k in group_keys)

    nvtx_count = len(list(nvtx_windows or []))
    meta_nvtx = f" | nvtx_scopes={nvtx_count}" if nvtx_count > 0 else ""
    rank_meta = f" | ranks={len(known_ranks) + (1 if has_unknown_rank else 0)}"

    all_stream_groups: List[Dict[str, object]] = []
    for rank, dev in group_keys:
        stream_rows: List[Dict[str, object]] = []
        streams = grouped.get((rank, dev), {})
        for sid in sorted(streams.keys()):
            kernels_payload: List[Dict[str, object]] = []
            for row in streams[sid]:
                ks = _to_int(row.get("start_ns"), 0)
                ke = _to_int(row.get("end_ns"), 0)
                kname = str(row.get("kernel_name") or "")
                kernels_payload.append(
                    {
                        "kernel_name": kname,
                        "start_ns": ks,
                        "end_ns": ke,
                        "kind": str(row.get("kind") or ""),
                        "duration_ms": row.get("duration_ms"),
                        "occupancy_pct_estimate": row.get("occupancy_pct_estimate"),
                        "color": _color_for_name(kname),
                    }
                )
            stream_rows.append(
                {
                    "stream_id": int(sid),
                    "kernels": kernels_payload,
                }
            )
        all_stream_groups.append(
            {
                "rank": rank,
                "device_id": int(dev),
                "streams": stream_rows,
            }
        )

    rank_heatmap_rows = _build_rank_heatmap_rows(kernels)
    roofline_proxy = _build_roofline_proxy_data(metric_series if include_metrics else [])
    gil_lane_series = _build_gil_lane_series(metric_series if include_metrics else [])

    payload = {
        "window_start_ns": int(window_start_ns),
        "window_end_ns": int(display_end_ns),
        "span_ns": int(span),
        "data_window_end_ns": int(window_end_ns),
        "data_span_ns": int(data_span),
        "display_span_ns": int(span),
        "metrics": metric_series if include_metrics else [],
        "chart_width": int(width_px),
        "all_stream_groups": all_stream_groups,
        "overlay_metrics_per_track": int(max(0, int(overlay_metrics_per_track))),
        "rank_heatmap_rows": rank_heatmap_rows,
        "roofline_proxy": roofline_proxy,
        "gil_lane_series": gil_lane_series,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    lines: List[str] = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<title>NSYS NVTX Timeline</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#0f1116;color:#e7e9ee;}",
        ".meta{margin-bottom:12px;color:#a8afbf;font-size:13px;}",
        ".card{background:#171c28;border:1px solid #2a3243;border-radius:8px;padding:12px;margin-bottom:14px;}",
        ".panel-title{font-size:13px;color:#c8d0df;margin:0 0 8px 0;}",
        ".legend{display:flex;flex-wrap:wrap;gap:8px 12px;margin-top:8px;font-size:12px;color:#b7c0d0;}",
        ".legend-item{display:flex;align-items:center;gap:6px;}",
        ".swatch{width:10px;height:10px;border-radius:2px;}",
        ".metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:10px;}",
        ".metric-panel{background:#121826;border:1px solid #2a3243;border-radius:6px;padding:8px;}",
        ".metric-title{font-size:12px;color:#d6deef;margin-bottom:4px;word-break:break-word;}",
        ".metric-sub{font-size:11px;color:#95a4bf;margin:4px 0 0 0;}",
        ".category-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px;}",
        ".category-card{background:#121826;border:1px solid #2a3243;border-radius:6px;padding:8px;}",
        ".category-title{font-size:12px;color:#d6deef;margin-bottom:4px;word-break:break-word;}",
        ".category-sub{font-size:11px;color:#95a4bf;margin:4px 0 0 0;}",
        ".category-bar{height:8px;background:#0f1522;border:1px solid #33435f;border-radius:999px;overflow:hidden;margin-top:6px;}",
        ".category-fill{height:100%;background:linear-gradient(90deg,#5f8bff,#60d3a5);}",
        ".simple-table{width:100%;border-collapse:collapse;font-size:11px;color:#dce5f8;margin-top:8px;}",
        ".simple-table th,.simple-table td{border-bottom:1px solid #2a3243;padding:4px 6px;text-align:left;vertical-align:top;}",
        ".simple-table th{color:#9cb0d0;font-weight:600;}",
        ".heatmap-table{width:100%;border-collapse:collapse;font-size:11px;color:#dce5f8;margin-top:8px;}",
        ".heatmap-table th,.heatmap-table td{border-bottom:1px solid #2a3243;padding:5px 7px;text-align:left;vertical-align:middle;}",
        ".heatmap-table th{color:#9cb0d0;font-weight:600;}",
        ".heat-cell{position:relative;border-radius:4px;padding:2px 6px;display:inline-block;min-width:72px;text-align:right;color:#eaf1ff;font-family:Consolas,Monaco,monospace;}",
        ".roofline-wrap{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;}",
        ".roofline-legend{font-size:11px;color:#96a5bf;display:flex;flex-direction:column;gap:4px;min-width:220px;}",
        ".gil-lane-grid{display:flex;flex-direction:column;gap:8px;}",
        ".gil-lane-item{background:#121826;border:1px solid #2a3243;border-radius:6px;padding:6px;}",
        ".gil-lane-title{font-size:11px;color:#d6deef;margin-bottom:4px;word-break:break-word;}",
        ".gil-lane-sub{font-size:11px;color:#95a4bf;margin-top:4px;}",
        ".row{display:flex;align-items:center;margin:8px 0;}",
        ".label{width:140px;color:#a8afbf;font-size:12px;}",
        f".track{{position:relative;height:24px;width:{int(width_px)}px;background:#1a1f2b;border-radius:4px;overflow:hidden;}}",
        ".bar{position:absolute;height:18px;top:3px;border-radius:3px;}",
        ".bar:hover{outline:1px solid #fff;}",
        ".stream-track{height:56px;background:#171d2b;}",
        ".stream-track .bar{top:34px;height:18px;}",
        ".metric-overlay{position:absolute;left:0;top:2px;width:100%;height:30px;pointer-events:none;}",
        ".metric-overlay line{stroke:#5b6c8e;stroke-width:1;opacity:0.7;}",
        ".metric-overlay polyline{fill:none;stroke-width:1.2;stroke-linejoin:round;stroke-linecap:round;opacity:0.92;}",
        ".stream-overlay-hint{font-size:11px;color:#94a3bd;margin-top:8px;}",
        ".allstream-root{display:flex;flex-direction:column;gap:10px;}",
        ".allstream-panel{position:relative;background:#121826;border:1px solid #2a3243;border-radius:6px;padding:8px;}",
        ".allstream-title{font-size:12px;color:#d6deef;margin:0 0 6px 0;}",
        ".allstream-scroll{overflow-x:auto;overflow-y:hidden;border:1px solid #29344a;border-radius:4px;background:#0f1522;}",
        ".allstream-controls{display:flex;flex-wrap:wrap;gap:6px 10px;margin-bottom:6px;}",
        ".shift-row{display:flex;align-items:center;gap:6px;font-size:11px;color:#9fb0ca;background:#101725;border:1px solid #2a3243;border-radius:4px;padding:3px 6px;}",
        ".shift-btn{background:#1a2538;color:#dbe6ff;border:1px solid #3d4f70;border-radius:3px;padding:1px 6px;cursor:pointer;font-size:11px;}",
        ".shift-btn:hover{background:#22314a;}",
        ".allstream-svg{display:block;}",
        ".allstream-hover-tip{position:absolute;z-index:20;min-width:220px;max-width:460px;background:rgba(10,16,28,0.96);border:1px solid #47608c;border-radius:6px;padding:6px 8px;pointer-events:none;box-shadow:0 4px 16px rgba(0,0,0,0.35);font-size:11px;color:#dce7ff;}",
        ".allstream-hover-head{font-size:11px;color:#a9bce0;margin-bottom:4px;font-family:Consolas,Monaco,monospace;}",
        ".allstream-hover-row{display:flex;align-items:center;gap:6px;line-height:1.25;margin:2px 0;}",
        ".allstream-hover-swatch{width:8px;height:8px;border-radius:2px;flex:none;}",
        ".allstream-hover-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}",
        ".allstream-hover-val{font-family:Consolas,Monaco,monospace;color:#f3f7ff;}",
        ".axis-note{font-size:11px;color:#8792a8;margin-top:6px;}",
        ".empty{color:#8f9ab0;font-size:12px;padding:6px 0;}",
        ".group-title{font-size:12px;color:#c9d3e8;margin:10px 0 4px 0;}",
        "</style></head><body>",
        "<h2>NSYS NVTX Window Timeline</h2>",
        (
            f"<div class='meta'>sqlite={html.escape(str(sqlite_path))} | "
            f"kernels={len(kernels)} | streams={stream_count}{rank_meta}{meta_nvtx}</div>"
        ),
    ]

    if include_metrics:
        lines.extend(
            [
                "<div class='card'>",
                "<h3 class='panel-title'>GPU Metrics In Window</h3>",
                "<div id='metrics-grid' class='metrics-grid'></div>",
                "<div class='axis-note'>Each metric/device is rendered in an independent panel (own Y-axis). X-axis is shared ns timeline.</div>",
                "</div>",
            ]
        )

    lines.extend(
        [
            "<div class='card'>",
            "<h3 class='panel-title'>Rank Heatmap</h3>",
            "<div id='rank-heatmap-root'></div>",
            "<div class='axis-note'>Aggregates compute/comm duration by rank-device pair for fast straggler detection.</div>",
            "</div>",
            "<div class='card'>",
            "<h3 class='panel-title'>Roofline Proxy</h3>",
            "<div id='roofline-root'></div>",
            "<div class='axis-note'>Proxy scatter: X=memory throughput (%), Y=compute activity (%), grouped by GPU device.</div>",
            "</div>",
            "<div class='card'>",
            "<h3 class='panel-title'>Python GIL Lane</h3>",
            "<div id='gil-lane-root'></div>",
            "<div class='axis-note'>Renders python/gil/thread-util metric lanes when such signals exist in the selected window.</div>",
            "</div>",
        ]
    )

    category_rows = list((kernel_category_summary or {}).get("rows") or [])
    if category_rows:
        wall_ms = _safe_float((kernel_category_summary or {}).get("wall_ms"))
        non_overlap_ms = _safe_float((kernel_category_summary or {}).get("non_overlap_ms"))
        busy_union_ms = _safe_float((kernel_category_summary or {}).get("busy_union_ms", non_overlap_ms))
        idle_ms = _safe_float((kernel_category_summary or {}).get("idle_ms"))
        busy_pct_wall = _safe_float((kernel_category_summary or {}).get("busy_pct_of_wall"))
        cross_overlap_ms = _safe_float((kernel_category_summary or {}).get("cross_category_overlap_ms"))
        raw_total_ms = _safe_float((kernel_category_summary or {}).get("raw_total_ms"))
        overlap_saved_ms = _safe_float((kernel_category_summary or {}).get("overlap_saved_ms"))
        profile_text = str(kernel_category_profile or "custom")
        lines.extend(
            [
                "<div class='card'>",
                (
                    "<h3 class='panel-title'>Kernel Category Breakdown "
                    "(Overlap-Aware Across All Streams)</h3>"
                ),
                (
                    "<div class='axis-note'>"
                    f"profile={html.escape(profile_text)} | wall_ms={wall_ms:.3f} | "
                    f"busy_union_ms={busy_union_ms:.3f} ({busy_pct_wall:.2f}% wall) | idle_ms={idle_ms:.3f} | "
                    f"cross_category_overlap_ms={cross_overlap_ms:.3f} | raw_total_ms={raw_total_ms:.3f} | "
                    f"overlap_saved_ms={overlap_saved_ms:.3f}. "
                    "Three ratio views are provided per category: "
                    "raw(sum, can exceed 100%), weighted(equal-slice share), and exclusive/overlap split."
                    "</div>"
                ),
                "<div class='category-grid'>",
            ]
        )
        for row in category_rows:
            cat_name = str(row.get("category") or "misc")
            weighted_pct = max(0.0, min(100.0, _safe_float(row.get("weighted_pct_of_nonoverlap"))))
            lines.extend(
                [
                    "<div class='category-card'>",
                    f"<div class='category-title'>{html.escape(cat_name)}</div>",
                    (
                        "<div class='category-sub'>"
                        f"raw_sum={_safe_float(row.get('raw_total_ms')):.3f} ms "
                        f"({_safe_float(row.get('raw_pct_of_wall')):.2f}% wall) | "
                        f"weighted={_safe_float(row.get('weighted_elapsed_ms')):.3f} ms "
                        f"({weighted_pct:.2f}% busy, {_safe_float(row.get('weighted_pct_of_wall')):.2f}% wall) | "
                        f"union={_safe_float(row.get('union_elapsed_ms')):.3f} ms ({_safe_float(row.get('union_pct_of_wall')):.2f}% wall) | "
                        f"exclusive={_safe_float(row.get('exclusive_elapsed_ms')):.3f} ms ({_safe_float(row.get('exclusive_pct_of_wall')):.2f}% wall) | "
                        f"overlap={_safe_float(row.get('overlap_elapsed_ms')):.3f} ms ({_safe_float(row.get('overlap_pct_of_wall')):.2f}% wall) | "
                        f"instances={int(row.get('instances') or 0)} | streams={int(row.get('stream_count') or 0)}"
                        "</div>"
                    ),
                    "<div class='category-bar'>",
                    f"<div class='category-fill' style='width:{weighted_pct:.2f}%;'></div>",
                    "</div>",
                    "</div>",
                ]
            )
        lines.extend(["</div>", "</div>"])

    nvtx_stats = dict(nvtx_window_category_stats or {})
    nvtx_stats_rows = list(nvtx_stats.get("category_summary_rows") or [])
    nvtx_window_count = int(nvtx_stats.get("window_count") or 0)
    nvtx_dur_stats = dict(nvtx_stats.get("window_duration_stats") or {})
    if nvtx_window_count > 0:
        cpu_dur = dict(nvtx_dur_stats.get("cpu_duration_ms") or {})
        gpu_dur = dict(nvtx_dur_stats.get("gpu_duration_ms") or {})
        non_overlap_dur = dict(nvtx_dur_stats.get("non_overlap_ms") or {})
        gpu_idle_dur = dict(nvtx_dur_stats.get("gpu_idle_ms") or {})
        raw_total_dur = dict(nvtx_dur_stats.get("raw_total_ms") or {})
        outlier_rows = list(nvtx_stats.get("outlier_windows") or [])
        lines.extend(
            [
                "<div class='card'>",
                "<h3 class='panel-title'>Per-Matched-NVTX Category Stability</h3>",
                (
                    "<div class='axis-note'>"
                    f"windows={int(nvtx_stats.get('window_count') or 0)} | categories={int(nvtx_stats.get('category_count') or 0)}. "
                    f"avg_cpu_nvtx_ms(clipped)={_safe_float(cpu_dur.get('avg_clipped')):.3f} | "
                    f"avg_gpu_envelope_ms(clipped)={_safe_float(gpu_dur.get('avg_clipped')):.3f} | "
                    f"avg_busy_nonoverlap_ms(clipped)={_safe_float(non_overlap_dur.get('avg_clipped')):.3f} | "
                    f"avg_gpu_idle_ms(clipped)={_safe_float(gpu_idle_dur.get('avg_clipped')):.3f} | "
                    f"avg_raw_sum_ms(clipped)={_safe_float(raw_total_dur.get('avg_clipped')):.3f}. "
                    "Each matched NVTX scope is analyzed independently, and each scope's category ratio is computed from "
                    "GPU kernel execution intervals (non-overlap weighted), not CPU-side NVTX duration. "
                    "avg/std uses IQR clipping (Q1-1.5IQR, Q3+1.5IQR) to reduce anomalous-window impact. "
                    "avg_ms(weighted,clipped) is overlap-aware category time mean per matched NVTX window."
                    "</div>"
                ),
            ]
        )
        if nvtx_stats_rows:
            lines.extend(
                [
                    "<table class='simple-table'>",
                    "<thead><tr><th>category</th><th>avg%(clipped)</th><th>avg_ms(weighted,clipped)</th><th>avg_ms(raw_sum,clipped)</th><th>avg%(raw)</th><th>std%(clipped)</th><th>std%(raw)</th><th>removed/windows</th><th>nonzero/windows</th></tr></thead>",
                    "<tbody>",
                ]
            )
            for row in nvtx_stats_rows[:14]:
                lines.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(row.get('category') or 'misc'))}</code></td>"
                    f"<td>{_safe_float(row.get('avg_pct')):.2f}</td>"
                    f"<td>{_safe_float(row.get('avg_weighted_ms')):.3f}</td>"
                    f"<td>{_safe_float(row.get('avg_raw_total_ms')):.3f}</td>"
                    f"<td>{_safe_float(row.get('avg_pct_raw')):.2f}</td>"
                    f"<td>{_safe_float(row.get('std_pct')):.2f}</td>"
                    f"<td>{_safe_float(row.get('std_pct_raw')):.2f}</td>"
                    f"<td>{int(row.get('removed_windows') or 0)}/{int(row.get('window_count') or 0)}</td>"
                    f"<td>{int(row.get('nonzero_windows') or 0)}/{int(row.get('window_count') or 0)}</td>"
                    "</tr>"
                )
            lines.extend(["</tbody>", "</table>"])
        else:
            lines.append("<div class='empty'>No category kernels matched in selected NVTX windows.</div>")
        if outlier_rows:
            lines.extend(
                [
                    "<div class='axis-note'>Potential anomalous scopes (|z|>=2, based on per-category ratio z-score):</div>",
                    "<table class='simple-table'>",
                    "<thead><tr><th>scope_idx</th><th>nvtx_text</th><th>outlier_category</th><th>z</th><th>gpu_dur_ms</th><th>kernels</th><th>warmup_head</th></tr></thead>",
                    "<tbody>",
                ]
            )
            for row in outlier_rows[:16]:
                lines.append(
                    "<tr>"
                    f"<td>{int(row.get('window_index') or 0)}</td>"
                    f"<td>{html.escape(str(row.get('nvtx_text') or ''))}</td>"
                    f"<td><code>{html.escape(str(row.get('outlier_category') or ''))}</code></td>"
                    f"<td>{_safe_float(row.get('outlier_z')):+.2f}</td>"
                    f"<td>{_safe_float(row.get('gpu_duration_ms')):.3f}</td>"
                    f"<td>{int(row.get('kernel_count') or 0)}</td>"
                    f"<td>{'yes' if bool(row.get('warmup_head_window')) else 'no'}</td>"
                    "</tr>"
                )
            lines.extend(["</tbody>", "</table>"])
        lines.append("</div>")

    lines.extend(
        [
            "<div class='card'>",
            "<h3 class='panel-title'>All Streams Overlap + Metrics Alignment</h3>",
            "<div id='allstream-root' class='allstream-root'></div>",
            (
                "<div class='axis-note'>"
                "All streams for the same rank/device are merged into one wide panel. "
                "Kernel Theoretical Occupancy Sum [%] is auto-derived by summing overlapping kernels, and "
                "you can use +/- controls to shift each metric curve vertically while keeping the same shared X-axis."
                "</div>"
            ),
            "</div>",
        ]
    )

    if nvtx_count > 0:
        lines.extend(["<div class='card'>", "<h3 class='panel-title'>Matched NVTX Scopes</h3>"])
        lines.append("<div class='row'>")
        lines.append("<div class='label'>nvtx</div>")
        lines.append("<div class='track'>")
        for i, scope in enumerate(nvtx_windows or []):
            s = _to_int(scope.get("start_ns"), 0)
            e = _to_int(scope.get("end_ns"), 0)
            if e <= s:
                continue
            left = (s - int(window_start_ns)) / float(span)
            width = max((e - s) / float(span), 1.0 / float(width_px))
            left_px = int(left * width_px)
            width_bar = max(1, int(width * width_px))
            name = str(scope.get("nvtx_text") or f"scope_{i}")
            color = _color_for_name(f"nvtx::{name}")
            title = f"{name} | start_ns={s} | end_ns={e} | duration_ms={scope.get('duration_ms')}"
            lines.append(
                f"<div class='bar' style='left:{left_px}px;width:{width_bar}px;background:{color};opacity:0.85;' "
                f"title='{html.escape(title)}'></div>"
            )
        lines.append("</div></div>")
        lines.append("</div>")

    lines.extend(["<div class='card'>", "<h3 class='panel-title'>Kernel Timeline By Stream</h3>"])
    if not kernels:
        lines.append("<div class='empty'>No kernels in selected window.</div>")
    else:
        for rank, dev in group_keys:
            if rank is None:
                group_title = "Rank Unknown"
            else:
                group_title = f"Rank {rank}"
            if int(dev) >= 0:
                group_title += f" | Device {int(dev)}"
            lines.append(f"<div class='group-title'>{html.escape(group_title)}</div>")
            streams = grouped.get((rank, dev), {})
            for sid in sorted(streams.keys()):
                rank_attr = "" if rank is None else str(rank)
                lines.append("<div class='row'>")
                lines.append(f"<div class='label'>stream {sid}</div>")
                lines.append(
                    "<div class='track stream-track' "
                    f"data-rank='{html.escape(rank_attr)}' "
                    f"data-device='{int(dev)}' "
                    f"data-stream='{int(sid)}'>"
                )
                for row in streams[sid]:
                    s = _to_int(row.get("start_ns"), 0)
                    e = _to_int(row.get("end_ns"), 0)
                    left = (s - int(window_start_ns)) / float(span)
                    width = max((e - s) / float(span), 1.0 / float(width_px))
                    left_px = int(left * width_px)
                    width_bar = max(1, int(width * width_px))
                    name = str(row.get("kernel_name") or "")
                    color = _color_for_name(name)
                    title = (
                        f"{name} | kind={row.get('kind')} | dur_ms={row.get('duration_ms')} | "
                        f"start_ns={s} | end_ns={e} | stream={sid} | device={dev} | rank={rank} | "
                        f"tpb={row.get('threads_per_block')} | regs={row.get('registers_per_thread')} | "
                        f"smem={row.get('total_shared_bytes')} | occ_theoretical_pct={row.get('occupancy_pct_estimate')}"
                    )
                    lines.append(
                        f"<div class='bar' style='left:{left_px}px;width:{width_bar}px;background:{color};' "
                        f"title='{html.escape(title)}'></div>"
                    )
                lines.append("</div></div>")
        if include_metrics and int(overlay_metrics_per_track) > 0:
            lines.append(
                "<div class='stream-overlay-hint'>"
                "Per-stream attribution overlay: metric curves are stacked above kernel bars in each stream lane "
                "(metrics are selected per device by highest variation)."
                "</div>"
            )
    lines.extend(["</div>"])

    lines.extend(
        [
            "<script>",
            f"const TIMELINE_DATA = {payload_json};",
            "(function(){",
            "  const d = TIMELINE_DATA;",
            "  const grid = document.getElementById('metrics-grid');",
            "  const series = [...(Array.isArray(d.metrics) ? d.metrics : [])].sort((a,b) => String(a.name||'').localeCompare(String(b.name||'')));",
            "  const span = Math.max(1, Number(d.span_ns || 1));",
            "  const startNs = Number(d.window_start_ns || 0);",
            "  const mk = (tag, attrs) => {",
            "    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);",
            "    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v));",
            "    return el;",
            "  };",
            "  const fmt = (v) => {",
            "    const x = Number(v);",
            "    if (!Number.isFinite(x)) return 'nan';",
            "    const ax = Math.abs(x);",
            "    if (ax >= 1e6 || (ax > 0 && ax < 1e-3)) return x.toExponential(2);",
            "    return x.toFixed(3);",
            "  };",
            "  if (grid) {",
            "    if (!series.length) {",
            "      grid.innerHTML = \"<div class='empty'>No GPU metric samples in this window.</div>\";",
            "    } else {",
            "      const W = Math.max(480, Math.min(920, Number(d.chart_width || 1200)));",
            "      const H = 190;",
            "      const padL = 52, padR = 16, padT = 12, padB = 26;",
            "      const x0 = padL, y0 = padT, cw = W - padL - padR, ch = H - padT - padB;",
            "      const x = (t) => x0 + ((Number(t) - startNs) / span) * cw;",
            "      for (const s of series) {",
            "        const ptsRaw = Array.isArray(s.points) ? s.points : [];",
            "        if (!ptsRaw.length) continue;",
            "        let minV = Number.POSITIVE_INFINITY;",
            "        let maxV = Number.NEGATIVE_INFINITY;",
            "        for (const p of ptsRaw) {",
            "          const v = Number(p[1]);",
            "          if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "        }",
            "        if (!Number.isFinite(minV) || !Number.isFinite(maxV)) { minV = 0; maxV = 1; }",
            "        if (Math.abs(maxV - minV) < 1e-12) { maxV = minV + 1.0; }",
            "        const y = (v) => y0 + (1.0 - ((Number(v) - minV) / (maxV - minV))) * ch;",
            "        const card = document.createElement('div'); card.className = 'metric-panel';",
            "        const title = document.createElement('div'); title.className = 'metric-title'; title.textContent = String(s.name || 'metric');",
            "        const svg = mk('svg', {width:W, height:H, viewBox:`0 0 ${W} ${H}`});",
            "        svg.appendChild(mk('rect', {x:x0, y:y0, width:cw, height:ch, fill:'#111725', stroke:'#32405b'}));",
            "        svg.appendChild(mk('line', {x1:x0, y1:y0+ch, x2:x0+cw, y2:y0+ch, stroke:'#55637f'}));",
            "        svg.appendChild(mk('line', {x1:x0, y1:y0, x2:x0, y2:y0+ch, stroke:'#55637f'}));",
            "        const yMinText = mk('text', {x:6, y:y0+ch, fill:'#9aa7be', 'font-size':11}); yMinText.textContent = fmt(minV); svg.appendChild(yMinText);",
            "        const yMaxText = mk('text', {x:6, y:y0+10, fill:'#9aa7be', 'font-size':11}); yMaxText.textContent = fmt(maxV); svg.appendChild(yMaxText);",
            "        const xStartText = mk('text', {x:x0, y:y0+ch+16, fill:'#7f8daa', 'font-size':10}); xStartText.textContent = String(Math.round(startNs)); svg.appendChild(xStartText);",
            "        const xEndText = mk('text', {x:x0+cw-92, y:y0+ch+16, fill:'#7f8daa', 'font-size':10}); xEndText.textContent = String(Math.round(startNs + span)); svg.appendChild(xEndText);",
            "        const pts = ptsRaw.map((p) => `${x(p[0]).toFixed(2)},${y(p[1]).toFixed(2)}`).join(' ');",
            "        if (pts) {",
            "          svg.appendChild(mk('polyline', {points:pts, fill:'none', stroke:s.color || '#88c', 'stroke-width':1.4, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "        }",
            "        const sub = document.createElement('div'); sub.className = 'metric-sub';",
            "        sub.textContent = `samples=${ptsRaw.length} | min=${fmt(minV)} | max=${fmt(maxV)}`;",
            "        card.appendChild(title);",
            "        card.appendChild(svg);",
            "        card.appendChild(sub);",
            "        grid.appendChild(card);",
            "      }",
            "    }",
            "  }",
            "  const parseGpuTag = (name) => {",
            "    const m = String(name || '').match(/\\[gpu\\s+(\\d+)\\]/i);",
            "    return m ? Number(m[1]) : null;",
            "  };",
            "  const normalizeMetricName = (name) => {",
            "    let s = String(name || '').toLowerCase();",
            "    s = s.replace(/\\[[^\\]]*\\]/g, ' ');",
            "    s = s.replace(/_/g, ' ');",
            "    s = s.replace(/\\s+/g, ' ').trim();",
            "    return s;",
            "  };",
            "  const isThroughputPercentName = (name) => {",
            "    const raw = String(name || '').toLowerCase();",
            "    const n = normalizeMetricName(name);",
            "    return raw.includes('%') || n.includes('throughput') || n.includes('pct') || n.includes('percent');",
            "  };",
            "  const isComputeWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('compute warps in flight') || (n.includes('warps in flight') && !n.includes('unallocated'))) && isThroughputPercentName(name);",
            "  };",
            "  const isUnallocatedWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('unallocated warps in active sms') || n.includes('unallocated warps')) && isThroughputPercentName(name);",
            "  };",
            "  const toNumericPoints = (seriesLike) => {",
            "    const ptsRaw = Array.isArray(seriesLike && seriesLike.points) ? seriesLike.points : [];",
            "    const out = [];",
            "    for (const p of ptsRaw) {",
            "      const t = Number(p[0]);",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(t) && Number.isFinite(v)) out.push([t, v]);",
            "    }",
            "    out.sort((a,b) => a[0] - b[0]);",
            "    return out;",
            "  };",
            "  const interpAt = (pts, t) => {",
            "    if (!pts.length) return 0;",
            "    const tt = Number(t);",
            "    if (tt <= pts[0][0]) return Number(pts[0][1]) || 0;",
            "    const last = pts[pts.length - 1];",
            "    if (tt >= last[0]) return Number(last[1]) || 0;",
            "    let lo = 0, hi = pts.length - 1;",
            "    while (lo + 1 < hi) {",
            "      const mid = (lo + hi) >> 1;",
            "      if (pts[mid][0] <= tt) lo = mid; else hi = mid;",
            "    }",
            "    const t0 = Number(pts[lo][0]);",
            "    const t1 = Number(pts[hi][0]);",
            "    const v0 = Number(pts[lo][1]);",
            "    const v1 = Number(pts[hi][1]);",
            "    if (!Number.isFinite(t0) || !Number.isFinite(t1) || Math.abs(t1 - t0) < 1e-12) return Number.isFinite(v1) ? v1 : 0;",
            "    const a = (tt - t0) / (t1 - t0);",
            "    const vv0 = Number.isFinite(v0) ? v0 : 0;",
            "    const vv1 = Number.isFinite(v1) ? v1 : 0;",
            "    return vv0 + (vv1 - vv0) * a;",
            "  };",
            "  const seriesScore = (s) => {",
            "    const ptsRaw = Array.isArray(s.points) ? s.points : [];",
            "    if (!ptsRaw.length) return 0;",
            "    let minV = Number.POSITIVE_INFINITY;",
            "    let maxV = Number.NEGATIVE_INFINITY;",
            "    for (const p of ptsRaw) {",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "    }",
            "    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return 0;",
            "    return Math.abs(maxV - minV);",
            "  };",
            "  const overlayCount = Math.max(0, Number(d.overlay_metrics_per_track || 0));",
            "  const tracks = Array.from(document.querySelectorAll('.stream-track'));",
            "  if (overlayCount > 0 && tracks.length > 0) {",
            "    for (const track of tracks) {",
            "      const dev = Number(track.dataset.device || -1);",
            "      let selected = series.filter((s) => {",
            "        const g = parseGpuTag(s.name);",
            "        return g !== null && g === dev;",
            "      });",
            "      if (selected.length === 0) selected = series;",
            "      selected = [...selected].sort((a,b) => seriesScore(b) - seriesScore(a));",
            "      if (overlayCount > 0) {",
            "        let trimmed = selected.slice(0, overlayCount);",
            "        if (overlayCount >= 2) {",
            "          const computePreferred = selected.find((s) => isComputeWarpsSeries(s.name));",
            "          const unallocPreferred = selected.find((s) => isUnallocatedWarpsSeries(s.name));",
            "          if (computePreferred && unallocPreferred) {",
            "            trimmed = [computePreferred, unallocPreferred, ...trimmed.filter((s) => s !== computePreferred && s !== unallocPreferred)].slice(0, overlayCount);",
            "          }",
            "        }",
            "        selected = trimmed;",
            "      } else {",
            "        selected = [];",
            "      }",
            "      if (!selected.length) continue;",
            "      const tw = Math.max(1, Number(track.clientWidth || d.chart_width || 1));",
            "      const svgH = 30;",
            "      const yTop = 3;",
            "      const yBottom = 27;",
            "      const svg = mk('svg', {class:'metric-overlay', width:tw, height:svgH, viewBox:`0 0 ${tw} ${svgH}`, preserveAspectRatio:'none'});",
            "      svg.appendChild(mk('line', {x1:0, y1:yBottom, x2:tw, y2:yBottom}));",
            "      const xTrack = (t) => ((Number(t) - startNs) / span) * tw;",
            "      const clampPct = (v) => Math.max(0, Math.min(100, Number(v) || 0));",
            "      const yPct = (vPct) => yBottom - (clampPct(vPct) / 100.0) * (yBottom - yTop);",
            "      const drawArea = (topPts, basePts, color, opacity) => {",
            "        if (topPts.length < 2 || basePts.length < 2) return;",
            "        const poly = [];",
            "        for (const p of topPts) poly.push(`${p[0].toFixed(2)},${p[1].toFixed(2)}`);",
            "        for (let i = basePts.length - 1; i >= 0; i--) poly.push(`${basePts[i][0].toFixed(2)},${basePts[i][1].toFixed(2)}`);",
            "        svg.appendChild(mk('polygon', {points:poly.join(' '), fill:String(color || '#88c'), opacity:String(opacity)}));",
            "      };",
            "      const computeSeries = selected.find((s) => isComputeWarpsSeries(s.name));",
            "      const unallocSeries = selected.find((s) => isUnallocatedWarpsSeries(s.name));",
            "      const handled = new Set();",
            "      if (computeSeries && unallocSeries) {",
            "        const cPts = toNumericPoints(computeSeries);",
            "        const uPts = toNumericPoints(unallocSeries);",
            "        if (cPts.length >= 2 && uPts.length >= 2) {",
            "          const cTop = [];",
            "          const cBase = [];",
            "          for (const p of cPts) {",
            "            const tx = xTrack(p[0]);",
            "            if (!Number.isFinite(tx)) continue;",
            "            cTop.push([tx, yPct(p[1])]);",
            "            cBase.push([tx, yBottom]);",
            "          }",
            "          drawArea(cTop, cBase, String(computeSeries.color || '#4fa5ff'), 0.42);",
            "          const uTop = [];",
            "          const uBase = [];",
            "          for (const p of uPts) {",
            "            const tx = xTrack(p[0]);",
            "            if (!Number.isFinite(tx)) continue;",
            "            const base = clampPct(interpAt(cPts, p[0]));",
            "            const top = clampPct(base + p[1]);",
            "            uTop.push([tx, yPct(top)]);",
            "            uBase.push([tx, yPct(base)]);",
            "          }",
            "          drawArea(uTop, uBase, String(unallocSeries.color || '#ffae57'), 0.58);",
            "          const cLine = cTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "          if (cLine) svg.appendChild(mk('polyline', {points:cLine, fill:'none', stroke:String(computeSeries.color || '#4fa5ff'), 'stroke-width':1.05, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "          const uLine = uTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "          if (uLine) svg.appendChild(mk('polyline', {points:uLine, fill:'none', stroke:String(unallocSeries.color || '#ffae57'), 'stroke-width':1.05, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "          handled.add(computeSeries);",
            "          handled.add(unallocSeries);",
            "        }",
            "      }",
            "      for (const ms of selected) {",
            "        if (handled.has(ms)) continue;",
            "        const ptsRaw = toNumericPoints(ms);",
            "        if (!ptsRaw.length) continue;",
            "        let minV = Number.POSITIVE_INFINITY;",
            "        let maxV = Number.NEGATIVE_INFINITY;",
            "        for (const p of ptsRaw) {",
            "          const v = Number(p[1]);",
            "          if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "        }",
            "        if (!Number.isFinite(minV) || !Number.isFinite(maxV)) continue;",
            "        if (Math.abs(maxV - minV) < 1e-12) maxV = minV + 1.0;",
            "        const pts = [];",
            "        for (const p of ptsRaw) {",
            "          const tx = xTrack(p[0]);",
            "          const v = Number(p[1]);",
            "          if (!Number.isFinite(tx) || !Number.isFinite(v)) continue;",
            "          const ty = yTop + (1.0 - ((v - minV) / (maxV - minV))) * (yBottom - yTop);",
            "          pts.push(`${tx.toFixed(2)},${ty.toFixed(2)}`);",
            "        }",
            "        if (pts.length >= 2) {",
            "          svg.appendChild(mk('polyline', {points:pts.join(' '), stroke:ms.color || '#88c'}));",
            "        }",
            "      }",
            "      track.appendChild(svg);",
            "      const names = selected.map((s) => String(s.name || '')).join(' | ');",
            "      if (names) {",
            "        const oldTitle = String(track.getAttribute('title') || '');",
            "        const title = oldTitle ? `${oldTitle} || overlay_metrics=${names}` : `overlay_metrics=${names}`;",
            "        track.setAttribute('title', title);",
            "      }",
            "    }",
            "  }",
            "})();",
            "</script>",
            "<script>",
            "(function(){",
            "  const d = TIMELINE_DATA;",
            "  const root = document.getElementById('allstream-root');",
            "  if (!root) return;",
            "  const groups = Array.isArray(d.all_stream_groups) ? d.all_stream_groups : [];",
            "  if (!groups.length) {",
            "    root.innerHTML = \"<div class='empty'>No kernels available for all-stream view.</div>\";",
            "    return;",
            "  }",
            "  const metricsAll = Array.isArray(d.metrics) ? d.metrics : [];",
            "  const overlayCount = Math.max(0, Number(d.overlay_metrics_per_track || 0));",
            "  const span = Math.max(1, Number(d.span_ns || 1));",
            "  const startNs = Number(d.window_start_ns || 0);",
            "  const basePlotW = Math.max(2200, Math.round(Number(d.chart_width || 1800) * 1.35));",
            "  const parseGpuTag = (name) => {",
            "    const m = String(name || '').match(/\\[gpu\\s+(\\d+)\\]/i);",
            "    return m ? Number(m[1]) : null;",
            "  };",
            "  const normalizeMetricName = (name) => {",
            "    let s = String(name || '').toLowerCase();",
            "    s = s.replace(/\\[[^\\]]*\\]/g, ' ');",
            "    s = s.replace(/_/g, ' ');",
            "    s = s.replace(/\\s+/g, ' ').trim();",
            "    return s;",
            "  };",
            "  const isThroughputPercentName = (name) => {",
            "    const raw = String(name || '').toLowerCase();",
            "    const n = normalizeMetricName(name);",
            "    return raw.includes('%') || n.includes('throughput') || n.includes('pct') || n.includes('percent');",
            "  };",
            "  const isComputeWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('compute warps in flight') || (n.includes('warps in flight') && !n.includes('unallocated'))) && isThroughputPercentName(name);",
            "  };",
            "  const isUnallocatedWarpsSeries = (name) => {",
            "    const n = normalizeMetricName(name);",
            "    return (n.includes('unallocated warps in active sms') || n.includes('unallocated warps')) && isThroughputPercentName(name);",
            "  };",
            "  const toNumericPoints = (seriesLike) => {",
            "    const ptsRaw = Array.isArray(seriesLike && seriesLike.points) ? seriesLike.points : [];",
            "    const out = [];",
            "    for (const p of ptsRaw) {",
            "      const t = Number(p[0]);",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(t) && Number.isFinite(v)) out.push([t, v]);",
            "    }",
            "    out.sort((a,b) => a[0] - b[0]);",
            "    return out;",
            "  };",
            "  const interpAt = (pts, t) => {",
            "    if (!pts.length) return 0;",
            "    const tt = Number(t);",
            "    if (tt <= pts[0][0]) return Number(pts[0][1]) || 0;",
            "    const last = pts[pts.length - 1];",
            "    if (tt >= last[0]) return Number(last[1]) || 0;",
            "    let lo = 0, hi = pts.length - 1;",
            "    while (lo + 1 < hi) {",
            "      const mid = (lo + hi) >> 1;",
            "      if (pts[mid][0] <= tt) lo = mid; else hi = mid;",
            "    }",
            "    const t0 = Number(pts[lo][0]);",
            "    const t1 = Number(pts[hi][0]);",
            "    const v0 = Number(pts[lo][1]);",
            "    const v1 = Number(pts[hi][1]);",
            "    if (!Number.isFinite(t0) || !Number.isFinite(t1) || Math.abs(t1 - t0) < 1e-12) return Number.isFinite(v1) ? v1 : 0;",
            "    const a = (tt - t0) / (t1 - t0);",
            "    const vv0 = Number.isFinite(v0) ? v0 : 0;",
            "    const vv1 = Number.isFinite(v1) ? v1 : 0;",
            "    return vv0 + (vv1 - vv0) * a;",
            "  };",
            "  const mkSvg = (tag, attrs) => {",
            "    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);",
            "    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v));",
            "    return el;",
            "  };",
            "  const metricRange = (series) => {",
            "    const pts = Array.isArray(series.points) ? series.points : [];",
            "    let minV = Number.POSITIVE_INFINITY;",
            "    let maxV = Number.NEGATIVE_INFINITY;",
            "    for (const p of pts) {",
            "      const v = Number(p[1]);",
            "      if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "    }",
            "    if (!Number.isFinite(minV) || !Number.isFinite(maxV)) return 0;",
            "    return Math.abs(maxV - minV);",
            "  };",
            "  const shortName = (name) => {",
            "    const s = String(name || 'metric');",
            "    return s.length <= 52 ? s : (s.slice(0, 49) + '...');",
            "  };",
            "  const fmtVal = (v) => {",
            "    const x = Number(v);",
            "    if (!Number.isFinite(x)) return 'nan';",
            "    const ax = Math.abs(x);",
            "    if (ax >= 1e6 || (ax > 0 && ax < 1e-3)) return x.toExponential(2);",
            "    return x.toFixed(3);",
            "  };",
            "  const stepSampleAt = (pts, t) => {",
            "    if (!Array.isArray(pts) || pts.length === 0) return null;",
            "    const tt = Number(t);",
            "    if (!Number.isFinite(tt)) return null;",
            "    let lo = 0;",
            "    let hi = pts.length;",
            "    while (lo < hi) {",
            "      const mid = (lo + hi) >> 1;",
            "      if (Number(pts[mid][0]) <= tt) lo = mid + 1;",
            "      else hi = mid;",
            "    }",
            "    const idx = Math.max(0, lo - 1);",
            "    const ts0 = Number(pts[idx][0]);",
            "    const value = Number(pts[idx][1]);",
            "    if (!Number.isFinite(ts0) || !Number.isFinite(value)) return null;",
            "    const ts1 = (idx + 1 < pts.length) ? Number(pts[idx + 1][0]) : null;",
            "    return {value:value, start_ns:ts0, end_ns:Number.isFinite(ts1) ? ts1 : null};",
            "  };",
            "  const buildKernelOccSumSeries = (streamRows) => {",
            "    const events = new Map();",
            "    const addEvent = (ts, delta) => {",
            "      const t = Number(ts);",
            "      const d = Number(delta);",
            "      if (!Number.isFinite(t) || !Number.isFinite(d)) return;",
            "      events.set(t, Number(events.get(t) || 0) + d);",
            "    };",
            "    for (const srow of (Array.isArray(streamRows) ? streamRows : [])) {",
            "      const kernels = Array.isArray(srow.kernels) ? srow.kernels : [];",
            "      for (const k of kernels) {",
            "        const s = Number(k.start_ns);",
            "        const e = Number(k.end_ns);",
            "        const occRaw = k.occupancy_pct_estimate;",
            "        if (occRaw === null || occRaw === undefined || occRaw === '') continue;",
            "        const occ = Number(occRaw);",
            "        if (!Number.isFinite(s) || !Number.isFinite(e) || !(e > s) || !Number.isFinite(occ)) continue;",
            "        addEvent(s, occ);",
            "        addEvent(e, -occ);",
            "      }",
            "    }",
            "    const times = [...events.keys()].sort((a,b) => a - b);",
            "    const points = [];",
            "    let running = 0.0;",
            "    const wndStart = Number(startNs);",
            "    const wndEnd = Number(startNs + span);",
            "    if (times.length > 0 && wndStart < times[0]) points.push([wndStart, 0.0]);",
            "    for (const t of times) {",
            "      running += Number(events.get(t) || 0);",
            "      if (Math.abs(running) < 1e-9) running = 0.0;",
            "      points.push([t, running]);",
            "    }",
            "    if (times.length > 0 && times[times.length - 1] < wndEnd) points.push([wndEnd, running]);",
            "    return {",
            "      name: 'Kernel Theoretical Occupancy Sum [%]',",
            "      color: '#7fd96b',",
            "      points: points,",
            "      metric_kind: 'kernel_occ_sum'",
            "    };",
            "  };",
            "  for (const g of groups) {",
                "    const panel = document.createElement('div');",
                "    panel.className = 'allstream-panel';",
            "    const title = document.createElement('div');",
            "    title.className = 'allstream-title';",
            "    const rankText = (g.rank === null || g.rank === undefined) ? 'Rank Unknown' : `Rank ${g.rank}`;",
            "    const devText = Number(g.device_id) >= 0 ? `Device ${g.device_id}` : 'Device Unknown';",
            "    title.textContent = `${rankText} | ${devText}`;",
            "    panel.appendChild(title);",
            "",
            "    const streamRows = Array.isArray(g.streams) ? g.streams : [];",
            "    if (!streamRows.length) {",
            "      const empty = document.createElement('div');",
            "      empty.className = 'empty';",
            "      empty.textContent = 'No stream rows in this group.';",
            "      panel.appendChild(empty);",
            "      root.appendChild(panel);",
            "      continue;",
            "    }",
            "",
            "    let selectedMetrics = [];",
            "    if (overlayCount > 0 && metricsAll.length > 0) {",
            "      const dev = Number(g.device_id);",
            "      const exact = metricsAll.filter((s) => parseGpuTag(s.name) === dev);",
            "      selectedMetrics = (exact.length > 0 ? exact : metricsAll).slice();",
            "      selectedMetrics.sort((a,b) => metricRange(b) - metricRange(a));",
            "      if (overlayCount > 0) {",
            "        let trimmed = selectedMetrics.slice(0, overlayCount);",
            "        if (overlayCount >= 2) {",
            "          const computePreferred = selectedMetrics.find((s) => isComputeWarpsSeries(s.name));",
            "          const unallocPreferred = selectedMetrics.find((s) => isUnallocatedWarpsSeries(s.name));",
            "          if (computePreferred && unallocPreferred) {",
            "            trimmed = [computePreferred, unallocPreferred, ...trimmed.filter((s) => s !== computePreferred && s !== unallocPreferred)].slice(0, overlayCount);",
            "          }",
            "        }",
            "        selectedMetrics = trimmed;",
            "      } else {",
            "        selectedMetrics = [];",
            "      }",
            "    }",
            "",
            "    const occSumSeries = buildKernelOccSumSeries(streamRows);",
            "    const metricState = [];",
            "    if (Array.isArray(occSumSeries.points) && occSumSeries.points.length > 0) {",
            "      metricState.push({series:occSumSeries, offset:0, visible:true});",
            "    }",
            "    for (const s of selectedMetrics) metricState.push({series:s, offset:0, visible:true});",
            "    for (const st of metricState) st._pts = toNumericPoints(st.series);",
            "    if (metricState.length > 0) {",
                "      const ctrls = document.createElement('div');",
                "      ctrls.className = 'allstream-controls';",
            "      panel.appendChild(ctrls);",
            "      for (const st of metricState) {",
            "        const row = document.createElement('div');",
            "        row.className = 'shift-row';",
            "        const sw = document.createElement('span');",
            "        sw.style.width = '10px';",
            "        sw.style.height = '10px';",
            "        sw.style.borderRadius = '2px';",
            "        sw.style.display = 'inline-block';",
            "        sw.style.background = String(st.series.color || '#88c');",
            "        row.appendChild(sw);",
            "        const lbl = document.createElement('span');",
            "        lbl.textContent = shortName(st.series.name);",
            "        row.appendChild(lbl);",
            "        const dy = document.createElement('span');",
            "        dy.textContent = 'dy=0';",
            "        row.appendChild(dy);",
            "        const mkBtn = (text) => {",
            "          const b = document.createElement('button');",
            "          b.type = 'button';",
            "          b.className = 'shift-btn';",
            "          b.textContent = text;",
            "          return b;",
            "        };",
            "        const up = mkBtn('up');",
            "        const down = mkBtn('down');",
            "        const reset = mkBtn('reset');",
            "        const toggle = mkBtn('hide');",
            "        row.appendChild(up);",
            "        row.appendChild(down);",
            "        row.appendChild(reset);",
            "        row.appendChild(toggle);",
            "        ctrls.appendChild(row);",
            "        st._dyEl = dy;",
            "        st._toggleEl = toggle;",
            "        st._btnUp = up;",
            "        st._btnDown = down;",
            "        st._btnReset = reset;",
            "      }",
            "    }",
            "",
            "    const scroll = document.createElement('div');",
            "    scroll.className = 'allstream-scroll';",
            "    panel.appendChild(scroll);",
            "    const laneH = 18;",
            "    const laneGap = 6;",
            "    const labelW = 88;",
            "    const rightPad = 18;",
            "    const metricBandH = metricState.length > 0 ? 132 : 0;",
            "    const lanesH = streamRows.length * (laneH + laneGap);",
            "    const svgH = metricBandH + lanesH + 38;",
            "    const svgW = labelW + basePlotW + rightPad;",
            "    const svg = mkSvg('svg', {class:'allstream-svg', width:svgW, height:svgH, viewBox:`0 0 ${svgW} ${svgH}`});",
            "    scroll.appendChild(svg);",
            "    const x = (t) => labelW + ((Number(t) - startNs) / span) * basePlotW;",
            "    const yMetricTop = 8;",
            "    const yMetricBottom = metricBandH > 0 ? (metricBandH - 14) : 0;",
            "    const hoverTip = document.createElement('div');",
            "    hoverTip.className = 'allstream-hover-tip';",
            "    hoverTip.style.display = 'none';",
            "    panel.appendChild(hoverTip);",
            "    let hoverNs = null;",
            "    let hoverLine = null;",
            "    const updateHoverLine = () => {",
            "      if (!hoverLine || hoverNs === null || metricBandH <= 0) {",
            "        if (hoverLine) hoverLine.setAttribute('visibility', 'hidden');",
            "        return;",
            "      }",
            "      const hx = Math.max(labelW, Math.min(labelW + basePlotW, x(hoverNs)));",
            "      hoverLine.setAttribute('x1', String(hx));",
            "      hoverLine.setAttribute('x2', String(hx));",
            "      hoverLine.setAttribute('y1', String(yMetricTop));",
            "      hoverLine.setAttribute('y2', String(yMetricBottom));",
            "      hoverLine.setAttribute('visibility', 'visible');",
            "    };",
            "    const hideHover = () => {",
            "      hoverNs = null;",
            "      hoverTip.style.display = 'none';",
            "      if (hoverLine) hoverLine.setAttribute('visibility', 'hidden');",
            "    };",
            "    const updateHoverAtEvent = (ev) => {",
            "      if (metricBandH <= 0) return;",
            "      const rect = svg.getBoundingClientRect();",
            "      const px = Number(ev.clientX) - rect.left;",
            "      const py = Number(ev.clientY) - rect.top;",
            "      if (!Number.isFinite(px) || !Number.isFinite(py) || py < 2 || py > (metricBandH - 2)) {",
            "        hideHover();",
            "        return;",
            "      }",
            "      const plotX = Math.max(labelW, Math.min(labelW + basePlotW, px));",
            "      hoverNs = startNs + ((plotX - labelW) / basePlotW) * span;",
            "      updateHoverLine();",
            "      const visibleMetricState = metricState.filter((st) => st.visible);",
            "      if (!visibleMetricState.length) {",
            "        hideHover();",
            "        return;",
            "      }",
            "      const rows = [];",
            "      for (const st of visibleMetricState) {",
            "        const sample = stepSampleAt(st._pts, hoverNs);",
            "        if (!sample) continue;",
            "        const endText = sample.end_ns === null ? 'end' : String(Math.round(sample.end_ns));",
            "        rows.push(`<div class='allstream-hover-row'><span class='allstream-hover-swatch' style='background:${String(st.series.color || '#88c')}'></span><span class='allstream-hover-name'>${shortName(st.series.name)}</span><span class='allstream-hover-val'>${fmtVal(sample.value)} @ [${Math.round(sample.start_ns)}, ${endText})</span></div>`);",
            "      }",
            "      if (!rows.length) {",
            "        hideHover();",
            "        return;",
            "      }",
            "      hoverTip.innerHTML = `<div class='allstream-hover-head'>t=${Math.round(hoverNs)} ns</div>${rows.join('')}`;",
            "      hoverTip.style.display = 'block';",
            "      const panelRect = panel.getBoundingClientRect();",
            "      let tx = Number(ev.clientX) - panelRect.left + 12;",
            "      let ty = Number(ev.clientY) - panelRect.top + 12;",
            "      const maxX = Math.max(6, panel.clientWidth - hoverTip.offsetWidth - 6);",
            "      const maxY = Math.max(6, panel.clientHeight - hoverTip.offsetHeight - 6);",
            "      if (tx > maxX) tx = maxX;",
            "      if (ty > maxY) ty = maxY;",
            "      hoverTip.style.left = `${Math.round(tx)}px`;",
            "      hoverTip.style.top = `${Math.round(ty)}px`;",
            "    };",
            "",
            "    const render = () => {",
                "      while (svg.firstChild) svg.removeChild(svg.firstChild);",
            "      svg.appendChild(mkSvg('rect', {x:0, y:0, width:svgW, height:svgH, fill:'#0f1522'}));",
            "      if (metricBandH > 0) {",
            "        svg.appendChild(mkSvg('rect', {x:labelW, y:2, width:basePlotW, height:metricBandH-4, fill:'#111a2b', stroke:'#30405c'}));",
            "      }",
            "      const yStreamsTop = metricBandH + 10;",
            "      svg.appendChild(mkSvg('rect', {x:labelW, y:yStreamsTop-2, width:basePlotW, height:lanesH+8, fill:'#131c2a', stroke:'#30405c'}));",
            "      svg.appendChild(mkSvg('line', {x1:labelW, y1:yStreamsTop-2, x2:labelW+basePlotW, y2:yStreamsTop-2, stroke:'#55637f'}));",
            "      const tStart = mkSvg('text', {x:labelW, y:svgH-6, fill:'#7f8daa', 'font-size':10});",
            "      tStart.textContent = String(Math.round(startNs));",
            "      svg.appendChild(tStart);",
            "      const tEnd = mkSvg('text', {x:labelW+basePlotW-95, y:svgH-6, fill:'#7f8daa', 'font-size':10});",
            "      tEnd.textContent = String(Math.round(startNs + span));",
            "      svg.appendChild(tEnd);",
            "",
            "      if (metricBandH > 0) {",
            "        const visibleMetricState = metricState.filter((st) => st.visible);",
            "        const computeState = visibleMetricState.find((st) => isComputeWarpsSeries(st.series && st.series.name));",
            "        const unallocState = visibleMetricState.find((st) => isUnallocatedWarpsSeries(st.series && st.series.name));",
            "        const handled = new Set();",
            "        const clampPct = (v) => Math.max(0, Math.min(100, Number(v) || 0));",
            "        const pairOffset = (computeState && unallocState) ? (Number(computeState.offset || 0) + Number(unallocState.offset || 0)) : 0;",
            "        const yPct = (vPct) => yMetricBottom - (clampPct(vPct) / 100.0) * (yMetricBottom - yMetricTop) + pairOffset;",
            "        const drawArea = (topPts, basePts, color, opacity) => {",
            "          if (topPts.length < 2 || basePts.length < 2) return;",
            "          const poly = [];",
            "          for (const p of topPts) poly.push(`${p[0].toFixed(2)},${p[1].toFixed(2)}`);",
            "          for (let i = basePts.length - 1; i >= 0; i--) poly.push(`${basePts[i][0].toFixed(2)},${basePts[i][1].toFixed(2)}`);",
            "          svg.appendChild(mkSvg('polygon', {points:poly.join(' '), fill:String(color || '#88c'), opacity:String(opacity)}));",
            "        };",
            "        if (computeState && unallocState) {",
            "          const cPts = computeState._pts;",
            "          const uPts = unallocState._pts;",
            "          if (cPts.length >= 2 && uPts.length >= 2) {",
            "            const cTop = [];",
            "            const cBase = [];",
            "            for (const p of cPts) {",
            "              const tx = x(p[0]);",
            "              if (!Number.isFinite(tx)) continue;",
            "              cTop.push([tx, yPct(p[1])]);",
            "              cBase.push([tx, yMetricBottom + pairOffset]);",
            "            }",
            "            drawArea(cTop, cBase, String(computeState.series.color || '#4fa5ff'), 0.42);",
            "            const uTop = [];",
            "            const uBase = [];",
            "            for (const p of uPts) {",
            "              const tx = x(p[0]);",
            "              if (!Number.isFinite(tx)) continue;",
            "              const base = clampPct(interpAt(cPts, p[0]));",
            "              const top = clampPct(base + p[1]);",
            "              uTop.push([tx, yPct(top)]);",
            "              uBase.push([tx, yPct(base)]);",
            "            }",
            "            drawArea(uTop, uBase, String(unallocState.series.color || '#ffae57'), 0.58);",
            "            const cLine = cTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "            if (cLine) svg.appendChild(mkSvg('polyline', {points:cLine, fill:'none', stroke:String(computeState.series.color || '#4fa5ff'), 'stroke-width':1.1, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "            const uLine = uTop.map((p) => `${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');",
            "            if (uLine) svg.appendChild(mkSvg('polyline', {points:uLine, fill:'none', stroke:String(unallocState.series.color || '#ffae57'), 'stroke-width':1.1, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "            handled.add(computeState);",
            "            handled.add(unallocState);",
            "          }",
            "        }",
            "        for (const st of visibleMetricState) {",
            "          if (handled.has(st)) continue;",
            "          const ptsRaw = st._pts;",
            "          if (!ptsRaw.length) continue;",
            "          let minV = Number.POSITIVE_INFINITY;",
            "          let maxV = Number.NEGATIVE_INFINITY;",
            "          for (const p of ptsRaw) {",
            "            const v = Number(p[1]);",
            "            if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }",
            "          }",
            "          if (!Number.isFinite(minV) || !Number.isFinite(maxV)) continue;",
            "          if (Math.abs(maxV - minV) < 1e-12) maxV = minV + 1.0;",
            "          const points = [];",
            "          for (const p of ptsRaw) {",
            "            const tx = x(p[0]);",
            "            const v = Number(p[1]);",
            "            if (!Number.isFinite(tx) || !Number.isFinite(v)) continue;",
            "            const tyNorm = yMetricTop + (1.0 - ((v - minV) / (maxV - minV))) * (yMetricBottom - yMetricTop);",
            "            const ty = tyNorm + st.offset;",
            "            points.push(`${tx.toFixed(2)},${ty.toFixed(2)}`);",
            "          }",
            "          if (points.length >= 2) {",
            "            svg.appendChild(mkSvg('polyline', {points:points.join(' '), fill:'none', stroke:String(st.series.color || '#88c'), 'stroke-width':1.2, 'stroke-linejoin':'round', 'stroke-linecap':'round'}));",
            "          }",
            "        }",
            "      }",
            "",
            "      for (let i = 0; i < streamRows.length; i++) {",
            "        const srow = streamRows[i];",
            "        const laneY = yStreamsTop + i * (laneH + laneGap);",
            "        const lbl = mkSvg('text', {x:4, y:laneY + 13, fill:'#98a9c5', 'font-size':11});",
            "        lbl.textContent = `stream ${srow.stream_id}`;",
            "        svg.appendChild(lbl);",
            "        svg.appendChild(mkSvg('line', {x1:labelW, y1:laneY + laneH + 1, x2:labelW + basePlotW, y2:laneY + laneH + 1, stroke:'#2d3a52'}));",
            "        const kernels = Array.isArray(srow.kernels) ? srow.kernels : [];",
            "        for (const k of kernels) {",
            "          const s = Number(k.start_ns || 0);",
            "          const e = Number(k.end_ns || 0);",
            "          if (!(e > s)) continue;",
            "          const left = x(s);",
            "          const width = Math.max(1, ((e - s) / span) * basePlotW);",
            "          const rect = mkSvg('rect', {x:left, y:laneY, width:width, height:laneH, rx:2, ry:2, fill:String(k.color || '#6699cc'), opacity:0.92});",
            "          const ttl = mkSvg('title', {});",
            "          ttl.textContent = `${k.kernel_name || ''} | stream=${srow.stream_id} | start_ns=${s} | end_ns=${e} | kind=${k.kind || ''} | occ_theoretical_pct=${k.occupancy_pct_estimate}`;",
            "          rect.appendChild(ttl);",
            "          svg.appendChild(rect);",
            "        }",
            "      }",
            "      if (metricBandH > 0) {",
            "        hoverLine = mkSvg('line', {x1:labelW, y1:yMetricTop, x2:labelW, y2:yMetricBottom, stroke:'#dbe6ff', 'stroke-width':1, opacity:0.92, visibility:'hidden'});",
            "        svg.appendChild(hoverLine);",
            "        updateHoverLine();",
            "      } else {",
            "        hoverLine = null;",
            "        hoverTip.style.display = 'none';",
            "      }",
            "    };",
            "",
            "    if (metricBandH > 0) {",
            "      svg.addEventListener('mousemove', updateHoverAtEvent);",
            "      svg.addEventListener('mouseleave', hideHover);",
            "    }",
            "    for (const st of metricState) {",
                "      if (st._btnUp) st._btnUp.onclick = () => { st.offset -= 10; st._dyEl.textContent = `dy=${st.offset}`; render(); };",
            "      if (st._btnDown) st._btnDown.onclick = () => { st.offset += 10; st._dyEl.textContent = `dy=${st.offset}`; render(); };",
            "      if (st._btnReset) st._btnReset.onclick = () => { st.offset = 0; st.visible = true; st._dyEl.textContent = 'dy=0'; if (st._toggleEl) st._toggleEl.textContent = 'hide'; render(); };",
            "      if (st._toggleEl) st._toggleEl.onclick = () => { st.visible = !st.visible; st._toggleEl.textContent = st.visible ? 'hide' : 'show'; render(); };",
            "    }",
            "    render();",
            "    root.appendChild(panel);",
            "  }",
            "})();",
            "</script>",
            "<script>",
            "(function(){",
            "  const d = TIMELINE_DATA || {};",
            "  const fmt = (v, digits=3) => {",
            "    const x = Number(v);",
            "    if (!Number.isFinite(x)) return 'n/a';",
            "    return x.toFixed(Number(digits));",
            "  };",
            "  const mk = (tag, attrs={}) => {",
            "    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);",
            "    for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v));",
            "    return el;",
            "  };",
            "",
            "  const rankRows = Array.isArray(d.rank_heatmap_rows) ? d.rank_heatmap_rows : [];",
            "  const rankRoot = document.getElementById('rank-heatmap-root');",
            "  if (rankRoot) {",
            "    if (!rankRows.length) {",
            "      rankRoot.innerHTML = \"<div class='empty'>No rank-attributed kernels in this window.</div>\";",
            "    } else {",
            "      const maxTotal = Math.max(...rankRows.map((r) => Number(r.total_ms || 0)), 1);",
            "      const table = document.createElement('table'); table.className = 'heatmap-table';",
            "      table.innerHTML = \"<thead><tr><th>rank</th><th>device</th><th>total ms</th><th>compute ms</th><th>comm ms</th><th>kernels</th><th>streams</th><th>avg occ %</th></tr></thead>\";",
            "      const body = document.createElement('tbody');",
            "      for (const r of rankRows) {",
            "        const tr = document.createElement('tr');",
            "        const rankText = (r.rank === null || r.rank === undefined) ? 'unknown' : String(r.rank);",
            "        const heat = Math.max(0, Math.min(1, Number(r.total_ms || 0) / maxTotal));",
            "        const bg = `rgba(83, 158, 255, ${0.15 + 0.55 * heat})`;",
            "        const occ = (r.avg_occupancy_pct === null || r.avg_occupancy_pct === undefined) ? 'n/a' : fmt(r.avg_occupancy_pct, 2);",
            "        tr.innerHTML = [",
            "          `<td><code>${rankText}</code></td>`,",
            "          `<td><code>${Number((r.device_id === null || r.device_id === undefined) ? -1 : r.device_id)}</code></td>`,",
            "          `<td><span class='heat-cell' style='background:${bg}'>${fmt(r.total_ms,3)}</span></td>`,",
            "          `<td>${fmt(r.compute_ms,3)}</td>`,",
            "          `<td>${fmt(r.comm_ms,3)}</td>`,",
            "          `<td>${Number(r.kernel_count || 0)}</td>`,",
            "          `<td>${Number(r.stream_count || 0)}</td>`,",
            "          `<td>${occ}</td>`,",
            "        ].join('');",
            "        body.appendChild(tr);",
            "      }",
            "      table.appendChild(body);",
            "      rankRoot.appendChild(table);",
            "    }",
            "  }",
            "",
            "  const roofRoot = document.getElementById('roofline-root');",
            "  const roof = (d.roofline_proxy && typeof d.roofline_proxy === 'object') ? d.roofline_proxy : {};",
            "  const roofPoints = Array.isArray(roof.points) ? roof.points : [];",
            "  const roofStats = Array.isArray(roof.stats) ? roof.stats : [];",
            "  if (roofRoot) {",
            "    if (!roofPoints.length) {",
            "      roofRoot.innerHTML = \"<div class='empty'>Roofline proxy requires both compute-activity and memory-throughput metric series.</div>\";",
            "    } else {",
            "      const wrap = document.createElement('div'); wrap.className = 'roofline-wrap';",
            "      const W = 780, H = 300, pad = 36;",
            "      const svg = mk('svg', {width:W, height:H, viewBox:`0 0 ${W} ${H}`});",
            "      const x0 = pad, y0 = pad, cw = W - pad * 2, ch = H - pad * 2;",
            "      const x = (v) => x0 + Math.max(0, Math.min(100, Number(v || 0))) / 100 * cw;",
            "      const y = (v) => y0 + (1 - Math.max(0, Math.min(100, Number(v || 0))) / 100) * ch;",
            "      svg.appendChild(mk('rect', {x:x0, y:y0, width:cw, height:ch, fill:'#111827', stroke:'#32405b'}));",
            "      svg.appendChild(mk('line', {x1:x0, y1:y0+ch, x2:x0+cw, y2:y0+ch, stroke:'#5f6f89'}));",
            "      svg.appendChild(mk('line', {x1:x0, y1:y0, x2:x0, y2:y0+ch, stroke:'#5f6f89'}));",
            "      const diag = mk('line', {x1:x0, y1:y0+ch, x2:x0+cw, y2:y0, stroke:'#4d9cff', 'stroke-dasharray':'4 4', opacity:'0.6'});",
            "      svg.appendChild(diag);",
            "      for (let t = 0; t <= 100; t += 25) {",
            "        const tx = x(t);",
            "        const ty = y(t);",
            "        svg.appendChild(mk('line', {x1:tx, y1:y0+ch, x2:tx, y2:y0+ch+4, stroke:'#6e7d98'}));",
            "        const lx = mk('text', {x:tx-8, y:y0+ch+16, fill:'#97a7c2', 'font-size':10}); lx.textContent = String(t); svg.appendChild(lx);",
            "        svg.appendChild(mk('line', {x1:x0-4, y1:ty, x2:x0, y2:ty, stroke:'#6e7d98'}));",
            "        const ly = mk('text', {x:4, y:ty+3, fill:'#97a7c2', 'font-size':10}); ly.textContent = String(t); svg.appendChild(ly);",
            "      }",
            "      const devColors = ['#63b3ff','#7dd3fc','#a3e635','#fbbf24','#fb7185','#c4b5fd'];",
            "      const colorByDev = (dev) => devColors[Math.abs(Number(dev || 0)) % devColors.length];",
            "      for (const p of roofPoints) {",
            "        const cx = x(p.x_mem_pct);",
            "        const cy = y(p.y_compute_pct);",
            "        const c = mk('circle', {cx, cy, r:2.5, fill:colorByDev(p.device_id), opacity:'0.72'});",
            "        const tt = mk('title', {});",
            "        tt.textContent = `gpu=${p.device_id} mem=${fmt(p.x_mem_pct,2)}% compute=${fmt(p.y_compute_pct,2)}% ts=${Math.round(Number(p.ts_ns || 0))}`;",
            "        c.appendChild(tt);",
            "        svg.appendChild(c);",
            "      }",
            "      const xLab = mk('text', {x:x0+cw/2-70, y:H-6, fill:'#9fb0ca', 'font-size':11}); xLab.textContent = 'Memory Throughput (%)'; svg.appendChild(xLab);",
            "      const yLab = mk('text', {x:8, y:14, fill:'#9fb0ca', 'font-size':11}); yLab.textContent = 'Compute Activity (%)'; svg.appendChild(yLab);",
            "      wrap.appendChild(svg);",
            "      const legend = document.createElement('div'); legend.className = 'roofline-legend';",
            "      if (roofStats.length) {",
            "        for (const row of roofStats) {",
            "          const div = document.createElement('div');",
            "          div.textContent = `gpu ${row.device_id}: avg_mem=${fmt(row.avg_mem_pct,2)}% avg_compute=${fmt(row.avg_compute_pct,2)}% gap=${fmt(row.gap_pct,2)}`;",
            "          div.style.color = colorByDev(row.device_id);",
            "          legend.appendChild(div);",
            "        }",
            "      } else {",
            "        legend.textContent = 'No per-device roofline stats';",
            "      }",
            "      wrap.appendChild(legend);",
            "      roofRoot.appendChild(wrap);",
            "    }",
            "  }",
            "",
            "  const gilRoot = document.getElementById('gil-lane-root');",
            "  const gilSeries = Array.isArray(d.gil_lane_series) ? d.gil_lane_series : [];",
            "  if (gilRoot) {",
            "    if (!gilSeries.length) {",
            "      gilRoot.innerHTML = \"<div class='empty'>No GIL/thread-utilization series detected.</div>\";",
            "    } else {",
            "      const grid = document.createElement('div'); grid.className = 'gil-lane-grid';",
            "      for (const s of gilSeries) {",
            "        const card = document.createElement('div'); card.className = 'gil-lane-item';",
            "        const title = document.createElement('div'); title.className = 'gil-lane-title'; title.textContent = String(s.name || 'series');",
            "        const W = 760, H = 90, padL = 38, padR = 8, padT = 10, padB = 20;",
            "        const svg = mk('svg', {width:W, height:H, viewBox:`0 0 ${W} ${H}`});",
            "        svg.appendChild(mk('rect', {x:padL, y:padT, width:W-padL-padR, height:H-padT-padB, fill:'#111725', stroke:'#32405b'}));",
            "        const ptsRaw = Array.isArray(s.points) ? s.points : [];",
            "        let minV = Number.POSITIVE_INFINITY, maxV = Number.NEGATIVE_INFINITY;",
            "        for (const p of ptsRaw) { const v = Number(p[1]); if (Number.isFinite(v)) { minV = Math.min(minV, v); maxV = Math.max(maxV, v);} }",
            "        if (!Number.isFinite(minV) || !Number.isFinite(maxV)) { minV = 0; maxV = 1; }",
            "        if (Math.abs(maxV - minV) < 1e-12) maxV = minV + 1.0;",
            "        const x0 = padL, y0 = padT, cw = W-padL-padR, ch = H-padT-padB;",
            "        const x = (i) => x0 + (i / Math.max(1, ptsRaw.length - 1)) * cw;",
            "        const y = (v) => y0 + (1 - ((Number(v)-minV)/(maxV-minV))) * ch;",
            "        const pts = [];",
            "        for (let i = 0; i < ptsRaw.length; i++) {",
            "          const v = Number(ptsRaw[i][1]);",
            "          if (!Number.isFinite(v)) continue;",
            "          pts.push(`${x(i).toFixed(2)},${y(v).toFixed(2)}`);",
            "        }",
            "        if (pts.length >= 2) svg.appendChild(mk('polyline', {points:pts.join(' '), fill:'none', stroke:String(s.color || '#7aa2ff'), 'stroke-width':1.4}));",
            "        const minTxt = mk('text', {x:4, y:y0+ch, fill:'#9aa7be', 'font-size':10}); minTxt.textContent = fmt(minV,2); svg.appendChild(minTxt);",
            "        const maxTxt = mk('text', {x:4, y:y0+10, fill:'#9aa7be', 'font-size':10}); maxTxt.textContent = fmt(maxV,2); svg.appendChild(maxTxt);",
            "        const sub = document.createElement('div'); sub.className = 'gil-lane-sub';",
            "        sub.textContent = `samples=${Number(s.samples || ptsRaw.length)} avg=${fmt(s.avg,3)} min=${fmt(s.min,3)} max=${fmt(s.max,3)}`;",
            "        card.appendChild(title); card.appendChild(svg); card.appendChild(sub);",
            "        grid.appendChild(card);",
            "      }",
            "      gilRoot.appendChild(grid);",
            "    }",
            "  }",
            "})();",
            "</script>",
            "</body></html>",
        ]
    )
    return "\n".join(lines)


def _collect_timeline_state(
    sqlite_path: str,
    *,
    output_path: str = "",
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 100000,
    nvtx_text: str = "",
    nvtx_index: int = -1,
    include_metrics: bool = False,
    metric_name_like: str = "%",
    metrics_limit: int = -1,
    metrics_max_points: int = -1,
    kernel_category_rules: Optional[Sequence[Tuple[Pattern[str], str]]] = None,
    kernel_category_profile: str = "",
    enable_kernel_category_breakdown: bool = True,
    default_focus_metrics: bool = True,
    include_all_metric_sources: bool = False,
    debug: bool = False,
    debug_rows: int = -1,
    debug_log_fn: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    debug_log = _build_debug_logger(enabled=bool(debug), log_fn=debug_log_fn)

    _phase_timings: List[Dict[str, float]] = []

    def _emit_phase(name: str, elapsed_ms: float) -> None:
        _phase_timings.append({"phase": name, "elapsed_ms": elapsed_ms})
        if progress_cb:
            progress_cb(f"  done:     {name}  [{elapsed_ms} ms]")
    try:
        debug_rows_i = int(debug_rows)
    except Exception:
        debug_rows_i = -1
    debug_log(
        "start sqlite={} output={} device_id={} start_ns={} end_ns={} nvtx_text={} nvtx_index={} include_metrics={} metrics_limit={} metrics_max_points={} overlay_metrics_per_track={} default_focus_metrics={} debug_rows={}".format(
            str(sqlite_path),
            str(output_path),
            int(device_id),
            int(start_ns),
            int(end_ns),
            str(nvtx_text or ""),
            int(nvtx_index),
            int(bool(include_metrics)),
            int(metrics_limit),
            int(metrics_max_points),
            0,
            int(bool(default_focus_metrics)),
            int(debug_rows_i),
        )
    )
    _t0 = time.perf_counter()
    try:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            schema = NsightSchema(conn)
            debug_log(
                "schema tables kernel={} runtime={} nvtx={} metrics={} string={}".format(
                    schema.kernel_table,
                    schema.runtime_table,
                    schema.nvtx_table,
                    schema.metrics_table,
                    schema.string_table,
                )
            )
            debug_log(
                "metrics columns timestamp={} id={} value={}".format(
                    schema.metrics_timestamp_col,
                    schema.metrics_id_col,
                    schema.metrics_value_col,
                )
            )
            for table_name in (
                schema.kernel_table,
                schema.runtime_table,
                schema.nvtx_table,
                schema.metrics_table,
                "TARGET_INFO_GPU_METRICS",
                "GENERIC_EVENT_SOURCES",
            ):
                if not table_name:
                    continue
                count = _safe_table_count(conn, table_name)
                debug_log("table count {}={}".format(table_name, "n/a" if count is None else count))
        finally:
            conn.close()
    except Exception as exc:
        debug_log("schema probe failed: {}".format(exc))
    _emit_phase("schema_probe", round((time.perf_counter() - _t0) * 1000, 1))

    provider = NsysSqliteMetricsProvider(sqlite_path)

    _t0 = time.perf_counter()
    selected_nvtx_windows: List[Dict[str, object]] = []
    effective_start_ns = int(start_ns)
    effective_end_ns = int(end_ns)
    nvtx_pattern = _normalize_nvtx_like_pattern(nvtx_text)
    if str(nvtx_text or "").strip():
        debug_log(
            "nvtx text filter raw={} normalized_like={}".format(
                str(nvtx_text or ""),
                str(nvtx_pattern),
            )
        )
        matched_nvtx = _select_nvtx_windows(
            provider,
            nvtx_text=str(nvtx_pattern),
        )
        selected_nvtx_windows = _pick_nvtx_windows(matched_nvtx, nvtx_index=int(nvtx_index))
        debug_log(
            "nvtx match count={} selected_count={}".format(
                len(matched_nvtx),
                len(selected_nvtx_windows),
            )
        )
        if selected_nvtx_windows:
            debug_log(
                "selected nvtx sample={}".format(
                    _preview_dict_rows(
                        selected_nvtx_windows,
                        keys=("nvtx_text", "start_ns", "end_ns", "duration_ms"),
                        limit=debug_rows_i,
                    )
                )
            )
        if selected_nvtx_windows:
            effective_start_ns = min(_to_int(item.get("start_ns"), -1) for item in selected_nvtx_windows)
            effective_end_ns = max(_to_int(item.get("end_ns"), -1) for item in selected_nvtx_windows)
            debug_log(
                "effective window from nvtx start_ns={} end_ns={}".format(
                    int(effective_start_ns),
                    int(effective_end_ns),
                )
            )

    _emit_phase("nvtx_window_selection", round((time.perf_counter() - _t0) * 1000, 1))

    _t0 = time.perf_counter()
    if effective_start_ns < 0 or effective_end_ns < 0 or effective_end_ns <= effective_start_ns:
        # If no explicit valid window, infer from full kernel rows.
        base_rows = collect_kernel_rows(
            provider,
            device_id=int(device_id),
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            limit=int(limit),
            attach_iteration=False,
        )
        debug_log("infer window from kernel_map rows={}".format(len(base_rows)))
        if not base_rows:
            render_start_ns = max(0, int(start_ns))
            render_end_ns = int(end_ns) if int(end_ns) > int(render_start_ns) else int(render_start_ns) + 1
            debug_log(
                "no kernels found; using fallback render window=[{}, {}]".format(
                    int(render_start_ns),
                    int(render_end_ns),
                )
            )
            return {
                "sqlite_path": str(sqlite_path),
                "kernels": [],
                "metric_series": [],
                "kernel_category_summary": {
                    "rows": [],
                    "wall_ms": 0.0,
                    "raw_total_ms": 0.0,
                    "non_overlap_ms": 0.0,
                    "busy_union_ms": 0.0,
                    "idle_ms": 0.0,
                    "busy_pct_of_wall": 0.0,
                    "cross_category_overlap_ms": 0.0,
                    "overlap_saved_ms": 0.0,
                },
                "kernel_category_kernel_rows": [],
                "kernel_category_profile": str(kernel_category_profile or ""),
                "nvtx_window_category_stats": {"window_count": 0, "windows": [], "category_summary_rows": [], "outlier_windows": []},
                "window_start_ns": int(render_start_ns),
                "window_end_ns": int(render_end_ns),
                "nvtx_windows": selected_nvtx_windows,
                "phase_timings": list(_phase_timings),
                "empty_reason": "No kernels",
            }
        effective_start_ns = min(int(item["start_ns"]) for item in base_rows)
        effective_end_ns = max(int(item["end_ns"]) for item in base_rows)
        debug_log(
            "effective window from kernels start_ns={} end_ns={}".format(
                int(effective_start_ns),
                int(effective_end_ns),
            )
        )
    _emit_phase("window_inference", round((time.perf_counter() - _t0) * 1000, 1))

    _t0 = time.perf_counter()
    # 按照nvtx range内的runtime correlationId到具体的kernel，得到kernel window
    kernels = _collect_kernels_in_window(
        provider,
        start_ns=int(effective_start_ns),
        end_ns=int(effective_end_ns),
        nvtx_text=str(nvtx_pattern),
        nvtx_windows=selected_nvtx_windows or None,
        device_id=int(device_id),
        limit=int(limit),
        debug_log=debug_log,
        debug_rows=debug_rows_i,
    )

    render_start_ns = int(effective_start_ns)
    render_end_ns = int(effective_end_ns)
    kernel_intervals = _merge_intervals(
        [
            (_to_int(row.get("start_ns"), -1), _to_int(row.get("end_ns"), -1))
            for row in kernels
        ]
    )
    if kernel_intervals:
        kernel_span_start = kernel_intervals[0][0]
        kernel_span_end = kernel_intervals[-1][1]
        new_render_start = min(int(render_start_ns), int(kernel_span_start))
        new_render_end = max(int(render_end_ns), int(kernel_span_end))
        if new_render_start != int(render_start_ns) or new_render_end != int(render_end_ns):
            debug_log(
                "extend render window with kernel span start_ns={} end_ns={} (old=[{}, {}], new=[{}, {}])".format(
                    int(kernel_span_start),
                    int(kernel_span_end),
                    int(render_start_ns),
                    int(render_end_ns),
                    int(new_render_start),
                    int(new_render_end),
                )
            )
            render_start_ns = int(new_render_start)
            render_end_ns = int(new_render_end)
    kernel_stream_count = len(
        {
            (_to_int(row.get("device_id"), -1), _to_int(row.get("stream_id"), 0))
            for row in kernels
        }
    )
    debug_log(
        "collect_kernels matched_kernels={} streams={} merged_gpu_intervals={}".format(
            len(kernels),
            kernel_stream_count,
            len(kernel_intervals),
        )
    )
    if progress_cb:
        progress_cb(
            "  info:     matched_kernels={}  streams={}  merged_gpu_intervals={}".format(
                len(kernels),
                kernel_stream_count,
                len(kernel_intervals),
            )
        )

    kernel_category_summary: Dict[str, object] = {
        "rows": [],
        "wall_ms": 0.0,
        "non_overlap_ms": 0.0,
        "busy_union_ms": 0.0,
        "idle_ms": 0.0,
        "busy_pct_of_wall": 0.0,
        "cross_category_overlap_ms": 0.0,
        "raw_total_ms": 0.0,
        "overlap_saved_ms": 0.0,
    }
    kernel_category_kernel_rows: List[Dict[str, object]] = []
    if kernel_category_rules:
        kernel_category_kernel_rows = _build_kernel_category_kernel_table(
            kernels,
            rules=list(kernel_category_rules or []),
        )
        debug_log("kernel-category kernel table rows={}".format(len(kernel_category_kernel_rows)))
    if bool(enable_kernel_category_breakdown):
        kernel_category_summary = _build_kernel_category_breakdown(
            kernels,
            rules=list(kernel_category_rules or []),
            wall_start_ns=int(render_start_ns),
            wall_end_ns=int(render_end_ns),
        )
        debug_log(
            "kernel-category profile={} rows={} non_overlap_ms={:.3f}".format(
                str(kernel_category_profile or ""),
                len(list(kernel_category_summary.get("rows") or [])),
                _safe_float(kernel_category_summary.get("non_overlap_ms")),
            )
        )
    nvtx_window_category_stats: Dict[str, object] = {
        "window_count": 0,
        "windows": [],
        "category_summary_rows": [],
        "outlier_windows": [],
    }
    if selected_nvtx_windows and kernel_category_rules:
        nvtx_window_category_stats = _build_nvtx_window_category_stats(
            nvtx_windows=list(selected_nvtx_windows or []),
            kernels=list(kernels or []),
            rules=list(kernel_category_rules or []),
        )
        debug_log(
            "nvtx-category stats windows={} categories={} outliers={}".format(
                int(nvtx_window_category_stats.get("window_count") or 0),
                int(nvtx_window_category_stats.get("category_count") or 0),
                len(list(nvtx_window_category_stats.get("outlier_windows") or [])),
            )
        )

    metric_series: List[Dict[str, object]] = []
    if bool(include_metrics):
        metric_query_start_ns = int(render_start_ns)
        metric_query_end_ns = int(render_end_ns)
        metric_restrict_intervals: Optional[List[Tuple[int, int]]] = None
        debug_log(
            "metrics query uses render window start_ns={} end_ns={} (no kernel-interval restriction, kernel_intervals={})".format(
                int(metric_query_start_ns),
                int(metric_query_end_ns),
                len(kernel_intervals),
            )
        )
        apply_focus_filter = bool(default_focus_metrics) and str(metric_name_like or "%").strip() in ("", "%")
        debug_log(
            "metrics focus filter active={} metric_name_like={}".format(
                int(apply_focus_filter),
                str(metric_name_like or "%"),
            )
        )
        metric_series = _collect_metric_samples(
            sqlite_path,
            start_ns=int(metric_query_start_ns),
            end_ns=int(metric_query_end_ns),
            metric_name_like=str(metric_name_like or "%"),
            include_all_sources=bool(include_all_metric_sources),
            device_id=int(device_id),
            limit=int(metrics_limit),
            max_points_per_series=int(metrics_max_points),
            apply_default_focus_filter=bool(apply_focus_filter),
            restrict_to_intervals=metric_restrict_intervals,
            debug_log=debug_log,
            debug_rows=debug_rows_i,
        )
    else:
        debug_log("metrics disabled (include_metrics=0)")
    _emit_phase("collect_kernels_and_metrics", round((time.perf_counter() - _t0) * 1000, 1))
    if progress_cb:
        total_ms = round(sum(float(p.get("elapsed_ms") or 0) for p in _phase_timings), 1)
        progress_cb(f"  total:    {total_ms} ms  (phases: {len(_phase_timings)})")
    return {
        "sqlite_path": str(sqlite_path),
        "kernels": kernels,
        "metric_series": metric_series,
        "kernel_category_summary": kernel_category_summary,
        "kernel_category_kernel_rows": kernel_category_kernel_rows,
        "kernel_category_profile": str(kernel_category_profile or ""),
        "nvtx_window_category_stats": nvtx_window_category_stats,
        "window_start_ns": int(render_start_ns),
        "window_end_ns": int(render_end_ns),
        "nvtx_windows": selected_nvtx_windows,
        "phase_timings": list(_phase_timings),
        "empty_reason": "",
    }


def export_timeline_html(
    sqlite_path: str,
    *,
    output_path: str,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 100000,
    width_px: int = 1800,
    nvtx_text: str = "",
    nvtx_index: int = -1,
    include_metrics: bool = False,
    metric_name_like: str = "%",
    metrics_limit: int = -1,
    metrics_max_points: int = -1,
    overlay_metrics_per_track: int = 7,
    display_span_ns: int = -1,
    kernel_category_map_json: str = "",
    kernel_category_engine: str = "sglang",
    kernel_category_model: str = "llama",
    enable_kernel_category_breakdown: bool = True,
    kernel_category_table_output: str = "",
    nvtx_category_stats_output: str = "",
    default_focus_metrics: bool = True,
    include_all_metric_sources: bool = False,
    debug: bool = False,
    debug_rows: int = -1,
    debug_log_fn: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Export an interactive HTML timeline from an nsys SQLite file.

    Parameters
    ----------
    progress_cb:
        Optional callback for per-phase progress messages printed to the caller.
        Each call receives a string like ``"  done: collect_kernels  [234 ms]"``.
        Pass ``lambda msg: print(msg, file=sys.stderr)`` for CLI progress output.
    """
    debug_log = _build_debug_logger(enabled=bool(debug), log_fn=debug_log_fn)
    kernel_rules, kernel_profile = _resolve_kernel_category_rules(
        map_json_path=str(kernel_category_map_json or ""),
        engine=str(kernel_category_engine or ""),
        model=str(kernel_category_model or ""),
        debug_log=debug_log,
    )
    state = _collect_timeline_state(
        sqlite_path,
        output_path=str(output_path),
        device_id=int(device_id),
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        limit=int(limit),
        nvtx_text=str(nvtx_text or ""),
        nvtx_index=int(nvtx_index),
        include_metrics=bool(include_metrics),
        metric_name_like=str(metric_name_like or "%"),
        metrics_limit=int(metrics_limit),
        metrics_max_points=int(metrics_max_points),
        kernel_category_rules=kernel_rules,
        kernel_category_profile=str(kernel_profile),
        enable_kernel_category_breakdown=bool(enable_kernel_category_breakdown),
        default_focus_metrics=bool(default_focus_metrics),
        include_all_metric_sources=bool(include_all_metric_sources),
        debug=bool(debug),
        debug_rows=int(debug_rows),
        debug_log_fn=debug_log_fn,
        progress_cb=progress_cb,
    )
    text = _render_html(
        sqlite_path=str(sqlite_path),
        kernels=list(state.get("kernels") or []),
        metric_series=list(state.get("metric_series") or []),
        kernel_category_summary=dict(state.get("kernel_category_summary") or {}),
        kernel_category_profile=str(state.get("kernel_category_profile") or ""),
        nvtx_window_category_stats=dict(state.get("nvtx_window_category_stats") or {}),
        window_start_ns=int(state.get("window_start_ns") or 0),
        window_end_ns=int(state.get("window_end_ns") or 1),
        display_span_ns=int(display_span_ns),
        nvtx_windows=list(state.get("nvtx_windows") or []),
        width_px=int(width_px),
        include_metrics=bool(include_metrics),
        overlay_metrics_per_track=int(overlay_metrics_per_track),
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    category_table_output = str(kernel_category_table_output or "").strip()
    if category_table_output:
        category_rows = list(state.get("kernel_category_kernel_rows") or [])
        table_path = _write_rows_table(category_table_output, category_rows)
        debug_log("kernel-category table wrote path={} rows={}".format(table_path, len(category_rows)))
        if progress_cb:
            progress_cb("  wrote:    kernel-category-table={} rows={}".format(table_path, len(category_rows)))
    stats_output = str(nvtx_category_stats_output or "").strip()
    if stats_output:
        stats_path = Path(stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_payload = dict(state.get("nvtx_window_category_stats") or {})
        stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        debug_log("nvtx-category stats wrote path={} windows={}".format(str(stats_path), int(stats_payload.get("window_count") or 0)))
        if progress_cb:
            progress_cb("  wrote:    nvtx-category-stats={} windows={}".format(str(stats_path), int(stats_payload.get("window_count") or 0)))
    return str(out)


def _extract_timeline_payload_from_html(html_text: str) -> Optional[Dict[str, object]]:
    m = re.search(r"const TIMELINE_DATA = (\{.*?\});", str(html_text or ""), flags=re.S)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_kernel_name_for_compare(name: object) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip())


def _short_kernel_name(name: object, limit: int = 72) -> str:
    text = _normalize_kernel_name_for_compare(name)
    if len(text) <= int(limit):
        return text
    return text[: max(8, int(limit) - 3)] + "..."


def _segment_duration_ms(segment: Sequence[Dict[str, object]]) -> float:
    total = 0.0
    for item in segment:
        try:
            total += float(item.get("duration_ms") or 0.0)
        except Exception:
            continue
    return total


def _stream_key_sort_key(key: Tuple[Optional[int], int, int]) -> Tuple[int, int, int]:
    rank, dev, sid = key
    return (10**9 if rank is None else int(rank), int(dev), int(sid))


def _kernel_anchor_key(name: object) -> str:
    return _normalize_kernel_name_for_compare(name).lower()


def _kernel_token_set(name: object) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", _kernel_anchor_key(name)):
        if raw.isdigit():
            continue
        token = re.sub(r"\d+", "#", raw)
        if len(token.replace("#", "")) < 2:
            continue
        out.add(token)
    return out


def _kernel_duration_ms_value(item: Dict[str, object]) -> float:
    try:
        value = float(item.get("duration_ms") or 0.0)
    except Exception:
        return 0.0
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return float(value)


def _sequence_time_bounds(segment: Sequence[Dict[str, object]]) -> Tuple[int, int]:
    starts: List[int] = []
    ends: List[int] = []
    for item in segment:
        s = _to_int(item.get("start_ns"), -1)
        e = _to_int(item.get("end_ns"), -1)
        if s >= 0 and e > s:
            starts.append(s)
            ends.append(e)
    if starts and ends:
        return min(starts), max(ends)
    return 0, max(1, len(segment))


def _kernel_center_norm(
    item: Dict[str, object],
    *,
    seq_start_ns: int,
    seq_end_ns: int,
    ordinal_idx: int,
    ordinal_count: int,
) -> float:
    s = _to_int(item.get("start_ns"), -1)
    e = _to_int(item.get("end_ns"), -1)
    span = max(1, int(seq_end_ns) - int(seq_start_ns))
    if s >= 0 and e > s:
        center = 0.5 * (float(s) + float(e))
        return max(0.0, min(1.0, (center - float(seq_start_ns)) / float(span)))
    denom = max(1, int(ordinal_count) - 1)
    return float(int(ordinal_idx) / float(denom))


def _ratio_similarity(lhs: float, rhs: float, *, max_ratio: float) -> float:
    lv = float(lhs)
    rv = float(rhs)
    if lv <= 0.0 or rv <= 0.0:
        return 0.5
    limit = math.log(max(1.000001, float(max_ratio)))
    diff = abs(math.log(lv / rv))
    return max(0.0, 1.0 - min(1.0, diff / limit))


def _kernel_token_overlap(lhs: object, rhs: object) -> float:
    left = _kernel_token_set(lhs)
    right = _kernel_token_set(rhs)
    if not left or not right:
        return 0.0
    return float(len(left & right)) / float(len(left | right))


def _anchor_match_score(
    base_item: Dict[str, object],
    target_item: Dict[str, object],
    *,
    base_idx: int,
    target_idx: int,
    base_count: int,
    target_count: int,
    base_start_ns: int,
    base_end_ns: int,
    target_start_ns: int,
    target_end_ns: int,
) -> float:
    if _kernel_anchor_key(base_item.get("kernel_name")) != _kernel_anchor_key(target_item.get("kernel_name")):
        return 0.0
    base_time = _kernel_center_norm(
        base_item,
        seq_start_ns=base_start_ns,
        seq_end_ns=base_end_ns,
        ordinal_idx=base_idx,
        ordinal_count=base_count,
    )
    target_time = _kernel_center_norm(
        target_item,
        seq_start_ns=target_start_ns,
        seq_end_ns=target_end_ns,
        ordinal_idx=target_idx,
        ordinal_count=target_count,
    )
    base_ord = float(int(base_idx) / float(max(1, int(base_count) - 1)))
    target_ord = float(int(target_idx) / float(max(1, int(target_count) - 1)))
    time_closeness = max(0.0, 1.0 - (abs(base_time - target_time) / 0.18))
    order_closeness = max(0.0, 1.0 - (abs(base_ord - target_ord) / 0.22))
    position_score = (0.6 * time_closeness) + (0.4 * order_closeness)
    if position_score < 0.20:
        return 0.0
    duration_score = _ratio_similarity(
        _kernel_duration_ms_value(base_item),
        _kernel_duration_ms_value(target_item),
        max_ratio=4.0,
    )
    return 0.72 + (0.20 * position_score) + (0.08 * duration_score)


def _align_stream_kernel_pairs(
    base_segment: Sequence[Dict[str, object]],
    target_segment: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    base_items = list(base_segment)
    target_items = list(target_segment)
    n = len(base_items)
    m = len(target_items)
    if n <= 0 or m <= 0:
        return []
    base_start_ns, base_end_ns = _sequence_time_bounds(base_items)
    target_start_ns, target_end_ns = _sequence_time_bounds(target_items)
    scores: List[List[float]] = []
    for i, base_item in enumerate(base_items):
        row: List[float] = []
        for j, target_item in enumerate(target_items):
            row.append(
                _anchor_match_score(
                    base_item,
                    target_item,
                    base_idx=i,
                    target_idx=j,
                    base_count=n,
                    target_count=m,
                    base_start_ns=base_start_ns,
                    base_end_ns=base_end_ns,
                    target_start_ns=target_start_ns,
                    target_end_ns=target_end_ns,
                )
            )
        scores.append(row)

    threshold = 0.82
    dp: List[List[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    action: List[List[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best = dp[i + 1][j]
            best_action = "skip_base"
            if dp[i][j + 1] > best + 1e-9:
                best = dp[i][j + 1]
                best_action = "skip_target"
            score = scores[i][j]
            if score >= threshold:
                candidate = score + dp[i + 1][j + 1]
                if candidate > best + 1e-9 or (
                    abs(candidate - best) <= 1e-9 and best_action != "match"
                ):
                    best = candidate
                    best_action = "match"
            dp[i][j] = best
            action[i][j] = best_action

    matches: List[Dict[str, object]] = []
    i = 0
    j = 0
    while i < n and j < m:
        step = action[i][j]
        if step == "match":
            matches.append(
                {
                    "base_idx": i,
                    "target_idx": j,
                    "base_kernel": base_items[i],
                    "target_kernel": target_items[j],
                    "score": round(scores[i][j], 4),
                }
            )
            i += 1
            j += 1
        elif step == "skip_target":
            j += 1
        else:
            i += 1
    return matches


def _segment_midpoint_norm(
    segment: Sequence[Dict[str, object]],
    *,
    full_segment: Sequence[Dict[str, object]],
) -> float:
    if not segment:
        return 0.5
    seg_start_ns, seg_end_ns = _sequence_time_bounds(segment)
    full_start_ns, full_end_ns = _sequence_time_bounds(full_segment)
    span = max(1, int(full_end_ns) - int(full_start_ns))
    center = 0.5 * (float(seg_start_ns) + float(seg_end_ns))
    return max(0.0, min(1.0, (center - float(full_start_ns)) / float(span)))


def _kernel_name_has_fusion_hint(name: object) -> bool:
    text = _kernel_anchor_key(name)
    return "fused" in text or "fusion" in text


def _single_vs_multi_name_evidence(
    *,
    single_kernel: Dict[str, object],
    multi_segment: Sequence[Dict[str, object]],
) -> Tuple[float, List[str]]:
    if not multi_segment:
        return 0.0, []
    single_name = single_kernel.get("kernel_name")
    single_key = _kernel_anchor_key(single_name)
    multi_keys = {_kernel_anchor_key(item.get("kernel_name")) for item in multi_segment}
    fusion_hint = _kernel_name_has_fusion_hint(single_name)
    overlap_scores = [_kernel_token_overlap(single_name, item.get("kernel_name")) for item in multi_segment]
    covered = float(sum(1 for score in overlap_scores if score >= 0.35)) / float(max(1, len(overlap_scores)))
    union_tokens: set[str] = set()
    for item in multi_segment:
        union_tokens.update(_kernel_token_set(item.get("kernel_name")))
    single_tokens = _kernel_token_set(single_name)
    union_overlap = 0.0
    if union_tokens and single_tokens:
        union_overlap = float(len(union_tokens & single_tokens)) / float(len(union_tokens | single_tokens))
    score = max(0.0, (0.65 * covered) + (0.35 * union_overlap))
    reasons: List[str] = []
    if covered >= 0.66:
        reasons.append("multi-name-overlap")
    if union_overlap >= 0.25:
        reasons.append("token-overlap")
    if fusion_hint:
        score = max(score, 1.0)
        reasons.append("fused-name")
    if single_key in multi_keys and not fusion_hint:
        score *= 0.35
        reasons.append("single-name-reused")
    return min(1.0, score), reasons


def _classify_fusion_gap(
    *,
    rank: Optional[int],
    dev: int,
    sid: int,
    prev_match: Dict[str, object],
    next_match: Dict[str, object],
    base_segment: Sequence[Dict[str, object]],
    target_segment: Sequence[Dict[str, object]],
    base_full_segment: Sequence[Dict[str, object]],
    target_full_segment: Sequence[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    seg_base = list(base_segment)
    seg_target = list(target_segment)
    base_count = len(seg_base)
    target_count = len(seg_target)
    if base_count <= 0 or target_count <= 0:
        return None
    if base_count == target_count:
        return None
    if base_count >= 2 and target_count == 1:
        kind = "target_fused"
        name_score, name_reasons = _single_vs_multi_name_evidence(
            single_kernel=seg_target[0],
            multi_segment=seg_base,
        )
    elif target_count >= 2 and base_count == 1:
        kind = "target_split"
        name_score, name_reasons = _single_vs_multi_name_evidence(
            single_kernel=seg_base[0],
            multi_segment=seg_target,
        )
    else:
        return None

    anchor_score = 0.5 * (
        float(prev_match.get("score") or 0.0) + float(next_match.get("score") or 0.0)
    )
    duration_score = _ratio_similarity(
        _segment_duration_ms(seg_base),
        _segment_duration_ms(seg_target),
        max_ratio=6.0,
    )
    position_score = max(
        0.0,
        1.0
        - (
            abs(
                _segment_midpoint_norm(seg_base, full_segment=base_full_segment)
                - _segment_midpoint_norm(seg_target, full_segment=target_full_segment)
            )
            / 0.16
        ),
    )
    if anchor_score < 0.88 or position_score < 0.55 or name_score < 0.22:
        return None

    score = (
        (0.34 * anchor_score)
        + (0.24 * duration_score)
        + (0.22 * position_score)
        + (0.20 * name_score)
    )
    if score < 0.78:
        return None

    evidence: List[str] = []
    if anchor_score >= 0.92:
        evidence.append("strong-anchors")
    if duration_score >= 0.55:
        evidence.append("duration-close")
    if position_score >= 0.75:
        evidence.append("position-aligned")
    for reason in name_reasons:
        if reason not in evidence:
            evidence.append(reason)

    confidence = "high" if score >= 0.88 and name_score >= 0.40 else "medium"
    return {
        "rank": rank,
        "device_id": dev,
        "stream_id": sid,
        "kind": kind,
        "confidence": confidence,
        "score": round(score, 4),
        "anchor_score": round(anchor_score, 4),
        "duration_score": round(duration_score, 4),
        "position_score": round(position_score, 4),
        "name_score": round(name_score, 4),
        "evidence": evidence,
        "prev_anchor": prev_match.get("base_kernel", {}).get("kernel_name"),
        "next_anchor": next_match.get("base_kernel", {}).get("kernel_name"),
        "base_segment": seg_base,
        "target_segment": seg_target,
        "base_duration_ms": round(_segment_duration_ms(seg_base), 6),
        "target_duration_ms": round(_segment_duration_ms(seg_target), 6),
    }


def _timeline_stream_sequences(payload: Dict[str, object]) -> Dict[Tuple[Optional[int], int, int], List[Dict[str, object]]]:
    out: Dict[Tuple[Optional[int], int, int], List[Dict[str, object]]] = {}
    for group in list(payload.get("all_stream_groups") or []):
        rank_raw = group.get("rank")
        rank = int(rank_raw) if isinstance(rank_raw, int) else None
        dev = _to_int(group.get("device_id"), -1)
        for stream in list(group.get("streams") or []):
            sid = _to_int(stream.get("stream_id"), 0)
            kernels = list(stream.get("kernels") or [])
            out[(rank, dev, sid)] = kernels
    return out


def _build_fusion_findings(
    base_payload: Dict[str, object],
    target_payload: Dict[str, object],
) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    base_streams = _timeline_stream_sequences(base_payload)
    target_streams = _timeline_stream_sequences(target_payload)
    common_keys = sorted(set(base_streams.keys()) & set(target_streams.keys()), key=_stream_key_sort_key)
    for key in common_keys:
        rank, dev, sid = key
        base_segment = list(base_streams.get(key) or [])
        target_segment = list(target_streams.get(key) or [])
        if len(base_segment) < 3 or len(target_segment) < 3:
            continue
        matches = _align_stream_kernel_pairs(base_segment, target_segment)
        if len(matches) < 2:
            continue
        for prev_match, next_match in zip(matches, matches[1:]):
            base_start_idx = int(prev_match.get("base_idx", -1)) + 1
            base_end_idx = int(next_match.get("base_idx", -1))
            target_start_idx = int(prev_match.get("target_idx", -1)) + 1
            target_end_idx = int(next_match.get("target_idx", -1))
            if base_start_idx >= base_end_idx or target_start_idx >= target_end_idx:
                continue
            finding = _classify_fusion_gap(
                rank=rank,
                dev=dev,
                sid=sid,
                prev_match=prev_match,
                next_match=next_match,
                base_segment=base_segment[base_start_idx:base_end_idx],
                target_segment=target_segment[target_start_idx:target_end_idx],
                base_full_segment=base_segment,
                target_full_segment=target_segment,
            )
            if finding:
                findings.append(finding)
    return findings


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(val):
        return float(default)
    return float(val)


def _metric_focus_rank(metric_name: object) -> Tuple[int, str]:
    text = str(metric_name or "").strip().lower()
    for idx, token in enumerate(_DEFAULT_FOCUS_METRIC_TOKENS):
        if token in text:
            return idx, text
    return len(_DEFAULT_FOCUS_METRIC_TOKENS) + 1, text


def _aggregate_kernel_hotspots(kernels: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_name: Dict[str, Dict[str, object]] = {}
    for row in kernels:
        name = str(row.get("kernel_name") or "").strip()
        if not name:
            continue
        entry = by_name.setdefault(
            name,
            {
                "name": name,
                "kind": str(row.get("kind") or ""),
                "count": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "fused_hint": _kernel_name_has_fusion_hint(name),
            },
        )
        dur_ms = _safe_float(row.get("duration_ms"))
        entry["count"] = int(entry.get("count") or 0) + 1
        entry["total_ms"] = _safe_float(entry.get("total_ms")) + dur_ms
        entry["max_ms"] = max(_safe_float(entry.get("max_ms")), dur_ms)
    return sorted(
        by_name.values(),
        key=lambda x: (-_safe_float(x.get("total_ms")), -int(x.get("count") or 0), str(x.get("name") or "")),
    )


def _summarize_metric_series(metric_series: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in metric_series:
        name = str(item.get("name") or "").strip()
        points = list(item.get("points") or [])
        values = [_safe_float(p[1], float("nan")) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2]
        values = [v for v in values if math.isfinite(v)]
        if not name or not values:
            continue
        rows.append(
            {
                "name": name,
                "samples": len(values),
                "avg": sum(values) / float(max(1, len(values))),
                "min": min(values),
                "max": max(values),
                "last": values[-1],
                "focus_rank": _metric_focus_rank(name)[0],
            }
        )
    rows.sort(
        key=lambda x: (
            int(x.get("focus_rank") or 0),
            -abs(_safe_float(x.get("avg"))),
            str(x.get("name") or ""),
        )
    )
    return rows


def _summarize_timeline_state(state: Dict[str, object]) -> Dict[str, object]:
    kernels = list(state.get("kernels") or [])
    metric_series = list(state.get("metric_series") or [])
    window_start_ns = _to_int(state.get("window_start_ns"), 0)
    window_end_ns = _to_int(state.get("window_end_ns"), window_start_ns + 1)
    window_ms = max(0.0, float(window_end_ns - window_start_ns) / 1e6)
    merged = _merge_intervals(
        [(_to_int(row.get("start_ns"), -1), _to_int(row.get("end_ns"), -1)) for row in kernels]
    )
    active_ms = sum(float(e - s) for s, e in merged) / 1e6
    stream_ids = {
        (_to_int(row.get("device_id"), -1), _to_int(row.get("stream_id"), 0))
        for row in kernels
    }
    unique_names = {str(row.get("kernel_name") or "").strip() for row in kernels if str(row.get("kernel_name") or "").strip()}
    hotspots = _aggregate_kernel_hotspots(kernels)
    compute_rows = [row for row in kernels if str(row.get("kind") or "") != "comm"]
    comm_rows = [row for row in kernels if str(row.get("kind") or "") == "comm"]
    occ_weight_num = 0.0
    occ_weight_den = 0.0
    for row in kernels:
        occ = row.get("occupancy_pct_estimate")
        if occ is None:
            continue
        dur = _safe_float(row.get("duration_ms"))
        occ_weight_num += dur * _safe_float(occ)
        occ_weight_den += dur
    metric_rows = _summarize_metric_series(metric_series)
    category_summary = dict(state.get("kernel_category_summary") or {})
    category_rows = list(category_summary.get("rows") or [])
    fused_hotspots = [row for row in hotspots if bool(row.get("fused_hint"))]
    return {
        "kernel_total": len(kernels),
        "window_ms": window_ms,
        "active_ms": active_ms,
        "stream_count": len(stream_ids),
        "unique_kernel_names": len(unique_names),
        "compute_count": len(compute_rows),
        "compute_total_ms": sum(_safe_float(row.get("duration_ms")) for row in compute_rows),
        "comm_count": len(comm_rows),
        "comm_total_ms": sum(_safe_float(row.get("duration_ms")) for row in comm_rows),
        "weighted_occupancy_pct": (occ_weight_num / occ_weight_den) if occ_weight_den > 0 else None,
        "top_kernels": hotspots[:8],
        "top_metrics": metric_rows[:8],
        "top_categories": category_rows[:8],
        "metric_map": {str(row.get("name") or ""): row for row in metric_rows},
        "kernel_map": {str(row.get("name") or ""): row for row in hotspots},
        "category_map": {str(row.get("category") or ""): row for row in category_rows},
        "category_non_overlap_ms": _safe_float(category_summary.get("non_overlap_ms")),
        "fused_kernel_groups": len(fused_hotspots),
        "fused_kernel_total_ms": sum(_safe_float(row.get("total_ms")) for row in fused_hotspots),
    }


def _build_metric_deltas(base_summary: Dict[str, object], target_summary: Dict[str, object]) -> List[Dict[str, object]]:
    base_map = dict(base_summary.get("metric_map") or {})
    target_map = dict(target_summary.get("metric_map") or {})
    out: List[Dict[str, object]] = []
    for name in sorted(set(base_map.keys()) & set(target_map.keys()), key=lambda x: _metric_focus_rank(x)):
        base_row = dict(base_map.get(name) or {})
        target_row = dict(target_map.get(name) or {})
        delta_avg = _safe_float(target_row.get("avg")) - _safe_float(base_row.get("avg"))
        delta_max = _safe_float(target_row.get("max")) - _safe_float(base_row.get("max"))
        out.append(
            {
                "name": name,
                "focus_rank": _metric_focus_rank(name)[0],
                "base_avg": _safe_float(base_row.get("avg")),
                "target_avg": _safe_float(target_row.get("avg")),
                "delta_avg": delta_avg,
                "base_max": _safe_float(base_row.get("max")),
                "target_max": _safe_float(target_row.get("max")),
                "delta_max": delta_max,
                "base_last": _safe_float(base_row.get("last")),
                "target_last": _safe_float(target_row.get("last")),
                "delta_last": _safe_float(target_row.get("last")) - _safe_float(base_row.get("last")),
            }
        )
    out.sort(
        key=lambda x: (
            int(x.get("focus_rank") or 0),
            -abs(_safe_float(x.get("delta_avg"))),
            -abs(_safe_float(x.get("delta_max"))),
            str(x.get("name") or ""),
        )
    )
    return out[:10]


def _build_kernel_deltas(base_summary: Dict[str, object], target_summary: Dict[str, object]) -> List[Dict[str, object]]:
    base_map = dict(base_summary.get("kernel_map") or {})
    target_map = dict(target_summary.get("kernel_map") or {})
    out: List[Dict[str, object]] = []
    for name in set(base_map.keys()) | set(target_map.keys()):
        base_row = dict(base_map.get(name) or {})
        target_row = dict(target_map.get(name) or {})
        delta_total_ms = _safe_float(target_row.get("total_ms")) - _safe_float(base_row.get("total_ms"))
        delta_count = int(target_row.get("count") or 0) - int(base_row.get("count") or 0)
        if abs(delta_total_ms) < 1e-12 and delta_count == 0:
            continue
        out.append(
            {
                "name": name,
                "kind": str(target_row.get("kind") or base_row.get("kind") or ""),
                "base_total_ms": _safe_float(base_row.get("total_ms")),
                "target_total_ms": _safe_float(target_row.get("total_ms")),
                "delta_total_ms": delta_total_ms,
                "base_count": int(base_row.get("count") or 0),
                "target_count": int(target_row.get("count") or 0),
                "delta_count": delta_count,
                "fused_hint": bool(target_row.get("fused_hint") or base_row.get("fused_hint")),
            }
        )
    out.sort(
        key=lambda x: (
            -abs(_safe_float(x.get("delta_total_ms"))),
            -abs(int(x.get("delta_count") or 0)),
            str(x.get("name") or ""),
        )
    )
    return out[:10]


def _build_category_deltas(base_summary: Dict[str, object], target_summary: Dict[str, object]) -> List[Dict[str, object]]:
    base_map = dict(base_summary.get("category_map") or {})
    target_map = dict(target_summary.get("category_map") or {})
    out: List[Dict[str, object]] = []
    for category in set(base_map.keys()) | set(target_map.keys()):
        base_row = dict(base_map.get(category) or {})
        target_row = dict(target_map.get(category) or {})
        delta_weighted_ms = _safe_float(target_row.get("weighted_elapsed_ms")) - _safe_float(base_row.get("weighted_elapsed_ms"))
        delta_weighted_pct = _safe_float(target_row.get("weighted_pct_of_nonoverlap")) - _safe_float(base_row.get("weighted_pct_of_nonoverlap"))
        if abs(delta_weighted_ms) < 1e-9 and abs(delta_weighted_pct) < 1e-9:
            continue
        out.append(
            {
                "category": category,
                "base_weighted_ms": _safe_float(base_row.get("weighted_elapsed_ms")),
                "target_weighted_ms": _safe_float(target_row.get("weighted_elapsed_ms")),
                "delta_weighted_ms": delta_weighted_ms,
                "base_weighted_pct": _safe_float(base_row.get("weighted_pct_of_nonoverlap")),
                "target_weighted_pct": _safe_float(target_row.get("weighted_pct_of_nonoverlap")),
                "delta_weighted_pct": delta_weighted_pct,
            }
        )
    out.sort(
        key=lambda x: (
            -abs(_safe_float(x.get("delta_weighted_ms"))),
            -abs(_safe_float(x.get("delta_weighted_pct"))),
            str(x.get("category") or ""),
        )
    )
    return out[:10]


def _fmt_float(value: object, digits: int = 3, suffix: str = "") -> str:
    return f"{_safe_float(value):.{int(digits)}f}{suffix}"


def _fmt_optional_float(value: object, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{_safe_float(value):.{int(digits)}f}{suffix}"


def _fmt_signed_float(value: object, digits: int = 3, suffix: str = "") -> str:
    val = _safe_float(value)
    return f"{val:+.{int(digits)}f}{suffix}"


def export_timeline_compare_html(
    sqlite_paths: Sequence[str],
    *,
    output_path: str,
    device_id: int = -1,
    start_ns: int = -1,
    end_ns: int = -1,
    limit: int = 100000,
    width_px: int = 1800,
    nvtx_text: str = "",
    nvtx_index: int = -1,
    include_metrics: bool = False,
    metric_name_like: str = "%",
    metrics_limit: int = -1,
    metrics_max_points: int = -1,
    overlay_metrics_per_track: int = 7,
    kernel_category_map_json: str = "",
    kernel_category_engine: str = "sglang",
    kernel_category_model: str = "llama",
    enable_kernel_category_breakdown: bool = True,
    kernel_category_table_output: str = "",
    nvtx_category_stats_output: str = "",
    default_focus_metrics: bool = True,
    include_all_metric_sources: bool = False,
    debug: bool = False,
    debug_rows: int = -1,
    debug_log_fn: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    items = [str(p) for p in sqlite_paths if str(p or "").strip()]
    if len(items) < 2:
        raise ValueError("export_timeline_compare_html requires at least two sqlite paths")

    compare_debug = _build_debug_logger(enabled=bool(debug), log_fn=debug_log_fn)
    kernel_rules, kernel_profile = _resolve_kernel_category_rules(
        map_json_path=str(kernel_category_map_json or ""),
        engine=str(kernel_category_engine or ""),
        model=str(kernel_category_model or ""),
        debug_log=compare_debug,
    )
    rendered: List[Dict[str, object]] = []
    total = len(items)

    def _build_section_srcdoc(full_html: str, *, panel_title: str) -> Optional[str]:
        if str(panel_title or "").strip() not in str(full_html or ""):
            return None
        injected = "\n".join(
            [
                "<style>",
                "body{margin:0;padding:0;background:#0f1116;overflow:hidden;}",
                "h2,.meta{display:none !important;}",
                ".card{margin:0 !important;}",
                "</style>",
                "<script>",
                "(function(){",
                f"  const target = {json.dumps(str(panel_title), ensure_ascii=False)};",
                "  const cards = Array.from(document.querySelectorAll('.card'));",
                "  let kept = 0;",
                "  for (const card of cards) {",
                "    const titleEl = card.querySelector('.panel-title');",
                "    const title = titleEl ? String(titleEl.textContent || '').trim() : '';",
                "    if (title === target) {",
                "      kept += 1;",
                "      continue;",
                "    }",
                "    card.remove();",
                "  }",
                "  const h2 = document.querySelector('h2');",
                "  if (h2) h2.remove();",
                "  const meta = document.querySelector('.meta');",
                "  if (meta) meta.remove();",
                "  if (!kept && document.body) {",
                "    document.body.innerHTML = \"<div class='card'><div class='empty'>Section not available.</div></div>\";",
                "  }",
                "})();",
                "</script>",
            ]
        )
        text = str(full_html or "")
        if "</body>" in text:
            return text.replace("</body>", injected + "\n</body>", 1)
        return text + injected

    normalized_compare_span_ns = -1
    collected: List[Dict[str, object]] = []
    for idx, sqlite_path in enumerate(items):
        label = Path(str(sqlite_path)).name or f"sqlite_{idx}"
        compare_debug(f"compare child collect start index={idx} sqlite={sqlite_path}")

        def _child_progress(msg: str, *, _idx: int = idx, _label: str = label) -> None:
            if progress_cb:
                progress_cb(f"[{_idx + 1}/{total}] {_label} {str(msg or '').strip()}")

        def _child_debug(msg: str, *, _idx: int = idx, _label: str = label) -> None:
            compare_debug(f"[{_idx + 1}/{total}] {_label} {msg}")

        state = _collect_timeline_state(
            str(sqlite_path),
            output_path="",
            device_id=int(device_id),
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            limit=int(limit),
            nvtx_text=str(nvtx_text or ""),
            nvtx_index=int(nvtx_index),
            include_metrics=bool(include_metrics),
            metric_name_like=str(metric_name_like or "%"),
            metrics_limit=int(metrics_limit),
            metrics_max_points=int(metrics_max_points),
            kernel_category_rules=kernel_rules,
            kernel_category_profile=str(kernel_profile),
            enable_kernel_category_breakdown=bool(enable_kernel_category_breakdown),
            default_focus_metrics=bool(default_focus_metrics),
            include_all_metric_sources=bool(include_all_metric_sources),
            debug=bool(debug),
            debug_rows=int(debug_rows),
            debug_log_fn=_child_debug,
            progress_cb=_child_progress,
        )
        collected.append(
            {
                "label": label,
                "sqlite_path": str(sqlite_path),
                "state": state,
                "summary": _summarize_timeline_state(state),
            }
        )
        compare_debug(
            "compare child collect done index={} sqlite={} kernels={} metrics={} span_ns={}".format(
                idx,
                sqlite_path,
                len(list(state.get("kernels") or [])),
                len(list(state.get("metric_series") or [])),
                int(_to_int(state.get("window_end_ns"), 1) - _to_int(state.get("window_start_ns"), 0)),
            )
        )

    natural_spans = [
        max(1, int(_to_int(item.get("state", {}).get("window_end_ns"), 1) - _to_int(item.get("state", {}).get("window_start_ns"), 0)))
        for item in collected
    ]
    if natural_spans:
        unique_spans = sorted(set(int(v) for v in natural_spans))
        if len(unique_spans) > 1:
            normalized_compare_span_ns = max(unique_spans)
            compare_debug(
                "normalize compare span_ns={} natural_spans={}".format(
                    int(normalized_compare_span_ns),
                    ",".join(str(int(v)) for v in unique_spans),
                )
            )

    rendered = []
    for idx, item in enumerate(collected):
        state = dict(item.get("state") or {})
        html_text = _render_html(
            sqlite_path=str(item.get("sqlite_path") or ""),
            kernels=list(state.get("kernels") or []),
            metric_series=list(state.get("metric_series") or []),
            kernel_category_summary=dict(state.get("kernel_category_summary") or {}),
            kernel_category_profile=str(state.get("kernel_category_profile") or ""),
            nvtx_window_category_stats=dict(state.get("nvtx_window_category_stats") or {}),
            window_start_ns=int(state.get("window_start_ns") or 0),
            window_end_ns=int(state.get("window_end_ns") or 1),
            display_span_ns=int(normalized_compare_span_ns),
            nvtx_windows=list(state.get("nvtx_windows") or []),
            width_px=int(width_px),
            include_metrics=bool(include_metrics),
            overlay_metrics_per_track=int(overlay_metrics_per_track),
        )
        payload = _extract_timeline_payload_from_html(html_text) or {}
        rendered.append(
            {
                **item,
                "srcdoc": html_text,
                "payload": payload,
            }
        )
        compare_debug(
            "compare child render done index={} sqlite={} payload_span_ns={} data_span_ns={}".format(
                idx,
                str(item.get("sqlite_path") or ""),
                int(payload.get("span_ns") or 0),
                int(payload.get("data_span_ns") or payload.get("span_ns") or 0),
            )
        )

    optimization_cards: List[str] = []
    for idx, item in enumerate(rendered):
        summary = dict(item.get("summary") or {})
        hotspot_rows = list(summary.get("top_kernels") or [])
        metric_rows = list(summary.get("top_metrics") or [])
        category_rows = list(summary.get("top_categories") or [])
        hotspot_html = "".join(
            [
                (
                    "<tr>"
                    f"<td><code title='{html.escape(str(row.get('name') or ''))}'>{html.escape(_short_kernel_name(row.get('name')))}</code></td>"
                    f"<td>{html.escape(str(row.get('kind') or ''))}</td>"
                    f"<td>{int(row.get('count') or 0)}</td>"
                    f"<td>{_fmt_float(row.get('total_ms'), 3, ' ms')}</td>"
                    "</tr>"
                )
                for row in hotspot_rows[:6]
            ]
        )
        metric_html = "".join(
            [
                (
                    "<tr>"
                    f"<td><code title='{html.escape(str(row.get('name') or ''))}'>{html.escape(_short_kernel_name(row.get('name')))}</code></td>"
                    f"<td>{_fmt_float(row.get('avg'), 3)}</td>"
                    f"<td>{_fmt_float(row.get('max'), 3)}</td>"
                    f"<td>{_fmt_float(row.get('last'), 3)}</td>"
                    "</tr>"
                )
                for row in metric_rows[:6]
            ]
        )
        category_html = "".join(
            [
                (
                    "<tr>"
                    f"<td><code>{html.escape(str(row.get('category') or 'misc'))}</code></td>"
                    f"<td>{_fmt_float(row.get('weighted_elapsed_ms'), 3, ' ms')}</td>"
                    f"<td>{_fmt_float(row.get('weighted_pct_of_nonoverlap'), 2, '%')}</td>"
                    f"<td>{int(row.get('instances') or 0)}</td>"
                    "</tr>"
                )
                for row in category_rows[:6]
            ]
        )
        optimization_cards.extend(
            [
                "<section class='compare-card summary-card'>",
                "<div class='compare-head'>",
                f"<div class='compare-index'>#{idx + 1}</div>",
                "<div class='compare-meta'>",
                f"<div class='compare-title'>{html.escape(str(item.get('label') or ''))}</div>",
                f"<div class='compare-path'>{html.escape(str(item.get('sqlite_path') or ''))}</div>",
                "</div>",
                "</div>",
                "<div class='summary-grid'>",
                f"<div class='summary-pill'><span>window</span><strong>{_fmt_float(summary.get('window_ms'), 3, ' ms')}</strong></div>",
                f"<div class='summary-pill'><span>gpu active</span><strong>{_fmt_float(summary.get('active_ms'), 3, ' ms')}</strong></div>",
                f"<div class='summary-pill'><span>kernels</span><strong>{int(summary.get('kernel_total') or 0)}</strong></div>",
                f"<div class='summary-pill'><span>streams</span><strong>{int(summary.get('stream_count') or 0)}</strong></div>",
                f"<div class='summary-pill'><span>unique kernels</span><strong>{int(summary.get('unique_kernel_names') or 0)}</strong></div>",
                f"<div class='summary-pill'><span>compute total</span><strong>{_fmt_float(summary.get('compute_total_ms'), 3, ' ms')}</strong></div>",
                f"<div class='summary-pill'><span>comm total</span><strong>{_fmt_float(summary.get('comm_total_ms'), 3, ' ms')}</strong></div>",
                f"<div class='summary-pill'><span>weighted occ</span><strong>{_fmt_optional_float(summary.get('weighted_occupancy_pct'), 2, '%')}</strong></div>",
                f"<div class='summary-pill'><span>fused groups</span><strong>{int(summary.get('fused_kernel_groups') or 0)}</strong></div>",
                f"<div class='summary-pill'><span>fused total</span><strong>{_fmt_float(summary.get('fused_kernel_total_ms'), 3, ' ms')}</strong></div>",
                f"<div class='summary-pill'><span>category non-overlap</span><strong>{_fmt_float(summary.get('category_non_overlap_ms'), 3, ' ms')}</strong></div>",
                "</div>",
                "<div class='summary-triple'>",
                "<div class='summary-box'>",
                "<div class='summary-box-title'>Kernel Hotspots</div>",
                (
                    "<table class='summary-table'><thead><tr><th>kernel</th><th>kind</th><th>count</th><th>total</th></tr></thead>"
                    f"<tbody>{hotspot_html or '<tr><td colspan=\"4\">No kernels</td></tr>'}</tbody></table>"
                ),
                "</div>",
                "<div class='summary-box'>",
                "<div class='summary-box-title'>Metric Snapshot</div>",
                (
                    "<table class='summary-table'><thead><tr><th>metric</th><th>avg</th><th>max</th><th>last</th></tr></thead>"
                    f"<tbody>{metric_html or '<tr><td colspan=\"4\">Metrics not enabled</td></tr>'}</tbody></table>"
                ),
                "</div>",
                "<div class='summary-box'>",
                "<div class='summary-box-title'>Category Breakdown (weighted, overlap-aware)</div>",
                (
                    "<table class='summary-table'><thead><tr><th>category</th><th>weighted</th><th>share</th><th>instances</th></tr></thead>"
                    f"<tbody>{category_html or '<tr><td colspan=\"4\">No category rows</td></tr>'}</tbody></table>"
                ),
                "</div>",
                "</div>",
                "</section>",
            ]
        )

    optimization_section: List[str] = []
    if optimization_cards:
        optimization_section = [
            "<section class='compare-section'>",
            "<h3 class='compare-section-title'>Optimization Summary</h3>",
            (
                "<div class='compare-section-note'>"
                "For each sqlite, summarize the matched NVTX window by GPU-active span, kernel mix, fused-kernel hints, and focused metric snapshots."
                "</div>"
            ),
            "<div class='compare-section-stack'>",
            *optimization_cards,
            "</div>",
            "</section>",
        ]

    delta_section: List[str] = []
    if len(rendered) >= 2:
        base_item = rendered[0]
        target_item = rendered[1]
        base_summary = dict(base_item.get("summary") or {})
        target_summary = dict(target_item.get("summary") or {})
        metric_deltas = _build_metric_deltas(base_summary, target_summary)
        kernel_deltas = _build_kernel_deltas(base_summary, target_summary)
        category_deltas = _build_category_deltas(base_summary, target_summary)
        metric_delta_rows = "".join(
            [
                (
                    "<tr>"
                    f"<td><code title='{html.escape(str(row.get('name') or ''))}'>{html.escape(_short_kernel_name(row.get('name')))}</code></td>"
                    f"<td>{_fmt_float(row.get('base_avg'), 3)}</td>"
                    f"<td>{_fmt_float(row.get('target_avg'), 3)}</td>"
                    f"<td>{_fmt_signed_float(row.get('delta_avg'), 3)}</td>"
                    f"<td>{_fmt_signed_float(row.get('delta_max'), 3)}</td>"
                    "</tr>"
                )
                for row in metric_deltas
            ]
        )
        kernel_delta_rows = "".join(
            [
                (
                    "<tr>"
                    f"<td><code title='{html.escape(str(row.get('name') or ''))}'>{html.escape(_short_kernel_name(row.get('name')))}</code></td>"
                    f"<td>{html.escape(str(row.get('kind') or ''))}</td>"
                    f"<td>{int(row.get('base_count') or 0)} -> {int(row.get('target_count') or 0)}</td>"
                    f"<td>{_fmt_float(row.get('base_total_ms'), 3, ' ms')} -> {_fmt_float(row.get('target_total_ms'), 3, ' ms')}</td>"
                    f"<td>{_fmt_signed_float(row.get('delta_total_ms'), 3, ' ms')}</td>"
                    "</tr>"
                )
                for row in kernel_deltas
            ]
        )
        category_delta_rows = "".join(
            [
                (
                    "<tr>"
                    f"<td><code>{html.escape(str(row.get('category') or 'misc'))}</code></td>"
                    f"<td>{_fmt_float(row.get('base_weighted_ms'), 3, ' ms')}</td>"
                    f"<td>{_fmt_float(row.get('target_weighted_ms'), 3, ' ms')}</td>"
                    f"<td>{_fmt_signed_float(row.get('delta_weighted_ms'), 3, ' ms')}</td>"
                    f"<td>{_fmt_signed_float(row.get('delta_weighted_pct'), 2, '%')}</td>"
                    "</tr>"
                )
                for row in category_deltas
            ]
        )
        delta_section = [
            "<section class='compare-section'>",
            "<h3 class='compare-section-title'>Pairwise Delta Summary</h3>",
            (
                "<div class='compare-section-note'>"
                "The first sqlite is treated as baseline and the second as target. Deltas are computed on the same matched NVTX window semantics."
                "</div>"
            ),
            "<div class='summary-grid'>",
            f"<div class='summary-pill'><span>active delta</span><strong>{_fmt_signed_float(_safe_float(target_summary.get('active_ms')) - _safe_float(base_summary.get('active_ms')), 3, ' ms')}</strong></div>",
            f"<div class='summary-pill'><span>kernel delta</span><strong>{int(target_summary.get('kernel_total') or 0) - int(base_summary.get('kernel_total') or 0):+d}</strong></div>",
            f"<div class='summary-pill'><span>unique delta</span><strong>{int(target_summary.get('unique_kernel_names') or 0) - int(base_summary.get('unique_kernel_names') or 0):+d}</strong></div>",
            f"<div class='summary-pill'><span>compute delta</span><strong>{_fmt_signed_float(_safe_float(target_summary.get('compute_total_ms')) - _safe_float(base_summary.get('compute_total_ms')), 3, ' ms')}</strong></div>",
            f"<div class='summary-pill'><span>comm delta</span><strong>{_fmt_signed_float(_safe_float(target_summary.get('comm_total_ms')) - _safe_float(base_summary.get('comm_total_ms')), 3, ' ms')}</strong></div>",
            f"<div class='summary-pill'><span>occ delta</span><strong>{_fmt_signed_float((_safe_float(target_summary.get('weighted_occupancy_pct')) if target_summary.get('weighted_occupancy_pct') is not None else 0.0) - (_safe_float(base_summary.get('weighted_occupancy_pct')) if base_summary.get('weighted_occupancy_pct') is not None else 0.0), 2, '%')}</strong></div>",
            f"<div class='summary-pill'><span>category non-overlap delta</span><strong>{_fmt_signed_float(_safe_float(target_summary.get('category_non_overlap_ms')) - _safe_float(base_summary.get('category_non_overlap_ms')), 3, ' ms')}</strong></div>",
            "</div>",
            "<div class='summary-triple'>",
            "<div class='summary-box'>",
            f"<div class='summary-box-title'>Metric Delta ({html.escape(str(base_item.get('label') or 'baseline'))} -> {html.escape(str(target_item.get('label') or 'target'))})</div>",
            (
                "<table class='summary-table'><thead><tr><th>metric</th><th>base avg</th><th>target avg</th><th>delta avg</th><th>delta max</th></tr></thead>"
                f"<tbody>{metric_delta_rows or '<tr><td colspan=\"5\">Metrics not enabled or no overlapping metric names</td></tr>'}</tbody></table>"
            ),
            "</div>",
            "<div class='summary-box'>",
            "<div class='summary-box-title'>Kernel Delta</div>",
            (
                "<table class='summary-table'><thead><tr><th>kernel</th><th>kind</th><th>count</th><th>total</th><th>delta</th></tr></thead>"
                f"<tbody>{kernel_delta_rows or '<tr><td colspan=\"5\">No changed kernels</td></tr>'}</tbody></table>"
            ),
            "</div>",
            "<div class='summary-box'>",
            "<div class='summary-box-title'>Category Delta (weighted, overlap-aware)</div>",
            (
                "<table class='summary-table'><thead><tr><th>category</th><th>base</th><th>target</th><th>delta</th><th>share delta</th></tr></thead>"
                f"<tbody>{category_delta_rows or '<tr><td colspan=\"5\">No changed categories</td></tr>'}</tbody></table>"
            ),
            "</div>",
            "</div>",
            "</section>",
        ]

    fusion_section: List[str] = []
    if len(rendered) >= 2:
        base_item = rendered[0]
        target_item = rendered[1]
        base_payload = dict(base_item.get("payload") or {})
        target_payload = dict(target_item.get("payload") or {})
        findings = _build_fusion_findings(base_payload, target_payload)
        compare_debug(
            "fusion heuristic baseline={} target={} findings={}".format(
                str(base_item.get("label") or ""),
                str(target_item.get("label") or ""),
                len(findings),
            )
        )
        fusion_cards: List[str] = []
        if findings:
            for idx, item in enumerate(findings):
                rank = item.get("rank")
                dev = _to_int(item.get("device_id"), -1)
                sid = _to_int(item.get("stream_id"), 0)
                base_segment = list(item.get("base_segment") or [])
                target_segment = list(item.get("target_segment") or [])
                base_names = [
                    f"<code title='{html.escape(_normalize_kernel_name_for_compare(k.get('kernel_name')))}'>{html.escape(_short_kernel_name(k.get('kernel_name')))}</code>"
                    for k in base_segment
                ]
                target_names = [
                    f"<code title='{html.escape(_normalize_kernel_name_for_compare(k.get('kernel_name')))}'>{html.escape(_short_kernel_name(k.get('kernel_name')))}</code>"
                    for k in target_segment
                ]
                if str(item.get("kind")) == "target_fused":
                    verdict = "Possible Fusion In Target"
                elif str(item.get("kind")) == "target_split":
                    verdict = "Possible Split In Target"
                else:
                    verdict = "Target Sequence Changed"
                rank_text = "Rank Unknown" if rank is None else f"Rank {int(rank)}"
                evidence = ", ".join(str(x) for x in list(item.get("evidence") or []) if str(x or "").strip())
                fusion_cards.extend(
                    [
                        "<div class='fusion-card'>",
                        (
                            f"<div class='fusion-head'><span class='fusion-badge'>{html.escape(verdict)}</span> "
                            f"<span class='fusion-meta'>{html.escape(rank_text)} | Device {dev} | stream {sid} | confidence={html.escape(str(item.get('confidence') or ''))} | score={float(item.get('score') or 0.0):.3f}</span></div>"
                        ),
                        (
                            f"<div class='fusion-anchor'>anchor: "
                            f"<code title='{html.escape(str(item.get('prev_anchor') or ''))}'>{html.escape(_short_kernel_name(item.get('prev_anchor')))}</code> "
                            f"&rarr; "
                            f"<code title='{html.escape(str(item.get('next_anchor') or ''))}'>{html.escape(_short_kernel_name(item.get('next_anchor')))}</code>"
                            "</div>"
                        ),
                        (
                            f"<div class='fusion-anchor'>evidence: {html.escape(evidence or 'none')} | "
                            f"anchor={float(item.get('anchor_score') or 0.0):.3f} | "
                            f"position={float(item.get('position_score') or 0.0):.3f} | "
                            f"duration={float(item.get('duration_score') or 0.0):.3f} | "
                            f"name={float(item.get('name_score') or 0.0):.3f}</div>"
                        ),
                        "<div class='fusion-grid'>",
                        (
                            f"<div class='fusion-side'><div class='fusion-side-title'>{html.escape(str(base_item.get('label') or 'baseline'))}</div>"
                            f"<div class='fusion-side-sub'>kernels={len(base_segment)} | total_ms={float(item.get('base_duration_ms') or 0.0):.6f}</div>"
                            f"<div class='fusion-flow'>{' <span class=\"fusion-arrow\">&rarr;</span> '.join(base_names)}</div></div>"
                        ),
                        (
                            f"<div class='fusion-side'><div class='fusion-side-title'>{html.escape(str(target_item.get('label') or 'target'))}</div>"
                            f"<div class='fusion-side-sub'>kernels={len(target_segment)} | total_ms={float(item.get('target_duration_ms') or 0.0):.6f}</div>"
                            f"<div class='fusion-flow'>{' <span class=\"fusion-arrow\">&rarr;</span> '.join(target_names)}</div></div>"
                        ),
                        "</div>",
                        "</div>",
                    ]
                )
        else:
            fusion_cards.extend(
                [
                    "<div class='fusion-empty'>",
                    "No strong fusion candidates detected between the first two sqlite files. "
                    "This heuristic only reports segments where the per-stream kernel sequence diverges "
                    "between the same preceding and following anchor kernels.",
                    "</div>",
                ]
            )
        fusion_section = [
            "<section class='compare-section'>",
            "<h3 class='compare-section-title'>Potential Fusion Mapping</h3>",
            (
                "<div class='compare-section-note'>"
                "Heuristic only: compare the first two sqlite files stream-by-stream, build a monotonic alignment from exact kernel-name anchors, "
                "and keep only gaps whose anchors, relative positions, durations, and fused-name/token evidence remain consistent."
                "</div>"
            ),
            "<div class='fusion-stack'>",
            *fusion_cards,
            "</div>",
            "</section>",
        ]

    section_titles: List[str] = [
        "All Streams Overlap + Metrics Alignment",
        "Matched NVTX Scopes",
        "Kernel Timeline By Stream",
    ]
    if bool(include_metrics):
        section_titles.append("GPU Metrics In Window")

    section_blocks: List[str] = []
    for section_title in section_titles:
        section_cards: List[str] = []
        for idx, item in enumerate(rendered):
            section_srcdoc = _build_section_srcdoc(str(item["srcdoc"]), panel_title=str(section_title))
            if not section_srcdoc:
                continue
            srcdoc_attr = html.escape(str(section_srcdoc), quote=True)
            sqlite_display = html.escape(str(item["sqlite_path"]))
            label_display = html.escape(str(item["label"]))
            section_cards.extend(
                [
                    "<section class='compare-card'>",
                    "<div class='compare-head'>",
                    f"<div class='compare-index'>#{idx + 1}</div>",
                    "<div class='compare-meta'>",
                    f"<div class='compare-title'>{label_display}</div>",
                    f"<div class='compare-path'>{sqlite_display}</div>",
                    "</div>",
                    "</div>",
                    (
                        "<iframe class='compare-frame' loading='lazy' "
                        "sandbox='allow-scripts allow-same-origin' "
                        f"srcdoc=\"{srcdoc_attr}\"></iframe>"
                    ),
                    "</section>",
                ]
            )
        if not section_cards:
            continue
        section_blocks.extend(
            [
                "<section class='compare-section'>",
                f"<h3 class='compare-section-title'>{html.escape(str(section_title))}</h3>",
                "<div class='compare-section-stack'>",
                *section_cards,
                "</div>",
                "</section>",
            ]
        )

    note = (
        "Each compare section groups the same timeline panel across all sqlite files. "
        "Each embedded timeline still keeps its own local matched window."
    )
    if int(normalized_compare_span_ns) > 0:
        note += (
            " The X-axis scale is normalized across sqlite files using a shared compare span of "
            f"{int(normalized_compare_span_ns)} ns so equal-duration kernels render with equal widths."
        )
    else:
        note += " The kernel/metrics X-axis scale is already identical across the compared sqlite files."
    if str(nvtx_text or "").strip():
        note += f" Shared NVTX filter: {str(nvtx_text)}."
    page = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'/>",
            "<title>NSYS NVTX Timeline Compare</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:20px;background:#0f1116;color:#e7e9ee;}",
            ".meta{margin-bottom:14px;color:#a8afbf;font-size:13px;}",
            ".compare-root{display:flex;flex-direction:column;gap:18px;align-items:center;}",
            ".compare-section{width:min(100%, 1900px);}",
            ".compare-section-title{font-size:15px;color:#dbe6ff;margin:0 0 10px 0;}",
            ".compare-section-note{font-size:12px;color:#96a6c2;margin:0 0 10px 0;}",
            ".compare-section-stack{display:flex;flex-direction:column;gap:14px;align-items:stretch;}",
            ".compare-card{width:min(100%, 1900px);background:#171c28;border:1px solid #2a3243;border-radius:8px;padding:12px;box-sizing:border-box;}",
            ".compare-head{display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;}",
            ".compare-index{font-size:12px;color:#9bb1d6;background:#121826;border:1px solid #34415a;border-radius:999px;padding:2px 8px;line-height:18px;}",
            ".compare-title{font-size:14px;color:#e7eefc;font-weight:600;word-break:break-word;}",
            ".compare-path{font-size:11px;color:#8fa1bf;word-break:break-all;margin-top:2px;}",
            ".compare-frame{display:block;width:100%;min-height:140px;border:1px solid #2f3b53;border-radius:6px;background:#0c111c;box-sizing:border-box;}",
            ".summary-card{padding-bottom:14px;}",
            ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin-bottom:12px;}",
            ".summary-pill{background:#111827;border:1px solid #2b3b57;border-radius:6px;padding:8px 10px;}",
            ".summary-pill span{display:block;font-size:11px;color:#8fa1bf;margin-bottom:4px;}",
            ".summary-pill strong{font-size:14px;color:#edf4ff;}",
            ".summary-dual{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px;}",
            ".summary-triple{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px;}",
            ".summary-box{background:#111827;border:1px solid #283550;border-radius:6px;padding:8px;min-width:0;}",
            ".summary-box-title{font-size:12px;color:#dbe6ff;margin-bottom:6px;}",
            ".summary-table{width:100%;border-collapse:collapse;font-size:11px;color:#d7e2f7;table-layout:fixed;}",
            ".summary-table th,.summary-table td{border-top:1px solid #26334a;padding:5px 6px;text-align:left;vertical-align:top;word-break:break-word;}",
            ".summary-table thead th{border-top:none;color:#8fa1bf;font-weight:600;}",
            ".summary-table code{background:#1b2740;border:1px solid #324868;border-radius:4px;padding:1px 4px;color:#edf4ff;}",
            ".fusion-stack{display:flex;flex-direction:column;gap:10px;}",
            ".fusion-card{background:#171c28;border:1px solid #2a3243;border-radius:8px;padding:10px;}",
            ".fusion-head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px;}",
            ".fusion-badge{font-size:11px;color:#eaf2ff;background:#23314b;border:1px solid #4a6186;border-radius:999px;padding:2px 8px;}",
            ".fusion-meta{font-size:12px;color:#9fb0ca;}",
            ".fusion-anchor{font-size:12px;color:#cfd8ea;margin-bottom:8px;}",
            ".fusion-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:10px;}",
            ".fusion-side{background:#111827;border:1px solid #283550;border-radius:6px;padding:8px;}",
            ".fusion-side-title{font-size:12px;color:#dbe6ff;margin-bottom:4px;}",
            ".fusion-side-sub{font-size:11px;color:#8ea0bd;margin-bottom:6px;}",
            ".fusion-flow{font-size:11px;line-height:1.8;color:#d7e2f7;word-break:break-word;}",
            ".fusion-flow code{background:#1b2740;border:1px solid #324868;border-radius:4px;padding:2px 5px;color:#edf4ff;}",
            ".fusion-arrow{color:#6f85aa;padding:0 5px;}",
            ".fusion-empty{background:#171c28;border:1px dashed #33435f;border-radius:8px;padding:12px;color:#9fb0ca;font-size:12px;}",
            "</style></head><body>",
            "<h2>NSYS NVTX Timeline Compare</h2>",
            (
                f"<div class='meta'>sqlite_count={len(rendered)} | include_metrics={int(bool(include_metrics))} | "
                f"device_id={int(device_id)} | note={html.escape(note)}</div>"
            ),
            "<div class='compare-root'>",
            *optimization_section,
            *delta_section,
            *fusion_section,
            *section_blocks,
            "</div>",
            "<script>",
            "(function(){",
            "  const frames = Array.from(document.querySelectorAll('.compare-frame'));",
            "  const resizeFrame = (frame) => {",
            "    try {",
            "      const doc = frame.contentDocument;",
            "      if (!doc || !doc.documentElement) return;",
            "      const body = doc.body;",
            "      const h = Math.max(",
            "        body ? body.scrollHeight : 0,",
            "        doc.documentElement.scrollHeight || 0,",
            "        140",
            "      );",
            "      frame.style.height = `${Math.min(Math.max(h + 12, 140), 6000)}px`;",
            "    } catch (_) {}",
            "  };",
            "  for (const frame of frames) {",
            "    frame.addEventListener('load', () => {",
            "      resizeFrame(frame);",
            "      setTimeout(() => resizeFrame(frame), 100);",
            "      setTimeout(() => resizeFrame(frame), 500);",
            "      setTimeout(() => resizeFrame(frame), 1500);",
            "    });",
            "  }",
            "})();",
            "</script>",
            "</body></html>",
        ]
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    category_table_output = str(kernel_category_table_output or "").strip()
    if category_table_output:
        merged_rows: List[Dict[str, object]] = []
        for idx, item in enumerate(collected):
            label = str(item.get("label") or "")
            sqlite_path = str(item.get("sqlite_path") or "")
            state = dict(item.get("state") or {})
            for row in list(state.get("kernel_category_kernel_rows") or []):
                merged_rows.append(
                    {
                        "sqlite_index": int(idx),
                        "sqlite_label": label,
                        "sqlite_path": sqlite_path,
                        **dict(row),
                    }
                )
        table_path = _write_rows_table(category_table_output, merged_rows)
        compare_debug("kernel-category table wrote path={} rows={}".format(table_path, len(merged_rows)))
        if progress_cb:
            progress_cb("kernel-category-table wrote: {} rows={}".format(table_path, len(merged_rows)))
    stats_output = str(nvtx_category_stats_output or "").strip()
    if stats_output:
        payload_rows: List[Dict[str, object]] = []
        for idx, item in enumerate(collected):
            state = dict(item.get("state") or {})
            payload_rows.append(
                {
                    "sqlite_index": int(idx),
                    "sqlite_label": str(item.get("label") or ""),
                    "sqlite_path": str(item.get("sqlite_path") or ""),
                    "stats": dict(state.get("nvtx_window_category_stats") or {}),
                }
            )
        stats_path = Path(stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(payload_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        compare_debug("nvtx-category stats wrote path={} sqlite_count={}".format(str(stats_path), len(payload_rows)))
        if progress_cb:
            progress_cb("nvtx-category-stats wrote: {} sqlite_count={}".format(str(stats_path), len(payload_rows)))
    compare_debug("compare html wrote output={} sqlite_count={}".format(str(out), len(rendered)))
    return str(out)
