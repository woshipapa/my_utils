from __future__ import annotations

from typing import Dict, Optional


# Common peak TFLOPS references used for MFU estimation.
# Values are theoretical peak throughput and can vary by SKU/clock/config.
_PEAK_TFLOPS_FP16: Dict[str, float] = {
    "h100": 989.0,
    "h200": 989.0,
    "a100": 312.0,
    "l40s": 362.0,
    "l40": 181.0,
    "a800": 312.0,
    "h800": 989.0,
}

_PEAK_TFLOPS_BF16: Dict[str, float] = {
    "h100": 989.0,
    "h200": 989.0,
    "a100": 312.0,
    "l40s": 362.0,
    "l40": 181.0,
    "a800": 312.0,
    "h800": 989.0,
}

_PEAK_TFLOPS_FP32: Dict[str, float] = {
    "h100": 67.0,
    "h200": 67.0,
    "a100": 19.5,
    "l40s": 91.0,
    "l40": 91.0,
    "a800": 19.5,
    "h800": 67.0,
}


def infer_peak_tflops(gpu_name: str, *, precision: str = "fp16") -> Optional[float]:
    text = str(gpu_name or "").strip().lower()
    if not text:
        return None
    if "h200" in text:
        key = "h200"
    elif "h100" in text:
        key = "h100"
    elif "h800" in text:
        key = "h800"
    elif "a800" in text:
        key = "a800"
    elif "a100" in text:
        key = "a100"
    elif "l40s" in text:
        key = "l40s"
    elif "l40" in text:
        key = "l40"
    else:
        return None

    p = str(precision or "fp16").strip().lower()
    if p == "fp32":
        return _PEAK_TFLOPS_FP32.get(key)
    if p == "bf16":
        return _PEAK_TFLOPS_BF16.get(key)
    return _PEAK_TFLOPS_FP16.get(key)


def compute_mfu_single(
    *,
    step_time_s: float,
    model_flops_per_step: float,
    peak_tflops: float,
) -> Dict[str, object]:
    if model_flops_per_step <= 0 or peak_tflops <= 0:
        return {
            "error": "model_flops_per_step and peak_tflops must be positive.",
            "formula": "MFU = ((model_flops_per_step / step_time_s) / 1e12) / peak_tflops",
        }
    if step_time_s <= 0:
        return {
            "error": "step_time_s must be positive.",
            "formula": "MFU = ((model_flops_per_step / step_time_s) / 1e12) / peak_tflops",
        }

    achieved_tflops = (model_flops_per_step / step_time_s) / 1e12
    mfu_pct = 100.0 * achieved_tflops / peak_tflops
    return {
        "step_time_s": float(step_time_s),
        "model_flops_per_step": float(model_flops_per_step),
        "peak_tflops": float(peak_tflops),
        "achieved_model_tflops": round(float(achieved_tflops), 4),
        "mfu_pct": round(float(mfu_pct), 4),
    }


def compute_mfu_compare(
    *,
    step_time_before_s: float,
    step_time_after_s: float,
    model_flops_per_step: float,
    peak_tflops: float,
) -> Dict[str, object]:
    before = compute_mfu_single(
        step_time_s=step_time_before_s,
        model_flops_per_step=model_flops_per_step,
        peak_tflops=peak_tflops,
    )
    if "error" in before:
        return before
    after = compute_mfu_single(
        step_time_s=step_time_after_s,
        model_flops_per_step=model_flops_per_step,
        peak_tflops=peak_tflops,
    )
    if "error" in after:
        return after
    return {
        "before": before,
        "after": after,
        "delta_mfu_pct": round(float(after["mfu_pct"]) - float(before["mfu_pct"]), 4),
        "delta_achieved_tflops": round(
            float(after["achieved_model_tflops"]) - float(before["achieved_model_tflops"]),
            4,
        ),
    }
