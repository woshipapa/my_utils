# SPDX-License-Identifier: Apache-2.0
"""Hardware capability tables shared by the nsys and ncu analysis paths."""

from .throttling import (
    CLOCK_EVENT_REASONS,
    DCGM_FIELDS,
    THROTTLING_MASK,
    ThrottleReading,
    analyze_throttling,
    decode_clock_event_mask,
)
from .gpu_specs import (
    DTYPE_ALIASES,
    GpuSpec,
    list_known_gpus,
    lookup_gpu_spec,
    normalize_dtype,
)

__all__ = [
    "GpuSpec",
    "lookup_gpu_spec",
    "list_known_gpus",
    "DTYPE_ALIASES",
    "normalize_dtype",
    # Clock throttling: was the GPU allowed to run at the speed you assumed?
    "analyze_throttling",
    "decode_clock_event_mask",
    "ThrottleReading",
    "CLOCK_EVENT_REASONS",
    "THROTTLING_MASK",
    "DCGM_FIELDS",
]
