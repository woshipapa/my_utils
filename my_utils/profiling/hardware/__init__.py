"""Hardware capability tables shared by the nsys and ncu analysis paths."""

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
]
