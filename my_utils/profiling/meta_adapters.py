# profiling/adapters.py
# Purpose: Extract lightweight profiling metadata (iter/microbatch info) from
# call args/kwargs to support capture matching logic.
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def extract_meta_from_call(
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    *,
    default_iter: Optional[int] = None
) -> Dict[str, Any]:
    """
    Best-effort extraction of meta info from common inputs:
    - DataProto.meta_info["microbatch_index"], ["num_microbatches"]
    - optionally fallback to default_iter
    """
    meta: Dict[str, Any] = {}
    if default_iter is not None:
        meta["iter"] = int(default_iter)

    # scan positional + keyword values for something with .meta_info dict
    for obj in list(args) + list(kwargs.values()):
        mi = getattr(obj, "meta_info", None)
        if isinstance(mi, dict):
            if "microbatch_index" in mi:
                meta["microbatch_index"] = mi["microbatch_index"]
            if "num_microbatches" in mi:
                meta["num_microbatches"] = mi["num_microbatches"]
            # you may add more keys here
            break

    return meta
