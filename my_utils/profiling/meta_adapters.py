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
    Best-effort extraction:
    - iter: from default_iter
    - microbatch_index/num_microbatches: from ANY arg/kw that has .meta_info dict
    Strategy:
      - scan all candidates, merge keys if present
      - do not stop at the first meta_info dict, because it might be empty
    """
    meta: Dict[str, Any] = {}
    if default_iter is not None:
        meta["iter"] = int(default_iter)

    candidates = list(args) + list(kwargs.values())

    for obj in candidates:
        mi = getattr(obj, "meta_info", None)
        if not isinstance(mi, dict):
            continue

        # 只在出现 key 时才写入（后面如果另一个参数也有，可以补全）
        if "microbatch_index" in mi and "microbatch_index" not in meta:
            meta["microbatch_index"] = mi["microbatch_index"]
        if "num_microbatches" in mi and "num_microbatches" not in meta:
            meta["num_microbatches"] = mi["num_microbatches"]

        # 如果已经拿齐了就提前结束
        if "microbatch_index" in meta and "num_microbatches" in meta:
            break

    return meta
