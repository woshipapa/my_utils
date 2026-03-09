import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn


def _tensor_bytes_sha256(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_bytes_md5(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    return hashlib.md5(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def _to_cpu_detached(x: Any) -> Any:
    """Detach tensors and move to CPU; keep structure for (list/tuple/dict)."""
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu_detached(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_cpu_detached(v) for k, v in x.items()}
    return x


def _flatten_tensors(x: Any, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    """
    Flatten nested structure to list of (key, tensor).
    key is a stable path like "tensor", "0", "0.key", etc.
    """
    out: List[Tuple[str, torch.Tensor]] = []
    if torch.is_tensor(x):
        out.append((prefix or "tensor", x))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.extend(_flatten_tensors(v, f"{prefix}.{i}" if prefix else str(i)))
    elif isinstance(x, dict):
        for k in sorted(x.keys(), key=lambda z: str(z)):
            v = x[k]
            out.extend(_flatten_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    return out


def _sample_tensor(
    t: torch.Tensor,
    mode: str = "none",
    max_elems: int = 200_000,
    seed: int = 0,
) -> torch.Tensor:
    """
    Reduce tensor size for saving.
    mode:
      - "none": save full tensor (may be huge)
      - "head": save first max_elems in flattened order
      - "rand": random sample of max_elems elements (deterministic with seed)
    """
    t = t.detach()
    if mode == "none":
        return t
    flat = t.flatten()
    n = flat.numel()
    if n <= max_elems:
        return t
    if mode == "head":
        return flat[:max_elems].clone()
    if mode == "rand":
        g = torch.Generator(device=flat.device)
        g.manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=flat.device)[:max_elems]
        return flat[idx].clone()
    raise ValueError(f"unknown sample mode: {mode}")


def _dtype_counts_str(dtypes: List[str]) -> Dict[str, int]:
    return dict(Counter(dtypes))


def _module_param_dtype_runtime(module: nn.Module) -> Dict[str, Any]:
    """
    Capture *current* (runtime) param dtypes for this module (recurse=False).
    """
    dtypes = [str(p.dtype) for p in module.parameters(recurse=False)]
    return {
        "first": dtypes[0] if dtypes else None,
        "counts": _dtype_counts_str(dtypes) if dtypes else {},
        "num_params": len(dtypes),
    }


def _module_buffer_dtype_runtime(module: nn.Module) -> Dict[str, Any]:
    dtypes = [str(b.dtype) for b in module.buffers(recurse=False)]
    return {
        "first": dtypes[0] if dtypes else None,
        "counts": _dtype_counts_str(dtypes) if dtypes else {},
        "num_buffers": len(dtypes),
    }


def _hash_named_tensors_md5(named_tensors: List[Tuple[str, torch.Tensor]]) -> Dict[str, Any]:
    """
    Stable MD5 for a list of (name, tensor):
      - per-tensor md5 is over raw bytes (cpu/contiguous)
      - module-level md5 combines: name + shape + dtype + bytes for each tensor, in sorted(name) order
    """
    # stable order
    items = sorted(named_tensors, key=lambda kv: kv[0])

    per = {}
    h = hashlib.md5()
    total_num = 0

    for name, t in items:
        if not torch.is_tensor(t):
            continue
        tt = t.detach().cpu().contiguous()
        total_num += 1

        # per-tensor md5
        md5_i = hashlib.md5(tt.view(torch.uint8).numpy().tobytes()).hexdigest()
        per[name] = {
            "md5": md5_i,
            "shape": list(tt.shape),
            "dtype": str(tt.dtype),
            "numel": int(tt.numel()),
        }

        # feed module md5 with metadata + raw bytes
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tt.shape)).encode("utf-8"))
        h.update(str(tt.dtype).encode("utf-8"))
        h.update(tt.view(torch.uint8).numpy().tobytes())

    return {
        "num_tensors": total_num,
        "module_md5": h.hexdigest() if total_num > 0 else None,
        "per_tensor": per,
    }


def _module_param_md5_runtime(module: nn.Module) -> Dict[str, Any]:
    """
    MD5 snapshot of this module's *own* params (recurse=False) at runtime.
    """
    named = [(n, p) for n, p in module.named_parameters(recurse=False)]
    return _hash_named_tensors_md5(named)


def _module_buffer_md5_runtime(module: nn.Module) -> Dict[str, Any]:
    """
    MD5 snapshot of this module's *own* buffers (recurse=False) at runtime.
    """
    named = [(n, b) for n, b in module.named_buffers(recurse=False)]
    return _hash_named_tensors_md5(named)


@dataclass
class TraceTensorMeta:
    name: str              # module qualified name
    kind: str              # "in" or "out"
    key: str               # nested key inside in/out structure
    shape: List[int]
    dtype: str
    device: str
    sha256: str
    min: float
    max: float
    mean: float


class ForwardTraceRecorder:
    """
    Ordered forward trace recorder:
    - stores per-call events in a list (execution order)
    - supports modules called multiple times
    - stores module param/buffer dtype + md5 snapshot at every hook call (recurse=False)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        name: str,
        record_inputs: bool = False,
        record_outputs: bool = True,
        include_name_regex: Optional[str] = None,
        exclude_name_regex: Optional[str] = None,
        include_module_types: Optional[Tuple[type, ...]] = None,
        exclude_module_types: Optional[Tuple[type, ...]] = (nn.Sequential, nn.ModuleList, nn.ModuleDict),
        sample_mode: str = "none",
        sample_max_elems: int = 200_000,
        sample_seed: int = 0,
        cast_float32: bool = False,
        save_stats_only: bool = False,
        max_modules: Optional[int] = None,
        verbose: bool = False,
        # dtype knobs
        record_param_dtypes_each_call: bool = True,
        record_buffer_dtypes_each_call: bool = True,
        # md5 knobs
        record_param_md5_each_call: bool = True,
        record_buffer_md5_each_call: bool = False,  # 默认关掉，开了更重
        record_param_md5_per_param: bool = True,
        record_buffer_md5_per_buffer: bool = False,
    ):
        self.model = model
        self.name = name
        self.record_inputs = record_inputs
        self.record_outputs = record_outputs
        self.include_re = re.compile(include_name_regex) if include_name_regex else None
        self.exclude_re = re.compile(exclude_name_regex) if exclude_name_regex else None
        self.include_types = include_module_types
        self.exclude_types = exclude_module_types
        self.sample_mode = sample_mode
        self.sample_max_elems = sample_max_elems
        self.sample_seed = sample_seed
        self.cast_float32 = cast_float32
        self.save_stats_only = save_stats_only
        self.max_modules = max_modules
        self.verbose = verbose

        self.record_param_dtypes_each_call = record_param_dtypes_each_call
        self.record_buffer_dtypes_each_call = record_buffer_dtypes_each_call
        self.record_param_md5_each_call = record_param_md5_each_call
        self.record_buffer_md5_each_call = record_buffer_md5_each_call
        self.record_param_md5_per_param = record_param_md5_per_param
        self.record_buffer_md5_per_buffer = record_buffer_md5_per_buffer

        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._module_count = 0
        self._start_time = None

        self.events: List[Dict[str, Any]] = []
        self._call_idx = 0

    def _should_hook(self, module_name: str, module: nn.Module) -> bool:
        if module is self.model:
            return False
        if self.include_re and not self.include_re.search(module_name):
            return False
        if self.exclude_re and self.exclude_re.search(module_name):
            return False
        if self.include_types and not isinstance(module, self.include_types):
            return False
        if self.exclude_types and isinstance(module, self.exclude_types):
            return False
        return True

    def _process(self, module_name: str, kind: str, obj: Any):
        cpu_obj = _to_cpu_detached(obj)
        flat = _flatten_tensors(cpu_obj, prefix="")

        store: Dict[str, torch.Tensor] = {}
        metas: List[Dict[str, Any]] = []

        for key, t in flat:
            if not torch.is_tensor(t):
                continue
            tt = t.contiguous()
            if self.cast_float32:
                tt = tt.float()

            sampled = _sample_tensor(
                tt,
                mode=self.sample_mode,
                max_elems=self.sample_max_elems,
                seed=self.sample_seed,
            )

            meta = TraceTensorMeta(
                name=module_name,
                kind=kind,
                key=key,
                shape=list(tt.shape),
                dtype=str(tt.dtype),
                device=str(tt.device),
                sha256=_tensor_bytes_sha256(sampled),
                min=float(tt.float().min().item()) if tt.numel() else 0.0,
                max=float(tt.float().max().item()) if tt.numel() else 0.0,
                mean=float(tt.float().mean().item()) if tt.numel() else 0.0,
            )
            metas.append(asdict(meta))

            if not self.save_stats_only:
                store[key] = sampled

        return store, metas

    def _hook_fn(self, module_name: str):
        def fn(module: nn.Module, inputs: Tuple[Any, ...], outputs: Any):
            call_idx = self._call_idx
            self._call_idx += 1

            ev: Dict[str, Any] = {
                "idx": call_idx,
                "module_name": module_name,
                "module_type": module.__class__.__name__,
                "time": time.time(),

                # dtype snapshots
                "param_dtypes": None,
                "buffer_dtypes": None,

                # md5 snapshots
                "param_md5": None,               # overall module md5 (params)
                "param_md5_per_param": None,     # per-param md5 map
                "buffer_md5": None,              # overall module md5 (buffers)
                "buffer_md5_per_buffer": None,   # per-buffer md5 map

                "inputs": None,
                "outputs": None,
                "meta": [],
            }

            # dtype snapshots
            if self.record_param_dtypes_each_call:
                ev["param_dtypes"] = _module_param_dtype_runtime(module)
            if self.record_buffer_dtypes_each_call:
                ev["buffer_dtypes"] = _module_buffer_dtype_runtime(module)

            # md5 snapshots (params / buffers)
            if self.record_param_md5_each_call:
                pm = _module_param_md5_runtime(module)
                ev["param_md5"] = pm["module_md5"]
                if self.record_param_md5_per_param:
                    ev["param_md5_per_param"] = pm["per_tensor"]
            if self.record_buffer_md5_each_call:
                bm = _module_buffer_md5_runtime(module)
                ev["buffer_md5"] = bm["module_md5"]
                if self.record_buffer_md5_per_buffer:
                    ev["buffer_md5_per_buffer"] = bm["per_tensor"]

            # tensor IO
            if self.record_inputs:
                store_in, metas_in = self._process(module_name, "in", inputs)
                ev["inputs"] = store_in
                ev["meta"].extend(metas_in)

            if self.record_outputs:
                store_out, metas_out = self._process(module_name, "out", outputs)
                ev["outputs"] = store_out
                ev["meta"].extend(metas_out)

            self.events.append(ev)

        return fn

    def install(self) -> None:
        self._start_time = time.time()
        for module_name, module in self.model.named_modules():
            if self.max_modules is not None and self._module_count >= self.max_modules:
                break
            if not self._should_hook(module_name, module):
                continue
            h = module.register_forward_hook(self._hook_fn(module_name))
            self.handles.append(h)
            self._module_count += 1
            if self.verbose:
                print(f"[trace:{self.name}] hook {module_name}: {module.__class__.__name__}")

        if self.verbose:
            print(f"[trace:{self.name}] installed hooks on {self._module_count} modules")

    def remove(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def save(self, path: Union[str, Path], extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "trace_name": self.name,
            "num_modules_hooked": self._module_count,
            "num_events": len(self.events),
            "elapsed_sec": (time.time() - self._start_time) if self._start_time else None,

            "record_inputs": self.record_inputs,
            "record_outputs": self.record_outputs,
            "cast_float32": self.cast_float32,
            "sample_mode": self.sample_mode,
            "sample_max_elems": self.sample_max_elems,
            "sample_seed": self.sample_seed,
            "save_stats_only": self.save_stats_only,

            # dtype knobs
            "record_param_dtypes_each_call": self.record_param_dtypes_each_call,
            "record_buffer_dtypes_each_call": self.record_buffer_dtypes_each_call,

            # md5 knobs
            "record_param_md5_each_call": self.record_param_md5_each_call,
            "record_buffer_md5_each_call": self.record_buffer_md5_each_call,
            "record_param_md5_per_param": self.record_param_md5_per_param,
            "record_buffer_md5_per_buffer": self.record_buffer_md5_per_buffer,

            "events": self.events,
            "extra": extra or {},
        }
        torch.save(payload, str(path))
        if self.verbose:
            print(f"[trace:{self.name}] saved to {path}")
