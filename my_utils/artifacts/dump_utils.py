import os
import json

import torch


class DumpTensorIO:
    def __init__(
        self,
        tensor_dir_env="WAN_DPO_PREVAE_TENSOR_DIR",
        default_tensor_dir="dpo_dumps",
    ):
        self.tensor_dir = os.environ.get(tensor_dir_env, default_tensor_dir)
        self._cache = {}

    def _safe_tag(self, tag):
        return str(tag).replace("/", "_").replace(" ", "_")

    def _tensor_path(self, dump_id, tag, rank):
        safe_tag = self._safe_tag(tag)
        return os.path.join(self.tensor_dir, f"{int(dump_id):04d}_{safe_tag}_rank{int(rank)}.pt")

    def tensor_path(self, dump_id, tag, rank):
        return self._tensor_path(dump_id, tag, rank)

    def _compare_path(self, rank):
        path = os.environ.get("WAN_DPO_PREVAE_COMPARE_FILE")
        if path:
            if "{rank}" in path:
                path = path.format(rank=int(rank))
            return path
        return os.path.join(self.tensor_dir, f"prevae_compare_rank{int(rank)}.jsonl")

    def load_tensors(self, dump_id, tag, rank, map_location="cpu"):
        path = self._tensor_path(dump_id, tag, rank)
        if path in self._cache:
            return self._cache[path]
        if not os.path.exists(path):
            return None
        payload = torch.load(path, map_location=map_location)
        self._cache[path] = payload
        return payload

    def write_compare_result(self, record, rank):
        path = self._compare_path(rank)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def compare_tensors(self, expected, actual, rtol=1e-5, atol=1e-8):
        if expected is None:
            return {"missing": True}
        if not torch.is_tensor(expected) or not torch.is_tensor(actual):
            return {"type_mismatch": True}
        if expected.shape != actual.shape:
            return {
                "shape_mismatch": True,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
            }
        diff = (actual - expected).abs()
        return {
            "missing": False,
            "allclose": bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
        }





import os
import re
import json
import time
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Optional imports (only used if available)
try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None


def _now_ts():
    # e.g. 20260123_142355_123456
    return time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time()%1)*1e6):06d}"


def _safe_name(s: str, max_len: int = 80) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:max_len].strip("_") or "x"


def _is_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def _short_type(x: Any) -> str:
    t = type(x)
    return f"{t.__module__}.{t.__name__}"


def _tensor_meta(t) -> Dict[str, Any]:
    meta = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "requires_grad": bool(getattr(t, "requires_grad", False)),
    }
    return meta


def _np_meta(a) -> Dict[str, Any]:
    return {"shape": list(a.shape), "dtype": str(a.dtype)}


def _pil_meta(img) -> Dict[str, Any]:
    return {"size": list(img.size), "mode": img.mode}


@dataclass
class DumpConfig:
    root_dir: str = "./dump_dataset"
    enable: bool = True
    # 控制避免 dump 太大
    max_list_elems: int = 64          # list/tuple 最多保存前 N 个元素
    max_dict_items: int = 128         # dict 最多保存前 N 个键
    max_str_len: int = 10_000         # 超长字符串截断写入
    save_tensor_cpu: bool = False     # True: tensor先搬到cpu再保存，降低GPU占用/失败风险
    tensor_save_format: str = "pt"    # "pt" 或 "pt+np"
    image_format: str = "png"         # "png" / "jpg"
    print_summary: bool = True
    # 每次 dump 是否附带一个 summary.json
    write_summary_json: bool = True


class UniversalDumper:
    """
    Universal dumper that recursively serializes intermediate data of a dataset pipeline.

    Layout:
      root_dir/
        sample_<data_id>/
          <stage>/
            manifest.json
            data/...
    """
    def __init__(self, cfg: DumpConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.root_dir, exist_ok=True)

    def dump(
        self,
        stage: str,
        obj: Any,
        data_id: int,
        branch: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Dump obj into:
          root_dir/sample_<data_id>/<ts>__<stage>__<branch?>/
        Return the dump directory path.
        """
        if not self.cfg.enable:
            return ""

        ts = _now_ts()
        stage_s = _safe_name(stage)
        branch_s = _safe_name(branch) if branch is not None else None

        sample_dir = os.path.join(self.cfg.root_dir, f"sample_{int(data_id)}")
        os.makedirs(sample_dir, exist_ok=True)

        folder_name = f"{ts}__{stage_s}"
        if branch_s:
            folder_name += f"__{branch_s}"

        dump_dir = os.path.join(sample_dir, folder_name)
        data_dir = os.path.join(dump_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        manifest = {
            "timestamp": ts,
            "stage": stage,
            "data_id": int(data_id),
            "branch": branch,
            "root": dump_dir,
            "object_type": _short_type(obj),
            "extra": extra or {},
        }

        if self.cfg.print_summary:
            print(self._format_log_prefix(manifest) + self._brief(obj))

        # Save object recursively
        saved_root = self._save_any(obj, data_dir, name="root")

        manifest["saved_root"] = saved_root

        # Write manifest
        with open(os.path.join(dump_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return dump_dir

    def _format_log_prefix(self, manifest: Dict[str, Any]) -> str:
        b = f", branch={manifest['branch']}" if manifest.get("branch") is not None else ""
        return f"[DUMP] id={manifest['data_id']}{b}, stage={manifest['stage']}: "

    def _brief(self, obj: Any) -> str:
        # lightweight structure summary for printing
        try:
            if _is_primitive(obj):
                s = obj if not isinstance(obj, str) else (obj[:200] + ("..." if len(obj) > 200 else ""))
                return f"type={_short_type(obj)}, value={repr(s)}"
            if Image is not None and isinstance(obj, Image.Image):
                return f"type=PIL.Image, meta={_pil_meta(obj)}"
            if torch is not None and isinstance(obj, torch.Tensor):
                return f"type=torch.Tensor, meta={_tensor_meta(obj)}"
            if np is not None and isinstance(obj, np.ndarray):
                return f"type=np.ndarray, meta={_np_meta(obj)}"
            if isinstance(obj, dict):
                keys = list(obj.keys())
                keys_show = keys[:20]
                return f"type=dict, nkeys={len(keys)}, keys(head)={keys_show}"
            if isinstance(obj, (list, tuple)):
                return f"type={type(obj).__name__}, len={len(obj)}"
            return f"type={_short_type(obj)}"
        except Exception as e:
            return f"type={_short_type(obj)}, brief_error={repr(e)}"

    def _save_any(self, obj: Any, out_dir: str, name: str) -> Dict[str, Any]:
        """
        Save obj to out_dir. Return a small metadata dict describing where/how it was saved.
        """
        os.makedirs(out_dir, exist_ok=True)
        safe = _safe_name(name)

        # 1) primitives -> json
        if _is_primitive(obj):
            path = os.path.join(out_dir, f"{safe}.json")
            val = obj
            if isinstance(val, str) and len(val) > self.cfg.max_str_len:
                val = val[: self.cfg.max_str_len] + "...<TRUNCATED>"
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"value": val, "type": _short_type(obj)}, f, ensure_ascii=False, indent=2)
            return {"kind": "primitive_json", "path": path}

        # 2) PIL Image -> image file
        if Image is not None and isinstance(obj, Image.Image):
            # 统一模式，保证可比（按需改成 "RGBA"）
            try:
                if obj.mode != "RGB":
                    obj = obj.convert("RGB")
            except Exception:
                pass

            # 强烈建议 PNG：无损、可逐像素比较
            ext = self.cfg.image_format.lower()
            if ext in ("jpg", "jpeg"):
                # 你也可以选择：直接不允许 jpg，强制 png
                # ext = "png"
                pass

            path = os.path.join(out_dir, f"{safe}.{ext}")
            try:
                if ext in ("jpg", "jpeg"):
                    obj.save(path, format="JPEG", quality=95)  # 有损，不适合 strict compare
                else:
                    obj.save(path, format="PNG")
            except Exception:
                path = os.path.join(out_dir, f"{safe}.png")
                obj.save(path, format="PNG")

            meta_path = os.path.join(out_dir, f"{safe}__meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"type": "PIL.Image", "meta": _pil_meta(obj)}, f, ensure_ascii=False, indent=2)

            return {"kind": "pil_image", "path": path, "meta": meta_path}


        # 3) torch.Tensor -> pt (and optional npy)
        if torch is not None and isinstance(obj, torch.Tensor):
            t = obj
            if self.cfg.save_tensor_cpu:
                try:
                    t = t.detach().cpu()
                except Exception:
                    pass
            else:
                try:
                    t = t.detach()
                except Exception:
                    pass

            pt_path = os.path.join(out_dir, f"{safe}.pt")
            torch.save(t, pt_path)
            meta_path = os.path.join(out_dir, f"{safe}__meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"type": "torch.Tensor", "meta": _tensor_meta(t)}, f, ensure_ascii=False, indent=2)

            saved = {"kind": "torch_tensor", "pt": pt_path, "meta": meta_path}

            if self.cfg.tensor_save_format == "pt+np" and np is not None:
                try:
                    npy_path = os.path.join(out_dir, f"{safe}.npy")
                    np.save(npy_path, t.cpu().numpy())
                    saved["npy"] = npy_path
                except Exception:
                    pass
            return saved

        # 4) numpy -> npy
        if np is not None and isinstance(obj, np.ndarray):
            path = os.path.join(out_dir, f"{safe}.npy")
            np.save(path, obj)
            meta_path = os.path.join(out_dir, f"{safe}__meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"type": "np.ndarray", "meta": _np_meta(obj)}, f, ensure_ascii=False, indent=2)
            return {"kind": "numpy", "path": path, "meta": meta_path}

        # 5) dict -> folder of entries + index
        if isinstance(obj, dict):
            d = obj
            items = list(d.items())
            if len(items) > self.cfg.max_dict_items:
                items = items[: self.cfg.max_dict_items]
                truncated = True
            else:
                truncated = False

            dir_path = os.path.join(out_dir, f"{safe}__dict")
            os.makedirs(dir_path, exist_ok=True)

            index = {"kind": "dict", "truncated": truncated, "nkeys_total": len(d), "items": []}
            for k, v in items:
                k_safe = _safe_name(k)
                child = self._save_any(v, dir_path, name=k_safe)
                index["items"].append({"key": str(k), "saved": child})

            idx_path = os.path.join(dir_path, "_index.json")
            with open(idx_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

            if self.cfg.write_summary_json:
                # also write a quick "keys + types"
                summary = {str(k): _short_type(v) for k, v in items}
                with open(os.path.join(dir_path, "_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {"truncated": truncated, "nkeys_total": len(d), "key_types": summary},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

            return {"kind": "dict_dir", "path": dir_path, "index": idx_path}

        # 6) list/tuple -> folder of elems + index
        if isinstance(obj, (list, tuple)):
            seq = obj
            elems = list(seq)
            if len(elems) > self.cfg.max_list_elems:
                elems = elems[: self.cfg.max_list_elems]
                truncated = True
            else:
                truncated = False

            dir_path = os.path.join(out_dir, f"{safe}__{type(seq).__name__}")
            os.makedirs(dir_path, exist_ok=True)

            index = {
                "kind": type(seq).__name__,
                "truncated": truncated,
                "len_total": len(seq),
                "elems": [],
            }
            for i, v in enumerate(elems):
                child = self._save_any(v, dir_path, name=f"{i:04d}")
                index["elems"].append({"i": i, "saved": child})

            idx_path = os.path.join(dir_path, "_index.json")
            with open(idx_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)

            return {"kind": "seq_dir", "path": dir_path, "index": idx_path}

        # 7) fallback -> pickle
        path = os.path.join(out_dir, f"{safe}.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(obj, f)
            return {"kind": "pickle", "path": path, "type": _short_type(obj)}
        except Exception as e:
            # worst-case: write repr
            txt_path = os.path.join(out_dir, f"{safe}__repr.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"type={_short_type(obj)}\nerror={repr(e)}\n\nrepr:\n{repr(obj)}\n")
            return {"kind": "repr_txt", "path": txt_path, "type": _short_type(obj)}



import os
import threading
from typing import Optional

# 假设你原来这两个类在同一个文件里
# from dump_utils import UniversalDumper, DumpConfig

class DumperSingleton:
    """
    Process-level singleton for UniversalDumper.
    - In one process: always returns the same dumper instance.
    - In multi-process (DataLoader workers / torchrun): each process has its own singleton.
    """
    _lock = threading.Lock()
    _instance: Optional["UniversalDumper"] = None
    _cfg_fingerprint: Optional[str] = None

    @classmethod
    def get(cls, cfg=None):
        """
        Get the singleton dumper. The first call can pass cfg; later calls will reuse it.

        Args:
            cfg: DumpConfig or None. If None and not initialized, a default DumpConfig is used.
        """
        with cls._lock:
            if cls._instance is None:
                if cfg is None:
                    cfg = DumpConfig()

                # （可选）给 root_dir 自动加上 pid，避免多进程写到同一个目录导致目录名极端碰撞/难排查
                # 你也可以注释掉这一段，继续让所有进程写同一个 root_dir
                try:
                    pid = os.getpid()
                    cfg.root_dir = os.path.join(cfg.root_dir, f"pid_{pid}")
                except Exception:
                    pass

                cls._instance = UniversalDumper(cfg)
                cls._cfg_fingerprint = repr(cfg)

            else:
                # 如果后续有人传了不同 cfg，直接忽略/或你也可以选择报错
                if cfg is not None:
                    new_fp = repr(cfg)
                    if cls._cfg_fingerprint != new_fp:
                        # 不报错也行，但打印一下很有用
                        print(
                            "[DumperSingleton] Warning: dumper already initialized with a different cfg. "
                            "New cfg is ignored."
                        )

            return cls._instance

    @classmethod
    def reset_for_tests(cls):
        """Optional: reset singleton (useful in unit tests)."""
        with cls._lock:
            cls._instance = None
            cls._cfg_fingerprint = None


def get_dumper(cfg=None):
    """Functional alias."""
    return DumperSingleton.get(cfg)






import os
import json
import glob
import pickle

try:
    import torch
except Exception:
    torch = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None


class UniversalLoader:
    """
    Load objects saved by UniversalDumper.
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def list_dumps(self, data_id: int):
        """Return all dump dirs for a sample id, sorted by folder name."""
        sample_dir = os.path.join(self.root_dir, f"sample_{int(data_id)}")
        if not os.path.isdir(sample_dir):
            return []
        dirs = [os.path.join(sample_dir, d) for d in os.listdir(sample_dir)]
        dirs = [d for d in dirs if os.path.isdir(d) and os.path.isfile(os.path.join(d, "manifest.json"))]
        return sorted(dirs)

    def find_dump(self, data_id: int, stage: str, branch: str = None, pick: str = "latest"):
        """
        Find a dump directory for (data_id, stage, branch).
        pick: "latest" or "earliest"
        """
        dirs = self.list_dumps(data_id)
        stage = str(stage)
        branch = None if branch is None else str(branch)

        matched = []
        for d in dirs:
            with open(os.path.join(d, "manifest.json"), "r", encoding="utf-8") as f:
                m = json.load(f)
            if m.get("stage") != stage:
                continue
            if branch is not None and m.get("branch") != branch:
                continue
            if branch is None and m.get("branch") is not None:
                # 如果你想 stage 不区分 branch，这里可以放宽；默认严格：branch=None 只匹配 branch=None
                continue
            matched.append((d, m))

        if not matched:
            return None

        matched = sorted(matched, key=lambda x: x[0])
        return matched[-1][0] if pick == "latest" else matched[0][0]

    def load(self, dump_dir: str):
        """Load the root object from a dump dir (the one referred by manifest['saved_root'])."""
        with open(os.path.join(dump_dir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)
        saved_root = manifest["saved_root"]
        return self._load_saved(saved_root)

    def load_stage(self, data_id: int, stage: str, branch: str = None, pick: str = "latest"):
        dump_dir = self.find_dump(data_id, stage=stage, branch=branch, pick=pick)
        if dump_dir is None:
            raise FileNotFoundError(f"No dump found for data_id={data_id}, stage={stage}, branch={branch}")
        return self.load(dump_dir)

    # ---------- internals ----------

    def _load_saved(self, saved: dict):
        kind = saved.get("kind")

        if kind == "primitive_json":
            with open(saved["path"], "r", encoding="utf-8") as f:
                return json.load(f)["value"]

        if kind == "pil_image":
            if Image is None:
                raise RuntimeError("PIL not available but trying to load pil_image")
            img = Image.open(saved["path"])
            img.load()
            return img

        if kind == "torch_tensor":
            if torch is None:
                raise RuntimeError("torch not available but trying to load torch_tensor")
            return torch.load(saved["pt"], map_location="cpu")

        if kind == "numpy":
            if np is None:
                raise RuntimeError("numpy not available but trying to load numpy")
            return np.load(saved["path"], allow_pickle=False)

        if kind == "dict_dir":
            # read _index.json then reconstruct dict
            idx_path = saved["index"]
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            out = {}
            for it in idx["items"]:
                key = it["key"]
                out[key] = self._load_saved(it["saved"])
            return out

        if kind == "seq_dir":
            idx_path = saved["index"]
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            elems = []
            for it in idx["elems"]:
                elems.append(self._load_saved(it["saved"]))
            # 原来是 list 还是 tuple，我们这里默认恢复 list（更常用）
            return elems

        if kind == "pickle":
            with open(saved["path"], "rb") as f:
                return pickle.load(f)

        if kind == "repr_txt":
            # 不可恢复对象，返回 repr 文本
            with open(saved["path"], "r", encoding="utf-8") as f:
                return f.read()

        raise ValueError(f"Unknown saved kind: {kind}")
