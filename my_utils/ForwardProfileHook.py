import torch
import torch.distributed as dist
from typing import List, Optional, Union

from .nvtx_utils import LabelerProtocol, create_labeler


class ForwardProfilerHook:
    def __init__(
        self,
        start_iter: int = 1,
        stop_iter: int = 2,
        rank_only: Union[int, List[int]] = [0, 2],
        nvtx_range: Optional[str] = None,
        *,
        labeler: Optional[LabelerProtocol] = None,
        nvtx_domain: Optional[str] = None,
    ):
        self.iter_count = 0
        self.start_iter = start_iter
        self.stop_iter = stop_iter
        self.rank_only = [rank_only] if isinstance(rank_only, int) else rank_only
        self._hook_handle = None
        self.nvtx_range = nvtx_range
        self._nvtx_token = None
        self._labeler = labeler if labeler is not None else create_labeler(
            enabled=bool(nvtx_range),
            default_domain=nvtx_domain,
        )

    def attach(self, module: torch.nn.Module):
        self._hook_handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if dist.is_initialized() and rank not in self.rank_only:
            return

        if self.iter_count == self.start_iter:
            print(f"[ProfilerHook] Rank {rank} START profiling at iter {self.iter_count}")
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStart()
            if self.nvtx_range:
                self._nvtx_token = self._labeler.start(self.nvtx_range)

        if self.iter_count == self.stop_iter:
            print(f"[ProfilerHook] Rank {rank} STOP profiling at iter {self.iter_count}")
            if self._nvtx_token is not None:
                self._labeler.stop(self._nvtx_token)
                self._nvtx_token = None
            if torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStop()
            self._hook_handle.remove()

        self.iter_count += 1
