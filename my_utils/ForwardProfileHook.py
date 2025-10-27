import torch
import torch.distributed as dist
from typing import Union, List, Optional

class ForwardProfilerHook:
    def __init__(
        self,
        start_iter: int = 1,
        stop_iter: int = 2,
        rank_only: Union[int, List[int]] = [0, 2],
        nvtx_range: Optional[str] = None
    ):
        self.iter_count = 0
        self.start_iter = start_iter
        self.stop_iter = stop_iter
        self.rank_only = [rank_only] if isinstance(rank_only, int) else rank_only
        self._hook_handle = None
        self.nvtx_range = nvtx_range
        self._nvtx_active = False

    def attach(self, module: torch.nn.Module):
        self._hook_handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        # 若启用分布式，检查当前 rank 是否在目标列表中
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank not in self.rank_only:
                return

        # 进入 profiling 区间
        if self.iter_count == self.start_iter:
            print(f"[ProfilerHook] Rank {dist.get_rank()} START profiling at iter {self.iter_count}")
            torch.cuda.cudart().cudaProfilerStart()
            if self.nvtx_range:
                torch.cuda.nvtx.range_push(f"[{self.nvtx_range}]")

            self._nvtx_active = True

        # 退出 profiling 区间
        if self.iter_count == self.stop_iter:
            print(f"[ProfilerHook] Rank {dist.get_rank()} STOP profiling at iter {self.iter_count}")
            if self._nvtx_active:
                torch.cuda.nvtx.range_pop()
                self._nvtx_active = False
            torch.cuda.cudart().cudaProfilerStop()
            self._hook_handle.remove()

        self.iter_count += 1
