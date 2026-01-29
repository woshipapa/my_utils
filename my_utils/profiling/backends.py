# profiling/backends.py
# Purpose: Define capture backends for profiling, including a no-op backend and
# a CUDA cudart-based backend that starts/stops profiling with optional sync.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class CaptureBackend:
    """Backend interface for starting/stopping capture."""
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


@dataclass
class NoOpBackend(CaptureBackend):
    """Does nothing; safe when CUDA/cudart is unavailable."""
    def start(self) -> None:
        return

    def stop(self) -> None:
        return


@dataclass
class CudaProfilerBackend(CaptureBackend):
    """
    Uses cudart cudaProfilerStart/Stop via torch.cuda.cudart().
    Note: start/stop must happen in the process you want to profile (actor process).
    """
    synchronize: bool = True

    def start(self) -> None:
        import torch
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        # cudart API
        torch.cuda.cudart().cudaProfilerStart()

    def stop(self) -> None:
        import torch
        if self.synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
