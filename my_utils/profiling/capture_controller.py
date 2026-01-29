# profiling/capture_controller.py
# Purpose: Orchestrate profiling capture windows by arming a spec, triggering
# start/stop on matching hook events, and optionally emitting NVTX window marks.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
import threading


@dataclass
class HookEvent:
    """
    A lightweight, framework-agnostic event.
    """
    profile_name: str
    meta: Dict[str, Any]
    # optional
    role: Optional[str] = None
    rank: Optional[int] = None


class CaptureController:
    """
    ARM/TRIGGER controller:
    - Driver broadcasts arm(spec) to all actors in a group
    - Actor triggers start() when entering a matching profiled function
    - Actor triggers stop() at function exit (default policy)
    """

    def __init__(
        self,
        backend,
        logger=None,
        *,
        enable_nvtx_window: bool = False,
        nvtx_domain: Optional[str] = None,
    ):
        self._backend = backend
        self._logger = logger

        # NVTX window markers (optional)
        self._enable_nvtx_window = enable_nvtx_window
        self._nvtx_domain = nvtx_domain

        # State
        self._armed = False
        self._active = False
        self._done = False
        self._spec: Optional[Dict[str, Any]] = None
        self._window_id: Optional[str] = None

        # Track whether current call is the one that triggered capture
        self._triggered_by_profile_name: Optional[str] = None

        # Thread safety (Ray actor is mostly single-threaded, but async/await can interleave)
        self._lock = threading.Lock()

    # -----------------------
    # Public control-plane API
    # -----------------------
    def arm(self, spec: Dict[str, Any]) -> None:
        """
        spec example:
          {
            "window_id": "iter120/G_BWD/mb31",
            "target_profile_names": ["generator_loss_and_backward"],
            "expected_iter": 120,          # optional
            "expected_mb": 31,             # optional
            "stop_policy": "ON_TARGET_FUNC_EXIT",  # recommended
            "enable_nvtx_window": True,    # optional override
          }
        """
        with self._lock:
            self._spec = dict(spec) if spec is not None else {}
            self._window_id = self._spec.get("window_id")
            self._armed = True
            self._active = False
            self._done = False
            self._triggered_by_profile_name = None

            # allow per-window nvtx setting override
            if "enable_nvtx_window" in self._spec:
                self._enable_nvtx_window = bool(self._spec["enable_nvtx_window"])

        self._log(f"[Capture] ARMED window={self._window_id} spec={self._spec}")

    def disarm(self, window_id: Optional[str] = None) -> None:
        with self._lock:
            if window_id is not None and window_id != self._window_id:
                return
            self._armed = False
            self._spec = None
            self._window_id = None
            self._triggered_by_profile_name = None
        self._log(f"[Capture] DISARM window={window_id}")

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "armed": self._armed,
                "active": self._active,
                "done": self._done,
                "window_id": self._window_id,
                "spec": self._spec,
            }

    # -----------------------
    # Data-plane hooks (called by wrapper/decorator)
    # -----------------------
    def on_enter(self, event: HookEvent) -> None:
        # decide trigger
        with self._lock:
            if not self._armed or self._done or self._active:
                return
            if not self._spec:
                return
            if not self._match(event, self._spec):
                return

            # Trigger start
            self._active = True
            self._triggered_by_profile_name = event.profile_name

        self._start_window(event)

    # capture_controller.py 里
    def on_exit(self, event: HookEvent) -> None:
        with self._lock:
            if not self._active:
                return
            if not self._spec:
                return
            stop_policy = self._spec.get("stop_policy", "ON_TRIGGER_FUNC_EXIT")

            if stop_policy == "ON_TRIGGER_FUNC_EXIT":
                if event.profile_name != self._triggered_by_profile_name:
                    return
            elif stop_policy == "MANUAL":
                return
            elif stop_policy == "ON_STOP_PROFILE_NAME":
                stop_names = set(self._spec.get("stop_profile_names", []) or [])
                if event.profile_name not in stop_names:
                    return
            else:
                # fallback：保持默认
                if event.profile_name != self._triggered_by_profile_name:
                    return

        self._stop_window(event)

        with self._lock:
            self._active = False
            self._done = True
            self._armed = False
            self._triggered_by_profile_name = None

    # -----------------------
    # Internal helpers
    # -----------------------
    def _match(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        targets: Set[str] = set(spec.get("target_profile_names", []) or [])
        if targets and event.profile_name not in targets:
            return False

        exp_it = spec.get("expected_iter", None)
        if exp_it is not None:
            it = event.meta.get("iter", None)
            if it is None or int(it) != int(exp_it):
                return False

        exp_mb = spec.get("expected_mb", None)
        if exp_mb is not None:
            mb = event.meta.get("microbatch_index", None)
            if mb is None or int(mb) != int(exp_mb):
                return False

        # optional role/rank filters
        exp_role = spec.get("expected_role", None)
        if exp_role is not None and event.role is not None:
            if str(event.role) != str(exp_role):
                return False

        ranks = spec.get("ranks_filter", None)
        if ranks is not None and event.rank is not None:
            if int(event.rank) not in set(map(int, ranks)):
                return False

        return True

    def _start_window(self, event: HookEvent) -> None:
        self._log(f"[Capture] START window={self._window_id} by={event.profile_name} meta={event.meta}")
        self._nvtx_mark_start()
        try:
            self._backend.start()
        except Exception as e:
            self._log(f"[Capture] ERROR on start: {e}")

    def _stop_window(self, event: HookEvent) -> None:
        self._log(f"[Capture] STOP window={self._window_id} by={event.profile_name} meta={event.meta}")
        try:
            self._backend.stop()
        except Exception as e:
            self._log(f"[Capture] ERROR on stop: {e}")
        finally:
            self._nvtx_mark_end()

    def _nvtx_mark_start(self) -> None:
        if not self._enable_nvtx_window:
            return
        try:
            import nvtx
            # keep it simple and robust: global API, not domain attrs
            nvtx.start_range(message=f"CAPTURE_WINDOW_START {self._window_id}", domain=self._nvtx_domain)
        except Exception:
            return

    def _nvtx_mark_end(self) -> None:
        if not self._enable_nvtx_window:
            return
        try:
            import nvtx
            nvtx.end_range()
        except Exception:
            return

    def _log(self, msg: str) -> None:
        if self._logger is None:
            return
        try:
            self._logger.info(msg)
        except Exception:
            pass
