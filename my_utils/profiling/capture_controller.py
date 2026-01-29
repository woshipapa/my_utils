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

            # arm() 里，在 self._spec = dict(spec) 后面追加
            if "start_profile_names" not in self._spec:
                self._spec["start_profile_names"] = self._spec.get("target_profile_names", [])
            if "start_expected_iter" not in self._spec:
                self._spec["start_expected_iter"] = self._spec.get("expected_iter", None)
            if "start_expected_mb" not in self._spec:
                self._spec["start_expected_mb"] = self._spec.get("expected_mb", None)

            if "stop_profile_names" not in self._spec:
                self._spec["stop_profile_names"] = self._spec.get("stop_profile_names", [])

            if "stop_expected_iter" not in self._spec:
                # 若用户没写 stop_expected_iter，就默认 None（不做 iter gate）
                self._spec["stop_expected_iter"] = self._spec.get("stop_expected_iter", None)

            if "stop_expected_mb" not in self._spec:
                self._spec["stop_expected_mb"] = self._spec.get("stop_expected_mb", None)

            # 兼容 stop_policy 名字
            if self._spec.get("stop_policy") == "ON_TARGET_FUNC_EXIT":
                self._spec["stop_policy"] = "ON_TRIGGER_FUNC_EXIT"

            if "stop_edge" not in self._spec:
                self._spec["stop_edge"] = "EXIT"

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
    def on_enter(self, event):
        with self._lock:
            if not self._armed or self._done:
                return
            if not self._spec:
                return

            # 如果 active 且 stop_edge=ENTER，就允许在 enter 上 stop
            if self._active:
                if self._spec.get("stop_edge", "EXIT") == "ENTER":
                    if self._match_stop(event, self._spec):
                        pass
                    else:
                        return
                else:
                    return  # active 且 stop_edge=EXIT 时，enter 不处理 stop
            else:
                # not active -> try start
                if not self._match_start(event, self._spec):
                    return
                self._active = True
                self._triggered_by_profile_name = event.profile_name

        # outside lock
        if self._active and self._triggered_by_profile_name == event.profile_name:
            self._start_window(event)

        # 若 stop_edge=ENTER 且匹配 stop -> stop
        if self._spec.get("stop_edge", "EXIT") == "ENTER":
            # 这里再 check 一遍 stop（避免竞态）
            if self._match_stop(event, self._spec):
                self._stop_window(event)
                with self._lock:
                    self._active = False
                    self._done = True
                    self._armed = False
                    self._triggered_by_profile_name = None


    # capture_controller.py 里
    def on_exit(self, event):
        with self._lock:
            if not self._active or not self._spec:
                return
            stop_edge = self._spec.get("stop_edge", "EXIT")
            if stop_edge != "EXIT":
                return

            stop_policy = self._spec.get("stop_policy", "ON_TRIGGER_FUNC_EXIT")
            if stop_policy == "ON_TRIGGER_FUNC_EXIT":
                if event.profile_name != self._triggered_by_profile_name:
                    return
            elif stop_policy == "MANUAL":
                return
            elif stop_policy == "ON_STOP_PROFILE_NAME":
                if not self._match_stop(event, self._spec):
                    return
            else:
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
        # debug 开关：默认不刷屏；你可以在 driver arm 的 spec 里加 "debug_match": true
        debug = bool(spec.get("debug_match", False))
        if debug:
            self._log(
                f"[Capture][Match] try window={self._window_id} "
                f"event={{name={event.profile_name}, role={event.role}, rank={event.rank}, meta={event.meta}}} "
                f"spec={{targets={spec.get('target_profile_names')}, expected_iter={spec.get('expected_iter')}, "
                f"expected_mb={spec.get('expected_mb')}, expected_role={spec.get('expected_role')}, "
                f"ranks_filter={spec.get('ranks_filter')}}}"
            )

        # 1) profile_name gate
        targets: Set[str] = set(spec.get("target_profile_names", []) or [])
        if targets and event.profile_name not in targets:
            if debug:
                self._log(f"[Capture][Match] FAIL name: {event.profile_name} not in {sorted(list(targets))}")
            return False
        if debug:
            self._log(f"[Capture][Match] PASS name")

        # 2) iteration gate
        exp_it = spec.get("expected_iter", None)
        if exp_it is not None:
            it = event.meta.get("iter", None)
            if it is None:
                if debug:
                    self._log(f"[Capture][Match] FAIL iter: event.meta['iter'] missing, expected={exp_it}")
                return False
            if int(it) != int(exp_it):
                if debug:
                    self._log(f"[Capture][Match] FAIL iter: got={it}, expected={exp_it}")
                return False
            if debug:
                self._log(f"[Capture][Match] PASS iter: {it}")

        # 3) microbatch gate
        exp_mb = spec.get("expected_mb", None)
        if exp_mb is not None:
            mb = event.meta.get("microbatch_index", None)
            if mb is None:
                if debug:
                    self._log(f"[Capture][Match] FAIL mb: event.meta['microbatch_index'] missing, expected={exp_mb}")
                return False
            if int(mb) != int(exp_mb):
                if debug:
                    self._log(f"[Capture][Match] FAIL mb: got={mb}, expected={exp_mb}")
                return False
            if debug:
                self._log(f"[Capture][Match] PASS mb: {mb}")

        # 4) role gate
        exp_role = spec.get("expected_role", None)
        if exp_role is not None:
            # event.role 可能是 None；这种情况建议直接 fail（避免误触发）
            if event.role is None:
                if debug:
                    self._log(f"[Capture][Match] FAIL role: event.role is None, expected={exp_role}")
                return False
            if str(event.role) != str(exp_role):
                if debug:
                    self._log(f"[Capture][Match] FAIL role: got={event.role}, expected={exp_role}")
                return False
            if debug:
                self._log(f"[Capture][Match] PASS role: {event.role}")

        # 5) ranks_filter gate
        ranks = spec.get("ranks_filter", None)
        if ranks is not None:
            if event.rank is None:
                if debug:
                    self._log(f"[Capture][Match] FAIL rank: event.rank is None, ranks_filter={ranks}")
                return False
            ranks_set = set(map(int, ranks))
            if int(event.rank) not in ranks_set:
                if debug:
                    self._log(f"[Capture][Match] FAIL rank: got={event.rank}, ranks_filter={sorted(list(ranks_set))}")
                return False
            if debug:
                self._log(f"[Capture][Match] PASS rank: {event.rank}")

        if debug:
            self._log(f"[Capture][Match] ✅ MATCHED window={self._window_id} by={event.profile_name}")
        return True


    def _match_start(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        # 复用 _match 的逻辑，但换成 start 字段
        spec2 = dict(spec)
        spec2["target_profile_names"] = spec.get("start_profile_names", []) or []
        spec2["expected_iter"] = spec.get("start_expected_iter", None)
        spec2["expected_mb"] = spec.get("start_expected_mb", None)
        return self._match(event, spec2)

    def _match_stop(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        spec2 = dict(spec)
        # stop 用 stop_profile_names；若为空则不能用 ON_STOP_PROFILE_NAME
        spec2["target_profile_names"] = spec.get("stop_profile_names", []) or []
        spec2["expected_iter"] = spec.get("stop_expected_iter", None)
        spec2["expected_mb"] = spec.get("stop_expected_mb", None)
        return self._match(event, spec2)



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
