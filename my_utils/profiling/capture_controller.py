from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set
import threading


@dataclass
class HookEvent:
    profile_name: str
    meta: Dict[str, Any]
    role: Optional[str] = None
    rank: Optional[int] = None


class CaptureController:
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

        self._enable_nvtx_window = enable_nvtx_window
        self._nvtx_domain = nvtx_domain

        self._armed = False
        self._active = False
        self._done = False
        self._spec: Optional[Dict[str, Any]] = None
        self._window_id: Optional[str] = None

        # 记录是哪个 profile_name 触发 start（用于 ON_TRIGGER_FUNC_EXIT）
        self._triggered_by_profile_name: Optional[str] = None

        self._lock = threading.Lock()

    # -----------------------
    # control-plane
    # -----------------------
    def arm(self, spec: Dict[str, Any]) -> None:
        with self._lock:
            self._spec = dict(spec) if spec is not None else {}
            self._window_id = self._spec.get("window_id")
            self._armed = True
            self._active = False
            self._done = False
            self._triggered_by_profile_name = None

            # allow per-window nvtx override
            if "enable_nvtx_window" in self._spec:
                self._enable_nvtx_window = bool(self._spec["enable_nvtx_window"])

            # ---- normalize defaults (统一字段名) ----
            self._spec.setdefault("start_profile_names", [])
            self._spec.setdefault("stop_profile_names", [])

            # start/stop iter/mb 默认 None = 不 gate
            self._spec.setdefault("start_iter", None)
            self._spec.setdefault("start_mb", None)
            self._spec.setdefault("stop_iter", None)
            self._spec.setdefault("stop_mb", None)

            # policy/edge 默认
            self._spec.setdefault("stop_policy", "ON_TRIGGER_FUNC_EXIT")
            self._spec.setdefault("stop_edge", "EXIT")

            self._log(
                f"[Capture] ARMED window={self._window_id} "
                f"start={{iter={self._spec.get('start_iter')}, mb={self._spec.get('start_mb')}, names={self._spec.get('start_profile_names')}}} "
                f"stop={{iter={self._spec.get('stop_iter')}, mb={self._spec.get('stop_mb')}, names={self._spec.get('stop_profile_names')}, "
                f"policy={self._spec.get('stop_policy')}, edge={self._spec.get('stop_edge')}}}"
            )

    def disarm(self, window_id: Optional[str] = None) -> None:
        with self._lock:
            if window_id is not None and window_id != self._window_id:
                return
            self._armed = False
            self._active = False
            self._done = False
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
    # data-plane hooks
    # -----------------------
    def on_enter(self, event: HookEvent) -> None:
        with self._lock:
            if not self._armed or self._done or not self._spec:
                return

            # active 时：只有 stop_edge=ENTER 才允许在 enter 上 stop
            if self._active:
                if self._spec.get("stop_edge", "EXIT") != "ENTER":
                    return
                # stop_edge=ENTER
                if not self._should_stop_on_event(event, edge="ENTER"):
                    return
                # will stop outside lock
                do_stop = True
                do_start = False
            else:
                # not active -> try start
                if not self._match_start(event):
                    return
                self._active = True
                self._triggered_by_profile_name = event.profile_name
                do_start = True
                do_stop = False

        # outside lock
        if do_start:
            self._start_window(event)

        if do_stop:
            self._stop_window(event)
            with self._lock:
                self._active = False
                self._done = True
                self._armed = False
                self._triggered_by_profile_name = None

    def on_exit(self, event: HookEvent) -> None:
        with self._lock:
            if not self._active or not self._spec:
                return
            if self._spec.get("stop_edge", "EXIT") != "EXIT":
                return
            if not self._should_stop_on_event(event, edge="EXIT"):
                return

        # outside lock
        self._stop_window(event)
        with self._lock:
            self._active = False
            self._done = True
            self._armed = False
            self._triggered_by_profile_name = None

    # -----------------------
    # matching
    # -----------------------
    def _match_common(
        self,
        event: HookEvent,
        *,
        profile_names: list[str],
        expected_iter: Optional[int],
        expected_mb: Optional[int],
    ) -> bool:
        spec = self._spec or {}
        debug = bool(spec.get("debug_match", False))

        if debug:
            self._log(
                f"[Capture][Match] window={self._window_id} "
                f"event={{name={event.profile_name}, role={event.role}, rank={event.rank}, meta={event.meta}}} "
                f"need={{names={profile_names}, iter={expected_iter}, mb={expected_mb}, "
                f"role={spec.get('expected_role')}, ranks={spec.get('ranks_filter')}}}"
            )

        # name gate
        targets: Set[str] = set(profile_names or [])
        if targets and event.profile_name not in targets:
            if debug:
                self._log(f"[Capture][Match] FAIL name")
            return False

        # iter gate
        if expected_iter is not None:
            it = event.meta.get("iter", None)
            if it is None or int(it) != int(expected_iter):
                if debug:
                    self._log(f"[Capture][Match] FAIL iter got={it} expected={expected_iter}")
                return False

        # mb gate
        if expected_mb is not None:
            mb = event.meta.get("microbatch_index", None)
            if mb is None or int(mb) != int(expected_mb):
                if debug:
                    self._log(f"[Capture][Match] FAIL mb got={mb} expected={expected_mb}")
                return False

        # role gate
        exp_role = spec.get("expected_role", None)
        if exp_role is not None:
            if event.role is None or str(event.role) != str(exp_role):
                if debug:
                    self._log(f"[Capture][Match] FAIL role got={event.role} expected={exp_role}")
                return False

        # rank gate
        ranks = spec.get("ranks_filter", None)
        if ranks is not None:
            if event.rank is None:
                if debug:
                    self._log(f"[Capture][Match] FAIL rank None expected in {ranks}")
                return False
            ranks_set = set(map(int, ranks))
            if int(event.rank) not in ranks_set:
                if debug:
                    self._log(f"[Capture][Match] FAIL rank got={event.rank} expected in {sorted(ranks_set)}")
                return False

        if debug:
            self._log(f"[Capture][Match] ✅ PASS")
        return True

    def _match_start(self, event: HookEvent) -> bool:
        spec = self._spec or {}
        return self._match_common(
            event,
            profile_names=spec.get("start_profile_names", []) or [],
            expected_iter=spec.get("start_iter", None),
            expected_mb=spec.get("start_mb", None),
        )

    def _match_stop(self, event: HookEvent) -> bool:
        spec = self._spec or {}
        return self._match_common(
            event,
            profile_names=spec.get("stop_profile_names", []) or [],
            expected_iter=spec.get("stop_iter", None),
            expected_mb=spec.get("stop_mb", None),
        )

    def _should_stop_on_event(self, event: HookEvent, *, edge: str) -> bool:
        """
        edge: "ENTER" or "EXIT"
        """
        spec = self._spec or {}
        stop_policy = spec.get("stop_policy", "ON_TRIGGER_FUNC_EXIT")

        if stop_policy == "MANUAL":
            return False

        if stop_policy == "ON_TRIGGER_FUNC_EXIT":
            # 只在 EXIT 边缘生效；ENTER 时不 stop
            if edge != "EXIT":
                return False
            return event.profile_name == self._triggered_by_profile_name

        if stop_policy == "ON_STOP_PROFILE_NAME":
            # stop_edge 由 on_enter/on_exit 的外层逻辑控制
            return self._match_stop(event)

        # fallback：等同 ON_TRIGGER_FUNC_EXIT
        if edge != "EXIT":
            return False
        return event.profile_name == self._triggered_by_profile_name

    # -----------------------
    # start/stop
    # -----------------------
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
            nvtx.start_range(
                message=f"CAPTURE_WINDOW_START {self._window_id}",
                domain=self._nvtx_domain,
            )
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