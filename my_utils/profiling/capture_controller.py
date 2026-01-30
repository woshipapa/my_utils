# profiling/capture_controller.py
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
    """
    ARM/TRIGGER controller:
    - Driver broadcasts arm(spec) to all actors in a group
    - Actor triggers start() when entering a matching profiled function
    - Actor triggers stop() at function exit or enter (stop_edge)
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

        self._enable_nvtx_window = enable_nvtx_window
        self._nvtx_domain = nvtx_domain

        self._armed = False
        self._active = False
        self._done = False
        self._spec: Optional[Dict[str, Any]] = None
        self._window_id: Optional[str] = None

        self._triggered_by_profile_name: Optional[str] = None
        self._lock = threading.Lock()

    # -----------------------
    # Public control-plane API
    # -----------------------
    def arm(self, spec: Dict[str, Any]) -> None:
        """
        spec (new convention) example:
          {
            "window_id": "iter5/G_iter5_mb0_to_iter6_mb3",
            "expected_role": "generator",   # optional but recommended

            # start condition
            "start_profile_names": ["run_forward_microbatch"],
            "start_iter": 5,                # optional
            "start_mb": 0,                  # optional (int)

            # stop condition
            "stop_policy": "ON_STOP_PROFILE_NAME" | "ON_TRIGGER_FUNC_EXIT" | "MANUAL",
            "stop_profile_names": ["generator_loss_and_backward"],  # used when stop_policy=ON_STOP_PROFILE_NAME
            "stop_iter": 5,                 # optional
            "stop_mb": 31,                  # optional
            "stop_edge": "EXIT" | "ENTER",  # default EXIT

            "ranks_filter": None | [0,1],
            "debug_match": True/False,
            "enable_nvtx_window": True/False,
          }
        """
        with self._lock:
            self._spec = dict(spec) if spec is not None else {}
            self._window_id = self._spec.get("window_id")

            self._armed = True
            self._active = False
            self._done = False
            self._triggered_by_profile_name = None

            # per-window override
            if "enable_nvtx_window" in self._spec:
                self._enable_nvtx_window = bool(self._spec["enable_nvtx_window"])

            # defaults
            self._spec.setdefault("stop_edge", "EXIT")
            self._spec.setdefault("stop_policy", "ON_TRIGGER_FUNC_EXIT")
            # 兼容旧名字（你 driver 里可能还会塞 ON_TARGET_FUNC_EXIT）
            if self._spec["stop_policy"] == "ON_TARGET_FUNC_EXIT":
                self._spec["stop_policy"] = "ON_TRIGGER_FUNC_EXIT"

            # normalize: ensure lists
            self._spec["start_profile_names"] = list(self._spec.get("start_profile_names") or [])
            self._spec["stop_profile_names"] = list(self._spec.get("stop_profile_names") or [])

            # normalize: iter/mb fields (allow None)
            if "start_iter" in self._spec and self._spec["start_iter"] is not None:
                self._spec["start_iter"] = int(self._spec["start_iter"])
            if "stop_iter" in self._spec and self._spec["stop_iter"] is not None:
                self._spec["stop_iter"] = int(self._spec["stop_iter"])
            if "start_mb" in self._spec and self._spec["start_mb"] is not None:
                self._spec["start_mb"] = int(self._spec["start_mb"])
            if "stop_mb" in self._spec and self._spec["stop_mb"] is not None:
                self._spec["stop_mb"] = int(self._spec["stop_mb"])

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
    # Data-plane hooks
    # -----------------------
    def on_enter(self, event: HookEvent) -> None:
        """
        - when inactive: try match start -> start()
        - when active and stop_edge=ENTER: try match stop -> stop()
        """
        with self._lock:
            if not self._armed or self._done or not self._spec:
                return

            # if not active: try start
            if not self._active:
                if not self._match_start(event, self._spec):
                    return
                self._active = True
                self._triggered_by_profile_name = event.profile_name
                do_start = True
            else:
                do_start = False

            # if active and stop_edge=ENTER: maybe stop on enter
            do_stop = False
            if self._active and self._spec.get("stop_edge", "EXIT") == "ENTER":
                if self._should_stop_on_event_enter(event, self._spec):
                    do_stop = True

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
        """
        stop_edge=EXIT only.
        """
        with self._lock:
            if not self._active or not self._spec:
                return
            if self._spec.get("stop_edge", "EXIT") != "EXIT":
                return

            if not self._should_stop_on_event_exit(event, self._spec):
                return

        # outside lock
        self._stop_window(event)
        with self._lock:
            self._active = False
            self._done = True
            self._armed = False
            self._triggered_by_profile_name = None

    # -----------------------
    # Matching helpers
    # -----------------------
    def _match_common_filters(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        """
        role + ranks_filter are always checked (if provided).
        """
        debug = bool(spec.get("debug_match", False))

        # role gate (recommended)
        exp_role = spec.get("expected_role", None)
        if exp_role is not None:
            if event.role is None or str(event.role) != str(exp_role):
                if debug:
                    self._log(f"[Capture][Match] FAIL role: got={event.role}, expected={exp_role}")
                return False

        # rank gate
        ranks = spec.get("ranks_filter", None)
        if ranks is not None:
            if event.rank is None:
                if debug:
                    self._log(f"[Capture][Match] FAIL rank: event.rank=None, ranks_filter={ranks}")
                return False
            ranks_set = set(map(int, ranks))
            if int(event.rank) not in ranks_set:
                if debug:
                    self._log(f"[Capture][Match] FAIL rank: got={event.rank}, ranks_filter={sorted(list(ranks_set))}")
                return False

        return True

    def _match_by_fields(
        self,
        *,
        event: HookEvent,
        spec: Dict[str, Any],
        profile_names: list[str],
        it: Optional[int],
        mb: Optional[int],
        tag: str,
    ) -> bool:
        debug = bool(spec.get("debug_match", False))
        if debug:
            self._log(
                f"[Capture][Match:{tag}] try window={self._window_id} "
                f"event={{name={event.profile_name}, role={event.role}, rank={event.rank}, meta={event.meta}}} "
                f"cond={{profiles={profile_names}, iter={it}, mb={mb}, expected_role={spec.get('expected_role')}, ranks_filter={spec.get('ranks_filter')}}}"
            )

        if not self._match_common_filters(event, spec):
            return False

        # profile gate
        targets: Set[str] = set(profile_names or [])
        if targets and event.profile_name not in targets:
            if debug:
                self._log(f"[Capture][Match:{tag}] FAIL name: {event.profile_name} not in {sorted(list(targets))}")
            return False

        # iter gate
        if it is not None:
            ev_it = event.meta.get("iter", None)
            if ev_it is None or int(ev_it) != int(it):
                if debug:
                    self._log(f"[Capture][Match:{tag}] FAIL iter: got={ev_it}, expected={it}")
                return False

        # mb gate
        if mb is not None:
            ev_mb = event.meta.get("microbatch_index", None)
            if ev_mb is None or int(ev_mb) != int(mb):
                if debug:
                    self._log(f"[Capture][Match:{tag}] FAIL mb: got={ev_mb}, expected={mb}")
                return False

        if debug:
            self._log(f"[Capture][Match:{tag}] ✅ MATCHED window={self._window_id}")
        return True

    def _match_start(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        return self._match_by_fields(
            event=event,
            spec=spec,
            profile_names=spec.get("start_profile_names", []) or [],
            it=spec.get("start_iter", None),
            mb=spec.get("start_mb", None),
            tag="START",
        )

    def _match_stop(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        # stop 条件只用于 ON_STOP_PROFILE_NAME 的路径
        return self._match_by_fields(
            event=event,
            spec=spec,
            profile_names=spec.get("stop_profile_names", []) or [],
            it=spec.get("stop_iter", None),
            mb=spec.get("stop_mb", None),
            tag="STOP",
        )

    def _should_stop_on_event_enter(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        """
        stop_edge=ENTER: allow stop on enter.
        """
        policy = spec.get("stop_policy", "ON_TRIGGER_FUNC_EXIT")

        if policy == "MANUAL":
            return False

        if policy == "ON_TRIGGER_FUNC_EXIT":
            return event.profile_name == self._triggered_by_profile_name

        if policy == "ON_STOP_PROFILE_NAME":
            # 如果没给 stop_profile_names，就 fallback 到 trigger-exit (防止永不 stop)
            stop_names = spec.get("stop_profile_names", []) or []
            if not stop_names:
                return event.profile_name == self._triggered_by_profile_name
            return self._match_stop(event, spec)

        # fallback
        return event.profile_name == self._triggered_by_profile_name

    def _should_stop_on_event_exit(self, event: HookEvent, spec: Dict[str, Any]) -> bool:
        """
        stop_edge=EXIT: stop evaluated on exit.
        """
        policy = spec.get("stop_policy", "ON_TRIGGER_FUNC_EXIT")

        if policy == "MANUAL":
            return False

        if policy == "ON_TRIGGER_FUNC_EXIT":
            return event.profile_name == self._triggered_by_profile_name

        if policy == "ON_STOP_PROFILE_NAME":
            stop_names = spec.get("stop_profile_names", []) or []
            if not stop_names:
                # fallback 防止永不 stop
                return event.profile_name == self._triggered_by_profile_name
            return self._match_stop(event, spec)

        return event.profile_name == self._triggered_by_profile_name

    # -----------------------
    # Start/Stop
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