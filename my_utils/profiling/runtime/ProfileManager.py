class ProfileManager:
    def __init__(self, profile_cfg: dict, logger=None):
        self.cfg = profile_cfg or {}
        self.logger = logger

        prof = self.cfg.get("profile", {}) or {}
        self._enabled = bool(prof.get("enabled", False))
        self._features = prof.get("features", {}) or {}
        self._capture_features = self._features.get("capture", {}) or {}
        self._capture_enabled = bool(self._capture_features.get("enabled", True))
        self._capture_nvtx_window = bool(
            self._capture_features.get("nvtx_window", False)
        )
        self._default_stop_policy = self._normalize_stop_policy(
            self._capture_features.get("default_stop_policy", "ON_TRIGGER_FUNC_EXIT")
        )

        cap = prof.get("capture", {}) or {}
        self._cap_cfg = cap
        self._schedule = cap.get("schedule", {}) or {}
        self._windows = cap.get("windows", []) or []

        self.debug_watch = bool(cap.get("debug_watch", False))
        self._steps = set(map(int, self._schedule.get("steps", []) or []))
        self._max_windows = int(self._schedule.get("max_windows_per_step", 1))
        self._roles = set(self._schedule.get("roles", []) or [])

        self._arm_iters = set()
        for w in self._windows:
            si = w.get("start_iter", None)
            if si is not None:
                self._arm_iters.add(int(si))

        self.metrics_collector = None
        metrics_cfg = prof.get("metrics", {}) or {}
        if bool(metrics_cfg.get("enabled", False)):
            try:
                from ..pipeline.metrics_collector import MetricsCollector

                self.metrics_collector = MetricsCollector(
                    output_dir=str(metrics_cfg.get("output_dir", "metrics"))
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        f"[ProfileManager] failed to init MetricsCollector: {exc}"
                    )

    def enabled(self) -> bool:
        return self._enabled and self._capture_enabled

    @staticmethod
    def _normalize_stop_policy(stop_policy: str) -> str:
        if stop_policy == "ON_TARGET_FUNC_EXIT":
            return "ON_TRIGGER_FUNC_EXIT"
        return stop_policy

    def should_capture_this_iter(self, it: int) -> bool:
        if not self.enabled():
            return False

        it = int(it)
        if self._steps and it not in self._steps:
            return False

        return it in self._arm_iters

    def resolve_mb(self, selector, num_microbatches: int):
        if selector is None:
            return None
        if selector == "last":
            return int(num_microbatches) - 1
        if selector == "first":
            return 0
        if isinstance(selector, int):
            return int(selector)
        if isinstance(selector, str) and selector.isdigit():
            return int(selector)
        return None

    def build_specs_to_arm_at_iter(self, it: int, num_microbatches: int) -> list[dict]:
        it = int(it)
        specs: list[dict] = []

        for w in self._windows:
            role = w.get("role")
            if self._roles and role not in self._roles:
                continue

            start_iter = w.get("start_iter", None)
            if start_iter is None or int(start_iter) != it:
                continue

            start_mb = self.resolve_mb(w.get("start_mb_selector"), num_microbatches)
            stop_mb = self.resolve_mb(w.get("stop_mb_selector"), num_microbatches)

            has_explicit_stop = any(
                k in w and w.get(k) is not None
                for k in (
                    "stop_iter",
                    "stop_profile_names",
                    "stop_mb_selector",
                    "stop_edge",
                )
            )

            stop_policy = w.get("stop_policy", None)
            if not stop_policy:
                stop_policy = (
                    "ON_STOP_PROFILE_NAME"
                    if has_explicit_stop
                    else self._default_stop_policy
                )

            stop_policy = self._normalize_stop_policy(stop_policy)
            stop_edge = w.get("stop_edge", None) or "EXIT"

            spec = {
                "window_id": (
                    f"{w.get('name')}"
                    f"/start_iter{int(start_iter)}_mb{start_mb}"
                    f"/stop_iter{int(w.get('stop_iter', start_iter))}_mb{stop_mb}"
                ),
                "start_profile_names": w.get("start_profile_names", []) or [],
                "start_iter": int(start_iter),
                "start_mb": start_mb,
                "stop_policy": stop_policy,
                "stop_profile_names": w.get("stop_profile_names", []) or [],
                "stop_iter": int(w.get("stop_iter", start_iter)),
                "stop_mb": stop_mb,
                "stop_edge": stop_edge,
                "expected_role": role,
                "ranks_filter": w.get("ranks_filter", None),
                "debug_match": bool(getattr(self, "debug_watch", False)),
                "enable_nvtx_window": bool(
                    w.get("enable_nvtx_window", self._capture_nvtx_window)
                ),
            }
            specs.append(spec)

        return specs[: self._max_windows]

    def arm_iter(self, it: int, num_microbatches: int, wg_by_role: dict):
        if not self.should_capture_this_iter(it):
            return

        specs = self.build_specs_to_arm_at_iter(it, num_microbatches)
        for spec in specs:
            role = spec.get("expected_role")
            wg = wg_by_role.get(role)
            if wg is None:
                continue
            wg.capture_arm(spec)
            if self.logger:
                self.logger.info(f"[Driver] ARM {role} spec={spec}")

        if self.metrics_collector is not None:
            try:
                self.metrics_collector.collect(
                    step=int(it), tags={"phase": "capture_arm"}
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        f"[ProfileManager] metrics collect failed at iter={it}: {exc}"
                    )

    def attach_metrics_provider(self, provider) -> bool:
        if self.metrics_collector is None:
            return False
        self.metrics_collector.register_provider(provider)
        return True

    def export_metrics_report(self, fmt: str = "html", output_path=None):
        if self.metrics_collector is None:
            return None
        return self.metrics_collector.export_report(fmt=fmt, output_path=output_path)
