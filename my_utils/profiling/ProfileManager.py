class ProfileManager:
    def __init__(self, profile_cfg: dict, logger=None):
        self.cfg = profile_cfg or {}
        self.logger = logger

        prof = (self.cfg.get("profile", {}) or {})
        self._enabled = bool(prof.get("enabled", False))

        cap = (prof.get("capture", {}) or {})
        self._cap_cfg = cap
        self._schedule = (cap.get("schedule", {}) or {})
        self._windows = (cap.get("windows", []) or [])

        self.debug_watch = bool(cap.get("debug_watch", False))

        # schedule.steps：你现在仍然可以保留（作为全局 gate）
        self._steps = set(map(int, self._schedule.get("steps", []) or []))
        self._max_windows = int(self._schedule.get("max_windows_per_step", 1))
        self._roles = set(self._schedule.get("roles", []) or [])  # optional

        # 预先收集“哪些 iter 需要 arm”（来自 windows.start_iter）
        self._arm_iters = set()
        for w in self._windows:
            si = w.get("start_iter", None)
            if si is not None:
                self._arm_iters.add(int(si))

    def enabled(self) -> bool:
        return self._enabled

    def should_capture_this_iter(self, it: int) -> bool:
        """
        什么时候需要在 driver 侧发起 arm：
        - 你可以用 schedule.steps 做总开关 gate（比如只想在 [5] 这一轮允许 arm）
        - 也可以不写 schedule.steps，只依赖 windows.start_iter
        """
        if not self._enabled:
            return False

        it = int(it)

        # 如果写了 schedule.steps，就当成总 gate
        if self._steps and it not in self._steps:
            return False

        # 在 start_iter 这一轮 arm（range window 只需要 arm 一次）
        return it in self._arm_iters

    def resolve_mb(self, selector, num_microbatches: int):
        """
        将 mb_selector 解析为具体 microbatch index。
        注意：这里按当前 iter 的 num_microbatches resolve。
        如果 stop_iter 的 num_microbatches 可能变化，需要把 selector 留到 worker 侧 resolve（更复杂）。
        """
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
        # 你也可以扩展 list/range
        return None

    def build_specs_to_arm_at_iter(self, it: int, num_microbatches: int) -> list[dict]:
        """
        只为“start_iter == it”的 window 构造 spec，并广播给对应 role 的 worker group。
        spec 使用你约定的统一字段名（start_iter/stop_iter/start_mb/stop_mb）。
        """
        it = int(it)
        specs: list[dict] = []

        for w in (self._windows or []):
            role = w.get("role")
            if self._roles and role not in self._roles:
                continue

            start_iter = w.get("start_iter", None)
            if start_iter is None or int(start_iter) != it:
                continue  # ✅ 只在 start_iter 当轮 arm

            # ---- resolve start/stop microbatch ----
            start_mb = self.resolve_mb(w.get("start_mb_selector"), num_microbatches)

            stop_iter = w.get("stop_iter", None)
            if stop_iter is None:
                stop_iter = start_iter
            stop_iter = int(stop_iter)

            stop_mb = self.resolve_mb(w.get("stop_mb_selector"), num_microbatches)

            # ---- stop policy default (关键修复点) ----
            stop_policy = w.get("stop_policy", None)
            has_explicit_stop = (
                w.get("stop_profile_names") is not None
                or w.get("stop_iter") is not None
                or w.get("stop_mb_selector") is not None
            )
            if not stop_policy:
                # ✅ 只要配置了 stop 条件，默认就应该是 ON_STOP_PROFILE_NAME
                stop_policy = "ON_STOP_PROFILE_NAME" if has_explicit_stop else "ON_TRIGGER_FUNC_EXIT"

            # 兼容旧名字：ON_TARGET_FUNC_EXIT == ON_TRIGGER_FUNC_EXIT
            if stop_policy == "ON_TARGET_FUNC_EXIT":
                stop_policy = "ON_TRIGGER_FUNC_EXIT"

            stop_edge = w.get("stop_edge", None) or "EXIT"

            # ---- spec ----
            spec = {
                "window_id": (
                    f"{w.get('name')}"
                    f"/start_iter{int(start_iter)}_mb{start_mb}"
                    f"/stop_iter{int(stop_iter)}_mb{stop_mb}"
                ),

                "expected_role": role,
                "ranks_filter": w.get("ranks_filter", None),

                # start 条件（统一字段）
                "start_iter": int(start_iter),
                "start_profile_names": w.get("start_profile_names", []) or [],
                "start_mb": start_mb,  # ✅ 已 resolve 的具体 mb index

                # stop 条件（统一字段）
                "stop_iter": int(stop_iter),
                "stop_profile_names": w.get("stop_profile_names", []) or [],
                "stop_mb": stop_mb,    # ✅ 已 resolve 的具体 mb index
                "stop_policy": stop_policy,
                "stop_edge": stop_edge,

                # debug
                "debug_match": bool(self.debug_watch),

                # window 级 NVTX 大标签
                "enable_nvtx_window": True,
            }
            specs.append(spec)

        return specs[: int(self._max_windows)]

    def arm_iter(self, it: int, num_microbatches: int, wg_by_role: dict):
        """
        wg_by_role: {"generator": self.generator_wg, "critic": self.critic_wg, ...}
        """
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