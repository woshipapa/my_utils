class ProfileManager:
    def __init__(self, profile_cfg: dict, logger=None):
        self.cfg = profile_cfg or {}
        self.logger = logger

        self._enabled = bool(self.cfg.get("profile", {}).get("enabled", False))
        self._cap_cfg = (self.cfg.get("profile", {}) or {}).get("capture", {}) or {}
        self._schedule = (self._cap_cfg.get("schedule", {}) or {})
        self._windows = (self._cap_cfg.get("windows", []) or [])
        self.debug_watch = bool(self._cap_cfg.get("debug_watch", False))
        self._steps = set(map(int, self._schedule.get("steps", []) or []))
        self._max_windows = int(self._schedule.get("max_windows_per_step", 1))
        self._roles = set(self._schedule.get("roles", []) or [])  # optional

    def enabled(self) -> bool:
        return self._enabled

    def should_capture_this_iter(self, it: int) -> bool:
        if not self._enabled:
            return False
        if not self._steps:
            return False
        return int(it) in self._steps

    def resolve_mb(self, selector, num_microbatches: int):
        # 支持：last/first/int/list/range dict
        if selector == "last":
            return num_microbatches - 1
        if selector == "first":
            return 0
        if isinstance(selector, int):
            return selector
        if isinstance(selector, list):
            # 你也可以返回 list -> 多 window，先简单取第一个
            return int(selector[0])
        if isinstance(selector, dict) and "range" in selector:
            # {"range":[0,3]} -> 0..3 你可以展开成多个 window
            a, b = selector["range"]
            return int(a)  # 先简单处理
        return None

    def build_specs_for_iter(self, it: int, num_microbatches: int) -> list[dict]:
        specs = []
        for w in self._windows:
            role = w.get("role")
            if self._roles and role not in self._roles:
                continue

            expected_mb = self.resolve_mb(w.get("mb_selector"), num_microbatches)
            spec = {
                "window_id": f"iter{it}/{w.get('name')}/mb{expected_mb}",
                "target_profile_names": w.get("target_profile_names", []) or [],
                "expected_iter": int(it),                 # ✅ 关键：注入 schedule 的 step
                "expected_mb": expected_mb,               # ✅ driver resolve
                "expected_role": role,                    # 让 worker 侧 role filter 更稳
                "ranks_filter": w.get("ranks_filter", None),
                "stop_policy": w.get("stop_policy", None) or "ON_TARGET_FUNC_EXIT",
                "stop_profile_names": w.get("stop_profile_names", None),
                "debug_match": self.debug_watch,
                # 可选：window 级 NVTX window 开关
                "enable_nvtx_window": True,
            }
            specs.append(spec)

        # 限制每轮最多 arm N 个 window，避免 nsys 太大
        return specs[: self._max_windows]

    def arm_iter(self, it: int, num_microbatches: int, wg_by_role: dict):
        """
        wg_by_role: {"generator": self.generator_wg, "critic": self.critic_wg, ...}
        """
        specs = self.build_specs_for_iter(it, num_microbatches)
        for spec in specs:
            role = spec.get("expected_role")
            wg = wg_by_role.get(role)
            if wg is None:
                continue
            # 广播到这个 role 的所有 actor
            wg.capture_arm(spec)
            if self.logger:
                self.logger.info(f"[Driver] ARM {role} spec={spec}")