import os
from contextlib import contextmanager
from typing import Optional, Tuple, Any


try:
    import nvtx as _nvtx

    NVTX_AVAILABLE = True
    print("[NvtxLabeler] nvtx import succeeded. NVTX hooks are available.")
except Exception:
    _nvtx = None
    NVTX_AVAILABLE = False
    print("[NvtxLabeler] nvtx import failed. NVTX hooks are unavailable.")


class NvtxLabeler:
    """
    Lightweight NVTX helper that can be used independently from MyTimer.

    Features:
    - Optional enable/disable switch (runtime no-op when disabled)
    - Optional default domain
    - Pre-registration of stable labels (cached attributes)
    - Context manager for scoped labels
    """

    def __init__(self, enabled: Optional[bool] = None, default_domain: Optional[str] = None):
        if enabled is None:
            enabled = os.environ.get("ENABLE_NVTX", "0") == "1"

        self.enabled = bool(enabled and NVTX_AVAILABLE)
        self.default_domain = default_domain
        self._domains = {}
        self._registered_attrs = {}
        self._active_stack = []
        if self.enabled:
            print(
                f"[NvtxLabeler] enabled=True, default_domain={self.default_domain!r}."
            )
        else:
            if not enabled:
                reason = "requested disabled (enabled=False or ENABLE_NVTX!=1)"
            elif not NVTX_AVAILABLE:
                reason = "nvtx backend unavailable"
            else:
                reason = "unknown"
            print(
                f"[NvtxLabeler] enabled=False, default_domain={self.default_domain!r}, reason={reason}."
            )

    def _get_domain(self, domain_name: Optional[str]):
        if not self.enabled or domain_name is None:
            return None
        if domain_name not in self._domains:
            self._domains[domain_name] = _nvtx.get_domain(domain_name)
        return self._domains[domain_name]

    def register_label(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None:
        if not self.enabled:
            return

        effective_domain = domain_name or self.default_domain
        if effective_domain is None:
            raise ValueError("register_label requires `domain_name` or `default_domain`.")

        domain = self._get_domain(effective_domain)
        attrs = domain.get_event_attributes(message=label, color=color, category=category)
        self._registered_attrs[label] = (domain, attrs)

    def _start_dynamic_range(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ):
        kwargs = {"message": label, "color": color}
        effective_domain = domain_name or self.default_domain
        if effective_domain is not None:
            kwargs["domain"] = effective_domain
        if category is not None:
            kwargs["category"] = category

        try:
            return _nvtx.start_range(**kwargs)
        except TypeError:
            # Some nvtx versions may not accept category in global start_range.
            kwargs.pop("category", None)
            return _nvtx.start_range(**kwargs)

    def start(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> Optional[Tuple[Any, Any]]:
        if not self.enabled:
            return None

        if label in self._registered_attrs:
            domain, attrs = self._registered_attrs[label]
            token = (domain, domain.start_range(attributes=attrs))
        else:
            token = (None, self._start_dynamic_range(label, color, domain_name, category))

        self._active_stack.append(token)
        return token

    def _remove_token(self, token: Tuple[Any, Any]) -> None:
        for idx in range(len(self._active_stack) - 1, -1, -1):
            if self._active_stack[idx] is token:
                self._active_stack.pop(idx)
                return

    def stop(self, token: Optional[Tuple[Any, Any]] = None) -> None:
        if not self.enabled:
            return

        if token is None:
            if not self._active_stack:
                return
            token = self._active_stack.pop()
        else:
            self._remove_token(token)

        domain, range_id = token
        if range_id is None:
            return

        if domain is not None:
            domain.end_range(range_id)
        else:
            _nvtx.end_range(range_id)

    @contextmanager
    def range(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ):
        token = self.start(label, color=color, domain_name=domain_name, category=category)
        try:
            yield
        finally:
            self.stop(token)
