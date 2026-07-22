from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Hashable, Iterator, Optional, Protocol, runtime_checkable

try:
    import nvtx as _nvtx

    NVTX_AVAILABLE = True
except Exception:
    _nvtx = None
    NVTX_AVAILABLE = False

# torch.cuda.nvtx is resolved on first use, not at module import. The old
# guarded module-level import was invisible without torch but eager with it:
# `import my_utils` pulled the whole torch stack (~half a second, 700+ modules)
# through profiling.runtime.capture_controller -> here, defeating the lazy
# design everywhere else. `TORCH_NVTX_AVAILABLE` is likewise decided lazily --
# a module-level constant would force the import it exists to avoid.
_torch_nvtx = None
_torch_nvtx_checked = False


def _get_torch_nvtx():
    """Return torch.cuda.nvtx, importing it on first call, or None."""
    global _torch_nvtx, _torch_nvtx_checked
    if not _torch_nvtx_checked:
        _torch_nvtx_checked = True
        try:
            import torch.cuda.nvtx as mod

            _torch_nvtx = mod
        except Exception:
            _torch_nvtx = None
    return _torch_nvtx


def _torch_nvtx_available() -> bool:
    return _get_torch_nvtx() is not None


class _TorchNvtxAvailability:
    """Bool-like deferral so `TORCH_NVTX_AVAILABLE` keeps working as a name."""

    def __bool__(self) -> bool:
        return _torch_nvtx_available()

    def __repr__(self) -> str:
        return repr(bool(self))


TORCH_NVTX_AVAILABLE = _TorchNvtxAvailability()


def _env_enabled() -> bool:
    return os.environ.get("ENABLE_NVTX", "0") == "1"


def _normalize_backend_name(backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized in {"auto", "nvtx", "torch", "torch_nvtx", "noop", "none"}:
        return normalized
    raise ValueError(f"Unsupported NVTX backend: {backend!r}")


def _category_key(category: Optional[Any]) -> Hashable:
    if category is None:
        return None
    try:
        hash(category)
    except TypeError:
        return repr(category)
    return category


def _registry_key(
    label: str,
    domain_name: Optional[str],
    category: Optional[Any],
) -> tuple[str, Optional[str], Hashable]:
    return (label, domain_name, _category_key(category))


@runtime_checkable
class LabelerProtocol(Protocol):
    enabled: bool
    backend_name: str
    default_domain: Optional[str]

    def register_label(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None: ...

    def start(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> object | None: ...

    def stop(self, token: object | None = None) -> None: ...

    def mark(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None: ...

    def range(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> Iterator[None]: ...


class NoOpLabeler:
    backend_name = "noop"

    def __init__(self, default_domain: Optional[str] = None):
        self.enabled = False
        self.default_domain = default_domain

    def register_label(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None:
        return

    def start(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None:
        return None

    def stop(self, token: object | None = None) -> None:
        return

    def mark(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None:
        return

    def push(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None:
        return None

    def pop(self) -> None:
        return None

    @contextmanager
    def range(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> Iterator[None]:
        yield


class NvtxLabeler:
    """
    NVTX helper backed by the NVIDIA Python nvtx package.

    The runtime surface is intentionally small so higher-level profiling code can
    depend on a framework-agnostic labeler interface.
    """

    backend_name = "nvtx"

    def __init__(
        self, enabled: Optional[bool] = None, default_domain: Optional[str] = None
    ):
        if enabled is None:
            enabled = _env_enabled()

        self.enabled = bool(enabled and NVTX_AVAILABLE)
        self.default_domain = default_domain
        self._domains: dict[str, Any] = {}
        self._registered_attrs: dict[
            tuple[str, Optional[str], Hashable], tuple[Any, Any]
        ] = {}
        self._active_stack: list[tuple[Any, Any]] = []

    def _get_domain(self, domain_name: Optional[str]) -> Any | None:
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
            raise ValueError(
                "register_label requires `domain_name` or `default_domain`."
            )

        domain = self._get_domain(effective_domain)
        attrs = domain.get_event_attributes(
            message=label, color=color, category=category
        )
        self._registered_attrs[_registry_key(label, effective_domain, category)] = (
            domain,
            attrs,
        )

    def _start_dynamic_range(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> Any:
        kwargs = {"message": label, "color": color}
        effective_domain = domain_name or self.default_domain
        if effective_domain is not None:
            kwargs["domain"] = effective_domain
        if category is not None:
            kwargs["category"] = category

        try:
            return _nvtx.start_range(**kwargs)
        except TypeError:
            kwargs.pop("category", None)
            return _nvtx.start_range(**kwargs)

    def start(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> tuple[Any, Any] | None:
        if not self.enabled:
            return None

        effective_domain = domain_name or self.default_domain
        token = None
        cache_key = _registry_key(label, effective_domain, category)
        if cache_key in self._registered_attrs:
            domain, attrs = self._registered_attrs[cache_key]
            token = (domain, domain.start_range(attributes=attrs))
        else:
            token = (
                None,
                self._start_dynamic_range(label, color, effective_domain, category),
            )

        self._active_stack.append(token)
        return token

    def _remove_token(self, token: tuple[Any, Any]) -> None:
        for idx in range(len(self._active_stack) - 1, -1, -1):
            if self._active_stack[idx] == token:
                self._active_stack.pop(idx)
                return

    def stop(self, token: object | None = None) -> None:
        if not self.enabled:
            return

        if token is None:
            if not self._active_stack:
                return
            domain, range_id = self._active_stack.pop()
        else:
            domain, range_id = token
            self._remove_token((domain, range_id))

        if range_id is None:
            return
        if domain is not None:
            domain.end_range(range_id)
        else:
            _nvtx.end_range(range_id)

    def mark(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None:
        if not self.enabled:
            return

        kwargs = {"message": label, "color": color}
        effective_domain = domain_name or self.default_domain
        if effective_domain is not None:
            kwargs["domain"] = effective_domain
        if category is not None:
            kwargs["category"] = category

        if hasattr(_nvtx, "mark"):
            try:
                _nvtx.mark(**kwargs)
                return
            except TypeError:
                kwargs.pop("category", None)
                _nvtx.mark(**kwargs)
                return

        token = self.start(
            label, color=color, domain_name=effective_domain, category=category
        )
        self.stop(token)

    @contextmanager
    def range(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> Iterator[None]:
        token = self.start(
            label, color=color, domain_name=domain_name, category=category
        )
        try:
            yield
        finally:
            self.stop(token)


class TorchNvtxLabeler:
    """
    Lightweight NVTX helper backed by torch.cuda.nvtx.

    torch.cuda.nvtx does not support true color/domain/category attributes, so
    metadata is folded into the emitted message.
    """

    backend_name = "torch_nvtx"

    def __init__(
        self, enabled: Optional[bool] = None, default_domain: Optional[str] = None
    ):
        if enabled is None:
            enabled = _env_enabled()

        self.enabled = bool(enabled and TORCH_NVTX_AVAILABLE)
        self.default_domain = default_domain
        self._registered_messages: dict[tuple[str, Optional[str], Hashable], str] = {}
        self._active_stack: list[int] = []

    def _compose_message(
        self,
        label: str,
        *,
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> str:
        effective_domain = domain_name or self.default_domain
        prefixes = []
        if effective_domain is not None:
            prefixes.append(str(effective_domain))
        if category is not None:
            prefixes.append(str(category))
        if not prefixes:
            return label
        return f"[{'/'.join(prefixes)}] {label}"

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
        self._registered_messages[_registry_key(label, effective_domain, category)] = (
            self._compose_message(
                label,
                domain_name=effective_domain,
                category=category,
            )
        )

    def start(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> int | None:
        if not self.enabled:
            return None

        effective_domain = domain_name or self.default_domain
        message = self._registered_messages.get(
            _registry_key(label, effective_domain, category)
        )
        if message is None:
            message = self._compose_message(
                label,
                domain_name=effective_domain,
                category=category,
            )
        token = _get_torch_nvtx().range_start(message)
        self._active_stack.append(token)
        return token

    def _remove_token(self, token: int) -> None:
        for idx in range(len(self._active_stack) - 1, -1, -1):
            if self._active_stack[idx] == token:
                self._active_stack.pop(idx)
                return

    def stop(self, token: object | None = None) -> None:
        if not self.enabled:
            return

        if token is None:
            if not self._active_stack:
                return
            token = self._active_stack.pop()
        else:
            self._remove_token(token)

        _get_torch_nvtx().range_end(token)

    def mark(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> None:
        if not self.enabled:
            return

        _mod = _get_torch_nvtx()
        if hasattr(_mod, "mark"):
            _mod.mark(
                self._compose_message(
                    label,
                    domain_name=domain_name or self.default_domain,
                    category=category,
                )
            )
            return

        token = self.start(
            label, color=color, domain_name=domain_name, category=category
        )
        self.stop(token)

    def push(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> int | None:
        if not self.enabled:
            return None
        return _get_torch_nvtx().range_push(
            self._compose_message(
                label,
                domain_name=domain_name or self.default_domain,
                category=category,
            )
        )

    def pop(self) -> int | None:
        if not self.enabled:
            return None
        return _get_torch_nvtx().range_pop()

    @contextmanager
    def range(
        self,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
    ) -> Iterator[None]:
        token = self.start(
            label, color=color, domain_name=domain_name, category=category
        )
        try:
            yield
        finally:
            self.stop(token)


def create_labeler(
    *,
    enabled: Optional[bool] = None,
    default_domain: Optional[str] = None,
    backend: str = "auto",
) -> LabelerProtocol:
    if enabled is None:
        enabled = _env_enabled()

    normalized_backend = _normalize_backend_name(backend)
    if not enabled or normalized_backend in {"noop", "none"}:
        return NoOpLabeler(default_domain=default_domain)

    if normalized_backend == "auto":
        if NVTX_AVAILABLE:
            return NvtxLabeler(enabled=True, default_domain=default_domain)
        if TORCH_NVTX_AVAILABLE:
            return TorchNvtxLabeler(enabled=True, default_domain=default_domain)
        return NoOpLabeler(default_domain=default_domain)

    if normalized_backend == "nvtx":
        if NVTX_AVAILABLE:
            return NvtxLabeler(enabled=True, default_domain=default_domain)
        return NoOpLabeler(default_domain=default_domain)

    if normalized_backend in {"torch", "torch_nvtx"}:
        if TORCH_NVTX_AVAILABLE:
            return TorchNvtxLabeler(enabled=True, default_domain=default_domain)
        return NoOpLabeler(default_domain=default_domain)

    return NoOpLabeler(default_domain=default_domain)
