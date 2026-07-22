from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, ContextManager, Dict, Optional, Tuple


_PATCH_FLAG = "_my_utils_method_patch_wrapped"
_PATCH_ORIGINAL = "_my_utils_method_patch_original"
_PATCH_TAG = "_my_utils_method_patch_tag"


WrapperFactory = Callable[[Callable[..., Any]], Callable[..., Any]]
ContextFactory = Callable[
    [Callable[..., Any], tuple[Any, ...], dict[str, Any]], ContextManager[Any]
]


@dataclass
class MethodPatchHandle:
    patcher: "MethodPatcher"
    target: Any
    attr_name: str
    tag: Optional[str] = None

    def unpatch(self) -> bool:
        return self.patcher.unpatch(self.target, self.attr_name)


class MethodPatcher:
    """
    Reversible monkey patch manager for object methods.

    The class is designed for profiling and debugging scenarios where a third-
    party library method needs to be wrapped without editing upstream source.
    """

    def __init__(self):
        self._handles: Dict[Tuple[int, str], MethodPatchHandle] = {}

    @staticmethod
    def _key(target: Any, attr_name: str) -> Tuple[int, str]:
        return (id(target), attr_name)

    @staticmethod
    def _get_original(callable_obj: Callable[..., Any]) -> Callable[..., Any]:
        return getattr(callable_obj, _PATCH_ORIGINAL, callable_obj)

    @staticmethod
    def is_patched(target: Any, attr_name: str) -> bool:
        if target is None or not hasattr(target, attr_name):
            return False
        return bool(getattr(getattr(target, attr_name), _PATCH_FLAG, False))

    def patch(
        self,
        target: Any,
        attr_name: str,
        wrapper_factory: WrapperFactory,
        *,
        tag: Optional[str] = None,
    ) -> MethodPatchHandle:
        if target is None:
            raise ValueError("`target` must not be None.")
        if not hasattr(target, attr_name):
            raise AttributeError(f"{target!r} has no attribute {attr_name!r}.")

        current = getattr(target, attr_name)
        if not callable(current):
            raise TypeError(f"{type(target).__name__}.{attr_name} is not callable.")

        original = self._get_original(current)
        wrapped = wrapper_factory(original)
        if not callable(wrapped):
            raise TypeError("`wrapper_factory` must return a callable object.")

        setattr(wrapped, _PATCH_FLAG, True)
        setattr(wrapped, _PATCH_ORIGINAL, original)
        setattr(wrapped, _PATCH_TAG, tag)
        setattr(target, attr_name, wrapped)

        handle = MethodPatchHandle(
            patcher=self,
            target=target,
            attr_name=attr_name,
            tag=tag,
        )
        self._handles[self._key(target, attr_name)] = handle
        return handle

    def patch_with_context(
        self,
        target: Any,
        attr_name: str,
        *,
        context_factory: ContextFactory,
        tag: Optional[str] = None,
    ) -> MethodPatchHandle:
        def wrapper_factory(original: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(original)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                with context_factory(original, args, kwargs):
                    return original(*args, **kwargs)

            return wrapped

        return self.patch(target, attr_name, wrapper_factory, tag=tag)

    def patch_with_labeler(
        self,
        target: Any,
        attr_name: str,
        *,
        labeler: Any,
        label: str,
        color: str = "blue",
        domain_name: Optional[str] = None,
        category: Optional[Any] = None,
        tag: Optional[str] = None,
    ) -> MethodPatchHandle:
        def context_factory(
            original: Callable[..., Any],
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> ContextManager[Any]:
            del original, args, kwargs
            if labeler is None:
                return nullcontext()
            return labeler.range(
                label,
                color=color,
                domain_name=domain_name,
                category=category,
            )

        return self.patch_with_context(
            target,
            attr_name,
            context_factory=context_factory,
            tag=tag or label,
        )

    def unpatch(self, target: Any, attr_name: str) -> bool:
        if target is None or not hasattr(target, attr_name):
            self._handles.pop(self._key(target, attr_name), None)
            return False

        current = getattr(target, attr_name)
        if not getattr(current, _PATCH_FLAG, False):
            self._handles.pop(self._key(target, attr_name), None)
            return False

        original = self._get_original(current)
        setattr(target, attr_name, original)
        self._handles.pop(self._key(target, attr_name), None)
        return True

    def unpatch_all(self) -> int:
        count = 0
        for handle in list(self._handles.values()):
            count += int(handle.unpatch())
        return count

    def clear(self) -> int:
        return self.unpatch_all()
