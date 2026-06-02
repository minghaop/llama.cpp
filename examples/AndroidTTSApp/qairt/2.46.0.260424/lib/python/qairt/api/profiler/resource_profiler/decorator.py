# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
@resource_profile decorator for class-scope and function-scope profiling.

Usage::

    @resource_profile
    class MyClass: ...

    @resource_profile(flush_ram=True)
    class MyClass: ...

    @resource_profile(scope="function")
    def my_function(): ...

    @resource_profile(scope="function", event="custom.event_name")
    def my_function(): ...

    @resource_profile(scope="function", flush_ram=True)
    def my_function(): ...
"""

from __future__ import annotations

import functools
import gc
import inspect
import logging
import os
from typing import Any, Callable, Optional, Union

from qairt.api.profiler.resource_profiler.events import ProfileEventType
from qairt.api.profiler.resource_profiler.resource_profiler import ResourceProfiler, _reset_gpu_peak

_logger = logging.getLogger("qairt.api.profiler.resource_profiler.decorator")

_ENV_VAR = "QAIRT_PROFILING"


def _profiling_enabled() -> bool:
    return os.environ.get(_ENV_VAR, "").strip() == "1"


def _wrap_function(
    fn: Callable,
    event_name: Union[str, Callable[..., str]],
    event_type: ProfileEventType,
    flush_ram: bool,
) -> Callable:
    """Wrap a single callable with profiling logic.

    Args:
        event_name: Static string or a callable that receives the function's
                    ``(*args, **kwargs)`` and returns the event name at runtime.
    """

    profiler = ResourceProfiler()

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _profiling_enabled():
            return fn(*args, **kwargs)
        resolved_name = event_name(*args, **kwargs) if callable(event_name) else event_name
        _logger.debug(f"Profiling started: {resolved_name}")
        if flush_ram:
            gc.collect()
        _reset_gpu_peak()
        start = profiler.snapshot(resolved_name, event_type)
        try:
            result = fn(*args, **kwargs)
        finally:
            end = profiler.snapshot(resolved_name, event_type)
            profiler.record(resolved_name, event_type, start, end)
            _logger.debug(f"Profiling completed: {resolved_name} ({end.t - start.t:.3f}s)")
        return result

    setattr(_wrapper, "_resource_profile", True)
    return _wrapper


def resource_profile(
    cls_or_fn: Any = None,
    *,
    scope: str = "class",
    event: Optional[Union[str, Callable[..., str]]] = None,
    flush_ram: bool = False,
) -> Any:
    """
    Profile a class or function.

    With ``scope="class"`` (default), all public methods are instrumented.
    With ``scope="function"``, only the decorated callable is instrumented.

    Args:
        scope:     ``"class"`` or ``"function"``.
        event:     Event name or callable returning one at runtime.
                   Defaults to ``ClassName.method_name`` or ``function.__qualname__``.
        flush_ram: If True, run ``gc.collect()`` before each snapshot.
    """

    def _decorate_class(cls: type) -> type:
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip private/dunder methods
            if name.startswith("_"):
                continue
            # Methods already wrapped by a per-method @profile are skipped here —
            # the per-method decorator's event name and options are used instead.
            if getattr(method, "_resource_profile", False):
                continue
            ev_name = f"{cls.__qualname__}.{name}"
            setattr(
                cls,
                name,
                _wrap_function(method, ev_name, ProfileEventType.CLASS_METHOD, flush_ram),
            )
        return cls

    def _decorate_function(fn: Callable) -> Callable:
        ev_name = event if event is not None else fn.__qualname__
        return _wrap_function(fn, ev_name, ProfileEventType.FUNCTION, flush_ram)

    # Bare @profile on a class or function (no parentheses)
    if cls_or_fn is not None:
        if inspect.isclass(cls_or_fn):
            return _decorate_class(cls_or_fn)
        return _decorate_function(cls_or_fn)

    # Called with arguments — return a decorator
    if scope not in ("class", "function"):
        raise ValueError(f"@resource_profile: invalid scope {scope!r}. Must be 'class' or 'function'.")

    def _decorator(target: Any) -> Any:
        if scope == "class":
            if not inspect.isclass(target):
                raise TypeError(f"@resource_profile(scope='class') requires a class, got {type(target)!r}")
            return _decorate_class(target)
        # scope == "function"
        if not callable(target):
            raise TypeError(f"@resource_profile(scope='function') requires a callable, got {type(target)!r}")
        return _decorate_function(target)

    return _decorator
