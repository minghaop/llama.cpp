# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Profiler core: ProfileMarker, global store, and ResourceProfiler facade.

By default, ``ResourceProfiler()`` instances share the module-level ``_GLOBAL_STORE``,
so profiling data from the ``@resource_profile`` decorator accumulates across the entire
process lifetime until reset.  Use ``ResourceProfiler.create_isolated()`` (or pass an
explicit ``store=``) to get a private store that does not mix with decorator
data — observers do this so their measurements stay independent.

Usage::

    # Shared global store (used by @resource_profile decorator)
    profiler = ResourceProfiler()
    profiler.reset_peak_memory_stats()
    start = profiler.snapshot("my_event", ProfileEventType.FUNCTION)
    # ... do work ...
    end = profiler.snapshot("my_event", ProfileEventType.FUNCTION)
    profiler.record("my_event", ProfileEventType.FUNCTION, start, end)
    report = profiler.report()

    # Isolated store (used by observers)
    profiler = ResourceProfiler.create_isolated()
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import psutil

from qairt.api.profiler.resource_profiler.events import ProfileEventType

if TYPE_CHECKING:
    from qairt.api.profiler.resource_profiler.report import MemoryProfilingReport


@dataclass
class _SnapshotPoint:
    """Internal: captures PSS/swap/GPU/time at a single moment in time."""

    event_name: str
    event_type: ProfileEventType
    ram_pss_bytes: int
    swap_bytes: int
    gpu_allocated_bytes: int
    gpu_peak_bytes: int
    t: float  # time.perf_counter()


#: Public type alias for snapshot return values.
#: Consumers should treat this as opaque — store it and pass it back to ``record()``.
SnapshotPoint = _SnapshotPoint


@dataclass
class ProfileMarker:
    """
    Immutable record capturing memory and timing for a single profiled event.

    Attributes:
        event_name:           Human-readable event identifier.
        event_type:           Classification of the event context.
        ram_pss_bytes:        PSS at event entry (bytes).
        ram_delta_bytes:      PSS delta (end minus start) in bytes. Positive = allocated.
        swap_bytes:           Swap at event entry (bytes).
        swap_delta_bytes:     Swap delta in bytes.
        gpu_allocated_bytes:  GPU memory allocated at event entry (bytes).
        gpu_delta_bytes:      GPU allocated delta (end minus start) in bytes.
        gpu_peak_bytes:       Peak GPU memory during this event (bytes).
        duration_s:           Wall-clock duration in seconds.
        timestamp:            perf_counter value at event end.
        start_timestamp:      perf_counter value at event start (for chronological ordering).
    """

    event_name: str
    event_type: ProfileEventType
    ram_pss_bytes: int
    ram_delta_bytes: int
    swap_bytes: int
    swap_delta_bytes: int
    duration_s: float
    gpu_allocated_bytes: int = 0
    gpu_delta_bytes: int = 0
    gpu_peak_bytes: int = 0
    timestamp: float = field(default_factory=time.perf_counter)
    start_timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize this marker to a JSON-compatible dict."""
        return {
            "event": self.event_name,
            "type": self.event_type.value,
            "ram_entry_bytes": self.ram_pss_bytes,
            "ram_delta_bytes": self.ram_delta_bytes,
            "swap_entry_bytes": self.swap_bytes,
            "swap_delta_bytes": self.swap_delta_bytes,
            "gpu_entry_bytes": self.gpu_allocated_bytes,
            "gpu_delta_bytes": self.gpu_delta_bytes,
            "gpu_peak_bytes": self.gpu_peak_bytes,
            "wall_time_s": self.duration_s,
        }


class _ProfilerStore:
    """
    Holds all raw snapshot data, computed markers, and peak watermarks.

    Prefer accessing through the Profiler facade.  ``Profiler.create_isolated()``
    creates new instances internally when isolated stores are needed.
    """

    def __init__(self) -> None:
        self._markers: list[ProfileMarker] = []
        self._peak_ram_bytes: int = 0
        self._peak_swap_bytes: int = 0
        self._peak_gpu_bytes: int = 0
        self._peak_ram_event: str = ""
        self._peak_swap_event: str = ""
        self._peak_gpu_event: str = ""

    def reset(self) -> None:
        """Clear all markers and reset peak watermarks to current PSS."""
        self._markers.clear()
        self._peak_ram_event = ""
        self._peak_swap_event = ""
        self._peak_gpu_event = ""
        self._reset_peak_from_current()

    def _reset_peak_from_current(self) -> None:
        ram, swap = self._current_pss_swap()
        self._peak_ram_bytes = ram
        self._peak_swap_bytes = swap
        gpu_alloc, _ = _current_gpu()
        self._peak_gpu_bytes = gpu_alloc

    def _current_pss_swap(self) -> tuple[int, int]:
        """Return current (PSS, swap) in bytes for this process.

        Falls back to (RSS, 0) when /proc/pid/smaps is not readable
        (e.g. unprivileged containers). Logs a debug message on fallback.
        """
        try:
            mem = psutil.Process(os.getpid()).memory_full_info()
            return mem.pss, mem.swap
        except Exception as e:
            logging.getLogger(__name__).debug(
                f"PSS unavailable ({e}), falling back to RSS. Swap will report as 0."
            )
            return psutil.Process(os.getpid()).memory_info().rss, 0

    def update_peak(self, ram: int, swap: int, gpu: int = 0, event_name: str = "") -> None:
        """Update peak watermarks if the given values exceed the current peaks.

        Args:
            ram:        Current RAM PSS in bytes.
            swap:       Current swap in bytes.
            gpu:        Current GPU allocated memory in bytes.
            event_name: Name of the event that produced these values.
        """
        if ram > self._peak_ram_bytes:
            self._peak_ram_bytes = ram
            self._peak_ram_event = event_name
        if swap > self._peak_swap_bytes:
            self._peak_swap_bytes = swap
            self._peak_swap_event = event_name
        if gpu > self._peak_gpu_bytes:
            self._peak_gpu_bytes = gpu
            self._peak_gpu_event = event_name

    @property
    def ram_peak(self) -> int:
        """Peak RAM PSS observed across all events (bytes)."""
        return self._peak_ram_bytes

    @property
    def swap_peak(self) -> int:
        """Peak swap observed across all events (bytes)."""
        return self._peak_swap_bytes

    @property
    def ram_peak_event(self) -> str:
        """Name of the event that set the RAM peak."""
        return self._peak_ram_event

    @property
    def swap_peak_event(self) -> str:
        """Name of the event that set the swap peak."""
        return self._peak_swap_event

    @property
    def gpu_peak(self) -> int:
        """Peak GPU memory observed across all events (bytes)."""
        return self._peak_gpu_bytes

    @property
    def gpu_peak_event(self) -> str:
        """Name of the event that set the GPU peak."""
        return self._peak_gpu_event

    @property
    def markers(self) -> list[ProfileMarker]:
        """Defensive copy of all recorded ProfileMarkers."""
        return list(self._markers)

    def add_marker(self, marker: ProfileMarker) -> None:
        """Append a ProfileMarker to the store."""
        self._markers.append(marker)


_GLOBAL_STORE: _ProfilerStore = _ProfilerStore()


def _current_gpu() -> tuple[int, int]:
    """Return current (allocated, peak) GPU memory in bytes.

    Returns (0, 0) when CUDA is unavailable or torch is not installed.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            return torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device)
    except ImportError:
        pass
    return 0, 0


def _reset_gpu_peak() -> None:
    """Reset GPU peak memory stats. No-op if CUDA is unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    except ImportError:
        pass


class ResourceProfiler:
    """
    Facade over a profiler store.

    By default all ResourceProfiler instances share the module-level ``_GLOBAL_STORE``
    (used by the ``@resource_profile`` decorator).  Pass ``store`` to give a component
    its own isolated store — observers do this so they don't interfere with
    each other or with decorator-based profiling.

    Args:
        store:  Optional isolated ``_ProfilerStore``.  When ``None`` (the
                default), the module-level global store is used.
    """

    def __init__(
        self,
        store: _ProfilerStore | None = None,
    ) -> None:
        self._store = store if store is not None else _GLOBAL_STORE

    @classmethod
    def create_isolated(cls) -> "ResourceProfiler":
        """Create a ResourceProfiler with its own isolated store.

        Use this when you need profiling data that doesn't mix with the
        global ``@resource_profile`` decorator data (e.g. in observers).
        """
        return cls(store=_ProfilerStore())

    def snapshot(
        self,
        event_name: str,
        event_type: ProfileEventType = ProfileEventType.CUSTOM,
    ) -> _SnapshotPoint:
        """
        Capture current PSS/swap/GPU/time as a snapshot point.

        Args:
            event_name:  Label for this snapshot.
            event_type:  Event classification.

        Returns:
            SnapshotPoint for use as a start or end argument to record().
        """
        ram, swap = self._store._current_pss_swap()
        gpu_alloc, gpu_peak = _current_gpu()
        self._store.update_peak(ram, swap, gpu_alloc, event_name)
        return _SnapshotPoint(
            event_name=event_name,
            event_type=event_type,
            ram_pss_bytes=ram,
            swap_bytes=swap,
            gpu_allocated_bytes=gpu_alloc,
            gpu_peak_bytes=gpu_peak,
            t=time.perf_counter(),
        )

    def record(
        self,
        event_name: str,
        event_type: ProfileEventType,
        start_marker: _SnapshotPoint,
        end_marker: _SnapshotPoint,
    ) -> ProfileMarker:
        """
        Compute a ProfileMarker from start/end snapshot points and append to store.

        Args:
            event_name:   Name for the recorded event.
            event_type:   Classification of the event.
            start_marker: Snapshot taken before the event.
            end_marker:   Snapshot taken after the event.

        Returns:
            Computed ProfileMarker (also stored internally).
        """
        self._store.update_peak(
            end_marker.ram_pss_bytes, end_marker.swap_bytes, end_marker.gpu_allocated_bytes, event_name
        )
        marker = ProfileMarker(
            event_name=event_name,
            event_type=event_type,
            ram_pss_bytes=start_marker.ram_pss_bytes,
            ram_delta_bytes=end_marker.ram_pss_bytes - start_marker.ram_pss_bytes,
            swap_bytes=start_marker.swap_bytes,
            swap_delta_bytes=end_marker.swap_bytes - start_marker.swap_bytes,
            duration_s=end_marker.t - start_marker.t,
            gpu_allocated_bytes=start_marker.gpu_allocated_bytes,
            gpu_delta_bytes=end_marker.gpu_allocated_bytes - start_marker.gpu_allocated_bytes,
            gpu_peak_bytes=end_marker.gpu_peak_bytes,
            timestamp=end_marker.t,
            start_timestamp=start_marker.t,
        )
        self._store.add_marker(marker)
        return marker

    def reset_peak_memory_stats(self) -> None:
        """Reset peak RAM, swap, and GPU watermarks to current values."""
        self._store._reset_peak_from_current()

    def report(self, print_report: bool = True) -> "MemoryProfilingReport":
        """
        Build and return a MemoryProfilingReport from the current store contents.

        Prints the rich table to stdout when profiling is enabled (``QAIRT_PROFILING=1``)
        and ``print_report`` is True.

        Args:
            print_report: Set to False to suppress terminal output (e.g. in tests).

        Returns:
            MemoryProfilingReport with all recorded events, sorted by start time.
        """
        # Lazy import to avoid circular dependency (report.py imports ProfileMarker from here)
        from qairt.api.profiler.resource_profiler.decorator import _profiling_enabled
        from qairt.api.profiler.resource_profiler.report import MemoryProfilingReport

        report_data = MemoryProfilingReport(
            ram_peak=self._store.ram_peak,
            swap_peak=self._store.swap_peak,
            gpu_peak=self._store.gpu_peak,
            ram_peak_event=self._store.ram_peak_event,
            swap_peak_event=self._store.swap_peak_event,
            gpu_peak_event=self._store.gpu_peak_event,
            events=sorted(self._store.markers, key=lambda m: m.start_timestamp),
        )
        if print_report and _profiling_enabled():
            report_data.print()
        return report_data
