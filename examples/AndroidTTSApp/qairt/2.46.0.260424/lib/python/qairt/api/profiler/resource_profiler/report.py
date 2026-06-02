# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
MemoryProfilingReport: structured output for profiling results.

Provides rich table output to the terminal and JSON serialization.
"""

from __future__ import annotations

from typing import Any, Dict

import psutil
from pydantic import ConfigDict, Field, model_serializer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qairt.api.bases.report_base import Report
from qairt.api.profiler.resource_profiler.resource_profiler import ProfileMarker


def _bytes_to_mb(b: int) -> float:
    """Convert bytes to megabytes."""
    return b / (1024 * 1024)


def _fmt_delta_mb(b: int) -> str:
    """Format a byte delta as a signed MB string (e.g. '+128.0 MB' or '-64.0 MB')."""
    mb = _bytes_to_mb(b)
    sign = "+" if b > 0 else ""
    return f"{sign}{mb:.1f} MB"


def _fmt_duration(s: float) -> str:
    """Format duration as human-readable string."""
    if s < 60:
        return f"{s:.3f}s"
    minutes = int(s // 60)
    seconds = s % 60
    if minutes < 60:
        return f"{minutes}m {seconds:.1f}s"
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {seconds:.0f}s"


class MemoryProfilingReport(Report):
    """
    Structured report of profiling results.

    Attributes:
        ram_peak:        Peak RAM PSS observed across all events (bytes).
        swap_peak:       Peak swap observed across all events (bytes).
        gpu_peak:        Peak GPU memory observed across all events (bytes).
        ram_peak_event:  Name of the event that set the RAM peak.
        swap_peak_event: Name of the event that set the swap peak.
        gpu_peak_event:  Name of the event that set the GPU peak.
        events:          List of ProfileMarker records in order of recording.
    """

    ram_peak: int
    swap_peak: int
    gpu_peak: int = 0
    ram_peak_event: str = ""
    swap_peak_event: str = ""
    gpu_peak_event: str = ""
    events: list[ProfileMarker] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_serializer
    def serialize(self) -> Dict[str, Any]:
        mem_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        cpu_freq = psutil.cpu_freq()

        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "data": {
                "system_capacity": {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_freq_mhz": cpu_freq.max if cpu_freq else None,
                    "total_ram_gb": mem_info.total / (1024**3),
                    "swap_total_gb": swap_info.total / (1024**3),
                },
                "summary": {
                    "ram_peak_bytes": self.ram_peak,
                    "ram_peak_event": self.ram_peak_event,
                    "swap_peak_bytes": self.swap_peak,
                    "swap_peak_event": self.swap_peak_event,
                    "gpu_peak_bytes": self.gpu_peak,
                    "gpu_peak_event": self.gpu_peak_event,
                },
                "events": [e.to_dict() for e in self.events],
            },
        }

    def print(self) -> None:
        """
        Print a rich-formatted report to stdout.

        Layout:
          - Summary panel at the top with peak stats and total duration
          - Events table below with per-event details
        """
        console = Console(force_terminal=True)

        # ── Pre-compute disambiguated display names ───────────────────
        event_counts: dict[str, int] = {}
        for evt in self.events:
            event_counts[evt.event_name] = event_counts.get(evt.event_name, 0) + 1

        occurrence: dict[str, int] = {}
        display_names: dict[int, str] = {}  # index → display name
        for i, evt in enumerate(self.events):
            occurrence[evt.event_name] = occurrence.get(evt.event_name, 0) + 1
            if event_counts[evt.event_name] > 1:
                display_names[i] = (
                    f"{evt.event_name} [{occurrence[evt.event_name]}/{event_counts[evt.event_name]}]"
                )
            else:
                display_names[i] = evt.event_name

        def _display_name_for(event_name: str, key_fn) -> str:
            """Return the display name for the *event_name* occurrence with the highest *key_fn* value."""
            best_idx = None
            best_val = -1
            for i, evt in enumerate(self.events):
                if evt.event_name == event_name and key_fn(evt) > best_val:
                    best_val = key_fn(evt)
                    best_idx = i
            if best_idx is not None:
                return display_names[best_idx]
            return event_name

        # ── Summary panel ────────────────────────────────────────────
        total_duration = sum(evt.duration_s for evt in self.events)
        total_ram_delta = sum(evt.ram_delta_bytes for evt in self.events)
        slowest_event = max(self.events, key=lambda e: e.duration_s) if self.events else None
        has_gpu = any(evt.gpu_allocated_bytes > 0 or evt.gpu_peak_bytes > 0 for evt in self.events)

        summary = Text()
        summary.append("Peak RAM:   ", style="bold")
        summary.append(f"{_bytes_to_mb(self.ram_peak):.1f} MB", style="bold red")
        if self.ram_peak_event:
            summary.append(
                f"  ({_display_name_for(self.ram_peak_event, lambda e: e.ram_pss_bytes)})", style="dim"
            )
        summary.append("\n")
        summary.append("Peak Swap:  ", style="bold")
        summary.append(f"{_bytes_to_mb(self.swap_peak):.1f} MB", style="bold yellow")
        if self.swap_peak_event:
            summary.append(
                f"  ({_display_name_for(self.swap_peak_event, lambda e: e.swap_bytes)})", style="dim"
            )
        summary.append("\n")
        summary.append("Peak GPU:   ", style="bold")
        if has_gpu:
            summary.append(f"{_bytes_to_mb(self.gpu_peak):.1f} MB", style="bold magenta")
            if self.gpu_peak_event:
                summary.append(
                    f"  ({_display_name_for(self.gpu_peak_event, lambda e: e.gpu_allocated_bytes)})",
                    style="dim",
                )
        else:
            summary.append("N/A", style="dim")
        summary.append("\n")
        summary.append("Total Time: ", style="bold")
        summary.append(_fmt_duration(total_duration), style="bold green")
        summary.append("\n")
        if slowest_event:
            slowest_idx = self.events.index(slowest_event)
            summary.append("Peak Time:  ", style="bold")
            summary.append(_fmt_duration(slowest_event.duration_s), style="bold green")
            summary.append(f"  ({display_names[slowest_idx]})", style="dim")
            summary.append("\n")
        summary.append("Total RAM:  ", style="bold")
        delta_style = "red" if total_ram_delta > 0 else "green"
        summary.append(_fmt_delta_mb(total_ram_delta), style=f"bold {delta_style}")

        console.print(
            Panel(
                summary,
                title="[bold bright_white]Memory Profiling Summary[/bold bright_white]",
                border_style="bright_white",
                expand=False,
            )
        )

        # ── Events table ─────────────────────────────────────────────
        table = Table(
            title="Events",
            show_header=True,
            header_style="bold bright_white",
            show_lines=True,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Event", style="cyan", no_wrap=True)
        table.add_column("Type", style="blue")
        table.add_column("RAM Entry (MB)", justify="right", style="white")
        table.add_column("RAM Delta", justify="right")
        table.add_column("RAM Exit (MB)", justify="right", style="white")
        table.add_column("GPU Entry (MB)", justify="right", style="white")
        table.add_column("GPU Delta", justify="right")
        table.add_column("GPU Peak (MB)", justify="right", style="magenta")
        table.add_column("Swap Entry (MB)", justify="right", style="white")
        table.add_column("Swap Delta", justify="right")
        table.add_column("Duration", justify="right", style="bold white")

        # Track occurrence counts to disambiguate repeated event names
        # (already computed above in display_names)
        for i, evt in enumerate(self.events):
            display_name = display_names[i]

            ram_delta_style = "red" if evt.ram_delta_bytes > 0 else "green"
            swap_delta_style = (
                "red" if evt.swap_delta_bytes > 0 else "green" if evt.swap_delta_bytes < 0 else "white"
            )
            ram_exit = evt.ram_pss_bytes + evt.ram_delta_bytes

            row = [
                str(i + 1),
                display_name,
                evt.event_type.value,
                f"{_bytes_to_mb(evt.ram_pss_bytes):.1f}",
                f"[{ram_delta_style}]{_fmt_delta_mb(evt.ram_delta_bytes)}[/{ram_delta_style}]",
                f"{_bytes_to_mb(ram_exit):.1f}",
            ]
            if has_gpu:
                gpu_delta_style = (
                    "red" if evt.gpu_delta_bytes > 0 else "green" if evt.gpu_delta_bytes < 0 else "white"
                )
                row.extend(
                    [
                        f"{_bytes_to_mb(evt.gpu_allocated_bytes):.1f}",
                        f"[{gpu_delta_style}]{_fmt_delta_mb(evt.gpu_delta_bytes)}[/{gpu_delta_style}]",
                        f"{_bytes_to_mb(evt.gpu_peak_bytes):.1f}",
                    ]
                )
            else:
                row.extend(["[dim]N/A[/dim]", "[dim]N/A[/dim]", "[dim]N/A[/dim]"])
            row.extend(
                [
                    f"{_bytes_to_mb(evt.swap_bytes):.1f}",
                    f"[{swap_delta_style}]{_fmt_delta_mb(evt.swap_delta_bytes)}[/{swap_delta_style}]",
                    _fmt_duration(evt.duration_s),
                ]
            )
            table.add_row(*row)

        console.print(table)
