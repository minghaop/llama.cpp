# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import shutil

from rich import table


class Colors:
    OKGREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


class PrettyPrintConstants:
    DEFAULT_LINE_LENGTH = 88
    GREEN = "40"
    RED = "160"
    Q_BLUE = "57"
    MAX_WIDTH = 4


def create_rich_table(
    title: str | None = None,
    caption: str | None = None,
    headers: list[str] = [],
    positions: list[float] = [],
    alignment: list[str] = [],
) -> table.Table:
    """Creates a blank rich table"""

    num_headers = len(headers)
    for attr in (positions, alignment):
        assert num_headers == len(attr)

    line_length = min(PrettyPrintConstants.DEFAULT_LINE_LENGTH, shutil.get_terminal_size().columns - 4)

    column_widths = []
    current = 0
    for pos in positions:
        width = int(pos * line_length) - current
        if width < PrettyPrintConstants.MAX_WIDTH:
            raise ValueError("Insufficient console width to print table.")
        column_widths.append(width)
        current += width

    columns = []
    for i, name in enumerate(headers):
        column = table.Column(
            name,
            justify=alignment[i],  # type: ignore[arg-type]
            width=column_widths[i],
        )
        columns.append(column)

    return table.Table(title=title, caption=caption, *columns, width=line_length, show_lines=True)


def bold_text(x, color=None):
    """Bolds text using rich markup."""
    if color:
        return f"[bold][color({color})]{x}[/][/]"
    return f"[bold]{x}[/]"
