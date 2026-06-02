# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
import re
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger


def _initialize_loggers():
    """
    Initializes and registers loggers based on the configuration file.

    This function loads a YAML-based logging configuration and registers
    loggers for each defined logging area using the QAIRTLogger utility.
    It is intended to be called automatically when the module is imported.
    """
    config_file_path = Path(__file__).resolve().parents[1] / "logging_config.yaml"
    config = QAIRTLogger.load_logging_config(config_file_path)

    loggers_config = config.get("loggers", {})
    log_root_dir = Path(os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir()))
    unique_suffix = uuid.uuid4().hex

    for logger_name, logger_settings in loggers_config.items():
        log_working_dir = log_root_dir / f"logs_{logger_name}_{unique_suffix}"
        logger = QAIRTLogger.register_area_logger(
            area=LogAreas.register_log_area(logger_name),
            level=logger_settings.get("level", "INFO"),
            formatter_val=logger_settings.get("formatter"),
            handler_list=logger_settings.get("handlers", []),
            log_file_path=log_working_dir,
        )


# Automatically initialize loggers when this module is imported
_initialize_loggers()


def get_logger(name: str = "", level: str = "INFO") -> logging.Logger:
    """
    Returns a logger instance with the specified level and name.

    Args:
        name (str): A name for the python logger instance
        level (Union[int, str]): A valid logging level for the python logging module
    """
    return QAIRTLogger.get_logger(name, level)


@contextmanager
def maybe_suppress_stdout(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
) -> Generator[None, None, None]:
    """Suppresses stdout at the file-descriptor level, including output from C++ backends.

    Suppression is skipped if the logger's effective level is at or below ``level``,
    allowing output through when the logger is sufficiently verbose.

    Args:
        logger: Optional logger to check. If None, stdout is always suppressed.
        level: The threshold level. If the logger's effective level is <= this value,
            suppression is skipped. Defaults to DEBUG — only passes through when
            the logger is set to DEBUG. Pass ``logging.INFO`` to also pass through
            when the logger is set to INFO.
    """
    if logger is not None and logger.getEffectiveLevel() <= level:
        yield
        return

    sys.stdout.flush()
    old_fd = os.dup(1)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
        finally:
            os.close(devnull)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(old_fd, 1)
    finally:
        os.close(old_fd)
