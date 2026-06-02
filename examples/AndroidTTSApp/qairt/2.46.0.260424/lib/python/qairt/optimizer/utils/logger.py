# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Logger utilities for ONNX Optimizer
"""

import logging

from qairt.utils import loggers

logger = loggers.get_logger("qairt.optimizer.onnx")


def set_log_level(level: str):
    """
    Configure logging level for all ONNX optimizer operations.

    This affects all passes, transformations, and operations within
    the ONNX optimizer module including MHA2SHA, splitter, and all
    graph rewriting passes.

    Args:
        level: Logging level. Supported values:
               - "DEBUG": Detailed diagnostic information
               - "INFO": General informational messages (default)
               - "WARNING": Warning messages only
               - "ERROR": Error messages only

    Example:
        >>> from qairt.optimizer.utils.logger import set_log_level
        >>> set_log_level("DEBUG")
        >>> # All subsequent optimizer operations will use DEBUG logging
    """
    logger.setLevel(getattr(logging, level.upper()))
