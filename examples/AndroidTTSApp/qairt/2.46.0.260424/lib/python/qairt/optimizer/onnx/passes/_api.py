# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Simple API for ONNX optimizer passes"""

import logging
import os
import tempfile

from qairt.optimizer.onnx.passes.adaptations.adapt_moe.adapt_moe import adapt_moe
from qairt.optimizer.onnx.passes.io_shape_rewriter.io_shape_rewriter import (
    change_context_length,
    change_seq_and_context_length,
    change_seq_length,
)

logger = logging.getLogger(__name__)


__all__ = [
    "change_seq_length",
    "change_context_length",
    "change_seq_and_context_length",
    "adapt_moe",
]
