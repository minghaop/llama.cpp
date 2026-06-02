# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
ONNX Optimizer

Provides simple APIs for common ONNX optimization tasks
"""

# Easy-export GraphContext
from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes._api import (
    adapt_moe,
    change_context_length,
    change_seq_and_context_length,
    change_seq_length,
)

# IO Shape Rewriter exports
from qairt.optimizer.onnx.passes.axis_denotation_infer.config import (
    AxisDenotationConfig,
    AxisDenotationSeedRule,
)
from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation

__all__ = [
    # Core graph class
    "GraphContext",
    # IO Shape Rewriter APIs
    "AxisDenotationConfig",
    "AxisDenotationSeedRule",
    "AxisDenotation",
    "change_seq_length",
    "change_context_length",
    "change_seq_and_context_length",
    "adapt_moe",
]
