# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
IOShapeRewriter pass for context-length retargeting
"""

from qairt.optimizer.onnx.passes.axis_denotation_infer.config import (
    AxisDenotationConfig,
    AxisDenotationSeedRule,
)
from qairt.optimizer.onnx.passes.io_shape_rewriter.config import IOShapeRewriterConfig
from qairt.optimizer.onnx.passes.io_shape_rewriter.io_shape_rewriter import IOShapeRewriter
from qairt.optimizer.onnx.passes.io_shape_rewriter.node_update_passes import register_node_update_pass

__all__ = [
    "IOShapeRewriter",
    "IOShapeRewriterConfig",
    "AxisDenotationConfig",
    "AxisDenotationSeedRule",
    "register_node_update_pass",
]
