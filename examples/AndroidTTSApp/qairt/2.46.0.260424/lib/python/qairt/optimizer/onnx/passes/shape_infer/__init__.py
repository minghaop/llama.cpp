# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Shape inference passes for ONNX models
"""

from qairt.optimizer.onnx.passes.shape_infer.shape_infer import ShapeInference

__all__ = [
    "ShapeInference",
]
