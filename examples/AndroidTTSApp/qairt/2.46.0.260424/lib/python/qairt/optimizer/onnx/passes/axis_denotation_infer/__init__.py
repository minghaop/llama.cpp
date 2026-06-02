# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Axis denotation inference pass for propagating axis denotations of tensors"""

from qairt.optimizer.onnx.passes.axis_denotation_infer.axis_denotation_infer import AxisDenotationInference
from qairt.optimizer.onnx.passes.axis_denotation_infer.config import (
    AxisDenotationConfig,
    AxisDenotationSeedRule,
)

__all__ = ["AxisDenotationInference", "AxisDenotationSeedRule", "AxisDenotationConfig"]
