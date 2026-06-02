# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Simplification passes.
"""

from qairt.optimizer.onnx.passes.simplification.parallelize_serial_ops import (
    ParallelizeSerialOpsRewriter,
)

__all__ = [
    "ParallelizeSerialOpsRewriter",
]
