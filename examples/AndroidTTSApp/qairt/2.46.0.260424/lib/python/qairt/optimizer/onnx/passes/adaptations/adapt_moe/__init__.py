# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
MoE adaption pass
"""

from qairt.optimizer.onnx.passes.adaptations.adapt_moe.adapt_moe import AdaptMoE

__all__ = [
    "AdaptMoE",
]
