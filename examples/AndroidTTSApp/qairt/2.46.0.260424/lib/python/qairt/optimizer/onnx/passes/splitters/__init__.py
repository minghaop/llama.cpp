# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Model splitting passes for ONNX models
"""

from qairt.optimizer.onnx.passes.splitters.llm_splitter import (
    LLMSplitter,
)

__all__ = [
    "LLMSplitter",
]
