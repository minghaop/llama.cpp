# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""ONNX optimizer utilities"""

from qairt.optimizer.onnx.utils.utils import (
    get_embedding_node,
    get_input_ids,
)

__all__ = [
    "get_embedding_node",
    "get_input_ids",
]
