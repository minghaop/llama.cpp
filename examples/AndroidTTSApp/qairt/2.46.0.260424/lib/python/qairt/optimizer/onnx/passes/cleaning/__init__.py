# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Graph cleaning passes for ONNX models
"""

from qairt.optimizer.onnx.passes.cleaning.eliminate_mha_reshapes import EliminateMHAReshapes
from qairt.optimizer.onnx.passes.cleaning.merge_reshape import MergeSequenceReshapeOps
from qairt.optimizer.onnx.passes.cleaning.merge_transposes import MergeSequenceTransposeOps
from qairt.optimizer.onnx.passes.cleaning.remove_dead_nodes import DeadCodeRemovalRewriter
from qairt.optimizer.onnx.passes.cleaning.remove_null_concat import NullConcatRemovalRewriter
from qairt.optimizer.onnx.passes.cleaning.remove_null_mul import NullMulRemovalRewriter
from qairt.optimizer.onnx.passes.cleaning.remove_null_slice import NullSliceRemovalRewriter
from qairt.optimizer.onnx.passes.cleaning.remove_null_tile import NullTileRemovalRewriter
from qairt.optimizer.onnx.passes.cleaning.remove_unused_functions import DeadFunctionRemovalRewriter
from qairt.optimizer.onnx.passes.cleaning.remove_unused_weights import DeadWeightRemovalRewriter

__all__ = [
    "DeadCodeRemovalRewriter",
    "DeadWeightRemovalRewriter",
    "DeadFunctionRemovalRewriter",
    "MergeSequenceReshapeOps",
    "MergeSequenceTransposeOps",
    "NullConcatRemovalRewriter",
    "NullMulRemovalRewriter",
    "NullSliceRemovalRewriter",
    "NullTileRemovalRewriter",
    "EliminateMHAReshapes",
]
