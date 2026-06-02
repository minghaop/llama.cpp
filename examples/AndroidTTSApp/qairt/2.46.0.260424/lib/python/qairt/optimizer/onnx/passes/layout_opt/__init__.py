# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Layout optimization passes for ONNX models
"""

from qairt.optimizer.onnx.passes.layout_opt.base_rewriter import LayoutBasePredicatePass
from qairt.optimizer.onnx.passes.layout_opt.layout_binelewise_rewriter import LayoutBinelewiseRewriter
from qairt.optimizer.onnx.passes.layout_opt.layout_concat_after_qkvmatmuls_rewriter import (
    LayoutConcatAfterQKVMatmulsRewriter,
)
from qairt.optimizer.onnx.passes.layout_opt.protect_layout_sensitive_ops import (
    ProtectLayoutSensitiveOps,
    UnProtectLayoutSensitiveOps,
)
from qairt.optimizer.onnx.passes.layout_opt.simplify_concat_transpose import SimplifyConcatTransposeRewriter
from qairt.optimizer.onnx.passes.layout_opt.simplify_reshape_transpose_seq import (
    SimplifyReshapeTransposeSeqRewriter,
)

__all__ = [
    "LayoutBasePredicatePass",
    "LayoutBinelewiseRewriter",
    "LayoutConcatAfterQKVMatmulsRewriter",
    "ProtectLayoutSensitiveOps",
    "SimplifyConcatTransposeRewriter",
    "SimplifyReshapeTransposeSeqRewriter",
    "UnProtectLayoutSensitiveOps",
]
