# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the entry pass for layout optimization
"""

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass
from qairt.optimizer.onnx.passes.cleaning import (
    DeadCodeRemovalRewriter,
    DeadWeightRemovalRewriter,
    MergeSequenceReshapeOps,
    MergeSequenceTransposeOps,
)
from qairt.optimizer.onnx.passes.layout_opt import (
    LayoutBinelewiseRewriter,
    LayoutConcatAfterQKVMatmulsRewriter,
    ProtectLayoutSensitiveOps,
    SimplifyConcatTransposeRewriter,
    SimplifyReshapeTransposeSeqRewriter,
    UnProtectLayoutSensitiveOps,
)


class LayoutOptRewriter(BasePass):
    """
    The entry pass for layout optimization
    """

    def apply(self, ctx: GraphContext) -> int:
        total_rewrite_count = 0
        total_rewrite_count += ProtectLayoutSensitiveOps().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += SimplifyReshapeTransposeSeqRewriter().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        # a SimplifyReshapeTransposeSeqRewriter pass is required
        # before apply LayoutBinelewiseRewriter, to not floating bin elewise above unnecessarily
        total_rewrite_count += LayoutBinelewiseRewriter().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += LayoutConcatAfterQKVMatmulsRewriter().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += SimplifyReshapeTransposeSeqRewriter().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += SimplifyConcatTransposeRewriter().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += UnProtectLayoutSensitiveOps().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += MergeSequenceTransposeOps().apply(ctx)
        total_rewrite_count += MergeSequenceReshapeOps().apply(ctx)

        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)
        total_rewrite_count += DeadWeightRemovalRewriter().apply(ctx)
        return total_rewrite_count
