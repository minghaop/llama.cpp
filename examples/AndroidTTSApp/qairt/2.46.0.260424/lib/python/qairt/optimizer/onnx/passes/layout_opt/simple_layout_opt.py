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
    EliminateMHAReshapes,
    MergeSequenceReshapeOps,
    MergeSequenceTransposeOps,
)
from qairt.optimizer.onnx.passes.layout_opt import (
    LayoutConcatAfterQKVMatmulsRewriter,
    SimplifyReshapeTransposeSeqRewriter,
)


class SimpleLayoutOptRewriter(BasePass):
    """
    The entry pass for layout optimization
    """

    def apply(self, ctx: GraphContext) -> int:
        total_rewrite_count = 0

        total_rewrite_count += SimplifyReshapeTransposeSeqRewriter().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += LayoutConcatAfterQKVMatmulsRewriter().apply(ctx)
        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)

        total_rewrite_count += MergeSequenceTransposeOps().apply(ctx)
        total_rewrite_count += MergeSequenceReshapeOps().apply(ctx)

        total_rewrite_count += DeadCodeRemovalRewriter().apply(ctx)
        total_rewrite_count += DeadWeightRemovalRewriter().apply(ctx)

        total_rewrite_count += EliminateMHAReshapes().apply(ctx)

        DeadCodeRemovalRewriter().apply(ctx)
        DeadWeightRemovalRewriter().apply(ctx)

        return total_rewrite_count
