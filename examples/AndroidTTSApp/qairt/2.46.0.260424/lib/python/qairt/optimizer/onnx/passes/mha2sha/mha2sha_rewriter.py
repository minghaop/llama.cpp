# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the entry pass for mha2sha optimization
"""

from typing import TypeAlias

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass
from qairt.optimizer.onnx.passes.cleaning import (
    DeadCodeRemovalRewriter,
    DeadWeightRemovalRewriter,
    NullConcatRemovalRewriter,
    NullMulRemovalRewriter,
    NullSliceRemovalRewriter,
    NullTileRemovalRewriter,
)
from qairt.optimizer.onnx.passes.mha2sha import (
    M2sFoldInitGroupSlice,
    M2sInsertGroupSliceManually,
    M2sInsertMHASliceAfterQKVMatmul,
    M2sMarkPackQKVSplittable,
    M2sReorderBinElewiseGroupslice,
    M2sReorderClipGroupslice,
    M2sReorderConcatGroupslice,
    M2sReorderConvGroupslice,
    M2sReorderCumSumGroupslice,
    M2sReorderExpandGroupslice,
    M2sReorderFastHadamardTransformGroupslice,
    M2sReorderGatherGroupslice,
    M2sReorderInstancenormGroupslice,
    M2sReorderLayernormGroupslice,
    M2sReorderMatmulGroupslice,
    M2sReorderPadGroupslice,
    M2sReorderReshapeGroupSlice,
    M2sReorderScatterElementsGroupslice,
    M2sReorderSliceGroupslice,
    M2sReorderSoftmaxGroupslice,
    M2sReorderSpaceToDepthGroupslice,
    M2sReorderTileGroupslice,
    M2sReorderTransposeGroupslice,
    M2sReorderUnaryGroupslice,
    M2sReorderUnaryReduceGroupslice,
    M2sReorderWhereGroupslice,
    M2sReplaceGroupSlice,
    M2sReplaceSliceGroupslice2GroupsliceGroupslice,
    M2sReplaceSplitGroupslice2GroupsliceGroupslice,
    M2sReplaceSplitGroupslice2SliceGroupslice,
    ReorderSqueezeGroupslice,
    ReorderUnsqueezeGroupslice,
)
from qairt.optimizer.onnx.passes.mha2sha.config import MHA2SHAConfig


class MHA2SHARewriter(BasePass):
    """
    The entry pass for mha2sha optimization
    It has these steps:
    - pre_stage:
        - Insert GroupSlice->Concat after QKV_Matmul of every attention block
    - proc_stage:
        - Reorder every possible (X->GroupSlice) pattern to (GroupSlice->X) in a loop
    - post_stage:
        - clean the graph
    """

    Config: TypeAlias = MHA2SHAConfig

    def __init__(self, config: Config | None = None):
        if config is None:
            config = MHA2SHARewriter.Config()
        super().__init__()
        self.config: MHA2SHARewriter.Config = config

    def apply(self, ctx: GraphContext) -> int:
        # Pre-stage passes
        pre_stages: list[BasePass] = [
            M2sInsertMHASliceAfterQKVMatmul(
                m2s_head_split_map=self.config.m2s_head_split_map,
                out_batch_size=self.config.out_batch_size,
            ),
            M2sInsertGroupSliceManually(self.config.m2s_additional_start_points),
            M2sMarkPackQKVSplittable(),
        ]

        # Processing stage passes
        proc_stages: list[BasePass] = [
            M2sReorderMatmulGroupslice(),
            M2sReorderSoftmaxGroupslice(),
            M2sReorderLayernormGroupslice(),
            M2sReorderInstancenormGroupslice(),
            M2sReorderUnaryReduceGroupslice(),
            M2sReorderBinElewiseGroupslice(),
            M2sReorderConcatGroupslice(),
            M2sReorderCumSumGroupslice(),
            M2sReplaceSliceGroupslice2GroupsliceGroupslice(),
            M2sReorderSliceGroupslice(),
            M2sReorderTileGroupslice(),
            M2sReorderTransposeGroupslice(),
            M2sReorderReshapeGroupSlice(),
            M2sReorderConvGroupslice(),
            M2sReorderUnaryGroupslice(),
            M2sReplaceSplitGroupslice2GroupsliceGroupslice(),
            M2sReplaceSplitGroupslice2SliceGroupslice(),
            M2sReorderExpandGroupslice(),
            M2sReorderWhereGroupslice(),
            M2sReorderScatterElementsGroupslice(),
            M2sReorderSpaceToDepthGroupslice(),
            M2sReorderClipGroupslice(),
            M2sReorderFastHadamardTransformGroupslice(),
            ReorderSqueezeGroupslice(),
            ReorderUnsqueezeGroupslice(),
            M2sReorderGatherGroupslice(),
            M2sReorderPadGroupslice(),
        ]

        # Post-stage passes
        post_stages: list[BasePass] = [
            DeadCodeRemovalRewriter(),
            M2sFoldInitGroupSlice(),
            M2sReplaceGroupSlice(),
            NullMulRemovalRewriter(),
            NullConcatRemovalRewriter(),
            NullSliceRemovalRewriter(),
            NullTileRemovalRewriter(),
            DeadCodeRemovalRewriter(),
            DeadWeightRemovalRewriter(),
        ]

        total_rewrite_count = 0
        for rewriter in pre_stages:
            total_rewrite_count += rewriter.apply(ctx)

        curr_rewrite_count = 1
        loop_count = 0
        loop_max = 10000
        while curr_rewrite_count > 0 and loop_count < loop_max:
            curr_rewrite_count = 0
            for proc_rewriter in proc_stages:
                count = proc_rewriter.apply(ctx)
                if count > 0:
                    # Clean the graph by removing unused nodes
                    # to prevent the optimizer from being misled.
                    DeadCodeRemovalRewriter().apply(ctx)
                curr_rewrite_count += count
            loop_count += 1
            total_rewrite_count += curr_rewrite_count

        if loop_count == loop_max:
            assert False, "dead loop, something wrong"

        for post_rewriter in post_stages:
            total_rewrite_count += post_rewriter.apply(ctx)

        return total_rewrite_count
