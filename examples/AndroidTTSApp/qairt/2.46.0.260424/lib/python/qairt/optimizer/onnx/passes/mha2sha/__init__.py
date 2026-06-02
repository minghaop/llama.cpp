# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
MHA to SHA transformation passes for ONNX models
"""

from qairt.optimizer.onnx.passes.mha2sha.fold_init_gslice import M2sFoldInitGroupSlice
from qairt.optimizer.onnx.passes.mha2sha.insert_group_slice_manually import M2sInsertGroupSliceManually
from qairt.optimizer.onnx.passes.mha2sha.insert_mhaslices_after_qkv_matmul import (
    M2sInsertMHASliceAfterQKVMatmul,
)
from qairt.optimizer.onnx.passes.mha2sha.mark_pack_qkv_splittable import M2sMarkPackQKVSplittable
from qairt.optimizer.onnx.passes.mha2sha.reorder_binelewise_gslice import M2sReorderBinElewiseGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_clip_gslice import M2sReorderClipGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_concat_gslice import M2sReorderConcatGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_conv_gslice import M2sReorderConvGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_cumsum_gslice import M2sReorderCumSumGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_expand_gslice import M2sReorderExpandGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_fasthadamardtransform_gslice import (
    M2sReorderFastHadamardTransformGroupslice,
)
from qairt.optimizer.onnx.passes.mha2sha.reorder_gather_gslice import M2sReorderGatherGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_instancenorm_gslice import M2sReorderInstancenormGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_layernorm_gslice import M2sReorderLayernormGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_matmul_gslice import M2sReorderMatmulGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_pad_gslice import M2sReorderPadGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_reduce_gslice import M2sReorderUnaryReduceGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_reshape_gslice import M2sReorderReshapeGroupSlice
from qairt.optimizer.onnx.passes.mha2sha.reorder_scatterelements_gslice import (
    M2sReorderScatterElementsGroupslice,
)
from qairt.optimizer.onnx.passes.mha2sha.reorder_slice_gslice import M2sReorderSliceGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_softmax_gslice import M2sReorderSoftmaxGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_spacetodepth_gslice import M2sReorderSpaceToDepthGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_squeeze_gslice import ReorderSqueezeGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_tile_gslice import M2sReorderTileGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_transpose_gslice import M2sReorderTransposeGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_unary_gslice import M2sReorderUnaryGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_unsqueeze_gslice import ReorderUnsqueezeGroupslice
from qairt.optimizer.onnx.passes.mha2sha.reorder_where_gslice import M2sReorderWhereGroupslice
from qairt.optimizer.onnx.passes.mha2sha.replace_gslice import M2sReplaceGroupSlice
from qairt.optimizer.onnx.passes.mha2sha.replace_slice_gslice_with_gslice_gslice import (
    M2sReplaceSliceGroupslice2GroupsliceGroupslice,
)
from qairt.optimizer.onnx.passes.mha2sha.replace_split_gslice_with_gslice_gslice import (
    M2sReplaceSplitGroupslice2GroupsliceGroupslice,
)
from qairt.optimizer.onnx.passes.mha2sha.replace_split_gslice_with_slice_gslice import (
    M2sReplaceSplitGroupslice2SliceGroupslice,
)

__all__ = [
    "M2sFoldInitGroupSlice",
    "M2sInsertGroupSliceManually",
    "M2sInsertMHASliceAfterQKVMatmul",
    "M2sMarkPackQKVSplittable",
    "M2sReorderBinElewiseGroupslice",
    "M2sReorderClipGroupslice",
    "M2sReorderConcatGroupslice",
    "M2sReorderConvGroupslice",
    "M2sReorderCumSumGroupslice",
    "M2sReorderExpandGroupslice",
    "M2sReorderFastHadamardTransformGroupslice",
    "M2sReorderInstancenormGroupslice",
    "M2sReorderLayernormGroupslice",
    "M2sReorderPadGroupslice",
    "M2sReorderGatherGroupslice",
    "M2sReorderMatmulGroupslice",
    "M2sReorderUnaryReduceGroupslice",
    "M2sReorderReshapeGroupSlice",
    "M2sReorderScatterElementsGroupslice",
    "M2sReorderSliceGroupslice",
    "M2sReorderSoftmaxGroupslice",
    "M2sReorderSpaceToDepthGroupslice",
    "M2sReorderTileGroupslice",
    "M2sReorderTransposeGroupslice",
    "M2sReorderUnaryGroupslice",
    "M2sReorderWhereGroupslice",
    "M2sReplaceGroupSlice",
    "M2sReplaceSliceGroupslice2GroupsliceGroupslice",
    "M2sReplaceSplitGroupslice2GroupsliceGroupslice",
    "M2sReplaceSplitGroupslice2SliceGroupslice",
    "ReorderSqueezeGroupslice",
    "ReorderUnsqueezeGroupslice",
]
