# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides M2sInsertMHASliceAfterQKVMatmul pass for mha2sha ir modification
"""

from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import GroupSliceAttrs
from qairt.optimizer.onnx.utils.utils import (
    OpTypePredicate,
    ValuePredicate,
    check_static_shape,
    get_slice_static_params,
    get_value_numeric_shape,
    has_static_shape_on_value,
    scan_previous_nearest_candidate,
)
from qairt.optimizer.utils.logger import logger


class AttnSinkTrimSlicePredicate(ValuePredicate):
    """
    Matches a Slice op that trims the attention-sink element from the last axis

    Attention sinks add 1 extra element to tgt_seq_len. After softmax, this element
    is sliced off before the matmul with value states. The pattern:
    - Producer is Slice
    - Slices on the last axis only
    - start=0 (trims from the tail)
    - Removes exactly 1 element (output_dim == input_dim - 1 on that axis)
    """

    def __call__(self, v: ir.Value) -> bool:
        producer = v.producer()
        if producer is None or producer.op_type != "Slice":
            return False
        if not has_static_shape_on_value(v):
            return False
        if producer.inputs[0] is None or not has_static_shape_on_value(producer.inputs[0]):
            return False

        params = get_slice_static_params(producer)

        input_shape = get_value_numeric_shape(producer.inputs[0])
        output_shape = get_value_numeric_shape(v)

        # No need to check if last_axis is -1
        # Since get_slice_static_params normalizes the slice axis
        last_axis = len(input_shape) - 1

        assert params is not None  # for mypy

        return (
            params["axes"] == [last_axis]
            and params["starts"] == [0]
            and params["steps"] == [1]
            and output_shape[last_axis] == input_shape[last_axis] - 1
        )


class M2sInsertMHASliceAfterQKVMatmul(M2sBasePass):
    """
    Pass to insert (GroupSlice -> Concat) after the QKVMatmul of every attention block
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for M2sInsertMHASliceAfterQKVMatmul pass"""

        qkv_matmul: ir.Node
        """The QKV matmul node"""

        softmax: ir.Node
        """The softmax node found in the attention pattern"""

    def __init__(self, m2s_head_split_map: dict[int, int] | None = None, out_batch_size: int | None = None):
        super().__init__()
        self.out_batch_size = out_batch_size
        self.m2s_head_split_map = m2s_head_split_map if m2s_head_split_map is not None else {}

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfo:
        if node.op_type != "MatMul":
            return False

        # Scan backward from qkv_matmul to find the Softmax, skipping over data-mover ops
        # and the attention-sink trim Slice (if present).
        # Ignored ops:
        # * Reshape/Transpose — pure data movers, do not affect attention computation
        # * Slice that removes exactly 1 element from the last axis (AttnSinkTrimSlicePredicate)
        #   — present in models that concatenate an attention sink token, expanding tgt_seq_len by 1
        #     before softmax, then slice it back off afterward

        # search bottom-up
        qkv_matmul = node
        softmax = None
        for candidate_v in scan_previous_nearest_candidate(
            start_values=list(node.inputs),
            check_fn=OpTypePredicate(["Softmax"]),
            ignore_fn=OpTypePredicate(["Reshape", "Transpose"]) | AttnSinkTrimSlicePredicate(),
        ):
            softmax = candidate_v.producer()

        if softmax is None:
            return False

        # check qkv_matmul output shape
        check_static_shape(qkv_matmul.outputs[0])
        assert qkv_matmul.outputs[0].shape is not None
        if qkv_matmul.outputs[0].shape.rank() not in (3, 4):
            return False

        return M2sInsertMHASliceAfterQKVMatmul.MatchInfo(qkv_matmul=qkv_matmul, softmax=softmax)

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, M2sInsertMHASliceAfterQKVMatmul.MatchInfo)  # For mypy

        qkv_matmul = match_info.qkv_matmul

        # get output shape
        value = qkv_matmul.outputs[0]
        qkv_matmul_out_shape = get_value_numeric_shape(value)
        out_rank = len(qkv_matmul_out_shape)

        if out_rank == 4:
            head_axis = 1
            head_num = qkv_matmul_out_shape[head_axis]
            # for batch splitting in the future
            # batch_axis = 0
            # batch_num = qkv_matmul_out_shape[batch_axis]
        elif out_rank == 3:
            head_axis = 0
            head_num = qkv_matmul_out_shape[head_axis]
            # for batch splitting in the future
            # batch_axis = None
            # batch_num = None
        else:
            assert False

        # split on head
        head_gslice_attrs = GroupSliceAttrs(axis=head_axis)

        if head_num in self.m2s_head_split_map:
            out_head_size = self.m2s_head_split_map[head_num]
        elif -1 in self.m2s_head_split_map:
            out_head_size = self.m2s_head_split_map[-1]
        else:
            out_head_size = 1

        for i, start_i in enumerate(range(0, head_num, out_head_size)):
            end_i = min(start_i + out_head_size, head_num)
            head_gslice_attrs.starts.append(start_i)
            head_gslice_attrs.ends.append(end_i)
            head_gslice_attrs.head_slice_ids.append(i)
            head_gslice_attrs.batch_slice_ids.append(-1)

        assert qkv_matmul.name is not None  # check for mypy
        # insert group slice and concat
        _, _ = self.gslice_then_concat(graph, value, head_gslice_attrs, qkv_matmul.name + "/head_gslice_node")

        logger.debug("found attention softmax '%s', added GroupSlice->Concat to it", match_info.softmax.name)
        return True
