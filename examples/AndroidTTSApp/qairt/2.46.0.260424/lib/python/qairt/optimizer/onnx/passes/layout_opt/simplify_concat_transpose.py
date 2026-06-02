# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides a pass to simplify Concat+Transpose to Concat
"""

from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.utils.reshape_transpose_analysis import ReshapeTransposeInfoSeq
from qairt.optimizer.onnx.utils.utils import (
    convert_attr_to_py,
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class SimplifyConcatTransposeRewriter(BasePredicatePass):
    """
    A graph rewriter pass that simplify reshape transpose consequences after concat

    simplify pattern
        Concat->Transpose
    To
        Concat

    For example, simplify
        Transpose(Concat(X0,X1,X2, axis=1), perm=[1,0,2,3]),
            where X0,X1,X2 has shape [1,1,128,64]
    To
        Concat(X0,X1,X2, axis=0)

    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for SimplifyConcatTransposeRewriter pass"""

        concat_node: ir.Node
        """The concat node"""

        transpose_node: ir.Node
        """The transpose node"""

        transpose_perm: list[int]
        """The transpose permutation"""

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfo:
        if node.op_type != "Concat":
            return False
        concat_node = node
        concat_out = node.outputs[0]

        # transpose node is the only one consumer of concat
        uses = list(concat_out.uses())
        if len(uses) != 1:
            return False

        if uses[0].node.op_type != "Transpose":
            return False
        transpose_node = uses[0].node

        if not have_static_shape_on_node_io(concat_node):
            return False
        if not have_static_shape_on_node_io(transpose_node):
            return False

        transpose_perm = list(transpose_node.attributes["perm"].as_ints())

        # case1, transpose can be eliminated if concat axis is permuted
        for v in concat_node.inputs:
            if v is not None and v.shape is not None:
                pre_info_seq = ReshapeTransposeInfoSeq(
                    [ReshapeTransposeInfoSeq.TransposeNodeInfo(transpose_perm)],
                    v.shape.numpy(),
                )
                pre_info_seq = pre_info_seq.simplify_seq()
                if len(pre_info_seq.seq) != 0:
                    return False

        return SimplifyConcatTransposeRewriter.MatchInfo(
            concat_node=concat_node, transpose_node=transpose_node, transpose_perm=transpose_perm
        )

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, SimplifyConcatTransposeRewriter.MatchInfo)  # For mypy
        concat_node = match_info.concat_node
        transpose_node = match_info.transpose_node
        transpose_perm = match_info.transpose_perm

        concat_axis = convert_attr_to_py(concat_node.attributes["axis"], "as_int")
        if concat_axis < 0:
            concat_axis += len(get_value_numeric_shape(concat_node.inputs[0]))
        new_concat_axis = transpose_perm.index(concat_axis)
        new_concat_node = ir.Node(
            "",
            "Concat",
            concat_node.inputs,
            attributes=[ir.AttrInt64("axis", new_concat_axis)],
            name=graph.meta["extra_info"].get_unique_name_with_suffix(concat_node.name, "/simplified"),
        )
        new_concat_node.outputs[0].name = graph.meta["extra_info"].get_unique_name_with_suffix(
            concat_node.outputs[0].name, "/simplified"
        )
        self.mark_value_as_copy(graph, transpose_node.outputs[0], new_concat_node.outputs[0])
        graph.insert_after(transpose_node, new_concat_node)
        safe_replace_all_uses_with(graph, transpose_node.outputs[0], new_concat_node.outputs[0])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), concat_node.name)
        return True
