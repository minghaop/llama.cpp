# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for merge a sequence of transpose ops
into a single transpose op.
"""

from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.utils.ir_extra_info import VariableExtraInfo
from qairt.optimizer.onnx.utils.utils import (
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class MergeSequenceTransposeOps(BasePredicatePass):
    """
    Transform subgraph:
        Subgraph(in_a) --> c
        {
            in_a1 = Transpose(in_a)
            in_a2 = Transpose(in_a2)
            ...
            c = Transpose(in_aX)
        }
    Into:
        Subgraph(in_a) --> c
        {
            c = Transpose(in_a)
        }
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for MergeSequenceTransposeOps pass"""

        transpose_seq: list[ir.Node]
        """Sequence of transpose nodes to merge"""

        v_extra_info: VariableExtraInfo
        """Variable extra info for encodings"""

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfo:
        if node.op_type != "Transpose":
            return False
        if not have_static_shape_on_node_io(node):
            return False

        transpose_seq = [node]
        v_extra_info = node.outputs[0].meta["extra_info"]

        # find top-down
        curr_node = node
        while True:
            uses = list(curr_node.outputs[0].uses())
            if len(uses) > 1 or len(uses) == 0:
                break
            if uses[0].node.op_type != "Transpose":
                break
            curr_node = uses[0].node

            if (
                v_extra_info.defined_encodings()
                and curr_node.outputs[0].meta["extra_info"].defined_encodings()
            ):
                if v_extra_info != curr_node.outputs[0].meta["extra_info"]:
                    break
            if (
                not v_extra_info.defined_encodings()
                and curr_node.outputs[0].meta["extra_info"].defined_encodings()
            ):
                v_extra_info = curr_node.outputs[0].meta["extra_info"]

            transpose_seq.append(curr_node)

        if len(transpose_seq) > 1:
            return MergeSequenceTransposeOps.MatchInfo(transpose_seq=transpose_seq, v_extra_info=v_extra_info)
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, MergeSequenceTransposeOps.MatchInfo)  # For mypy
        transpose_seq = match_info.transpose_seq
        v_extra_info = match_info.v_extra_info

        rank = len(get_value_numeric_shape(node.inputs[0]))
        merged_perm = list(range(rank))

        for transpose_node in transpose_seq[::-1]:
            curr_perm = transpose_node.attributes["perm"].as_ints()
            for i in range(rank):
                merged_perm[i] = curr_perm[merged_perm[i]]

        if merged_perm == list(range(rank)):
            safe_replace_all_uses_with(graph, transpose_seq[-1].outputs[0], transpose_seq[0].inputs[0])
        else:
            node_name = graph.meta["extra_info"].get_unique_name_with_suffix(node.name, ".merged")
            output_name = graph.meta["extra_info"].get_unique_name_with_suffix(
                transpose_seq[-1].outputs[0].name, ".merged"
            )
            new_node = ir.Node("", "Transpose", [node.inputs[0]], name=node_name)
            new_node.outputs[0].name = output_name
            new_node.outputs[0].shape = transpose_seq[-1].outputs[0].shape

            assert transpose_seq[-1].outputs[0].dtype is not None  # check for mypy, definitely true
            new_node.outputs[0].dtype = transpose_seq[-1].outputs[0].dtype
            new_node.attributes["perm"] = ir.AttrInt64s("perm", merged_perm)
            new_node.outputs[0].meta["extra_info"] = v_extra_info.copy(ignore_safetensors=True)
            graph.insert_before(transpose_seq[-1], new_node)
            graph.meta["extra_info"].record_sharing_encodings(
                transpose_seq[-1].outputs[0].name, new_node.outputs[0].name, self.get_curr_pass_name()
            )
            safe_replace_all_uses_with(graph, transpose_seq[-1].outputs[0], new_node.outputs[0])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)

        return True
