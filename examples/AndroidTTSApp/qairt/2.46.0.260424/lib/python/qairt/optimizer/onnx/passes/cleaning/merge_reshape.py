# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for merge a sequence of reshape ops
into a single reshape op.
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


class MergeSequenceReshapeOps(BasePredicatePass):
    """
    Transform subgraph:
        Subgraph(in_a) --> c
        {
            # allow in_a/in_a1/in_a2... have multiple users

            in_a1 = Reshape(in_a)
            in_a2 = Reshape(in_a2)
            ...
            c = Reshape(in_aX)
        }
    Into:
        Subgraph(in_a) --> c
        {
            c = Reshape(in_a)
        }
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for MergeSequenceReshapeOps pass"""

        reshape_seq: list[ir.Node]
        """Sequence of reshape nodes to merge"""

        v_extra_info: VariableExtraInfo
        """Variable extra info for encodings"""

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfo:
        if node.op_type != "Reshape":
            return False
        if not have_static_shape_on_node_io(node):
            return False

        reshape_seq = [node]
        v_extra_info = node.outputs[0].meta["extra_info"]

        # find bottom-up
        curr_node = node
        while True:
            curr_node_input = curr_node.inputs[0]

            # check for mypy, definitely true, since curr_node can only be Reshape
            assert curr_node_input is not None

            producer = curr_node_input.producer()
            if producer is None:
                break
            if producer.op_type != "Reshape":
                break
            curr_node = producer

            # do not merge reshape if encodings are not equal
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

            reshape_seq.insert(0, curr_node)

        if len(reshape_seq) > 1:
            return MergeSequenceReshapeOps.MatchInfo(reshape_seq=reshape_seq, v_extra_info=v_extra_info)
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, MergeSequenceReshapeOps.MatchInfo)  # For mypy

        reshape_seq = match_info.reshape_seq
        v_extra_info = match_info.v_extra_info

        input_shape = get_value_numeric_shape(reshape_seq[0].inputs[0])
        output_shape = get_value_numeric_shape(reshape_seq[-1].outputs[0])
        if tuple(input_shape) == tuple(output_shape):
            # reshape is useless, remove it
            safe_replace_all_uses_with(
                graph,
                reshape_seq[-1].outputs[0],
                reshape_seq[0].inputs[0],
                except_users=reshape_seq,
            )
        else:
            reshape_seq[-1].replace_input_with(0, reshape_seq[0].inputs[0])
        reshape_seq[-1].outputs[0].meta["extra_info"].merge(v_extra_info)
        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)

        return True
