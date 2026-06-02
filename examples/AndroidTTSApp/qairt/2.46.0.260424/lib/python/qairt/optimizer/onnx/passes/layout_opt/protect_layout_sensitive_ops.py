# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to protect/unprotect the layout sensitive ops,
such as Conv
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.layout_opt.base_rewriter import LayoutBasePredicatePass
from qairt.optimizer.onnx.utils.ir_extra_info import VariableExtraInfo
from qairt.optimizer.onnx.utils.utils import (
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger

CONV_CHN_LAST = "M2S_Conv_ChnLast"


def insert_transpose(graph: ir.Graph, input_v: ir.Value, perm, name_suffix):
    """
    Helper function, insert transpose to the input_v, and return the transposed_v

    Args:
        graph: ir.graph
        input_v: the input value to be transposed
        perm: the perm to be used for transpose
        name_suffix: the suffix to be used for the transpose node name
    Returns:
        transposed_v: the transposed value
    """
    op_name = graph.meta["extra_info"].get_unique_name(input_v.name + name_suffix + ".transpose")
    transpose_node = ir.Node("", "Transpose", [input_v], num_outputs=1, name=op_name)
    transpose_node.attributes["perm"] = ir.AttrInt64s("perm", perm)
    transposed_v = transpose_node.outputs[0]
    transposed_v.name = graph.meta["extra_info"].get_unique_name(input_v.name + name_suffix)

    transposed_v.meta["extra_info"] = VariableExtraInfo()
    transposed_v.meta["extra_info"] = input_v.meta["extra_info"].copy(ignore_safetensors=True)

    # infershape
    if input_v.shape is not None and input_v.shape.is_static():
        input_shape = input_v.shape.numpy()
        output_shape = [input_shape[perm[x]] for x in range(len(input_shape))]
        transposed_v.shape = ir.Shape(output_shape)
    if input_v.dtype is not None:
        transposed_v.dtype = input_v.dtype

    producer = input_v.producer()
    if producer is not None:
        graph.insert_after(producer, transpose_node)
    else:
        # value producer is None, so it should be input or initializers
        graph.insert_before(graph[0], transpose_node)

    return transposed_v


class ProtectLayoutSensitiveOps(LayoutBasePredicatePass):
    """
    Protect layout sensitive ops, such as Conv
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        return node.op_type in ["Conv"]

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        if node.op_type in ["Conv"]:
            # convert conv to conv.chn_last
            if not have_static_shape_on_node_io(node):
                logger.warning(
                    "cannot get shape of input/output of '%s', ignore to protect its layout", node.name
                )
                return False

            input_shape = get_value_numeric_shape(node.inputs[0])
            output_shape = get_value_numeric_shape(node.outputs[0])

            # insert permutation before conv
            input_rank = len(input_shape)  # N C D1 D2 D3 ...

            # pre_perm:
            pre_perm = list(range(input_rank))
            # transpose [N C D1 D2 D3 ...] to [N D1 D2 D3 ... C]
            input_perm = pre_perm[0:1] + pre_perm[2:] + pre_perm[1:2]
            # transpose [Cout Cin K1 K2 K3 ...] to [K1 K2 K3 ... Cin Cout]
            weight_perm = pre_perm[2:] + [1, 0]
            input_0 = node.inputs[0]
            input_1 = node.inputs[1]
            if input_0 is None or input_1 is None:
                logger.warning("Conv node '%s' has None inputs", node.name)
                return False
            transposed_input = insert_transpose(graph, input_0, input_perm, "/to_chn_last")
            transposed_weight = insert_transpose(graph, input_1, weight_perm, "/to_chn_last")

            # mark conv node is chn_last
            node.op_type = CONV_CHN_LAST
            node.replace_input_with(0, transposed_input)
            node.replace_input_with(1, transposed_weight)
            # new output shape is [N D1 D2 D3 ... C]
            new_output_shape = output_shape[0:1] + output_shape[2:] + output_shape[1:2]
            node.outputs[0].shape = ir.Shape(new_output_shape)

            # post_perm: transpose [N D1 D2 D3 ... C] to [N C D1 D2 D3 ...]
            post_perm = list(range(input_rank))
            post_perm = post_perm[0:1] + post_perm[-1:] + post_perm[1:-1]
            recovered_output = insert_transpose(graph, node.outputs[0], post_perm, "/from_chn_last")

            safe_replace_all_uses_with(
                graph, node.outputs[0], recovered_output, except_users=[recovered_output.producer()]
            )
            graph.meta["extra_info"].record_sharing_encodings(
                node.outputs[0].name, recovered_output.name, self.get_curr_pass_name()
            )

            logger.debug(
                "applied pass %s, protected %s %s", self.get_curr_pass_name(), node.op_type, node.name
            )
            return True
        return False


class UnProtectLayoutSensitiveOps(LayoutBasePredicatePass):
    """
    Unprotect layout sensitive ops, such as Conv
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        return node.op_type in [CONV_CHN_LAST]

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        if node.op_type in [CONV_CHN_LAST]:
            # convert conv to conv.chn_last
            if not have_static_shape_on_node_io(node):
                logger.warning(
                    "cannot get shape of input/output of '%s', ignore to unprotect its layout", node.name
                )
                return False

            input_shape = get_value_numeric_shape(node.inputs[0])
            output_shape = get_value_numeric_shape(node.outputs[0])

            # insert permutation before conv
            input_rank = len(input_shape)  # N D1 D2 D3 C ...

            # pre_perm:
            pre_perm = list(range(input_rank))
            # transpose [N D1 D2 D3 ... C] to [N C D1 D2 D3 ...]
            input_perm = pre_perm[0:1] + pre_perm[-1:] + pre_perm[1:-1]
            # transpose [K1 K2 K3 ... Cin Cout] to [Cout Cin K1 K2 K3 ...]
            weight_perm = [pre_perm[-1], pre_perm[-2]] + pre_perm[0:-2]
            input_0 = node.inputs[0]
            input_1 = node.inputs[1]
            if input_0 is None or input_1 is None:
                logger.warning("Conv node '%s' has None inputs", node.name)
                return False
            transposed_input = insert_transpose(graph, input_0, input_perm, "/axis_unprotect")
            transposed_weight = insert_transpose(graph, input_1, weight_perm, "/axis_unprotect")

            # mark conv node is the normal conv
            node.op_type = "Conv"
            node.replace_input_with(0, transposed_input)
            node.replace_input_with(1, transposed_weight)
            # new output shape is [N C D1 D2 D3 ... ], origin output is [N D1 D2 D3 ... C]
            new_output_shape = output_shape[0:1] + output_shape[-1:] + output_shape[1:-1]
            node.outputs[0].shape = ir.Shape(new_output_shape)

            # post_perm: transpose [N C D1 D2 D3 ...] to [N D1 D2 D3 ... C]
            post_perm = list(range(input_rank))
            post_perm = post_perm[0:1] + post_perm[2:] + post_perm[1:2]
            recovered_output = insert_transpose(graph, node.outputs[0], post_perm, "/axis_unprotect")

            safe_replace_all_uses_with(
                graph, node.outputs[0], recovered_output, except_users=[recovered_output.producer()]
            )
            graph.meta["extra_info"].record_sharing_encodings(
                node.outputs[0].name, recovered_output.name, self.get_curr_pass_name()
            )
            logger.debug(
                "applied pass %s, unprotected %s %s", self.get_curr_pass_name(), node.op_type, node.name
            )
            return True
        return False
