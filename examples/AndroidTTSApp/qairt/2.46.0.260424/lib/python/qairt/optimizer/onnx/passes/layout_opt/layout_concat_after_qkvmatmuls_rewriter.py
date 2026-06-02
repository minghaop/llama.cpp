# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides a pass to optimize concat node,
especially for the concat of qkv-matmuls in SHA graph
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.layout_opt.base_rewriter import LayoutBasePredicatePass
from qairt.optimizer.onnx.utils.reshape_transpose_analysis import (
    ReshapeTransposeInfoSeq,
)
from qairt.optimizer.onnx.utils.reshape_transpose_seq_utils import (
    ReshapeTransposeOpSeq,
    determine_seq_complexity,
    find_reshape_transpose_seq_top_down,
)
from qairt.optimizer.onnx.utils.utils import (
    convert_attr_to_py,
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class LayoutConcatAfterQKVMatmulsRewriter(LayoutBasePredicatePass):
    """
    Optimize layout of the concat of qkv matmuls in SHA graph

    Transform subgraph:
        Subgraph(in_h1, in_h2, in_h3) --> c_X
        {
            c = Concat(in_h1, in_h2, in_h3, axis=1)

            c_2 = Reshape(c_1)
            c_3 = Transpose(c_3)
            ... (a sequence of reshape/transpose that applied on c)
        }
    to:

        Subgraph(in_h1, in_h2, in_h3) --> c_X
        {
            c = Concat(in_h1, in_h2, in_h3, axis=3)  # AXIS CHANGED!

            c_2 = Reshape(c_1)
            c_3 = Transpose(c_3)
            ... (a SIMPILFIED sequence of reshape/transpose that applied on c, MAYBE EMPTY)
        }
    Also, encodings are handled
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type != "Concat":
            return False

        if not have_static_shape_on_node_io(node):
            return False
        input_shapes = [get_value_numeric_shape(x) for x in node.inputs]
        if any(x != input_shapes[0] for x in input_shapes):
            return False  # inputs shape must be same
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:  # pylint: disable=[too-many-locals]
        assert node.name is not None

        concat_axis = convert_attr_to_py(node.attributes["axis"], "as_int")
        out_shape = get_value_numeric_shape(node.outputs[0])

        if concat_axis < 0:
            concat_axis += len(out_shape)

        if concat_axis != len(out_shape) - 3:
            return False

        out_reshape_transpose_seq = find_reshape_transpose_seq_top_down(node.outputs[0], set())

        # try to remove concat axis to last
        new_concat_axis = len(out_shape) - 1
        src_info_seq = out_reshape_transpose_seq.build_info_seq()
        if src_info_seq is None:
            return False

        # split at current concat axis
        concat_num = len(node.inputs)
        new_concat_out_shape = list(get_value_numeric_shape(node.inputs[0]))
        new_concat_out_shape[new_concat_axis] *= concat_num

        reshape1_shape = list(get_value_numeric_shape(node.inputs[0]))
        reshape1_shape.insert(new_concat_axis, concat_num)

        reshape_axis_map = {}
        for i in range(len(out_shape)):
            if i <= new_concat_axis:
                reshape_axis_map[i] = i
            else:
                reshape_axis_map[i] = i + 1

        new_perm = list(range(len(reshape1_shape)))
        new_perm = (
            new_perm[: reshape_axis_map[new_concat_axis]] + new_perm[reshape_axis_map[new_concat_axis] + 1 :]
        )
        new_perm.insert(concat_axis, reshape_axis_map[new_concat_axis])

        reshape2_shape = list(get_value_numeric_shape(node.inputs[0]))
        reshape2_shape[concat_axis] *= concat_num

        try_info_seq = ReshapeTransposeInfoSeq(
            [
                ReshapeTransposeInfoSeq.ReshapeNodeInfo(reshape1_shape),
                ReshapeTransposeInfoSeq.TransposeNodeInfo(new_perm),
                ReshapeTransposeInfoSeq.ReshapeNodeInfo(reshape2_shape),
            ]
            + src_info_seq.seq[:],
            new_concat_out_shape,
        )

        opt_info_seq = try_info_seq.simplify_seq()

        if determine_seq_complexity(src_info_seq) > determine_seq_complexity(opt_info_seq):
            new_concat_name = graph.meta["extra_info"].get_unique_name_with_suffix(node.name, "/Simplify")
            new_concat = ir.Node(
                "",
                "Concat",
                node.inputs,
                num_outputs=1,
                name=new_concat_name,
                attributes=node.attributes.values(),
            )
            new_concat.attributes["axis"] = ir.AttrInt64("axis", new_concat_axis)
            assert node.outputs[0].name is not None
            new_concat.outputs[0].name = graph.meta["extra_info"].get_unique_name_with_suffix(
                node.outputs[0].name, ".simplify"
            )
            new_concat.outputs[0].shape = ir.Shape(new_concat_out_shape)
            if node.outputs[0].dtype is not None:
                new_concat.outputs[0].type = ir.TensorType(node.outputs[0].dtype)
            new_concat.outputs[0].meta["extra_info"] = (
                node.outputs[0].meta["extra_info"].copy(ignore_safetensors=True)
            )
            graph.insert_before(node, new_concat)

            opt_out_seq = ReshapeTransposeOpSeq.create_by_info_seq(
                graph, new_concat.outputs[0], out_reshape_transpose_seq.v_extra_info, opt_info_seq
            )
            if opt_out_seq is None:
                return False
            safe_replace_all_uses_with(graph, out_reshape_transpose_seq.output_v, opt_out_seq.output_v)

            graph.meta["extra_info"].record_copy(
                out_reshape_transpose_seq.output_v.name, opt_out_seq.output_v.name, self.get_curr_pass_name()
            )

            logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)
            return True
        return False
