# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides a pass to optimize in/out layout of the binary elewise op
"""

import copy

import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.layout_opt.base_rewriter import LayoutBasePredicatePass
from qairt.optimizer.onnx.utils.reshape_transpose_seq_utils import (
    ReshapeTransposeInfoSeq,
    ReshapeTransposeOpSeq,
    determine_seq_complexity,
    find_reshape_transpose_seq_bottom_up,
    find_reshape_transpose_seq_top_down,
    remove_common_ancesters_of_reshape_transpose_op_seq,
)
from qairt.optimizer.onnx.utils.utils import (
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class LayoutBinelewiseRewriter(LayoutBasePredicatePass):
    """
    Optimize input/output layout of the binary elewise op

    Transform subgraph:
        Subgraph(in_a, in_b) --> c_X
        {
            in_a_1 = Transpose(in_a)
            in_a_2 = Reshape(in_a_1)
            ... (a sequence of reshape/transpose that applied on in_a)

            in_b_1 = Transpose(in_b)
            in_b_2 = Reshape(in_b_1)
            ... (a sequence of reshape/transpose that applied on in_b)

            c = BinElewiseOp(in_a_X, in_b_X)


            c_1 = Transpose(c)
            c_2 = Reshape(c_1)
            ... (a sequence of reshape/transpose that applied on c)
        }
    to:
        Subgraph(in_a, in_b) --> c,c0,c1,c2,...
        {

            in_a_1 = Transpose(in_a)
            in_a_2 = Reshape(in_a_1)
            ... (a SIMPLIFIED sequence of reshape/transpose that applied on in_a, MAYBE EMPTY)

            in_b_1 = Transpose(in_b)
            in_b_2 = Reshape(in_b_1)
            ... (a SIMPLIFIED sequence of reshape/transpose that applied on in_b, MAYBE EMPTY)

            c = BinElewiseOp(in_a_X, in_b_X)

            c_1 = Transpose(c)
            c_2 = Reshape(c_1)
            ... (a SIMPLIFIED sequence of reshape/transpose that applied on c, MAYBE EMPTY)
        }

    Also, encodings are handled


    Visualization of the optimization:

    Optimize the subgraph

    [reshape/transpose sequence A] --> X --\\

                                        BinaryOp --> Z --> [reshape/transpose sequence C]

    [reshape/transpose sequence B] --> Y --/

    Into

    [simplified reshape/transpose sequence A_sim] --> X --\\

                                                        BinaryOp --> Z --> [simplified reshape/transpose
                                                                            sequence C_sim]

    [simplified reshape/transpose sequence B_sim] --> Y --/
    """

    SUPPORT_OP_TYPES = {"Add", "Minus", "Mul", "Div"}

    def pre_rewrite_hook(self, ctx: GraphContext):
        if "_layout_binelewise_exclude_names" not in ctx.graph_ir.meta:
            ctx.graph_ir.meta["_layout_binelewise_exclude_names"] = set()
        else:
            ctx.graph_ir.meta["_layout_binelewise_exclude_names"].clear()

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type not in self.SUPPORT_OP_TYPES:
            return False
        if not have_static_shape_on_node_io(node):
            return False

        input_shapes = [get_value_numeric_shape(node.inputs[0]), get_value_numeric_shape(node.inputs[1])]

        # special case, one input is one-dim
        if any(x in ((1,), tuple()) for x in input_shapes) and not all(
            x in ((1,), tuple()) for x in input_shapes
        ):
            return True

        # do not support broadcast for now
        if input_shapes[0] != input_shapes[1]:
            return False

        return True

    def rewrite_tensor_scalar(self, graph: ir.Graph, node: ir.Node):  # pylint: disable=[too-many-locals]
        """
        Optimize for special case
        one input of binary op is a scalar,
        Another input is a tensor
        """
        assert node.name is not None  # check for mypy
        assert node.outputs[0].name is not None  # check for mypy

        exclude_names = graph.meta.get("_layout_binelewise_exclude_names", set())

        input_shapes = [get_value_numeric_shape(node.inputs[0]), get_value_numeric_shape(node.inputs[1])]
        if input_shapes[0] in ((1,), tuple()):
            in_tensor_id = 1
            in_scalar_id = 0
        else:
            in_tensor_id = 0
            in_scalar_id = 1

        in_tensor_v = node.inputs[in_tensor_id]
        assert in_tensor_v is not None  # check for mypy
        in_layout_transform_op_seq = find_reshape_transpose_seq_bottom_up(in_tensor_v, exclude_names)

        in_info_seq = in_layout_transform_op_seq.build_info_seq()
        if in_info_seq is None:
            return False

        out_layout_transform_op_seq = find_reshape_transpose_seq_top_down(node.outputs[0], exclude_names)
        out_info_seq = out_layout_transform_op_seq.build_info_seq()
        if out_info_seq is None:
            return False

        origin_complexity = determine_seq_complexity(in_info_seq) + determine_seq_complexity(out_info_seq)

        if len(out_info_seq.seq) > 0:
            # try to remove all output layout transform seq to input a and b

            merged_in_info_seq = ReshapeTransposeInfoSeq(
                in_info_seq.seq + out_info_seq.seq, in_info_seq.input_shape
            )

            opt_in_info_seq = merged_in_info_seq.simplify_seq()
            opt_out_info_seq = ReshapeTransposeInfoSeq([], out_info_seq.infer_shape())
        else:
            opt_in_info_seq = in_info_seq.simplify_seq()
            opt_out_info_seq = out_info_seq.simplify_seq()

        opt_out_info_seq = opt_out_info_seq.simplify_seq()
        opt_complexity = determine_seq_complexity(opt_in_info_seq) + determine_seq_complexity(
            opt_out_info_seq
        )
        if opt_complexity < origin_complexity:
            new_in_op_seq = ReshapeTransposeOpSeq.create_by_info_seq(
                graph,
                in_layout_transform_op_seq.input_v,
                in_layout_transform_op_seq.v_extra_info,
                opt_in_info_seq,
            )
            if new_in_op_seq is None:
                return False

            new_node = ir.Node(
                "",
                node.op_type,
                [new_in_op_seq.output_v, node.inputs[in_scalar_id]],
                num_outputs=1,
                name=graph.meta["extra_info"].get_unique_name_with_suffix(node.name, "/simplify"),
            )
            for k, v in node.attributes.items():
                new_node.attributes[k] = copy.deepcopy(v)

            new_node.outputs[0].name = graph.meta["extra_info"].get_unique_name_with_suffix(
                node.outputs[0].name, "/simplify"
            )
            new_node.outputs[0].shape = ir.Shape(new_in_op_seq.output_v.shape.numpy())
            new_node.outputs[0].dtype = new_in_op_seq.output_v.dtype
            new_node.outputs[0].meta["extra_info"] = out_layout_transform_op_seq.input_v.meta[
                "extra_info"
            ].copy(ignore_safetensors=True)
            graph.insert_before(node, new_node)
            graph.meta["extra_info"].record_sharing_encodings(
                node.outputs[0].name, new_node.outputs[0].name, self.get_curr_pass_name()
            )
            assert out_layout_transform_op_seq.output_v.shape.numpy() == new_node.outputs[0].shape.numpy()
            safe_replace_all_uses_with(graph, out_layout_transform_op_seq.output_v, new_node.outputs[0])

            logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)
            return True

        return False

    # pylint: disable=[too-many-locals,too-many-branches]
    # pylint: disable=[too-many-statements,too-many-return-statements]
    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert node.name is not None  # check for mypy

        exclude_names = graph.meta.get("_layout_binelewise_exclude_names", set())

        # Check if one input is scalar
        input_shapes = [get_value_numeric_shape(node.inputs[0]), get_value_numeric_shape(node.inputs[1])]
        one_input_is_scalar = any(x in ((1,), tuple()) for x in input_shapes) and not all(
            x in ((1,), tuple()) for x in input_shapes
        )

        if one_input_is_scalar:
            return self.rewrite_tensor_scalar(graph, node)

        assert node.inputs[0] is not None  # make mypy happy, definitely true
        assert node.inputs[1] is not None  # make mypy happy, definitely true
        in_a_layout_transform_op_seq = find_reshape_transpose_seq_bottom_up(node.inputs[0], exclude_names)
        in_b_layout_transform_op_seq = find_reshape_transpose_seq_bottom_up(node.inputs[1], exclude_names)
        in_a_layout_transform_op_seq, in_b_layout_transform_op_seq = (
            remove_common_ancesters_of_reshape_transpose_op_seq(
                in_a_layout_transform_op_seq, in_b_layout_transform_op_seq
            )
        )
        in_a_info_seq = in_a_layout_transform_op_seq.build_info_seq()
        in_b_info_seq = in_b_layout_transform_op_seq.build_info_seq()
        if in_a_info_seq is None:
            return False
        if in_b_info_seq is None:
            return False

        out_layout_transform_op_seq = find_reshape_transpose_seq_top_down(node.outputs[0], exclude_names)
        out_info_seq = out_layout_transform_op_seq.build_info_seq()
        if out_info_seq is None:
            return False

        origin_complexity = (
            determine_seq_complexity(in_a_info_seq)
            + determine_seq_complexity(in_b_info_seq)
            + determine_seq_complexity(out_info_seq)
        )

        if len(out_info_seq.seq) > 0:
            # try to remove all output layout transform seq to input a and b

            merged_in_a_info_seq = ReshapeTransposeInfoSeq(
                in_a_info_seq.seq + out_info_seq.seq, in_a_info_seq.input_shape
            )
            merged_in_b_info_seq = ReshapeTransposeInfoSeq(
                in_b_info_seq.seq + out_info_seq.seq, in_b_info_seq.input_shape
            )

            opt_in_a_info_seq = merged_in_a_info_seq.simplify_seq()
            opt_in_b_info_seq = merged_in_b_info_seq.simplify_seq()
            opt_out_info_seq = ReshapeTransposeInfoSeq([], out_info_seq.infer_shape())
        else:
            opt_in_a_info_seq = in_a_info_seq.simplify_seq()
            opt_in_b_info_seq = in_b_info_seq.simplify_seq()
            opt_out_info_seq = out_info_seq.simplify_seq()

        # if possible, move some layout transform to output if they are same in a and b
        opt_in_a_intermedate_shapes = opt_in_a_info_seq.get_intermediate_shapes()
        opt_in_b_intermedate_shapes = opt_in_b_info_seq.get_intermediate_shapes()
        while len(opt_in_a_info_seq.seq) > 0 and len(opt_in_b_info_seq.seq) > 0:
            last_a_n_info = opt_in_a_info_seq.seq[-1]
            last_b_n_info = opt_in_b_info_seq.seq[-1]
            last_a_n_info_input = opt_in_a_intermedate_shapes[-2]
            last_b_n_info_input = opt_in_b_intermedate_shapes[-2]
            if last_a_n_info == last_b_n_info and last_a_n_info_input == last_b_n_info_input:
                opt_in_a_info_seq.seq = opt_in_a_info_seq.seq[:-1]
                opt_in_b_info_seq.seq = opt_in_b_info_seq.seq[:-1]
                opt_in_a_intermedate_shapes = opt_in_a_intermedate_shapes[:-1]
                opt_in_b_intermedate_shapes = opt_in_b_intermedate_shapes[:-1]
                opt_out_info_seq.seq.insert(0, last_a_n_info)
                opt_out_info_seq.input_shape = opt_in_a_info_seq.infer_shape()
            else:
                break
        # opt_in_a_info_seq = opt_in_a_info_seq.simplify_seq()
        # opt_in_b_info_seq = opt_in_b_info_seq.simplify_seq()
        opt_out_info_seq = opt_out_info_seq.simplify_seq()
        opt_complexity = (
            determine_seq_complexity(opt_in_a_info_seq)
            + determine_seq_complexity(opt_in_b_info_seq)
            + determine_seq_complexity(opt_out_info_seq)
        )
        if opt_complexity < origin_complexity:
            new_a_op_seq = ReshapeTransposeOpSeq.create_by_info_seq(
                graph,
                in_a_layout_transform_op_seq.input_v,
                in_a_layout_transform_op_seq.v_extra_info,
                opt_in_a_info_seq,
            )
            new_b_op_seq = ReshapeTransposeOpSeq.create_by_info_seq(
                graph,
                in_b_layout_transform_op_seq.input_v,
                in_b_layout_transform_op_seq.v_extra_info,
                opt_in_b_info_seq,
            )
            if new_a_op_seq is None or new_b_op_seq is None:
                return False

            new_node = ir.Node(
                "",
                node.op_type,
                [new_a_op_seq.output_v, new_b_op_seq.output_v],
                num_outputs=1,
                name=graph.meta["extra_info"].get_unique_name_with_suffix(node.name, "/simplify"),
            )
            for k, v in node.attributes.items():
                new_node.attributes[k] = copy.deepcopy(v)

            assert new_a_op_seq.output_v.shape.numpy() == new_b_op_seq.output_v.shape.numpy()
            new_node.outputs[0].name = graph.meta["extra_info"].get_unique_name_with_suffix(
                node.outputs[0].name, "/simplify"
            )
            new_node.outputs[0].shape = ir.Shape(new_b_op_seq.output_v.shape.numpy())
            new_node.outputs[0].dtype = new_b_op_seq.output_v.dtype
            new_node.outputs[0].meta["extra_info"] = out_layout_transform_op_seq.v_extra_info.copy(
                ignore_safetensors=True
            )

            graph.insert_before(node, new_node)
            graph.meta["extra_info"].record_sharing_encodings(
                node.outputs[0].name, new_node.outputs[0].name, self.get_curr_pass_name()
            )
            new_out_op_seq = ReshapeTransposeOpSeq.create_by_info_seq(
                graph, new_node.outputs[0], out_layout_transform_op_seq.v_extra_info, opt_out_info_seq
            )

            assert out_layout_transform_op_seq.output_v.shape.numpy() == new_out_op_seq.output_v.shape.numpy()
            safe_replace_all_uses_with(graph, out_layout_transform_op_seq.output_v, new_out_op_seq.output_v)
            graph.meta["extra_info"].record_copy(
                out_layout_transform_op_seq.output_v.name,
                new_out_op_seq.output_v.name,
                self.get_curr_pass_name(),
            )

            logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)
            return True

        return False
