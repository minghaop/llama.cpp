# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Expand->GroupSlice) -> (GroupSlice->Expand)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    BroadcastHelper,
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.ir_extra_info import VariableExtraInfo
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    clone_node_attribute,
    get_value_numeric_shape,
    make_initializer,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderExpandGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a, in_b) --> c, c0,c1,c2...
        {
            c = Expand(in_a, in_b)
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(in_a, in_b) --> c, c0,c1,c2...
        {
            a0,a1,a2,... = GroupSlice(in_a)

            c0 = Expand(in_a0, in_b)
            c1 = Expand(in_a1, in_b)
            c2 = Expand(in_a2, in_b)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = Expand(in_a, in_b)

        }

    Broadcasting will be automatically updated
    Also, encodings are updated
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False

        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        if op_node.op_type != "Expand":
            return False

        check_static_shape_of_node_io(op_node)
        return True

    def create_mini_pattern(
        self,  # pylint: disable=R0913,R0917
        graph: ir.Graph,
        origin_op: ir.Node,
        mini_inputs: list[ir.Value | None],
        head_slice_id,
        batch_slice_id,
        slice_i,
        custom_kwargs: dict | None = None,
    ) -> list[ir.Value | None]:
        # Get current output shape and gslice attrs from the origin op
        out_shape = get_value_numeric_shape(origin_op.outputs[0])
        gslice_node = None
        for user, _ in origin_op.outputs[0].uses():
            if user.op_type == "GroupSlice":
                gslice_node = user
                break

        assert gslice_node is not None
        output_gslice_attrs = get_gslice_attrs(gslice_node)

        mini_out_shape_list = list(out_shape)[:]
        mini_out_shape_list[output_gslice_attrs.axis] = (
            output_gslice_attrs.ends[slice_i] - output_gslice_attrs.starts[slice_i]
        )
        mini_out_shape: tuple[int] = tuple(mini_out_shape_list)  # type: ignore

        if mini_out_shape == tuple(get_value_numeric_shape(mini_inputs[0])):
            # no need to expand
            return [mini_inputs[0]]

        if "_expand_constant_cache" not in graph.meta:
            graph.meta["_expand_constant_cache"] = {}
        constant_cache = graph.meta["_expand_constant_cache"]

        # in_b is the shape (integer), so it cannot have encodings
        # so we can safely create new b
        if mini_out_shape in constant_cache:
            in_b = constant_cache[mini_out_shape]
        else:
            assert origin_op.name is not None  # check for mypy, definitely true
            in_b = make_initializer(
                graph,
                graph.meta["extra_info"].get_unique_name_with_suffix(origin_op.name, ".sha.expand_constant"),
                mini_out_shape,
            )
            constant_cache[mini_out_shape] = in_b
            in_b.meta["extra_info"] = VariableExtraInfo()

        mini_inputs[1] = in_b

        # create mini op
        mini_op = ir.Node(
            domain="",
            op_type=origin_op.op_type,
            inputs=mini_inputs,
            num_outputs=len(origin_op.outputs),
            name=self.get_mini_op_name(graph, origin_op.name, head_slice_id, batch_slice_id),
            attributes=[clone_node_attribute(attr) for attr in origin_op.attributes.values()],
        )

        graph.insert_before(origin_op, mini_op)
        return list(mini_op.outputs)

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        in_shape = get_value_numeric_shape(op_node.inputs[0])
        out_shape = get_value_numeric_shape(op_node.outputs[0])

        output_gslice_attrs = get_gslice_attrs(gslice_node)

        bc_helper = BroadcastHelper(
            in_shape,
            out_shape,
            get_value_numeric_shape(op_node.outputs[0]),
        )
        in_a_groupslice_attrs = bc_helper.get_input_group_attrs(0, output_gslice_attrs)
        in_b_groupslice_attrs = FullGroupSliceAttrs(
            output_gslice_attrs.num_outputs(),
            [-1] * output_gslice_attrs.num_outputs(),
            [-1] * output_gslice_attrs.num_outputs(),
        )

        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=[in_a_groupslice_attrs, in_b_groupslice_attrs]
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
