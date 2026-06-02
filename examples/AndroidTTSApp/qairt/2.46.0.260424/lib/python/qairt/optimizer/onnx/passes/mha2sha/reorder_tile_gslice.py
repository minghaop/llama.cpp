# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Tile->GroupSlice) -> (GroupSlice->Tile)
"""

import copy

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    BroadcastHelper,
    FullGroupSliceAttrs,
    GroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.ir_extra_info import VariableExtraInfo
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    get_constant_np,
    get_value_numeric_shape,
    make_initializer,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderTileGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a, in_b) --> c, c0,c1,c2...
        {
            c = Tile(in_a, in_b)
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(in_a, in_b) --> c, c0,c1,c2...
        {
            a0,a1,a2,... = GroupSlice(in_a)

            c0 = Tile(in_a0, in_b)
            c1 = Tile(in_a1, in_b)
            c2 = Tile(in_a2, in_b)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = Tile(in_a, in_b)

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

        if op_node.op_type != "Tile":
            return False

        if op_node.domain not in ["", "ai.onnx", "main"]:
            return False
        op_version = graph.opset_imports[op_node.domain]
        if op_version < 6:
            # Tile has different input definition for opset_version < 6
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
        assert custom_kwargs is not None
        if custom_kwargs["slice_on_tile_repeat_axis"]:
            mini_inputs = list(mini_inputs)
            assert mini_inputs[1] is not None
            new_repeats = get_constant_np(origin_op.inputs[1]).tolist()
            new_repeats[custom_kwargs["slice_axis"]] = 1
            new_repeats_v = make_initializer(
                graph,
                graph.meta["extra_info"].get_unique_name_with_suffix(mini_inputs[1].name, ".sha.repeats"),
                new_repeats,
            )
            # new_repeats are integers, so it cannot have encodings
            new_repeats_v.meta["extra_info"] = VariableExtraInfo()
            mini_inputs[1] = new_repeats_v

        return super().create_mini_pattern(
            graph, origin_op, mini_inputs, head_slice_id, batch_slice_id, slice_i
        )

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        in_shape = get_value_numeric_shape(op_node.inputs[0])
        out_shape = get_value_numeric_shape(op_node.outputs[0])

        output_gslice_attrs = get_gslice_attrs(gslice_node)

        repeats = get_constant_np(op_node.inputs[1])
        if repeats is None:
            return False

        gslice_axis = output_gslice_attrs.axis
        if repeats[output_gslice_attrs.axis] == 1:
            # gslice on non-repeat axis
            in_a_groupslice_attrs = get_gslice_attrs(gslice_node)

            in_b_groupslice_attrs = FullGroupSliceAttrs(
                output_gslice_attrs.num_outputs(),
                [-1] * output_gslice_attrs.num_outputs(),
                [-1] * output_gslice_attrs.num_outputs(),
            )

            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                op_node,
                [gslice_node],
                inputs_gslice_attrs=[in_a_groupslice_attrs, in_b_groupslice_attrs],
                mini_pattern_custom_kwargs={"slice_on_tile_repeat_axis": False, "slice_axis": gslice_axis},
            )
        else:
            # gslice on repeat axis
            in_a_groupslice_attrs = copy.deepcopy(output_gslice_attrs)
            in_b_groupslice_attrs = FullGroupSliceAttrs(
                output_gslice_attrs.num_outputs(),
                [-1] * output_gslice_attrs.num_outputs(),
                [-1] * output_gslice_attrs.num_outputs(),
            )

            for i in range(output_gslice_attrs.num_outputs()):
                original_start = output_gslice_attrs.starts[i]
                new_start = original_start % in_shape[gslice_axis]
                original_end = output_gslice_attrs.ends[i]
                new_end = (original_end - 1) % in_shape[gslice_axis] + 1

                if original_start // in_shape[gslice_axis] != (original_end - 1) // in_shape[gslice_axis]:
                    # not on same span
                    return False

                in_a_groupslice_attrs.starts[i] = new_start
                in_a_groupslice_attrs.ends[i] = new_end

            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                op_node,
                [gslice_node],
                inputs_gslice_attrs=[in_a_groupslice_attrs, in_b_groupslice_attrs],
                mini_pattern_custom_kwargs={"slice_on_tile_repeat_axis": True, "slice_axis": gslice_axis},
            )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
