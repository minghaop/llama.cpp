# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for replacing
(Split->GroupSlice) -> (Slice->GroupSlice)
so that we don't need to re-implement Split-GroupSlice reordering,
just reuse Slice->GroupSlice reordering
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import is_reorderable_group_slice
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    convert_attr_to_py,
    get_value_numeric_shape,
    make_initializer,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class M2sReplaceSplitGroupslice2SliceGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a) --> c, c00,c01,c02,c10,c11,c12,c20,c21,c22...
        {
            b1,b2,b3 = Split(in_a)
            c00,c01,c02... = GroupSlice(b1)
            c10,c11,c12... = GroupSlice(b2)
            c20,c21,c22... = GroupSlice(b3)
        }
    Into:
        Subgraph(in_a) --> c, c00,c01,c02,c10,c11,c12,c20,c21,c22...
        {
            b1 = Slice(in_a)
            b2 = Slice(in_a)
            b3 = Slice(in_a)
            c00,c01,c02... = GroupSlice(b1)
            c10,c11,c12... = GroupSlice(b2)
            c20,c21,c22... = GroupSlice(b3)
        }
    Do not handle groupslice, it will be handled by reorder_slice_gslice pass
    Also, encodings are updated
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        if op_node is None:
            return False
        if op_node.op_type != "Split":
            return False

        check_static_shape_of_node_io(op_node)

        for split_param in op_node.outputs[0].meta["extra_info"].splittable:
            if split_param.axis == convert_attr_to_py(op_node.attributes["axis"], "as_int"):
                # M2sReplaceSplitGroupslice2GroupSliceGroupslice will handle this situation
                return False
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy
        assert op_node.name is not None  # check for mypy

        split_axis = convert_attr_to_py(op_node.attributes["axis"], "as_int")
        outputs_shapes = [get_value_numeric_shape(v) for v in op_node.outputs]
        split_sizes = [s[split_axis] for s in outputs_shapes]

        accum_size = 0
        for v_i, _ in enumerate(op_node.outputs):
            slice_node_name = graph.meta["extra_info"].get_unique_name_with_suffix(
                op_node.name, "/slice_" + str(v_i)
            )
            v_i_start = make_initializer(
                graph,
                graph.meta["extra_info"].get_unique_name_with_suffix(slice_node_name, ".start"),
                [accum_size],
            )
            v_i_end = make_initializer(
                graph,
                graph.meta["extra_info"].get_unique_name_with_suffix(slice_node_name, ".end"),
                [accum_size + split_sizes[v_i]],
            )
            v_i_axes = make_initializer(
                graph,
                graph.meta["extra_info"].get_unique_name_with_suffix(slice_node_name, ".axes"),
                [split_axis],
            )
            slice_node = ir.Node(
                "",
                op_type="Slice",
                inputs=[op_node.inputs[0], v_i_start, v_i_end, v_i_axes],
                name=slice_node_name,
            )
            slice_node.outputs[0].name = graph.meta["extra_info"].get_unique_name(op_node.outputs[v_i].name)
            self.mark_value_as_copy(graph, op_node.outputs[v_i], slice_node.outputs[0])

            accum_size += split_sizes[v_i]
            graph.insert_after(op_node, slice_node)
            safe_replace_all_uses_with(graph, op_node.outputs[v_i], slice_node.outputs[0])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), gslice_node.name)

        return True
