# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for replacing
(Split->GroupSlice) -> (GroupSlice->GroupSlice)
"""

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import GroupSliceAttrs, is_reorderable_group_slice
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    convert_attr_to_py,
    get_value_numeric_shape,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class M2sReplaceSplitGroupslice2GroupsliceGroupslice(M2sBasePass):
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
            b1,b2,b3 = GroupSlice(in_a)
            c00,c01,c02... = GroupSlice(b1)
            c10,c11,c12... = GroupSlice(b2)
            c20,c21,c22... = GroupSlice(b3)
        }
    Also, encodings are updated
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        if op_node is None:
            return False
        if op_node.op_type != "Split":
            return False
        if len(op_node.inputs) < 1 or op_node.inputs[0] is None:
            return False

        check_static_shape_of_node_io(op_node)

        for split_param in op_node.inputs[0].meta["extra_info"].splittable:
            if split_param.axis == convert_attr_to_py(op_node.attributes["axis"], "as_int"):
                return True
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy

        split_axis = op_node.attributes["axis"].as_int()
        if split_axis < 0:
            split_axis += len(get_value_numeric_shape(op_node.inputs[0]))
        outputs_shapes = [get_value_numeric_shape(v) for v in op_node.outputs]
        split_sizes = [s[split_axis] for s in outputs_shapes]

        accum_sizes = np.cumsum([0] + split_sizes).tolist()
        node_namehint = (op_node.name if op_node.name else "") + "/as_gslice"
        assert op_node.inputs[0] is not None
        new_gslice_node = self._create_groupslice_node(
            graph,
            op_node.inputs[0],
            GroupSliceAttrs(
                axis=split_axis,
                starts=accum_sizes[0:-1],
                ends=accum_sizes[1:],
                head_slice_ids=[-1] * len(split_sizes),
                batch_slice_ids=[-1] * len(split_sizes),
            ),
            node_namehint=node_namehint,
        )
        graph.insert_after(op_node, new_gslice_node)
        for v_i in range(len(op_node.outputs)):
            new_gslice_node.outputs[v_i].name = graph.meta["extra_info"].get_unique_name(
                op_node.outputs[v_i].name
            )
            self.mark_value_as_copy(graph, op_node.outputs[v_i], new_gslice_node.outputs[v_i])
            safe_replace_all_uses_with(graph, op_node.outputs[v_i], new_gslice_node.outputs[v_i])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), new_gslice_node.name)
        return True
