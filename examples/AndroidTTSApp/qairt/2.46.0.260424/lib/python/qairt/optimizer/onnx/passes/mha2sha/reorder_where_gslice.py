# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Where->GroupSlice) -> (GroupSlice->Where)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    BroadcastHelper,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import check_static_shape_of_node_io, get_value_numeric_shape
from qairt.optimizer.utils.logger import logger


class M2sReorderWhereGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_cond, in_a, in_b) --> c, c0,c1,c2...
        {
            c = Where(in_cond, in_a, in_b)
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(in_cond, in_a, in_b) --> c, c0,c1,c2...
        {
            con1,con2,con3,... = GroupSlice(in_cond)
            a0,a1,a2,... = GroupSlice(in_a)
            b0,b1,b2,... = GroupSlice(in_b)

            c0 = Where(con1, in_a0, in_b0)
            c1 = Where(con2, in_a1, in_b1)
            c2 = Where(con3, in_a2, in_b2)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = Where(in_a, in_b)

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
        if op_node.op_type != "Where":
            return False

        check_static_shape_of_node_io(op_node)
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        output_shape = get_value_numeric_shape(op_node.outputs[0])

        # the broadcast is multidirectional among condition/a/b
        in_cond_bc_helper = BroadcastHelper(
            get_value_numeric_shape(op_node.inputs[0]), output_shape, output_shape
        )
        in_a_bc_helper = BroadcastHelper(
            get_value_numeric_shape(op_node.inputs[1]), output_shape, output_shape
        )
        in_b_bc_helper = BroadcastHelper(
            get_value_numeric_shape(op_node.inputs[2]), output_shape, output_shape
        )

        gslice_attrs = get_gslice_attrs(gslice_node)

        cond_groupslice_attrs = in_cond_bc_helper.get_input_group_attrs(0, gslice_attrs)
        a_groupslice_attrs = in_a_bc_helper.get_input_group_attrs(0, gslice_attrs)
        b_groupslice_attrs = in_b_bc_helper.get_input_group_attrs(0, gslice_attrs)

        _ = self.rewrite_based_on_gslice_attrs(
            graph,
            op_node,
            [gslice_node],
            inputs_gslice_attrs=[cond_groupslice_attrs, a_groupslice_attrs, b_groupslice_attrs],
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
