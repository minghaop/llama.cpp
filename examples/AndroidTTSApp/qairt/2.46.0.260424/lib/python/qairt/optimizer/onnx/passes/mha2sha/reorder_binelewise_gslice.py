# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(binary_elewise_op->GroupSlice) -> (GroupSlice->binary_elewise_op)
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


class M2sReorderBinElewiseGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a, in_b) --> c, c0,c1,c2...
        {
            c = BinElewiseOp(in_a, in_b)
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(in_a, in_b) --> c, c0,c1,c2...
        {
            a0,a1,a2,... = Slice(in_a)
            b0,b1,b2,... = Slice(in_b)

            c0 = BinElewiseOp(in_a0, in_b0)
            c1 = BinElewiseOp(in_a1, in_b1)
            c2 = BinElewiseOp(in_a2, in_b2)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = BinElewiseOp(in_a, in_b)

        }

    Broadcasting will be automatically updated
    Also, encodings are updated
    """

    SUPPORTED_OP_TYPES = ["Add", "Sub", "Mul", "Div", "Equal", "Pow", "Max", "Min"]

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true
        if op_node.op_type not in self.SUPPORTED_OP_TYPES:
            return False

        check_static_shape_of_node_io(op_node)
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        in_a, in_b = op_node.inputs

        bc_helper = BroadcastHelper(
            get_value_numeric_shape(in_a),
            get_value_numeric_shape(in_b),
            get_value_numeric_shape(op_node.outputs[0]),
        )
        gslice_attrs = get_gslice_attrs(gslice_node)
        a_groupslice_attrs = bc_helper.get_input_group_attrs(0, gslice_attrs)
        b_groupslice_attrs = bc_helper.get_input_group_attrs(1, gslice_attrs)

        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=[a_groupslice_attrs, b_groupslice_attrs]
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
