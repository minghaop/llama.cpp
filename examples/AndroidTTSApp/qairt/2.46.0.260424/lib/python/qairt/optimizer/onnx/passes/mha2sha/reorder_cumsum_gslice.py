# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(CumSum->GroupSlice) -> (GroupSlice->CumSum)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import get_constant_np, have_static_shape_on_node_io
from qairt.optimizer.utils.logger import logger


class M2sReorderCumSumGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a_1, in_a_2, in_a_3, ...) --> b,b0,b1,b2
        {
            b = CumSum(in_a_1, in_a_2, in_a_3, ...)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a_1, in_a_2, in_a_3, ...) --> b,b0,b1,b2
        {

            in_a_1.0, in_a_1.1, in_a_1.1 ... = GroupSlice(in_a_1)
            in_a_2.0, in_a_2.1, in_a_2.2 ... = GroupSlice(in_a_2)
            in_a_3.0, in_a_3.1, in_a_3.2 ... = GroupSlice(in_a_3)
            ...

            b0 = CumSum(in_a_1.0, in_a_2.0, in_a_3.0, ...)
            b1 = CumSum(in_a_1.1, in_a_2.1, in_a_3.1, ...)
            b2 = CumSum(in_a_1.2, in_a_2.2, in_a_3.2, ...)
            ...

            # if possible
            b = CumSum(b0,b1,b2,...)
            # or b = CumSum(in_a_1, in_a_2, in_a_3, ...)

        }

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

        if op_node.op_type != "CumSum":
            return False

        if not have_static_shape_on_node_io(op_node):
            return False
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true
        cumsum_axis = get_constant_np(op_node.inputs[1])
        if cumsum_axis is None:
            return False

        output_gslice_attrs = get_gslice_attrs(gslice_node)
        if output_gslice_attrs.axis == cumsum_axis:
            return False

        in_a_gslice_attrs = get_gslice_attrs(gslice_node)
        in_axis_gslice_attrs = FullGroupSliceAttrs(
            output_gslice_attrs.num_outputs(),
            [-1] * output_gslice_attrs.num_outputs(),
            [-1] * output_gslice_attrs.num_outputs(),
        )
        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=[in_a_gslice_attrs, in_axis_gslice_attrs]
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
