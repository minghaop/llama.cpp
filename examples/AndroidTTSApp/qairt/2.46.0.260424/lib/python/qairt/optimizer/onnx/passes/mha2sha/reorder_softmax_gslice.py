# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Softmax->GroupSlice) -> (GroupSlice->Softmax)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import get_gslice_attrs, is_reorderable_group_slice
from qairt.optimizer.onnx.utils.utils import (
    convert_attr_to_py,
    get_attribute_with_default,
    has_static_shape_on_value,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderSoftmaxGroupslice(M2sBasePass):
    """
    Reorder subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Softmax(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            a0,a1,a2... = GroupSlice(in_a)

            b0 = Softmax(a0)
            b1 = Softmax(a1)
            b2 = Softmax(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Softmax(in_a)
        }

    Also, encodings are updated
    """

    SUPPORTED_OP_TYPES = ["Softmax"]

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False

        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        if op_node.op_type not in self.SUPPORTED_OP_TYPES:
            return False

        assert op_node.inputs[0] is not None  # check for mypy, definitely true
        if not has_static_shape_on_value(op_node.inputs[0]):
            return False
        # check for mypy, definitely true
        assert op_node.inputs[0].shape is not None

        reduce_axis = get_attribute_with_default(op_node, "axis", -1)
        if reduce_axis < 0:
            reduce_axis += len(op_node.inputs[0].shape)

        if reduce_axis == convert_attr_to_py(gslice_node.attributes["axis"], "as_int"):
            return False

        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        # when reduce_axis != gslice_axis
        # input gslice attrs should be the same on output gslice
        a_groupslice_attrs = get_gslice_attrs(gslice_node)

        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=[a_groupslice_attrs]
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
