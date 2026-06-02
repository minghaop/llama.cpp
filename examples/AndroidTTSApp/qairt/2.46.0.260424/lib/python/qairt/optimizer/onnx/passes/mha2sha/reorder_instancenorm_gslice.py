# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(InstanceNormalization->GroupSlice) -> (GroupSlice->InstanceNormalization)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import convert_attr_to_py, has_static_shape_on_value
from qairt.optimizer.utils.logger import logger


class M2sReorderInstancenormGroupslice(M2sBasePass):
    """
    Reorder subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = InstanceNormalization(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            a0,a1,a2... = GroupSlice(in_a)

            b0 = InstanceNormalization(a0)
            b1 = InstanceNormalization(a1)
            b2 = InstanceNormalization(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = InstanceNormalization(in_a)
        }

    Also, encodings are updated
    """

    SUPPORTED_OP_TYPES = ["InstanceNormalization"]

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true
        op_type = op_node.op_type
        if op_type not in self.SUPPORTED_OP_TYPES:
            return False

        assert op_node.inputs[0] is not None  # check for mypy, definitely true
        if not has_static_shape_on_value(op_node.inputs[0]):
            return False

        if convert_attr_to_py(gslice_node.attributes["axis"], "as_int") not in [0, 1]:
            return False

        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        gslice_axis = convert_attr_to_py(gslice_node.attributes["axis"], "as_int")
        if gslice_axis == 0:
            # slice on batch
            a_groupslice_attrs = get_gslice_attrs(gslice_node)
            other_inputs_gslice_attrs = []
            for _ in op_node.inputs[1:]:
                # scale and bias
                other_inputs_gslice_attrs.append(
                    FullGroupSliceAttrs(
                        a_groupslice_attrs.num_outputs(),
                        a_groupslice_attrs.head_slice_ids[:],
                        a_groupslice_attrs.batch_slice_ids[:],
                    )
                )
            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                op_node,
                [gslice_node],
                inputs_gslice_attrs=[a_groupslice_attrs] + other_inputs_gslice_attrs,
            )
        elif gslice_axis == 1:
            # slice on channel
            a_groupslice_attrs = get_gslice_attrs(gslice_node)
            other_inputs_gslice_attrs = []
            for _ in op_node.inputs[1:]:
                # scale and bias
                groupslice_attrs = get_gslice_attrs(gslice_node)
                groupslice_attrs.axis = 0
                other_inputs_gslice_attrs.append(groupslice_attrs)

            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                op_node,
                [gslice_node],
                inputs_gslice_attrs=[a_groupslice_attrs] + other_inputs_gslice_attrs,
            )
        else:
            assert False, "something wrong"

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
