# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Gather->GroupSlice) -> (GroupSlice->Gather)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import (
    get_attribute_with_default,
    get_value_numeric_shape,
    has_static_shape_on_value,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderGatherGroupslice(M2sBasePass):
    """
    Reorder subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Gather(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            a0,a1,a2... = GroupSlice(in_a)

            b0 = Gather(a0)
            b1 = Gather(a1)
            b2 = Gather(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Gather(in_a)
        }

    Also, encodings are updated
    """

    SUPPORTED_OP_TYPES = ["Gather"]

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

        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        gather_axis = get_attribute_with_default(op_node, "axis", 0)
        assert op_node.inputs[0] is not None
        if gather_axis < 0:
            assert op_node.inputs[0].shape is not None  # check for mypy, definitely true
            gather_axis += op_node.inputs[0].shape.rank()

        out_gslice_attrs = get_gslice_attrs(gslice_node)

        indices = op_node.inputs[1]
        indices_rank = len(get_value_numeric_shape(indices))

        # by definition, the output shape should be
        # data_shape[0:gather_axis] + indices_shape + data_shape[gather_axis+1:]
        # output rank can be decomposed by these three parts:
        #  - part1: [0:gather_axis]
        #  - part2: [gather_axis:gather_axis+indices_rank]
        #  - part3: [gather_axis+indices_rank:]

        # note: indices can be a scalar, so indices_shape=[]

        if gather_axis <= out_gslice_attrs.axis and out_gslice_attrs.axis < gather_axis + indices_rank:
            # gather_axis is in part2
            # so we should slice the indices
            indices_gslice_attrs = get_gslice_attrs(gslice_node)
            indices_gslice_attrs.axis = out_gslice_attrs.axis - gather_axis

            in_gslice_attrs = FullGroupSliceAttrs(
                out_gslice_attrs.num_outputs(),
                list(out_gslice_attrs.head_slice_ids),
                list(out_gslice_attrs.batch_slice_ids),
            )
        else:
            # gather_axis is in part1 and part3
            # so we should slice the data

            in_gslice_attrs = get_gslice_attrs(gslice_node)
            if out_gslice_attrs.axis < gather_axis:
                # gather axis is in part1
                in_gslice_attrs.axis = out_gslice_attrs.axis
            else:
                # gather axis is in part3
                in_gslice_attrs.axis = out_gslice_attrs.axis - indices_rank + 1

            indices_gslice_attrs = FullGroupSliceAttrs(
                out_gslice_attrs.num_outputs(),
                list(out_gslice_attrs.head_slice_ids),
                list(out_gslice_attrs.batch_slice_ids),
            )

        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=[in_gslice_attrs, indices_gslice_attrs]
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
