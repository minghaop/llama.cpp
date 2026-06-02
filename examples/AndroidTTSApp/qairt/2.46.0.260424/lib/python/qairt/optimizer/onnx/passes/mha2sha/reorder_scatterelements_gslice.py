# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(ScatterElements->GroupSlice) -> (GroupSlice->ScatterElements)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import get_gslice_attrs, is_reorderable_group_slice
from qairt.optimizer.onnx.utils.utils import check_static_shape_of_node_io, convert_attr_to_py
from qairt.optimizer.utils.logger import logger


class M2sReorderScatterElementsGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(data, indices, updates) --> c, c0,c1,c2...
        {
            c = ScatterElements(data, indices, updates)
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(data, indices, updates) --> c, c0,c1,c2...
        {
            data0,data1,data2,... = GroupSlice(data)
            indices0,indices1,indices2,... = GroupSlice(indices)
            updates0,updates1,updates2,... = GroupSlice(updates)

            c0 = ScatterElements(data0, indices0, updates0)
            c1 = ScatterElements(data1, indices1, updates1)
            c2 = ScatterElements(data2, indices2, updates2)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = ScatterElements(data, indices, updates)

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
        if op_node.op_type != "ScatterElements":
            return False

        check_static_shape_of_node_io(op_node)
        assert op_node.inputs[0] is not None  # check for mypy, definitely true
        # check for mypy, definitely true
        assert op_node.inputs[0].shape is not None

        if "axis" in op_node.attributes:
            scatter_axis = convert_attr_to_py(op_node.attributes["axis"], "as_int")
        else:
            scatter_axis = 0
        if scatter_axis < 0:
            scatter_axis += len(op_node.inputs[0].shape)
        gslice_attrs = get_gslice_attrs(gslice_node)

        if gslice_attrs.axis == scatter_axis:
            return False
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        # according to the definition of ScatterElements
        # data/indices/updates should have same rank
        # indices/updates should have same shape

        # for gslice_axis != scatter_axis (already checked),
        # to reorder gslice, we should gslice all the inputs with the same attributes
        data_groupslice_attrs = get_gslice_attrs(gslice_node)
        indices_groupslice_attrs = get_gslice_attrs(gslice_node)
        updates_groupslice_attrs = get_gslice_attrs(gslice_node)

        _ = self.rewrite_based_on_gslice_attrs(
            graph,
            op_node,
            [gslice_node],
            inputs_gslice_attrs=[data_groupslice_attrs, indices_groupslice_attrs, updates_groupslice_attrs],
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
