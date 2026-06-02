# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(SpaceToDepth->GroupSlice) -> (GroupSlice->SpaceToDepth)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import get_gslice_attrs, is_reorderable_group_slice
from qairt.optimizer.onnx.utils.utils import check_static_shape_of_node_io, get_value_numeric_shape
from qairt.optimizer.utils.logger import logger


class M2sReorderSpaceToDepthGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a) --> c, c0,c1,c2...
        {
            c = SpaceToDepth(in_a)
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(in_a) --> c, c0,c1,c2...
        {
            a0,a1,a2,... = GroupSlice(in_a)

            c0 = SpaceToDepth(a0)
            c1 = SpaceToDepth(a1)
            c2 = SpaceToDepth(a2)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = SpaceToDepth(in_a)
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
        if op_node.op_type != "SpaceToDepth":
            return False

        check_static_shape_of_node_io(op_node)

        blocksize = op_node.attributes["blocksize"].as_int()
        gslice_attrs = get_gslice_attrs(gslice_node)

        output_shape = get_value_numeric_shape(op_node.outputs[0])
        if len(output_shape) == 4 and gslice_attrs.axis != 1:
            return False

        blocksize_square = blocksize * blocksize
        for start, end in zip(gslice_attrs.starts, gslice_attrs.ends):
            if start % blocksize_square != 0 or end % blocksize_square != 0:
                return False

        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        blocksize = op_node.attributes["blocksize"].as_int()
        blocksize_square = blocksize * blocksize

        gslice_attrs = get_gslice_attrs(gslice_node)
        for slice_i in range(gslice_attrs.num_outputs()):
            gslice_attrs.starts[slice_i] = gslice_attrs.starts[slice_i] // blocksize_square
            gslice_attrs.ends[slice_i] = gslice_attrs.ends[slice_i] // blocksize_square

        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=[gslice_attrs]
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
