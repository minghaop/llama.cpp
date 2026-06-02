# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Clip->GroupSlice) -> (GroupSlice->Clip)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import check_static_shape_of_node_io
from qairt.optimizer.utils.logger import logger


class M2sReorderClipGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a, in_max, in_min) --> c, c0,c1,c2...
        {
            c = Clip(in_a, in_max, in_min)
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(in_a, in_max, in_min) --> c, c0,c1,c2...
        {
            in_a1,in_a2,in_a3,... = GroupSlice(in_a)
            c0 = Clip(in_a1, in_max, in_min)
            c1 = Clip(in_a2, in_max, in_min)
            c2 = Clip(in_a3, in_max, in_min)
            # if possible
            c = concat(c0,c1,c2,...)
            # or c = Clip(in_a, in_max, in_min)
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
        if op_node.op_type != "Clip":
            return False

        check_static_shape_of_node_io(op_node)
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        input_groupslice_attrs = get_gslice_attrs(gslice_node)

        min_groupslice_attrs = FullGroupSliceAttrs(
            input_groupslice_attrs.num_outputs(),
            input_groupslice_attrs.head_slice_ids,
            input_groupslice_attrs.batch_slice_ids,
        )
        max_groupslice_attrs = FullGroupSliceAttrs(
            input_groupslice_attrs.num_outputs(),
            input_groupslice_attrs.head_slice_ids,
            input_groupslice_attrs.batch_slice_ids,
        )

        _ = self.rewrite_based_on_gslice_attrs(
            graph,
            op_node,
            [gslice_node],
            inputs_gslice_attrs=[input_groupslice_attrs, min_groupslice_attrs, max_groupslice_attrs],
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
