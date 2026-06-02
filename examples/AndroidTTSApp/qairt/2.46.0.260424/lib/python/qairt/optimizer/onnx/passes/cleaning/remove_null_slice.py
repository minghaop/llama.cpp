# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to remove useless Slice in the graph
Useless Slice is the Slice with output == input
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.utils.utils import (
    get_slice_static_params,
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class NullSliceRemovalRewriter(BasePredicatePass):
    """
    A graph rewriter pass that removes useless Slice from the graph.
    Useless Slice is the Slice with output == input
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type != "Slice":
            return False
        if not have_static_shape_on_node_io(node):
            return False
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert node.inputs[0] is not None  # check for mypy
        assert node.outputs[0] is not None  # check for mypy

        slice_param = get_slice_static_params(node)
        if slice_param is None:
            return False

        input_shape = get_value_numeric_shape(node.inputs[0])

        for axis, start, end, step in zip(
            slice_param["axes"], slice_param["starts"], slice_param["ends"], slice_param["steps"]
        ):
            if not (end == input_shape[axis] and start == 0 and step == 1):
                return False

        # this slice can be removed
        safe_replace_all_uses_with(graph, node.outputs[0], node.inputs[0])
        node.outputs[0].meta["extra_info"].merge(node.inputs[0].meta["extra_info"])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)
        return True
