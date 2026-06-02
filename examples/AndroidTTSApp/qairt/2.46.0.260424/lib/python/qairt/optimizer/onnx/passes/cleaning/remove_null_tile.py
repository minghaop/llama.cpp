# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to remove useless Tile in the graph
Useless Tile is the Tile with output == input
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.utils.utils import (
    get_constant_np,
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class NullTileRemovalRewriter(BasePredicatePass):
    """
    A graph rewriter pass that removes useless Tile from the graph.
    Useless Tile is the Tile with output == input
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type != "Tile":
            return False
        if not have_static_shape_on_node_io(node):
            return False
        repeats = get_constant_np(node.inputs[1])
        if repeats is None:
            return False
        if not (repeats == 1).all():
            return False
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert node.inputs[0] is not None  # check for mypy
        assert node.outputs[0] is not None  # check for mypy

        # this tile can be removed
        safe_replace_all_uses_with(graph, node.outputs[0], node.inputs[0])
        node.outputs[0].meta["extra_info"].merge(node.inputs[0].meta["extra_info"])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)
        return True
