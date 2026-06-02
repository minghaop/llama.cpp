# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to remove all dead node in the graph
"""

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass
from qairt.optimizer.onnx.utils.utils import is_used
from qairt.optimizer.utils.logger import logger


class DeadCodeRemovalRewriter(BasePass):
    """
    A graph rewriter pass that removes dead code from the graph.

    Dead code is defined as nodes that have no outputs or nodes
    that are not reachable from the graph's outputs.
    """

    def apply(self, ctx: GraphContext) -> int:
        """
        Removes dead code from the graph.
        Assume graph.nodes are topologically sorted

        Returns the number of nodes removed.
        """
        removed_count = 0
        for node in reversed(ctx.graph_ir):
            if all(not is_used(v) for v in node.outputs):
                ctx.graph_ir.remove(node, safe=True)

                logger.debug("Removed dead node '%s'", node.name)
                removed_count += 1
        return removed_count
