# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to remove dead weight in the graph
"""

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass


class DeadWeightRemovalRewriter(BasePass):
    """
    A graph rewriter pass that removes dead weight from the graph.
    """

    def apply(self, ctx: GraphContext) -> int:
        """
        Removes dead weight from the graph.

        Returns the number of weights removed.
        """
        value_nameset = set()
        for node in ctx.graph_ir:
            for v in node.inputs:
                if v is not None:
                    value_nameset.add(v.name)
        values_to_remove = set(x for x in ctx.graph_ir.initializers) - value_nameset
        for v_name in values_to_remove:
            del ctx.graph_ir.initializers[v_name]

        return len(values_to_remove)
