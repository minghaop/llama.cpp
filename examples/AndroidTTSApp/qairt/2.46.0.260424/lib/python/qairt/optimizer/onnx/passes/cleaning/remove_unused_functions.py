# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to remove dead functions in the graph
"""

from onnx_ir.traversal import RecursiveGraphIterator

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass


class DeadFunctionRemovalRewriter(BasePass):
    """
    A graph rewriter pass that removes dead function from the graph.
    """

    def apply(self, ctx: GraphContext) -> int:
        """
        Removes dead function from the graph.

        Returns the number of functions removed.
        """
        all_func_identifier = set(x.identifier() for x in ctx.model_ir.functions.values())
        used_func_identifier = set()

        # scan all graph nodes, note: a function graph may call other functions)
        to_scan_graph = [ctx.graph_ir]
        while len(to_scan_graph):
            graph = to_scan_graph.pop()
            for node in RecursiveGraphIterator(graph):
                identifier = node.op_identifier()
                if identifier in all_func_identifier and identifier not in used_func_identifier:
                    used_func_identifier.add(identifier)
                    to_scan_graph.append(ctx.model_ir.functions[identifier].graph)

        removed_func_num = 0

        for id in all_func_identifier - used_func_identifier:
            del ctx.model_ir.functions[id]
            removed_func_num += 1

        return removed_func_num
