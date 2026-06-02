# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the base ir visitor class
"""

from abc import ABC

import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext


class BaseTreeVisitor(ABC):
    """
    Base class for stateless graph visitors

    This class provides the foundation for all graph traversal passes
    and shouldn't be used to modify the underlying IR topology
    """

    VISIT_SUBGRAPHS = False

    def apply(self, ctx: GraphContext):
        """
        Apply the visitor on the model context

        Args:
            ctx: The model context containing the IR graph and metadata
        """
        for n in ctx.graph_ir:
            self.visit_node(ctx.graph_ir, n)

        if self.VISIT_SUBGRAPHS:
            for func in ctx.model_ir.functions.values():
                for n in func.graph:
                    self.visit_node(func.graph, n)

    def visit_node(self, graph: ir.Graph, node: ir.Node):
        """
        Entry for visit each node
        if visit_node_{op_type} method is defined, then call this method
        otherwise call the default method visit_general_node

        Args:
            graph: ir.Graph instance
            node: ir node to be visited
        """
        if hasattr(self, f"visit_node_{node.op_type}"):
            getattr(self, f"visit_node_{node.op_type}")(graph, node)
        else:
            self.visit_general_node(graph, node)

    def visit_general_node(self, graph: ir.Graph, node: ir.Node):
        """
        The default method to visit the node

        Args:
            graph: ir.Graph instance
            node: ir node to be visited
        """
        pass
