# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Pass to clear intermediate shapes of tensors
Preserves only the shapes of graph inputs and outputs
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass


class ClearIntermediateShapes(BasePredicatePass):
    """
    Clear shapes from all tensors except inputs/outputs

    This prepares the graph for shape inference by removing intermediate
    shape information that may be stale after updates
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        """Check if node has intermediate outputs with shapes"""
        for out in node.outputs:
            if out not in graph.outputs and out.shape is not None:
                return True
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        """Clear shapes from intermediate outputs"""
        modified = False
        for out in node.outputs:
            if out not in graph.outputs and out.shape is not None:
                out.shape = None
                modified = True
        return modified
