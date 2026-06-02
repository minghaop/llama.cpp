# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to remove useless concat in the graph
Useless concat is the concat with only one input
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.utils.utils import safe_replace_all_uses_with
from qairt.optimizer.utils.logger import logger


class NullConcatRemovalRewriter(BasePredicatePass):
    """
    A graph rewriter pass that removes useless concat from the graph.
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type != "Concat":
            return False
        if len(node.inputs) == 1:
            return True
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert node.inputs[0] is not None  # check for mypy
        assert node.outputs[0] is not None  # check for mypy

        safe_replace_all_uses_with(graph, node.outputs[0], node.inputs[0])
        node.outputs[0].meta["extra_info"].merge(node.inputs[0].meta["extra_info"])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)

        return True
