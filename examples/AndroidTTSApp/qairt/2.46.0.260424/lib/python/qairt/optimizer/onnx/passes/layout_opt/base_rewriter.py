# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides basic rewriters for layout optimization ir modification
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol


class LayoutBasePredicatePass(BasePredicatePass):
    """Base rewriter for Layout optimization"""

    def match(self, graph: ir.Graph, node: ir.Node):
        raise NotImplementedError

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None):
        raise NotImplementedError
