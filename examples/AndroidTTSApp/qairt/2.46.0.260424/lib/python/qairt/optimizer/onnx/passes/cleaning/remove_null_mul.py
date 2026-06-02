# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass to remove useless mul in the graph
Useless mul is the mul with one input that is 1.0
"""

from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.utils.utils import (
    get_constant_np,
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    is_constant,
    safe_replace_all_uses_with,
    scan_least_common_ancestor,
)
from qairt.optimizer.utils.logger import logger


class NullMulRemovalRewriter(BasePredicatePass):
    """
    A graph rewriter pass that removes useless Mul from the graph.
    Useless mul is the mul with one input that is 1.0
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for NullMulRemovalRewriter pass"""

        cst_one_id: int
        """Index of the constant input that is 1.0"""

        input_id: int
        """Index of the non-constant input"""

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfo:
        if node.op_type != "Mul":
            return False
        if not have_static_shape_on_node_io(node):
            return False

        cst_one_id: int | None = None
        input_id: int | None = None
        if is_constant(node.inputs[0]):
            cst_one_id = 0
            input_id = 1
        elif is_constant(node.inputs[1]):
            cst_one_id = 1
            input_id = 0

        if cst_one_id is None or input_id is None:
            return False

        cst_v = node.inputs[cst_one_id]
        assert cst_v is not None  # check for mypy

        cst_v_np = get_constant_np(node.inputs[cst_one_id])
        if not (cst_v_np == 1.0).all():
            return False

        if cst_v.meta["extra_info"].is_updatable_weight():
            return False

        # skip this optimization if this node may be a part of layernorm/rmsnorm
        # removing the Mul node in this case can break the matching rules in Converter/Quantizer
        if self.is_potential_affine_mul_in_norm(node, input_id):
            return False

        input_shape = get_value_numeric_shape(node.inputs[input_id])
        output_shape = get_value_numeric_shape(node.outputs[0])

        if input_shape == output_shape:
            # non broadcast
            return NullMulRemovalRewriter.MatchInfo(cst_one_id=cst_one_id, input_id=input_id)

        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, NullMulRemovalRewriter.MatchInfo)  # For mypy
        input_id = match_info.input_id
        assert node.inputs[0] is not None  # check for mypy
        assert node.outputs[0] is not None  # check for mypy

        safe_replace_all_uses_with(graph, node.outputs[0], node.inputs[input_id])
        node.outputs[0].meta["extra_info"].merge(node.inputs[0].meta["extra_info"])

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), node.name)

        return True

    def is_potential_affine_mul_in_norm(self, node: ir.Node, non_cst_input_id: int) -> bool:
        """
        Checks if the given node could be part of a normalization (e.g. rmsnorm, layernorm).

        Typically the rmsnorm and layernorm has this kind of subgraph connection:
            OP-----------
            |           |
            |          Mul/Pow
            |           |
            |      ReduceMean
            |           |
            |         Sqrt
            |           |
            |          Add
            |           |
            Div--------/
            |
            Mul (the affine mul that this function concerns)
        The whole subgraph is not checked strictly, since there are so many varieties.
        Only the two way paths connection are checked.

        Note: This is not an exact check, if with any possibility it is part of a normalization,
        it will return True.

        Note: one of the inputs should be constant, this will not be checked by this function.

        Args:
            node: The node to check.
            non_cst_input_id: Index of the non-constant input

        Returns:
            True if the node is potentially part of layernorm, False otherwise.
        """
        non_cst_v = node.inputs[non_cst_input_id]
        if non_cst_v is None:  # check for mypy
            return False
        converge_node = non_cst_v.producer()
        if converge_node is None:
            return False
        if len(converge_node.inputs) != 2:
            return False
        lca_v = scan_least_common_ancestor(
            converge_node.inputs[0],
            converge_node.inputs[1],
            max_layers_to_traverse=10,  # RMSNorm/LayerNorm should be very small subgraph, so limit the traversal
        )
        if lca_v not in converge_node.inputs:
            # lca should also be a input of the converge node
            return False
        return True
