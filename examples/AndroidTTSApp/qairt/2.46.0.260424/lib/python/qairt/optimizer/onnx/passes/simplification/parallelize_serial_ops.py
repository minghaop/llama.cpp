# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Parallelize serial associative binary ops by re-associating them into balanced trees.
"""

from __future__ import annotations

from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol, PassConfig
from qairt.optimizer.onnx.passes.mha2sha.utils import BroadcastHelper
from qairt.optimizer.onnx.utils.binop_binary_tree_utils import (
    build_balanced_binary_tree,
    collect_connected_binelewise_ops,
    have_same_or_no_encodings,
    is_balanced_binary_tree,
    is_bottom_node_of_connected_bin_ops,
)
from qairt.optimizer.onnx.utils.utils import (
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class ParallelizeSerialOpsRewriter(BasePredicatePass):
    """
    Transform subgraph:
        Subgraph(x0, x1, x2, x3, ...) --> out
        {
            t0 = op(x0, x1)
            t1 = op(t0, x2)
            out = op(t1, x3)
            ...
        }
    Into:
        Subgraph(x0, x1, x2, x3, ...) --> out
        {
            a0 = op(x0, x1)
            a1 = op(x2, x3)
            ...
            out = op(a0, a1, ...)
        }

    Constraints:
        only associative binary ops are supported;
        only serial binary nodes with same encodings (or no encodings) across
        each node's inputs/output can be transformed.
    """

    ASSOCIATIVE_OPS = {
        "Add",
        "Mul",
        "Max",
        "Min",
        "And",
        "Or",
        "Xor",
    }

    @dataclass
    class Config(PassConfig):
        """Configuration for ParallelizeSerialOpsRewriter pass."""

        op_types: tuple[str, ...] = (
            "Add",
            "Mul",
            "Max",
            "Min",
            "And",
            "Or",
            "Xor",
        )
        """
        Binary op types to rewrite.

        Must be a subset of `ASSOCIATIVE_OPS`.
        """

        # Rewriting only starts when the matched binary tree has at least this many
        # binary-op nodes. A default of 3 avoids tiny transformations while still
        # capturing the first non-balanced serial pattern.
        op_num_threshold: int = 3

    def __init__(self, config: Config | None = None):
        super().__init__()
        if config is None:
            config = ParallelizeSerialOpsRewriter.Config()

        invalid_ops = set(config.op_types) - self.ASSOCIATIVE_OPS
        if invalid_ops:
            invalid_ops_str = ", ".join(sorted(invalid_ops))
            raise ValueError(f"Unsupported op types for parallelization: {invalid_ops_str}")
        if config.op_num_threshold < 1:
            raise ValueError("`op_num_threshold` must be >= 1")

        self.config: ParallelizeSerialOpsRewriter.Config = config
        self.supported_ops = set(self.config.op_types)

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        root: ir.Node
        target_op_type: str
        target_domain: str
        leaf_values: list[ir.Value]

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfoProtocol:
        if node.op_type not in self.supported_ops:
            return False
        if node.domain not in ("", "ai.onnx"):
            return False
        if len(node.inputs) != 2 or any(inp is None for inp in node.inputs):
            return False
        if not have_static_shape_on_node_io(node):
            return False
        if not is_bottom_node_of_connected_bin_ops(node, require_same_encodings=True):
            return False

        leave_values, tree_nodes, max_height = collect_connected_binelewise_ops(
            node, require_same_encodings=True
        )

        if len(tree_nodes) < self.config.op_num_threshold:
            return False
        if is_balanced_binary_tree(leave_values, max_height):
            return False

        match_info = ParallelizeSerialOpsRewriter.MatchInfo(
            root=node,
            target_op_type=node.op_type,
            target_domain=node.domain,
            leaf_values=leave_values,
        )
        return match_info

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, ParallelizeSerialOpsRewriter.MatchInfo)

        root = match_info.root
        root_out = root.outputs[0]
        assert root_out is not None

        tree = build_balanced_binary_tree(
            graph=graph,
            leaves=match_info.leaf_values,
            op_type=match_info.target_op_type,
            domain=match_info.target_domain,
        )
        if tree is None:
            return False

        new_output = tree.output
        if new_output is root_out:
            return False

        new_output.shape = root_out.shape
        if root_out.dtype is not None:
            new_output.dtype = root_out.dtype

        graph.insert_before(root, tree.new_nodes)

        if root_out.name is not None:
            for new_node in tree.new_nodes:
                traced_output = new_node.outputs[0]
                graph.meta["extra_info"].record_sharing_encodings(
                    root_out.name, traced_output.name, self.get_curr_pass_name()
                )
        safe_replace_all_uses_with(graph, root_out, new_output)

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), root.name)
        return True
