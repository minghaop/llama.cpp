# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Utility helpers for transforming serial binary ops into balanced trees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import onnx_ir as ir

from qairt.optimizer.onnx.passes.mha2sha.utils import BroadcastHelper
from qairt.optimizer.onnx.utils.ir_extra_info import VariableExtraInfo
from qairt.optimizer.onnx.utils.utils import get_value_numeric_shape, have_static_shape_on_node_io


def have_same_or_no_encodings(values: Sequence[ir.Value | None]) -> bool:
    """
    Return True if all values share identical encodings, or none of them has encodings.
    """
    if not values:
        return True

    if any(v is None for v in values):
        return False

    assert values[0] is not None  # check for mypy
    ref_extra_info = values[0].meta["extra_info"]
    ref_has_encodings = ref_extra_info.defined_encodings()
    ref_named_encodings = ref_extra_info.named_encodings if ref_has_encodings else None

    for value in values[1:]:
        assert value is not None  # check for mypy
        curr_extra_info = value.meta["extra_info"]
        curr_has_encodings = curr_extra_info.defined_encodings()
        if curr_has_encodings != ref_has_encodings:
            return False
        if curr_has_encodings and curr_extra_info.named_encodings != ref_named_encodings:
            return False
    return True


@dataclass
class BalancedBinaryTree:
    """
    Result of building a balanced binary tree.

    Attributes:
        new_nodes: All newly created IR nodes that form the balanced tree.
        output: The final output value produced by the root node of the tree.
    """

    new_nodes: list[ir.Node]
    output: ir.Value


def build_balanced_binary_tree(
    graph: ir.Graph,
    leaves: list[ir.Value],
    op_type: str,
    domain: str,
    extra_info: VariableExtraInfo | None = None,
) -> BalancedBinaryTree | None:
    """
    Build a balanced binary-op tree from a flattened list of leaf values.

    This helper is intentionally generic so other passes can reuse it for
    "serial-to-parallel" rewrites where the target op is associative.

    For example, given leaves ``[a0, a1, a2, a3]`` and ``op_type=Add``,
    the function builds the following tree::

        a0   a1   a2   a3
          \\  /      \\  /
          Add        Add
            \\       /
              Add
               |
             new_out

    Args:
        graph: The graph IR that the new nodes will be inserted into.
        leaves: All leaf values of the binary-op tree.
        op_type: The op-type name of the binary operation (e.g. ``"Add"``).
        domain: The op domain name (e.g. ``""`` for the default ONNX domain).
        extra_info: Optional encoding override.
            - If *not* provided, the tree is built only when all values share
              the same encodings (returns ``None`` otherwise).
            - If provided, every value in the tree is overwritten with this
              ``extra_info`` (including its encodings).

    Behavior:
        1. **Validate preconditions** – ``leaves`` must be non-empty; all
           leaves must have compatible encodings (unless ``extra_info`` is
           supplied).
        2. **Build level-by-level (pairwise reduce)** – each adjacent pair is
           replaced by one new binary node; an odd tail element at any level is
           forwarded unchanged to the next level.
        3. **For each created node**:
           - The output name is uniquely allocated from root-name hints.
           - The output shape is inferred with :class:`BroadcastHelper`; if
             inference fails, the root output shape is used as a fallback.
           - The output dtype follows the root output dtype.
           - The output ``extra_info`` is copied from the root output (or from
             the provided ``extra_info``) to preserve encoding metadata
             expected by downstream passes.
        4. **Return** all newly created nodes and the final tree output value.

    Returns:
        A :class:`BalancedBinaryTree` when construction succeeds, or ``None``
        when the encoding compatibility check fails or ``leaves`` is empty.
    """
    if len(leaves) == 0:
        return None
    if len(leaves) == 1:
        return BalancedBinaryTree(new_nodes=[], output=leaves[0])

    root_name_hint = "parallelize_serial_ops"
    root_output = leaves[0]
    root_out_name_hint = root_output.name if root_output.name else "parallelize_serial_ops/out"
    root_out_shape = tuple(get_value_numeric_shape(root_output))
    root_out_dtype = root_output.dtype

    new_nodes: list[ir.Node] = []
    curr_level = list(leaves)
    binary_index = 0

    while len(curr_level) > 1:
        next_level: list[ir.Value] = []
        for i in range(0, len(curr_level), 2):
            if i + 1 >= len(curr_level):
                # Odd tail element: forward to the next level unchanged.
                next_level.append(curr_level[i])
                continue

            in_a = curr_level[i]
            in_b = curr_level[i + 1]
            if extra_info:
                # Overwrite every node's encoding with the caller-supplied extra_info.
                pair_ref_extra_info = extra_info
            else:
                # Only proceed when all three values share the same encodings.
                if not have_same_or_no_encodings([root_output, in_a, in_b]):
                    return None
                pair_ref_extra_info = in_a.meta["extra_info"]

            binary_index += 1
            new_node = ir.Node(
                domain,
                op_type,
                [in_a, in_b],
                num_outputs=1,
                name=graph.meta["extra_info"].get_unique_name_with_suffix(
                    root_name_hint, f"/parallel_{binary_index}"
                ),
            )
            new_output = new_node.outputs[0]
            new_output.name = graph.meta["extra_info"].get_unique_name_with_suffix(
                root_out_name_hint, f"/parallel_{binary_index}"
            )

            out_shape = BroadcastHelper.get_broadcast_shape(
                get_value_numeric_shape(in_a),
                get_value_numeric_shape(in_b),
            )
            new_output.shape = ir.Shape(out_shape)

            if root_out_dtype is not None:
                new_output.dtype = root_out_dtype
            new_output.meta["extra_info"] = pair_ref_extra_info.copy(ignore_safetensors=True)

            new_nodes.append(new_node)
            next_level.append(new_output)
        curr_level = next_level

    return BalancedBinaryTree(new_nodes=new_nodes, output=curr_level[0])


def _is_compatible_binary_op(
    curr_node: ir.Node,
    prev_node: ir.Node,
    require_same_encodings: bool = True,
) -> bool:
    """
    Return True if ``prev_node`` is a compatible binary op that can be merged
    with ``curr_node`` into the same connected binary elementwise op subgraph.

    ``prev_node`` is expected to be a direct predecessor of
    ``curr_node`` in the graph.  Compatibility requires:

    - Identical op-type and domain.
    - Exactly two non-``None`` inputs.
    - Static shapes on all inputs and outputs.
    - Matching encodings across both nodes (when ``require_same_encodings``
      is ``True``).
    """
    if prev_node.op_type != curr_node.op_type or prev_node.domain != curr_node.domain:
        return False
    if len(prev_node.inputs) != 2 or any(inp is None for inp in prev_node.inputs):
        return False
    if not have_static_shape_on_node_io(prev_node):
        return False
    if require_same_encodings:
        if not have_same_or_no_encodings(list(curr_node.outputs) + list(prev_node.outputs)):
            # note: we don't need prev_node.inputs has same encodings,
            # for example,
            #       a0 = Add0(x1, x2)
            #       a1 = Add1(a0, x3)
            #  Add0 and Add1 are compatible if a1,a0 share same encodings
            #  we don't need x1/x2/x3 have same encodings with a0/a1
            return False
    return True


def is_bottom_node_of_connected_bin_ops(
    candidate: ir.Node,
    require_same_encodings: bool = True,
) -> bool:
    """
    Return True if ``candidate`` is the root (bottom) node of a connected
    binary elementwise op subgraph, meaning none of its successors belongs to
    the same subgraph.

    Notes:
        - This function does **not** check the op type of ``candidate``; the
          caller is responsible for ensuring it is a binary elementwise op.
        - This function does **not** check predecessors; predecessor traversal
          is handled by :func:`collect_connected_binary_elementwise_ops`.
    """

    if len(candidate.outputs[0].uses()) > 1:
        # The output is consumed by multiple nodes.  We currently disallow this
        # for any intermediate node in the connected subgraph (even when all
        # consumers are part of the same subgraph), so the candidate is treated
        # as the root and its successors are not part of the subgraph.
        return True

    # Verify encoding compatibility for the candidate's own inputs and output.
    if require_same_encodings and not have_same_or_no_encodings(
        list(candidate.inputs) + list(candidate.outputs)
    ):
        return False

    # If any successor is a compatible binary op, the candidate is not the root.
    successors: list[ir.Node] = [use.node for use in candidate.outputs[0].uses()]
    for successor in successors:
        if _is_compatible_binary_op(successor, candidate, require_same_encodings=require_same_encodings):
            return False

    return True


def collect_connected_binelewise_ops(
    root: ir.Node, require_same_encodings: bool = True, skip_op_types: list[str] | None = None
) -> tuple[list[ir.Value], list[ir.Node], int]:
    """
    Collect all nodes and leaf values of a connected binary elementwise op
    subgraph rooted at ``root``.

    Algorithm:
        Bottom-up DFS starting from ``root``'s two inputs.  For each visited
        node confirmed as part of the subgraph, its predecessor is also
        included if it satisfies **all** of the following:

        - It is a compatible binary op with the current node (same op-type,
          domain, static shapes, and matching encodings).
        - Its output has exactly one consumer, so it is not shared externally.
          (Multi-consumer predecessors are treated as leaf values; this
          restriction may be relaxed in the future.)

    Args:
        root: The root (bottom) node of the connected binary elementwise op
            subgraph, as identified by
            :func:`is_bottom_node_of_connected_bin_ops`.
        require_same_encodings: When ``True``, nodes with mismatched encodings
            are treated as leaf boundaries rather than subgraph members.
        skip_op_types: Optional list of op types to skip during predecessor
            traversal (e.g. ``["Identity"]``).  When a predecessor matches a
            skipped op type, the traversal continues through its single input.

    Returns:
        A 3-tuple ``(leaf_values, connected_nodes, max_height)`` where:

        - ``leaf_values``: Deduplicated list of leaf :class:`ir.Value` objects
          in top-down order.
        - ``connected_nodes``: All :class:`ir.Node` objects that form the
          subgraph, in top-down order.
        - ``max_height``: The maximum depth of the subgraph tree.

        Returns ``None`` if ``root`` has no output.
    """
    root_out = root.outputs[0]
    if root_out is None:
        return [], [], 0

    visited: set[int] = set()
    connected_nodes: list[ir.Node] = []
    nodes_to_check: list[tuple[ir.Node, int]] = [(root, 1)]  # (node, depth)
    leave_values: list[ir.Value] = []
    max_height = 0

    while len(nodes_to_check) > 0:
        curr_node, height = nodes_to_check.pop(-1)
        # curr_node is confirmed as part of the connected binary elementwise op
        # subgraph; now check whether each of its predecessors also belongs.

        if id(curr_node) in visited:
            continue
        visited.add(id(curr_node))
        connected_nodes.append(curr_node)

        if height > max_height:
            max_height = height

        # Reverse the input order so that the stack-based DFS visits inputs in
        # the same left-to-right order as a recursive DFS would.
        for v in curr_node.inputs[::-1]:
            assert v is not None  # Binary op inputs must not be None.
            predecessor = v.producer()
            if skip_op_types is not None and predecessor is not None and predecessor.op_type in skip_op_types:
                # Transparently skip through this op (e.g. Identity) and
                # continue the traversal from its single input's producer.
                predecessor_input_v = predecessor.inputs[0]
                assert predecessor_input_v is not None
                predecessor = predecessor_input_v.producer()

            if predecessor is None:
                # v is a graph input or initializer — it is a leaf value.
                leave_values.append(v)
            elif len(v.uses()) > 1:
                # v is consumed by multiple nodes — treat it as a leaf value
                # to avoid pulling shared tensors into the subgraph.
                leave_values.append(v)
            elif _is_compatible_binary_op(curr_node, predecessor, require_same_encodings):
                # predecessor is part of the subgraph; recurse into it.
                nodes_to_check.append((predecessor, height + 1))
            else:
                # predecessor is not part of the subgraph — v is a leaf value.
                leave_values.append(v)

    # Deduplicate leaf values while preserving insertion order.
    leave_value_id_set = set()
    leave_values_unique: list[ir.Value] = []
    for v in leave_values:
        if id(v) in leave_value_id_set:
            continue
        leave_value_id_set.add(id(v))
        leave_values_unique.append(v)

    # Leaf values and nodes were collected in bottom-up DFS order (deepest
    # first); reverse both lists to restore a natural top-down order.
    leave_values_unique = leave_values_unique[::-1]
    connected_nodes = connected_nodes[::-1]
    return leave_values_unique, connected_nodes, max_height


def is_balanced_binary_tree(leaves: list, original_height: int) -> bool:
    """
    Return True if the binary tree described by ``leaves`` and
    ``original_height`` is already a balanced binary tree.

    A binary tree is considered balanced when its height does not exceed the
    theoretical minimum, i.e. ``ceil(log2(len(leaves)))``.

    Args:
        leaves: The leaf values of the tree (or chain).
        original_height: The current height (depth) of the tree or chain.

    Returns:
        ``True`` if the tree is already balanced; ``False`` otherwise.
    """
    min_height = math.ceil(math.log2(len(leaves)))
    if original_height <= min_height:
        return True
    return False
