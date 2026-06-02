# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Reshape->GroupSlice) -> (GroupSlice->Reshape)
"""

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.reorder_reshape_slice_utils import (
    get_reshape_slice_reordered_slice_attrs,
)
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    GroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    get_value_numeric_shape,
    make_initializer,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class SliceTreeNode:
    """
    Helper class that represents how multi-axes slices are organized
    """

    def __init__(
        self,  # pylint: disable=R0913,R0917
        parent,
        axis: int | None,
        start: int | None,
        end: int | None,
        original_indices: list[int],
    ):
        self.parent = parent
        self.axis = axis
        self.start = start
        self.end = end
        self.original_indices = original_indices
        self.children: dict[int, list[SliceTreeNode]] = {}  # key is the axis
        self.graph_value: ir.Value | None = None

    def add_child(self, node):
        """
        add child to current slice tree node
        """
        if node.axis not in self.children:
            self.children[node.axis] = [node]
        else:
            self.children[node.axis].append(node)

    def find_child(self, axis: int, start: int | None, end: int | None):
        """
        find child from current slice tree node
        """
        if axis not in self.children:
            return None
        for child in self.children[axis]:
            if child.start == start and child.end == end:
                return child
        return None

    def as_string_lines(self, indent=0):
        """
        pretty print as string lines
        """
        curr_str = (
            " " * indent
            + f"- SliceTreeNode(axis={self.axis}, start={self.start}, "
            + f"end={self.end}, original_indices={self.original_indices})"
        )
        children_strs = []
        for axis, axis_children in self.children.items():
            children_strs.append(" " * (indent) + f"  * axis={axis}")
            for child in axis_children:
                children_strs += child.as_string_lines(indent + 4)
        return [curr_str, *children_strs]

    def __str__(self):
        all_str = "\n".join(self.as_string_lines())
        return all_str

    def __repr__(self):
        return str(self)


class SliceTreeRoot(SliceTreeNode):
    """
    Helper class that represents how multi-axes slices are organized
    Root node of the SliceTree
    """

    def __init__(self):
        super().__init__(None, None, None, None, [])

    def construct_node(self, axes: list[int], starts: list[int], ends: list[int], original_id: int):
        """
        construct neccesary nodes of the given slices in multiple axes
        """
        parent: SliceTreeNode = self
        for axis, start, end in zip(axes, starts, ends, strict=True):
            child = parent.find_child(axis, start, end)
            if child is not None:
                child.original_indices.append(original_id)
                parent = child
                continue
            child = SliceTreeNode(parent, axis, start, end, [original_id])
            parent.add_child(child)
            parent = child
        return parent

    def bfs(self):
        """
        bfs iterate on the tree
        """
        queue: list[SliceTreeNode] = [self]
        while len(queue) > 0:
            curr = queue.pop(0)
            yield curr
            for _, axis_children in curr.children.items():
                for child in axis_children:
                    queue.append(child)


class M2sReorderReshapeGroupSlice(M2sBasePass):
    """
    Reorder subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Reshape(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            a0,a1,a2... = GroupSlice(in_a) # multiple slices on different axis may be required

            b0 = Reshape(a0)
            b1 = Reshape(a1)
            b2 = Reshape(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Reshape(in_a)
        }


    Also, encodings are updated
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true
        if op_node.op_type != "Reshape":
            return False
        check_static_shape_of_node_io(op_node)
        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:  # pylint: disable=R0914
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        # get the input and output of the reshape node
        v_a = op_node.inputs[0]
        v_b = op_node.outputs[0]

        # get the shape of the input and output
        shape_a = get_value_numeric_shape(v_a)
        shape_b = get_value_numeric_shape(v_b)

        # get the group slice attributes
        b_gslice_attrs = get_gslice_attrs(gslice_node)

        # iterate over the slices, build SliceTree
        slice_tree = SliceTreeRoot()
        slice_leaf_nodes = []
        for slice_i in range(b_gslice_attrs.num_outputs()):
            # for each slice, get the reordered slice attributes
            #  tmp = v_a
            #  for (a, s, e) in zip(reodered_slice_axes, reodered_slice_starts, reodered_slice_ends):
            #       tmp = Slice(tmp, a, s, e)
            #  where the final tmp is the Slice(v_b, axis, start, end)

            can_reorder, reodered_slice_axes, reodered_slice_starts, reodered_slice_ends = (
                get_reshape_slice_reordered_slice_attrs(
                    shape_a,
                    shape_b,
                    [b_gslice_attrs.axis],
                    [b_gslice_attrs.starts[slice_i]],
                    [b_gslice_attrs.ends[slice_i]],
                )
            )
            if not can_reorder:
                return False
            slice_leaf_nodes.append(
                slice_tree.construct_node(
                    reodered_slice_axes, reodered_slice_starts, reodered_slice_ends, slice_i
                )
            )

        # bfs on slice_tree
        slice_tree.graph_value = v_a
        for tree_node in slice_tree.bfs():
            if len(tree_node.children) == 0:
                # leaf node
                continue
            for axis, children in tree_node.children.items():
                group_slice_attrs = GroupSliceAttrs(axis)
                for child in children:
                    group_slice_attrs.starts.append(child.start)
                    group_slice_attrs.ends.append(child.end)
                    group_slice_attrs.head_slice_ids.append(
                        b_gslice_attrs.head_slice_ids[child.original_indices[0]]
                    )
                    group_slice_attrs.batch_slice_ids.append(
                        b_gslice_attrs.batch_slice_ids[child.original_indices[0]]
                    )

                curr_group_slice_node, outputs = self.create_simplified_gslice_node(
                    graph, tree_node.graph_value, group_slice_attrs
                )
                if curr_group_slice_node:
                    graph.insert_before(op_node, curr_group_slice_node)
                for child, sliced_v in zip(children, outputs, strict=True):
                    child.graph_value = sliced_v
        curr_v_list = [tree_node.graph_value for tree_node in slice_leaf_nodes]

        # reshape the final output
        assert len(curr_v_list) == b_gslice_attrs.num_outputs()
        new_sliced_outputs = []
        for i in range(b_gslice_attrs.num_outputs()):
            shape_v = make_initializer(
                graph,
                graph.meta["extra_info"].get_unique_name_with_suffix(op_node.name, "/shape"),
                np.array(get_value_numeric_shape(gslice_node.outputs[i]), dtype=np.int64),
            )
            b_slice_reshape = ir.Node("", "Reshape", inputs=[curr_v_list[i], shape_v])
            graph.insert_before(op_node, b_slice_reshape)

            self.mark_value_as_copy(graph, gslice_node.outputs[i], b_slice_reshape.outputs[0])
            safe_replace_all_uses_with(graph, gslice_node.outputs[i], b_slice_reshape.outputs[0])
            new_sliced_outputs.append(b_slice_reshape.outputs[0])

        self.try_replace_out_with_concat(graph, op_node, b_gslice_attrs, new_sliced_outputs)

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
