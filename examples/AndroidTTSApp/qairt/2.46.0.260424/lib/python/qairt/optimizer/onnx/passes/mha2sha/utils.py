# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the helper functions of mha2sha passes
"""

import onnx_ir as ir

from qairt.optimizer.onnx.utils.utils import (
    convert_attr_to_py,
    get_value_numeric_shape,
)
from qairt.optimizer.utils.logger import logger


class BaseGroupSliceAttrs:
    """
    Base class for group slice attributes
    """

    def __init__(self, head_slice_ids: list[int] | None = None, batch_slice_ids: list[int] | None = None):
        if head_slice_ids is None:
            head_slice_ids = []
        if batch_slice_ids is None:
            batch_slice_ids = []
        self.head_slice_ids: list[int] = head_slice_ids
        self.batch_slice_ids: list[int] = batch_slice_ids

    def is_each_output_full(self, in_v: ir.Value):
        """
        Check if every slice output is same as in_v
        """
        raise NotImplementedError

    def num_outputs(self):
        """
        Get number of the outputs
        """
        raise NotImplementedError

    def complete(self, dim_of_axis):
        """
        Check if the GroupSlice is complete
        complete is defined as that the concatenation result of all outputs
        is the same as the input
        """
        raise NotImplementedError


class GroupSliceAttrs(BaseGroupSliceAttrs):
    """
    Normal Group slice attributes
    """

    def __init__(
        self,  # pylint: disable=R0913,R0917
        axis,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        head_slice_ids: list[int] | None = None,
        batch_slice_ids: list[int] | None = None,
    ):
        super().__init__(head_slice_ids, batch_slice_ids)
        self.axis: int = axis
        self.starts: list[int] = starts if starts is not None else []
        self.ends: list[int] = ends if ends is not None else []

    def is_each_output_full(self, in_v: ir.Value):
        if any(x != 0 for x in self.starts):
            return False
        in_v_shape = get_value_numeric_shape(in_v)
        dim = in_v_shape[self.axis]
        if any(x != dim for x in self.ends):
            return False
        return True

    def num_outputs(self):
        return len(self.starts)

    def complete(self, dim_of_axis):
        if len(self.starts) == 0:
            return False
        if self.starts[0] != 0:
            return False
        for i in range(1, len(self.starts)):
            if self.ends[i - 1] != self.starts[i]:
                return False
        if self.ends[-1] != dim_of_axis:
            return False
        return True


class FullGroupSliceAttrs(BaseGroupSliceAttrs):
    """
    Special Group slice attributes to represent a full slice (so same as Identity)
    """

    def __init__(self, output_num, head_slice_ids=None, batch_slice_ids=None):
        super().__init__(head_slice_ids, batch_slice_ids)
        self.output_num = output_num

    def is_each_output_full(self, in_v: ir.Value):
        return True

    def num_outputs(self):
        return self.output_num

    def complete(self, dim_of_axis):
        return True


def get_gslice_attrs(node: ir.Node):
    """
    Get GroupSliceAttrs from the GroupSlice node attributes
    the GroupSliceAttrs is easier to access rather than the node attributes
    """
    axis = convert_attr_to_py(node.attributes["axis"], "as_int")
    attrs = GroupSliceAttrs(axis=axis)
    attrs.starts = list(convert_attr_to_py(node.attributes["starts"], "as_ints"))
    attrs.ends = list(convert_attr_to_py(node.attributes["ends"], "as_ints"))
    attrs.batch_slice_ids = list(convert_attr_to_py(node.attributes["batch_slice_ids"], "as_ints"))
    attrs.head_slice_ids = list(convert_attr_to_py(node.attributes["head_slice_ids"], "as_ints"))
    return attrs


def get_slice_out_name(graph: ir.Graph, src_name: str | None, batch_slice_id, head_slice_id):
    """
    Get meaningful sliced output name

    Args:
        graph: ir graph
        src_name: original name (group slice input)
        batch_slice_id: batch_id of the current slice
        head_slice_id: head id of the current slice
    Returns:
        unique name of the current slice
    """
    if src_name is None:
        src_name = "tmp"
    out_name = src_name
    if batch_slice_id >= 0:
        out_name = out_name + "/b" + str(batch_slice_id)
    if head_slice_id >= 0:
        out_name = out_name + "/head" + str(head_slice_id)
    out_name = graph.meta["extra_info"].get_unique_name(out_name)

    return out_name


def simplify_group_slice_attrs(gslice_attrs: GroupSliceAttrs):
    """
    Simplify GroupSliceAttrs by removing duplicates and sorting slices.

    Args:
        gslice_attrs (GroupSliceAttrs): The original GroupSliceAttrs.

    Returns:
        GroupSliceAttrs: The simplified GroupSliceAttrs.
        origin2simplified_idx_map (list): A list to store the mapping from original index to simplified index.
    """
    # Create a set to store unique slices
    slices_set = set((start, end) for start, end in zip(gslice_attrs.starts, gslice_attrs.ends))

    # remove None slice (start == end)
    slices_set = set((start, end) for (start, end) in slices_set if start != end)

    # Convert the set to a list and sort it by starts
    slices_vec = sorted(list(slices_set), key=lambda x: x[0])

    # Create a dictionary to store the index of each slice
    slices_idx_map = {slice: i for i, slice in enumerate(slices_vec)}

    # Create a list to store the mapping from original index to simplified index
    origin2simplified_idx_map = {
        i: slices_idx_map[(start, end)]
        for i, (start, end) in enumerate(zip(gslice_attrs.starts, gslice_attrs.ends))
        if start != end
    }

    # Create lists to store the simplified starts, ends, batch_slice_ids, and head_slice_ids
    simplified_starts = []
    simplified_ends = []
    simplified_batch_slice_ids = []
    simplified_head_slice_ids = []

    # Populate the simplified lists
    for i, (start, end) in enumerate(slices_vec):
        origin_idx = [
            j
            for j, (s, e) in enumerate(zip(gslice_attrs.starts, gslice_attrs.ends))
            if (s, e) == (start, end)
        ][0]
        simplified_starts.append(gslice_attrs.starts[origin_idx])
        simplified_ends.append(gslice_attrs.ends[origin_idx])
        simplified_batch_slice_ids.append(
            gslice_attrs.batch_slice_ids[origin_idx] if gslice_attrs.batch_slice_ids else -1
        )
        simplified_head_slice_ids.append(
            gslice_attrs.head_slice_ids[origin_idx] if gslice_attrs.head_slice_ids else -1
        )

    # Create a new GroupSliceAttrs object with the simplified attributes
    simplified_gslice_attrs = GroupSliceAttrs(
        axis=gslice_attrs.axis,
        starts=simplified_starts,
        ends=simplified_ends,
        batch_slice_ids=simplified_batch_slice_ids,
        head_slice_ids=simplified_head_slice_ids,
    )

    return simplified_gslice_attrs, origin2simplified_idx_map


def create_concat_node(
    graph: ir.Graph, inputs: list[ir.Value], axis: int, node_namehint: str, output_namehint: str
):
    """
    Create a Concat node in the graph.

    Args:
        graph (ir.Graph): The graph to create the node in.
        inputs (list[ir.Value]): The input values to the Concat node.
        axis (int): The axis to concatenate along.
        node_namehint (str): A hint for the name of the node.
        output_namehint (str): A hint for the name of the output.

    Returns:
        ir.Node: The created Concat node.
    """
    # Create a Concat node with the given inputs and axis
    concat_out_name = graph.meta["extra_info"].get_unique_name(output_namehint)
    concat_node = ir.Node(
        domain="", op_type="Concat", inputs=inputs, num_outputs=1, outputs=[ir.Value(name=concat_out_name)]
    )
    concat_node.attributes["axis"] = ir.AttrInt64("axis", axis)
    concat_node.name = graph.meta["extra_info"].get_unique_name(node_namehint)

    # Shape infer
    if len(inputs) > 0:
        can_infer_shape = all(input.shape and input.shape.is_static() for input in inputs)
        if can_infer_shape:
            assert axis >= 0, "Axis must be non-negative"
            first_elem_shape = get_value_numeric_shape(inputs[0])
            result_shape = list(first_elem_shape)
            for i in range(1, len(inputs)):
                elem_shape = get_value_numeric_shape(inputs[i])
                result_shape[axis] += elem_shape[axis]
            concat_node.outputs[0].shape = ir.Shape(result_shape)
        else:
            logger.warning("cannot infer shape for '%s'", concat_node.name)

    if inputs[0].dtype is not None:
        concat_node.outputs[0].dtype = inputs[0].dtype
    return concat_node


def is_reorderable_group_slice(node: ir.Node):
    """
    Helper function to determinate whether a group slice is reorderable
    Every groupslice reordering should call this function

    Args:
        node: candidate group slice node
    Returns:
        whether the node is a reorderable group slice
    """
    if node.op_type != "GroupSlice":
        return False
    assert node.inputs[0] is not None  # check for mypy
    if node.inputs[0].producer() is None:
        # no op can be reordered
        return False
    return True


class BroadcastHelper:
    """
    Helper class to determinate the broadcast behavior
    Only support multidirectional broadcast in binary op.

    For inputs more than 2, we should use two BroadcastHelper to determinate the broadcast behavior,
    for example, check reorder_where_gslice.py

    """

    class BroadcastError(ValueError):
        pass

    @classmethod
    def get_broadcast_shape(cls, *shapes):
        shapes_list = list(shapes)
        output_shape = None
        while shapes_list:
            s1 = output_shape if output_shape else shapes_list.pop()
            s2 = shapes_list.pop()
            try:
                output_shape = cls(s1, s2).output_shape_
            except cls.BroadcastError:
                return None
        return output_shape

    def __init__(self, a_shape, b_shape, output_shape=None, ignore_last_dim_num=0):
        self.a_shape_ = a_shape
        self.b_shape_ = b_shape
        self.output_shape_ = list(output_shape) if output_shape else []
        self.broadcast_on_a_ = []
        self.broadcast_on_b_ = []
        self.ignore_last_dim_num_ = ignore_last_dim_num
        self.init()

    def init(self):
        """
        internall called init function
        """
        rank_a = len(self.a_shape_)
        rank_b = len(self.b_shape_)
        rank_out = max(rank_a, rank_b)

        if len(self.output_shape_) == 0:
            self.output_shape_ = [0] * rank_out

        self.broadcast_on_a_ = [False] * rank_out
        self.broadcast_on_b_ = [False] * rank_out

        for axis_i_from_right in range(self.ignore_last_dim_num_, rank_out):
            axis_i = rank_out - 1 - axis_i_from_right
            can_broadcast, broadcast_out_dim, curr_axis_broadcast_on_a, curr_axis_broadcast_on_b = (
                self.get_broadcast_dim(self.a_shape_, self.b_shape_, axis_i_from_right)
            )
            self.output_shape_[axis_i] = broadcast_out_dim
            self.broadcast_on_a_[axis_i] = curr_axis_broadcast_on_a
            self.broadcast_on_b_[axis_i] = curr_axis_broadcast_on_b
            if not can_broadcast:
                raise self.BroadcastError(
                    f"Cannot broadcast shapes {self.a_shape_} and {self.b_shape_} at axis {axis_i}"
                )

        if self.output_shape_:
            for axis_i_from_right in range(self.ignore_last_dim_num_):
                axis_i = rank_a - 1 - axis_i_from_right
                self.output_shape_[axis_i] = self.output_shape_[axis_i]

        if self.output_shape_:
            assert len(self.output_shape_) == len(self.output_shape_)
            for i, _ in enumerate(self.output_shape_):
                assert self.output_shape_[i] == self.output_shape_[i]

    def get_broadcast_dim(self, shape_a, shape_b, axis_i_from_right):
        """
        Compute the broadcast output dimension in the specific axis

        Args:
            shape_a: the shape of the first input
            shape_b: the shape of the second input
            axis_i_from_right: which axis we want to perform the computation
        Returns:
            can_broadcast: whether the broadcast can be performed in the specific axis
            broadcast_out_dim: the broadcast output dimension in the specific axis
            curr_axis_broadcast_on_a: whether the first input need to be broadcast in the specific axis
            curr_axis_broadcast_on_b: whether the first input need to be broadcast in the specific axis
        """
        axis_i_for_a = len(shape_a) - 1 - axis_i_from_right
        axis_i_for_b = len(shape_b) - 1 - axis_i_from_right
        broadcast_on_a = False
        broadcast_on_b = False
        if axis_i_for_a < 0:
            if axis_i_for_b < 0:
                return False, None, None, None
            # axis_i_for_b >= 0
            broadcast_on_a = True
            broadcast_out_dim = shape_b[axis_i_for_b]
        else:  # axis_i_for_a >= 0
            if axis_i_for_b < 0:
                broadcast_on_b = True
                broadcast_out_dim = shape_a[axis_i_for_a]
            else:
                if shape_a[axis_i_for_a] <= 0 or shape_b[axis_i_for_b] <= 0:
                    return False, None, None, None
                if shape_a[axis_i_for_a] == shape_b[axis_i_for_b]:
                    broadcast_out_dim = shape_a[axis_i_for_a]
                elif shape_a[axis_i_for_a] == 1:
                    broadcast_on_a = True
                    broadcast_out_dim = shape_b[axis_i_for_b]
                elif shape_b[axis_i_for_b] == 1:
                    broadcast_on_b = True
                    broadcast_out_dim = shape_a[axis_i_for_a]
                else:
                    return False, None, None, None
        return True, broadcast_out_dim, broadcast_on_a, broadcast_on_b

    def get_input_group_attrs(self, in_id: int, output_gslice_attrs: GroupSliceAttrs):
        """
        Get the input slice group attrs if we want to propagate groupslice
        of the output to the inputs

        broadcast behavior will be considered and handled automatically

        Args:
            in_id: which input we want to get the attrs
            output_gslice_attrs: the slice group attrs of the output of the broadcastable op
        Returns:
            the group slice attrs of the choseen input
        """
        broadcast = False

        if in_id == 0:
            broadcast = self.broadcast_on_a_[output_gslice_attrs.axis]
            rank = len(self.a_shape_)
        elif in_id == 1:
            broadcast = self.broadcast_on_b_[output_gslice_attrs.axis]
            rank = len(self.b_shape_)
        else:
            assert False, "invalid in_id"

        slice_axis_from_right = len(self.output_shape_) - output_gslice_attrs.axis - 1

        if broadcast:
            input_gslice_attrs: BaseGroupSliceAttrs = FullGroupSliceAttrs(
                output_gslice_attrs.num_outputs(),
                head_slice_ids=output_gslice_attrs.head_slice_ids,
                batch_slice_ids=output_gslice_attrs.batch_slice_ids,
            )
            return input_gslice_attrs

        in_slice_axis = rank - 1 - slice_axis_from_right
        input_gslice_attrs = GroupSliceAttrs(
            in_slice_axis,
            output_gslice_attrs.starts,
            output_gslice_attrs.ends,
            output_gslice_attrs.head_slice_ids,
            output_gslice_attrs.batch_slice_ids,
        )

        return input_gslice_attrs

    def broadcast_on(self, axis_of_output, input_id):
        """
        Determinate whether broadcast is happend in the specified input at the specified axis

        Args:
            axis_of_output: which axis we want to determine
            input_id: which input we want to determine
        Returns:
            whether the broadcast is happend
        """
        if input_id == 0:
            return self.broadcast_on_a_[axis_of_output]
        if input_id == 1:
            return self.broadcast_on_b_[axis_of_output]

        assert False, "invalid input_id"
