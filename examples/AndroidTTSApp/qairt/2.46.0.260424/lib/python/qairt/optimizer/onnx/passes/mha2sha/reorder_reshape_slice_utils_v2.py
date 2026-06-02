# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the analysis functions for Reshape->GroupSlice reordering
This is V2 version, will not use isl and islpy
"""

from qairt.optimizer.onnx.utils.reshape_transpose_analysis import (
    DimNode,
    ReshapeTransposeInfoSeq,
)


def get_memory_stride_of_tensor(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Assume tensor memory is flattern, get the memory stride of a tensor
    """
    reversed_stride: list[int] = [1]
    reversed_shape = list(shape)
    reversed_shape.reverse()
    for s in reversed_shape[:-1]:
        reversed_stride.append(reversed_stride[-1] * s)
    stride = reversed_stride[:]
    stride.reverse()
    return tuple(stride)


def get_index_of_position(pos: int, shape: tuple[int, ...], stride: tuple[int, ...]) -> tuple[int, ...]:
    rest_pos = pos
    index = [0] * len(shape)

    for dim_i in range(len(shape)):
        index[dim_i] = rest_pos // stride[dim_i]
        rest_pos = rest_pos % stride[dim_i]

    return tuple(index)


class DimNodeSliceInfo:
    def __init__(self, at_node: DimNode, start: int, last: int):
        self.at_node = at_node
        self.start = start
        self.last = last


def get_slice_infos_at_node(slice_info_list: list[DimNodeSliceInfo], node: DimNode) -> list[DimNodeSliceInfo]:
    all_founds = []
    for slice_info in slice_info_list:
        if slice_info.at_node is node:
            all_founds.append(slice_info)
    return all_founds


def get_reshape_slice_reordered_slice_attrs(  # pylint: disable=R0914
    o_shape: list[int] | tuple[int, ...],
    a_shape: list[int] | tuple[int, ...],
    a2b_slice_axes: list[int],
    a2b_slice_starts: list[int],
    a2b_slice_ends: list[int],
) -> tuple[bool, list[int], list[int], list[int]]:
    """
    Try to determine whether we can reorder
    subgraph:
        To --(Reshape)--> Ta --(Slice)--> Tb
    into:
        To --(Slice)--> Tc --(Reshape)--> Tb

    and determine the attributes of reordered_slice (from T_o to T_c)

    index convention of this function:
        - "start","end" use the same convention as numpy, that is end is not included
        - "last" is the last one index that can be accesed, that is last=end-1
    """

    info_seq = ReshapeTransposeInfoSeq(
        [ReshapeTransposeInfoSeq.ReshapeNodeInfo(a_shape)], input_shape=o_shape
    )
    dim_trees, node_seqs = info_seq.build_dim_tree()

    if len(dim_trees) != 1:
        return False, [], [], []
    dim_tree = dim_trees[0]
    if dim_tree is None:
        return False, [], [], []

    a_dims: list[list[DimNode]] = dim_tree.curr_dims
    o_dims: list[list[DimNode]] = [x.get_flatten_edge_nodes() for x in dim_tree.src_root.children]

    a2b_slice_infos = []
    for axis, start, end in zip(a2b_slice_axes, a2b_slice_starts, a2b_slice_ends):
        assert start >= 0
        assert end <= a_shape[axis]
        internal_dims = a_dims[axis]
        internal_shape = tuple(x.size for x in a_dims[axis])
        internal_stride = get_memory_stride_of_tensor(internal_shape)
        internal_start_index = get_index_of_position(start, internal_shape, internal_stride)
        internal_last_index = get_index_of_position(end - 1, internal_shape, internal_stride)

        # internal_last_index is the last index reachable.
        # we expect the polyhedral bounded by internal_start_index/internal_last_index should be a box
        #   e.g. given internal_shape [2,8]
        #      good cases:
        #               internal_start_index:[0,0], internal_last_index:[0,0]
        #               internal_start_index:[0,2], internal_last_index:[0,7]
        #               internal_start_index:[1,0], internal_last_index:[2,7]
        #      bad cases:
        #               internal_start_index:[0,2], internal_last_index:[1,3]

        # check if the polyhedral is a box
        first_diff_i = len(internal_shape)
        for internal_i in range(len(internal_shape)):
            if internal_start_index[internal_i] != internal_last_index[internal_i]:
                first_diff_i = internal_i
                break

        # get bounding box of the polyhedral
        internal_index_mins = []
        internal_index_maxs = []  # included
        for internal_i in range(len(internal_shape)):
            if internal_i > first_diff_i:
                internal_index_mins.append(0)
                internal_index_maxs.append(internal_shape[internal_i] - 1)
            else:
                internal_index_mins.append(
                    min(internal_start_index[internal_i], internal_last_index[internal_i])
                )
                internal_index_maxs.append(
                    max(internal_start_index[internal_i], internal_last_index[internal_i])
                )

        point_num_of_bounding_box = 1
        for internal_i in range(len(internal_shape)):
            point_num_of_bounding_box = point_num_of_bounding_box * (
                internal_index_maxs[internal_i] - internal_index_mins[internal_i] + 1
            )

        if point_num_of_bounding_box != end - start:
            # not a box!
            return False, [], [], []

        for internal_i in range(len(internal_shape)):
            internal_start_index_i = internal_start_index[internal_i]
            internal_last_index_i = internal_last_index[internal_i]
            if internal_start_index_i == 0 and internal_last_index_i == internal_shape[internal_i] - 1:
                # full slice
                continue
            else:
                a2b_slice_infos.append(
                    DimNodeSliceInfo(internal_dims[internal_i], internal_start_index_i, internal_last_index_i)
                )

    if len(a2b_slice_infos) == 0:
        # full slice
        return True, [], [], []

    # Prepare output
    o2c_slice_axes = []
    o2c_slice_starts = []
    o2c_slice_ends = []

    for o_axis, o_internal_dims in enumerate(o_dims):
        slice_infos_with_index: list[tuple[int, DimNodeSliceInfo]] = []
        for d_i, d in enumerate(o_internal_dims):
            slice_infos_with_index += [(d_i, x) for x in get_slice_infos_at_node(a2b_slice_infos, d)]

        if len(slice_infos_with_index) == 0:
            continue
        if len(slice_infos_with_index) > 1:
            return False, [], [], []

        o_internal_index_id, slice_info = slice_infos_with_index[0]
        if o_internal_index_id != 0:
            # this behavior aligns with islpy version of get_reshape_slice_reordered_slice_attrs
            # it can actually be represented if we support slice_step
            return False, [], [], []

        # map o_internal_index(nd) to pos (1d) of o_axis
        o_internal_stride = get_memory_stride_of_tensor(tuple(x.size for x in o_internal_dims))
        o_internal_pos_start = slice_info.start * o_internal_stride[o_internal_index_id]
        o_internal_pos_end = (slice_info.last + 1) * o_internal_stride[o_internal_index_id]

        o2c_slice_axes.append(o_axis)
        o2c_slice_starts.append(o_internal_pos_start)
        o2c_slice_ends.append(o_internal_pos_end)

    return True, o2c_slice_axes, o2c_slice_starts, o2c_slice_ends
