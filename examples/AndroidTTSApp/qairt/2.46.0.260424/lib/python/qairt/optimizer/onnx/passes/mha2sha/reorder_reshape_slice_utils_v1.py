# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the analysis functions for Reshape->GroupSlice reordering
"""

import sys

try:
    import islpy as isl
except ImportError as e:
    if sys.platform.startswith("win"):
        raise ImportError(
            "Windows platform detected and islpy package is required for this module. "
            "However islpy is not available via pip on Windows.\n"
            "Please explore one of these options:\n"
            "  1. Use conda: install Miniconda; conda create -n poly python=3.11; "
            "conda activate poly; conda install -c conda-forge islpy\n"
            "  2. Use WSL (Ubuntu): inside WSL run: pip install islpy\n"
        ) from e
    raise ImportError("Please install islpy (e.g. pip install islpy) to use this module.") from e


def get_memory_stride_of_tensor(shape: list[int]) -> list[int]:
    """
    Assume tensor memory is flattern, get the memory stride of a tensor
    """
    reversed_stride = [1]
    reversed_shape = list(shape[:])
    reversed_shape.reverse()
    for s in reversed_shape[:-1]:
        reversed_stride.append(reversed_stride[-1] * s)
    stride = reversed_stride[:]
    stride.reverse()
    return stride


def add_constraint_by_min_max(set_: isl.Set, input_dim_i: int, min_value: int, max_value: int):
    """
    Add min/max bound to Set
    """
    set_ = set_.lower_bound_val(isl.dim_type.set, input_dim_i, min_value)
    set_ = set_.upper_bound_val(isl.dim_type.set, input_dim_i, max_value)
    return set_


def create_set_by_shape(ctx: isl.Context, shape: list[int]):
    """
    Create index Set by tensor shape
    """
    space = isl.Space.set_alloc(ctx, 0, len(shape))
    set_ = isl.Set.universe(space)

    # add shape constraint for each dimension
    for i, dim_size in enumerate(shape):
        set_ = add_constraint_by_min_max(set_, i, 0, dim_size - 1)
    return set_


def get_reshape_relation(
    src_set: isl.Set, dst_set: isl.Set, src_shape: list[int], dst_shape: list[int]
) -> isl.Map:
    """
    Get index relationship between Reshape input and output,
    represented in isl.Map
    """
    src2dst_map = isl.Map.from_domain_and_range(src_set.copy(), dst_set.copy())

    src_mem_stride = get_memory_stride_of_tensor(src_shape)
    dst_mem_stride = get_memory_stride_of_tensor(dst_shape)

    # add the constraint of memory position of reshape op:
    #      dot(src_shape, src_mem_stride.transpose()) == dot(dst_shape, dst_mem_stride.transpose())

    constraint = isl.Constraint.equality_alloc(src2dst_map.get_space())

    # in form "const + coeff_i*var_i =0"
    for src_i in range(len(src_shape)):
        constraint = constraint.set_coefficient_val(isl.dim_type.in_, src_i, src_mem_stride[src_i])

    for dst_i in range(len(dst_shape)):
        constraint = constraint.set_coefficient_val(isl.dim_type.out, dst_i, -dst_mem_stride[dst_i])

    src2dst_map = src2dst_map.add_constraint(constraint)
    return src2dst_map


def is_fixed_box(set_0: isl.Set) -> tuple[bool, list[int], list[int]]:
    """
    Check if a set is a bounded fixed box

    Get the constrained box by get_simple_fixed_box_hull is simplier
    but islpy doesn't export function of FixedBox
    so we have to get min/max manually

    """

    # may require to call polyhedral_hull() first, not sure if it's necessary
    # such as set_0 = set_0.polyhedral_hull().to_set()
    # polyhedral_hull is an expensive operation
    if not set_0.is_bounded():
        return False, [], []

    dim_num = set_0.get_space().dim(isl.dim_type.set)
    offsets = []
    sizes = []
    for i in range(dim_num):
        curr_dim_min = set_0.dim_min_val(i).get_num_si()
        curr_dim_max = set_0.dim_max_val(i).get_num_si()
        offsets.append(curr_dim_min)
        sizes.append(curr_dim_max - curr_dim_min + 1)

    # Check if the box is full (no hole inside)
    rebuilt_box = isl.Set.universe(set_0.get_space())
    for i in range(dim_num):
        rebuilt_box = add_constraint_by_min_max(rebuilt_box, i, offsets[i], offsets[i] + sizes[i] - 1)
    if not set_0.is_equal(rebuilt_box):
        return False, [], []

    return True, offsets, sizes


def get_slice_relation(
    src_set: isl.Set, dst_set: isl.Set, slice_starts: list[int], dst_shape: list[int]
) -> isl.Map:
    """
    Get index relationship between Slice input and output,
    represented in isl.Map
    """
    src2dst_map = isl.Map.from_domain_and_range(src_set.copy(), dst_set.copy())

    # add the constraint of memory position of slice op:
    for i in range(len(dst_shape)):
        # src_i = dst_i + slice_start_i
        space = src2dst_map.get_space()
        constraint = isl.Constraint.equality_alloc(space)
        constraint = constraint.set_coefficient_val(isl.dim_type.in_, i, -1)
        constraint = constraint.set_coefficient_val(isl.dim_type.out, i, 1)
        constraint = constraint.set_constant_val(slice_starts[i])
        src2dst_map = src2dst_map.add_constraint(constraint)

    return src2dst_map


def get_reshape_slice_reordered_slice_attrs(  # pylint: disable=R0914
    o_shape: list[int] | tuple[int],
    a_shape: list[int] | tuple[int],
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

    Analysis this is not so easy, it is not linear transformation

    We need the help of polyhedron tool (ISL)
    """
    o_shape = list(o_shape)
    a_shape = list(a_shape)

    # Create a context
    ctx = isl.Context()
    o_set = create_set_by_shape(ctx, o_shape)
    a_set = create_set_by_shape(ctx, a_shape)

    # Reshape
    # create mapping from o_set to a_set
    o2a_map = get_reshape_relation(o_set, a_set, o_shape, a_shape)

    # Slice
    # create mapping from a_set to b_set
    b_shape = a_shape[:]
    a2b_slice_starts_full = [0] * len(b_shape)
    for i, axis in enumerate(a2b_slice_axes):
        b_shape[axis] = a2b_slice_ends[i] - a2b_slice_starts[i]
        a2b_slice_starts_full[axis] = a2b_slice_starts[i]
    b_set = create_set_by_shape(ctx, b_shape)
    a2b_map = get_slice_relation(a_set, b_set, a2b_slice_starts_full, b_shape)

    # Merge Reshape and Slice
    o2b_map = o2a_map.apply_range(a2b_map)

    # project out variables of b
    o_prime = o2b_map.project_out(isl.dim_type.out, 0, len(b_shape)).domain()

    # Get the constrained box by get_simple_fixed_box_hull is simplier
    # but islpy doesn't export function of FixedBox,
    # so we have to get min/max manually
    can_reorder, o_constrained_box_offset, o_constrained_box_size = is_fixed_box(o_prime)

    # Prepare output
    o2c_slice_axes = []
    o2c_slice_starts = []
    o2c_slice_ends = []

    if can_reorder:
        for i, o_dim_size in enumerate(o_shape):
            if o_constrained_box_size[i] == o_dim_size:
                # no need to slice in this dimension
                continue
            o2c_slice_axes.append(i)
            o2c_slice_starts.append(o_constrained_box_offset[i])
            o2c_slice_ends.append(o_constrained_box_offset[i] + o_constrained_box_size[i])

    return can_reorder, o2c_slice_axes, o2c_slice_starts, o2c_slice_ends
