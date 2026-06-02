# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Scan the graph, find out all pack qkv possible candidates,
mark the split or slice of pack qkv as reorderable

"""

import itertools

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.utils.ir_extra_info import SplittableParam
from qairt.optimizer.onnx.utils.reshape_transpose_analysis import DimTree
from qairt.optimizer.onnx.utils.reshape_transpose_seq_utils import find_reshape_transpose_seq_top_down
from qairt.optimizer.onnx.utils.utils import (
    find_largest_element_smaller_than,
    find_smallest_element_larger_than,
    get_slice_static_params,
    get_value_numeric_shape,
    is_constant,
)
from qairt.optimizer.utils.logger import logger


class M2sMarkPackQKVSplittable(M2sBasePass):
    """
    Pass to mark the split/slice of possible PackQKV candidates as reorderable.

    The pattern of Pack QKV is like

                    (Conv)/(Matmul+Bias)
                        |
                    Transpose/Reshape sequence
                        |
                    Split/Slice                  (note: can be one Split or three Slices)
                    /     |       \
        Q_proj_out  K_Proj_out  V_proj_out

    We need to mark the input of split/slice as Splittable

    Algo:
    - Search (Conv)/(MatMul+Bias) + Transpose/Reshape | None + Split/Slice pattern
    - Map the split/slice axis to Conv/Matmul by analyzing the Transpose/Reshape Sequence
    - Check whether slicing on the mapped axis of Conv/Matmul affect only the weights
        - if so, mark the input of Split/Slice as "Splittable"

    Note:
        Marking them Splittable does not mean it will be reordered (propagated bottom-up).
        Only when we see Split->GroupSlice then this split will be transformed into GroupSlice->GroupSlice, and then it will start the reorder process
        This ensures that Attention non-related subgraphs will not be affected.
    """

    def verify_slicing_proj_weight_after_reshape_transpose(
        self, slice_axis: int, slice_start: int, slice_end: int, dim_tree: DimTree | None, proj_op: ir.Node
    ):
        if dim_tree is None:
            return self.verify_slicing_proj_weight(slice_axis, proj_op)

        minidims_sizes = [x.size for x in dim_tree.curr_dims[slice_axis]]
        minidims_accm = list(itertools.accumulate(minidims_sizes, lambda a, b: a + b, initial=0))
        minidims_start_id = find_largest_element_smaller_than(minidims_accm, slice_start)
        minidims_start_id = max(0, minidims_start_id)
        minidims_end_id = find_smallest_element_larger_than(minidims_accm, slice_end)
        minidims_to_slice = dim_tree.curr_dims[slice_axis][minidims_start_id:minidims_end_id]

        # mapping minidims_to_slice to the dim of proj op output
        # all minidim of minidims_to_slice should be inside of the same dim of proj op output
        proj_out_dims = dim_tree.src_root.children
        proj_out_slice_dim = None
        for minidim in minidims_to_slice:
            for i, x in enumerate(proj_out_dims):
                if x.contains(minidim):
                    if proj_out_slice_dim is None:
                        proj_out_slice_dim = i
                    elif proj_out_slice_dim != i:
                        return False
                    break
        if proj_out_slice_dim is None:
            return False
        return self.verify_slicing_proj_weight(proj_out_slice_dim, proj_op)

    def verify_slicing_proj_weight(self, slice_axis: int, proj_op: ir.Node):
        proj_op_out_rank = len(get_value_numeric_shape(proj_op.outputs[0]))
        if proj_op.op_type == "Conv":
            if slice_axis == proj_op_out_rank - 3:
                # actually slicing the weight of Conv, so always sliceable
                return True
        elif proj_op.op_type == "MatMul":
            if slice_axis == proj_op_out_rank - 2:
                # actually slicing the input0 of MatMul, so we need to ensure it is constant
                if is_constant(proj_op.inputs[0]):
                    return True
            elif slice_axis == proj_op_out_rank - 1:
                # actually slicing the input1 of MatMul, so we need to ensure it is constant
                if is_constant(proj_op.inputs[1]):
                    return True
        return False

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        # top-down scan

        if node.op_type not in ["Conv", "MatMul"]:
            return False
        search_after = node
        if node.op_type == "MatMul":
            # maybe there is a bias
            uses = list(node.outputs[0].uses())
            if len(uses) == 1 and uses[0].node.op_type in ["Add", "Sub"]:
                search_after = uses[0].node

        # search for reshape/transpose after Conv or MatMul+Add
        reshape_transpose_op_seq = find_reshape_transpose_seq_top_down(search_after.outputs[0], set())

        # build dim-tree of reshape_transpose_seq
        info_seq = reshape_transpose_op_seq.build_info_seq()
        dim_trees, node_seqs = info_seq.build_dim_tree()

        if len(dim_trees) == 0:
            # this means that no reshape/transpose op are found
            dim_tree = None
        elif len(dim_trees) > 1 or dim_trees[0] is None:
            # this means that we can't track the reshape/transpose sequence by just one DimTree
            # like reshape [1,4,5,1024] to [1,5,4,1024]
            # this is a rare case, and probably not PackQKV related
            return False
        else:
            dim_tree = dim_trees[0]

        out_v: ir.Value = reshape_transpose_op_seq.output_v
        out_v_shape = get_value_numeric_shape(out_v)
        # check if out_v are used only by Slices or Split
        splittable: set[tuple[int, int, int]] = set()

        uses = list(out_v.uses())
        if len(uses) == 1 and uses[0].node.op_type == "Split":
            split_axis = uses[0].node.attributes["axis"].as_int()
            if not self.verify_slicing_proj_weight_after_reshape_transpose(
                split_axis, 0, out_v_shape[split_axis], dim_tree, node
            ):
                return False
            # mark output of split node as splittable
            splittable.add((split_axis, 0, out_v_shape[split_axis]))

        elif len(uses) == 3 and all(x.node.op_type == "Slice" for x in uses):
            for use in uses:
                slice_params = get_slice_static_params(use.node)
                if slice_params is None:
                    return False
                for slice_i in range(len(slice_params["starts"])):
                    axis = slice_params["axes"][slice_i]
                    start = slice_params["starts"][slice_i]
                    end = slice_params["ends"][slice_i]

                    if not self.verify_slicing_proj_weight_after_reshape_transpose(
                        slice_params["axes"][slice_i],
                        slice_params["starts"][slice_i],
                        slice_params["ends"][slice_i],
                        dim_tree,
                        node,
                    ):
                        return False
                    splittable.add((axis, start, end))
        else:
            return False

        # sort the splittable to make it stable (good for test)
        splittable_sorted = list(splittable)
        splittable_sorted.sort(key=lambda x: x[1])

        # set splittable to out_v
        for axis, start, end in splittable_sorted:
            out_v.meta["extra_info"].splittable.append(SplittableParam(axis, start, end))

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), out_v.name)

        # We don't need to call rewrite for this Pass
        # so return False
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        """
        This pass only marks nodes in match() and doesn't need rewrite.
        This method is required by BasePredicatePass but never called since match() returns False.
        """
        return False
