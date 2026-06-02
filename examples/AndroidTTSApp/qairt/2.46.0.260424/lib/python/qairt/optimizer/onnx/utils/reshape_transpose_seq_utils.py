# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Module to provide helper functions for the analysis of reshape transpose sequence
"""

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.utils.ir_extra_info import (
    VariableExtraInfo,
)
from qairt.optimizer.onnx.utils.reshape_transpose_analysis import (
    ReshapeTransposeInfoSeq,
    group_transpose,
)
from qairt.optimizer.onnx.utils.utils import (
    convert_attr_to_py,
    get_value_numeric_shape,
    has_static_shape_on_value,
    have_static_shape_on_node_io,
    make_initializer,
)


def find_reshape_transpose_seq_top_down(in_value: ir.Value, exclude_node_names: set[str]):
    """
    Find reshape or transpose op in the graph, top down
    """
    accept_types = set(["Transpose", "Reshape", "Squeeze", "Unsqueeze"])
    # attention, top_down searching only works for node that has only one user

    op_list = []

    curr_in_value = in_value
    v_extra_info = in_value.meta["extra_info"].copy()
    # found seq from the top
    while True:
        in_value_uses = list(curr_in_value.uses())
        if len(in_value_uses) != 1:
            break
        curr_n = in_value_uses[0].node
        if curr_n.op_type not in accept_types:
            break
        if curr_n.name in exclude_node_names:
            break
        curr_out_value = curr_n.outputs[0]
        # search top-down, so we need to check output encodings
        if (not v_extra_info.defined_encodings()) and curr_out_value.meta["extra_info"].defined_encodings():
            v_extra_info = curr_out_value.meta["extra_info"].copy()
        elif (
            curr_out_value.meta["extra_info"].defined_encodings()
            and v_extra_info != curr_out_value.meta["extra_info"]
        ):
            break
        op_list.append(curr_n)
        curr_in_value = curr_out_value
    out_value = curr_in_value
    op_seq = ReshapeTransposeOpSeq(in_value, out_value, op_list, v_extra_info)
    return op_seq


def find_reshape_transpose_seq_bottom_up(out_value: ir.Value, exclude_node_names: set[str]):
    """
    Find reshape or transpose op in the graph, bottom up
    attention, currently bottom-up searching only works for node that has only one user
    """
    accept_types = set(["Transpose", "Reshape", "Squeeze", "Unsqueeze"])
    # found seq from the bottom
    curr_out_v = out_value
    v_extra_info = out_value.meta["extra_info"].copy()
    op_list = []
    while True:
        # this is bottom-up search, so node with multi users are allowed
        producer = curr_out_v.producer()
        if producer is None:
            break
        if producer.op_type not in accept_types:
            break
        if producer.name in exclude_node_names:
            break
        # search bottom-up, so we need to check input encodings
        curr_in_v = producer.inputs[0]
        assert curr_in_v is not None  # check for mypy, definitely true

        if (not v_extra_info.defined_encodings()) and curr_in_v.meta["extra_info"].defined_encodings():
            v_extra_info = curr_in_v.meta["extra_info"].copy()
        elif (
            curr_in_v.meta["extra_info"].defined_encodings() and v_extra_info != curr_in_v.meta["extra_info"]
        ):
            break
        op_list.append(producer)
        curr_out_v = curr_in_v
    # reverse seq, so that the first one is the first node in the seq
    op_list = op_list[::-1]

    assert all(x.op_type in accept_types for x in op_list)
    op_seq = ReshapeTransposeOpSeq(curr_out_v, out_value, op_list, v_extra_info)
    return op_seq


class ReshapeTransposeOpSeq:
    """
    A helper class to represent transpose / reshape sequence in ir.Node format
    """

    def __init__(self, input_v: ir.Value, output_v: ir.Value, op_list: list[ir.Node], v_extra_info=None):
        self.input_v = input_v
        self.output_v = output_v
        self.op_list = op_list
        self.info_seq: ReshapeTransposeInfoSeq | None = None
        self.v_extra_info = v_extra_info

    @classmethod
    def create_by_info_seq(  # pylint: disable=[too-many-locals]
        cls,
        graph: ir.Graph,
        input_v: ir.Value,
        v_extra_info: VariableExtraInfo | None,
        info_seq: ReshapeTransposeInfoSeq,
    ):
        """
        Build new ir.Node sequence (ReshapeTransposeOpSeq)
        with the information of ReshapeTransposeInfoSeq

        Args:
            graph: ir graph
            op_seq: original op sequence
            input_v: the input of the original op sequence
            v_extra_info: the extra info of the tensors in the sequence,
                        they should share same encodings
            info_seq: the info sequence
        Return:
            ReshapeTransposeOpSeq

        """
        if not has_static_shape_on_value(input_v):
            return None
        input_shape = get_value_numeric_shape(input_v)

        assert input_v.name is not None  # check for mypy
        # rewrite the graph
        curr_shape = input_shape
        new_nodes = []
        name_hint = input_v.name + "/simplify/"
        curr_input_v = input_v
        for info in info_seq.seq:
            if isinstance(info, ReshapeTransposeInfoSeq.ReshapeNodeInfo):
                op_name = graph.meta["extra_info"].get_unique_name_with_suffix(name_hint, "/Reshape")
                shape_v_name = graph.meta["extra_info"].get_unique_name("shape_v")
                shape_v = make_initializer(graph, shape_v_name, np.array(info.shape, dtype=np.int64))
                node = ir.Node("", "Reshape", [curr_input_v, shape_v], num_outputs=1, name=op_name)
                output_v = node.outputs[0]
                output_v.name = graph.meta["extra_info"].get_unique_name_with_suffix(
                    name_hint, "/Reshape_out"
                )

            elif isinstance(info, ReshapeTransposeInfoSeq.TransposeNodeInfo):
                op_name = graph.meta["extra_info"].get_unique_name_with_suffix(name_hint, "Transpose")
                node = ir.Node("", "Transpose", [curr_input_v], num_outputs=1, name=op_name)
                node.attributes["perm"] = ir.AttrInt64s("perm", info.perm)
                output_v = node.outputs[0]
                output_v.name = graph.meta["extra_info"].get_unique_name_with_suffix(
                    name_hint, "/Transpose_out"
                )

            else:
                assert False

            curr_input_v_producer = curr_input_v.producer()
            if curr_input_v_producer is None:
                graph.insert_before(graph[0], node)
            else:
                graph.insert_after(curr_input_v_producer, node)
            new_nodes.append(node)
            curr_shape = info.infer_shape(curr_shape)
            output_v = node.outputs[0]
            output_v.shape = ir.Shape(curr_shape)
            if curr_input_v.dtype is not None:
                output_v.dtype = curr_input_v.dtype
            if v_extra_info is not None:
                output_v.meta["extra_info"] = v_extra_info.copy(ignore_safetensors=True)
            curr_input_v = output_v

        op_seq = ReshapeTransposeOpSeq(input_v, curr_input_v, new_nodes)
        op_seq.info_seq = info_seq
        return op_seq

    def build_info_seq(self):
        """
        Build ReshapeTransposeInfoSeq from ReshapeTransposeOpSeq

        Return:
            ReshapeTransposeInfoSeq

        """

        if not has_static_shape_on_value(self.input_v):
            return None

        info_node_list: list[ReshapeTransposeInfoSeq.BaseNodeInfo] = []
        for x in self.op_list:
            if x.op_type in ("Reshape", "Squeeze", "Unsqueeze"):
                if not have_static_shape_on_node_io(x):
                    return None
                info_node_list.append(
                    ReshapeTransposeInfoSeq.ReshapeNodeInfo(get_value_numeric_shape(x.outputs[0]))
                )
            elif x.op_type == "Transpose":
                info_node_list.append(
                    ReshapeTransposeInfoSeq.TransposeNodeInfo(convert_attr_to_py(x.attributes["perm"]))
                )
            else:
                raise ValueError(f"Unknown op type: {x.op_type}")
        info_seq = ReshapeTransposeInfoSeq(info_node_list, get_value_numeric_shape(self.input_v))
        self.info_seq = info_seq
        return info_seq


def determine_seq_complexity(seq: ReshapeTransposeInfoSeq) -> int:
    """
    Determine the reshape transpose sequence complexity

    Args:
        ReshapeTransposeInfoSeq: info seq
    Return:
        complexity

    """
    complexity = 0
    curr_shape = seq.input_shape
    for s in seq.seq:
        if isinstance(s, ReshapeTransposeInfoSeq.ReshapeNodeInfo):
            complexity += 1
        elif isinstance(s, ReshapeTransposeInfoSeq.TransposeNodeInfo):
            complexity += get_complexity_of_transpose(s.perm, curr_shape)
        else:
            assert False
        curr_shape = s.infer_shape(curr_shape)
    return complexity


def get_complexity_of_transpose(perm, input_shape):
    """
    Determine the complexity of the transpose op

    Args:
        perm: transpose permutation
        input_shape: the input of the transpose
    Return:
        complexity

    """
    if len(input_shape) == 5:
        return 5
    if len(input_shape) > 5:
        return 20
    # group the transpose
    # e.g. [0,3,1,2] ==> [0,2,1]
    grouped_perm, _ = group_transpose(perm, input_shape)

    # remove the same axis before/after the transpose
    # e.g. [0,2,1,3] ==> [2,1]
    # e.g. [0,3,2,1] ==> [3,1]
    # e.g. [2,0,3,1] ==> [2,0,3,1]
    diff_axis = np.abs(np.array(grouped_perm) - np.array(range(len(grouped_perm)))).sum()
    if diff_axis == 0:
        return 0
    if diff_axis == 2:
        return 1
    return 2


def remove_common_ancesters_of_reshape_transpose_op_seq(
    in_a_op_seq: ReshapeTransposeOpSeq,
    in_b_op_seq: ReshapeTransposeOpSeq,
):
    """
    Remove the common ancesters of in_a_op_seq and in_b_op_seq

    maybe in_a_layout_transform_op_seq intersect with in_b_layout_transform_op_seq
    we don't allow this intersection, so remove the common nodes

    Args:
        in_a_op_seq: op seq a
        in_b_op_seq: op seq b
    Return:
        new_in_a_op_seq
        new_in_b_op_seq

    """

    intersect_node_names = set(x.name for x in in_a_op_seq.op_list).intersection(
        set(x.name for x in in_b_op_seq.op_list)
    )
    if len(intersect_node_names) == 0:
        return in_a_op_seq, in_b_op_seq

    # has common ancesters! remove them
    # op_list are sorted, so we should find the last common node
    nearest_common_ancester = None
    for node in in_a_op_seq.op_list:
        if node.name in intersect_node_names:
            nearest_common_ancester = node

    assert nearest_common_ancester is not None  # check for mypy, definitely true

    new_a_op_list = in_a_op_seq.op_list[in_a_op_seq.op_list.index(nearest_common_ancester) + 1 :]
    new_b_op_list = in_b_op_seq.op_list[in_b_op_seq.op_list.index(nearest_common_ancester) + 1 :]

    return ReshapeTransposeOpSeq(
        nearest_common_ancester.outputs[0], in_a_op_seq.output_v, new_a_op_list, in_a_op_seq.v_extra_info
    ), ReshapeTransposeOpSeq(
        nearest_common_ancester.outputs[0], in_b_op_seq.output_v, new_b_op_list, in_b_op_seq.v_extra_info
    )
