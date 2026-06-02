# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Concat->GroupSlice) -> (GroupSlice->Concat)
"""

import itertools

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    GroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    convert_attr_to_py,
    find_largest_element_smaller_than,
    find_smallest_element_larger_than,
    get_value_numeric_shape,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderConcatGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a_1, in_a_2, in_a_3, ...) --> b,b0,b1,b2
        {
            b = Concat(in_a_1, in_a_2, in_a_3, ...)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a_1, in_a_2, in_a_3, ...) --> b,b0,b1,b2
        {

            in_a_1.0, in_a_1.1, in_a_1.1 ... = GroupSlice(in_a_1)
            in_a_2.0, in_a_2.1, in_a_2.2 ... = GroupSlice(in_a_2)
            in_a_3.0, in_a_3.1, in_a_3.2 ... = GroupSlice(in_a_3)
            ...

            b0 = Concat(in_a_1.0, in_a_2.0, in_a_3.0, ...)
            b1 = Concat(in_a_1.1, in_a_2.1, in_a_3.1, ...)
            b2 = Concat(in_a_1.2, in_a_2.2, in_a_3.2, ...)
            ...

            # if possible
            b = Concat(b0,b1,b2,...)
            # or b = Concat(in_a_1, in_a_2, in_a_3, ...)

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

        if op_node.op_type != "Concat":
            return False

        check_static_shape_of_node_io(op_node)
        return True

    def create_mini_pattern(
        self,  # pylint: disable=R0913,R0917
        graph: ir.Graph,
        origin_op: ir.Node,
        mini_inputs: list[ir.Value | None],
        head_slice_id,
        batch_slice_id,
        slice_i,
        custom_kwargs: dict | None = None,
    ) -> list[ir.Value | None]:
        # filtere mini inputs
        mini_inputs = [x for x in mini_inputs if x is not None]
        if len(mini_inputs) > 1:
            return super().create_mini_pattern(
                graph, origin_op, mini_inputs, head_slice_id, batch_slice_id, slice_i, custom_kwargs
            )
        if len(mini_inputs) == 1:
            # no need to concat
            return mini_inputs

        assert False, "should not happen"

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true
        concat_axis = convert_attr_to_py(op_node.attributes["axis"], "as_int")

        out_gslice_attrs = get_gslice_attrs(gslice_node)

        if concat_axis != convert_attr_to_py(gslice_node.attributes["axis"], "as_int"):
            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                op_node,
                [gslice_node],
                inputs_gslice_attrs=[out_gslice_attrs for x in op_node.inputs],
            )

        else:
            # concat axis is the same as gslice axis
            # we should handle carefully for this case
            concat_in_dim_list = [get_value_numeric_shape(x)[concat_axis] for x in op_node.inputs]

            concat_in_dim_accum = [0] + list(itertools.accumulate(concat_in_dim_list))

            inputs_gslice_attrs = [GroupSliceAttrs(axis=concat_axis) for x in range(len(op_node.inputs))]

            for slice_i in range(out_gslice_attrs.num_outputs()):
                out_start = out_gslice_attrs.starts[slice_i]
                out_end = out_gslice_attrs.ends[slice_i]

                in_i_start = find_largest_element_smaller_than(concat_in_dim_accum, out_start)
                in_i_start = max(0, in_i_start)
                in_i_end = find_smallest_element_larger_than(concat_in_dim_accum, out_end)

                for input_i in range(len(op_node.inputs)):
                    if in_i_start <= input_i < in_i_end:
                        inputs_gslice_attrs[input_i].starts.append(
                            max(0, out_start - concat_in_dim_accum[input_i])
                        )
                        inputs_gslice_attrs[input_i].ends.append(
                            min(out_end, concat_in_dim_accum[input_i + 1]) - concat_in_dim_accum[input_i]
                        )
                        inputs_gslice_attrs[input_i].head_slice_ids.append(
                            out_gslice_attrs.head_slice_ids[slice_i]
                        )
                        inputs_gslice_attrs[input_i].batch_slice_ids.append(
                            out_gslice_attrs.batch_slice_ids[slice_i]
                        )
                    else:
                        # empty
                        inputs_gslice_attrs[input_i].starts.append(0)
                        inputs_gslice_attrs[input_i].ends.append(0)
                        inputs_gslice_attrs[input_i].head_slice_ids.append(
                            out_gslice_attrs.head_slice_ids[slice_i]
                        )
                        inputs_gslice_attrs[input_i].batch_slice_ids.append(
                            out_gslice_attrs.batch_slice_ids[slice_i]
                        )

            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                op_node,
                [gslice_node],
                inputs_gslice_attrs=list(inputs_gslice_attrs),  # list() to make mypy happy
            )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
