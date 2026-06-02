# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(ReduceOp->GroupSlice) -> (GroupSlice->ReduceOp)
"""

from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import (
    VariableExtraInfo,
    convert_attr_to_py,
    get_attribute_with_default,
    get_constant_np,
    get_value_numeric_shape,
    have_static_shape_on_node_io,
    is_constant,
    make_initializer,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderUnaryReduceGroupslice(M2sBasePass):
    """
    Reorder subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = ReduceOp(in_a) # such as ReduceSum
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            a0,a1,a2... = GroupSlice(in_a)

            b0 = ReduceOp(a0)
            b1 = ReduceOp(a1)
            b2 = ReduceOp(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = ReduceOp(in_a)
        }

    Also, encodings are updated
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        axes_as_input: bool
        gslice_axis: int

        # inputs/attributes of Reduce op
        data_v: ir.Value
        reduce_axes: list[int]
        keepdims: int

    SUPPORTED_OP_TYPES = {
        "ReduceMin": {"min_op_ver_axes_as_input": 18},
        "ReduceMean": {"min_op_ver_axes_as_input": 18},
        "ReduceMax": {"min_op_ver_axes_as_input": 18},
        "ReduceProd": {"min_op_ver_axes_as_input": 18},
        "ReduceSum": {"min_op_ver_axes_as_input": 13},
    }

    def is_axes_as_input(self, graph: ir.Graph, reduce_node: ir.Node):
        op_version = graph.opset_imports[reduce_node.domain]
        if op_version < self.SUPPORTED_OP_TYPES[reduce_node.op_type]["min_op_ver_axes_as_input"]:
            return False
        return True

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfoProtocol:  # pylint: disable=R0911
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False

        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        reduce_node = gslice_node.inputs[0].producer()
        assert reduce_node is not None  # check for mypy, definitely true
        if reduce_node.op_type not in self.SUPPORTED_OP_TYPES:
            return False
        if not have_static_shape_on_node_io(reduce_node):
            return False
        if reduce_node.domain not in ["", "ai.onnx", "main"]:
            return False
        axes_as_input = self.is_axes_as_input(graph, reduce_node)

        keepdims = get_attribute_with_default(reduce_node, "keepdims", 1)
        data_v = reduce_node.inputs[0]
        if data_v is None:
            return False
        reduce_axes: list[int] = []
        if axes_as_input:
            if len(reduce_node.inputs) > 1 and reduce_node.inputs[1] is not None:
                if not is_constant(reduce_node.inputs[1]):
                    return False
                reduce_axes = get_constant_np(reduce_node.inputs[1]).tolist()
            else:
                reduce_axes = []
            noop_with_empty_axes = get_attribute_with_default(reduce_node, "noop_with_empty_axes", 0)

            if len(reduce_axes) == 0:
                # no axes
                if not noop_with_empty_axes:
                    # reduce on all axes, can't reorder groupslice
                    return False
                # for else case, act as an Identity
        else:
            reduce_axes = list(convert_attr_to_py(reduce_node.attributes["axes"], "as_ints"))

        # normalize reduce_axes
        if len(reduce_axes) > 0:
            input_rank = len(get_value_numeric_shape(data_v))
            for i in range(len(reduce_axes)):
                if reduce_axes[i] < 0:
                    reduce_axes[i] += input_rank

        gslice_axis = convert_attr_to_py(gslice_node.attributes["axis"], "as_int")
        if gslice_axis in reduce_axes:
            return False

        return self.MatchInfo(
            axes_as_input=axes_as_input,
            gslice_axis=gslice_axis,
            data_v=data_v,
            reduce_axes=reduce_axes,
            keepdims=keepdims,
        )

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, self.MatchInfo)
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        reduce_node = gslice_node.inputs[0].producer()
        assert reduce_node is not None  # check for mypy, definitely true

        reduce_axes = match_info.reduce_axes
        assert reduce_node.inputs[0] is not None  # already checked in match
        assert reduce_node.inputs[0].shape is not None  # already checked in match

        # when reduce_axis != gslice_axis
        # input gslice attrs should be the same on output gslice
        a_groupslice_attrs = get_gslice_attrs(gslice_node)

        if match_info.keepdims == 0:
            out2int_map = {}
            j = 0
            for i in range(len(reduce_node.inputs[0].shape)):
                if i in reduce_axes:
                    continue
                out2int_map[j] = i
                j += 1
            a_groupslice_attrs.axis = out2int_map[a_groupslice_attrs.axis]

        inputs_gslice_attrs = [a_groupslice_attrs]
        if match_info.axes_as_input:
            axes_groupslice_attrs = FullGroupSliceAttrs(
                a_groupslice_attrs.num_outputs(),
                a_groupslice_attrs.head_slice_ids,
                a_groupslice_attrs.batch_slice_ids,
            )
            inputs_gslice_attrs.append(axes_groupslice_attrs)
        new_outputs = self.rewrite_based_on_gslice_attrs(
            graph, reduce_node, [gslice_node], inputs_gslice_attrs=inputs_gslice_attrs
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), reduce_node.name)
        return True
