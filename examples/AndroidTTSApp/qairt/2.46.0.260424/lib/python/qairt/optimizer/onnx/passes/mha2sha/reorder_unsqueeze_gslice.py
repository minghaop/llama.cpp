# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Transpose->GroupSlice) -> (GroupSlice->Transpose)
"""

import onnx
import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    convert_attr_to_py,
    get_constant_np,
    get_value_numeric_shape,
    is_constant,
)
from qairt.optimizer.utils.logger import logger


class _ReorderUnSqueezeGroupsliceOrSqueezeGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Unsqueeze(in_a) // or squeeze
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {

            a0,a1,a2... = GroupSlice(in_a)

            b0 = Unsqueeze(a0) // or squeeze
            b1 = Unsqueeze(a1) // or squeeze
            b2 = Unsqueeze(a2) // or squeeze
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Unsqueeze(in_a) // or squeeze
        }

    Also, encodings are updated
    """

    def __init__(self, supported_op_type: str):
        super().__init__()
        self.supported_op_type = supported_op_type

    @classmethod
    def get_output_axis_to_input_axis_map(cls, op_axes, output_rank):
        raise NotImplementedError()

    def get_op_axes(
        self, graph: ir.Graph, op_node: ir.Node, input_rank: int, output_rank: int
    ) -> tuple | None:
        if op_node.domain not in ["", "ai.onnx", "main"]:
            return None
        op_version = graph.opset_imports[op_node.domain]
        if not onnx.defs.has(op_node.op_type, op_version, op_node.domain):
            return None
        op_schema = onnx.defs.get_schema(op_node.op_type, op_version, op_node.domain)

        if "axes" in op_schema.attributes:
            # axes is the attributes of Squeeze/Unsqueeze (opset <= 11)
            if "axes" not in op_node.attributes:
                return None
            axes = op_node.attributes["axes"].as_ints()
        else:
            # axes is the second input of Squeeze/Unsqueeze (opset > 11)
            if not is_constant(op_node.inputs[1]):
                return None
            axes = get_constant_np(op_node.inputs[1]).tolist()

        # handle negative axes
        if op_node.op_type == "Squeeze":
            rank = input_rank
        elif op_node.op_type == "Unsqueeze":
            rank = output_rank
        else:
            assert False
        axes_list = list(axes)
        for i, axis in enumerate(axes_list):
            if axis < 0:
                axes_list[i] = axis + rank
        return tuple(axes_list)

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        if op_node.op_type != self.supported_op_type:
            return False

        check_static_shape_of_node_io(op_node)

        gslice_axis = convert_attr_to_py(gslice_node.attributes["axis"], "as_int")

        input_rank = len(get_value_numeric_shape(op_node.inputs[0]))
        output_rank = len(get_value_numeric_shape(op_node.outputs[0]))
        op_axes = self.get_op_axes(graph, op_node, input_rank, output_rank)

        if op_axes is None:
            return False

        axis_map = self.get_output_axis_to_input_axis_map(op_axes, output_rank)
        if gslice_axis not in axis_map:
            return False

        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        a_groupslice_attrs = get_gslice_attrs(gslice_node)
        output_rank = len(get_value_numeric_shape(op_node.outputs[0]))

        input_rank = len(get_value_numeric_shape(op_node.inputs[0]))
        op_axes = self.get_op_axes(graph, op_node, input_rank, output_rank)
        axis_map = self.get_output_axis_to_input_axis_map(op_axes, output_rank)
        a_groupslice_attrs.axis = axis_map[a_groupslice_attrs.axis]

        inputs_gslice_attrs = [a_groupslice_attrs]

        if not "axes" in op_node.attributes:
            # axes is the second input
            axes_groupslice_attrs = FullGroupSliceAttrs(
                a_groupslice_attrs.num_outputs(),
                a_groupslice_attrs.head_slice_ids,
                a_groupslice_attrs.batch_slice_ids,
            )
            inputs_gslice_attrs.append(axes_groupslice_attrs)

        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=inputs_gslice_attrs
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True


class ReorderUnsqueezeGroupslice(_ReorderUnSqueezeGroupsliceOrSqueezeGroupslice):
    """
    Transform subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Unsqueeze(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {

            a0,a1,a2... = GroupSlice(in_a)

            b0 = Unsqueeze(a0)
            b1 = Unsqueeze(a1)
            b2 = Unsqueeze(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Unsqueeze(in_a)
        }

    Also, encodings are updated
    """

    def __init__(self):
        super().__init__("Unsqueeze")

    @classmethod
    def get_output_axis_to_input_axis_map(cls, op_axes, output_rank):
        # for unsqueeze
        axis_map = {}  # key is the output axis id, value is the input axis id
        input_i = 0
        for output_i in range(output_rank):
            if output_i in op_axes:
                continue
            axis_map[output_i] = input_i
            input_i += 1
        return axis_map
