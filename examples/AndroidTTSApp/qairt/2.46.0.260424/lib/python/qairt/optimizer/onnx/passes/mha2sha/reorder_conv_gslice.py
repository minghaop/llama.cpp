# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Conv->GroupSlice) -> (GroupSlice->Conv)
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    GroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape,
    check_static_shape_of_node_io,
    get_attribute_with_default,
    get_value_numeric_shape,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderConvGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a, in_b) --> c,c0,c1,c2,...
        {
            c = Conv(in_a, in_w)
            c0,c1,c2... = GroupSlice(c)

        }
    to:
        Subgraph(in_a, in_b) --> c,c0,c1,c2,...
        {

            a0,a1,a2... = GroupSlice(in_a) # for some case, no groupslice is required
            b0,b1,b2... = GroupSlice(in_b)

            c0 = Conv(a0, b0)
            c1 = Conv(a1, b1)
            c2 = Conv(a2, b2)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = Conv(in_a, in_b)
        }

    Also, encodings are updated
    """

    def get_kernel_shape(self, op_node):
        """
        Get kernel shape from op_node attribute. If not found, get it from weight
        """
        if "kernel_shape" in op_node.attributes:
            kernel_shape = op_node.attributes["kernel_shape"].as_ints()
        else:
            weight = op_node.inputs[1]
            check_static_shape(weight)
            kernel_shape = get_value_numeric_shape(weight)[2:]

        return kernel_shape

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if not is_reorderable_group_slice(node):
            return False

        assert node.inputs[0] is not None  # check for mypy, definitely true

        op_node = node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        if op_node.op_type != "Conv":
            return False

        check_static_shape_of_node_io(op_node)

        # check for mypy, definitely true
        assert op_node.outputs[0].shape is not None
        gslice_attrs = get_gslice_attrs(node)
        gslice_axis = gslice_attrs.axis
        reorderable: bool = False
        if gslice_axis in [0, 1]:
            # general case
            # in onnx definition, input's shape is defined as [N x C x D1 x D2 … x Dn]
            # so only N,C is generally slice-able
            reorderable = True
        else:
            # special case
            # conv kernel size should be 1 on slice axis
            kernel_shape = self.get_kernel_shape(op_node)
            spatial_axis = gslice_axis - 2
            reorderable = True
            if kernel_shape[spatial_axis] != 1:
                reorderable = False
            pads = get_attribute_with_default(op_node, "pads", [0] * (len(kernel_shape) * 2))
            if pads[spatial_axis * 2] != 0 or pads[spatial_axis * 2 + 1] != 0:
                reorderable = False
            strides = get_attribute_with_default(op_node, "strides", [1] * (len(kernel_shape)))
            if strides[spatial_axis] != 1:
                reorderable = False
            dilations = get_attribute_with_default(op_node, "dilations", [1] * (len(kernel_shape)))
            if dilations[spatial_axis] != 1:
                reorderable = False
        return reorderable

    def create_mini_pattern(
        self,  # pylint: disable=R0913,R0917
        graph: ir.Graph,
        origin_op: ir.Node,
        mini_inputs: list[ir.Value | None],
        head_slice_id,
        batch_slice_id,
        slice_i,  # pylint: disable=W0613
        custom_kwargs: dict | None = None,
    ) -> list[ir.Value | None]:
        mini_op_outputs = super().create_mini_pattern(
            graph, origin_op, mini_inputs, head_slice_id, batch_slice_id, slice_i, custom_kwargs
        )

        if mini_op_outputs[0] is not None:
            mini_op = mini_op_outputs[0].producer()
            if mini_op is not None and "group" in mini_op.attributes:
                # we need to fix op's group attribute here (since it was directly copied from origin_op)
                # by definition, X shape is (N x C x D1 x D2 … x Dn), W shape is (M x C/group x k1 x k2 x … x kn)
                # so group = X.shape[1] // W.shape[1]
                mini_op_group = (
                    get_value_numeric_shape(mini_inputs[0])[1] // get_value_numeric_shape(mini_inputs[1])[1]
                )
                mini_op.attributes["group"] = ir.AttrInt64("group", mini_op_group)
        return mini_op_outputs

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        gslice_attrs = get_gslice_attrs(gslice_node)
        gslice_axis = gslice_attrs.axis
        output_gslice_attrs = get_gslice_attrs(gslice_node)

        out_num = len(gslice_node.outputs)
        in_a = op_node.inputs[0]
        in_weight = op_node.inputs[1]

        in_bias = None
        if len(op_node.inputs) >= 3:
            in_bias = op_node.inputs[2]

        out_shape = get_value_numeric_shape(op_node.outputs[0])
        group = get_attribute_with_default(op_node, "group", 1)

        if gslice_axis == 0 or gslice_axis >= 2:
            # if axis==0, slice on batch dim,
            # if axis==1, special case, kernel size should be 1 on this axis (verified in match())
            # in both cases, slice directly on input of conv

            a_gslice_attrs = get_gslice_attrs(gslice_node)

            weight_gslice_attrs = GroupSliceAttrs(
                0,
                starts=[0] * out_num,
                ends=[get_value_numeric_shape(in_weight)[0]] * out_num,
                head_slice_ids=output_gslice_attrs.head_slice_ids,
                batch_slice_ids=output_gslice_attrs.batch_slice_ids,
            )

            inputs_gslice_attrs = [a_gslice_attrs, weight_gslice_attrs]
            if in_bias is not None:
                inputs_gslice_attrs.append(
                    GroupSliceAttrs(
                        0,
                        starts=[0] * out_num,
                        ends=[get_value_numeric_shape(in_bias)[0]] * out_num,
                        head_slice_ids=output_gslice_attrs.head_slice_ids,
                        batch_slice_ids=output_gslice_attrs.batch_slice_ids,
                    )
                )

        elif group == 1:  # gslice_axis == 1
            # slice on output channel
            # so the input data is not needed to be sliced
            # but we need to slice weight/bias

            a_gslice_attrs = GroupSliceAttrs(
                0,
                starts=[0] * out_num,
                ends=[get_value_numeric_shape(in_a)[0]] * out_num,
                head_slice_ids=output_gslice_attrs.head_slice_ids,
                batch_slice_ids=output_gslice_attrs.batch_slice_ids,
            )

            weight_gslice_attrs = get_gslice_attrs(gslice_node)
            weight_gslice_attrs.axis = 0
            inputs_gslice_attrs = [a_gslice_attrs, weight_gslice_attrs]
            if in_bias is not None:
                bias_gslice_attrs = get_gslice_attrs(gslice_node)
                bias_gslice_attrs.axis = 0
                inputs_gslice_attrs.append(bias_gslice_attrs)
        elif group == out_shape[1]:
            # special case, depthwise conv and slice on output channel
            # input/weight/bias should be all split correspondingly
            a_gslice_attrs = get_gslice_attrs(gslice_node)
            weight_gslice_attrs = get_gslice_attrs(gslice_node)
            weight_gslice_attrs.axis = 0
            inputs_gslice_attrs = [a_gslice_attrs, weight_gslice_attrs]
            if in_bias is not None:
                bias_gslice_attrs = get_gslice_attrs(gslice_node)
                bias_gslice_attrs.axis = 0
                inputs_gslice_attrs.append(bias_gslice_attrs)

        else:
            # for other group situation, it is not easy to write a general solution
            # advance analysis may be required, so todo
            return False

        _ = self.rewrite_based_on_gslice_attrs(
            graph, op_node, [gslice_node], inputs_gslice_attrs=inputs_gslice_attrs
        )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
