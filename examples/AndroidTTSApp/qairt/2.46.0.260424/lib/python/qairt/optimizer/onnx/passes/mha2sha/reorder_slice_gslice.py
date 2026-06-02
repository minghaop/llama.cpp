# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Slice->GroupSlice) -> (GroupSlice->Slice)
"""

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
    get_slice_static_params,
    is_constant,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderSliceGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Slice(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {

            a0,a1,a2... = GroupSlice(in_a)

            b0 = Slice(a0)
            b1 = Slice(a1)
            b2 = Slice(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Slice(in_a)
        }

    Also, encodings are updated
    """

    def _get_slice_and_gslice_axes(self, slice_node: ir.Node, gslice_node: ir.Node) -> tuple[list[int], int]:
        """
        Extract slice axes and gslice axis from the nodes.

        Args:
            slice_node: The Slice node
            gslice_node: The GroupSlice node

        Returns:
            Tuple of (slice_axes, gslice_axis)
        """
        slice_params = get_slice_static_params(slice_node)
        assert slice_params is not None  # already checked in match
        slice_axes = slice_params["axes"]
        gslice_axis = convert_attr_to_py(gslice_node.attributes["axis"], "as_int")
        return slice_axes, gslice_axis

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if not is_reorderable_group_slice(gslice_node):
            return False
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        if op_node is None:
            return False
        if op_node.op_type != "Slice":
            return False

        check_static_shape_of_node_io(op_node)
        if get_slice_static_params(op_node) is None:
            return False

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
        # Find the gslice node to get the axis
        gslice_node = None
        for user, _ in origin_op.outputs[0].uses():
            if user.op_type == "GroupSlice":
                gslice_node = user
                break

        assert gslice_node is not None
        slice_axes, gslice_axis = self._get_slice_and_gslice_axes(origin_op, gslice_node)

        if tuple(slice_axes) == (gslice_axis,):
            # no need to slice anymore
            return [mini_inputs[0]]

        return super().create_mini_pattern(
            graph, origin_op, mini_inputs, head_slice_id, batch_slice_id, slice_i, custom_kwargs
        )

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:  # pylint: disable=R0911,R0912
        gslice_node = node
        assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true

        slice_axes, gslice_axis = self._get_slice_and_gslice_axes(op_node, gslice_node)

        if gslice_axis not in slice_axes:
            out_gslice_attrs = get_gslice_attrs(gslice_node)
            inputs_gslice_attrs = [out_gslice_attrs]
            for _ in op_node.inputs[1:]:
                # full slices for other inputs (starts, ends, axes, steps)
                inputs_gslice_attrs.append(
                    FullGroupSliceAttrs(
                        out_gslice_attrs.num_outputs(),
                        head_slice_ids=[-1] * out_gslice_attrs.num_outputs(),
                        batch_slice_ids=[-1] * out_gslice_attrs.num_outputs(),
                    )
                )

            _ = self.rewrite_based_on_gslice_attrs(
                graph, op_node, [gslice_node], inputs_gslice_attrs=inputs_gslice_attrs
            )
        elif gslice_axis in slice_axes and len(slice_axes) > 1:
            # multi-slices
            # to simplify the code, let's firstly transform it into a sequnce of
            # - a slice that slice_axes don't include gslice_axis
            # - a slice that slice_axes has only gslice_axis
            # complexe and rare, support it in the future if required
            return False
        elif tuple(slice_axes) == (gslice_axis,):
            # gslice_axis in slice_axes
            # for example,
            #   Slice(start=0, end=384, axes=2)
            #   Groupslice(starts=[0,24],ends=[24,48], axes=2)

            if not is_constant(op_node.inputs[1]):
                return False
            if not is_constant(op_node.inputs[2]):
                return False
            if len(op_node.inputs) >= 5:
                steps_v = op_node.inputs[4]
                if not is_constant(steps_v):
                    return False
                steps_cst = get_constant_np(steps_v)[0]
                if steps_cst != 1:
                    return False
            else:
                steps_cst = 1

            slice_start_cst = int(get_constant_np(op_node.inputs[1])[0])
            slice_end_cst = int(get_constant_np(op_node.inputs[2])[0])

            out_gslice_attrs = get_gslice_attrs(gslice_node)
            data_gslice_attrs = out_gslice_attrs
            data_gslice_attrs.starts = [x + slice_start_cst for x in data_gslice_attrs.starts]
            data_gslice_attrs.ends = [x + slice_start_cst for x in data_gslice_attrs.ends]

            # check end
            for curr_end in data_gslice_attrs.ends:
                if curr_end > slice_end_cst:
                    return False

            inputs_gslice_attrs = [data_gslice_attrs]
            for _ in op_node.inputs[1:]:
                # full slices for other inputs (starts, ends, axes, steps)
                # not used actually (see self.create_mini_pattern)
                inputs_gslice_attrs.append(
                    FullGroupSliceAttrs(
                        out_gslice_attrs.num_outputs(),
                        head_slice_ids=[-1] * out_gslice_attrs.num_outputs(),
                        batch_slice_ids=[-1] * out_gslice_attrs.num_outputs(),
                    )
                )

            _ = self.rewrite_based_on_gslice_attrs(
                graph, op_node, [gslice_node], inputs_gslice_attrs=inputs_gslice_attrs
            )
        else:
            # unknown situation
            return False

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
