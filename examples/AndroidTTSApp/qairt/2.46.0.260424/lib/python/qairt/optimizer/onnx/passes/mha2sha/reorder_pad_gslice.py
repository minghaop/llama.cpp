# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(Pad->GroupSlice) -> (GroupSlice->Pad)
"""

import copy

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    FullGroupSliceAttrs,
    get_gslice_attrs,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import (
    get_constant_np,
    get_value_numeric_shape,
    has_static_shape_on_value,
    is_constant,
)
from qairt.optimizer.utils.logger import logger


class M2sReorderPadGroupslice(M2sBasePass):
    """
    Reorder subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = Pad(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            a0,a1,a2... = GroupSlice(in_a)

            b0 = Pad(a0)
            b1 = Pad(a1)
            b2 = Pad(a2)
            ...

            # if possible
            b = concat(b0,b1,b2,...)
            # or b = Pad(in_a)
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

        if op_node.op_type != "Pad":
            return False

        assert op_node.inputs[0] is not None  # check for mypy, definitely true
        if not has_static_shape_on_value(op_node.inputs[0]):
            return False

        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None
        op_node = gslice_node.inputs[0].producer()
        assert op_node is not None  # check for mypy, definitely true
        out_gslice_attrs = get_gslice_attrs(gslice_node)

        in_shape = get_value_numeric_shape(op_node.inputs[0])
        pads_np = get_constant_np(op_node.inputs[1], use_infered_value=True)

        if len(op_node.inputs) > 3:
            pad_axes_v = op_node.inputs[3]
            if not is_constant(pad_axes_v, use_infered_value=True):
                return False
            pad_axes = get_constant_np(pad_axes_v, use_infered_value=True)

            # normalize pads_np to axes=list(range(in_rank))
            normalized_pads_np = np.zeros(len(in_shape) * 2, dtype=np.int64)
            for i, axis in enumerate(pad_axes):
                if axis < 0:
                    axis += len(in_shape)
                normalized_pads_np[axis] = pads_np[i]
                normalized_pads_np[axis + len(in_shape)] = pads_np[i + len(pad_axes)]
            pads_np = np.array(normalized_pads_np)

        assert len(pads_np) == len(in_shape) * 2, "pads should be a 1D tensor of shape [2*num_axes]"
        # we only care about the pads for gslice axis
        pad_begin = pads_np[out_gslice_attrs.axis]
        pad_end = pads_np[out_gslice_attrs.axis + len(in_shape)]

        # simple case
        if pad_begin == 0 and pad_end == 0:
            in_gslice_attrs = get_gslice_attrs(gslice_node)

            full_gslice_attrs = FullGroupSliceAttrs(
                in_gslice_attrs.num_outputs(),
                list(out_gslice_attrs.head_slice_ids),
                list(out_gslice_attrs.batch_slice_ids),
            )
            inputs_gslice_attrs = [in_gslice_attrs, full_gslice_attrs]

            if len(op_node.inputs) > 2:
                # for constant_value
                inputs_gslice_attrs.append(copy.deepcopy(full_gslice_attrs))
            if len(op_node.inputs) > 3:
                # for axes
                inputs_gslice_attrs.append(copy.deepcopy(full_gslice_attrs))

            _ = self.rewrite_based_on_gslice_attrs(
                graph, op_node, [gslice_node], inputs_gslice_attrs=inputs_gslice_attrs
            )
        else:
            return False

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)
        return True
