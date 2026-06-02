# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides M2sFoldInitGroupSlice pass for mha2sha ir modification
"""

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import get_gslice_attrs
from qairt.optimizer.onnx.utils.utils import (
    get_constant_np,
    make_initializer,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class M2sFoldInitGroupSlice(M2sBasePass):
    """
    Pass to fold (initializer -> GroupSlice) into (sliced_initializers)
    """

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if gslice_node.op_type != "GroupSlice":
            return False
        gslice_in_v = gslice_node.inputs[0]
        assert gslice_in_v is not None  # check for mypy
        if gslice_in_v.const_value is not None:
            return True

        producer = gslice_in_v.producer()
        if producer is not None and producer.op_type == "Constant":
            return True
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        gslice_attrs = get_gslice_attrs(gslice_node)
        gslice_in_v = gslice_node.inputs[0]
        in_value = get_constant_np(gslice_in_v)
        out_np_values = []
        for slice_i in range(gslice_attrs.num_outputs()):
            np_slices = [slice(None, None, None)] * len(in_value.shape)
            np_slices[gslice_attrs.axis] = slice(gslice_attrs.starts[slice_i], gslice_attrs.ends[slice_i])
            out_np_values.append(
                in_value.__getitem__(tuple(np_slices))  # pylint: disable=C2801
            )

        for slice_i in range(gslice_attrs.num_outputs()):
            new_out_v = make_initializer(
                graph,
                self.get_mini_tensor_name(
                    graph,
                    gslice_node.outputs[slice_i].name,
                    gslice_attrs.head_slice_ids[slice_i],
                    gslice_attrs.batch_slice_ids[slice_i],
                ),
                out_np_values[slice_i],
            )

            safe_replace_all_uses_with(graph, gslice_node.outputs[slice_i], new_out_v)
            assert gslice_node.inputs[0] is not None  # check for mypy, definitely true
            self.mark_value_as_slice(
                graph,
                gslice_node.inputs[0],
                new_out_v,
                gslice_attrs.axis,
                gslice_attrs.starts[slice_i],
                gslice_attrs.ends[slice_i],
                gslice_attrs.head_slice_ids[slice_i],
                gslice_attrs.batch_slice_ids[slice_i],
            )

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), gslice_node.name)
        graph.remove(gslice_node)

        return True
