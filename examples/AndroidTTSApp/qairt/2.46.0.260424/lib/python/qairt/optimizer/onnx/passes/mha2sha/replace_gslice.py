# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for replacing GroupSlice with a serial of Slice Op,
so that ourself defined GroupSlice will not exist anymore in the graph
"""

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import BasePredicatePass, MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import get_gslice_attrs
from qairt.optimizer.onnx.utils.utils import (
    get_constant_np,
    get_value_numeric_shape,
    make_initializer,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class M2sReplaceGroupSlice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(c) --> c0,c1,c2...
        {
            c0,c1,c2... = GroupSlice(c)
        }
    Into:
        Subgraph(c) --> c0,c1,c2...
        {
            c0 = Slice(c)
            c1 = Slice(c)
            c2 = Slice(c)
        }

    Broadcasting will be automatically updated
    Also, encodings are updated
    """

    def __init__(self, no_validation=False):
        """
        Args:
            no_validation: if True, no validation will be performed (should be only used in unit testing)
        """
        super().__init__()
        self.no_validation = no_validation

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        gslice_node = node
        if gslice_node.op_type == "GroupSlice":
            return True
        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        gslice_node = node
        # check for mypy, definitely true
        assert gslice_node.inputs[0] is not None

        gslice_attrs = get_gslice_attrs(gslice_node)

        # slice node may be already created
        # key is (axis,start, end), value is the slice node
        already_created_slices = {}
        for user, _ in gslice_node.inputs[0].uses():
            if user.op_type == "Slice" and len(user.inputs) == 5:
                axis = get_constant_np(user.inputs[3])[0]
                start = get_constant_np(user.inputs[1])[0]
                end = get_constant_np(user.inputs[2])[0]
                already_created_slices[(axis, start, end)] = user

        # create slice nodes

        for slice_i in range(gslice_attrs.num_outputs()):
            start = gslice_attrs.starts[slice_i]
            end = gslice_attrs.ends[slice_i]
            axis = gslice_attrs.axis

            if start == 0 and end == get_value_numeric_shape(gslice_node.inputs[0])[axis]:
                # no need to slice
                safe_replace_all_uses_with(graph, gslice_node.outputs[slice_i], gslice_node.inputs[0])
            elif (axis, start, end) in already_created_slices:
                safe_replace_all_uses_with(
                    graph,
                    gslice_node.outputs[slice_i],
                    already_created_slices[(axis, start, end)].outputs[0],
                )
            else:
                slice_n = ir.Node(
                    "",
                    op_type="Slice",
                    inputs=[
                        gslice_node.inputs[0],
                        make_initializer(
                            graph,
                            graph.meta["extra_info"].get_unique_name("slice_start"),
                            np.array([gslice_attrs.starts[slice_i]], dtype=np.int64),
                        ),
                        make_initializer(
                            graph,
                            graph.meta["extra_info"].get_unique_name("slice_end"),
                            np.array([gslice_attrs.ends[slice_i]], dtype=np.int64),
                        ),
                        make_initializer(
                            graph,
                            graph.meta["extra_info"].get_unique_name("slice_axis"),
                            np.array([gslice_attrs.axis], dtype=np.int64),
                        ),
                        make_initializer(
                            graph,
                            graph.meta["extra_info"].get_unique_name("slice_step"),
                            np.array([1], dtype=np.int64),
                        ),
                    ],
                    name=graph.meta["extra_info"].get_unique_name(gslice_node.name),
                )
                safe_replace_all_uses_with(graph, gslice_node.outputs[slice_i], slice_n.outputs[0])
                graph.insert_after(gslice_node, slice_n)
                self.mark_value_as_copy(graph, gslice_node.outputs[slice_i], slice_n.outputs[0])

                already_created_slices[(axis, start, end)] = slice_n

                if not self.no_validation:
                    # ideally, for MHA2SHA transformation,
                    # there are should only some gslice nodes left for inputs
                    if gslice_node.inputs[0] not in graph.inputs:
                        logger.warning(
                            "applied %s with non-inputs/non-weights value '%s'"
                            + "the MHA2SHA transformation may not be perfect, "
                            + "please verify the output graph manually",
                            self.get_curr_pass_name(),
                            gslice_node.inputs[0].name,
                        )
                    else:
                        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), gslice_node.name)
                else:
                    logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), gslice_node.name)

        graph.remove(gslice_node)

        return True
