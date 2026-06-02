# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for reordering
(LayerNormalization->GroupSlice) -> (GroupSlice->LayerNormalization)
"""

import copy
from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    BroadcastHelper,
    FullGroupSliceAttrs,
    GroupSliceAttrs,
    get_gslice_attrs,
    get_value_numeric_shape,
    is_reorderable_group_slice,
)
from qairt.optimizer.onnx.utils.utils import check_static_shape_of_node_io
from qairt.optimizer.utils.logger import logger


class M2sReorderMatmulGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a, in_b) --> c,c0,c1,c2,...
        {
            c = Matmul(in_a, in_b)
            c0,c1,c2... = GroupSlice(c)

        }
    to:
        Subgraph(in_a, in_b) --> c,c0,c1,c2,...
        {

            a0,a1,a2... = GroupSlice(in_a) # for some case, no groupslice is required
            b0,b1,b2... = GroupSlice(in_b)

            c0 = Matmul(a0, b0)
            c1 = Matmul(a1, b1)
            c2 = Matmul(a2, b2)

            # if possible
            c = concat(c0,c1,c2,...)
            # or c = Matmul(in_a, in_b)
        }

    Also, encodings are updated
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for M2sReorderMatmulGroupslice pass"""

        matmul_node: ir.Node
        rank: int
        gslice_attrs: GroupSliceAttrs

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfo:
        if not is_reorderable_group_slice(node):
            return False
        assert node.inputs[0] is not None  # check for mypy, definitely true
        matmul_node = node.inputs[0].producer()
        assert matmul_node is not None  # check for mypy, definitely true
        if matmul_node.op_type != "MatMul":
            return False

        check_static_shape_of_node_io(matmul_node)
        # check for mypy, definitely true
        assert matmul_node.outputs[0] is not None
        # check for mypy, definitely true
        assert matmul_node.outputs[0].shape is not None

        rank = matmul_node.outputs[0].shape.rank()
        gslice_attrs = get_gslice_attrs(node)

        if (
            gslice_attrs.axis == rank - 1  # slice on last dim, so it can always be sliced.
            or gslice_attrs.axis == rank - 2  # slice on second last dim, so it can always be sliced.
            or gslice_attrs.axis < rank - 2
        ):  # slice on broadcastable dim, so we need to handle broadcast.
            return M2sReorderMatmulGroupslice.MatchInfo(
                matmul_node=matmul_node, rank=rank, gslice_attrs=gslice_attrs
            )

        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:
        assert isinstance(match_info, M2sReorderMatmulGroupslice.MatchInfo)  # For mypy
        gslice_node = node
        matmul_node = match_info.matmul_node
        gslice_attrs = match_info.gslice_attrs
        rank = match_info.rank

        in_a, in_b = matmul_node.inputs

        if gslice_attrs.axis == rank - 1:
            # slice on last dim, so it can always be sliced.
            a_groupslice_attrs_full = FullGroupSliceAttrs(
                gslice_attrs.num_outputs(),
                head_slice_ids=gslice_attrs.head_slice_ids,
                batch_slice_ids=gslice_attrs.batch_slice_ids,
            )
            b_groupslice_attrs = copy.deepcopy(gslice_attrs)
            b_groupslice_attrs.axis = len(get_value_numeric_shape(in_b)) - 1

            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                matmul_node,
                [gslice_node],
                inputs_gslice_attrs=[a_groupslice_attrs_full, b_groupslice_attrs],
            )
        elif gslice_attrs.axis == rank - 2:
            # slice on second last dim, so it can always be sliced.
            b_groupslice_attrs_full = FullGroupSliceAttrs(
                gslice_attrs.num_outputs(),
                head_slice_ids=gslice_attrs.head_slice_ids,
                batch_slice_ids=gslice_attrs.batch_slice_ids,
            )
            a_groupslice_attrs = copy.deepcopy(gslice_attrs)
            a_groupslice_attrs.axis = len(get_value_numeric_shape(in_a)) - 2

            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                matmul_node,
                [gslice_node],
                inputs_gslice_attrs=[a_groupslice_attrs, b_groupslice_attrs_full],
            )
        elif gslice_attrs.axis < rank - 2:
            bc_helper = BroadcastHelper(
                get_value_numeric_shape(in_a),
                get_value_numeric_shape(in_b),
                get_value_numeric_shape(matmul_node.outputs[0]),
                ignore_last_dim_num=2,
            )

            a_groupslice_attrs = bc_helper.get_input_group_attrs(0, gslice_attrs)
            b_groupslice_attrs = bc_helper.get_input_group_attrs(1, gslice_attrs)

            _ = self.rewrite_based_on_gslice_attrs(
                graph,
                matmul_node,
                [gslice_node],
                inputs_gslice_attrs=[a_groupslice_attrs, b_groupslice_attrs],
            )

        else:
            raise ValueError()

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), matmul_node.name)
        return True
