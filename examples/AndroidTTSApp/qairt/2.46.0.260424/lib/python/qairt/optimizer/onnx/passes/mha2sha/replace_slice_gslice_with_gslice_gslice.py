# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides the pass for replacing
(Slice->GroupSlice) -> (GroupSlice->GroupSlice)
"""

from dataclasses import dataclass

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.utils import GroupSliceAttrs, is_reorderable_group_slice
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape_of_node_io,
    get_slice_static_params,
    make_initializer,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class M2sReplaceSliceGroupslice2GroupsliceGroupslice(M2sBasePass):
    """
    Transform subgraph:
        Subgraph(in_a) --> b,b0,b1,b2
        {
            b = slice(in_a)
            b0,b1,b2... = GroupSlice(b)
        }
    Into:
        Subgraph(in_a) --> c,b0,b1,b2
        {
            c = GroupSlice(in_a)
            b0,b1,b2... = GroupSlice(c)
        }

    Also, encodings are updated
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for M2sReplaceSliceGroupslice2GroupsliceGroupslice pass"""

        op_node: ir.Node
        """The Slice node"""

        slice_params: dict
        """The slice parameters"""

        splittable_axis_idx: list[int]
        """List of splittable axis indices"""

    def match(self, graph: ir.Graph, node: ir.Node) -> bool | MatchInfo:
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
        if len(op_node.inputs) < 1 or op_node.inputs[0] is None:
            return False

        check_static_shape_of_node_io(op_node)

        slice_params = get_slice_static_params(op_node)
        if slice_params is None:
            return False

        splittable_axis_idx = []

        for i in range(len(slice_params["axes"])):
            if slice_params["steps"][i] != 1:
                continue
            for param in op_node.inputs[0].meta["extra_info"].splittable:
                if (
                    slice_params["axes"][i] == param.axis
                    and slice_params["starts"][i] >= param.start
                    and slice_params["ends"][i] <= param.end
                ):
                    splittable_axis_idx.append(i)

        if len(splittable_axis_idx) == 0:
            return False

        return M2sReplaceSliceGroupslice2GroupsliceGroupslice.MatchInfo(
            op_node=op_node, slice_params=slice_params, splittable_axis_idx=splittable_axis_idx
        )

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info: MatchInfoProtocol | None = None) -> bool:  # pylint: disable=R0911,R0912
        assert isinstance(match_info, M2sReplaceSliceGroupslice2GroupsliceGroupslice.MatchInfo)  # For mypy
        op_node = match_info.op_node
        slice_params = match_info.slice_params
        splittable_axis_idx = match_info.splittable_axis_idx

        current_v = op_node.inputs[0]
        if current_v is None:
            return False  # check for mypy

        for id in splittable_axis_idx:
            node_namehint = (op_node.name if op_node.name is not None else "") + f"/as_gslice_{id}"
            gslice_node = self._create_groupslice_node(
                graph,
                current_v,
                GroupSliceAttrs(
                    axis=slice_params["axes"][id],
                    starts=[slice_params["starts"][id]],
                    ends=[slice_params["ends"][id]],
                    head_slice_ids=[-1],
                    batch_slice_ids=[-1],
                ),
                node_namehint=node_namehint,
            )
            graph.insert_before(op_node, gslice_node)
            current_v = gslice_node.outputs[0]

        remain_slice_idx = [id for id in range(len(slice_params["axes"])) if id not in splittable_axis_idx]
        get_unique_name_with_suffix = lambda x, suffix: graph.meta["extra_info"].get_unique_name_with_suffix(
            x, suffix
        )
        if len(remain_slice_idx) > 0:
            new_slice_node = ir.Node(
                "",
                "Slice",
                [
                    current_v,
                    make_initializer(
                        graph,
                        get_unique_name_with_suffix(op_node.name, "/starts"),
                        np.array(slice_params["starts"][remain_slice_idx], dtype=np.int64),
                    ),
                    make_initializer(
                        graph,
                        get_unique_name_with_suffix(op_node.name, "/ends"),
                        np.array(slice_params["ends"][remain_slice_idx], dtype=np.int64),
                    ),
                    make_initializer(
                        graph,
                        get_unique_name_with_suffix(op_node.name, "/axes"),
                        np.array(slice_params["axes"][remain_slice_idx], dtype=np.int64),
                    ),
                    make_initializer(
                        graph,
                        get_unique_name_with_suffix(op_node.name, "/steps"),
                        np.array(slice_params["steps"][remain_slice_idx], dtype=np.int64),
                    ),
                ],
                name=get_unique_name_with_suffix(op_node.name, "/new_slice"),
            )
            new_slice_node.outputs[0].name = get_unique_name_with_suffix(op_node.name, "/new_outputs")
            current_v = new_slice_node.outputs[0]
            graph.insert_before(op_node, new_slice_node)

        self.mark_value_as_copy(graph, op_node.outputs[0], current_v)
        safe_replace_all_uses_with(graph, op_node.outputs[0], current_v)

        logger.debug("applied pass %s on '%s'", self.get_curr_pass_name(), op_node.name)

        return True
