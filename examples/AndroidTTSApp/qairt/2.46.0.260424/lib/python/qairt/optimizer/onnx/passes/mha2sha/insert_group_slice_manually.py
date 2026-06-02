# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides M2sInsertGroupSliceManually pass for mha2sha ir modification
"""

import re
from dataclasses import dataclass

import onnx_ir as ir

from qairt.optimizer.onnx.passes.base import MatchInfoProtocol
from qairt.optimizer.onnx.passes.mha2sha.base_rewriter import M2sBasePass
from qairt.optimizer.onnx.passes.mha2sha.config import M2sStartPoint
from qairt.optimizer.onnx.passes.mha2sha.utils import (
    GroupSliceAttrs,
)
from qairt.optimizer.onnx.utils.utils import (
    check_static_shape,
    get_value_numeric_shape,
    iter_all_values,
)


class M2sInsertGroupSliceManually(M2sBasePass):
    """
    Pass to insert (GroupSlice -> Concat) manually, for attention patterns
    that don't follow the standard QKV MatMul pattern (Softmax -> MatMul pattern)

    Example - Start the splitting from  "past_(key|value)_(\d+)_out" in Samsung Gauss3 LLM model
    """

    @dataclass
    class MatchInfo(MatchInfoProtocol):
        """Match information for M2sInsertGroupSliceManually pass"""

        split_axis: int
        split_map: dict

    def __init__(self, m2s_additional_start_points: list[M2sStartPoint] | None = None):
        super().__init__()
        if m2s_additional_start_points is None:
            m2s_additional_start_points = []
        self.m2s_additional_start_points = m2s_additional_start_points

    def match_value(self, graph: ir.Graph, v: ir.Value) -> bool | MatchInfo:
        if v.name is None:
            return False

        for start_point in self.m2s_additional_start_points:
            # Get the naming prefix from the graph's extra_info
            if "extra_info" in graph.meta:
                naming_prefix = graph.meta["extra_info"].naming_policy.prefix
            else:
                naming_prefix = ""

            # Create regex pattern where prefix and .protect suffix appear together or neither do
            # Pattern matches: prefix/name_pattern.protect or name_pattern
            escaped_prefix = re.escape(naming_prefix)

            # Remove anchors: ^, $, \A, \Z, to use the name_prefix regex patten twice
            user_pattern = re.sub(r"^(?:\^|\\A)+|(?:\$|\\Z)+$", "", start_point.name_pattern)

            # If user passes in a pattern like "past_key_\d+_out|past_value_\d+_out"
            # grouping it can make '.protect' bind to all alternations instead of just the last one
            user_pattern = f"(?:{user_pattern})"

            name_pattern = rf"^(?:{user_pattern}|{escaped_prefix}/{user_pattern}\.protect)$"
            compiled_name_pattern = re.compile(name_pattern)

            if compiled_name_pattern.fullmatch(v.name):
                split_axis: int = start_point.split_axis
                split_map: dict = start_point.split_map if start_point.split_map is not None else {-1: 1}

                check_static_shape(v)

                # Validate that split_axis is within the valid range [0, rank-1]
                v_shape = get_value_numeric_shape(v)
                tensor_rank = len(v_shape)
                if split_axis < 0 or split_axis >= tensor_rank:
                    raise ValueError(
                        f"split_axis {split_axis} is out of range for tensor '{v.name}' "
                        f"with rank {tensor_rank}. Valid range is [0, {tensor_rank - 1}]."
                    )

                return M2sInsertGroupSliceManually.MatchInfo(split_axis=split_axis, split_map=split_map)

        return False

    def apply(self, ctx):
        count = 0
        for v in iter_all_values(ctx.graph_ir):
            match_res = self.match_value(ctx.graph_ir, v)
            if isinstance(match_res, MatchInfoProtocol):
                self.insert_groupslice_concat(ctx.graph_ir, v, match_res)
                count += 1
        return count

    def match(self, graph, node):
        raise NotImplementedError

    def rewrite(self, graph, node, match_info=None):
        raise NotImplementedError

    def insert_groupslice_concat(self, graph: ir.Graph, v: ir.Value, match_info: MatchInfoProtocol) -> bool:
        assert isinstance(match_info, M2sInsertGroupSliceManually.MatchInfo)  # For mypy
        v_shape = get_value_numeric_shape(v)

        head_axis = match_info.split_axis
        head_num = v_shape[head_axis]

        # Split on head axis
        head_gslice_attrs = GroupSliceAttrs(axis=head_axis)

        if head_num in match_info.split_map:
            out_head_size = match_info.split_map[head_num]
        elif -1 in match_info.split_map:
            out_head_size = match_info.split_map[-1]
        else:
            out_head_size = 1

        for i, start_i in enumerate(range(0, head_num, out_head_size)):
            end_i = min(start_i + out_head_size, head_num)
            head_gslice_attrs.starts.append(start_i)
            head_gslice_attrs.ends.append(end_i)
            head_gslice_attrs.head_slice_ids.append(i)
            head_gslice_attrs.batch_slice_ids.append(-1)

        node = v.producer()
        if node is not None:
            node_namehint = (node.name if node.name else "") + "/head_gslice_node"
        else:
            node_namehint = (v.name if v.name else "") + "/head_gslice_node"

        # insert group slice and concat
        _, _ = self.gslice_then_concat(graph, v, head_gslice_attrs, node_namehint)

        return True
