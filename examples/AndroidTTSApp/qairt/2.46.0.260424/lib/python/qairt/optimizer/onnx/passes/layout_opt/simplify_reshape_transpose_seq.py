# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides a pass to simplify the sequence of reshape/transpose ops
"""

import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass
from qairt.optimizer.onnx.utils.reshape_transpose_seq_utils import (
    ReshapeTransposeOpSeq,
    determine_seq_complexity,
    find_reshape_transpose_seq_bottom_up,
)
from qairt.optimizer.onnx.utils.utils import (
    is_used,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class SimplifyReshapeTransposeSeqRewriter(BasePass):
    """
    A graph rewriter pass that simplify reshape transpose consequences

    """

    def apply(self, ctx: GraphContext) -> int:
        # found all reshape/transpose seq and try to optimize them
        all_node = list(ctx.graph_ir)[:]
        optimized_out_nodes: set[ir.Node] = set()
        accept_types = set(["Transpose", "Reshape", "Squeeze", "Unsqueeze"])
        opt_count = 0
        # iterate from the end to the start
        for n in all_node[::-1]:  # pylint: disable=[too-many-nested-blocks]
            if n in optimized_out_nodes:
                continue
            if n.op_type not in accept_types:
                continue
            op_seq = find_reshape_transpose_seq_bottom_up(
                n.outputs[0], set(x.name for x in optimized_out_nodes if x.name is not None)
            )
            if len(op_seq.op_list) > 0:
                # check every non-empty list,
                # even 1-len sequences can be eliminated if possible

                # found a seq
                # try to optimize it
                if self.optimize_seq(ctx, op_seq):
                    opt_count += 1
                    # remove old seq
                    for x in op_seq.op_list[::-1]:
                        if not is_used(x.outputs[0]):
                            optimized_out_nodes.add(x)
                            for in_v_i in range(len(x.inputs)):
                                x.replace_input_with(in_v_i, None)

        return opt_count

    def optimize_seq(self, ctx: GraphContext, op_seq: ReshapeTransposeOpSeq):
        """
        Try to optimize the given transpose/reshape op sequence
        If possible, replace the op sequence with a simplified one

        Args:
            ctx: model context
            op_seq: the transpose/reshape op sequence
        Returns:
            True if the sequence is optimized, False otherwise
        """
        info_seq = op_seq.build_info_seq()
        if info_seq is None:
            return False

        new_info_seq = info_seq.simplify_seq()
        if info_seq == new_info_seq:
            return False

        origin_complexity = determine_seq_complexity(info_seq)
        opt_complexity = determine_seq_complexity(new_info_seq)

        if opt_complexity < origin_complexity:
            # rewrite the graph
            new_op_seq = ReshapeTransposeOpSeq.create_by_info_seq(
                ctx.graph_ir, op_seq.input_v, op_seq.v_extra_info, new_info_seq
            )
            if new_op_seq is None:
                return False
            safe_replace_all_uses_with(ctx.graph_ir, op_seq.output_v, new_op_seq.output_v)
            ctx.graph_ir.meta["extra_info"].record_copy(
                op_seq.output_v.name, new_op_seq.output_v.name, self.get_curr_pass_name()
            )

            logger.debug(
                "applied pass %s, transformed '%s' to '%s'",
                self.get_curr_pass_name(),
                info_seq.as_oneline_str(),
                new_info_seq.as_oneline_str(),
            )
            return True
        return False
