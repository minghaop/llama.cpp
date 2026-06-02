# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides passes to protect/unprotect the input/output names of the graph
"""

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass


class ProtectIO(BasePass):
    """
    Rename all outputs of the graph to protect them

    Renames all outputs with a .protect suffix and stores original names in graph metadata
    Use UnprotectIO to restore original names
    """

    META_INFO_KEY_NAME = "ProtectIO.original_output_names.stack"

    def apply(self, ctx: GraphContext) -> int:
        """
        Rename all outputs of the graph to protect them
        Stores original names in graph metadata
        """
        graph = ctx.graph_ir
        origin_output_names = []

        for v in graph.outputs:
            new_name = graph.meta["extra_info"].get_unique_name_with_suffix(v.name, ".protect")
            origin_output_names.append(v.name)
            graph.meta["extra_info"].record_copy(v.name, new_name, self.get_curr_pass_name())
            v.name = new_name

        # Store original names in graph metadata for UnprotectIO to use
        if self.META_INFO_KEY_NAME not in graph.meta:
            graph.meta[self.META_INFO_KEY_NAME] = []
        graph.meta[self.META_INFO_KEY_NAME].append(origin_output_names)

        # NOTE:
        #   graph.meta[self.META_INFO_KEY_NAME] is treated as a stack.
        #   - Each time ProtectIO is applied, the current graph output names
        #     are pushed onto the stack.
        #   - Each time UnprotectIO is applied, the stack is popped.
        #
        # This makes ProtectIO re-entrant, allowing calls like the following:
        #
        #   ProtectIO().apply()
        #   Xxx1Pass().apply()
        #   ProtectIO().apply()
        #   Xxx2Pass().apply()
        #   UnprotectIO().apply()
        #   Xxx3Pass().apply()
        #   UnprotectIO().apply()

        return len(origin_output_names)


class UnprotectIO(BasePass):
    """
    Pass to restore original graph IO names after protection

    Retrieves original names from graph metadata and restores them
    Must be used after ProtectIO
    """

    def apply(self, ctx: GraphContext) -> int:
        """
        Rename all outputs of the graph to their original names
        Retrieves original names from graph metadata
        """
        graph = ctx.graph_ir

        # Retrieve original names from graph metadata
        if ProtectIO.META_INFO_KEY_NAME not in graph.meta:
            raise RuntimeError("Must call ProtectIO before UnprotectIO")

        origin_output_names = graph.meta[ProtectIO.META_INFO_KEY_NAME].pop(-1)

        if len(graph.outputs) != len(origin_output_names):
            raise RuntimeError(
                f"Mismatch between current outputs ({len(graph.outputs)}) "
                f"and original protected names ({len(origin_output_names)}). "
                "Graph structure may have been modified after protection."
            )

        for i, v in enumerate(graph.outputs):
            original_name = origin_output_names[i]
            graph.meta["extra_info"].record_copy(v.name, original_name, self.get_curr_pass_name())
            v.name = original_name

        return len(origin_output_names)
