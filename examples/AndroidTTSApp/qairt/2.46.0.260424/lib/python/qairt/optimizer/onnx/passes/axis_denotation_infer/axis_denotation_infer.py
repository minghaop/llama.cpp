# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Pass for inferring axis denotations for all tensors across the ONNX graph"""

from typing import TypeAlias

import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.axis_denotation_infer.axis_denotation_initializer import (
    AxisDenotationInitializer,
)
from qairt.optimizer.onnx.passes.axis_denotation_infer.config import AxisDenotationConfig
from qairt.optimizer.onnx.passes.axis_denotation_infer.propagation_passes import (
    get_registered_passes,
)
from qairt.optimizer.onnx.passes.axis_denotation_infer.utils import get_axis_denotations
from qairt.optimizer.onnx.passes.base import BasePass
from qairt.optimizer.utils.logger import logger


class AxisDenotationInference(BasePass):
    """
    Pass to infer axis denotations of all tensors in the graph

    This pass:
    1. Applies seed rules to bootstrap axis denotations for graph inputs
    2. Processes nodes in topological order, applying the appropriate propagation pass to each node
    3. Stores inferred denotations in tensor's metadata (via extra_info)
    """

    Config: TypeAlias = AxisDenotationConfig

    def __init__(self, config: Config | None = None):
        if config is None:
            config = AxisDenotationInference.Config()
        super().__init__()
        self.config: AxisDenotationInference.Config = config

    def apply(self, ctx: GraphContext) -> int:
        """
        Apply axis denotation inference to the graph

        Args:
            ctx: Graph context

        Returns:
            Number of nodes whose output denotations were inferred
        """
        # Apply seed rules to graph inputs
        AxisDenotationInitializer(self.config).apply(ctx)

        # Propagation passes for various op types — built from the registry so
        # new passes added with @register_pass are automatically included.
        propagation_passes = get_registered_passes()

        total_inferred = 0

        # Process nodes in topological order
        for node in ctx.graph_ir:
            for pass_instance in propagation_passes:
                if pass_instance.match(ctx.graph_ir, node):
                    if pass_instance.rewrite(ctx.graph_ir, node):
                        total_inferred += 1
                    break  # Only one pass should match per node
            else:
                # No pass matched. Warn if any input had denotations — propagation is broken here.
                if any(len(get_axis_denotations(inp)) > 0 for inp in node.inputs if inp is not None):
                    logger.warning(
                        "No axis denotation propagation rule for op '%s' (node '%s'). "
                        "Denotations will not propagate past this node; downstream tensors "
                        "may not be resized correctly.",
                        node.op_type,
                        node.name or "unnamed",
                    )

        return total_inferred
