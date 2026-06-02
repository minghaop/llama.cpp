# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Configuration for IOShapeRewriter pass"""

from dataclasses import dataclass

from qairt.optimizer.onnx.passes.axis_denotation_infer.config import AxisDenotationConfig
from qairt.optimizer.onnx.passes.base import PassConfig


@dataclass
class IOShapeRewriterConfig(PassConfig):
    """
    Configuration for IOShapeRewriter pass

    Args:
        new_seq_length: New sequence length to apply. If None, uses original from graph.
        new_context_length: New context length to apply. If None, uses original from graph.
        axis_denotation_config: Configuration for axis denotation inference.
                              If None, uses default AxisDenotationConfig.

    Example:
        # Simple usage
        config = IOShapeRewriterConfig(
            new_seq_length=128,
            new_context_length=4096
        )

        # Advanced usage with custom seed rules
        from qairt.optimizer.onnx.passes.io_shape_rewriter import (
            IOShapeRewriterConfig,
            AxisDenotationConfig,
            AxisDenotationSeedRule,
        )
        from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation

        axis_config = AxisDenotationConfig(
            custom_seed_rules=[
                AxisDenotationSeedRule(
                    name_pattern=r"my_input",
                    denotations=[AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH]
                )
            ]
        )

        config = IOShapeRewriterConfig(
            new_seq_length=128,
            new_context_length=4096,
            axis_denotation_config=axis_config
        )
    """

    new_seq_length: int | None = None  # If None, don't change sequence length
    new_context_length: int | None = None  # If None, don't change context length
    axis_denotation_config: AxisDenotationConfig | None = None  # If None, use default seed rules
