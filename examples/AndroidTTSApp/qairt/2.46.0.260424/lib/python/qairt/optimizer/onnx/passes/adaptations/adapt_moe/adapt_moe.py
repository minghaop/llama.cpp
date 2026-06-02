# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module is the entry point of the MoE transformation

"""

import os
import tempfile
from dataclasses import dataclass

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.adaptations.adapt_moe.adapt_moe_components import (
    AdaptMoEComponents,
    InlineInternalFunctions,
)
from qairt.optimizer.onnx.passes.adaptations.adapt_moe.extract_moe_components import (
    ExtractMoEComponents,
)
from qairt.optimizer.onnx.passes.base import BasePass, PassConfig
from qairt.optimizer.onnx.passes.cleaning import (
    DeadCodeRemovalRewriter,
    DeadFunctionRemovalRewriter,
    DeadWeightRemovalRewriter,
)
from qairt.optimizer.onnx.passes.protect_io import ProtectIO, UnprotectIO
from qairt.optimizer.onnx.passes.shape_infer import ShapeInference
from qairt.utils.loggers import get_logger

_logger = get_logger(__name__)


class AdaptMoE(BasePass):
    @dataclass
    class Config(PassConfig):
        overridden_subselection: int | None = None
        remove_op_predicate: bool = False

    def apply(self, ctx: GraphContext) -> int:
        assert isinstance(self.config, AdaptMoE.Config)

        ProtectIO().apply(ctx)  # ProtectIO is re-entrant now

        count = 0
        ExtractMoEComponents().apply(ctx)

        # ExtractMoEComponents will produce many unused nodes
        # remove them to speedup next passes
        DeadCodeRemovalRewriter().apply(ctx)
        DeadWeightRemovalRewriter().apply(ctx)

        count += AdaptMoEComponents(
            AdaptMoEComponents.Config(
                overridden_subselection=self.config.overridden_subselection,
                remove_op_predicate=self.config.remove_op_predicate,
            )
        ).apply(ctx)

        DeadCodeRemovalRewriter().apply(ctx)
        DeadWeightRemovalRewriter().apply(ctx)
        InlineInternalFunctions().apply(ctx)

        DeadFunctionRemovalRewriter().apply(ctx)
        ShapeInference().apply(ctx)

        UnprotectIO().apply(ctx)

        return count


def adapt_moe(
    ctx: GraphContext,
    *,
    overridden_subselection: int | None = None,
    remove_op_predicate: bool = False,
    enable_validation: bool = False,
) -> GraphContext:
    """High-level API for Mixture-of-Experts (MoE) model adaptation.

    Adapt a Mixture-of-Experts (MoE) ONNX model, in-place.

    Extracts and adapts the AR=N and AR=1 MoE components, inlines internal
    functions, removes dead code, and runs shape inference.

    When ``enable_validation`` is True, the adapted model is saved to a
    temporary directory and compared against the post-transform model using
    ONNX Runtime with random inputs. The temporary directory is cleaned up
    automatically after validation.

    Args:
        ctx: The model context to adapt.
        overridden_subselection: Override the number of experts selected per
            token. If ``None``, the value is inferred from the model.
        remove_op_predicate: Whether to remove the op-predicate ``Where`` ops
            (default ``False``).
        enable_validation: Whether to verify the transformed model against the
            original using ONNX Runtime. Defaults to False.

    Returns:
        The same ``GraphContext``, modified in-place.

    Usage::

        from qairt.optimizer.onnx import adapt_moe

        adapt_moe(ctx)
        adapt_moe(ctx, remove_op_predicate=True)
    """
    config = AdaptMoE.Config(
        overridden_subselection=overridden_subselection,
        remove_op_predicate=remove_op_predicate,
    )

    from qairt.optimizer.onnx.validation.ort_accuracy_checker import verify_onnx_with_random_inputs

    with tempfile.TemporaryDirectory(dir=os.environ.get("QAIRT_TMP_DIR", None)) as tmp:
        pre_transform_path = os.path.join(tmp, "pre_transform", "model.onnx")

        if enable_validation:
            _logger.info(
                "Validation enabled: saving pre-transform model for comparison. "
                "Requires a server with at least 64 GB RAM to avoid OOM."
            )
            ctx.save_onnx(pre_transform_path)

        AdaptMoE(config).apply(ctx)

        if enable_validation:
            _logger.info("Validating AdaptMoE transformation with ONNX Runtime...")
            post_transform_path = os.path.join(tmp, "post_transform", "model.onnx")
            ctx.save_onnx(post_transform_path)
            verify_onnx_with_random_inputs(pre_transform_path, post_transform_path)

    return ctx
