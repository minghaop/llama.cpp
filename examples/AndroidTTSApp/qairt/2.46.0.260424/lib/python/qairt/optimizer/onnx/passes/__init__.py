# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
ONNX optimizer passes

This module provides the base classes and main entry points for ONNX graph optimization.
Individual passes are available in their respective submodules.
"""

# Base classes for extension
# Adaptation passes (commonly used directly)
from qairt.optimizer.onnx.passes.adaptations.extract_lora_alpha import LoraAlphaExtractor
from qairt.optimizer.onnx.passes.adaptations.permute_kv_cache import PermuteKVCacheRewriter
from qairt.optimizer.onnx.passes.base import BasePass, BasePredicatePass, BaseTreeVisitor

# Layout optimization (commonly used directly)
from qairt.optimizer.onnx.passes.layout_opt.layout_opt import LayoutOptRewriter

# Main orchestration passes
from qairt.optimizer.onnx.passes.mha2sha.mha2sha_rewriter import MHA2SHARewriter
from qairt.optimizer.onnx.passes.protect_io import ProtectIO, UnprotectIO
from qairt.optimizer.onnx.passes.shape_infer.shape_infer import ShapeInference

__all__ = [
    # Base classes for extension
    "BasePass",
    "BasePredicatePass",
    "BaseTreeVisitor",
    # Main entry points
    "MHA2SHARewriter",
    "ShapeInference",
    "ProtectIO",
    "UnprotectIO"
    # Common adaptation passes
    "LoraAlphaExtractor",
    "PermuteKVCacheRewriter",
    # Layout optimization
    "LayoutOptRewriter",
]
