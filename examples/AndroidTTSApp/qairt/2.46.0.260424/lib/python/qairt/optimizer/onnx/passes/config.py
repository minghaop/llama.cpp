# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module re-exports all configuration classes for the ONNX optimizer passes.
"""

from qairt.optimizer.onnx.passes.adaptations.permute_kv_cache import PermuteKVCacheRewriter
from qairt.optimizer.onnx.passes.mha2sha.config import M2sStartPoint, MHA2SHAConfig
from qairt.optimizer.onnx.passes.shape_infer import ShapeInference
from qairt.optimizer.onnx.passes.splitters.config import LLMSplitterConfig

# Aliases for convenience
PermuteKVCacheConfig = PermuteKVCacheRewriter.Config
ShapeInferConfig = ShapeInference.Config

__all__ = [
    "M2sStartPoint",
    "MHA2SHAConfig",
    "PermuteKVCacheConfig",
    "ShapeInferConfig",
    "LLMSplitterConfig",
]
