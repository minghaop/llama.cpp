# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Adaptation passes for ONNX models
"""

from qairt.optimizer.onnx.passes.adaptations.extract_lora_alpha import LoraAlphaExtractor
from qairt.optimizer.onnx.passes.adaptations.permute_kv_cache import PermuteKVCacheRewriter

__all__ = [
    "LoraAlphaExtractor",
    "PermuteKVCacheRewriter",
]
