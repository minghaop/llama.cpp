# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
Splitter Pass Configuration Classes
This module contains configuration classes specific to model splitting passes.
"""

from dataclasses import dataclass

from qairt.optimizer.onnx.passes.base import PassConfig


@dataclass
class LLMSplitterConfig(PassConfig):
    """Configuration for LLMSplitter pass"""

    num_splits: int = 1
    """
    Number of splits to divide the model into.
    The model will be partitioned into this many subgraphs, with each
    subgraph containing a portion of the original model's layers
    """

    split_embedding: bool = False
    """
    Whether to split embedding layers into their own subgraph.
    If True, embedding layers will be extracted into a separate model,
    which can be useful for deployment scenarios where embeddings
    need special handling or caching.
    """

    split_lm_head: bool = False
    """
    Whether to split the lm_head into its own subgraph.
    """
