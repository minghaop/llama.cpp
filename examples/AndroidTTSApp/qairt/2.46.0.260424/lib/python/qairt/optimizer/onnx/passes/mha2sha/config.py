# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Configuration classes for MHA2SHA passes
"""

from dataclasses import dataclass, field

from qairt.optimizer.onnx.passes.base import PassConfig


@dataclass
class M2sStartPoint:
    """
    Configuration for custom starting points used in MHA2SHA transformation

    This dataclass defines where the MHA2SHA transformation should begin for attention patterns
    that don't follow the standard QKV MatMul pattern

    Example:
        For a model with past_key/past_value tensors that need custom MHA2SHA conversion:

        start_point = M2sStartPoint(
            name_pattern="past_(key|value)_(\d+)_out",
            split_axis=1,  # Head axis for 4D tensors
            split_map={32: 8, -1: 1}  # Optiona. 32 heads -> 8 heads, others -> 1 head
        )
    """

    name_pattern: str
    """
    Regular expression pattern to match tensor names where MHA2SHA conversion should start

    The pattern identifies tensors that require GroupSlice insertion after their producer nodes
    to enable multi-head to single-head attention conversion. This handles non-standard attention
    patterns or special cases like past_key/past_value tensors in models like Samsung Gauss3

    Example pattern:
        - "past_(key|value)_(\d+)_out" - Matches past key/value output tensors
    """

    split_axis: int
    """
    The head axis along which to split the multi-head attention into single heads

    This specifies which dimension contains the head information that should be split during
    the MHA2SHA conversion. Common values:
        - 0: Head axis for 3D tensors [heads, seq_len, head_dim]
        - 1: Head axis for 4D tensors [batch, heads, seq_len, head_dim]

    The choice depends on the tensor's shape and the model's attention architecture
    """

    split_map: dict[int, int] | None = None
    """
    Optional mapping from input head count to output head count for the MHA2SHA conversion

    Defines how many heads the multi-head attention should be converted into based on its
    current head count. If not provided, defaults to {-1: 1} (convert any multi-head
    attention into single-head attention).

    Format: {input_head_count: output_head_count}
    Special key -1 acts as a wildcard for any head count not explicitly specified.

    Examples:
        - {32: 8, 16: 4, -1: 1}: 32->8 heads, 16->4 heads, others->1 head
        - {-1: 1}: Convert any multi-head attention to single-head (default behavior)
        - {128: 8}: Only convert 128-head attention to 8 heads, leave others unchanged
    """


@dataclass
class MHA2SHAConfig(PassConfig):
    """Configuration for MHA2SHARewriter pass"""

    m2s_head_split_map: dict[int, int] = field(default_factory=dict)
    """
    Mapping for splitting multi-head attention to single-head attention
    Defines how to convert different head counts during MHA2SHA transformation.
    Format: {input_head_count: output_head_count}
    Examples:
        - {32: 1, 16: 1}: Convert 32-head and 16-head attention to single-head
        - {64: 8, 32: 4, -1: 1}: Progressive reduction, with -1 as wildcard
        - {}: Use default behavior (convert all to single-head)
    """

    m2s_additional_start_points: list[M2sStartPoint] = field(default_factory=list)
    """
    List of additional starting points for MHA2SHA conversion.
    Allows specification of custom tensor patterns where MHA2SHA conversion
    should begin, beyond the default QKV MatMul detection. Useful for
    non-standard attention architectures or special tensor patterns
    """

    out_batch_size: int | None = None
    """
    Output batch size for batch splitting (future feature).
    Currently not used but reserved for future batch splitting functionality.
    When implemented, this will control how batches are split during MHA2SHA conversion.
    """
