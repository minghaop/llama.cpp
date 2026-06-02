# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Configuration for axis denotation inference"""

from dataclasses import dataclass, field

from qairt.optimizer.onnx.passes.base import PassConfig
from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation


@dataclass
class AxisDenotationSeedRule:
    """
    Seed rule for bootstrapping axis denotations based on ONNX graph input name patterns

    This rule maps tensor name patterns to lists of axis denotations. When an ONNX graph
    input's name matches the pattern (using regex fullmatch), the specified denotations
    are assigned to its axes. This bootstraps the denotation inference process.

    How seed rules work:
    --------------------
    1. The name_pattern is compiled as a Python regex pattern
    2. If a graph input name matches the name pattern, the denotations is assigned to the tensor
    3. The length of denotations must match the tensor's rank

    Args:
        name_pattern: Regex pattern to match ONNX graph input names (case-insensitive)
                      Uses Python regex syntax. The pattern must match the entire name
        denotations: List of AxisDenotation values to assign to matching tensors
                     Length must equal the tensor's rank

    Example::

        # For a custom input named "my_input_0" with shape [batch, seq_len]:
        AxisDenotationSeedRule(
            name_pattern=r"my_input_\\d+",
            denotations=[AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH],
        )

        # For KV cache inputs with shape [batch, heads, past_seq, head_dim]:
        AxisDenotationSeedRule(
            name_pattern=r"past_key_\\d+_in",
            denotations=[
                AxisDenotation.BATCH,
                AxisDenotation.UNKNOWN,
                AxisDenotation.PAST_SEQ_LENGTH,
                AxisDenotation.UNKNOWN,
            ],
        )
    """

    name_pattern: str
    denotations: list[AxisDenotation]


@dataclass
class AxisDenotationConfig(PassConfig):
    """
    Configuration for axis denotation inference passes

    This config specifies regex patterns to match ONNX graph input names and determine
    what each dimension represents. It's used to bootstrap the denotation inference
    process by identifying known tensor patterns in LLM models

    How it works:
    -------------
    1. The name_pattern is compiled as a Python regex pattern
    2. If a graph input name matches the name pattern, the tensor is initialized with the denotations
    3. These initial denotations propagate through the graph via inference passes

    The config includes built-in patterns for common LLM inputs (input_ids,
    attention_mask, KV caches, etc.) and supports custom patterns for non-standard
    models via custom_seed_rules

    Pattern Matching Priority:
    --------------------------
    - Custom seed rules (custom_seed_rules) are checked FIRST
    - Rules are evaluated in the order they appear in the list
    - The FIRST matching rule is used (subsequent rules are skipped)
    - If no custom rule matches, built-in patterns are tried
    - This allows overriding built-in patterns or handling non-standard naming

    Example::

        # Basic usage with defaults
        config = AxisDenotationConfig()

        # Custom model with non-standard names
        config = AxisDenotationConfig(
            custom_seed_rules=[
                AxisDenotationSeedRule(
                    name_pattern=r"my_custom_input",
                    denotations=[AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH],
                ),
                AxisDenotationSeedRule(
                    name_pattern=r"my_cache_\\d+",
                    denotations=[
                        AxisDenotation.BATCH,
                        AxisDenotation.UNKNOWN,
                        AxisDenotation.PAST_SEQ_LENGTH,
                        AxisDenotation.UNKNOWN,
                    ],
                ),
            ],
        )

    Attributes:
        All pattern attributes use Python regex syntax and are case-insensitive
        Patterns must match the entire input name (fullmatch, not search)
    """

    input_ids_name_pattern: str = "input_ids"
    """
    input_ids_name_pattern would apply only to the first split in an LLM
    This input has the shape [BATCH, SEQ_LENGTH]
    """

    inputs_embeds_name_pattern: str = "(input|inputs)_embeds"
    """
    Naming pattern for pre-computed embedding inputs.

    Some LLMs accept pre-computed embeddings instead of token IDs, bypassing the
    embedding lookup (Gather) layer entirely. Both ``inputs_embeds`` and
    ``input_embeds`` naming conventions are found in practice.

    This input has shape [BATCH, SEQ_LENGTH, UNKNOWN] where the last dimension is
    the hidden/embedding dimension.
    """

    hidden_states_name_pattern: str = "hidden_states"
    """
    Naming pattern for hidden_states input used in speculative decoding techniques like EAGLE/EAGLET

    EAGLE/EAGLET (Extrapolation Algorithm for Greater Language-model Efficiency) is a speculative
    decoding technique that speeds up LLM inference by predicting future tokens directly from
    the "hidden states" of the target model. Instead of using a separate draft model, EAGLE uses
    a lightweight "EAGLE head" (small transformer layer) that takes hidden states from the main
    LLM to generate candidate tokens.

    This input has shape [BATCH, SEQ_LENGTH, UNKNOWN]
    """

    layer_output_name_pattern: str = "/?model_(layers_\\d+_Add/Add|embed_tokens/Gather)_output_0"
    """
    layer_output_name_pattern would apply only to splits apart from the first
    Matches:
    - Residual Add outputs: /model_layers_0_Add/Add_output_0
    - Embedding Gather outputs: /model/embed_tokens/Gather_output_0

    These inputs have shape [BATCH, SEQ_LENGTH, UNKNOWN]
    """

    position_ids_name_pattern: str = "(swa_)?position_ids(_sin|_cos)?"
    """
    This should cover both RoPE and absolute position encodings
    """

    key_cache_name_pattern: str = "past_key_(\\d)+_in"
    value_cache_name_pattern: str = "past_value_(\\d)+_in"
    """
    Naming pattern for KV cache input
    """

    attention_mask_name_pattern: str = "attention_mask"
    """
    Naming pattern for attention mask
    """

    cache_index_name_pattern: str = "(swa_)?cache_index"
    """
    Naming pattern for cache_index, including for Sliding Window Attention(SWA) models
    """

    # SWA-specific patterns
    swa_mask_name_pattern: str = "swa_attention_mask"
    """
    Naming pattern for attention mask for Sliding Window Attention(SWA) models
    """

    swa_key_name_pattern: str = "swa_key_(\\d)+_in"
    swa_value_name_pattern: str = "swa_value_(\\d)+_in"
    """
    Naming pattern for SWA KV cache
    """

    # Pre-quant adaptations
    transposed_key_cache: bool = True

    # Custom seed rules for user-specific patterns
    custom_seed_rules: list[AxisDenotationSeedRule] = field(default_factory=list)
    """
    Custom seed rules for models with non-standard tensor names
    Checked first before built-in patterns
    """
