# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Initialize axis denotations for graph inputs using pattern-based seed rules

This pass applies seed rules to match graph input names against known patterns
(e.g., input_ids, attention_mask, past_key_*) and assigns initial axis denotation
values (e.g., BATCH, SEQ_LENGTH, CONTEXT_LENGTH). These denotations serve as the
starting point for propagation passes that infer denotations for all tensors in
the graph
"""

import re
from textwrap import dedent

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.axis_denotation_infer.config import AxisDenotationConfig
from qairt.optimizer.onnx.passes.axis_denotation_infer.utils import set_axis_denotations
from qairt.optimizer.onnx.passes.base import BasePass
from qairt.optimizer.onnx.passes.utilities.compute_seq_and_context_length import ComputeSeqAndContextLength
from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation
from qairt.optimizer.utils.logger import logger


class AxisDenotationInitializer(BasePass):
    def __init__(self, config: AxisDenotationConfig | None = None):
        if config is None:
            config = AxisDenotationConfig()
        super().__init__()
        self.config: AxisDenotationConfig = config

        # Compile regex patterns
        self.input_ids_pattern = re.compile(self.config.input_ids_name_pattern, re.IGNORECASE)
        self.inputs_embeds_pattern = re.compile(self.config.inputs_embeds_name_pattern, re.IGNORECASE)
        self.hidden_states_pattern = re.compile(self.config.hidden_states_name_pattern, re.IGNORECASE)
        self.layer_output_pattern = re.compile(self.config.layer_output_name_pattern, re.IGNORECASE)
        self.position_ids_pattern = re.compile(self.config.position_ids_name_pattern, re.IGNORECASE)
        self.key_cache_pattern = re.compile(self.config.key_cache_name_pattern, re.IGNORECASE)
        self.value_cache_pattern = re.compile(self.config.value_cache_name_pattern, re.IGNORECASE)
        self.attention_mask_pattern = re.compile(self.config.attention_mask_name_pattern, re.IGNORECASE)
        self.cache_index_pattern = re.compile(self.config.cache_index_name_pattern, re.IGNORECASE)
        self.swa_mask_pattern = re.compile(self.config.swa_mask_name_pattern, re.IGNORECASE)
        self.swa_key_pattern = re.compile(self.config.swa_key_name_pattern, re.IGNORECASE)
        self.swa_value_pattern = re.compile(self.config.swa_value_name_pattern, re.IGNORECASE)

        # Compile custom seed rule patterns
        self.custom_rules = []
        for rule in self.config.custom_seed_rules:
            compiled_pattern = re.compile(rule.name_pattern, re.IGNORECASE)
            self.custom_rules.append((compiled_pattern, rule.denotations))

    def apply(self, ctx: GraphContext):
        """
        Apply high-confidence seed rules to bootstrap axis denotations for graph inputs.

        Examples:
        - input_ids -> [BATCH, SEQ_LENGTH]
        - hidden_states -> [BATCH, SEQ_LENGTH, UNKNOWN]
        - position_ids_cos/sin -> [BATCH, UNKNOWN, SEQ_LENGTH, UNKNOWN]
        - past_key_\d+_in:
            -> [BATCH, UNKNOWN, UNKNOWN, PAST_SEQ_LENGTH] if transposed_key_cache is True
            -> [BATCH, UNKNOWN, PAST_SEQ_LENGTH, UNKNOWN] if transposed_key_cache is False
        - past_value_\d+_in → [BATCH, UNKNOWN, PAST_SEQ_LENGTH, UNKNOWN]
        - attention_mask → [BATCH, UNKNOWN, UNKNOWN, CONTEXT_LENGTH]
        ...
        """
        # Step 1: Ensure lengths are computed (context_length, seq_length, sliding_context_length)
        if "context_length" not in ctx.graph_ir.meta or "seq_length" not in ctx.graph_ir.meta:
            compute_config = ComputeSeqAndContextLength.Config(
                attention_mask_name_pattern=self.config.attention_mask_name_pattern,
                swa_mask_name_pattern=self.config.swa_mask_name_pattern,
                input_ids_name_pattern=self.config.input_ids_name_pattern,
                inputs_embeds_name_pattern=self.config.inputs_embeds_name_pattern,
                layer_output_name_pattern=self.config.layer_output_name_pattern,
            )
            ComputeSeqAndContextLength(compute_config).apply(ctx)

        graph = ctx.graph_ir

        # Get computed lengths for shape-based detection
        seq_length = graph.meta.get("seq_length")
        context_length = graph.meta.get("context_length")
        sliding_context_length = graph.meta.get("sliding_context_length")

        for input_tensor in graph.inputs:
            name = input_tensor.name

            assert name is not None

            assert seq_length is not None
            assert context_length is not None

            assert input_tensor.shape is not None
            input_shape = input_tensor.shape

            rank = len(input_shape)

            denotations = []

            # Try custom seed rules first (user-specified patterns take precedence)
            for pattern, rule_denotations in self.custom_rules:
                if pattern.fullmatch(name):
                    denotations = rule_denotations
                    break

            if denotations:
                pass  # Custom rule matched

            # If no custom rule matched, try built-in rules
            elif self.input_ids_pattern.fullmatch(name):
                denotations = [AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH]

            # Rule: inputs_embeds -> [BATCH, SEQ_LENGTH, UNKNOWN]
            # Pre-computed embeddings bypassing the embedding lookup layer.
            # Covers both "inputs_embeds" and "input_embeds" naming conventions.
            elif self.inputs_embeds_pattern.fullmatch(name):
                if rank == 3:
                    denotations = [AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH, AxisDenotation.UNKNOWN]

            # Rule: hidden_states -> [BATCH, SEQ_LENGTH, UNKNOWN]
            # Used in speculative decoding techniques like EAGLE/EAGLET where hidden states from
            # the target model are passed to a lightweight draft head to predict future tokens
            elif self.hidden_states_pattern.fullmatch(name):
                denotations = [AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH, AxisDenotation.UNKNOWN]

            # Rule: layer_output -> [BATCH, SEQ_LENGTH, UNKNOWN]
            elif self.layer_output_pattern.fullmatch(name):
                denotations = [AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH, AxisDenotation.UNKNOWN]

            # Rule:
            # * position_ids_cos/sin -> [BATCH, UNKNOWN, SEQ_LENGTH, UNKNOWN]
            # * Regular position_ids -> [BATCH, SEQ_LENGTH]
            # * Alibi position_ids -> [SEQ_LENGTH, CONTEXT_LENGTH]
            elif self.position_ids_pattern.fullmatch(name):
                if rank == 4:
                    denotations = [
                        AxisDenotation.BATCH,
                        AxisDenotation.UNKNOWN,
                        AxisDenotation.SEQ_LENGTH,
                        AxisDenotation.UNKNOWN,
                    ]
                elif rank == 2:
                    skip_op_types = {"Reshape", "Cast"}  # Regular position ids
                    while input_tensor.consumers() and input_tensor.consumers()[0].op_type in skip_op_types:
                        input_tensor = input_tensor.consumers()[0].outputs[0]

                    if input_tensor.consumers() and input_tensor.consumers()[0].op_type == "Gather":
                        denotations = [AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH]
                    else:
                        # Alibi position embeddings
                        denotations = [AxisDenotation.SEQ_LENGTH, AxisDenotation.CONTEXT_LENGTH]

            # Rule: swa_key_in -> [BATCH, UNKNOWN, UNKNOWN, SLIDING_CONTEXT_LENGTH] or [BATCH, UNKNOWN, SLIDING_CONTEXT_LENGTH, UNKNOWN]
            elif self.swa_key_pattern.fullmatch(name):
                if rank == 4:
                    if self.config.transposed_key_cache:
                        denotations = [
                            AxisDenotation.BATCH,
                            AxisDenotation.UNKNOWN,
                            AxisDenotation.UNKNOWN,
                            AxisDenotation.SLIDING_CONTEXT_LENGTH,
                        ]
                    else:
                        denotations = [
                            AxisDenotation.BATCH,
                            AxisDenotation.UNKNOWN,
                            AxisDenotation.SLIDING_CONTEXT_LENGTH,
                            AxisDenotation.UNKNOWN,
                        ]

            # Rule: swa_value_in -> [BATCH, UNKNOWN, SLIDING_CONTEXT_LENGTH, UNKNOWN]
            elif self.swa_value_pattern.fullmatch(name):
                if rank >= 4:
                    denotations = [
                        AxisDenotation.BATCH,
                        AxisDenotation.UNKNOWN,
                        AxisDenotation.SLIDING_CONTEXT_LENGTH,
                        AxisDenotation.UNKNOWN,
                    ]
                    denotations.extend([AxisDenotation.UNKNOWN] * (rank - 4))

            # Rule: past_key -> [BATCH, UNKNOWN, UNKNOWN, PAST_SEQ_LENGTH or CONTEXT_LENGTH or SLIDING_CONTEXT_LENGTH]
            #                or [BATCH, UNKNOWN, PAST_SEQ_LENGTH or CONTEXT_LENGTH or SLIDING_CONTEXT_LENGTH, UNKNOWN]
            # Shape-based detection: compare cache dimension against known lengths
            elif self.key_cache_pattern.fullmatch(name):
                if rank == 4:
                    # Get cache dimension based on transposed_key_cache setting
                    cache_dim_idx = 3 if self.config.transposed_key_cache else 2
                    cache_dim = input_shape[cache_dim_idx]

                    # Shape-based detection: compare against known lengths
                    if cache_dim == sliding_context_length:
                        cl_denotation = AxisDenotation.SLIDING_CONTEXT_LENGTH
                    elif cache_dim == context_length:
                        cl_denotation = AxisDenotation.CONTEXT_LENGTH
                    elif cache_dim == context_length - seq_length:
                        cl_denotation = AxisDenotation.PAST_SEQ_LENGTH
                    else:
                        raise ValueError(
                            dedent(
                                f"""
                                Could not determine axis denotation for input '{name}' with shape {list(input_shape)}:
                                  With transposed_key_cache={self.config.transposed_key_cache}, at axis {cache_dim_idx}
                                  Expected one of:
                                    - sliding_context_length: {sliding_context_length}
                                    - context_length: {context_length}
                                    - past_seq_length: {context_length - seq_length}
                                  But found: {cache_dim}
                                """
                            )
                        )

                    if self.config.transposed_key_cache:
                        denotations = [
                            AxisDenotation.BATCH,
                            AxisDenotation.UNKNOWN,
                            AxisDenotation.UNKNOWN,
                            cl_denotation,
                        ]
                    else:
                        denotations = [
                            AxisDenotation.BATCH,
                            AxisDenotation.UNKNOWN,
                            cl_denotation,
                            AxisDenotation.UNKNOWN,
                        ]

            # Rule: past_value -> [BATCH, UNKNOWN, PAST_SEQ_LENGTH or CONTEXT_LENGTH or SLIDING_CONTEXT_LENGTH, UNKNOWN]
            # Shape-based detection: compare cache dimension against known lengths
            elif self.value_cache_pattern.fullmatch(name):
                if rank == 4:
                    # Value cache dimension is always at index 2
                    cache_dim = input_shape[2]

                    # Shape-based detection: compare against known lengths
                    if sliding_context_length and cache_dim == sliding_context_length:
                        cl_denotation = AxisDenotation.SLIDING_CONTEXT_LENGTH
                    elif context_length and cache_dim == context_length:
                        cl_denotation = AxisDenotation.CONTEXT_LENGTH
                    elif cache_dim == context_length - seq_length:
                        cl_denotation = AxisDenotation.PAST_SEQ_LENGTH
                    else:
                        raise ValueError(
                            dedent(
                                f"""
                                Could not determine axis denotation for input '{name}' with shape {list(input_shape)}:
                                  At axis 2
                                  Expected one of:
                                    - sliding_context_length: {sliding_context_length}
                                    - context_length: {context_length}
                                    - past_seq_length: {context_length - seq_length}
                                  But found: {cache_dim}
                                """
                            )
                        )

                    denotations = [
                        AxisDenotation.BATCH,
                        AxisDenotation.UNKNOWN,
                        cl_denotation,
                        AxisDenotation.UNKNOWN,
                    ]

            # Rule: cache_index followed by Add -> Add output gets [SEQ_LENGTH]
            # Note: We do NOT set denotations on the cache_index input itself (it's a scalar index)
            # We only set denotations on the Add output and its constant input
            elif self.cache_index_pattern.fullmatch(name):
                if input_tensor.consumers() and input_tensor.consumers()[0].op_type == "Add":
                    add_node = input_tensor.consumers()[0]
                    if add_node.outputs[0]:
                        output = add_node.outputs[0]
                        set_axis_denotations(output, [AxisDenotation.SEQ_LENGTH])

                    constant_input = add_node.inputs[1]
                    if constant_input:
                        set_axis_denotations(constant_input, [AxisDenotation.SEQ_LENGTH])

                # Skip setting denotations on the input itself - continue to next input
                continue

            # Rule: attention_mask -> [BATCH, UNKNOWN, SEQ_LENGTH, CONTEXT_LENGTH]
            elif self.attention_mask_pattern.fullmatch(name):
                if rank == 2:
                    denotations = [AxisDenotation.BATCH, AxisDenotation.CONTEXT_LENGTH]
                elif rank == 3:
                    denotations = [
                        AxisDenotation.BATCH,
                        AxisDenotation.SEQ_LENGTH,
                        AxisDenotation.CONTEXT_LENGTH,
                    ]
                elif rank == 4:
                    denotations = [
                        AxisDenotation.BATCH,
                        AxisDenotation.UNKNOWN,
                        AxisDenotation.SEQ_LENGTH,
                        AxisDenotation.CONTEXT_LENGTH,
                    ]

            # Rule: attention_mask -> [BATCH, UNKNOWN, SEQ_LENGTH, SLIDING_CONTEXT_LENGTH]
            elif self.swa_mask_pattern.fullmatch(name):
                denotations = [
                    AxisDenotation.BATCH,
                    AxisDenotation.UNKNOWN,
                    AxisDenotation.SEQ_LENGTH,
                    AxisDenotation.SLIDING_CONTEXT_LENGTH,
                ]

            # Validate denotations if a rule matched
            if denotations:
                # Check if denotation count matches tensor rank
                if len(denotations) != rank:
                    raise ValueError(
                        dedent(
                            f"""
                            Rank mismatch for graph input '{name}':
                              Tensor rank: {rank}
                              Denotations provided: {denotations} (length {len(denotations)})

                            The number of axis denotations must match the tensor rank.
                            To fix this, add a custom seed rule with the correct denotations:

                            Example:
                              from qairt.optimizer.onnx.passes.axis_denotation_infer import (
                                  AxisDenotationSeedRule,
                                  AxisDenotationInference,
                              )
                              from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation

                              config = AxisDenotationInference.Config(
                                  custom_seed_rules=[
                                      AxisDenotationSeedRule(
                                          name_pattern=r"{name}",
                                          denotations=[AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH, ...]  # {rank} denotations total
                                      )
                                  ]
                              )
                            """
                        )
                    )
                set_axis_denotations(input_tensor, denotations)
                # Log successful initialization with full enum names
                denotation_str = "[" + ", ".join(d.name for d in denotations) + "]"
                logger.info(f"Initialized '{name}': {denotation_str}")
            else:
                # No rule matched
                raise ValueError(
                    dedent(
                        f"""
                        Could not match graph input '{name}' (rank {rank}) to any built-in or custom seed rule pattern.
                        To fix this, add a custom seed rule to AxisDenotationInference.Config:

                        Example:
                          from qairt.optimizer.onnx.passes.axis_denotation_infer import (
                              AxisDenotationSeedRule,
                              AxisDenotationInference,
                          )
                          from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation

                          config = AxisDenotationInference.Config(
                              custom_seed_rules=[
                                  AxisDenotationSeedRule(
                                      name_pattern=r"{name}",  # or use a regex pattern
                                      denotations=[AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH, ...]  # {rank} denotations total
                                  )
                              ]
                          )
                        """
                    )
                )
