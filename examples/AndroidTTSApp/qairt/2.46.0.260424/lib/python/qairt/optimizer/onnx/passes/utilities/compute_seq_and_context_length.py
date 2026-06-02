# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Compute sequence length and context length from graph inputs.

This module extracts seq_length, context_length, and optionally
sliding_context_length from attention mask and input_ids
"""

import re
from dataclasses import dataclass
from textwrap import dedent

import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass, PassConfig


class ComputeSeqAndContextLength(BasePass):
    """
    Extract seq_length, context_length, and sliding_context_length(for SWA models only) from graph.

    This pass examines graph inputs to extract dimensional information:
    - context_length: From attention_mask shape (last dimension)
    - sliding_context_length: From swa_attention_mask shape (last dimension, if present)
    - seq_length: From attention_mask, input_ids or layer_output (residual add)

    Results are stored in graph.meta:
    - graph.meta["seq_length"]
    - graph.meta["context_length"]
    - graph.meta["sliding_context_length"] (optional)
    """

    @dataclass
    class Config(PassConfig):
        """Configuration for ComputeSeqAndContextLength pass"""

        attention_mask_name_pattern: str = "attention_mask"
        """Pattern to match attention mask input"""

        swa_mask_name_pattern: str = "swa_attention_mask"
        """Pattern to match SWA attention mask input"""

        input_ids_name_pattern: str = "input_ids"
        """Pattern to match input_ids (fallback for seq_length)"""

        inputs_embeds_name_pattern: str = "(input|inputs)_embeds"
        """Pattern to match inputs_embeds (fallback for seq_length)"""

        layer_output_name_pattern: str = "/?model_(layers_\\d+_Add/Add|embed_tokens/Gather)_output_0"
        """Pattern to match attention layer output (fallback for seq_length)"""

    def __init__(self, config: Config | None = None):
        if config is None:
            config = ComputeSeqAndContextLength.Config()
        super().__init__()
        self.config: ComputeSeqAndContextLength.Config = config

        # Compile patterns
        self.attention_mask_pattern = re.compile(self.config.attention_mask_name_pattern, re.IGNORECASE)
        self.swa_mask_pattern = re.compile(self.config.swa_mask_name_pattern, re.IGNORECASE)
        self.input_ids_pattern = re.compile(self.config.input_ids_name_pattern, re.IGNORECASE)
        self.inputs_embeds_pattern = re.compile(self.config.inputs_embeds_name_pattern, re.IGNORECASE)
        self.layer_output_pattern = re.compile(self.config.layer_output_name_pattern, re.IGNORECASE)

    def _extract_seq_length_from_input_ids(self, graph: ir.Graph):
        """Fallback: extract seq_length from input_ids shape"""
        for input_tensor in graph.inputs:
            if not input_tensor.name or not input_tensor.shape:
                continue

            if self.input_ids_pattern.fullmatch(input_tensor.name):
                # input_ids: [batch, seq_length]
                if len(input_tensor.shape) == 2:
                    return input_tensor.shape[1]

        return None

    def _extract_seq_length_from_inputs_embeds(self, graph: ir.Graph):
        """Fallback: extract seq_length from inputs_embeds shape"""
        for input_tensor in graph.inputs:
            if not input_tensor.name or not input_tensor.shape:
                continue

            if self.inputs_embeds_pattern.fullmatch(input_tensor.name):
                # inputs_embeds: [batch, seq_length, hidden_dim]
                if len(input_tensor.shape) == 3:
                    return input_tensor.shape[1]

        return None

    def _extract_seq_length_from_layer_output(self, graph: ir.Graph):
        """Fallback: extract seq_length from layer_output shape"""
        for input_tensor in graph.inputs:
            if not input_tensor.name or not input_tensor.shape:
                continue

            if self.layer_output_pattern.fullmatch(input_tensor.name):
                # layer_output: [batch, seq_length, hidden_dim]
                if len(input_tensor.shape) == 3:
                    return input_tensor.shape[1]

        return None

    def apply(self, ctx: GraphContext) -> int:
        """
        Extract lengths from graph inputs and store in metadata.

        Returns:
            1 if lengths were extracted, 0 if already present
        """
        graph = ctx.graph_ir

        # Skip if already computed
        if "context_length" in graph.meta and "seq_length" in graph.meta:
            return 0

        seq_length = None
        context_length = None
        sliding_context_length = None

        for input_tensor in graph.inputs:
            if not input_tensor.name or not input_tensor.shape:
                continue

            name = input_tensor.name
            shape = input_tensor.shape
            rank = len(shape)

            if self.attention_mask_pattern.fullmatch(name):
                # Last dimension is ALWAYS context_length
                context_length = shape[-1]

                # If rank >= 3, second-to-last dim is seq_length
                if rank >= 3 and seq_length is None:
                    seq_length = shape[-2]

            # SWA attention mask
            elif self.swa_mask_pattern.fullmatch(name):
                # Last dimension is sliding_context_length
                sliding_context_length = shape[-1]

                # Second-to-last is seq_length (if rank >= 3)
                if rank >= 3 and seq_length is None:
                    seq_length = shape[-2]

        # Fallback: Extract seq_length from input_ids
        if seq_length is None:
            seq_length = self._extract_seq_length_from_input_ids(graph)

        # Fallback: Extract seq_length from inputs_embeds
        if seq_length is None:
            seq_length = self._extract_seq_length_from_inputs_embeds(graph)

        # Fallback: Extract seq_length from layer_output
        if seq_length is None:
            seq_length = self._extract_seq_length_from_layer_output(graph)

        # Validate we found required lengths
        if context_length is None:
            raise ValueError(
                dedent(
                    f"""
                    Could not find attention_mask to extract CONTEXT_LENGTH.
                    Ensure your model has an input matching pattern: "
                    '{self.config.attention_mask_name_pattern}'
                    """
                )
            )

        if seq_length is None:
            raise ValueError(
                "Could not determine SEQ_LENGTH from attention_mask, input_ids, inputs_embeds, or layer_output"
            )

        # Store in metadata
        graph.meta["seq_length"] = seq_length
        graph.meta["context_length"] = context_length
        if sliding_context_length is not None:
            graph.meta["sliding_context_length"] = sliding_context_length

        return 1
