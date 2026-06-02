# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides a pass to permute kv cache input/output
to make head dim as the first dim
"""

import copy
import re
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass, PassConfig
from qairt.optimizer.onnx.utils.utils import (
    get_value_numeric_shape,
    safe_replace_all_uses_with,
)
from qairt.optimizer.utils.logger import logger


class PermuteKVCacheRewriter(BasePass):
    """
    Pass to permute head dim and batch dim on kv cache input/output
    """

    @dataclass
    class Config(PassConfig):
        """Configuration for PermuteKVCacheRewriter pass"""

        key_cache_name_pattern: str = "past_key_(\\d)+_in|past_key_(\\d)+_out"
        value_cache_name_pattern: str = "past_value_(\\d)+_in|past_value_(\\d)+_out"

    def __init__(self, config: Config | None = None):
        super().__init__()
        if config is None:
            config = PermuteKVCacheRewriter.Config()
        self.config: PermuteKVCacheRewriter.Config = config

        self.key_name_pattern = re.compile(self.config.key_cache_name_pattern)
        self.value_name_pattern = re.compile(self.config.value_cache_name_pattern)

    def _permute_input_kv(self, graph: ir.Graph, value: ir.Value):
        # set head dim on batch
        origin_shape = get_value_numeric_shape(value)
        perm = list(range(len(origin_shape)))
        perm[0:2] = [1, 0]
        new_shape = np.array(origin_shape)[perm].tolist()
        value.shape = ir.Shape(new_shape)
        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[value],
            attributes=[ir.AttrInt64s("perm", perm)],
            name=graph.meta["extra_info"].get_unique_name_with_suffix(value.name, "/permute"),
        )
        transpose_node.outputs[0].name = graph.meta["extra_info"].get_unique_name_with_suffix(
            value.name, "/origin_layout"
        )
        graph.insert_before(graph[0], transpose_node)
        safe_replace_all_uses_with(graph, value, transpose_node.outputs[0], except_users=[transpose_node])
        transpose_node.outputs[0].shape = ir.Shape(origin_shape)
        transpose_node.outputs[0].type = copy.deepcopy(value.type)
        transpose_node.outputs[0].meta["extra_info"] = value.meta["extra_info"].copy(ignore_safetensors=True)

        logger.debug("permute input kv cache on '%s'", value.name)
        return value

    def _permute_output_kv(self, graph: ir.Graph, value: ir.Value):
        # set head dim on batch
        assert value.name is not None  # check for mypy
        value_name = value.name
        value_shape = get_value_numeric_shape(value)
        perm = list(range(len(value_shape)))
        perm[0:2] = [1, 0]
        transpose_node = ir.Node(
            "",
            "Transpose",
            inputs=[value],
            attributes=[ir.AttrInt64s("perm", perm)],
            name=graph.meta["extra_info"].get_unique_name_with_suffix(value.name, "/permute"),
        )
        graph.insert_after(graph[-1], transpose_node)
        value.name = graph.meta["extra_info"].get_unique_name_with_suffix(value.name, "/permute_layout")
        transpose_node.outputs[0].name = value_name
        transpose_node.outputs[0].shape = ir.Shape(np.array(value_shape)[perm].tolist())
        transpose_node.outputs[0].type = copy.deepcopy(value.type)
        transpose_node.outputs[0].meta["extra_info"] = value.meta["extra_info"].copy(ignore_safetensors=True)

        graph_outputs = graph.outputs
        graph_outputs[graph_outputs.index(value)] = transpose_node.outputs[0]

        logger.debug("permute output kv cache on '%s'", value_name)
        return value

    def apply(self, ctx: GraphContext) -> int:
        """
        Apply the KV cache permutation to the graph.

        Args:
            ctx: The model context containing the graph and metadata

        Returns:
            Number of KV cache tensors permuted
        """
        graph = ctx.graph_ir
        rewrite_count = 0
        for input_v in graph.inputs[:]:
            assert input_v.name is not None  # for mypy
            if self.key_name_pattern.match(input_v.name) or self.value_name_pattern.match(input_v.name):
                self._permute_input_kv(graph, input_v)
                rewrite_count += 1
        for output_v in graph.outputs[:]:
            assert output_v.name is not None  # for mypy
            if self.key_name_pattern.match(output_v.name) or self.value_name_pattern.match(output_v.name):
                self._permute_output_kv(graph, output_v)
                rewrite_count += 1
        return rewrite_count

    def preproc_inputs(self, inputs: dict[str, np.ndarray]):
        """
        Preprocess the inputs data of the original graph,
        return the corresponding inputs data of the KV cache-permuted graph.
        Args:
            inputs: Inputs of the original graph
        Returns:
            Corresponding inputs of the KV cache-permuted graph
        """
        for name, value in inputs.items():
            if self.key_name_pattern.match(name) or self.value_name_pattern.match(name):
                perm = list(range(len(value.shape)))
                perm[0:2] = [1, 0]
                inputs[name] = value.transpose(*perm)
        return inputs

    def postproc_outputs(self, outputs: dict[str, np.ndarray]):
        """
        Postprocess the outputs data of the KV cache-permuted graph,
        return the corresponding outputs data of the original graph.
        Args:
            outputs: Outputs of the KV cache-permuted graph
        Returns:
            Corresponding outputs of the original graph
        """

        for name, value in outputs.items():
            if self.key_name_pattern.match(name) or self.value_name_pattern.match(name):
                perm = list(range(len(value.shape)))
                perm[0:2] = [1, 0]
                outputs[name] = value.transpose(*perm)
        return outputs
