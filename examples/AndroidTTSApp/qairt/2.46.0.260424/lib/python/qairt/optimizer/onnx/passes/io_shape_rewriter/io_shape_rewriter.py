# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""IOShapeRewriter pass for updating model I/O based on new sequence/context lengths"""

from typing import TypeAlias

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.axis_denotation_infer import AxisDenotationConfig, AxisDenotationInference
from qairt.optimizer.onnx.passes.axis_denotation_infer.utils import get_axis_denotations
from qairt.optimizer.onnx.passes.base import BasePass, BasePredicatePass
from qairt.optimizer.onnx.passes.cleaning import DeadCodeRemovalRewriter, DeadWeightRemovalRewriter
from qairt.optimizer.onnx.passes.io_shape_rewriter.config import IOShapeRewriterConfig
from qairt.optimizer.onnx.passes.io_shape_rewriter.node_update_passes import (
    get_node_update_passes,
    register_node_update_pass,
)
from qairt.optimizer.onnx.passes.shape_infer.shape_infer import ShapeInference
from qairt.optimizer.onnx.passes.utilities import ClearIntermediateShapes, ComputeSeqAndContextLength
from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation
from qairt.optimizer.onnx.utils.utils import (
    OpTypePredicate,
    get_constant_np,
    make_initializer,
    safe_replace_all_uses_with,
    scan_previous_nearest_candidate,
)
from qairt.optimizer.utils.logger import logger


def is_sequential_array(values: np.ndarray) -> bool:
    """Check if array is sequential [0,1,2,...,n-1] with at least 2 elements"""
    if len(values.shape) != 1 or len(values) < 2:
        return False
    return (values == np.arange(len(values))).all()


def is_uniform_array(values: np.ndarray) -> bool:
    """Check if all elements in the array have the same value (e.g., all-ones, all-zeros)"""
    return bool(values.size > 0 and np.all(values == values.flat[0]))


class UpdateGraphIO(BasePass):
    """
    Update graph inputs and outputs based on axis denotations

    Reads seq_length and context_length from graph_ir.meta and updates
    tensor shapes according to their axis denotations
    """

    def __init__(self, config: IOShapeRewriterConfig):
        super().__init__()
        self.config: IOShapeRewriterConfig = config

    def _update_tensor_shape(self, tensor: ir.Value, seq_length: int, context_length: int) -> bool:
        """
        Update a single tensor's shape based on its axis denotations

        Returns:
            True if shape was modified, False otherwise
        """
        denotations = get_axis_denotations(tensor)
        if not denotations or not tensor.shape:
            return False

        modified = False
        new_shape = list(tensor.shape)

        for i, denotation in enumerate(denotations):
            if denotation == AxisDenotation.SEQ_LENGTH and new_shape[i] != seq_length:
                new_shape[i] = seq_length
                modified = True
            elif denotation == AxisDenotation.CONTEXT_LENGTH and new_shape[i] != context_length:
                new_shape[i] = context_length
                modified = True
            elif denotation == AxisDenotation.PAST_SEQ_LENGTH:
                past_seq_length = context_length - seq_length
                if new_shape[i] != past_seq_length:
                    new_shape[i] = past_seq_length
                    modified = True

        if modified:
            old_shape = list(tensor.shape)
            tensor.shape = ir.Shape(new_shape)

            if tensor.is_graph_input():
                logger.info(f"Updated graph input '{tensor.name}': {old_shape} → {new_shape}")
            elif tensor.is_graph_output():
                logger.info(f"Updated graph output '{tensor.name}': {old_shape} → {new_shape}")

        return modified

    def apply(self, ctx: GraphContext) -> int:
        # Get target lengths from config or use original from metadata
        seq_length = self.config.new_seq_length or ctx.graph_ir.meta["seq_length"]
        context_length = self.config.new_context_length or ctx.graph_ir.meta["context_length"]

        changes = 0

        # Update inputs
        for tensor in ctx.graph_ir.inputs:
            if self._update_tensor_shape(tensor, seq_length, context_length):
                changes += 1

        # Update outputs
        for tensor in ctx.graph_ir.outputs:
            if self._update_tensor_shape(tensor, seq_length, context_length):
                changes += 1

        return changes


@register_node_update_pass
class UpdateShapeInputTensors(BasePredicatePass):
    """
    Update shape constant inputs (input[1]) on Reshape and Expand nodes based on
    output axis denotations.

    Skips special shape values (<= 0): -1 means "infer" and 0 means "copy from input".
    """

    op_types = {"Reshape", "Expand"}

    def __init__(self, config: IOShapeRewriterConfig):
        super().__init__()
        self.config: IOShapeRewriterConfig = config

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type not in self.op_types:
            return False

        # Check if output has axis denotations
        denotations = get_axis_denotations(node.outputs[0])
        if not denotations:
            return False

        # Check if shape input (input[1]) is a constant
        shape_input = node.inputs[1]
        if get_constant_np(shape_input) is None:
            return False

        return True

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        seq_length = self.config.new_seq_length or graph.meta["seq_length"]
        context_length = self.config.new_context_length or graph.meta["context_length"]

        denotations = get_axis_denotations(node.outputs[0])
        if not denotations:
            return False

        shape_input = node.inputs[1]
        if not shape_input:  # For mypy
            return False

        shape_array = get_constant_np(shape_input).copy()

        modified = False

        # Update shape values based on denotations
        for i, denotation in enumerate(denotations):
            if i >= len(shape_array):
                break

            # Skip special values: -1 means "infer", 0 means "copy from input"
            if shape_array[i] <= 0:
                continue

            if denotation == AxisDenotation.SEQ_LENGTH and shape_array[i] != seq_length:
                shape_array[i] = seq_length
                modified = True
            elif denotation == AxisDenotation.CONTEXT_LENGTH and shape_array[i] != context_length:
                shape_array[i] = context_length
                modified = True
            elif denotation == AxisDenotation.PAST_SEQ_LENGTH:
                past_seq_length = context_length - seq_length
                if shape_array[i] != past_seq_length:
                    shape_array[i] = past_seq_length
                    modified = True

        if modified:
            # Create new initializer with updated shape array
            new_shape_tensor = make_initializer(graph, f"{shape_input.name}_updated", shape_array)

            # Replace all uses of old shape input with new one
            safe_replace_all_uses_with(graph, shape_input, new_shape_tensor)

        return modified


@register_node_update_pass
class UpdateConstantTensors(BasePredicatePass):
    """
    Update constant tensors based on axis denotations

    Handles sequential arrays [0,1,2,...,n-1] with SEQ_LENGTH denotation

    Operates on node inputs (constants) rather than iterating all values
    """

    def __init__(self, config: IOShapeRewriterConfig):
        super().__init__()
        self.config: IOShapeRewriterConfig = config

    _LENGTH_DENOTATIONS = {
        AxisDenotation.SEQ_LENGTH,
        AxisDenotation.CONTEXT_LENGTH,
        AxisDenotation.PAST_SEQ_LENGTH,
    }

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        for inp in node.inputs:
            if inp is None or inp.const_value is None:
                continue

            denotations = get_axis_denotations(inp)
            if not denotations:
                continue

            values = inp.const_value.numpy()

            if is_sequential_array(values):
                return True

            if is_uniform_array(values) and len(values.shape) > 1:
                if any(d in self._LENGTH_DENOTATIONS for d in denotations):
                    return True

        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        seq_length = self.config.new_seq_length or graph.meta["seq_length"]
        context_length = self.config.new_context_length or graph.meta["context_length"]

        modified = False

        for inp in node.inputs:
            if inp is None or inp.const_value is None:
                continue

            denotations = get_axis_denotations(inp)
            if not denotations:
                continue

            values = inp.const_value.numpy()

            # Sequential array [0,1,2,...,n-1] with SEQ_LENGTH denotation
            if is_sequential_array(values):
                if AxisDenotation.SEQ_LENGTH in denotations:
                    old_len = len(values)
                    if old_len != seq_length:
                        inp.const_value = ir.Tensor(np.arange(seq_length, dtype=values.dtype))
                        inp.shape = ir.Shape([seq_length])
                        logger.info(f"Updated constant tensor '{inp.name}': [{old_len}] → [{seq_length}]")
                        modified = True

            # Uniform (broadcast) tensor — resize axes with length-related denotations
            elif is_uniform_array(values) and len(values.shape) > 1:
                fill_value = values.flat[0]
                new_shape = list(values.shape)
                shape_modified = False

                for i, denotation in enumerate(denotations):
                    if i >= len(new_shape):
                        break
                    if denotation == AxisDenotation.SEQ_LENGTH and new_shape[i] != seq_length:
                        new_shape[i] = seq_length
                        shape_modified = True
                    elif denotation == AxisDenotation.CONTEXT_LENGTH and new_shape[i] != context_length:
                        new_shape[i] = context_length
                        shape_modified = True
                    elif denotation == AxisDenotation.PAST_SEQ_LENGTH:
                        past_seq_length = context_length - seq_length
                        if new_shape[i] != past_seq_length:
                            new_shape[i] = past_seq_length
                            shape_modified = True

                if shape_modified:
                    old_shape = list(values.shape)
                    new_values = np.full(new_shape, fill_value, dtype=values.dtype)
                    inp.const_value = ir.Tensor(new_values)
                    inp.shape = ir.Shape(new_shape)
                    logger.info(f"Updated uniform constant '{inp.name}': {old_shape} → {new_shape}")
                    modified = True

        return modified


@register_node_update_pass
class UpdateScaledReshapeNodes(BasePredicatePass):
    """
    Update Reshape node shape constants that contain scaled AR/CL dimensions.

    Handles the MoE pattern where a Reshape constant contains a dimension that
    is an exact multiple of the original seq/context length (e.g. ``AR * K``
    for some integer ``K``).  These dimensions have no axis denotation because
    the denotation chain is broken by a flatten, but they still need to be
    updated when AR or CL changes.

    Example: ``[-1, 584]`` where ``584 = 73 * 8`` (AR=73, experts_per_token=8)
    must become ``[-1, 512]`` when AR changes to 64.
    """

    def __init__(self, config: IOShapeRewriterConfig):
        super().__init__()
        self.config: IOShapeRewriterConfig = config

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        if node.op_type != "Reshape":
            return False

        # Skip if output has actionable denotations — UpdateShapeInputTensors handles those.
        # UNKNOWN-only denotations don't block us: the denotation chain propagated
        # through the node but couldn't assign a meaningful label, so UpdateShapeInputTensors
        # won't update the constant either.
        _ACTIONABLE = {
            AxisDenotation.SEQ_LENGTH,
            AxisDenotation.CONTEXT_LENGTH,
            AxisDenotation.PAST_SEQ_LENGTH,
        }
        if any(d in _ACTIONABLE for d in get_axis_denotations(node.outputs[0])):
            return False

        # Only act when at least one length is changing
        old_seq = graph.meta.get("seq_length")
        old_ctx = graph.meta.get("context_length")
        if old_seq is None and old_ctx is None:
            return False

        shape_input = node.inputs[1]
        shape_array = get_constant_np(shape_input)
        if shape_array is None:
            return False

        data_input = node.inputs[0]
        if data_input is None or not data_input.shape:
            return False

        # Guard 1: data flow must pass through a TopK (MoE routing signal)
        if not any(
            scan_previous_nearest_candidate(
                [data_input],
                check_fn=OpTypePredicate(["TopK"]),
                ignore_fn=~OpTypePredicate(["TopK"]),
            )
        ):
            return False

        # Guard 2: candidate dim must appear in data input's actual shape
        data_dims = {d for d in data_input.shape if isinstance(d, int) and d > 0}

        new_seq = self.config.new_seq_length or old_seq
        new_ctx = self.config.new_context_length or old_ctx

        for dim in shape_array:
            if dim <= 0 or dim not in data_dims:
                continue
            if old_seq and old_seq > 1 and dim % old_seq == 0 and dim != new_seq:
                return True
            if old_ctx and old_ctx > 1 and dim % old_ctx == 0 and dim != new_ctx:
                return True

        return False

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        old_seq = graph.meta.get("seq_length")
        old_ctx = graph.meta.get("context_length")
        new_seq = self.config.new_seq_length or old_seq
        new_ctx = self.config.new_context_length or old_ctx

        shape_input = node.inputs[1]
        if not shape_input:
            return False

        data_input = node.inputs[0]
        data_dims = (
            {d for d in data_input.shape if isinstance(d, int) and d > 0}
            if data_input and data_input.shape
            else set()
        )

        shape_array = get_constant_np(shape_input).copy()
        modified = False

        for i, dim in enumerate(shape_array):
            if dim <= 0 or dim not in data_dims:
                continue
            # Scale by new_seq / old_seq
            if old_seq and old_seq > 1 and dim % old_seq == 0:
                factor = dim // old_seq
                new_dim = new_seq * factor
                if new_dim != dim:
                    shape_array[i] = new_dim
                    modified = True
                    continue
            # Scale by new_ctx / old_ctx
            if old_ctx and old_ctx > 1 and dim % old_ctx == 0:
                factor = dim // old_ctx
                new_dim = new_ctx * factor
                if new_dim != dim:
                    shape_array[i] = new_dim
                    modified = True

        if modified:
            new_shape_tensor = make_initializer(graph, f"{shape_input.name}_scaled", shape_array)
            safe_replace_all_uses_with(graph, shape_input, new_shape_tensor)

        return modified


class IOShapeRewriter(BasePass):
    """
    Updates model I/O shapes and constants based on new sequence/context lengths

    This pass uses axis denotations to intelligently update tensor shapes throughout
    the graph when changing sequence length (AR) or context length (CL)

    Steps:
    1. Compute original sequence length and context length from graph
    2. Validate config for new sequence length and/or new context length
    3. Run AxisDenotationInference pass to label all axes
    4. Update graph inputs/outputs based on axis denotations
    5. Update relevant nodes
    6. Update graph metadata with new values
    7. Run cleanup passes
    8. Run shape inference to recompute shapes
    """

    Config: TypeAlias = IOShapeRewriterConfig

    def __init__(self, config: Config | None = None):
        if config is None:
            config = IOShapeRewriter.Config()
        super().__init__()
        self.config: IOShapeRewriter.Config = config

    @staticmethod
    def validate_config(seq_length: int, context_length: int) -> None:
        """
        Validate that 1 <= seq_length <= context_length - 1

        Args:
            seq_length: Sequence length to validate
            context_length: Context length to validate

        Raises:
            ValueError: If validation fails
        """
        if seq_length <= 0:
            raise ValueError("Sequence length must be greater than 0")
        if context_length <= 0:
            raise ValueError("Context length must be greater than 0")
        if seq_length > context_length - 1:
            raise ValueError(
                f"Invalid sequence length '{seq_length}' for context length '{context_length}' "
                f"Supported range is 1 <= SEQ_LENGTH <= {context_length - 1}"
            )

    def apply(self, ctx: GraphContext) -> int:
        # No-op
        if self.config.new_seq_length is None and self.config.new_context_length is None:
            return 0

        # Step 1: Label and infer axis denotations of all the tensors in the graph
        AxisDenotationInference(self.config.axis_denotation_config).apply(ctx)

        # Step 2: Compute original lengths from graph (If not already computed)
        if "seq_length" not in ctx.graph_ir.meta or "context_length" not in ctx.graph_ir.meta:
            axis_config = self.config.axis_denotation_config
            axis_config = axis_config if axis_config else AxisDenotationConfig()
            compute_config = ComputeSeqAndContextLength.Config(
                attention_mask_name_pattern=axis_config.attention_mask_name_pattern,
                swa_mask_name_pattern=axis_config.swa_mask_name_pattern,
                input_ids_name_pattern=axis_config.input_ids_name_pattern,
                layer_output_name_pattern=axis_config.layer_output_name_pattern,
            )
            ComputeSeqAndContextLength(compute_config).apply(ctx)

        # Step 3: Resolve and validate config (use original if None)
        original_seq_length = ctx.graph_ir.meta["seq_length"]
        original_context_length = ctx.graph_ir.meta["context_length"]

        seq_length = self.config.new_seq_length or original_seq_length
        context_length = self.config.new_context_length or original_context_length

        # Check if no-op
        if seq_length == original_seq_length and context_length == original_context_length:
            return 0

        IOShapeRewriter.validate_config(seq_length, context_length)

        # Step 4: Update graph I/O
        # NOTE: Should graph IO updates be counted as a graph change?
        total_changes = UpdateGraphIO(self.config).apply(ctx)

        # Step 5: Update relevant nodes and clear all intermediate tensor shapes
        for update_pass in [*get_node_update_passes(self.config), ClearIntermediateShapes()]:
            total_changes += update_pass.apply(ctx)

        # Step 6: Update metadata with new values
        ctx.graph_ir.meta["seq_length"] = seq_length
        ctx.graph_ir.meta["context_length"] = context_length

        # Step 7: Cleanup - remove unused nodes and constants
        DeadCodeRemovalRewriter().apply(ctx)
        DeadWeightRemovalRewriter().apply(ctx)

        # Step 8: Run shape inference
        ShapeInference().apply(ctx)

        return total_changes


def change_seq_length(
    ctx: GraphContext,
    new_seq_length: int,
    axis_denotation_config: AxisDenotationConfig | None = None,
) -> GraphContext:
    """
    Change the sequence length(AR) of an LLM

    Modifies the graph in-place. If you need to preserve the original, pass a
    ``copy.deepcopy(ctx)`` instead.

    Args:
        ctx: Graph context containing the model
        new_seq_length: New sequence length to apply
        axis_denotation_config: Optional configuration for axis denotation inference
                              See :class:`~qairt.optimizer.onnx.passes.axis_denotation_infer.config.AxisDenotationConfig`
                              for details. Provide this if your model has non-standard input
                              names that don't match the built-in patterns

    Returns:
        The same GraphContext, modified in-place

    Example::

        from qairt.optimizer.onnx import change_seq_length
        from qairt.optimizer.onnx import GraphContext

        ctx = GraphContext.from_files("model.onnx")
        change_seq_length(ctx, 128)
        ctx.save("model_modified.onnx")

        # With custom seed rules
        from qairt.optimizer.onnx import (
            change_seq_length,
            AxisDenotationConfig,
            AxisDenotationSeedRule,
            AxisDenotation
        )

        axis_denotation_config = AxisDenotationConfig(
            custom_seed_rules=[
                AxisDenotationSeedRule(
                    name_pattern=r"my_custom_input",
                    denotations=[AxisDenotation.BATCH, AxisDenotation.SEQ_LENGTH]
                )
            ]
        )
        change_seq_length(ctx, 128, axis_denotation_config)
    """
    config = IOShapeRewriter.Config(
        new_seq_length=new_seq_length,
        axis_denotation_config=axis_denotation_config,
    )
    IOShapeRewriter(config).apply(ctx)
    return ctx


def change_context_length(
    ctx: GraphContext,
    new_context_length: int,
    axis_denotation_config: AxisDenotationConfig | None = None,
) -> GraphContext:
    """
    Change the context length(CL) of an LLM

    Modifies the graph in-place. If you need to preserve the original, pass a
    ``copy.deepcopy(ctx)`` instead.

    Args:
        ctx: Graph context containing the model
        new_context_length: New context length to apply
        axis_denotation_config: Optional configuration for axis denotation inference
                              See :class:`~qairt.optimizer.onnx.passes.axis_denotation_infer.config.AxisDenotationConfig`
                              for details. Provide this if your model has non-standard input
                              names that don't match the built-in patterns

    Returns:
        The same GraphContext, modified in-place

    Example::

        from qairt.optimizer.onnx import change_context_length
        from qairt.optimizer.onnx import GraphContext

        ctx = GraphContext.from_files("model.onnx")
        change_context_length(ctx, 2048)
        ctx.save("model_modified.onnx")
    """
    config = IOShapeRewriter.Config(
        new_context_length=new_context_length,
        axis_denotation_config=axis_denotation_config,
    )
    IOShapeRewriter(config).apply(ctx)
    return ctx


def change_seq_and_context_length(
    ctx: GraphContext,
    new_seq_length: int,
    new_context_length: int,
    axis_denotation_config: AxisDenotationConfig | None = None,
) -> GraphContext:
    """
    Change both sequence length(AR) and context length(CL) of an LLM

    Modifies the graph in-place. If you need to preserve the original, pass a
    ``copy.deepcopy(ctx)`` instead.

    Args:
        ctx: Graph context containing the model
        new_seq_length: New sequence length to apply
        new_context_length: New context length to apply
        axis_denotation_config: Optional configuration for axis denotation inference
                              See :class:`~qairt.optimizer.onnx.passes.axis_denotation_infer.config.AxisDenotationConfig`
                              for details. Provide this if your model has non-standard input
                              names that don't match the built-in patterns

    Returns:
        The same GraphContext, modified in-place

    Example::

        from qairt.optimizer.onnx import change_seq_and_context_length
        from qairt.optimizer.onnx import GraphContext

        ctx = GraphContext.from_files("model.onnx")
        change_seq_and_context_length(ctx, 128, 2048)
        ctx.save("model_modified.onnx")
    """
    config = IOShapeRewriter.Config(
        new_seq_length=new_seq_length,
        new_context_length=new_context_length,
        axis_denotation_config=axis_denotation_config,
    )
    IOShapeRewriter(config).apply(ctx)
    return ctx
