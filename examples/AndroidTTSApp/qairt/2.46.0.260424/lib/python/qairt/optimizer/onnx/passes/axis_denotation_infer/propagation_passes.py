# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Axis denotation propagation passes for various op types"""

from abc import abstractmethod

import numpy as np
import onnx_ir as ir

from qairt.optimizer.onnx.passes.axis_denotation_infer.utils import get_axis_denotations, set_axis_denotations
from qairt.optimizer.onnx.passes.base import BasePredicatePass
from qairt.optimizer.onnx.utils.ir_extra_info import AxisDenotation
from qairt.optimizer.onnx.utils.utils import get_constant_np
from qairt.optimizer.utils.logger import logger

_REGISTRY: dict[str, type["BaseAxisDenotationPass"]] = {}


def register_pass(cls: type["BaseAxisDenotationPass"]) -> type["BaseAxisDenotationPass"]:
    """Class decorator that registers a propagation pass in the global op-type registry.

    Apply to every ``Infer*AxisDenotation`` subclass so that
    :func:`get_registered_op_types` can report all covered op types and
    :func:`get_registered_passes` can instantiate all passes without requiring
    a manually maintained list.
    """
    _REGISTRY[cls.__name__] = cls
    return cls


def get_registered_op_types() -> dict[str, set[str]]:
    """Return a mapping of pass class name → set of op types it handles.

    Useful for quickly checking which ops have propagation rules and which do not.

    Example::

        from qairt.optimizer.onnx.passes.axis_denotation_infer.propagation_passes import (
            get_registered_op_types,
        )
        table = get_registered_op_types()
        for pass_name, ops in sorted(table.items()):
            print(f"{pass_name}: {sorted(ops)}")
    """
    return {name: cls.op_types for name, cls in _REGISTRY.items()}


def get_registered_passes() -> list["BaseAxisDenotationPass"]:
    """Return one instance of every registered propagation pass.

    The order matches registration order (i.e. class definition order in this
    module), which is the topological-processing order used by
    :class:`AxisDenotationInference`.
    """
    return [cls() for cls in _REGISTRY.values()]


ABBREV_MAP = {
    AxisDenotation.BATCH: "B",
    AxisDenotation.SEQ_LENGTH: "AR",
    AxisDenotation.CONTEXT_LENGTH: "CL",
    AxisDenotation.SLIDING_CONTEXT_LENGTH: "SCL",
    AxisDenotation.PAST_SEQ_LENGTH: "P",
    AxisDenotation.UNKNOWN: "-",
}


def format_denotations(denotations: list[AxisDenotation] | None) -> str:
    """Format denotations compactly using abbreviations"""
    if not denotations:
        return "[]"

    abbrevs = [ABBREV_MAP.get(d, d.name[:3]) for d in denotations]
    return "[" + ",".join(abbrevs) + "]"


def log_propagation_success(node: ir.Node) -> None:
    """Log successful axis denotation propagation in compact format"""
    inputs_part = " × ".join(format_denotations(get_axis_denotations(inp)) for inp in node.inputs if inp)
    outputs_part = ", ".join(format_denotations(get_axis_denotations(out)) for out in node.outputs if out)
    logger.debug(f"{node.op_type} '{node.name or 'unnamed'}': {inputs_part} → {outputs_part}")


class BaseAxisDenotationPass(BasePredicatePass):
    """Base class for axis denotation propagation passes

    - op_types: set of op types this pass handles
    - infer_output_denotations(): logic to compute output denotations
    """

    # Subclasses must override this
    op_types: set[str] = set()

    def match(self, graph: ir.Graph, node: ir.Node) -> bool:
        """Check op_type and if required inputs have denotations"""
        return (node.op_type in self.op_types) and self._has_required_input_denotations(node)

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        """
        Get input denotations, infer and set output tensor denotations
        Handles the case of a single output that applies to most ONNX ops
        Overridden for operators that have multiple outputs (Split, TopK, etc)
        """
        output = node.outputs[0]
        if not output:
            return False

        output_denotations = self.infer_output_denotations(node)
        if output_denotations:
            set_axis_denotations(output, output_denotations)
            log_propagation_success(node)
            return True
        return False

    def _has_required_input_denotations(self, node: ir.Node) -> bool:
        """Check if required inputs have denotations. Override if needed"""
        return node.inputs[0] is not None and len(get_axis_denotations(node.inputs[0])) != 0

    def _get_input_denotations(self, node: ir.Node, index: int):
        """Get denotations from input at index"""
        if index < len(node.inputs) and (node_input := node.inputs[index]) is not None:
            return get_axis_denotations(node_input)
        return None

    @abstractmethod
    def infer_output_denotations(self, node: ir.Node) -> list[AxisDenotation] | None:
        """Compute output denotations from input denotations"""
        pass


@register_pass
class InferIdentityAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for operations that preserve shape and axis denotations"""

    op_types = {
        "Sigmoid",
        "Sqrt",
        "Tanh",
        "Clip",
        "Identity",
        "Softmax",
        "Cast",
        "Slice",
        "ScatterElements",
        "Relu",
        "Gelu",
        "Silu",
        "Swish",
        "Elu",
        "LeakyRelu",
        "Dropout",
        "LayerNormalization",
        "BatchNormalization",
        "GroupNormalization",
        "RMSNorm",
        "Neg",
        "Abs",
        "Exp",
        "Log",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Not",
        "IsNaN",
        "IsInf",
        "Sign",
        "Floor",
        "Ceil",
        "Round",
        "Erf",
        "Sin",
        "Cos",
        "Equal",
        "Greater",
        "Less",
        "GreaterOrEqual",
        "LessOrEqual",
        "Not",
        "Pad",
        "Tile",
    }

    def infer_output_denotations(self, node):
        return self._get_input_denotations(node, 0)


@register_pass
class InferElementwiseAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for elementwise ops"""

    op_types = {"Add", "Sub", "Mul", "Div", "Pow"}

    def _has_required_input_denotations(self, node):
        return any(inp and get_axis_denotations(inp) for inp in node.inputs)

    def infer_output_denotations(self, node):
        # Get all inputs with denotations
        denotations_1 = get_axis_denotations(node.inputs[0])
        denotations_2 = get_axis_denotations(node.inputs[1])

        # Handle cases where one or both inputs don't have denotations

        if not denotations_2:
            # Only input 1 has denotations - need to check ranks for broadcasting
            if node.inputs[1] and node.inputs[1].shape:
                rank2 = len(node.inputs[1].shape)
                rank1 = len(denotations_1)
                if rank1 != rank2:
                    # Different ranks - apply broadcasting (pad shorter with UNKNOWN)
                    max_rank = max(rank1, rank2)
                    best_denotations = [AxisDenotation.UNKNOWN] * (max_rank - rank1) + denotations_1
                else:
                    best_denotations = denotations_1
            else:
                best_denotations = denotations_1
        elif not denotations_1:
            # Only input 2 has denotations - need to check ranks for broadcasting
            if node.inputs[0] and node.inputs[0].shape:
                rank1 = len(node.inputs[0].shape)
                rank2 = len(denotations_2)
                if rank1 != rank2:
                    # Different ranks - apply broadcasting (pad shorter with UNKNOWN)
                    max_rank = max(rank1, rank2)
                    best_denotations = [AxisDenotation.UNKNOWN] * (max_rank - rank2) + denotations_2
                else:
                    best_denotations = denotations_2
            else:
                best_denotations = denotations_2
        else:
            # Both have denotations - use broadcasting semantics to merge them
            # Align to the right (trailing dimensions match)
            rank1 = len(denotations_1)
            rank2 = len(denotations_2)
            max_rank = max(rank1, rank2)

            # Pad shorter denotations with UNKNOWN on the left
            padded_1 = [AxisDenotation.UNKNOWN] * (max_rank - rank1) + denotations_1
            padded_2 = [AxisDenotation.UNKNOWN] * (max_rank - rank2) + denotations_2

            # Merge denotations dimension by dimension
            best_denotations = []
            for d1, d2 in zip(padded_1, padded_2):
                if d1 == AxisDenotation.UNKNOWN and d2 == AxisDenotation.UNKNOWN:
                    best_denotations.append(AxisDenotation.UNKNOWN)
                elif d1 == AxisDenotation.UNKNOWN:
                    best_denotations.append(d2)
                elif d2 == AxisDenotation.UNKNOWN:
                    best_denotations.append(d1)
                elif d1 == d2:
                    # Both have same denotation
                    best_denotations.append(d1)
                else:
                    # Conflicting denotations - use UNKNOWN
                    best_denotations.append(AxisDenotation.UNKNOWN)

        # Label constant inputs with the same denotations
        # NOTE: A way to label and change constant tensors for Expand and cache_index
        for inp in node.inputs:
            if inp and not get_axis_denotations(inp):
                is_constant = inp.const_value or (inp.producer() and inp.producer().op_type == "Constant")
                if is_constant and inp.shape and len(inp.shape) == len(best_denotations):
                    set_axis_denotations(inp, best_denotations)

        return best_denotations


@register_pass
class InferTransposeAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Transpose"""

    op_types = {"Transpose"}

    def infer_output_denotations(self, node):
        input_denotations = self._get_input_denotations(node, 0)

        # Get permutation
        perm = node.attributes.get("perm")

        # By default, reverse the dimensions
        if not perm:
            return input_denotations[::-1]

        # Permute denotations
        perm_ints = perm.as_ints()
        return [input_denotations[i] for i in perm_ints]


@register_pass
class InferMatMulAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for MatMul"""

    op_types = {"MatMul"}

    def _has_required_input_denotations(self, node):
        return (
            node.inputs[0]
            and get_axis_denotations(node.inputs[0])
            and node.inputs[1]
            and get_axis_denotations(node.inputs[1])
        )

    def infer_output_denotations(self, node):
        left_denotations = self._get_input_denotations(node, 0)
        right_denotations = self._get_input_denotations(node, 1)

        if len(left_denotations) >= 2 and len(right_denotations) >= 2:
            # [..., M, K] @ [..., K, N] -> [..., M, N]
            return left_denotations[:-1] + [right_denotations[-1]]
        return None


@register_pass
class InferGatherAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Gather"""

    op_types = {"Gather"}

    def _has_required_input_denotations(self, node):
        # Either data or indices should have denotations
        data_has_denotations = node.inputs[0] and get_axis_denotations(node.inputs[0])
        indices_has_denotations = node.inputs[1] and get_axis_denotations(node.inputs[1])

        return data_has_denotations or indices_has_denotations

    def infer_output_denotations(self, node):
        data_tensor = node.inputs[0]
        indices_tensor = node.inputs[1]

        if not data_tensor or not data_tensor.shape:
            return

        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr else 0  # Default axis

        rank = len(data_tensor.shape)
        if axis < 0:
            axis = axis + rank

        data_denotations = self._get_input_denotations(node, 0)
        indices_denotations = self._get_input_denotations(node, 1)

        if not data_denotations:
            data_denotations = [AxisDenotation.UNKNOWN] * rank

        if not indices_denotations:
            if indices_tensor and indices_tensor.shape:
                indices_denotations = [AxisDenotation.UNKNOWN] * len(indices_tensor.shape)
            else:
                indices_denotations = []

        output_denotations = (
            data_denotations[:axis]  # Prefix: dimensions before axis
            + indices_denotations  # Indices dimensions
            + data_denotations[axis + 1 :]  # Suffix: dimensions after axis
        )

        return output_denotations


@register_pass
class InferConvAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Conv"""

    op_types = {"Conv"}

    def infer_output_denotations(self, node):
        input_denotations = self._get_input_denotations(node, 0)

        # Build output denotations
        output_denotations = []
        for i, denotation in enumerate(input_denotations):
            if i == 0:
                # Batch dimension preserved
                output_denotations.append(denotation)
            elif i == 1:
                # Channel dimension transformed by weights
                output_denotations.append(AxisDenotation.UNKNOWN)
            else:
                # Spatial dimensions preserved
                output_denotations.append(denotation)

        return output_denotations


@register_pass
class InferSqueezeAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Squeeze"""

    op_types = {"Squeeze"}

    def infer_output_denotations(self, node):
        input_denotations = self._get_input_denotations(node, 0)

        axes = self._get_squeeze_axes(node)

        # Remove denotations at squeezed axes
        output_denotations = [denotation for i, denotation in enumerate(input_denotations) if i not in axes]

        return output_denotations

    def _get_squeeze_axes(self, node):
        """Get squeeze axis handling opset version >= 13 and opset version < 13"""
        axes = None
        input_shape = node.inputs[0].shape

        # Opset >= 13: axes is second input (optional)
        if len(node.inputs) == 2 and node.inputs[1]:
            axes = get_constant_np(node.inputs[1])

        # Opset < 13: axes is an attribute
        if axes is None:
            axes_attr = node.attributes.get("axes")
            if axes_attr:
                axes = list(axes_attr.as_ints())

        # If no axes specified, squeeze all size-1 dimensions
        if axes is None or len(axes) == 0:
            axes = [i for i, dim in enumerate(input_shape) if dim == 1]

        # Normalize negative axes
        input_rank = len(input_shape)
        axes = [ax if ax >= 0 else input_rank + ax for ax in axes]
        return axes


@register_pass
class InferUnsqueezeAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Unsqueeze"""

    op_types = {"Unsqueeze"}

    def infer_output_denotations(self, node):
        input_denotations = self._get_input_denotations(node, 0)
        output = node.outputs[0]

        if not output or not output.shape:
            return None

        # Get axes to unsqueeze - handle both opset versions
        axes = self._get_unsqueeze_axes(node)
        if axes is None:
            return None

        # Insert UNKNOWN denotations at unsqueezed axes
        output_denotations = []
        input_idx = 0
        for i in range(len(output.shape)):
            if i in axes:
                output_denotations.append(AxisDenotation.UNKNOWN)
            else:
                if input_idx < len(input_denotations):
                    output_denotations.append(input_denotations[input_idx])
                    input_idx += 1
                else:
                    output_denotations.append(AxisDenotation.UNKNOWN)

        return output_denotations

    def _get_unsqueeze_axes(self, node):
        """Get unsqueeze axis handling opset version >= 13 and opset version < 13"""
        axes = None

        # Opset >= 13: axes is second input (required)
        if len(node.inputs) == 2 and node.inputs[1]:
            axes_value = get_constant_np(node.inputs[1])
            if axes_value is not None:
                if isinstance(axes_value, np.ndarray):
                    axes = axes_value.tolist()
                elif isinstance(axes_value, (list, tuple)):
                    axes = list(axes_value)

        # Opset < 13: axes is an attribute
        if axes is None:
            axes_attr = node.attributes.get("axes")
            if axes_attr:
                axes = list(axes_attr.as_ints())

        if axes is None:
            return None

        output_rank = len(node.outputs[0].shape)

        # Normalize negative axes relative to output rank
        axes = [ax if ax >= 0 else output_rank + ax for ax in axes]
        return axes


@register_pass
class InferExpandAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Expand"""

    op_types = {"Expand"}

    def infer_output_denotations(self, node):
        input_denotations = self._get_input_denotations(node, 0)
        input_tensor = node.inputs[0]
        output = node.outputs[0]

        if not input_tensor.shape or not output.shape:
            return None

        input_rank = len(input_tensor.shape)
        output_rank = len(output.shape)

        output_denotations = [AxisDenotation.UNKNOWN] * output_rank

        # If the shape constant (input[1]) is available, use it to locate where
        # each input dimension lands in the output. This handles non-trailing
        # placements such as expanding [AR] → [1, n_heads, AR, head_dim].
        shape_input = node.inputs[1] if len(node.inputs) > 1 else None
        shape_const = get_constant_np(shape_input) if shape_input is not None else None
        if shape_const is not None and len(shape_const) == output_rank:
            input_dims = [int(d) for d in input_tensor.shape]
            shape_dims = [int(v) for v in shape_const]
            # Match each input dim to an output dim by value, right-to-left to
            # respect standard broadcasting precedence for ambiguous values.
            used: set[int] = set()
            for inp_i in range(input_rank - 1, -1, -1):
                denotation = input_denotations[inp_i]
                if denotation == AxisDenotation.UNKNOWN:
                    continue
                inp_dim = input_dims[inp_i]
                best = -1
                for out_i in range(output_rank - 1, -1, -1):
                    if out_i not in used and shape_dims[out_i] == inp_dim:
                        best = out_i
                        break
                if best >= 0:
                    output_denotations[best] = denotation
                    used.add(best)
            return output_denotations

        # Fallback: align input denotations to the right (trailing dimensions)
        rank_diff = output_rank - input_rank
        for i in range(input_rank):
            output_denotations[rank_diff + i] = input_denotations[i]

        return output_denotations


@register_pass
class InferSplitAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Split"""

    op_types = {"Split"}

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        """Override to handle multiple outputs"""
        input_denotations = self._get_input_denotations(node, 0)
        if not input_denotations:
            return False

        # Get split axis
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr else 0

        # Normalize negative axis
        input_tensor = node.inputs[0]
        if not input_tensor or not input_tensor.shape:
            return False

        input_rank = len(input_tensor.shape)
        if axis < 0:
            axis = input_rank + axis

        # All outputs have same denotations as input (split doesn't change denotations)
        changed = False
        for output in node.outputs:
            if output:
                set_axis_denotations(output, input_denotations)
                changed = True

        return changed

    def infer_output_denotations(self, node):
        # Not used
        pass


@register_pass
class InferWhereAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Where"""

    op_types = {"Where"}

    def _has_required_input_denotations(self, node):
        # Need denotations from at least one of the value inputs (X or Y)
        return len(node.inputs) >= 3 and (
            (node.inputs[1] and get_axis_denotations(node.inputs[1]))
            or (node.inputs[2] and get_axis_denotations(node.inputs[2]))
        )

    def infer_output_denotations(self, node):
        # Where(condition, X, Y) - output has same denotations as X or Y
        # Prefer X (input[1]) if it has denotations, otherwise use Y (input[2])
        if len(node.inputs) >= 3 and node.inputs[1] and node.inputs[2]:
            x_denotations = get_axis_denotations(node.inputs[1])
            if x_denotations:
                best_input, best_denotations = node.inputs[1], x_denotations
            else:
                y_denotations = get_axis_denotations(node.inputs[2])
                if y_denotations:
                    best_input, best_denotations = node.inputs[2], y_denotations
                else:
                    return None

            # Label constant inputs with the same denotations
            for inp in node.inputs:
                if inp and inp != best_input and not get_axis_denotations(inp):
                    is_constant = inp.const_value or (inp.producer() and inp.producer().op_type == "Constant")
                    if is_constant and inp.shape and len(inp.shape) == len(best_denotations):
                        set_axis_denotations(inp, best_denotations)

            return best_denotations

        return None


@register_pass
class InferGemmAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Gemm"""

    op_types = {"Gemm"}

    def _has_required_input_denotations(self, node):
        return (
            node.inputs[0]
            and get_axis_denotations(node.inputs[0])
            and node.inputs[1]
            and get_axis_denotations(node.inputs[1])
        )

    def infer_output_denotations(self, node):
        # Gemm: Y = alpha * A' * B' + beta * C
        # A: [M, K] or [K, M] if transA
        # B: [K, N] or [N, K] if transB
        # Output: [M, N]

        a_denotations = self._get_input_denotations(node, 0)
        b_denotations = self._get_input_denotations(node, 1)

        if len(a_denotations) < 2 or len(b_denotations) < 2:
            return None

        # Get transpose attributes
        trans_a_attr = node.attributes.get("transA")
        trans_b_attr = node.attributes.get("transB")
        trans_a = trans_a_attr.as_int() if trans_a_attr else 0
        trans_b = trans_b_attr.as_int() if trans_b_attr else 0

        # Determine output denotations based on transposes
        if trans_a:
            # A is [K, M], so M is at index 1
            m_denotation = a_denotations[1]
        else:
            # A is [M, K], so M is at index 0
            m_denotation = a_denotations[0]

        if trans_b:
            # B is [N, K], so N is at index 0
            n_denotation = b_denotations[0]
        else:
            # B is [K, N], so N is at index 1
            n_denotation = b_denotations[1]

        return [m_denotation, n_denotation]


@register_pass
class InferEinsumAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Einsum"""

    op_types = {"Einsum"}

    def _has_required_input_denotations(self, node):
        # Need at least one input with denotations
        return any(inp and get_axis_denotations(inp) for inp in node.inputs)

    def infer_output_denotations(self, node):
        # Einsum is complex - equation determines output denotations
        # Conservative approach: mark as UNKNOWN unless we can parse equation
        output = node.outputs[0]
        if output and output.shape:
            # TODO: Could parse equation attribute to infer denotations
            return [AxisDenotation.UNKNOWN for _ in output.shape]
        return None


@register_pass
class InferTopKAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for TopK"""

    op_types = {"TopK"}

    def rewrite(self, graph: ir.Graph, node: ir.Node, match_info=None) -> bool:
        """Override rewrite to handle two outputs"""
        input_denotations = self._get_input_denotations(node, 0)
        if not input_denotations:
            return False

        input_tensor = node.inputs[0]
        if input_tensor is None:  # For mypy
            return False

        if not input_tensor.shape:
            return False

        # Get axis attribute (default is -1)
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr else -1

        # Normalize negative axis
        input_rank = len(input_tensor.shape)
        if axis < 0:
            axis = input_rank + axis

        # Output denotations: same as input except the TopK axis becomes UNKNOWN
        output_denotations = input_denotations.copy()
        if 0 <= axis < len(output_denotations):
            output_denotations[axis] = AxisDenotation.UNKNOWN

        # Apply to both outputs (values and indices)
        changed = False
        for output in node.outputs:
            if output:
                set_axis_denotations(output, output_denotations)
                changed = True

        return changed

    def infer_output_denotations(self, node):
        # Not used
        pass


@register_pass
class InferConcatAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Concat"""

    op_types = {"Concat"}

    def _has_required_input_denotations(self, node):
        # At least one input must have denotations
        return any(inp and get_axis_denotations(inp) for inp in node.inputs)

    def infer_output_denotations(self, node):
        # Get concat axis
        axis_attr = node.attributes.get("axis")
        axis = axis_attr.as_int() if axis_attr else 0

        # Get valid input denotations
        valid_inputs = [
            (inp, get_axis_denotations(inp)) for inp in node.inputs if inp and get_axis_denotations(inp)
        ]
        if not valid_inputs:
            return None

        first_denotations = valid_inputs[0][1]

        if not all(len(denots) == len(first_denotations) for _, denots in valid_inputs):
            return None

        # Propagate denotations
        output_denotations = []
        for i in range(len(first_denotations)):
            if i == axis:
                # Special handling for concat axis
                # Collect all denotations at this axis from all inputs
                axis_denotations = [denots[i] for _, denots in valid_inputs]
                all_denotations = set(axis_denotations)

                # Check for PAST_SEQ_LENGTH + SEQ_LENGTH → CONTEXT_LENGTH pattern
                if (
                    AxisDenotation.PAST_SEQ_LENGTH in all_denotations
                    and AxisDenotation.SEQ_LENGTH in all_denotations
                ):
                    output_denotations.append(AxisDenotation.CONTEXT_LENGTH)
                elif len(all_denotations) == 1:
                    # All inputs have same denotation
                    output_denotations.append(axis_denotations[0])
                else:
                    # Mixed denotations - use UNKNOWN
                    output_denotations.append(AxisDenotation.UNKNOWN)
            else:
                # Non-concat axis: denotations must match
                denotations_at_axis = [denots[i] for _, denots in valid_inputs]
                if len(set(denotations_at_axis)) == 1:
                    output_denotations.append(denotations_at_axis[0])
                else:
                    output_denotations.append(AxisDenotation.UNKNOWN)

        return output_denotations


@register_pass
class InferReductionAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Reduce* ops"""

    op_types = {"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"}

    def infer_output_denotations(self, node):
        input_denotations = self._get_input_denotations(node, 0)
        input_tensor = node.inputs[0]
        output = node.outputs[0]

        if not output or not output.shape or not input_tensor.shape:
            return None

        input_rank = len(input_tensor.shape)

        # Get reduction axes - handle both opset versions
        axes = self._get_reduction_axes(node, input_rank)
        if axes is None:
            return None

        # Get keepdims attribute (default is 1)
        keepdims_attr = node.attributes.get("keepdims")
        keepdims = keepdims_attr.as_int() if keepdims_attr else 1

        # Build output denotations
        if keepdims:
            # Reduced dimensions become UNKNOWN, others preserved
            output_denotations = []
            for i, denotation in enumerate(input_denotations):
                if i in axes:
                    output_denotations.append(AxisDenotation.UNKNOWN)
                else:
                    output_denotations.append(denotation)
        else:
            # Reduced dimensions are removed
            output_denotations = [
                denotation for i, denotation in enumerate(input_denotations) if i not in axes
            ]

        return output_denotations

    def _get_reduction_axes(self, node, input_rank):
        """Get Reduce* axis handling opset version >= 13 and opset version < 13"""
        axes = None

        # Opset >= 18: axes is second input (optional)
        if len(node.inputs) == 2 and node.inputs[1]:
            axes_array = get_constant_np(node.inputs[1])
            if axes_array is not None:
                axes = axes_array.tolist()

        # Opset < 18: axes is an attribute
        if axes is None:
            axes_attr = node.attributes.get("axes")
            if axes_attr:
                axes = list(axes_attr.as_ints())

        # Handle noop_with_empty_axes (opset >= 18)
        noop_attr = node.attributes.get("noop_with_empty_axes")
        noop_with_empty_axes = noop_attr.as_int() if noop_attr else 0

        # Determine axes to reduce
        if axes is None or len(axes) == 0:
            if noop_with_empty_axes:
                return []  # No-op case
            else:
                axes = list(range(input_rank))  # Reduce all

        # Normalize negative axes
        axes = [ax if ax >= 0 else input_rank + ax for ax in axes]
        return axes


@register_pass
class InferReshapeAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for Reshape"""

    op_types = {"Reshape"}

    def _infer_reshape_denotations(self, input_shape, output_shape, input_denotations):
        """Infer output denotations for reshape using input/output tensor shapes"""

        # Special case: [1] -> [1,1,1,...,1] preserves denotation in last position
        if (
            len(input_shape) == 1
            and input_shape[0] == 1
            and len(output_shape) > 1
            and all(d == 1 for d in output_shape)
        ):
            output_denotations = [AxisDenotation.UNKNOWN] * len(output_shape)
            output_denotations[-1] = input_denotations[0]
            return output_denotations

        # Build output denotations by matching input to output dimensions
        output_denotations = []
        in_idx = 0
        out_idx = 0

        while out_idx < len(output_shape):
            out_dim = output_shape[out_idx]

            # Skip size-1 output dimensions (unsqueeze)
            if out_dim == 1:
                output_denotations.append(AxisDenotation.UNKNOWN)
                out_idx += 1
                continue

            # Skip squeezed input dimensions (size 1)
            while in_idx < len(input_shape) and input_shape[in_idx] == 1:
                in_idx += 1

            # Check if we've run out of input dimensions
            if in_idx >= len(input_shape):
                output_denotations.append(AxisDenotation.UNKNOWN)
                out_idx += 1
                continue

            in_dim = input_shape[in_idx]

            # Case 1: Exact match
            if in_dim == out_dim:
                output_denotations.append(input_denotations[in_idx])
                in_idx += 1
                out_idx += 1

            # Case 2: Split - one input dim becomes multiple output dims
            elif in_dim > out_dim:
                product = 1
                temp_out_idx = out_idx

                while temp_out_idx < len(output_shape) and product < in_dim:
                    if output_shape[temp_out_idx] == 1:
                        temp_out_idx += 1
                        continue
                    product *= output_shape[temp_out_idx]
                    temp_out_idx += 1

                if product == in_dim:
                    while out_idx < temp_out_idx:
                        if output_shape[out_idx] == 1:
                            output_denotations.append(AxisDenotation.UNKNOWN)
                        else:
                            output_denotations.append(input_denotations[in_idx])
                        out_idx += 1
                    in_idx += 1
                else:
                    return [AxisDenotation.UNKNOWN for _ in output_shape]

            # Case 3: Merge - multiple input dims become one output dim
            else:  # in_dim < out_dim
                # Calculate how many input dims are needed
                product = 1
                start_in_idx = in_idx

                while in_idx < len(input_shape) and product < out_dim:
                    # Skip size-1 dimensions in merge
                    if input_shape[in_idx] != 1:
                        product *= input_shape[in_idx]
                    in_idx += 1

                if product == out_dim:
                    # Collect denotations from non-size-1 merged dimensions
                    merged_denotations = [
                        input_denotations[i] for i in range(start_in_idx, in_idx) if input_shape[i] != 1
                    ]

                    if len(merged_denotations) <= 1:
                        # Single dim (or all size-1): safe to propagate
                        if not merged_denotations:
                            output_denotations.append(AxisDenotation.UNKNOWN)
                        else:
                            output_denotations.append(merged_denotations[0])
                    else:
                        # Multiple non-size-1 dims merged (flatten) — the output
                        # dimension is a product, so no single denotation applies.
                        output_denotations.append(AxisDenotation.UNKNOWN)
                    out_idx += 1
                else:
                    # Can't match exactly
                    return [AxisDenotation.UNKNOWN for _ in output_shape]

        return output_denotations

    def infer_output_denotations(self, node):
        input_denotations = self._get_input_denotations(node, 0)
        input_tensor = node.inputs[0]
        output = node.outputs[0]

        if not input_tensor.shape or not output.shape:
            return None

        return self._infer_reshape_denotations(input_tensor.shape, output.shape, input_denotations)


@register_pass
class InferGatherElementsAxisDenotation(BaseAxisDenotationPass):
    """Infer axis denotations for GatherElements.

    GatherElements output shape == indices shape, so output denotations
    are taken directly from the indices tensor (input 1).
    """

    op_types = {"GatherElements"}

    def _has_required_input_denotations(self, node: ir.Node) -> bool:
        # Propagate if either data or indices has denotations
        data_has = node.inputs[0] is not None and len(get_axis_denotations(node.inputs[0])) != 0
        indices_has = node.inputs[1] is not None and len(get_axis_denotations(node.inputs[1])) != 0
        return data_has or indices_has

    def infer_output_denotations(self, node: ir.Node) -> list[AxisDenotation] | None:
        indices_tensor = node.inputs[1]
        if not indices_tensor or not indices_tensor.shape:
            return None

        indices_denotations = self._get_input_denotations(node, 1)
        if indices_denotations:
            return indices_denotations

        # Indices has no denotations — fall back to data denotations if same rank
        data_tensor = node.inputs[0]
        if data_tensor and data_tensor.shape and len(data_tensor.shape) == len(indices_tensor.shape):
            data_denotations = self._get_input_denotations(node, 0)
            if data_denotations:
                return data_denotations

        return [AxisDenotation.UNKNOWN] * len(indices_tensor.shape)
