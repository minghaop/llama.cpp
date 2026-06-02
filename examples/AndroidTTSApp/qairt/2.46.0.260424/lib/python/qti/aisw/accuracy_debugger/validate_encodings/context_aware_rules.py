# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""Context-aware validation rules for encoding validation.

These rules require access to the full model encoding and DLC graph structure
to validate relationships between tensors (e.g., reshape/transpose operations
should match predecessor encodings).
"""

from typing import Dict, List, Tuple

from qti.aisw.accuracy_debugger.encodings.encodings import ModelEncoding, TensorEncoding
from qti.aisw.accuracy_debugger.encodings.encodings_utils import TensorType
from qti.aisw.accuracy_debugger.graph_op.dlc_graph_utils import DLCConnectedGraph


def _encodings_match(
    tensor1: TensorEncoding, tensor2: TensorEncoding, tolerance: float = 1e-6
) -> bool:
    """Check if two tensor encodings have the same quantization range.

    Args:
        tensor1: First TensorEncoding object.
        tensor2: Second TensorEncoding object.
        tolerance: Tolerance for scale comparison.

    Returns:
        bool: True if encodings match (same scale and offset).
    """
    # Check if both have scales
    if not tensor1.scale or not tensor2.scale:
        return True  # Can't compare if no scale

    # Check if number of channels match
    if len(tensor1.scale) != len(tensor2.scale):
        return False

    # Check if scales and offsets match within tolerance
    for s1, s2, o1, o2 in zip(tensor1.scale, tensor2.scale, tensor1.offset, tensor2.offset):
        if abs(s1 - s2) > tolerance or abs(o1 - o2) > tolerance:
            return False

    return True


def _find_predecessor_tensor(
    tensor_name: str, all_encodings: Dict[str, TensorEncoding], dlc_graph: DLCConnectedGraph | None
) -> TensorEncoding | None:
    """Find the predecessor tensor for a given tensor using DLC graph.

    Args:
        tensor_name: Name of the tensor to find predecessor for.
        all_encodings: Dictionary of all tensor encodings.
        dlc_graph: DLC graph information object.

    Returns:
        TensorEncoding | None: Predecessor tensor encoding if found, None otherwise.
    """
    if not dlc_graph:
        # Fallback to heuristic if no DLC graph available
        return _find_predecessor_tensor_heuristic(tensor_name, all_encodings)

    # Use DLC graph to find predecessor
    predecessors = dlc_graph.find_predecessor_tensors(tensor_name)
    if predecessors:
        # Return the first predecessor that has an encoding
        for pred_name in predecessors:
            if pred_name in all_encodings:
                return all_encodings[pred_name]

    return None


def _find_predecessor_tensor_heuristic(
    tensor_name: str, all_encodings: Dict[str, TensorEncoding]
) -> TensorEncoding | None:
    """Find the predecessor tensor using heuristic approach (fallback).

    This is a heuristic approach that tries to find the input tensor based on naming patterns.

    Args:
        tensor_name: Name of the tensor to find predecessor for.
        all_encodings: Dictionary of all tensor encodings.

    Returns:
        TensorEncoding | None: Predecessor tensor encoding if found, None otherwise.
    """
    # Try to find predecessor by removing operation suffix
    # E.g., "layer.reshape" -> "layer", "layer.transpose" -> "layer"
    name_lower = tensor_name.lower()

    # Common operation patterns
    operations = ["reshape", "transpose", "permute", "view", "flatten"]

    for op in operations:
        if op in name_lower:
            # Try to find the base name
            parts = tensor_name.split(".")
            for i in range(len(parts) - 1, -1, -1):
                if op in parts[i].lower():
                    # Try predecessor name without this part
                    if i > 0:
                        predecessor_name = ".".join(parts[:i])
                        if predecessor_name in all_encodings:
                            return all_encodings[predecessor_name]
                    break

    # Try to find by removing last part
    if "." in tensor_name:
        parts = tensor_name.rsplit(".", 1)
        predecessor_name = parts[0]
        if predecessor_name in all_encodings:
            return all_encodings[predecessor_name]

    return None


def _find_concat_inputs(
    tensor_name: str, all_encodings: Dict[str, TensorEncoding], dlc_graph: DLCConnectedGraph | None
) -> List[TensorEncoding]:
    """Find input tensors for a concat operation using DLC graph.

    Args:
        tensor_name: Name of the concat tensor.
        all_encodings: Dictionary of all tensor encodings.
        dlc_graph: DLC graph information object.

    Returns:
        List[TensorEncoding]: List of input tensor encodings.
    """
    if not dlc_graph:
        # Fallback to heuristic if no DLC graph available
        return _find_concat_inputs_heuristic(tensor_name, all_encodings)

    # Use DLC graph to find concat inputs
    input_names = dlc_graph.find_concat_inputs(tensor_name)
    inputs = []
    for input_name in input_names:
        if input_name in all_encodings:
            inputs.append(all_encodings[input_name])

    return inputs


def _find_concat_inputs_heuristic(
    tensor_name: str, all_encodings: Dict[str, TensorEncoding]
) -> List[TensorEncoding]:
    """Find input tensors for a concat operation using heuristic approach (fallback).

    This is a heuristic approach that tries to find input tensors based on naming patterns.

    Args:
        tensor_name: Name of the concat tensor.
        all_encodings: Dictionary of all tensor encodings.

    Returns:
        List[TensorEncoding]: List of input tensor encodings.
    """
    inputs = []

    # Try to find inputs by looking for numbered inputs
    # E.g., "layer.concat" might have inputs "layer.input.0", "layer.input.1", etc.
    base_name = tensor_name.replace("concat", "input").replace("cat", "input")

    # Try numbered inputs
    for i in range(10):  # Check up to 10 inputs
        input_name = f"{base_name}.{i}"
        if input_name in all_encodings:
            inputs.append(all_encodings[input_name])
        else:
            # Also try without the dot
            input_name = f"{base_name}{i}"
            if input_name in all_encodings:
                inputs.append(all_encodings[input_name])

    # If no numbered inputs found, try to find by prefix matching
    if not inputs:
        # Remove the concat part and look for tensors with similar prefix
        if "." in tensor_name:
            prefix = tensor_name.rsplit(".", 1)[0]
            for name, encoding in all_encodings.items():
                if name.startswith(prefix) and name != tensor_name:
                    # Check if it's likely an input (not the output)
                    if "input" in name.lower() or name.count(".") == tensor_name.count("."):
                        inputs.append(encoding)

    return inputs


def rule_reshape_encoding_matches_predecessor(
    tensor_name: str,
    tensor: TensorEncoding,
    model_encoding: ModelEncoding,
    dlc_graph: DLCConnectedGraph | None = None,
) -> Tuple[bool, str | None]:
    """Validation rule: Reshape operation encoding should match its predecessor.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.
        model_encoding: Full ModelEncoding object for context.
        dlc_graph: Optional DLC graph information for accurate graph traversal.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
    """
    # Check if this is a reshape operation using DLC graph or heuristic
    is_reshape = False
    if dlc_graph:
        is_reshape = dlc_graph.is_reshape_op(tensor_name)
    else:
        # Fallback to name-based heuristic
        name_lower = tensor_name.lower()
        is_reshape = "reshape" in name_lower

    if not is_reshape:
        return True, None

    # Get all activation encodings
    all_encodings = model_encoding.get_type_encodings(tensor_type=TensorType.ActivationEncodings)
    if not all_encodings:
        all_encodings = model_encoding.get_type_encodings(tensor_type=TensorType.Encodings)

    # Find predecessor
    predecessor = _find_predecessor_tensor(tensor_name, all_encodings, dlc_graph)
    if not predecessor:
        # Can't validate without predecessor
        return True, None

    # Check if encodings match
    is_valid = _encodings_match(tensor, predecessor)
    if not is_valid:
        return (
            False,
            f"Reshape operation encoding should match its predecessor "
            f"(predecessor scale: {predecessor.scale}, current scale: {tensor.scale})",
        )

    return True, None


def rule_transpose_encoding_matches_predecessor(
    tensor_name: str,
    tensor: TensorEncoding,
    model_encoding: ModelEncoding,
    dlc_graph: DLCConnectedGraph | None = None,
) -> Tuple[bool, str | None]:
    """Validation rule: Transpose operation encoding should match its predecessor.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.
        model_encoding: Full ModelEncoding object for context.
        dlc_graph: Optional DLC graph information for accurate graph traversal.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
    """
    # Check if this is a transpose operation using DLC graph or heuristic
    is_transpose = False
    if dlc_graph:
        is_transpose = dlc_graph.is_transpose_op(tensor_name)
    else:
        # Fallback to name-based heuristic
        name_lower = tensor_name.lower()
        is_transpose = "transpose" in name_lower or "permute" in name_lower

    if not is_transpose:
        return True, None

    # Get all activation encodings
    all_encodings = model_encoding.get_type_encodings(tensor_type=TensorType.ActivationEncodings)
    if not all_encodings:
        all_encodings = model_encoding.get_type_encodings(tensor_type=TensorType.Encodings)

    # Find predecessor
    predecessor = _find_predecessor_tensor(tensor_name, all_encodings, dlc_graph)
    if not predecessor:
        # Can't validate without predecessor
        return True, None

    # Check if encodings match
    is_valid = _encodings_match(tensor, predecessor)
    if not is_valid:
        return (
            False,
            f"Transpose operation encoding should match its predecessor "
            f"(predecessor scale: {predecessor.scale}, current scale: {tensor.scale})",
        )

    return True, None


def rule_concat_inputs_same_range(
    tensor_name: str,
    tensor: TensorEncoding,
    model_encoding: ModelEncoding,
    dlc_graph: DLCConnectedGraph | None = None,
) -> Tuple[bool, str | None]:
    """Validation rule: Concat operation inputs should have the same quantization range.

    Args:
        tensor_name: Name of the tensor.
        tensor: TensorEncoding object.
        model_encoding: Full ModelEncoding object for context.
        dlc_graph: Optional DLC graph information for accurate graph traversal.

    Returns:
        Tuple[bool, str | None]: (is_valid, rule_description)
    """
    # Check if this is a concat operation using DLC graph or heuristic
    is_concat = False
    if dlc_graph:
        is_concat = dlc_graph.is_concat_op(tensor_name)
    else:
        # Fallback to name-based heuristic
        name_lower = tensor_name.lower()
        is_concat = "concat" in name_lower or "cat" in name_lower

    if not is_concat:
        return True, None

    # Get all activation encodings
    all_encodings = model_encoding.get_type_encodings(tensor_type=TensorType.ActivationEncodings)
    if not all_encodings:
        all_encodings = model_encoding.get_type_encodings(tensor_type=TensorType.Encodings)

    # Find concat inputs
    inputs = _find_concat_inputs(tensor_name, all_encodings, dlc_graph)
    if len(inputs) < 2:
        # Can't validate without at least 2 inputs
        return True, None

    # Check if all inputs have the same quantization range
    first_input = inputs[0]
    for i, input_tensor in enumerate(inputs[1:], start=1):
        if not _encodings_match(first_input, input_tensor):
            return (
                False,
                f"Concat operation inputs should have the same quantization range "
                f"(input 0 scale: {first_input.scale}, input {i} scale: {input_tensor.scale})",
            )

    return True, None


# Context-aware rule collection
CONTEXT_AWARE_RULES = [
    rule_reshape_encoding_matches_predecessor,
    rule_transpose_encoding_matches_predecessor,
    rule_concat_inputs_same_range,
]
