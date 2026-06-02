# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.encodings.encodings import TensorEncoding
from qti.aisw.accuracy_debugger.encodings.encodings_utils import TensorDtype


def get_resolved_names(tensor_name: str) -> list:
    """Generates a list of resolved tensor names based on the input tensor name.

    Args:
        tensor_name (str): QNN tensor name.

    Returns:
        list: A list of resolved tensor names derived from the input tensor name.
    """
    # TODO: use framework_op_trace to resolve the target name once the feature is stable

    resolved_names = []

    if "_" in tensor_name:
        resolved_names.append("_".join(tensor_name.split("_")[:-1]))

    if "." in tensor_name:
        resolved_names.append(".".join(tensor_name.split(".")[:-1]))

    return resolved_names


def needs_encoding_update(
    tensor_encoding1: TensorEncoding, tensor_encoding2: TensorEncoding
) -> bool:
    """Determines whether encoding1 should overwrite encoding2 based on precedence.

    The precedence order is: int16 > int8 > int4 > fp32 > fp16 > fp8.

    Args:
        tensor_encoding1 (TensorEncoding): Encoding for a tensor
        tensor_encoding2 (TensorEncoding): Encoding for a tensor

    Returns:
        bool: True if encoding1 should overwrite encoding2, False otherwise.

    """
    dtype_order = {
        TensorDtype.SFXP: [4, 8, 16],
        TensorDtype.UFXP: [4, 8, 16],
        TensorDtype.FLOAT: [8, 16, 32],
    }

    # float only support [8, 16, 32] bitwidths
    # int/uint only support [4, 8, 16] bitwidths
    if (
        tensor_encoding1.bitwidth not in dtype_order[tensor_encoding1.dtype]
        or tensor_encoding2.bitwidth not in dtype_order[tensor_encoding2.dtype]
    ):
        raise ValueError("Check dtype and bitwidth combination of tensor encoding.")

    int_dtypes = [TensorDtype.SFXP, TensorDtype.UFXP]

    # Check if both data types are the same or both of type int and compare bitwidths
    if tensor_encoding1.dtype == tensor_encoding2.dtype or (
        tensor_encoding1.dtype in int_dtypes and tensor_encoding2.dtype in int_dtypes
    ):
        return dtype_order[tensor_encoding1.dtype].index(tensor_encoding1.bitwidth) > dtype_order[
            tensor_encoding2.dtype
        ].index(tensor_encoding2.bitwidth)

    # Precedence: int types have higher precedence over float types
    return tensor_encoding1.dtype in int_dtypes


def identify_inter_activations_path(
    current_activation: str, parent_activation_name: str, target_activation_op_map: dict, depth: int
) -> list:
    """Identifies the path between child op activation and target op activation in the target graph.

    Args:
        current_activation: Child op activation in the target graph.
        parent_activation_name: Parent op activation in the target graph.
        target_activation_op_map: A mapping of target activations (keys) to target ops (values).
        depth: The current number of ops in the path between parent and child ops. If greater than
            10, the path is dropped as it may indicate loops.

    Returns:
        list: A list representing the path of activations.
    """
    # Base case: if the current activation is the parent activation
    if current_activation == parent_activation_name:
        return [parent_activation_name]

    # Base case: if the depth exceeds 10, return an empty path to avoid potential loops
    if depth >= 10:
        return []

    # Initialize the smallest path as empty
    shortest_path = []

    if current_activation in target_activation_op_map:
        current_target_op = target_activation_op_map[current_activation]

        # Iterate through the inputs of the current target op
        for op_input in current_target_op.inputs:
            path = identify_inter_activations_path(
                op_input, parent_activation_name, target_activation_op_map, depth + 1
            )
            #         |------------------>|
            # 100 --->|                   |-----> 103
            #         |--> 101 --> 102 -->|
            # Incase of residual connections, path between 100 and 103
            # should be {100, 103} but one other possible path is
            # {100, 101, 102, 103}. Therefore, we need to take the shortest path

            if path:
                if not shortest_path:
                    shortest_path = path
                else:
                    shortest_path = shortest_path if len(shortest_path) < len(path) else path

            # # Only update the smallest path if a valid path is found and it's shorter
            # if path and (not shortest_path or len(path) < len(shortest_path)):
            #     shortest_path = path

    if shortest_path:
        shortest_path.append(current_activation)

    return shortest_path


def is_convert_op_in_path(path: list, target_activation_op_map: dict) -> tuple:
    """Checks if there exists a convert operation in the path.

    Args:
        path: List of target activations.
        target_activation_op_map: Dictionary of target activations to target ops.

    Returns:
        tuple: (bool, str) indicating if 'Convert' op is found, and the activation name.
    """
    for activation in path:
        op = target_activation_op_map.get(activation)
        if op.op_type == "Convert":
            # if op and "convert" in op:
            return True, activation
    return False, None
