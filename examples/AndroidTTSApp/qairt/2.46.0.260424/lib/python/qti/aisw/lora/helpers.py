# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import re
from pathlib import Path
import onnx
from safetensors.numpy import load_file, save_file
from onnx.numpy_helper import to_array
from onnx import numpy_helper
import json
import numpy as np


def is_attach_point(pytorch_module, adapter_target_modules, delimiter="_"):
    """
    Checks if a pytorch module is an attach point given the adapter config's "target_modules".

    Parameters:
        pytorch_module (str): Name of pytorch module.
        adapter_target_modules (list[str]): List of keywords from adapter config to determine if pytorch_module is an attach point.

    Returns:
        bool: True if pytorch_module is an attach point, False otherwise.
    """

    if pytorch_module in adapter_target_modules:
        # this module is specified directly in adapter_target_modules
        return True
    else:
        return any(pytorch_module.endswith(f"{delimiter}{target_key}") or pytorch_module.endswith(f"{delimiter}{target_key}/Conv") for target_key in adapter_target_modules)

def get_attach_points(adapter_target_module, pytorch_modules, delimiter="_"):
    attach_points = [module for module in pytorch_modules if is_attach_point(module, [adapter_target_module], delimiter)]
    return attach_points

def validate_file_path(file_path):
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("The path {} does not exist or is not a file.".format(file_path))


def create_base_graph_name(name):
    base_name = name
    if "_base_layer" in name:
        base_name = name.replace("_base_layer", "")
    elif ".base_layer" in name:
        base_name = name.replace(".base_layer", "")
    elif "/base_layer" in name:
        base_name = name.replace("/base_layer", "")

    return base_name




def apply_safetensors_to_onnx(model, safetensors_dict):
    """
    Apply safetensors into an ONNX model and return the updated model.
    :param model: The original ONNX model to be patched.
    :param safetensors_dict: A dictionary containing safetensors to be applied to the model.
    :return: onnx.ModelProto: The patched ONNX model with safetensors integrated.
    """

    # Overwrite tensors in the ONNX model with those from the safetensors file
    for i, tensor in enumerate(model.graph.initializer):
        if tensor.name in safetensors_dict:
            original_tensor = to_array(tensor)
            # Validate Datatype and shape of the tensor
            if original_tensor.shape != safetensors_dict[tensor.name].shape:
                raise ValueError("Invalid Safetensors file {}: Unable to patched onnx model."
                                 "Shape of the tensor {} in the safetensors file is not matching with the "
                                 "onnx model.".format(safetensors_path, tensor.name))
            if original_tensor.dtype != safetensors_dict[tensor.name].dtype:
                raise ValueError("Invalid Safetensors file {}: Unable to patched onnx model."
                                 "Datatype of the tensor {} in the safetensors file is not matching with the "
                                 "onnx model.".format(safetensors_path, tensor.name))

            # Create a new tensor with the data from the safetensors file
            new_tensor = numpy_helper.from_array(safetensors_dict[tensor.name], name=tensor.name)

            # Replace the old tensor with the new tensor
            model.graph.initializer.remove(tensor)
            model.graph.initializer.insert(i, new_tensor)

    return model


def gate_min_max(min_val: float, max_val: float):
    """
    Gates min and max encoding values to retain zero in the range representation.
    Rules : min at maximum can be zero, max at minimum can be zero and
    if max and min are equal, adds epsilon to maintain range.
    :param min_val: min encoding value
    :param max_val: max encoding value
    :return: gated min and max values
    """

    epsilon = 1e-5
    # For per channel quantization
    if isinstance(min_val, np.ndarray):
        gated_min = np.clip(min_val, None, 0.0)
        gated_max = np.clip(max_val, 0.0, None)
        gated_max = np.clip(gated_max, gated_min + epsilon, None)
    else:
        gated_min = min(min_val, 0.0)
        gated_max = max(max_val, 0.0)
        gated_max = max(gated_max, gated_min + epsilon)

    return gated_min, gated_max


def calculate_v1_scale_offset(
        min_val: float,
        max_val: float,
        bitwidth: int,
        use_symmetric_encodings: bool,
        use_strict_symmetric: bool
):
    """
    Calculates scale and offset given min and max.

    :param min_val: min encoding value
    :param max_val: max encoding value
    :param bitwidth: bitwidth used for quantization
    :param use_symmetric_encodings: use_symmetric_encodings flag
    :param use_strict_symmetric: use_strict_symmetric flag
    :return: scale and offset values computed
    """
    num_steps = 2**bitwidth - 1
    if use_symmetric_encodings and use_strict_symmetric:
        num_steps -= 1

    min_val, max_val = gate_min_max(min_val, max_val)

    # Use only max val to compute delta in the case of signed symmetric
    if use_symmetric_encodings and min_val < 0:
        num_positive_steps = np.floor(num_steps / 2)
        delta = max_val / num_positive_steps
        offset = -num_positive_steps
        if not use_strict_symmetric:
            offset -= 1
    else:
        delta = (max_val - min_val) / num_steps
        offset = round(min_val / delta)

    return delta, offset

def calculate_v2_scale_zero_point(
    min_val: float,
    max_val: float,
    bitwidth: int,
    use_symmetric_encodings: bool,
    use_strict_symmetric: bool,
    dtype: str
):
    v1_scale, v1_offset = calculate_v1_scale_offset(
        min_val,
        max_val,
        bitwidth,
        use_symmetric_encodings,
        use_strict_symmetric)

    if dtype.lower() == "uint":
        v2_zero_point = -v1_offset
    elif dtype.lower() == "int":
        v2_zero_point = (-v1_offset) - 2**(bitwidth - 1)
    else:
        raise ValueError(f"Dtype {dtype} is unrecognized.")

    v2_scale = v1_scale

    return v2_scale, v2_zero_point


def open_and_load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
