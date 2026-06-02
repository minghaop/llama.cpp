# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
this module provides helper functions for the validation
"""

import os
from collections import OrderedDict

import numpy as np
import onnx

from qairt.optimizer.onnx.utils.utils import (
    get_shape_from_value_info_proto,
)


def _check_static_numerical_shape(shape, name) -> list[int]:
    for s in shape:
        assert isinstance(s, int), f"shape should be statical number, got {shape} for {name}"
    return shape


def read_inputs_raw_paths(input_raw_list_path: str, input_base_dir: str | None = None):
    """
    get inputs raw paths from the input list

    Args:
        input_raw_list_path: input list path
        input_base_dir: the overriding input raw file base directory
    Return:
        list of inputs raw paths
    """
    inputs_list = []
    with open(input_raw_list_path, "r", encoding="utf-8") as f:
        for line in f:
            inputs = {}
            for kv in line.strip().split(" "):
                k, v = kv.strip().split(":=")
                if input_base_dir is not None:
                    inputs[k] = os.path.join(input_base_dir, v)
                else:
                    inputs[k] = v
            inputs_list.append(inputs)

    return inputs_list


def generate_random_input(
    model: onnx.ModelProto, special_inputs=None | dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Generate random input of the model
    Some inputs can be directly indicated by special_inputs

    Args:
        model: onnx model proto
        special_inputs: inputs indicate directly
    Returns:
        generated inputs
    """
    if special_inputs is None:
        special_inputs = {}
    initializer_names = [x.name for x in model.graph.initializer]
    input_node = [ipt for ipt in model.graph.input if ipt.name not in initializer_names]
    inputs_data = OrderedDict()
    for model_input in input_node:
        input_name = model_input.name
        input_shape_ = get_shape_from_value_info_proto(model_input, False)
        input_shape = _check_static_numerical_shape(input_shape_, input_name)
        input_dtype = onnx.helper.tensor_dtype_to_np_dtype(model_input.type.tensor_type.elem_type)

        if input_name == "attention_mask":
            inputs_data[input_name] = np.zeros(input_shape).astype(input_dtype)
        elif input_name == "input_ids":
            inputs_data[input_name] = np.random.randint(0, 10, size=input_shape).astype(input_dtype)
        elif np.issubdtype(input_dtype, np.integer):
            inputs_data[input_name] = np.random.randint(0, 10, size=input_shape).astype(input_dtype)
        else:
            data = np.random.rand(*input_shape)
            if not isinstance(data, np.ndarray):
                # when input_shape = [], data is a float number
                data = np.array(data)
            inputs_data[input_name] = data.astype(input_dtype)

        if input_name in special_inputs:
            inputs_data[input_name] = (
                np.array(special_inputs[input_name]).reshape(input_shape).astype(input_dtype)
            )

    return inputs_data


def get_cosine_similarity(a: np.ndarray, b: np.ndarray):
    """
    compute cosine similarity between two tensors

    Args:
        A: tensor A
        B: tensor B
    Returns:
        cosine similarity
    """
    return (np.dot(a, b) + 1e-9) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
