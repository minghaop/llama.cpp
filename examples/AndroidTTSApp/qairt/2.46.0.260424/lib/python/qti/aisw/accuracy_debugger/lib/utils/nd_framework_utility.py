# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
from typing import Tuple
import json
from importlib import import_module
import os
from logging import Logger
import psutil
import onnx

from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import UnsupportedError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import FrameworkExtension
from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_objects import get_available_frameworks
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import MaxLimits, DataType

import sys
if sys.version_info < (3, 8):
    # distutils deprecated for Python 3.8 and up
    from distutils.version import StrictVersion as Version
else:
    # packaging requires Python 3.8 and up
    from packaging.version import Version as Version

GB = 1024 * 1024 * 1024
MAX_RAM_LIMIT = psutil.virtual_memory()[1]  # Available RAM memory in the system


def dump_intermediate_tensors(output_dir: str, result: dict, logger: Logger, use_native_output_files: bool) -> None:
    '''
    dumps the tensors present in the result dictionary in the output directory in <sanitized_tensor_name.raw> format

    :param output_dir: path to the output directory where the tensors will be dumped in .raw format
    :param result: dictionary of tensor name and tensor data as values
    :param logger: object of logging.Logger
    :param use_native_output_files: dumps outputs as per framework model's actual data types
    '''
    logger.debug(f'Dumping outputs with use_native_output_files={use_native_output_files}')
    data_path = os.path.join(output_dir, '{}{}')
    for output_tensor_name, data in result.items():
        # Sanitize output tensor name to avoid dumping outputs in sub-folders.
        sanitized_output_tensor_name = santize_node_name(output_tensor_name)

        # Most of the systems does not allow to create a file name, more than 255 bytes, hence skipping
        # to dump those tensor files.
        if len((sanitized_output_tensor_name +
                '.raw').encode('utf-8')) > MaxLimits.max_file_name_size.value:
            logger.warning(
                f"Skipping dumping of tensor {sanitized_output_tensor_name} as filename exceeds max limit {MaxLimits.max_file_name_size.value}"
            )
            continue
        file_path = data_path.format(sanitized_output_tensor_name, '.raw')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            save_outputs(data, file_path, use_native_output_files)
        except Exception as e:
            logger.error(
                f"Dumping output raw file {sanitized_output_tensor_name + '.raw'} for tensor {output_tensor_name} failed"
            )


def dump_profile_json(output_dir: str, result: str, use_native_output_files: bool = True) -> None:
    '''
    dumps the tensors present in the result dictionary in the output directory in <sanitized_tensor_name.raw> format

    :param output_dir: path to the output directory where the tensors will be dumped in .raw format
    :param result: dictionary of tensor name and tensor data as values
    :param use_native_output_files: when True, function returns datatypes as per framework model's
                                    actual data types, otherwise returns datatype as FP32
    '''
    profile_info_json_path = os.path.join(output_dir, 'profile_info.json')
    if os.path.exists(profile_info_json_path):
        # For the case for model > 2GB
        tensor_info = read_json(profile_info_json_path)
    else:
        tensor_info = {}

    for output_tensor_name, data in result.items():
        santized_tensor_name = santize_node_name(output_tensor_name)
        try:
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
        except Exception as e:
            raise Exception(f"Encountered Error: {e}")

        if not data.size or data.dtype == bool:
            if data.size == 0:
                tensor_info[santized_tensor_name] = (
                    '-',
                    '-',
                    '-',
                    '-',
                    '-',
                )
            else:
                # When --use_native_output_files is used, outputs are expected to be in same datatype as in framework model
                # When --use_native_output_files is not used, outputs are expected to be in FP32
                data_type = str(data.dtype) if use_native_output_files else DataType.DEFAULT_OUTPUTS_DATATYPE.value
                tensor_info[santized_tensor_name] = (data_type,
                                                     data.shape, data.tolist(),
                                                     data.tolist(), data.tolist())
        else:
            # When --use_native_output_files is used, outputs are expected to be in same datatype as in framework model
            # When --use_native_output_files is not used, outputs are expected to be in FP32
            data_type = str(data.dtype) if use_native_output_files else DataType.DEFAULT_OUTPUTS_DATATYPE.value
            tensor_info[santized_tensor_name] = (data_type,
                                                 data.shape,
                                                 str(round(np.min(data),
                                                           3)), str(round(np.max(data), 3)),
                                                 str(round(np.median(data), 3)))

    dump_json(tensor_info, profile_info_json_path)


def load_inputs(data_path, data_type, data_dimension=None):
    # type:  (str, str, Tuple) -> np.ndarray
    data = np.fromfile(data_path, data_type)
    if data_dimension is not None:
        data = data.reshape(data_dimension)
    return data


def save_outputs(data: np.ndarray, data_path: str, use_native_output_files: bool) -> None:
    '''
    Dumps given numpy data to disk, if use_native_output_files is False then data will be typecasted
    to float32 before dumping

    :param data: output tensor as numpy array
    :param data_path: path to dump given data
    :param use_native_output_files: dumps outputs as per framework model's actual data types
    '''
    if use_native_output_files is False:
        data = data.astype(DataType.DEFAULT_OUTPUTS_DATATYPE.value)

    data.tofile(data_path)


def dequantize_tflite_tensor(tensor_data: np.ndarray, tensor_detail: dict) -> np.ndarray:
    """
    Dequantizes TFLite tensor data from quantized integer format to floating-point format.

    This function handles both per-tensor and per-channel quantization schemes commonly used
    in TensorFlow Lite models. It extracts quantization parameters (scale and zero_point) from
    the tensor details and applies the dequantization formula:
        dequantized_value = (quantized_value - zero_point) * scale

    For per-channel quantization, different scale and zero_point values are applied to each
    channel along the specified quantized dimension.

    :param tensor_data: Raw tensor data from TFLite interpreter, typically in int8/uint8 format
                       for quantized tensors or float32 for non-quantized tensors
    :param tensor_detail: Dictionary containing tensor metadata from interpreter.get_tensor_details(),
                         which includes 'quantization_parameters' with 'scales', 'zero_points',
                         and optionally 'quantized_dimension' for per-channel quantization

    :return: Dequantized tensor in float32 format. If the tensor is not quantized (already float
            or missing quantization parameters), returns the original tensor unchanged.

    :raises: None - function handles edge cases by returning original data

    Example usage:
        tensor_details = interpreter.get_tensor_details()
        tensor_detail = next((t for t in tensor_details if t['index'] == tensor_index), None)
        raw_output = interpreter.get_tensor(tensor_index)
        dequantized_output = dequantize_tflite_tensor(raw_output, tensor_detail)

    Notes:
        - Per-tensor quantization: Single scale/zero_point applied to entire tensor
        - Per-channel quantization: Different scale/zero_point per channel (common for Conv/Dense weights)
        - Only processes integer dtypes (int8, uint8, int16, uint16); returns float types unchanged
        - Handles missing or empty quantization parameters
    """
    if tensor_detail is None:
        return tensor_data

    # Extract quantization parameters from tensor details
    quantization = tensor_detail.get('quantization_parameters', {})
    scales = quantization.get('scales', [])
    zero_points = quantization.get('zero_points', [])

    # Only dequantize if:
    # 1. Quantization parameters exist (non-empty scales and zero_points)
    # 2. Data is in quantized integer format (int8, uint8, int16, uint16)
    if len(scales) > 0 and len(zero_points) > 0 and tensor_data.dtype in [np.int8, np.uint8, np.int16, np.uint16]:

        # Convert quantized integer data to float32 for dequantization computation
        dequantized = tensor_data.astype(np.float32)

        if len(scales) == 1:
            # Per-Tensor Quantization:
            # Single scale and zero_point applied uniformly to all tensor elements
            # Formula: dequantized = (quantized - zero_point) * scale
            scale = scales[0]
            zero_point = zero_points[0]
            dequantized = (dequantized - zero_point) * scale

        else:
            # Per-Channel Quantization:
            # Different scale and zero_point for each channel along a specific dimension
            # This is Common in convolutional and dense layer weights for better accuracy,
            # its not for activations generally.

            # Get the dimension along which quantization varies (typically 0 for output channels)
            quantized_dimension = quantization.get('quantized_dimension', 0)

            # Reshape scales and zero_points for numpy broadcasting
            # Create a shape with 1s everywhere except the quantized dimension
            # Example: For tensor shape [32, 3, 3, 16] with quantized_dimension=0,
            #          reshape scales from [32] to [32, 1, 1, 1] for broadcasting
            shape = [1] * len(tensor_data.shape)
            shape[quantized_dimension] = len(scales)

            scales_reshaped = np.array(scales).reshape(shape)
            zero_points_reshaped = np.array(zero_points).reshape(shape)

            # Apply per-channel dequantization using numpy broadcasting
            # Each channel gets its own scale and zero_point automatically
            dequantized = (dequantized - zero_points_reshaped) * scales_reshaped

        return dequantized

    # Return original data if:
    # - Tensor is not quantized (no quantization parameters)
    # - Tensor is already in floating-point format
    # - Quantization parameters are empty or invalid
    return tensor_data


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data


def dump_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def transpose_to_nhwc(data, data_dimension):
    # type:  (np.ndarray, list) ->np.ndarray
    if len(data_dimension) == 4:
        data = np.reshape(
            data, (data_dimension[0], data_dimension[1], data_dimension[2], data_dimension[3]))
        data = np.transpose(data, (0, 2, 3, 1))
        data = data.flatten()
    return data


class ModelHelper:

    @classmethod
    def onnx_type_to_numpy(cls, type):
        """
        This method gives the corresponding numpy datatype for given onnx tensor element type
        Args:
            type : onnx tensor element type
        Returns:
            corresponding onnx datatype
        """
        onnx_to_numpy = {
            '1': (np.float32, 4),
            '2': (np.uint8, 1),
            '3': (np.int8, 1),
            '4': (np.uint16, 2),
            '5': (np.int16, 2),
            '6': (np.int32, 4),
            '7': (np.int64, 8),
            '9': (np.bool_, 1)
        }
        if type in onnx_to_numpy:
            return onnx_to_numpy[type]
        else:
            raise UnsupportedError('Unsupported type : {}'.format(str(type)))


def get_framework_info(model_path):
    """
    Tries to find framework name of given model_path based on it's extension.
    Returns: Framework name (None if not able to find framework name)
    """
    if model_path is None:
        return None
    extenstion_framework_map = {
        v: k
        for k, v in FrameworkExtension.framework_extension_mapping.items()
    }
    model_extension = '.' + model_path.rsplit('.', 1)[-1]
    return extenstion_framework_map.get(model_extension, None)


def extract_input_information(input_tensor):
    input_info = {}
    in_list = list(zip(*input_tensor))
    if len(in_list) == 4:
        (in_names, in_dims, in_data_paths, in_types) = in_list
    elif len(in_list) == 3:
        (in_names, in_dims, in_data_paths) = in_list
        in_types = None
    else:
        raise FrameworkError(get_message('ERROR_FRAMEWORK_RUNNER_INPUT_TENSOR_LENGHT_ERROR'))

    input_names = list(in_names)
    input_dims = [[int(x) for x in dim.split(',')] for dim in in_dims]

    if len(input_names) != len(input_dims):
        return None

    for i, input_name in enumerate(input_names):
        input_info[input_name] = input_dims[i]

    return input_info


def max_version(framework, available_frameworks):
    versions = available_frameworks.get(framework, {})
    return max(versions.keys(), key=lambda x: Version(x))


def simplify_onnx_model(logger, model_path=None, input_tensor=None, output_dir=None,
                        custom_op_lib=None):
    framework = 'onnx'
    available_frameworks = get_available_frameworks()
    version = max_version(framework, available_frameworks)
    module, framework_class = available_frameworks[framework][version]
    framework_type = getattr(import_module(module), framework_class)
    framework_instance = framework_type(logger, custom_op_lib=custom_op_lib)

    optimized_model_path = os.path.join(
        output_dir, "optimized_model" + FrameworkExtension.framework_extension_mapping[framework])
    input_information = extract_input_information(input_tensor)
    _, optimized_model_path = framework_instance.optimize(model_path, optimized_model_path,
                                                          input_information)

    return optimized_model_path


def fetch_onnx_intermediate_info(model_path: str, sanitize_node_names: bool = False, use_native_output_files: bool = True) -> dict:
    '''
    This function tries to fetch details like datatypes, dimensions/shapes of all intermediate
    nodes present in the given onnx model.

    :param model_path: path to the framework model file
    :param sanitize_node_names: when True, returns tensor details with santized names
    :param use_native_output_files: when True, function returns datatypes as per framework model's
                                    actual data types, otherwise returns datatype as FP32

    Returns: Dictionary containing datatypes, dimensions of all intermediate outputs
    '''

    DATATYPE_MAP = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.BOOL: np.bool_,
    }

    intermediates_info = {}
    model = onnx.load(model_path)
    model = onnx.shape_inference.infer_shapes(model)
    model.graph.value_info.extend(model.graph.output) # Appends final outputs info as well
    for value_info in model.graph.value_info:
        node_name = value_info.name
        node_name = santize_node_name(node_name) if sanitize_node_names else node_name

        shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]

        if use_native_output_files:
            # When --use_native_output_files is used, outputs are expected to be in same datatype
            # as in framework model
            datatype_in_onnx = value_info.type.tensor_type.elem_type
            datatype = DATATYPE_MAP.get(datatype_in_onnx, None).__name__
        else:
            # When --use_native_output_files is not used, outputs are expected to be in FP32
            datatype = DataType.DEFAULT_OUTPUTS_DATATYPE.value

        # Intermediate info data format: [<DATA_TYPE>, <SHAPE>, <MIN>, <MAX>, <MEDIAN>]
        intermediates_info[node_name] = [datatype, shape, '-', '-', '-']

    return intermediates_info
