# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import contextlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from qairt.api.converter.converter_config import CalibrationConfig
from qairt.utils.loggers import get_logger
from qti.aisw.tools.core.modules.converter.converter_module import (
    InputTensorConfig as ModuleInputTensorConfig,
)
from qti.aisw.tools.core.modules.converter.converter_module import (
    OutputTensorConfig as ModuleOutputTensorConfig,
)

_convert_logger = get_logger("qairt.convert")


def _rename_arg(args: Dict, api_arg: str, module_arg: str) -> None:
    """
    Renames an argument from API naming to module naming.

    Args:
        args (Dict): The arguments dict containing the API arguments.
        api_arg (str): The name of the API argument.
        module_arg (str): The name of the module argument.

    Returns:
        None: Modifies args dictionary in-place
    """
    args[module_arg] = args.pop(api_arg)


def _tensor_config_arg_adapter(
    tensor_dict: Dict, tensor_type: str
) -> Union[ModuleInputTensorConfig, ModuleOutputTensorConfig]:
    """
    Converts TensorConfig dict to ModuleTensorConfig object.

    Args:
        tensor_dict: Dict containing the tensor configuration.
        tensor_type: str indicating the type of tensor ('input' or 'output').

    Returns:
        Union[ModuleInputTensorConfig, ModuleOutputTensorConfig]: The converted ModuleTensorConfig object.
    """

    # Check if tensor_type is valid
    if tensor_type not in ["input", "output"]:
        raise ValueError(f"Invalid tensor_type: {tensor_type}. Must be 'input' or 'output'.")

    tensor_args_map = {
        "layout": f"source_model_{tensor_type}_layout",
        "desired_datatype": f"desired_model_{tensor_type}_datatype",
        "desired_layout": f"desired_model_{tensor_type}_layout",
    }

    if tensor_type == "input":
        tensor_args_map.update(
            {
                "shape": "source_model_input_shape",
                "datatype": "source_model_input_datatype",
            }
        )

    # Rename tensor args
    for api_arg, module_arg in tensor_args_map.items():
        if api_arg in tensor_dict:
            _rename_arg(tensor_dict, api_arg, module_arg)

    try:
        from qairt.api.converter.torch_convert_util import is_torch_dtype, is_torch_shape
    except (ImportError, OSError):
        _convert_logger.warning("Could not import torch. Torch dtype/data conversion will not be supported.")
        is_torch_dtype = lambda _: False
        is_torch_shape = lambda _: False

    # Handle source_model_shape arg for input tensors only
    if tensor_type == "input":
        shape = tensor_dict.get("source_model_input_shape")

        if shape:
            if isinstance(shape, list):
                shape_list = [
                    _shape_to_str(tuple(s)) if is_torch_shape(s) else _shape_to_str(s) for s in shape
                ]
                tensor_dict["source_model_input_shape"] = shape_list
            else:
                shape_str = _shape_to_str(tuple(shape)) if is_torch_shape(shape) else _shape_to_str(shape)
                tensor_dict["source_model_input_shape"] = shape_str

    # Handling for datatype args
    for key in [f"source_model_{tensor_type}_datatype", f"desired_model_{tensor_type}_datatype"]:
        datatype = tensor_dict.get(key)
        if datatype:
            if is_torch_dtype(datatype):
                tensor_dict[key] = str(datatype).split(".")[-1]
            elif isinstance(datatype, np.dtype):
                tensor_dict[key] = str(datatype)

    layout_aliases = {
        "channels_first": {3: "NCF", 4: "NCHW", 5: "NCDHW"},
        "channels_last": {3: "NFC", 4: "NHWC", 5: "NDHWC"},
    }

    # Handling for layout args
    for key in [f"source_model_{tensor_type}_layout", f"desired_model_{tensor_type}_layout"]:
        layout = tensor_dict.get(key)
        if layout in layout_aliases:
            if tensor_type == "input" and shape is None:
                _convert_logger.warning(
                    f"{layout} cannot be resolved without a shape provided. Defaulting to None..."
                )
                tensor_dict[key] = None
            elif tensor_type == "input" and shape is not None:
                tensor_rank = _get_tensor_rank(shape)
                if tensor_rank:
                    tensor_dict[key] = layout_aliases[layout].get(tensor_rank, layout)
        else:
            tensor_dict[key] = layout

    # Create module-level TensorConfig
    if tensor_type == "input":
        return ModuleInputTensorConfig(**tensor_dict)
    else:
        return ModuleOutputTensorConfig(**tensor_dict)


def _shape_to_str(shape: Tuple[int, ...]) -> str:
    """
    Converts a tuple representing a tensor shape into a string.

    Args:
        shape (Tuple[int, ...]): A tuple of integers representing the
            dimensions of a tensor.

    Returns:
        str: A string representation of the tensor shape, with
            comma separated dimensions.
    """
    return ",".join(str(dim) for dim in shape)


def _get_tensor_rank(shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) -> Optional[int]:
    """
    Returns the rank of a tensor given its shape.

    Args:
        shape (Union[Tuple[int, ...], List[Tuple[int, ...]]]): The shape of the tensor, which can
            be either a tuple of integers or a list of tuples of integers.

    Returns:
        Optional[int]: The rank of the tensor, or None if the shape is invalid.

    Notes:
        If the input shape is a list of tuples, the function assumes that all tuples have the same rank,
        or it takes the rank of the first tuple.
    """
    if isinstance(shape, list) and shape and isinstance(shape[0], tuple):
        return len(shape[0])
    elif isinstance(shape, tuple):
        return len(shape)
    return None


def _converter_config_arg_adapter(extra_args: Dict) -> Dict:
    """Converts extra args names to converter module internal names"""

    converter_config_args_map = {
        "input_tensor_config": "input_tensors",
        "output_tensor_config": "output_tensors",
        "float_precision": "float_bitwidth",
        "float_bias_precision": "float_bias_bitwidth",
        "batch": "onnx_batch",
        "define_symbol": "onnx_define_symbol",
        "defer_loading": "onnx_defer_loading",
        "op_package_lib": "converter_op_package_lib",
        "lora_tensor_names": "lora_weight_list",
    }

    for api_arg, module_arg in converter_config_args_map.items():
        if api_arg in extra_args:
            _rename_arg(extra_args, api_arg, module_arg)

    # Handle input_tensors args
    input_tensors = []
    for input_tensor in extra_args.get("input_tensors", []):
        module_input_tensor_config = _tensor_config_arg_adapter(input_tensor, "input")
        input_tensors.append(module_input_tensor_config)
    if input_tensors:
        extra_args["input_tensors"] = input_tensors

    # Handle output_tensors args
    output_tensors = []
    for output_tensor in extra_args.get("output_tensors", []):
        module_output_tensor_config = _tensor_config_arg_adapter(output_tensor, "output")
        output_tensors.append(module_output_tensor_config)
    if output_tensors:
        extra_args["output_tensors"] = output_tensors

    if extra_args.get("io_config"):
        extra_args["io_config"] = str(extra_args["io_config"])

    return extra_args


def _calibration_config_arg_adapter(calibration_config: CalibrationConfig) -> Dict:
    """Converts calibration config attributes to quantizer module internal names"""

    quantizer_config_args_map = {
        "act_precision": "act_bitwidth",
        "bias_precision": "bias_bitwidth",
        "weights_precision": "weights_bitwidth",
        "param_calibration_method": "param_quantizer_calibration",
        "act_calibration_method": "act_quantizer_calibration",
        "per_channel_quantization": "use_per_channel_quantization",
        "per_row_quantization": "use_per_row_quantization",
        "per_row_quantization_bias": "enable_per_row_quantized_bias",
    }

    quantizer_args = calibration_config.model_dump()

    for api_arg, module_arg in quantizer_config_args_map.items():
        if api_arg in quantizer_args:
            _rename_arg(quantizer_args, api_arg, module_arg)

    for key in ["dataset", "batch_size", "num_of_samples"]:
        quantizer_args.pop(key)

    return quantizer_args


@contextlib.contextmanager
def disable_spawn_process_and_exec():
    """Apply temporary patch to spawn_process_and_exec and restore original method"""

    import qti.aisw.converters.common.utils.framework_utils as fw_utils

    def _spawn_process_and_exec_override(func, *args, **kwargs):
        kwargs.pop("process_name", "Process")
        res = func(*args, **kwargs)
        status = res is not None
        return status, res

    # Patch spawn_process_and_exec
    spawn_process_and_exec_name = "spawn_process_and_exec"
    original_spawn_process_and_exec = getattr(fw_utils, spawn_process_and_exec_name)
    patched_spawn_process_and_exec = _spawn_process_and_exec_override

    try:
        setattr(fw_utils, spawn_process_and_exec_name, patched_spawn_process_and_exec)
        yield

    finally:
        setattr(fw_utils, spawn_process_and_exec_name, original_spawn_process_and_exec)
