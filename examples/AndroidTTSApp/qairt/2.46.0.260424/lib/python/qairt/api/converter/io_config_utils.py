# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import copy
import json
import logging
import tempfile
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import onnx
import yaml

from qairt.api.converter.converter_config import InputTensorConfig, OutputTensorConfig

logger = logging.getLogger(__name__)


def convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings in nested data structures.

    This utility function traverses dictionaries, lists, and other nested structures
    to convert all pathlib.Path objects to their string representations.

    Args:
        obj: Any object (dict, list, Path, or primitive type)

    Returns:
        The same structure with all Path objects converted to strings

    Example:
        >>> data = {"config": Path("/tmp/file.yaml"), "items": [Path("/tmp/a"), "text"]}
        >>> convert_paths_to_strings(data)
        {'config': '/tmp/file.yaml', 'items': ['/tmp/a', 'text']}
    """
    if isinstance(obj, dict):
        return {k: convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj


# YAML template structures for converter IO configuration
_INPUT_TENSOR_YAML_TEMPLATE = {
    "Name": None,
    "Src Model Parameters": {
        "DataType": None,
        "Layout": None,
    },
    "Desired Model Parameters": {
        "DataType": None,
        "Layout": None,
        "Shape": None,
        "Color Conversion": None,
        "QuantParams": {
            "Scale": None,
            "Offset": None,
        },
    },
}

_OUTPUT_TENSOR_YAML_TEMPLATE = {
    "Name": None,
    "Src Model Parameters": {
        "DataType": None,
        "Layout": None,
    },
    "Desired Model Parameters": {
        "DataType": None,
        "Layout": None,
        "QuantParams": {
            "Scale": None,
            "Offset": None,
        },
    },
}


def get_onnx_data_type(elem_type: int) -> str:
    """Convert ONNX element type enum to string representation.

    Args:
        elem_type: ONNX tensor element type enum value

    Returns:
        String representation of the data type
    """
    type_mapping = {
        1: "float32",
        2: "uint8",
        3: "int8",
        4: "uint16",
        5: "int16",
        6: "int32",
        7: "int64",
        8: "string",
        9: "bool",
        10: "float16",
        11: "double",
        12: "uint32",
        13: "uint64",
        14: "complex64",
        15: "complex128",
        16: "bfloat16",
    }

    if elem_type not in type_mapping:
        logger.warning(f"Unknown ONNX data type encountered: {elem_type}. Using 'unknown_type_{elem_type}'.")

    return type_mapping.get(elem_type, f"unknown_type_{elem_type}")


def get_tensor_shape(tensor_type) -> Optional[List[Union[int, str]]]:
    """Extract actual shape values from ONNX tensor type.

    Args:
        tensor_type: ONNX tensor type object

    Returns:
        List of shape dimensions, or None if no shape available
    """
    if not tensor_type.HasField("shape"):
        return None

    shape_values = []
    has_dynamic = False

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape_values.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape_values.append(f"dynamic_{dim.dim_param}")
            has_dynamic = True
        else:
            shape_values.append(-1)  # Unknown dimension
            has_dynamic = True

    if has_dynamic:
        logger.warning(f"Tensor has dynamic dimensions: {shape_values}")

    return shape_values


def infer_layout_from_rank(rank: int, tensor_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer source and desired layouts based on tensor rank.

    Args:
        rank: Number of tensor dimensions
        tensor_name: Name of the tensor (for logging)

    Returns:
        Tuple of (source_layout, desired_layout)
    """
    layout_mapping = {
        1: ("F", "F"),  # 1D: Feature vector
        2: ("NF", "NF"),  # 2D: Batch x Features
        3: ("NCF", "NFC"),  # 3D: Batch x Channels x Features (or sequence)
        4: ("NCHW", "NHWC"),  # 4D: Batch x Channels x Height x Width
        5: ("NCDHW", "NDHWC"),  # 5D: Batch x Channels x Depth x Height x Width
    }

    if rank in layout_mapping:
        return layout_mapping[rank]
    else:
        logger.warning(f"Unsupported tensor rank {rank} for tensor '{tensor_name}'. Layout will be None.")
        return None, None


def _process_onnx_tensor(node, is_input: bool = True) -> Dict:
    """Helper to process ONNX tensor node into config.

    Args:
        node: ONNX graph input or output node
        is_input: True for input tensors, False for output tensors

    Returns:
        Dictionary with tensor configuration (InputTensorConfig or OutputTensorConfig compatible)
    """
    tensor_type = node.type.tensor_type
    tensor_name = node.name

    config: Dict = {"name": tensor_name}

    # Extract data type (only for inputs)
    if is_input and tensor_type.HasField("elem_type"):
        datatype = get_onnx_data_type(tensor_type.elem_type)
        config["datatype"] = datatype
        logger.debug(f"{'Input' if is_input else 'Output'} tensor '{tensor_name}' datatype: {datatype}")

    # Extract rank and infer layouts
    if tensor_type.HasField("shape"):
        rank = len(tensor_type.shape.dim)

        # Get shape for logging/debugging
        shape = get_tensor_shape(tensor_type)
        logger.debug(f"{'Input' if is_input else 'Output'} tensor '{tensor_name}' shape: {shape}")

        # Infer layouts based on rank
        src_layout, desired_layout = infer_layout_from_rank(rank, tensor_name)
        if src_layout:
            config["layout"] = src_layout
            config["desired_layout"] = desired_layout

    return config


def infer_io_config_from_onnx(
    onnx_file: Union[str, PathLike],
) -> Tuple[List[InputTensorConfig], List[OutputTensorConfig]]:
    """Extract IO configurations directly from ONNX model.

    This function analyzes an ONNX model and automatically generates input and output
    tensor configurations with support for:
    - All tensor dimensions (1D through 5D)
    - Actual data type extraction from ONNX
    - Dynamic shape detection and warnings
    - Automatic layout inference

    Args:
        onnx_file: Path to ONNX model file

    Returns:
        Tuple of (input_tensor_configs, output_tensor_configs)

    Raises:
        ValueError: If ONNX model cannot be loaded

    Example:
        >>> input_configs, output_configs = infer_io_config_from_onnx("model.onnx")
        >>> print(input_configs[0])
        {'name': 'input', 'datatype': 'float32', 'layout': 'NCHW', 'desired_layout': 'NHWC'}
    """
    # Load ONNX model
    try:
        logger.debug(f"Loading ONNX model: {onnx_file}")
        model = onnx.load(str(onnx_file), load_external_data=False)
        g = onnx.shape_inference.infer_shapes(model)
        logger.debug("ONNX model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise ValueError(f"Failed to load ONNX model from {onnx_file}: {e}")

    # Get initializer names (weights/constants) to filter them out from inputs
    initializer_names = {init.name for init in g.graph.initializer}
    logger.debug(f"Found {len(initializer_names)} initializers to filter out")

    # Process input tensors (excluding initializers)
    input_configs = cast(
        List[InputTensorConfig],
        [
            _process_onnx_tensor(node, is_input=True)
            for node in g.graph.input
            if node.name not in initializer_names
        ],
    )

    # Process output tensors
    output_configs = cast(
        List[OutputTensorConfig], [_process_onnx_tensor(node, is_input=False) for node in g.graph.output]
    )

    logger.debug(f"Generated {len(input_configs)} input configs and {len(output_configs)} output configs")
    return input_configs, output_configs


def _populate_tensor_dict(
    tensor_config: Union[InputTensorConfig, OutputTensorConfig], template: dict, is_input: bool = True
) -> dict:
    """Helper function to populate tensor dictionary from config.

    Args:
        tensor_config: Input or output tensor configuration
        template: Dictionary template to populate
        is_input: True for input tensors, False for output tensors

    Returns:
        Populated dictionary with tensor configuration
    """
    dict_content = copy.deepcopy(template)
    dict_content["Name"] = tensor_config.get("name")

    # Src Model Parameters
    if is_input and (datatype := tensor_config.get("datatype")) is not None:
        dict_content["Src Model Parameters"]["DataType"] = str(datatype)

    if (layout := tensor_config.get("layout")) is not None:
        dict_content["Src Model Parameters"]["Layout"] = layout

    # Desired Model Parameters
    if (desired_datatype := tensor_config.get("desired_datatype")) is not None:
        dict_content["Desired Model Parameters"]["DataType"] = str(desired_datatype)

    if (desired_layout := tensor_config.get("desired_layout")) is not None:
        dict_content["Desired Model Parameters"]["Layout"] = desired_layout

    if is_input and (shape := tensor_config.get("shape")) is not None:
        dict_content["Desired Model Parameters"]["Shape"] = str(shape)

    # Handle quant params
    if (quant_param := tensor_config.get("quant_param")) is not None:
        if hasattr(quant_param, "scale") and quant_param.scale is not None:
            dict_content["Desired Model Parameters"]["QuantParams"]["Scale"] = quant_param.scale
        if hasattr(quant_param, "offset") and quant_param.offset is not None:
            dict_content["Desired Model Parameters"]["QuantParams"]["Offset"] = quant_param.offset

    return dict_content


def generate_yaml_io_config(
    input_tensor_configs: List[InputTensorConfig],
    output_tensor_configs: List[OutputTensorConfig],
    output_path: str | PathLike | None = None,
) -> str:
    """Generate a YAML IO config file from input/output tensor configurations.

    This function creates a YAML file compatible with the qairt-converter's --config option.
    The YAML structure follows the format expected by generate_converter_config.py.

    Args:
        input_tensor_configs: List of input tensor configurations
        output_tensor_configs: List of output tensor configurations
        output_path: Optional path where to save the YAML file. If None, creates a temp file.

    Returns:
        str: Path to the generated YAML file

    Raises:
        ValueError: If both input and output configs are empty
        IOError: If file write operation fails
    """
    # Validate that configs are not empty
    if not input_tensor_configs and not output_tensor_configs:
        logger.error("Cannot generate YAML config: both input and output tensor configs are empty.")
        raise ValueError("At least one input or output tensor configuration must be provided.")

    if not input_tensor_configs:
        logger.warning("No input tensor configurations provided. YAML will only contain output configs.")
    elif not output_tensor_configs:
        logger.warning("No output tensor configurations provided. YAML will only contain input configs.")

    logger.debug(
        f"Generating YAML config with {len(input_tensor_configs)} input(s) and {len(output_tensor_configs)} output(s)"
    )

    yaml.add_representer(
        type(None), lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:null", "")
    )

    converter_io_config_yaml: Dict[str, List[Any]] = {
        "Input Tensor Configuration": [],
        "Output Tensor Configuration": [],
    }

    # Populate input tensors
    for input_tensor in input_tensor_configs:
        dict_content = _populate_tensor_dict(input_tensor, _INPUT_TENSOR_YAML_TEMPLATE, is_input=True)
        converter_io_config_yaml["Input Tensor Configuration"].append(dict_content)

    # Populate output tensors
    for output_tensor in output_tensor_configs:
        dict_content = _populate_tensor_dict(output_tensor, _OUTPUT_TENSOR_YAML_TEMPLATE, is_input=False)
        converter_io_config_yaml["Output Tensor Configuration"].append(dict_content)

    # Create YAML file with error handling
    try:
        if output_path:
            yaml_path = str(output_path)
            logger.debug(f"Writing YAML config to: {yaml_path}")
            with open(yaml_path, "w") as f:
                yaml.safe_dump(converter_io_config_yaml, f, default_flow_style=False, sort_keys=False)
            logger.debug(f"Successfully wrote YAML config to: {yaml_path}")
        else:
            # Create temporary YAML file
            logger.debug("Creating temporary YAML config file")
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix="_converter_io_config.yaml", delete=False
            )
            yaml.safe_dump(converter_io_config_yaml, temp_file, default_flow_style=False, sort_keys=False)
            temp_file.close()
            yaml_path = temp_file.name
            logger.debug(f"Successfully created temporary YAML config: {yaml_path}")
    except Exception as e:
        logger.error(f"Failed to write YAML config file: {e}")
        raise IOError(f"Failed to write YAML config file: {e}")

    return yaml_path


def convert_config_dict_to_json(
    config_dict: Dict[str, Any],
    output_path: Optional[str | PathLike] = None,
    indent: int = 2,
) -> str:
    """Convert a configuration dictionary to JSON file.

    This is a utility function that can serialize any config dictionary to JSON,
    with automatic Path object conversion to strings.

    Args:
        config_dict: Configuration dictionary to serialize
        output_path: Path to save JSON file. If None, creates a temp file.
        indent: JSON indentation spaces. Default is 2.

    Returns:
        Path to the generated JSON file

    Raises:
        IOError: If file write operation fails

    Example:
        >>> config_dict = {"input_tensor_config": [...], "io_config": Path("/tmp/io.yaml")}
        >>> json_path = convert_config_dict_to_json(config_dict)
    """
    # Convert Path objects to strings for JSON serialization
    config_dict = convert_paths_to_strings(config_dict)

    # Create JSON file with error handling
    try:
        if output_path:
            json_path = str(output_path)
            logger.debug(f"Writing JSON config to: {json_path}")
            with open(json_path, "w") as f:
                json.dump(config_dict, f, indent=indent, default=str)
            logger.debug(f"Successfully wrote JSON config to: {json_path}")
        else:
            # Create temporary JSON file
            logger.debug("Creating temporary JSON config file")
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix="_converter_config.json", delete=False)
            json.dump(config_dict, temp_file, indent=indent, default=str)
            temp_file.close()
            json_path = temp_file.name
            logger.debug(f"Successfully created temporary JSON config: {json_path}")
    except Exception as e:
        logger.error(f"Failed to write JSON config file: {e}")
        raise IOError(f"Failed to write JSON config file: {e}")

    return json_path
