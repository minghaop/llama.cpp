# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""ConverterConfig class"""

from os import PathLike
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import ConfigDict, model_serializer

try:
    from torch import Size, dtype
    from torch.utils.data import DataLoader, Dataset

    ShapeTypes = Union[Tuple[int, ...], List[Tuple[int, ...]], Size]
    DatatypeTypes = Union[str, np.dtype, dtype]
    DatasetTypes = Union[str, PathLike, List, DataLoader, Dataset]
except (ImportError, OSError):
    # mypy complains of multiple types assignments
    ShapeTypes = Union[Tuple[int, ...], List[Tuple[int, ...]]]  # type: ignore
    DatatypeTypes = Union[str, np.dtype]  # type: ignore
    DatasetTypes = Union[str, PathLike, List]  # type: ignore

from typing_extensions import TypedDict

from qairt.api.configs.common import AISWBaseModel
from qti.aisw.tools.core.modules.converter.converter_module import QuantParam


class InputTensorConfig(TypedDict, total=False):
    """
    TypedDict of input tensor configuration. Any of the keys can be omitted, except name.
    """

    name: str
    """
    Name of tensor. This is required.
    """

    shape: ShapeTypes
    """
    Shape of input tensor. Default is None.
    """

    datatype: DatatypeTypes
    """
    Data type of input tensor. Default is float32.
    """

    layout: Optional[str]
    """
    Layout of each input tensor. Valid layouts include "channels_first", "channels_last", "NCDHW", "NDHWC",
    "NCHW", "NHWC", "HWIO", "OIHW", "NFC", "NCF", "NTF", "TNF", "NF", "NC", "F", "NONTRIVIAL".
    Default is None.
    """

    desired_layout: Optional[str]
    """
    Desired layout of each input tensor. Valid layouts include "channels_first", "channels_last", "NCDHW", "NDHWC",
    "NCHW", "NHWC", "HWIO", "OIHW", "NFC", "NCF", "NTF", "TNF", "NF", "NC", "F", "NONTRIVIAL".
    """

    desired_datatype: Optional[DatatypeTypes]
    """
    Desired data type of each input tensor.
    """

    quant_param: Optional[QuantParam]
    """
    Quantization parameters for each input tensor. QuantParam field (optional) has two sub fields: Scale and Offset. Default is None.
    """

    optional: Optional[bool]
    """
    Marks the tensor as optional, allowing it to be omitted when execution is triggered.
    """


class OutputTensorConfig(TypedDict, total=False):
    """
    TypedDict of output tensor configuration. Any of the keys can be omitted, except name.
    """

    name: str
    """
    Name of tensor. This is required.
    """

    layout: Optional[str]
    """
    Layout of each output tensor. Valid layouts include "NCDHW", "NDHWC",
    "NCHW", "NHWC", "HWIO", "OIHW", "NFC", "NCF", "NTF", "TNF", "NF", "NC", "F", "NONTRIVIAL".
    Default is None.
    """

    desired_layout: Optional[str]
    """
    Desired layout of each output tensor. Valid layouts include "NCDHW", "NDHWC",
    "NCHW", "NHWC", "HWIO", "OIHW", "NFC", "NCF", "NTF", "TNF", "NF", "NC", "F", "NONTRIVIAL".
    """

    desired_datatype: Optional[DatatypeTypes]
    """
    Desired data type of each output tensor.
    """

    quant_param: Optional[QuantParam]
    """
    Quantization parameters for each output tensor. QuantParam field (optional) has two sub fields: Scale and Offset. Default is None.
    """

    optional: Optional[bool]
    """
    Marks the tensor as optional. If set to true, the tensors will not be written by the backend.
    """


class ConverterConfig(AISWBaseModel):
    """
    Pydantic class of parameters for model conversion.
    """

    # Override model_config for this class
    model_config = AISWBaseModel.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "extra": "allow"})

    input_tensor_config: Optional[List[InputTensorConfig]] = None
    """
    A list of input tensor configurations containing the name, shape, datatype, layout, desired layout,
    desired datatype, quantization parameters, and optional flag of the input tensors.
    """

    output_tensor_config: Optional[List[OutputTensorConfig]] = None
    """
    A list of output tensor configurations containing the name, layout, desired layout, desired datatype,
    quantization parameters, and optional flag of the output tensors.
    """

    float_precision: Literal[32, 16] = 32
    """
    The floating point precision to use for the model. Note the floating point precision will be applied to
    all tensors (including static tensors). Users should ensure that the precision is supported by each
    operation according to QAIRT spec.
    """

    float_bias_precision: Literal[16, 32] = 32
    """
    Option to select the precision to use for float bias tensor.
    """

    preserve_io_datatype: Optional[Union[str, List[str]]] = None
    """
    Set this option to maintain the source framework datatype for input and output tensors. This option
    is particularly useful when sequences of unsupported static operations are present. To preserve datatype
    for all input and output tensors, use preserve_io_datatype = "all". For select input and output tensors,
    use preserve_io_datatype = ["input1", "output1", ...]
    """

    onnx_simplification: bool = True
    """
    Do not attempt to simplify the model automatically. This may prevent some models from converting when
    sequences of unsupported static operations are present. Default is True.
    """

    batch: Optional[int] = None
    """
    The batch dimension override. This will take the first dimension of all inputs and treat it as a batch
    dim, overriding it with the value provided here.
    """

    define_symbol: Optional[List[Tuple[str, int]]] = None
    """
    Option to override specific input dimension symbols.
    """

    defer_loading: bool = False
    """
    Option to have the model not load weights. If False, the model will be loaded eagerly.
    """

    enable_framework_trace: bool = False
    """
    Use this option to enable converter to trace the o/p tensor change information.
    """

    op_package_config: Optional[List[str | PathLike]] = None

    """
    List of absolute paths to a Qnn Op Package XML configuration file that contains user defined custom
    operations.
    """

    op_package_lib: Optional[List[str | PathLike]] = None
    """
    List of absolute paths to converter op package library compiled by the OpPackage generator.
    """

    io_config: Optional[str | PathLike] = None
    """
    The path to the custom I/O configuration file. This file defines the custom input and output tensors.
    This file takes precedence over input_tensor_config and output_tensor_config if any of them are also specified.
    """

    export_format: Literal["DLC_DEFAULT", "DLC_STRIP_QUANT"] = "DLC_DEFAULT"
    """
    Specifies the export format for the model.
    "DLC_DEFAULT":
    - Produces a Float graph from a Float source graph.
    - Produces a Quantized graph from a source graph using provided encodings.
    "DLC_STRIP_QUANT":
    - Produces a Float graph by removing all quantization data from the source graph.
    """

    create_custom_io: bool = False
    """
    When set to True, automatically infers input_tensor_config and output_tensor_config from ONNX models.
    Extracts tensor names, datatypes, shapes, and infers layouts.
    User-provided configs take precedence if already specified.

    Example:
        >>> converted_model = qairt.convert("model.onnx", create_custom_io=True)
    """

    @model_serializer
    def serialize_io_config(self, info):
        """Serialize ConverterConfig to dictionary with optional YAML IO config auto-generation.

        When context={'auto_generate_io': True}, generates a temp YAML file from tensor configs
        and populates the io_config field.

        Example:
            >>> config_dict = config.model_dump(context={"auto_generate_io": True})
            >>> # config.io_config now contains path to auto-generated YAML
        """
        context = info.context or {}

        # Check if auto-generation of IO config is requested
        if context.get("auto_generate_io", False):
            # Auto-generate YAML IO config if tensor configs exist
            if self.input_tensor_config or self.output_tensor_config:
                try:
                    from qairt.api.converter.io_config_utils import generate_yaml_io_config

                    # Generate temporary YAML file
                    yaml_path = generate_yaml_io_config(
                        input_tensor_configs=self.input_tensor_config or [],
                        output_tensor_configs=self.output_tensor_config or [],
                        output_path=None,  # Always create temp file
                    )

                    # Update io_config field with the generated YAML path
                    self.io_config = yaml_path

                except Exception as e:
                    # If YAML generation fails, log and continue without it
                    import logging

                    logging.getLogger(__name__).warning(f"Failed to auto-generate YAML IO config: {e}")

        return self.__dict__

    def dump_json(self, output_path: Optional[str | PathLike] = None, indent: int = 2) -> str:
        """Generate JSON file from ConverterConfig with auto-generated YAML IO config.

        Auto-generates a temp YAML IO config file (if tensor configs exist) and populates
        config.io_config with the YAML path, then serializes to JSON.

        Args:
            output_path: Path to save JSON file. If None, creates a temp file.
            indent: JSON indentation spaces. Default is 2.

        Returns:
            Path to the generated JSON file

        Example:
            >>> config = ConverterConfig(input_tensor_config=[...])
            >>> json_path = config.dump_json()
            >>> qairt.convert(model_path, io_config=config.io_config)
        """
        from qairt.api.converter.io_config_utils import convert_config_dict_to_json

        # Use model_dump with auto_generate_io context to trigger YAML generation
        # This calls serialize_io_config() which auto-generates the YAML file
        config_dict = self.model_dump(context={"auto_generate_io": True})

        # Use the utility function to convert to JSON
        return convert_config_dict_to_json(config_dict, output_path=output_path, indent=indent)


class CalibrationConfig(AISWBaseModel):
    """
    Configuration for calibration process.
    """

    # Override model_config for this class
    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, validate_assignment=True, protected_namespaces=()
    )

    dataset: Optional[DatasetTypes] = None
    """
    The dataset to be used for calibration.
        It can be a string, a PathLike object, or a list of datasets.
    """

    batch_size: int = 1
    """
    The size of the batch to be used during calibration.
            Default is 1.
    """

    num_of_samples: int = 512
    """
    The number of samples to be used for calibration.
            Default is 512.
    """

    act_precision: Literal[8, 16] = 8
    """
    Integer precision value to use while quantizing activations
    """

    bias_precision: Literal[8, 32] = 8
    """
    Precision value to use while quantizing biases
    """

    weights_precision: Literal[4, 8, 16] = 8
    """
    Precision value to use while quantizing weights
    """

    param_calibration_method: str = "min-max"
    """
    Calibration method to use for parameters. Valid methods are "min-max", "sqnr", "entropy", "mse", "percentile"
    """

    act_calibration_method: str = "min-max"
    """
    Calibration method to use for activations. Valid methods are "min-max", "sqnr", "entropy", "mse", "percentile"
    """

    per_channel_quantization: bool = True
    """
    Enable per channel quantization for convolution based op weights
    """

    per_row_quantization: bool = False
    """
    Enable per row quantization for Matmul and FullyConnected ops
    """

    per_row_quantization_bias: bool = False
    """
    Enable per row quantization of bias for FullyConnected ops, when weights are per-row quantized
    """

    keep_weights_quantized: bool = False
    """
    Enable wFxp_actFP configurations according to the provided bitwidth for weights and activations.
    """
