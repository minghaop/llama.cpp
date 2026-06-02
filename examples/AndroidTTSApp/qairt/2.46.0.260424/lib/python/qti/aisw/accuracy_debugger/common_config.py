# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from typing import List, Literal, Optional, Union

from pydantic import Field, FilePath, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.modules.converter import (
    ConverterInputConfig,
    QuantizerInputConfig,
)
from qti.aisw.tools.core.modules.net_runner import InferenceConfig
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    RemoteDeviceInfo,
)


class EncodingInputConfig(AISWBaseModel):
    """Encoding class to be used for passing encoding details to Compare Encodings"""

    encoding_path: FilePath = Field(
        description="Path to an encoding file or quantized dlc file. Encoding file must end with "
        "either .json or .encodings. Dlc file must end with .dlc. Incase if dlc file provided, "
        "encodings are extracted from quantized dlc file."
    )
    quantized_dlc_path: Optional[FilePath] = Field(
        description="Path to quantized dlc file related to encoding_file_path being passed."
        "If passed along side with framework model, "
        "it performs following operations on the encodings from encoding_file_path:"
        "1.  Propagates convert_ops encodings to the its parent op considering the fact that"
        "parent op exists in the framework model"
        "2.  Resolves any activation name changes done. For e.g. matmul+add in framework"
        "model becomes fc in the dlc graph and the tensor name gets _fc suffix."
        "It also performs supergroup mapping which maps each QNN tensor to a set of tensors in"
        "framework graph which got fused to form the fused QNN op.",
        default=None,
    )

    @model_validator(mode="after")
    def validate_encoding_path(self):
        """Validates the encoding_path ends with one of the following:
        1. .json
        2. .encodings
        3. .dlc
        """
        supported_extensions = [".json", ".dlc", ".encodings"]
        if self.encoding_path.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"encoding_path :{self.encoding_path} must have one of the supported file "
                f"extension: {supported_extensions}"
            )

        return self


class LayerOptions(AISWBaseModel):
    add_layer_outputs: Optional[List[str]] = []
    add_layer_types: Optional[List[str]] = []
    skip_layer_types: Optional[List[str]] = []
    skip_layer_outputs: Optional[List[str]] = []
    start_layer: Optional[str] = None
    end_layer: Optional[str] = None


class ConverterInputArguments(ConverterInputConfig):
    input_network: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    dry_run: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    output_path: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    float_bitwidth: Optional[Literal[32, 16]] = Field(
        default=None,
        description="Convert the graph to specified float bitwidth.",
    )
    onnx_batch: SkipJsonSchema[int] = Field(default=None, init=False, exclude=True)
    preserve_io_datatype: SkipJsonSchema[Union[str, List[str]]] = Field(
        default=None, init=False, exclude=True
    )

    @field_validator("input_network")
    @classmethod
    def validate_framework(cls, v):
        pass

    @model_validator(mode="after")
    def validate_input_arguments(self):
        return self


class QuantizerInputArguments(QuantizerInputConfig):
    input_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    output_dlc: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    backend_info: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
    use_native_input_files: SkipJsonSchema[bool] = Field(default=False, init=False, exclude=True)


class NetRunnerInputArguments(InferenceConfig):
    use_native_input_data: SkipJsonSchema[bool] = Field(default=None, init=False, exclude=True)


class RemoteHostDetails(RemoteDeviceInfo):
    platform_type: SkipJsonSchema[str] = Field(default="", init=False, exclude=True)
