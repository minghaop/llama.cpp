# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from pathlib import Path
from typing import Any, Optional

import numpy as np
from qti.aisw.accuracy_debugger.encodings.encodings_utils import (
    EncodingType,
    EncodingVersion,
    TensorDtype,
    TensorType,
    UnsupportedEncodingsVersionError,
    get_encodings_version,
)
from qti.aisw.accuracy_debugger.utils.file_utils import dump_json, read_json
from qti.aisw.converters.common import ir_graph
from qti.aisw.dlc_utils import modeltools


# ------------------------------------------------------
# Code block for Tensor encoding
# ------------------------------------------------------


class TensorEncoding:
    """Base class for storing encoding information for an tensor"""

    def __init__(self, tensor_name: str, tensor_type: TensorType) -> None:
        """Initializes TensorEncoding class"""
        self.tensor_name: str = tensor_name
        self.tensor_type: TensorType = tensor_type
        self.encoding_type: EncodingType = None
        self.dtype: TensorDtype = None
        self.bitwidth: int = None
        self.min: list = []
        self.max: list = []
        self.scale: list = []  # per_channel_float_scale incase of LPBQ
        self.offset: list = []
        self.is_symm: str = ""
        self.channels: int = None
        self.axis: int = None
        self.block_size: int = None  # Used incase of LPBQ
        self.compressed_bw: int = None  # Used incase of LPBQ
        self.per_block_int_scale: list = None  # Used incase of LPBQ

    @property
    def V0(self) -> list[dict[str, Any]]:
        """Version 0.6.0 representation of an tensor encoding"""
        # Can not create V0 encoding representation if tensor is of type TensorType.Encodings
        if self.tensor_type == TensorType.Encodings:
            raise ValueError(
                f"Tensor of type: {TensorType.Encodings} can not be represented in"
                f"{EncodingVersion.V0.value} version."
            )

        encoding_repr = []

        # If encoding is of type float
        if self.encoding_type == EncodingType.UNDEFINED:
            encoding_repr = [{"dtype": self.dtype.value, "bitwidth": self.bitwidth}]
        else:
            if self.encoding_type in [
                EncodingType.PER_TENSOR,
                EncodingType.PER_CHANNEL,
            ]:
                for channel_index in range(self.channels):
                    channel_encoding = {
                        "bitwidth": self.bitwidth,
                        "is_symmetric": self.is_symm,
                        "max": self.max[channel_index],
                        "min": self.min[channel_index],
                        "offset": self.offset[channel_index],
                        "scale": self.scale[channel_index],
                    }
                    encoding_repr.append(channel_encoding)
            else:
                raise Exception(
                    f"Tensor encoding representation not supported for type: {self.encoding_type.value}"
                )

        return encoding_repr

    @property
    def V1(self) -> dict[str, Any]:
        """Version 1.0.0 representation of an tensor encoding"""
        # Can not create V1 encoding representation if tensor is of type TensorType.Encodings
        if self.tensor_type == TensorType.Encodings:
            raise ValueError(
                f"Tensor of type: {TensorType.Encodings} can not be represented in"
                f"{EncodingVersion.V1.value} version."
            )

        if self.encoding_type == EncodingType.UNDEFINED:
            # FLOAT encoding
            encoding_repr = {
                "dtype": self.dtype.value.upper(),
                "bw": self.bitwidth,
                "name": self.tensor_name,
            }
        elif self.encoding_type in [EncodingType.PER_TENSOR, EncodingType.PER_CHANNEL]:
            encoding_repr = {
                "bw": self.bitwidth,
                "name": self.tensor_name,
                "dtype": "INT",
                "enc_type": self.encoding_type.value,
                "is_sym": self.is_symm,
                "offset": self.offset,
                "scale": self.scale,
            }
        elif self.encoding_type == EncodingType.LPBQ:
            encoding_repr = {
                "bw": self.bitwidth,
                "name": self.tensor_name,
                "block_size": self.block_size,
                "compressed_bw": self.compressed_bw,
                "dtype": "INT",
                "enc_type": self.encoding_type.value,
                "is_sym": self.is_symm,
                "offset": self.offset,
                "scale": self.scale,
                "per_block_int_scale": self.per_block_int_scale,
            }
        else:
            raise Exception(
                f"Tensor encoding representation not supported for type: {self.encoding_type.value}"
            )

        return encoding_repr

    @property
    def V2(self) -> dict[str, Any]:
        """Version 2.0.0 representation of an tensor encoding"""
        # Channel Wise or Row Wise TensorEncoding created with V0 or V1 encoding json file can not
        # be converted to V2 encoding version because axis information is not present
        if self.channels and self.channels > 1 and (self.axis is None):
            raise Exception(
                "Per Channel/Row TensorEncoding created with V1 or V0 encoding json file can not be"
                "converted to V2 encoding version"
            )

        # If encoding is of type float and general initiation
        encoding_repr = {
            "output_dtype": f"{self.dtype.value}{self.bitwidth}",
            "name": self.tensor_name,
        }
        if self.encoding_type == EncodingType.UNDEFINED:
            return encoding_repr
        elif self.encoding_type == EncodingType.PER_TENSOR:
            encoding_repr["y_scale"] = self.scale[0]
            if self.offset[0] != 0:
                encoding_repr["y_zero_point"] = self.offset[0]
        elif self.encoding_type == EncodingType.PER_CHANNEL:
            encoding_repr["y_scale"] = self.scale
            encoding_repr["axis"] = self.axis
            # If all offsets are 0, skip it. Not requried.
            if not all(offset == 0 for offset in self.offset):
                encoding_repr["y_zero_point"] = self.offset
        elif self.encoding_type == EncodingType.LPBQ:
            per_channel_float_scale = (
                np.asarray(self.scale).reshape((len(self.scale), 1, 1, 1)).tolist()
            )
            encoding_repr["per_channel_float_scale"] = per_channel_float_scale
            encoding_repr["output_dtype"] = f"{self.dtype.value}{self.compressed_bw}"

            # Number of blocks = number of channels * number of block per channel
            # Number of channels = len(self.scale) = afterall one scale per channel in LPBQ
            # Number of blocks = len(per_block_int_scale)
            num_blocks_per_channel = len(self.per_block_int_scale) // len(self.scale)
            per_block_int_scale = np.asarray(self.per_block_int_scale)
            per_block_int_scale = per_block_int_scale.reshape(
                (len(self.scale), num_blocks_per_channel, 1, 1)
            ).tolist()
            encoding_repr["per_block_int_scale"] = per_block_int_scale

            encoding_repr["axis"] = self.axis
            encoding_repr["block_size"] = self.block_size
            # If all offsets are 0, skip it. Not requried.
            if not all(offset == 0 for offset in self.offset):
                encoding_repr["y_zero_point"] = np.asarray(self.offset).reshape(
                    (len(self.offset), 1, 1, 1)
                )

        return encoding_repr


class DlcTensorEncoding(TensorEncoding):
    """Encoding class for a Dlc Tensor"""

    def __init__(self, ir_tensor: Any, op_type: str | None = None) -> None:
        """Initializes DlcTensorEncoding class

        Args:
            ir_tensor: Object of libPyIrGraph.IrStaticTensor or libPyIrGraph.IrTensor.
            op_type: The type of the op that produces this tensor (e.g. "Conv2d",
                "MatMul").  Stored on the encoding so that validation rules can
                perform accurate layer-type matching without needing to traverse
                the DLC graph separately.
        """
        tensor_name = ir_tensor.name()
        tensor_type = (
            TensorType("param_encodings")
            if ir_tensor.is_static_tensor()
            else TensorType("activation_encodings")
        )
        super().__init__(tensor_name=tensor_name, tensor_type=tensor_type)
        self.op_type: str | None = op_type

        # Check if the tensor is running in float or int
        float_types = [ir_graph.QNN_DATATYPE_FLOAT_16, ir_graph.QNN_DATATYPE_FLOAT_32]
        if ir_tensor.data_type() in float_types:
            self._set_float_encodings(ir_tensor=ir_tensor)
        else:
            self._set_quantized_encodings(ir_tensor=ir_tensor)

    def _consume_qnn_quantization_encoding_scale_offset(self, encInfo: Any) -> None:
        """Given a Per Tensor Encoding info, extract the info

        Args:
            encInfo: Object of Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
        """
        self.bitwidth = encInfo.bw
        self.min = [encInfo.min]
        self.max = [encInfo.max]
        self.scale = [encInfo.scale]
        self.offset = [encInfo.offset]
        self.is_symm = str(encInfo.is_symmetric).lower()
        self.channels = 1

    def _consume_qnn_quantization_encoding_axis_scale_offset(self, axisEncInfo: Any) -> None:
        """Given a Per Channel Encoding info, extract the info

        Args:
            axisEncInfo: Object of Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
        """
        encInfos = axisEncInfo.encInfos
        self.bitwidth = encInfos[0].bw
        self.is_symm = str(encInfos[0].is_symmetric).lower()
        self.channels = len(encInfos)

        for encInfo in encInfos:
            self.min.append(encInfo.min)
            self.max.append(encInfo.max)
            self.scale.append(encInfo.scale)
            self.offset.append(encInfo.offset)

    def _consume_qnn_lpbq_encodings(self, lpbqEncInfo: Any) -> None:
        """Given a LPBQ Encoding info, extract the info

        Args:
            lpbqEncInfo: Object of Qnn_QuantizationEncoding_t.QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION
        """
        self.block_size = lpbqEncInfo.blockSize
        self.compressed_bw = lpbqEncInfo.blockScaleBitwidth
        self.axis = lpbqEncInfo.axis
        self.per_block_int_scale = (
            lpbqEncInfo.blocksScale8 if lpbqEncInfo.blocksScale8 else lpbqEncInfo.blocksScale16
        )

        # Populate attributes like scale, offset, bitwidth, is_symm etc
        # Use the existing function for PER_CHANNEL as they have the same structure
        self._consume_qnn_quantization_encoding_axis_scale_offset(axisEncInfo=lpbqEncInfo)

    def _set_quantized_encodings(self, ir_tensor: Any) -> None:
        """Given ir_tensor extract the quantization info for the tensor

        Args:
            ir_tensor: Object of libPyIrGraph.IrStaticTensor or libPyIrGraph.IrTensor.

        Raises:
            Exception: If quantized tensor encoding type is not one of the following:
                1. ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
                2. ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
        """
        int_type = [
            ir_graph.QNN_DATATYPE_SFIXED_POINT_8,
            ir_graph.QNN_DATATYPE_SFIXED_POINT_16,
            ir_graph.QNN_DATATYPE_SFIXED_POINT_32,
        ]
        if ir_tensor.data_type() in int_type:
            self.dtype = TensorDtype.SFXP
        else:
            self.dtype = TensorDtype.UFXP

        tensor_encoding = ir_tensor.get_encoding()
        # QNN_QUANTIZATION_ENCODING_SCALE_OFFSET and QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET
        # encoding information are treated the same as per IrTensor.hpp definitions
        if tensor_encoding.type in [
            ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
            ir_graph.QNN_QUANTIZATION_ENCODING_BW_SCALE_OFFSET,
        ]:
            self.encoding_type = EncodingType.PER_TENSOR
            self._consume_qnn_quantization_encoding_scale_offset(encInfo=tensor_encoding.encInfo)
        # QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET and
        # QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET
        # encoding information are treated the same as per IrTensor.hpp definitions
        elif tensor_encoding.type in [
            ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET,
            ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET,
        ]:
            self.encoding_type = EncodingType.PER_CHANNEL
            self.axis = (
                tensor_encoding.axisEncInfo.axis
            )  # This axis value is not same as framework axis.
            self._consume_qnn_quantization_encoding_axis_scale_offset(
                axisEncInfo=tensor_encoding.axisEncInfo
            )
        elif tensor_encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION:
            self.encoding_type = EncodingType.LPBQ
            self._consume_qnn_lpbq_encodings(lpbqEncInfo=tensor_encoding.lpbqEncInfo)
        else:
            raise Exception(
                f"Tensor encoding of type {tensor_encoding.type.name} is not supported."
            )

    def _set_float_encodings(self, ir_tensor: Any) -> None:
        """Given ir_tensor extract the float info for the tensor

        Args:
            ir_tensor: Object of libPyIrGraph.IrStaticTensor or libPyIrGraph.IrTensor.
        """
        self.encoding_type = EncodingType.UNDEFINED
        self.dtype = TensorDtype.FLOAT
        if ir_tensor.data_type() == ir_graph.QNN_DATATYPE_FLOAT_16:
            self.bitwidth = 16
        else:
            self.bitwidth = 32


class V0JsonTensorEncoding(TensorEncoding):
    """Encoding class for a Version 0.6.0 tensor encoding loaded from json file"""

    def __init__(self, tensor_name: str, tensor_encoding: list, tensor_type: TensorType) -> None:
        """Initializes V0JsonTensorEncoding class

        Args:
            tensor_name: Name of the tensor.
            tensor_encoding: Version "0.6.0" tensor encoding.
            tensor_type: Type of the tensor: [param or activation]
        """
        super().__init__(tensor_name=tensor_name, tensor_type=tensor_type)
        self._tensor_encoding = tensor_encoding
        self.dtype = self._get_dtype()
        self.bitwidth = self._tensor_encoding[0]["bitwidth"]
        self.scale = self._get_value(field="scale")
        self.offset = self._get_value(field="offset")
        self.min = self._get_value(field="min")
        self.max = self._get_value(field="max")
        self.is_symm = str(self._tensor_encoding[0].get("is_symmetric", "")).lower()
        self.channels = len(self.scale)
        if self.dtype == TensorDtype.FLOAT:
            self.encoding_type = EncodingType.UNDEFINED
        elif self.channels == 1:
            self.encoding_type = EncodingType.PER_TENSOR
        else:
            self.encoding_type = EncodingType.PER_CHANNEL

    def _get_dtype(self) -> TensorDtype:
        """Extracts dtype from V0 tensor encoding

        Returns:
            (TensorDtype): dtype of the tensor encoding
        """
        dtype = TensorDtype.FLOAT
        if "dtype" in self._tensor_encoding[0]:
            dtype = TensorDtype(self._tensor_encoding[0]["dtype"].lower())
        elif "scale" in self._tensor_encoding[0]:
            if self._tensor_encoding[0]["scale"] == 0:
                dtype = TensorDtype.FLOAT
            else:
                dtype = TensorDtype.SFXP

        return dtype

    def _get_value(self, field: str) -> list:
        """Extarct the scale or offset from the V0 tensor encoding

        Args:
            field (str): either scale or offset

        Returns:
            (list): list of field values
        """
        value = []

        if field in self._tensor_encoding[0]:
            for channel in self._tensor_encoding:
                value.append(channel[field])

        return value


class V1JsonTensorEncoding(TensorEncoding):
    """Encoding class for a Version 1.0.0 tensor encoding loaded from json file"""

    def __init__(self, tensor_encoding: dict, tensor_type: TensorType) -> None:
        """Initializes V1JsonTensorEncoding class

        Args:
            tensor_encoding: Version "1.0.0" tensor encoding.
            tensor_type: Type of the tensor: [param or activation]
        """
        super().__init__(tensor_name=tensor_encoding["name"], tensor_type=tensor_type)

        # Set encoding type and related values
        if tensor_encoding["dtype"].lower() == "float":
            self.encoding_type = EncodingType.UNDEFINED
        elif tensor_encoding["enc_type"] == EncodingType.PER_TENSOR.value:
            self.encoding_type = EncodingType.PER_TENSOR
        elif tensor_encoding["enc_type"] == EncodingType.PER_CHANNEL.value:
            self.encoding_type = EncodingType.PER_CHANNEL
        elif tensor_encoding["enc_type"] == EncodingType.LPBQ.value:
            self.encoding_type = EncodingType.LPBQ
            self.block_size = tensor_encoding["block_size"]
            self.compressed_bw = tensor_encoding["compressed_bw"]
            self.per_block_int_scale = tensor_encoding["per_block_int_scale"]
        else:
            raise Exception(
                f"Tensor encoding of type {tensor_encoding['enc_type']} is not supported."
            )

        self._tensor_encoding = tensor_encoding
        self.dtype = TensorDtype(self._tensor_encoding["dtype"].lower())
        self.bitwidth = self._tensor_encoding["bw"]
        self.scale = self._tensor_encoding.get("scale", [])
        self.offset = self._tensor_encoding.get("offset", [])
        self.is_symm = str(self._tensor_encoding.get("is_sym", "")).lower()
        self.channels = len(self.scale)
        if self.dtype in [TensorDtype.SFXP, TensorDtype.UFXP]:
            self.min = [offset * scale for offset, scale in zip(self.offset, self.scale)]
            self.max = [
                (2 ** (self.bitwidth) - 1) * scale + min for scale, min in zip(self.scale, self.min)
            ]


class V2JsonTensorEncoding(TensorEncoding):
    """Encoding class for a Version 2.0.0 tensor encoding loaded from json file"""

    def __init__(self, tensor_encoding: dict) -> None:
        """Initializes V2JsonTensorEncoding class

        Args:
            tensor_encoding: Version 2.0.0 tensor encoding
        """
        super().__init__(tensor_name=tensor_encoding["name"], tensor_type=TensorType("encodings"))
        # AIMET: Positive offset value
        # QNN: Negative offset value
        # Since Qairt never dumps in V2 format, we assume this behavior will remain going ahead.
        # Hence, we multiply offset with -1 assuming this class will be used only by AIMET enc file

        if "float" in tensor_encoding["output_dtype"]:
            self.dtype = TensorDtype.FLOAT
            self.encoding_type = EncodingType.UNDEFINED
            self.bitwidth = int(tensor_encoding["output_dtype"].split("float")[1])
            self.scale = []
            self.offset = []
            self.channels = 0
        else:
            if "block_size" in tensor_encoding:
                self.encoding_type = EncodingType.LPBQ
                self.block_size = tensor_encoding["block_size"]
                self.scale = (
                    np.asarray(tensor_encoding.get("per_channel_float_scale", []))
                    .flatten()
                    .tolist()
                )
                self.per_block_int_scale = (
                    np.asarray(tensor_encoding.get("per_block_int_scale", [])).flatten().tolist()
                )
            elif "y_scale" in tensor_encoding:
                if isinstance(tensor_encoding["y_scale"], (int, float)):
                    self.encoding_type = EncodingType.PER_TENSOR
                else:
                    self.encoding_type = EncodingType.PER_CHANNEL
                # Handle y_scale being a list, a single float, or missing
                y_scale = tensor_encoding.get("y_scale", [])
                if isinstance(y_scale, list):
                    self.scale = y_scale
                elif isinstance(y_scale, (int, float)):
                    self.scale = [y_scale]
                else:
                    self.scale = []

            self.channels = len(self.scale)

            # Handle y_zero_point being a list, a single number, or missing
            y_zero_point = tensor_encoding.get("y_zero_point", [0] * self.channels)
            if isinstance(y_zero_point, list):
                y_zero_point = np.asarray(y_zero_point).flatten().tolist()  # for LPBQ flattening
                offset = y_zero_point
            elif isinstance(y_zero_point, (int, float)):
                offset = [y_zero_point]
            else:
                offset = [0] * self.channels
            self.offset = [-1 * o for o in offset]  # Convert AIMET offset to QNN format

            if "uint" in tensor_encoding["output_dtype"]:
                self.dtype = TensorDtype.UFXP
                self.bitwidth = int(tensor_encoding["output_dtype"].split("uint")[1])
                if self.channels > 0 and self.offset[0] == -1 * 2 ** (self.bitwidth - 1):
                    self.is_symm = "true"
                else:
                    self.is_symm = "false"
                min_point = 0  # E.g. 0 for int8
                max_point = pow(2, self.bitwidth) - 1  # E.g. 255 for int8
            else:
                self.dtype = TensorDtype.SFXP
                self.bitwidth = int(tensor_encoding["output_dtype"].split("int")[1])
                if self.channels > 0 and self.offset[0] == 0:
                    self.is_symm = "true"
                else:
                    self.is_symm = "false"
                min_point = -1 * pow(2, self.bitwidth - 1)  # E.g. -128 for int8
                max_point = pow(2, self.bitwidth - 1) - 1  # E.g. 127 for int8

            for channel_scale, channel_offset in zip(self.scale, self.offset):
                self.min.append((min_point + channel_offset) * channel_scale)
                self.max.append((max_point + channel_offset) * channel_scale)

        # Incase of LPBQ set the bitwidth to compressed bitwidth
        if self.encoding_type == EncodingType.LPBQ:
            self.compressed_bw = self.bitwidth
        self.axis = tensor_encoding.get("axis", None)


# ------------------------------------------------------
# Code block for base Encoding class
# ------------------------------------------------------


class Encoding:
    """Encoding Base Class"""

    def __init__(self) -> None:
        """Initializes Encoding class"""
        self.tensor_encodings: dict[str:TensorEncoding] = {}
        self.name: str = None

    @staticmethod
    def get_encodings_structure(version: EncodingVersion) -> dict:
        """Given encodings version returns empty encodings data structure

        Args:
            version (EncodingVersion): encodings version enum

        Returns:
            (dict or None) encodings data structure

        Raises:
            UnsupportedEncodingsVersionError: If encodings version is not supported
        """
        if version == EncodingVersion.V0:
            return {"activation_encodings": {}, "param_encodings": {}}
        elif version == EncodingVersion.V1:
            return {"version": version.value, "activation_encodings": [], "param_encodings": []}
        elif version == EncodingVersion.V2:
            return {"version": version.value, "encodings": []}

        raise UnsupportedEncodingsVersionError(
            f"Could not create encodings data structure for the given {version} version"
        )

    def _generate_V0_encoding(self) -> dict:
        """Generates an encoding dictionary with list of tensor_encodings in V0 format."""
        encoding = Encoding.get_encodings_structure(version=EncodingVersion.V0)
        for tensor_encoding in self.tensor_encodings.values():
            encoding[tensor_encoding.tensor_type.value][tensor_encoding.tensor_name] = (
                tensor_encoding.V0
            )

        return encoding

    def _generate_V1_encoding(self) -> dict:
        """Generates an encoding dictionary with list of tensor_encodings in V1 format."""
        encoding = Encoding.get_encodings_structure(version=EncodingVersion.V1)
        for tensor_encoding in self.tensor_encodings.values():
            encoding[tensor_encoding.tensor_type.value].append(tensor_encoding.V1)

        return encoding

    def _generate_V2_encoding(self) -> dict:
        """Generates an encoding dictionary with list of tensor_encodings in V2 format."""
        encoding = Encoding.get_encodings_structure(version=EncodingVersion.V2)
        for tensor_encoding in self.tensor_encodings.values():
            encoding[TensorType.Encodings.value].append(tensor_encoding.V2)

        return encoding

    def get_type_encodings(self, tensor_type: TensorType) -> dict[str:TensorEncoding]:
        """Return the dict of all tensor encodings of the given tensor_type with name as
        keys and TensorEncoding object as value.

        For V0/V1 encoding files and DLC files, tensors carry an explicit
        tensor_type (ActivationEncodings or ParamEncodings) so filtering is
        exact.  For V2 (AIMET 2.0.0) encoding files every tensor carries
        TensorType.Encodings, so we fall back to name-based heuristics:
          - ParamEncodings  → names containing "weight", "bias", or "kernel"
          - ActivationEncodings → all remaining tensors
        Requesting TensorType.Encodings always returns the full dict.

        Args:
            tensor_type: TensorType Enum to indicate which tensor encodings is needed.
        """
        if tensor_type == TensorType.Encodings:
            return self.tensor_encodings

        if tensor_type == TensorType.ParamEncodings:
            # Exact match for V0/V1/DLC tensors.
            param_encodings = {
                name: enc
                for name, enc in self.tensor_encodings.items()
                if enc.tensor_type == TensorType.ParamEncodings
            }
            # V2 fallback: all tensors carry TensorType.Encodings, so use
            # name-based heuristics to identify param tensors.
            if not param_encodings:
                param_encodings = {
                    name: enc
                    for name, enc in self.tensor_encodings.items()
                    if enc.tensor_type == TensorType.Encodings
                    and (
                        "weight" in name.lower()
                        or "bias" in name.lower()
                        or "kernel" in name.lower()
                    )
                }
            return param_encodings

        if tensor_type == TensorType.ActivationEncodings:
            # Exact match for V0/V1/DLC tensors.
            activation_encodings = {
                name: enc
                for name, enc in self.tensor_encodings.items()
                if enc.tensor_type == TensorType.ActivationEncodings
            }
            # V2 fallback: exclude name-based param tensors from the full set.
            if not activation_encodings:
                activation_encodings = {
                    name: enc
                    for name, enc in self.tensor_encodings.items()
                    if enc.tensor_type == TensorType.Encodings
                    and "weight" not in name.lower()
                    and "bias" not in name.lower()
                    and "kernel" not in name.lower()
                }
            return activation_encodings

        raise ValueError(f"Incorrect tensor_type: {tensor_type}")

    def get_tensor_encoding(self, tensor_name: str) -> TensorEncoding:
        """Given tensor name return the object of TensorEncoding

        Args:
            tensor_name: Tensor name for which encoding object is needed.

        Returns:
            (TensorEncoding): Object of TensorEncoding containg the quant info for the tensor.
            (None): If tensor_name not present in the present encoding list
        """
        return self.tensor_encodings.get(tensor_name, None)

    def encoding(self, version: EncodingVersion) -> dict:
        """Generates encoding dictionary with the list of tensor encodings.

        Args:
            version: Encoding Version in which encoding dictionary needs to be generated.

        Returns:
            (dict): encoding dictionary in the provided version.

        Raises:
            (UnsupportedEncodingsVersionError): If the provided version is not supported.
        """
        if version == EncodingVersion.V0:
            return self._generate_V0_encoding()
        elif version == EncodingVersion.V1:
            return self._generate_V1_encoding()
        elif version == EncodingVersion.V2:
            return self._generate_V2_encoding()
        else:
            raise UnsupportedEncodingsVersionError(f"Unsupported encoding {version}")

    def dump(self, version: EncodingVersion, file_path: Path | str) -> None:
        """Dumps the graph encoding consisting of list of tensor encodings according to the
        provided encoding version.

        Args:
            version: EncodingVersion in which encodings should be dumped.
            file_path: path to which encoding file should be dumped.
        """
        encoding = self.encoding(version=version)
        dump_json(data=encoding, json_path=file_path)

    def add(
        self,
        tensor_encoding: Optional[TensorEncoding] = None,
        subgraph_tensor_encodings: Optional[dict[TensorEncoding]] = None,
        overwrite_existing: bool = True,
    ) -> None:
        """Adds the given tensor encoding or a dictionary of subgraph tensor encodings to
        list of graph tensor encodings.

        Args:
            tensor_encoding: Object of TensorEncoding representing an tensor encoding.
            subgraph_tensor_encodings: Dictionary of subgraph tensor encodings with tensor names as
                keys and TensorEncoding object as value.
            overwrite_existing: If True, overwrites the existing tensor encoding with given the
                tensor encoding for a tensor name.

        Raises:
            ValueError: If both tensor_encoding and subgraph_tensor_encodings are None.
        """
        if not (tensor_encoding or subgraph_tensor_encodings):
            raise ValueError(
                "One of tensor_encoding or subgraph_tensor_encodings must be provided."
            )
        if tensor_encoding and (
            tensor_encoding.tensor_name not in self.tensor_encodings or overwrite_existing
        ):
            self.tensor_encodings[tensor_encoding.tensor_name] = tensor_encoding

        if subgraph_tensor_encodings:
            for tensor_name, tensor_encoding in subgraph_tensor_encodings.items():
                if tensor_encoding.tensor_name not in self.tensor_encodings or overwrite_existing:
                    self.tensor_encodings[tensor_name] = tensor_encoding


# ------------------------------------------------------
# Code block for full model encodings
# ------------------------------------------------------


class ModelEncoding(Encoding):
    """Class for model encodings"""

    def __init__(self) -> None:
        """Initializes ModelEncoding class"""
        super().__init__()
        self.activation_tensors: set[str] = set()
        self.param_tensors: str[str] = set()

    def load(self, artifact: str | Path, load_dlc: bool = False, load_json: bool = False) -> None:
        """Loads the model encodings from the given artifact.

        Args:
            artifact: Path to a quantized dlc file or encoding json file. If both load_json and
                load_dlc are False, encoding is loaded as per file extension. Supported file
                extensions for auto load are: .json, .dlc, .encodings. If load_dlc or load_json
                is passed, irespective of the file extension we try to load the encodings as per
                passed load instruction.
            load_dlc: Pass True if the artifact provided is dlc file.
            load_json: Pass True if the artifact provided is encoding json file.

        Raises:
            ValueError: If both load_dlc and load_json are False and artifact extension not one of
                .json, .dlc, .encodings for auto model encoding loading.
        """
        artifact = Path(artifact)
        self.name = artifact.name.split(".")[0]

        # Validate that both flags are not set to True
        if load_dlc and load_json:
            raise ValueError("A artifact can not be both dlc and encoding file at the same time.")

        # If both flags are False, auto-detect based on file extension
        if not load_json and not load_dlc:
            if artifact.suffix.lower() in [".json", ".encodings"]:
                load_json = True
            elif artifact.suffix.lower() == ".dlc":
                load_dlc = True
            else:
                raise ValueError(
                    f"Unsupported file extension '{artifact.suffix}' for auto model encoding load. "
                    "Supported extensions are: .json, .encodings, .dlc"
                )

        if load_dlc:
            self.param_tensors, self.activation_tensors = self._load_from_dlc(
                quantized_dlc_path=artifact.as_posix()
            )
        elif load_json:
            self._load_from_json(encoding_json=read_json(artifact))

    def _load_v0_json(self, encoding_json: dict) -> None:
        """Loads model encodings from version 0.6.0 encoding file.

        Args:
            encoding_json: dictionary of 0.6.0 version model encoding json.
        """
        for encoding_type in encoding_json:
            if encoding_type in ["activation_encodings", "param_encodings"]:
                tensor_type = TensorType(encoding_type)
                for tensor_name, tensor_encoding in encoding_json[encoding_type].items():
                    self.tensor_encodings[tensor_name] = V0JsonTensorEncoding(
                        tensor_name=tensor_name,
                        tensor_encoding=tensor_encoding,
                        tensor_type=tensor_type,
                    )

    def _load_v1_json(self, encoding_json: dict) -> None:
        """Loads model encodings from version 1.0.0 encoding file.

        Args:
            encoding_json: dictionary of 1.0.0 version model encoding json.
        """
        for encoding_type in encoding_json:
            if encoding_type in ["activation_encodings", "param_encodings"]:
                tensor_type = TensorType(encoding_type)
                for tensor_encoding in encoding_json[encoding_type]:
                    tensor_name = tensor_encoding["name"]
                    self.tensor_encodings[tensor_name] = V1JsonTensorEncoding(
                        tensor_encoding=tensor_encoding, tensor_type=tensor_type
                    )

    def _load_v2_json(self, encoding_json: dict) -> None:
        """Loads model encodings from version 2.0.0 encoding file.

        Args:
            encoding_json: dictionary of 2.0.0 version model encoding json.
        """
        for tensor_encoding in encoding_json["encodings"]:
            self.tensor_encodings[tensor_encoding["name"]] = V2JsonTensorEncoding(
                tensor_encoding=tensor_encoding
            )

    def _load_from_json(self, encoding_json: dict) -> None:
        """Loads model encodings from an encoding file.

        Args:
            encoding_json: dictionary model encoding json.
        """
        json_version = get_encodings_version(encodings=encoding_json)

        if json_version == EncodingVersion.V0:
            self._load_v0_json(encoding_json=encoding_json)
        elif json_version == EncodingVersion.V1:
            self._load_v1_json(encoding_json=encoding_json)
        elif json_version == EncodingVersion.V2:
            self._load_v2_json(encoding_json=encoding_json)

    def _load_from_dlc(self, quantized_dlc_path: Path | str) -> tuple[set, set]:
        """Loads model encodings from quantized dlc.

        Args:
            quantized_dlc_path: Path to the quantized dlc file.

        Returns: tuple of:
            1. set: set of param tensor names
            2. set: set of activation tensor names
        """
        param_tensors = set()
        activation_tensors = set()
        model_reader = modeltools.IrDlcReader()
        model_reader.open(str(quantized_dlc_path))  # Convert to string
        ir_graph = model_reader.get_ir_graph()

        # Build a map from tensor name -> producing op type.
        tensor_to_op_type: dict[str, str] = {}
        for ir_op in ir_graph.get_ops():
            for ir_tensor in ir_op.outputs():
                tensor_to_op_type[ir_tensor.name()] = ir_op.type
            # Static (param) tensors are inputs to the op that consumes them.
            for ir_tensor in ir_op.inputs():
                if ir_tensor.is_static_tensor():
                    tensor_to_op_type[ir_tensor.name()] = ir_op.type

        tensor_map = ir_graph.get_tensor_map()
        for tensor_name, ir_tensor in tensor_map.items():
            # Create DlcTensorEncoding object if tensor is quantizable. This eliminates tensors
            # like int64, int32, boolean etc
            if ir_tensor.is_quantizable():
                self.tensor_encodings[tensor_name] = DlcTensorEncoding(
                    ir_tensor=ir_tensor,
                    op_type=tensor_to_op_type.get(tensor_name),
                )
                if ir_tensor.is_static_tensor():
                    param_tensors.update([ir_tensor.name()])
                else:
                    activation_tensors.update([ir_tensor.name()])

        return param_tensors, activation_tensors


# ------------------------------------------------------
# Code block for Subgraph encoding
# ------------------------------------------------------


class SubgraphEncoding(Encoding):
    """Class for Subgraph encodings"""

    def __init__(self):
        """Initializes SubgraphEncoding class"""
        super().__init__()
