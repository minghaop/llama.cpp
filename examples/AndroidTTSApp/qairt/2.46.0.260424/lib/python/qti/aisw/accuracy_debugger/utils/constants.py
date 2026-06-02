# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from enum import Enum

import numpy as np
import onnx
from qti.aisw.tools.core.modules.api import BackendType
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.framework.utils.constants import OnnxFrameworkInfo


class InferenceStatus(Enum):
    ConversionFailure = "ConversionFailure"
    OptimizationFailure = "OptimizationFailure"
    QuantizationFailure = "QuantizationFailure"
    SerializationFailure = "SerializationFailure"
    GenerateBinaryFailure = "GenerateBinaryFailure"
    ExecutionFailure = "ExecutionFailure"
    SUCCESS = "SUCCESS"


class Algorithm(str, Enum):
    ONESHOT = "oneshot"
    LAYERWISE = "layerwise"
    CUMULATIVE = "cumulative_layerwise"


class Component(Enum):
    SNOOPING = "snooping"


class DataType(Enum):
    INTERMEDIATE_OUT_DATATYPE = "float32"


class MaxLimits(Enum):
    max_file_name_size = 255

    # Model has to fit in one DSP of 3.75 GB = 3840 MB
    # To be on safe side we will leave 840MB has buffer and use 3000MB
    max_model_size_with_intermediates = 3000


class HTP_SUPPORTED_PLATFORMS(Enum):
    ANDROID = DevicePlatformType.ANDROID
    X86_64_LINUX = DevicePlatformType.X86_64_LINUX
    WOS = DevicePlatformType.WOS
    QNX = DevicePlatformType.QNX
    LINUX_EMBEDDED = DevicePlatformType.LINUX_EMBEDDED

    @classmethod
    def values(cls):
        """Return the underlying DevicePlatformType values as a set."""
        return {member.value for member in cls}


MATH_INVARIANT_OPS = [
    "cast",
    "constant",
    "reshape",
    "shape",
    "squeeze",
    "transpose",
    "unsqueeze",
    "maxpool",
    "flatten",
    "resize",
    "expand",
    "tile",
    "convert",
    "branch",
    "gather",
    "split",
    "compress",
    "stridedslice",
]

# Relu op types in onnx and tensorflow, if in future some new type is found -> add here
RELU_OPS = ["clip", "relu", "relu6", "leakyrelu", "prelu", "thresholdedrelu", "leaky_relu"]

supported_backends = [
    BackendType.AIC,
    BackendType.HTP,
    BackendType.CPU,
    BackendType.HTP_MCP,
    BackendType.GPU,
]

supported_platforms = [
    DevicePlatformType.ANDROID,
    DevicePlatformType.X86_64_LINUX,
    DevicePlatformType.WOS,
    DevicePlatformType.QNX,
    DevicePlatformType.LINUX_EMBEDDED,
]

supported_frameworks = [OnnxFrameworkInfo.name]

ONNX_NUMPY_DTYPE_MAP = {
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

CSV_TO_JSON_FIELDS_MAP = {
    "Source Name": "source_name",
    "Target Name": "target_name",
    "STATUS": "status",
    "Layer Type": "layer_type",
    "Source Shape": "source_shape",
    "Target Shape": "target_shape",
    "Source(Min, Max, Median)": "source_min_max_median",
    "Target(Min, Max, Median)": "target_min_max_median",
    "INFO": "info",
    "Layer Name": "layer_name",
    "Layer Output": "layer_output",
    "Layer Shape": "layer_shape",
}

GRAPH_FINALIZE_FAILURE_MSG = "Failed to generate binaries! (<NetRunErrorCode.FINALIZE_GRAPHS: 15>, 'Failed to finalize graphs')"
