# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import List, Optional

from pydantic import Field

from qairt.api.configs.common import AISWBaseModel


class EncodingsInfo(AISWBaseModel):
    """
    Represents the encodings information of a tensor.
    """

    bitwidth: int
    """The number of bits used for quantization"""

    max: float
    """The maximum value of the quantized range"""

    min: float
    """The minimum value of the quantized range"""

    scale: float
    """The scale factor used in quantization. This factor is used to map the original values to the quantized range."""

    offset: int
    """The offset used in quantization. This is typically used to shift the range of values."""

    is_symmetric: bool
    """Indicates whether the quantization is symmetric."""

    is_fixed_point: bool
    """Indicates whether the quantization uses fixed-point arithmetic"""


class TensorInfo(AISWBaseModel):
    """
    Represents an input or output tensor in a graph.
    """

    name: str
    """The unique name of the tensor, used to identify it in the graph."""
    dimensions: List[int]
    """List representation of the tensor dimensions"""
    data_type: str
    """The string representation of the tensor dimensions"""
    is_quantized: bool = False
    """Determines if a model is quantized."""
    encodings: Optional[EncodingsInfo] = None
    """Encapsulates the quantization encoding details for a tensor"""


class GraphInfo(AISWBaseModel):
    """
    Represents one graph in the cache, with name
    and lists of input and output tensors.
    """

    name: str
    """The name or identifier of the graph."""
    inputs: List[TensorInfo]
    """A list of input tensors for the graph."""
    outputs: List[TensorInfo]
    """A list of output tensors for the graph."""
