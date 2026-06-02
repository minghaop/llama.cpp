# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
""" File contains components used by the MAD library to represent a Graph """

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union
import numpy as np


class TensorTypes(Enum):
    """Enumerates the types of tensors: STATIC or DYNAMIC."""
    STATIC = 1
    DYNAMIC = 2

class ProducerTypes(Enum):
    """Enumerates the types of producers for a tensor."""
    MODULE_INPUT = 1
    CONSTANT = 2

class AllowedDtypes(Enum):
    """Enumerates the allowed data types for tensors."""
    FLOAT32 = np.float32,
    INT32 = np.int32,
    BOOL = np.bool_,
    UINT32 = np.uint32,
    INT64 = np.int64,
    FLOAT16 = np.float16,

@dataclass
class Op:
    """
    Represents an operation in the graph.

    :param name[str]: The name of the operation.
    :param type[str]: The type of the operation (e.g., "conv", "add").
    :param attrs[dict]: A dictionary of attributes for the operation.
    :param inputs[list]: A list of input tensor names to this operation.
    :param outputs[list]: A list of output tensor names from this operation.
    :param static_inputs[list]: A list of static input tensor names to this operation.
    """
    name: str
    type: str
    attrs: dict  = field(default_factory=dict)
    inputs: list  = field(default_factory=list)
    outputs: list  = field(default_factory=list)
    static_inputs: list  = field(default_factory=list)

@dataclass
class TensorDataInfo:
    """
    Holds information about a tensor's data.

    :param shape[tuple]: The shape of the tensor.
    :param dtype[AllowedDtypes]: The data type of the tensor. Defaults to FLOAT32.
    """
    shape: tuple
    dtype: AllowedDtypes = AllowedDtypes.FLOAT32

@dataclass
class Tensor:
    """
    Represents a tensor in the graph.

    :param name[str]: The name of the tensor.
    :param type[TensorTypes]: The type of the tensor (STATIC or DYNAMIC).
    :param producer[Union[Op, ProducerTypes]]: The operation or type that produced this tensor.
    :param data_info[TensorDataInfo]: Information about the tensor's data (shape, dtype).
    :param consumer[list]: A list of operations that consume this tensor.
    :param value[Any]: The actual value of the tensor if it's static, otherwise None.
    """
    name: str
    type: TensorTypes
    producer: Union[Op, ProducerTypes]
    data_info: TensorDataInfo
    consumer: list  = field(default_factory=list)
    value: Any = None

@dataclass
class Graph:
    """
    Represents a computational graph.

    :param name[str]: The name of the graph.
    :param ops[dict]: A dictionary mapping operation names to Op objects.
    :param tensors[dict]: A dictionary mapping tensor names to Tensor objects.
    :param inputs[list]: A list of input tensor names for the graph.
    :param outputs[list]: A list of output tensor names for the graph.
    :param ordered_ops[list]: A topologically sorted list of operation names.
    :param split_info[list]: Information regarding graph splitting, if applicable.
    """
    name: str
    ops: dict  = field(default_factory=dict)
    tensors:dict  = field(default_factory=dict)
    inputs:list =  field(default_factory=list)
    outputs:list = field(default_factory=list)
    ordered_ops: list = field(default_factory=list)
    split_info: list = field(default_factory=list)
