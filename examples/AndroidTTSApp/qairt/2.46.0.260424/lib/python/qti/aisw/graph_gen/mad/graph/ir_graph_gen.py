# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
""" File contains the code to export a MAD Graph representation to an QNN IR Graph """

from typing import Dict

import numpy as np

from qti.aisw.graph_gen.mad.graph.components import TensorTypes, ProducerTypes
from qti.aisw.graph_gen.mad.lib.op import *
from qti.aisw.converters.common import ir_graph as IrGraph
from qti.aisw.converters.common import modeltools
from qti.aisw.graph_gen.mad.utils import apply_encoding, save_ir_graph_to_dlc, quantize_graph

def is_quantizable(dtype):
    """
    Checks if a given data type is quantizable (float16 or float32).

    :param dtype[AllowedDtypes]: The data type to check.
    :return: True if the dtype is quantizable, False otherwise.
    """
    return dtype in (AllowedDtypes.FLOAT16, AllowedDtypes.FLOAT32)


class IrGraphGenerator:
    """
    Generates an IR (Intermediate Representation) graph from a given MAD graph definition.

    This class takes a MAD graph, along with optional weights and encoding information,
    and constructs a corresponding IR graph that can be serialized or further processed.

    :param op_type_to_opdef[dict]: A mapping from MAD operation types to IR operation definitions.
    :param allowedDtypes_to_QNN_TYPES[dict]: A mapping from AllowedDtypes to QNN data types.
    :param quantize[bool]: Whether to apply quantization during graph generation.
    :param graph[Graph]: The input MAD graph definition.
    :param weight_dict[Dict]: A dictionary of weights to be used in the graph.
    :param encoding_file_path[str]: Path to the encoding file for quantization.
    :param ir_graph[IrGraph.IrGraph]: The generated IR graph object.
    """

    op_type_to_opdef = OP_TYPE_TO_OP_CLASS
    allowedDtypes_to_QNN_TYPES = {
        AllowedDtypes.FLOAT32: IrGraph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,
        AllowedDtypes.FLOAT16: IrGraph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_16,
        AllowedDtypes.INT32: IrGraph.Qnn_DataType_t.QNN_DATATYPE_INT_32,
        AllowedDtypes.BOOL: IrGraph.Qnn_DataType_t.QNN_DATATYPE_BOOL_8,
        AllowedDtypes.UINT32: IrGraph.Qnn_DataType_t.QNN_DATATYPE_UINT_32
    }
    def __init__(self, graph: Graph, weight_dict: Dict, encoding_file_path: str = None, quantize: bool = False, graph_name = None):
        """
        Initializes the IrGraphGenerator.

        :param graph[Graph]: The MAD graph definition.
        :param weight_dict[Dict]: Dictionary containing tensor weights.
        :param encoding_file_path[str, optional]: Path to a JSON file containing quantization encodings. Defaults to None.
        :param quantize[bool, optional]: If True, the graph will be quantized after applying encodings. Defaults to False.
        """
        self.quantize = quantize
        self.graph = graph
        self.weight_dict = weight_dict
        self.encoding_file_path = encoding_file_path
        self.ir_graph = IrGraph.IrGraph(
            name=graph.name if graph_name is None else graph_name
        )

    def generate(self):
        """
        Generates the IR graph by creating IR graph elements and setting I/O types.
        """
        # Generate the ir_graph elements for graph object
        self.create_ir_graph_elements(self.graph)

        # Set input and output tensors types
        for input in self.graph.inputs:
            # self.ir_graph.get_tensor(input).set_tensor_type(IrGraph.IR_TENSOR_TYPE_APP_WRITE)
            self.ir_graph.update_tensor_type(input, IrGraph.IR_TENSOR_TYPE_APP_WRITE)

        for output in self.graph.outputs:
            # self.ir_graph.get_tensor(output).set_tensor_type(IrGraph.IR_TENSOR_TYPE_APP_READ)
            self.ir_graph.update_tensor_type(output, IrGraph.IR_TENSOR_TYPE_APP_READ)


    def process_overrides(self):
        """
        If the encoding file is present apply the override and optionally quantizes the graph.
        """
        # Apply the encoding if present
        if self.encoding_file_path is not None:
            apply_encoding(self.ir_graph, self.encoding_file_path)

            # Quantize in float fallback mode
            if self.quantize:
                quantize_graph(self.ir_graph)


    def save(self, save_path):
        """
        Saves the generated IR graph to a DLC file.

        :param save_path[str]: The path where the DLC file will be saved.
        """
        # serialize the graph
        save_ir_graph_to_dlc(self.ir_graph, save_path)


    def create_ir_graph_elements(self, graph):
        """
        Recursively creates IR graph elements (tensors and operations) from a MAD graph.

        :param graph[Graph]: The MAD graph or subgraph to process.
        """
        # Create IrTensors for Tensors
        for name, tensor in graph.tensors.items():
            # Check if the tensor is already added to the ir_graph.
            # This will happen when the tensor is already created in the parent graph.
            if self.ir_graph.has_tensor(name):
                continue

            if tensor.type == TensorTypes.STATIC:
                # TODO: Make the dtype assignment generic. Create a mapping from np types to Qnn types
                if tensor.producer == ProducerTypes.CONSTANT:
                    val = tensor.value
                    # Check for scalar data and convert in to tensor as QNN does not support scalar constant values
                    if isinstance(tensor.data_info.shape, int):
                        val = np.array([val]).astype(np.float32)

                    self.ir_graph.add_static_tensor(name=tensor.name,dataType=self.allowedDtypes_to_QNN_TYPES[tensor.data_info.dtype],  data=val)
                else:
                    self.ir_graph.add_static_tensor(name=tensor.name,dataType=self.allowedDtypes_to_QNN_TYPES[tensor.data_info.dtype],  data=np.ascontiguousarray(self.get_tensor_data(tensor)))
            else:
                shape = tensor.data_info.shape
                if isinstance(shape, int):
                    shape = [shape]
                self.ir_graph.add_tensor(name=tensor.name, dims=shape, dataType=self.allowedDtypes_to_QNN_TYPES[tensor.data_info.dtype], quantizable = is_quantizable(tensor.data_info.dtype))

        # Create Ops
        for name, op in graph.ops.items():
            if isinstance(op, Op):
                try:
                    self.op_type_to_opdef[op.type].generate_ir_graph_op(self.ir_graph, op)
                except ValueError as e:
                    raise ValueError(f"Invalid configuration for {op}. \nAdditional Info: " +  str(e))
            elif isinstance(op, Graph):
                self.create_ir_graph_elements(op)

    def get_tensor_data(self, tensor:Tensor):
        """
        Retrieves the data for a given tensor.

        If `weight_dict` is provided during initialization, it attempts to fetch the tensor's
        value from there. Otherwise, it generates random data.

        :param tensor[Tensor]: The tensor for which to retrieve data.
        :return: The tensor's data.
        """
        if self.weight_dict is not None:
            return self.weight_dict[tensor.name]
        else:
            return np.random.rand(*tensor.data_info.shape).astype(np.float32)
