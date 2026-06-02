# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from logging import Logger
from pathlib import Path
from typing import Optional

import onnx
from onnx import ModelProto
from qti.aisw.accuracy_debugger.graph_op.framework_op import FrameworkOp
from qti.aisw.accuracy_debugger.utils.constants import ONNX_NUMPY_DTYPE_MAP
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_framework import OnnxFramework
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import (
    OnnxModelHelper,
)
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper


class CustomOnnxFramework(OnnxFramework):
    """Class representing the Onnx framework.

    This class provides methods for loading and creating connectd graph
    """

    def __init__(self, logger: Logger):
        """Init function for onnx framework class
        Args:
            logger (Logger): A python Logger instance
        """
        super().__init__(logger)

    def get_output_tensor_names(self, model: ModelProto) -> list:
        """Get the output tensor names from the model

        Args:
            model(ModelProto) : The loaded model.

        Returns:
            list: A list of output tensor names
        """
        output_tensor_names = OnnxModelHelper.get_output_names(model)
        return output_tensor_names

    def create_connected_graph(self, model: ModelProto) -> dict:
        """Create a map of node name to Op object

        Args:
            model(ModelProto) : The loaded model.

        Returns:
            dict: A dictionary mapping node names to their corresponding Op objects
        """
        graph = model.graph
        connected_graph = {}
        for idx, inp in enumerate(graph.input):
            name = f"input_{idx}"
            op = FrameworkOp(name)
            op.inputs = []
            op.outputs = [inp.name]
            op.op_type = "input"
            connected_graph[name] = op

        for idx, node in enumerate(graph.node):
            name = node.name if node.name else f"op_{idx}"
            op = FrameworkOp(name)
            op.inputs = node.input
            op.outputs = node.output
            op.op_type = node.op_type
            connected_graph[name] = op

        # Now set the children and parent ops for each op
        for _, node1 in connected_graph.items():
            for _, node2 in connected_graph.items():
                for output in node1.outputs:
                    if output in node2.inputs:
                        # node1 -> node2
                        node1.children_ops = [node2]
                        node2.parent_ops = [node1]

        return connected_graph

    def get_intermediate_outputs_info(self, model_path: Path) -> dict:
        """Fetches datatypes and shapes information of all tensors present in the given model

        Args:
            model_path: Path to model file

        Returns:
            dict: A dictionary containing datatypes and shapes information for each tensor
        """
        model = onnx.load(model_path)
        model = onnx.shape_inference.infer_shapes(model)
        model.graph.value_info.extend(model.graph.output)
        intermediate_outputs_info = {}
        for value_info in model.graph.value_info:
            sanitized_node_name = Helper.transform_node_names(value_info.name)
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            datatype_in_onnx = value_info.type.tensor_type.elem_type
            datatype = ONNX_NUMPY_DTYPE_MAP.get(datatype_in_onnx, None).__name__
            intermediate_outputs_info[sanitized_node_name] = {"dtype": datatype, "shape": shape}

        return intermediate_outputs_info

    def get_input_tensor_details(
        self, model_path: Path, onnx_symbols: Optional[list] = None
    ) -> list[tuple]:
        """Fetches names, dimensions and data type of input tensor

        Args:
            model_path: Path to model file.
            onnx_symbols: List of tuple containing the mapping of input symbols to their actual values.

        Returns:
            list[dict]: A list of dictionary containing datatypes and shapes information for each
                        tensor.

        Raises:
            ValueError: If symbol value is not supplied for the input tensor.
        """
        model = onnx.load(model_path)
        input_tensor_details = []
        symbols_dict = {}
        if onnx_symbols:
            symbols_dict = {symbol[0]: int(symbol[1]) for symbol in onnx_symbols}

        initializer_names = set(init.name for init in model.graph.initializer)

        # Extract the input tensor information from the model
        for input_tensor in model.graph.input:
            input_tensor_name = input_tensor.name
            # Filter initializers from input tensors.
            if input_tensor_name in initializer_names:
                continue

            dim_len = len(input_tensor.type.tensor_type.shape.dim)
            shape = []
            for i in range(dim_len):
                _symbol = input_tensor.type.tensor_type.shape.dim[i].dim_param
                if len(_symbol) > 0:
                    if _symbol in symbols_dict:
                        shape.append(symbols_dict[_symbol])
                    else:
                        raise ValueError("Please supply value for onnx symbol: {}".format(_symbol))

                else:
                    shape.append(input_tensor.type.tensor_type.shape.dim[i].dim_value)

            data_type = input_tensor.type.tensor_type.elem_type
            resolved_data_type = None

            # Old mapping (onnx <= 1.18.x): maybe useful in some model debugging
            mapping_mod = getattr(onnx, "mapping", None)
            if mapping_mod is not None and hasattr(mapping_mod, "TENSOR_TYPE_TO_NP_TYPE"):
                resolved_data_type = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[data_type]

            # For onnx >= 1.19.0
            helper = getattr(onnx, "helper", None)
            if (
                not resolved_data_type
                and helper is not None
                and hasattr(helper, "tensor_dtype_to_np_dtype")
            ):
                resolved_data_type = onnx.helper.tensor_dtype_to_np_dtype(data_type)

            input_tensor_details.append(
                {"name": input_tensor_name, "shape": shape, "data_type": str(resolved_data_type)}
            )
        return input_tensor_details

    @classmethod
    def is_qdq_model(cls, model_path: Path) -> bool:
        """Check if an ONNX model is in QDQ (Quantize-Dequantize) format.

        This method determines whether a model uses quantization by scanning for
        QuantizeLinear or DequantizeLinear operations in the model graph. QDQ models
        explicitly represent quantization operations as graph nodes, as opposed to
        models with quantized weights stored in a different format.

        The QDQ format is commonly used for:
        - Quantization-aware training (QAT)
        - Post-training quantization (PTQ) with explicit Q/DQ nodes

        Args:
            model_path: Path to ONNX model

        Returns:
            bool: True if the model contains at least one QuantizeLinear or
                DequantizeLinear operation, False otherwise.

        Note:
            - This method only checks for the presence of Q/DQ nodes, not their correctness
            - A model with even a single Q/DQ node is considered a QDQ model
        """
        model = onnx.load(model_path)
        for node in model.graph.node:
            # Check if the node's operation type starts with 'Quantize' or 'Dequantize'
            # This catches both 'QuantizeLinear' and 'DequantizeLinear' operations
            # which are the standard ONNX quantization operators
            if node.op_type.startswith(("Quantize", "Dequantize")):
                # Found at least one quantization operation - model is QDQ format
                return True
        else:
            # No quantization operations found after checking all nodes
            # Model is either FP32 or uses a different quantization representation
            return False

    @classmethod
    def validate_model(cls, model_path, validate_qdq=False):
        """Validate the model file path, format, and QDQ nodes (when validate_qdq is set to True).

        This validator performs three critical checks on the QDQ model:
        1. Verifies that the model file exists at the specified path
        2. Ensures the file has a .onnx extension indicating ONNX format
        3. Confirms the model contains QDQ (QuantizeLinear/DequantizeLinear) nodes (if specified)

        Raises:
            FileNotFoundError: If the QDQ model file does not exist at the specified path.
            ValueError: If the file is not an ONNX model or lacks QDQ nodes required
                for quantization-aware inference when validate_qdq is set to True.
        """
        # Check if the model file exists on the filesystem
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

        # Verify the file has the correct .onnx extension
        if model_path.suffix != ".onnx":
            raise ValueError("Invalid model format specified. Please provide a valid ONNX file.")

        # Verify it contains QDQ nodes when validate_qdq is set to True
        if validate_qdq and not cls.is_qdq_model(model_path):
            raise ValueError(
                "Model doesn't contain QDQ nodes. Please provide a valid QDQ ONNX file."
            )
