# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
File contains all the operations supported by the MAD library
All op has the same kernel specifications as that of the QNN Op.
For more details about the QNN OP Specifications visit:
https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/MasterOpDef.html
"""

import abc
import math
from abc import ABC
from typing import List

import numpy as np

from qti.aisw.graph_gen.mad.graph.components import Op, Graph, Tensor, TensorDataInfo, AllowedDtypes
from qti.aisw.converters.common import ir_graph as IrGraph

OP_TYPE_TO_OP_CLASS = {}
def register_op_class(cls):
    """
    Decorator to register an operation class with its type.

    :param cls[type]: The operation class to register.
    :return: The registered class.
    """
    op_type = getattr(cls, 'type')
    OP_TYPE_TO_OP_CLASS[op_type] = cls
    return cls


SHARE_BIAS_TENSOR = False
class QcOp(ABC):
    """
    Abstract base class for Quantized Compute Operations.

    All specific operations (like MatMul, ReLU, etc.) should inherit from this class.
    It defines the common interface for operations, including attribute retrieval,
    static input handling, output tensor information, and IR graph generation.

    Attributes:
        leaf_node (bool): Always True for operations, indicating they are leaf nodes.
        type (str): The string identifier for the operation type.
    """
    leaf_node = True
    type = "BASE_OP"

    def __init__(self, *args, **kwargs):
        """
        Initializes the QcOp.

        Args:
            *args: Variable positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        pass
    def __call__(self, *args, **kwargs):
        """
        Enables the operation instance to be called as a function.

        Args:
            *args: Variable positional arguments.
            **kwargs: Arbitrary keyword arguments.
        """
        print(f"Method in Class {self.__class__.__name__} was called")

    def get_attributes(self):
        """
        Retrieves the attributes of the operation.

        Returns:
            dict or None: A dictionary of attributes, or None if no attributes are present.
        """
        return None

    def get_static_inputs(self):
        """
        Retrieves information about static inputs (e.g., weights, biases) for the operation.

        Returns:
            list or None: A list of tuples, where each tuple contains the name and
                          TensorDataInfo for a static input, or None if no static
                          inputs are present.
        """
        return None

    def output_tensor_info(self, tensor_infos):
        """
        Calculates and returns the TensorDataInfo for the output tensor(s) of the operation.

        Args:
            tensor_infos (list): A list of TensorDataInfo objects for the input tensors.

        Returns:
            TensorDataInfo: The TensorDataInfo for the primary output tensor.
        """
        return tensor_infos[0]


    @staticmethod
    @abc.abstractmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the corresponding IR graph operation for this QcOp.

        Args:
            graph: The IR graph object to which the operation will be added.
            op (Op): The MAD Op object representing this operation.
        """
        pass

@register_op_class
class QcMatMul(QcOp):
    """
    Represents a Matrix Multiplication operation.
    """
    type = 'MatMul'

    def output_tensor_info(self, tensor_infos):
        """
        Calculates the output tensor information for a matrix multiplication.

        The output shape is determined by the last dimension of the first input
        and the last dimension of the second input.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the two input matrices.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output matrix.
        """
        out_shape = list(tensor_infos[0].shape)
        out_shape[-1] = tensor_infos[1].shape[-1]

        return TensorDataInfo(out_shape, tensor_infos[0].dtype)

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the MatMul operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
            'transpose_in0': IrGraph.IrAttribute(data_bool = False),
            'transpose_in1': IrGraph.IrAttribute(data_bool = False)
        })

        graph.add_op(IrGraph.MatMulOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

@register_op_class
class QcRelu(QcOp):
    """
    Represents a Rectified Linear Unit (ReLU) activation function.
    """
    type = 'ReLU'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the ReLU operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseNeuronOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )



@register_op_class
class QcGelu(QcOp):
    """
    Represents a Gaussian Error Linear Unit (GELU) activation function.
    """
    type = 'Gelu'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the GELU operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseNeuronOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )


@register_op_class
class QcSigmoid(QcOp):
    """
    Represents a Sigmoid activation function.
    """
    type = 'Sigmoid'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Sigmoid operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseNeuronOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

@register_op_class
class QcSoftmax(QcOp):
    """
    Represents a Softmax activation function.
    """
    type = 'Softmax'
    def __init__(self, dim):
        """
        Initializes the QcSoftmax operation.

        Args:
            dim (int): The dimension along which to apply the Softmax.
        """
        self.dim = dim

    def get_attributes(self):
        """
        Returns the attributes of the Softmax operation.

        Returns:
            dict: A dictionary containing the 'dim' attribute.
        """
        return {"dim":self.dim}

    def output_tensor_info(self, tensor_infos):
        """
        Calculates the output tensor information for a Softmax operation.

        The output shape is the same as the input shape.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.
        """
        return tensor_infos[0] # Output shape is same as input

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Softmax operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'axis': IrGraph.IrAttribute(data_int32 = op.attrs['dim']),
            'beta': IrGraph.IrAttribute(data_float = 1.0),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.SoftmaxOp(op.name, attr), op.inputs, op.outputs)

@register_op_class
class QcTranspose(QcOp):
    """
    Represents a Transpose operation.
    """
    type = 'Transpose'
    def __init__(self, perm):
        """
        Initializes the QcTranspose operation.

        Args:
            perm (list): The permutation of the dimensions.
        """
        # QNN does not support negative indexing .. so in case of negative indexes is there convert it.
        rank = len(perm)
        perm = [(p if p >= 0 else rank + p) for p in perm]
        self.perm = perm

    def get_attributes(self):
        """
        Returns the attributes of the Transpose operation.

        Returns:
            dict: A dictionary containing the 'perm' attribute.
        """
        return {"perm": self.perm}

    def output_tensor_info(self, tensor_infos):
        """
        Calculates the output tensor information for a Transpose operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.
        """
        in_shape = list(tensor_infos[0].shape)
        return TensorDataInfo(np.array(self.perm).choose(in_shape).tolist(), tensor_infos[0].dtype)

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Transpose operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'perm': IrGraph.IrAttribute(name_array = f'{op.name}_perm',data_array = np.array(list(op.attrs['perm']), dtype=np.int32)),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.TransposeOp(op.name, attr), op.inputs, op.outputs)

@register_op_class
class QcConcat(QcOp):
    """
    Represents a Concatenate operation.
    """
    type = 'Concat'
    def __init__(self, dim):
        """
        Initializes the QcConcat operation.

        Args:
            dim (int): The dimension along which to concatenate.
        """
        self.dim = dim

    def get_attributes(self):
        """
        Returns the attributes of the Concat operation.

        Returns:
            dict: A dictionary containing the 'dim' attribute.
        """
        return {'dim': self.dim}

    def output_tensor_info(self, tensor_infos):
        """
        Calculates the output tensor information for a Concat operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensors.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.

        Raises:
            ValueError: If no input tensor information is provided.
        """
        if not tensor_infos:
            raise ValueError("Concat operation requires at least one input tensor")
        out_shape = list(tensor_infos[0].shape)
        concatenated_dim_length = 0
        for tensor_info in tensor_infos:
            concatenated_dim_length += tensor_info.shape[self.dim]

        out_shape[self.dim] = concatenated_dim_length
        return TensorDataInfo(tuple(out_shape), tensor_infos[0].dtype)

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Concat operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'axis': IrGraph.IrAttribute(data_int32 = op.attrs['dim']),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ConcatOp(op.name, attr), op.inputs, op.outputs)


@register_op_class
class QcLinear(QcOp):
    """
    Represents a Fully Connected (Linear) layer operation.
    """
    type = 'FullyConnected'
    def __init__(self,in_features, out_features, bias=True):
        """
        Initializes the QcLinear operation.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def get_attributes(self):
        """
        Returns the attributes of the Linear operation.

        Returns:
            dict: A dictionary containing 'in_features' and 'out_features'.
        """
        return {
            'in_features': self.in_features,
            'out_features': self.out_features,
        }

    def get_static_inputs(self):
        """
        Returns information about static inputs (weight and optional bias) for the Linear operation.

        Returns:
            list: A list of tuples, where each tuple contains the name and
                  TensorDataInfo for a static input.
        """
        static_weight_lists = [('weight', TensorDataInfo(self._get_weight_shape()))]
        if self.bias:
            static_weight_lists.append(('bias', TensorDataInfo(self._get_bias_shape())))

        return static_weight_lists

    def _get_weight_shape(self):
        """
        Calculates the shape of the weight tensor.

        Returns:
            tuple: The shape of the weight tensor.
        """
        return self.out_features, self.in_features

    def _get_bias_shape(self):
        """
        Calculates the shape of the bias tensor.

        Returns:
            tuple or None: The shape of the bias tensor if `bias` is True, otherwise None.
        """
        if self.bias:
            return self.out_features,
        else:
            return None

    def output_tensor_info(self,tensor_infos: List[TensorDataInfo]) -> TensorDataInfo:
        """
        Calculates the output tensor information for a Linear operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.
        """
        out_shape = list(tensor_infos[0].shape)
        out_shape[-1] = self.out_features
        return TensorDataInfo(tuple(out_shape), tensor_infos[0].dtype)

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the FullyConnected (Linear) operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        # Handle bias tensor.
        # QNN OP def requires bias tensor. So in case bias is not present add a zero bias tensor
        # zero bias tensor can be reused. So check if it already exists before adding
        zero_bias_inputs = []
        if len(op.static_inputs) == 1:
            # Get the bias tensor length
            out_features = op.attrs['out_features']
            if SHARE_BIAS_TENSOR:
                bias_tensor_name = f'shared_bias_zeros_{out_features}'
            else:
                bias_tensor_name = f'{op.name}.bias'
            zero_bias_inputs.append(bias_tensor_name)

            # Created tensor only if it does not exist
            if not graph.has_tensor(bias_tensor_name):
                graph.add_static_tensor(name=bias_tensor_name,dataType=IrGraph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,  data=np.zeros((out_features,), dtype=np.float32))
        else:
            bias_tensor_name = op.static_inputs[-1]

        # Create attrs
        attr = IrGraph.IrAttributes({
            'bias_op_name': IrGraph.IrAttribute(data_string = bias_tensor_name, attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })

        graph.add_op(IrGraph.FullyConnectedOp(op.name, attr),
                     op.inputs + op.static_inputs + zero_bias_inputs,
                     op.outputs
                     )

@register_op_class
class QcSqrt(QcOp):
    """
    Represents a Square Root operation.
    """
    type = 'Sqrt'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Sqrt operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseUnaryOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

@register_op_class
class QcDiv(QcOp):
    """
    Represents an Element-wise Division operation.
    """
    type = 'Div'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Div operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseBinaryOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

@register_op_class
class QcAdd(QcOp):
    """
    Represents an Element-wise Addition operation.
    """
    type = 'Add'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Add operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseBinaryOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

@register_op_class
class QcMul(QcOp):
    """
    Represents an Element-wise Multiplication operation.
    """
    type = 'Mul'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Mul operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseBinaryOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

@register_op_class
class QcSub(QcOp):
    """
    Represents an Element-wise Subtraction operation.
    """
    type = 'Sub'

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Sub operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseBinaryOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

@register_op_class
class QcStridedSliceOp(QcOp):
    """
    Represents a Strided Slice operation.
    """
    type = 'StridedSlice'
    def __init__(self, axes = [], slice_ranges = []):
        """
        Initializes the QcStridedSliceOp.

        Args:
            axes (list, optional): The axes along which to slice. Defaults to [].
            slice_ranges (list, optional): The start, stop, and step for each slice. Defaults to [].
        """
        self.axes = axes
        self.slice_range = slice_ranges
        assert len(axes) == len(slice_ranges)

    def get_attributes(self):
        """
        Returns the attributes of the StridedSlice operation.

        Returns:
            dict: A dictionary containing 'axes' and 'ranges'.
        """
        return {'axes':self.axes,
                'ranges': self.slice_range}

    def output_tensor_info(self,tensor_infos: List[TensorDataInfo]) -> TensorDataInfo:
        """
        Calculates the output tensor information for a StridedSlice operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.
        """
        out_shape = list(tensor_infos[0].shape)
        for i, range in zip(self.axes, self.slice_range):
            out_shape[i] = self._count_elements_in_slice(out_shape[i], range[0], range[1], range[2])
        return TensorDataInfo(out_shape, tensor_infos[0].dtype)

    @staticmethod
    def _count_elements_in_slice(length, start, stop, step):
        """
        Helper method to count elements in a slice.

        Args:
            length (int): The length of the dimension.
            start (int): The start index of the slice.
            stop (int): The stop index of the slice.
            step (int): The step of the slice.

        Returns:
            int: The number of elements in the slice.

        Raises:
            ValueError: If the slice step is zero.
        """
        if step == 0:
            raise ValueError("Slice step cannot be zero")
        return max(0, (min(stop, length) - start + step - 1) // step)

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the StridedSlice operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        def get_slice_ranges():
            axes = op.attrs['axes']
            ranges = op.attrs['ranges']
            slice_ranges = []

            # Get input dimension
            in_shape = graph.get_tensor(op.inputs[0]).dims()
            for dim_len in in_shape:
                slice_ranges.append([0, dim_len, 1])

            for i, range in zip(axes, ranges):
                slice_ranges[i] = range
            return slice_ranges

        slice_ranges = get_slice_ranges()
        attr = IrGraph.IrAttributes({
            'begin_mask': IrGraph.IrAttribute(data_uint32 = 0),
            'end_mask': IrGraph.IrAttribute(data_uint32 = 0),
            'new_axes_mask': IrGraph.IrAttribute(data_uint32 = 0),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
            'ranges': IrGraph.IrAttribute(name_array = f'{op.name}_ranges',data_array = np.array(slice_ranges,dtype=np.int32)),
            'shrink_axes': IrGraph.IrAttribute(data_uint32 = 0)
        })

        graph.add_op(IrGraph.StridedSliceOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )



@register_op_class
class QcReshape(QcOp):
    """
    Represents a Reshape operation.
    """
    type = 'Reshape'
    def __init__(self, out_shape):
        """
        Initializes the QcReshape operation.

        Args:
            out_shape (list or tuple): The target output shape.
        """
        self.out_shape = out_shape

    def output_tensor_info(self,tensor_infos: List[TensorDataInfo]) -> TensorDataInfo:
        """
        Calculates the output tensor information for a Reshape operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.
        """
        return TensorDataInfo(self.out_shape, tensor_infos[0].dtype)

    def get_attributes(self):
        """
        Returns the attributes of the Reshape operation.

        Returns:
            dict: A dictionary containing the 'shape' attribute.
        """
        return {'shape': self.out_shape}

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Reshape operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
            'shape': IrGraph.IrAttribute(name_array = f'{op.name}_shape',data_array = np.array(op.attrs['shape'], dtype=np.int32), attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ReshapeOp(op.name, attr),
                     op.inputs,
                     op.outputs
                     )

# TODO: Below Implementation of RMS norm only supports normalization on last axis. Need to update it to make it generic.

@register_op_class
class QcRMSNormOp(QcOp):
    """
    Represents a Root Mean Square Normalization (RMSNorm) operation.
    """
    type = 'RMSNorm'

    def __init__(self, eps, axis, out_channel, bias = True):
        """
        Initializes the QcRMSNormOp.

        Args:
            eps (float): A small value added to the denominator for numerical stability.
            axis (int): The axis along which to perform normalization.
            out_channel (int): The number of output channels.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        self.eps = eps
        self.axis = axis
        self.out_channel = out_channel
        self.bias = bias

    def get_attributes(self):
        """
        Returns the attributes of the RMSNorm operation.

        Returns:
            dict: A dictionary containing 'eps', 'axis', and 'out_channel'.
        """
        return {'eps': self.eps, 'axis': self.axis, 'out_channel': self.out_channel}


    def get_static_inputs(self):
        """
        Returns information about static inputs (weight and optional bias) for the RMSNorm operation.

        Returns:
            list: A list of tuples, where each tuple contains the name and
                  TensorDataInfo for a static input.
        """
        static_weight_lists = [('weight', TensorDataInfo((self.out_channel,)))]
        if self.bias:
            static_weight_lists.append(('bias', TensorDataInfo((self.out_channel,))))

        return static_weight_lists

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the RMSNorm operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        # Add bias tensor is bias is not present in the op. Similar to as that of Linear op
        zero_bias_inputs = []
        if len(op.static_inputs) == 1:
            # Get the bias tensor length
            out_features = op.attrs['out_channel']
            if SHARE_BIAS_TENSOR:
                bias_tensor_name = f'shared_bias_zeros_{out_features}'
            else:
                bias_tensor_name = f'{op.name}.bias'
            zero_bias_inputs.append(bias_tensor_name)

            # Created tensor only if it does not exist
            if not graph.has_tensor(bias_tensor_name):
                graph.add_static_tensor(name=bias_tensor_name,dataType=IrGraph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,  data=np.zeros((out_features,), dtype=np.float32))

        attr = IrGraph.IrAttributes({
            'axes': IrGraph.IrAttribute(name_array = f'{op.name}_axes',data_array = np.array([op.attrs['axis']],dtype=np.uint32)),
            'epsilon': IrGraph.IrAttribute(data_float = op.attrs['eps']),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.RMSNormOp(op.name, attr),
                     op.inputs + op.static_inputs + zero_bias_inputs,
                     op.outputs
                     )

@register_op_class
class QcEmbedding(QcOp):
    """
    Represents an Embedding lookup operation.
    """
    type = 'Embedding'

    def __init__(self, num_embeddings, embedding_dimension):
        """
        Initializes the QcEmbedding operation.

        Args:
            num_embeddings (int): The size of the vocabulary.
            embedding_dimension (int): The size of each embedding vector.
        """
        self.num_embeddings = num_embeddings
        self.embedding_dimension = embedding_dimension


    def get_attributes(self):
        """
        Returns the attributes of the Embedding operation.

        Returns:
            dict: A dictionary containing 'num_embeddings' and 'embedding_dimension'.
        """
        return {'num_embeddings': self.num_embeddings, 'embedding_dimension': self.embedding_dimension}

    def get_static_inputs(self):
        """
        Returns information about static inputs (weight) for the Embedding operation.

        Returns:
            list: A list of tuples, where each tuple contains the name and
                  TensorDataInfo for the weight input.
        """
        static_weight_lists = [('weight', TensorDataInfo((self.num_embeddings, self.embedding_dimension)))]
        return static_weight_lists

    def output_tensor_info(self,tensor_infos: List[TensorDataInfo]) -> TensorDataInfo:
        """
        Calculates the output tensor information for an Embedding operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.
        """
        out_shape = list(tensor_infos[0].shape)
        out_shape.append(self.embedding_dimension)
        return TensorDataInfo(tuple(out_shape))

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Gather operation (used for Embedding).

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        attr = IrGraph.IrAttributes({
            'axis': IrGraph.IrAttribute(data_int32 = 0),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.GatherOp(op.name, attr), op.static_inputs + op.inputs, op.outputs)

@register_op_class
class QcConv(QcOp):
    """
    Represents a 2D Convolution operation.
    """
    type = 'Conv2D'

    def __init__(self, in_channel, out_channel, kernel_size = (1,1), stride = (1,1), dilation = (1,1), group = 1, pad_amount = [[0,0],[0,0]], bias = True):
        """
        Initializes the QcConv operation.

        Args:
            in_channel (int): The number of input channels.
            out_channel (int): The number of output channels.
            kernel_size (tuple, optional): The size of the convolution kernel. Defaults to (1,1).
            stride (tuple, optional): The stride of the convolution. Defaults to (1,1).
            dilation (tuple, optional): The dilation rate of the convolution. Defaults to (1,1).
            group (int, optional): The number of groups for grouped convolution. Defaults to 1.
            pad_amount (list, optional): The padding amount for height and width. Defaults to [[0,0],[0,0]].
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.pad_amount = pad_amount
        self.bias = bias


    def get_attributes(self):
        """
        Returns the attributes of the Conv2D operation.

        Returns:
            dict: A dictionary containing various convolution parameters.
        """
        return {
            'in_channel': self.in_channel,
            'out_channel': self.out_channel,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dilation': self.dilation,
            'group': self.group,
            'pad_amount': self.pad_amount,
        }

    def get_static_inputs(self):
        """
        Returns information about static inputs (weight and optional bias) for the Conv2D operation.

        Returns:
            list: A list of tuples, where each tuple contains the name and
                  TensorDataInfo for a static input.
        """
        static_weight_lists = [('weight', TensorDataInfo(self._get_weight_shape()))]
        if self.bias:
            static_weight_lists.append(('bias', TensorDataInfo(self._get_bias_shape())))

        return static_weight_lists

    def _get_weight_shape(self):
        """
        Calculates the shape of the weight tensor for Conv2D.

        Returns:
            tuple: The shape of the weight tensor.
        """
        return *self.kernel_size, self.in_channel, self.out_channel

    def _get_bias_shape(self):
        """
        Calculates the shape of the bias tensor for Conv2D.

        Returns:
            tuple or None: The shape of the bias tensor if `bias` is True, otherwise None.
        """
        if self.bias:
            return self.out_channel,
        else:
            return None

    def output_tensor_info(self,tensor_infos: List[TensorDataInfo]) -> TensorDataInfo:
        """
        Calculates the output tensor information for a Conv2D operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor.
        """
        in_shape = tensor_infos[0].shape
        dilated_filter_height = (self.kernel_size[0] - 1) * self.dilation[0] + 1
        dilated_filter_width = (self.kernel_size[1] - 1) * self.dilation[1] + 1
        height_out = math.floor((self.pad_amount[0][0] + in_shape[1] + self.pad_amount[0][1] - dilated_filter_height) / self.stride[0] + 1)
        width_out = math.floor((self.pad_amount[1][0] + in_shape[2] + self.pad_amount[1][1] - dilated_filter_width) / self.stride[1] + 1)

        out_shape = list(in_shape)
        out_shape[1] = height_out
        out_shape[2] = width_out
        out_shape[-1] = self.out_channel

        return TensorDataInfo(tuple(out_shape), tensor_infos[0].dtype)

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Conv2d operation.

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        zero_bias_inputs = []
        if len(op.static_inputs) == 1:
            # Get the bias tensor length
            out_channel = op.attrs['out_channel']
            if SHARE_BIAS_TENSOR:
                bias_tensor_name = f'shared_bias_zeros_{out_channel}'
            else:
                bias_tensor_name = f'{op.name}.bias'
            zero_bias_inputs.append(bias_tensor_name)

            # Created tensor only if it does not exist
            if not graph.has_tensor(bias_tensor_name):
                graph.add_static_tensor(name=bias_tensor_name,dataType=IrGraph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,  data=np.zeros((out_channel,), dtype=np.float32))
        else:
            bias_tensor_name = op.static_inputs[-1]

        attr = IrGraph.IrAttributes({
            'bias_op_name': IrGraph.IrAttribute(data_string = bias_tensor_name, attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
            'dilation': IrGraph.IrAttribute(name_array = f'{op.name}_dilation',data_array = np.array(op.attrs['dilation'],dtype=np.uint32)),
            'group': IrGraph.IrAttribute(data_uint32 = op.attrs['group']),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
            'pad_amount': IrGraph.IrAttribute(name_array = f'{op.name}_pad_amount',data_array = np.array(op.attrs['pad_amount'],dtype=np.uint32)),
            # 'padding_size_strategy': IrGraph.IrAttribute(data_uint8 = 5, attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL),
            'stride': IrGraph.IrAttribute(name_array = f'{op.name}_stride',data_array = np.array(op.attrs['stride'],dtype=np.uint32))
        })
        graph.add_op(IrGraph.Conv2dOp(op.name, attr),
                     op.inputs + op.static_inputs + zero_bias_inputs,
                     op.outputs
                     )

@register_op_class
class QcExpand(QcOp):
    """
    Represents an Expand operation.

    This operation is implemented in terms of element-wise multiplication with a tensor of ones
    that has the target shape.
    """
    type = 'Expand'
    def __init__(self, shape):
        """
        Initializes the QcExpand operation.

        Args:
            shape (list or tuple): The target shape to expand the input tensor to.
        """
        self.shape = shape

    def get_attributes(self):
        """
        Returns the attributes of the Expand operation.

        Returns:
            dict: A dictionary containing the 'shape' attribute.
        """
        return {'shape': self.shape}

    def output_tensor_info(self,tensor_infos: List[TensorDataInfo]) -> TensorDataInfo:
        """
        Calculates the output tensor information for an Expand operation.

        Args:
            tensor_infos (list): A list containing TensorDataInfo for the input tensor.

        Returns:
            TensorDataInfo: The TensorDataInfo for the output tensor, which has the target shape.
        """
        return TensorDataInfo(self.shape, tensor_infos[0].dtype)

    @staticmethod
    def generate_ir_graph_op(graph, op: Op):
        """
        Generates the IR graph for the Expand operation, using an ElementwiseBinaryOp (Multiply).

        Args:
            graph: The IR graph object.
            op (Op): The MAD Op object.
        """
        # Create a tensor for expand coeff
        coeff_tensor_name = f'{op.name}_coeff'
        data = np.ascontiguousarray(np.ones(op.attrs['shape']).astype(np.float32))
        graph.add_static_tensor(name=coeff_tensor_name,dataType=IrGraph.Qnn_DataType_t.QNN_DATATYPE_FLOAT_32,  data=data)
        attr = IrGraph.IrAttributes({
            'operation': IrGraph.IrAttribute(data_uint32 = IrGraph.QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY),
            'packageName': IrGraph.IrAttribute(data_string = 'qti.aisw', attrUsage=IrGraph.IrAttrUsageType.IR_ATTR_USAGE_SUPPLEMENTAL)
        })
        graph.add_op(IrGraph.ElementwiseBinaryOp(op.name, attr),
                     op.inputs + [coeff_tensor_name],
                     op.outputs
                     )