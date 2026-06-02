# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
This module provides a pass for shape inference
"""

import copy
import os
from dataclasses import dataclass

import numpy as np
import onnx
import onnx_ir as ir
import onnxscript.evaluator

from qairt.optimizer.onnx.graph import GraphContext
from qairt.optimizer.onnx.passes.base import BasePass, BaseTreeVisitor, PassConfig
from qairt.optimizer.onnx.passes.mha2sha.utils import BroadcastHelper
from qairt.optimizer.onnx.utils.utils import (
    clean_model_proto,
    convert_attr_to_py,
    convert_attrs_to_py,
    get_constant_np,
    get_shape_from_value_info_proto,
    get_value_numeric_shape,
    has_static_shape_on_value,
    iter_all_values,
)
from qairt.optimizer.utils.logger import logger

# Shape-related size threshold: 1MB
# Tensors smaller than this will be preloaded into memory for shape inference
CONSTANT_FOLDING_SIZE_THRESHOLD = 1024 * 1024  # 1 MB in bytes


def serialize_model_without_data(model: ir.Model, size_threshold_bytes=128) -> onnx.ModelProto:
    """Serialize ir.Model into onnx.ModelProto, without large data
    The large data (larger than size_threshold_bytes) will be set to a dummy empty external file.
    modified from ir.external_data.unload_from_model and ir.save

    Args:
        model: model to serialize ir.Model
        size_threshold_bytes: size threshold, in byte
    Returns:
        proto: serialized protobuf
    """
    # Store the original initializer values so they can be restored
    initializer_values = tuple(model.graph.initializers.values())
    tensors = [v.const_value for v in initializer_values]

    try:
        # In-memory or external tensors, if equal to or above the threshold,
        # should be converted to or re-saved as external tensors

        for value in model.graph.initializers.values():
            if value.const_value is None:
                # Filter out the uninitialized initializer values
                continue
            if value.const_value.nbytes > size_threshold_bytes:
                dummy_external_tensor = ir.ExternalTensor(
                    os.path.normpath("./dummy_data_for_shape_infer.data"),
                    0,
                    value.const_value.nbytes,
                    value.dtype,  # type: ignore[arg-type]
                    shape=value.shape,  # type: ignore[arg-type]
                    name=value.name,  # type: ignore[arg-type]
                )
                value.const_value = dummy_external_tensor

        proto = ir.serde.serialize_model(model)

    finally:
        # Restore the original initializer values so the model is unchanged
        for initializer, tensor in zip(initializer_values, tensors, strict=True):
            initializer.const_value = tensor
    return proto


def serialize_model_with_folded_constant(model: ir.Model, with_data=True) -> onnx.ModelProto:
    """Serialize ir.Model into onnx.ModelProto, constant will be folded

    Args:
        model: model to serialize ir.Model
        with_data: whether to serialize with large data
    Returns:
        proto: serialized protobuf
    """
    if with_data:
        proto = ir.serde.serialize_model(model)
    else:
        proto = serialize_model_without_data(model)

    constant_values = {}
    # iterate for activations
    for node in model.graph:
        if node.op_type == "Constant":
            continue
        for output in node.outputs:
            if output.meta["extra_info"].infered_constant_value is not None:
                constant_values[output.name] = output.meta["extra_info"].infered_constant_value

    new_cst_nodes = []
    new_node_lists = []
    for node in proto.graph.node:
        # remove node whose output is constant
        all_output_cst = True
        for v_i, v in enumerate(node.output):
            if v in constant_values:
                cst_node = onnx.helper.make_node(
                    "Constant",
                    [],
                    [v],
                    name=v + ".cst",
                    domain="",
                    value=onnx.numpy_helper.from_array(constant_values[v], name=v),
                )
                new_cst_nodes.append(cst_node)
                # rename origin v to other name to avoid conflict
                # TODO, handle renaming properly
                node.output[v_i] = v + "_###.##_origin##_$"
            else:
                all_output_cst = False
        if not all_output_cst:
            new_node_lists.append(node)

    while len(proto.graph.node) > 0:
        proto.graph.node.pop()
    proto.graph.node.extend(new_cst_nodes)
    proto.graph.node.extend(new_node_lists)

    # after setting some nodes to constant,
    # some nodes maybe not used, so remove them

    proto = clean_model_proto(proto)

    return proto


class PreloadSmallExternalTensors(BasePass):
    """Pass to preload small external tensors into memory before shape inference.

    This pass addresses two issues:
    1. Too many file handlers open during shape inference
    2. Large tensors causing memory issues

    By preloading only small tensors (< shape_related_size_threshold) into memory
    and replacing ExternalTensor with in-memory tensors, we:
    - Avoid keeping many file handlers open
    - Prevent loading large tensors unnecessarily
    - File handlers are released manually once the tensors are loaded
    """

    def __init__(self, size_threshold=CONSTANT_FOLDING_SIZE_THRESHOLD):
        """Initialize an instance of PreloadSmallExternalTensors

        Args:
            size_threshold: size threshold in bytes for preloading tensors
        """
        super().__init__()
        self.size_threshold = size_threshold

    def apply(self, ctx: GraphContext) -> int:
        """Preload small external tensors into memory.

        Iterates through all values and Constant nodes, replacing ExternalTensor
        instances with in-memory numpy arrays if their size is below the threshold.

        Explicitly releases file handles after loading each tensor.
        """
        preloaded_count = 0
        skipped_count = 0

        # Handle external tensors in value.const_value (initializers and some activations)
        for value in iter_all_values(ctx.graph_ir):
            if value.const_value is None:
                continue

            # Check if this is an ExternalTensor
            if isinstance(value.const_value, ir.ExternalTensor):
                external_tensor = value.const_value

                # Only load if size is below threshold
                if external_tensor.nbytes <= self.size_threshold:
                    try:
                        # Load the external tensor data into memory
                        # Make a copy to ensure data is fully loaded before releasing
                        loaded_array = external_tensor.numpy().copy()

                        # Explicitly release the file handle
                        # This is needed to avoid "Too many files open" issue
                        # This can potentially happen in a large QDQ model
                        # where each Q/DQ node can open upto two mmap files for
                        # its quantization fields y_scale and y_zero_point
                        external_tensor.release()

                        # Replace the ExternalTensor with the in-memory array
                        value.const_value = ir.Tensor(loaded_array)
                        preloaded_count += 1

                    except Exception as e:
                        logger.warning(f"Failed to preload external tensor '{value.name}': {e}")
                        skipped_count += 1
                else:
                    # Skip large tensors
                    skipped_count += 1

        # Handle external tensors in Constant node attributes
        for node in ctx.graph_ir:
            if node.op_type == "Constant":
                if "value" not in node.attributes:
                    continue

                attr = node.attributes["value"]
                if attr.type != ir.AttributeType.TENSOR:
                    continue

                if isinstance(attr.value, ir.ExternalTensor):
                    external_tensor = attr.value

                    # Only load if size is below threshold
                    if external_tensor.nbytes <= self.size_threshold:
                        try:
                            # Load the external tensor data into memory
                            loaded_array = external_tensor.numpy().copy()

                            # Explicitly release the file handle
                            external_tensor.release()

                            # Replace the ExternalTensor with the in-memory Tensor
                            node.attributes["value"] = ir.AttrTensor(
                                name="value",
                                value=ir.Tensor(loaded_array, name=external_tensor.name),
                            )
                            preloaded_count += 1

                        except Exception as e:
                            logger.warning(
                                f"Failed to preload external tensor from Constant node '{node.name}': {e}"
                            )
                            skipped_count += 1
                    else:
                        # Skip large tensors
                        skipped_count += 1

        if preloaded_count > 0:
            logger.debug(
                f"Preloaded {preloaded_count} small external tensors into memory "
                f"(skipped {skipped_count} large tensors)"
            )

        return preloaded_count


class ShapeInference(BasePass):
    """Pass to inference shape"""

    @dataclass
    class Config(PassConfig):
        """Configuration for ShapeInference pass"""

        """
        Size threshold in bytes for preloading external tensors into memory during constant folding

        Tensors smaller than this threshold will be preloaded into memory before
        shape inference to avoid file handler issues. Larger tensors will remain
        as external tensors to prevent memory issues

        Default: 1MB (1024 * 1024 bytes)
        """
        constant_folding_size_threshold: int = CONSTANT_FOLDING_SIZE_THRESHOLD

        """
        The shape of tensors maybe incorrect for some cases (for example graph is modified but shape is not updated)
        With default behavior (overwrite_shape=False), we will keep the incorrect values.
        But we can explicitly set overwrite_shape=True to update the correct values (note: this consumes much more computation resource)
        """
        overwrite_shape: bool = False

    def __init__(self, config: Config | None = None):
        """Initialize an instance of ShapeInference

        Args:
            config: Configuration for shape inference
        """
        super().__init__()
        if config is None:
            config = ShapeInference.Config()
        self.config: ShapeInference.Config = config

    def infer_by_onnx(self, ctx: GraphContext, overwrite_shape=False):  # pylint: disable=R0912,R0914
        """Shape inference by onnx.shape_inference.infer_shapes,
        constant will be firstly folded before the inference

        """
        origin_value_infos = {}
        inferred_value_infos = {}

        # convert model into proto, and fold constants to help onnx infer shape
        proto = serialize_model_with_folded_constant(ctx.model_ir, with_data=False)

        if overwrite_shape:
            # remove all exisiting shape
            while len(proto.graph.value_info) > 0:
                proto.graph.value_info.pop()

        origin_value_infos = {info.name: info for info in proto.graph.value_info}
        origin_value_infos.update({info.name: info for info in proto.graph.output})
        inferred_proto = onnx.shape_inference.infer_shapes(
            proto,
            check_type=True,
            strict_mode=False,
            data_prop=True,
        )
        inferred_value_infos = {info.name: info for info in inferred_proto.graph.value_info}
        inferred_value_infos.update({info.name: info for info in inferred_proto.graph.output})

        # check which tensors should be updated shape
        shape_to_update = {}
        type_to_update = {}
        for name, value_info in inferred_value_infos.items():
            new_shape = get_shape_from_value_info_proto(value_info, allow_symbols=False)
            if new_shape is None:
                continue

            if name not in origin_value_infos:
                shape_to_update[name] = new_shape
                type_to_update[name] = ir.serde.deserialize_type_proto_for_type(value_info.type)
            else:
                origin_value_info = origin_value_infos[name]
                origin_shape = get_shape_from_value_info_proto(origin_value_info, allow_symbols=False)

                if origin_shape is not None and new_shape is not None:
                    if origin_shape != new_shape:
                        raise ValueError(
                            f"mismatch shape at {name}, inferred {new_shape}, origin {origin_shape}"
                        )
                elif origin_shape is None and new_shape is not None:
                    shape_to_update[name] = new_shape

                value_type = (
                    ir.serde.deserialize_type_proto_for_type(value_info.type)
                    if value_info.type is not None
                    else None
                )
                origin_value_type = (
                    ir.serde.deserialize_type_proto_for_type(origin_value_info.type)
                    if value_info.type is not None
                    else None
                )
                if value_type is not None and origin_value_type is not None:
                    if value_type != origin_value_type:
                        raise ValueError(
                            f"mismatch type at {name}, inferred {value_type}, origin {origin_value_type}"
                        )
                elif value_type is not None:
                    type_to_update[name] = value_type

        # update the shapes
        updated_counts = 0
        for value in iter_all_values(ctx.graph_ir):
            updated = False
            if value.name in shape_to_update:
                value.shape = ir.Shape(shape_to_update[value.name])
                updated = True
            if value.name in type_to_update:
                value.type = type_to_update[value.name]
                updated = True
            if updated:
                updated_counts += 1
        return updated_counts

    def apply(self, ctx: GraphContext) -> int:
        """Inference the shape on the whole graph
        Iteratively do the constant propagation and shape inference
        until no extra information(shape/type/constant value) can be inferred.

        """
        # step0: Preload small external tensors into memory before shape inference
        # This avoids file handler issues and prevents loading large tensors
        PreloadSmallExternalTensors(self.config.constant_folding_size_threshold).apply(ctx)

        loop_counts_max = 100
        loop_i = 0
        all_updated_counts = 0
        while loop_i < loop_counts_max:
            # step1: propagate some constants, such as some nodes that construct a shape
            ConstantPropagation().apply(ctx)
            # step2: infer shape by onnx
            updated_counts = self.infer_by_onnx(
                ctx, overwrite_shape=self.config.overwrite_shape and loop_i == 0
            )
            # step3: infer shape on specific ops manually
            specific_op_infer = ShapeInferForSpecificOp()
            specific_op_infer.apply(ctx)
            updated_counts += specific_op_infer.updated_count

            if updated_counts == 0:
                break

            all_updated_counts += updated_counts
            loop_i += 1

        if loop_i == loop_counts_max:
            logger.warning("shape inference failed, loop counts max reached")

        if all_updated_counts > 0:
            logger.debug("Shape inference completed with updates")
            return 1
        return 0


class ShapeInferForSpecificOp(BaseTreeVisitor):
    """Pass to inference shape on specific Ops

    for example, currently onnx.shape_inference.infer_shapes doesn't handle QDQ nodes
    so we need to handle it manually.

    Also, the shape inference on self defined Op can be handled in this pass.
    """

    def __init__(self):
        """Initialize an instance of ShapeInferForSpecificOp"""
        super().__init__()
        self.updated_count = 0

    def visit_node_DequantizeLinear(self, graph: ir.Graph, node: ir.Node):  # pylint: disable=C0103
        """Inference the shape on the DequantizeLinear op"""
        updated = False
        if (
            node.inputs[0] is not None
            and node.outputs[0] is not None
            and node.outputs[0].shape is None
            and node.inputs[0].shape is not None
        ):
            node.outputs[0].shape = copy.deepcopy(node.inputs[0].shape)
            updated = True

        if node.outputs[0].dtype is None:
            output_dtype_attr = node.attributes.get("output_dtype", 0)
            if isinstance(output_dtype_attr, ir.Attr):
                output_dtype_value = convert_attr_to_py(output_dtype_attr, "as_int")
            else:
                output_dtype_value = output_dtype_attr
            output_dtype = ir.DataType(output_dtype_value)
            if output_dtype == ir.DataType(0):
                # If not supplied, the output data type is inferred from x_scale
                if node.inputs[1] is not None and node.inputs[1].dtype is not None:
                    output_dtype = node.inputs[1].dtype
            if output_dtype is not None:
                node.outputs[0].type = ir.TensorType(output_dtype)
                updated = True

        if updated:
            self.updated_count += 1

    def visit_node_QuantizeLinear(self, graph: ir.Graph, node: ir.Node):  # pylint: disable=C0103
        """Inference the shape on the QuantizeLinear op"""
        updated = False
        if (
            node.inputs[0] is not None
            and node.outputs[0] is not None
            and node.outputs[0].shape is None
            and node.inputs[0].shape is not None
        ):
            node.outputs[0].shape = copy.deepcopy(node.inputs[0].shape)
            updated = True

        if node.outputs[0].dtype is None:
            output_dtype_attr = node.attributes.get("output_dtype", 0)
            if isinstance(output_dtype_attr, ir.Attr):
                output_dtype_value = convert_attr_to_py(output_dtype_attr, "as_int")
            else:
                output_dtype_value = output_dtype_attr
            output_dtype = ir.DataType(output_dtype_value)
            if output_dtype == ir.DataType(0):
                # If not supplied, the output data type is inferred from y_zero_point
                if node.inputs[2] is not None and node.inputs[2].dtype is not None:
                    output_dtype = node.inputs[2].dtype
            if output_dtype is not None:
                node.outputs[0].type = ir.TensorType(output_dtype)
                updated = True

        if updated:
            self.updated_count += 1

    def visit_node_GroupSlice(self, graph: ir.Graph, node: ir.Node):  # pylint: disable=C0103
        """Inference the shape on the GroupSlice op"""
        updated = False
        assert node.inputs[0] is not None  # check for mypy
        for i, output in enumerate(node.outputs):
            if output.shape is not None and output.type is not None:
                return
            if output.shape is None and node.inputs[0].shape is not None:
                input_shape = get_value_numeric_shape(node.inputs[0])
                output_shape = list(input_shape)[:]
                axis = convert_attr_to_py(node.attributes["axis"], "as_int")
                start = convert_attr_to_py(node.attributes["starts"], "as_ints")[i]
                end = convert_attr_to_py(node.attributes["ends"], "as_ints")[i]
                output_shape[axis] = end - start
                output.shape = ir.Shape(output_shape)
                updated = True
            if output.type is None and node.inputs[0].dtype is not None:
                output.type = ir.TensorType(node.inputs[0].dtype)
                updated = True
        if updated:
            self.updated_count += 1

    def visit_node_FastHadamardTransform(self, graph: ir.Graph, node: ir.Node):  # pylint: disable=C0103
        """Inference the shape on the FastHadamardTransform op"""
        updated = False
        if node.inputs[0] is None:
            return

        if node.outputs[0].shape is None and node.inputs[0].shape is not None:
            node.outputs[0].shape = copy.deepcopy(node.inputs[0].shape)
            updated = True
        if node.outputs[0].type is None and node.inputs[0].type is not None:
            node.outputs[0].type = copy.deepcopy(node.inputs[0].type)
            updated = True
        if updated:
            self.updated_count += 1

    def visit_node_ElementWiseMux(self, graph: ir.Graph, node: ir.Node):
        """Inference the shape on the ElementWiseMux op"""

        if node.outputs[0].shape is not None and node.outputs[0].type is not None:
            return

        assert not any(x is None for x in node.inputs)
        input_shapes = [
            get_value_numeric_shape(x)
            for x in node.inputs[1:]
            if x is not None and has_static_shape_on_value(x)
        ]
        input_dtypes = [x.dtype for x in node.inputs[1:] if x is not None and x.type is not None]

        if len(input_shapes) == 0 and len(input_dtypes) == 0:
            return

        broadcast_shape = input_shapes[1]
        for shape in input_shapes[1:]:
            broadcast_shape = BroadcastHelper.get_broadcast_shape(broadcast_shape, shape)
            if broadcast_shape is None:
                assert False
        node.outputs[0].shape = ir.Shape(broadcast_shape)
        if input_dtypes[0] is not None:
            node.outputs[0].dtype = input_dtypes[0]
        self.updated_count += 1


def get_constant_inputs(graph: ir.Graph, node: ir.Node):
    """Get constant inputs to a node, if possible

    Args:
        graph: the graph containing the node
        node: the target node
    Returns:
        all_inputs_cst: whether all inputs are constant
        inputs_const_values: the constant values of the inputs (None for none constant input)

    Small external tensors should be preloaded by PreloadSmallExternalTensors pass.
    Large external tensors should remain external and will be treated as non-constant.
    """
    inputs_const_values = []
    all_inputs_cst = True
    for x in node.inputs:
        if x is not None:
            cst = x.meta["extra_info"].infered_constant_value
            # Check if this is an initializer with const_value (only if already loaded)
            if cst is None and x.name in graph.initializers:
                cst = get_constant_np(x, load_external_data=False)

            inputs_const_values.append(cst)
            if cst is None:
                all_inputs_cst = False
        else:
            inputs_const_values.append(None)

    return all_inputs_cst, inputs_const_values


def set_shape_dtype(value: ir.Value, shape: list[int], dtype: np.dtype | ir.DataType | None):
    """Set shape and dtype of a value

    Args:
        Value: the target value
        shape: the shape to set
        dtype: the datatype to set

    Returns: None
    """

    if isinstance(dtype, np.dtype):
        tensor_type = ir.TensorType(ir.DataType.from_numpy(dtype))
    elif isinstance(dtype, ir.DataType):
        tensor_type = ir.TensorType(dtype)
    elif isinstance(dtype, ir.TensorType):
        tensor_type = dtype
    else:
        raise ValueError(f"unhandled type {dtype}")

    origin_tensor_type = None
    if value.type is not None:
        origin_tensor_type = value.type
    if origin_tensor_type is not None and tensor_type is not None:
        if origin_tensor_type != tensor_type:
            raise ValueError(
                f"infered type {tensor_type} mismatch with original shape {origin_tensor_type}"
                f" at '{value.name}'"
            )
    elif origin_tensor_type is None:
        value.type = tensor_type

    if not has_static_shape_on_value(value):
        value.shape = ir.Shape(shape)
    else:
        origin_shape = get_value_numeric_shape(value)
        if tuple(origin_shape) != tuple(shape):
            raise ValueError(
                f"infered shape {tuple(shape)} mismatch with original shape {tuple(origin_shape)}"
                f" at '{value.name}'"
            )


def set_infered_cst_value(value: ir.Value, value_np: np.ndarray):
    """Set constant to a value

    Args:
        Value: the target value
        value_np: the numpy array to set

    Returns: None
    """
    value.meta["extra_info"].infered_constant_value = value_np
    set_shape_dtype(value, list(value_np.shape), value_np.dtype)


class ConstantPropagation(BaseTreeVisitor):
    """Pass to propagate constant"""

    def __init__(self):
        """Initialize an instance of ConstantPropagation"""
        super().__init__()

    def visit_general_node(self, graph: ir.Graph, node: ir.Node):
        """General method to propagate constant on the node
        if no specific method exists for the node, this method will be called

        Args:
            graph: ir.Graph instance
            node: ir node to be visited
        """
        constant_value = get_constant_np(node.outputs[0], load_external_data=False)
        if constant_value is not None:
            # constant node
            set_infered_cst_value(node.outputs[0], constant_value)
        else:
            # if all inputs have infered value
            # then using onnxscript's evaluator to infer the output value
            all_inputs_cst, inputs_const_values = get_constant_inputs(graph, node)
            if all_inputs_cst:
                has_schema = onnx.defs.has(node.op_type, graph.opset_imports[node.domain], domain=node.domain)
                if not has_schema:
                    return

                # Check if any input has a custom dtype that's not supported by onnxscript evaluator
                # Custom dtypes like int4, uint4, etc. cannot be converted to TensorProto element types
                has_unsupported_dtype = False
                for input_val in inputs_const_values:
                    if input_val is not None and isinstance(input_val, np.ndarray):
                        try:
                            # Try to convert the dtype to check if it's supported
                            onnx.helper.np_dtype_to_tensor_dtype(input_val.dtype)
                        # For unsupported dtypes:
                        # * onnx <= 1.16 throws KeyError
                        # * onnx >= 1.17 throws ValueError
                        except (KeyError, ValueError):
                            # This dtype is not supported by ONNX, skip evaluation
                            has_unsupported_dtype = True
                            break

                if has_unsupported_dtype:
                    # Skip constant propagation for nodes with unsupported dtypes
                    return

                op_schema = onnx.defs.get_schema(
                    node.op_type, graph.opset_imports[node.domain], domain=node.domain
                )
                attributes_py = convert_attrs_to_py(dict(node.attributes))
                evaluator = onnxscript.evaluator.default()
                use_legacy_eval = hasattr(evaluator, "eval")
                if use_legacy_eval:
                    # onnxscript < 0.6
                    outputs_cst = evaluator.eval(op_schema, inputs_const_values, attributes=attributes_py)
                elif hasattr(evaluator, "eval_op"):
                    # onnxscript >= 0.6 removed eval();
                    # We should use the new function eval_op instead now.
                    from onnxscript import values

                    opset_version = graph.opset_imports.get(node.domain, onnx.defs.onnx_opset_version())
                    opset = values.Opset(node.domain or None, opset_version)
                    op = values.Op(opset, node.op_type, op_schema)
                    outputs_cst = evaluator.eval_op(op, inputs_const_values, attributes_py)
                else:
                    logger.error(
                        "Onnx script evaluator do not has proper function. Please check your onnxscript and onnx-ir version."
                    )
                    raise NotImplementedError
                if len(node.outputs) == 1:
                    outputs_cst = [outputs_cst]
                assert len(outputs_cst) == len(node.outputs)
                for output_cst, output_v in zip(outputs_cst, node.outputs):
                    set_infered_cst_value(output_v, output_cst.value)

    def visit_node_Constant(self, graph: ir.Graph, node: ir.Node):
        """Constant propagation for Constant op

        Args:
            graph: ir.Graph instance
            node: ir node to be visited
        """
        # do not load constant when it is stored in the external data
        out_np = get_constant_np(node.outputs[0], load_external_data=False)
        if out_np is not None:
            set_infered_cst_value(node.outputs[0], out_np)

    def visit_node_Shape(self, graph: ir.Graph, node: ir.Node):  # pylint: disable=C0103
        """Constant propagation for Shape op

        Args:
            graph: ir.Graph instance
            node: ir node to be visited
        """
        input_v = node.inputs[0]
        assert input_v is not None  # check for mypy, definitely true
        if has_static_shape_on_value(input_v):
            shape = get_value_numeric_shape(input_v)
            end = None
            start = None
            if "end" in node.attributes:
                end = convert_attr_to_py(node.attributes["end"], "as_int")
            if "start" in node.attributes:
                start = convert_attr_to_py(node.attributes["start"], "as_int")
            sliced_shape = np.array(shape, dtype=np.int64)[slice(start, end)]
            set_infered_cst_value(node.outputs[0], sliced_shape)
        else:
            self.visit_general_node(graph, node)

    def visit_node_ElementWiseMux(self, graph: ir.Graph, node: ir.Node):
        """Constant propagation for ElementWiseMux op

        Args:
            graph: ir.Graph instance
            node: ir node to be visited
        """
        # we can't apply cosnt prop on ElementWiseMux
        return
