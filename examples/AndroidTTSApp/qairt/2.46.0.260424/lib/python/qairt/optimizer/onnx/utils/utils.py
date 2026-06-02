# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Helper functions
"""

import copy
import hashlib
import json
import os
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import networkx
import numpy as np
import onnx
import onnx_ir as ir
import safetensors.numpy

from qairt.optimizer.onnx.utils.ir_extra_info import VariableExtraInfo
from qairt.optimizer.utils.logger import logger


def load_safetensors(safetensor_path):
    """
    Load safetensor from the file
    """
    return safetensors.numpy.load_file(safetensor_path)


def save_safetensors(tensor_dict, safetensor_path):
    """
    Save safetensor to the file
    """
    return safetensors.numpy.save_file(tensor_dict, safetensor_path)


def save_named_safetensors(named_safetensors: dict[str, dict], dir_path: str, enable_log=True):
    """
    Save named safetensors to the directory
    Args:
        named_safetensors: a dict of {encset_name: safetensors}
        dir_path: the directory path
        enable_log: wheter to print log infomation
    """
    saved_paths = {}
    # extract safetensors from extra_info
    for encset_name, graph_safetensors in named_safetensors.items():
        safetensor_path = os.path.join(dir_path, encset_name + ".safetensors")
        save_safetensors(graph_safetensors, safetensor_path)
        saved_paths[encset_name] = safetensor_path

        if enable_log:
            logger.info("saved '%s' safetensors to %s", encset_name, safetensor_path)
    return saved_paths


def save_named_encodings(named_graph_encodings: dict[str, dict], dir_path: str, enable_log=True):
    """
    Save named encodings to the directory
    Args:
        named_graph_encodings: a dict of {encset_name: encodings}
        dir_path: the directory path
        enable_log: wheter to print log infomation
    """
    # extract encodings from extra_info
    saved_encodings_paths = {}

    for encset_name, graph_encodings in named_graph_encodings.items():
        enc_path = os.path.join(dir_path, encset_name + ".json")
        with open(enc_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(graph_encodings, indent=4))
        saved_encodings_paths[encset_name] = enc_path

        if enable_log:
            logger.info("saved '%s' encodings to %s", encset_name, enc_path)
    return saved_encodings_paths


def save_updatable_tensor_names(updatable_tensor_names: list[str], path: str):
    """
    Save updatable_tensor_names to the file
    Args:
        updatable_tensor_names: a list of updatable tensor names
        path: the file path
    """
    with open(path, "w", encoding="utf-8") as f:
        for n in updatable_tensor_names:
            f.write(n + "\n")


def safe_replace_all_uses_with(
    graph: ir.Graph, old_value: ir.Value, new_value: ir.Value | None, except_users=None
):
    """
    Replace all the uses of old_value by new_value
    automatically handle the graph outputs if old_value is one of them
    In some case, we don't want to replace some users, we can add them in excpet_users

    Args:
        graph: ir graph
        old_value: the old value
        new_value: the new value to replace
        except_users: the users to be excepted
    """
    assert new_value is not None  # easier to call this function with mypy check
    if old_value is new_value:
        return
    is_graph_output = False
    if old_value.is_graph_output():
        is_graph_output = True

    for user_node, index in tuple(old_value.uses()):
        if except_users and user_node in except_users:
            continue
        user_node.replace_input_with(index, new_value)

    if is_graph_output:
        graph_outputs = graph.outputs
        graph_outputs[graph_outputs.index(old_value)] = new_value


def safe_insert_node_after(
    graph: ir.Graph, node: ir.Node | None, new_nodes: Iterable[ir.Node] | ir.Node, /
) -> None:
    """
    Insert new_nodes after node in the graph
    if node is None, then assert new_nodes in the beginning

    Args:
        graph: ir graph
        node: the position to insert after
        new_nodes: the new node to insert
    """
    if node is not None:
        graph.insert_after(node, new_nodes)
    else:
        # node is None (for example node=v.producer(), where v is an initializer),
        # we add new_nodes as the first nodes
        if len(graph) == 0:
            if isinstance(new_nodes, ir.Node):
                graph.append(new_nodes)
            else:
                graph.extend(new_nodes)
        else:
            graph.insert_before(graph[0], new_nodes)


def get_shape_of_slice(src_shape, axes, starts, ends, steps=None):
    """
    Calculate the shape of a slice given the source shape, axes, starts, ends, and steps.

    Args:
        src_shape (list): The shape of the source array.
        axes (list): The axes to slice.
        starts (list): The start indices for each axis.
        ends (list): The end indices for each axis.
        steps (list, optional): The step sizes for each axis. Defaults to None.

    Returns:
        list: The shape of the slice.
    """
    if steps is None:
        steps = [1] * len(axes)

    assert len(axes) == len(starts) == len(ends) == len(steps), (
        "Axes, starts, ends, and steps must have the same length"
    )

    out_shape = list(src_shape)[:]

    for axis, start, end, step in zip(axes, starts, ends, steps):
        assert axis >= 0, "Axis must be non-negative"
        assert start >= 0, "Start index must be non-negative"
        assert end >= 0, "End index must be non-negative"
        assert step > 0, "Step size must be positive"

        out_shape[axis] = (end - start) // step

    return out_shape


def check_static_shape(value: ir.Value | None):
    """
    Check whether a value has a static shape
    a ValueError will be raised if the value has not static shape

    Args:
        value: the value to check
    """
    if value is None:
        return
    if not has_static_shape_on_value(value):
        raise ValueError(f"Tensor {value.name} has no static shape. Please run ONNX shape inference first.")


def check_static_shape_of_node_io(node: ir.Node):
    """
    Check whether all inputs/outputs of the specified node have static shape
    a ValueError will be raised if not

    Args:
        node: the node to check
    """
    for v in node.inputs:
        check_static_shape(v)
    for v in node.outputs:
        check_static_shape(v)


def have_static_shape_on_node_io(node: ir.Node) -> bool:
    """
    Check whether all inputs/outputs of the specified node have static shape

    Args:
        node: the node to check
    """
    for v in node.inputs:
        if v is None:
            continue
        if v.shape is None:
            return False
        if not v.shape.is_static():
            return False
    for v in node.outputs:
        if v is None:
            continue
        if v.shape is None:
            return False
        if not v.shape.is_static():
            return False
    return True


def has_static_shape_on_value(value: ir.Value) -> bool:
    """
    Check whether the specified value has static shape

    Args:
        value: the value to check
    """
    if value.shape is None:
        return False
    if not value.shape.is_static():
        return False
    return True


def get_value_numeric_shape(value: ir.Value | None):
    """
    Get numeric shape of the value
    """
    assert value is not None
    shape = value.shape
    assert shape is not None
    return shape.numpy()


def get_constant_np(value: ir.Value | None, load_external_data: bool = True, use_infered_value: bool = False):
    """
    Get value's constant numpy value
    a ValueError will be raised if the value is not a constant

    Args:
        value: the target value
        load_external_data: if False, will not load ExternalTensor instances
                          (returns None for external tensors instead)
        use_infered_value: if True, will return the value infered by ShapeInference pass
                          (only small tensors are infered)
    Returns:
        numpy array if the value is a constant, None otherwise
    """
    if value is None:
        return None

    # Check const_value but optionally skip ExternalTensor
    if value.const_value is not None:
        if not load_external_data and isinstance(value.const_value, ir.ExternalTensor):
            # Don't load external tensors when load_external_data=False
            return None
        return value.const_value.numpy()

    producer = value.producer()
    if producer is not None:
        if producer.op_type == "Constant":
            if not load_external_data and isinstance(producer.attributes["value"].value, ir.ExternalTensor):
                # Don't load external tensors when load_external_data=False
                return None
            return convert_attr_to_py(producer.attributes["value"])
        if producer.op_type == "Identity":
            # check for mypy, definitely true
            assert producer.inputs[0] is not None
            return get_constant_np(
                producer.inputs[0], load_external_data=load_external_data, use_infered_value=use_infered_value
            )

    if use_infered_value and "extra_info" in value.meta:
        v_extra_info: VariableExtraInfo = value.meta["extra_info"]
        return v_extra_info.infered_constant_value
    return None


def is_constant(value: ir.Value | None, use_infered_value: bool = False) -> bool:
    """
    Check if the given value is constant
    """
    if value is None:
        return False
    if value.const_value is not None:
        return True
    producer = value.producer()
    if producer is not None:
        if producer.op_type == "Constant":
            return True
        if producer.op_type == "Identity":
            # check for mypy, definitely true
            assert producer.inputs[0] is not None
            return is_constant(producer.inputs[0])

    if use_infered_value and "extra_info" in value.meta:
        v_extra_info: VariableExtraInfo = value.meta["extra_info"]
        return v_extra_info.infered_constant_value is not None
    return False


def make_constant_node(
    graph: ir.Graph, namehint: str | None, data: ir.TensorProtocol | np.ndarray | list | tuple
):
    if namehint is None:
        namehint = ""
    name = get_unique_name(graph, namehint)
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    if isinstance(data, np.ndarray):
        tensor_proto = onnx.numpy_helper.from_array(data, name=name)
        tensor = ir.TensorProtoTensor(tensor_proto)
        v = ir.Value(name=name, type=ir.TensorType(tensor.dtype), shape=tensor.shape, const_value=tensor)
    elif isinstance(data, ir.TensorProtocol):
        ir_shape = ir.Shape(data.shape.dims)  # type: ignore
        v = ir.Value(name=name, type=ir.TensorType(data.dtype), shape=ir_shape, const_value=data)
    else:
        assert False

    assert v.const_value is not None  # check for mypy
    v.meta["extra_info"] = VariableExtraInfo()

    n = ir.Node(
        "",
        "Constant",
        [],
        attributes=[ir.AttrTensor("value", v.const_value)],
        name=get_unique_name(graph, namehint + "/node"),
        outputs=[v],
    )
    return n


def make_initializer(graph: ir.Graph, name: str, np_array: np.ndarray | list | tuple):
    """
    Create initializer to the graph
    name should be unique
    """

    if isinstance(np_array, (list, tuple)):
        if all(isinstance(x, int) for x in np_array):
            # on windows, array of int will be converted to dtype=np.int32 by default
            # on linux, array of int will be converted to dtype=np.int64 by default
            # to align the behavior, here we set dtype=np.int64 explicitly
            np_array = np.array(np_array, dtype=np.int64)
        else:
            np_array = np.array(np_array)
    tensor = ir.TensorProtoTensor(onnx.numpy_helper.from_array(np_array, name=name))
    v = ir.Value(name=name, type=ir.TensorType(tensor.dtype), shape=tensor.shape, const_value=tensor)
    v.shape = ir.Shape(tuple(np_array.shape))
    v.type = ir.TensorType(tensor.dtype)
    v.meta["extra_info"] = VariableExtraInfo()
    graph.initializers[name] = v
    return v


def join_name(*args):
    out = []
    for x in args:
        if x is None:
            continue
        elif isinstance(x, str):
            out.append(x)
        else:
            raise ValueError()
    return "".join(out)


def make_initializer_with_namehint(graph: ir.Graph, namehint: str, np_array: np.ndarray | list | tuple):
    name = get_unique_name(graph, namehint)
    return make_initializer(graph, name, np_array)


def get_unique_name(graph: ir.Graph, namehint: str | None):
    if namehint is None:
        # v.name may be None, so to facilitate the usage, we support namehint = None
        namehint = "tmp"
    return graph.meta["extra_info"].get_unique_name(namehint)


def copy_value(graph: ir.Graph, copy_from: ir.Value, namehint: str | None):
    assert isinstance(copy_from.meta["extra_info"], VariableExtraInfo)
    if namehint is None:
        namehint = "tmp"
    value = ir.Value(producer=None, name=get_unique_name(graph, namehint))
    value.meta["extra_info"] = copy_from.meta["extra_info"].copy()
    value.shape = copy_from.shape
    if copy_from.dtype is not None:
        value.dtype = copy_from.dtype
    return value


def iter_all_values(graph_ir: ir.Graph):
    """
    Iterate all values in the graph
    """
    # iterate inputs
    yield from graph_ir.inputs

    # iterate for initializers
    yield from graph_ir.initializers.values()

    # iterate for activations
    for node in list(graph_ir):
        yield from node.outputs


def convert_attr_to_py(value, schema: str | None = None):
    # pylint: disable=[too-many-return-statements, too-many-branches]
    """
    Convert attribute value to python object automatically
    """
    if schema is not None:
        # schema can be only applied with ir.Attr
        assert isinstance(value, ir.Attr)
        if schema == "as_int":
            return value.as_int()
        if schema == "as_ints":
            return value.as_ints()
        if schema == "as_float":
            return value.as_float()
        if schema == "as_floats":
            return value.as_floats()
        if schema == "as_string":
            return value.as_string()
        if schema == "as_strings":
            return value.as_strings()
        if schema == "as_tensor":
            return value.as_tensor().numpy()
        if schema == "as_tensors":
            return [x.numpy() for x in value.as_tensors()]

    if isinstance(value, ir.TensorProtocol):
        return value.numpy()
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, Sequence):
        # copy to prevent mutation
        return [convert_attr_to_py(x) for x in value]
    if isinstance(value, ir.Attr):
        return convert_attr_to_py(value.value)

    raise NotImplementedError


def convert_attrs_to_py(attrs: dict[str, ir.Attr]):
    """
    Convert attributes to dictionary with python objects automatically
    """
    py_attrs = {}
    for k, v in attrs.items():
        py_attrs[k] = convert_attr_to_py(v.value)
    return py_attrs


def clone_node_attribute(attr: ir.Attr, is_deep_copy_graph: bool = False) -> ir.Attr:
    """
    Clone node attributes safely. The new version of onnxscript >= 0.6,
    the attr contains node, but the node contains the attr again.

    is_deep_copy_graph:
        If False (default), Graph/Graphs attributes are shallow-copied to avoid
        deep-copying the graph structure. Other attributes are deep-copied.
    """

    is_graph_attr = attr.type in (ir.AttributeType.GRAPH, ir.AttributeType.GRAPHS)
    if is_graph_attr and is_deep_copy_graph:
        raise ValueError("Do not support. It is a complicated case, maybe support it in the future.")

    if is_graph_attr:
        # Except for `value`, which may be a graph or a sequence of graphs, all other fields are immutable,
        # e.g., `attr.name` is a string.
        return copy.copy(attr)
    return copy.deepcopy(attr)


def clean_model_proto(proto: onnx.ModelProto):  # pylint: disable=[too-many-branches]
    """
    Remove unused node from the given proto
    """
    # remove unused node
    act_use_counts = {}
    for n in proto.graph.node:
        for v in n.input:
            if v not in act_use_counts:
                act_use_counts[v] = 0
            act_use_counts[v] += 1
    for v in proto.graph.output:
        if v.name not in act_use_counts:
            act_use_counts[v.name] = 0
        act_use_counts[v.name] += 1

    clean_node_lists = []
    for n in proto.graph.node[::-1]:
        all_unused = True
        for v in n.output:
            if act_use_counts.get(v, 0) > 0:
                all_unused = False
                break
        if all_unused:
            # all outputs are unused
            # so remove this node
            for v in n.input:
                act_use_counts[v] -= 1
        else:
            clean_node_lists.append(n)

    clean_node_lists.reverse()

    # remove unused initializers
    clean_initializer = []
    for v in proto.graph.initializer:
        if act_use_counts.get(v.name, 0) <= 0:
            continue
        clean_initializer.append(v)

    while len(proto.graph.initializer) > 0:
        proto.graph.initializer.pop()
    proto.graph.initializer.extend(clean_initializer)

    while len(proto.graph.node) > 0:
        proto.graph.node.pop()
    proto.graph.node.extend(clean_node_lists)
    return proto


def get_shape_from_value_info_proto(
    val_info: onnx.ValueInfoProto,
    allow_symbols: bool = False,
) -> list[str | int] | None:
    """
    Copied from MHA2SHA repo
    Function to get the shape from value info proto.
    """
    tensor_shape = []
    tensor_type = val_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        return None

    # iterate through dimensions of the shape:
    for d in tensor_type.shape.dim:
        # the dimension may have a definite (integer) value or a symbolic identifier or neither:
        if d.HasField("dim_value"):
            tensor_shape.append(d.dim_value)
        elif d.HasField("dim_param"):
            # unknown dimension with symbolic name
            if allow_symbols:
                tensor_shape.append(d.dim_param)
            else:
                return None
        else:
            return None
    return tensor_shape


def load_updatable_tensor_list(updatable_tensors_path: str) -> list[str]:
    """
    Load updatable tensor list from the given file path
    """
    updatable_tensors = []
    if updatable_tensors_path is not None:
        with open(updatable_tensors_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    updatable_tensors.append(line)
    return updatable_tensors


def get_base_enc_name(named_encodings_paths, named_safetensors_paths):
    """
    Infer the base encodings encset_name
    """
    if len(named_encodings_paths):
        return None

    base_enc_name_candidates = set(named_encodings_paths.keys()) - set(named_safetensors_paths.keys())
    if len(base_enc_name_candidates) == 1:
        return base_enc_name_candidates.pop()
    if len(base_enc_name_candidates) == 0:
        logger.warning(
            "Cannot find base enc name, please ensure encodings path for base is correctly set"
            "Note: base graph has no safetensors"
        )
        return None
    if len(base_enc_name_candidates) > 1:
        base_enc_name = base_enc_name_candidates.pop()
        logger.warning(
            "Found multiple base enc candiates %s, using %s" + "Note: base graph has no safetensors",
            base_enc_name_candidates,
            base_enc_name,
        )
        return base_enc_name
    raise ValueError("cannot infer the base encoding name")


def get_attribute_with_default(node: ir.Node, attr_name: str, default_value):
    """
    Get node attribute, if it has not the specified attribute, then return default_value

    Args:
        node:
        attr_name: the attribute name
        default_value: the default attribute value
    Return:
        attribute with python object
    """
    if attr_name in node.attributes:
        return convert_attr_to_py(node.attributes[attr_name])

    return default_value


def scan_previous_nearest_candidate(
    start_values: list[ir.Value | None],
    check_fn: Callable[[ir.Value], bool],
    ignore_fn: Callable[[ir.Value], bool],
):
    """
    Scan the nearest acceptable nodes bottom-up from the start values

    Args:
        start_values: searching start nodes
        check_fn: whether the node is acceptable
        ignore_fn: whether the node should be ignored, and continue to scan its input
    """
    check_values: list[ir.Value | None] = []
    check_values += start_values
    while len(check_values) > 0:
        candidate_v = check_values.pop(0)
        if candidate_v is None:
            continue
        if ignore_fn is not None and ignore_fn(candidate_v):
            candidate_v_producer = candidate_v.producer()
            if candidate_v_producer is not None:
                check_values += list(candidate_v_producer.inputs)
        elif check_fn(candidate_v):
            yield candidate_v


class ValuePredicate:
    """Base class for composable predicates on ir.Value used in scan_previous_nearest_candidate"""

    def __call__(self, v: ir.Value) -> bool:
        raise NotImplementedError

    def __or__(self, other: "ValuePredicate") -> "ValuePredicate":
        return _OrPredicate(self, other)

    def __and__(self, other: "ValuePredicate") -> "ValuePredicate":
        return _AndPredicate(self, other)

    def __invert__(self) -> "ValuePredicate":
        return _NotPredicate(self)


class _NotPredicate(ValuePredicate):
    def __init__(self, inner: ValuePredicate):
        self._inner = inner

    def __call__(self, v: ir.Value) -> bool:
        return not self._inner(v)


class _OrPredicate(ValuePredicate):
    def __init__(self, left: ValuePredicate, right: ValuePredicate):
        self._left = left
        self._right = right

    def __call__(self, v: ir.Value) -> bool:
        return self._left(v) or self._right(v)


class _AndPredicate(ValuePredicate):
    def __init__(self, left: ValuePredicate, right: ValuePredicate):
        self._left = left
        self._right = right

    def __call__(self, v: ir.Value) -> bool:
        return self._left(v) and self._right(v)


class OpTypePredicate(ValuePredicate):
    """
    Checks if the value's producer op_type is in a given list
    NOTE: This class was previously named ConditionOnValueProducer
    """

    def __init__(self, valid_producer_types: list[str]):
        self.valid_producer_types = valid_producer_types

    def __call__(self, v: ir.Value) -> bool:
        producer = v.producer()
        if producer is None:
            return False
        return producer.op_type in self.valid_producer_types


def load_json(json_path: str):
    """
    load json file
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def autocomplete_opset(model: ir.Model, default_possible_opsets: dict[str, int]):
    """
    Opset for custom domain (for example qti.aisw) may not be well defined in the model.
    In this case onnx.shape_inference will fail, so we need to autocomplete the opset for it

    Args:
        model: The ONNX IR model
        default_possible_opsets: Dictionary mapping domain names to default opset versions
    """
    # model.opset_imports is a dict mapping domain (str) to version (int)
    opset_imports = model.opset_imports

    opsets_to_add = {}
    for node in model.graph:
        domain = node.domain if node.domain else ""
        if domain not in opset_imports and domain in default_possible_opsets and domain not in opsets_to_add:
            logger.warning(
                'opset version for domain "%s" is not set, autocomplete it to version "%d"',
                domain,
                default_possible_opsets[domain],
            )
            opsets_to_add[domain] = default_possible_opsets[domain]

    # Add missing opsets to the model (just set the version integer)
    for domain, version in opsets_to_add.items():
        model.opset_imports[domain] = version


def get_lora_weight_shapes(src_named_safetensors: dict[str, dict]) -> dict[str, list[int]]:
    """Helper to extract original LoRA weight shapes from safetensors data."""
    original_lora_weight_shapes: dict[str, list[int]] = {}
    if not src_named_safetensors:
        return original_lora_weight_shapes

    for set_name, safetensors_in_set in src_named_safetensors.items():
        if not isinstance(safetensors_in_set, dict):
            logger.warning(f"Safetensors for set '{set_name}' is not a dictionary. Skipping.")
            continue
        for lora_weight_name, lora_weight_data in safetensors_in_set.items():
            try:
                if isinstance(lora_weight_data, np.ndarray):
                    original_lora_weight_shapes[lora_weight_name] = list(lora_weight_data.shape)
                else:
                    logger.warning(
                        f"LoRA weight data for '{lora_weight_name}' in set '{set_name}' "
                        f"is not a NumPy array. Cannot extract shape. Data type: {type(lora_weight_data)}."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to extract shape for LoRA weight '{lora_weight_name}' in set '{set_name}': {e}"
                )
    return original_lora_weight_shapes


def infer_slice_output_shape(
    src_shape: list[int], axes: list[int], starts: list[int], ends: list[int], src_name: str, entry_idx: int
) -> list[int] | None:
    """Helper to infer the shape of a sliced tensor."""
    dest_shape: list[int] = list(src_shape)

    if not (len(axes) == len(starts) == len(ends)):
        logger.warning(
            f"Mismatched lengths for axes, starts, and ends in trace entry {entry_idx} "
            f"(src: '{src_name}'). Dest shape set to None."
        )
        return None

    for i in range(len(axes)):
        axis = axes[i]
        start = starts[i]
        end = ends[i]

        if not (0 <= axis < len(src_shape)):
            logger.warning(
                f"Slice axis {axis} is out of bounds for source shape {src_shape} "
                f"of '{src_name}' in trace entry {entry_idx}. Dest shape set to None."
            )
            return None

        slice_len = end - start
        if slice_len >= 0:
            dest_shape[axis] = slice_len
        else:
            logger.warning(
                f"Invalid slice length ({slice_len}) for axis {axis} in trace entry {entry_idx} "
                f"(src: '{src_name}'). Dest shape set to None."
            )
            return None
    return dest_shape


def move_external_constant_to_initializer(model: ir.Model):
    """
    Move constant with external data to initializer.
    currently, onnx_ir will not handle external data for constant node in the serialization,
    this is a workaround to fix this
    """
    nodes_to_remove = []
    for n in list(model.graph):
        if n.op_type == "Constant":
            if "value" not in n.attributes:
                continue
            value = n.attributes["value"]
            if value is None:
                continue
            if value.type != ir.AttributeType.TENSOR:
                continue
            external_tensor = value.value
            cst_v = n.outputs[0]
            if isinstance(external_tensor, ir.ExternalTensor):
                # load data from file
                # note: we need to keep name, since this function is called after encodings having been fixed
                init_v = ir.Value(None, name=cst_v.name)
                init_v.const_value = external_tensor
                init_v.shape = cst_v.shape
                if cst_v.dtype is not None:
                    init_v.dtype = cst_v.dtype
                init_v.doc_string = cst_v.doc_string
                init_v.meta.update(cst_v.meta)
                model.graph.initializers.add(init_v)

                safe_replace_all_uses_with(model.graph, cst_v, init_v)
                nodes_to_remove.append(n)
    model.graph.remove(nodes_to_remove)


def is_used(v: ir.Value):
    """
    Check if the indicated value is used or not
    """
    return len(v.uses()) > 0 or v.is_graph_output()


def iter_ancestors_with_budget(start_v: ir.Value | None, max_layers_to_traverse: int | None):
    if start_v is None:
        return

    visited: set[int] = set()
    queue: deque[tuple[ir.Value | None, int]] = deque([(start_v, 0)])  # (node, current_layer_id)

    while queue:
        current_v, current_layer_id = queue.popleft()

        if current_v is None:
            continue
        if id(current_v) in visited:
            continue

        visited.add(id(current_v))
        yield current_v

        if max_layers_to_traverse and current_layer_id >= max_layers_to_traverse:
            continue
        producer = current_v.producer()
        if producer is None:
            continue
        for input_value in producer.inputs:
            if id(input_value) not in visited:
                queue.append((input_value, current_layer_id + 1))


def scan_ancestors_with_budget(
    start_v: ir.Value | None, max_layers_to_traverse: int | None
) -> list[ir.Value]:
    if start_v is None:
        return []

    ancestors: list[ir.Value] = list(iter_ancestors_with_budget(start_v, max_layers_to_traverse))
    return ancestors


def scan_least_common_ancestor(
    start_v1: ir.Value | None,
    start_v2: ir.Value | None,
    max_layers_to_traverse: int | None,
    lca_checker: Callable[[ir.Value], bool] | None = None,
):
    """
    Scan ancestors of a value, return the common least ancestors
    """

    # Collect all ancestors of start_v1 and start_v2
    ancestors_of_v1 = scan_ancestors_with_budget(start_v1, max_layers_to_traverse)
    ancestors_of_v2 = scan_ancestors_with_budget(start_v2, max_layers_to_traverse)

    # common nodes
    common_ancestors_set = set(id(x) for x in ancestors_of_v1).intersection(
        set(id(x) for x in ancestors_of_v2)
    )

    # find lca
    lca = None
    for ancestor in ancestors_of_v1:
        if id(ancestor) in common_ancestors_set:
            if lca_checker is not None and not lca_checker(ancestor):
                continue
            lca = ancestor
            break
    return lca


def scan_lca_interleaved(
    start_v_list: list[ir.Value],
    max_layers_to_traverse: int | None,
    lca_checker: Callable[[ir.Value], bool] | None = None,
) -> ir.Value | None:
    """
    Find the Lowest Common Ancestor (LCA) of multiple values using an interleaved
    (lock-step) traversal strategy.

    All per-value ancestor iterators are advanced one BFS layer at a time in parallel.
    This means a value that is at depth D from every start value is discovered after
    exactly D rounds, regardless of how many start values there are.  As a result, this
    function is significantly faster than calling :func:`scan_least_common_ancestor`
    repeatedly when the true LCA lies at a **similar depth** from all start values
    (i.e. the "balanced" case).  In the worst case (highly unbalanced depths) it
    degrades gracefully to a full ancestor enumeration.

    Args:
        start_v_list: The values whose LCA is to be found.
        max_layers_to_traverse: Maximum number of ancestor layers to explore from each
            start value.  Pass ``None`` to search without a depth limit.
        lca_checker: Optional predicate applied to each LCA candidate.  When provided,
            a candidate is accepted only if ``lca_checker(candidate)`` returns ``True``;
            otherwise the search continues to the next common ancestor.

    Returns:
        The nearest (deepest) common ancestor that satisfies ``lca_checker`` (if given),
        or ``None`` if no qualifying ancestor is found within the traversal budget.
    """
    # Advance all per-value ancestor iterators simultaneously so that values closer to
    # the start nodes are examined before deeper ancestors.  This guarantees we return
    # the *lowest* (i.e. closest to the leaves) common ancestor first.
    ancestor_iterators = {
        i: iter(iter_ancestors_with_budget(v, max_layers_to_traverse)) for i, v in enumerate(start_v_list)
    }
    ancestor_set_list: list[set[int]] = [set() for _ in start_v_list]
    ancestor_set_union: set[int] = set()

    while ancestor_iterators:
        empty_iterators_indices = []
        for idx, it in ancestor_iterators.items():
            curr_v = next(it, None)
            if curr_v is None:
                empty_iterators_indices.append(idx)
                continue

            curr_v_id = id(curr_v)
            ancestor_set_list[idx].add(curr_v_id)
            if curr_v_id in ancestor_set_union:
                # curr_v has already been seen for at least one other start value;
                # check whether it is now reachable from all start values.
                if all(curr_v_id in ancestor_set for ancestor_set in ancestor_set_list):
                    if lca_checker is None or lca_checker(curr_v):
                        return curr_v

            ancestor_set_union.add(curr_v_id)

        for idx in empty_iterators_indices:
            del ancestor_iterators[idx]

    return None


def find_smallest_element_larger_than(ascending_list, value):
    """
    Helper function
    find smalles element in the ascending_list that larger than value
    """
    for i, x in enumerate(ascending_list):
        if x >= value:
            return i
    return len(ascending_list)


def find_largest_element_smaller_than(ascending_list, value):
    """
    Helper function
    find largest element in the ascending_list that smaller than value
    """
    for i in range(len(ascending_list) - 1, -1, -1):
        x = ascending_list[i]
        if x <= value:
            return i
    return -1


def get_slice_static_params(op_node: ir.Node) -> dict | None:
    assert op_node.op_type == "Slice"

    if not have_static_shape_on_node_io(op_node):
        return None

    assert op_node.inputs[0] is not None  # check for mypy, definitely true
    # check for mypy, definitely true
    assert op_node.inputs[0].shape is not None
    input_shape = get_value_numeric_shape(op_node.inputs[0])

    starts = get_constant_np(op_node.inputs[1])
    ends = get_constant_np(op_node.inputs[2])

    if starts is None or ends is None:
        return None

    starts = starts.tolist()
    ends = ends.tolist()
    axis_num = len(starts)

    if len(op_node.inputs) >= 4:
        slice_axes_v = op_node.inputs[3]
        slice_axes = get_constant_np(slice_axes_v)

        if slice_axes is None:
            return None
        slice_axes = slice_axes.tolist()
    else:
        slice_axes = [0] * axis_num

    # normalize slice_axes
    for i, _ in enumerate(slice_axes):
        if slice_axes[i] < 0:
            slice_axes[i] += len(input_shape)

    # normalize slice end
    for i, _ in enumerate(ends):
        # some onnx model exported by torch will have end=9223372036854775807 (which is 2**63-1)
        axis = slice_axes[i]
        ends[i] = min(ends[i], input_shape[axis])

    if len(op_node.inputs) >= 5:
        steps = get_constant_np(op_node.inputs[4])
        if steps is None:
            return None
        steps = steps.tolist()
    else:
        steps = [1] * axis_num

    return {"starts": starts, "ends": ends, "axes": slice_axes, "steps": steps}


def get_input_ids(model: ir.Model) -> ir.Value:
    """
    Attempts to locate the 'input_ids' input value from the model's graph.

    Args:
        model (ir.Model): The intermediate representation of the model.

    Returns:
        ir.Value: The resolved input value corresponding to 'input_ids' or a fallback input.
    """
    input_ids = None
    try:
        input_ids = [inp for inp in model.graph.inputs if inp.name == "input_ids"][0]
    except IndexError:
        for node in model.graph:
            if (
                node.op_type == "Gather"
                and not any(inp and inp.name and "lora" in inp.name for inp in node.inputs)
                and any(inp in model.graph.inputs for inp in node.inputs)
            ):
                input_ids = node.inputs[0]

    if input_ids is None:
        first_input = model.graph.inputs[0]
        logger.warning(
            f"Unable to find 'input_ids' graph input. Using the first graph input: {first_input.name}"
        )
        input_ids = first_input
    return input_ids


def get_embedding_node(input_ids: ir.Value) -> ir.Node:
    """
    Traverses the graph to locate the embedding node associated with the given input value.

    Args:
        input_ids (ir.Value): The input value representing token IDs in the model graph.

    Returns:
        ir.Node: The node performing the 'Gather' operation, typically used for embedding lookup.
    """
    input_ids_consumer = input_ids.consumers()[0]
    while input_ids_consumer.op_type != "Gather":
        input_ids_consumer = input_ids_consumer.outputs[0].consumers()[0]
    return input_ids_consumer


def get_immediate_dominator_of_value(
    values: list[ir.Value] | ir.Value, max_layer_to_traverse: int = 50
) -> ir.Value | None:
    """
    Search the immediate dominator of the given values
    Immediate dominator is the closest node that appears on every path from graph inputs to these node.

    if len(values) > 0, a virtual consumer will be created for all values,
    and the immediate dominator of this virtual consumer will be return

    Args:
        value: The value to find the immediate dominator for
        max_layer_to_traverse: Maximum number of layers to traverse backwards

    Returns:
        The immediate dominator value, or None if not found
    """

    if isinstance(values, ir.Value):
        values = [values]

    # Build networkx directed graph for subgraph with max layers to traverse
    name2v_map: dict[str | None, ir.Value] = {x.name: x for x in values}

    for v in values:
        curr_ancestors = scan_ancestors_with_budget(v, max_layer_to_traverse)

        # ignore all constant ancestors
        curr_ancestors = [x for x in curr_ancestors if x.name not in name2v_map and not is_constant(x)]

        for a in curr_ancestors:
            name2v_map[a.name] = a

    graph = networkx.DiGraph()

    # in networkx's graph, node is ir.Value, edge is ir.Node
    for name in name2v_map.keys():
        graph.add_node(name)

    # add edges (from producer to consumer)
    for v in name2v_map.values():
        producer = v.producer()
        if producer is not None:
            for input_value in producer.inputs:
                if input_value is not None and input_value.name in graph:
                    # edge from producer to consumer
                    graph.add_edge(input_value.name, v.name)

    # networkx.immediate_dominators needs a start point
    # so we need to create a virtual root
    entries = [node for node in graph.nodes() if graph.in_degree(node) == 0]

    if not entries:
        return None

    # Create a virtual root node that connects to all entry nodes
    virtual_root = -1
    graph.add_node(virtual_root)
    for entry in entries:
        graph.add_edge(virtual_root, entry)

    # for multi values, a virtual consumer will be added
    if len(values) > 1:
        virtual_consumer = -2
        graph.add_node(virtual_consumer)
        for v in values:
            graph.add_edge(v.name, virtual_consumer)
        end_consumer: int | str = virtual_consumer
    else:
        assert values[0].name is not None  # check for mypy
        end_consumer = values[0].name

    dominators = networkx.immediate_dominators(graph, virtual_root)

    if end_consumer not in dominators:
        return None

    idom_name = dominators[end_consumer]
    if idom_name == virtual_root:
        return None
    return name2v_map[idom_name]


def get_constant_tensor_proto(value: ir.Value) -> ir.TensorProtocol | None:
    # unlike get_constant_np
    # this function will not parse/load explicitly the constants into memory
    # making it more efficient to copy directly

    if value is None:
        return None

    # Check const_value but optionally skip ExternalTensor
    if value.const_value is not None:
        return value.const_value

    producer = value.producer()
    if producer is not None:
        if producer.op_type == "Constant":
            return producer.attributes["value"].value
        if producer.op_type == "Identity":
            # check for mypy, definitely true
            assert producer.inputs[0] is not None
            return get_constant_tensor_proto(producer.inputs[0])

    return None


def hash_values(v_list: Iterable[ir.Value]):
    hasher = hashlib.md5()
    for v in v_list:
        v_name = v.name if v.name is not None else ""
        hasher.update(v_name.encode())

    hash_str = hasher.hexdigest()
    return hash_str


def get_mmap_count(graph: ir.Graph):
    # count the mmap number that currently opened in graph
    # too many mmap can result to OS error
    count = 0
    for v in iter_all_values(graph):
        cst_value = get_constant_tensor_proto(v)
        if isinstance(cst_value, ir.ExternalTensor):
            if cst_value.raw is not None:
                count += 1
    return count
