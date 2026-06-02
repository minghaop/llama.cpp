# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Helper functions related to encodings

This module uses the Adapter Pattern to handle different encoding versions (V1.0.0, V2.0.0)
Each version has its own adapter class that encapsulates version-specific serialization/deserialization logic
This abstracts the internal representation (TensorEncodingInfo) from external encoding formats
"""

import copy
import enum
import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Type

import numpy as np
import onnx_ir as ir

EXPORT_DYNAMIC_MULTI_QUANT_ENCODINGS = os.getenv("EXPORT_DYNAMIC_MULTI_QUANT_ENCODINGS", "False").lower() in (
    "true",
    "1",
)


class EncType(enum.Enum):
    """
    Encodings Type
    """

    PER_CHANNEL = "PER_CHANNEL"
    PER_TENSOR = "PER_TENSOR"
    PER_BLOCK = "PER_BLOCK"
    LPBQ = "LPBQ"


class EncKind(enum.Enum):
    """
    Encodings Kind, param or activation
    """

    PARAM = "param_encodings"
    ACTIVATION = "activation_encodings"
    UNKNOWN = "unknown"


class GraphEncodingInfo:
    """
    Encodings of the whole graph
    """

    def __init__(self, version: str):
        self._validate_version(version)
        self._version = version

        self.encodings: dict[str, TensorEncodingInfo] = {}
        self.quantizer_args = None

        # for MoE model, we need to store encodings for each experts
        # key: tensor name, value: list of tensor encodings, index is the expert id
        self.MoE_expert_encodings: dict[str, list[TensorEncodingInfo | None]] = {}
        # key: tensor name, value: the value name(typically output of gather) in the graph that will be used as the dynamic expert id
        self._MoE_dyn_expert_id_map: dict[
            str, str
        ] = {}  # do not set this manually, it should be infered by infer_MoE_dyn_expert_id_map()

    @property
    def version(self) -> str:
        """Get the immutable version of this encoding"""
        return self._version

    def add_tensor_encodings(self, name, encodings):
        """
        Add tensor encodings to the graph encoding info

        Args:
            name: name of the tensor
            encodings: TensorEncInfo
        """
        if name in self.encodings:
            assert False, f"already added {name}"
        self.encodings[name] = encodings

    def add_MoE_expert_tensor_encodings(
        self, tensor_name: str, expert_id: int, encodings: "TensorEncodingInfo"
    ):
        if tensor_name not in self.MoE_expert_encodings:
            self.MoE_expert_encodings[tensor_name] = []
        expert_enc_list = self.MoE_expert_encodings[tensor_name]
        if expert_id >= len(expert_enc_list):
            expert_enc_list.extend([None] * (expert_id - len(expert_enc_list) + 1))
        assert expert_enc_list[expert_id] is None, f"expert {expert_id} of {tensor_name} is realdy added"
        expert_enc_list[expert_id] = encodings

    def infer_MoE_dyn_expert_id_map(self, graph: ir.Graph):
        from qairt.optimizer.onnx.utils.utils import scan_previous_nearest_candidate

        self._MoE_dyn_expert_id_map = {}
        if len(self.MoE_expert_encodings) == 0:
            return

        for n in graph:
            if n.op_type == "ElementWiseMux" and n.domain == "qti_aisw":
                assert n.inputs[0] is not None
                assert n.inputs[0].name is not None
                assert n.outputs[0].name is not None
                self._MoE_dyn_expert_id_map[n.inputs[0].name] = n.inputs[0].name
                self._MoE_dyn_expert_id_map[n.outputs[0].name] = n.inputs[0].name

        # assume the graph is topo-sorted
        for n in graph:
            if all(v.name not in self.MoE_expert_encodings for v in n.outputs):
                continue

            prev_v = None  # scan for the nearst previous value that has marked expert dynamic id name
            for candidate in scan_previous_nearest_candidate(
                list(n.inputs),
                check_fn=lambda v: v.name in self._MoE_dyn_expert_id_map,
                ignore_fn=lambda v: v.name not in self._MoE_dyn_expert_id_map
                and (
                    v.name not in self.MoE_expert_encodings or len(v.meta["extra_info"].named_encodings) == 0
                ),
            ):
                prev_v = candidate
                break
            assert prev_v is not None, "Cannot infer the dynamic index name of tensor '{}'".format(
                [v.name for v in n.outputs if v.name in self.MoE_expert_encodings]
            )

            for v in n.outputs:
                if v.name in self.MoE_expert_encodings:
                    self._MoE_dyn_expert_id_map[v.name] = self._MoE_dyn_expert_id_map[prev_v.name]

    def get_tensor_encodings(self, name):
        """
        Get tensor encodings by name

        Args:
            name: name of the tensor
        Returns:
            TensorEncInfo
        """
        if name in self.encodings:
            return self.encodings[name]
        raise ValueError(f"cannot find tensor encodings {name}")

    def create_similar(self) -> "GraphEncodingInfo":
        """
        Create a new GraphEncodingInfo instance with the same version and quantizer_args.

        Returns:
            GraphEncodingInfo: New instance with same version and copied quantizer_args
        """
        new_instance = GraphEncodingInfo(self._version)
        new_instance.quantizer_args = copy.deepcopy(self.quantizer_args)
        return new_instance

    @staticmethod
    def _validate_version(version: str):
        """
        Validate that the version is supported by the encoding adapter factory

        Args:
            version: Version string to validate

        Raises:
            ValueError: If version is not supported
        """
        # Import here to avoid circular import
        if version not in EncodingAdapterFactory._adapters:
            raise ValueError(f"Unsupported encoding version: {version}")


class TensorEncodingInfo:  # pylint: disable=[too-many-instance-attributes]
    """
    Encodings of the tensor
    """

    # pylint: disable=[too-many-arguments,too-many-positional-arguments,redefined-builtin]
    def __init__(
        self,
        enc_type: EncType,
        bw: int,
        dtype: str,
        is_sym: bool | None,
        offset: np.ndarray | None,
        scale: np.ndarray,
        max: np.ndarray | None,
        min: np.ndarray | None,
        enc_kind: EncKind = EncKind.UNKNOWN,
        zero_point_shift: np.ndarray | None = None,
    ):
        assert isinstance(enc_type, EncType)
        assert isinstance(enc_kind, EncKind)

        self.enc_kind = enc_kind
        self.enc_type = enc_type
        self.bw = bw
        self.dtype = dtype
        self.is_sym = is_sym
        self.offset = offset
        self.scale = scale

        # optional
        self.max: np.ndarray | None = max
        self.min: np.ndarray | None = min
        self.zero_point_shift: np.ndarray | None = zero_point_shift

        # extra attributes for LPBQ
        self.compressed_bw: int | None = None
        self.block_size: int | None = None
        self.per_block_int_scale: np.ndarray | None = None

        # channel_axis and block_axis should be inferred automatically
        self.channel_axis: int | None = 0
        self.block_axis: int | None = 1

        self.is_signed: bool | None = None  # optional

        # Track op_type for V1 serialization (needed to reverse transpose for MatMul/Gemm)
        self.op_type: str | None = None

    def slice(self, axis: int, start: int, end: int) -> "TensorEncodingInfo":
        """
        Slice encodings along the specified axis
        """

        if self.enc_type == EncType.PER_TENSOR:
            return copy.deepcopy(self)
        elif self.enc_type == EncType.PER_CHANNEL:
            return self._slice_per_channel(start, end, axis)
        elif self.enc_type == EncType.PER_BLOCK:
            return self._slice_per_block(start, end, axis)
        elif self.enc_type == EncType.LPBQ:
            return self._slice_lpbq(start, end, axis)
        else:
            raise ValueError(f"Unknown encoding type: {self.enc_type}")

    def _slice_per_channel(self, start: int, end: int, axis: int) -> "TensorEncodingInfo":
        """Slice PER_CHANNEL encodings along channel_axis"""
        new_enc = copy.deepcopy(self)

        # If slice axis is not the same as channel axis,
        # No slicing, just return the original encodings
        if axis != self.channel_axis:
            return new_enc

        # If slice axis is equal to channel axis, slice the (1D) TensorEncodingInfo fields
        new_enc.scale = self.scale[start:end]
        if self.offset is not None:
            new_enc.offset = self.offset[start:end]
        if self.min is not None:
            new_enc.min = self.min[start:end]
        if self.max is not None:
            new_enc.max = self.max[start:end]
        if self.zero_point_shift is not None:
            new_enc.zero_point_shift = self.zero_point_shift[start:end]
        return new_enc

    def _slice_per_block(self, start: int, end: int, axis: int) -> "TensorEncodingInfo":
        """Slice PER_BLOCK encodings"""
        new_enc = copy.deepcopy(self)

        # If slice axis is equal to block axis, slice the encodings field for every block
        # This requires dividing the start and end by block_size
        if axis == self.block_axis:
            assert self.block_size is not None, "Block size is a required field for PER_BLOCK encodings"
            start = start // self.block_size
            end = end // self.block_size

        # For other cases, including the case where slice axis == channel_axis
        # Slice the fields from start to end at the slice axis
        new_enc.scale = np.take(self.scale, range(start, end), axis=axis)
        if self.offset is not None:
            new_enc.offset = np.take(self.offset, range(start, end), axis=axis)
        if self.min is not None:
            new_enc.min = np.take(self.min, range(start, end), axis=axis)
        if self.max is not None:
            new_enc.max = np.take(self.max, range(start, end), axis=axis)
        if self.zero_point_shift is not None:
            new_enc.zero_point_shift = np.take(self.zero_point_shift, range(start, end), axis=axis)
        return new_enc

    def _slice_lpbq(self, start: int, end: int, axis: int) -> "TensorEncodingInfo":
        """Slice LPBQ encodings along channel_axis"""
        new_enc = copy.deepcopy(self)

        # If slice axis is neither channel axis or block axis
        # No slicing, just return the original encodings
        if axis != self.channel_axis and axis != self.block_axis:
            return new_enc

        # If slice axis is equal to block axis, slice the per_block_int_scale field for every block
        # This requires dividing the start and end by block_size
        # No other field needs to be sliced, since they are "per_channel" fields
        if axis == self.block_axis:
            assert self.per_block_int_scale is not None, (
                "per_block_int_scale is a required field for LPBQ encodings"
            )
            assert self.block_size is not None, "Block size is a required field for LPBQ encodings"

            start = start // self.block_size
            end = end // self.block_size

            new_enc.per_block_int_scale = np.take(self.per_block_int_scale, range(start, end), axis=axis)
            return new_enc

        # axis == self.channel_axis
        else:
            # If slice axis is equal to channel axis, slice the encoding fields at channel axis
            if self.per_block_int_scale is not None:
                new_enc.per_block_int_scale = np.take(self.per_block_int_scale, range(start, end), axis=axis)
            if self.scale is not None:
                new_enc.scale = np.take(self.scale, range(start, end), axis=axis)
            if self.offset is not None:
                new_enc.offset = np.take(self.offset, range(start, end), axis=axis)
            if self.min is not None:
                new_enc.min = np.take(self.min, range(start, end), axis=axis)
            if self.max is not None:
                new_enc.max = np.take(self.max, range(start, end), axis=axis)
            if self.zero_point_shift is not None:
                new_enc.zero_point_shift = np.take(self.zero_point_shift, range(start, end), axis=axis)

            return new_enc

    def __eq__(self, other):
        for attr in self.__dict__:
            v = getattr(self, attr)
            other_v = getattr(other, attr)
            if isinstance(v, np.ndarray) and isinstance(other_v, np.ndarray):
                if v.shape != other_v.shape:
                    return False
                if not (v == other_v).all():
                    return False
            elif getattr(self, attr) != getattr(other, attr):
                return False
        return True


class EncodingAdapter(ABC):
    """Abstract base class for encoding version adapters"""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version string"""
        pass

    @staticmethod
    @abstractmethod
    def deserialize(src_enc: dict, model_ir: ir.Model | None = None) -> GraphEncodingInfo:
        """Deserialize encodings from JSON format"""
        pass

    @staticmethod
    @abstractmethod
    def deserialize_tensor_encodings(t_enc_dict: dict, tensor: ir.Value | None = None) -> TensorEncodingInfo:
        """Deserialize a single tensor encoding. Tensor is optional (only needed for V1)"""
        pass

    @staticmethod
    @abstractmethod
    def serialize(encodings: GraphEncodingInfo) -> dict:
        """Serialize encodings to JSON format"""
        pass

    @staticmethod
    @abstractmethod
    def serialize_tensor_encodings(t_enc: TensorEncodingInfo, name: str) -> dict:
        """Serialize a single tensor encoding"""
        pass

    @classmethod
    def serialize_dynamic_tensor_encodings(
        cls, t_enc_list: list[TensorEncodingInfo], tensor_name: str, index_tensor_name: str
    ) -> dict:
        # this is typically used for MoE models
        assert len(t_enc_list) >= 1

        for d in t_enc_list[1:]:
            if d.enc_type != t_enc_list[0].enc_type:
                raise ValueError(f"dynamic encodings of tensor {tensor_name} should have same enc_type")
            if d.channel_axis != t_enc_list[0].channel_axis:  # can be None
                raise ValueError(f"dynamic encodings of tensor {tensor_name} should have same channel_axis")
            if d.block_axis != t_enc_list[0].block_axis:  # can be None
                raise ValueError(f"dynamic encodings of tensor {tensor_name} should have same block_axis")

        enc_dict_list = [cls.serialize_tensor_encodings(x, tensor_name) for x in t_enc_list]
        dynamic_enc_dict: dict[str, Any] = {}
        # Put these entries first to improve encodings file readability
        dynamic_enc_dict["is_dynamic"] = True
        dynamic_enc_dict["index"] = index_tensor_name
        for k, v in enc_dict_list[0].items():
            if k not in dynamic_enc_dict:
                dynamic_enc_dict[k] = copy.deepcopy(v)

        def merge_list_attr(
            dict_list: list[dict], attr_name: str, out_dict: dict, default_creator: Callable | None = None
        ):
            if all(attr_name not in d for d in dict_list):
                return
            merged_attrs = []
            for d in dict_list:
                if attr_name in d:
                    merged_attrs.append(d[attr_name])
                elif default_creator is not None:
                    merged_attrs.append(default_creator())
                else:
                    raise ValueError(
                        f"not all experts of {tensor_name} has same encodings field: {attr_name}"
                    )

            out_dict[attr_name] = merged_attrs

        # fields for v1
        merge_list_attr(enc_dict_list, "scale", dynamic_enc_dict)
        merge_list_attr(enc_dict_list, "offset", dynamic_enc_dict)
        merge_list_attr(enc_dict_list, "zero_point_shift", dynamic_enc_dict)
        merge_list_attr(enc_dict_list, "max", dynamic_enc_dict)
        merge_list_attr(enc_dict_list, "min", dynamic_enc_dict)

        # fields for v2
        merge_list_attr(enc_dict_list, "y_scale", dynamic_enc_dict)

        def get_default_y_zero_point():
            y_scale = dynamic_enc_dict["y_scale"][0]
            if t_enc_list[0].enc_type == EncType.PER_TENSOR and np.array(y_scale).size == 1:
                return 0
            else:
                return np.zeros_like(y_scale, dtype=np.int64).tolist()

        merge_list_attr(
            enc_dict_list, "y_zero_point", dynamic_enc_dict, default_creator=get_default_y_zero_point
        )
        merge_list_attr(enc_dict_list, "per_block_int_scale", dynamic_enc_dict)  # v1/v2
        return dynamic_enc_dict


class EncodingAdapterV1(EncodingAdapter):
    """Adapter for V1.0.0 encoding format"""

    @property
    def version(self) -> str:
        return "1.0.0"

    @staticmethod
    def deserialize(src_enc: dict, model_ir: ir.Model | None = None) -> GraphEncodingInfo:
        """Deserialize V1.0.0 encodings"""
        if model_ir is None:
            raise ValueError("model_ir required for V1 deserialization")
        elif not isinstance(model_ir, ir.Model):
            raise TypeError(
                f"Invalid type for parameter model_ir: {type(model_ir)}. Expected an argument of type ir.Model"
            )

        # Import here to avoid circular import
        from qairt.optimizer.onnx.utils.utils import iter_all_values

        if src_enc["version"] == "0.6.1":
            src_enc = convert_v0_6_1_to_v1(src_enc)

        graph_enc = GraphEncodingInfo(version="1.0.0")
        graph_enc.quantizer_args = src_enc.get("quantizer_args")

        name_to_tensor = {v.name: v for v in iter_all_values(model_ir.graph)}

        for enc_kind in [EncKind.PARAM, EncKind.ACTIVATION]:
            for t_enc_dict in src_enc[enc_kind.value]:
                name = t_enc_dict["name"]
                tensor = name_to_tensor.get(name)
                if tensor is None:
                    continue

                new_t_enc = EncodingAdapterV1.deserialize_tensor_encodings(t_enc_dict, tensor)
                new_t_enc.enc_kind = enc_kind
                graph_enc.encodings[name] = new_t_enc
        return graph_enc

    @staticmethod
    def deserialize_tensor_encodings(t_enc_dict: dict, tensor: ir.Value | None = None) -> TensorEncodingInfo:
        """Deserialize a single tensor encoding from V1.0.0 format"""
        if tensor is None:
            raise ValueError("V1 deserialization requires tensor parameter")

        enc_type = EncType[t_enc_dict["enc_type"]]
        offset = _get_field_as_np_array(t_enc_dict, "offset")
        scale = _get_field_as_np_array(t_enc_dict, "scale")
        max_val = _get_field_as_np_array(t_enc_dict, "max")
        min_val = _get_field_as_np_array(t_enc_dict, "min")
        zero_point_shift = _get_field_as_np_array(t_enc_dict, "zero_point_shift")

        # Only infer axes for non-PER_TENSOR types
        if enc_type != EncType.PER_TENSOR:
            channel_axis, block_axis = EncodingAdapterV1._infer_channel_and_block_axis(tensor)
        else:
            channel_axis, block_axis = None, None

        per_block_int_scale = None
        block_size = None

        if enc_type == EncType.LPBQ:
            block_size = t_enc_dict["block_size"]

            # For LPBQ, these fields are only defined per-channel
            assert tensor.shape is not None, (
                f"Tensor {tensor.name} has not static shape, please run shape infer firstly"
            )
            assert channel_axis is not None, f"channel_axis None for LPBQ encodings for tensor {tensor.name}"

            scale = EncodingAdapterV1._reshape_1d_per_channel_to_nd(scale, tensor.shape.numpy(), channel_axis)
            if offset is not None:
                offset = EncodingAdapterV1._reshape_1d_per_channel_to_nd(
                    offset, tensor.shape.numpy(), channel_axis
                )
            if min_val is not None:
                min_val = EncodingAdapterV1._reshape_1d_per_channel_to_nd(
                    min_val, tensor.shape.numpy(), channel_axis
                )
            if max_val is not None:
                max_val = EncodingAdapterV1._reshape_1d_per_channel_to_nd(
                    max_val, tensor.shape.numpy(), channel_axis
                )
            if zero_point_shift is not None:
                zero_point_shift = EncodingAdapterV1._reshape_1d_per_channel_to_nd(
                    zero_point_shift, tensor.shape.numpy(), channel_axis
                )

            per_block_int_scale = _get_field_as_np_array(t_enc_dict, "per_block_int_scale")
            per_block_int_scale = EncodingAdapterV1._reshape_1d_per_block_to_nd(
                per_block_int_scale, tensor, block_size
            )

        elif enc_type == EncType.PER_BLOCK:
            block_size = t_enc_dict["block_size"]
            scale = EncodingAdapterV1._reshape_1d_per_block_to_nd(scale, tensor, block_size)
            if offset is not None:
                offset = EncodingAdapterV1._reshape_1d_per_block_to_nd(offset, tensor, block_size)
            if min_val is not None:
                min_val = EncodingAdapterV1._reshape_1d_per_block_to_nd(min_val, tensor, block_size)
            if max_val is not None:
                max_val = EncodingAdapterV1._reshape_1d_per_block_to_nd(max_val, tensor, block_size)
            if zero_point_shift is not None:
                zero_point_shift = EncodingAdapterV1._reshape_1d_per_block_to_nd(
                    zero_point_shift, tensor, block_size
                )

        new_t_enc = TensorEncodingInfo(
            enc_type=enc_type,
            bw=t_enc_dict["bw"],
            dtype=t_enc_dict["dtype"],
            is_sym=t_enc_dict.get("is_sym", None),
            offset=offset,
            scale=scale,
            max=max_val,
            min=min_val,
            zero_point_shift=zero_point_shift,
        )
        new_t_enc.channel_axis = channel_axis
        new_t_enc.block_axis = block_axis

        if enc_type in [EncType.PER_BLOCK, EncType.LPBQ]:
            new_t_enc.block_size = block_size
            # Store op_type for V1 serialization (needed to reverse transpose)
            consumer = EncodingAdapterV1._get_weight_consumer(tensor)

            if consumer:
                new_t_enc.op_type = consumer.op_type

            if enc_type == EncType.LPBQ:
                new_t_enc.per_block_int_scale = per_block_int_scale
                new_t_enc.compressed_bw = t_enc_dict["compressed_bw"]

        return new_t_enc

    @staticmethod
    def serialize(encodings: GraphEncodingInfo) -> dict:
        """Serialize to V1.0.0 format"""
        data: dict = {
            "quantizer_args": encodings.quantizer_args,
            "activation_encodings": [],
            "param_encodings": [],
            "version": "1.0.0",
        }

        for name, t_enc in encodings.encodings.items():
            if name in encodings.MoE_expert_encodings and EXPORT_DYNAMIC_MULTI_QUANT_ENCODINGS:
                assert all(x is not None for x in encodings.MoE_expert_encodings[name])
                moe_expert_enc: list[TensorEncodingInfo] = [x for x in encodings.MoE_expert_encodings[name]]  # type: ignore[misc]
                curr_enc = EncodingAdapterV1.serialize_dynamic_tensor_encodings(
                    moe_expert_enc, name, encodings._MoE_dyn_expert_id_map[name]
                )
            else:
                curr_enc = EncodingAdapterV1.serialize_tensor_encodings(t_enc, name)
            data[t_enc.enc_kind.value].append(curr_enc)
        return data

    @staticmethod
    def serialize_tensor_encodings(t_enc: TensorEncodingInfo, name: str) -> dict:
        """Serialize TensorEncodingInfo to V1.0.0 format"""

        def _maybe_transpose_and_flatten(array: np.ndarray | None) -> list | None:
            """
            Helper to transpose (if needed) and flatten array for V1 serialization

            This reverses the transpose done in _reshape_1d_per_block_to_nd during deserialization.
            For MatMul/Gemm where channel_axis > block_axis, we need to transpose back to
            [out_channels, num_blocks] order before flattening to 1D.

            Code: scale = scale.swapaxes(0, 1).flatten()
            """
            if array is None:
                return None

            # For PER_BLOCK and LPBQ with MatMul/Gemm, need to reverse transpose before flattening
            needs_transpose = (
                t_enc.enc_type in (EncType.PER_BLOCK, EncType.LPBQ)
                and t_enc.op_type in ("MatMul", "Gemm")
                and t_enc.channel_axis is not None
                and t_enc.block_axis is not None
                and t_enc.channel_axis > t_enc.block_axis
            )

            if needs_transpose:
                # Swap first two dimensions to reverse the transpose from deserialization
                array = np.swapaxes(array, 0, 1)

            return array.flatten().tolist()

        curr_enc: dict = {
            "name": name,
            "enc_type": t_enc.enc_type.value,
            "bw": t_enc.bw,
            "dtype": t_enc.dtype,
        }
        if t_enc.is_sym is not None:
            curr_enc["is_sym"] = t_enc.is_sym
        if t_enc.offset is not None:
            curr_enc["offset"] = _maybe_transpose_and_flatten(t_enc.offset)
        if t_enc.scale is not None:
            curr_enc["scale"] = _maybe_transpose_and_flatten(t_enc.scale)
        if t_enc.max is not None:
            curr_enc["max"] = _maybe_transpose_and_flatten(t_enc.max)
        if t_enc.min is not None:
            curr_enc["min"] = _maybe_transpose_and_flatten(t_enc.min)
        if t_enc.zero_point_shift is not None:
            curr_enc["zero_point_shift"] = _maybe_transpose_and_flatten(t_enc.zero_point_shift)
        if t_enc.enc_type == EncType.PER_BLOCK:
            curr_enc["block_size"] = t_enc.block_size
        if t_enc.enc_type == EncType.LPBQ:
            assert isinstance(t_enc.per_block_int_scale, np.ndarray)
            assert t_enc.block_size is not None
            assert t_enc.compressed_bw is not None

            curr_enc["per_block_int_scale"] = _maybe_transpose_and_flatten(t_enc.per_block_int_scale)
            curr_enc["block_size"] = t_enc.block_size
            curr_enc["compressed_bw"] = t_enc.compressed_bw
        return curr_enc

    @staticmethod
    def _reshape_1d_per_block_to_nd(array_1d: np.ndarray, tensor: ir.Value, block_size: int) -> np.ndarray:
        """
        Reshape 1D array to ND for PER_BLOCK and LPBQ quantization

        V1 Encoding Storage Format:
        ---------------------------
        V1 encodings always store per-block data as a 1D array with length (out_channels * num_blocks).
        The 1D array is ALWAYS stored in [out_channels, num_blocks] order, regardless of the actual
        tensor layout or op type (Conv, MatMul, Gemm, etc.).

        Reshaping Strategy:
        ------------------
        1. First, reshape 1D array to intermediate shape: [out_channels, num_blocks, 1, 1, ...]
        2. Then, if channel_axis > block_axis (e.g., MatMul, Gemm with transB=False):
           - Transpose first two dimensions to match tensor layout
           - Code: scale = scale.reshape(out_channels, num_blocks).swapaxes(0, 1)

        Examples:
        ---------
        Conv weight (3072, 3072, 1, 1), block_size=64:
          - channel_axis=0, block_axis=1
          - 1D array [147456] -> reshape to (3072, 48, 1, 1)
          - No transpose needed (channel_axis < block_axis)

        MatMul weight (3072, 3072), block_size=64:
          - channel_axis=1, block_axis=0
          - 1D array [147456] -> reshape to (3072, 48) -> transpose to (48, 3072)
          - Transpose needed (channel_axis > block_axis)
          - This matches V2 format where scale would be stored at block_axis

        Serialization (see _maybe_transpose_and_flatten):
        -------------------------------------------------
        When serializing back to V1, we reverse this process:
        - If channel_axis > block_axis (e.g., MatMul, Gemm with transB=False): transpose back before flattening
        - This ensures round-trip consistency: deserialize -> serialize -> deserialize

        ConvTranspose Special Case:
        ---------------------------
        ConvTranspose is handled separately because V1 stores it differently:
        - V1 stores as [num_blocks, out_channels, ...] directly
        - No intermediate reshape + transpose needed

        NOTE: Difference between V1.0.0 and V2.0.0
        ------------------------------------------
        V2.0.0 encodings align with ONNX QuantizeLinear/DequantizeLinear format.
        For a Conv weight (32, 64, 3, 3) with block_size=8:
        - V2: scale shape is (32, 8, 3, 3) - matches ONNX requirement
        - V1: 1D array length is (32 * 8 * 1 * 1) = 256 - more compact

        V1 doesn't need to match ONNX format, so it uses a more compact representation.
        """
        from qairt.optimizer.onnx.utils.utils import (
            check_static_shape,
        )

        check_static_shape(tensor)

        assert tensor.shape is not None
        tensor_shape = tensor.shape.numpy()

        consumer = EncodingAdapterV1._get_weight_consumer(tensor)
        if consumer:
            op_type = consumer.op_type
        else:
            op_type = None

        channel_axis, block_axis = EncodingAdapterV1._infer_channel_and_block_axis(tensor)

        out_channels = tensor_shape[channel_axis]
        num_blocks = tensor_shape[block_axis] // block_size

        if op_type == "ConvTranspose":
            # ConvTranspose: V1 stores as [num_blocks, out_channels, ...]
            nd_shape = [1] * len(tensor_shape)
            nd_shape[block_axis] = num_blocks
            nd_shape[channel_axis] = out_channels

            return array_1d.reshape(nd_shape)

        else:
            # Conv, MatMul, Gemm: V1 always stores as [out_channels, num_blocks, ...]
            # Step 1: Reshape to intermediate shape
            intermediate_shape = [out_channels, num_blocks] + [1] * (len(tensor_shape) - 2)
            reshaped = array_1d.reshape(intermediate_shape)

            # Step 2: Transpose if needed to match tensor layout
            # For MatMul and Gemm with transB=False: channel_axis > block_axis
            # So we need to swap dimensions to get [num_blocks, out_channels, ...]
            if channel_axis > block_axis:
                # Swap the first two dimensions using np.swapaxes
                # This matches the V2 format where scale is stored at block_axis
                reshaped = np.swapaxes(reshaped, 0, 1)

            return reshaped

    @staticmethod
    def _reshape_1d_per_channel_to_nd(
        array_1d: np.ndarray, tensor_shape: tuple, channel_axis: int
    ) -> np.ndarray:
        """Reshape 1D per-channel array to ND"""
        nd_shape = [1] * len(tensor_shape)
        nd_shape[channel_axis] = tensor_shape[channel_axis]
        return array_1d.reshape(nd_shape)

    @staticmethod
    def _get_weight_consumer(tensor: ir.Value) -> ir.Node | None:
        """
        Get the node that uses this tensor as a weight

        For weight tensors, this looks for usage at arg_id==1 (weight position)
        Falls back to first consumer if no weight usage found

        Args:
            tensor: The tensor to find the weight consumer for

        Returns:
            ir.Node: The consumer operation node

        Raises:
            ValueError: If tensor has no consumers
        """
        # First, try to find usage as a weight (arg_id == 1)
        for user, arg_id in tensor.uses():
            if arg_id == 1:
                return user

        # Fallback: use first consumer if no weight usage found
        if tensor.consumers():
            return tensor.consumers()[0]

        return None

    @staticmethod
    def _infer_channel_and_block_axis(tensor: ir.Value) -> tuple[int, int]:
        """
        Infer encodings' channel axis and block axis for the given tensor
        """
        if tensor.shape is None:
            raise ValueError(
                f"cannot get the shape of {tensor.name}, shape inference should be called firstly"
            )

        tensor_shape = tensor.shape.numpy()

        consumer = EncodingAdapterV1._get_weight_consumer(tensor)

        if consumer and consumer.op_type == "MatMul":
            rank = len(tensor_shape)
            return (rank - 1, rank - 2)
        elif consumer and consumer.op_type == "Gemm":
            # Gemm: output channel axis depends on transB
            # If transB=False: weight is (in, out), so out_channel_axis=1
            # If transB=True: weight is (out, in), so out_channel_axis=0
            trans_b = consumer.attributes.get("transB", 0)
            return (0, 1) if trans_b else (1, 0)
        else:
            # For other cases, for example Conv/ConvTranspose/RMSnorm,
            # we assume the first dim is the channel dim
            # and the second dim is the block dim
            return (0, 1)


class EncodingAdapterV2(EncodingAdapter):
    """Adapter for V2.0.0 encoding format"""

    @property
    def version(self) -> str:
        return "2.0.0"

    @staticmethod
    def deserialize(src_enc: dict, model_ir: ir.Model | None = None) -> GraphEncodingInfo:
        """Deserialize V2.0.0 encodings"""
        graph_enc = GraphEncodingInfo(version="2.0.0")

        for t_enc_dict in src_enc.get("encodings", []):
            name = t_enc_dict["name"]
            new_t_enc = EncodingAdapterV2.deserialize_tensor_encodings(t_enc_dict)
            graph_enc.encodings[name] = new_t_enc
        return graph_enc

    @staticmethod
    def deserialize_tensor_encodings(t_enc_dict: dict, tensor: ir.Value | None = None) -> TensorEncodingInfo:
        """Deserialize a single tensor encoding from V2.0.0 format"""
        enc_type = EncodingAdapterV2._infer_enc_type_from_v2(t_enc_dict)
        output_dtype = t_enc_dict["output_dtype"]
        bw = EncodingAdapterV2._extract_bitwidth(output_dtype)
        dtype = output_dtype.replace(str(bw), "").upper()

        y_scale = _get_field_as_np_array(t_enc_dict, "y_scale")
        y_zero_point = _get_field_as_np_array(t_enc_dict, "y_zero_point")
        per_block_int_scale = _get_field_as_np_array(t_enc_dict, "per_block_int_scale")
        per_channel_float_scale = _get_field_as_np_array(t_enc_dict, "per_channel_float_scale")

        if enc_type == EncType.LPBQ:
            scale = per_channel_float_scale
        else:
            scale = y_scale

        # NOTE: Our definition of "offset" differs from that of "zero_point" in ONNX. (offset := -zero_point)
        if y_zero_point is None:
            # y zero point is optional, default to zero with y_scale's shape
            y_zero_point = np.zeros_like(y_scale, dtype=np.int64)

        offset = -y_zero_point

        new_t_enc = TensorEncodingInfo(
            enc_type=enc_type,
            bw=bw,
            dtype=dtype,
            is_sym=None,
            offset=offset,
            scale=scale,
            max=None,
            min=None,
            enc_kind=EncKind.UNKNOWN,
        )

        # Set axis fields based on encoding type
        axis = t_enc_dict.get("axis")
        if enc_type == EncType.PER_CHANNEL:
            # For PER_CHANNEL: axis is the channel_axis
            new_t_enc.channel_axis = axis
        elif enc_type == EncType.PER_BLOCK:
            # For PER_BLOCK: axis is the block_axis, channel_axis not needed
            new_t_enc.block_axis = axis
        elif enc_type == EncType.LPBQ:
            # For LPBQ: axis is the block_axis
            new_t_enc.block_axis = axis
            assert axis is not None, f"axis is None for LPBQ encodings for tensor"
            # Derive channel_axis by comparing shapes of per_block_int_scale and per_channel_float_scale
            if per_block_int_scale is not None and per_channel_float_scale is not None:
                new_t_enc.channel_axis = EncodingAdapterV2._infer_channel_axis_for_lpbq(
                    per_block_int_scale, per_channel_float_scale, axis
                )

        if enc_type in (EncType.PER_BLOCK, EncType.LPBQ):
            new_t_enc.block_size = t_enc_dict.get("block_size")
        if enc_type == EncType.LPBQ:
            new_t_enc.per_block_int_scale = per_block_int_scale

        return new_t_enc

    @staticmethod
    def serialize(encodings: GraphEncodingInfo) -> dict:
        """Serialize to V2.0.0 format"""
        data: dict = {"version": "2.0.0", "encodings": []}

        for name, t_enc in encodings.encodings.items():
            if name in encodings.MoE_expert_encodings and EXPORT_DYNAMIC_MULTI_QUANT_ENCODINGS:
                assert all(x is not None for x in encodings.MoE_expert_encodings[name])
                moe_expert_enc: list[TensorEncodingInfo] = [x for x in encodings.MoE_expert_encodings[name]]  # type: ignore[misc]
                curr_enc = EncodingAdapterV2.serialize_dynamic_tensor_encodings(
                    moe_expert_enc, name, encodings._MoE_dyn_expert_id_map[name]
                )
            else:
                curr_enc = EncodingAdapterV2.serialize_tensor_encodings(t_enc, name)
            data["encodings"].append(curr_enc)  # type: ignore[union-attr]
        return data

    @staticmethod
    def serialize_tensor_encodings(t_enc: TensorEncodingInfo, name: str) -> dict:
        """Serialize TensorEncodingInfo to V2.0.0 format"""
        enc_dict: dict = {
            "name": name,
            "output_dtype": f"{t_enc.dtype.lower()}{t_enc.bw}",
        }

        # Add scale field (y_scale or per_channel_float_scale for LPBQ)
        if t_enc.scale is not None:
            if t_enc.enc_type == EncType.LPBQ:
                enc_dict["per_channel_float_scale"] = t_enc.scale.tolist()
            elif t_enc.enc_type == EncType.PER_TENSOR and t_enc.scale.size == 1:
                enc_dict["y_scale"] = t_enc.scale.item()
            else:
                enc_dict["y_scale"] = t_enc.scale.tolist()

        # NOTE: Our definition of "offset" differs from that of "zero_point" in ONNX. (offset := -zero_point)
        # As defined in onnx::QuantizedLinear, y_zero_point is optional if it's all 0
        if t_enc.offset is not None and not (t_enc.offset == 0).all():
            zero_point = -t_enc.offset
            if not (zero_point == 0).all():
                if t_enc.enc_type == EncType.PER_TENSOR and zero_point.size == 1:
                    enc_dict["y_zero_point"] = zero_point.item()
                else:
                    enc_dict["y_zero_point"] = zero_point.tolist()

        # Add axis field
        if t_enc.enc_type == EncType.PER_CHANNEL and t_enc.channel_axis is not None:
            enc_dict["axis"] = t_enc.channel_axis
        elif t_enc.enc_type in (EncType.PER_BLOCK, EncType.LPBQ) and t_enc.block_axis is not None:
            enc_dict["axis"] = t_enc.block_axis

        # Add block_size for PER_BLOCK and LPBQ
        if t_enc.enc_type in (EncType.PER_BLOCK, EncType.LPBQ) and t_enc.block_size is not None:
            enc_dict["block_size"] = t_enc.block_size

        # Add per_block_int_scale for LPBQ
        if t_enc.enc_type == EncType.LPBQ and t_enc.per_block_int_scale is not None:
            enc_dict["per_block_int_scale"] = t_enc.per_block_int_scale.tolist()

        return enc_dict

    @staticmethod
    def _extract_bitwidth(dtype_str: str) -> int:
        """Extract bitwidth from dtype string"""
        match = re.search(r"\d+", dtype_str)
        if match:
            return int(match.group())
        raise ValueError(f"Cannot extract bitwidth from dtype string: {dtype_str}")

    @staticmethod
    def _infer_enc_type_from_v2(t_enc_dict: dict) -> EncType:
        """Infer encoding type from V2 structure"""
        if "per_block_int_scale" in t_enc_dict and "per_channel_float_scale" in t_enc_dict:
            return EncType.LPBQ
        elif "block_size" in t_enc_dict:
            return EncType.PER_BLOCK
        elif "axis" in t_enc_dict:
            return EncType.PER_CHANNEL
        else:
            return EncType.PER_TENSOR

    @staticmethod
    def _infer_channel_axis_for_lpbq(
        per_block_int_scale: np.ndarray, per_channel_float_scale: np.ndarray, block_axis: int
    ) -> int:
        """
        Infer channel_axis for LPBQ by comparing shapes of per_block_int_scale and per_channel_float_scale

        Logic:
        - Both arrays have the same number of dimensions as the tensor
        - Both have the same number of elements in the channel_axis
        - Different elements in the block_axis (per_block has more due to blocking)
        - per_channel_float_scale has 1s everywhere except channel_axis
        - per_block_int_scale may have 1s or other values in other dimensions

        Example: For tensor shape (128256, 3072, 3, 3) with block_size=64:
        - per_block_int_scale shape: (128256, 48, 3, 3)  # 3072/64 = 48
        - per_channel_float_scale shape: (128256, 1, 1, 1)
        - channel_axis = 0 (both have 128256 elements)
        - block_axis = 1 (48 vs 1)

        Args:
            per_block_int_scale: The per-block integer scale array
            per_channel_float_scale: The per-channel float scale array
            block_axis: The axis along which blocking is performed

        Returns:
            The inferred channel axis
        """
        pb_shape = per_block_int_scale.shape
        pc_shape = per_channel_float_scale.shape

        if len(pb_shape) != len(pc_shape):
            raise ValueError(
                f"Shape mismatch: per_block_int_scale has {len(pb_shape)} dims, "
                f"per_channel_float_scale has {len(pc_shape)} dims"
            )

        # Find the axis where:
        # 1. Both arrays have the same size (not 1)
        # 2. It's not the block_axis
        # 3. per_channel_float_scale has size > 1 (not a broadcast dimension)
        for axis in range(len(pb_shape)):
            if axis == block_axis:
                continue
            if pb_shape[axis] == pc_shape[axis] and pc_shape[axis] > 1:
                return axis

        # If we can't find a clear channel axis, this is an error
        raise ValueError(
            f"Cannot infer channel_axis for LPBQ encoding. "
            f"per_block_int_scale shape: {pb_shape}, "
            f"per_channel_float_scale shape: {pc_shape}, "
            f"block_axis: {block_axis}. "
            f"Expected to find an axis (other than block_axis) where both arrays have matching size > 1."
        )


class EncodingAdapterFactory:
    """Factory for creating encoding adapters"""

    _adapters: dict[str, Type[EncodingAdapter]] = {
        "0.6.1": EncodingAdapterV1,
        "1.0.0": EncodingAdapterV1,
        "2.0.0": EncodingAdapterV2,
    }

    @classmethod
    def get_adapter(cls, version: str) -> EncodingAdapter:
        """Get adapter for a specific version"""
        if version not in cls._adapters:
            raise ValueError(f"Unsupported encoding version: {version}")
        return cls._adapters[version]()

    @classmethod
    def detect_and_get_adapter(cls, src_enc: dict) -> EncodingAdapter:
        """Detect version from encoding dict and return appropriate adapter"""
        try:
            version = src_enc["version"]
            return cls.get_adapter(version)
        except KeyError:
            raise KeyError("Unable to determine encoding version from json")


def load_encodings(encodings_src: str, model_ir: ir.Model | None = None) -> GraphEncodingInfo:
    """
    Load encodings from file
    """
    with open(encodings_src, "r", encoding="utf-8") as f:
        src_enc = json.load(f)
    return deserialize_encodings(src_enc, model_ir)


def deserialize_encodings(src_enc: dict, model_ir: ir.Model | None = None) -> GraphEncodingInfo:
    """
    Deserialize json encodings to GraphEncodingInfo using appropriate adapter
    """
    adapter = EncodingAdapterFactory.detect_and_get_adapter(src_enc)
    return adapter.deserialize(src_enc, model_ir)


def serialize_graph_encodings(encodings: GraphEncodingInfo) -> dict:
    """
    Serialize GraphEncodingInfo to json encodings using appropriate adapter
    """
    adapter = EncodingAdapterFactory.get_adapter(encodings.version)
    return adapter.serialize(encodings)


def save_encodings(encodings: GraphEncodingInfo, dst_path: str | os.PathLike):
    """
    Save GraphEncodingInfo to json file
    """
    data = serialize_graph_encodings(encodings)

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=4))


def convert_v0_6_1_to_v1(src_enc: dict) -> dict:
    """
    Convert v0.6.1 aimet encodings to v1 encodings
    """
    if src_enc["version"] == "1.0.0":
        return src_enc
    assert src_enc["version"] == "0.6.1"
    dst_enc: dict = {"version": "1.0.0"}

    is_symmetric_map = {"False": False, "True": True, "false": False, "true": True}

    for e_type in ["activation_encodings", "param_encodings"]:
        dst_enc[e_type] = []
        for name, t_enc in src_enc[e_type].items():
            new_t_enc: dict = {
                "bw": t_enc[0]["bitwidth"],
                "dtype": t_enc[0]["dtype"].upper(),
                "name": name,
            }
            if "is_symmetric" in t_enc[0]:
                new_t_enc["is_sym"] = is_symmetric_map[t_enc[0]["is_symmetric"]]
            if "offset" in t_enc[0]:
                new_t_enc["offset"] = [x["offset"] for x in t_enc]
            if "scale" in t_enc[0]:
                new_t_enc["scale"] = [x["scale"] for x in t_enc]
            if "max" in t_enc[0]:
                new_t_enc["max"] = [x["max"] for x in t_enc]
            if "min" in t_enc[0]:
                new_t_enc["min"] = [x["min"] for x in t_enc]

            if "offset" in new_t_enc and len(new_t_enc["offset"]) > 1:
                new_t_enc["enc_type"] = "PER_CHANNEL"
            else:
                new_t_enc["enc_type"] = "PER_TENSOR"
            dst_enc[e_type].append(new_t_enc)

    dst_enc["quantizer_args"] = src_enc["quantizer_args"]
    return dst_enc


def _get_field_as_np_array(dict_json: dict, name: str, default=None):
    """Get json field as np array"""
    if name not in dict_json:
        return default
    if dict_json[name] is None:
        return default
    return np.array(dict_json[name])
