# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import numpy as np
from math import isclose
from typing import Any
import os
import json
import copy
import yaml
import onnx
from safetensors.numpy import load_file, save_file
from .helpers import (open_and_load_json,
                      create_base_graph_name, validate_file_path, apply_safetensors_to_onnx, calculate_v1_scale_offset, calculate_v2_scale_zero_point)
from qti.aisw.converters.common.converter_ir.op_graph import QuantUpdatableMode
from qti.aisw.converters.common.tensor_transforms.transform_manager import TransformManager
from qti.aisw.converters.common.tensor_transforms.operator import Operator
from qti.aisw.converters.common.utils.converter_utils import (
    log_debug, log_warning, log_info, log_assert, log_error)
from qti.aisw.converters.common.encoding_handler.encoding_handler_factory import EncodingHandlerFactory
from qti.aisw.converters.common.encoding_handler.encoding_handler import TensorType, EncodingVersion, EncodingType



class DefaultEncoding(object):
    """
    This class contains the default values for all fields related to encoding.
    These min/max values will be used as encodings of weights and activation tensors(zero tensor)
    of the inactive lora branches.
    """
    MAX = 0.0039061307907104492
    MIN = -0.00390625

    @staticmethod
    def get_v1_scale_offset(quant_info):
        scale, offset = calculate_v1_scale_offset(
            DefaultEncoding.MIN,
            DefaultEncoding.MAX,
            quant_info.bitwidth,
            quant_info.is_symmetric,
            False
        )
        return scale, offset

    @staticmethod
    def get_v2_scale_zero_point(quant_info):
        scale, zero_point = calculate_v2_scale_zero_point(
            DefaultEncoding.MIN,
            DefaultEncoding.MAX,
            quant_info.bitwidth,
            quant_info.is_symmetric,
            False,
            quant_info.dtype
        )
        return scale, zero_point


class EncodingGenerator(object):

    def __init__(self, attach_point_info_map, max_rank_attach_point_map, base_lora_alpha_encoding=None):
        """
        This class handles the generation of the updated use case encoding
        for concurrences (including base concurrency).
        :param attach_point_info_map: A dictionary mapping attach point names to AttachPointInfo objects
        :param max_rank_attach_point_map: A dictionary mapping attach point names to MaxRankAttachPoint objects
        :param base_lora_alpha_encoding: Encoding for lora alpha for base concurrency.
        """
        self.attach_point_info_map = attach_point_info_map
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.base_lora_alpha_encoding = base_lora_alpha_encoding

    def get_default_weight_encoding(self, version: str, tensor_name: str, num_channels: int, quant_info) -> Any:
        """
        Create default per-channel weight encodings for this version.

        Args:
            tensor_name: Name of the tensor
            num_channels: Number of channels for per-channel quantization
            quant_info: Quantization information object with bitwidth, dtype, is_symmetric attributes

        Returns:
            Default weight encoding in version-specific format
        """
        if version == EncodingVersion.ENCODING_VERSION_V_0_6_1:
            scale, offset = DefaultEncoding.get_v1_scale_offset(quant_info)
            per_channel_encoding = {
                "bitwidth": quant_info.bitwidth,
                "dtype": quant_info.dtype.lower(),
                "is_symmetric": str(quant_info.is_symmetric),
                "max": DefaultEncoding.MAX,
                "min": DefaultEncoding.MIN,
                "offset": offset,
                "scale": scale
            }
            default_encoding = [per_channel_encoding for _ in range(num_channels)]
        elif version == EncodingVersion.ENCODING_VERSION_V_1_0_0:
            scale, offset = DefaultEncoding.get_v1_scale_offset(quant_info)
            default_encoding = {
                "bw": quant_info.bitwidth,
                "dtype": quant_info.dtype.upper(),
                "enc_type": EncodingType.PER_CHANNEL.value,
                "is_sym": str(quant_info.is_symmetric),
                "name": tensor_name,
                "offset": [offset] * num_channels,
                "scale": [scale] * num_channels
            }
        elif version == EncodingVersion.ENCODING_VERSION_V_2_0_0:
            scale, zero_point = DefaultEncoding.get_v2_scale_zero_point(quant_info)
            default_encoding = {
                "name": tensor_name,
                "output_dtype": f"{quant_info.dtype.lower()}{quant_info.bitwidth}",
                "y_scale": [scale] * num_channels,
                "y_zero_point": [zero_point] * num_channels
            }
            if num_channels > 1:
                if quant_info.axis is None:
                    raise ValueError(f"{tensor_name} is not per-tensor quantization but axis is None.")
                default_encoding["axis"] = quant_info.axis
        else:
            raise ValueError(f"Unsupported version {version}.")

        return default_encoding

    def get_default_activation_encoding(self, version: str, tensor_name: str, quant_info) -> Any:
        """
        Create default activation encodings for this version.

        Args:
            tensor_name: Name of the tensor
            quant_info: Quantization information object with bitwidth, dtype, is_symmetric attributes

        Returns:
            Default activation encoding in version-specific format
        """
        if version == EncodingVersion.ENCODING_VERSION_V_0_6_1:
            scale, offset = DefaultEncoding.get_v1_scale_offset(quant_info)
            default_encoding = [{
                "bitwidth": quant_info.bitwidth,
                "dtype": quant_info.dtype.lower(),
                "is_symmetric": str(quant_info.is_symmetric),
                "max": DefaultEncoding.MAX,
                "min": DefaultEncoding.MIN,
                "offset": offset,
                "scale": scale
            }]
        elif version == EncodingVersion.ENCODING_VERSION_V_1_0_0:
            scale, offset = DefaultEncoding.get_v1_scale_offset(quant_info)
            default_encoding = {
                "bw": quant_info.bitwidth,
                "dtype": quant_info.dtype.upper(),
                "enc_type": EncodingType.PER_TENSOR.value,
                "is_sym": str(quant_info.is_symmetric),
                "name": tensor_name,
                "offset": [offset],
                "scale": [scale]
            }
        elif version == EncodingVersion.ENCODING_VERSION_V_2_0_0:
            scale, zero_point = DefaultEncoding.get_v2_scale_zero_point(quant_info)
            default_encoding = {
                "name": tensor_name,
                "output_dtype": f"{quant_info.dtype.lower()}{quant_info.bitwidth}",
                "y_scale": scale,  # Single value for per-tensor
                "y_zero_point": zero_point  # Single value for per-tensor
            }
        else:
            raise ValueError(f"Unsupported version {version}.")

        return default_encoding

    def _generate_tensor_mapping(self, concurrency_info):
        """
        Generate mapping of the lora tensor for this concurrency where key is name of the lora tensor in the
        concurrency graph and the value is the name of the tensor in the max concatenated graph.
        :param concurrency_info: A concurrency info object
        :return: A dict mapping original tensor name to its corresponding name in the max concatenated graph.
        """

        tensor_mapping = {}
        concurrency_name = concurrency_info.name
        for attach_point_name in concurrency_info.attach_point_names:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            original_tensor_names = attach_point_info.tensor_names[concurrency_name]
            updated_tensor_names = max_rank_attach_point.lora_tensor_names
            tensor_mapping[original_tensor_names.lora_a_weight_name] = updated_tensor_names.lora_a_weight_name
            tensor_mapping[original_tensor_names.lora_a_act] = updated_tensor_names.lora_a_act
            tensor_mapping[original_tensor_names.mul_scale] = updated_tensor_names.mul_scale
            tensor_mapping[original_tensor_names.mul] = updated_tensor_names.mul
            tensor_mapping[original_tensor_names.lora_b_weight_name] = updated_tensor_names.lora_b_weight_name
            tensor_mapping[original_tensor_names.lora_b_act] = updated_tensor_names.lora_b_act
            tensor_mapping[original_tensor_names.add] = updated_tensor_names.add
            tensor_mapping[original_tensor_names.attach_point_act] = updated_tensor_names.attach_point_act

        return tensor_mapping

    def is_attach_point_tensor(self, tensor_name):
        """
        Return whether the tensor is a base graph tensor or not
        :param tensor_name: name of the tensor
        :return: True if the tensor is a base graph tensor, otherwise false
        """
        # Check whether the tensor contains base_layer substring.
        # In concurrency graph, base_layer is added in the name of the attach point nodes and tensor.
        if tensor_name != create_base_graph_name(tensor_name):
            return True
        return False


    def _update_names(self, encoding_handler, tensor_mapping):
        """
        Update names of the lora branches in the encoding handler based on the provided mapping.
        :param encoding_handler: EncodingHandler instance to update
        :param tensor_mapping: A dictionary mapping tensor name to the tensor name in the max concatenated graph.
        :return: None (updates handler in place)
        """
        # Get all current tensor names to avoid modifying dict during iteration
        current_tensor_names = list(encoding_handler.encodings.keys())

        for tensor_name in current_tensor_names:
            updated_tensor_name = tensor_name

            # Check whether the tensor is a lora branch tensor or attach point tensor.
            if tensor_name in tensor_mapping:
                updated_tensor_name = tensor_mapping[tensor_name]
            elif self.is_attach_point_tensor(tensor_name):
                # Update the name with the name in the base_graph.
                updated_tensor_name = create_base_graph_name(tensor_name)

            if tensor_name != updated_tensor_name:
                # Get the encoding info and update the name
                encoding_info = encoding_handler.get_encoding_info(tensor_name)
                if encoding_info:
                    # Update encoding info with updated name
                    encoding_info.name = updated_tensor_name

                    # Add new encoding
                    encoding_handler.set_encoding(updated_tensor_name, encoding_info)

                    # Remove old encoding
                    encoding_handler.remove_encoding(tensor_name)

    def _update_lora_a_weight_encoding(self, encoding_handler, concurrency_info):
        """
        Update the encoding of the Lora A weight in the encoding handler to
        align it to the max concatenated graph.
        For example, suppose the number of channels in the encoding is m and
        number of channels in the graph is n.
        If these are not equal (m != n), then pad or trim the encoding to make it equal to n channels.
        Note: This function updates the encoding handler in place. It does not return anything.
        :param encoding_handler: EncodingHandler instance to update
        :param concurrency_info: A Concurrency info object.
        :return: None
        """
        for attach_point_name in concurrency_info.attach_point_names:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            updated_tensor_names = max_rank_attach_point.lora_tensor_names
            lora_a_weight = updated_tensor_names.lora_a_weight_name
            max_rank = attach_point_info.max_rank
            quant_info = attach_point_info.lora_quant_info.lora_a_weight

            # Get current encoding
            current_encoding_info = encoding_handler.get_encoding_info(lora_a_weight)
            if current_encoding_info:
                # Update the encoding based on max rank
                updated_scale = current_encoding_info.scale
                updated_zero_point = current_encoding_info.zero_point
                updated_min = current_encoding_info.min
                updated_max = current_encoding_info.max

                # Handle per-channel quantization padding/trimming
                if len(updated_scale) > 1:  # Per-channel
                    if max_rank > len(updated_scale):
                        # Pad with default values
                        encoding_version = encoding_handler.get_version()
                        if encoding_version == EncodingVersion.ENCODING_VERSION_V_2_0_0:
                            scale, zero_point = DefaultEncoding.get_v2_scale_zero_point(quant_info)
                        else:
                            scale, offset = DefaultEncoding.get_v1_scale_offset(quant_info)
                            zero_point = -offset

                        remaining_channels = max_rank - len(updated_scale)
                        updated_scale.extend([scale] * remaining_channels)
                        updated_zero_point.extend([zero_point] * remaining_channels)
                        if updated_min is not None:
                            updated_min.extend([DefaultEncoding.MIN] * remaining_channels)
                        if updated_max is not None:
                            updated_max.extend([DefaultEncoding.MAX] * remaining_channels)
                    elif max_rank < len(updated_scale):
                        # Trim to max_rank
                        updated_scale = updated_scale[:max_rank]
                        updated_zero_point = updated_zero_point[:max_rank]
                        if updated_min is not None:
                            updated_min = updated_min[:max_rank]
                        if updated_max is not None:
                            updated_max = updated_max[:max_rank]

                current_encoding_info.scale = updated_scale
                current_encoding_info.zero_point = updated_zero_point
                current_encoding_info.min = updated_min
                current_encoding_info.max = updated_max

    def _update_encodings_with_default_values(self, max_rank_attach_point, encoding_handler):
        """
        Update the encoding dict with default weight and activation encoding for all
        the lora branch tensors (activation and weights).
        For add node, copy the encoding from the preceding attach point node in the main branch.
        Note: This function updates the encoding in place. It does not return anything.
        :param encodings: Encodings dictionary
        :param max_rank_attach_point: A MaxRankAttachPoint object.
        :param version: encoding version (Encoding_VERSION_V1 or ENCODING_VERSION_V2)
        :return: None.
        """
        # Num of output channels. It is required for per channel quantization
        if max_rank_attach_point.op.op_type == 'Conv':
            num_channels_lora_a = max_rank_attach_point.lora_a_weight_shape[0]
            num_channels_lora_b = max_rank_attach_point.lora_b_weight_shape[0]
        elif max_rank_attach_point.op.op_type == "MatMul":
            num_channels_lora_a = max_rank_attach_point.lora_a_weight_shape[1]
            num_channels_lora_b = max_rank_attach_point.lora_b_weight_shape[1]
        else:
            raise ValueError("Invalid Attach Point {} : "
                             "op_type for attach_point should be Conv or MatMul but got {}".
                             format(max_rank_attach_point.op.name, max_rank_attach_point.op.op_type))

        # The output of the attach point is one of the input of the add node
        attach_point_output_name = max_rank_attach_point.op.output[0]
        lora_tensor_names = max_rank_attach_point.lora_tensor_names
        lora_quant_info = self.attach_point_info_map[max_rank_attach_point.op.name].lora_quant_info
        # Generate and add weight encodings using the encoding handler
        lora_a_weight_name = lora_tensor_names.lora_a_weight_name
        loraA_weight_encoding = self.get_default_weight_encoding(encoding_handler.get_version(), lora_a_weight_name, num_channels_lora_a, lora_quant_info.lora_a_weight)
        loraA_weight_encoding_info = encoding_handler.convert_encoding_to_info(lora_a_weight_name, loraA_weight_encoding, TensorType.PARAM)
        encoding_handler.set_encoding(lora_a_weight_name, loraA_weight_encoding_info)

        lora_b_weight_name = lora_tensor_names.lora_b_weight_name
        loraB_weight_encoding = self.get_default_weight_encoding(encoding_handler.get_version(), lora_b_weight_name, num_channels_lora_b, lora_quant_info.lora_b_weight)
        loraB_weight_encoding_info = encoding_handler.convert_encoding_to_info(lora_b_weight_name, loraB_weight_encoding, TensorType.PARAM)
        encoding_handler.set_encoding(lora_b_weight_name, loraB_weight_encoding_info)

        # Generate and add activation encodings using the encoding handler
        lora_a_act = lora_tensor_names.lora_a_act
        loraA_activation_encoding = self.get_default_activation_encoding(encoding_handler.get_version(), lora_a_act, lora_quant_info.lora_a_act)
        loraA_activation_encoding_info = encoding_handler.convert_encoding_to_info(lora_a_act, loraA_activation_encoding, TensorType.ACTIVATION)
        encoding_handler.set_encoding(lora_a_act, loraA_activation_encoding_info)

        lora_b_act = lora_tensor_names.lora_b_act
        loraB_activation_encoding = self.get_default_activation_encoding(encoding_handler.get_version(), lora_b_act, lora_quant_info.lora_b_act)
        loraB_activation_encoding_info = encoding_handler.convert_encoding_to_info(lora_b_act, loraB_activation_encoding, TensorType.ACTIVATION)
        encoding_handler.set_encoding(lora_b_act, loraB_activation_encoding_info)

        mul_output = lora_tensor_names.mul
        lora_mul_encoding = self.get_default_activation_encoding(encoding_handler.get_version(), mul_output, lora_quant_info.mul)
        lora_mul_encoding_info = encoding_handler.convert_encoding_to_info(mul_output, lora_mul_encoding, TensorType.ACTIVATION)
        encoding_handler.set_encoding(mul_output, lora_mul_encoding_info)

        # Copy the encoding from attach point output to add node output
        if encoding_handler.has_encoding(attach_point_output_name):
            encoding_handler.copy_encoding(attach_point_output_name, lora_tensor_names.add)
        elif encoding_handler.has_encoding(lora_tensor_names.add):
            encoding_handler.copy_encoding(lora_tensor_names.add, attach_point_output_name)

    def _add_lora_alpha_input_encoding(self, encoding_handler, concurrency_info):
        """
        Add encoding of the lora_alpha input vector using handler.
        :param encoding_handler: EncodingHandler instance to update
        :param concurrency_info: A Concurrency Info object.
        :return: None
        """
        lora_alpha = "lora_alpha"
        lora_alpha_encoding_info = None

        if concurrency_info.is_base():
            # For base concurrency, use the `base_lora_alpha_encoding` provided in the constructor
            if self.base_lora_alpha_encoding is not None:
                lora_alpha_encoding_info = copy.deepcopy(self.base_lora_alpha_encoding)
        else:
            # lora_alpha encoding is same as the encoding of mul scale of any lora branch.
            if len(concurrency_info.attach_point_names) >= 1:
                attach_point_name = concurrency_info.attach_point_names[0]
                attach_point_info = self.attach_point_info_map[attach_point_name]
                mul_scale = attach_point_info.tensor_names[concurrency_info.name].mul_scale

                if encoding_handler.has_encoding(mul_scale):
                    lora_alpha_encoding_info = copy.deepcopy(encoding_handler.get_encoding_info(mul_scale))
                    lora_alpha_encoding_info.name = lora_alpha

                    log_debug("Mul Scale encoding of the attach point {} is used as encodings of "
                              "lora_alpha for the use-case {}".format(attach_point_name, concurrency_info.name))

        if lora_alpha_encoding_info is not None:
            encoding_handler.set_encoding(lora_alpha, lora_alpha_encoding_info)
        else:
            log_warning("lora_alpha encoding not found for the use-case {}, lora mul_scale encoding "
                        "was not provided.".format(concurrency_info.name))

    def _add_default_values(self, encoding_handler, concurrency_info):
        """
        Add default encodings for the Lora branches which are not the part of the specified concurrency.
        Note: This function updates the encoding handler in place. It does not return anything.
        :param encoding_handler: EncodingHandler instance to update
        :param concurrency_info: A Concurrency Info object.
        :return: None
        """
        for attach_point_name in self.max_rank_attach_point_map:
            # Only add the default values if the attach-point is not the part of the concurrency
            if attach_point_name not in concurrency_info.attach_point_names:
                max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
                self._update_encodings_with_default_values(max_rank_attach_point, encoding_handler)

    def generate_encodings_for_concurrency(self, encoding_handler, concurrency_info):
        """
        Generate new encoding handler for the given concurrency
        :param encoding_handler: EncodingHandler instance with use case encodings
        :param concurrency_info: A concurrency info object
        :return: EncodingHandler instance with updated encodings for max-rank graph
        """

        # Step 1: Create tensor mapping of the lora branch tensors for the given concurrency
        tensor_mapping = self._generate_tensor_mapping(concurrency_info)

        # Step 2: Add lora_alpha input encoding using handler
        self._add_lora_alpha_input_encoding(encoding_handler, concurrency_info)

        # Step 3: Update lora tensor names in the concurrency encoding using the tensor mapping
        self._update_names(encoding_handler, tensor_mapping)

        # Step 4: Update Lora A weight encoding in the concurrency encoding
        self._update_lora_a_weight_encoding(encoding_handler, concurrency_info)

        # Step 5: Add default encoding for the lora branches which are not the part of the given concurrency
        self._add_default_values(encoding_handler, concurrency_info)

        return encoding_handler


class TensorNamesSerializer(object):
    def __init__(self, max_rank_attach_point_map, quant_updatable_mode):
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.quant_updatable_mode = quant_updatable_mode

    def generate_tensor_names(self):
        def get_indices():
            indices = []
            for attach_point in self.max_rank_attach_point_map.values():
                indices_tensor_name = attach_point.alpha_separation_gather_indices_tensor_name
                indices.append(indices_tensor_name)
            return indices

        def get_weights():
            weights = []
            for attach_point in self.max_rank_attach_point_map.values():
                attach_point_tensors_names = attach_point.lora_tensor_names
                weights.append(attach_point_tensors_names.lora_a_weight_name)
                weights.append(attach_point_tensors_names.lora_b_weight_name)
            return weights

        def get_activations():
            activations = []
            for attach_point in self.max_rank_attach_point_map.values():
                attach_point_tensors_names = attach_point.lora_tensor_names
                activations.append(attach_point_tensors_names.mul)
                activations.append(attach_point_tensors_names.lora_a_act)
                activations.append(attach_point_tensors_names.lora_b_act)
            return activations

        if self.quant_updatable_mode in [QuantUpdatableMode.NONE, QuantUpdatableMode.ALL,
                                         QuantUpdatableMode.FLOAT_ONLY]:
            return get_weights() + get_indices()
        elif self.quant_updatable_mode == QuantUpdatableMode.ADAPTER_ONLY:
            return get_weights() + get_indices() + get_activations()
        else:
            raise RuntimeError("Invalid quant_updatable_mode: {}".format(self.quant_updatable_mode))

    def save_tensor_names(self, path):
        tensor_names = self.generate_tensor_names()
        with open(path, 'w') as tensor_name_file:
            for name in tensor_names:
                tensor_name_file.write(f"{name}\n")
        log_info("Lora tensor name list saved at {}".format(path))


class EncodingSerializer(object):
    def __init__(self,
                 concurrency_infos,
                 attach_point_info_map,
                 max_rank_attach_point_map,
                 quant_updatable_mode,
                 output_dir):
        self.concurrency_infos = concurrency_infos
        self.attach_point_info_map = attach_point_info_map
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.output_dir = output_dir
        self.quant_updatable_mode = quant_updatable_mode
        self._validate_quant_updatable_mode()
        self.concurrency_encodings_file_path = {}
        self.none_quant_encodings = None
        self.is_lora_alpha_updateable = False
        self.base_lora_alpha_encoding = None
        self.__check_alpha_updateable()
        self.encoding_generator = EncodingGenerator(
            self.attach_point_info_map,
            self.max_rank_attach_point_map,
            self.base_lora_alpha_encoding
        )

    def _validate_quant_updatable_mode(self):
        """
        Raise error if encoding serializer is initialized with FLOAT_ONLY mode
        :return: None
        """
        if self.quant_updatable_mode == QuantUpdatableMode.FLOAT_ONLY:
            raise ValueError("EncodingSerializer cannot be initialized with FLOAT_ONLY updatable mode. "
                             "Encoding serialization is not supported for FLOAT_ONLY mode.")

    def __check_alpha_updateable(self, threshold=None):
        """
        Determine if `lora_alpha` is updatable by verifying if the encodings for `mul_scale` for
        the attach-point vary across use cases.
        :return: None
        """

        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                base_concurrency = concurrency_info

        # If the encodings is not present for the base graph then skip this step
        if not base_concurrency.quant_overrides:
            return

        mul_scale_encoding = None
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                continue
            attach_point_name = concurrency_info.attach_point_names[0]
            attach_point_info = self.attach_point_info_map[attach_point_name]
            lora_tensor_names = attach_point_info.tensor_names[concurrency_info.name]
            mul_scale = lora_tensor_names.mul_scale
            encodings = open_and_load_json(concurrency_info.quant_overrides)
            encoding_handler = EncodingHandlerFactory.create_handler(encodings)
            current_mul_scale_encodings = encoding_handler.get_encoding_info(mul_scale)

            if current_mul_scale_encodings is None:
                self.is_lora_alpha_updateable = True
            elif mul_scale_encoding is not None:
                if (not isclose(mul_scale_encoding.scale[0], current_mul_scale_encodings.scale[0], abs_tol=threshold if threshold else 0) or
                    not isclose(mul_scale_encoding.zero_point[0], current_mul_scale_encodings.zero_point[0], abs_tol=threshold if threshold else 0)):
                    self.is_lora_alpha_updateable = True

            mul_scale_encoding = current_mul_scale_encodings
            # Mul scale of any attach point from any use-case can be used as the encoding of the lora_alpha
            if current_mul_scale_encodings is not None and self.base_lora_alpha_encoding is None:
                self.base_lora_alpha_encoding = copy.deepcopy(current_mul_scale_encodings)
                self.base_lora_alpha_encoding.name = "lora_alpha"
                log_debug("Mul Scale encoding of the attach point {} for use case {} is used as "
                          "encodings of lora_alpha for base graph".format(attach_point_name, concurrency_info.name))

        log_debug("Updateable flag is set to {} for lora_alpha input of the base graph".
                  format(self.is_lora_alpha_updateable))

    def get_base_concurrency(self):
        """
        Get the concurrency info for the base use-case.
        :return: A ConcurrencyInfo object.
        """
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                base_concurrency = concurrency_info
        return base_concurrency

    def get_concurrency_info(self, concurrency_name):
        """
        Get the Concurrency info for the given concurrency name
        :param concurrency_name: name of the concurrency
        :return: ConcurrencyInfo object
        """
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.name == concurrency_name:
                return concurrency_info

    def generate_none_mode_encoding(self):
        """
        Generate the encoding dict for none quant mode using EncodingHandler instances.
        :return: Encoding dict
        """

        # None quant encoding is same across all the concurrency. No need to calculate it again.
        if self.none_quant_encodings is not None:
            return self.none_quant_encodings

        base_concurrency = self.get_base_concurrency()

        # Create EncodingHandler for base encodings
        base_encodings_dict = open_and_load_json(base_concurrency.quant_overrides)
        base_handler = EncodingHandlerFactory.create_handler(base_encodings_dict)

        # Create a map where the key is the concurrency name and the value is a list of attach points.
        # For each attach point in the values, extract the encoding from the concurrency encoding and
        # add it to the base encoding.
        concurrency_to_attach_point = {}
        for attach_point_name in self.attach_point_info_map:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            if attach_point_info.max_concurrency not in concurrency_to_attach_point:
                concurrency_to_attach_point[attach_point_info.max_concurrency] = [attach_point_name]
            else:
                concurrency_to_attach_point[attach_point_info.max_concurrency].append(attach_point_name)

        for concurrency_name in concurrency_to_attach_point:
            concurrency_info = self.get_concurrency_info(concurrency_name)

            # Load usecase encodings and create handler
            usecase_encodings_dict = open_and_load_json(concurrency_info.quant_overrides)
            usecase_handler = EncodingHandlerFactory.create_handler(usecase_encodings_dict)

            # Generate updated encodings for this concurrency using handler
            updated_usecase_handler = self.encoding_generator.generate_encodings_for_concurrency(
                usecase_handler,
                concurrency_info
            )

            for attach_point_name in concurrency_to_attach_point[concurrency_name]:
                max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
                lora_tensor_names = max_rank_attach_point.lora_tensor_names

                # Extract and add Lora branch tensor encodings using handlers
                tensor_names = [
                    lora_tensor_names.lora_a_weight_name,
                    lora_tensor_names.lora_b_weight_name,
                    lora_tensor_names.lora_a_act,
                    lora_tensor_names.lora_b_act,
                    lora_tensor_names.mul,
                    lora_tensor_names.add
                ]

                for tensor_name in tensor_names:
                    encoding_info = updated_usecase_handler.get_encoding_info(tensor_name)
                    if encoding_info:
                        base_handler.set_encoding(tensor_name, encoding_info)

            log_debug("Encodings for the following LoRA branches corresponding are extracted from the use-case {} : "
                      "{}".format(concurrency_name, concurrency_to_attach_point[concurrency_name]))

        # Add lora alpha encoding using handler
        if self.base_lora_alpha_encoding is not None:
            base_handler.set_encoding("lora_alpha", self.base_lora_alpha_encoding)

        # Return encodings in native format
        self.none_quant_encodings = base_handler.to_dict()
        return self.none_quant_encodings

    def _add_lora_branch_encodings(self, base_handler, usecase_handler, concurrency_info):
        """
        Copy the lora branch encodings from updated use case handler to the base handler.
        Use case encodings are properly aligned as per the max rank graph and also contain default values.
        :param base_handler: EncodingHandler instance containing base graph encodings
        :param usecase_handler: EncodingHandler instance containing updated use-case encodings
        :param concurrency_info: A ConcurrencyInfo object
        :return: EncodingHandler instance containing final encodings for this concurrency
        """
        for attach_point_name in self.attach_point_info_map:
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            lora_tensor_names = max_rank_attach_point.lora_tensor_names

            # Add the encoding of Lora branch tensors from the use-case handler to the base handler
            tensor_names = [
                lora_tensor_names.lora_a_weight_name,
                lora_tensor_names.lora_b_weight_name,
                lora_tensor_names.lora_a_act,
                lora_tensor_names.lora_b_act,
                lora_tensor_names.mul,
                lora_tensor_names.add
            ]

            for tensor_name in tensor_names:
                encoding_info = usecase_handler.get_encoding_info(tensor_name)
                if encoding_info:
                    base_handler.set_encoding(tensor_name, encoding_info)

            # For default case, use the attach point output encoding from the base handler for lora-add.
            attach_point_output_name = max_rank_attach_point.op.output[0]
            if attach_point_name not in concurrency_info.attach_point_names:
                if base_handler.has_encoding(attach_point_output_name):
                    base_handler.copy_encoding(attach_point_output_name, lora_tensor_names.add)

            # If the encoding does not exist for attach point output but exists for add output,
            # then copy the encoding from the add output to the attach point output.
            # This scenario can occur when the attach point is the output node of the graph.
            add_encoding = base_handler.get_encoding_info(lora_tensor_names.add)
            attach_point_encoding = base_handler.get_encoding_info(attach_point_output_name)
            if add_encoding and not attach_point_encoding:
                base_handler.copy_encoding(lora_tensor_names.add, attach_point_output_name)

        # Add lora alpha encoding
        lora_alpha_encoding = usecase_handler.get_encoding_info("lora_alpha")
        if lora_alpha_encoding:
            base_handler.set_encoding("lora_alpha", lora_alpha_encoding)

        return base_handler

    def generate_encoding_for_all_mode(self, concurrency_info):
        """
        Generate Encodings for the specified Concurrency based for ALL Updateable mode.
        :param concurrency_info: A concurrency info object
        :return: Encoding dict for this use-case.
        """
        # load usecase encodings and create handler
        f = open(concurrency_info.quant_overrides)
        usecase_encodings = json.load(f)
        usecase_handler = EncodingHandlerFactory.create_handler(usecase_encodings)

        # generate new encodings for this concurrency using handler
        updated_handler = self.encoding_generator.generate_encodings_for_concurrency(
            usecase_handler,
            concurrency_info
        )
        return updated_handler.to_dict()

    def generate_encoding_for_adapter_only_mode(self, concurrency_info):
        """
        Generate Encodings for the specified Concurrency based for Adapter Only mode.
        :param concurrency_info: A concurrency info object
        :return: Encoding dict for this use-case.
        """
        base_concurrency = self.get_base_concurrency()
        # load usecase encodings and create handler
        validate_file_path(concurrency_info.quant_overrides)
        f = open(concurrency_info.quant_overrides)
        usecase_encodings = json.load(f)
        usecase_handler = EncodingHandlerFactory.create_handler(usecase_encodings)

        # generate new encodings for this concurrency using handler
        updated_handler = self.encoding_generator.generate_encodings_for_concurrency(
            usecase_handler,
            concurrency_info
        )

        if concurrency_info.is_base():
            return updated_handler.to_dict()

        # Take the base encoding and add the LoRa branch tensors from the
        # use-case encodings. This ensures that the encodings for the base graph tensors remain
        # consistent across all use cases.
        # This will be the final encoding for this concurrency.
        f = open(base_concurrency.quant_overrides)
        base_encodings_dict = json.load(f)
        base_handler = EncodingHandlerFactory.create_handler(base_encodings_dict)

        # Add lora branch encodings using handlers
        final_handler = self._add_lora_branch_encodings(base_handler, updated_handler, concurrency_info)

        return final_handler.to_dict()

    def generate_encoding_for_concurrency(self, concurrency_info):
        """
        Generate Encodings for the specified Concurrency based on the mode.
        :param concurrency_info: A concurrency info object
        :return: Encoding dict for this use-case.
        """
        if self.quant_updatable_mode == QuantUpdatableMode.NONE:
            return self.generate_none_mode_encoding()
        elif self.quant_updatable_mode == QuantUpdatableMode.ALL:
            return self.generate_encoding_for_all_mode(concurrency_info)
        elif self.quant_updatable_mode == QuantUpdatableMode.ADAPTER_ONLY:
            return self.generate_encoding_for_adapter_only_mode(concurrency_info)
        else:
            raise ValueError("Unsupported Quant Updateable Mode: {}".format(self.quant_updatable_mode))

    def serialize(self):
        """
        This method generates and serializes the encoding for the mode specified during instantiation.
        :return: None
        """
        def save_encodings(encodings, concurrency_info):
            if concurrency_info.is_base():
                path = os.path.join(self.output_dir, "base_encodings.json".format(concurrency_info.name))
            else:
                path = os.path.join(self.output_dir, "{}_encodings.json".format(concurrency_info.name))

            with open(path, 'w') as json_file:
                json.dump(encodings, json_file, indent=4)
            self.concurrency_encodings_file_path[concurrency_info.name] = path
            log_info("Encoding file for {} concurrency saved at {}".format(concurrency_info.name, path))

        log_info("Generating Encodings for the {} quant-updateable mode".format(self.quant_updatable_mode.value))

        base_concurrency = self.get_base_concurrency()
        # If the encodings is not present for the base graph then skip the encoding generation step.
        if not base_concurrency.quant_overrides:
            log_info("Quantization overrides have not been provided for the base graph. "
                     "Consequently, encodings will not be generated for any use cases")
            return None

        for concurrency_info in self.concurrency_infos:
            encodings = self.generate_encoding_for_concurrency(concurrency_info)
            save_encodings(encodings, concurrency_info)


class TransformMetadataSerializer(object):
    def __init__(self, concurrency_infos, attach_point_info_map, max_rank_attach_point_map, indices_map, transforms_metadata):
        self.concurrency_infos = concurrency_infos
        self.attach_point_info_map = attach_point_info_map
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.indices_map = indices_map
        self.transforms_metadata = transforms_metadata
        self.order_suffix = "_stage0"
        self.transform_manager = TransformManager()

    def validate_input_metadata(self):
        concurrency_ct = 0
        for concurrency in self.concurrency_infos:
            if concurrency.name != "base":
                concurrency_ct += 1

        graph_ids = self.transform_manager.get_graph_ids()
        if len(graph_ids) != concurrency_ct:
            raise ValueError(
                f"Expected one transform graph per concurrency, Found: ({len(graph_ids)}), Expected: ({len(self.concurrency_infos)})"
            )
        return True

    def save_metadata(self, output_path):
        input_metadata_flag = False
        if self.transforms_metadata:
            # check if transforms metadata is a file path
            if os.path.isfile(self.transforms_metadata):
                input_metadata_flag = True
                self.transform_manager.load_metadata(self.transforms_metadata)
                self.validate_input_metadata()
                for graph_name in self.transform_manager.get_graph_ids():
                    self.transform_manager.remove_graph(graph_name)

        self._add_transforms()
        if input_metadata_flag:
            self._add_existing_transforms()
        self.transform_manager.save_metadata(output_path)

    def _add_existing_transforms(self):
        input_metadata_manager = TransformManager()
        input_metadata_manager.load_metadata(self.transforms_metadata)

        graph_concurrency_map = dict()
        for uc_id in self.transform_manager.get_graph_ids():
            uc_graph = self.transform_manager.get_transform_graph(uc_id)
            graph_concurrency_map[uc_graph.concurrency_name] = uc_id

        input_graph_ids = input_metadata_manager.get_graph_ids()
        for input_graph_id in input_graph_ids:
            input_graph = input_metadata_manager.get_transform_graph(input_graph_id)
            curr_graph_id = graph_concurrency_map[input_graph.concurrency_name]
            curr_graph = self.transform_manager.get_transform_graph(curr_graph_id)
            if input_graph.output_tensors != curr_graph.input_tensors:
                mismatched_tensors = input_graph.output_tensors.symmetric_difference(curr_graph.input_tensors)
                raise ValueError(f"Transforms metadata output tensors for graph '{input_graph_id}' do not match with creator graph '{curr_graph_id}' input tensors. "
                                 "Input artifacts may be incorrect or mismatched. "
                                 "Confirm the correct file is passed to --transforms_metadata, "
                                 "target_modules in each adapter config file capture all the attach points for that adapter, "
                                 "and the correct adapter configs and onnx concurrency models are used together.\n"
                                 f"Mismatched tensors: {mismatched_tensors}")

            for transform in input_graph.transforms:
                self.transform_manager.add_transform(
                    src_tensors=transform.src_tensors,
                    dest_tensors=transform.dest_tensors,
                    operator=transform.operator,
                    tensor_dtypes=transform.tensor_dtypes,
                    graph_id=curr_graph_id
                )

    def _add_transforms(self):
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.name == "base":
                continue
            self.transform_manager.create_transform_graph(concurrency_info.name + self.order_suffix, order_num=0, concurrency_name=concurrency_info.name)
            self._add_usecase_transforms(concurrency_info)

    def _add_usecase_transforms(self, concurrency_info):
        usecase_name = concurrency_info.name

        for attach_point_name, attach_point_info in self.attach_point_info_map.items():
            self._add_lora_weight_transforms(attach_point_name, concurrency_info)
            self._add_indices(attach_point_name, concurrency_info)


    def _add_indices(self, attach_point_name, concurrency_info):
        indices_tensor = self.indices_map[concurrency_info.name].attach_pt_indices[attach_point_name].alpha_indices
        indices_tensor = np.array(indices_tensor)
        indices_tensor_name = self.max_rank_attach_point_map[attach_point_name].alpha_separation_gather_indices_tensor_name
        tensor_dtypes_info = {indices_tensor_name: indices_tensor.dtype.name}
        self.transform_manager.add_transform(
            src_tensors=[],
            dest_tensors=[f"{indices_tensor_name}"],
            graph_id=concurrency_info.name + self.order_suffix,
            tensor_dtypes=tensor_dtypes_info,
            operator=Operator(
                op_type="Constant",
                attributes={
                    "value_ints":indices_tensor.tolist()[0]
                },
                input_shapes=[],  # Original shape
                output_shapes=[list(indices_tensor.shape)]      # Flattened shape
            )
        )

    def _add_lora_weight_transforms(self, attach_point_name, concurrency_info):
        usecase = concurrency_info.name

        if attach_point_name in concurrency_info.attach_point_names:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            attach_point_adapters = [adapter for adapter in attach_point_info.adapter_names if adapter in concurrency_info.adapter_infos]

            # Initialize the member variable
            self.current_transforms = {"a_transforms": [], "b_transforms": []}

            # Each function updates self.current_transforms
            self._populate_adapter_tensor_info(usecase, attach_point_adapters, attach_point_info)
            self._add_concat_transforms(usecase, attach_point_info)
            self._add_pad_transforms(usecase, attach_point_info)
            self._rename_output_tensors(usecase, attach_point_info)

        else:
            self._add_zero_tensor(usecase, attach_point_name)

    def _add_zero_tensor(self, usecase, attach_point_name):
        def add_transform(output_name, output_shape):
            num_of_zero = np.prod(output_shape)
            const_transform = self.transform_manager.add_transform(
                graph_id=usecase + self.order_suffix,
                src_tensors=[],
                dest_tensors=[output_name],
                operator=Operator(
                    op_type="Constant",
                    attributes={
                        "value_floats": [0]*num_of_zero
                    },
                    input_shapes=[],
                    output_shapes=[output_shape]
                )
            )
            return const_transform

        max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]

        add_transform(
            output_name=max_rank_attach_point.lora_tensor_names.lora_a_weight_name,
            output_shape=max_rank_attach_point.lora_a_weight_shape
        )

        add_transform(
            output_name=max_rank_attach_point.lora_tensor_names.lora_b_weight_name,
            output_shape=max_rank_attach_point.lora_b_weight_shape
        )

    def _rename_output_tensors(self, usecase, attach_point_info):
        def add_identity_transform(transform_info, output_tensor_name):
            """
            Create an identity transform with the given input and output tensor names.
            """
            return self.transform_manager.add_transform(
                graph_id=usecase + self.order_suffix,
                src_tensors=[transform_info["name"]],
                dest_tensors=[output_tensor_name],
                operator=Operator(
                    op_type="Identity",
                    attributes={},
                    input_shapes=[transform_info["shape"]],
                    output_shapes=[transform_info["shape"]]
                )
            )

        # Read from member variable
        a_transform_info = self.current_transforms["a_transforms"][0]
        b_transform_info = self.current_transforms["b_transforms"][0]

        max_rank_attach_point = self.max_rank_attach_point_map[attach_point_info.name]

        # Get target names
        new_a_tensor_name = max_rank_attach_point.lora_tensor_names.lora_a_weight_name
        new_b_tensor_name = max_rank_attach_point.lora_tensor_names.lora_b_weight_name

        graph_id = usecase + self.order_suffix

        # Handle A tensor
        old_a_tensor_name = a_transform_info["name"]
        if old_a_tensor_name in self.input_tensor_names["a_transforms"]:
            # Each transform subgraph should output the onnx max rank name.
            # If a tensor has no transforms, an identity is needed to rename its output.
            add_identity_transform(a_transform_info, new_a_tensor_name)
            # Update current_transforms
            self.current_transforms["a_transforms"] = [{"name": new_a_tensor_name, "shape": a_transform_info["shape"]}]
        else:
            # Use rename_tensor method for derived tensor
            self.transform_manager.rename_tensor(graph_id, old_a_tensor_name, new_a_tensor_name)

        # Handle B tensor
        old_b_tensor_name = b_transform_info["name"]
        if old_b_tensor_name in self.input_tensor_names["b_transforms"]:
            # Each transform subgraph should output the onnx max rank name.
            # If a tensor has no transforms, an identity is needed to rename its output.
            add_identity_transform(b_transform_info, new_b_tensor_name)
            # Update current_transforms
            self.current_transforms["b_transforms"] = [{"name": new_b_tensor_name, "shape": b_transform_info["shape"]}]
        else:
            # Use rename_tensor method for derived tensor
            self.transform_manager.rename_tensor(graph_id, old_b_tensor_name, new_b_tensor_name)

    def _add_pad_transforms(self, usecase, attach_point_info):
        def add_transform(transform_info, rank_axis):
            def add_pad_constant(pad_value):
                transform_graph = self.transform_manager.get_transform_graph(usecase + self.order_suffix)
                pad_name = "pad_" + "_".join([str(val) for val in pad_value])
                pad_shape = [1, len(pad_value)]
                if pad_name not in transform_graph.produced_tensors:
                    self.transform_manager.add_transform(
                        graph_id=usecase + self.order_suffix,
                        src_tensors=[],
                        dest_tensors=[pad_name],
                        operator=Operator(
                            op_type="Constant",
                            attributes={
                                "value_ints":pad_value
                            },
                            input_shapes=[],
                            output_shapes=[pad_shape]
                        )
                    )

                return pad_name, pad_shape

            tensor_name = transform_info['name']
            input_shape = transform_info['shape']
            usecase_rank = input_shape[rank_axis]
            max_rank = attach_point_info.max_rank
            output_shape = list(input_shape)
            output_shape[rank_axis] = max_rank
            pad_amount = max_rank - usecase_rank

            src_name = tensor_name
            dest_name = src_name + "_padded"

            num_axes = len(input_shape)
            pads = [0 for _ in range(num_axes * 2)]
            pads[num_axes + rank_axis] = pad_amount
            pad_name, pad_shape = add_pad_constant(pad_value=pads)

            pad_dtype = np.dtype(type(pads[0])).name
            tensor_dtypes_info = {
                pad_name: pad_dtype
            }

            pad_transform = self.transform_manager.add_transform(
                graph_id=usecase + self.order_suffix,
                src_tensors=[src_name, pad_name],
                tensor_dtypes=tensor_dtypes_info,
                dest_tensors=[dest_name],
                operator=Operator(
                    op_type="Pad",
                    input_shapes=[input_shape, pad_shape],
                    output_shapes=[output_shape]
                )
            )

            return pad_transform

        a_transform_info = self.current_transforms["a_transforms"][0]
        a_rank_axis = next(iter(attach_point_info.weight_info.values())).lora_weightA_rank_axis

        usecase_rank = a_transform_info['shape'][a_rank_axis]
        max_rank = attach_point_info.max_rank
        if usecase_rank == max_rank:
            return

        b_transform_info = self.current_transforms["b_transforms"][0]
        b_rank_axis = next(iter(attach_point_info.weight_info.values())).lora_weightB_rank_axis

        a_pad_transform = add_transform(
            transform_info=a_transform_info,
            rank_axis=a_rank_axis
        )

        b_pad_transform = add_transform(
            transform_info=b_transform_info,
            rank_axis=b_rank_axis
        )

        # Update member variable
        self.current_transforms["a_transforms"] = [{
            "name": a_pad_transform.dest_tensors[0],
            "shape": a_pad_transform.operator.output_shapes[0],
        }]

        self.current_transforms["b_transforms"] = [{
            "name": b_pad_transform.dest_tensors[0],
            "shape": b_pad_transform.operator.output_shapes[0],
        }]


    def _add_concat_transforms(self, usecase, attach_point_info):
        def add_transform(transform_infos, dest_tensors, concat_axis):
            src_tensors = [info["name"] for info in transform_infos]
            input_shapes = [info["shape"] for info in transform_infos]

            output_shape = list(input_shapes[0])
            usecase_rank = sum([shape[concat_axis] for shape in input_shapes])
            output_shape[concat_axis] = usecase_rank

            concat_transform = self.transform_manager.add_transform(
                graph_id=usecase + self.order_suffix,
                src_tensors=src_tensors,
                dest_tensors=dest_tensors,
                operator=Operator(
                    op_type="Concat",
                    attributes={"axis": concat_axis},
                    input_shapes=input_shapes,
                    output_shapes=[output_shape]
                )
            )

            return concat_transform

        # Read from member variable
        weight_A_transform_infos = self.current_transforms["a_transforms"]
        weight_B_transform_infos = self.current_transforms["b_transforms"]

        if len(weight_A_transform_infos) == 1:
            # single adapter attach point so nothing to concatenate
            return

        ap_lora_tensor_names = next(iter(attach_point_info.tensor_names.values()))

        dest_tensors = [ap_lora_tensor_names.lora_a_weight_name]
        concat_axis = next(iter(attach_point_info.weight_info.values())).lora_weightA_rank_axis
        weight_A_concat_transform = add_transform(weight_A_transform_infos, dest_tensors, concat_axis)

        dest_tensors = [ap_lora_tensor_names.lora_b_weight_name]
        concat_axis = next(iter(attach_point_info.weight_info.values())).lora_weightB_rank_axis
        weight_B_concat_transform = add_transform(weight_B_transform_infos, dest_tensors, concat_axis)

        # Update member variable with new transform info
        self.current_transforms["a_transforms"] = [{"name": weight_A_concat_transform.dest_tensors[0], "shape": weight_A_concat_transform.operator.output_shapes[0]}]
        self.current_transforms["b_transforms"] = [{"name": weight_B_concat_transform.dest_tensors[0], "shape": weight_B_concat_transform.operator.output_shapes[0]}]

    def _populate_adapter_tensor_info(self, usecase, adapters, attach_point_info):
        # Initialize tracking of input tensor names
        self.input_tensor_names = {"a_transforms": [], "b_transforms": []}

        weight_A_transform_info = []
        weight_B_transform_info = []

        for adapter in adapters:
            # A weight tensor info
            original_a_name = attach_point_info.tensor_names[usecase].lora_a_weight_name
            input_a_name = f"{original_a_name}_{adapter}"
            a_shape = attach_point_info.weight_info[adapter].lora_weightA_shape
            weight_A_transform_info.append({"name": input_a_name, "shape": a_shape})

            # Track the input tensor names
            self.input_tensor_names["a_transforms"].append(input_a_name)

            # B weight tensor info
            original_b_name = attach_point_info.tensor_names[usecase].lora_b_weight_name
            input_b_name = f"{original_b_name}_{adapter}"
            b_shape = attach_point_info.weight_info[adapter].lora_weightB_shape
            weight_B_transform_info.append({"name": input_b_name, "shape": b_shape})

            # Track the input tensor names
            self.input_tensor_names["b_transforms"].append(input_b_name)

        # Update the member variable with tensor info
        self.current_transforms["a_transforms"] = weight_A_transform_info
        self.current_transforms["b_transforms"] = weight_B_transform_info


class LoraSerializer(object):
    def __init__(
            self,
            concurrency_infos,
            safe_tensor_path,
            indices_map,
            attach_point_info_map,
            max_rank_attach_point_map,
            output_dir,
            max_graph_path,
            quant_updatable_mode,
            transforms_metadata,
            dump_onnx=False
    ):
        self.concurrency_infos = concurrency_infos
        self.safe_tensor_path = safe_tensor_path
        self.indices_map = indices_map
        self.attach_point_info_map = attach_point_info_map
        self.max_rank_attach_point_map = max_rank_attach_point_map
        self.output_dir = output_dir
        self.concurrency_weight_file_path = {}
        self.concurrency_encodings_file_path = {}
        self.max_graph_path = max_graph_path
        self.quant_updatable_mode = quant_updatable_mode
        if self.quant_updatable_mode != QuantUpdatableMode.FLOAT_ONLY:
            self.encoding_serializer = EncodingSerializer(
                self.concurrency_infos,
                self.attach_point_info_map,
                self.max_rank_attach_point_map,
                quant_updatable_mode,
                self.output_dir,
            )
        else:
            self.encoding_serializer = None
        self.dump_onnx = dump_onnx
        self.transforms_metadata = transforms_metadata


    @staticmethod
    def __concatenate_weights(weight_list, axis, max_rank):
        """
        Concatenates a list of n-dimensional array along the specified axis and
        adds padding to the concatenated vector till specified max rank
        :param weight_list: a list of n-d arrays
        :param axis: axis along with array will be concatenated
        :param max_rank: the required rank for the given axis
        :return: A concatenated array
        """
        # concatenate all the weights along the axis
        concatenated_weight = np.concatenate(weight_list, axis=axis)

        # calculate the required padding
        padding_needed = max_rank - concatenated_weight.shape[axis]
        if padding_needed > 0:
            padding_shape = list(concatenated_weight.shape)
            padding_shape[axis] = padding_needed
            padding = np.full(padding_shape, 0, dtype=np.float32)
            # concatenate the padding tensor to the weights
            padded_concatenated_weight = np.concatenate([concatenated_weight, padding], axis=axis)
        else:
            padded_concatenated_weight = concatenated_weight

        return padded_concatenated_weight

    def is_lora_alpha_updateable(self):
        # Return false for FLOAT_ONLY mode where encoding_serializer is None
        if self.encoding_serializer is None:
            return False
        return self.encoding_serializer.is_lora_alpha_updateable

    def generate_safetensors_for_concurrency(self, concurrency_info, adapter_weights):
        """
        Generate concatenated weights for max rank graph for the given concurrency(or use case)
        and save it in the safetensor format.
        Along with lora A and lora B node weights, it also includes the indices tensor required for this concurrency
        :param concurrency_info: ConcurrencyInfo object
        :param adapter_weights: A dict containing all the adapter weights.
        :return: A dict containing all the weights for this concurrency
        """
        concurrency_safetensors = {}
        for attach_point_name in self.attach_point_info_map:
            attach_point_info = self.attach_point_info_map[attach_point_name]
            max_rank_attach_point = self.max_rank_attach_point_map[attach_point_name]
            # Initialize lora A and lora B weight with default values
            lora_a_weight = np.zeros(max_rank_attach_point.lora_a_weight_shape, dtype=np.float32)
            lora_b_weight = np.zeros(max_rank_attach_point.lora_b_weight_shape, dtype=np.float32)
            # update lora A and lora B weight if this attach point exists in this concurrency
            if attach_point_name in concurrency_info.attach_point_names:
                lora_a_weights = []
                lora_b_weights = []
                # Extract all the adapter weights for this attach point required for this concurrency
                for adapter_name in concurrency_info.adapter_names:
                    if adapter_name in attach_point_info.adapter_names:
                        lora_a_name = attach_point_info.weight_info[adapter_name].lora_weightA_name
                        lora_b_name = attach_point_info.weight_info[adapter_name].lora_weightB_name
                        lora_a_weights.append(adapter_weights[lora_a_name])
                        lora_b_weights.append(adapter_weights[lora_b_name])

                # Max rank for this attach point
                max_rank = attach_point_info.max_rank

                # Concatenate all the extracted weights along the required dimension.
                # ------------------------------------------------------------------------------|
                # |   op-type    |   lora A concatenated axis   |    lora B Concatenated axis   |
                # |--------------|--------------------------------------------------------------|
                # |    Conv      |           0                  |              1                |
                # |    MatMul    |           1                  |              0                |
                # -------------------------------------------------------------------------------

                if attach_point_info.op_type == "Conv":
                    lora_a_weight = self.__concatenate_weights(lora_a_weights, 0, max_rank)
                    lora_b_weight = self.__concatenate_weights(lora_b_weights, 1, max_rank)
                elif attach_point_info.op_type == "MatMul":
                    lora_a_weight = self.__concatenate_weights(lora_a_weights, 1, max_rank)
                    lora_b_weight = self.__concatenate_weights(lora_b_weights, 0, max_rank)

            # Extract the new name of Lora A and B node for this attach point in the max graph.
            lora_a_weight_name = max_rank_attach_point.lora_tensor_names.lora_a_weight_name
            lora_b_weight_name = max_rank_attach_point.lora_tensor_names.lora_b_weight_name

            concurrency_safetensors[lora_a_weight_name] = lora_a_weight
            concurrency_safetensors[lora_b_weight_name] = lora_b_weight

            # Get the indices tensor for this attach point
            indices_tensor = self.indices_map[concurrency_info.name].attach_pt_indices[attach_point_name].alpha_indices
            indices_tensor = np.array(indices_tensor)
            indices_tensor_name = max_rank_attach_point.alpha_separation_gather_indices_tensor_name
            concurrency_safetensors[indices_tensor_name] = indices_tensor

        return concurrency_safetensors

    def generate_and_save_safetensors(self):
        """
        Function to generate safe tensor for each concurrency.
        return: None
        """
        adapter_weights = load_file(self.safe_tensor_path)
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                continue
            concurrency_safetensors = self.generate_safetensors_for_concurrency(concurrency_info, adapter_weights)
            # save the weights in the safe tensor file
            path = os.path.join(self.output_dir, "{}.safetensors".format(concurrency_info.name))
            save_file(concurrency_safetensors, path)
            self.concurrency_weight_file_path[concurrency_info.name] = path
            log_info("Safetensors file for {} concurrency saved at {}".format(concurrency_info.name, path))

    def dump_patched_onnx_model(self):
        """
        Dump the patched Onnx file for all the concurrences.
        :return: None
        """
        for concurrency_info in self.concurrency_infos:
            if concurrency_info.is_base():
                continue
            # Load the ONNX model
            model = onnx.load(self.max_graph_path, load_external_data=True)
            concurrency_name = concurrency_info.name
            # load the safetensors file for this concurrency
            safetensor_file_path = self.concurrency_weight_file_path[concurrency_name]
            safetensors_dict = load_file(safetensor_file_path)

            # Apply the safetensors file and generate the patched Onnx model
            patched_model = apply_safetensors_to_onnx(model, safetensors_dict)

            # Save the patched Onnx model
            path = os.path.join(self.output_dir, "{}.onnx".format(concurrency_name))
            onnx.save(patched_model, path, save_as_external_data=True,
                      all_tensors_to_one_file=True, location=concurrency_name+".data")
            log_info("Patched ONNX model for use-case {} saved at {}".format(concurrency_name, path))

    def generate_and_save_importer_config(self):
        """
        Generate lora config for qairt-lora-importer tool. It will contain the information about each use case.
        :return: None
        """
        importer_config_data = {"use_case": []}
        for concurrency_name in self.concurrency_weight_file_path:
            use_case_dict = dict()
            use_case_dict['model_name'] = self.max_graph_path
            use_case_dict['name'] = concurrency_name
            use_case_dict['lora_weights'] = self.concurrency_weight_file_path[concurrency_name]
            if concurrency_name in self.concurrency_encodings_file_path:
                use_case_dict['quant_overrides'] = self.concurrency_encodings_file_path[concurrency_name]
            importer_config_data["use_case"].append(use_case_dict)

        save_path = os.path.join(self.output_dir, "lora_importer_config.yaml")
        log_info("Importer Config saved at {}".format(save_path))

        with open(save_path, "w") as f:
            yaml.dump(importer_config_data, f)

    def serialize(self):
        """
        Generate the following output files of the qairt creator tool.
          - Safetensors file and encodings file per concurrency
          - Encodings file and encodings file per concurrency
          - A txt file containing lora tensor names
          - Lora Importer YAML config file
        :return: None
        """
        # generate concatenated weight per concurrency and save in safetensor file
        log_debug("Generating safetensors file and encoding file for each concurrency...")
        self.generate_and_save_safetensors()
        log_debug("Successfully generated safetensors file for each concurrency.")

        if self.dump_onnx:
            # Dump the patched ONNX model
            self.dump_patched_onnx_model()

        # Only generate encodings if quant updateable mode not in float only
        if self.encoding_serializer is not None:
            # Generate encodings per concurrency and dump the encodings in a json file.
            # This steps also generates encoding for the base model.
            self.encoding_serializer.serialize()
            self.concurrency_encodings_file_path = self.encoding_serializer.concurrency_encodings_file_path
            log_debug("Successfully generated encoding file for each concurrency.")
        else:
            log_info("Skipping encoding generation for FLOAT_ONLY mode")

        # Generate and save all the updateable tensor in the max rank graph
        log_debug("Generating lora tensor names list...")
        tensor_names_serializer = TensorNamesSerializer(self.max_rank_attach_point_map, self.quant_updatable_mode)
        tensor_names_serializer.save_tensor_names(path=os.path.join(self.output_dir, "lora_tensor_names.txt"))
        log_debug("Successfully generated lora tensor names list.")

        # Create Lora importer yaml config
        log_debug("Generating yaml config for lora importer tool...")
        self.generate_and_save_importer_config()
        log_debug("Successfully generated yaml config for lora importer tool.")

        # Generate transform metadata
        if self.transforms_metadata:

            log_debug("Generating lora creator metadata...")
            transform_metadata_serializer = TransformMetadataSerializer(
                self.concurrency_infos,
                self.attach_point_info_map,
                self.max_rank_attach_point_map,
                self.indices_map,
                self.transforms_metadata
            )
            transform_metadata_serializer.save_metadata(output_path=os.path.join(self.output_dir, "transforms_metadata.json"))
            log_debug("Successfully saved transform metadata.")
