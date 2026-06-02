# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

""" File contains the logic to convert the Weighs and the encodings of a MHA model to its SHA equivalent """

import os
import pickle
import json
from string import Template
import copy
import logging

from qti.aisw.graph_gen.architectures import AutoConfig, AutoModel
from qti.aisw.graph_gen.mad.utils import load_encodings, save_encodings
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MHAtoSHAConverter:
    """
    A class used to convert MHA (Multi-Head Attention) weights and encodings into SHA (Split-Head Attention) format.

    The conversion process involves splitting the weights and encodings into multiple sub-weights and sub-encodings,
    each corresponding to a different attention head. This is done by splitting the last axis of the weights and encodings
    into equal chunks, where the number of chunks is determined by the number of attention heads.

    # Example usage:
    converter = MHAtoSHAConverter(num_attention_heads=32, num_key_value_heads=8)

    # From file
    converter.split_weights('sample_run/mha_weights.pkl', 'sample_run/weights_sha.pkl')
    converter.split_encodings('sample_run/gguf_data.encodings', 'sample_run/gguf_data_sha.encodings')


    # In memory data
    data = pickle.load(open('sample_run/mha_weights.pkl', 'rb'))
    updated_weights = converter.split_data(data)
    data = json.load(open('sample_run/gguf_data.encodings', 'r'))['param_encodings']
    updated_encodings = converter.split_encoding_data(data)

    """

    def __init__(self, num_attention_heads: int, num_key_value_heads: int):
        """
        Initializes the MHAtoSHAConverter with the specified number of attention and key-value heads.

        :param num_attention_heads: The total number of attention heads.
        :param num_key_value_heads: The number of key-value heads.
        """
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.parameter_mha_to_sha_maps = {
            "k_proj": (Template("k_heads.$num.k_proj"), num_key_value_heads),
            "v_proj": (Template("v_heads.$num"), num_key_value_heads),
            "q_proj": (Template("q_heads.$num.q_proj"), num_attention_heads),
        }
        self.activation_mha_to_sha_maps = None

    def split_weights(self, mha_weight_file: str, sha_weight_file: str):
        """
        Splits the weights from an MHA weight file into SHA format and saves them to a new file.

        :param mha_weight_file: The path to the input MHA weight file (pickle format).
        :param sha_weight_file: The path where the converted SHA weights will be saved.
        :raises FileNotFoundError: If the MHA weight file does not exist.
        """
        if not os.path.exists(mha_weight_file):
            raise FileNotFoundError(f"MHA weight file not found: {mha_weight_file}")

        with open(mha_weight_file, 'rb') as f:
            weight_dict = pickle.load(f)

        updated_weight_dict = self.split_data(weight_dict)
        with open(sha_weight_file, 'wb') as f:
            pickle.dump(updated_weight_dict, f)

    def split_encodings(self, encoding_file:str, output_file:str):
        """
        Splits the encodings from an MHA encoding file into SHA format and saves them to a new file.

        :param encoding_file: The path to the input MHA encoding file (JSON format).
        :param output_file: The path where the converted SHA encodings will be saved.
        """

        if not os.path.exists(encoding_file):
            raise FileNotFoundError(f"MHA encoding file not found: {encoding_file}")

        with open(encoding_file, 'r') as f:
            encoding_dict = json.load(f)
            if 'param_encodings' not in encoding_dict:
                raise KeyError("Invalid Encoding File:: Does not contains `param_encodings` ")
            param_encoding = encoding_dict['param_encodings']

        updated_encoding_info = self.split_encoding_data(param_encoding)
        with open(output_file, 'w') as f:
            encoding_dict['param_encodings'] = updated_encoding_info
            json.dump(encoding_dict, f)

    def split_data(self, weight_dict):
        """
        Splits the given weights from MHA to SHA format.

        This method iterates through the weights dictionary. If a key corresponds to
        an attention projection ('k_proj', 'v_proj', 'q_proj'), it splits the
        associated value along its last dimension into chunks corresponding to
        individual attention heads.

        :param weight_dict: A dictionary containing model weights.
        :return: A new dictionary with the weights split into SHA format.
        :raises ValueError: If dimensions of weight are not divisible by the split count.
        """
        updated_data = {}
        for key, value in tqdm(weight_dict.items(), "Splitting Parameter Tensors"):
            key_splits = key.split('.')

            #
            if len(key_splits) > 2 and key_splits[-2] in self.parameter_mha_to_sha_maps.keys():
                template, split_cnt = self.parameter_mha_to_sha_maps[key_splits[-2]]

                # Check if the data can be split along the last axis equally or not
                if not hasattr(value, 'shape'):
                    raise ValueError(f"Value for key {key} does not have a shape attribute")
                if value.shape[-1] % split_cnt != 0:
                    raise ValueError(f"Last dimension {value.shape[-1]} is not divisible by split count {split_cnt} for key: {key}")

                chunk_size = len(value['scale']) // split_cnt if isinstance(value, dict) else value.shape[-1] // split_cnt
                logger.debug(f"Splitting {key}...")
                for i in range(split_cnt):
                    start = i * chunk_size
                    end = start + chunk_size
                    updated_key = ".".join(key_splits[:-2] + [template.substitute(num=i), key_splits[-1]])
                    split_value = value[..., start:end]
                    updated_data[updated_key] = split_value
                    logger.debug(f"Split {i} ==> {updated_key}")
            else:
                updated_data[key] = value
        return updated_data


    def split_encoding_data(self, param_encoding):
        """
        Splits a list of parameter encoding dictionaries from MHA to SHA format.

        This method processes each encoding dictionary. If the 'name' field in
        the encoding dictionary corresponds to an attention projection, the
        'scale' and 'offset' arrays are split into chunks for each attention head.

        :param param_encoding: A list of dictionaries, each representing parameter encoding information.
        :return: A new list of dictionaries with encoding data split into SHA format.
        :raises ValueError: If 'scale' or 'offset' are missing in encoding data,
                            or if dimensions are not divisible by the split count.
        """
        updated_encoding_info = []
        for encoding in tqdm(param_encoding, "Splitting Parameter Encoding"):
            key = encoding['name']

            key_splits = key.split('.')
            if len(key_splits) > 2 and key_splits[-2] in self.parameter_mha_to_sha_maps.keys():
                template, split_cnt = self.parameter_mha_to_sha_maps[key_splits[-2]]

                # Check if the encoding is valid (contains scale and offset) and can be split along the last axis equally or not
                if 'scale' not in encoding or 'offset' not in encoding:
                    raise ValueError(f"Missing 'scale' or 'offset' in encoding data for key: {key}")

                if len(encoding['scale']) % split_cnt != 0:
                    raise ValueError(f"Scale length {len(encoding['scale'])} is not divisible by split count {split_cnt} for key: {key}")

                if len(encoding['offset']) % split_cnt != 0:
                    raise ValueError(f"Offset length {len(encoding['offset'])} is not divisible by split count {split_cnt} for key: {key}")

                chunk_size = len(encoding['scale']) // split_cnt
                int_scale_chunk_size = None
                if 'per_block_int_scale' in encoding:
                    if len(encoding['per_block_int_scale']) % split_cnt != 0:
                        raise ValueError(f"per_block_int_scale length {len(encoding['per_block_int_scale'])} is not divisible by split count {split_cnt} for key: {key}")
                    int_scale_chunk_size = len(encoding['per_block_int_scale']) //split_cnt

                logger.debug(f"Splitting {key}...")
                for i in range(split_cnt):
                    start = i * chunk_size
                    end = start + chunk_size
                    updated_key = ".".join(key_splits[:-2] + [template.substitute(num = i), key_splits[-1]])
                    split_data = {key:value for key, value in encoding.items() if key not in ['scale', 'offset', 'name', 'per_block_int_scale']}
                    split_data['scale'] = encoding['scale'][start:end]
                    split_data['offset'] = encoding['offset'][start:end]
                    split_data['name'] = copy.deepcopy(updated_key)
                    if int_scale_chunk_size is not None:
                        block_start = i * int_scale_chunk_size
                        block_end = block_start + int_scale_chunk_size
                        split_data['per_block_int_scale'] = encoding['per_block_int_scale'][block_start:block_end]
                    updated_encoding_info.append(split_data)
                    logger.debug(f"Split {i} ==> {updated_key}")
            else:
                updated_encoding_info.append(encoding)

        return updated_encoding_info

    @classmethod
    def from_config(cls, config: AutoConfig):
        """
        Creates an MHAtoSHAConverter instance from an AutoConfig object.

        This class method initializes a converter by extracting the number of attention heads
        and key-value heads from the provided config. It also validates that the model
        implements the required 'parameter_encoding_map' and 'activation_encoding_map' methods,
        and uses these methods to set up the parameter and activation encoding mappings.

        :param config: An AutoConfig object containing model configuration including
                    num_attention_heads and num_key_value_heads.
        :return: An initialized MHAtoSHAConverter instance.
        :raises AttributeError: If the model does not implement the required
                                'parameter_encoding_map' or 'activation_encoding_map' methods.
        """
        model_obj = AutoModel(config)
        obj = cls.__new__(cls)
        obj.num_key_value_heads = config.num_key_value_heads
        obj.num_attention_heads = config.num_attention_heads
        # Validate that model has required methods
        if not hasattr(model_obj, 'parameter_encoding_map') or not callable(getattr(model_obj, 'parameter_encoding_map')):
            raise AttributeError(f"Model {type(model_obj).__name__} does not implement parameter_encoding_map method. Can not create converter object.")
        if not hasattr(model_obj, 'activation_encoding_map') or not callable(getattr(model_obj, 'activation_encoding_map')):
            raise AttributeError(f"Model {type(model_obj).__name__} does not implement activation_encoding_map method. Can not create converter object.")

        obj.parameter_mha_to_sha_maps = model_obj.parameter_encoding_map(config.num_attention_heads, config.num_key_value_heads)
        obj.activation_mha_to_sha_maps = model_obj.activation_encoding_map(config.num_attention_heads, config.num_key_value_heads)
        return obj

    def generate_sha_encodings(self, encoding_file, output_encoding_file_path):
        """
        Generates SHA format encodings from an MHA encoding file and saves them.

        This method loads encodings from the input file, splits parameter encodings
        into SHA format, converts activation encodings, and saves the updated encodings
        to the output file.

        :param encoding_file: The path to the input MHA encoding file.
        :param output_encoding_file_path: The path where the converted SHA encodings will be saved.
        :raises ValueError: If the encoding file is invalid or does not contain required keys.
        """
        encodings = load_encodings(encoding_file)
        try:
            encodings["param_encodings"] = self.split_encoding_data(encodings["param_encodings"])
            encodings["activation_encodings"] = self.convert_activation_encoding(encodings["activation_encodings"])
            save_encodings(encodings, output_encoding_file_path)
        except KeyError as e:
            raise ValueError(f"Invalid encoding file {encoding_file} provided.\n Error Message: {str(e)}")

    def convert_activation_encoding(self, activation_encoding):
        """
        Converts activation encodings from MHA to SHA format.

        This method processes each activation encoding dictionary and applies the
        activation encoding mappings to generate SHA format encodings. For each
        applicable encoding, it creates multiple copies corresponding to each
        attention head, updating the tensor names according to the mapping templates.

        :param activation_encoding: A list of dictionaries, each representing activation encoding information.
        :return: A new list of dictionaries with activation encoding data converted to SHA format.
        :raises AttributeError: If activation_mha_to_sha_maps is not initialized. Use from_config() to initialize.
        """

        if self.activation_mha_to_sha_maps is None:
            raise AttributeError("Activation Mapping not present. Please initialize the converter object"
                                 " using corresponding model config using `from_config` class method.")

        updated_encoding = []
        for encoding in tqdm(activation_encoding, "Converting Activation Encoding from MHA to SHA."):
            tensor_name = encoding["name"]
            key_splits = tensor_name.split('.')
            if len(key_splits) > 3:
                mha_key = ".".join(key_splits[3:])
                if mha_key in self.activation_mha_to_sha_maps.keys():
                    mapping = self.activation_mha_to_sha_maps[mha_key]
                    if isinstance(mapping, tuple):
                        mapping = [mapping]
                    for template, split_cnt in mapping:
                        for i in range(split_cnt):
                            if template.template == "past_value_out_output_0":
                                updated_key = f"past_value_{key_splits[2]}_out"
                            else:
                                updated_key = ".".join(key_splits[:3] + [template.substitute(num=i)])
                            logger.debug(f"Encoding copied {tensor_name} =>  {updated_key}")
                            updated_enc = copy.deepcopy(encoding)
                            updated_enc["name"] = copy.deepcopy(updated_key)
                            updated_encoding.append(updated_enc)

                    continue
            updated_encoding.append(encoding)

        return updated_encoding
