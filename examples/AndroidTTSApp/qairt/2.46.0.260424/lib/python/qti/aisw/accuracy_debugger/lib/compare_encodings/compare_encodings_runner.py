# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


import json
import os
from collections import OrderedDict

import numpy as np
import xlsxwriter
from qti.aisw.accuracy_debugger.common_config import EncodingInputConfig
from qti.aisw.accuracy_debugger.compare_encodings.compare_encodings import (
    CompareEncodings,
)
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name


class CompareEncodingsRunner(object):
    def __init__(self, logger, args):
        # type: (Logger, namespace) -> None

        self.args = args
        self._logger = logger
        self.encoding_diff_path = os.path.join(args.output_dir, "encodings_diff.xlsx")
        self.extracted_encodings_path = os.path.join(args.output_dir, "extracted_encodings.json")
        self.filtered_encodings_path = os.path.join(args.output_dir, "filtered_encodings.json")
        self.engine_type = None

    def run(self, engine_type):
        self.engine_type = engine_type
        self._logger.info(f"Arguments received to encodings comparison tool: {self.args}")

        if self.engine_type == Engine.QNN.value:
            self.compare_encodings_qnn()
        elif self.engine_type == Engine.SNPE.value:
            self.compare_encodings_snpe()
        elif self.engine_type == Engine.QAIRT.value:
            self._compare_encodings_qairt()
        else:
            raise ParameterError(
                f"Given engine type {self.engine_type} does not support Compare encodings feature."
            )

    def _compare_encodings_qairt(self):
        encoding_config1 = EncodingInputConfig(
            encoding_file_path=self.args.encoding1_file_path,
            quantized_dlc_path=self.args.quantized_dlc1_path,
        )
        encoding_config2 = EncodingInputConfig(
            encoding_file_path=self.args.encoding2_file_path,
            quantized_dlc_path=self.args.quantized_dlc2_path,
        )
        comapre_encodings = CompareEncodings(logger=self._logger)
        comapre_encodings.run(
            encoding_config1=encoding_config1,
            encoding_config2=encoding_config2,
            output_dir=self.args.output_dir,
            framework_model_path=self.args.framework_model_path,
            scale_threshold=self.args.scale_threshold,
        )

    def check_missing_encodings(self, extracted_encodings=None, aimet_encodings=None):
        """
        Helper function to find encodings present in AIMET but not in Target(QNN/SNPE) and vice-versa
        """
        self._logger.info(
            f"Finding encodings present only in AIMET encodings but not in {self.engine_type} encodings:"
        )
        for enc_type in aimet_encodings:
            if enc_type in extracted_encodings:
                self._logger.info(f"Checking {enc_type}...")
                for layer in aimet_encodings[enc_type]:
                    if all(
                        alias not in extracted_encodings[enc_type]
                        for alias in [layer, layer + "_permute"]
                    ):
                        self._logger.warning(f"{layer} present only in AIMET encodings")
            else:
                self._logger.warning(f"{enc_type} present only in AIMET encodings")

        self._logger.info(
            f"Finding encodings present only in {self.engine_type} encodings but not in AIMET encodings:"
        )
        for enc_type in extracted_encodings:
            if enc_type in aimet_encodings:
                self._logger.info(f"Checking {enc_type}...")
                for layer in extracted_encodings[enc_type]:
                    if all(
                        alias not in aimet_encodings[enc_type]
                        for alias in [layer, layer.replace("_permute", "")]
                    ):
                        self._logger.warning(
                            f"{layer} present only in {self.engine_type} encodings"
                        )
            else:
                self._logger.warning(f"{enc_type} present only in {self.engine_type} encodings")

    def compare_encodings_qnn(self):
        extracted_encodings = self.extract_model_net_encodings()
        with open(self.extracted_encodings_path, "w") as json_write:
            json.dump(extracted_encodings, json_write, indent=4)
        aimet_encodings, sanitize_usantize_map = self.get_aimet_encodings()
        # Filter the extracted encodings from model_net_json
        filtered_encodings = self.filter_encodings(extracted_encodings, sanitize_usantize_map)
        # Dump the filtered encodings in json file
        with open(self.filtered_encodings_path, "w") as json_write:
            json.dump(filtered_encodings, json_write, indent=4)
        sanitized_filtered_encodings = {}
        sanitized_filtered_encodings["activation_encodings"] = {}
        sanitized_filtered_encodings["param_encodings"] = {}
        for encodings in filtered_encodings.keys():
            if encodings == "activation_encodings" or encodings == "param_encodings":
                for layer in filtered_encodings[encodings].keys():
                    sanitized_filtered_encodings[encodings][santize_node_name(layer)] = (
                        filtered_encodings[encodings][layer]
                    )
        self.generate_excel_sheet(aimet_encodings, sanitized_filtered_encodings, "qnn")

        self.check_missing_encodings(
            extracted_encodings=sanitized_filtered_encodings, aimet_encodings=aimet_encodings
        )

    def get_dtype(self, data_type):
        # hex value to dtype conversion map
        dtype_map = {
            "0x008": "int",
            "0x016": "int",
            "0x032": "int",
            "0x064": "int",
            "0x108": "int",
            "0x116": "int",
            "0x132": "int",
            "0x164": "int",
            "0x308": "int",
            "0x316": "int",
            "0x332": "int",
            "0x408": "int",
            "0x416": "int",
            "0x432": "int",
            "0x216": "float",
            "0x232": "float",
            "0x508": "bool",
        }
        dtype = hex(data_type)
        dtype = dtype_map.get(dtype, "")
        return dtype

    def generate_encoding_dict(
        self, min_value, max_value, scale, offset, bitwidth, data_type=None, is_symmetric=None
    ):
        """
        Helper function to create a dictionary with given encodings data
        """
        # Using OrderedDict to maintain same order as AIMET encodings
        encoding_dict = OrderedDict()
        encoding_dict["bitwidth"] = bitwidth
        if data_type:
            encoding_dict["dtype"] = data_type
        if is_symmetric is not None:
            encoding_dict["is_symmetric"] = str(is_symmetric)
        encoding_dict["max"] = max_value
        encoding_dict["min"] = min_value
        encoding_dict["offset"] = offset
        encoding_dict["scale"] = scale

        return encoding_dict

    def get_activation_encodings(self, data, op_list):
        """
        Helper function to extract activation encodings
        """
        activation_encodings = OrderedDict()
        data_tensor = data["graph"]["tensors"]
        data_nodes = data["graph"]["nodes"]
        try:
            for layer in data_tensor:
                if "params_count" in data_tensor[layer].keys() or (
                    layer in data_nodes and data_nodes[layer]["type"] in op_list
                ):
                    continue
                datatype = self.get_dtype(data_tensor[layer]["data_type"])
                encoding_info = data_tensor[layer]["quant_params"]["scale_offset"]
                encoding_dict = self.generate_encoding_dict(
                    encoding_info["minimum"],
                    encoding_info["maximum"],
                    encoding_info["scale"],
                    encoding_info["offset"],
                    encoding_info["bitwidth"],
                    datatype,
                    encoding_info["is_symmetric"],
                )
                activation_encodings[layer] = [encoding_dict]
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )
        return activation_encodings

    def get_param_encodings(self, data, op_list):
        """
        Helper function to extract param encodings
        """
        param_encodings = OrderedDict()
        data_tensor = data["graph"]["tensors"]
        data_nodes = data["graph"]["nodes"]
        try:
            for layer in data_tensor:
                if "params_count" not in data_tensor[layer].keys() or (
                    layer in data_nodes and data_nodes[layer]["type"] in op_list
                ):
                    continue
                datatype = self.get_dtype(data_tensor[layer]["data_type"])
                reset_offset = False
                if np.right_shift(data_tensor[layer]["data_type"], 8) == 3:
                    reset_offset = True
                if "axis_scale_offset" in data_tensor[layer]["quant_params"]:
                    channel_encodings = []
                    if "scale_offsets" in data_tensor[layer]["quant_params"]["axis_scale_offset"]:
                        num_channels = len(
                            data_tensor[layer]["quant_params"]["axis_scale_offset"]["scale_offsets"]
                        )
                        encoding_type = "scale_offsets"
                    else:
                        num_channels = len(
                            data_tensor[layer]["quant_params"]["axis_scale_offset"][
                                "bw_scale_offset"
                            ]
                        )
                        encoding_type = "bw_scale_offset"
                    for axis in range(num_channels):
                        encoding_info = data_tensor[layer]["quant_params"]["axis_scale_offset"][
                            encoding_type
                        ][axis]
                        if reset_offset:
                            encoding_info["offset"] = 0
                        encoding_dict = self.generate_encoding_dict(
                            encoding_info["minimum"],
                            encoding_info["maximum"],
                            encoding_info["scale"],
                            encoding_info["offset"],
                            encoding_info["bitwidth"],
                            datatype,
                            encoding_info["is_symmetric"],
                        )
                        channel_encodings.append(encoding_dict)
                    param_encodings[layer] = channel_encodings
                else:
                    for encoding_type in ["scale_offset", "bw_scale_offset"]:
                        if encoding_type in data_tensor[layer]["quant_params"]:
                            encoding_info = data_tensor[layer]["quant_params"][encoding_type]
                            if reset_offset:
                                encoding_info["offset"] = 0
                            encoding_dict = self.generate_encoding_dict(
                                encoding_info["minimum"],
                                encoding_info["maximum"],
                                encoding_info["scale"],
                                encoding_info["offset"],
                                encoding_info["bitwidth"],
                                datatype,
                                encoding_info["is_symmetric"],
                            )
                            param_encodings[layer] = [encoding_dict]
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting param encodings from the given json file, error: {e}"
            )
        return param_encodings

    def extract_model_net_encodings(self):
        """
        Helper function to extract encodings from model_net.json file
        """
        with open(self.args.input) as json_file:
            data = json.load(json_file)
            op_list = [
                "Reduce",
                "Transpose",
                "CropAndResize",
                "Gather",
                "GatherElements",
                "GatherND",
                "Pad",
                "Pool2d",
                "Pool3d",
                "Reshape",
                "Resize",
                "StridedSlice",
                "SpaceToDepth",
                "DepthToSpace",
                "ChannelShuffle",
                "Split",
                "TopK",
                "Conv2d",
                "Conv3d",
                "TransposeConv2d",
                "DepthwiseConv2d",
                "FullyConnected",
                "MatMul",
            ]
            extracted_encodings = {}
            extracted_encodings["activation_encodings"] = self.get_activation_encodings(
                data, op_list
            )
            extracted_encodings["param_encodings"] = self.get_param_encodings(data, op_list)
            return extracted_encodings

    def get_aimet_encodings(self):
        """
        Helper function extract aimet encodings from file
        """
        aimet_encodings = {}
        aimet_encodings["activation_encodings"] = {}
        aimet_encodings["param_encodings"] = {}
        sanitize_unsanitize_map = OrderedDict()
        with open(self.args.aimet_encodings_json) as json_file:
            aimet_encodings_json = json.load(json_file)
            for encodings in aimet_encodings_json.keys():
                if encodings == "activation_encodings" or encodings == "param_encodings":
                    for layer in aimet_encodings_json[encodings].keys():
                        sanitize_unsanitize_map[santize_node_name(layer)] = layer
                        aimet_encodings[encodings][santize_node_name(layer)] = aimet_encodings_json[
                            encodings
                        ][layer]
        return aimet_encodings, sanitize_unsanitize_map

    def filter_encodings(self, encodings, sanitize_unsanitize_map):
        """
        Helper function to filter extracted encodings
        """
        filtered_encodings = {}
        filtered_encodings["activation_encodings"] = {}
        filtered_encodings["param_encodings"] = {}
        for encoding in encodings.keys():
            if encoding == "activation_encodings" or encoding == "param_encodings":
                for tensor in encodings[encoding].keys():
                    try:
                        if tensor in sanitize_unsanitize_map.keys():
                            filtered_encodings[encoding][sanitize_unsanitize_map[tensor]] = (
                                encodings[encoding][tensor]
                            )
                        elif tensor.endswith("_permute"):
                            new_tensor = tensor.removesuffix("_permute")
                            filtered_encodings[encoding][sanitize_unsanitize_map[new_tensor]] = (
                                encodings[encoding][tensor]
                            )
                    except:
                        continue
        return filtered_encodings

    def compare_encodings_snpe(self):
        try:
            from qti.aisw.dlc_utils import snpe_dlc_utils
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        # Load given SNPE DLC file
        snpe_model = snpe_dlc_utils.ModelInfo(self.args.input)

        # Fetch model's meta data
        (
            model_version,
            converter_command,
            quantizer_command,
            converter_version,
            model_copyright,
        ) = snpe_model.get_meta_data()

        # Find Major version using value of converter_version
        # Sample value of converter_version variable is 'DLC created with converter version: 2.16.0.231027072756_64280'
        converter_major_version = converter_version.split(":")[-1].strip().split(".")[0]
        self._logger.info(converter_version)

        # Extract both activation and param encodings from the given DLC
        DLC_helper = DLCHelper(self.args.input, converter_major_version)
        extracted_encodings = DLC_helper.extract_dlc_encodings()

        # Dump Extracted SNPE encodings to json file
        with open(self.extracted_encodings_path, "w") as json_write:
            json.dump(extracted_encodings, json_write, indent=4)

        # load AIMET encodings
        with open(self.args.aimet_encodings_json) as json_file:
            aimet_encodings = json.load(json_file)

        # Generate excel sheet highlighting any mismatches between AIMET and SNPE encodings
        self.generate_excel_sheet(
            aimet_encodings, extracted_encodings, "snpe", converter_major_version
        )

        # Log warnings if any encodings are present in AIMET but not in SNPE and vice-versa
        self.check_missing_encodings(
            extracted_encodings=extracted_encodings, aimet_encodings=aimet_encodings
        )

        self._logger.info(
            "Extracted SNPE encodings are saved at {}".format(
                os.path.abspath(self.extracted_encodings_path)
            )
        )
        self._logger.info(
            "Differences in SNPE encodings and AIMET encodings are written to {}".format(
                os.path.abspath(self.encoding_diff_path)
            )
        )

    def generate_excel_sheet(
        self, aimet_encodings, target_encodings, engine, converter_major_version=0
    ):
        """
        Helper function to find differences between AIMET and Target encodings.
        """
        with xlsxwriter.Workbook(self.encoding_diff_path) as workbook:
            # Initialize Excel sheet
            worksheet = workbook.add_worksheet()
            # Writer headers to Excel sheet
            if converter_major_version == 1:
                headers = [
                    "Encoding_type",
                    "buffer_name",
                    "bitwidth",
                    "max",
                    "min",
                    "offset",
                    "scale",
                ]
            else:
                headers = [
                    "Encoding_type",
                    "buffer_name",
                    "bitwidth",
                    "dtype",
                    "is_symmetric",
                    "max",
                    "min",
                    "offset",
                    "scale",
                ]

            headers_idx = {}
            for idx, header in enumerate(headers):
                worksheet.write(0, idx, header)
                headers_idx[header] = idx

            sheet_idx = 1
            warning_format_1 = workbook.add_format({"bold": True, "font_color": "red"})
            warning_format_2 = workbook.add_format({"bold": True, "font_color": "blue"})
            diff_counts = {}
            dlc_version = "dlcv3"
            if converter_major_version != 1:
                dlc_version = "dlcv4"
            if engine == "qnn":
                target_encoding_type = "QNN"
            else:
                target_encoding_type = dlc_version
            # Loop for activations and params
            for encoding_type in aimet_encodings.keys():
                diff_counts[encoding_type] = 0

                if (self.args.params_only and encoding_type == "activation_encodings") or (
                    self.args.activations_only and encoding_type == "param_encodings"
                ):
                    continue

                if encoding_type not in target_encodings.keys():
                    continue
                """
                Loop for encodings list present in activations/params.
                if a layer has per-channel quantization then aimet_encoding_list will contain multiple encoding dictionaries corresponding to each channel,
                otherwise only one encoding dictionary will present in aimet_encoding_list
                """
                for encoding_name, aimet_encoding_list in aimet_encodings[encoding_type].items():
                    if self.args.specific_node and encoding_name != self.args.specific_node:
                        continue

                    if encoding_name not in target_encodings[encoding_type].keys():
                        continue

                    target_encoding_list = target_encodings[encoding_type][encoding_name]

                    if len(aimet_encoding_list) != len(target_encoding_list):
                        self._logger.error(
                            f"Encodings channels count mismatch for {encoding_name}, AIMET has {len(aimet_encoding_list)} channel encodings while Target has {len(target_encoding_list)} channel encodings"
                        )
                        continue

                    for idx, aimet_encoding_dict in enumerate(aimet_encoding_list):
                        target_encoding_dict = target_encoding_list[idx]
                        errors_observed = 0
                        # Indicate whether scale and offset need to be corrected according to the bitwidths before comparing
                        correction = None
                        for key in aimet_encoding_dict.keys():
                            if key not in target_encoding_dict.keys():
                                continue

                            # convert below encodings to strings since dtype and is_symmetric are strings in AIMET encodings
                            if key in ["dtype", "is_symmetric"]:
                                target_encoding_dict[key] = str(target_encoding_dict[key])
                            pre = self.args.precision

                            # if encoding is either scale or offset and If correction is needed then modify the aimet_encoding before comparing.
                            if key in ["scale", "offset"] and correction:
                                if (correction == "up" and key == "offset") or (
                                    correction == "down" and key == "scale"
                                ):
                                    compare_encoding = round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] * 256.0), pre) or round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] * 257.0), pre)
                                else:
                                    compare_encoding = round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] / 256.0), pre) or round(
                                        target_encoding_dict[key], pre
                                    ) == round((aimet_encoding_dict[key] / 257.0), pre)
                            elif key in ["offset", "is_symmetric"]:
                                if (
                                    target_encoding_dict["is_symmetric"] == "False"
                                    and target_encoding_dict["offset"] == 0
                                    and aimet_encoding_dict["is_symmetric"] == "True"
                                    and aimet_encoding_dict["offset"] == -128
                                ):
                                    compare_encoding = True
                                else:
                                    if key in ["offset"]:
                                        compare_encoding = round(
                                            target_encoding_dict[key], pre
                                        ) == round(aimet_encoding_dict[key], pre)
                                    else:
                                        compare_encoding = (
                                            target_encoding_dict[key] == aimet_encoding_dict[key]
                                        )
                            elif key in ["max", "min", "scale"]:
                                # Compare the encodings by rounding with the specified precision
                                compare_encoding = round(target_encoding_dict[key], pre) == round(
                                    aimet_encoding_dict[key], pre
                                )
                            else:
                                compare_encoding = (
                                    target_encoding_dict[key] == aimet_encoding_dict[key]
                                )
                            # Compare current iteration's encoding and they are not equal
                            if not compare_encoding:
                                # Highlight entry for encoding since AIMET and Target is not matching
                                diff_counts[encoding_type] += 1
                                errors_observed += 1
                                # Default error message if the encodings are not equal
                                diff_warning = f"* {target_encoding_type} encoding={str(target_encoding_dict[key])} aimet encoding={str(aimet_encoding_dict[key])}"
                                if key == "bitwidth":
                                    # Warning message if bitwidths are not equal and either 8 or 16
                                    diff_warning = f"| {target_encoding_type} encoding={str(target_encoding_dict[key])} aimet encoding={str(aimet_encoding_dict[key])}"
                                    if (
                                        target_encoding_dict[key] == 16
                                        and aimet_encoding_dict[key] == 8
                                    ):
                                        correction = "up"
                                    elif (
                                        target_encoding_dict[key] == 8
                                        and aimet_encoding_dict[key] == 16
                                    ):
                                        correction = "down"
                                    else:
                                        # Invalid bitwidths, neither of the bitwidth equal to 8 or 16
                                        diff_warning = f"* Activation bitwidth conversions from aimet encoding={str(aimet_encoding_dict[key])} to {target_encoding_type} encoding={str(target_encoding_dict[key])} not supported"
                                # If encoding is either scale/offset and correction was applied
                                elif key in ["scale", "offset"] and correction:
                                    diff_warning = f"* {key} not consistent according to bitwidth conversion {target_encoding_type} encoding={str(target_encoding_dict[key])} aimet encoding={str(aimet_encoding_dict[key])}"
                                # if the warning message starts with "|", apply warning_format_2
                                if diff_warning[0] == "|":
                                    worksheet.write(
                                        sheet_idx, headers_idx[key], diff_warning, warning_format_2
                                    )
                                else:
                                    worksheet.write(
                                        sheet_idx, headers_idx[key], diff_warning, warning_format_1
                                    )
                        if errors_observed:
                            worksheet.write(sheet_idx, 0, encoding_type)
                            worksheet.write(sheet_idx, 1, encoding_name)
                            sheet_idx = sheet_idx + 1

                    if self.args.specific_node:
                        break

        self._logger.info(
            f"Number of activation encoding differences observed: {diff_counts['activation_encodings']}"
        )
        self._logger.info(
            f"Number of param encoding differences observed: {diff_counts['param_encodings']}"
        )
        self._logger.info(
            f"Total number of encoding differences observed: {diff_counts['activation_encodings'] + diff_counts['param_encodings']}"
        )


class DLCHelper:
    def __init__(self, dlc, converter_major_version):
        try:
            from qti.aisw.dlc_utils import modeltools
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        self.converter_major_version = converter_major_version
        if converter_major_version == "1":
            self.model = modeltools.Model()
            self.model.load(dlc)
        else:
            self.model = modeltools.IrDlcReader()
            self.cache_reader = modeltools.IrDlcCacheRecordReader()
            self.model.open(dlc)

    def extract_dlc_encodings(self):
        """
        Extracts both activation and param encodings from the given dlc.
        """
        extracted_encodings = {}
        extracted_encodings["activation_encodings"] = self.get_activation_encodings()
        extracted_encodings["param_encodings"] = self.get_param_encodings()

        return extracted_encodings

    def generate_encoding_dict(
        self, min_value, max_value, delta, offset, bitwidth, data_type=None, is_symmetric=None
    ):
        """
        Helper function to create a dictionary with given encodings data
        """
        # Using OrderedDict to maintain same order as AIMET encodings
        encoding_dict = OrderedDict()
        encoding_dict["bitwidth"] = bitwidth
        if data_type:
            encoding_dict["dtype"] = data_type
        if is_symmetric:
            encoding_dict["is_symmetric"] = is_symmetric
        encoding_dict["max"] = max_value
        encoding_dict["min"] = min_value
        encoding_dict["offset"] = offset
        encoding_dict["scale"] = delta

        return encoding_dict

    def get_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc.
        """
        if self.converter_major_version == "1":
            return self.extract_dlcv3_activation_encodings()
        else:
            return self.extract_dlcv4_activation_encodings()

    def extract_dlcv3_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc with converter version 1.*
        """
        activation_encodings = OrderedDict()
        try:
            for layer in self.model.get_layers():
                if ":0" in layer["name"]:
                    continue

                min_value, max_value, delta, offset, bitwidth = self.model.get_tf_output_encoding(
                    layer["name"]
                )[:5]
                encoding_name = layer["output_names"][0]
                encoding_dict = self.generate_encoding_dict(
                    min_value, max_value, delta, offset, bitwidth
                )
                activation_encodings[encoding_name] = [encoding_dict]
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )

        return activation_encodings

    def extract_dlcv4_activation_encodings(self):
        """
        Extracts activation encodings from the given dlc with converter version 2.*
        """
        try:
            from qti.aisw.converters.common import ir_graph
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        def extract_encodings(encoding_name, encoding, dtype):
            """
            Helper function to extract bitwidth, min, max, scale and offset params from the given encoding
            """
            encoding_info = None
            if encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
                encoding_info = encoding.encInfo
            elif encoding.type == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET:
                encoding_info = encoding.encInfo.axisEncInfo.encInfos[0]

            if encoding_info != None:
                data_type = self.get_aimet_datatype(dtype)
                encoding_dict = self.generate_encoding_dict(
                    encoding_info.min,
                    encoding_info.max,
                    encoding_info.scale,
                    encoding_info.offset,
                    encoding_info.bw,
                    data_type=data_type,
                    is_symmetric=str(bool(encoding.axisEncInfo.axis)),
                )
                activation_encodings[encoding_name] = [encoding_dict]

        graph = self.model.get_ir_graph()
        activation_encodings = OrderedDict()
        try:
            for op in graph.get_ops():
                if ":0" in op.name:
                    continue

                # Extract encodings from inputs of the Op
                for input in op.inputs():
                    if input.is_app_write_tensor():
                        extract_encodings(
                            input.name(), input.get_encoding(), input.data_type_string()
                        )

                # Extract encodings from outputs of the Op
                for output in op.outputs():
                    extract_encodings(
                        output.name(), output.get_encoding(), output.data_type_string()
                    )
        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting activation encodings from the given DLC file, error: {e}"
            )

        return activation_encodings

    def get_aimet_datatype(self, snpe_dtype):
        """
        Returns AIMET equivalent datatype for given SNPE datatype
        """
        if snpe_dtype in [
            "Int_8",
            "Uint_8",
            "sFxp_8",
            "uFxp_8",
            "Int_16",
            "Uint_16",
            "sFxp_16",
            "uFxp_16",
            "Int_32",
            "Uint_32",
            "sFxp_32",
            "uFxp_32",
            "Int_64",
            "Uint_64",
        ]:
            data_type = "int"
        elif snpe_dtype in ["Float_16", "Float_32"]:
            data_type = "float"
        elif snpe_dtype == "Bool_8":
            data_type = "bool"
        else:
            data_type = "undefined"
        return data_type

    def get_param_encodings(self):
        """
        Extracts param encodings from the given dlc.
        """
        if self.converter_major_version == "1":
            return self.extract_dlcv3_param_encodings()
        else:
            return self.extract_dlcv4_param_encodings()

    def extract_dlcv4_param_encodings(self):
        """
        Extracts param encodings from the given dlc with converter version 2.*
        """
        try:
            from qti.aisw.converters.common import ir_graph
        except ImportError as ie:
            raise Exception(
                f"Failed to import necessary packages: {str(ie)}. Please ensure that $SNPE_ROOT/lib/python is added to your PYTHONPATH."
            )

        graph = self.model.get_ir_graph()
        param_encodings = OrderedDict()
        try:
            for op in graph.get_ops():
                if ":0" in op.name:
                    continue

                for input in op.inputs():
                    # consider only static tensors(weights)
                    if input.is_static_tensor():
                        data_type = self.get_aimet_datatype(input.data_type_string())

                        if (
                            input.get_encoding().type
                            == ir_graph.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET
                        ):
                            # extract per-tensor weight encodings
                            encoding_info = input.get_encoding().encInfo
                            encoding_dict = self.generate_encoding_dict(
                                encoding_info.min,
                                encoding_info.max,
                                encoding_info.scale,
                                encoding_info.offset,
                                encoding_info.bw,
                                data_type=data_type,
                                is_symmetric=str(bool(input.get_encoding().axisEncInfo.axis)),
                            )
                            param_encodings[input.name()] = [encoding_dict]
                        elif (
                            input.get_encoding().type
                            == ir_graph.QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET
                            or input.get_encoding().type
                            == ir_graph.QNN_QUANTIZATION_ENCODING_BW_AXIS_SCALE_OFFSET
                        ):
                            # extract per-channel weight encodings
                            channel_encodings = []
                            for axis in range(len(input.get_encoding().axisEncInfo.encInfos)):
                                encoding_info = input.get_encoding().axisEncInfo.encInfos[axis]
                                encoding_dict = self.generate_encoding_dict(
                                    encoding_info.min,
                                    encoding_info.max,
                                    encoding_info.scale,
                                    encoding_info.offset,
                                    encoding_info.bw,
                                    data_type=data_type,
                                    is_symmetric=str(bool(input.get_encoding().axisEncInfo.axis)),
                                )
                                channel_encodings.append(encoding_dict)
                            param_encodings[input.name()] = channel_encodings

        except Exception as e:
            raise Exception(
                f"Failure occurred while extracting param encodings from the given DLC file, error: {e}"
            )

        return param_encodings

    def extract_dlcv3_param_encodings(self):
        """
        Extracts param encodings from the given dlc with converter version 1.*
        """

        param_encodings = OrderedDict()
        for layer in self.model.get_layers():
            if ":0" in layer["name"]:
                continue

            try:
                weight_encoding = self.model.get_tf_weight_encoding(layer["name"], 0)
                if weight_encoding is not None:
                    axis = self.model.get_tf_weight_encoding_axis(layer["name"], 0)

                    if axis >= 0:
                        # extract per-channel weight encodings
                        num_elements = self.model.get_tf_weight_encoding_num_elements(
                            layer["name"], 0
                        )

                        channel_encodings = []
                        for channel in range(num_elements):
                            min_value, max_value, delta, offset, bitwidth = (
                                self.model.get_tf_weight_encoding_by_element(
                                    layer["name"], 0, channel
                                )[:5]
                            )
                            encoding_dict = self.generate_encoding_dict(
                                min_value, max_value, delta, offset, bitwidth
                            )
                            channel_encodings.append(encoding_dict)
                        encoding_name = layer["name"] + ".weight"
                        param_encodings[encoding_name] = channel_encodings
                    else:
                        # extract per-tensor weight encodings
                        min_value, max_value, delta, offset, bitwidth = weight_encoding[:5]
                        encoding_name = layer["name"] + ".weight"
                        encoding_dict = self.generate_encoding_dict(
                            min_value, max_value, delta, offset, bitwidth
                        )
                        param_encodings[encoding_name] = [encoding_dict]
            except:
                try:
                    # extract bias encodings
                    bias_encoding = self.model.get_tf_bias_encoding(layer["name"])
                    if bias_encoding is not None:
                        min_value, max_value, delta, offset, bitwidth = bias_encoding[:5]
                        encoding_name = layer["name"] + ".bias"
                        encoding_dict = self.generate_encoding_dict(
                            min_value, max_value, delta, offset, bitwidth
                        )
                        param_encodings[encoding_name] = [encoding_dict]
                except:
                    pass

        return param_encodings
