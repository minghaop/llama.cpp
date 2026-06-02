# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
import re
from typing import Any

from qti.aisw.accuracy_debugger.argparser.parser import Parser
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    NetRunnerInputArguments,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.lora import (
    LoRAModelCreatorInputConfig,
    LoRAImporterInputConfig,
)
from qti.aisw.accuracy_debugger.utils.constants import (
    supported_backends,
    supported_platforms,
)
from qti.aisw.tools.core.modules.api.definitions.common import BackendType, OpPackageIdentifier
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import GenerateConfig
from qti.aisw.tools.core.modules.converter.converter_module import (
    InputTensorConfig,
    OutputTensorConfig,
)
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DeviceCredentials,
    DevicePlatformType,
    RemoteDeviceIdentifier,
)
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper


class InferenceEngineParser(Parser):
    """This is a parser for Inference Engine tool."""

    def __init__(self, component="inference_engine"):
        super().__init__(component)

    def _initialize(self):
        """Create parser with inference engine tool specific arguments"""
        super()._initialize()
        converter_args = self._parser.add_argument_group("converter arguments")
        quantizer_args = self._parser.add_argument_group("quantizer arguments")
        net_run_args = self._parser.add_argument_group("netrun arguments")
        offline_prepare_args = self._parser.add_argument_group("offline prepare arguments")

        self.required.add_argument(
            "--input_model", type=str, required=True, help="Path to the source model/dlc/bin file"
        )
        converter_args.add_argument(
            "--desired_input_shape",
            "--input_tensor",
            dest="desired_input_shape",
            nargs="+",
            action="append",
            required=False,
            default=None,
            help="The name,dimension,datatype and layout of all the input buffers to the network "
            "specified in the format [input_name comma-separated-dimensions data-type layout]. "
            "Dimension, datatype and layout are optional."
            "for example: 'data' 1,224,224,3. Note that the quotes should always be included in "
            "order to handle special characters, spaces, etc. "
            "For multiple inputs, specify multiple --desired_input_shape on the command line like: "
            '--desired_input_shape "data1" 1,224,224,3 float32 '
            '--desired_input_shape "data2" 1,50,100,3 int64 ',
        )
        converter_args.add_argument(
            "--output_tensor",
            type=str,
            required=False,
            action="append",
            help="Name of the graph's specified output tensor(s).",
        )
        converter_args.add_argument(
            "--converter_float_bitwidth",
            type=int,
            required=False,
            default=32,
            choices=[32, 16],
            help="Use this option to convert the graph to the specified float \
                bitwidth, either 32 (default) or 16.",
        )
        converter_args.add_argument(
            "--float_bias_bitwidth",
            default=32,
            required=False,
            type=int,
            choices=[32, 16],
            help="Option to select the bitwidth to use for float bias tensor, either 32(default) \
                or 16",
        )
        converter_args.add_argument(
            "--quantization_overrides",
            required=False,
            default=None,
            type=str,
            help="Path to quantization overrides json file.",
        )
        converter_args.add_argument(
            "--onnx_define_symbol",
            default=None,
            nargs=2,
            action="append",
            required=False,
            metavar=("SYMBOL", "VALUE"),
            help="Option to override specific input dimension symbols.",
        )
        converter_args.add_argument(
            "--onnx_defer_loading",
            default=False,
            action="store_true",
            required=False,
            help="Option to have the model not load weights. "
            "If False, the model will be loaded eagerly.",
        )
        converter_args.add_argument(
            "--enable_framework_trace",
            default=False,
            action="store_true",
            required=False,
            help="Use this option to enable converter to trace the o/p tensor change information.",
        )
        converter_args.add_argument(
            "--op_package_config",
            default=[],
            required=False,
            nargs="+",
            type=str,
            help="Absolute paths to Qnn Op Package XML configuration file that "
            "contains user defined custom operations."
            "Note: Only one of: {'op_package_config', 'package_name'} can be specified.",
        )
        converter_args.add_argument(
            "--converter_op_package_lib",
            default=[],
            type=str,
            help="Absolute path to converter op package library compiled by the OpPackage "
            "generator. Must be separated by a comma for multiple package libraries. "
            "Note: Libraries must follow the same order as the xml files. "
            "E.g.1: --converter_op_package_lib absolute_path_to/libExample.so "
            "E.g.2: --converter_op_package_lib "
            "absolute_path_to/libExample1.so,absolute_path_to/libExample2.so",
        )
        converter_args.add_argument(
            "--disable_onnx_simplification",
            default=False,
            action="store_true",
            required=False,
            help="Flag to disable onnx simplification. Note: This will disable simplification "
            "even at framework level in snooping",
        )
        converter_args.add_argument(
            "--package_name",
            default="",
            required=False,
            type=str,
            help="A global package name to be used for each node in the Model.cpp file. "
            "Defaults to Qnn header defined package name. "
            "Note: Only one of: {'op_package_config', 'package_name'} can be specified.",
        )
        converter_args.add_argument(
            "--extra_converter_args",
            default=None,
            required=False,
            type=str,
            help="Any specific converter argument which is not explicitly exposed "
            "can be provided using this argument. "
            "Possible values an argument can take: "
            "1. single value: value1 or store_true "
            "2. list of values: [value1, value2, value3] "
            "3. list of list: [[value11, value12], [value21, value22]] "
            "Example: --extra_converter_args 'arg1=value1;arg2;arg3=value1 value2 value3;arg4=value11 value12;arg4=value21 value22'",
        )

        quantizer_args.add_argument(
            "--calibration_input_list",
            type=str,
            required=False,
            default=None,
            help="Path to the inputs list text file to run quantization(used with qairt-quantizer)",
        )
        quantizer_args.add_argument(
            "--bias_bitwidth",
            type=int,
            required=False,
            default=8,
            choices=[8, 32],
            help="Option to select the bitwidth to use when quantizing the bias. default 8",
        )
        quantizer_args.add_argument(
            "--act_bitwidth",
            type=int,
            required=False,
            default=8,
            choices=[8, 16],
            help="Option to select the bitwidth to use when quantizing the activations. default 8",
        )
        quantizer_args.add_argument(
            "--weights_bitwidth",
            type=int,
            required=False,
            default=8,
            choices=[8, 4],
            help="Option to select the bitwidth to use when quantizing the weights. default 8",
        )
        quantizer_args.add_argument(
            "--quantizer_float_bitwidth",
            type=int,
            required=False,
            default=32,
            choices=[32, 16],
            help="Use this option to select the bitwidth to use for float tensors, \
                either 32 (default) or 16.",
        )
        quantizer_args.add_argument(
            "--act_quantizer_calibration",
            type=str.lower,
            required=False,
            default="min-max",
            choices=["min-max", "sqnr", "entropy", "mse", "percentile"],
            help="Specify which quantization calibration method to use for activations. "
            "Supported values: min-max (default), sqnr, entropy, mse, percentile. "
            "This option can be paired with --act_quantizer_schema to override the "
            "quantization schema to use for activations otherwise the default "
            "schema (asymmetric) will be used.",
        )
        quantizer_args.add_argument(
            "--param_quantizer_calibration",
            type=str.lower,
            required=False,
            default="min-max",
            choices=["min-max", "sqnr", "entropy", "mse", "percentile"],
            help="Specify which quantization calibration method to use for parameters. "
            "Supported values: min-max (default), sqnr, entropy, mse, percentile. "
            "This option can be paired with --act_quantizer_schema to override the "
            "quantization schema to use for activations otherwise the default "
            "schema (asymmetric) will be used.",
        )
        quantizer_args.add_argument(
            "--act_quantizer_schema",
            type=str.lower,
            required=False,
            default="asymmetric",
            choices=["asymmetric", "symmetric", "unsignedsymmetric"],
            help="Specify which quantization schema to use for activations. \
                Note: Default is asymmetric.",
        )
        quantizer_args.add_argument(
            "--param_quantizer_schema",
            type=str.lower,
            required=False,
            default="asymmetric",
            choices=["asymmetric", "symmetric", "unsignedsymmetric"],
            help="Specify which quantization schema to use for parameters. \
                Note: Default is asymmetric.",
        )
        quantizer_args.add_argument(
            "--percentile_calibration_value",
            type=float,
            required=False,
            default=99.99,
            help="Value must lie between 90 and 100. Default is 99.99",
        )
        quantizer_args.add_argument(
            "--use_per_channel_quantization",
            action="store_true",
            default=False,
            help="Use per-channel quantization for convolution-based op weights. \
                Note: This will replace built-in model QAT encodings when used for a given weight.",
        )

        quantizer_args.add_argument(
            "--use_per_row_quantization",
            action="store_true",
            default=False,
            help="Use this option to enable rowwise quantization of Matmul and FullyConnected ops.",
        )

        quantizer_args.add_argument(
            "--float_fallback",
            action="store_true",
            default=False,
            help="Use this option to enable fallback to floating point (FP) instead of fixed point."
            "This option can be paired with --quantizer_float_bitwidth to indicate the bitwidth for"
            "FP (by default 32). If this option is enabled, then input list must "
            "not be provided and --ignore_encodings must not be provided. "
            "The external quantization encodings (encoding file/FakeQuant encodings) "
            "might be missing quantization parameters for some interim tensors. "
            "First it will try to fill the gaps by propagating across math-invariant "
            "functions. If the quantization parameters are still missing, "
            "then it will apply fallback to nodes to floating point.",
        )
        quantizer_args.add_argument(
            "--quantization_algorithms",
            required=False,
            default=[],
            type=str,
            nargs="+",
            help="Use this option to select quantization algorithms. Usage is: \
                --quantization_algorithms <algo_name1> ... ",
        )

        quantizer_args.add_argument(
            "--restrict_quantization_steps",
            required=False,
            default=[],
            type=str,
            help="Specifies the number of steps to use for computing"
            'quantization encodings E.g.--restrict_quantization_steps "-0x80 0x7F" indicates an \
            example 8 bit range,',
        )
        quantizer_args.add_argument(
            "--dump_encodings_json",
            required=False,
            default=False,
            action="store_true",
            help="Dump encoding of all the tensors in a json file",
        )

        quantizer_args.add_argument(
            "--ignore_encodings",
            required=False,
            default=False,
            action="store_true",
            help="Use only quantizer generated encodings, "
            "ignoring any user or model provided encodings.",
        )
        quantizer_args.add_argument(
            "--op_package_lib",
            type=str,
            default=[],
            required=False,
            help="Use this argument to pass an op package library for quantization. "
            "Must be in the form <op_package_lib_path:interfaceProviderName> and "
            "be separated by a comma for multiple package libs",
        )
        quantizer_args.add_argument(
            "--extra_quantizer_args",
            default=None,
            required=False,
            type=str,
            help="Any specific quantizer argument which is not explicitly exposed "
            "can be provided using this argument. "
            "Possible values an argument can take: "
            "1. single value: value1 or store_true "
            "2. list of values: [value1, value2, value3] "
            "3. list of list: [[value11, value12], [value21, value22]] "
            "Example: --extra_quantizer_args 'arg1=value1;arg2;arg3=value1 value2 value3;arg4=value11 value12;arg4=value21 value22'",
        )

        net_run_args.add_argument(
            "--perf_profile",
            type=str.lower,
            required=False,
            default="balanced",
            choices=[
                "low_balanced",
                "balanced",
                "default",
                "high_performance",
                "sustained_high_performance",
                "burst",
                "low_power_saver",
                "power_saver",
                "high_power_saver",
                "extreme_power_saver",
                "system_settings",
            ],
            help='Specifies performance profile to set. Valid settings are "low_balanced" ,'
            '"balanced", "default", high_performance" ,"sustained_high_performance", "burst", '
            '"low_power_saver", "power_saver", "high_power_saver", "extreme_power_saver", and '
            '"system_settings". Note: perf_profile argument is now deprecated for '
            "HTP backend, user can specify performance profile through "
            "backend extension config now.",
        )
        net_run_args.add_argument(
            "--profiling_level",
            type=str.lower,
            required=False,
            default=None,
            help="Enables profiling and sets its level. "
            'For QNN executor, valid settings are "basic", "detailed" and "client" '
            "Default is detailed.",
        )
        net_run_args.add_argument(
            "--input_list",
            type=str,
            required=False,
            help="Path to the input list text file to run inference(used with net-run). "
            "Note: When having multiple entries in text file, in order to save "
            "memory and time.",
        )
        net_run_args.add_argument(
            "--netrun_backend_extension_config",
            type=str,
            required=False,
            default=None,
            help="Path to config to be used with qnn-net-run",
        )
        net_run_args.add_argument(
            "--extra_net_run_args",
            default=None,
            required=False,
            type=str,
            help="Any specific net_run argument which is not explicitly exposed "
            "can be provided using this argument. "
            "Possible values an argument can take: "
            "1. single value: value1 or store_true "
            "2. list of values: [value1, value2, value3] "
            "3. list of list: [[value11, value12], [value21, value22]] "
            "Example: --extra_net_run_args 'arg1=value1;arg2;arg3=value1 value2 value3;arg4=value11 value12;arg4=value21 value22'",
        )

        offline_prepare_args.add_argument(
            "--offline_prepare_backend_extension_config",
            type=str,
            required=False,
            default=None,
            help="Path to config to be used with qnn-context-binary-generator.",
        )
        offline_prepare_args.add_argument(
            "--extra_context_bin_args",
            default=None,
            required=False,
            type=str,
            help="Any specific contextbinary generator argument which is not explicitly exposed "
            "can be provided using this argument. "
            "Possible values an argument can take: "
            "1. single value: value1 or store_true "
            "2. list of values: [value1, value2, value3] "
            "3. list of list: [[value11, value12], [value21, value22]] "
            "Example: --extra_context_bin_args 'arg1=value1;arg2;arg3=value1 value2 value3;arg4=value11 value12;arg4=value21 value22'",
        )

        self.optional.add_argument(
            "--backend",
            type=str.upper,
            required=False,
            choices=[backend.value for backend in supported_backends],
            default=None,
            help="Backend type for inference to be run",
        )
        self.optional.add_argument(
            "--platform",
            type=str,
            required=False,
            choices=[platform.value for platform in supported_platforms],
            default=None,
            help="The type of device platform to be used for inference",
        )
        self.optional.add_argument(
            "--offline_prepare",
            action="store_true",
            required=False,
            default=False,
            help=" Boolean to indicate offline preapre of the graph",
        )
        self.optional.add_argument(
            "--working_directory",
            type=str,
            required=False,
            default=None,
            help="Path to the directory to store the output result",
        )
        self.optional.add_argument(
            "--output_directory",
            type=str,
            required=False,
            default=None,
            help="Name of the output directory. If not specified a directory with name \
                <timestamp> will be created in the working directory.",
        )
        self.optional.add_argument(
            "--device_id",
            type=str,
            required=False,
            default=None,
            help="The serial number of the device to use. If not available, "
            "the first in a list of queried devices will be used for inference.",
        )
        self.optional.add_argument(
            "--username",
            type=str,
            required=False,
            default=None,
            help="The username for the device to be used for QNX platform.",
        )
        self.optional.add_argument(
            "--password",
            type=str,
            required=False,
            default=None,
            help="The password for the device to be used for QNX platform.",
        )
        self.optional.add_argument(
            "--ip_address",
            type=str,
            required=False,
            default=None,
            help="The IP address for the device to be used for QNX platform.",
        )
        self.optional.add_argument(
            "--log_level",
            type=str.upper,
            required=False,
            default="INFO",
            choices=["ERROR", "WARN", "INFO", "DEBUG", "VERBOSE"],
            help="Enable verbose logging.",
        )
        self.optional.add_argument(
            "--op_packages",
            type=str,
            required=False,
            default=[],
            help="Provide a comma separated list of op package and interface providers to "
            "register during graph preparation."
            "Usage: op_package_path:interface_provider[,op_package_path:interface_provider...]",
        )
        self.optional.add_argument(
            "--soc_model",
            type=str,
            required=False,
            default="",
            help="Option to specify the SOC on which the model needs to run. "
            "This can be found from SOC info of the device and it starts with strings "
            "such as SDM, SM, QCS, IPQ, SA, QC, SC, SXR, SSG, STP, QRB, or AIC.",
        )
        self.optional.add_argument(
            "--set_output_tensors",
            type=lambda s: [item.strip() for item in s.split(",")],
            required=False,
            default=None,
            help="Option to provide Comma-separated list of tensor names to set as outputs"
            "to the context binary generation stage or the net-run stage.",
        )

        # LoRA Model Creator arguments
        lora_model_creator_group = self._parser.add_argument_group(
            "LoRA Model Creator Arguments", "Arguments for qairt-lora-model-creator tool"
        )
        lora_model_creator_group.add_argument(
            "--lora_config",
            type=str,
            required=False,
            default=None,
            help="Path to the YAML config file for LoRA (required for LoRA Model Creator)",
        )
        lora_model_creator_group.add_argument(
            "--lora_output_dir",
            type=str,
            required=False,
            default=None,
            help="Path to store the output of the LoRA Model Creator tool",
        )
        lora_model_creator_group.add_argument(
            "--lora_quant_updatable_mode",
            type=str,
            choices=["none", "adapter_only", "all"],
            required=False,
            default="adapter_only",
            help="Specify whether/for which tensors the quantization encodings change across use-cases",
        )
        lora_model_creator_group.add_argument(
            "--lora_debug",
            type=int,
            required=False,
            default=-1,
            help="Run the LoRA Model Creator in debug mode",
        )
        lora_model_creator_group.add_argument(
            "--lora_skip_validation",
            action="store_true",
            required=False,
            default=False,
            help="Skip validation checks in LoRA Model Creator",
        )
        lora_model_creator_group.add_argument(
            "--lora_dump_usecase_onnx",
            action="store_true",
            required=False,
            default=False,
            help="Dump per-usecase ONNX models",
        )
        lora_model_creator_group.add_argument(
            "--lora_transforms_metadata",
            type=str,
            required=False,
            default=None,
            help="Path to a JSON file for storing transformation metadata",
        )

        # LoRA Importer arguments
        lora_importer_group = self._parser.add_argument_group(
            "LoRA Importer Arguments", "Arguments for qairt-lora-importer tool"
        )
        lora_importer_group.add_argument(
            "--lora_importer_config",
            type=str,
            required=False,
            default=None,
            help="Path to the YAML config file for LoRA Importer (use when running importer standalone)",
        )
        lora_importer_group.add_argument(
            "--lora_importer_input_dlc",
            type=str,
            required=False,
            default=None,
            help="Path to the Float or Quantized DLC for LoRA Importer (use when running importer standalone)",
        )
        lora_importer_group.add_argument(
            "--lora_importer_input_network",
            type=str,
            required=False,
            default=None,
            help="Path to the source ONNX model for LoRA Importer (use when running importer standalone)",
        )
        lora_importer_group.add_argument(
            "--lora_importer_input_list",
            type=str,
            required=False,
            default=None,
            help="Path to file specifying input data for LoRA Importer (use when running importer standalone)",
        )
        lora_importer_group.add_argument(
            "--lora_importer_output_dir",
            type=str,
            required=False,
            default=None,
            help="Directory to store all LoRA Importer output artifacts",
        )
        lora_importer_group.add_argument(
            "--lora_importer_float_fallback",
            action="store_true",
            required=False,
            default=False,
            help="Enable fallback to floating point in LoRA Importer",
        )
        lora_importer_group.add_argument(
            "--lora_importer_debug",
            type=int,
            required=False,
            default=-1,
            help="Run the LoRA Importer in debug mode",
        )
        lora_importer_group.add_argument(
            "--lora_importer_skip_validation",
            action="store_true",
            required=False,
            default=False,
            help="Skip validation checks in LoRA Importer",
        )
        lora_importer_group.add_argument(
            "--lora_importer_dump_usecase_dlc",
            action="store_true",
            required=False,
            default=False,
            help="Dump per-usecase DLC files",
        )
        lora_importer_group.add_argument(
            "--lora_importer_dump_usecase_onnx",
            action="store_true",
            required=False,
            default=False,
            help="Dump per-usecase ONNX models",
        )
        lora_importer_group.add_argument(
            "--lora_importer_skip_apply_graph_transforms",
            action="store_true",
            required=False,
            default=False,
            help="Skip applying graph transforms in LoRA Importer",
        )

        # Top-level LoRA arguments
        lora_general_group = self._parser.add_argument_group(
            "LoRA General Arguments", "General LoRA arguments for execution"
        )
        lora_general_group.add_argument(
            "--use_case_names",
            type=str,
            nargs="+",
            required=False,
            default=None,
            help="List of LoRA use cases to run (space-separated). Used to create binary_updates file for net runner.",
        )
        lora_general_group.add_argument(
            "--lora_alpha_tensor",
            type=str,
            required=False,
            default=None,
            help="Path to the LoRA alpha tensor raw file. This tensor will be automatically prepended as the first input "
            "to both calibration_input_list and input_list. Format: path/to/lora_alpha.raw",
        )

        # Add adapter_weight_config_file to offline_prepare_args for standalone usage
        offline_prepare_args.add_argument(
            "--adapter_weight_config_file",
            type=str,
            required=False,
            default=None,
            help="Path to LoRA adapter weight config file (lora_output_files.yaml) for context binary generation",
        )

    def _get_converter_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validate and build converter args from the parsed args

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            ConverterInputArguments: The converter arguments.
        """
        # Validate desired input shape argument
        if args.desired_input_shape:
            tensor_list = []
            for tensor in args.desired_input_shape:
                input_shape = ""
                input_datatype = "float32"
                input_layout = None
                if len(tensor) > 1:
                    input_shape = tensor[1]
                if len(tensor) > 2:
                    input_datatype = tensor[2]
                if len(tensor) > 3:
                    input_layout = tensor[3]
                tensor_list.append(
                    InputTensorConfig(
                        name=tensor[0],
                        source_model_input_shape=input_shape,
                        source_model_input_datatype=input_datatype,
                        source_model_input_layout=input_layout,
                    )
                )
            args.desired_input_shape = tensor_list

        # Update output tensor argument
        if args.output_tensor:
            output_tensors = []
            for tensor in args.output_tensor:
                output_tensors.append(OutputTensorConfig(name=tensor))
            args.output_tensor = output_tensors

        # Update converter_op_package argument
        if args.converter_op_package_lib:
            args.converter_op_package_lib = args.converter_op_package_lib.split(",")

        # Update onnx_define_symbol argument
        if args.onnx_define_symbol:
            symbols = []
            for symbol in args.onnx_define_symbol:
                symbols.append((symbol[0], int(symbol[1])))
            args.onnx_define_symbol = symbols
        # Define converter_args by including all the arguments that are passed to the converter
        converter_args = ConverterInputArguments(
            input_tensors=args.desired_input_shape,
            output_tensors=args.output_tensor,
            float_bitwidth=args.converter_float_bitwidth,
            float_bias_bitwidth=args.float_bias_bitwidth,
            quantization_overrides=args.quantization_overrides,
            onnx_define_symbol=args.onnx_define_symbol,
            onnx_defer_loading=args.onnx_defer_loading,
            enable_framework_trace=args.enable_framework_trace,
            op_package_config=args.op_package_config,
            converter_op_package_lib=args.converter_op_package_lib,
            package_name=args.package_name,
            onnx_simplification=not args.disable_onnx_simplification,
        )

        # Consume extra_converter_args if provided
        if args.extra_converter_args:
            qairt_debugger_arg_map = self._get_parser_arguments(group_name="converter arguments")

            # converter has some arguments with modified names
            del qairt_debugger_arg_map["converter_float_bitwidth"]

            qairt_debugger_arg_map["float_bitwidth"] = "converter_float_bitwidth"

            converter_args = parse_extra_args(
                extra_args=args.extra_converter_args,
                qairt_debugger_arg_map=qairt_debugger_arg_map,
                qairt_module_input_args=converter_args,
            )

        return converter_args

    def _get_quantizer_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validate and build quantizer arguments from the parsed arguments.

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            QuantizerInputArguments: The quantizer arguments.
        """
        # In case of calibration input list or float_fallback passed, create quantizer_args
        # to be passed to the quantizer
        if args.calibration_input_list or args.float_fallback:
            # Update restrict_quantizaiton_steps argument
            if args.restrict_quantization_steps:
                args.restrict_quantization_steps = args.restrict_quantization_steps.split()
            # Update op_package_lib argument
            if args.op_package_lib:
                args.op_package_lib = args.op_package_lib.split(",")
            quantizer_args = QuantizerInputArguments(
                input_list=args.calibration_input_list,
                bias_bitwidth=args.bias_bitwidth,
                act_bitwidth=args.act_bitwidth,
                weights_bitwidth=args.weights_bitwidth,
                float_bitwidth=args.quantizer_float_bitwidth,
                act_quantizer_calibration=args.act_quantizer_calibration,
                param_quantizer_calibration=args.param_quantizer_calibration,
                act_quantizer_schema=args.act_quantizer_schema,
                param_quantizer_schema=args.param_quantizer_schema,
                percentile_calibration_value=args.percentile_calibration_value,
                use_per_channel_quantization=args.use_per_channel_quantization,
                use_per_row_quantization=args.use_per_row_quantization,
                float_fallback=args.float_fallback,
                algorithms=args.quantization_algorithms,
                restrict_quantization_steps=args.restrict_quantization_steps,
                dump_encoding_json=args.dump_encodings_json,
                ignore_encodings=args.ignore_encodings,
                op_package_lib=args.op_package_lib,
            )
            # Consume extra_quantizer_args if provided
            if args.extra_quantizer_args:
                qairt_debugger_arg_map = self._get_parser_arguments(
                    group_name="quantizer arguments"
                )

                # quantizer has some arguments with modified names
                del qairt_debugger_arg_map["quantizer_float_bitwidth"]
                del qairt_debugger_arg_map["calibration_input_list"]

                qairt_debugger_arg_map["float_bitwidth"] = "quantizer_float_bitwidth"
                qairt_debugger_arg_map["input_list"] = "calibration_input_list"

                quantizer_args = parse_extra_args(
                    extra_args=args.extra_quantizer_args,
                    qairt_debugger_arg_map=qairt_debugger_arg_map,
                    qairt_module_input_args=quantizer_args,
                )

            # V2 encodings does not dump encodings
            if quantizer_args.use_quantize_v2:
                quantizer_args.dump_encoding_json = False

            return quantizer_args

        else:
            return None

    def _get_context_bin_and_net_run_args(self, args: argparse.Namespace) -> tuple:
        """Validate and build context bin and net run arguments from the parsed arguments.

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            tuple: A tuple containing (context_bin_args, net_run_args) where each can be None.
                - context_bin_args (GenerateConfig | None): The context bin arguments.
                - net_run_args (NetRunnerInputArguments | None): The net run arguments.
        """
        context_bin_args = None
        net_run_args = None
        if args.set_output_tensors and not args.backend == BackendType.AIC:
            args.set_output_tensors = list(
                map(Helper.transform_node_names, args.set_output_tensors)
            )
        # Create context_bin_args if offline_prepare is enabled
        if args.offline_prepare:
            context_bin_args = GenerateConfig(
                profiling_level=args.profiling_level,
                set_output_tensors=args.set_output_tensors,
                enable_intermediate_outputs=not args.set_output_tensors,
                op_packages=args.op_packages,
            )

            # Add LoRA adapter weight config file if provided
            if getattr(args, 'adapter_weight_config_file', None):
                context_bin_args.adapter_weight_config_file = args.adapter_weight_config_file

            # Consume extra_context_bin_args if provided
            if args.extra_context_bin_args:
                qairt_debugger_arg_map = self._get_parser_arguments(
                    group_name="offline prepare arguments"
                )

                del qairt_debugger_arg_map["offline_prepare_backend_extension_config"]
                qairt_debugger_arg_map["config_file"] = "offline_prepare_backend_extension_config"

                context_bin_args = parse_extra_args(
                    extra_args=args.extra_context_bin_args,
                    qairt_debugger_arg_map=qairt_debugger_arg_map,
                    qairt_module_input_args=context_bin_args,
                )

        # Create net_run_args if input_list/input_sample is provided
        if args.input_list or getattr(args, "input_sample", None):
            # When offline_prepare is enabled, disable debug and set_output_tensors
            # as enable_intermediate_outputs is set in context_bin_gen args
            use_debug = not (args.offline_prepare or args.set_output_tensors)
            output_tensors = args.set_output_tensors if not args.offline_prepare else None

            net_run_args = NetRunnerInputArguments(
                perf_profile=args.perf_profile,
                profiling_level=args.profiling_level,
                set_output_tensors=output_tensors,
                debug=use_debug,
                op_packages=args.op_packages,
            )

            # Consume extra_net_run_args if provided
            if args.extra_net_run_args:
                qairt_debugger_arg_map = self._get_parser_arguments(group_name="netrun arguments")

                del qairt_debugger_arg_map["netrun_backend_extension_config"]
                qairt_debugger_arg_map["config_file"] = "netrun_backend_extension_config"

                net_run_args = parse_extra_args(
                    extra_args=args.extra_net_run_args,
                    qairt_debugger_arg_map=qairt_debugger_arg_map,
                    qairt_module_input_args=net_run_args,
                )

        return (context_bin_args, net_run_args)

    def _get_lora_model_creator_args(self, args: argparse.Namespace) -> LoRAModelCreatorInputConfig:
        """Validate and build LoRA Model Creator arguments from the parsed arguments.

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            LoRAModelCreatorInputConfig | None: The LoRA Model Creator arguments or None if not provided.
        """
        # Only create config if lora_config is provided (required for LoRA Model Creator)
        if args.lora_config:
            return LoRAModelCreatorInputConfig(
                lora_config=args.lora_config,
                output_dir=args.lora_output_dir,
                quant_updatable_mode=args.lora_quant_updatable_mode,
                debug=args.lora_debug,
                skip_validation=args.lora_skip_validation,
                dump_usecase_onnx=args.lora_dump_usecase_onnx,
                transforms_metadata=args.lora_transforms_metadata,
            )
        return None

    def _get_lora_importer_args(self, args: argparse.Namespace) -> LoRAImporterInputConfig:
        """Validate and build LoRA Importer arguments from the parsed arguments.

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            LoRAImporterInputConfig | None: The LoRA Importer arguments or None if not provided.
        """
        # Create config if any LoRA Importer argument is provided
        if any(
            [
                args.lora_importer_config,
                args.lora_importer_input_dlc,
                args.lora_importer_input_network,
                args.lora_importer_input_list,
                args.lora_importer_output_dir,
                args.lora_importer_float_fallback,
                args.lora_importer_debug != -1,
                args.lora_importer_skip_validation,
                args.lora_importer_dump_usecase_dlc,
                args.lora_importer_dump_usecase_onnx,
                args.lora_importer_skip_apply_graph_transforms,
            ]
        ):
            return LoRAImporterInputConfig(
                lora_importer_config=args.lora_importer_config,
                input_dlc=args.lora_importer_input_dlc,
                input_network=args.lora_importer_input_network,
                input_list=args.lora_importer_input_list,
                output_dir=args.lora_importer_output_dir,
                float_fallback=args.lora_importer_float_fallback,
                debug=args.lora_importer_debug,
                skip_validation=args.lora_importer_skip_validation,
                dump_usecase_dlc=args.lora_importer_dump_usecase_dlc,
                dump_usecase_onnx=args.lora_importer_dump_usecase_onnx,
                skip_apply_graph_transforms=args.lora_importer_skip_apply_graph_transforms,
            )
        return None

    def _verify_and_update_parsed_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validates and updates parsed args
        Args:
            args (argparse.Namespace): parsed arguments
        Returns:
            argparse.Namespace: Verified and updated arguments
        """
        args = super()._verify_and_update_parsed_args(args)

        args.converter_args = self._get_converter_args(args)
        args.quantizer_args = self._get_quantizer_args(args)

        # Parse LoRA arguments
        args.lora_model_creator_args = self._get_lora_model_creator_args(args)
        args.lora_importer_args = self._get_lora_importer_args(args)

        # Update op_package argument which is passed to context_bin and net_run args
        if args.op_packages:
            op_packages = []
            for op_package in args.op_packages.split(","):
                if ":" not in op_package:
                    raise ValueError(
                        f"Invalid op_package format: {op_package}. Expected format: 'package_path:interface_provider'"
                    )
                package_path, interface = op_package.rsplit(":", 1)
                op_packages.append(
                    OpPackageIdentifier(package_path=package_path, interface_provider=interface)
                )
            args.op_packages = op_packages

        args.context_bin_args, args.net_run_args = self._get_context_bin_and_net_run_args(args)

        # For QNX platform, create DeviceCredentials object and pass the username and password arguments
        device_credentials = None
        if args.platform and DevicePlatformType(args.platform) == DevicePlatformType.QNX:
            device_credentials = (
                DeviceCredentials(username=args.username, password=args.password)
                if args.username
                else None
            )

        # Create RemoteHostDetails object and pass the serial_id, IP address and device credentials
        # as arguments
        if args.device_id or args.ip_address or device_credentials:
            args.remote_host_details = RemoteHostDetails(
                identifier=RemoteDeviceIdentifier(
                    serial_id=args.device_id, ip_addr=args.ip_address
                ),
                credentials=device_credentials,
            )
        else:
            args.remote_host_details = None

        # Update backend to BackendType object
        if args.backend:
            args.backend = BackendType(args.backend)

        # Update platform to DevicePlatformType object
        if args.platform:
            args.platform = DevicePlatformType(args.platform)

        return args

    def _get_parser_arguments(self, group_name: str) -> dict:
        """Returns all argument names in the parser and part of the given group name
        Args:
            group_name: Parser group name for which all argument names are needed
        Returns:
            dict: dictionary of argument names as keys and values.
        """
        argument_names = set()
        for group in self._parser._action_groups:
            if group.title == group_name:
                for action in group._group_actions:
                    _argument_names = list(
                        map(
                            lambda option_str: option_str.strip("-"),
                            action.option_strings,
                        )
                    )
                    argument_names.update(_argument_names)
        return {argument_name: argument_name for argument_name in argument_names}


def parse_extra_args(
    extra_args: str,
    qairt_debugger_arg_map: dict,
    qairt_module_input_args: ConverterInputArguments
    | QuantizerInputArguments
    | GenerateConfig
    | NetRunnerInputArguments,
) -> Any:
    """Parses extra args provided in the string form
    Args:
        extra_args: extra args
        Possible values an argument can take:
        1. single value: value1 or store_true
        2. list of values: [value1, value2, value3]
        3. list of list: [[value11, value12], [value21, value22]]
        e.g: 'arg1=value1;arg2;arg3=value1 value2 value3;arg4=value11 value12;arg4=value21 value22'
    Returns:
        dictionary of str as key representing argument and value being anything
    """
    # If extra_args is empty string such as: "  ", "", " "
    if not extra_args.strip():
        raise ValueError(
            "Empty extra arguments passed. Please pass a valid extra args else remove it."
        )

    parsed_extra_args = {}
    arguments = extra_args.strip().split(";")
    for argument in arguments:
        argument = argument.strip()
        # If argument is empty
        if not argument:
            raise ValueError(f"extra_args: {extra_args} has an empty argument. Please validate.")

        argument = argument.split("=")
        argument_name = argument[0].strip()
        # If argument name is empty
        if not argument_name:
            raise ValueError(
                f"extra_args: {extra_args} has invalid argument: {'='.join(argument)}, Please validate."
            )

        # According to argument length:
        # len=1: argument has no value, it is treated as argument as type boolean True
        # len>2: Invalid argument, arg=val, there must be only one "=" for an argument
        # len=2: Valid argument, arg=val or arg=val1 val2 val3
        if len(argument) == 1:
            argument_value = True
        elif len(argument) > 2:
            raise ValueError(
                f"extra_args: {extra_args} has invalid argument: {'='.join(argument)}, Please validate."
            )
        else:
            argument_value = argument[1].strip().split(" ")

        # An argument value can be nested, for e.g:[[val11, val12, val13], [val21, val22, val23]]
        # which can be passed as: "arg1=val11 val12 val13;arg1=val21 val22 val23"
        if argument_name not in parsed_extra_args:
            parsed_extra_args[argument_name] = [argument_value]
        else:
            parsed_extra_args[argument_name].append(argument_value)

    for argument_name, argument_value in parsed_extra_args.items():
        if argument_name in qairt_debugger_arg_map:
            raise ValueError(
                f"Argument '{argument_name}' is already exposed directly, please use the corresponding "
                f"argument: '--{qairt_debugger_arg_map[argument_name]}' instead. "
                f"Remove it from extra_args."
            )

        # Incase argument value is incorrect, let the underlying input module handle the error
        if hasattr(qairt_module_input_args, argument_name):
            if len(argument_value) == 1:
                # [["value"]], [True], [["val1", "val2", "val3"]]
                if isinstance(argument_value[0], list):
                    if len(argument_value[0]) == 1:
                        argument_value = argument_value[0][0]
                    else:
                        argument_value = argument_value[0]
                else:
                    argument_value = argument_value[0]
            setattr(qairt_module_input_args, argument_name, argument_value)
        else:
            module_name = type(qairt_module_input_args).__name__
            raise ValueError(
                f"Argument '{argument_name}' is not supported for {module_name}. "
                f"Please check the module documentation for valid arguments."
            )

    return qairt_module_input_args
