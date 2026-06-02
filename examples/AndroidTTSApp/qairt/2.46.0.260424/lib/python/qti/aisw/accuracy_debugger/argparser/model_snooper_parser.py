# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
import json
import os

from qti.aisw.accuracy_debugger.argparser.framework_runner_parser import FrameworkRunnerParser
from qti.aisw.accuracy_debugger.argparser.inference_engine_parser import InferenceEngineParser
from qti.aisw.accuracy_debugger.common_config import NetRunnerInputArguments
from qti.aisw.accuracy_debugger.lora import (
    LoRAModelCreatorInputConfig,
    LoRAImporterInputConfig,
)
from qti.aisw.accuracy_debugger.model_snooper.config import (
    validate_quantization_params,
)
from qti.aisw.accuracy_debugger.utils.constants import (
    Algorithm,
    supported_backends,
    supported_platforms,
)
from qti.aisw.accuracy_debugger.utils.helper import InputSample
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import GenerateConfig
from qti.aisw.tools.core.utilities.comparators.common import COMPARATORS
from qti.aisw.tools.core.utilities.comparators.factory import get_comparator


class ModelSnooperParser(FrameworkRunnerParser, InferenceEngineParser):
    """This is a parser for Snooper tool"""

    def __init__(self, component="snooping"):
        super().__init__(component)

    def _initialize(self):
        """Create parser with snooper tool specific arguments"""
        # Add config file option before other arguments
        self.optional.add_argument(
            "--config",
            type=str,
            required=False,
            help="Specifies the path to a JSON configuration file that defines debugger CLI options."
            "When this option is used, no other CLI arguments should be proivded. "
            "The configuration file is required to compare two QNN backends using the oneshot algorithm.\n",
        )
        self.optional.add_argument(
            "--algorithm",
            type=Algorithm,
            required=False,
            choices=[algo.value for algo in Algorithm],
            default=Algorithm.ONESHOT,
            help="Algorithm to use to debug the model.",
        )
        super()._initialize()
        self.required.add_argument(
            "--backend",
            type=str.upper,
            required=True,
            choices=[backend.value for backend in supported_backends],
            help="Backend type for inference to be run",
        )
        self.required.add_argument(
            "--platform",
            type=str,
            required=True,
            choices=[platform.value for platform in supported_platforms],
            help="The type of device platform to be used for inference",
        )
        self.optional.add_argument("--input_list", help=argparse.SUPPRESS)

        self.optional.add_argument(
            "--golden_reference",
            required=False,
            help="The path of directory where golden reference tensor files are saved.",
        )
        self.optional.add_argument(
            "--is_qnn_golden_reference",
            action="store_true",
            required=False,
            default=False,
            help="""Specifies that outputs passed with --golden_reference are dumped by QNN.
            This option should be used only when --golden_reference is supplied.""",
        )
        self.optional.add_argument(
            "--retain_compilation_artifacts",
            action="store_true",
            required=False,
            default=False,
            help="Flag to retain compilation artifacts.",
        )
        self.optional.add_argument(
            "--comparator",
            type=COMPARATORS,
            nargs="+",
            required=False,
            choices=[comp.value for comp in COMPARATORS],
            default=[COMPARATORS.MSE],
            help="Comparator to use to compare tensors. For multiple comparators, "
            "specify as follows: --comparator mse std",
        )
        self.optional.add_argument(
            "--offline_prepare",
            action="store_true",
            required=False,
            default=None,
            help=" Boolean to indicate offline preapre of the graph",
        )
        self.optional.add_argument(
            "--debug_subgraph_inputs",
            type=str,
            default=None,
            required=False,
            help="Pass comma separated inputs for the subgraph which is to be debugged.",
        )
        self.optional.add_argument(
            "--debug_subgraph_outputs",
            type=str,
            default=None,
            required=False,
            help="Pass comma separated outputs for the subgraph which is to be debugged.",
        )
        self.optional.add_argument(
            "--skip_layer_types",
            type=lambda s: [item.strip() for item in s.split(",")],
            default=None,
            required=False,
            help="Pass comma separated op_types for the layers to be skipped."
            "Currently supported for Oneshot Layerwise snooping."
            "e.g., --skip_layer_types Conv,Relu,Add",
        )
        self.optional.add_argument(
            "--include_layer_types",
            type=lambda s: [item.strip() for item in s.split(",")],
            default=None,
            required=False,
            help="Pass comma separated op_types for the layers to be debugged."
            "Currently supported for Oneshot Layerwise snooping."
            "e.g., --include_layer_types Conv,Relu,Add",
        )
        self.optional.add_argument(
            "--compulsory_overrides",
            type=str,
            required=False,
            default=None,
            help="""The path to json file containing a compulsory override of the encodings passed.
            This is to be used with Layerwise and Cumulative Layerwise algorithm only. Maintain
            the structure of encodings as per the supplied encodings or the encodings generated by
            QNN""",
        )
        self.optional.add_argument(
            "--max_parallel_compilations",
            type=int,
            default=None,
            required=False,
            help="""The number of parallel executions of subgraphs in layerwise and cumulative
            snooping. The provided max_parallel_compilations will set as follows:
            max_parallel_compilations = min(user provided max compilations, max compilations
            supported by host device). Maximum of 16 parallel compilations supported.""",
        )

        # Supress arguments related to custom ops. Enabled them when custom op supported enabled in
        # snooping algorithms.
        self.optional.add_argument("--op_package_config", default=[], help=argparse.SUPPRESS)
        self.optional.add_argument("--converter_op_package_lib", default=[], help=argparse.SUPPRESS)
        self.optional.add_argument("--package_name", default="", type=str, help=argparse.SUPPRESS)
        self.optional.add_argument("--op_package_lib", default=[], help=argparse.SUPPRESS)
        self.optional.add_argument("--op_packages", default=[], help=argparse.SUPPRESS)

    def _get_context_bin_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validate and build context bin arguments from the parsed arguments.

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            GenerateConfig: The context bin arguments.
        """
        # In case of offline_prepare, create context_bin_args to pass to context_bin_gen
        if args.offline_prepare:
            context_bin_args = GenerateConfig(
                profiling_level=args.profiling_level,
                op_packages=args.op_packages,
            )
            return context_bin_args
        else:
            return None

    def _get_net_run_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validate and build net run arguments from the parsed arguments.

        Args:
            args (argparse.Namespace): The parsed arguments.

        Returns:
            NetRunnerInputArguments: The net run arguments.
        """
        net_run_args = NetRunnerInputArguments(
            perf_profile=args.perf_profile,
            profiling_level=args.profiling_level,
            op_packages=args.op_packages,
        )
        return net_run_args

    def parse(self, args: list) -> argparse.Namespace:
        """Parse the arguments.

        Args:
            args (list): Arguments from the user to be parsed
        Returns:
            argparse.Namespace | dict: parsed arguments
        """
        # First, create a temporary parser just to check if --config is present
        temp_parser = argparse.ArgumentParser(add_help=False)
        temp_parser.add_argument("--config", type=str)
        temp_args, _ = temp_parser.parse_known_args(args)

        if temp_args.config:
            # Config file is provided
            config_file = temp_args.config

            # Ensure no other arguments are provided except --config
            if len(args) > 2:  # More than just --config and its value
                raise argparse.ArgumentError(
                    None,
                    "When --config is provided, no other command line arguments should be specified.",
                )

            if not os.path.exists(config_file):
                raise argparse.ArgumentError(None, f"Config file '{config_file}' does not exist.")

            # Parse the config file
            with open(config_file, "r") as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError as e:
                    raise argparse.ArgumentError(None, f"Invalid JSON in config file: {str(e)}")

            return self._parse_json_config(config)

        else:
            # Normal parsing without config file
            self._initialize()
            parsed_args = self._parser.parse_args(args)
            return self._verify_and_update_parsed_args(parsed_args)

    def _parse_json_config(self, config: dict) -> dict:
        """Parse and validate a JSON configuration file.

        This method processes a configuration dictionary loaded from a JSON file,
        validates required fields, and converts the configuration into the format
        expected by the model snooper input config.

        Args:
            config (dict): Dictionary containing configuration parameters loaded from a JSON file

        Returns:
            dict: Processed configuration dictionary with validated parameters

        Raises:
            argparse.ArgumentError: If required fields are missing or invalid in the config
        """
        parsed_dict = {}

        # Optional top-level input_model (if provided)
        if "input_model" in config:
            parsed_dict["input_model"] = config["input_model"]

        # Optional algorithm (default: oneshot)
        parsed_dict["algorithm"] = config.get("algorithm", "oneshot")

        # Process the reference config JSON structure
        if "reference_config" not in config:
            raise argparse.ArgumentError(
                None, "Missing required field 'reference_config' in config file"
            )

        parsed_dict["reference_config"] = self._parse_backend_config(
            config["reference_config"], "reference"
        )

        # Process the target config JSON structure
        if "target_config" not in config:
            raise argparse.ArgumentError(
                None, "Missing required field 'target_config' in config file"
            )

        parsed_dict["target_config"] = self._parse_backend_config(config["target_config"], "target")

        # Handle required top-level field
        if "input_sample" in config:
            parsed_dict["input_sample"] = self._get_input_sample(config["input_sample"])
        else:
            raise argparse.ArgumentError(
                None, "Missing required field 'input_sample' in config file"
            )

        # Handle optional top-level fields
        parsed_dict["comparators"] = (
            [get_comparator(comp) for comp in config["comparators"]]
            if "comparators" in config
            else [get_comparator(COMPARATORS.MSE)]
        )

        parsed_dict["working_directory"] = (
            config["working_directory"] if "working_directory" in config else None
        )

        parsed_dict["retain_compilation_artifacts"] = config.get(
            "retain_compilation_artifacts", False
        )

        parsed_dict["dump_output_tensors"] = config.get("dump_output_tensors", True)

        parsed_dict["log_level"] = config.get("log_level", "info")

        # Handle LoRA-specific fields
        parsed_dict["lora_alpha_tensor"] = config.get("lora_alpha_tensor", None)
        parsed_dict["use_case_names"] = config.get("use_case_names", None)

        # Parse LoRA Model Creator arguments if present
        if "lora_model_creator_args" in config:
            parsed_dict["lora_model_creator_args"] = self._parse_lora_model_creator_args(
                config["lora_model_creator_args"]
            )
        else:
            parsed_dict["lora_model_creator_args"] = None

        # Parse LoRA Importer arguments if present
        if "lora_importer_args" in config:
            parsed_dict["lora_importer_args"] = self._parse_lora_importer_args(
                config["lora_importer_args"]
            )
        else:
            parsed_dict["lora_importer_args"] = None

        return argparse.Namespace(**parsed_dict)

    def _parse_backend_config(self, backend_config: dict, config_type: str) -> dict:
        """Parse backend configuration from JSON.

        Args:
            backend_config: Backend config dictionary
            config_type: "reference" or "target" (for error messages)

        Returns:
            Parsed backend config dictionary
        """
        parsed = {}

        # Required fields
        required_fields = ["backend", "platform"]
        for field in required_fields:
            if field not in backend_config:
                raise argparse.ArgumentError(
                    None, f"Missing required field '{field}' in {config_type}_config"
                )

        parsed["backend"] = backend_config["backend"].upper()
        parsed["platform"] = backend_config["platform"]

        # Optional: dlc_file
        if "dlc_file" in backend_config:
            parsed["dlc_file"] = backend_config["dlc_file"]

        # Optional: soc_model
        if "soc_model" in backend_config:
            parsed["soc_model"] = backend_config["soc_model"]

        # Optional: remote_host_details
        if "remote_host_details" in backend_config:
            parsed["remote_host_details"] = backend_config["remote_host_details"]

        # Optional: Compilation arguments (only if dlc_file not provided)
        if "converter_arguments" in backend_config:
            parsed["converter_arguments"] = backend_config["converter_arguments"]

        if "quantizer_arguments" in backend_config:
            parsed["quantizer_arguments"] = backend_config["quantizer_arguments"]

        if "context_bin_gen_arguments" in backend_config:
            parsed["context_bin_gen_arguments"] = backend_config["context_bin_gen_arguments"]

        if "context_bin_backend_extension" in backend_config:
            parsed["context_bin_backend_extension"] = backend_config[
                "context_bin_backend_extension"
            ]

        if "offline_prepare" in backend_config:
            parsed["offline_prepare"] = backend_config["offline_prepare"]

        # Optional: Execution arguments
        if "net_run_arguments" in backend_config:
            parsed["net_run_arguments"] = backend_config["net_run_arguments"]

        if "net_run_backend_extension" in backend_config:
            parsed["net_run_backend_extension"] = backend_config["net_run_backend_extension"]

        # Legacy support for netrun_backend_extension_config
        if "netrun_backend_extension_config" in backend_config:
            parsed["net_run_backend_extension"] = backend_config["netrun_backend_extension_config"]

        return parsed

    def _get_input_sample(self, input_sample: list) -> list[InputSample]:
        """Parse input sample configuration from JSON into InputSample objects.

        Args:
            input_sample (list): List of dictionaries containing input sample configurations
                                with keys like 'name', 'raw_file', 'dimensions', and optionally 'data_type'

        Returns:
            list[InputSample]: List of InputSample objects created from the input configurations
        """
        input_sample_objs = []

        for item in input_sample:
            input_sample_objs.append(
                InputSample(
                    name=item["name"],
                    raw_file=item["raw_file"],
                    dimensions=item["dimensions"],
                    data_type=item["data_type"] if "data_type" in item else "float32",
                )
            )
        return input_sample_objs

    def _verify_and_update_parsed_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validates and updates parsed args
        Args:
            args (argparse.Namespace): parsed arguments
        Returns:
            argparse.Namespace: Verified and updated arguments
            - Sets offline_prepare=True if backend supports it
            - Sets float_fallback=True if quantization_overrides provided without calibration_input_list
        """
        backend = BackendType(args.backend)
        # Enable offline prepare if backend supports.
        if args.offline_prepare is None and backend in BackendType.offline_preparable_backends():
            args.offline_prepare = True

        # Either quantization_overrides or calibration_input_list should be provided
        validate_quantization_params(
            args.calibration_input_list, args.quantization_overrides, args.algorithm, backend
        )

        # If user has provided quantization_overrides and no calibration data then
        # make float_fallback True
        if args.quantization_overrides and not args.calibration_input_list:
            args.float_fallback = True

        args = super()._verify_and_update_parsed_args(args)

        # Update comparator by getting the relevant comparator class
        args.comparator = [get_comparator(comp) for comp in args.comparator]

        if args.is_qnn_golden_reference and args.golden_reference is None:
            raise Exception(
                "--is_qnn_golden_reference is allowed only when --golden_reference is supplied."
            )

        # Parse debug_subgraph_input and outputs
        if args.debug_subgraph_inputs:
            args.debug_subgraph_inputs = args.debug_subgraph_inputs.split(",")
        if args.debug_subgraph_outputs:
            args.debug_subgraph_outputs = args.debug_subgraph_outputs.split(",")

        # Validate lora_alpha_tensor is provided when LoRA is enabled
        lora_config = getattr(args, "lora_config", None)
        lora_alpha_tensor = getattr(args, "lora_alpha_tensor", None)
        if lora_config and not lora_alpha_tensor:
            raise argparse.ArgumentError(
                None,
                "--lora_alpha_tensor is required when --lora_config is provided for LoRA snooping.",
            )

        return args

    def _parse_lora_model_creator_args(self, lora_config: dict) -> LoRAModelCreatorInputConfig:
        """Parse LoRA Model Creator arguments from config dictionary.

        Args:
            lora_config (dict): Dictionary containing LoRA Model Creator configuration

        Returns:
            LoRAModelCreatorInputConfig: Parsed LoRA Model Creator configuration object
        """
        return LoRAModelCreatorInputConfig(
            lora_config=lora_config.get("lora_config"),
            output_dir=lora_config.get("output_dir"),
            quant_updatable_mode=lora_config.get("quant_updatable_mode", "adapter_only"),
            debug=lora_config.get("debug", -1),
            skip_validation=lora_config.get("skip_validation", False),
            dump_usecase_onnx=lora_config.get("dump_usecase_onnx", False),
            transforms_metadata=lora_config.get("transforms_metadata"),
        )

    def _parse_lora_importer_args(self, lora_importer_config: dict) -> LoRAImporterInputConfig:
        """Parse LoRA Importer arguments from config dictionary.

        Args:
            lora_importer_config (dict): Dictionary containing LoRA Importer configuration

        Returns:
            LoRAImporterInputConfig: Parsed LoRA Importer configuration object
        """
        return LoRAImporterInputConfig(
            lora_importer_config=lora_importer_config.get("lora_importer_config"),
            input_dlc=lora_importer_config.get("input_dlc"),
            input_network=lora_importer_config.get("input_network"),
            input_list=lora_importer_config.get("input_list"),
            output_dir=lora_importer_config.get("output_dir"),
            float_fallback=lora_importer_config.get("float_fallback", False),
            debug=lora_importer_config.get("debug", -1),
            skip_validation=lora_importer_config.get("skip_validation", False),
            dump_usecase_dlc=lora_importer_config.get("dump_usecase_dlc", False),
            dump_usecase_onnx=lora_importer_config.get("dump_usecase_onnx", False),
            skip_apply_graph_transforms=lora_importer_config.get(
                "skip_apply_graph_transforms", False
            ),
        )
