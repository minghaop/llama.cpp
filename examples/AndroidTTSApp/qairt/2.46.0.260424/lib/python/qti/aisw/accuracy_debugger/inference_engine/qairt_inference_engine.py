# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from pydantic import (
    DirectoryPath,
    Field,
    FilePath,
    model_validator,
)
from qairt.api.configs.device import Device
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    NetRunnerInputArguments,
    QuantizerInputArguments,
    RemoteHostDetails,
)

# LoRA-specific imports
from qti.aisw.accuracy_debugger.lora import (
    LoRAImporterInputConfig,
    LoRAImporterOutputConfig,
    LoRAModelCreatorInputConfig,
    LoRAModelCreatorOutputConfig,
)
from qti.aisw.accuracy_debugger.utils.constants import (
    HTP_SUPPORTED_PLATFORMS,
    InferenceStatus,
    supported_backends,
    supported_platforms,
)
from qti.aisw.accuracy_debugger.utils.exceptions import (
    ConversionFailure,
    ExecutionFailure,
    GenerateBinaryFailure,
    OptimizationFailure,
    ParameterError,
    QuantizationFailure,
    SerializationFailure,
)
from qti.aisw.accuracy_debugger.utils.file_utils import dump_json, read_json
from qti.aisw.accuracy_debugger.utils.helper import DebuggerConfig
from qti.aisw.tools.core.modules.api.definitions.common import (
    BackendType,
    ModelConfig,
    Target,
)
from qti.aisw.tools.core.modules.context_bin_gen import (
    GenerateConfig,
    context_bin_gen_module,
)
from qti.aisw.tools.core.modules.converter import (
    BackendInfoConfig,
    converter_module,
    optimizer_module,
    quantizer_module,
    serializer_module,
)
from qti.aisw.tools.core.modules.net_runner import net_runner_module
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.utils.constants import (
    OnnxFrameworkInfo,
    PytorchFrameworkInfo,
    TensorflowFrameworkInfo,
    TFLiteFrameworkInfo,
)
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


# Constants for binary updates file versioning
BINARY_UPDATE_VERSION = 1


def sanitize_dict(argument: dict) -> dict:
    for arg, value in argument.items():
        argument[arg] = str(value)

    return argument


def sanitize_keys_of_dict(input_dict: dict) -> dict:
    """Sanitize dictionary keys by transforming node names to a standardized format.

    Args:
        input_dict: Dictionary with potentially unsanitized keys (e.g., tensor names from
                   framework models or inference outputs)

    Returns:
        dict: New dictionary with sanitized keys and original values preserved

    Example:
        >>> input_dict = {"tensor:0": np.array([1, 2, 3]), "layer/output": np.array([4, 5, 6])}
        >>> sanitized = sanitize_keys_of_dict(input_dict)
        >>> # Keys are transformed to standardized format while values remain unchanged

    Note:
        - Original dictionary is not modified; a new dictionary is returned
        - Only keys are transformed; values are preserved as-is
        - Commonly used for sanitizing tensor names in inference inputs/outputs
    """
    sanitized_input_dict = {
        Helper.transform_node_names(key): value for key, value in input_dict.items()
    }
    return sanitized_input_dict


def validate_backend_platform(backend, platform, offline_prepare) -> None:
    """Validations for combination of backend and platform."""
    if platform and platform not in supported_platforms:
        raise ValueError(
            f"Platform type {platform} is not supported."
            f"Supported platforms are {supported_platforms}"
        )
    if (
        platform in (DevicePlatformType.QNX, DevicePlatformType.LINUX_EMBEDDED)
        and offline_prepare is False
    ):
        raise ValueError(
            f"Offline graph preparation is mandatory for platform type {platform}, "
            f"but is set as {offline_prepare}"
        )
    if backend:
        if backend not in supported_backends:
            raise ValueError(
                f"Backend type {backend} is not supported."
                f"Supported backends are {supported_backends}"
            )
        if offline_prepare and backend not in BackendType.offline_preparable_backends():
            raise ValueError(
                f"Offline graph preparation is unsupported for {backend} backend."
                f"Supported backends are {BackendType.offline_preparable_backends()}"
            )

    if backend and platform:
        if backend == BackendType.AIC and platform != DevicePlatformType.X86_64_LINUX:
            raise ValueError(f"AIC backend is unsupported for {platform} platform.")

        if backend == BackendType.HTP and platform not in HTP_SUPPORTED_PLATFORMS.values():
            raise ValueError(
                f"HTP backend is unsupported for {platform} platform."
                f" Supported platforms are {HTP_SUPPORTED_PLATFORMS.values()}"
            )

        if backend == BackendType.GPU and platform != DevicePlatformType.ANDROID:
            raise ValueError(
                f"GPU backend is supported only for android platform but {platform} platform given."
            )


def validate_platform_remote_host_details(platform, remote_host_details):
    """Validations for combination of platform and remote host details."""
    if platform and remote_host_details:
        ip_address = (
            remote_host_details.identifier.ip_addr if remote_host_details.identifier else None
        )
        credentials = remote_host_details.credentials
        if platform == DevicePlatformType.QNX:
            if ip_address is None:
                raise ValueError(f"IP address not provided for platform {platform}")
            if credentials is None:
                raise ValueError(f"User credentials not provided for platform {platform}")


def validate_float_fallback(
    converter_arguments: ConverterInputArguments, quantizer_arguments: QuantizerInputArguments
):
    """Validations for converter and quantizer arguments
    Args:
        converter_arguments: ConverterInputArguments object containing args for converter
        quantizer_arguments: QuantizerInputArguments object containing args for quantizer

    Raises:
        ValueError: If float_fallback is True and quantization_overrides is None
    """
    quantization_overrides = (
        converter_arguments.quantization_overrides if converter_arguments else None
    )
    if quantizer_arguments:
        if quantizer_arguments.float_fallback and not quantization_overrides:
            raise ValueError(
                "External quantization overrides must be provided when using 'float_fallback'."
            )


def create_working_directory():
    """Create a default working directory for inference engine"""
    path = Path.cwd() / "inference_engine_work_dir"
    path.mkdir(exist_ok=True)
    return path


class InferenceEngineInputConfig(DebuggerConfig):
    """Input configuration class for Inference Engine

    Attributes:
        input_model: Path to the source model/dlc/bin file
        converter_arguments: Input arguments required by the converter module
        quantizer_arguments: Input arguments required by quantizer module
        backend: Backend type for inference to be run
        platform: The type of device platform to be used for inference
        context_bin_gen_arguments: Input arguments required by the context_bin_gen_module
        context_bin_backend_extension: Backend extension config for context_bin_gen_module
        offline_prepare: Boolean to indicate offline prepare of graph
        net_run_arguments: Input arguments required by the net_run module
        net_run_input_data: Input data to net-runner
        net_run_backend_extension: Backend extension config for net-runner
        dump_output: Enable to dump the results of the netrun into a raw file
        remote_host_details: Details and credentials of the remote host
        working_directory: Path to the directory to store artifacts and outputs.
        soc_model : Name of SOC model on target device.
        lora_config: Path to LoRA configuration file (enables LoRA mode)
        use_case_names: List of LoRA use cases to run
        skip_lora_validation: Skip LoRA configuration validation
        quant_updatable_mode: Quantization updatable mode for LoRA adapters
        dump_lora_artifacts: Dump intermediate LoRA artifacts for debugging
    """

    _source_model: bool
    input_model: FilePath
    converter_arguments: Optional[ConverterInputArguments] = None
    quantizer_arguments: Optional[QuantizerInputArguments] = None
    backend: Optional[BackendType] = None
    platform: Optional[DevicePlatformType] = None
    context_bin_gen_arguments: Optional[GenerateConfig] = None
    context_bin_backend_extension: Optional[FilePath | dict] = None
    offline_prepare: Optional[bool] = False
    net_run_arguments: Optional[NetRunnerInputArguments] = None
    net_run_input_data: Optional[net_runner_module.NetRunnerInputData] = None
    net_run_backend_extension: Optional[FilePath | dict] = None
    dump_output: Optional[bool] = False
    remote_host_details: Optional[RemoteHostDetails] = None
    working_directory: DirectoryPath = Field(default_factory=create_working_directory)
    soc_model: Optional[str] = ""

    # LoRA-specific attributes
    lora_model_creator_args: Optional[LoRAModelCreatorInputConfig] = None
    lora_importer_args: Optional[LoRAImporterInputConfig] = None
    use_case_names: Optional[list[str]] = None
    lora_alpha_tensor: Optional[FilePath] = None

    @model_validator(mode="after")
    def validate_input_model(self):
        """Validation for the type of input_model provided based on arguments in input config"""
        self._source_model = False
        try:
            if self.input_model.suffix.lower() not in [".dlc", ".bin"]:
                FrameworkManager.infer_framework_type(self.input_model)
                self._source_model = True
        except Exception:
            if self.converter_arguments:
                raise ValueError(
                    "Invalid source model for converter. Support model format are "
                    f"{OnnxFrameworkInfo.name, TensorflowFrameworkInfo.name},"
                    f"{TFLiteFrameworkInfo.name} and {PytorchFrameworkInfo.name}"
                )

        if not self._source_model:
            input_model_suffix = self.input_model.suffix
            if self.quantizer_arguments or self.offline_prepare:
                if input_model_suffix != ".dlc":
                    raise ValueError(
                        "DLC file should be given for quantization or offline prepare but, "
                        f"'{input_model_suffix}' file given."
                    )
            else:
                if input_model_suffix not in [".bin", ".dlc"]:
                    raise ValueError(
                        "'.bin' or '.dlc' file is expected for net-run but, "
                        f"'{input_model_suffix}' file given."
                    )

        return self

    @model_validator(mode="after")
    def validate_arguments(self):
        """Validation for the backend and platform provided"""
        validate_backend_platform(self.backend, self.platform, self.offline_prepare)
        if self.offline_prepare:
            # backend is need to prepare offline graph
            if not self.backend:
                raise ValueError("Backend is required to prepare offline graph")
            if self.context_bin_gen_arguments:
                if (
                    self.context_bin_gen_arguments.enable_intermediate_outputs
                    and self.context_bin_gen_arguments.set_output_tensors
                ):
                    raise ValueError(
                        "Either enable_intermediate_outputs or set_output_tensors must be set at a "
                        "time in contex binary module"
                    )
        else:
            if self.context_bin_gen_arguments:
                raise ValueError(
                    "Context binary generation arguments should supplied only when offline prepare "
                    "is enabled"
                )

        if self.net_run_input_data:
            # backend and platform are needed for
            if not self.backend:
                raise ValueError("Backend is required to execute graph")
            if not self.platform:
                raise ValueError("Platform is required to execute graph")
            if self.net_run_arguments:
                if self.net_run_arguments.debug and self.net_run_arguments.set_output_tensors:
                    raise ParameterError(
                        "Either debug or set_output_tensors parameter should be set at a time in "
                        "net-runner module."
                    )
                if self.net_run_arguments.debug or self.net_run_arguments.set_output_tensors:
                    if self.offline_prepare:
                        raise ParameterError(
                            "In offline prepare, the debug or set_output_tensors parameters for "
                            "netrunner should not be set"
                        )
        else:
            if self.net_run_arguments:
                raise ValueError(
                    "Net run input data should be supplied when net_run_arguments are supplied"
                )
        return self

    @model_validator(mode="after")
    def validate_float_fallback(self):
        """Validation for the float fallback"""
        validate_float_fallback(self.converter_arguments, self.quantizer_arguments)
        return self

    @model_validator(mode="after")
    def validate_platform_remote_host_details(self):
        """Validation for platform and remote host details."""
        validate_platform_remote_host_details(self.platform, self.remote_host_details)
        return self

    @model_validator(mode="after")
    def validate_lora_importer_args(self):
        """Validate LoRA importer arguments based on model creator presence."""
        if self.lora_importer_args:
            if self.lora_model_creator_args:
                # Scenario 1: After model creator - certain fields should not be provided
                forbidden = [
                    "lora_config",
                    "input_dlc",
                    "input_network",
                    "input_list",
                    "output_dir",
                ]
                for field in forbidden:
                    if getattr(self.lora_importer_args, field, None) is not None:
                        raise ValueError(
                            f"'{field}' should not be provided in lora_importer_args "
                            f"when lora_model_creator_args is provided. "
                            f"It will be auto-populated from model creator output."
                        )
            else:
                # Scenario 2: Standalone importer - required fields must be provided
                required = ["lora_config", "input_dlc", "input_network", "input_list"]
                for field in required:
                    if getattr(self.lora_importer_args, field, None) is None:
                        raise ValueError(
                            f"'{field}' is required in lora_importer_args "
                            f"when lora_model_creator_args is not provided."
                        )
        return self


class InferenceEngineOutputConfig(DebuggerConfig):
    """Output configuration class for Inference Engine

    Attributes:
        output_data: Inference output data. Can be either:
            - Standard mode: list[dict[str, np.ndarray]] - List of output dictionaries
            - LoRA mode: Dict[str, list[dict[str, np.ndarray]]] - Dictionary mapping model names to lists of output dictionaries
        output_dir: Path to the dumped raw files when dump_output is set.
        converter_dlc: Path to the generated DLC after conversion
        quantizer_dlc: Path to the quantized DLC
        offline_graph: Path to the generated context binary
        lora_model_creator_output: Output from LoRA Model Creator
        lora_importer_output: Output from LoRA Importer
        lora_adapter_binaries: Dictionary mapping use case names to adapter binary paths
    """

    output_data: Optional[
        Union[list[dict[str, np.ndarray]], Dict[str, list[dict[str, np.ndarray]]]]
    ] = None
    output_dir: Optional[DirectoryPath] = None
    converter_dlc: Optional[FilePath] = None
    quantizer_dlc: Optional[FilePath] = None
    offline_graph: Optional[FilePath] = None

    # LoRA-specific outputs
    lora_model_creator_output: Optional[LoRAModelCreatorOutputConfig] = None
    lora_importer_output: Optional[LoRAImporterOutputConfig] = None
    lora_adapter_binaries: Optional[Dict[str, Dict[str, FilePath]]] = None

    def cleanup_artifacts(self) -> None:
        """Delete the artifacts generated by the inference engine."""
        import shutil

        # Delete LoRA Model Creator artifacts
        if self.lora_model_creator_output:
            output_dir = self.lora_model_creator_output.output_directory
            if output_dir:
                output_dir_path = Path(output_dir)
                if output_dir_path.exists():
                    shutil.rmtree(output_dir_path, ignore_errors=False)

        # Delete LoRA Importer artifacts
        if self.lora_importer_output:
            output_dir = self.lora_importer_output.output_directory
            if output_dir:
                output_dir_path = Path(output_dir)
                if output_dir_path.exists():
                    shutil.rmtree(output_dir_path, ignore_errors=False)

        # Delete LoRA adapter binaries
        if self.lora_adapter_binaries:
            for graph_name, use_case_dict in self.lora_adapter_binaries.items():
                for use_case_name, binary_path in use_case_dict.items():
                    if binary_path:
                        binary_path_obj = Path(binary_path)
                        if binary_path_obj.exists():
                            binary_path_obj.unlink(missing_ok=True)

        # Delete standard artifacts (converter_dlc, quantizer_dlc, offline_graph)
        attr = self.model_dump()
        exclude_keys = [
            "output_data",
            "output_dir",
            "lora_model_creator_output",
            "lora_importer_output",
            "lora_adapter_binaries",
        ]

        for k, v in attr.items():
            if k not in exclude_keys and v:
                if isinstance(v, (Path, str)):
                    artifact_path = Path(v)
                    if artifact_path.exists():
                        artifact_path.unlink(missing_ok=True)

class InferenceEngine:
    """User interface class for model inference.
    Contains methods to convert, quantize, generate_binary and execute the model,
    based on the backend and platform provided in the InferenceEngineInputConfig.
    """

    def __init__(self, logger: Any = None) -> None:
        """Initialize InferenceEngine with LoRA support
        Args:
            logger (Any): Desired python logger
        """
        if logger:
            self.logger = logger
        else:
            self.log_area = LogAreas.register_log_area("Inference")
            self.logger = QAIRTLogger.register_area_logger(area=self.log_area, level="INFO")
        self.status_dict = {}

        # LoRA-specific attributes
        self.lora_pipeline = None
        self.lora_imporer_output_artifacts = None

    def run(self, config: InferenceEngineInputConfig, **kwargs) -> InferenceEngineOutputConfig:
        """Execute Inference Engine
        Args:
            config: InferenceEngineInputConfig object containing args for inference

        Returns:
            InferenceEngineOutputConfig: Compilation artifacts and inference results
        """
        # Get netrun_lock object from kwargs, if passed
        netrun_lock = kwargs.get("netrun_lock", None)

        # Initialize the inference engine execution status dict
        status_file_path = config.working_directory / "status.json"
        if status_file_path.exists():
            self.status_dict = read_json(status_file_path)
        else:
            self.status_dict = {
                "status": InferenceStatus.SUCCESS.value,
                "converter_dlc_path": "",
                "quantizer_dlc_path": "",
                "context_binary_bin_path": "",
                "tensor_info": {},
                "result_dir": "",
                "msg": "",
                "arguments": {
                    "converter": {},
                    "optimizer": {},
                    "serializer": {},
                    "quantizer": {},
                    "context_bin": {},
                    "netrun": {},
                },
                "InferenceEngine_Input_Arguments": sanitize_dict(config.model_dump()),
            }
        try:
            input_model = config.input_model
            result = InferenceEngineOutputConfig()

            # Check if any LoRA configuration is provided
            lora_enabled = (
                config.lora_model_creator_args is not None
                or config.lora_importer_args is not None
                or (
                    config.context_bin_gen_arguments
                    and config.context_bin_gen_arguments.adapter_weight_config_file is not None
                )
            )

            # Initialize LoRA pipeline if needed (for both model creator and importer)
            if lora_enabled:
                self._initialize_lora_pipeline(config.working_directory)

            # Step 1: Execute LoRA Model Creator (if arguments provided)
            if config.lora_model_creator_args:
                self.logger.info("Executing LoRA Model Creator...")
                lora_model_creator_output = self.lora_pipeline.run_lora_model_creator(
                    config.lora_model_creator_args
                )
                result.lora_model_creator_output = lora_model_creator_output
                # Update input_model to use the generated base model
                input_model = lora_model_creator_output.base_model_path

            # Step 2: Convert model to DLC (if source model)
            backend_info = None
            if config.backend:
                backend_info = BackendInfoConfig(
                    backend=config.backend.value, soc_model=config.soc_model
                )

            if config._source_model:
                self.logger.info("Converting model to DLC...")
                output_dlc_path = os.path.join(
                    str(config.working_directory), "lora_base.dlc" if lora_enabled else "base.dlc"
                )

                # Create enhanced converter configuration with LoRA tensor names if available
                converter_config = config.converter_arguments or ConverterInputArguments()
                lora_tensor_names_file = None
                if result.lora_model_creator_output:
                    # Extract LoRA-specific files from model creator output
                    lora_tensor_names_file = result.lora_model_creator_output.lora_tensor_names_file
                    quantization_overrides = result.lora_model_creator_output.base_encodings_path

                    # Set lora_weight_list if tensor names file exists
                    if lora_tensor_names_file and lora_tensor_names_file.exists():
                        converter_config.lora_weight_list = lora_tensor_names_file
                        self.logger.debug(
                            f"Converter using LoRA tensor names file: {lora_tensor_names_file}"
                        )

                    # Set quantization_overrides if encodings file exists
                    if quantization_overrides and quantization_overrides.exists():
                        converter_config.quantization_overrides = quantization_overrides
                        self.logger.debug(
                            f"Converter using quantization encoding file: {quantization_overrides}"
                        )

                # Set quant_updatable_mode if LoRA model creator args are provided
                quant_updatable_mode = None
                if config.lora_model_creator_args:
                    quant_updatable_mode = config.lora_model_creator_args.quant_updatable_mode
                    converter_config.quant_updatable_mode = quant_updatable_mode
                    self.logger.debug(
                        f"Converter using LoRA quant_updatable_mode: {quant_updatable_mode}"
                    )

                # Running Converter
                converter_output = self._convert(
                    model=input_model,
                    output_path=output_dlc_path,
                    converter_args=converter_config,
                    status_file_path=status_file_path,
                )

                # Running Optimizer
                optimizer_output = self._optimize(
                    converter_output=converter_output,
                    status_file_path=status_file_path,
                    backend_info=backend_info,
                )

                # Clear converter ir_graph
                del converter_output.ir_graph

                # Running serializer
                dlc_path = self._serialize(
                    optimized_graph=optimizer_output.optimized_graph,
                    optimizer_args=optimizer_output.optimizer_args,
                    output_path=output_dlc_path,
                    converter_output=converter_output,
                    status_file_path=status_file_path,
                    lora_weight_list=lora_tensor_names_file,
                    quant_updatable_mode=quant_updatable_mode,
                )

                input_model = dlc_path
                result.converter_dlc = dlc_path
                self.status_dict["converter_dlc_path"] = str(dlc_path)

            # Step 3: Quantize the DLC (if quantizer arguments provided)
            if config.quantizer_arguments:
                self.logger.info("Quantizing model...")

                # Validate LoRA alpha tensor is provided when LoRA is enabled
                if lora_enabled and config.quantizer_arguments.input_list and not config.lora_alpha_tensor:
                    raise ValueError(
                        "LoRA is enabled but --lora_alpha_tensor is not provided. "
                        "The LoRA alpha tensor is required for quantization when using LoRA models. "
                        "Please provide the path to the LoRA alpha tensor using --lora_alpha_tensor argument."
                    )

                # Handle LoRA alpha tensor prepending for quantization
                if config.lora_alpha_tensor and config.quantizer_arguments.input_list:
                    self.logger.info(
                        f"LoRA alpha tensor provided: {config.lora_alpha_tensor}. "
                        "Prepending it to calibration input list for quantization."
                    )
                    # Create modified calibration input list with LoRA alpha prepended
                    modified_calibration_list = self._prepend_lora_alpha_to_input_list(
                        config.quantizer_arguments.input_list,
                        config.lora_alpha_tensor,
                        config.working_directory,
                        "calibration"
                    )
                    # Update quantizer arguments with modified input list
                    config.quantizer_arguments.input_list = str(modified_calibration_list)
                    self.logger.debug(
                        f"Modified calibration input list created at: {modified_calibration_list}"
                    )

                # VALIDATION: Check quantizer input_list matches DLC inputs
                if config.quantizer_arguments.input_list:
                    num_dlc_inputs, dlc_input_names = self._inspect_dlc_inputs(Path(input_model))

                    if num_dlc_inputs > 0:  # Only validate if DLC inspection succeeded
                        num_provided_inputs = self._count_inputs_in_file(
                            Path(config.quantizer_arguments.input_list)
                        )

                        # Validate input count matches
                        if num_provided_inputs != num_dlc_inputs:
                            error_msg = f"Input count mismatch for quantization: DLC expects {num_dlc_inputs} inputs but calibration input_list provides {num_provided_inputs} inputs."

                            if lora_enabled:
                                error_msg += "\n\nLoRA Model Detected: LoRA models typically add an additional input tensor (usually for alpha values) as the first input."
                                error_msg += "\nPlease ensure your calibration input_list includes data for ALL model inputs, including any LoRA-specific inputs."
                                error_msg += f"\nDLC input names: {dlc_input_names}"
                                error_msg += (
                                    "\nRefer to LoRA documentation for the expected input format."
                                )

                            raise ValueError(error_msg)
                        else:
                            self.logger.debug(
                                f"Quantizer input validation passed: {num_provided_inputs} inputs provided for {num_dlc_inputs} DLC inputs"
                            )

                quantized_dlc_path = os.path.join(
                    str(config.working_directory),
                    "lora_base_quantized.dlc" if lora_enabled else "base_quantized.dlc",
                )
                if config.backend and config.backend not in BackendType.quantizable_backends():
                    backend_info = None

                # Running Quantizer
                quantized_dlc = self._quantize(
                    input_dlc=input_model,
                    output_dlc_path=quantized_dlc_path,
                    quantizer_args=config.quantizer_arguments,
                    status_file_path=status_file_path,
                    backend_info=backend_info,
                )

                # Clear converter_output and optimizer_output
                del optimizer_output
                del converter_output

                input_model = quantized_dlc
                result.quantizer_dlc = quantized_dlc
                self.status_dict["quantizer_dlc_path"] = str(quantized_dlc)

            # Step 4: Execute LoRA Importer (if arguments provided)
            if config.lora_importer_args:
                self.logger.info("Executing LoRA Importer...")

                # Validate LoRA alpha tensor is provided when LoRA is enabled
                if config.lora_importer_args.input_list and not config.lora_alpha_tensor:
                    raise ValueError(
                        "LoRA is enabled but --lora_alpha_tensor is not provided. "
                        "The LoRA alpha tensor is required for LoRA importer when using LoRA models. "
                        "Please provide the path to the LoRA alpha tensor using --lora_alpha_tensor argument."
                    )

                # Handle LoRA alpha tensor prepending for importer input list
                if config.lora_alpha_tensor and config.lora_importer_args.input_list:
                    self.logger.info(
                        f"LoRA alpha tensor provided: {config.lora_alpha_tensor}. "
                        "Prepending it to input list for LoRA importer."
                    )
                    # Create modified input list with LoRA alpha prepended
                    modified_importer_input_list = self._prepend_lora_alpha_to_input_list(
                        config.lora_importer_args.input_list,
                        config.lora_alpha_tensor,
                        config.working_directory,
                        "importer"
                    )
                    # Update importer arguments with modified input list
                    config.lora_importer_args.input_list = str(modified_importer_input_list)
                    self.logger.debug(
                        f"Modified importer input list created at: {modified_importer_input_list}"
                    )

                lora_importer_output = self.lora_pipeline.run_lora_importer(
                    config.lora_importer_args
                )
                result.lora_importer_output = lora_importer_output

            # Step 5: Generate context binary (if offline_prepare enabled)
            if config.offline_prepare:
                self.logger.info("Generating context binary...")
                model_obj = ModelConfig(path=input_model)

                # Use adapter_weight_config_file from LoRA importer output if available
                context_bin_args = config.context_bin_gen_arguments or GenerateConfig()

                if context_bin_args.adapter_weight_config_file:
                    self.logger.debug(
                        f"Using LoRA adapter weight config file: {context_bin_args.adapter_weight_config_file}"
                    )

                # Running offline prepare
                model_obj = self._generate_binary(
                    model_obj=model_obj,
                    output_dir=config.working_directory,
                    backend=config.backend,
                    status_file_path=status_file_path,
                    context_bin_args=context_bin_args,
                    context_bin_backend_extension=config.context_bin_backend_extension,
                )

                result.offline_graph = model_obj.path
                input_model = model_obj.path
                self.status_dict["context_binary_bin_path"] = str(model_obj.path)

                # Scan for adapter binaries after context binary generation (LoRA-specific)
                if context_bin_args.adapter_weight_config_file:
                    adapter_binaries = self._scan_for_adapter_binaries(
                        config.working_directory, context_bin_args.adapter_weight_config_file
                    )
                    if adapter_binaries:
                        result.lora_adapter_binaries = adapter_binaries
                        self.logger.debug(
                            f"Found {len(adapter_binaries)} adapter binaries: {list(adapter_binaries.keys())}"
                        )
                    else:
                        result.lora_adapter_binaries = None
                        self.logger.warning(
                            "LoRA adapter_weight_config_file provided but no adapter binaries found after context binary generation"
                        )
                else:
                    result.lora_adapter_binaries = None

                # clear context binary outputs
                del model_obj

            # Step 6: Execute net runner (if input data provided)
            if config.net_run_input_data:
                self.logger.info("Executing net runner...")
                if config.platform is None:
                    raise ValueError("platform is mandatory for inference")

                # Validate LoRA alpha tensor is provided when LoRA is enabled
                if lora_enabled and isinstance(config.net_run_input_data, str) and not config.lora_alpha_tensor:
                    raise ValueError(
                        "LoRA is enabled but --lora_alpha_tensor is not provided. "
                        "The LoRA alpha tensor is required for net run when using LoRA models. "
                        "Please provide the path to the LoRA alpha tensor using --lora_alpha_tensor argument."
                    )

                # Handle LoRA alpha tensor prepending for net run input data
                if config.lora_alpha_tensor and isinstance(config.net_run_input_data, str):
                    self.logger.info(
                        f"LoRA alpha tensor provided: {config.lora_alpha_tensor}. "
                        "Prepending it to input list for net run."
                    )
                    # Create modified input list with LoRA alpha prepended
                    modified_input_list = self._prepend_lora_alpha_to_input_list(
                        config.net_run_input_data,
                        config.lora_alpha_tensor,
                        config.working_directory,
                        "netrun"
                    )
                    # Update net run input data with modified input list
                    config.net_run_input_data = str(modified_input_list)
                    self.logger.debug(
                        f"Modified net run input list created at: {modified_input_list}"
                    )

                model_obj = ModelConfig(path=input_model)
                target = self._create_target(config.platform, config.remote_host_details)

                # Enhanced net runner configuration with binary_updates (LoRA-specific)
                net_runner_args = config.net_run_arguments or NetRunnerInputArguments()

                # Create binary_updates file if use_case_names provided and adapter binaries available
                if config.use_case_names and result.lora_adapter_binaries:
                    binary_updates_file = self._create_binary_updates_file(config, result)
                    if binary_updates_file:
                        net_runner_args.binary_updates = binary_updates_file
                        self.logger.debug(f"Using LoRA binary_updates file: {binary_updates_file}")

                # Running net run
                netrun_output = self._execute(
                    model_obj,
                    config.working_directory,
                    config.backend,
                    target,
                    config.net_run_input_data,
                    status_file_path,
                    net_runner_args,
                    config.net_run_backend_extension,
                    netrun_lock=netrun_lock,
                )

                # Handle different output formats based on whether binary_updates was used
                if net_runner_args.binary_updates:
                    # LoRA mode: netrun_output is dict of {model_name: [list of output dicts]}
                    self.logger.debug("Processing LoRA netrun output with binary_updates")

                    # Sanitize output tensor names for each model's outputs
                    for model_name, model_outputs in netrun_output.items():
                        sanitized_model_outputs = []
                        for idx, output_dict in enumerate(model_outputs):
                            sanitized_output_dict = sanitize_keys_of_dict(output_dict)
                            sanitized_model_outputs.append(sanitized_output_dict)
                        netrun_output[model_name] = sanitized_model_outputs
                        self.logger.debug(
                            f"Sanitized outputs for model '{model_name}': {len(sanitized_model_outputs)} output sets"
                        )

                    if config.dump_output:
                        result.output_dir = self._dump_lora_inference_outputs(
                            netrun_output, config.working_directory
                        )
                else:
                    # Standard mode: netrun_output is list of output dicts
                    self.logger.debug("Processing standard netrun output")

                    # Sanitize output tensor names obtained from netrun
                    for idx, output_dict in enumerate(netrun_output):
                        sanitized_output_dict = {}
                        for key, value in output_dict.items():
                            sanitized_output_name = Helper.transform_node_names(key)
                            sanitized_output_dict[sanitized_output_name] = value
                            self.status_dict["tensor_info"][sanitized_output_name] = {}
                            self.status_dict["tensor_info"][sanitized_output_name]["shape"] = (
                                value.shape
                            )
                            self.status_dict["tensor_info"][sanitized_output_name]["dtype"] = (
                                value.dtype.name
                            )

                        netrun_output[idx] = sanitized_output_dict

                    if config.dump_output:
                        result.output_dir = self._dump_inference_outputs(
                            netrun_output, config.working_directory
                        )

                self.status_dict["result_dir"] = str(result.output_dir)
                result.output_data = netrun_output

            dump_json(self.status_dict, status_file_path)
            self.logger.debug("Inference engine completed successfully!")
            return result

        except Exception as e:
            self.logger.error(f"Inference engine failed: {e}")
            raise e

    def _create_target(
        self, platform_type: DevicePlatformType, remote_host_details: RemoteHostDetails
    ) -> Target:
        """Get module variant of Target class
        Args:
            platform (Target): Target parameter of type DevicePlatformType
        Returns:
            module_target: Module defined Target object
        """
        if remote_host_details:
            identifier = remote_host_details.identifier
            credentials = remote_host_details.credentials
            return Target(type=platform_type, identifier=identifier, credentials=credentials)
        return Target(type=platform_type)

    def _dump_inference_outputs(
        self,
        inference_outputs: list[net_runner_module.NamedTensorMapping],
        output_path: Path,
    ) -> Path:
        """Dump the generated outputs into raw file
        Args:
            inference_outputs: Netrunner output of type list[net_runner_module.NamedTensorMapping]
            output_path: Path to dump the raw outputs
        Returns:
            Path: Path to output directory
        """
        output_path = output_path / "Output"
        output_path.mkdir(parents=True, exist_ok=True)
        for idx, output_dict in enumerate(inference_outputs):
            base_dir = output_path / f"Result_{idx}"
            base_dir.mkdir(parents=True, exist_ok=True)
            for output_name, out_tensor in output_dict.items():
                """
                Most of the systems does not allow to create a file name, more than 255 bytes,
                hence skipping to dump those tensor files.
                """
                if len((output_name + ".raw").encode("utf-8")) > 255:
                    self.logger.warning(
                        f"Skipping output tensor '{output_name}' as filename exceeds 255 bytes"
                    )
                    continue
                out_tensor.tofile(base_dir / f"{output_name}.raw")
        return output_path

    def _dump_lora_inference_outputs(
        self,
        lora_inference_outputs: Dict[str, list[net_runner_module.NamedTensorMapping]],
        output_path: Path,
    ) -> Path:
        """Dump the generated LoRA outputs into raw files organized by model name
        Args:
            lora_inference_outputs: LoRA netrunner output of type Dict[str, list[NamedTensorMapping]]
                                   Format: {model_name: [list of output dicts]}
            output_path: Path to dump the raw outputs
        Returns:
            Path: Path to output directory
        """
        output_path = output_path / "Output"
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, model_outputs in lora_inference_outputs.items():
            # Create a directory for each model (e.g., 'base', 'lora_base_function')
            model_dir = output_path / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            for idx, output_dict in enumerate(model_outputs):
                # Create subdirectory for each result set within the model
                result_dir = model_dir / f"Result_{idx}"
                result_dir.mkdir(parents=True, exist_ok=True)

                for output_name, out_tensor in output_dict.items():
                    """
                    Most of the systems does not allow to create a file name, more than 255 bytes,
                    hence skipping to dump those tensor files.
                    """
                    if len((output_name + ".raw").encode("utf-8")) > 255:
                        self.logger.warning(
                            f"Skipping output tensor '{output_name}' for model '{model_name}' as filename exceeds 255 bytes"
                        )
                        continue
                    out_tensor.tofile(result_dir / f"{output_name}.raw")

            self.logger.debug(
                f"Dumped {len(model_outputs)} result sets for model '{model_name}' to {model_dir}"
            )

        self.logger.debug(
            f"Dumped LoRA inference outputs for {len(lora_inference_outputs)} models to {output_path}"
        )
        return output_path

    def _convert(
        self,
        model: Path,
        output_path: Path,
        converter_args: ConverterInputArguments,
        status_file_path: Path,
    ) -> converter_module.ConverterOutputConfig:
        """Perform model conversion
        Args:
            model: Path to the source framework model
            output_path: Path where the converted output model should be saved
            converter_args: ConverterInputArguments object containing arguments for conversion
            status_file_path: Path to status json file to store the failure status
            self.status_dict: Dictionary to store InferenceEngine execution status

        Returns:
            ConverterOutputConfig: Object with IR graph and framework of source model

        Raises:
            ConversionFailure: If conversion model fails
        """
        try:
            self.logger.debug("Converting source model to IR")
            if converter_args is None:
                converter_args = converter_module.ConverterInputConfig(input_network=model)
            else:
                converter_args = converter_module.ConverterInputConfig(
                    input_network=model,
                    **converter_args.model_dump(exclude_unset=True),
                )
            converter_args.output_path = str(output_path)
            self.logger.debug(f"Conversion parameters: {converter_args.model_dump()}")
            self.status_dict["arguments"]["converter"] = sanitize_dict(
                converter_args.model_dump(exclude_unset=True)
            )
            converter = converter_module.QAIRTConverter(logger=self.logger)
            converter_output = converter.convert(converter_args)
            self.logger.debug("Completed converting to IR")
        except Exception as exception:
            exc_msg = f"Failed to convert the model! {exception}"
            self.status_dict["status"] = InferenceStatus.ConversionFailure.value
            self.status_dict["msg"] = exc_msg
            dump_json(self.status_dict, status_file_path)
            raise ConversionFailure(exc_msg) from exception
        return converter_output

    def _optimize(
        self,
        converter_output: converter_module.ConverterOutputConfig,
        status_file_path: Path,
        backend_info: Optional[BackendInfoConfig] = None,
    ) -> Any:
        """Perform model optimization
        Args:
            converter_output : ConverterOutputConfig object containing irgraph and framework
            backend_info: backend specific information required for backend aware optimization.
            status_file_path: Path to status json file to store the failure status
            self.status_dict: Dictionary to store InferenceEngine execution status

        Returns:
            Any: Optimizer output object containing optimized graph and optimizer arguments.

        Raises:
            OptimizationFailure: If optimization of IRgraph fails
        """
        try:
            self.logger.debug("Optimizing IR graph")
            ir_graph = converter_output.ir_graph
            framework = converter_output.framework
            optimizer_args = optimizer_module.OptimizerInputConfig(
                ir_graph=ir_graph,
                framework=framework,
                backend_info=backend_info,
            )
            self.logger.debug(f"Optimization parameters: {optimizer_args.model_dump()}")
            self.status_dict["arguments"]["optimizer"] = sanitize_dict(
                optimizer_args.model_dump(exclude_unset=True)
            )
            optimizer = optimizer_module.QAIRTOptimizer(logger=self.logger)
            optimizer_output = optimizer.optimize(optimizer_args)
            self.logger.debug("Completed optimization of IR graph")
        except Exception as exception:
            exc_msg = f"Failed to optimize the graph! {exception}"
            self.status_dict["status"] = InferenceStatus.OptimizationFailure.value
            self.status_dict["msg"] = exc_msg
            dump_json(self.status_dict, status_file_path)
            raise OptimizationFailure(exc_msg) from exception
        return optimizer_output

    def _serialize(
        self,
        optimized_graph: Any,
        optimizer_args: Dict[str, Any],
        output_path: str | Path,
        converter_output: converter_module.ConverterOutputConfig,
        status_file_path: Path,
        lora_weight_list: Optional[FilePath] = None,
        quant_updatable_mode: Optional[Literal["none", "adapter_only", "all"]] = None,
    ) -> str:
        """Perform model serialization using the QAIRTSerializer.

        Args:
            optimized_graph (Any): The optimized (IR) graph to be serialized.
            optimizer_args (Dict[str, Any]): The optimizer arguments used for optimization.
            output_path (str | Path): The file path where the serialized DLC model will be saved.
            converter_output (ConverterOutputConfig): The ConverterOutputConfig object containing
                the DLC backend config and framework.
            status_file_path: Path to status json file to store the failure status
            self.status_dict: Dictionary to store InferenceEngine execution status

        Returns:
            str: The path of serialized DLC model.

        Raises:
            SerializationFailure: If serialization of the graph fails.
        """
        try:
            self.logger.debug("Serializing IR graph")
            qairt_serializer = serializer_module.QAIRTSerializer(logger=self.logger)
            serializer_args = serializer_module.SerializerInputConfig(
                optimized_graph=optimized_graph,
                optimizer_args=optimizer_args,
                output_dlc=str(output_path),
                dlc_backend_config=converter_output.dlc_backend_config,
                framework=converter_output.framework,
                lora_weight_list=lora_weight_list,
                quant_updatable_mode=quant_updatable_mode,
            )
            self.status_dict["arguments"]["serializer"] = sanitize_dict(
                serializer_args.model_dump(exclude_unset=True)
            )
            serializer_output = qairt_serializer.serialize(config=serializer_args)
            self.logger.debug("Completed serialization of graph")
        except Exception as exception:
            exc_msg = f"Failed to serialize the graph! {exception}"
            self.status_dict["status"] = InferenceStatus.SerializationFailure.value
            self.status_dict["msg"] = exc_msg
            dump_json(self.status_dict, status_file_path)
            raise SerializationFailure(exc_msg) from exception
        return serializer_output.dlc_path

    def _quantize(
        self,
        input_dlc: Path,
        output_dlc_path: Path,
        quantizer_args: QuantizerInputArguments,
        status_file_path: Path,
        backend_info: Optional[BackendInfoConfig] = None,
    ) -> str:
        """Perform model quantization
        Args:
            output_dlc_path : File path to be used for saving the Quantized DLC
            input_dlc : Path to the DLC file that needs to be quantized
            quantizer_args: Arguments required for quantization
            status_file_path: Path to status json file to store the failure status
            self.status_dict: Dictionary to store InferenceEngine execution status
            backend_info: backend specific information required for backend aware quantization.

        Returns:
            str: Path to the quantized DLC

        Raises:
            QuantizationFailure: If quantization of DLC fails
        """
        try:
            self.logger.debug("Performing quantization")
            quant_args = quantizer_module.QuantizerInputConfig(
                input_dlc=input_dlc,
                **quantizer_args.model_dump(exclude_unset=True),
                output_dlc=str(output_dlc_path),
                backend_info=backend_info,
            )
            self.logger.debug(f"Quantization parameters: {quant_args.model_dump()}")
            self.status_dict["arguments"]["quantizer"] = sanitize_dict(
                quant_args.model_dump(exclude_unset=True)
            )
            quantizer = quantizer_module.QAIRTQuantizer(logger=self.logger)
            quantizer_output = quantizer.quantize(quant_args)
            quantized_dlc = quantizer_output.dlc_output
            self.logger.debug(f"Quantized graph is saved at {quantized_dlc}")
            self.logger.debug("Completed quantization")
        except Exception as exception:
            exc_msg = f"Failed to quantize the graph! {exception}"
            self.status_dict["status"] = InferenceStatus.QuantizationFailure.value
            self.status_dict["msg"] = exc_msg
            dump_json(self.status_dict, status_file_path)
            raise QuantizationFailure(exc_msg) from exception
        return quantized_dlc

    def _generate_binary(
        self,
        model_obj: ModelConfig,
        output_dir: Path,
        backend: BackendType,
        status_file_path: Path,
        context_bin_args: Optional[GenerateConfig] = None,
        context_bin_backend_extension: Optional[Path | dict] = None,
    ) -> ModelConfig:
        """Perform offline graph preparation to generate binary

        Args:
            model_obj: Module defined ModelConfig object having path to a .dlc file set
            output_dir: output path to export the generated context_bin file
            backend: The backend to use for execution (e.g., CPU, GPU, etc.).
            status_file_path: Path to status json file to store the failure status
            self.status_dict: Dictionary to store InferenceEngine execution status
            context_bin_backend_extension: Backend extension configuration file or dictionary.
            context_bin_args: Arguments for context_bin_generation
        Returns:
            ModelConfig: ModelConfig object with the path to the generated context binary

        Raises:
            ContextBinGenerationFailure: If offline preparation graph fails
        """
        try:
            self.logger.debug("Preparing offline graph")
            input_config = context_bin_gen_module.ContextBinGenArgConfig(
                backend=backend,
                model=model_obj,
                output_dir=output_dir,
                generate_config=context_bin_args,
            )
            self.logger.debug(f"Offline graph prepare parameters: {input_config.model_dump()}")
            if context_bin_backend_extension:
                if isinstance(context_bin_backend_extension, dict):
                    input_config.backend_config_dict = context_bin_backend_extension
                else:
                    input_config.backend_config_file = context_bin_backend_extension
            self.status_dict["arguments"]["context_bin"] = sanitize_dict(
                input_config.model_dump(exclude_unset=True)
            )

            context_bin_gen = context_bin_gen_module.ContextBinGen(logger=self.logger)
            output_config = context_bin_gen.generate(input_config)
            offline_graph = output_config.context_binary
            self.logger.debug(f"Offline graph saved at {offline_graph}")
            self.logger.debug("Completed offline graph preparation")
        except Exception as exception:
            exc_msg = f"Failed to generate binaries! {exception}"
            self.status_dict["status"] = InferenceStatus.GenerateBinaryFailure.value
            self.status_dict["msg"] = exc_msg
            dump_json(self.status_dict, status_file_path)
            raise GenerateBinaryFailure(exc_msg) from exception
        return offline_graph

    def _scan_for_adapter_binaries(
        self, working_directory: Path, adapter_weight_config_file: Path
    ) -> Dict[str, Dict[str, Path]]:
        """Scan working directory for adapter binary files created by context binary generation.

        Args:
            working_directory: Directory where context binary generation was run
            adapter_weight_config_file: Path to the adapter weight config file (lora_output_files.yaml)

        Returns:
            Dict mapping graph names to dict of use case names to their corresponding binary file paths
            Format: {graph_name: {use_case_name: binary_file_path}}
        """
        adapter_binaries = {}

        try:
            import yaml

            # Load the adapter weight config file to get use case information
            with open(adapter_weight_config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # Extract use case information from the config
            use_cases = config_data.get("use_case", [])

            for use_case in use_cases:
                use_case_name = use_case.get("name")
                graph_name = use_case.get("graph")

                if use_case_name and graph_name:
                    # The naming convention for adapter binaries: {graph_name}_{use_case_name}.bin
                    binary_file_path = working_directory / f"{graph_name}_{use_case_name}.bin"

                    if binary_file_path.exists():
                        # Initialize graph_name dict if it doesn't exist
                        if graph_name not in adapter_binaries:
                            adapter_binaries[graph_name] = {}

                        adapter_binaries[graph_name][use_case_name] = binary_file_path
                        self.logger.debug(
                            f"Found adapter binary for graph '{graph_name}', use case '{use_case_name}': {binary_file_path}"
                        )
                    else:
                        self.logger.warning(
                            f"Expected adapter binary not found: {binary_file_path}"
                        )

        except Exception as e:
            self.logger.error(f"Failed to scan for adapter binaries: {e}")

        return adapter_binaries

    def _execute(
        self,
        model_config: ModelConfig,
        output_dir: Path,
        backend: BackendType,
        target: Target,
        net_run_input_data: net_runner_module.NetRunnerInputData,
        status_file_path: Path,
        net_run_args: Optional[NetRunnerInputArguments] = None,
        net_run_backend_extension: Optional[Path | dict] = None,
        netrun_lock=None,
    ) -> Union[list[dict[str, np.ndarray]], Dict[str, list[dict[str, np.ndarray]]]]:
        """Perform model inference
        Args:
            model_obj: Module defined ModelConfig object having either .bin or .dlc path
            output_dir: Output path to dump artifacts like, profiling logs, generated during
                        inference
            backend: The backend to use for execution (e.g., CPU, GPU, etc.)
            target: The target platform for execution (e.g.,android)
            net_run_args: Arguments for net-runner
            net_run_input_data: Inputs for inference
            status_file_path: Path to status json file to store the failure status
            self.status_dict: Dictionary to store InferenceEngine execution status
            net_run_backend_config_file: Backend extension config for net-runner
            netrun_lock: Object of multiprocessing.Lock to enable synchronized ondevice model execution

        Returns:
            Union[list[dict[str, np.ndarray]], Dict[str, list[dict[str, np.ndarray]]]]:
                Standard mode: list of output dictionaries
                LoRA mode (with binary_updates): dictionary mapping model names to lists of output dictionaries

        Raises:
            InferenceFailure: if inference fails
        """
        self.status_dict["arguments"]["netrun"] = {}
        successful_netrun_load = False
        netrun_lock_acquired = False
        try:
            # Create NetRunner Object
            net_runner = net_runner_module.NetRunner(logger=self.logger)

            self.logger.debug(
                f"Running inference for {backend} backend on {target.type.value} target"
            )

            # Load the artifacts on the device
            if target.type in (DevicePlatformType.ANDROID, DevicePlatformType.LINUX_EMBEDDED):
                device_obj = Device(type=DevicePlatformType.ANDROID)
                target.soc_model = device_obj.get_chipset()
                self.logger.debug(f"SOC Model found: {target.soc_model}")
            identifier = net_runner_module.InferenceIdentifier(
                model=model_config, target=target, backend=backend
            )

            net_runner_load_config = net_runner_module.NetRunnerLoadArgConfig(
                identifier=identifier,
                inference_config=net_run_args,
                output_dir=str(output_dir),
            )

            if net_run_backend_extension is not None:
                if isinstance(net_run_backend_extension, dict):
                    net_runner_load_config.backend_config_dict = net_run_backend_extension
                else:
                    net_runner_load_config.backend_config_file = net_run_backend_extension

            self.logger.debug(f"Net-run load parameters: {net_runner_load_config.model_dump()}")
            self.status_dict["arguments"]["netrun"]["load"] = sanitize_dict(
                net_runner_load_config.model_dump(exclude_unset=True)
            )

            net_runner_load_output_config = net_runner.load(config=net_runner_load_config)
            successful_netrun_load = True

            # Create unload config
            net_run_unload_config = net_runner_module.NetRunnerUnloadArgConfig(
                handle=net_runner_load_output_config.handle, output_dir=str(output_dir)
            )

            # Sanitize the input names only when net_run_input_data is a dictionary
            if isinstance(net_run_input_data, dict):
                net_run_input_data = sanitize_keys_of_dict(net_run_input_data)
            # Execute the graph
            net_run_execute_config = net_runner_module.NetRunnerRunArgConfig(
                identifier=net_runner_load_output_config.handle,
                output_dir=str(output_dir),
                input_data=net_run_input_data,
            )
            self.logger.debug(f"Net-run execute parameters: {net_run_execute_config.model_dump()}")
            self.status_dict["arguments"]["netrun"]["execute"] = sanitize_dict(
                net_run_execute_config.model_dump(exclude_unset=True)
            )

            # If netrun lock is passed, acquire it before executing the ondevice inference
            if netrun_lock:
                netrun_lock.acquire()
                netrun_lock_acquired = True
            output_config = net_runner.run(net_run_execute_config)
            self.logger.debug("Completed inference")

        except Exception as exception:
            if netrun_lock and netrun_lock_acquired:
                netrun_lock.release()
            exc_msg = f"Failed to execute the model! {exception}"
            self.status_dict["status"] = InferenceStatus.ExecutionFailure.value
            self.status_dict["msg"] = exc_msg
            dump_json(self.status_dict, status_file_path)
            # Unload
            if successful_netrun_load:
                self.logger.debug(
                    f"Net-run unload parameters: {net_run_unload_config.model_dump()}"
                )
                self.status_dict["arguments"]["netrun"]["unload"] = sanitize_dict(
                    net_run_unload_config.model_dump(exclude_unset=True)
                )
                net_runner.unload(config=net_run_unload_config)
            raise ExecutionFailure(exc_msg) from exception

        # If netrun_lock is passed and has been acquired, release it
        if netrun_lock and netrun_lock_acquired:
            netrun_lock.release()

        # Unload
        if successful_netrun_load:
            self.logger.debug(f"Net-run unload parameters: {net_run_unload_config.model_dump()}")
            self.status_dict["arguments"]["netrun"]["unload"] = sanitize_dict(
                net_run_unload_config.model_dump(exclude_unset=True)
            )
            net_runner.unload(config=net_run_unload_config)

        return output_config.output_data

    def _initialize_lora_pipeline(self, working_directory: Path):
        """Initialize LoRA pipeline components when LoRA config is provided."""
        try:
            from qti.aisw.accuracy_debugger.lora.lora_pipeline_manager import LoRAPipelineManager

            self.lora_pipeline = LoRAPipelineManager(
                working_directory=working_directory, logger=self.logger
            )
            self.logger.info("LoRA pipeline initialized")
        except ImportError as e:
            self.logger.error(f"LoRA support not available: {e}")
            raise ValueError("LoRA configuration provided but LoRA support not available")

    def _prepend_lora_alpha_to_input_list(
        self,
        original_input_list: str,
        lora_alpha_tensor: Path,
        working_directory: Path,
        list_type: str = "input"
    ) -> Path:
        """Prepend LoRA alpha tensor to an input list file if not already present.

        Args:
            original_input_list: Path to the original input list file
            lora_alpha_tensor: Path to the LoRA alpha tensor raw file
            working_directory: Working directory to store the modified input list
            list_type: Type of list being modified ("calibration", "importer", or "netrun")

        Returns:
            Path: Path to the modified input list file with LoRA alpha prepended,
                  or original path if lora_alpha is already present
        """
        try:
            # Read the original input list
            with open(original_input_list, "r") as f:
                original_lines = f.readlines()

            # Check if lora_alpha is already present in any line
            for line in original_lines:
                line_stripped = line.strip()
                if line_stripped:
                    # Check if line contains lora_alpha.raw or lora_alpha:=
                    entries = line_stripped.split()
                    for entry in entries:
                        # Check for patterns: "lora_alpha.raw" or "lora_alpha:=..."
                        if "lora_alpha.raw" in entry or entry.startswith("lora_alpha:="):
                            self.logger.info(
                                f"LoRA alpha tensor already present in {list_type} input list. "
                                f"Skipping modification."
                            )
                            return Path(original_input_list)

            # Create modified input list file
            modified_input_list_path = working_directory / f"modified_{list_type}_list.txt"

            # Determine the format to use based on the first entry in the original list
            # Check if the first line uses named format (contains ":=") or unnamed format
            use_named_format = False
            if original_lines:
                first_line = original_lines[0].strip()
                if first_line and ":=" in first_line:
                    use_named_format = True

            # Create lora_alpha entry based on the detected format
            if use_named_format:
                lora_alpha_entry = f"lora_alpha:={lora_alpha_tensor}"
            else:
                lora_alpha_entry = str(lora_alpha_tensor)

            # Write modified input list with LoRA alpha prepended to each line
            with open(modified_input_list_path, "w") as f:
                for line in original_lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        # Prepend lora_alpha to the line
                        modified_line = f"{lora_alpha_entry} {line}\n"
                        f.write(modified_line)

            self.logger.info(
                f"Created modified {list_type} list with LoRA alpha tensor prepended: {modified_input_list_path}"
            )
            return modified_input_list_path

        except Exception as e:
            self.logger.error(f"Failed to prepend LoRA alpha tensor to {list_type} list: {e}")
            raise

    def _inspect_dlc_inputs(self, dlc_path: Path) -> tuple[int, list[str]]:
        """Inspect DLC file to get input information.

        Args:
            dlc_path: Path to the DLC file to inspect

        Returns:
            tuple: (num_inputs, input_names)
        """
        try:
            from qti.aisw.dlc_utils import modeltools

            model_reader = modeltools.IrDlcReader()
            model_reader.open(str(dlc_path))

            # Get the first graph (assuming single graph)
            graph_names = model_reader.get_ir_graph_names()

            # Convert to list if it's a set or other iterable
            if isinstance(graph_names, set):
                graph_names = list(graph_names)
            elif not isinstance(graph_names, list):
                graph_names = list(graph_names)

            if not graph_names:
                self.logger.warning(f"No graphs found in DLC {dlc_path}")
                return 0, []

            graph = model_reader.get_ir_graph(graph_names[0])

            # Get input tensors (APP_WRITE tensors)
            input_tensors = graph.get_input_tensors_to_graph()

            num_inputs = len(input_tensors)
            input_names = [tensor.name() for tensor in input_tensors]

            self.logger.debug(f"DLC {dlc_path} has {num_inputs} inputs: {input_names}")
            return num_inputs, input_names

        except Exception as e:
            self.logger.error(f"Failed to inspect DLC inputs: {e}")
            return 0, []

    def _count_inputs_in_file(self, input_file_path: Path) -> int:
        """Count number of input entries in the first line of an input list file.

        The input list file format:
        - line: Space-separated tensor files for a single inference

        Examples:
        - "file1.raw file2.raw" -> 2 inputs
        - "tensor1:=file1.raw tensor2:=file2.raw" -> 2 inputs

        Args:
            input_file_path: Path to the input list text file

        Returns:
            int: Number of tensor entries in the first line
        """
        try:
            with open(input_file_path, "r") as f:
                first_line = f.readline().strip()

            if not first_line:
                self.logger.warning(
                    f"Input file {input_file_path} is empty or has no valid first line"
                )
                return 0

            # Split by whitespace to get individual tensor entries in the first line
            entries = first_line.split()

            # Count non-empty entries
            count = len([entry for entry in entries if entry.strip()])

            self.logger.debug(
                f"Input file {input_file_path} first line: '{first_line}' -> {count} tensor entries"
            )
            return count
        except Exception as e:
            raise ValueError(f"Failed to read input file {input_file_path}: {e}")

    def _create_binary_updates_v1(self, graph_name: str, bin_files: list[str]) -> dict:
        """Create binary updates configuration for version 1 format.

        Args:
            graph_name: Name of the graph
            bin_files: List of binary file paths

        Returns:
            dict: Binary updates configuration in version 1 format
        """
        return {"version": 1, "graphs": [{graph_name: bin_files}]}

    def _create_binary_updates_file(
        self, config: InferenceEngineInputConfig, result: InferenceEngineOutputConfig
    ) -> Optional[Path]:
        """Create binary updates file for the specified use cases using adapter binaries from context binary generation."""
        try:
            import yaml

            context_bin_output_dir = config.working_directory

            if config.use_case_names and result.lora_adapter_binaries:
                # Create a binary updates YAML file for all specified use cases
                use_cases_str = "_".join(config.use_case_names)
                binary_updates_file = (
                    context_bin_output_dir / f"binary_updates_{use_cases_str}.yaml"
                )

                # Extract graph names and their use cases from lora_adapter_binaries
                # Format: {graph_name: {use_case_name: binary_file_path}}
                bin_files = []
                missing_use_cases = []
                graph_name = None

                # Find the graph name and collect binary files for specified use cases
                for graph, use_case_dict in result.lora_adapter_binaries.items():
                    if graph_name is None:
                        graph_name = graph  # Use the first graph name found

                    for use_case_name in config.use_case_names:
                        if use_case_name in use_case_dict:
                            binary_path = use_case_dict[use_case_name]
                            if binary_path.exists():
                                bin_files.append(str(binary_path))
                                self.logger.debug(
                                    f"Using adapter binary for graph '{graph}', use case '{use_case_name}': {binary_path}"
                                )
                            else:
                                missing_use_cases.append(use_case_name)
                                self.logger.warning(
                                    f"Adapter binary file not found for use case '{use_case_name}': {binary_path}"
                                )
                        else:
                            missing_use_cases.append(use_case_name)
                            self.logger.warning(
                                f"Use case '{use_case_name}' not found in adapter binaries for graph '{graph}'"
                            )

                if graph_name and bin_files:
                    # Create binary updates configuration using version-specific method
                    if BINARY_UPDATE_VERSION == 1:
                        binary_updates_config = self._create_binary_updates_v1(
                            graph_name, bin_files
                        )
                    else:
                        raise ValueError(
                            f"Unsupported binary updates version: {BINARY_UPDATE_VERSION}"
                        )

                    # Write the binary updates file
                    with open(binary_updates_file, "w") as f:
                        yaml.dump(binary_updates_config, f, default_flow_style=False)

                    self.logger.debug(f"Created binary updates file: {binary_updates_file}")
                    self.logger.debug(
                        f"Included {len(bin_files)} use case bin files for graph '{graph_name}': {bin_files}"
                    )

                    if missing_use_cases:
                        self.logger.warning(f"Some use cases not available: {missing_use_cases}")

                    return binary_updates_file
                else:
                    if not graph_name:
                        self.logger.warning("No graph name found in adapter binaries")
                    if not bin_files:
                        self.logger.warning(
                            f"No binary files available for any of the specified use cases: {config.use_case_names}"
                        )
            else:
                self.logger.warning("No use case names specified or no adapter binaries available")

        except Exception as e:
            self.logger.error(f"Failed to create binary updates file: {e}")

        return None
