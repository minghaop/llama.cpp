# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import onnx  # noqa
import json
from pathlib import Path
from typing import List, Optional, Literal
from numpy.typing import NDArray
from pydantic import DirectoryPath, Field, FilePath, field_validator, model_validator

from qti.aisw.accuracy_debugger.lora import (
    LoRAModelCreatorInputConfig,
    LoRAImporterInputConfig,
)
from qti.aisw.accuracy_debugger.encodings.encodings_utils import (
    EncodingVersion,
    get_encodings_version,
)
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.accuracy_debugger.utils.helper import DebuggerConfig
from qti.aisw.tools.core.modules.context_bin_gen import GenerateConfig
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.comparators.mse import MSEComparator
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.accuracy_debugger.framework_runner.framework_factory import get_framework_type
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    validate_backend_platform,
    validate_float_fallback,
    validate_platform_remote_host_details,
)
from qti.aisw.tools.core.modules.api import (
    AISWBaseModel,
    BackendType,
    ModuleSchema,
    ModuleSchemaVersion,
)

from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    NetRunnerInputArguments,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.utils.helper import InputSample


def create_working_directory():
    """Create a default working directory for snooper module"""
    path = Path.cwd() / "snooper_work_dir"
    path.mkdir(exist_ok=True)
    return path


def validate_encoding_json_file(file_path: Path | str, file_description: str) -> None:
    """Validates that an encoding JSON file is not empty and has required fields based on version.

    Args:
        file_path: Path to the encoding JSON file to validate.
        file_description: Description of the file (e.g., 'compulsory_overrides',
            'quantization_overrides') for use in error messages.

    Raises:
        ValueError: If the file is invalid, empty, or missing required fields.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            if not data:
                raise ValueError(f"{file_description} JSON file '{file_path}' is empty.")

            # Validate encoding structure based on version
            try:
                version = get_encodings_version(encodings=data)

                # Check if required fields are present and not empty based on version
                if version in [EncodingVersion.V0, EncodingVersion.V1]:
                    if not data.get("activation_encodings") and not data.get("param_encodings"):
                        raise ValueError(
                            f"{file_description} JSON file '{file_path}' "
                            f"(version {version.value}) must contain non-empty "
                            f"'activation_encodings' or 'param_encodings' fields."
                        )
                elif version == EncodingVersion.V2:
                    if not data.get("encodings"):
                        raise ValueError(
                            f"{file_description} JSON file '{file_path}' "
                            f"(version {version.value}) must contain non-empty 'encodings' field."
                        )
            except Exception as version_error:
                raise ValueError(
                    f"Error validating {file_description} file '{file_path}': {version_error}"
                )

    except json.JSONDecodeError as e:
        raise ValueError(f"{file_description} file '{file_path}' is not a valid JSON file: {e}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading {file_description} file '{file_path}': {e}")


def validate_quantization_params(
    calibration_list: Path | str,
    quant_overrides: Path | str,
    algorithm: Algorithm,
    backend: BackendType,
):
    """Validation for calibration list and quantization override params
    based on backend and algorithm
    """
    # Backend: GPU => Two cases
    # 1. If algorithm is not ONESHOT => block it
    # 2. If either calibration_list or quant_overrides is provided => block it
    if backend == BackendType.GPU:
        if algorithm != Algorithm.ONESHOT:
            raise ValueError(
                f"{algorithm.value} snooping algorithm is not supported for GPU backend."
            )
        if calibration_list or quant_overrides:
            raise ValueError("GPU backend does not support quantized model execution.")

    # Backend: CPU => Two cases
    # 1. If algorithm is not ONESHOT => block it
    # 2. If quant_overrides is provided => block it
    if backend == BackendType.CPU:
        if algorithm != Algorithm.ONESHOT:
            raise ValueError(
                f"{algorithm.value} snooping algorithm is not supported for CPU backend."
            )
        if quant_overrides:
            raise ValueError("CPU backend does not support quantized model execution.")

    # Layerwise and Cumulative works iff calibration or override is provided
    if not (calibration_list or quant_overrides) and algorithm in [
        Algorithm.LAYERWISE,
        Algorithm.CUMULATIVE,
    ]:
        raise ValueError(
            "Either quantization overrides or calibration list must be provided to run "
            f"{algorithm.value} snooping algorithm."
        )


class BackendConfig(AISWBaseModel):
    """Define the backend config for snooper module
    Attributes:
        dlc_file: Ready to excute dlc file on given backend and platform.
        converter_arguments: Input arguments required by the converter module.
        quantizer_arguments: Input arguments required by the quantizer module.
        context_bin_gen_arguments: Input arguments required by context_bin_gen module.
        context_bin_backend_extension: Backend extension config for context binary generator.
        offline_prepare: Boolean to indicate offline prepare of graph.
        net_run_arguments: Input arguments required by the netrunner module.
        net_run_backend_extension: Backend extension config for net-runner.
        backend: Type of Backend.
        platform: Platform of target device like, android, x86_64_linux, etc.
        soc_model: Name of SOC model on target device.
        remote_host_details: Details of the remote host.
    """

    dlc_file: Optional[FilePath] = None
    converter_arguments: Optional[ConverterInputArguments] = None
    quantizer_arguments: Optional[QuantizerInputArguments] = None
    context_bin_gen_arguments: Optional[GenerateConfig] = None
    context_bin_backend_extension: Optional[FilePath | dict] = None
    offline_prepare: Optional[bool] = None
    net_run_arguments: Optional[NetRunnerInputArguments] = None
    net_run_backend_extension: Optional[FilePath | dict] = None
    backend: BackendType
    platform: DevicePlatformType
    soc_model: str = ""
    remote_host_details: Optional[RemoteHostDetails] = None

    @model_validator(mode="after")
    def validate_options(self):
        """Validate the backend configuration options.

        This method checks if a DLC file is provided, and if so, it ensures that
        converter, quantizer, offline prepare, and other related options are disabled.

        Raises:
            ValueError: If invalid configuration is detected.
        """
        # When dlc file is supplied, it is assumed dlc is ready to execute. Hence converter, quantizer
        # quantizer and offline prepare option should not be given.
        if self.dlc_file:
            if (
                self.converter_arguments
                or self.quantizer_arguments
                or self.context_bin_gen_arguments
                or self.context_bin_backend_extension
                or self.offline_prepare
            ):
                raise ValueError(
                    "Invalid configuration. When dlc_file is provided, converter_arguments, "
                    "quantizer_arguments, context_bin_gen_arguments, context_bin_backend_extension,"
                    " and offline_prepare must not be set."
                )
        else:
            # If DLC file is not provided, then we must be start from source model.
            # In this case, converter arguments are mandatory (implicitly, though
            # ModelSnooperInputConfig validation will enforce input_model existence).
            pass
        return self

    @model_validator(mode="before")
    @classmethod
    def set_offline_prepare(cls, values):
        """If offline_prepare is None, set it based on backend."""
        dlc_file = values.get("dlc_file")
        # If DLC file is supplied it assumed to ready to execute. Don't set offline_prepare.
        if dlc_file:
            return values
        backend = values.get("backend")
        offline_prepare = values.get("offline_prepare")

        # Enable offline prepare if backend supports.
        if offline_prepare is None and backend in BackendType.offline_preparable_backends():
            values["offline_prepare"] = True
        return values

    @model_validator(mode="after")
    def validate_backend_platform(self):
        """Validations for combination of backend and platform."""
        validate_backend_platform(self.backend, self.platform, self.offline_prepare)
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


class ModelSnooperInputConfig(DebuggerConfig):
    """Defines input arguments for Accuracy Debugger

    Attributes:
        input_model: Path to the source model file.
        input_sample: List of InputSample objects or dictionary of tensor name to numpy array.
        algorithm: Algorithm to use to debug the model.
        reference_config: Configuration for reference backend.
        target_config: Configuration for target backend.
        comparators: List of comparators to use in verification stage.
        debug_subgraph_inputs: list of inputs to debug subgraph.
        debug_subgraph_outputs: list of outputs of debug subgraph.
        skip_layer_types: list of op_types to be ignored.
            Note: Currently supported for Oneshot Layerwise snooping.
        include_layer_types: list of op_types to be considered for debugging.
            Note: Currently supported for Oneshot Layerwise snooping.
        working_directory: Path to the directory to store the artifacts and results.
        golden_reference_path: Directory containing golden reference outputs.
        retain_compilation_artifacts: Flag to retain the compilation artifacts. Default is False.
        dump_output_tensors: Boolean to indicate whether to dump output tensors.
        is_qnn_golden_reference: Whether given golden outputs are from QNN.
        compulsory_overrides: Path to the compulsory overrides. This is to be used only for
           layerwise and cumulative-layerwise snooping.
        max_parallel_compilations: The number of parallel executions of subgraphs in layerwise and
            cumulative snooping. The provided max_parallel_compilations will set as follows:
            max_parallel_compilations = min(user provided max compilations, max compilations supported by host device)
            Maximum of 16 parallel compilations supported.
    """

    input_model: Optional[FilePath] = None
    input_sample: list[InputSample] | dict[str, NDArray]
    algorithm: Algorithm = Algorithm.ONESHOT
    reference_config: Optional[BackendConfig] = None
    target_config: Optional[BackendConfig]
    comparators: List[Comparator] = [MSEComparator()]
    debug_subgraph_inputs: Optional[List[str]] = None
    debug_subgraph_outputs: Optional[List[str]] = None
    skip_layer_types: Optional[List[str]] = None
    include_layer_types: Optional[List[str]] = None
    working_directory: DirectoryPath = Field(default_factory=create_working_directory)
    golden_reference_path: Optional[DirectoryPath] = None
    retain_compilation_artifacts: bool = False
    dump_output_tensors: bool = False
    is_qnn_golden_reference: bool = False
    compulsory_overrides: Optional[Path] = None
    max_parallel_compilations: Optional[int] = None

    # LoRA-specific attributes for LoRA-aware snooping
    lora_model_creator_args: Optional[LoRAModelCreatorInputConfig] = None
    lora_importer_args: Optional[LoRAImporterInputConfig] = None
    use_case_names: Optional[List[str]] = None
    lora_alpha_tensor: Optional[FilePath] = None

    @model_validator(mode="after")
    def validate_lora_alpha_tensor(self):
        """Validate that lora_alpha_tensor is provided when LoRA is enabled."""
        lora_enabled = (
            self.lora_model_creator_args is not None or self.lora_importer_args is not None
        )
        if lora_enabled and self.lora_alpha_tensor is None:
            raise ValueError(
                "lora_alpha_tensor is required when lora_model_creator_args or "
                "lora_importer_args is provided for LoRA snooping."
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_options(cls, values):
        """Validate the input configuration options.

        This validates:
        1. input_model requirement when DLC files are missing
        2. Algorithm restriction when DLC is provided for either reference or target
        """
        input_model = values.get("input_model")
        reference_config = values.get("reference_config")
        target_config = values.get("target_config")
        algorithm = values.get("algorithm", Algorithm.ONESHOT)

        # Extract DLC file paths from configs
        ref_dlc = None
        target_dlc = None

        if reference_config:
            ref_dlc = (
                reference_config.dlc_file
                if hasattr(reference_config, "dlc_file")
                else reference_config.get("dlc_file")
            )

        if target_config:
            target_dlc = (
                target_config.dlc_file
                if hasattr(target_config, "dlc_file")
                else target_config.get("dlc_file")
            )

        # Validation 1: input_model requirement
        # This validation is for config mode (when reference_config is present)
        if reference_config is not None:
            # Config mode: If either reference or target DLC is missing, input_model is required
            if not ref_dlc or not target_dlc:
                if not input_model:
                    raise ValueError(
                        "input_model is required when dlc_file is missing in "
                        "reference_config or target_config."
                    )

        # Validation 2: Algorithm restriction when DLC is provided
        # Only OneShot algorithm is supported when DLC is provided for either reference or target
        if (ref_dlc or target_dlc) and algorithm != Algorithm.ONESHOT:
            raise ValueError(
                f"Only '{Algorithm.ONESHOT.value}' algorithm is supported when dlc_file is provided in "
                f"reference_config or target_config. Got algorithm: '{algorithm.value}'"
            )

        return values

    @model_validator(mode="after")
    def validate_max_parallel_compilations(self):
        """Verify max_parallel_compilations > 0"""
        if self.max_parallel_compilations is not None and self.max_parallel_compilations <= 0:
            raise ValueError(
                "Argument --max_parallel_compilations must be a positive integer greater than 0. "
                f"Invalid input value: {self.max_parallel_compilations}."
            )
        return self

    @model_validator(mode="after")
    def validate_compulsory_overrides(self):
        """Validation for compulsory_overrides parameter based on algorithm"""
        if self.compulsory_overrides and self.algorithm == Algorithm.ONESHOT:
            raise ValueError(
                "compulsory_overrides can only be used with layerwise or cumulative-layerwise"
                "snooping algorithms."
            )

        # Validate that compulsory_overrides JSON file is not empty and has required fields
        if self.compulsory_overrides:
            validate_encoding_json_file(self.compulsory_overrides, "compulsory_overrides")

        return self

    @field_validator("input_model", mode="after")
    @classmethod
    def validate_input_model(cls, input_model):
        """Validation for the type of input_model provided"""
        if input_model:
            _ = get_framework_type(input_model)
        return input_model

    @model_validator(mode="after")
    def validate_quantization_params(self):
        """Validation for calibration list and quantization override params
        based on algorithm and backend
        """
        # Validate Target Config
        target_quant_overrides = (
            self.target_config.converter_arguments.quantization_overrides
            if self.target_config.converter_arguments
            else None
        )
        target_calibration_list = (
            self.target_config.quantizer_arguments.input_list
            if self.target_config.quantizer_arguments
            else None
        )

        if target_quant_overrides:
            validate_encoding_json_file(target_quant_overrides, "quantization_overrides")

        validate_quantization_params(
            target_calibration_list,
            target_quant_overrides,
            self.algorithm,
            self.target_config.backend,
        )

        # Validate Reference Config (if present)
        if self.reference_config:
            ref_quant_overrides = (
                self.reference_config.converter_arguments.quantization_overrides
                if self.reference_config.converter_arguments
                else None
            )
            ref_calibration_list = (
                self.reference_config.quantizer_arguments.input_list
                if self.reference_config.quantizer_arguments
                else None
            )

            if ref_quant_overrides:
                validate_encoding_json_file(ref_quant_overrides, "quantization_overrides")

            validate_quantization_params(
                ref_calibration_list,
                ref_quant_overrides,
                self.algorithm,
                self.reference_config.backend,
            )

        return self

    @model_validator(mode="after")
    def set_qnn_golden_reference(self):
        """Automatically set is_qnn_golden_reference when reference_config is provided.

        When reference_config is provided, the reference outputs will be from QNN backend,
        so we automatically set is_qnn_golden_reference to True.
        """
        if self.reference_config:
            object.__setattr__(self, "is_qnn_golden_reference", True)
        return self

    @model_validator(mode="after")
    def validate_tensor_and_layer_configurations(self):
        """Validate tensor output and layer filtering configurations.

        This method validates the following scenarios:
        1. skip_layer_types and include_layer_types are mutually exclusive
        2. set_output_tensors cannot be used with debug_subgraph_inputs,
           debug_subgraph_outputs, skip_layer_types or include_layer_types
        3. set_output_tensors, include_layer_types or skip_layer_types can be
           used when algorithm is Algorithm.ONESHOT only

        Raises:
            ValueError: If any validation scenario fails
        """
        # Scenario 1: Validate that skip_layer_types and include_layer_types are mutually exclusive
        if self.skip_layer_types is not None and self.include_layer_types is not None:
            raise ValueError(
                "skip_layer_types and include_layer_types are mutually exclusive. "
                "Please provide only one of these parameters."
            )

        # Get set_output_tensors from target_config.context_bin_gen_arguments or net_run_arguments
        set_output_tensors = None
        if self.target_config.context_bin_gen_arguments:
            set_output_tensors = getattr(
                self.target_config.context_bin_gen_arguments, "set_output_tensors", None
            )
        if set_output_tensors is None and self.target_config.net_run_arguments:
            set_output_tensors = getattr(
                self.target_config.net_run_arguments, "set_output_tensors", None
            )

        # Scenario 2: set_output_tensors cannot be used with debug_subgraph_inputs,
        # debug_subgraph_outputs, skip_layer_types or include_layer_types
        if set_output_tensors is not None:
            conflicting_params = []
            if self.debug_subgraph_inputs is not None:
                conflicting_params.append("debug_subgraph_inputs")
            if self.debug_subgraph_outputs is not None:
                conflicting_params.append("debug_subgraph_outputs")
            if self.skip_layer_types is not None:
                conflicting_params.append("skip_layer_types")
            if self.include_layer_types is not None:
                conflicting_params.append("include_layer_types")

            if conflicting_params:
                raise ValueError(
                    f"set_output_tensors cannot be used together with "
                    f"{', '.join(conflicting_params)}. Please use only one approach "
                    f"to specify output tensors or layer filtering."
                )

        # Scenario 3: set_output_tensors, include_layer_types or skip_layer_types
        # can be used when algorithm is Algorithm.ONESHOT only
        if self.algorithm != Algorithm.ONESHOT:
            invalid_params = []
            if set_output_tensors is not None:
                invalid_params.append("set_output_tensors")
            if self.include_layer_types is not None:
                invalid_params.append("include_layer_types")
            if self.skip_layer_types is not None:
                invalid_params.append("skip_layer_types")

            if invalid_params:
                raise ValueError(
                    f"{', '.join(invalid_params)} can only be used with "
                    f"oneshot-layerwise. Current algorithm is {self.algorithm.value}."
                )

        return self


class ModelSnooperOutputConfig(AISWBaseModel):
    """Defines Accuracy Debugger output format

    Attributes:
        snooping_report: Report generated by the snooping algorithm executed.
    """

    csv_snooping_report: FilePath
    json_snooping_report: FilePath


class ModelSnooperSchemaV1(ModuleSchema):
    """Schema for accuracy debugger module."""

    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["ModelSnooperModule"] = "ModelSnooperModule"
    path: Path = Path(__file__)
    arguments: ModelSnooperInputConfig
    outputs: Optional[ModelSnooperOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION
