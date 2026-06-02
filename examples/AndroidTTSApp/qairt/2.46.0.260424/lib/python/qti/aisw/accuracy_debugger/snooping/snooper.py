# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    NetRunnerInputArguments,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.framework_runner.framework_factory import (
    get_framework_type,
)
from qti.aisw.accuracy_debugger.framework_runner.frameworks.onnx_framework import (
    CustomOnnxFramework,
)
from qti.aisw.accuracy_debugger.inference_engine.inference_engine_wrapper import (
    InferenceEngineWrapper,
)
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    InferenceEngine,
    InferenceEngineInputConfig,
    InferenceEngineOutputConfig,
)
from qti.aisw.accuracy_debugger.model_snooper.config import (
    BackendConfig,
    ModelSnooperInputConfig,
)
from qti.aisw.accuracy_debugger.snooping.snooper_utils import convert_data
from qti.aisw.accuracy_debugger.utils.helper import (
    ActivationInfo,
    create_working_directory_with_timestamp,
    generate_reference_data_with_framework,
    get_logger,
    load_data_from_directory,
)
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import (
    GenerateConfig,
)
from qti.aisw.tools.core.utilities.devices.api.device_definitions import (
    DevicePlatformType,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_framework import (
    OnnxTransformModel,
)
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import (
    OnnxModelHelper,
)
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.verifier.verifier import TensorLayout


class Snooper(ABC):
    """Abstract class for all the snooper algorithms."""

    def __init__(self, name: str, logger: logging.Logger):
        """Initializes the Snooper.

        Args:
            name (str): The name of the snooper algorithm
            logger (logging.Logger): A python logger instance
        """
        self.framework_type = None
        self._name = name
        self._logger = logger

    def _setup(self, modelsnooper_input_config: ModelSnooperInputConfig) -> ModelSnooperInputConfig:
        """This method contains the intial setup instructions for running snooper

        Args:
            modelsnooper_input_config: Model Snooper input configuration.

        Returns:
           modelsnooper_input_config: Updated Model Snooper input configuration

        """
        # Initialize working directory if not provided
        if not modelsnooper_input_config.working_directory:
            modelsnooper_input_config.working_directory = create_working_directory_with_timestamp(
                sub_directory=self._name + "_snooping",
            )

        if modelsnooper_input_config.input_model:
            self.model = modelsnooper_input_config.input_model
            self.framework_type = get_framework_type(self.model)

            # Skip optimization if explicitly disabled or if quantization overrides are provided
            converter_args = modelsnooper_input_config.target_config.converter_arguments
            if converter_args:
                if (
                    converter_args.onnx_simplification is False
                    or converter_args.quantization_overrides
                ):
                    return modelsnooper_input_config

            # Build input shape dictionary
            input_sample = modelsnooper_input_config.input_sample
            if isinstance(input_sample, list):
                input_shape_dict = {sample.name: sample.dimensions for sample in input_sample}
            else:
                input_shape_dict = {name: data.shape for name, data in input_sample.items()}

            # Optimize model
            optimized_model = OnnxTransformModel.optimize_by_simplifier(
                self.model,
                overwrite_input_shapes=input_shape_dict,
            )

            if optimized_model:
                self._logger.info("ONNX model simplification completed")
                optimized_model_path = Path(
                    modelsnooper_input_config.working_directory
                    / f"{self.model.stem}_optimized.onnx"
                )
                OnnxModelHelper.save_model(optimized_model, optimized_model_path)
                self._logger.debug(f"Optimized model saved at {optimized_model_path}")
                self.model = optimized_model_path
            else:
                self._logger.warning(
                    "ONNX model simplification failed. Continuing without onnx simplification"
                )
        return modelsnooper_input_config

    @abstractmethod
    def run(self, modelsnooper_input_config: ModelSnooperInputConfig) -> tuple[Path, Path]:
        """This is abstract method, and it should be implemented by the child class algorithms.

        Args:
            modelsnooper_input_config: Model Snooper input configuration.

        Returns:
           tuple: File paths to csv and json snooping report.
        """
        pass

    @abstractmethod
    def _generate_json_report(self, data_frame: pd.DataFrame, output_dir: Path) -> Path:
        """Generate a JSON version of the snooping report.

        Creates a structured JSON report from the DataFrame containing snooping results.
        The JSON format includes a header with version information and a list of layers
        with their comparison metrics.

        Args:
            data_frame (pd.DataFrame): DataFrame containing the snooping report data.
            output_dir (Path): Directory where the JSON report will be saved.

        Returns:
            Path: File path to json snooping report
        """
        pass

    def _resolve_unsupported_dtypes(
        self,
        input_tensors: list,
        input_sample: dict,
        quantizer_args: QuantizerInputArguments,
        output_dir: Path,
    ) -> tuple[dict, QuantizerInputArguments]:
        """Function to resolve unsupported data types in input tensors for the given backend
        Args:
            input_tensors: List of input tensors provided
            input_sample: Dict containing input tensor names and numpy data
            quantizer_args: Quantizer arguments
            output_dir: Output directory to store the results
        Returns:
            tuple: Updated Input sample and quantizer arguments
        """
        # Get unsupported tensors
        unsupported_tensor_names = [
            tensor.name
            for tensor in input_tensors
            if tensor.data_type and tensor.data_type != "float32"
        ]

        if not unsupported_tensor_names:
            return (input_sample, quantizer_args)

        # Update quantizer args
        if quantizer_args and quantizer_args.input_list:
            quantizer_args.input_list = convert_data(
                quantizer_args.input_list,
                [tensor.data_type for tensor in input_tensors],
                output_dir,
            )
        # Update input sample
        for tensor in unsupported_tensor_names:
            input_sample[tensor] = input_sample[tensor].astype(np.float32)

        return (input_sample, quantizer_args)

    def _get_reference_outputs(
        self,
        model: Path = None,
        reference_backend_config: BackendConfig = None,
        golden_reference_path: Path = None,
        input_sample: dict[str, NDArray] = None,
        dump_output_tensors: bool = False,
        working_directory: Path = None,
        intermediate_outputs_list: Optional[list[str]] = None,
    ) -> tuple:
        """Get reference outputs from either golden reference path, reference backend, or framework.

        Args:
            model: Path to ONNX model file (for backward compatibility)
            reference_backend_config: Configuration for reference backend.
            golden_reference_path: Path to pre-generated golden reference data.
            input_sample: Input tensors for model execution.
            dump_output_tensors: Flag to enable dumping of output tensors.
            working_directory: Directory to store intermediate files.
            intermediate_outputs_list: List of nodes to debug

        Returns:
            dict: Dictionary containing reference output tensors.

        Raises:
            Exception: If both model and reference_backend_config are None
        """
        # Use intermediate_outputs_list for debug nodes
        debug_nodes = intermediate_outputs_list

        # Use self.model if model parameter is not provided
        model_path = model if model is not None else getattr(self, "model", None)

        if model_path is None and reference_backend_config is None:
            raise Exception("Either model or reference_backend_config needs to be provided.")

        # If golden reference path is provided, load data from there
        if golden_reference_path:
            self._logger.info(f"Loading reference data from given {golden_reference_path}")
            # Load raw files into memory using ONNX framework
            custom_onnx_obj = CustomOnnxFramework(self._logger)
            intermediate_outputs_info = custom_onnx_obj.get_intermediate_outputs_info(model_path)
            reference_outputs = load_data_from_directory(
                golden_reference_path, intermediate_outputs_info
            )

        elif reference_backend_config:
            # Generate reference data using inference engine
            reference_output_directory = working_directory / "reference_outputs"
            reference_output_directory.mkdir(exist_ok=True)

            if reference_backend_config.dlc_file:
                # Scenario 1: DLC provided - use directly
                reference_net_run_arguments = reference_backend_config.net_run_arguments
                if not reference_net_run_arguments:
                    reference_net_run_arguments = NetRunnerInputArguments(debug=True)
                else:
                    if not reference_net_run_arguments.set_output_tensors:
                        reference_net_run_arguments.debug = True

                inference_output_config = self._execute_inference(
                    model=reference_backend_config.dlc_file,
                    net_runner_args=reference_net_run_arguments,
                    net_run_backend_extension=reference_backend_config.net_run_backend_extension,
                    input_sample=input_sample,
                    backend=reference_backend_config.backend,
                    platform=reference_backend_config.platform,
                    remote_host_details=reference_backend_config.remote_host_details,
                    working_directory=reference_output_directory,
                    soc_model=reference_backend_config.soc_model,
                    dump_output_tensors=dump_output_tensors,
                    wrapper_mode=True,
                )
                reference_outputs = inference_output_config.output_data[0]
            else:
                # Scenario 2: DLC not provided - generate from input_model (Enhanced functionality)
                reference_outputs = self._generate_reference_outputs_from_input_model(
                    reference_backend_config=reference_backend_config,
                    model_path=model_path,
                    debug_nodes=debug_nodes,
                    input_sample=input_sample,
                    reference_output_directory=reference_output_directory,
                    dump_output_tensors=dump_output_tensors,
                )
        else:
            self._logger.info(f"Generating reference data for {model_path}")
            # Generate reference outputs using framework manager
            reference_outputs = generate_reference_data_with_framework(
                self._logger,
                model_path,
                input_sample,
                dump_output_tensors,
                working_directory,
                intermediate_output_tensors=debug_nodes,
            )

        return reference_outputs

    def _generate_reference_outputs_from_input_model(
        self,
        reference_backend_config: BackendConfig,
        model_path: Path,
        debug_nodes: Optional[list[str]],
        input_sample: dict[str, NDArray],
        reference_output_directory: Path,
        dump_output_tensors: bool,
    ) -> dict[str, NDArray]:
        """Generate reference outputs from input model using inference engine.

        This method handles the scenario where DLC is not provided and needs to be generated
        from the input model. It configures the backend appropriately for offline prepare
        or net runner modes and executes the full inference pipeline.

        Args:
            reference_backend_config: Configuration for reference backend.
            model_path: Path to the input model.
            debug_nodes: List of debug nodes to extract outputs for.
            input_sample: Input tensors for model execution.
            reference_output_directory: Directory to store reference outputs.
            dump_output_tensors: Flag to enable dumping of output tensors.

        Returns:
            dict[str, NDArray]: Dictionary containing reference output tensors.
        """
        # Enable framework traces in converter arguments
        if reference_backend_config.converter_arguments:
            reference_backend_config.converter_arguments.enable_framework_trace = True
        else:
            reference_backend_config.converter_arguments = ConverterInputArguments(
                enable_framework_trace=True
            )

        # Determine if offline preparable
        is_offline_preparable = (
            reference_backend_config.offline_prepare
            and reference_backend_config.backend in BackendType.offline_preparable_backends()
        )

        if is_offline_preparable:
            # Configure context bin generation
            if not reference_backend_config.context_bin_gen_arguments:
                if debug_nodes:
                    sanitized_debug_subgraph = [
                        Helper.transform_node_names(name) for name in debug_nodes
                    ]
                    reference_backend_config.context_bin_gen_arguments = GenerateConfig(
                        set_output_tensors=sanitized_debug_subgraph,
                        enable_intermediate_outputs=False,
                    )
                else:
                    reference_backend_config.context_bin_gen_arguments = GenerateConfig(
                        enable_intermediate_outputs=True
                    )
            else:
                if debug_nodes:
                    sanitized_debug_subgraph = [
                        Helper.transform_node_names(name) for name in debug_nodes
                    ]
                    reference_backend_config.context_bin_gen_arguments.set_output_tensors = (
                        sanitized_debug_subgraph
                    )
                reference_backend_config.context_bin_gen_arguments.enable_intermediate_outputs = (
                    False
                    if reference_backend_config.context_bin_gen_arguments.set_output_tensors
                    else True
                )
        else:
            # Configure net runner
            if not reference_backend_config.net_run_arguments:
                if debug_nodes:
                    sanitized_debug_subgraph = [
                        Helper.transform_node_names(name) for name in debug_nodes
                    ]
                    reference_backend_config.net_run_arguments = NetRunnerInputArguments(
                        set_output_tensors=sanitized_debug_subgraph,
                        debug=False,
                    )
                else:
                    reference_backend_config.net_run_arguments = NetRunnerInputArguments(debug=True)
            else:
                if debug_nodes:
                    sanitized_debug_subgraph = [
                        Helper.transform_node_names(name) for name in debug_nodes
                    ]
                    reference_backend_config.net_run_arguments.set_output_tensors = (
                        sanitized_debug_subgraph
                    )
                reference_backend_config.net_run_arguments.debug = (
                    False if reference_backend_config.net_run_arguments.set_output_tensors else True
                )

        # Execute full pipeline
        inference_output_config = self._execute_inference(
            model=model_path,
            converter_args=reference_backend_config.converter_arguments,
            quantizer_args=reference_backend_config.quantizer_arguments,
            context_bin_args=reference_backend_config.context_bin_gen_arguments,
            context_bin_backend_extension=reference_backend_config.context_bin_backend_extension,
            offline_prepare=reference_backend_config.offline_prepare,
            net_runner_args=reference_backend_config.net_run_arguments,
            net_run_backend_extension=reference_backend_config.net_run_backend_extension,
            input_sample=input_sample,
            backend=reference_backend_config.backend,
            platform=reference_backend_config.platform,
            remote_host_details=reference_backend_config.remote_host_details,
            working_directory=reference_output_directory,
            soc_model=reference_backend_config.soc_model,
            dump_output_tensors=dump_output_tensors,
            wrapper_mode=True,
        )
        return inference_output_config.output_data[0]

    @staticmethod
    def _execute_inference(
        model: Path,
        logger: logging.Logger = None,
        converter_args: Optional[ConverterInputArguments] = None,
        quantizer_args: Optional[QuantizerInputArguments] = None,
        context_bin_args: Optional[GenerateConfig] = None,
        context_bin_backend_extension: Optional[Path | dict] = None,
        offline_prepare: Optional[bool] = None,
        net_runner_args: Optional[NetRunnerInputArguments] = None,
        net_run_backend_extension: Optional[Path | dict] = None,
        input_sample: Optional[dict[str, NDArray]] = None,
        backend: Optional[BackendType] = None,
        platform: Optional[DevicePlatformType] = None,
        remote_host_details: Optional[RemoteHostDetails] = None,
        working_directory: Optional[Path] = None,
        soc_model: str = "",
        dump_output_tensors: bool = False,
        wrapper_mode: bool = False,
        netrun_lock=None,
        log_level: str = None,
        logger_name: str = None,
    ) -> InferenceEngineOutputConfig:
        """Compile and execute model using inference engine.

        Args:
            model: Path to model
            logger: Logger object for logging
            converter_args: Input arguments required by the converter module
            quantizer_args: Input arguments required by the quantizer module
            context_bin_args: Input arguments required by context_bin_gen module
            context_bin_backend_extension: Backend extension config file or dictionary for
                                           context binary generator.
            offline_prepare: Boolean to indicate offline prepare of graph
            net_runner_args: Input arguments required by the netrunner module
            net_run_backend_extension: Backend extension config file or dictionary for net-runner
            input_sample: Input to netrunner module
            backend: Backed type
            platform: Target platform
            remote_host_details: Details of remote host
            working_directory: Path to directory to store artifacts.
            soc_model : Name of SOC model on target device.
            dump_output_tensors: Boolean to indicate whether to dump output tensors.
            wrapper_mode: Boolean to indicate whether to run inference engine in wrapper mode.

        Returns:
            InferenceEngineOutputConfig: Inference engine output config containing inference data
                                         and artifacts like, converted dlc, quantized dlc and,
                                         context binary.

        """
        if not logger:
            # Get logger for Inference Engine.
            logger = get_logger(
                logger_name=logger_name if logger_name else "Inference Engine",
                log_file_path=working_directory,
                log_file_name="execution",
                level=log_level.upper() if log_level else "INFO",
            )

        inf_engine_input_config = InferenceEngineInputConfig(
            input_model=model,
            converter_arguments=converter_args,
            quantizer_arguments=quantizer_args,
            context_bin_gen_arguments=context_bin_args,
            context_bin_backend_extension=context_bin_backend_extension,
            offline_prepare=offline_prepare,
            net_run_arguments=net_runner_args,
            net_run_input_data=input_sample,
            net_run_backend_extension=net_run_backend_extension,
            backend=backend,
            platform=platform,
            remote_host_details=remote_host_details,
            working_directory=working_directory,
            soc_model=soc_model,
            dump_output=dump_output_tensors,
        )
        if wrapper_mode:
            inf_engine = InferenceEngineWrapper(logger)
            inf_output_config = inf_engine.run(inf_engine_input_config)
        else:
            inf_engine = InferenceEngine(logger)
            inf_output_config = inf_engine.run(inf_engine_input_config, netrun_lock=netrun_lock)

        return inf_output_config

    def _get_profile_info(
        self, golden_output: dict[str, NDArray], dlc_path: Path = None
    ) -> tuple[dict[str, ActivationInfo], dict[str, ActivationInfo], dict]:
        """Reads profile_info.json from framework runner and layout_data to
        populate framework and target related tensor informations.

        Args:
            golden_output (dict[str, NDArray]): Dictionary of output name to numpy array
            dlc_path (Path): Path to DLC file.

        Returns:
            tuple[dict[str, ActivationInfo], dict[str, ActivationInfo], dict]: Returns
                the framework activation, target activation and layout info.
        """
        # Create Profile Info
        profile_info = {}
        for output_tensor_name, data in golden_output.items():
            santized_tensor_name = Helper.transform_node_names(output_tensor_name)

            if not data.size or data.dtype == bool:
                if data.size == 0:
                    profile_info[santized_tensor_name] = (
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    )
                else:
                    profile_info[santized_tensor_name] = (
                        str(data.dtype),
                        data.shape,
                        data.tolist(),
                        data.tolist(),
                        data.tolist(),
                    )
            else:
                profile_info[santized_tensor_name] = (
                    str(data.dtype),
                    data.shape,
                    round(np.min(data), 3),
                    round(np.max(data), 3),
                    round(np.median(data), 3),
                )

        framework_activation_info = {}
        for sanitized_activation_name, value in profile_info.items():
            activation_info = ActivationInfo(
                dtype=value[0], shape=value[1], distribution=tuple(value[2:])
            )
            framework_activation_info[sanitized_activation_name] = activation_info

        target_activation_info = {}
        layout_info = {}
        if dlc_path:
            tensor_layout = TensorLayout()
            layout_info = tensor_layout.get_layout_info_from_dlc(dlc_path)

            for sanitized_activation_name, value in layout_info.items():
                activation_info = ActivationInfo(dtype=None, shape=value["dims"], distribution=None)
                target_activation_info[sanitized_activation_name] = activation_info

        return framework_activation_info, target_activation_info, layout_info
