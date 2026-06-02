# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnx  # noqa
import pandas as pd
from numpy.typing import NDArray
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    NetRunnerInputArguments,
)
from qti.aisw.accuracy_debugger.framework_runner.frameworks.model_traverser import ModelTraverser
from qti.aisw.accuracy_debugger.inference_engine.inference_engine_wrapper import InferenceEngineWrapper
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    InferenceEngineInputConfig,
)
from qti.aisw.accuracy_debugger.lora import LoRASnooper
from qti.aisw.accuracy_debugger.model_snooper.config import BackendConfig, ModelSnooperInputConfig
from qti.aisw.accuracy_debugger.snooping.snooper import Snooper
from qti.aisw.accuracy_debugger.snooping.snooper_utils import filter_snooping_report
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.accuracy_debugger.utils.exceptions import VerificationError
from qti.aisw.accuracy_debugger.utils.helper import (
    ActivationInfo,
    dump_json_report,
    load_input_tensors,
    plot_graphs,
    verify,
)
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import GenerateConfig
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.tensor_mapping.tensor_mapping import (
    TensorMapper,
    TensorMapperInputConfig,
)


class OneshotLayerwiseSnooper(Snooper):
    """Subclass for oneshot layerwise algorithm."""

    def __init__(self, logger: logging.Logger):
        """Initializes the OneshotLayerwiseSnooper.

        Args:
            logger (logging.Logger): A python logger instance
        """
        # Initialize the parent Snooper class with the oneshot algorithm name and logger
        super().__init__(name=Algorithm.ONESHOT, logger=logger)
        # Initialize inference_output_config to None, will be set during execution
        self.inference_output_config = None
        # Cache for ModelTraverser instance to avoid redundant model loading
        self._model_traverser_cache = None

    def run(self, modelsnooper_input_config: ModelSnooperInputConfig) -> tuple[Path, Path]:
        """This is entry point method of oneshot algorithm. It accepts data necessary to debug the
        model via modelsnooper input config and generates report.

        Args:
            modelsnooper_input_config: Model Snooper input configuration.

        Returns:
           tuple: File paths to csv and json snooping report.
        """
        # Call parent's run method to perform initial setup
        modelsnooper_input_config = self._setup(modelsnooper_input_config=modelsnooper_input_config)

        # Check if LoRA mode is enabled
        if self._is_lora_enabled(modelsnooper_input_config):
            self._logger.info("LoRA mode detected - running LoRA-aware snooping")
            return self._run_lora_snooping(modelsnooper_input_config)

        # Extract configuration parameters from the input config
        comparators = modelsnooper_input_config.comparators
        dump_output_tensors = modelsnooper_input_config.dump_output_tensors
        golden_reference_path = modelsnooper_input_config.golden_reference_path
        input_tensors = modelsnooper_input_config.input_sample
        is_qnn_golden_reference = modelsnooper_input_config.is_qnn_golden_reference
        reference_backend_config = modelsnooper_input_config.reference_config
        retain_compilation_artifacts = modelsnooper_input_config.retain_compilation_artifacts
        target_backend_config = modelsnooper_input_config.target_config
        working_directory = modelsnooper_input_config.working_directory
        # Get subgraph to be debugged based on the following options
        # 1. set_output_tensors
        # 2. debug_subgraph_inputs
        # 3. debug_subgraph_outputs
        # 4. skip_layer_types
        # 5. include_layer_types
        if modelsnooper_input_config.input_model:
            debug_subgraph = self._get_debug_subgraph(modelsnooper_input_config)
        else:
            debug_subgraph = None

        # Load input tensors from the provided input sample
        input_sample = load_input_tensors(input_tensors)

        # Get reference outputs from either golden reference, reference backend, or framework
        reference_outputs = self._get_reference_outputs(
            reference_backend_config=reference_backend_config,
            golden_reference_path=golden_reference_path,
            input_sample=input_sample,
            dump_output_tensors=dump_output_tensors,
            working_directory=working_directory,
            intermediate_outputs_list=debug_subgraph,
        )
        self._logger.info("Completed reference output generation.")

        # Modify input tensors to handle datatypes not supported by the given backend
        # If all the datatypes are supported, the input remains unchanged
        input_sample, target_backend_config.quantizer_arguments = self._resolve_unsupported_dtypes(
            input_tensors,
            input_sample,
            target_backend_config.quantizer_arguments,
            working_directory,
        )

        # Set up inference output directory
        inference_output_directory = working_directory / "inference_engine"
        inference_output_directory.mkdir(exist_ok=True)

        # Get target outputs by running inference on the target backend
        inference_outputs, dlc_file = self._get_target_outputs(
            target_backend_config,
            input_sample,
            dump_output_tensors,
            inference_output_directory,
            debug_subgraph=debug_subgraph,
        )
        # Check if golden reference names are already sanitized.
        # This may happens when the golden reference was generated by the accuracy debugger
        # framework runner, which dumps tensor files using sanitized names
        # (via Helper.transform_node_names). In that case, the reference tensor names
        # already match the sanitized QNN inference tensor names, so identity mapping is correct.
        golden_reference_sanitized = (
            golden_reference_path is not None
            and bool(reference_outputs)
            and all(
                Helper.transform_node_names(name) == name for name in reference_outputs.keys()
            )
        )

        # Set up graph info for tensor mapping if using golden reference or reference backend.
        # Also use identity mapping when golden reference names are already sanitized, which
        # happens when the golden reference may be generated by the accuracy debugger framework runner.
        if (
            (golden_reference_path and is_qnn_golden_reference)
            or reference_backend_config
            or golden_reference_sanitized
        ):
            graph_info = {"tensor_mapping": {name: name for name in reference_outputs.keys()}}
        else:
            graph_info = None

        # Compare framework outputs with inference output and generate snooping report
        verifier_output = verify(
            reference_outputs,
            inference_outputs,
            self._logger,
            dlc_file,
            comparators,
            graph_info=graph_info,
            is_qnn_golden_reference=is_qnn_golden_reference,
        )

        # Raise error if verification failed
        if not verifier_output:
            raise VerificationError(
                "Verification of tensors failed. Please check the logs for more details."
            )

        # Get tensor mapping between reference and target tensors.
        tensor_mapping_dict = self._get_tensor_mapping(
            dlc_file,
            graph_info
        )

        # Get profile information for both framework and target activations
        framework_activation_info, target_activation_info, _ = self._get_profile_info(
            reference_outputs, dlc_file
        )

        # Set target min, max, median values in target_activation_info
        for inference_tensor, tensor_data in inference_outputs.items():
            if inference_tensor in target_activation_info:
                target_activation_info[inference_tensor].distribution = (
                    np.min(tensor_data),
                    np.max(tensor_data),
                    np.median(tensor_data),
                )
        # Generate the snooping report with all collected data
        csv_snooping_report_path, json_snooping_report_path = self._generate_snooping_report(
            verifier_output,
            comparators,
            working_directory,
            inference_outputs,
            framework_activation_info,
            target_activation_info,
            tensor_mapping_dict,
        )

        # Plot graphs based on the snooping report data
        plot_graphs(
            csv_path=csv_snooping_report_path,
            layer_names_column="Source Name",
            comparator_columns=[comparator.name for comparator in comparators],
            output_dir=working_directory,
            algorithm=self._name,
            logger=self._logger,
        )

        # Clean up artifacts if not needed
        if not retain_compilation_artifacts and self.inference_output_config:
            self._logger.info(
                f"Cleaning up compilation artifacts:\n"
                f"Converter DLC: {self.inference_output_config.converter_dlc}\n"
                f"Quantizer DLC: {self.inference_output_config.quantizer_dlc}\n"
                f"Offline Graph: {self.inference_output_config.offline_graph}"
            )
            self.inference_output_config.cleanup_artifacts()

        # Return the path to the generated snooping report
        return csv_snooping_report_path, json_snooping_report_path

    def _get_debug_subgraph(
        self, modelsnooper_input_config: ModelSnooperInputConfig
    ) -> list[str] | None:
        """Get the debug subgraph based on configuration options.

        Determines which nodes/tensors of the model should be debugged based on the
        following configuration options:
        - debug_subgraph_inputs: Input tensors defining subgraph start
        - debug_subgraph_outputs: Output tensors defining subgraph end
        - skip_layer_types: Layer types to exclude from debugging
        - include_layer_types: Layer types to include in debugging

        Args:
            modelsnooper_input_config: Model Snooper input configuration containing
                debug subgraph options.

        Returns:
            list[str] | None: List of tensor names to debug, or None if no filtering
                is specified.

        Note:
            Uses a cached ModelTraverser instance to avoid redundant model loading
            from disk on repeated calls.
        """
        # If no model is available (e.g., when both DLCs are provided), return None
        if not hasattr(self, "model") or self.model is None:
            return None

        set_output_tensors: list[str] | None = None
        target_config = modelsnooper_input_config.target_config
        if target_config.context_bin_gen_arguments:
            set_output_tensors = getattr(
                target_config.context_bin_gen_arguments, "set_output_tensors", None
            ) or None
        if not set_output_tensors and target_config.net_run_arguments:
            set_output_tensors = getattr(
                target_config.net_run_arguments, "set_output_tensors", None
            ) or None

        if set_output_tensors:
            return list(set_output_tensors)

        # Reuse cached ModelTraverser instance if available, otherwise create and cache it
        if self._model_traverser_cache is None:
            self._model_traverser_cache = ModelTraverser(model_path=self.model)

        model_traverser = self._model_traverser_cache
        debug_subgraph = None

        # Check if we need to filter by layer type along with subgraph inputs/outputs
        if (
            modelsnooper_input_config.debug_subgraph_inputs
            or modelsnooper_input_config.debug_subgraph_outputs
        ):
            # Scenario 3: If there are subgraph inputs/outputs along with skip_layer_types or include_layer_types
            if modelsnooper_input_config.skip_layer_types:
                # For skip_layer_types: Get all subgraph tensors, then filter out the specified types
                all_subgraph_tensors = model_traverser.get_subgraph_output_tensors(
                    input_tensors=modelsnooper_input_config.debug_subgraph_inputs,
                    output_tensors=modelsnooper_input_config.debug_subgraph_outputs,
                )
                # Get tensors from nodes of the type to skip
                skip_tensors = model_traverser.get_subgraph_nodes_by_type(
                    op_type=modelsnooper_input_config.skip_layer_types,
                    input_tensors=modelsnooper_input_config.debug_subgraph_inputs,
                    output_tensors=modelsnooper_input_config.debug_subgraph_outputs,
                )
                # Filter out the skip tensors from all subgraph tensors
                debug_subgraph = [
                    tensor for tensor in all_subgraph_tensors if tensor not in skip_tensors
                ]
            elif modelsnooper_input_config.include_layer_types:
                debug_subgraph = model_traverser.get_subgraph_nodes_by_type(
                    op_type=modelsnooper_input_config.include_layer_types,
                    input_tensors=modelsnooper_input_config.debug_subgraph_inputs,
                    output_tensors=modelsnooper_input_config.debug_subgraph_outputs,
                )
            else:
                # No type filtering, just get subgraph output tensors
                debug_subgraph = model_traverser.get_subgraph_output_tensors(
                    input_tensors=modelsnooper_input_config.debug_subgraph_inputs,
                    output_tensors=modelsnooper_input_config.debug_subgraph_outputs,
                )
        # Scenario 1: Check if skip_layer_types is specified (without subgraph inputs/outputs)
        elif modelsnooper_input_config.skip_layer_types:
            debug_subgraph = model_traverser.get_nodes_not_of_type(
                op_type=modelsnooper_input_config.skip_layer_types
            )
        # Scenario 2: Check if include_layer_types is specified (without subgraph inputs/outputs)
        elif modelsnooper_input_config.include_layer_types:
            debug_subgraph = model_traverser.get_nodes_by_type(
                op_type=modelsnooper_input_config.include_layer_types
            )

        return debug_subgraph

    def _get_target_outputs(
        self,
        target_backend_config: BackendConfig,
        input_sample: dict[str, NDArray],
        dump_output_tensors: bool,
        inference_output_directory: Path,
        debug_subgraph: Optional[list[str]] = None,
    ) -> tuple:
        """Get inference outputs from target backend and return the DLC file.

        This method executes inference using either a pre-compiled DLC file or by compiling
        the model on-the-fly, depending on the provided configuration. It handles both offline
        preparation and direct execution scenarios.

        Args:
            target_backend_config: Configuration for the target backend.
            input_sample: Input tensors for model execution.
            dump_output_tensors: Flag to enable dumping of output tensors.
            inference_output_directory: Directory to store inference outputs.

        Returns:
            tuple: A tuple containing:
                - dict: Dictionary of inference output tensors
                - Path: Path to the DLC file used for inference
        """
        target_net_run_args = target_backend_config.net_run_arguments

        if debug_subgraph:
            debug_subgraph = [Helper.transform_node_names(name) for name in debug_subgraph]

        if target_backend_config.dlc_file:
            # If a pre-compiled DLC file is provided, use it directly
            # Set debug flag to true to enable tensor dumping
            if not target_net_run_args:
                target_net_run_args = NetRunnerInputArguments(debug=True)
            else:
                if not target_net_run_args.set_output_tensors:
                    target_net_run_args.debug = True
            # Execute inference using the provided DLC file
            inference_output_config = Snooper._execute_inference(
                model=target_backend_config.dlc_file,
                logger=self._logger,
                input_sample=input_sample,
                net_runner_args=target_net_run_args,
                net_run_backend_extension=target_backend_config.net_run_backend_extension,
                backend=target_backend_config.backend,
                platform=target_backend_config.platform,
                remote_host_details=target_backend_config.remote_host_details,
                working_directory=inference_output_directory,
                soc_model=target_backend_config.soc_model,
                dump_output_tensors=dump_output_tensors,
                wrapper_mode=True,
            )
            # Extract inference outputs from the execution results
            inference_outputs = inference_output_config.output_data[0]
            dlc_file = target_backend_config.dlc_file
        else:
            # If no DLC file is provided, we need to compile the model source model
            # Enable framework traces in converter arguments.
            if target_backend_config.converter_arguments:
                target_backend_config.converter_arguments.enable_framework_trace = True
            else:
                target_backend_config.converter_arguments = ConverterInputArguments(
                    enable_framework_trace=True
                )
            target_context_bin_args = target_backend_config.context_bin_gen_arguments
            target_converter_args = target_backend_config.converter_arguments

            # Check if the backend supports offline preparation
            is_offline_preparable = (
                target_backend_config.offline_prepare
                and target_backend_config.backend in BackendType.offline_preparable_backends()
            )

            if not is_offline_preparable:
                # If offiline prepare enabled, enable debug option to dump intermediate outputs.
                if not target_net_run_args:
                    if debug_subgraph:
                        target_net_run_args = NetRunnerInputArguments(
                            set_output_tensors=debug_subgraph, debug=False
                        )
                    else:
                        target_net_run_args = NetRunnerInputArguments(debug=True)
                else:
                    if debug_subgraph:
                        target_net_run_args.set_output_tensors = debug_subgraph
                    target_net_run_args.debug = (
                        False if target_net_run_args.set_output_tensors else True
                    )
            else:
                # For offline preparable backends, configure context bin generation
                if not target_context_bin_args:
                    if debug_subgraph:
                        target_context_bin_args = GenerateConfig(
                            set_output_tensors=debug_subgraph, enable_intermediate_outputs=False
                        )
                    else:
                        target_context_bin_args = GenerateConfig(enable_intermediate_outputs=True)
                else:
                    if debug_subgraph:
                        target_context_bin_args.set_output_tensors = debug_subgraph
                    target_context_bin_args.enable_intermediate_outputs = (
                        False if target_context_bin_args.set_output_tensors else True
                    )

                # Ensure framework traces are enabled in converter arguments
                if not target_converter_args:
                    target_converter_args = ConverterInputArguments(enable_framework_trace=True)
                else:
                    target_converter_args.enable_framework_trace = True

            inference_output_config = self._execute_inference(
                model=self.model,
                logger=self._logger,
                converter_args=target_converter_args,
                quantizer_args=target_backend_config.quantizer_arguments,
                context_bin_args=target_context_bin_args,
                context_bin_backend_extension=target_backend_config.context_bin_backend_extension,
                offline_prepare=target_backend_config.offline_prepare,
                net_runner_args=target_net_run_args,
                net_run_backend_extension=target_backend_config.net_run_backend_extension,
                input_sample=input_sample,
                backend=target_backend_config.backend,
                platform=target_backend_config.platform,
                remote_host_details=target_backend_config.remote_host_details,
                working_directory=inference_output_directory,
                soc_model=target_backend_config.soc_model,
                dump_output_tensors=dump_output_tensors,
                wrapper_mode=True,
            )

            # Extract inference outputs and DLC file path from execution results
            inference_outputs = inference_output_config.output_data[0]
            dlc_file = inference_output_config.converter_dlc

        # Store the inference output config for potential cleanup later
        self.inference_output_config = inference_output_config
        return inference_outputs, dlc_file

    def _get_tensor_mapping(
        self,
        dlc_file,
        graph_info=None,
    ) -> dict:
        """Get tensor mapping from source to target.

        When the reference outputs use sanitized tensor names (i.e., names that have already
        been transformed via Helper.transform_node_names, such as those dumped by the accuracy
        debugger framework runner), the mapping between reference and inference tensors is an
        identity mapping because both sides use the same sanitized naming convention, and
        already present in graph_info.

        Args:
            dlc_file: Path to the DLC file used for inference.
            graph_info: Dictionary containing graph information like, tensor mapping,
            graph structure and layout information.

        Returns:
            dict: Mapping from reference tensor names to target tensor names.
        """

        tensor_mapping = (
            graph_info["tensor_mapping"]
            if graph_info and ("tensor_mapping" in graph_info)
            else None
        )
        if tensor_mapping:
            return tensor_mapping # Return tensor mapping from graph info
        else:
            # Get tensor mapping from source name to target dlc name
            tensor_mapper_input_config = TensorMapperInputConfig(
                dlc_path=dlc_file, transform_dlc_node_name=False
            )
            tensor_mapping_dict = TensorMapper.run_tensor_mapping_on_dlc(tensor_mapper_input_config)
            # Swap key-value as the mapping is from target dlc -> source tensor
            return {value: key for key, value in tensor_mapping_dict.items()}

    def _generate_snooping_report(
        self,
        verifier_output: dict[tuple[str, str], dict],
        comparators: list[Comparator],
        working_directory: Path,
        inference_data: dict[str, NDArray],
        framework_activation_info: dict[str, ActivationInfo],
        target_activation_info: dict[str, ActivationInfo],
        tensor_mapping: dict,
    ) -> tuple[Path, Path]:
        """Generate a CSV snooping report from verification output data.

        Args:
            verifier_output: Output from verifier module.
            comparators: List of comparator used for verification.
            working_directory: Directory to store report.
            inference_data: Dictionary containing outputs of the model
            framework_activation_info: Dict containing info for framework activations
            target_activation_info (dict[str, ActivationInfo]): Target activation information
            tensor_mapping: dict containing mapping from source to target tensor name

        Returns:
            tuple: File paths to csv and json snooping report.
        """
        self._logger.debug(f"DEBUG: First 10 entries of tensor_mapping: {dict(list(tensor_mapping.items())[:10])}")

        snooping_report = []
        for key, value in verifier_output.items():
            framework_activation = key[1]
            sanitized_framework_activation = Helper.transform_node_names(framework_activation)
            selected_fw_activation_info = framework_activation_info.get(
                sanitized_framework_activation, None
            )

            self._logger.debug(f"DEBUG: Target Tensor Name (key[0]): {key[0]}, Reference Tensor Name (framework_activation/key[1]): {framework_activation}")
            if framework_activation not in tensor_mapping:
                self._logger.warning(f"DEBUG: framework_activation '{framework_activation}' not found in tensor_mapping!")

            report_entry = {
                "Source Name": framework_activation,
                "Target Name": tensor_mapping[framework_activation],
                "Layer Type": value["op_type"],
                "Source Shape": selected_fw_activation_info.shape
                if selected_fw_activation_info
                else [],
                "Target Shape": value["dimensions"],
                "Source(Min, Max, Median)": selected_fw_activation_info.distribution
                if selected_fw_activation_info
                else [],
                "Target(Min, Max, Median)": target_activation_info[key[0]].distribution,
            }
            # Add comparator results
            for comparator in comparators:
                report_entry[comparator.name] = value[comparator.name]
            snooping_report.append(report_entry)

        snooping_report = pd.DataFrame(snooping_report)
        snooping_report = filter_snooping_report(snooping_report, inference_data)

        csv_snooper_report_file = working_directory / "oneshot_layerwise.csv"
        snooping_report.to_csv(csv_snooper_report_file, index=False)
        json_snooper_report_file = self._generate_json_report(snooping_report, working_directory)

        return csv_snooper_report_file, json_snooper_report_file

    def _generate_json_report(self, data_frame: pd.DataFrame, output_dir: Path) -> Path:
        """Generate a JSON version of the snooping report.

        Creates a structured JSON report from the DataFrame containing snooping results.
        The JSON format includes a header with version information and a list of layers
        with their comparison metrics.

        Args:
            data_frame (pd.DataFrame): DataFrame containing the snooping report data.
            output_dir (Path): Directory where the JSON report will be saved.

        Returns:
            Path: File path to json snooper report
        """
        json_report_path = dump_json_report(data_frame, output_dir / "oneshot_layerwise.json")
        return json_report_path

    def _is_lora_enabled(self, config: ModelSnooperInputConfig) -> bool:
        """Check if LoRA mode is enabled.

        LoRA mode is enabled if either lora_model_creator_args or lora_importer_args
        is provided in the configuration.

        Args:
            config: Model Snooper input configuration.

        Returns:
            bool: True if LoRA mode is enabled, False otherwise.
        """
        return (
            config.lora_model_creator_args is not None or config.lora_importer_args is not None
        )

    def _get_lora_target_outputs(
        self,
        config: ModelSnooperInputConfig,
        input_sample: Dict[str, NDArray],
        dump_output_tensors: bool,
        inference_output_directory: Path,
        debug_subgraph: Optional[list[str]] = None,
    ) -> tuple[Dict[str, list[dict[str, NDArray]]], Path]:
        """Run inference engine with LoRA enabled.

        Creates an InferenceEngineInputConfig with all LoRA arguments and runs
        the inference engine wrapper to get outputs for all use cases.

        Args:
            config: Model Snooper input configuration.
            input_sample: Input tensors for model execution.
            dump_output_tensors: Flag to enable dumping of output tensors.
            inference_output_directory: Directory to store inference outputs.
            debug_subgraph: Optional list of intermediate output tensors to collect.

        Returns:
            tuple: A tuple containing:
                - Dict[str, list[dict]]: LoRA mode output format with keys as model names
                  ("base", "lora_base_{use_case_name}")
                - Path: Path to the DLC file used for inference
        """
        target_backend_config = config.target_config

        # Create InferenceEngineInputConfig with LoRA arguments
        inference_config = InferenceEngineInputConfig(
            input_model=self.model,
            converter_arguments=target_backend_config.converter_arguments,
            quantizer_arguments=target_backend_config.quantizer_arguments,
            backend=target_backend_config.backend,
            platform=target_backend_config.platform,
            context_bin_gen_arguments=target_backend_config.context_bin_gen_arguments,
            context_bin_backend_extension=target_backend_config.context_bin_backend_extension,
            offline_prepare=target_backend_config.offline_prepare,
            net_run_arguments=target_backend_config.net_run_arguments,
            net_run_input_data=input_sample,
            net_run_backend_extension=target_backend_config.net_run_backend_extension,
            dump_output=dump_output_tensors,
            remote_host_details=target_backend_config.remote_host_details,
            working_directory=inference_output_directory,
            soc_model=target_backend_config.soc_model,
            # LoRA-specific arguments
            lora_model_creator_args=config.lora_model_creator_args,
            lora_importer_args=config.lora_importer_args,
            use_case_names=config.use_case_names,
            lora_alpha_tensor=config.lora_alpha_tensor
        )

        # Run inference engine wrapper
        inference_wrapper = InferenceEngineWrapper(logger=self._logger)
        inference_output_config = inference_wrapper.run(inference_config)

        # Store the inference output config for potential cleanup later
        self.inference_output_config = inference_output_config

        # Extract LoRA mode outputs (Dict[str, list[dict]])
        lora_outputs = inference_output_config.output_data

        # Get DLC path
        dlc_file = (
            inference_output_config.quantizer_dlc
            if inference_output_config.quantizer_dlc
            else inference_output_config.converter_dlc
        )

        return lora_outputs, dlc_file

    def _run_lora_snooping(
        self, modelsnooper_input_config: ModelSnooperInputConfig
    ) -> tuple[Path, Path]:
        """Main LoRA snooping orchestration.

        This method handles the complete LoRA snooping workflow:
        1. Load LoRA configuration
        2. Get use cases to debug
        3. For each use case: generate framework reference outputs
        4. Run inference engine with LoRA to get all target outputs
        5. For each use case: map outputs, verify, and collect results
        6. Generate combined report with use case information

        Args:
            modelsnooper_input_config: Model Snooper input configuration.

        Returns:
            tuple: File paths to csv and json snooping report.
        """
        # Extract configuration parameters
        comparators = modelsnooper_input_config.comparators
        dump_output_tensors = modelsnooper_input_config.dump_output_tensors
        input_tensors = modelsnooper_input_config.input_sample
        retain_compilation_artifacts = modelsnooper_input_config.retain_compilation_artifacts
        working_directory = modelsnooper_input_config.working_directory

        # Get debug subgraph if specified
        debug_subgraph = None
        if modelsnooper_input_config.input_model:
            debug_subgraph = self._get_debug_subgraph(modelsnooper_input_config)

        # Load input tensors
        input_sample = load_input_tensors(input_tensors)

        # Get lora_config path
        lora_config_path = None
        if modelsnooper_input_config.lora_model_creator_args:
            lora_config_path = modelsnooper_input_config.lora_model_creator_args.lora_config
        elif modelsnooper_input_config.lora_importer_args:
            lora_config_path = modelsnooper_input_config.lora_importer_args.lora_config

        if not lora_config_path:
            raise ValueError("LoRA config path not found in lora_model_creator_args or lora_importer_args")

        # Create LoRASnooper instance
        lora_snooper = LoRASnooper(lora_config_path=lora_config_path, logger=self._logger)

        # Validate LoRA configuration
        lora_snooper.validate_lora_config()

        # Get use cases to debug
        if modelsnooper_input_config.use_case_names:
            use_cases_to_debug = modelsnooper_input_config.use_case_names
            self._logger.info(f"Debugging specified use cases: {use_cases_to_debug}")
        else:
            # Debug all non-base use cases
            use_cases_to_debug = lora_snooper.get_non_base_use_case_names()
            self._logger.info(f"Debugging all non-base use cases: {use_cases_to_debug}")

        # Check if reference backend config is provided (config mode)
        reference_backend_config = modelsnooper_input_config.reference_config

        # Generate reference outputs for each use case
        reference_outputs_per_use_case = {}

        if reference_backend_config:
            # Config mode: Generate reference outputs using inference engine with reference backend
            # Run inference engine without LoRA arguments for each use case model separately
            self._logger.info("Config mode detected: Generating reference outputs using inference engine with reference backend")

            # Set up reference inference output directory
            reference_inference_output_directory = working_directory / "reference_inference_engine"
            reference_inference_output_directory.mkdir(exist_ok=True)

            # Run inference engine for each use case model separately (without LoRA args)
            for use_case_name in use_cases_to_debug:
                self._logger.info(f"Running reference inference engine for use case: {use_case_name}")

                # Get the model path for this use case from lora_config
                use_case_model_path = lora_snooper.get_use_case_model_path(use_case_name)
                if use_case_model_path is None:
                    self._logger.warning(
                        f"Could not find model path for use case '{use_case_name}', skipping"
                    )
                    continue

                # Create use case specific output directory
                use_case_output_dir = reference_inference_output_directory / use_case_name
                use_case_output_dir.mkdir(exist_ok=True)

                # Run inference engine without LoRA arguments for this specific use case model
                self._logger.info(f"Executing inference for use case '{use_case_name}' model: {use_case_model_path}")

                inference_output_config = self._execute_inference(
                    model=use_case_model_path,
                    logger=self._logger,
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
                    working_directory=use_case_output_dir,
                    soc_model=reference_backend_config.soc_model,
                    dump_output_tensors=dump_output_tensors,
                    wrapper_mode=True,
                )

                # Extract outputs for this use case
                reference_outputs = inference_output_config.output_data[0]
                reference_outputs_per_use_case[use_case_name] = reference_outputs

                self._logger.info(f"Generated {len(reference_outputs)} reference outputs for use case '{use_case_name}'")
        else:
            # Generate reference outputs using framework runner
            self._logger.info("Generating reference outputs using framework runner")
            for use_case_name in use_cases_to_debug:
                self._logger.info(f"Generating framework reference outputs for use case: {use_case_name}")
                framework_outputs = lora_snooper.generate_framework_reference_outputs(
                    use_case_name=use_case_name,
                    input_sample=input_sample,
                    debug_subgraph=debug_subgraph,
                    dump_output_tensors=dump_output_tensors,
                    output_dir=working_directory,
                )
                reference_outputs_per_use_case[use_case_name] = framework_outputs

        # Set up inference output directory
        inference_output_directory = working_directory / "inference_engine"
        inference_output_directory.mkdir(exist_ok=True)

        # Prepend lora_alpha tensor to input_sample for inference engine.
        # The LoRA model creator adds lora_alpha as the first input to the DLC model,
        # so we must prepend it to every inference input set.
        lora_inference_input = self._prepend_lora_alpha(
            input_sample, modelsnooper_input_config.lora_alpha_tensor
        )

        # Run inference engine with LoRA to get all target outputs
        self._logger.info("Running inference engine with LoRA enabled")
        lora_target_outputs, dlc_file = self._get_lora_target_outputs(
            modelsnooper_input_config,
            lora_inference_input,
            dump_output_tensors,
            inference_output_directory,
            debug_subgraph=debug_subgraph,
        )

        # Collect verification results for all use cases
        all_verifier_outputs = []
        all_tensor_mappings = []
        all_framework_activation_infos = []
        all_target_activation_infos = []
        all_inference_outputs = []

        # Process each use case
        for use_case_name in use_cases_to_debug:
            self._logger.info(f"Processing use case: {use_case_name}")

            # Get reference outputs for this use case
            reference_outputs = reference_outputs_per_use_case.get(use_case_name)
            is_qnn_golden_reference = False
            graph_info = None
            if reference_outputs:
                graph_info = {"tensor_mapping": {name: name for name in reference_outputs.keys()}}
                is_qnn_golden_reference = True
            else:
                self._logger.warning(
                    f"Could not find reference outputs for use case '{use_case_name}', skipping"
                )

            # Map inference engine output to this use case
            inference_outputs = lora_snooper.map_inference_output_to_use_case(
                lora_target_outputs, use_case_name
            )

            if inference_outputs is None:
                self._logger.warning(
                    f"Could not find inference outputs for use case '{use_case_name}', skipping"
                )
                continue

            # Verify outputs for this use case
            verifier_output = verify(
                reference_outputs,
                inference_outputs,
                self._logger,
                dlc_file,
                comparators,
                graph_info=graph_info,
                is_qnn_golden_reference=is_qnn_golden_reference,
            )

            if not verifier_output:
                self._logger.warning(
                    f"Verification failed for use case '{use_case_name}'"
                )
                continue

            tensor_mapping_dict = self._get_tensor_mapping(dlc_file, graph_info)

            # Get profile information
            framework_activation_info, target_activation_info, _ = self._get_profile_info(
                reference_outputs, dlc_file
            )

            # Set target min, max, median values
            for inference_tensor, tensor_data in inference_outputs.items():
                if inference_tensor in target_activation_info:
                    target_activation_info[inference_tensor].distribution = (
                        np.min(tensor_data),
                        np.max(tensor_data),
                        np.median(tensor_data),
                    )

            # Store results with use case information
            all_verifier_outputs.append((use_case_name, verifier_output))
            all_tensor_mappings.append((use_case_name, tensor_mapping_dict))
            all_framework_activation_infos.append((use_case_name, framework_activation_info))
            all_target_activation_infos.append((use_case_name, target_activation_info))
            all_inference_outputs.append((use_case_name, inference_outputs))

        # Generate combined report with use case information
        csv_snooping_report_path, json_snooping_report_path = self._generate_lora_snooping_report(
            all_verifier_outputs,
            comparators,
            working_directory,
            all_inference_outputs,
            all_framework_activation_infos,
            all_target_activation_infos,
            all_tensor_mappings,
        )

        # Plot graphs based on the snooping report data
        plot_graphs(
            csv_path=csv_snooping_report_path,
            layer_names_column="Source Name",
            comparator_columns=[comparator.name for comparator in comparators],
            output_dir=working_directory,
            algorithm=self._name,
            logger=self._logger,
        )

        # Clean up artifacts if not needed
        if not retain_compilation_artifacts and self.inference_output_config:
            self._logger.info(
                f"Cleaning up compilation artifacts:\n"
                f"Converter DLC: {self.inference_output_config.converter_dlc}\n"
                f"Quantizer DLC: {self.inference_output_config.quantizer_dlc}\n"
                f"Offline Graph: {self.inference_output_config.offline_graph}"
            )
            self.inference_output_config.cleanup_artifacts()

        return csv_snooping_report_path, json_snooping_report_path

    def _generate_lora_snooping_report(
        self,
        all_verifier_outputs: list[tuple[str, dict]],
        comparators: list[Comparator],
        working_directory: Path,
        all_inference_outputs: list[tuple[str, dict[str, NDArray]]],
        all_framework_activation_infos: list[tuple[str, dict[str, ActivationInfo]]],
        all_target_activation_infos: list[tuple[str, dict[str, ActivationInfo]]],
        all_tensor_mappings: list[tuple[str, dict]],
    ) -> tuple[Path, Path]:
        """Generate a combined CSV snooping report for all LoRA use cases.

        Args:
            all_verifier_outputs: List of (use_case_name, verifier_output) tuples.
            comparators: List of comparators used for verification.
            working_directory: Directory to store report.
            all_inference_outputs: List of (use_case_name, inference_data) tuples.
            all_framework_activation_infos: List of (use_case_name, framework_activation_info) tuples.
            all_target_activation_infos: List of (use_case_name, target_activation_info) tuples.
            all_tensor_mappings: List of (use_case_name, tensor_mapping) tuples.

        Returns:
            tuple: File paths to csv and json snooping report.
        """
        snooping_report = []

        # Process each use case
        for idx, (use_case_name, verifier_output) in enumerate(all_verifier_outputs):
            tensor_mapping = all_tensor_mappings[idx][1]
            framework_activation_info = all_framework_activation_infos[idx][1]
            target_activation_info = all_target_activation_infos[idx][1]
            inference_data = all_inference_outputs[idx][1]

            for key, value in verifier_output.items():
                framework_activation = key[1]
                sanitized_framework_activation = Helper.transform_node_names(framework_activation)
                selected_fw_activation_info = framework_activation_info.get(
                    sanitized_framework_activation, None
                )

                report_entry = {
                    "Use Case": use_case_name,
                    "Source Name": framework_activation,
                    "Target Name": tensor_mapping[framework_activation],
                    "Layer Type": value["op_type"],
                    "Source Shape": selected_fw_activation_info.shape
                    if selected_fw_activation_info
                    else [],
                    "Target Shape": value["dimensions"],
                    "Source(Min, Max, Median)": selected_fw_activation_info.distribution
                    if selected_fw_activation_info
                    else [],
                    "Target(Min, Max, Median)": target_activation_info[key[0]].distribution,
                }
                # Add comparator results
                for comparator in comparators:
                    report_entry[comparator.name] = value[comparator.name]
                snooping_report.append(report_entry)

        snooping_report = pd.DataFrame(snooping_report)

        # Filter report (combine all inference outputs for filtering)
        combined_inference_data = {}
        for _, inference_data in all_inference_outputs:
            combined_inference_data.update(inference_data)
        snooping_report = filter_snooping_report(snooping_report, combined_inference_data)

        csv_snooper_report_file = working_directory / "oneshot_layerwise.csv"
        snooping_report.to_csv(csv_snooper_report_file, index=False)
        json_snooper_report_file = self._generate_json_report(
            snooping_report, working_directory)

        return csv_snooper_report_file, json_snooper_report_file

    def _prepend_lora_alpha(
        self, input_sample: Dict[str, NDArray], lora_alpha_tensor_path: Path
    ) -> Dict[str, NDArray]:
        """Prepend lora_alpha tensor to the input sample for inference engine.

        The LoRA model creator adds lora_alpha as the first input to the DLC model.
        This method loads the lora_alpha raw file and prepends it to the input dict
        so the inference engine receives the correct number of inputs.

        Args:
            input_sample: Dictionary mapping input names to numpy arrays (framework inputs).
            lora_alpha_tensor_path: Path to the lora_alpha raw tensor file.

        Returns:
            New dict with lora_alpha as the first key, followed by the original inputs.
        """
        # Load lora_alpha tensor from raw file as float32
        lora_alpha_array = np.fromfile(str(lora_alpha_tensor_path), dtype=np.float32)
        self._logger.debug(
            f"Loaded lora_alpha tensor from {lora_alpha_tensor_path}, "
            f"shape: {lora_alpha_array.shape}, dtype: {lora_alpha_array.dtype}"
        )

        # Create new dict with lora_alpha as the first key (Python 3.7+ preserves insertion order)
        lora_input = {"lora_alpha": lora_alpha_array}
        lora_input.update(input_sample)
        return lora_input
