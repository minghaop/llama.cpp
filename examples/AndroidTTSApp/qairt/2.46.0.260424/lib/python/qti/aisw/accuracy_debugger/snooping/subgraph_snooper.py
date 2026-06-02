# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import logging
import multiprocessing
import re
import time
from abc import ABC, abstractmethod
from multiprocessing import Lock
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psutil
from pandas import DataFrame
from qti.aisw.accuracy_debugger.common_config import (
    ConverterInputArguments,
    NetRunnerInputArguments,
    QuantizerInputArguments,
    RemoteHostDetails,
)
from qti.aisw.accuracy_debugger.encodings.encodings import ModelEncoding
from qti.aisw.accuracy_debugger.encodings.encodings_utils import (
    EncodingVersion,
    get_encodings_version,
)
from qti.aisw.accuracy_debugger.encodings_converter.qairt_encodings_converter import (
    QairtEncodingsConverter,
)
from qti.aisw.accuracy_debugger.framework_runner.framework_factory import get_framework_instance
from qti.aisw.accuracy_debugger.model_snooper.config import (
    ModelSnooperInputConfig,
)
from qti.aisw.accuracy_debugger.snooping.snooper import Snooper
from qti.aisw.accuracy_debugger.snooping.snooper_utils import (
    COMPILATIONS_LIMIT,
    InferenceEngineProcess,
    get_max_parallel_compilations,
)
from qti.aisw.accuracy_debugger.utils.constants import CSV_TO_JSON_FIELDS_MAP, MATH_INVARIANT_OPS
from qti.aisw.accuracy_debugger.utils.exceptions import VerificationError
from qti.aisw.accuracy_debugger.utils.file_utils import dump_csv, dump_json, read_json
from qti.aisw.accuracy_debugger.utils.graph_utils import (
    get_common_parent_activations,
    get_subgraph,
    get_supergroup_activations,
    get_topological_order,
)
from qti.aisw.accuracy_debugger.utils.helper import (
    ActivationInfo,
    ActivationStatus,
    load_input_tensors,
    plot_graphs,
)
from qti.aisw.tools.core.modules.api.definitions.common import BackendType
from qti.aisw.tools.core.modules.context_bin_gen.context_bin_gen_module import GenerateConfig
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.verifier.verifier import Verifier


class SubgraphSnooper(Snooper, ABC):
    """SubgraphSnooper class for snooping subgraphs."""

    def __init__(self, name: str, logger: logging.Logger, is_cumulative: bool):
        """Initializes the SubgraphSnooper.

        Args:
            name (str): The name of the snooper algorithm
            logger (logging.Logger): A python logger instance
            is_cumulative (bool): Specifies whether the algorithm is layerwise or cumulative.
        """
        super().__init__(name, logger)
        self.is_cumulative = is_cumulative

    @abstractmethod
    def _get_subgraph_info(
        self,
        target_activation: str,
        target_activation_op_map: dict,
        framework_activation_op_map: dict,
        supergroup_activations: set,
    ) -> tuple[set, set, set, set]:
        """Abstract method which must be implemented by the snoopers which inherites
        SubgraphSnooper. This is required because subgraph construction can be different
        for different algorithms

        Args:
            target_activation (str): activation name in the target graph.
            target_activation_op_map (dict): Target activations to Target op map.
            framework_activation_op_map (dict): Framework activation to framework op map.
            supergroup_activations (set): Activations of supergroups.

        Returns:
            tuple(set, set, set, set): tuple of following informations:
                1.set of all subgraph inputs
                2.set of all subgraph outputs
                3.set of all target subgraph activations(inputs to the subgraph are not included)
                4.set of all framework subgraph activations(inputs to the subgraph are not included)
        """
        pass

    def _get_subgraph_inputs(
        self,
        target_activation: str,
        target_activation_op_map: dict,
        framework_activation_op_map: dict,
        supergroup_activations: set,
    ) -> set:
        """Given target_activation return the layerwise and cumulative subgraph's input activations
        Args:
            target_activation (str): activation name in the target graph.
            target_activation_op_map (dict): Target activations to Target op map.
            framework_activation_op_map (dict): Framework activation to framework op map.
            supergroup_activations (set): Activations of supergroups.

        Returns:
            (set): set of all subgraph inputs
        """
        target_op = target_activation_op_map[target_activation]

        # Get subgraph inputs
        subgraph_inputs = set()
        for input_name in target_op.inputs:
            # some input_name can be param
            if input_name in target_activation_op_map:
                common_parent_activations = get_common_parent_activations(
                    input_name,
                    target_activation_op_map,
                    framework_activation_op_map,
                    supergroup_activations,
                )
                subgraph_inputs.update(common_parent_activations)

        return subgraph_inputs

    def _get_all_subgraphs(
        self,
        framework_activation_op_map: dict,
        debug_graph_activations: list,
        target_activation_op_map: dict,
        supergroup_activations: set,
        resolved_target_activations: dict,
        target_activation_info: dict[str, ActivationInfo],
        framework_activation_info: dict[str, ActivationInfo],
        verifier_scores: dict,
        comparators: list,
        data_frame: DataFrame,
        output_tensor: list,
        qairt_encodings_converter: QairtEncodingsConverter,
        output_dir: Path,
        encoding_version: EncodingVersion,
        compulsory_overrides: Optional[Path] = None,
    ) -> tuple[dict, dict]:
        """Each op in the target graph represents a subgraph post optimizations.
        Those optimizations can be backend aware depends upon such implementations
        at the qairt-converter level.
        Maps each op in the target graph to the framework subgraph, subsequently
        finds out the subgraph quantization overrides.

        Args:
            framework_activation_op_map: Framework activation to framework op map.
            debug_graph_activations: Activations from the debug graph.
            target_activation_op_map: Target activations to Target op map.
            supergroup_activations: Activations of supergroups.
            resolved_target_activations: Framework name to target activation name.
            target_activation_info: Target Activation information.
            verifier_scores: Comparator scores from verifier.
            comparators: List of comparators.
            framework_activation_info: Framework activation info.
            data_frame: DataFrame to be updated with the results.
            output_tensor: Output tensor name.
            qairt_encodings_converter: QairtEncodingsConverter object.
            output_dir: Path to output directory.
            encoding_version: Version in which subgraph encoding override needs to be dumped.
            compulsory_overrides: Path to compulsory overrides.

        Returns:
            tuple[dict, dict]: All Subgraph, Data going into the csv report
        """
        subgraphs = {}
        for activation in debug_graph_activations:
            target_op = target_activation_op_map[activation]
            self._activation_status[activation] = ActivationStatus(activation)

            (
                subgraph_inputs,
                subgraph_outputs,
                target_subgraph_activations,
                framework_subgraph_activations,
            ) = self._get_subgraph_info(
                activation,
                target_activation_op_map,
                framework_activation_op_map,
                supergroup_activations,
            )

            subgraphs[activation] = {
                "Inputs": ",".join(subgraph_inputs),
                "Outputs": ",".join(subgraph_outputs),
                "Target Tensors": ",".join(target_subgraph_activations),
                "Framework Tensors": ",".join(framework_subgraph_activations),
                "layer_type": target_op.op_type,
            }

            skip, reason = self._should_be_skipped(
                target_subgraph_activations, activation, target_activation_op_map
            )

            if skip:
                status_msg = (
                    f"Skipping target subgraph {target_subgraph_activations} "
                    f"as it is of type {reason}"
                )
                self._activation_status[activation].set_status(ActivationStatus.SKIP, status_msg)
                self._logger.info(status_msg)
                data_frame = self._build_data_frame_for_subgraph(
                    activation,
                    resolved_target_activations,
                    target_activation_info,
                    verifier_scores,
                    subgraphs[activation]["layer_type"],
                    comparators,
                    framework_activation_info,
                    data_frame,
                    output_tensor,
                )
                subgraphs[activation]["status"] = ActivationStatus.SKIP
                subgraphs[activation]["status_msg"] = status_msg
                subgraphs[activation]["override_file_path"] = ""
            else:
                # handle cases like split node
                # subgraph_last_op ---> (out1, out2)
                # add all subgraph outputs to the subgraph
                target_subgraph_activations.update(subgraph_outputs)

                self._logger.debug(
                    f"Subgraph {target_subgraph_activations} starts "
                    f"with {subgraph_inputs} and ends with {subgraph_outputs}"
                )
                try:
                    self._logger.debug("Creating subgraph override")
                    subgraph_override_file_path = self._create_subgraph_quantization_override(
                        target_subgraph_activations,
                        target_op.outputs,
                        supergroup_activations,
                        qairt_encodings_converter,
                        output_dir,
                        encoding_version,
                        compulsory_overrides,
                    )
                    subgraphs[activation]["override_file_path"] = str(subgraph_override_file_path)
                except Exception as e:
                    self._logger.error(
                        f"Failed to create quantization overrides for {target_subgraph_activations}"
                        f" with error: {str(e)}"
                    )
                    status_msg = ""
                    subgraphs[activation]["override_file_path"] = ""
                    subgraphs[activation]["status"] = (
                        ActivationStatus.CUSTOM_OVERRIDE_GENERATION_FAILURE
                    )
                    subgraphs[activation]["status_msg"] = status_msg
                    self._activation_status[activation].set_status(
                        ActivationStatus.CUSTOM_OVERRIDE_GENERATION_FAILURE, status_msg
                    )
                    data_frame = self._build_data_frame_for_subgraph(
                        activation,
                        resolved_target_activations,
                        target_activation_info,
                        verifier_scores,
                        subgraphs[activation]["layer_type"],
                        comparators,
                        framework_activation_info,
                        data_frame,
                        output_tensor,
                    )

        return subgraphs, data_frame

    def _calculate_compilation_memory(
        self,
        output_dir: Path,
        backend: BackendType,
        platform: DevicePlatformType,
        remote_host_details: RemoteHostDetails,
        soc_model: str,
        converter_args: ConverterInputArguments,
        quantizer_args: QuantizerInputArguments,
        context_bin_args: GenerateConfig,
        context_bin_backend_extension: Path | dict,
        quantization_overrides: Path,
    ) -> int:
        """Computes the maximum RAM required by a subgraph process
        Args:
            output_dir: Output directory path
            converter_args: Input arguments required by the converter module.
            quantizer_args: Input arguments required by the quantizer module.
            context_bin_args: Input arguments required by the context_bin_gen module.
            context_bin_backend_extension: Backend extension config file or dictionary for context
                                           binary generator.
            backend: Type of backend
            platform: Platform of target device (android, wos, x86_64_linux, etc.)
            remote_host_details: Details of remote host
            soc_model: Name of SOC model on target device.
            quantization_overrides: Path to a quantization override file.

        Returns:
            int: RAM usage by a subgraph IE process in bytes
        """
        compilation_run_dir = output_dir / "compilation_run"
        quantizer_args = quantizer_args.model_copy(
            update={
                "input_list": None,
                "float_fallback": True,
            }
        )
        converter_args = converter_args.model_copy(
            update={"quantization_overrides": quantization_overrides}
        )
        inference_engine_config = {
            "model": self.model,
            "converter_args": converter_args,
            "quantizer_args": quantizer_args,
            "offline_prepare": True,
            "context_bin_args": context_bin_args,
            "backend": backend,
            "working_directory": compilation_run_dir,
            "soc_model": soc_model,
            "context_bin_backend_extension": context_bin_backend_extension,
            "platform": platform,
            "remote_host_details": remote_host_details,
            "log_level": logging.getLevelName(self._logger.level),
            "logger_name": "Inference Engine(Compilation Run)",
        }
        compilation_memory = 0
        initial_ie_process = multiprocessing.Process(
            target=Snooper._execute_inference,
            kwargs=inference_engine_config,
        )
        try:
            initial_ie_process.start()
            child = psutil.Process(initial_ie_process.pid)
            while initial_ie_process.is_alive():
                compilation_memory = max(compilation_memory, child.memory_info().rss)
                time.sleep(1)
        except Exception as e:
            self._logger.error(f"Error during compilation memory calculation: {e}")
            if initial_ie_process.is_alive():
                initial_ie_process.terminate()
                initial_ie_process.join(timeout=10)
            raise
        finally:
            initial_ie_process.join()
        self._logger.debug(f"Maximum RAM usage for one subgraph: {compilation_memory} Bytes")

        return compilation_memory

    def run(self, modelsnooper_input_config: ModelSnooperInputConfig) -> tuple[Path, Path]:
        """Run method used by layerwise and cumulative layerwise snooping classes.

        Args:
            modelsnooper_input_config: Model Snooper input configuration.

        Returns:
            tuple: File paths to csv and json snooping report.

        Raises:
            Exception: If inference engine fails to generate qairt encodings (or)
                Fails to create QairtEncodingsConverter Object
        """
        modelsnooper_input_config = self._setup(modelsnooper_input_config=modelsnooper_input_config)

        # Extract configuration from modelsnooper_input_config
        comparators = modelsnooper_input_config.comparators
        compulsory_overrides = modelsnooper_input_config.compulsory_overrides
        debug_subgraph_inputs = modelsnooper_input_config.debug_subgraph_inputs
        debug_subgraph_outputs = modelsnooper_input_config.debug_subgraph_outputs
        dump_output_tensors = modelsnooper_input_config.dump_output_tensors
        golden_reference_path = modelsnooper_input_config.golden_reference_path
        input_tensors = modelsnooper_input_config.input_sample
        working_directory = modelsnooper_input_config.working_directory

        # Extract target configuration
        target_config = modelsnooper_input_config.target_config
        converter_args = target_config.converter_arguments
        quantizer_args = target_config.quantizer_arguments
        context_bin_args = target_config.context_bin_gen_arguments
        context_bin_backend_extension = target_config.context_bin_backend_extension
        offline_prepare = target_config.offline_prepare
        net_runner_args = target_config.net_run_arguments
        net_run_backend_extension = target_config.net_run_backend_extension
        backend = target_config.backend
        platform = target_config.platform
        soc_model = target_config.soc_model
        remote_host_details = target_config.remote_host_details

        # Store the working directory path.
        net_run_backend_extension = (
            modelsnooper_input_config.target_config.net_run_backend_extension
        )
        retain_compilation_artifacts = modelsnooper_input_config.retain_compilation_artifacts
        is_qnn_golden_reference = modelsnooper_input_config.is_qnn_golden_reference

        self._logger.info(f"Started {self._name} snooping")

        csv_filename = "cumulative_layerwise.csv" if self.is_cumulative else "layerwise.csv"
        csv_path = working_directory / csv_filename

        verifier_scores = {}
        if converter_args and converter_args.output_tensors:
            output_tensor = [output_tensor.name for output_tensor in converter_args.output_tensors]
        else:
            output_tensor = self._get_output_tensor_names()

        input_sample = load_input_tensors(input_tensors)

        # Generate or load golden reference outputs
        golden_reference_output = self._get_reference_outputs(
            model=self.model,
            golden_reference_path=golden_reference_path,
            input_sample=input_sample,
            dump_output_tensors=dump_output_tensors,
            working_directory=working_directory,
        )

        all_subgraphs = {}

        # Modify input tensors to handle datatypes not supported by the given backend
        # If all the datatypes are supported, the input remains unchanged
        input_sample, quantizer_args = self._resolve_unsupported_dtypes(
            input_tensors,
            input_sample,
            quantizer_args,
            working_directory,
        )

        comparator_names = [comparator.name for comparator in comparators]
        data_frame = self._initialize_data_frame(comparator_names, output_tensor)

        # Generate quantization encodings
        quantizer_args.dump_encoding_json = True
        inference_output_directory = working_directory / "inference_engine"
        inference_output_directory.mkdir(exist_ok=True)
        initial_run_output_directory = inference_output_directory / "initial_run"
        initial_run_output_directory.mkdir(exist_ok=True)
        try:
            self._logger.info("Generate quantization encodings")
            inference_output_config = Snooper._execute_inference(
                model=self.model,
                converter_args=converter_args,
                quantizer_args=quantizer_args,
                backend=backend,
                working_directory=initial_run_output_directory,
                soc_model=soc_model,
                remote_host_details=remote_host_details,
                log_level=logging.getLevelName(self._logger.level),
                logger_name="Inference Engine(Initial Run)",
            )
        except Exception as exception:
            self._logger.error(f"Failed to generate encodings. Reason: {exception}")
            raise exception

        # If quantization_overrides has been provided by the user, subgraph encoding
        # override must be generated in the same version.
        if converter_args.quantization_overrides:
            encoding_version = get_encodings_version(
                read_json(converter_args.quantization_overrides)
            )
        elif quantizer_args.use_quantize_v2:
            # In case of use_quantize_v2 = True, we need to make it False hereforth for all subgraph
            # as it will no longer be needed since no calibration data will be passed to the
            # quantizer
            encoding_version = EncodingVersion.V2
            quantizer_args.use_quantize_v2 = False
        else:
            encoding_version = EncodingVersion.V0

        quantized_dlc_path = inference_output_config.quantizer_dlc
        qairt_encodings_converter = self._get_encodings_converter(
            working_directory, self.model, quantized_dlc_path, encoding_version
        )
        framework_activation_op_map = qairt_encodings_converter.get_framework_activation_op_map()
        target_activation_op_map = qairt_encodings_converter.get_target_activation_op_map()
        resolved_target_activations = qairt_encodings_converter.get_resolved_target_activation()

        supergroup_activations = get_supergroup_activations(
            framework_activation_op_map, target_activation_op_map
        )
        all_subgraphs["ignore_activations"] = ",".join(supergroup_activations)

        self._logger.info("Generating debug subgraph")
        debug_graph_activations, debug_graph_input_names, debug_graph_output_names = (
            self._get_debug_graph(
                framework_activation_op_map,
                target_activation_op_map,
                supergroup_activations,
                debug_subgraph_inputs,
                debug_subgraph_outputs,
            )
        )

        # Populate topologically sorted activations and set them as the index.
        data_frame["Source Name"] = debug_graph_activations
        data_frame.set_index("Source Name", inplace=True)

        all_subgraphs["debug_graph_input_tensors"] = ",".join(debug_graph_input_names)
        all_subgraphs["debug_graph_output_tensors"] = ",".join(debug_graph_output_names)
        all_subgraphs["debug_graph_activations"] = ",".join(debug_graph_activations)
        all_subgraphs["subgraphs"] = {}

        # Get Activation info for framework and target as well as layout data
        framework_activation_info, target_activation_info, layout_info = self._get_profile_info(
            golden_reference_output, quantized_dlc_path
        )

        # Generate subgraphs
        self._logger.info("Generating subgraphs from debug graph")
        all_subgraphs["subgraphs"], data_frame = self._get_all_subgraphs(
            framework_activation_op_map,
            debug_graph_activations,
            target_activation_op_map,
            supergroup_activations,
            resolved_target_activations,
            target_activation_info,
            framework_activation_info,
            verifier_scores,
            comparators,
            data_frame,
            output_tensor,
            qairt_encodings_converter,
            working_directory,
            encoding_version,
            compulsory_overrides,
        )

        # Dump all_subgraphs for user to check the identified subgraphs
        all_subgraphs_path = working_directory / "all_subgraphs.json"
        dump_json(all_subgraphs, all_subgraphs_path)

        # Calculating Compilation memory
        compilation_dir = inference_output_directory / "compilation_run"
        compilation_dir.mkdir(exist_ok=True)
        compilation_override_path = compilation_dir / "compilation_override.json"
        qairt_encodings_converter._model_encoding.dump(
            version=encoding_version, file_path=compilation_override_path
        )
        compilation_memory = self._calculate_compilation_memory(
            output_dir=inference_output_directory,
            backend=backend,
            platform=platform,
            remote_host_details=remote_host_details,
            soc_model=soc_model,
            converter_args=converter_args,
            quantizer_args=quantizer_args,
            context_bin_args=context_bin_args,
            context_bin_backend_extension=context_bin_backend_extension,
            quantization_overrides=compilation_override_path,
        )

        # Execute the generated subgraphs
        self._logger.info("Executing generated subgraphs")
        csv_path, data_frame = self._execute_all_sub_graphs(
            output_dir=inference_output_directory,
            all_subgraphs=all_subgraphs,
            debug_graph_activations=debug_graph_activations,
            resolved_target_activations=resolved_target_activations,
            converter_args=converter_args,
            quantizer_args=quantizer_args,
            context_bin_args=context_bin_args,
            context_bin_backend_extension=context_bin_backend_extension,
            offline_prepare=offline_prepare,
            net_runner_args=net_runner_args,
            net_run_backend_extension=net_run_backend_extension,
            input_sample=input_sample,
            backend=backend,
            platform=platform,
            remote_host_details=remote_host_details,
            soc_model=soc_model,
            comparators=comparators,
            layout_info=layout_info,
            golden_reference_output=golden_reference_output,
            data_frame=data_frame,
            output_tensor=output_tensor,
            target_activation_info=target_activation_info,
            framework_activation_info=framework_activation_info,
            verifier_scores=verifier_scores,
            csv_path=csv_path,
            retain_compilation_artifacts=retain_compilation_artifacts,
            dump_output_tensors=dump_output_tensors,
            is_qnn_golden_reference=is_qnn_golden_reference,
            user_max_parallel_compilations=modelsnooper_input_config.max_parallel_compilations,
            compilation_memory=compilation_memory,
        )

        json_report_path = self._generate_json_report(data_frame, working_directory)

        # Re-dump the all_subgraphs to update the 'status' field.
        dump_json(all_subgraphs, all_subgraphs_path)

        self._plot_comparator_scores(csv_path, comparators, output_tensor, working_directory)

        return csv_path, json_report_path

    def _plot_comparator_scores(
        self,
        csv_path: Path,
        comparators: list[Comparator],
        output_tensor: list[str],
        output_dir: Path,
    ) -> None:
        """Plots and dumps scores of all verifiers/comparators for both current layer of each
        subgraph and actual original outputs of the model.

        Args:
            csv_path: Path to the CSV snooper report
            comparators: List of comparators to use in verification stage.,
            output_tensor: Output tensors of the framework model
            output_dir: Output directory path
        """
        comparator_columns = []
        for comparator in comparators:
            comparator_columns.append(f"{comparator.name}(current_layer)")
            for output_name in output_tensor:
                comparator_columns.append(f"{comparator.name}({output_name})")

        plot_graphs(
            csv_path=csv_path,
            layer_names_column="Source Name",
            comparator_columns=comparator_columns,
            output_dir=output_dir,
            algorithm=self._name,
            logger=self._logger,
        )

    def _get_encodings_converter(
        self,
        output_dir: Path,
        model_path: Path,
        quantized_dlc_path: Path,
        encoding_version: EncodingVersion,
    ) -> QairtEncodingsConverter:
        """Method consumes the model, quantized dlc and quantization overrides
        to return the qairt encodings converter object.

        Args:
            output_dir (Path): Output directory.
            model_path (Path): Path to framework model.
            quantized_dlc_path (Path): Path to quantized DLC.
            encoding_version (EncodingVersion): Version in which subgraph encoding override needs
                to be dumped.

        Returns:
            QairtEncodingsConverter: QAIRT Encoding converter object

        Raises:
            Exception: If QairtEncodingsConverter object creation fails
        """
        working_dir = output_dir / "encodings_converter"
        try:
            qairt_encodings_converter = QairtEncodingsConverter(
                str(model_path),
                str(quantized_dlc_path),
                str(working_dir),
                self._logger,
            )
            quantized_model_encoding = qairt_encodings_converter.create_subgraph_encodings()
        except Exception as exception:
            raise Exception(
                f"QairtEncodingsConverter object creation failed with error: {exception}"
            )
        converted_encodings_file_path = working_dir / "converted_encodings.json"
        quantized_model_encoding.dump(
            version=encoding_version, file_path=converted_encodings_file_path
        )

        return qairt_encodings_converter

    def _get_debug_graph(
        self,
        framework_activation_op_map: dict,
        target_activation_op_map: dict,
        supergroup_activations: set,
        debug_subgraph_inputs: Optional[list] = None,
        debug_subgraph_outputs: Optional[list] = None,
    ) -> tuple[list, set, set]:
        """Generate subgraph for debugging

        Args:
            framework_activation_op_map: Framework activation to framework op map.
            target_activation_op_map: Target activations to Target op map.
            supergroup_activations: Activations of supergroups.
            debug_subgraph_inputs: list of inputs to debug subgraph
            debug_subgraph_outputs: list of outputs of debug subgraph

        Returns:
            tuple[list, set, set]: Activations, input names and output names of debug graph

        Raises:
            Exception: if debug subgraph turns out to be empty
        """
        debug_framework_graph_inputs = (
            set(debug_subgraph_inputs) if debug_subgraph_inputs else set()
        )
        debug_framework_graph_outputs = (
            set(debug_subgraph_outputs) if debug_subgraph_outputs else set()
        )

        for activation_name, framework_op in framework_activation_op_map.items():
            # If debug subgraph inputs are not provided initialize with framework model inputs
            if not debug_subgraph_inputs and framework_op.op_type == "input":
                debug_framework_graph_inputs.update([activation_name])

            # If debug subgraph outputs are not provided initialize with framework model outputs
            if not debug_subgraph_outputs and not framework_op.children_ops:
                debug_framework_graph_outputs.update([activation_name])

        # Align debug framework subgraph inputs with dlc graph such that we do not break
        # any fusion patterns.
        debug_graph_input_names = set()
        for input_name in debug_framework_graph_inputs:
            partial_inputs = get_common_parent_activations(
                input_name,
                framework_activation_op_map,
                target_activation_op_map,
                supergroup_activations,
            )
            debug_graph_input_names.update(partial_inputs)

        # Align debug framework subgraph outputs with dlc graph such that we do not break
        # any fusion patterns.
        debug_graph_output_names = set()
        for output_name in debug_framework_graph_outputs:
            partial_outputs = get_common_parent_activations(
                output_name,
                framework_activation_op_map,
                target_activation_op_map,
                supergroup_activations,
            )
            debug_graph_output_names.update(partial_outputs)

        # Following situations will occur:
        # 1. Both of the debug_subgraph_inputs and debug_subgraph_outputs not provided
        # 2. Either of the debug_subgraph_inputs or debug_subgraph_outputs are provided
        if not (debug_subgraph_inputs or debug_subgraph_outputs):
            target_activations = set(target_activation_op_map.keys()) - debug_graph_input_names
            # filter out all the target activations which are not part of framework graph like
            # convert ops/ extra target ops added as no point debugging them bcz they do not
            # have corresponding framework ops.
            target_activations = target_activations.intersection(framework_activation_op_map.keys())
            # filter out all intermediate supergroup activations as they should not be debugged
            debug_graph_activations = target_activations - supergroup_activations
            visited_debug_graph_inputs = debug_graph_input_names
            visited_debug_graph_outputs = debug_graph_output_names
        else:
            # user passed either of the debug subgraph inputs or outputs
            debug_graph_activations, visited_debug_graph_inputs, visited_debug_graph_outputs = (
                get_subgraph(
                    debug_graph_input_names,
                    debug_graph_output_names,
                    target_activation_op_map,
                    framework_activation_op_map,
                    supergroup_activations,
                )
            )

        # Topological sort the target activations
        target_topological_sort = get_topological_order(target_activation_op_map)
        debug_graph_topological_activations = [
            activation
            for activation in target_topological_sort
            if activation in debug_graph_activations
        ]

        if not debug_graph_activations:
            raise RuntimeError("Failed to generate debug graph activations")

        self._logger.debug(
            f"Debug Graph: {debug_graph_topological_activations} with Inputs: "
            f"{visited_debug_graph_inputs} and Outputs: {visited_debug_graph_outputs}"
        )

        return (
            debug_graph_topological_activations,
            visited_debug_graph_inputs,
            visited_debug_graph_outputs,
        )

    def _should_be_skipped(
        self, subgraph_activations: set, subgraph_output_name: str, target_activation_op_map: dict
    ) -> tuple[bool, str]:
        """Determines whether a subgraph should be skipped.

        Args:
            subgraph_activations (set): Set of subgraph activations
            subgraph_output_name (str): Last output of the subgraph
            target_activation_op_map (dict): Mapping of target activation names to op.

        Returns:
            tuple[bool, str]: Indicates whether the node should be skipped and the reason.
        """
        # Skip if subgraph_activations is an empty set
        if not subgraph_activations:
            return True, "Target subgraph is empty"

        target_op = target_activation_op_map[subgraph_output_name]
        target_op_type = target_op.op_type.lower()

        self._logger.debug(f"Activation: {subgraph_output_name},  Target op_type: {target_op_type}")

        # Skip if all ops in the target subgraph are MATH_INVARIANT ops
        subgraph_op_types = set()
        for activation_name in subgraph_activations:
            target_op = target_activation_op_map[activation_name]
            target_op_type = target_op.op_type.lower()
            subgraph_op_types.update([target_op_type])

        # Skip if all ops in the subgraph are classified as MATH_INVARIANT ops
        if subgraph_op_types.issubset(MATH_INVARIANT_OPS):
            return True, "MATH_INVARIANT"

        return False, ""

    def _build_data_frame_for_subgraph(
        self,
        framework_activation_name: str,
        resolved_target_activations: dict,
        target_activation_info: dict[str, ActivationInfo],
        verifier_scores: dict,
        layer_type: str,
        comparators: list[Comparator],
        framework_activation_info: dict[str, ActivationInfo],
        data_frame: DataFrame,
        output_tensor: list[str],
    ) -> dict:
        """Build the result dataframe for the given activation in the framework graph

        Args:
            framework_activation_name: activation to build the dataframe
            resolved_target_activations: mapping between framework and target activations
            target_activation_info: Activation information of target output.
            verifier_scores: Comparator scores from verifier.
            layer_type: Layer type of the activation
            comparators: list of comparators to be used in verification stage
            framework_activation_info: Framework activations information
            data_frame: dataframe to be dumped in csv report
            output_tensor: output tensors of the framework model

        Returns:
            dict: dataframe to be dumped in the csv report
        """
        comparator_names = [comparator.name for comparator in comparators]
        sanitized_framework_activation = Helper.transform_node_names(framework_activation_name)
        selected_fw_activation_info = framework_activation_info.get(
            sanitized_framework_activation, None
        )

        status = self._activation_status[framework_activation_name].get_status()
        info = self._activation_status[framework_activation_name].get_msg()

        data_frame.loc[framework_activation_name, "STATUS"] = status
        data_frame.loc[framework_activation_name, "INFO"] = info
        data_frame.loc[framework_activation_name, "Layer Type"] = layer_type
        if selected_fw_activation_info:
            data_frame.at[framework_activation_name, "Source Shape"] = (
                selected_fw_activation_info.shape
            )
            data_frame.at[framework_activation_name, "Source(Min, Max, Median)"] = (
                selected_fw_activation_info.distribution
            )
        else:
            data_frame.at[framework_activation_name, "Source Shape"] = []
            data_frame.at[framework_activation_name, "Source(Min, Max, Median)"] = []
        resolved_target_activation = resolved_target_activations[framework_activation_name]
        sanitized_target_activation = Helper.transform_node_names(resolved_target_activation)
        selected_target_activation_info = target_activation_info[sanitized_target_activation]
        data_frame.at[framework_activation_name, "Target Name"] = resolved_target_activation
        data_frame.at[framework_activation_name, "Target Shape"] = (
            selected_target_activation_info.shape
        )

        if status == ActivationStatus.SUCCESS:
            data_frame.at[framework_activation_name, "Target(Min, Max, Median)"] = (
                selected_target_activation_info.distribution
            )
            for comp in comparator_names:
                data_frame.loc[framework_activation_name, f"{comp}(current_layer)"] = (
                    verifier_scores[framework_activation_name]["self"][comp]
                )
                for original_output in output_tensor:
                    data_frame.loc[framework_activation_name, f"{comp}({original_output})"] = (
                        verifier_scores[framework_activation_name]["original_outputs"][
                            original_output
                        ][comp]
                    )

        else:
            data_frame.at[framework_activation_name, "Target(Min, Max, Median)"] = []

            for comp in comparator_names:
                data_frame.loc[framework_activation_name, f"{comp}(current_layer)"] = "NaN"
                for original_output in output_tensor:
                    data_frame.loc[framework_activation_name, f"{comp}({original_output})"] = "NaN"
        return data_frame

    def _execute_all_sub_graphs(
        self,
        output_dir: Path,
        all_subgraphs: dict,
        debug_graph_activations: list,
        resolved_target_activations: dict,
        converter_args: ConverterInputArguments,
        quantizer_args: QuantizerInputArguments,
        context_bin_args: GenerateConfig,
        context_bin_backend_extension: Path | dict,
        offline_prepare: bool,
        net_runner_args: NetRunnerInputArguments,
        net_run_backend_extension: Path | dict,
        input_sample: dict,
        backend: BackendType,
        platform: DevicePlatformType,
        remote_host_details: RemoteHostDetails,
        soc_model: str,
        comparators: list[Comparator],
        layout_info: dict,
        golden_reference_output: dict,
        data_frame: DataFrame,
        output_tensor: list[str],
        target_activation_info: dict[str, ActivationInfo],
        framework_activation_info: dict[str, ActivationInfo],
        verifier_scores: dict,
        csv_path: Path,
        retain_compilation_artifacts: bool = False,
        dump_output_tensors: bool = False,
        is_qnn_golden_reference: bool = False,
        user_max_parallel_compilations: int = None,
        compilation_memory: float = None,
    ) -> tuple[Path, pd.DataFrame]:
        """Executes all subgraph on QAIRT inference engine

        Args:
            output_dir: Output directory path
            all_subgraphs: Information of all subgraphs.
            debug_graph_activations: topologically sorted graph activations for debugging.
            resolved_target_activations: Output target activation.
            converter_args: Input arguments required by the converter module.
            quantizer_args: Input arguments required by the quantizer module.
            context_bin_args: Input arguments required by the context_bin_gen module.
            context_bin_backend_extension: Backend extension config file or dictionary for context
                                           binary generator.
            offline_prepare: Boolean to indicate offline prepare of graph.
            net_runner_args: Input arguments required by the netrunner module.
            net_run_backend_extension: Backend extension config file or dictionary for net-runner
            input_sample: Input to netrunner module
            backend: Type of backend
            platform: Platform of target device (android, wos, x86_64_linux, etc.)
            remote_host_details: Details of remote host
            soc_model: Name of SOC model on target device.
            comparators: List of comparators to use in verification stage.
            layout_info: Layout information used for verification
            golden_reference_output: Golden reference output data
            data_frame: Dataframe with information on subgraphs
            output_tensor: Output tensors of the framework model
            target_activation_info: Target Activation information.
            framework_activation_info: Framework activation info.
            verifier_scores: Comparator scores from verifier.
            csv_path: Path to CSV snooper report
            retain_compilation_artifacts: Flag to retain compilation artifacts. Default is set to
                                          False
            dump_output_tensors: Boolean to indicate whether to dump output tensors.
            max_parallel_compilations: Maximum number of processes that can be executed in parallel.
                                       Minimum of 1 core is enabled.
            is_qnn_golden_reference: Whether given golden outputs are from QNN.
            user_max_parallel_compilations: User provided max parallel compilations.
            compilation_memory: RAM memory used for a single subgraph compilation

        Returns:
            Path: Path to the given CSV snooper report
            pd.DataFrame: DataFrame object containing the snooping report
        """
        quantizer_args = quantizer_args.model_copy(
            update={"input_list": None, "float_fallback": True}
        )

        compile_template = {
            "model": self.model,
            "converter_args": converter_args,
            "quantizer_args": quantizer_args,
            "context_bin_args": context_bin_args,
            "context_bin_backend_extension": context_bin_backend_extension,
            "offline_prepare": offline_prepare,
            "backend": backend,
            "soc_model": soc_model,
            "platform": platform,
            "remote_host_details": remote_host_details,
        }
        net_run_template = {
            "input_sample": input_sample,
            "net_runner_args": net_runner_args,
            "net_run_backend_extension": net_run_backend_extension,
            "backend": backend,
            "platform": platform,
            "remote_host_details": remote_host_details,
            "soc_model": soc_model,
            "dump_output_tensors": dump_output_tensors,
        }
        subgraph_processes = {}
        ongoing_processes = {}

        # You do not require netrun locks for x86 execution, update the list if more such
        # platforms optimizations are valid.
        platforms = [DevicePlatformType.X86_64_LINUX]
        if platform in platforms:
            netrun_lock = None
        else:
            netrun_lock = Lock()

        # Compute max parallel compilations
        max_parallel_compilations = get_max_parallel_compilations(
            degree_of_freedom=3, compilation_memory=compilation_memory
        )
        if user_max_parallel_compilations:
            max_parallel_compilations = min(
                max_parallel_compilations, user_max_parallel_compilations
            )
        max_parallel_compilations = min(max_parallel_compilations, COMPILATIONS_LIMIT)

        self._logger.info(
            f"Proceeding with {max_parallel_compilations} parallel subgraph executions."
        )
        initialized_debug_activations = []
        self._logger.debug(f"debug_graph_activations: {debug_graph_activations[:10]}")

        # Create Process object for each subgraph and initialize initial set of ongoing processes
        for framework_activation_name in debug_graph_activations:
            if (
                self._activation_status[framework_activation_name].get_status()
                == ActivationStatus.INITIALIZED
            ):
                initialized_debug_activations.append(framework_activation_name)
                subgraph_override_file_path = all_subgraphs["subgraphs"][framework_activation_name][
                    "override_file_path"
                ]
                sanitized_framework_activation_name = Helper.transform_node_names(
                    framework_activation_name
                )
                working_dir = output_dir / sanitized_framework_activation_name
                working_dir.mkdir(exist_ok=True)

                resolved_target_activation = resolved_target_activations[framework_activation_name]
                output_tensors = [resolved_target_activation] + output_tensor

                converter_args_subgraph = converter_args.model_copy(
                    update={"quantization_overrides": subgraph_override_file_path}
                )

                context_bin_args_subgraph = (
                    context_bin_args.model_copy() if context_bin_args else None
                )
                net_runner_args_subgraph = net_runner_args.model_copy() if net_runner_args else None

                # Set output tensors for offline preparation
                if backend == BackendType.AIC:
                    set_output_tensors = output_tensors
                else:
                    set_output_tensors = list(map(Helper.transform_node_names, output_tensors))

                if (
                    offline_prepare is False
                    or backend not in BackendType.offline_preparable_backends()
                ):
                    if net_runner_args_subgraph:
                        net_runner_args_subgraph.debug = False
                        net_runner_args_subgraph.set_output_tensors = set_output_tensors
                    else:
                        net_runner_args_subgraph = NetRunnerInputArguments(
                            debug=False, set_output_tensors=set_output_tensors
                        )

                else:
                    if context_bin_args_subgraph:
                        context_bin_args_subgraph.enable_intermediate_outputs = False
                        context_bin_args_subgraph.set_output_tensors = set_output_tensors
                    else:
                        context_bin_args_subgraph = GenerateConfig(
                            enable_intermediate_outputs=False, set_output_tensors=set_output_tensors
                        )

                compile_config = {
                    **compile_template,
                    "working_directory": working_dir,
                    "converter_args": converter_args_subgraph,
                    "context_bin_args": context_bin_args_subgraph,
                }

                inference_engine_config = {
                    **compile_config,
                    **net_run_template,
                    "net_runner_args": net_runner_args_subgraph,
                    "netrun_lock": netrun_lock,
                    "log_level": logging.getLevelName(self._logger.level),
                    "logger_name": f"Inference Engine(Subgraph {sanitized_framework_activation_name})",
                }
                self._logger.info(f"INF CONFIG: {inference_engine_config}")
                process = InferenceEngineProcess(
                    target=Snooper._execute_inference,
                    kwargs=inference_engine_config,
                    inference_output_directory=working_dir,
                )
                subgraph_processes[framework_activation_name] = process

                # Initialize the initial set of processes
                if len(ongoing_processes) < max_parallel_compilations:
                    ongoing_processes[framework_activation_name] = process
        # Start the initial set of processes
        for framework_activation_name in ongoing_processes:
            self._logger.debug(f"Starting process for activation: {framework_activation_name}")
            ongoing_processes[framework_activation_name].start()

        self._logger.debug(f"Current Process Queue: {ongoing_processes.keys()}")

        index = len(ongoing_processes)

        self._logger.debug(f"initialized_debug_activations: {initialized_debug_activations[:10]}")
        self._logger.debug(f"Index: {index}")

        while ongoing_processes:
            new_processes = {}
            joined_processes_names = []
            for framework_activation_name, subgraph_process_obj in ongoing_processes.items():
                if not subgraph_process_obj.is_alive():
                    joined_processes_names.append(framework_activation_name)
                    # Join the current process as it is not alive
                    self._logger.debug(
                        f"Joining process for activation: {framework_activation_name}"
                    )
                    status, status_message, result = subgraph_process_obj.join()

                    if status != ActivationStatus.INFERENCE_DONE:
                        status_message = f"Exception occurred while executing subgraph \
                            {framework_activation_name}: {status_message}"
                    else:
                        try:
                            verifier_scores, target_activation_info = (
                                self._run_verification_for_subgraph(
                                    framework_activation_name,
                                    golden_reference_output,
                                    result.output_data[0],
                                    verifier_scores,
                                    resolved_target_activations,
                                    comparators,
                                    layout_info,
                                    target_activation_info,
                                    output_tensor,
                                    is_qnn_golden_reference=is_qnn_golden_reference,
                                )
                            )
                            status = ActivationStatus.SUCCESS
                        except VerificationError as err:
                            status_message = f"Verification failed while verifying subgraph \
                                {framework_activation_name}. Reason: {err}"
                            status = ActivationStatus.VERIFICATION_FAILURE

                    # Unified status update
                    all_subgraphs["subgraphs"][framework_activation_name]["status"] = status
                    all_subgraphs["subgraphs"][framework_activation_name]["status_msg"] = (
                        status_message
                    )
                    self._activation_status[framework_activation_name].set_status(
                        status, status_message
                    )

                    # Cleanup artifacts
                    if not retain_compilation_artifacts and result:
                        self._logger.info(
                            f"Cleaning up artifacts for subgraph {framework_activation_name}:\n"
                            f"Converter DLC: {result.converter_dlc}\n"
                            f"Quantizer DLC: {result.quantizer_dlc}\n"
                            f"Offline Graph: {result.offline_graph}"
                        )
                        result.cleanup_artifacts()
                    # TODO: clean up the raw files by default

                    data_frame = self._build_data_frame_for_subgraph(
                        framework_activation_name,
                        resolved_target_activations,
                        target_activation_info,
                        verifier_scores,
                        all_subgraphs["subgraphs"][framework_activation_name]["layer_type"],
                        comparators,
                        framework_activation_info,
                        data_frame,
                        output_tensor,
                    )

                    # Dump snooper data into CSV
                    dump_csv(data_frame, csv_path, index=True)
                    self._logger.info(
                        f"STATUS for activation {framework_activation_name}: \
                            {self._activation_status[framework_activation_name].get_status()}"
                    )

                    # Initiate a new process
                    if index < len(initialized_debug_activations):
                        self._logger.debug(f"Index: {index}")
                        new_subgraph_name = initialized_debug_activations[index]
                        index += 1

                        self._logger.debug(
                            f"initialized_debug_activations: {initialized_debug_activations[:10]}"
                        )
                        self._logger.debug(f"New Index: {index}")
                        new_processes[new_subgraph_name] = subgraph_processes[new_subgraph_name]
                        new_processes[new_subgraph_name].start()
                        self._logger.debug(f"Starting process for activation: {new_subgraph_name}")

            # Delete the joined processes from the ongoing processes window
            if joined_processes_names:
                for joined_process_name in joined_processes_names:
                    del ongoing_processes[joined_process_name]

            # Add the new processes to the ongoing processes window
            if new_processes:
                for new_process_name, process in new_processes.items():
                    ongoing_processes[new_process_name] = process

                self._logger.debug(f"Current Process Queue: {ongoing_processes.keys()}")

        return csv_path, data_frame

    def _initialize_data_frame(self, comparator_names: list, output_tensor: list) -> DataFrame:
        """Initialize result DataFrame for storing activation comparison data.

        Args:
            comparator_names (list): List of comparator names.
            output_tensor (list[str]): List of output tensor names.

        Returns:
            DataFrame: Initialized DataFrame with appropriate columns.
        """
        columns = [
            "Source Name",
            "Target Name",
            "STATUS",
            "Layer Type",
            "Source Shape",
            "Target Shape",
            "Source(Min, Max, Median)",
            "Target(Min, Max, Median)",
        ]

        for comp in comparator_names:
            columns.append(f"{comp}(current_layer)")
            for original_output in output_tensor:
                columns.append(f"{comp}({original_output})")
        columns.append("INFO")

        # Create an empty DataFrame with the specified columns
        data_frame = DataFrame(columns=columns)

        return data_frame

    def _run_verification_for_subgraph(
        self,
        framework_activation_name: str,
        golden_reference_output: dict,
        target_output: dict,
        verifier_scores: dict,
        resolved_target_activations: dict,
        comparators: list[Comparator],
        layout_info: dict,
        target_activation_info: dict[str, ActivationInfo],
        output_tensor: list[str],
        is_qnn_golden_reference: bool = False,
    ) -> tuple[dict, dict]:
        """Calculates the verifier score for the given subgraph's final output and model's
        final output between target and framework tensors.

        Args:
            framework_activation_name (str): Name of the activation as per framework model
            golden_reference_output (dict): Dictionary of output names to tensor value
            target_output (dict): Final output of the subgraph.
            verifier_scores (dict): Verifier scores based on comparator.
            resolved_target_activations (dict): Output target activation.
            comparators (list[Comparator]): List of comparators for verification.
            layout_info (dict): Dictionary of layout data for each tensor.
            target_activation_info (dict[str, ActivationInfo]): Target activation information.
            output_tensor (list[str]): List of output tensor names
            is_qnn_golden_reference: Whether given golden outputs are from QNN.

        Returns:
            tuple[dict, dict]: verifier_scores, target_activation_info
        """
        resolved_target_activation = resolved_target_activations[framework_activation_name]
        sanitized_target_activation = Helper.transform_node_names(resolved_target_activation)

        # First compute verification for the intermediate nodes
        if framework_activation_name in golden_reference_output:
            reference_output_name = framework_activation_name
        elif Helper.transform_node_names(framework_activation_name) in golden_reference_output:
            reference_output_name = Helper.transform_node_names(framework_activation_name)
        else:
            raise VerificationError(
                f"Golden reference not available for target activation: {sanitized_target_activation}."
            )

        if sanitized_target_activation not in target_output:
            raise VerificationError(
                f"Target output not available for framework activation: {framework_activation_name}."
            )

        reference_activation_output = {
            reference_output_name: golden_reference_output[reference_output_name]
        }
        target_activation_output = {
            sanitized_target_activation: target_output[sanitized_target_activation]
        }
        intermediate_node_verifier_scores, target_activation_info = (
            self._compute_verification_score(
                reference_activation_output,
                target_activation_output,
                layout_info,
                target_activation_info,
                comparators,
                is_qnn_golden_reference=is_qnn_golden_reference,
            )
        )

        verifier_scores[framework_activation_name] = {"self": intermediate_node_verifier_scores}

        # Now compute for its original model outputs
        original_outputs_verifier_scores = {}
        for original_output in output_tensor:
            resolved_target_original_output = resolved_target_activations[original_output]
            santized_original_target_output = Helper.transform_node_names(
                resolved_target_original_output
            )

            if original_output in golden_reference_output:
                reference_output_name = original_output
            elif Helper.transform_node_names(original_output) in golden_reference_output:
                reference_output_name = Helper.transform_node_names(original_output)
            else:
                raise VerificationError(
                    f"Golden reference not available for target_activation: {sanitized_target_activation}."
                )

            reference_activation_output = {
                reference_output_name: golden_reference_output[reference_output_name]
            }
            target_activation_output = {
                santized_original_target_output: target_output[santized_original_target_output]
            }

            original_output_verifier_score, target_activation_info = (
                self._compute_verification_score(
                    reference_activation_output,
                    target_activation_output,
                    layout_info,
                    target_activation_info,
                    comparators,
                    is_qnn_golden_reference=is_qnn_golden_reference,
                )
            )
            original_outputs_verifier_scores[original_output] = original_output_verifier_score

        verifier_scores[framework_activation_name]["original_outputs"] = (
            original_outputs_verifier_scores
        )

        return verifier_scores, target_activation_info

    def _compute_verification_score(
        self,
        reference_output: dict,
        target_output: dict,
        layout_info: dict,
        target_activation_info: dict[str, ActivationInfo],
        comparators: list[Comparator],
        is_qnn_golden_reference: bool = False,
    ) -> tuple[dict, dict]:
        """Computes the verifier score between two given tensor outputs.

        Args:
            reference_output (dict): Reference output tensors from framework.
            target_output (dict): Target output
            layout_info (dict): dictionary of layout information.
            target_activation_info (dict[str, ActivationInfo]): Target activation information.
            comparators (list[Comparator]): List of selected comparators
            is_qnn_golden_reference: Whether given golden outputs are from QNN.

        Raises:
            VerificationError: If verification fails.

        Returns:
            tuple[dict, dict]: Verification scores and Target activation info
        """
        reference_output_name = list(reference_output.keys())[0]
        target_output_name = list(target_output.keys())[0]

        target_min = np.min(target_output[target_output_name])
        target_max = np.max(target_output[target_output_name])
        target_median = np.median(target_output[target_output_name])

        target_activation_info[target_output_name].distribution = (
            target_min,
            target_max,
            target_median,
        )

        verifier_scores = {}
        if target_output is not None and reference_output is not None:
            verifier = Verifier(comparators, logger=self._logger)
            tensor_mapping_dict = {target_output_name: reference_output_name}
            graph_info = {
                "layout_info": layout_info,
                "tensor_mapping": tensor_mapping_dict,
            }

            try:
                verifier_score_dict = verifier.verify_dictionary_of_tensors(
                    reference_output,
                    target_output,
                    graph_info=graph_info,
                    disable_layout_transform=is_qnn_golden_reference,
                )
            except Exception as exception:
                raise VerificationError(
                    f"Verification failed for reference activation: {reference_output_name} "
                    f"and target activation: {target_output_name}. Reason: {exception}"
                ) from exception

            verifier_score_dict = list(verifier_score_dict.values())[0]

            for comparator in comparators:
                verifier_score = verifier_score_dict[comparator.name]
                verifier_scores[comparator.name] = verifier_score

        return verifier_scores, target_activation_info

    def _create_subgraph_quantization_override(
        self,
        subgraph: set,
        subgraph_output_names: list,
        supergroup_activations: set,
        qairt_encodings_converter: QairtEncodingsConverter,
        output_dir: Path,
        encoding_version: EncodingVersion,
        compulsory_overrides: Optional[Path] = None,
    ) -> Path:
        """Creates quantization overrides file for the given subgraph intermediate tensor names.

        Args:
            subgraph (set): Subgraph under execution
            subgraph_output_names (list): Outputs of the subgraph under execution.
            supergroup_activations (set): Activations that are a supergroup.
            qairt_encodings_converter (QairtEncodingsConverter): QAIRT encodings converter object.
            output_dir (Path): Path to output directory.
            encoding_version: Version in which subgraph encoding override needs to be dumped.
            compulsory_overrides (Path): Path to compulsory overrides file.

        Returns:
            Path: Path to quantization override for given subgraph
        """
        subgraph_encoding = qairt_encodings_converter.create_subgraph_encodings(
            subgraph, supergroup_activations
        )
        if compulsory_overrides:
            compulsory_encoding = ModelEncoding()
            compulsory_encoding.load(artifact=compulsory_overrides, load_json=True)
            subgraph_encoding.add(
                subgraph_tensor_encodings=compulsory_encoding.tensor_encodings,
                overwrite_existing=True,
            )

        subgraph_output_names = list(map(Helper.transform_node_names, subgraph_output_names))
        subgraph_override_dir = output_dir / "sub_graph_node_precision_files"
        subgraph_override_dir.mkdir(parents=True, exist_ok=True)
        file_name = "#".join(sorted(subgraph_output_names)) + ".json"
        subgraph_override_file_path = subgraph_override_dir / file_name
        subgraph_encoding.dump(version=encoding_version, file_path=subgraph_override_file_path)

        return subgraph_override_file_path

    def _get_output_tensor_names(self) -> list[str]:
        """Get output tensor names from model

        Returns:
            list: A list of output tensor names
        """
        framework_instance = get_framework_instance(
            framework=self.framework_type, logger=self._logger
        )
        model_proto = framework_instance.load_model(self.model)
        return framework_instance.get_output_tensor_names(model_proto)

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
        # Convert DataFrame to json format
        # reset_index() to ensure row indices (source name) is part of the data
        json_layers = json.loads(data_frame.reset_index().to_json(orient="records"))

        # Path of json snooping report based on the snooping algorithm used
        json_path = output_dir / f"{self._name}.json"

        new_layers = []
        for layer in json_layers:
            transformed_layer = {}
            comparators = {}

            for key, value in layer.items():
                if key in CSV_TO_JSON_FIELDS_MAP.keys():
                    transformed_layer[CSV_TO_JSON_FIELDS_MAP[key]] = value
                else:
                    # Handle dynamic comparator columns (format: "ComparatorName(LayerName)")
                    # Example: "mse(conv1)" -> comparator="mse", layer_name="conv1"
                    match = re.match(r"([\w]+)\(([^)]+)\)", key)
                    comparator = match.group(1)
                    layer_name = match.group(2)

                    # Group comparator results by comparator type
                    # This creates nested structure: {"COSINE": {"conv1": value, "conv2": value}}
                    if comparator in comparators.keys():
                        comparators[comparator][layer_name] = "NaN" if value is None else value

                    else:
                        comparators[comparator] = {layer_name: "NaN" if value is None else value}

            transformed_layer["comparators"] = comparators
            new_layers.append(transformed_layer)

        # Create the final JSON structure with header metadata and layer data
        json_data = {
            "header": {
                "header_version": {"major": 0, "minor": 1, "patch": 0},
                "artifact_type": "LAYERWISE_SNOOPING_REPORT",
                "algorithm": f"{self._name}",
                "version": {"major": 0, "minor": 1, "patch": 0},
            },
            "layers": new_layers,
        }

        # Write the JSON data to file
        dump_json(json_data, json_path)
        return json_path
