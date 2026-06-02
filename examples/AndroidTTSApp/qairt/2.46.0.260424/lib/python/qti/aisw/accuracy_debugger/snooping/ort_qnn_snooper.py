# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""ORT QNN Snooper Module

This module provides the core snooping functionality for debugging and analyzing
ONNX models running on the ONNXRuntime with QNN Execution provider.
It compares model outputs between reference (FP32/QDQ) and target (QDQ) models
to identify accuracy issues.

The snooper supports multiple debugging algorithms:
- ONESHOT: Analyzes the entire model in a single pass
- LAYERWISE: Analyzes the model layer by layer

Key Features:
- Tensor comparison using multiple comparators (MSE, SQNR, etc.)
- Support for both QDQ and FP32 as reference models
- Intermediate layer output analysis
- Elementwise comparison logs
- Comprehensive CSV and JSON reporting
- Visualization of comparison metrics
"""

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
import pandas as pd
from pydantic import DirectoryPath, Field, FilePath, model_validator
from qti.aisw.accuracy_debugger.argparser.ort_qnn_snooper_parser import OrtQnnSnooperParser
from qti.aisw.accuracy_debugger.framework_runner.frameworks.onnx_framework import (
    CustomOnnxFramework,
)
from qti.aisw.accuracy_debugger.inference_engine.ort_qnn_inference_engine import (
    OrtQnnInferenceEngine,
    OrtQnnInferenceEngineInputConfig,
)
from qti.aisw.accuracy_debugger.lib.framework_runner.nd_framework_runner import FrameworkRunner
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.accuracy_debugger.utils.exceptions import VerificationError
from qti.aisw.accuracy_debugger.utils.file_utils import dump_csv
from qti.aisw.accuracy_debugger.utils.helper import (
    ActivationInfo,
    ActivationStatus,
    DebuggerConfig,
    InputSample,
    Namespace,
    create_working_directory,
    create_working_directory_with_timestamp,
    dump_json_report,
    generate_reference_data_with_framework,
    get_logger,
    load_data_from_directory,
    load_input_tensors,
    plot_graphs,
    verify,
)
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.comparators.mse import MSEComparator
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper


class OrtQnnSnooperInputConfig(DebuggerConfig):
    """Input configuration class for ORT/QNN-based Accuracy Debugger (Snooper).

    Overview:
    - Debug ONNX models executed with ONNXRuntime's QNN Execution Provider (QNN-EP)
    - Compare a reference model (FP32 or QDQ) against a quantized QDQ model to localize accuracy issues
    - Support ONESHOT and LAYERWISE workflows with optional intermediate tensor collection and metrics

    Important constraints:
    - reference_model is supported only with Algorithm.ONESHOT.
    - qdq_model must be an ONNX model that contains QDQ (QuantizeLinear/DequantizeLinear) ops.
    - set_cpu_layers is valid only with Algorithm.LAYERWISE.
    - dump_elementwise_stats may be enabled only when reference_model is a quantized (QDQ) model.

    Attributes:
        qdq_model (FilePath): Path to the quantized-dequantized (QDQ) ONNX model to be debugged.
            Must be a valid ONNX model that contains QDQ operations.
        reference_model (Optional[FilePath]): Path to the reference ONNX model used as ground truth.
            May be FP32 or QDQ. Supported only when algorithm == Algorithm.ONESHOT.
        input_sample (list[InputSample] | dict[str, NDArray]): Input tensors for inference;
            either a list of InputSample objects or a dict of name->numpy array.
        algorithm (Algorithm): Snooping algorithm to use (default: Algorithm.ONESHOT).
        comparators (List[Comparator]): Metrics to compare reference vs. target outputs
            (default: [MSEComparator()]).
        working_directory (DirectoryPath): Directory used to persist artifacts and results.
            Created automatically when not provided.
        golden_reference_path (Optional[DirectoryPath]): Directory containing pre-computed
            reference outputs. When provided, running the reference model can be skipped.
        set_intermediate_layers (Optional[List[str]]): Layer names/types to extract intermediate
            tensors from; when None, defaults to dumping all outputs.
        set_cpu_layers (Optional[List[str]]): Layer names/types to execute on CPU EP instead
            of QNN-HTP-EP for isolation. Supported only with Algorithm.LAYERWISE.
        dump_elementwise_stats (Optional[Dict[str, int]]): Per-layer configuration to dump
            element-wise statistics; allowed only if reference_model is a QDQ model.
        dump_output_tensors (bool): Dump final inference outputs to binary files (default: False).

    Examples:
        # Example 1: ONESHOT with separate reference model (reference_model allowed)
        config = OrtQnnSnooperInputConfig(
            reference_model="float_model.onnx",
            qdq_model="quantized_model.onnx",
            input_sample={"input": np.random.randn(1, 3, 224, 224)},
            algorithm=Algorithm.ONESHOT,
            dump_output_tensors=True,
        )

        # Example 2: LAYERWISE without reference_model, using CPU fallbacks for isolation
        config = OrtQnnSnooperInputConfig(
            qdq_model="quantized_model.onnx",
            input_sample={"input": np.random.randn(1, 3, 224, 224)},
            algorithm=Algorithm.LAYERWISE,
            set_intermediate_layers=["Conv_0", "Relu_1"],
            set_cpu_layers=["Conv_0"],
            dump_elementwise_stats={"Conv_0": 3, "Relu_1": 1},
        )

    Notes:
    - Both reference_model and qdq_model must be valid ONNX files.
    - The qdq_model must contain QDQ (QuantizeLinear/DequantizeLinear) nodes
    - set_cpu_layers requires algorithm == Algorithm.LAYERWISE.
    - When golden_reference_path is provided, reference_model execution can be skipped.
    """

    qdq_model: FilePath
    reference_model: Optional[FilePath]
    input_sample: List[InputSample] | Dict[str, np.ndarray]
    algorithm: Algorithm = Algorithm.ONESHOT
    comparators: List[Comparator] = [MSEComparator()]
    working_directory: DirectoryPath = Field(default_factory=create_working_directory)
    golden_reference_path: Optional[DirectoryPath] = None
    set_intermediate_layers: Optional[List[str]] = None
    set_cpu_layers: Optional[List[str]] = None
    dump_elementwise_stats: Optional[Dict[str, int]] = None
    dump_output_tensors: bool = False

    @model_validator(mode="after")
    def validate_reference_model(self):
        """Validate reference_model constraints and integrity.

        - Allowed only when algorithm == Algorithm.ONESHOT (layer-wise/cumulative do not use a
          separate reference model).
        - When present, validates model integrity via CustomOnnxFramework.validate_model.

        Raises:
            ValueError: If algorithm is not ONESHOT while reference_model is provided.
            Exception: Propagated from CustomOnnxFramework.validate_model on invalid ONNX.
        """
        if self.reference_model:
            # Only ONESHOT supports executing a separate reference model; layer-wise mode
            # operate on the QDQ target only and therefore reject reference_model.
            if self.algorithm == Algorithm.ONESHOT:
                # Validate basic ONNX integrity of the reference model.
                CustomOnnxFramework.validate_model(self.reference_model)
            else:
                raise ValueError(
                    f"reference_model is supported only with {Algorithm.ONESHOT.name.lower()} algorithm."
                )
        return self

    @model_validator(mode="after")
    def validate_qdq_model(self):
        """Validate qdq_model integrity and presence of QDQ nodes.

        Uses CustomOnnxFramework.validate_model(..., validate_qdq=True) to ensure:
        - The file is a valid ONNX model
        - The graph contains QDQ (QuantizeLinear/DequantizeLinear) nodes

        Raises:
            Exception: Propagated from the underlying validator on invalid ONNX/QDQ.
        """
        # Validate ONNX integrity and enforce presence of QDQ (QuantizeLinear/DequantizeLinear) nodes.
        CustomOnnxFramework.validate_model(self.qdq_model, validate_qdq=True)
        return self

    @model_validator(mode="after")
    def validate_set_cpu_layers(self):
        """Validate the set_cpu_layers configuration against the selected algorithm.

        The LAYERWISE algorithm processes the model layer-by-layer, making it
        possible to switch execution providers per layer. Other algorithms like
        ONESHOT process the entire model at once and don't support per-layer
        execution provider selection.

        Raises:
            ValueError: If set_cpu_layers is specified but the algorithm is not
                Algorithm.LAYERWISE, as this feature requires layer-by-layer
                processing capability.
        """
        # Validate that set_cpu_layers is only used with LAYERWISE algorithm
        if self.set_cpu_layers and self.algorithm != Algorithm.LAYERWISE:
            raise ValueError(
                f"set_cpu_layers option is supported only with {Algorithm.LAYERWISE.name.lower()} algorithm."
            )

        return self

    @model_validator(mode="after")
    def validate_dump_elementwise_stats(self):
        """Validate dump_elementwise_stats usage.

        This option is allowed only when reference_model is a quantized (QDQ) model,
        because element-wise stats are defined between quantized tensors.

        Raises:
            ValueError: When dump_elementwise_stats is used along with a non-qdq reference model
        """
        if self.reference_model:
            # Ensure elementwise stats are requested only when reference is QDQ.
            reference_model_is_qdq = CustomOnnxFramework.is_qdq_model(self.reference_model)
            if self.dump_elementwise_stats and not reference_model_is_qdq:
                raise ValueError(
                    "dump_elementwise_stats option is supported only when reference_model is a quantized model."
                )

        return self


class OrtQnnSnooperOutputConfig(AISWBaseModel):
    """Defines OrtQnnSnooper output format

    Attributes:
        snooping_report: Report generated by the snooping algorithm executed.
    """

    csv_snooping_report: FilePath
    json_snooping_report: FilePath


class OrtQnnSnooper:
    """ORT/QNN Execution Provider Snooper for Model Debugging.

    This class provides comprehensive debugging capabilities for ONNX models running
    on the ONNXRuntime with QNN Execution provider. It compares outputs between a reference
    model (FP32 or QDQ) and a target quantized model to identify accuracy degradation
    and analyze tensor distributions.

    The snooper supports multiple debugging workflows:
    - Full model analysis (ONESHOT algorithm)
    - Layer-by-layer analysis (LAYERWISE algorithm - future implementation)
    - Custom intermediate layer selection
    - CPU fallback layer configuration

    Attributes:
        _logger (logging.Logger): Logger instance for debugging and status messages
        name (str): Algorithm name (ONESHOT or LAYERWISE)

    Inherits:
        Snooper: Base snooper class providing common functionality
        ABC: Abstract base class for enforcing interface contracts

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> snooper = OrtQnnSnooper(logger, name=Algorithm.ONESHOT)
        >>> output = snooper.run(config)
        >>> print(output.csv_snooping_report, output.json_snooping_report)
    """

    def __init__(self, logger: logging.Logger, name: str = Algorithm.ONESHOT):
        """Initialize the OrtQnnSnooper instance.

        Sets up the snooper with the specified algorithm and logger for tracking
        the debugging process.

        Args:
            logger (logging.Logger): Python logger instance for outputting debug
                information, warnings, and errors during the snooping process.
            name (str, optional): The debugging algorithm to use. Defaults to
                Algorithm.ONESHOT. Valid options are:
                - Algorithm.ONESHOT: Analyze entire model in one pass
                - Algorithm.LAYERWISE: Analyze model layer by layer
        Raises:
            None: Initialization errors are deferred to the run() method
        """
        self._logger = logger
        self._name = name

    def run(self, config: OrtQnnSnooperInputConfig) -> OrtQnnSnooperOutputConfig:
        """Execute the ORT/QNN-EP snooping process.

        This is the main entry point for the snooping workflow. It orchestrates the
        entire debugging process including:
        1. Loading and validating input data
        2. Generating reference outputs from the reference model
        3. Generating target outputs from the QDQ model on ORT/QNN-EP
        4. Comparing outputs using specified comparators
        5. Generating comprehensive reports (CSV and JSON)
        6. Creating visualization plots

        The method supports both ONESHOT (full model) and LAYERWISE (incremental)
        debugging algorithms.

        Args:
            config (OrtQnnSnooperInputConfig): Configuration object containing:
                - reference_model (Path): Path to reference ONNX model (FP32 or QDQ)
                - qdq_model (Path): Path to quantized QDQ ONNX model
                - input_sample: Input tensor data for inference
                - algorithm (Algorithm): Debugging algorithm (ONESHOT/LAYERWISE)
                - comparators (list[Comparator]): Metrics for tensor comparison
                - working_directory (Path): Directory for intermediate files
                - golden_reference_path (Path): Pre-computed reference outputs
                - dump_output_tensors (bool): Whether to save tensor outputs
                - set_intermediate_layers (list): Specific layers to analyze
                - set_cpu_layers (list): Layers to run on CPU instead of QNN
                - dump_elementwise_stats (dict[str, int]): Layers to dump elementwise comparison stats

        Returns:
            OrtQnnSnooperOutputConfig: Output configuration with paths to:
                - CSV snooping report with detailed metrics
                - JSON snooping report for programmatic access

        Raises:
            Exception: If algorithm is not ONESHOT or LAYERWISE
            VerificationError: If tensor verification fails
            FileNotFoundError: If model files or input samples don't exist
            RuntimeError: If inference engines fail to execute

        Example:
            >>> config = OrtQnnSnooperInputConfig(
            ...     reference_model=Path("model_fp32.onnx"), # or "quantized_model.onnx"
            ...     qdq_model=Path("model_qdq.onnx"),
            ...     input_sample={"input": np.random.randn(1, 3, 224, 224)},
            ...     algorithm=Algorithm.ONESHOT,
            ...     set_intermediate_layers=["conv1", "conv2"],
            ...     set_cpu_layers=["conv1"],
            ...     dump_elementwise_stats={"Conv_272": "3", "Add": "1", "LeakyRelu_300": "5"},
            ...     dump_output_tensors=True
            ... )
            >>> output = snooper.run(config)
        """
        # Extract configuration parameters from config object
        qdq_model = config.qdq_model
        input_tensors = config.input_sample
        algorithm = config.algorithm
        comparators = config.comparators
        working_directory = config.working_directory
        golden_reference_path = config.golden_reference_path
        dump_output_tensors = config.dump_output_tensors
        set_intermediate_layers = config.set_intermediate_layers
        set_cpu_layers = config.set_cpu_layers
        dump_elementwise_stats = config.dump_elementwise_stats
        # If user didn't pass reference_model then use qdq_model for reference
        reference_model = config.reference_model if config.reference_model else qdq_model

        # Convert None to empty lists for easier handling downstream
        intermediates_layers_list = (
            [] if set_intermediate_layers is None else set_intermediate_layers
        )
        cpu_layers_list = [] if set_cpu_layers is None else set_cpu_layers

        self._logger.info(f"Started {algorithm} snooping")

        # Create working directory with timestamp if not provided
        if not working_directory:
            working_directory = create_working_directory_with_timestamp(
                sub_directory=algorithm + "_snooping",
            )

        # Load input tensors from file or use provided tensors
        input_sample = load_input_tensors(input_tensors)

        # Determine if reference model is QDQ format (affects tensor mapping strategy)
        reference_model_is_qdq = CustomOnnxFramework.is_qdq_model(reference_model)

        reference_intermediates_list = None
        if algorithm == Algorithm.ONESHOT:
            # Get valid intermediate layer outputs based on user specification and model type
            reference_intermediates_list = self._get_valid_intermediates(
                intermediates_layers_list, reference_model, is_qdq_model=reference_model_is_qdq
            )

        # If golden reference path is provided, load data from there
        if golden_reference_path:
            self._logger.info(f"Loading reference data from given {golden_reference_path}")
            # Load raw files into memory using ONNX framework
            custom_onnx_obj = CustomOnnxFramework(self._logger)
            intermediate_outputs_info = custom_onnx_obj.get_intermediate_outputs_info(
                reference_model
            )
            reference_outputs = load_data_from_directory(
                golden_reference_path, intermediate_outputs_info
            )
        else:
            self._logger.info(f"Generating reference data for {reference_model}")
            # Generate reference outputs using framework manager
            reference_outputs = generate_reference_data_with_framework(
                self._logger,
                reference_model,
                input_sample,
                dump_output_tensors,
                working_directory,
                intermediate_output_tensors=reference_intermediates_list,
            )

        # Execute the appropriate debugging algorithm
        if algorithm == Algorithm.ONESHOT:
            # ONESHOT: Analyze entire model in one pass
            csv_path, json_report_path = self._execute_oneshot(
                qdq_model=qdq_model,
                input_sample=input_sample,
                reference_outputs=reference_outputs,
                dump_output_tensors=dump_output_tensors,
                comparators=comparators,
                working_directory=working_directory,
                intermediates_layers_list=intermediates_layers_list,
                reference_model_is_qdq=reference_model_is_qdq,
                dump_elementwise_stats=dump_elementwise_stats,
            )
        elif algorithm == Algorithm.LAYERWISE:
            # LAYERWISE: Analyze model layer by layer
            csv_path, json_report_path = self._execute_layerwise(
                reference_model=reference_model,
                qdq_model=qdq_model,
                input_sample=input_sample,
                reference_outputs=reference_outputs,
                dump_output_tensors=dump_output_tensors,
                comparators=comparators,
                working_directory=working_directory,
                intermediates_layers_list=intermediates_layers_list,
                cpu_layers_list=cpu_layers_list,
                reference_model_is_qdq=reference_model_is_qdq,
                dump_elementwise_stats=dump_elementwise_stats,
            )
        else:
            # Invalid algorithm specified
            raise ValueError(
                f"Invalid algorithm '{algorithm}' supplied. "
                f"Supported options are {Algorithm.ONESHOT.name.lower()} and {Algorithm.LAYERWISE.name.lower()}"
            )

        output_config = OrtQnnSnooperOutputConfig(
            csv_snooping_report=csv_path, json_snooping_report=json_report_path
        )
        return output_config

    def _execute_oneshot(
        self,
        qdq_model: Path,
        input_sample: dict,
        reference_outputs: dict[str, np.ndarray],
        dump_output_tensors: bool,
        comparators: list[Comparator],
        working_directory: Path,
        intermediates_layers_list: list,
        reference_model_is_qdq: bool,
        dump_elementwise_stats: dict[str, int],
    ) -> tuple[Path, Path]:
        """Execute ONESHOT debugging algorithm.

        This method implements the ONESHOT algorithm which analyzes the entire model
        in a single pass. It performs the following steps:
        1. Runs inference on the QDQ model using QNN backend
        2. Maps target tensors to reference tensors
        3. Compares outputs using specified comparators
        4. Generates activation distribution statistics
        5. Creates comprehensive CSV and JSON reports
        6. Generates visualization plots

        The ONESHOT approach is faster than LAYERWISE but provides less granular
        debugging information.

        Args:
            qdq_model (Path): Path to the quantized QDQ ONNX model to be analyzed
            input_sample (dict): Dictionary mapping input names to numpy arrays
                containing the input data for inference
            reference_outputs (dict[str, np.ndarray]): Dictionary of reference model
                outputs mapping tensor names to numpy arrays (ground truth)
            dump_output_tensors (bool): If True, saves all intermediate tensor
                outputs to disk for detailed analysis
            comparators (list[Comparator]): List of comparator instances to use
                for tensor comparison (e.g., MSE, SQNR, CosineSimilarity)
            working_directory (Path): Directory path for storing intermediate
                files and final reports
            intermediates_layers_list (list): List of layer names or types to
                specifically analyze. Empty list means analyze all layers
            reference_model_is_qdq (bool): Flag indicating if reference model is
                in QDQ format (affects tensor name mapping strategy)
            dump_elementwise_stats (dict[str, int]): Layers to dump elementwise comparison stats

        Returns:
            tuple[Path, Path]: A tuple containing:
                - csv_snooping_report_path: Path to CSV report with detailed metrics
                - json_snooping_report_path: Path to JSON report for automation

        Raises:
            VerificationError: If tensor verification fails or outputs don't match
            RuntimeError: If QNN inference engine fails to execute
            KeyError: If expected tensors are missing from outputs
        """
        # Get valid intermediate outputs for the QDQ model
        # Determine which intermediate outputs to capture from the QDQ model.
        # When the reference is also QDQ, we capture both QuantizeLinear and DequantizeLinear outputs.
        # When the reference is FP32, we only capture DequantizeLinear outputs to compare post-dequantized values.
        qdq_ops_idx = {"QuantizeLinear": 1, "DequantizeLinear": 2}
        if not reference_model_is_qdq:
            qdq_ops_idx = {"DequantizeLinear": 2}
        outputs_list = self._get_valid_intermediates(
            intermediates_layers_list,
            qdq_model,
            is_qdq_model=True,
            qdq_ops_index=qdq_ops_idx,
        )

        # Configure QNN inference engine with model and input parameters
        inference_config = OrtQnnInferenceEngineInputConfig(
            qdq_model=qdq_model,
            input_sample=input_sample,
            dump_output_tensors=dump_output_tensors,
            working_directory=working_directory,
            outputs_list=outputs_list,
        )

        # Execute inference on ORT/QNN-EP backend
        ort_qnn = OrtQnnInferenceEngine(self._logger)
        ort_qnn_output = ort_qnn.run(inference_config)
        target_outputs = ort_qnn_output.output_data

        # Create mapping between target and reference tensor names
        # This handles naming differences between FP32 and quantized models
        reference_outputs_names = list(reference_outputs.keys())
        target_outputs_names = list(target_outputs.keys())
        tensor_mapping = self._get_onnx_tensor_mapping(
            reference_outputs_names, target_outputs_names, reference_model_is_qdq
        )

        # Compare reference outputs with target outputs using specified comparators
        verifier_output = verify(
            reference_outputs,
            target_outputs,
            self._logger,
            comparators=comparators,
            graph_info={"tensor_mapping": tensor_mapping},
        )

        # Raise error if verification failed
        if not verifier_output:
            raise VerificationError(
                "Verification of tensors failed. Please check the logs for more details."
            )

        # Dump elementwise stats if user specified
        if dump_elementwise_stats:
            for layer, steps_allowed in dump_elementwise_stats.items():
                elementwise_outputs_list = self._get_valid_intermediates(
                    [layer], qdq_model, is_qdq_model=True
                )
                target_outputs_subset = {}
                reference_outputs_subset = {}
                for output_name in elementwise_outputs_list:
                    if output_name in target_outputs and output_name in reference_outputs:
                        target_outputs_subset[output_name] = target_outputs[output_name]
                        reference_outputs_subset[output_name] = reference_outputs[output_name]

                self._dump_elementwise_stats(
                    reference_outputs_subset,
                    target_outputs_subset,
                    steps_allowed,
                    working_directory,
                )

        # Get activation statistics distribution for reference and target outputs
        # Distribution includes (min, max, median) for each tensor along with dtype and shapes
        reference_activation_info = self._get_activations_info(reference_outputs)
        target_activation_info = self._get_activations_info(target_outputs)

        # Generate comprehensive snooping reports in CSV and JSON formats
        csv_snooping_report_path, json_snooping_report_path = self._generate_snooping_report(
            verifier_output,
            comparators,
            working_directory,
            reference_activation_info,
            target_activation_info,
            qdq_model,
        )

        # Generate visualization plots for comparison metrics
        # Creates line plots for each comparator
        plot_graphs(
            csv_path=csv_snooping_report_path,
            layer_names_column="Layer Output",
            comparator_columns=[comparator.name for comparator in comparators],
            output_dir=working_directory,
            algorithm=self._name,
            logger=self._logger,
        )

        # Return paths to generated reports
        return csv_snooping_report_path, json_snooping_report_path

    def _execute_layerwise(
        self,
        reference_model: Path,
        qdq_model: Path,
        input_sample: dict,
        reference_outputs: dict[str, np.ndarray],
        dump_output_tensors: bool,
        comparators: list[Comparator],
        working_directory: Path,
        intermediates_layers_list: list,
        cpu_layers_list: list,
        reference_model_is_qdq: bool,
        dump_elementwise_stats: dict[str, int],
    ) -> tuple[Path, Path]:
        """Execute LAYERWISE debugging algorithm.

        This method analyzes the model layer-by-layer by extracting per-layer subgraphs
        and running them through the ORT/QNN execution backend. For each layer:
        1. A subgraph is extracted from the layer's input to the selected dequantized output
        2. Subgraph inputs are resolved from original inputs, golden reference outputs,
           or previously computed target outputs (for mixed CPU/QNN execution)
        3. The subgraph is executed (optionally forcing CPU for a layer if user specifies)
        4. Outputs are compared against the reference using provided comparators
        5. A CSV row is appended with per-layer status, shapes, distributions, and metrics

        Args:
            reference_model (Path): Path to the reference ONNX model (FP32 or QDQ).
            qdq_model (Path): Path to the quantized QDQ ONNX model to be analyzed.
            input_sample (dict): Mapping of input tensor names to numpy arrays.
            reference_outputs (dict[str, np.ndarray]): Golden/reference outputs for comparison.
            dump_output_tensors (bool): Whether to dump intermediate outputs to disk.
            comparators (list[Comparator]): Comparators to compute verification metrics.
            working_directory (Path): Output folder where artifacts/reports are saved.
            intermediates_layers_list (list): Optional filter for layers/op types to analyze.
            cpu_layers_list (list): Layers or op types to force execution on CPU.
            reference_model_is_qdq (bool): Whether the reference model is QDQ (affects mapping).
            dump_elementwise_stats (dict[str, int]): Layers to dump elementwise comparison stats

        Returns:
            tuple[Path, Path]: A tuple of:
                - CSV file path containing per-layer results
                - JSON report path with structured results

        Raises:
            RuntimeError: If subgraph extraction or execution fails unexpectedly.
            KeyError: If expected tensors are missing when preparing inputs.
            Exception: Any unforeseen errors in per-layer processing are logged and
                       recorded in the CSV under the Exception column.

        Notes:
            - This workflow allows mixing CPU and QNN execution for isolating issues.
            - Layers with math-invariant ops are skipped to prioritize meaningful comparisons.
        """
        # Collect comparator metric names for CSV column headers
        comparator_names = [comparator.name for comparator in comparators]

        # Define CSV schema: identifiers, metadata, distributions, and comparator columns
        columns = [
            "Layer Name",
            "Layer Output",
            "Layer Status",
            "Layer Type",
            "Layer Shape",
            "Source(Min, Max, Median)",
            "Target(Min, Max, Median)",
        ]
        columns.extend(comparator_names)
        columns.append("Exception")

        # Pre-compute distributions for reference activations (used per-layer)
        reference_activation_info = self._get_activations_info(reference_outputs)

        # Initialize empty results table and output locations
        data_frame = pd.DataFrame(columns=columns)
        csv_path = working_directory / f"{Algorithm.LAYERWISE.name.lower()}.csv"
        subgraphs_dir = working_directory / "subgraph_data"
        # Ensure subgraph output directory exists for artifacts from each layer
        os.makedirs(subgraphs_dir, exist_ok=True)

        # Initialize framework runner to enable per-layer subgraph extraction
        framework_args = Namespace(
            framework="onnx", version=None, model_path=qdq_model, output_dir=subgraphs_dir
        )
        model_handler = FrameworkRunner(self._logger, framework_args)
        model_handler.load_framework()

        # Skip ops that are math-invariant or not useful for accuracy analysis
        skip_ops = ["QuantizeLinear", "DequantizeLinear", "Transpose", "Identity", "Constant"]

        # Accumulator for target activations to support mixed backend chaining
        target_outputs = {}

        # Load model to iterate nodes for subgraph extraction
        model = onnx.load(qdq_model)
        for idx, current_layer in enumerate(model.graph.node):
            # Skip no-op/math-invariant nodes to focus on meaningful compute ops
            if current_layer.op_type in skip_ops:
                self._logger.debug(f"Skipping layer {current_layer.name} as it is MATH INVARIANT")
                continue

            # Optional filter: process only specified nodes/op types if provided
            if (
                intermediates_layers_list
                and current_layer.name not in intermediates_layers_list
                and current_layer.op_type not in intermediates_layers_list
            ):
                continue

            # For the current node, identify the corresponding QDQ output nodes
            # that represents post-quantization values for fair comparison with reference outputs
            subgraph_intermediates = self._get_valid_intermediates(
                [current_layer.name],
                qdq_model,
                is_qdq_model=True,
            )

            # If there are no QDQ nodes associated, skip that layer
            if not subgraph_intermediates:
                self._logger.debug(
                    f"Skipping layer {current_layer.name} as it doesn't have QDQ nodes"
                )
                continue

            # Take farthest output in the graph topology which is -1
            subgraph_end_layer = subgraph_intermediates[-1]

            # Create a per-layer working directory to store extracted model and dumps
            current_subgraph_dir = subgraphs_dir / current_layer.name
            os.makedirs(current_subgraph_dir, exist_ok=True)

            err_msg = ""
            try:
                # Extract subgraph from the layer's input to its dequantized output
                self._logger.info(
                    f"Extracting subgraph from layer {current_layer.name} to {subgraph_end_layer}"
                )
                ret_status, extracted_model, graph_input_names = model_handler.extract_sub_graph(
                    current_layer.input[0], subgraph_end_layer, current_subgraph_dir
                )
            except Exception as exception:
                # Capture extraction failure and continue with next layer
                err_msg = f"Extraction failed with error: {exception}"
                self._logger.error(err_msg)
                ret_status = False

            if not ret_status:
                # Mark this layer as SKIP in the report with the error message
                self._logger.error("Couldn't extract current layer, proceeding to next layer.")
                data_frame = self._build_data_frame_for_subgraph(
                    status=ActivationStatus.SKIP,
                    layer_name=current_layer.name,
                    output_name=subgraph_end_layer,
                    verifier_scores=None,
                    layer_type=current_layer.op_type,
                    layer_shape=(),
                    source_distribution=(),
                    target_distribution=(),
                    comparator_names=comparator_names,
                    data_frame=data_frame,
                    exception=err_msg,
                )
                dump_csv(data_frame, csv_path, index=False)
                continue

            try:
                # Resolve inputs for the subgraph with precedence: original -> target(chained) -> golden
                self._logger.info(
                    f"Fetching inputs for the subgraph with names: {graph_input_names}"
                )
                subgraph_inputs = self._get_subgraph_inputs(
                    graph_input_names,
                    input_sample,
                    target_outputs,
                    reference_outputs,
                    True if cpu_layers_list else False,
                )

                # Decide backend for this subgraph: force CPU if configured for this node/op type.
                # This allows mixing CPU/QNN execution to isolate problematic layers while preserving
                # target outputs for subsequent subgraphs.
                use_cpu = (
                    current_layer.name in cpu_layers_list
                    or current_layer.op_type in cpu_layers_list
                )

                # Configure and invoke the ORT/QNN engine on the extracted subgraph
                inference_config = OrtQnnInferenceEngineInputConfig(
                    qdq_model=extracted_model,
                    input_sample=subgraph_inputs,
                    use_cpu=use_cpu,
                    dump_output_tensors=dump_output_tensors,
                    working_directory=current_subgraph_dir,
                    outputs_list=subgraph_intermediates,
                )

                # Execute inference and collect outputs
                self._logger.info("Executing subgraph")
                ort_qnn = OrtQnnInferenceEngine(self._logger)
                # Run ORT inference as a process so that segmentation fault can be caught,
                # otherwise it will stop the layerwise snooping
                with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
                    future = pool.submit(ort_qnn.run, inference_config)
                    try:
                        inference_output_config = future.result()
                    except Exception as e:
                        raise RuntimeError(f"ORT session error - {e}") from e

                if cpu_layers_list:
                    # Support mixed backend by chaining outputs as inputs to subsequent layers
                    target_outputs.update(inference_output_config.output_data)
            except Exception as exception:
                # Mark inference failure for this layer and proceed
                err_msg = f"Couldn't execute extracted model, proceeding to next layer. Reason: {exception}"
                self._logger.error(err_msg)
                data_frame = self._build_data_frame_for_subgraph(
                    status=ActivationStatus.INFERENCE_FAILURE,
                    layer_name=current_layer.name,
                    output_name=subgraph_end_layer,
                    verifier_scores=None,
                    layer_type=current_layer.op_type,
                    layer_shape=(),
                    source_distribution=(),
                    target_distribution=(),
                    comparator_names=comparator_names,
                    data_frame=data_frame,
                    exception=err_msg,
                )
                dump_csv(data_frame, csv_path, index=False)
                continue

            try:
                # Align target outputs with corresponding reference tensors and verify
                target_output = inference_output_config.output_data

                # Slice reference outputs to only those produced by this subgraph
                reference_output = {}
                for output_name in target_output.keys():
                    reference_name = None
                    if output_name in reference_outputs:
                        reference_name = output_name
                    elif Helper.transform_node_names(output_name) in reference_outputs:
                        reference_name = Helper.transform_node_names(output_name)
                    else:
                        raise Exception(f"Couldn't fetch golden reference for: {output_name}")

                    reference_output[output_name] = reference_outputs[reference_name]

                # Map names across reference/target models to account for quantization name changes
                reference_output_names = list(reference_output.keys())
                target_output_names = list(target_output.keys())
                tensor_mapping = self._get_onnx_tensor_mapping(
                    reference_output_names, target_output_names, reference_model_is_qdq
                )

                # Compute comparator metrics for this subgraph
                verifier_output = verify(
                    reference_output,
                    target_output,
                    self._logger,
                    comparators=comparators,
                    graph_info={"tensor_mapping": tensor_mapping},
                )

                # Dump elementwise stats if user specified
                if dump_elementwise_stats:
                    steps_allowed = None
                    if current_layer.name in dump_elementwise_stats:
                        steps_allowed = dump_elementwise_stats[current_layer.name]
                    elif current_layer.op_type in dump_elementwise_stats:
                        steps_allowed = dump_elementwise_stats[current_layer.op_type]
                    if steps_allowed is not None:
                        self._dump_elementwise_stats(
                            reference_output, target_output, steps_allowed, working_directory
                        )

            except Exception as exception:
                # Mark verification failure for this layer and proceed
                err_msg = f"Couldn't compare outputs, proceeding to next layer. Reason: {exception}"
                self._logger.error(err_msg)
                data_frame = self._build_data_frame_for_subgraph(
                    status=ActivationStatus.VERIFICATION_FAILURE,
                    layer_name=current_layer.name,
                    output_name=subgraph_end_layer,
                    verifier_scores=None,
                    layer_type=current_layer.op_type,
                    layer_shape=(),
                    source_distribution=(),
                    target_distribution=(),
                    comparator_names=comparator_names,
                    data_frame=data_frame,
                    exception=err_msg,
                )
                dump_csv(data_frame, csv_path, index=False)
                continue

            # Gather target distributions and build one CSV row per produced tensor
            target_activation_info = self._get_activations_info(target_output)
            for key, verifier_scores in verifier_output.items():
                target_tensor_name, reference_tensor_name = key[0], key[1]

                # Distribution summaries for target and reference (if available)
                target_distribution = target_activation_info[target_tensor_name].distribution
                reference_distribution = ()
                if reference_tensor_name in reference_activation_info:
                    reference_distribution = reference_activation_info[
                        reference_tensor_name
                    ].distribution
                if Helper.transform_node_names(reference_tensor_name) in reference_activation_info:
                    reference_distribution = reference_activation_info[
                        Helper.transform_node_names(reference_tensor_name)
                    ].distribution

                # Capture resulting tensor shape for metadata
                tensor_shape = target_activation_info[target_tensor_name].shape

                # Append the successful comparison row to the DataFrame
                data_frame = self._build_data_frame_for_subgraph(
                    status=ActivationStatus.SUCCESS,
                    layer_name=current_layer.name,
                    output_name=target_tensor_name,
                    verifier_scores=verifier_scores,
                    layer_type=current_layer.op_type,
                    layer_shape=tensor_shape,
                    source_distribution=reference_distribution,
                    target_distribution=target_distribution,
                    comparator_names=comparator_names,
                    data_frame=data_frame,
                )
                dump_csv(data_frame, csv_path, index=False)

        # After iterating all layers, create JSON and plots for visualization
        json_path = self._generate_json_report(data_frame, working_directory, Algorithm.LAYERWISE)

        plot_graphs(
            csv_path=csv_path,
            layer_names_column="Layer Output",
            comparator_columns=comparator_names,
            output_dir=working_directory,
            algorithm=self._name,
            logger=self._logger,
        )

        return csv_path, json_path

    def _get_subgraph_inputs(
        self, graph_inputs, original_inputs, target_outputs, golden_outputs, mixed_backend=False
    ):
        """Resolve input tensors required by a subgraph.

        Inputs are resolved with the following precedence:
        1. Original model inputs (provided by the user)
        2. Previously computed target outputs (only if mixed_backend is True)
        3. Golden/reference outputs

        Args:
            graph_inputs (list[str]): Names of the subgraph input tensors.
            original_inputs (dict[str, np.ndarray]): Original model input tensors.
            target_outputs (dict[str, np.ndarray]): Accumulated target activations from
                previously executed subgraphs (used when mixing CPU/QNN).
            golden_outputs (dict[str, np.ndarray]): Reference activations for fallback.
            mixed_backend (bool, optional): If True, allows pulling inputs from target_outputs
                to support chaining subgraphs executed on different backends. Defaults to False.

        Returns:
            dict[str, np.ndarray]: Mapping of subgraph input names to resolved numpy arrays.

        Raises:
            Exception: If an input cannot be resolved from any of the sources.
        """
        # Container for resolved inputs to the extracted subgraph
        subgraph_inputs = {}
        for ip in graph_inputs:
            # Highest precedence: use original model inputs if the subgraph expects them
            if ip in original_inputs:
                subgraph_inputs[ip] = original_inputs[ip]
            # Next: if mixing backends, allow feeding outputs of previously executed subgraphs
            elif ip in target_outputs and mixed_backend:
                subgraph_inputs[ip] = target_outputs[ip]
            # Fallback: use golden/reference activations for remaining inputs
            elif ip in golden_outputs:
                subgraph_inputs[ip] = golden_outputs[ip]
            # In case user passed reference outputs with sanitized names
            elif Helper.transform_node_names(ip) in golden_outputs:
                subgraph_inputs[ip] = golden_outputs[Helper.transform_node_names(ip)]
            else:
                # Input could not be sourced from any known location; abort this subgraph
                raise Exception(f"Couldn't fetch input: {ip}")
        return subgraph_inputs

    def _build_data_frame_for_subgraph(
        self,
        status: str,
        layer_name: str,
        output_name: str,
        verifier_scores: dict,
        layer_type: str,
        layer_shape: tuple,
        source_distribution: tuple,
        target_distribution: tuple,
        comparator_names: list[str],
        data_frame: pd.DataFrame,
        exception: str = "",
    ) -> pd.DataFrame:
        """Append a per-layer result row to the DataFrame.

        Builds a single row with layer metadata, activation distributions, comparator
        scores, and optional exception message. Non-success statuses populate comparator
        columns with "NaN" to clearly indicate uncomputed metrics.

        Args:
            status (str): ActivationStatus for the layer (e.g., SUCCESS, SKIP).
            layer_name (str): Logical layer name.
            output_name (str): Output tensor under analysis.
            verifier_scores (dict): Comparator scores keyed by comparator name.
            layer_type (str): ONNX op_type of the layer.
            layer_shape (tuple): Shape of the output tensor.
            source_distribution (tuple): (min, max, median) for reference activations.
            target_distribution (tuple): (min, max, median) for target activations.
            comparator_names (list[str]): Ordered list of comparator names (columns).
            data_frame (pd.DataFrame): The DataFrame to append the row to.
            exception (str, optional): Error/exception details if any. Defaults to "".

        Returns:
            pd.DataFrame: Updated DataFrame with the new row appended.
        """
        # Core per-row metadata and distribution summaries
        row_data = {
            "Layer Name": layer_name,
            "Layer Output": output_name,
            "Layer Status": status,
            "Layer Type": layer_type,
            "Layer Shape": layer_shape,
            "Source(Min, Max, Median)": source_distribution,
            "Target(Min, Max, Median)": target_distribution,
        }

        # Populate comparator metric columns when verification succeeded, otherwise "NaN"
        for comp in comparator_names:
            row_data[comp] = verifier_scores[comp] if status == ActivationStatus.SUCCESS else "NaN"

        # Attach any error/exception message captured during extraction/inference/verification
        row_data["Exception"] = exception

        # Append as the next row in the DataFrame
        data_frame.loc[len(data_frame)] = row_data
        return data_frame

    def _dump_elementwise_stats(
        self,
        reference_output: dict[str, np.ndarray],
        target_output: dict[str, np.ndarray],
        steps_allowed: int,
        working_directory: Path,
    ):
        """Dump elementwise comparison statistics for selected outputs.

        Compares each corresponding element in target vs reference outputs and logs
        detailed differences that exceed an allowed step threshold. This is useful
        for pinpointing exact indices where quantization deviates beyond tolerance.

        Args:
            reference_output (dict[str, np.ndarray]): Reference activations mapped by output name.
            target_output (dict[str, np.ndarray]): Target activations mapped by output name.
            steps_allowed (int): Maximum allowed absolute difference between corresponding
                elements. Accepts int/float or string-like values and is coerced to float.
            working_directory (Path): Directory where the 'elementwise_stats.log' file
                will be appended with findings.

        Behavior:
            - Validates shapes match before comparison.
            - Tracks and reports total out-of-tolerance elements and the maximum difference.
            - Appends human-readable logs to 'elementwise_stats.log' for later inspection.

        Notes:
            - If steps_allowed cannot be coerced to a float, it falls back to 0.0.
            - Missing outputs in either reference or target are skipped gracefully.
        """
        # Convert the allowed step threshold to float to ensure numeric comparisons are valid.
        # This supports values provided as strings (e.g. "2"), ints, or floats.
        try:
            allowed = float(steps_allowed)
        except Exception:
            self._logger.warning(f"Invalid steps_allowed='{steps_allowed}'. Falling back to 0.")
            allowed = 0.0

        try:
            # Iterate over outputs present in target_output; we will only compare those
            # that are also present in reference_output to avoid KeyErrors.
            for output_name in target_output:
                # Retrieve the candidate tensors. If either is missing, skip gracefully.
                reference = reference_output.get(output_name)
                target = target_output.get(output_name)
                if reference is None or target is None:
                    self._logger.warning(
                        f"Skipping elementwise stats for '{output_name}' as it is missing "
                        f"in reference/target outputs."
                    )
                    continue

                # Elementwise comparison requires shape equality so that indexing aligns.
                if reference.shape != target.shape:
                    self._logger.warning(
                        f"Skipping elementwise stats for '{output_name}' as reference shape: "
                        f"{reference.shape} and target shape: {target.shape} are not matching."
                    )
                    continue

                log = f"Starting Elementwise comparison for {output_name}\n"
                self._logger.debug(log)
                max_diff = -np.inf  # Track the largest absolute difference encountered
                diff_counter = 0  # Count how many elements exceed the allowed threshold

                # Iterate elementwise using ndarray indexing
                for idx, target_val in np.ndenumerate(target):
                    reference_val = reference[idx]
                    # Compute absolute difference; dtype casting rules are handled by NumPy.
                    abs_diff = abs(target_val - reference_val)
                    # Update maximum difference tracker
                    if abs_diff > max_diff:
                        max_diff = abs_diff
                    # If difference exceeds the allowed threshold, append a detailed log line.
                    if abs_diff > allowed:
                        log += (
                            f"Difference={abs_diff} at index {idx}, Steps allowed={allowed}, "
                            f"Target value: {target_val}, Reference value: {reference_val}\n"
                        )
                        diff_counter += 1

                # Persist findings only when at least one element exceeded the threshold.
                if diff_counter:
                    log += f"Total number of elements that are further than {allowed} steps: {diff_counter}\n"
                    log += f"Maximum step difference: {max_diff}\n\n\n"
                    # Append to a single log file in the working directory to accumulate findings.
                    save_path = os.path.join(working_directory, "elementwise_stats.log")
                    try:
                        with open(save_path, "a") as file:
                            file.write(log)
                    except Exception as file_err:
                        # File system errors should not abort the snooping run; warn and continue.
                        self._logger.warning(
                            f"Couldn't write elementwise stats for '{output_name}' to '{save_path}'. "
                            f"Reason: {file_err}"
                        )
        except Exception as e:
            # Top-level guard to ensure any unexpected failure does not break the overall flow.
            self._logger.warning(f"Dumping elementwise stats failed. Reason: {e}")

    def _get_valid_intermediates(
        self,
        intermediates_layers_list: list,
        model: Path,
        is_qdq_model: bool = False,
        qdq_ops_index: dict[str, int] = {"QuantizeLinear": 1, "DequantizeLinear": 2},
    ) -> list:
        """Extract valid intermediate layer output names from an ONNX model.

        Determines which intermediate outputs to capture during inference based on:
        - User-specified filter of node names or op types
        - Whether the model follows a QDQ pattern
        - A configurable mapping of QDQ ops to their relative indices

        QDQ Models:
        - Typical pattern: Op(i) → QuantizeLinear(i+1) → DequantizeLinear(i+2)
        - By default, both QuantizeLinear and DequantizeLinear outputs are captured
          via qdq_ops_index = {"QuantizeLinear": 1, "DequantizeLinear": 2}.
        - Callers may override qdq_ops_index to capture only specific QDQ outputs
          (e.g., only DequantizeLinear when comparing against an FP32 reference).

        FP32 Models:
        - Captures the direct outputs of the selected nodes (no QDQ traversal).

        Decision Logic:
        1. If intermediates_layers_list is provided and node name or op_type matches:
              - QDQ model → capture outputs of the downstream QDQ nodes defined by qdq_ops_index
              - FP32 model → capture node outputs directly
        2. If intermediates_layers_list is empty:
           - QDQ model → capture outputs from nodes whose op_type is present in qdq_ops_index
           - FP32 model → capture all node outputs

        Args:
            intermediates_layers_list (list): Filter of node names or op_types to analyze.
                Examples: ["conv1", "Conv", "MatMul"]. Empty list means "capture all".
            model (Path): Path to the ONNX model file (.onnx).
            is_qdq_model (bool, optional): Set to True for quantized models with QDQ nodes.
            qdq_ops_index (dict[str, int], optional): Mapping of QDQ op_type to relative
                index offset from the current node (i). Defaults to {"QuantizeLinear": 1,
                "DequantizeLinear": 2}. Override to narrow selection, e.g. {"DequantizeLinear": 2}.

        Returns:
            list: Output tensor names to capture during inference.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            onnx.onnx_cpp2py_export.checker.ValidationError: If model is invalid.
            IndexError: If a referenced downstream QDQ node is out of bounds.

        Notes:
            - Uses the caller-provided qdq_ops_index to traverse QDQ nodes instead of
              a hard-coded lookahead, making selection configurable and robust.
            - When reference is FP32, callers typically pass {"DequantizeLinear": 2}
              to capture post-dequantized values only.

        Example:
            >>> # Capture all Conv and MatMul layer outputs from a QDQ model (DQ only)
            >>> outputs = snooper._get_valid_intermediates(
            ...     ["Conv", "MatMul"],
            ...     Path("model_qdq.onnx"),
            ...     is_qdq_model=True,
            ...     qdq_ops_index={"DequantizeLinear": 2}
            ... )
        """
        # Load ONNX model from file
        model = onnx.load(model)
        valid_intermediates = []

        # Iterate through all nodes in the model graph
        for idx, node in enumerate(model.graph.node):
            # Condition #1 - User specified specific layers to analyze
            if intermediates_layers_list:
                if (
                    node.name in intermediates_layers_list
                    or node.op_type in intermediates_layers_list
                ):
                    # Handle QDQ vs FP32 models differently
                    if is_qdq_model:
                        # For QDQ models, select downstream QDQ outputs based on the caller-provided
                        # qdq_ops_index mapping. Typical pattern: Op(i) -> Q(i+1) -> DQ(i+2).
                        # Example:
                        #   - When reference is QDQ: {"QuantizeLinear": 1, "DequantizeLinear": 2}
                        #   - When reference is FP32: {"DequantizeLinear": 2} (capture post-dequantized values only)
                        try:
                            for op_type, op_idx in qdq_ops_index.items():
                                possible_qdq_op = model.graph.node[idx + op_idx]
                                if possible_qdq_op.op_type == op_type:
                                    # Capture the outputs of the matched QDQ node
                                    valid_intermediates.extend(possible_qdq_op.output)
                        except IndexError:
                            # If the current node is too close to the end, the expected Q/DQ nodes
                            # may not exist; warn and continue without failing the entire pass.
                            self._logger.warning(
                                f"Node '{node.name}' of type '{node.op_type}' at index {idx} "
                                f"doesn't have downstream QDQ nodes per qdq_ops_index={qdq_ops_index}"
                            )
                    else:
                        # For FP32 models, capture node outputs directly
                        valid_intermediates.extend(node.output)

            # Condition #2 - No specific layers specified, capture based on model type
            else:
                if is_qdq_model:
                    # For QDQ models, select downstream QDQ outputs based on qdq_ops_index
                    if node.op_type in qdq_ops_index:
                        valid_intermediates.extend(node.output)
                else:
                    # For FP32 models, capture node outputs directly
                    valid_intermediates.extend(node.output)

        return valid_intermediates

    def _get_onnx_tensor_mapping(
        self,
        reference_outputs_names: list,
        target_outputs_names: list,
        reference_model_is_qdq: bool,
    ) -> dict[str, str]:
        """Create a mapping between target model outputs and reference model outputs.

        This function establishes a correspondence between tensor names in the target
        (quantized) model and the reference (floating-point or QDQ) model. The mapping
        strategy differs based on whether the reference model is in QDQ format.

        Mapping Strategies:
        -------------------
        1. **Exact Name Match** (Always attempted first):
           - Direct match: "conv1_output" → "conv1_output"
           - Applied to both QDQ and FP32 reference models

        2. **Transformed Name Match** (Attempted second):
           - Handles external golden output files scenario
           - Uses Helper.transform_node_names() to normalize names
           - Example: "/conv1/output" → "conv1_output"

        3. **Prefix Matching** (Only when reference is FP32):
           - Handles quantization-added suffixes
           - Example: "conv1_output_quantized" → "conv1_output"
           - Example: "layer2_dequantized" → "layer2"
           - First match wins (no ambiguity resolution)

        Rationale:
        ----------
        Quantization often modifies tensor names by adding suffixes like:
        - "_quantized", "_dequantized", "_qdq"
        - "_scale", "_zero_point"

        While the underlying computational layer remains the same, these naming
        changes require intelligent matching to compare outputs correctly.

        Args:
            reference_outputs_names (list): List of tensor names from the reference
                model. These represent the ground truth outputs (FP32 or QDQ reference).
            target_outputs_names (list): List of tensor names from the target model.
                These are the quantized model outputs to be verified against reference.
            reference_model_is_qdq (bool): Flag indicating whether the reference model
                is in QDQ (Quantize-Dequantize) format.
                - True: Only exact and transformed name matches are used
                - False: Prefix matching is also enabled to handle quantization naming

        Returns:
            dict[str, str]: Dictionary mapping target tensor names (keys) to their
                corresponding reference tensor names (values). Only successfully
                mapped tensors are included in the result.

                Example return value:
                {
                    "conv1_output_qdq": "conv1_output",
                    "fc_dequant": "fc",
                    "layer2": "layer2"  # exact match
                }

        Raises:
            None: Function does not raise exceptions. Unmapped tensors are silently
                excluded from the result.

        Notes:
            - Unmapped target tensors are silently excluded from the result
            - Prefix matching only occurs when reference_model_is_qdq is False
            - The function assumes quantization adds suffixes, not prefixes
            - First match wins in prefix matching (no ambiguity resolution)
            - Empty lists are handled gracefully (returns empty mapping)
            - Logging provides visibility into mapping success rate

        Example:
            >>> # Example 1: FP32 reference with quantized target
            >>> ref_outputs = ["layer1", "layer2"]
            >>> tgt_outputs = ["layer1_qdq", "layer2_quantized", "layer3_new"]
            >>> mapping = self._get_onnx_tensor_mapping(ref_outputs, tgt_outputs, False)
            >>> print(mapping)
            {"layer1_qdq": "layer1", "layer2_quantized": "layer2"}

            >>> # Example 2: QDQ reference (exact match only)
            >>> ref_outputs = ["layer1_qdq"]
            >>> tgt_outputs = ["layer1_qdq"]
            >>> mapping = self._get_onnx_tensor_mapping(ref_outputs, tgt_outputs, True)
            >>> print(mapping)
            {"layer1_qdq": "layer1_qdq"}
        """
        # Initialize empty mapping dictionary to store target->reference correspondences
        tensor_mapping = {}

        # Iterate through all target model outputs to find corresponding reference outputs
        # Each target tensor needs to be matched to exactly one reference tensor
        for target_output_name in target_outputs_names:
            # ============================================================
            # Strategy 1: Exact Name Match
            # ============================================================
            # Check for direct name match between target and reference
            # This is the most reliable matching strategy and works for:
            # - Models where quantization preserves tensor names
            # - Both QDQ and FP32 reference models
            # - Tensors that haven't been renamed during quantization
            if target_output_name in reference_outputs_names:
                tensor_mapping[target_output_name] = target_output_name
                continue  # Move to next target output (match found)

            # ============================================================
            # Strategy 2: Transformed Name Match
            # ============================================================
            # Apply name transformation to handle framework-specific conventions
            # Helper.transform_node_names() may perform operations like:
            # - Removing special characters (/, :, etc.)
            # - Normalizing path-like names to flat names
            transformed_target_name = Helper.transform_node_names(target_output_name)
            if transformed_target_name in reference_outputs_names:
                tensor_mapping[target_output_name] = transformed_target_name
                continue  # Move to next target output (match found)

            # ============================================================
            # Strategy 3: Prefix Matching (FP32 Reference Only)
            # ============================================================
            # If no exact match and reference is FP32, try prefix matching
            # This handles quantization-introduced naming changes such as:
            # - "layer_output" (FP32) → "layer_output_quantized" (Quantized)
            # - "conv1" (FP32) → "conv1_dequantized" (Quantized)
            # - "fc" (FP32) → "fc_qdq" (Quantized)
            #
            # Note: This is only done for FP32 references because:
            # - QDQ references already have quantization naming
            # - Prevents incorrect matches between two quantized models
            elif not reference_model_is_qdq:
                # Search through all reference outputs for potential prefix matches
                for reference_output_name in reference_outputs_names:
                    # Check if target name starts with reference name
                    # This assumes quantization adds suffixes, not prefixes
                    # Example matches:
                    # - "conv1" matches "conv1_quantized"
                    # - "layer2" matches "layer2_dequantized"
                    # - "fc" matches "fc_qdq"
                    if target_output_name.startswith(reference_output_name):
                        # Additional validation: ensure it's a true suffix match
                        # Prevent false matches like "conv1" matching "conv10"
                        # by checking if the next character is a separator or end of string
                        suffix_start = len(reference_output_name)
                        if suffix_start == len(target_output_name) or target_output_name[
                            suffix_start
                        ] in ["_", "/", ".", ":"]:
                            # Map the target tensor to its reference counterpart
                            tensor_mapping[target_output_name] = reference_output_name
                            # Use first match (assumes no ambiguous prefixes)
                            # If multiple matches are possible, the first one wins
                            break
                    # Also check if the transformed target name starts with reference name
                    # This handles cases where transformation is needed before prefix matching
                    # Example: "/layer1/output_quantized" transforms to "layer1_output_quantized"
                    # which then matches "layer1_output"
                    elif transformed_target_name.startswith(reference_output_name):
                        # Validate suffix boundary for transformed name as well
                        suffix_start = len(reference_output_name)
                        if suffix_start == len(transformed_target_name) or transformed_target_name[
                            suffix_start
                        ] in ["_", "/", ".", ":"]:
                            tensor_mapping[target_output_name] = reference_output_name
                            break

        # ============================================================
        # Logging and Diagnostics
        # ============================================================
        # Calculate mapping statistics for debugging and monitoring
        mapped_count = len(tensor_mapping)
        total_target_count = len(target_outputs_names)
        unmapped_count = total_target_count - mapped_count

        # Log mapping statistics for debugging and quality assurance
        self._logger.info(
            f"Tensor mapping created: {mapped_count} out of {total_target_count} target tensors mapped to reference."
        )

        # Log unmapped tensors for debugging (only if there are unmapped tensors)
        if unmapped_count > 0 and self._logger.isEnabledFor(logging.DEBUG):
            unmapped_tensors = set(target_outputs_names) - set(tensor_mapping.keys())
            self._logger.debug(f"Unmapped target tensors: {sorted(unmapped_tensors)}")

        return tensor_mapping

    def _get_activations_info(self, outputs: dict[str, np.ndarray]) -> dict[str, ActivationInfo]:
        """Extract and compute statistical information from model's outputs data.

        This method processes a dictionary of activation tensors and generates comprehensive
        metadata for each tensor, including data type, shape, and statistical distribution
        metrics (min, max, median).

        Args:
            outputs (dict[str, np.ndarray]): A dictionary mapping tensor names to their
                corresponding NumPy arrays. Each array represents the output of a layer or
                operation in the neural network.

        Returns:
            dict[str, ActivationInfo]: A dictionary mapping tensor names to ActivationInfo
                objects containing metadata about each activation tensor. Each ActivationInfo
                includes:
                - dtype: String representation of the tensor's data type
                - shape: Tuple representing the tensor's dimensions
                - distribution: Tuple of (min, max, median) statistical values

        Raises:
            ValueError: If the outputs dictionary is empty or contains invalid data.
            TypeError: If tensor_data is not a NumPy array.

        Example:
            >>> outputs = {'layer1': np.array([[1.0, 2.0], [3.0, 4.0]])}
            >>> info = self._get_activations_info(outputs)
            >>> info['layer1'].shape
            (2, 2)
            >>> info['layer1'].distribution
            (1.0, 4.0, 2.5)
        """
        # Validate input to ensure we have data to process
        if not outputs:
            raise ValueError("outputs dictionary cannot be empty")

        # Initialize dictionary to store activation metadata for each tensor
        activation_info = {}

        # Iterate through each tensor in the outputs dictionary
        for tensor_name, tensor_data in outputs.items():
            # Validate that the tensor data is a NumPy array
            if not isinstance(tensor_data, np.ndarray):
                raise TypeError(
                    f"Expected np.ndarray for tensor '{tensor_name}', "
                    f"got {type(tensor_data).__name__}"
                )

            # Handle empty arrays gracefully to avoid computation errors
            if tensor_data.size == 0:
                min_val, max_val, median_val = np.nan, np.nan, np.nan
            else:
                # Compute statistical distribution metrics for the tensor
                # These metrics help understand the activation value ranges
                min_val = np.min(tensor_data)  # Minimum activation value
                max_val = np.max(tensor_data)  # Maximum activation value
                median_val = np.median(tensor_data)  # Median for central tendency

            # Create ActivationInfo object with comprehensive tensor metadata
            activation_info[tensor_name] = ActivationInfo(
                dtype=str(tensor_data.dtype),  # Convert dtype to string for serialization
                shape=tensor_data.shape,  # Preserve original tensor dimensions
                distribution=(min_val, max_val, median_val),  # Statistical summary
            )

        return activation_info

    def _generate_snooping_report(
        self,
        verifier_output: dict[tuple[str, str], dict],
        comparators: list[Comparator],
        working_directory: Path,
        reference_activation_info: dict,
        target_activation_info: dict,
        qdq_model: Path,
    ) -> tuple[Path, Path]:
        """Generate comprehensive CSV and JSON snooping reports from verification data.

        This method creates detailed reports containing:
        - Layer-wise comparison metrics (MSE, SQNR, etc.)
        - Tensor distribution statistics (min, max, median)
        - Layer metadata (name, type, shape)
        - Parent layer information for DequantizeLinear nodes

        The reports are generated in two formats:
        1. CSV: Human-readable tabular format for analysis in spreadsheet tools
        2. JSON: Machine-readable format for automation and further processing

        Args:
            verifier_output (dict[tuple[str, str], dict]): Verification results where
                keys are (target_tensor_name, reference_tensor_name) tuples and values
                are dictionaries containing:
                - Comparator results (MSE, SQNR, etc.)
                - Layer metadata (op_type, dimensions)
            comparators (list[Comparator]): List of comparator instances used for
                verification. Their names are used as column headers in the report.
            working_directory (Path): Directory path where reports will be saved.
                Reports are named based on the algorithm (e.g., "ONESHOT.csv").
            reference_activation_info (dict): Dictionary mapping reference tensor names
                to ActivationInfo objects containing distribution statistics.
            target_activation_info (dict): Dictionary mapping target tensor names to
                ActivationInfo objects containing distribution statistics.
            qdq_model (Path): Path to the QDQ ONNX model file. Used to extract layer
                hierarchy and parent node information.

        Returns:
            tuple[Path, Path]: A tuple containing:
                - csv_snooper_report_file: Path to generated CSV report
                - json_snooper_report_file: Path to generated JSON report

        Raises:
            FileNotFoundError: If qdq_model file doesn't exist
            PermissionError: If unable to write to working_directory
            KeyError: If expected keys are missing from verifier_output

        Note:
            - For DequantizeLinear nodes, the parent operation name is used as "Layer Name"
            - Distribution tuples are formatted as (min, max, median)
            - Empty distributions are represented as empty tuples ()

        Example:
            CSV Report Structure:
            | Layer Name | Layer Output   | Layer Type | Layer Shape | Source(...) | Target(...) | MSE | SQNR |
            |------------|----------------|------------|-------------|-------------|-------------|-----|------|
            | conv1      | conv1_dequant  | Conv       | [1,64,56,56]| (0,1,0.5)   | (0,1,0.5)   | 0.01| 45.2 |
        """
        # Load QDQ model to access node hierarchy and metadata
        model = onnx.load(qdq_model)
        model_nodes = model.graph.node

        # Initialize list to store report entries (one per tensor comparison)
        snooping_report = []

        # Iterate through verification results to build report entries
        for key, value in verifier_output.items():
            # Extract tensor names from the key tuple
            target_tensor_name, reference_tensor_name = key[0], key[1]

            # Get corresponding parent Op details for QDQ output nodes
            parent_op = self._get_qdq_parent_map(target_tensor_name, model_nodes)
            parent_op_name = parent_op_type = ""
            if parent_op:
                parent_op_name, parent_op_type = parent_op.name, parent_op.op_type

            # Get distribution statistics for target tensor
            target_distribution = target_activation_info[target_tensor_name].distribution

            # Get distribution statistics for reference tensor (if available)
            reference_distribution = ()
            if reference_tensor_name in reference_activation_info:
                reference_distribution = reference_activation_info[
                    reference_tensor_name
                ].distribution

            # Create report entry with layer information and statistics
            report_entry = {
                "Layer Name": parent_op_name,
                "Layer Output": target_tensor_name,
                "Layer Type": parent_op_type,
                "Layer Shape": target_activation_info[target_tensor_name].shape,
                "Source(Min, Max, Median)": reference_distribution,
                "Target(Min, Max, Median)": target_distribution,
            }

            # Add comparator results to report entry
            # Each comparator adds its metric as a column (e.g., MSE, SQNR)
            for comparator in comparators:
                report_entry[comparator.name] = value[comparator.name]

            # Append completed entry to report list
            snooping_report.append(report_entry)

        # Convert list of dictionaries to pandas DataFrame for easy manipulation
        snooping_report = pd.DataFrame(snooping_report)

        # Generate CSV report file path and save
        csv_snooper_report_file = working_directory / f"{Algorithm.ONESHOT.name.lower()}.csv"
        snooping_report.to_csv(csv_snooper_report_file, index=False)

        # Generate JSON report from the DataFrame
        json_snooper_report_file = self._generate_json_report(
            snooping_report, working_directory, Algorithm.ONESHOT
        )

        return csv_snooper_report_file, json_snooper_report_file

    def _get_qdq_parent_map(self, tensor_name: str, model_nodes: list):
        """Find the parent operation node for a QDQ node's output.

        In Quantization-Dequantization (QDQ) models, the typical pattern is:
        Op(i) → QuantizeLinear(i+1) → DequantizeLinear(i+2)

        This method traces back from a tensor produced by either QuantizeLinear or
        DequantizeLinear to identify the originating operation node. This yields
        more meaningful layer names in reports (e.g., "conv1" instead of "conv1_dequantize").

        Args:
            tensor_name (str): Name of the tensor output to trace back from.
                Typically an output of a QuantizeLinear or DequantizeLinear node,
                but can also be produced by any other node.
            model_nodes (list): List of ONNX node objects from model.graph.node.
                Each node has attributes: name, op_type, input, output.

        Returns:
            node or None: The parent operation node if found, otherwise None.
                - For DequantizeLinear: Returns the operation node 2 steps back (idx-2),
                  skipping QuantizeLinear.
                - For QuantizeLinear: Returns the operation node 1 step back (idx-1).
                - For other nodes: Returns the node that produces the tensor directly.
                - If tensor not found in any node outputs: Returns None.

        Notes:
            - Assumes a standard QDQ pattern with contiguous nodes; guards index bounds.
            - For nodes near the start of the graph where lookback is not possible,
              returns the QDQ node itself as a fallback.

        Example:
            >>> # For tensor "conv1_dequant_output" from DequantizeLinear at idx=5:
            >>> # Conv(idx=3) → Quantize(idx=4) → Dequantize(idx=5) → output
            >>> parent = self._get_qdq_parent_map("conv1_dequant_output", model_nodes)
            >>> print(parent.name)  # "conv1"
            >>> print(parent.op_type)  # "Conv"

            >>> # For tensor "conv1_quant_output" from QuantizeLinear at idx=4:
            >>> # Conv(idx=3) → Quantize(idx=4) → output
            >>> parent = self._get_qdq_parent_map("conv1_quant_output", model_nodes)
            >>> print(parent.name)  # "conv1"
            >>> print(parent.op_type)  # "Conv"
        """
        # Iterate through all nodes to find which one produces the target tensor
        for idx, node in enumerate(model_nodes):
            # Check if this node produces the target tensor as one of its outputs
            if tensor_name in node.output:
                # Check if this is a QuantizeLinear node (or variant)
                if node.op_type.startswith("Quantize"):
                    # Trace back to parent operation (typically 1 nodes before)
                    # Pattern: Op (idx-1) → QuantizeLinear (idx)
                    if idx >= 1:
                        return model_nodes[idx - 1]
                    else:
                        # Return the QuantizeLinear node itself as a fallback
                        return node
                elif node.op_type.startswith("Dequantize"):
                    # Trace back to the parent operation (typically 2 nodes before).
                    # Pattern: Op (idx-2) → QuantizeLinear (idx-1) → DequantizeLinear (idx)
                    if idx >= 2:
                        return model_nodes[idx - 2]
                    else:
                        # Return the DequantizeLinear node itself as a fallback
                        return node
                else:
                    # Not a QDQ node, return the current node directly
                    # This handles regular operations that produce the tensor
                    return node

        # Tensor name not found in any node's outputs
        return None

    def _generate_json_report(
        self, data_frame: pd.DataFrame, output_dir: Path, algorithm: str = Algorithm.ONESHOT
    ) -> Path:
        """Generate a JSON version of the snooping report.

        Creates a structured JSON report from the DataFrame containing snooping results.
        The JSON format is designed for:
        - Programmatic access and automation
        - Integration with CI/CD pipelines
        - Further analysis with custom tools

        The JSON structure typically includes:
        - Header with metadata (version, timestamp, model info)
        - List of layers with their comparison metrics
        - Nested structure for easy parsing

        Args:
            data_frame (pd.DataFrame): DataFrame containing the snooping report data
                with columns like:
                - Layer Name: Name of the layer
                - Layer Output: Output tensor name
                - Layer Type: Operation type
                - Layer Shape: Tensor dimensions
                - Comparator metrics: MSE, SQNR, etc.
                - Distribution statistics: Min, Max, Median
            output_dir (Path): Directory where the JSON report will be saved.
                The file will be named based on the algorithm (e.g., "ONESHOT.json" or "LAYERWISE.json").
            algorithm (str): Algorithm name to use for the output filename. Defaults to Algorithm.ONESHOT.

        Returns:
            Path: File path to the generated JSON snooper report.

        Raises:
            PermissionError: If unable to write to output_dir
            ValueError: If data_frame is empty or malformed

        Note:
            - Delegates to dump_json_report for actual generation
            - File naming follows pattern: {algorithm}.json
            - JSON is formatted with indentation for readability

        Example:
            JSON Structure:
            {
                "header": {
                    "version": "1.0",
                    "algorithm": "ONESHOT",
                    "timestamp": "2024-01-15T10:30:00"
                },
                "layers": [
                    {
                        "layer_name": "conv1",
                        "layer_type": "Conv",
                        "metrics": {"MSE": 0.01, "SQNR": 45.2}
                    }
                ]
            }
        """
        # Generate JSON report using the oneshot-specific method
        json_report_path = dump_json_report(
            data_frame, output_dir / f"{algorithm.name.lower()}.json"
        )

        return json_report_path


def execute_ort_qnn_snooper(args: list) -> None:
    """Parses arguments for OrtQnnSnooper and executes it.

    Args:
        args (list): List of arguments to be parsed.
    """
    # Parse arguments for model snooper
    ort_qnn_args = OrtQnnSnooperParser().parse(args)
    algorithm = getattr(ort_qnn_args, "algorithm", Algorithm.ONESHOT)
    working_directory = create_working_directory(
        working_directory=ort_qnn_args.working_directory,
        sub_directory=algorithm + "_snooping",
        output_directory=ort_qnn_args.output_directory
        if hasattr(ort_qnn_args, "output_directory")
        else None,
    )

    # Create Logger using get_logger method for Model Snooper
    logger = get_logger(
        logger_name="Snooping_OrtQnn",
        log_file_path=working_directory,
        log_file_name="accuracy_debugger",
        level=ort_qnn_args.log_level.upper(),
    )

    # Remove log_level argument from snooper args
    delattr(ort_qnn_args, "log_level")

    logger.info("Running ORT/QNN-EP snooping...")
    try:
        ort_qnn_snooper_config = OrtQnnSnooperInputConfig(
            qdq_model=ort_qnn_args.qdq_model,
            reference_model=ort_qnn_args.reference_model,
            input_sample=ort_qnn_args.input_sample,
            algorithm=ort_qnn_args.algorithm,
            comparators=ort_qnn_args.comparator,
            working_directory=working_directory,
            golden_reference_path=ort_qnn_args.golden_reference,
            set_intermediate_layers=ort_qnn_args.set_intermediate_layers,
            set_cpu_layers=ort_qnn_args.set_cpu_layers,
            dump_elementwise_stats=ort_qnn_args.dump_elementwise_stats,
            dump_output_tensors=True,
        )
        # Run ORT/QNN-EP Snooper
        ort_qnn_snooper = OrtQnnSnooper(logger)
        output = ort_qnn_snooper.run(ort_qnn_snooper_config)
    except Exception:
        logger.exception("Error occurred while executing ORT/QNN-EP Snooping.")
        raise
    else:
        logger.info(f"Snooping Completed. CSV Report generated at: {output.csv_snooping_report}")
        logger.info(f"Json Report generated at: {output.json_snooping_report}")
