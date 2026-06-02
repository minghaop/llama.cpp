# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""ORT QNN Inference Engine Module

This module provides an inference engine for running ONNX models with Quantization-Dequantization (QDQ)
nodes using ONNXRuntime's QNN Execution Provider (QNN-EP). It supports both QNN hardware acceleration
and CPU fallback execution.

Classes:
    OrtQnnInferenceEngineInputConfig: Configuration for inference engine inputs
    OrtQnnInferenceEngineOutputConfig: Configuration for inference engine outputs
    OrtQnnInferenceEngine: Main inference engine class for model execution
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from pydantic import DirectoryPath, Field, FilePath, model_validator
from qti.aisw.accuracy_debugger.framework_runner.frameworks.onnx_framework import (
    CustomOnnxFramework,
)
from qti.aisw.accuracy_debugger.utils.helper import DebuggerConfig, create_working_directory
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class OrtQnnInferenceEngineInputConfig(DebuggerConfig):
    """Input configuration class for ORT/QNN-EP Inference Engine.

    This class defines and validates all input parameters required for running
    inference using the ONNXRuntime QNN Execution Provider. It ensures that the
    provided model is a valid QDQ ONNX model and all paths are accessible.

    Attributes:
        qdq_model (FilePath): Path to the quantized-dequantized (QDQ) ONNX model file.
            Must be a valid .onnx file containing QDQ nodes for quantization-aware inference.
        input_sample (dict[str, NDArray]): Dictionary mapping input node names to
            their corresponding numpy arrays. If None, inference cannot be executed.
            Default is None.
        outputs_list (Optional[list]): List of intermediate layer output names to be extracted
            during inference. If None or empty, only final model outputs are returned.
            Default is None.
        use_cpu (Optional[bool]): Flag to use CPU Execution Provider instead of QNN-EP.
            Set to True for CPU-only execution, False for QNN hardware acceleration.
            Default is False.
        dump_output_tensors (Optional[bool]): Flag to enable dumping inference results to
            raw binary files in the working directory. Useful for debugging and validation.
            Default is False.
        working_directory (DirectoryPath): Path to the directory for storing artifacts and
            outputs. Created automatically if it doesn't exist.
            Default is created via create_working_directory factory.

    Raises:
        FileNotFoundError: If the specified qdq_model path does not exist.
        Exception: If the model is not in ONNX format or doesn't contain QDQ nodes.

    Example:
        >>> config = OrtQnnInferenceEngineInputConfig(
        ...     qdq_model="model.onnx",
        ...     input_sample={"input": np.random.randn(1, 3, 224, 224)},
        ...     outputs_list=["layer1_output", "layer2_output"],
        ...     use_cpu=False,
        ...     dump_output_tensors=True
        ... )
    """

    qdq_model: FilePath
    input_sample: dict[str, NDArray]
    outputs_list: Optional[list] = None
    use_cpu: Optional[bool] = False
    dump_output_tensors: Optional[bool] = False
    working_directory: DirectoryPath = Field(default_factory=create_working_directory)

    @model_validator(mode="after")
    def validate_qdq_model(self):
        """Validate the QDQ model file and its format.

        This validator performs three critical checks:
        1. Verifies that the model file exists at the specified path
        2. Ensures the file has a .onnx extension
        3. Confirms the model contains QDQ (Quantize-Dequantize) nodes

        Returns:
            self: The validated instance

        Raises:
            FileNotFoundError: If the model file does not exist at the specified path.
            Exception: If the file is not an ONNX model or lacks QDQ nodes.

        Note:
            This method is automatically called after model initialization due to
            the @model_validator decorator with mode="after".
        """
        CustomOnnxFramework.validate_model(self.qdq_model, validate_qdq=True)
        return self


class OrtQnnInferenceEngineOutputConfig(DebuggerConfig):
    """Output configuration class for ORT/QNN-EP Inference Engine.

    This class encapsulates the results of inference execution, including both
    the inference output tensors and the location of any dumped output files.

    Attributes:
        output_data (Optional[dict[str, np.ndarray]]): Dictionary mapping output node names
            to their corresponding inference results as numpy arrays. All outputs are
            converted to float32 format for consistency. None if inference hasn't been run.
        output_dir (Optional[DirectoryPath]): Path to the directory containing dumped
            raw output files. Only populated when dump_output_tensors is enabled in the
            input configuration. None otherwise.

    Example:
        >>> output_config = OrtQnnInferenceEngineOutputConfig(
        ...     output_data={"output": np.array([0.1, 0.9])},
        ...     output_dir=Path("/path/to/outputs")
        ... )
    """

    output_data: Optional[dict[str, np.ndarray]] = None
    output_dir: Optional[DirectoryPath] = None


class OrtQnnInferenceEngine:
    """User interface class for model inference using ONNXRuntime's QNN Execution Provider.

    This class provides a high-level interface for executing ONNX models with QDQ nodes
    using Qualcomm's QNN (Qualcomm Neural Network) Execution Provider. It supports:
    - Adding intermediate layer outputs for debugging and analysis
    - Hardware-accelerated inference via QNN-EP (default)
    - CPU fallback execution via CPU-EP
    - Output tensor dumping for validation

    The engine automatically modifies the model graph to include requested intermediate
    outputs while preserving the original model outputs.

    Attributes:
        logger (Any): Logger instance for tracking execution progress and debugging.
        log_area (str): Registered log area identifier (only if custom logger not provided).

    Example:
        >>> engine = OrtQnnInferenceEngine()
        >>> config = OrtQnnInferenceEngineInputConfig(
        ...     qdq_model="model.onnx",
        ...     input_sample={"input": np.random.randn(1, 3, 224, 224)}
        ... )
        >>> output = engine.run(config)
        >>> print(output.output_data)
    """

    def __init__(self, logger: Any = None) -> None:
        """Initialize OrtQnnInferenceEngine with optional custom logger.

        Args:
            logger (Any, optional): Custom python logger instance for logging inference
                operations. If None, a default QAIRT logger is created and registered
                with the "OrtQnnInference" log area at INFO level. Default is None.

        Note:
            When using the default logger, it will be registered with the QAIRT logging
            system and can be configured through standard QAIRT logging mechanisms.
        """
        if logger:
            # Use the provided custom logger
            self.logger = logger
        else:
            # Create and register a default QAIRT logger
            self.log_area = LogAreas.register_log_area("OrtQnnInference")
            self.logger = QAIRTLogger.register_area_logger(area=self.log_area, level="INFO")

    def run(self, config: OrtQnnInferenceEngineInputConfig) -> OrtQnnInferenceEngineOutputConfig:
        """Execute inference using ORT/QNN-EP on the provided QDQ model.

        This method performs the following operations:
        1. Loads the QDQ ONNX model from the specified path
        2. Modifies the model graph to include intermediate layer outputs
        3. Creates an ONNXRuntime session with QNN-EP or CPU-EP
        4. Executes inference with the provided input samples
        5. Organizes and optionally dumps the output tensors

        The method preserves all original model outputs while adding any requested
        intermediate layer outputs. All outputs are converted to float32 for consistency.

        Args:
            config (OrtQnnInferenceEngineInputConfig): Configuration object containing:
                - Model path and input data
                - List of intermediate outputs to extract
                - Execution provider preference (QNN or CPU)
                - Output dumping preferences

        Returns:
            OrtQnnInferenceEngineOutputConfig: Output configuration containing:
                - output_data: Dictionary of output tensors (all in float32 format)
                - output_dir: Path to dumped outputs (if dump_output_tensors is True)

        Raises:
            RuntimeError: If inference execution fails or session creation fails.
            ValueError: If input_sample is None or doesn't match model inputs.

        Note:
            - All output tensors are converted to float32 regardless of their original dtype
            - Intermediate outputs are prepended to the output list before original outputs
            - When dump_output_tensors is True, outputs are saved as .raw binary files

        Example:
            >>> engine = OrtQnnInferenceEngine()
            >>> config = OrtQnnInferenceEngineInputConfig(
            ...     qdq_model="model.onnx",
            ...     input_sample={"input": np.random.randn(1, 3, 224, 224)},
            ...     outputs_list=["conv1_output"],
            ...     dump_output_tensors=True
            ... )
            >>> result = engine.run(config)
            >>> print(f"Outputs: {list(result.output_data.keys())}")
        """
        # Handle None case for outputs_list
        outputs_list = config.outputs_list
        outputs_list = [] if outputs_list is None else outputs_list

        # Load the QDQ ONNX model from disk
        model = onnx.load(config.qdq_model)

        if outputs_list:
            # Preserve original model outputs by extracting them from the graph
            # Note: We pop() to clear the output list, then re-add them later
            original_outputs = []
            for output in range(len(model.graph.output)):
                original_outputs.append(model.graph.output.pop())

            # Add intermediate layer outputs to the model graph
            self.logger.info(f"Adding {len(outputs_list)} intermediate output nodes to the model")
            outputs_list = [onnx.ValueInfoProto(name=output) for output in outputs_list]
            model.graph.output.extend(outputs_list)

            # Re-add original outputs to maintain model's intended outputs
            model.graph.output.extend(original_outputs)
        else:
            self.logger.warning(
                "No intermediate outputs specified. Only final model outputs will be captured."
            )

        import onnxruntime_qnn as qnn_ep

        # Configure execution provider based on user preference
        if config.use_cpu:
            ep_registration_name = "CPUExecutionProvider"
            ep_backend = qnn_ep.get_qnn_cpu_path()
        else:
            ep_registration_name = "QNNExecutionProvider"
            ep_backend = qnn_ep.get_qnn_htp_path()

        # Register QNN EP library
        # The registration name is used as the EP name for QNN EP
        ep_lib_path = qnn_ep.get_library_path()
        ort.register_execution_provider_library(ep_registration_name, ep_lib_path)

        # Select OrtEpDevice(s) matching the registration name (which is the EP name for QNN EP)
        all_ep_devices = ort.get_ep_devices()
        selected_ep_devices = [
            ep_device for ep_device in all_ep_devices if ep_device.ep_name == ep_registration_name
        ]

        if len(selected_ep_devices) == 0:
            raise RuntimeError("QNN EP device not found.")

        # Create session options
        options = ort.SessionOptions()

        # Configure QNN EP options
        ep_options = {"backend_path": ep_backend}

        # Add QNN EP to session
        options.add_provider_for_devices(selected_ep_devices, ep_options)

        # # (Optional) Enable configuration that raises an exception if the model can't be
        # # run entirely on the QNN HTP backend.
        # options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

        # Create ONNXRuntime inference session with the modified model
        self.logger.info(f"Creating ORT session with following options: {options}")
        session = ort.InferenceSession(model.SerializeToString(), sess_options=options)

        # Execute inference with the provided input samples
        self.logger.info("Executing inference with ORT/QNN-EP")
        result = session.run(None, config.input_sample)

        # Clean up
        del session

        # Unregister the library after all sessions using it have been released
        ort.unregister_execution_provider_library(ep_registration_name)

        # Organize inference results into a dictionary mapping output names to tensors
        self.logger.info("Organizing generated inference outputs")
        node_names = [node.name for node in model.graph.output]
        inference_outputs = {}
        for name, output in zip(node_names, result):
            # Convert all outputs to float32 for consistency across different quantization schemes
            inference_outputs[name] = output.astype("float32")

        # Initialize output_dir to None (will be set if dumping is enabled)
        output_dir = None

        # Optionally dump inference outputs to raw binary files
        if config.dump_output_tensors:
            output_dir = self._dump_inference_outputs(inference_outputs, config.working_directory)

        # Create and populate the output configuration object
        inference_output_config = OrtQnnInferenceEngineOutputConfig(
            output_data=inference_outputs, output_dir=output_dir
        )

        self.logger.info("ORT_QNN-EP Inference completed successfully!")
        return inference_output_config

    def _dump_inference_outputs(
        self,
        inference_outputs: dict[str, np.ndarray],
        output_path: Path,
    ) -> Path:
        """Dump inference output tensors to raw binary files.

        This private method saves each output tensor as a separate .raw binary file
        in the specified output directory. Files are named using the output node names
        with a .raw extension. The method handles filesystem limitations by skipping
        outputs with names that would exceed the 255-byte filename limit.

        Args:
            inference_outputs (dict[str, np.ndarray]): Dictionary mapping output node names
                to their corresponding numpy array tensors. All tensors will be saved in
                their current dtype (typically float32).
            output_path (Path): Base directory path where outputs should be saved.
                An "Output" subdirectory will be created within this path.

        Returns:
            Path: Path to the "Output" directory containing all dumped tensor files.
                This directory is created if it doesn't exist.

        Note:
            - Output files are saved in raw binary format (numpy's tofile format)
            - Filenames exceeding 255 bytes (UTF-8 encoded) are skipped with a warning
            - The output directory structure is: output_path/Output/<tensor_name>.raw
            - Existing files with the same name will be overwritten
        """
        # Create the output directory structure
        output_path = output_path / "target_output"
        output_path.mkdir(parents=True, exist_ok=True)

        # Iterate through each output tensor and save to disk
        for output_name, out_tensor in inference_outputs.items():
            # Check filename length to avoid filesystem errors
            # Most filesystems have a 255-byte limit for filenames
            if len((output_name + ".raw").encode("utf-8")) > 255:
                self.logger.warning(
                    f"Skipping output tensor '{output_name}' as filename exceeds 255 bytes"
                )
                continue

            # Save tensor as raw binary file
            out_tensor.tofile(output_path / f"{Helper.transform_node_names(output_name)}.raw")

        return output_path
