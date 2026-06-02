# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import json
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import ConfigDict, FilePath
from qti.aisw.accuracy_debugger.tensor_visualizer.visualizers import line_plot
from qti.aisw.accuracy_debugger.utils.constants import CSV_TO_JSON_FIELDS_MAP
from qti.aisw.accuracy_debugger.utils.file_utils import dump_json
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel
from qti.aisw.tools.core.utilities.comparators.comparator import Comparator
from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger
from qti.aisw.tools.core.utilities.verifier.verifier import Verifier


class InputSample(AISWBaseModel):
    name: str
    dimensions: Optional[List[int]] = None
    raw_file: FilePath
    data_type: Optional[str] = None


class DebuggerConfig(AISWBaseModel):
    """Base pydantic class for DebuggerConfig"""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
    )


class Namespace(Namespace):
    def __init__(self, data=None, **kwargs):
        if data is not None:
            kwargs.update(data)
        super(Namespace, self).__init__(**kwargs)


@dataclass
class ActivationInfo:
    """Represents activation information of a tensor.

    Attributes:
        dtype (str): Data type of the tensor.
        shape (list[int]): Shape of the tensor.
        distribution (tuple[float, float, float]): Distribution of the tensor.
    """

    dtype: str
    shape: list[int]
    distribution: tuple[float, float, float]


class ActivationStatus:
    """Activation Status"""

    INITIALIZED = "INITIALIZED"
    SKIP = "SKIP"
    CONVERTER_FAILURE = "CONVERTER_FAILURE"
    OPTIMIZER_FAILURE = "OPTIMIZER_FAILURE"
    SERIALIZER_FAILURE = "SERIALIZER_FAILURE"
    QUANTIZER_FAILURE = "QUANTIZER_FAILURE"
    OFFLINE_PREPARE_FAILURE = "OFFLINE_PREPARE_FAILURE"
    NET_RUN_FAILURE = "NET_RUN_FAILURE"
    CUSTOM_OVERRIDE_GENERATION_FAILURE = "CUSTOM_OVERRIDE_GENERATION_FAILURE"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    OUTPUT_DUMP_FAILURE = "OUTPUT_DUMP_FAILURE"
    INFERENCE_DONE = "INFERENCE_DONE"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    SUCCESS = "SUCCESS"
    UNKOWN_STATUS = "UNKOWN_STATUS"

    def __init__(self, activation_name, msg="initialize") -> None:
        self._current_status = ActivationStatus.INITIALIZED
        self._msg = msg
        self._activation_name = activation_name

    def set_status(self, status, msg):
        self._current_status = status
        self._msg = msg

    def get_status(self):
        return self._current_status

    def get_msg(self):
        return self._msg


def get_logger(
    logger_name: str, log_file_path: Path, log_file_name: str, level: str = "INFO"
) -> Logger:
    """Function to get a component specific logger

    Args:
        logger_name (str): Logger name.
        log_file_path (Path): Path to story log file.
        log_file_name (str): Log file name.
        level (str, optional): Log level. Defaults to "INFO".

    Returns:
        Logger: Logger object.
    """
    log_area = LogAreas.register_log_area(logger_name)
    logger = QAIRTLogger.register_area_logger(
        area=log_area,
        level=level,
        formatter_val="simple",
        log_file_path=log_file_path,
        log_file_name=log_file_name,
        handler_list=["user_console", "user_file_handler"],
    )
    return logger


def create_working_directory(
    sub_directory: str,
    working_directory: Optional[Path] = None,
    output_directory: Optional[str] = None,
) -> Path:
    """Create a working directory. If `output_directory` is not provided, delegate to
    `create_working_directory_with_timestamp`. Otherwise, create the directory
    using `output_directory` as the leaf name.

    Args:
        sub_directory (str): Subdirectory under the working directory.
        working_directory (Optional[Path]): Base working directory. If None, defaults to
                                            `<cwd>/working_directory` (and is created if missing).
        output_directory (Optional[str]): Name for the final directory. If None, a timestamp is used.

    Returns:
        Path: Path to the created directory.
    """
    # If no explicit name is provided, defer to the timestamp-based function.
    if output_directory is None:
        return create_working_directory_with_timestamp(sub_directory, working_directory)

    # Otherwise, mirror the original function's behavior for base handling.
    if working_directory is None:
        working_directory = Path.cwd() / "working_directory"
        working_directory.mkdir(parents=True, exist_ok=True)
    elif not Path(working_directory).exists():
        raise FileNotFoundError(f"Working directory {working_directory} does not exist.")

    working_dir = Path(working_directory) / sub_directory / output_directory
    working_dir.mkdir(parents=True, exist_ok=True)
    return working_dir


def create_working_directory_with_timestamp(
    sub_directory: str, working_directory: Optional[Path] = None
) -> Path:
    """Creates a working directory with a timestamp.

    This function creates a new directory with a timestamp in the format 'YYYY-MM-DD_HH-MM-SS'
    inside the provided working directory. If the working directory does not exist, it will
    be created. If the provided working directory is not a directory, a FileNotFoundError will be
    raised.

    Args:
        working_directory (Path): The path to the working directory. If None, the current working
                                  directory will be used.
        sub_directory (str): The name of the subdirectory to be created inside the working directory

    Returns:
        Path: The path to the newly created working directory with a timestamp.

    Raises:
        FileNotFoundError: If the provided working directory does not exist and is not None.
    """
    if working_directory is None:
        working_directory = Path.cwd() / "working_directory"
        working_directory.mkdir(parents=True, exist_ok=True)

    # If the provided working directory is not a directory, raise an error

    elif not Path(working_directory).exists():
        raise FileNotFoundError(f"Working directory {working_directory} does not exist.")

    working_directory = Path(working_directory) / sub_directory

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    working_dir = working_directory / timestamp
    working_dir.mkdir(parents=True, exist_ok=True)
    return working_dir


def load_input_tensors(
    input_tensors: list[InputSample] | dict[str, NDArray],
) -> dict[str, NDArray] | None:
    """Load input tensors from raw files to numpy arrays.

    Args:
        input_tensors: List of input tensors.

    Returns:
        dict[str, NDArray]: Dictionary of numpy arrays containing input tensor data.
    """
    if isinstance(input_tensors, list):
        input_tensors_dict = {}
        for tensor in input_tensors:
            dtype = tensor.data_type if tensor.data_type else np.float32
            input_tensors_dict[tensor.name] = np.resize(
                np.fromfile(tensor.raw_file, dtype), tensor.dimensions
            )
        return input_tensors_dict
    else:
        return input_tensors


def plot_graphs(
    csv_path: Path,
    layer_names_column: str,
    comparator_columns: list,
    output_dir: Path,
    algorithm: str,
    logger: Logger,
) -> None:
    """Plots graphs for verifiers/comparators scores present in the given CSV data.

    Args:
        csv_path: Path to the CSV snooper
        layer_names_column: The name of the column in the CSV that lists the names of the
                            layers in the model
        comparator_columns: List of comparator columns names in the given CSV
        output_dir: Output directory path
        algorithm: Algorithm name
        logger: Logger object
    """
    plots_save_dir = output_dir / "plots"
    plots_save_dir.mkdir(exist_ok=True)
    snooping_report_df = pd.read_csv(csv_path)

    for column in comparator_columns:
        try:
            logger.debug(f"Plotting graph for {column} scores...")
            line_plot(
                x=snooping_report_df[layer_names_column],
                y=[float(item) for item in snooping_report_df[column]],
                plot_name=column,
                save_dir=plots_save_dir,
            )
        except Exception as e:
            logger.warning(f"Plotting graph failed with error: {e}")

    logger.info(f"{algorithm} report plots saved at {plots_save_dir}")


def verify(
    reference_output: dict[str, NDArray],
    inference_output: dict[str, NDArray],
    logger: Logger,
    dlc_file: Path = None,
    comparators: list[Comparator] = None,
    graph_info: dict = None,
    is_qnn_golden_reference: bool = False,
) -> dict[tuple[str, str], dict[str, Any]]:
    """This method verifies the inference output with reference output using given comparators.

    Args:
        reference_output: Reference output.
        inference_output: Inference output.
        logger: Logger object
        dlc_file: Path to dlc file.
        comparators: List of comparators.
        graph_info: Dictionary containing graph information like, tensor mapping,
                    graph structure and layout information.
        is_qnn_golden_reference: Whether given golden outputs are from QNN.

    Returns:
        dict: A dictionary containing the verification results for each comparator.
    """
    verifier = Verifier(comparators, logger=logger)
    verifier_output = verifier.verify_dictionary_of_tensors(
        reference_output,
        inference_output,
        dlc_file=dlc_file,
        graph_info=graph_info,
        disable_layout_transform=is_qnn_golden_reference,
    )
    return verifier_output


def load_data_from_directory(
    directory: Path, intermediate_outputs_info: dict = {}
) -> dict[str, NDArray]:
    """This method recursively visits the directory and loads raw files as numpy array.
    If intermediate_outputs_info dictionary is provided, then raw files will be loaded as per
    datatypes and shapes specified in the given dictionary.

    Args:
        directory: Path to directory containing raw files
        intermediate_outputs_info: A dictionary containing datatypes and shapes information

    Returns:
        dict: A dictionary mapping from file name to numpy array.
    """
    tensor_data = {}
    for file in directory.rglob("*.raw"):
        tensor_name = file.stem
        sanitized_tensor_name = Helper.transform_node_names(tensor_name)

        if sanitized_tensor_name in intermediate_outputs_info:
            dtype = intermediate_outputs_info[sanitized_tensor_name]["dtype"]
            shape = intermediate_outputs_info[sanitized_tensor_name]["shape"]
        else:
            dtype = np.float32
            shape = None

        numpy_tensor = np.fromfile(file, np.float32).astype(dtype)
        tensor_data[tensor_name] = numpy_tensor.reshape(shape) if shape else numpy_tensor

    return tensor_data


def generate_reference_data_with_framework(
    logger: Logger,
    model: Path,
    inputs: dict[str, NDArray],
    dump_output_tensors: bool = False,
    output_dir: Optional[Path] = None,
    intermediate_output_tensors: Optional[list] = None,
) -> dict[str, NDArray]:
    """Generate golden reference outputs for the model using framework manager utility.

    Args:
        logger: Logger object
        model: Path to model file.
        inputs: Dictionary of input tensors.
        dump_output_tensors: Boolean to indicate whether to dump output tensors.
        output_dir: Path to directory to store output tensors
        intermediate_output_tensors: List of intermediate outputs to be dumped

    Returns:
        Dictionary containing reference outputs.
    """
    framework_manager = FrameworkManager(logger)
    framework_model = framework_manager.load(model)
    infer_shape = len(framework_model.graph.value_info) == 0
    reference_outputs = framework_manager.generate_intermediate_outputs(
        input_model=framework_model,
        input_data=inputs,
        infer_shape=infer_shape,
        intermediate_output_tensors=intermediate_output_tensors,
    )
    if dump_output_tensors:
        if output_dir is None:
            raise ValueError("Output directory must be specified when dump_output_tensors is True.")
        output_dir = output_dir / "reference_output"
        Helper.save_output_to_file(reference_outputs, output_dir)

    return reference_outputs


def dump_json_report(data_frame: pd.DataFrame, json_report_path: Path) -> Path:
    """Generates and dumps a JSON version of the oneshot snooping report.

    Creates a structured JSON report from the DataFrame containing snooping results.
    The JSON format includes a header with version information and a list of layers
    with their comparison metrics.

    Args:
        data_frame (pd.DataFrame): DataFrame containing the snooping report data.
        json_report_path (Path): Path to save JSON report.

    Returns:
        Path: File path to json snooper report
    """
    # Convert DataFrame to json format
    json_layers = json.loads(data_frame.to_json(orient="records"))

    new_layers = []
    for layer in json_layers:
        transformed_layer = {}
        comparators = {}

        for key, value in layer.items():
            if key in CSV_TO_JSON_FIELDS_MAP.keys():
                transformed_layer[CSV_TO_JSON_FIELDS_MAP[key]] = value
            else:
                # Handle comparator columns (e.g., "mse", "cosine", etc.)
                comparators[key] = "NaN" if value is None else value

        transformed_layer["comparators"] = comparators
        new_layers.append(transformed_layer)

    # Create the final JSON structure with header metadata and layer data
    json_data = {
        "header": {
            "header_version": {"major": 0, "minor": 1, "patch": 0},
            "artifact_type": "ONESHOT_SNOOPING_REPORT",
            "algorithm": "oneshot",
            "version": {"major": 0, "minor": 1, "patch": 0},
        },
        "layers": new_layers,
    }

    # Write the JSON data to file
    dump_json(json_data, json_report_path)
    return json_report_path
