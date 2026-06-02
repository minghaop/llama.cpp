# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import math
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from qti.aisw.accuracy_debugger.inference_engine.qairt_inference_engine import (
    InferenceEngineOutputConfig,
)
from qti.aisw.accuracy_debugger.utils.constants import InferenceStatus
from qti.aisw.accuracy_debugger.utils.file_utils import read_json
from qti.aisw.accuracy_debugger.utils.helper import ActivationStatus
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper


COMPILATIONS_LIMIT = 16  # To prevent bus error


def get_max_parallel_compilations(
    degree_of_freedom: int = 1,
    compilation_memory: float = None,
) -> int:
    """Computes the maximum number of parallel subgraph compilations that can be done based on:
    1. Available RAM Size
    2. Number of free cpu cores
    3. Degree of freedom
    4. Compilation memory

    max_parallel_compilations is calculated as:

    min(floor(available_ram/compilation_memory), free_cpu_cores) - degree_of_freedom

    If it comes out to be -ve, it returns 1

    Args:
        degree_of_freedom: The margin between the max process that can be run and should be run
        compilation_memory: Known memory requriement for one compilation (in bytes)

    Returns:
        int: maximum number of parallel compilations that can be done
    """
    available_ram = psutil.virtual_memory().available  # in bytes
    free_cpu_cores = get_free_cpu_cores()

    max_parallel_compilations = (
        min(math.floor(available_ram / compilation_memory), free_cpu_cores) - degree_of_freedom
    )

    return max(1, max_parallel_compilations)


class InferenceEngineProcess(multiprocessing.Process):
    """Implements Inference Engine Process which can be used to implement multiprocessing over
    Inference Engine. Use case is specific to subgraph snooper
    """

    def __init__(
        self,
        target,
        kwargs,
        inference_output_directory: Path,
    ) -> None:
        """Initializes InferenceEngineProcess object.

        Args:
            inference_output_directory: path to the inference engine working directory.
        """
        super().__init__(target=target, kwargs=kwargs)
        self._inference_output_directory = inference_output_directory

    def join(self) -> tuple[ActivationStatus, str, InferenceEngineOutputConfig]:
        """Joins the process"""
        # Join the Process
        super().join()
        result = InferenceEngineOutputConfig()

        # Post processing of the Inference Engine process status
        status_file = self._inference_output_directory / "status.json"

        # If status file does not exists -> IE failed
        if not status_file.exists():
            status = ActivationStatus.INFERENCE_FAILURE
            status_message = f"Execution failure. {status.value}"
            return status, status_message, result

        status_json = read_json(status_file)

        if status_json["converter_dlc_path"]:
            result.converter_dlc = status_json["converter_dlc_path"]
        if status_json["quantizer_dlc_path"]:
            result.quantizer_dlc = status_json["quantizer_dlc_path"]
        if status_json["context_binary_bin_path"]:
            result.offline_graph = status_json["context_binary_bin_path"]

        inference_status = InferenceStatus(status_json["status"])
        if inference_status == InferenceStatus.ConversionFailure:
            status = ActivationStatus.CONVERTER_FAILURE
            status_message = f"Conversion failure. {status_json['msg']}"
        elif inference_status == InferenceStatus.OptimizationFailure:
            status = ActivationStatus.OPTIMIZER_FAILURE
            status_message = f"Optimization failure. {status_json['msg']}"
        elif inference_status == InferenceStatus.SerializationFailure:
            status = ActivationStatus.SERIALIZER_FAILURE
            status_message = f"Serialization failure. {status_json['msg']}"
        elif inference_status == InferenceStatus.QuantizationFailure:
            status = ActivationStatus.QUANTIZER_FAILURE
            status_message = f"Quantization failure. {status_json['msg']}"
        elif inference_status == InferenceStatus.GenerateBinaryFailure:
            status = ActivationStatus.OFFLINE_PREPARE_FAILURE
            status_message = f"Offline graph preparation failure. {status_json['msg']}"
        elif inference_status == InferenceStatus.ExecutionFailure:
            status = ActivationStatus.NET_RUN_FAILURE
            status_message = f"Execution failure. {status_json['msg']}"
        elif inference_status == InferenceStatus.SUCCESS:
            status_message = ""
            result_dir = Path(status_json["result_dir"])
            if result_dir.exists():
                status = ActivationStatus.INFERENCE_DONE
                tensors = {}
                for tensor_name, info in status_json["tensor_info"].items():
                    tensor_file_path = result_dir / "Result_0" / f"{tensor_name}.raw"
                    tensors[tensor_name] = np.fromfile(
                        tensor_file_path, dtype=np.dtype(info["dtype"])
                    ).reshape(info["shape"])

                result.output_data = [tensors]
                result.output_dir = result_dir
            else:
                status = ActivationStatus.OUTPUT_DUMP_FAILURE
        else:
            status = ActivationStatus.UNKOWN_STATUS
            status_message = "Subgraph inference had unkown execution status."

        return status, status_message, result


def filter_snooping_report(
    snooping_report: pd.DataFrame, inference_data: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Filters given snooping report and returns filtered report.

    Filtering is applied to below scenarios:
    1. Conv -> Relu
    2. Add -> Relu
    In both cases, target graphs dump Relu output for Conv/Add nodes, leading to inconsistencies
    between framework outputs or AIMET outputs. Conv/Add entries that match the subsequent
    Relu node will be removed from snooping report.

    Args:
        snooping_report: Snooping report dataframe
        inference_data: Inference outputs corresponding to each entry in Snooping report

    Returns:
        pd.DataFrame: A Dataframe containing filtered snooping report
    """
    remove_indexes = []
    for index in range(0, len(snooping_report.index) - 1):
        if (
            snooping_report["Layer Type"][index] in ["Conv2d", "Eltwise_Binary"]
            and snooping_report["Layer Type"][index + 1] == "ElementWiseNeuron"
        ):
            current_node_name = snooping_report["Source Name"][index]
            next_node_name = snooping_report["Source Name"][index + 1]
            current_node_data = inference_data[Helper.transform_node_names(current_node_name)]
            next_node_data = inference_data[Helper.transform_node_names(next_node_name)]

            if current_node_data.shape == next_node_data.shape:
                unique_data = np.unique(current_node_data == next_node_data)
                if len(unique_data) == 1 and unique_data[0] == True:
                    remove_indexes.append(index)

    return snooping_report.drop(labels=remove_indexes, axis=0)


def get_free_cpu_cores(threshold: float = 20.0) -> int:
    """Get cpu cores with utilization within threshold.

    Args:
        threshold (float, optional): cpu utilization percentage. Defaults to 20.0.

    Returns:
        int: Number of free cores.
    """
    # Get the CPU usage for each core
    usage_per_core = psutil.cpu_percent(percpu=True, interval=1)

    # Count how many cores are under the usage threshold
    free_cores = len([core for core in usage_per_core if core < threshold])

    return free_cores


def convert_raw_file(
    input_file: Path, input_dtype: str, target_dtype: str, output_file: Path = None
) -> Path:
    """This function loads data from a file, converts it to the target data type,
    and dumps it to a new file.

    Args:
        input_file (Path): The path to the input file.
        input_dtype (str): The data type of the input data.
        target_dtype (str): The target data type.
        output_file (Path): The path to the output file. If not provided,
            the output file will be created in the same directory as the input file.

    Returns:
        Path: The path to the output file.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        IOError: If there's an issue reading from or writing to files.
    """
    try:
        # Load the data from the input file
        data = np.fromfile(input_file, dtype=input_dtype)

        # Convert the data to the target data type
        converted_data = data.astype(target_dtype)

        # If the output file is not provided, create a new file name
        if output_file is None:
            output_file = input_file.parent / f"converted_{input_file.name}"

        # Dump the converted data to the output file
        converted_data.tofile(output_file)

        return output_file
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file {input_file} not found")
    except Exception as e:
        raise IOError(f"Error converting file {input_file}: {str(e)}")


def convert_data(input_list: Path, user_provided_dtypes: list, output_dir: Path) -> Path:
    """This function converts all the tensors present in input_list such that they will be
    supported by converter. The converted tensors are dumped into new files.The paths of the new
    input tensors are stored in a list file created inside converted_inputs directory
    Args:
        input_list: input list provided
        user_provided_dtypes: List containing the datatypes of input tensors
        output_dir: Directory path to store the converted inputs
    Returns:
        Path to the new input list file
    Raises:
        FileNotFoundError: If the input_list file doesn't exist
        ValueError: If the number of input files doesn't match the dtypes provided
        IOError: If there's an issue reading from or writing to files
    """
    # Check if input_list file exists
    if not input_list.exists():
        raise FileNotFoundError(f"Input list file {input_list} not found")

    try:
        # Create a directory to dump the converted input files
        converted_input_file_dump_path = output_dir / "converted_calib_data"
        converted_input_file_dump_path.mkdir(parents=True, exist_ok=True)

        # Create a new input list file in the dump directory
        new_input_list_file_path = converted_input_file_dump_path / "input_list.txt"

        # Open the original and new input list files
        with open(input_list, "r") as old_file, open(new_input_list_file_path, "w") as new_file:
            # Iterate over each line in the original input list file
            for line in old_file:
                line = line.strip().split()
                if line:
                    if len(line) != len(user_provided_dtypes):
                        raise ValueError("Number of input files doesn't match the dtypes provided")
                    new_file_name_and_path = []
                    # Iterate over each file name and path in the line
                    for user_provided_dtype, file_name_and_path in zip(user_provided_dtypes, line):
                        file_name_and_path = (
                            file_name_and_path.split(":=")
                            if ":=" in file_name_and_path
                            else [None, file_name_and_path]
                        )
                        # If user provided dtype is None, set it to float32
                        if user_provided_dtype is None:
                            user_provided_dtype = "float32"
                        # Convert the tensor to 32 bit if necessary
                        target_dtype = "float32"
                        # Convert and dump the tensor to a new file
                        output_file = convert_raw_file(
                            Path(file_name_and_path[1]),
                            user_provided_dtype,
                            target_dtype,
                            converted_input_file_dump_path / Path(file_name_and_path[1]).name,
                        )

                        # Add the new file name and path to the new line
                        new_file_name_and_path.append(
                            (file_name_and_path[0] + ":=" if file_name_and_path[0] else "")
                            + str(output_file)
                        )
                    # Write the new line to the new input list file
                    new_file.write(" ".join(new_file_name_and_path) + "\n")

        # Update new input list file path
        return new_input_list_file_path
    except (IOError, OSError) as e:
        raise IOError(f"Error processing input list file: {str(e)}")
