# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import numpy as np
import onnxruntime as ort
import os
from typing import Dict
from pathlib import Path
import argparse
from contextlib import contextmanager
import logging

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model import OnnxModel
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper

logger = logging.getLogger(__name__)

ONNX_TYPE_TO_NUMPY  = {
        'tensor(float)': np.float32,
        'tensor(float16)': np.float16,
        'tensor(double)': np.float64,
        'tensor(int8)': np.int8,
        'tensor(int16)': np.int16,
        'tensor(int32)': np.int32,
        'tensor(int64)': np.int64,
        'tensor(uint8)': np.uint8,
        'tensor(uint16)': np.uint16,
        'tensor(uint32)': np.uint32,
        'tensor(uint64)': np.uint64,
        'tensor(bool)': np.bool_,
        'tensor(string)': np.object_,
    }

@contextmanager
def onnx_session(model_path: str, providers=None):
    """Context manager for ONNX Runtime sessions."""
    session = None
    try:
        session = ort.InferenceSession(
            model_path,
            providers=providers or ['CPUExecutionProvider']
        )
        yield session
    finally:
        if session is not None:
            session = None  # Release reference

def onnx_type_to_numpy_dtype(onnx_type_str: str) -> np.dtype:
    """
    Convert ONNX Runtime type string to numpy dtype.

    Args:
        onnx_type_str: ONNX type string like 'tensor(float)', 'tensor(int64)', etc.

    Returns:
        numpy dtype object
    """
    # Mapping from ONNX type strings to numpy dtypes

    if onnx_type_str not in ONNX_TYPE_TO_NUMPY:
        raise ValueError(f"Unsupported ONNX type: {onnx_type_str}")

    return np.dtype(ONNX_TYPE_TO_NUMPY[onnx_type_str])


def read_input_list(input_list: str) -> Dict[str, str]:
    """
    Read input list file and parse tensor names and file paths.
    If tensor_list entries are separated by newlines, only the first entry is used.

    Args:
        input_list: Path to input_list.txt file

    Returns:
        Dictionary mapping tensor names to file paths
    """
    input_list_path = Path(input_list)
    if not input_list_path.exists():
        logger.error(f"Input List not found: {input_list}")
        return None

    input_dict = {}

    input_sample_line = ""
    with open(input_list, "r") as file:
        for line in file:
            strip_line = line.strip()
            if strip_line:
                input_sample_line = strip_line
                break
    if not input_sample_line:
        raise ValueError(f"Invalid input_list provided")

    tensor_list = input_sample_line.split()
    base_path = Path(os.path.abspath(input_list)).parent
    for tensor in tensor_list:
        if not tensor:
            continue
        if ':=' in tensor:
            tensor_name, file_path = tensor.split(':=', 1)
            tensor_name = tensor_name.strip()
            file_path = file_path.strip()
            absolute_path = (Path(file_path) if Path(file_path).exists() else Path(base_path) / file_path).resolve()
            input_dict[tensor_name] = absolute_path
        else:
            raise ValueError(f"Invalid format in input_list: {tensor}. Expected format: tensor_name:=file_path")

    return input_dict


def load_tensor_from_file(file_path: Path) -> np.ndarray:
    """
    Load tensor from file. Supports raw binary files.

    Args:
        file_path: Path to tensor file

    Returns:
        Numpy array containing the tensor data
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {file_path}")

    return np.fromfile(file_path, dtype=np.float32)


def save_tensor_to_file(tensor: np.ndarray, output_path: Path) -> str:
    """
    Save tensor to file in .raw format (raw binary).

    Args:
        tensor: Numpy array to save
        output_path: Path where to save the tensor

    Returns:
        Path where the tensor was saved
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor.tofile(output_path)
    return str(output_path)


def save_input_list(
    input_feed: Dict[str, np.ndarray],
    output_path: str,
    prefix: str,
) -> None:
    """
    Save input tensors to files and create an input_list.txt file.

    Args:
        input_feed: Dictionary mapping tensor names to numpy arrays
        output_path: Base directory where files will be saved
        prefix: prefix to be added to input list and input folder
    Returns:
        None
    """
    output_dir_path = Path(output_path).resolve()
    test_inputs_dir = output_dir_path / f"test_inputs_{prefix}"
    test_inputs_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors and collect paths
    tensor_paths = []
    for tensor_name, tensor in input_feed.items():
        # Sanitize tensor name for filename
        safe_name = Helper.transform_node_names(tensor_name)
        tensor_file_path = test_inputs_dir / f"{safe_name}.raw"
        saved_path = save_tensor_to_file(tensor, tensor_file_path)
        tensor_paths.append(f"{tensor_name}:={saved_path}")

    # Create input_list.txt file
    input_list_path = output_dir_path / f"input_list_{prefix}.txt"
    with open(input_list_path, 'w') as f:
        f.write(' '.join(tensor_paths))


def run_onnx_inference(
    onnx_model_path: str,
    model_input_tensors: dict,
    previous_layer_output: dict,
    prefix: str,
    output_path: str
) -> Dict[str, str]:
    """
    Run ONNX inference using inputs from input_list.txt file.

    Args:
        onnx_model_path: Path to ONNX model file
        model_input_tensors: Dict with model input tensor_name and tensor
        previous_layer_output: Dict with previous layer output_name and tensor.
        prefix: prefix to be added to input list and input folder
        output_path: Directory where output tensors will be saved

    Returns:
        Dictionary mapping output tensor names to their saved file paths
    """

    with onnx_session(onnx_model_path) as session:

        input_data = [(inp.name, inp.shape, onnx_type_to_numpy_dtype(inp.type)) for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        input_feed = {}
        for required_input, inp_shape, inp_dtype in input_data:
            if required_input in model_input_tensors:
                tensor = model_input_tensors[required_input].astype(inp_dtype)
                expected_size = np.prod(inp_shape)
                if tensor.size != expected_size:
                    raise ValueError(
                        f"Tensor '{required_input}' size mismatch: "
                        f"expected {expected_size} elements (shape {inp_shape}), "
                        f"got {tensor.size} elements"
                    )
                input_feed[required_input] = tensor.reshape(inp_shape)
            elif previous_layer_output is not None and required_input in previous_layer_output.keys():
                input_feed[required_input] = previous_layer_output[required_input]
            else:
                raise ValueError(f"Missing required input: {required_input}")

        save_input_list(input_feed, output_path, prefix)

        outputs = session.run(output_names, input_feed)
        output_dict = dict(zip(output_names, outputs))

    return output_dict

def extract_subgraph(
    onnxfile: str,
    encoding_file: str,
    num_splits: int,
    input_list: str,
    split_embedding: bool = False,
    split_lmhead: bool = False,
    output_dir: str = './',
    generate_input: bool = True,
) -> None:
    """
    Split an ONNX model into multiple parts.

    Args:
        onnxfile: Path to input ONNX model file
        encoding_file: Path to encoding file
        num_splits: Number of splits to create
        input_list: Path to input_list.txt file containing initial input tensors
        split_embedding: Whether to split embedding layer (default: False)
        split_lmhead: Whether to split language model head (default: False)
        output_dir: Output directory for split models (default: './')
        generate_input: Whether to generate test inputs for each split (default: True)

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If num_splits is invalid
        RuntimeError: If splitting fails
    """
    # Validate inputs
    onnxfile_path = Path(onnxfile)
    encoding_file_path = Path(encoding_file)
    output_path = Path(output_dir)

    if not onnxfile_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnxfile}")
    if not encoding_file_path.exists():
        raise FileNotFoundError(f"Encoding file not found: {encoding_file}")
    if num_splits < 1:
        raise ValueError(f"num_splits must be >= 1, got {num_splits}")

    logger.info(f"Starting ONNX model splitting into {num_splits} parts")

    # Load ONNX model once for efficiency
    try:
        onnxmodel = OnnxModel.load(
            model_path=onnxfile,
            encodings_path=encoding_file
        )
    except (IOError, ValueError, RuntimeError) as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

    # Run splitting
    try:
        splits = onnxmodel.split(
            num_splits=num_splits,
            split_embedding=split_embedding,
            split_lm_head=split_lmhead
        )
    except Exception as e:
        raise RuntimeError(f"Failed to split ONNX model: {e}")

    # Create output directory
    split_output_dir = output_path / 'split_onnx'
    split_output_dir.mkdir(parents=True, exist_ok=True)
    previous_layer_output = None

    # Read input list
    input_file_dict = read_input_list(input_list)

    if input_file_dict:
        model_input_tensors = {}
        for tensor_name, file_path in input_file_dict.items():
            model_input_tensors[tensor_name] = load_tensor_from_file(file_path)

    # Process each split
    for idx, split_model in enumerate(splits):

        split_num = idx + 1
        prefix = f"split_model_{split_num}_of_{num_splits}"

        # Export split model
        try:
            split_model.export(str(split_output_dir), prefix=prefix)
            logger.info(f"Exported split {split_num}/{num_splits}")
        except Exception as e:
            logger.error(f"Failed to export split {split_num}: {e}")
            continue

        if generate_input and input_file_dict:
            # Load the exported model to get input/output names
            split_model_path = split_output_dir / f"{prefix}.onnx"
            if not split_model_path.exists():
                logger.warning(f"Split model file not found: {split_model_path}")
                continue
            prefix = f"{split_num}_of_{num_splits}"
            previous_layer_output = run_onnx_inference(
                split_model_path,
                model_input_tensors,
                previous_layer_output,
                prefix,
                output_dir)
