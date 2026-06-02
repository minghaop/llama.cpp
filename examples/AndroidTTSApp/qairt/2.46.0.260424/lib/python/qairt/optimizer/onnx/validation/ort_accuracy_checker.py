# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""
Check accuracy with onnx runtime
"""

import copy
from collections.abc import Callable

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort

from qairt.optimizer.onnx.validation.utils import (
    generate_random_input,
    get_cosine_similarity,
    read_inputs_raw_paths,
)
from qairt.optimizer.utils.logger import logger


def run_onnx(onnx_path: str, inputs) -> dict[str, np.ndarray]:
    """
    get outputs of the model with onnxruntime

    Args:
        onnx_path:  onnx model path
        inputs: the inputs of the model
    Returns:
        outputs
    """
    sess_opt = ort.SessionOptions()
    logger.debug("  create session")
    ort_sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
        sess_options=sess_opt,
    )

    logger.debug("  run session")
    outputs = ort_sess.run(None, inputs)
    output_names = [x.name for x in ort_sess.get_outputs()]
    return dict(zip(output_names, outputs))


def get_sensitive_op_types(model_ir: ir.Model) -> set[str]:
    """
    get unstable op types existed in the model

    Here, "unstable" means the op outputs will be significantly influenced by the small variance of the inputs
    for example sort, topk

    Args:
        model_ir:  onnx model ir
    Returns:
        contain_sensitive_op_types
    """
    sensitive_op_types = set(["TopK", "Sort"])
    contain_sensitive_op_types = set()
    for n in model_ir.graph:
        if n.op_type in sensitive_op_types:
            contain_sensitive_op_types.add(n.op_type)
    return contain_sensitive_op_types


def verify_onnx_on_ort(
    origin_onnx_path,
    optimized_onnx_path,
    origin_inputs: dict[str, np.ndarray],
    optimized_inputs: dict[str, np.ndarray],
    optimized_inputs_preprocs: list[Callable] | None = None,
    optimized_outputs_postprocs: list[Callable] | None = None,
):
    # pylint: disable=[too-many-positional-arguments,too-many-locals,too-many-arguments,too-many-branches]
    """
    verify two onnx similairty by running onnx runtime with given inputs

    Args:
        origin_onnx_path:  origin onnx model path
        optimized_onnx_path:  optimized onnx model path
        origin_inputs:  origin onnx model inputs
        optimized_inputs:  optimized onnx model inputs
        optimized_inputs_preprocs:  preproccessor for optimized inputs
        optimized_outputs_postprocs:  postproccessor for optimized outputs

    """
    origin_model_ir = ir.load(origin_onnx_path)

    for input_v in origin_model_ir.graph.inputs:
        if input_v.name not in origin_inputs:
            raise ValueError(f"input '{input_v.name}' not found in origin_inputs")
        if isinstance((input_path := origin_inputs[input_v.name]), str):
            origin_inputs[input_v.name] = (
                np.fromfile(input_path, dtype=np.float32)
                .reshape(input_v.shape.numpy())
                .astype(input_v.dtype.numpy())
            )

    logger.debug("start to run original onnx...")
    origin_out_dict = run_onnx(origin_onnx_path, origin_inputs)
    logger.debug("finished to run original onnx.")

    optimized_model_ir = ir.load(optimized_onnx_path)
    for input_v in optimized_model_ir.graph.inputs:
        if input_v.name not in optimized_inputs:
            raise ValueError(f"input '{input_v.name}' not found in optimized_inputs")
        if isinstance((input_path := optimized_inputs[input_v.name]), str):
            optimized_inputs[input_v.name] = (
                np.fromfile(input_path, dtype=np.float32)
                .reshape(input_v.shape.numpy())
                .astype(input_v.dtype.numpy())
            )

    if optimized_inputs_preprocs is not None:
        for preproc in optimized_inputs_preprocs:
            optimized_inputs = preproc(optimized_inputs)

    logger.debug("start to run optimized onnx...")
    optimized_out_dict = run_onnx(optimized_onnx_path, optimized_inputs)
    logger.debug("finished to run optimized onnx.")

    if optimized_outputs_postprocs is not None:
        for preproc in optimized_outputs_postprocs:
            optimized_out_dict = preproc(optimized_out_dict)

    logger.info("Validation Reports:")
    max_mad = 0
    min_cosine_similarity = 1.0
    all_output_names = set(optimized_out_dict.keys()).union(set(origin_out_dict.keys()))
    intersect_output_names = [x for x in optimized_out_dict.keys() if x in origin_out_dict.keys()]
    for name in intersect_output_names:
        x = origin_out_dict[name]
        y = optimized_out_dict[name]
        mad = np.abs(x - y).max()
        cosine_similarity = get_cosine_similarity(x.reshape(-1), y.reshape(-1))
        max_mad = max(mad, max_mad)
        min_cosine_similarity = min(min_cosine_similarity, cosine_similarity)
        logger.info("for '%s': origin %s and optimized %s", name, str(x.shape), str(y.shape))
        logger.info("    MAD %f , cosine similarity %f", mad, cosine_similarity)

    # raise warnings for non-intersected outputs
    for name in all_output_names.difference(intersect_output_names):
        logger.warning("output '%s' not found in both origin and optimized onnx", name)

    sensitive_op_types = get_sensitive_op_types(origin_model_ir)

    if max_mad < 1e-2 or min_cosine_similarity > 0.999:
        logger.info(
            "Validation passed with onnxruntime with max-MAD %f, min-cosine %f",
            max_mad,
            min_cosine_similarity,
        )
    else:
        if len(sensitive_op_types) == 0:
            raise ValueError("failed to pass validation with onnxruntime")

        logger.warning(
            "failed to pass validation with onnxruntime with max-MAD %f, "
            + "min-cosine-similarity %f, but this is expected. "
            + "Since the model contains sensitive operations %s, "
            + "which are significantly sensitive to small variations in input data",
            max_mad,
            min_cosine_similarity,
            str(sensitive_op_types),
        )


def verify_onnx_with_random_inputs(
    origin_onnx_path,
    optimized_onnx_path,
    original_special_inputs: dict[str, np.ndarray] | None = None,
    optimized_special_inputs: dict[str, np.ndarray] | None = None,
    optimized_inputs_preprocs: list[Callable] | None = None,
    optimized_outputs_postprocs: list[Callable] | None = None,
):
    # pylint: disable=[too-many-positional-arguments,too-many-locals,too-many-arguments]
    """
    verify two cosine similairty by running onnx runtime with random inputs

    Args:
        origin_onnx_path:  origin onnx model path
        optimized_onnx_path:  optimized onnx model path
        original_special_inputs:  special inputs for the origin onnx model inputs
        optimized_special_inputs:  special inputs for the optimized onnx model inputs
        optimized_inputs_preprocs:  preproccessor for optimized inputs
        optimized_outputs_postprocs:  postproccessor for optimized outputs

    """
    origin_model = onnx.load(origin_onnx_path, load_external_data=False)
    random_in = generate_random_input(origin_model, original_special_inputs)

    optimized_onnx_inputs = random_in.copy()
    if optimized_special_inputs is not None:
        optimized_onnx_inputs.update(optimized_special_inputs)

    return verify_onnx_on_ort(
        origin_onnx_path=origin_onnx_path,
        optimized_onnx_path=optimized_onnx_path,
        origin_inputs=random_in,
        optimized_inputs=optimized_onnx_inputs,
        optimized_inputs_preprocs=optimized_inputs_preprocs,
        optimized_outputs_postprocs=optimized_outputs_postprocs,
    )


def verify_onnx_with_inputs_list(
    origin_onnx_path: str,
    optimized_onnx_path: str,
    origin_inputs_list_path,
    origin_input_raw_base_dir=None,
    optimized_onnx_special_inputs: dict[str, np.ndarray] | None = None,
    optimized_inputs_preprocs: list[Callable] | None = None,
    optimized_outputs_postprocs: list[Callable] | None = None,
):
    # pylint: disable=[too-many-arguments,too-many-positional-arguments]
    """
    verify two cosine similairty by running onnx runtime with given input lists

    Args:
        origin_onnx_path:  origin onnx model path
        optimized_onnx_path:  optimized onnx model path
        origin_inputs_list_path: input list of original model
        origin_input_raw_base_dir: base directory of original models' input raw files
        optimized_onnx_special_inputs:  special inputs for the optimized onnx model inputs
        optimized_inputs_preprocs:  preproccessor for optimized inputs
        optimized_outputs_postprocs:  postproccessor for optimized outputs
    """
    origin_inputs = read_inputs_raw_paths(origin_inputs_list_path, origin_input_raw_base_dir)[0]
    optimized_inputs = copy.deepcopy(origin_inputs)

    if optimized_onnx_special_inputs is not None:
        optimized_inputs.update(optimized_onnx_special_inputs)

    if optimized_inputs_preprocs is not None:
        for preproc in optimized_inputs_preprocs:
            optimized_inputs = preproc(optimized_inputs)

    return verify_onnx_on_ort(
        origin_onnx_path=origin_onnx_path,
        optimized_onnx_path=optimized_onnx_path,
        origin_inputs=origin_inputs,
        optimized_inputs=optimized_inputs,
        optimized_inputs_preprocs=optimized_inputs_preprocs,
        optimized_outputs_postprocs=optimized_outputs_postprocs,
    )
