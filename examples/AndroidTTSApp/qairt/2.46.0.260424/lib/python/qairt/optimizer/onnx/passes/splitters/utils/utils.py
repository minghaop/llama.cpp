# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import logging
import os
import tempfile

import numpy as np
import onnx
import onnxruntime

from qairt.optimizer.onnx.passes.splitters.utils.pretty_print import Colors

QAIRT_TMP_DIR = os.getenv("QAIRT_TMP_DIR", tempfile.gettempdir())


def run_model_on_ort(
    model: str | bytes | os.PathLike,
    inputs: dict[str, np.ndarray],
    output_names: list[str],
) -> dict[str, np.ndarray]:
    """
    Run the ONNX model on ONNXRT.

    Args:
        model: onnx.ModelProto bytes or path to model
        inputs: dict to be passed as inputs to the model
        output_names: Ordered output names of the model
    Returns:
        Outputs of the model, corresponding to the given inputs
    """

    ort_session = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
    ort_outputs = ort_session.run(output_names, inputs)

    return dict(zip(output_names, ort_outputs))


def _generate_random_inputs(model: onnx.ModelProto):
    def get_tensor_proto_shape(tp: onnx.ValueInfoProto):
        return [dim.dim_value for dim in tp.type.tensor_type.shape.dim]

    def get_tensor_proto_dtype(tp: onnx.ValueInfoProto):
        return onnx.helper.tensor_dtype_to_np_dtype(tp.type.tensor_type.elem_type)

    inputs = {}
    for inp in model.graph.input:
        shape = get_tensor_proto_shape(inp)
        dtype = get_tensor_proto_dtype(inp)
        if inp.name == "input_ids":
            inputs["input_ids"] = np.random.randint(1, 500, shape).astype(dtype)
        elif inp.name == "lora_alpha":
            inputs["lora_alpha"] = np.zeros(shape).astype(dtype)
        else:
            inputs[inp.name] = np.random.rand(*shape).astype(dtype)
    return inputs


def validate_splits(
    model: str | onnx.ModelProto,
    splits: list[str | onnx.ModelProto],
    logger: logging.Logger,
):
    logger.info("Validating model splits by executing on ONNX Runtime")

    logger.debug("Generating random inputs for model")
    if isinstance(model, str):
        tmp_model = onnx.load(model, load_external_data=False)
        random_inputs = _generate_random_inputs(tmp_model)
        output_names = [output.name for output in tmp_model.graph.output]
    else:
        random_inputs = _generate_random_inputs(model)
        output_names = [output.name for output in model.graph.output]

    logger.debug("Generating golden outputs for full model")
    if isinstance(model, str):
        goldens = run_model_on_ort(model, random_inputs, output_names)
    else:
        with tempfile.TemporaryDirectory(dir=QAIRT_TMP_DIR) as tmpdir:
            temp_model_path = os.path.join(tmpdir, "model.onnx")
            onnx.save(model, temp_model_path, save_as_external_data=True)
            try:
                goldens = run_model_on_ort(temp_model_path, random_inputs, output_names)
                onnx.load_external_data_for_model(model, tmpdir)
            except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException:  # No external data loaded
                logger.warning("External data not loaded for model. Skipping ONNX Runtime validation!")
                return

    split_outputs: dict[str, np.ndarray] = {}

    with tempfile.TemporaryDirectory(dir=QAIRT_TMP_DIR) as tmpdir:
        for split_model in splits:
            if isinstance(split_model, str):
                path = split_model
                _tmp_model = onnx.load(path, load_external_data=False)
                split_input_names = [inp.name for inp in _tmp_model.graph.input]
                split_output_names = [out.name for out in _tmp_model.graph.output]
            else:
                path = os.path.join(tmpdir, "model.onnx")
                onnx.save(split_model, path, save_as_external_data=True)
                split_input_names = [inp.name for inp in split_model.graph.input]
                split_output_names = [output.name for output in split_model.graph.output]

            split_inputs = {
                input_name: np_input
                for input_name, np_input in random_inputs.items()
                if input_name in split_input_names
            }

            # Output of previous split as input to this split
            for out in split_outputs:
                if out in split_input_names:
                    split_inputs[out] = split_outputs[out]

            outputs = run_model_on_ort(path, split_inputs, split_output_names)
            split_outputs.update(outputs)

            if not isinstance(split_model, str):
                onnx.load_external_data_for_model(split_model, tmpdir)

    status = True
    for tensor_name, tensor in goldens.items():
        split_output = split_outputs[tensor_name]
        logger.debug(f"Output name: {tensor_name}. MAD: {np.abs(tensor - split_output).max()}")
        if not np.allclose(tensor, split_output, atol=1e-4):
            logger.warning(
                f"{Colors.FAIL}Output name: {tensor_name}. MAD: {np.abs(tensor - split_output).max()}{Colors.ENDC}"
            )
            status = False

    verification_str = f"{Colors.OKGREEN if status else Colors.FAIL}{'OK' if status else 'FAIL'}{Colors.ENDC}"
    logger.info(f"Verification Status ----- {verification_str} -----")
