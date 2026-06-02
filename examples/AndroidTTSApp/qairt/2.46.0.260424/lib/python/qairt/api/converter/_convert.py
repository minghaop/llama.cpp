# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import tempfile
from os import PathLike
from pathlib import Path
from typing import List, Optional, cast

from jsonschema import ValidationError
from typing_extensions import Unpack

from qairt.api.configs.common import BackendType
from qairt.api.converter.convert_util import (
    _calibration_config_arg_adapter,
    _converter_config_arg_adapter,
    disable_spawn_process_and_exec,
)
from qairt.api.converter.converter_config import CalibrationConfig, ConverterConfig
from qairt.api.model import Model
from qairt.modules.dlc_module import DlcModule
from qairt.utils.exceptions import (
    ApplyEncodingsError,
    ApplyLoraUpdatesError,
    ConversionError,
    OptimizationError,
    SerializationError,
)
from qairt.utils.loggers import get_logger
from qti.aisw.core.model_level_api.utils.subprocess_executor import (
    generate_input_list,
    get_name_input_pairs_from_input_list,
)
from qti.aisw.tools.core.modules.converter.common import BackendInfoConfig
from qti.aisw.tools.core.modules.converter.converter_module import (
    ConverterInputConfig,
    QAIRTConverter,
    get_input_dimensions,
)
from qti.aisw.tools.core.modules.converter.optimizer_module import OptimizerInputConfig, QAIRTOptimizer
from qti.aisw.tools.core.modules.converter.quantizer_module import (
    QAIRTQuantizer,
    QuantizerInputConfig,
)
from qti.aisw.tools.core.modules.converter.serializer_module import QAIRTSerializer, SerializerInputConfig
from qti.aisw.tools.core.modules.converter.utils import get_graph_configs

_convert_logger = get_logger("qairt.convert")

# TODO:
# - AISW-112821: Add support for in-memory model in QAIRT converter
# - AISW-115745: Add support for in-memory DLC output in QAIRT optimizer
# - AISW-115738: Add support for in-memory DLC in QAIRT quantizer
# - Add support for in-memory encodings object (TBD)


def _handle_custom_io_generation(model: str | PathLike, extra_args: dict) -> None:
    """
    Handle automatic IO configuration generation when create_custom_io flag is set.

    Args:
        model: The framework model path.
        extra_args: Dictionary of extra arguments that may contain IO configurations.
                   This dict will be modified in-place if IO configs are auto-generated.

    Raises:
        ConversionError: If auto-generation of IO config fails.
    """
    create_custom_io = extra_args.pop("create_custom_io", False)
    if create_custom_io:
        # Auto-generate IO configs if not already provided
        if (
            not extra_args.get("input_tensor_config")
            and not extra_args.get("output_tensor_config")
            and not extra_args.get("io_config")
        ):
            _convert_logger.info("create_custom_io=True: Auto-generating IO configurations from model")
            try:
                from qairt.api.converter.io_config_utils import infer_io_config_from_onnx

                input_configs, output_configs = infer_io_config_from_onnx(model)
                extra_args["input_tensor_config"] = input_configs
                extra_args["output_tensor_config"] = output_configs

                _convert_logger.debug(
                    f"Auto-generated {len(input_configs)} input tensor configs and "
                    f"{len(output_configs)} output tensor configs"
                )
            except Exception as e:
                raise ConversionError(f"Failed to auto-generate IO config: {e}")
        else:
            _convert_logger.warning(
                "create_custom_io=True but input_tensor_config/output_tensor_config/io_config already provided. "
                "Skipping auto-generation."
            )


def convert(
    model: str | PathLike,
    encodings: Optional[str | PathLike] = None,
    calibration_config: Optional[CalibrationConfig] = None,
    backend: str | BackendType = "HTP",
    **extra_args,
) -> Model:
    """
    Convert a framework model into a Model object.

    Args:
        model: The framework model path (frameworks supported: ONNX, TFLite and PyTorch).
        encodings: The encoding information to be applied to the graph.
        calibration_config: Configuration for calibration process. Use this option to pass in sample input data
                           for calibration.
                           See :class:`qairt.api.converter.converter_config.CalibrationConfig` for details.
        backend: Specifies the backend used for serialization. This arg is required when using V2 encodings.
                Available options are HTP (default), CPU, AIC, and LPAI.
                Note: For V2 encodings, only HTP is supported. For non-V2 encodings, all options are valid.
        **extra_args: Extra keyword arguments for conversion options.
                      See :class:`qairt.api.converter.converter_config.ConverterConfig` for details.

                      Additional supported arguments include:
                        - `lora_importer_config`: Optional configuration for LoRA Importer's
                            apply_lora_updates
                        - `lora_tensor_names`: Optional file specifying a list of tensor names
                            that should be updatable.
                        - `quant_updatable_mode`: Mode for quant-updatable tensors.
                        - `disable_batchnorm_folding`: (Optimization) Disables batch normalization folding.
                        - `expand_lstm_op_structure`: (Optimization) Expands LSTM operator structure.
                        - `multi_time_steps_lstm`: (Optimization) Enables multi-time steps LSTM optimization.
                        - `multi_time_steps_gru`: (Optimization) Enables multi-time steps GRU optimization.

    Examples:
        .. code-block:: python

            converted_model = qairt.convert("path/to/model")

            # For applying encodings -
            converted_model = qairt.convert("path/to/model", encodings="path/to/encodings")

            # For calibration
            calib_config = CalibrationConfig(dataset=input_data, batch_size = 4, act_precision = 16)
            converted_model = qairt.convert("/path/to/model", calibration_config=calib_config)

            # Using extra args
            converted_model = qairt.convert("path/to/model",
                                            float_precision=16,
                                            input_tensor_config=[{"name": "input", "shape": (1, 3, 224, 224), "desired_datatype"="int8"}])

            # Advance IO customization
            converted_model = qairt.convert("path/to/model",
                                            input_tensor_config=[InputTensorConfig(name="input", shape=(1,3,224,224), desired_datatype="int8",
                                            layout="NCHW", desired_layout="NHWC", quant_param = QuantParam(scale=1.0, offset=0), optional = True)],
                                            output_tensor_config=[OutputTensorConfig(name="output", layout="NCHW", desired_datatype="uint16", optional = True)])

            # Using optimizer args
            converted_model = qairt.convert("path/to/model",
                                            disable_batchnorm_folding=True,
                                            multi_time_steps_lstm=True)

            # See qairt.api.converter.converter_config.InputTensorConfig and qairt.api.converter.converter_config.OutputTensorConfig for details.

    Returns:
        Model: A Model instance that is executable on a QAIRT Runtime.

    Raises:
        ValidationError: If provided extra args are invalid.
        ConversionError: If model conversion fails.
        OptimizationError: If model optimization fails.
        SerializationError: If model serialization fails.
        ApplyEncodingsError: If apply encodings fails.

    .. note::

        To convert pytorch models, the `input_tensor_config` argument must be passed in to specify
        the input tensor shape and data type:

        .. code-block:: python

            converted_model = qairt.convert("model.pt", input_tensor_config= [InputTensorConfig(name="input",
            shape=(1,3,224,224))])
    """

    _convert_logger.info("Starting model conversion...")

    # Validating extra args
    _ = ConverterConfig(**extra_args)

    # Handle create_custom_io flag
    _handle_custom_io_generation(model, extra_args)

    # Extract parameters that are not part of ConverterInputConfig
    lora_importer_config: Optional[str | PathLike] = extra_args.pop("lora_importer_config", None)

    quant_updatable_mode = extra_args.get("quant_updatable_mode", None)
    lora_tensor_names: Optional[str | PathLike] = extra_args.get("lora_tensor_names", None)

    # Defining the keys that belong to the Optimizer
    OPTIMIZER_ARG_KEYS = [
        "disable_batchnorm_folding",
        "expand_lstm_op_structure",
        "multi_time_steps_lstm",
        "multi_time_steps_gru",
    ]

    optimizer_params = {}
    for key in OPTIMIZER_ARG_KEYS:
        if key in extra_args:
            optimizer_params[key] = extra_args.pop(key)

    converter_args = _converter_config_arg_adapter(extra_args)

    converter_input_config = ConverterInputConfig(
        input_network=str(model),
        quantization_overrides=encodings,
        **converter_args,
    )

    dlc_path = ""
    input_tensors = getattr(converter_input_config, "input_tensors", None)
    io_config = converter_args.get("io_config", None)

    input_dim = []
    if input_tensors or io_config:
        input_dim = get_input_dimensions(input_tensors=input_tensors, io_config=io_config)

    if len(input_dim) == 0:
        graph_configs: list[dict] = [{}]
    else:
        graph_configs = get_graph_configs({"input_dim": input_dim})

    # Varying input shapes for a tensor lead to multiple graph_configs, each requiring separate conversion to generate specialized DLC
    network_specialization = True if len(graph_configs) > 1 else False

    if network_specialization:
        _convert_logger.debug(
            "Multiple graphs detected. Model will be converted using Network specialization Flow."
        )

    _convert_logger.debug("Initializing converter, optimizer and serializer modules")

    qairt_converter = QAIRTConverter(logger=_convert_logger)
    qairt_optimizer = QAIRTOptimizer(logger=_convert_logger)
    qairt_serializer = QAIRTSerializer(logger=_convert_logger)

    # TODO: Implement global config to set QAIRT_TMP_DIR environment variable
    tmp_root_dir = os.getenv("QAIRT_TMP_DIR", default=tempfile.gettempdir())
    temp_working_dir = Path(tempfile.mkdtemp(prefix="temp_working_dir_", dir=tmp_root_dir))

    # set converter tmp dir to QAIRT_TMP_DIR
    os.environ["TMPDIR"] = tmp_root_dir

    for idx, config in enumerate(graph_configs):
        model_suffix = f"Model {idx + 1}" if len(graph_configs) > 1 else "Model"
        ### STEP 1: Conversion ###
        if config:
            converter_input_config.graph_specific_input_dims = config

        try:
            if getattr(converter_input_config, "onnx_simplification"):
                with disable_spawn_process_and_exec():
                    _convert_logger.debug(
                        f"Running conversion for {model_suffix} with spawn_process_and_exec disabled."
                    )
                    converter_output = qairt_converter.convert(converter_input_config)
            else:
                converter_output = qairt_converter.convert(converter_input_config)

            _convert_logger.debug(f"Completed conversion for {model_suffix} with output: {converter_output}")
        except Exception as exc:
            raise ConversionError(f"Model conversion failed: {exc}")

        ### STEP 2: Optimization ###
        optimizer_input_config = OptimizerInputConfig(
            ir_graph=converter_output.ir_graph,
            framework=converter_output.framework,
            backend_info=BackendInfoConfig(backend=backend),
            **optimizer_params,
        )

        try:
            optimizer_output = qairt_optimizer.optimize(optimizer_input_config)
            _convert_logger.debug(f"Completed optimization for {model_suffix} with output {optimizer_output}")
        except Exception as exc:
            raise OptimizationError(f"Model optimization failed: {exc}")

        ### STEP 3: Serialization ###
        output_dlc_name = Path(model).stem + ".dlc"
        serializer_input_config = SerializerInputConfig(
            output_dlc=str(temp_working_dir / output_dlc_name),
            dlc_backend_config=converter_output.dlc_backend_config,
            framework=converter_output.framework,
            optimized_graph=optimizer_output.optimized_graph,
            optimizer_args=optimizer_output.optimizer_args,
            network_specialization=network_specialization,
            lora_weight_list=lora_tensor_names,
            quant_updatable_mode=quant_updatable_mode,
        )

        try:
            serializer_output = qairt_serializer.serialize(config=serializer_input_config)
            _convert_logger.debug(
                f"Completed model serialization for {model_suffix} with serializer output: {serializer_output}"
            )
        except Exception as exc:
            raise SerializationError(f"Model serialization failed: {exc}")

        if not network_specialization:
            dlc_path = serializer_output.dlc_path

        # Cleanup
        del converter_output
        del optimizer_output

    if network_specialization:
        dlc_path = qairt_serializer.finalize_backend()
        _convert_logger.info("Network specialization completed successfully.")

    ### STEP 4: Apply Encodings ###
    data = None
    export_format = converter_args.get("export_format", None)

    if (
        encodings
        and not (calibration_config and calibration_config.dataset)
        and export_format == "DLC_STRIP_QUANT"
    ):
        _convert_logger.debug(
            "Encodings and export format 'DLC_STRIP_QUANT' provided without calibration config. Skipping quantization."
        )
    elif encodings or (calibration_config and calibration_config.dataset):
        if encodings and calibration_config:
            _convert_logger.debug(
                "Encodings and calibration config provided. Performing calibration and quantization"
            )
        elif encodings:
            _convert_logger.debug("Encodings information was provided. Applying encodings")
        elif calibration_config:
            _convert_logger.debug("Calibration config provided. Performing calibration and quantization")

        if calibration_config:
            data = calibration_config.dataset
            quantizer_args = _calibration_config_arg_adapter(calibration_config)
            quantizer_args["backend_info"] = optimizer_params.get("backend_info", None)
        else:
            data = None
            quantizer_args = {}

        if data:
            if isinstance(data, (PathLike, str)):
                resolved_paths = get_name_input_pairs_from_input_list(Path(data))

                # Check if any input names are empty
                has_empty_names = any(
                    not input_name for input_pair_list in resolved_paths for input_name, _ in input_pair_list
                )

                # Write input list to temporary file with appropriate format
                if has_empty_names:
                    _convert_logger.debug("Input list uses simple format (no input names)")
                else:
                    _convert_logger.debug("Input list uses named format")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                    updated_file_path = Path(temp_file.name)

                    for input_pair_list in resolved_paths:
                        if has_empty_names:
                            # Simple format: write paths without name prefix
                            line = " ".join(str(input_path) for _, input_path in input_pair_list)
                        else:
                            # Named format: write with name:=path format
                            line = " ".join(
                                f"{input_name}:={input_path}" for input_name, input_path in input_pair_list
                            )
                        temp_file.write((line + "\n").encode())

                    data = updated_file_path
            elif isinstance(data, List):
                data, _ = generate_input_list(data, temp_working_dir)
            else:
                try:
                    from torch.utils.data import DataLoader, Dataset

                    from qairt.api.converter.torch_convert_util import _convert_to_list

                    if isinstance(data, (DataLoader, Dataset)):
                        data = _convert_to_list(
                            data,
                            batch_size=cast(CalibrationConfig, calibration_config).batch_size,
                            num_of_samples=cast(CalibrationConfig, calibration_config).num_of_samples,
                        )
                        data, _ = generate_input_list(data, temp_working_dir)
                    else:
                        raise ValueError("Invalid dataset object passed of unknown type.")
                except ImportError:
                    raise ImportError("torch is not installed. Please install torch to use this function.")
        else:
            quantizer_args["float_fallback"] = True

        try:
            quantizer_obj = QAIRTQuantizer()
            quantizer_input_config = QuantizerInputConfig(
                input_dlc=dlc_path,
                output_dlc=str(temp_working_dir / output_dlc_name),
                input_list=data,
                **quantizer_args,
            )

            # Create quantizer module object.
            quantizer_output_config = quantizer_obj.quantize(quantizer_input_config)
            dlc_path = quantizer_output_config.dlc_output
            _convert_logger.debug("Completed quantization, output dlc path = {}".format(dlc_path))
        except Exception as exc:
            raise ApplyEncodingsError("IRQuantization failed: %s", exc)

    ### (Optional) STEP 5: Apply LoRA updates ###
    lora_importer_output = None

    if lora_importer_config is not None:
        try:
            with disable_spawn_process_and_exec():
                from qairt.modules.lora.lora_module import apply_lora_updates

                # Apply lora updates under a spawn process guard to avoid issues with converter's multiprocessing use
                lora_importer_output = apply_lora_updates(
                    model,
                    dlc_path,
                    lora_importer_config=lora_importer_config,
                    input_list=cast(Optional[PathLike], data),
                    output_dir=temp_working_dir,
                )
        except Exception as exc:
            raise ApplyLoraUpdatesError("apply_lora_updates failed: %s" % exc)

        if lora_importer_output is None:
            raise ApplyLoraUpdatesError("apply_lora_updates returned None despite valid config")

    # Create Model object
    dlc_module = DlcModule.load(dlc_path, working_dir=temp_working_dir)
    qairt_model = Model(module=dlc_module)

    # Sets the LoRA use case config if present
    if lora_importer_output:
        qairt_model.lora_use_cases = lora_importer_output

    _convert_logger.info("Convert completed successfully!")
    return qairt_model
