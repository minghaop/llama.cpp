# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml  # type: ignore

import qti.aisw.lora.lora_importer_app as importer_app
from qairt.modules.lora.lora_config import (
    LoraBuilderInputConfig,
    LoraBuilderOutputConfig,
    UseCaseOutputConfig,
    load_use_case_config,
    serialize_lora_input_config,
)
from qairt.utils.exceptions import MissingConfigFileError
from qairt.utils.loggers import get_logger
from qti.aisw.lora.lora_importer_app import (
    apply_lora_updates as module_apply_lora_updates,
)
from qti.aisw.lora.lora_mapper_app import LoraMapperAppConfig, resolve_attach_point_name
from qti.aisw.lora.lora_model_creator_app import *

_logger = get_logger(name="LoraModule")

importer_app.module_lora_flow = True


def _get_lora_importer_namespace_args(
    model: Union[str, os.PathLike],
    dlc_path: Union[str, os.PathLike],
    lora_config: Union[str, os.PathLike, List[UseCaseOutputConfig]],
    input_list: Optional[os.PathLike],
    output_dir: Union[str, os.PathLike],
) -> Namespace:
    """
    Constructs a Namespace object for LoRA importer using individual parameters.

    Args:
        model: Path to the base model directory or file.
        dlc_path: Path to the input DLC file.
        lora_config: LoRA config path or list of use-case configs.
        input_list: Path to YAML file specifying use-cases.
        output_dir: Directory to save updated model artifacts.

    Returns:
        Namespace: Arguments packaged for module_apply_lora_updates.
    """
    args = Namespace()
    args.input_network = str(model)
    args.input_dlc = str(dlc_path)
    args.lora_config = lora_config
    args.input_list = str(input_list) if input_list else ""
    args.output_dir = str(output_dir)

    # Setting below to default values
    args.debug = -1
    args.skip_validation = False
    args.dump_usecase_dlc = False
    args.dump_usecase_onnx = False
    args.skip_apply_graph_transforms = False
    args.float_fallback = input_list is None

    return args


def _relativize_importer_config_paths(config_dir: Union[str, os.PathLike]) -> None:
    """
    Rewrites absolute or external paths in lora_importer_config.yaml to be relative to the given config directory.

    Args:
        config_dir (Union[str, os.PathLike]): The directory containing lora_importer_config.yaml and related files.

    Returns:
        None
    """
    config_dir = Path(config_dir)
    importer_config_path = config_dir / "lora_importer_config.yaml"

    if not importer_config_path.exists():
        return

    with open(importer_config_path, "r") as f:
        importer_data = yaml.safe_load(f)

    for uc in importer_data.get("use_case", []):
        for key in ["model_name", "lora_weights", "quant_overrides"]:
            if key in uc:
                abs_path = Path(uc[key])
                if not abs_path.is_absolute():
                    abs_path = config_dir / uc[key]
                uc[key] = os.path.relpath(abs_path, config_dir)

    with open(importer_config_path, "w") as f:
        yaml.safe_dump(importer_data, f)


def _relativize_lora_output_paths(output_dir: Union[str, os.PathLike]) -> None:
    """
    Rewrites absolute paths in lora_output_files.yaml to be relative to the given output directory.

    Args:
        output_dir (Union[str, os.PathLike]): The directory containing lora_output_files.yaml and related files.

    Returns:
        None
    """
    output_dir = Path(output_dir)
    output_yaml_path = output_dir / "lora_output_files.yaml"
    if not output_yaml_path.exists():
        return
    with open(output_yaml_path, "r") as f:
        output_data = yaml.safe_load(f)

    for uc in output_data.get("use_case", []):
        for key in ["encodings", "weights"]:
            if key in uc:
                abs_path = Path(uc[key])
                if not abs_path.is_absolute():
                    abs_path = output_dir / uc[key]
                uc[key] = os.path.relpath(abs_path, output_dir)

    with open(output_yaml_path, "w") as f:
        yaml.safe_dump(output_data, f)


def build_lora_graph(
    lora_input_config: LoraBuilderInputConfig,
    output_dir: str | os.PathLike,
) -> LoraBuilderOutputConfig:
    """
    Builds a max rank, concatenated LoRA configuration graph by processing the input configuration
    through a mapper and model creator pipeline.

    Args:
        lora_input_config (LoraBuilderInputConfig): Input configuration, either as a path or an object.
        output_dir (str | os.PathLike): Directory to store final outputs.

    Returns:
        LoraBuilderOutputConfig: Output config object.
    """
    root_dir = os.getenv("QAIRT_TMP_DIR", tempfile.gettempdir())

    with tempfile.TemporaryDirectory(dir=root_dir) as tmp_dir:
        # Handle input config
        if lora_input_config.lora_config_obj:
            _logger.debug("Serializing LoRA config object to YAML in temp dir.")
            lora_config_path = serialize_lora_input_config(lora_input_config.lora_config_obj, tmp_dir)
        else:
            _logger.debug("Using provided LoRA config path.")
            lora_config_path = str(lora_input_config.lora_config_path)

        # Run lora_mapper
        try:
            _logger.debug("Running LoRA mapper.")
            lora_mapper_output_dir = os.path.join(tmp_dir, "lora_mapper_output")
            os.makedirs(lora_mapper_output_dir, exist_ok=True)

            app_config = LoraMapperAppConfig(
                lora_config=lora_config_path, output_dir=lora_mapper_output_dir, debug=False
            )
            resolve_attach_point_name(app_config)
            _logger.debug("lora_mapper ran successfully")
        except Exception as e:
            raise RuntimeError(f"LoRA mapper failed: {e}")

        # Locate the updated config file
        model_creator_lora_config_path = next(
            (
                os.path.join(lora_mapper_output_dir, f)
                for f in os.listdir(lora_mapper_output_dir)
                if f.endswith((".yaml", ".yml"))
            ),
            None,
        )

        if not model_creator_lora_config_path:
            raise MissingConfigFileError("No YAML config file found in mapper output.")

        # Run lora_model_creator (final output goes to output_dir)
        try:
            _logger.debug("Running LoRA model creator.")
            lora_model_creator_output_dir = os.path.join(output_dir, "lora_model_creator_output")
            os.makedirs(lora_model_creator_output_dir, exist_ok=True)

            lora_model_creator_app = LoraModelCreatorApp(  # type: ignore
                model_creator_lora_config_path,
                lora_model_creator_output_dir,
                skip_validation=False,
                quant_updatable_mode=lora_input_config.quant_updatable_mode,
            )

            lora_model_creator_app.run()
            _logger.debug("lora_model_creator ran successfully")

            _relativize_importer_config_paths(lora_model_creator_output_dir)
        except Exception as e:
            raise RuntimeError(f"LoRA model creator failed: {e}")

    # Load final output config
    lora_tensor_names_path = os.path.join(lora_model_creator_output_dir, "lora_tensor_names.txt")
    lora_tensor_names_path = os.path.relpath(lora_tensor_names_path, lora_model_creator_output_dir)

    use_case_outputs: List[UseCaseOutputConfig] = load_use_case_config(
        Path(lora_model_creator_output_dir) / "lora_importer_config.yaml"
    )

    # Capture base model artifacts by scanning for .onnx and .data files
    base_model_artifacts = {}
    for fname in os.listdir(lora_model_creator_output_dir):
        if fname.endswith(".onnx"):
            base_model_artifacts["onnx"] = os.path.relpath(
                os.path.join(lora_model_creator_output_dir, fname), lora_model_creator_output_dir
            )
        elif fname.endswith(".data"):
            base_model_artifacts["data"] = os.path.relpath(
                os.path.join(lora_model_creator_output_dir, fname), lora_model_creator_output_dir
            )
        elif fname == "base_encodings.json":
            base_model_artifacts["encodings"] = os.path.relpath(
                os.path.join(lora_model_creator_output_dir, fname), lora_model_creator_output_dir
            )

    lora_output_config: LoraBuilderOutputConfig = LoraBuilderOutputConfig(
        use_case=use_case_outputs,
        lora_tensor_names=lora_tensor_names_path,
        base_model_artifacts=base_model_artifacts or {},
    )

    return lora_output_config


def apply_lora_updates(
    model: str | os.PathLike,
    dlc_path: str | os.PathLike,
    lora_importer_config: str | os.PathLike | List[UseCaseOutputConfig],
    input_list: Optional[os.PathLike],
    output_dir: str | os.PathLike,
) -> List[UseCaseOutputConfig]:
    """
    Applies LoRA (Low-Rank Adaptation) updates to a QAIRT model using the qairt-lora-importer tool.
    This function prepares and runs the LoRA importer pipeline, which updates lora safetensors
        and encodings as per QNN graph
    Args:
        model (str | os.PathLike): Path to the base model directory or file.
        dlc_path (str | os.PathLike): Path to the input DLC file containing the base model.
        lora_importer_config (str | os.PathLike | List[UseCaseOutputConfig]): Path of configuration for LoRA Importer's apply_lora_updates
        input_list (str | os.PathLike): Path to the YAML file specifying use-cases for conversion and quantization.
        output_dir (str | os.PathLike): Directory where the updated model artifacts will be saved.
    Returns:
        List[UseCaseOutputConfig]: Parsed use case configurations from the output YAML.
    Raises:
        RuntimeError: If the LoRA importer process fails.
    """

    try:
        _logger.debug("Running LoRA importer.")

        importer_args = _get_lora_importer_namespace_args(
            model=model,
            dlc_path=dlc_path,
            lora_config=lora_importer_config,
            input_list=input_list,
            output_dir=output_dir,
        )

        module_apply_lora_updates(importer_args)
        _relativize_lora_output_paths(output_dir)
        _logger.debug("lora_importer ran successfully")

        # Parse output YAML
        yaml_path = os.path.join(output_dir, "lora_output_files.yaml")

        return load_use_case_config(yaml_path)
    except Exception as e:
        raise RuntimeError(f"Apply LoRA updates failed: {e}")
