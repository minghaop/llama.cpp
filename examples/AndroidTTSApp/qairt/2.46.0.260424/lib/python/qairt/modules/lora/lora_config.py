# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import json
import os
from os import PathLike
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Union

import yaml  # type: ignore
from pydantic import Field, field_validator, model_validator

from qairt.api.configs.common import AISWBaseModel


class AdapterParamsConfig(AISWBaseModel):
    """
    Configuration for individual adapter parameters.
    """

    name: str
    """
    Name of the adapter.
    """

    rank: int
    """
    Rank used in the adapter configuration.
    """

    alpha: int
    """
    Alpha value for scaling.
    """

    target_modules: List[str]
    """
    List of source framework module names where the adapter is applied.
    """


class AdapterConfig(AISWBaseModel):
    """
    Configuration for an adapter, including its name and associated LoRA configurations.
    """

    name: str
    """
    Name of the adapter.
    """

    adapter_lora_config: Annotated[
        List[Union[str, PathLike, AdapterParamsConfig]], Field(alias="lora_config")
    ]
    """
    List of LoRA configuration paths or objects.
    """

    @field_validator("adapter_lora_config", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """
        Ensures that the adapter_lora_config is always a list.

        Args:
            v: The input value to validate.

        Returns:
            List: A list containing the input value if it was not already a list.
        """
        if isinstance(v, (str, PathLike, AdapterParamsConfig)):
            return [v]
        return v


class UseCaseInputConfig(AISWBaseModel):
    """
    Configuration for a specific use case of the model.
    """

    name: str
    """
    Name of the use case.
    """

    adapter_names: List[str]
    """
    List of adapter names used in this use case.
    """

    model: Union[str, PathLike] = Field(alias="model_name")
    """
    Path or name of the model (aliased as 'model_name').
    """

    adapter_alphas: List[float]
    """
    List of alpha values for each adapter.
    """

    encodings: Union[str, PathLike] = Field(alias="quant_overrides")
    """
    Path to quantization overrides (aliased as 'quant_overrides').
    """

    quant_updatable_tensors: Optional[Union[str, PathLike]] = None
    """
    Path to quant-updatable tensors file.
    """


class LoraConfig(AISWBaseModel):
    """
    Top-level configuration for LoRA, including adapters and use cases.
    """

    adapter: List[AdapterConfig]
    """
    List of adapter configurations.
    """

    attach_point_onnx_mapping: Union[str, PathLike]
    """
    Path to ONNX mapping file.
    """

    use_cases: List[UseCaseInputConfig] = Field(alias="use-case")
    """
    List of use case configurations (aliased as 'use-case').
    """


class LoraBuilderInputConfig(AISWBaseModel):
    """
    Input configuration for LoRA, allowing either a path or an object.
    """

    lora_config_path: Optional[Union[str, PathLike]] = None
    """
    Path to the LoRA configuration file.
    """

    lora_config_obj: Optional[LoraConfig] = None
    """
    LoRA configuration object.
    """

    create_lora_graph: bool = True
    """
    Whether to create LoRA max rank-concatenated graph
    """

    quant_updatable_mode: Literal["none", "adapter_only", "all"] = "adapter_only"
    """
    Mode for quant-updatable tensors.
    """

    alpha_tensor_name: str
    """
    Name of the tensor where LoRA adapter is being applied.
    """

    @model_validator(mode="after")
    def check_exclusive_inputs(cls, values):
        """
        Validates that only one of lora_config_path or lora_config_obj is provided.

        Args:
            values (dict): Dictionary of field values.

        Raises:
            ValueError: If both or neither of the fields are provided.

        Returns:
            dict: Validated field values.
        """
        if values.lora_config_path and values.lora_config_obj:
            raise ValueError("Provide either `lora_config_path` or `lora_config_obj`, not both.")
        if not values.lora_config_path and not values.lora_config_obj:
            raise ValueError("You must provide either `lora_config_path` or `lora_config_obj`.")
        return values


class UseCaseOutputConfig(AISWBaseModel):
    """
    Configuration for the output of a specific use case after LoRA processing.
    """

    name: str
    """
    Name of the use case.
    """

    model: Optional[Union[str, PathLike]] = Field(default=None, alias="model_name")
    """
    Path or name of the model (mapped to 'model_name' when serialized).
    """

    graph: Optional[str] = ""
    """
    Name of the graph for the use case
    """

    lora_weights: Union[str, PathLike] = Field(alias="weights")
    """
    Path to the LoRA weights file (in safetensors format).
    """

    encodings: Optional[Union[str, PathLike]] = Field(default=None, alias="quant_overrides")
    """
    Path to quantization overrides (mapped to 'quant_overrides' when serialized).
    """

    output_path: Optional[Union[str, os.PathLike]] = None
    """
    Path where the importer output should be saved.
    """


class LoraBuilderOutputConfig(AISWBaseModel):
    """
    Defines the output configuration from the LoRA `build_lora_graph` process.

    This configuration can be serialized into a YAML file
    and passed to subsequent steps in the pipeline.
    """

    # Use case needs to be serialized to lora_importer_config.yaml
    use_case: List[UseCaseOutputConfig]
    """
    A list of use case configurations that describe how the LoRA model will be used.
    This is serialized to `lora_importer_config.yaml`.
    """

    lora_tensor_names: Union[str, PathLike]
    """
    Path or string reference to the tensor names used in the LoRA model.
    """

    base_model_artifacts: Dict[str, Union[str, PathLike]]
    """
    Dictionary containing paths to base model artifacts like ONNX, encodings and data files.
    """


class AdapterRunConfig(AISWBaseModel):
    """
    Defines the configuration parameters for executing a LoRA (Low-Rank Adaptation) model.

    This configuration is used to control how the LoRA adapter is applied during model inference.
    """

    adapter_name: str
    """
    The name or identifier of the LoRA adapter to be used during execution.
    """

    alpha: float = 1.0
    """
    A scaling factor applied to the LoRA weights.
    """


class UseCaseRunConfig(AISWBaseModel):
    """
    Defines the configuration for a specific use case involving one or more LoRA adapters.
    """

    use_case_name: str
    """
     A unique identifier for the use case, representing a single adapter or a group of adapters.
    """

    adapters: List[AdapterRunConfig]
    """
    A list of LoRA adapter configurations to be used in this use case.
        Each adapter is defined by its own `AdapterRunConfig`, specifying parameters
        such as adapter name and scaling factor.
    """


def _resolve_path(path: str | os.PathLike | None, base_dir: str) -> str:
    if path is None:
        raise ValueError("Path cannot be None")
    return str(Path(path).resolve()) if Path(path).is_absolute() else str(Path(base_dir, path).resolve())


def serialize_lora_input_config(lora_config: LoraConfig, base_directory: Union[str, PathLike]) -> str:
    """
    Serializes a LoraConfig object into a YAML file and saves adapter parameter configs as JSON.

    Args:
        lora_config (LoraConfig): The configuration object to serialize.
        base_directory (Union[str, PathLike]): Directory where the files will be saved.

    Returns:
        str: Path to the generated YAML configuration file.
    """

    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    yaml_dict: dict = {
        "adapter": [],
        "attach_point_onnx_mapping": lora_config.attach_point_onnx_mapping,
        "use-case": [],
    }

    for adapter in lora_config.adapter:
        lora_config_paths = []
        for config in adapter.adapter_lora_config:
            if isinstance(config, AdapterParamsConfig):
                config_path = os.path.join(base_directory, f"{config.name}.json")
                with open(config_path, "w") as f:
                    json.dump(config.model_dump(), f)
                lora_config_paths.append(config_path)
            else:
                lora_config_paths.append(str(config))

        yaml_dict["adapter"].append(
            {
                "name": adapter.name,
                "lora_config": lora_config_paths if len(lora_config_paths) > 1 else lora_config_paths[0],
            }
        )

    for use_case in lora_config.use_cases:
        yaml_dict["use-case"].append(
            {
                "name": use_case.name,
                "adapter_names": use_case.adapter_names,
                "model_name": use_case.model,
                "adapter_alphas": use_case.adapter_alphas,
                "quant_overrides": use_case.encodings,
                "quant_updatable_tensors": use_case.quant_updatable_tensors or f"{base_directory}/null",
            }
        )

    yaml_path = os.path.join(base_directory, "lora_config.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f)

    return yaml_path


def serialize_lora_importer_config(
    lora_uc_output_config: List[UseCaseOutputConfig], yaml_path: str, base_dir: str
) -> None:
    """
    Serializes fields from List[UseCaseOutputConfig] to a YAML file.

    Only the fields 'name', 'model_name', 'weights', 'quant_overrides', and 'output_path'
    are serialized for each use case. Relative paths are resolved using the provided base_dir.

    Args:
        lora_uc_output_config (List[UseCaseOutputConfig]): The configuration object containing use case data.
        yaml_path (str): The file path where the YAML output should be saved.
        base_dir (str): The base directory to resolve relative paths.

    Returns:
        None
    """
    use_case_dicts = []

    for uc in lora_uc_output_config:
        if not uc.model:
            raise ValueError(f"Missing 'model_name' for use case '{uc.name}'")

        uc_dict = {
            "name": uc.name,
            "model_name": _resolve_path(uc.model, base_dir),
            "lora_weights": _resolve_path(uc.lora_weights, base_dir),
            "quant_overrides": _resolve_path(uc.encodings, base_dir) if uc.encodings else None,
            "output_path": (
                _resolve_path(uc.output_path, base_dir)
                if uc.output_path and not os.path.isabs(str(uc.output_path))
                else str(uc.output_path)
                if uc.output_path
                else None
            ),
        }
        use_case_dicts.append(uc_dict)

    with open(yaml_path, "w") as f:
        yaml.dump({"use_case": use_case_dicts}, f, sort_keys=False)


def serialize_lora_adapter_weight_config(
    use_cases: List[UseCaseOutputConfig], yaml_path: str, base_dir: str
) -> None:
    """
    Serializes selected fields from UseCaseOutputConfig objects to a YAML file for compile API.

    Only the fields 'name', 'graph', 'lora_weights' (as 'weights'), and 'encodings' are serialized.
    Relative paths are resolved using the provided base_dir.

    Args:
        use_cases (List[UseCaseOutputConfig]): The configuration object containing use case data.
        yaml_path (str): The file path where the YAML output should be saved.
        base_dir (str): The base directory to resolve relative paths.

    Returns:
        None
    """
    use_case_dicts = []

    for uc in use_cases:
        uc_dict = {
            "name": uc.name,
            "graph": uc.graph,
            "weights": _resolve_path(uc.lora_weights, base_dir),
            "encodings": _resolve_path(uc.encodings, base_dir),
        }
        use_case_dicts.append(uc_dict)

    with open(yaml_path, "w") as f:
        yaml.dump({"use_case": use_case_dicts}, f, sort_keys=False)


def load_use_case_config(yaml_path: Union[str, Path]) -> List[UseCaseOutputConfig]:
    """
    Loads use case configuration from a specified YAML file.

    Args:
        yaml_path (Union[str, Path]): The path to the YAML configuration file.

    Returns:
        List[UseCaseOutputConfig]: A list of use case configuration objects parsed from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file is not found at the specified path.
        ValueError: If the YAML content is invalid or missing the 'use_case' key.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_path} not found.")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if "use_case" not in data:
        raise ValueError(f"'use_case' key not found in {yaml_path}")

    return [UseCaseOutputConfig(**uc) for uc in data["use_case"]]


def get_adapter_count_by_use_case(lora_config: LoraBuilderInputConfig) -> Dict[str, int]:
    """
    Constructs a dictionary mapping each use case to the count of LoRA adapters it contains.

    Args:
        lora_config (LoraBuilderInputConfig): The configuration object containing LoRA adapter information.

    Returns:
        Dict[str, int]: A dictionary where keys are use case names and values are the count of LoRA adapters in each use case.
    """
    adapter_count_by_use_case = {}

    if lora_config.lora_config_path:
        with open(lora_config.lora_config_path, "r") as f:
            data = yaml.safe_load(f)
            use_cases = data.get("use-case", [])
    else:
        if lora_config.lora_config_obj is not None:
            use_cases = [use_case.model_dump() for use_case in lora_config.lora_config_obj.use_cases]
        else:
            raise ValueError("lora_config_obj is None")

    for uc in use_cases:
        adapter_count_by_use_case[uc["name"]] = len(uc["adapter_names"])

    return adapter_count_by_use_case
