# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""
ModelTransformerConfig class.
"""

from dataclasses import dataclass, field
from enum import Enum
from types import NoneType
from typing import Any, List, Optional, Type, Union

from qairt.optimizer.onnx.passes.config import M2sStartPoint


class QuantizationStage(Enum):
    """
    Represents the quantization stage for running the transformations.

    Attributes:
        PRE_QUANT (str): Indicates the stage before quantization.
        POST_QUANT (str): Indicates the stage after quantization.
    """

    PRE_QUANT = "PRE_QUANT"
    POST_QUANT = "POST_QUANT"


@dataclass
class SplitModelConfig:
    """
    Configuration for splitting a model into multiple subgraphs.
    """

    num_splits: int = 1
    """
    Number of splits we want to divide the model up into.
    """
    split_embedding: bool = False
    """
    Split embedding into its own subgraph/model.
    """
    split_lm_head: bool = False
    """
    Split LM head into its own subgraph/model.
    """
    skip_verification: bool = False
    """
    Skip ONNXRT verification of comparing full model outputs to splits outputs
    """
    log_level: str = "info"
    """
    Level/Severity of events to log.
    """


@dataclass
class ARn_ContextLengthConfig:
    context_length: List[int] = field(default_factory=lambda: [4096])
    auto_regression_number: List[int] = field(default_factory=lambda: [1, 128])
    """Auto-regression numbers to generate."""
    arn_model_paths: dict[int, str] = field(default_factory=dict)
    """
    Optional per-AR model path overrides. If a path is provided for a given AR value,
    that model is used as the source for AR/CL conversion instead of the base framework model.
    Both models must share the same architecture.
    """
    arn_encodings_paths: dict[int, str] = field(default_factory=dict)
    """
    Optional per-AR encodings path overrides. If a path is provided for a given AR value,
    that encodings file is used during transform instead of the base encodings path.
    """


@dataclass
class MhaConfig:
    """
    Collection of configuration options used (exclusively) for the alternate (version 2) MHA to SHA conversion algorithm.
    If this configuration is not present, the legacy (v1) algorithm will be used.
    """

    extract_lorav2_alpha: bool = False
    permute_kv_cache_io: bool = False
    key_cache_name_pattern: str = r"past_key_(\d)+_in|past_key_(\d)+_out"
    value_cache_name_pattern: str = r"past_value_(\d)+_in|past_value_(\d)+_out"
    m2s_head_split_map: dict[int, int] | None = None
    m2s_additional_start_points: list[M2sStartPoint] | None = None
    enable_validation: bool = False
    validation_kwargs: dict | None = None  # Optional dictionary of arguments for validation.
    #    It can contain:
    #      - input_raw_list_path: str - Path to a file listing input files.
    #      - input_raw_base_dir: str - Base directory for input files.


# TODO: Change this name to transforms and allow a user to pass in the options when they
# Do not add any new transforms as dataclasses because it creates two sources of truth. Only use a config
# if the API function contains a config, otherwise use a dictionary.
@dataclass
class ModelTransformerConfig:
    """
    Parent configuration for all transformation settings.

    Attributes:
        arn_cl_options (ARN_CL_Options): Configuration for ARN (Auto Regression) and Context Length (CL)
        split_model (SplitModel): Configuration for splitting a model into multiple subgraphs.
        mha_config (Optional[MhaConfig]): Configuration for MHA2SHA v2 conversion.
        adapt_moe (Optional[dict]): Configuration for MoE transformations.
            See :function:`qairt.optimizer.onnx.api.adapt_moe` for valid options.
    """

    arn_cl_options: ARn_ContextLengthConfig = field(default_factory=ARn_ContextLengthConfig)
    split_model: SplitModelConfig = field(default_factory=SplitModelConfig)
    mha_config: MhaConfig = field(default_factory=MhaConfig)
    adapt_moe: Optional[dict] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ModelTransformerConfig":
        # Mapping of transformation names to their corresponding configuration classes
        transformation_config_classes: dict[str, Type] = {
            "arn_cl_options": ARn_ContextLengthConfig,
            "split_model": SplitModelConfig,
            "mha_config": MhaConfig,
            "adapt_moe": dict,
        }

        # Initialize TransformConfig dynamically
        transform_config_kwargs = {}
        for key, config_class in transformation_config_classes.items():
            if key in config_dict:
                item_value = config_dict[key]  # Get the value once

                # Check if item_value is already an instance of the expected class
                if isinstance(item_value, config_class):
                    # It's already the correct type, just assign it directly
                    transform_config_kwargs[key] = item_value
                elif isinstance(item_value, dict):
                    # It's a dictionary, so instantiate the class with it
                    transform_config_kwargs[key] = config_class(**item_value)
                elif isinstance(item_value, NoneType):
                    transform_config_kwargs[key] = None
                else:
                    # Handle unexpected types if necessary, e.g., raise an error or log
                    raise TypeError(
                        f"Expected '{key}' to be a dictionary or an instance of "
                        f"'{config_class.__name__}', but got '{type(item_value).__name__}'."
                    )
        return cls(**transform_config_kwargs)
