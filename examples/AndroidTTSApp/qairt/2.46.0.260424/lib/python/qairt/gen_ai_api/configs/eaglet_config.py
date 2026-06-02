# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from pathlib import Path
from typing import List, Optional

from pydantic import FilePath, field_validator

from qairt.api.configs.common import AISWBaseModel
from qairt.modules.genie_execution.genie_config import EagletConfig


class EagletRunConfig(EagletConfig):
    """Configuration options to run the eaglet model"""

    draft_container: Optional[Path] = None
    draft_token_map: Optional[Path] = None


class EagletBuilderConfig(EagletConfig):
    """
    Configuration for building an Eaglet model.

    The builder creates the model in ``draft_model_path`` The resulting container
    path is later set on an :class:`EagletRunConfig` instance for execution.
    """

    draft_model_path: Path
    draft_token_map: Optional[FilePath] = None
    draft_model_arn: List[int] = [8, 128]

    @field_validator("draft_model_path")
    def validate_draft_model_path(cls, v):
        if not v.exists():
            raise ValueError(f"draft_model_path must exist: {v}")
        return v

    @field_validator("draft_token_map")
    def validate_draft_token_map(cls, v):
        if v is not None and v.suffix.lower() != ".json":
            raise ValueError(f"draft_token_map must be a JSON file: {v}")
        return v
