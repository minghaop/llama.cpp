# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from pathlib import Path

from pydantic import field_validator

from qairt.modules.genie_execution.genie_config import SsdConfig
from qti.aisw.tools.core.modules.api.definitions.common import AISWBaseModel


class SsdBuilderConfig(SsdConfig):
    """Configuration for SSD inference."""

    ssd_tensor_file: str

    @field_validator("ssd_tensor_file")
    def check_file_extension(cls, v: str) -> str:
        path = Path(v)
        if path.suffix != ".pt":
            raise ValueError("ssd_tensor_file must be a .pt file")
        return v
