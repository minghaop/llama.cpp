# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import onnx  # noqa
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from qti.aisw.accuracy_debugger.snooping.factory import get_snooper_class
from qti.aisw.accuracy_debugger.model_snooper.config import (
    ModelSnooperInputConfig,
    ModelSnooperOutputConfig,
)
from qti.aisw.tools.core.modules.api import (
    Module,
    ModuleSchema,
    ModuleSchemaVersion,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class ModelSnooperSchemaV1(ModuleSchema):
    """Schema for accuracy debugger module."""

    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["ModelSnooperModule"] = "ModelSnooperModule"
    path: Path = Path(__file__)
    arguments: ModelSnooperInputConfig
    outputs: Optional[ModelSnooperOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


class ModelSnooper(Module):
    """User interface class for accuracy debugger API."""

    _SCHEMA = ModelSnooperSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger: logging.Logger = None) -> None:
        """Initialize Debugger module

        Args:
            logger: A logger instance to be used by the ModelSnooper module
        """
        if logger is None:
            log_area = LogAreas.register_log_area("Accuracy Debugger")
            self.logger = QAIRTLogger.register_area_logger(area=log_area, level=logging.INFO)
        else:
            self.logger = logger
        super().__init__(logger)

    def run(self, config: ModelSnooperInputConfig) -> ModelSnooperOutputConfig:
        """Run the accuracy debugger module
        Args:
             config: Accuracy debugger input configuration.

        Returns:
             ModelSnooperOutputConfig: Accuracy debugger output configuration
             with the snooping report.
        """
        snooping_cls = get_snooper_class(config.algorithm)
        snooping_obj = snooping_cls(logger=self.logger)

        csv_snooping_report, json_snooping_report = snooping_obj.run(config)
        output_config = ModelSnooperOutputConfig(
            csv_snooping_report=csv_snooping_report, json_snooping_report=json_snooping_report
        )
        return output_config

    def get_logger(self) -> Any:
        """This should return an instance of the logger that is used.

        Returns:
            The logger used by the module.
        """
        return self._logger

    def properties(self) -> Dict[str, Any]:
        return self._schema.model_json_schema()

    @property
    def _schema(self):
        return self._SCHEMA

    def enable_debug(self, debug_level: int, **kwargs) -> None:
        """Enable debugging behaviour for the module

        Args:
            debug_level: The level of debugging to be enabled.
        """
        pass
