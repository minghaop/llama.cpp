# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

from qti.aisw.accuracy_debugger.utils.exceptions import (
    LoRAModelCreatorFailure,
    LoRAImporterFailure,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class LoRAPipelineManager:
    """Manages LoRA pipeline execution within accuracy debugger."""

    def __init__(self, working_directory: Path, logger: Any = None):
        """Initialize LoRA Pipeline Manager
        Args:
            working_directory (Path): Working directory for LoRA operations
            logger (Any): Desired python logger
        """
        self.working_dir = working_directory

        if logger:
            self.logger = logger
        else:
            self.log_area = LogAreas.register_log_area("LoRA_Pipeline")
            self.logger = QAIRTLogger.register_area_logger(area=self.log_area, level="INFO")

        self.lora_tensor_names_file = None

    def run_lora_model_creator(
        self, config: "LoRAModelCreatorInputConfig"
    ) -> "LoRAModelCreatorOutputConfig":
        """Execute LoRA Model Creator using standardized config objects."""
        try:
            from qti.aisw.lora.lora_model_creator_app import LoraModelCreatorApp
            from qti.aisw.accuracy_debugger.lora import LoRAModelCreatorOutputConfig

            # Prepare output directory
            output_dir = config.output_dir or (self.working_dir / "lora_model_creator_output")
            output_dir.mkdir(exist_ok=True)

            # Extract arguments using keyword arguments matching SDK tool constructor
            creator_args = {
                "config_path": str(config.lora_config),
                "output_dir": str(output_dir),
                "skip_validation": config.skip_validation,
                "quant_updatable_mode": config.quant_updatable_mode,
                "transforms_metadata": str(config.transforms_metadata)
                if config.transforms_metadata
                else None,
                "dump_onnx": config.dump_usecase_onnx,
            }

            # Configure LoRA Model Creator with explicit keyword arguments
            creator_app = LoraModelCreatorApp(**creator_args)

            # Execute model creator
            self.logger.info("Running LoRA Model Creator...")
            creator_app.run()

            # Return standardized output config
            return LoRAModelCreatorOutputConfig(
                base_model_path=output_dir / "base_model.onnx",
                base_encodings_path=output_dir / "base_encodings.json",
                lora_tensor_names_file=output_dir / "lora_tensor_names.txt",
                importer_config=output_dir / "lora_importer_config.yaml",
                output_directory=output_dir,
            )

        except ImportError as e:
            self.logger.error(f"LoRA Model Creator not available in SDK: {e}")
            raise LoRAModelCreatorFailure(
                "LoRA Model Creator utility not found. Please ensure QAIRT SDK is properly installed.",
                e,
            )
        except Exception as e:
            self.logger.error(f"LoRA Model Creator execution failed: {e}")
            raise LoRAModelCreatorFailure(f"Failed to execute LoRA Model Creator! {e}", e)

    def run_lora_importer(self, config: "LoRAImporterInputConfig") -> "LoRAImporterOutputConfig":
        """Execute LoRA Importer using standardized config objects."""
        try:
            from qti.aisw.lora.lora_importer_app import apply_lora_updates
            from qti.aisw.accuracy_debugger.lora import LoRAImporterOutputConfig, UseCaseArtifacts

            # Prepare output directory
            output_dir = config.output_dir or (self.working_dir / "lora_importer_output")
            output_dir.mkdir(exist_ok=True)

            # Create a namespace object similar to argparse.Namespace with all SDK tool arguments
            importer_args = type(
                "Args",
                (),
                {
                    "lora_config": str(config.lora_config),
                    "input_dlc": str(config.input_dlc),
                    "output_dir": str(output_dir),
                    "input_network": str(config.input_network) if config.input_network else None,
                    "input_list": str(config.input_list) if config.input_list else None,
                    "float_fallback": config.float_fallback,
                    "debug": config.debug if config.debug != -1 else None,
                    "skip_validation": config.skip_validation,
                    "skip_apply_graph_transforms": config.skip_apply_graph_transforms,
                    "dump_usecase_onnx": config.dump_usecase_onnx,
                    "dump_usecase_dlc": config.dump_usecase_dlc,
                },
            )()

            # Execute LoRA Importer
            self.logger.info("Running LoRA Importer...")
            apply_lora_updates(importer_args)

            # Load output configuration
            output_config_path = output_dir / "lora_output_files.yaml"
            with open(output_config_path, "r") as f:
                output_config = yaml.safe_load(f)

            # Parse use case artifacts into standardized format
            use_case_artifacts = {}
            for use_case in output_config.get("use_case", []):
                use_case_name = use_case["name"]
                use_case_artifacts[use_case_name] = UseCaseArtifacts(
                    name=use_case_name,
                    graph_name=use_case["graph"],
                    weights=Path(use_case["weights"]),
                    encodings=Path(use_case.get("encodings"))
                    if use_case.get("encodings")
                    else None,
                )

            # Return standardized output config
            return LoRAImporterOutputConfig(
                lora_output_files=output_config_path,
                use_case_artifacts=use_case_artifacts,
                output_directory=output_dir,
            )

        except ImportError as e:
            self.logger.error(f"LoRA Importer not available in SDK: {e}")
            raise LoRAImporterFailure(
                "LoRA Importer utility not found. Please ensure QAIRT SDK is properly installed.", e
            )
        except Exception as e:
            self.logger.error(f"LoRA Importer execution failed: {e}")
            raise LoRAImporterFailure(f"Failed to execute LoRA Importer! {e}", e)
