# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
LoRA Snooper module for Phase 2 - LoRA-aware OneShot snooping.

This module provides LoRA-specific snooping logic for the accuracy debugger,
enabling comparison of LoRA-adapted models between framework (ONNX runtime)
and target device (QNN inference engine) for each use case.
"""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from qti.aisw.tools.core.utilities.framework.framework_manager import FrameworkManager
from qti.aisw.tools.core.utilities.framework.utils.helper import Helper


class LoRASnooper:
    """Handles LoRA-aware snooping logic.

    This class manages the LoRA configuration, generates framework reference
    outputs for each use case, and maps inference engine outputs to use cases
    for comparison.

    The LoRA config structure:
        adapter:
          - name: <adapter_name>
            lora_config: <path_to_adapter_config>
        use-case:
          - name: base
            adapter_names: []
            model_name: <path_to_base_onnx>
          - name: <use_case_name>
            adapter_names: [<adapter_name>]
            model_name: <path_to_adapted_onnx>

    Inference engine output naming convention (LoRA mode):
        - "base": outputs for the base model (no adapters)
        - "lora_base_{use_case_name}": outputs for use case with adapters
    """

    def __init__(self, lora_config_path: Path, logger: logging.Logger):
        """Initialize LoRA Snooper.

        Args:
            lora_config_path: Path to the LoRA configuration YAML file
            logger: Python logger instance
        """
        self.lora_config_path = Path(lora_config_path)
        self.logger = logger
        self.lora_config = self._load_lora_config()

    def _load_lora_config(self) -> Dict[str, Any]:
        """Load LoRA configuration from YAML file.

        Returns:
            Dictionary containing the LoRA configuration.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file is not valid YAML.
        """
        try:
            with open(self.lora_config_path, "r") as f:
                config = yaml.safe_load(f)
            self.logger.debug(f"Loaded LoRA config from {self.lora_config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"LoRA config file not found: {self.lora_config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse LoRA config YAML: {e}")
            raise

    def get_use_cases(self) -> List[Dict[str, Any]]:
        """Get list of all use cases from LoRA config.

        Returns:
            List of use case dictionaries.
        """
        return self.lora_config.get("use-case", [])

    def get_base_use_case(self) -> Optional[Dict[str, Any]]:
        """Get the base use case (use case with no adapters).

        Returns:
            Base use case dictionary, or None if not found.
        """
        for uc in self.get_use_cases():
            if not uc.get("adapter_names"):
                return uc
        return None

    def get_non_base_use_cases(self) -> List[Dict[str, Any]]:
        """Get non-base use cases (use cases with adapters).

        Returns:
            List of non-base use case dictionaries.
        """
        return [uc for uc in self.get_use_cases() if uc.get("adapter_names")]

    def get_non_base_use_case_names(self) -> List[str]:
        """Get names of non-base use cases.

        Returns:
            List of non-base use case names.
        """
        return [uc["name"] for uc in self.get_non_base_use_cases()]

    def get_use_case_model_path(self, use_case_name: str) -> Optional[Path]:
        """Get the ONNX model path for a specific use case.

        Args:
            use_case_name: Name of the use case.

        Returns:
            Path to the ONNX model, or None if not found.
        """
        for uc in self.get_use_cases():
            if uc["name"] == use_case_name:
                return Path(uc["model_name"])
        return None

    def generate_framework_reference_outputs(
        self,
        use_case_name: str,
        input_sample: Dict[str, NDArray],
        debug_subgraph: Optional[List[str]] = None,
        dump_output_tensors: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, NDArray]:
        """Generate framework reference outputs for a specific use case.

        Runs the use-case specific ONNX model on ONNX runtime to generate
        reference outputs for comparison with inference engine outputs.

        Args:
            use_case_name: Name of the use case.
            input_sample: Input tensors for model execution.
            debug_subgraph: List of intermediate output tensors to collect.
                If None, all intermediate outputs are collected.
            dump_output_tensors: Whether to dump output tensors to files.
            output_dir: Directory to save output tensors (required if
                dump_output_tensors is True).

        Returns:
            Dictionary mapping tensor names to numpy arrays.

        Raises:
            ValueError: If the use case is not found in the LoRA config.
            FileNotFoundError: If the use case model file does not exist.
        """
        model_path = self.get_use_case_model_path(use_case_name)
        if model_path is None:
            raise ValueError(f"Use case '{use_case_name}' not found in LoRA config")

        if not model_path.exists():
            raise FileNotFoundError(f"Use case model not found: {model_path}")

        self.logger.info(
            f"Generating framework reference outputs for use case '{use_case_name}' "
            f"using model: {model_path}"
        )

        framework_manager = FrameworkManager(self.logger)
        framework_model = framework_manager.load(model_path)
        infer_shape = len(framework_model.graph.value_info) == 0

        reference_outputs = framework_manager.generate_intermediate_outputs(
            input_model=framework_model,
            input_data=input_sample,
            infer_shape=infer_shape,
            intermediate_output_tensors=debug_subgraph,
        )

        if dump_output_tensors and output_dir:
            use_case_output_dir = output_dir / f"reference_output_{use_case_name}"
            Helper.save_output_to_file(reference_outputs, use_case_output_dir)
            self.logger.info(
                f"Reference outputs for use case '{use_case_name}' saved to {use_case_output_dir}"
            )

        self.logger.info(
            f"Generated {len(reference_outputs)} reference outputs for use case '{use_case_name}'"
        )
        return reference_outputs

    def map_inference_output_to_use_case(
        self,
        inference_outputs: Dict[str, List[Dict[str, NDArray]]],
        use_case_name: str,
    ) -> Optional[Dict[str, NDArray]]:
        """Map inference engine output to a specific use case.

        The inference engine in LoRA mode returns outputs organized by model name:
        - "base": outputs for the base model (no adapters)
        - "lora_base_{use_case_name}": outputs for use cases with adapters

        Args:
            inference_outputs: Dict mapping model names to list of output dicts.
                Format: {model_name: [list of output dicts]}
            use_case_name: Name of the use case to find.

        Returns:
            Dictionary of tensor name to numpy array for the use case,
            or None if not found.
        """
        base_uc = self.get_base_use_case()
        is_base = base_uc and use_case_name == base_uc["name"]

        if is_base:
            # Base use case - look for "base" key
            if "base" in inference_outputs:
                outputs = inference_outputs["base"]
                return outputs[0] if outputs else None

            # Fallback: look for key that doesn't contain any non-base use case name
            non_base_names = self.get_non_base_use_case_names()
            for key in inference_outputs:
                if not any(name in key for name in non_base_names):
                    outputs = inference_outputs[key]
                    return outputs[0] if outputs else None
        else:
            # Non-base use case - look for key containing use case name
            # Primary pattern: "lora_base_{use_case_name}"
            expected_key = f"lora_base_{use_case_name}"
            if expected_key in inference_outputs:
                outputs = inference_outputs[expected_key]
                return outputs[0] if outputs else None

            # Fallback: look for any key containing the use case name
            for key in inference_outputs:
                if use_case_name in key:
                    outputs = inference_outputs[key]
                    return outputs[0] if outputs else None

        self.logger.warning(
            f"Could not find inference output for use case '{use_case_name}'. "
            f"Available keys: {list(inference_outputs.keys())}"
        )
        return None

    def validate_lora_config(self) -> None:
        """Validate the LoRA configuration structure.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if "adapter" not in self.lora_config:
            raise ValueError("LoRA config missing required 'adapter' field")

        if "use-case" not in self.lora_config:
            raise ValueError("LoRA config missing required 'use-case' field")

        use_cases = self.get_use_cases()
        if not use_cases:
            raise ValueError("LoRA config must have at least one use case")

        # Validate each use case has a model_name
        for uc in use_cases:
            if "name" not in uc:
                raise ValueError("Each use case must have a 'name' field")
            if "model_name" not in uc:
                raise ValueError(
                    f"Use case '{uc['name']}' missing required 'model_name' field"
                )
            model_path = Path(uc["model_name"])
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Use case '{uc['name']}' model not found: {model_path}"
                )

        self.logger.debug("LoRA config validation passed")
