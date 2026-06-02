# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""
Rule configuration loader for JSON-based validation rules.

This module provides functionality to load validation rules from JSON configuration files,
allowing users to extend or suppress rules without modifying code.
"""

import inspect
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from qti.aisw.accuracy_debugger.encodings.encodings import ModelEncoding, TensorEncoding
from qti.aisw.accuracy_debugger.encodings.encodings_utils import TensorType
from qti.aisw.accuracy_debugger.utils.file_utils import read_json
from qti.aisw.accuracy_debugger.validate_encodings import validation_rules, context_aware_rules


class RuleConfigLoader:
    """Loader for JSON-based validation rule configuration."""

    def __init__(self, logger=None):
        """Initialize the rule config loader.

        Args:
            logger: Optional logger instance.
        """
        self._logger = logger
        self._builtin_rules = self._get_builtin_rules()
        self._builtin_context_rules = self._get_builtin_context_rules()

    def _get_builtin_rules(self) -> Dict[str, Callable]:
        """Get dictionary of built-in validation rules.

        Returns:
            Dict[str, Callable]: Dictionary mapping rule names to rule functions.
        """
        return {
            "conv_weights_symmetric": validation_rules.rule_conv_weights_symmetric,
            "rmsnorm_weights_16bit": validation_rules.rule_rmsnorm_weights_16bit,
            "rmsnorm_weights_asymmetric": validation_rules.rule_rmsnorm_weights_asymmetric,
            "layernorm_weights_asymmetric": validation_rules.rule_layernorm_weights_asymmetric,
            "batchnorm_weights_asymmetric": validation_rules.rule_batchnorm_weights_asymmetric,
            "layernorm_weights_16bit": validation_rules.rule_layernorm_weights_16bit,
            "all_weights_8bit": validation_rules.rule_all_weights_8bit,
            "linear_weights_per_channel": validation_rules.rule_linear_weights_per_channel,
        }

    def _get_builtin_context_rules(self) -> Dict[str, Callable]:
        """Get dictionary of built-in context-aware validation rules.

        Returns:
            Dict[str, Callable]: Dictionary mapping rule names to rule functions.
        """
        return {
            "reshape_encoding_matches_predecessor": context_aware_rules.rule_reshape_encoding_matches_predecessor,
            "transpose_encoding_matches_predecessor": context_aware_rules.rule_transpose_encoding_matches_predecessor,
            "concat_inputs_same_range": context_aware_rules.rule_concat_inputs_same_range,
        }

    def load_config(self, config_path: Path | str) -> Dict[str, Any]:
        """Load rule configuration from JSON file.

        Args:
            config_path: Path to the JSON configuration file.

        Returns:
            Dict[str, Any]: Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If config file does not exist.
            json.JSONDecodeError: If config file is not valid JSON.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config = read_json(config_path)

        if self._logger:
            self._logger.info(f"Loaded rule configuration from {config_path}")

        return config

    def create_rules_from_config(
        self, config: Dict[str, Any]
    ) -> Tuple[List[Callable], List[Callable]]:
        """Create validation rules from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Tuple[List[Callable], List[Callable]]: (validation_rules, context_aware_rules)
        """
        validation_rule_list = []
        context_rule_list = []

        # Process validation rules
        if "validation_rules" in config:
            for rule_config in config["validation_rules"]:
                if not rule_config.get("enabled", True):
                    if self._logger:
                        self._logger.debug(f"Skipping disabled rule: {rule_config.get('name')}")
                    continue

                rule = self._create_validation_rule_from_config(rule_config)
                if rule:
                    validation_rule_list.append(rule)

        # Process context-aware rules
        if "context_aware_rules" in config:
            for rule_config in config["context_aware_rules"]:
                if not rule_config.get("enabled", True):
                    if self._logger:
                        self._logger.debug(
                            f"Skipping disabled context rule: {rule_config.get('name')}"
                        )
                    continue

                rule = self._create_context_rule_from_config(rule_config)
                if rule:
                    context_rule_list.append(rule)

        if self._logger:
            self._logger.info(
                f"Created {len(validation_rule_list)} validation rules and "
                f"{len(context_rule_list)} context-aware rules from configuration"
            )

        return validation_rule_list, context_rule_list

    def _create_validation_rule_from_config(self, rule_config: Dict[str, Any]) -> Callable | None:
        """Create a validation rule from configuration.

        Args:
            rule_config: Rule configuration dictionary.

        Returns:
            Callable | None: Rule function or None if creation failed.
        """
        rule_type = rule_config.get("type", "builtin")

        if rule_type == "builtin":
            return self._create_builtin_rule(rule_config)
        elif rule_type == "custom":
            return self._create_custom_rule(rule_config)
        elif rule_type == "parametric":
            return self._create_parametric_rule(rule_config)
        else:
            if self._logger:
                self._logger.warning(f"Unknown rule type: {rule_type}")
            return None

    def _create_builtin_rule(self, rule_config: Dict[str, Any]) -> Callable | None:
        """Create a built-in validation rule.

        Args:
            rule_config: Rule configuration dictionary.

        Returns:
            Callable | None: Rule function or None if not found.
        """
        rule_name = rule_config.get("name")
        if rule_name not in self._builtin_rules:
            if self._logger:
                self._logger.warning(f"Built-in rule not found: {rule_name}")
            return None

        rule_func = self._builtin_rules[rule_name]

        # Sanity-check that the rule function actually accepts the
        # supplied parameters before wrapping it, to give a clear error instead
        # of a confusing TypeError at call time.
        params = rule_config.get("parameters", {})
        if params:
            sig = inspect.signature(rule_func)
            accepted = set(sig.parameters.keys()) - {"tensor_name", "tensor"}
            unsupported = set(params.keys()) - accepted
            if unsupported:
                if self._logger:
                    self._logger.warning(
                        f"Built-in rule '{rule_name}' does not accept parameter(s) "
                        f"{unsupported}. Supported extra parameters: {accepted}. "
                        "Rule will be used without parameters."
                    )
                return rule_func

            def rule_wrapper(tensor_name: str, tensor: TensorEncoding) -> Tuple[bool, str | None]:
                return rule_func(tensor_name, tensor, **params)

            return rule_wrapper

        return rule_func

    def _create_custom_rule(self, rule_config: Dict[str, Any]) -> Callable | None:
        """Create a custom validation rule from configuration.

        Args:
            rule_config: Rule configuration dictionary with custom rule definition.

        Returns:
            Callable | None: Rule function or None if creation failed.
        """
        description = rule_config.get("description", "Custom validation rule")
        conditions = rule_config.get("conditions", {})

        # Extract conditions
        tensor_pattern = conditions.get("tensor_name_pattern")
        tensor_type = conditions.get("tensor_type")  # "weight", "activation", "bias"
        layer_types = conditions.get("layer_types", [])  # ["conv", "linear", etc.]

        # Extract checks
        checks = rule_config.get("checks", {})
        bitwidth = checks.get("bitwidth")
        is_symm = checks.get("is_symm")
        min_channels = checks.get("min_channels")
        max_channels = checks.get("max_channels")
        scale_range = checks.get("scale_range")  # {"min": 0.001, "max": 0.1}
        offset_range = checks.get("offset_range")

        def custom_rule(tensor_name: str, tensor: TensorEncoding) -> Tuple[bool, str | None]:
            # Check if rule applies
            if tensor_pattern and not re.search(tensor_pattern, tensor_name, re.IGNORECASE):
                return True, None

            # Use tensor.tensor_type when it is unambiguous (V0/V1/DLC).
            # Only fall back to name-based string matching for V2 (AIMET 2.0.0)
            # encodings where every tensor carries TensorType.Encodings.
            if tensor_type:
                if tensor.tensor_type != TensorType.Encodings:
                    # Reliable: tensor_type was set explicitly during loading.
                    if tensor_type == "weight" and tensor.tensor_type != TensorType.ParamEncodings:
                        return True, None
                    if tensor_type == "bias" and tensor.tensor_type != TensorType.ParamEncodings:
                        return True, None
                    if tensor_type == "activation" and tensor.tensor_type != TensorType.ActivationEncodings:
                        return True, None
                else:
                    # V2 fallback: tensor_type is TensorType.Encodings for all tensors.
                    name_lower = tensor_name.lower()
                    if tensor_type == "weight" and not ("weight" in name_lower or "kernel" in name_lower):
                        return True, None
                    elif tensor_type == "bias" and "bias" not in name_lower:
                        return True, None
                    elif tensor_type == "activation" and any(
                        kw in name_lower for kw in ["weight", "bias", "kernel"]
                    ):
                        return True, None

            # For DLC-loaded encodings, op_type is stored directly on
            # the DlcTensorEncoding object, giving accurate layer-type matching.
            # For JSON-loaded encodings op_type is None, so fall back to
            # tensor-name string matching.
            if layer_types:
                op_type = getattr(tensor, "op_type", None)
                if op_type is not None:
                    # DLC path: match against the actual op type string.
                    if not any(lt.lower() in op_type.lower() for lt in layer_types):
                        return True, None
                else:
                    # JSON path: fall back to tensor-name heuristic.
                    name_lower = tensor_name.lower()
                    if not any(lt in name_lower for lt in layer_types):
                        return True, None

            # Perform checks
            violations = []

            if bitwidth is not None and tensor.bitwidth != bitwidth:
                violations.append(f"bitwidth must be {bitwidth} (got {tensor.bitwidth})")

            if is_symm is not None:
                expected_symm = "true" if is_symm else "false"
                actual_symm = tensor.is_symm.lower() if tensor.is_symm else "unknown"
                if actual_symm != expected_symm:
                    violations.append(f"is_symm must be {expected_symm} (got {actual_symm})")

            if min_channels is not None and tensor.channels < min_channels:
                violations.append(
                    f"channels must be >= {min_channels} (got {tensor.channels})"
                )

            if max_channels is not None and tensor.channels > max_channels:
                violations.append(
                    f"channels must be <= {max_channels} (got {tensor.channels})"
                )

            if scale_range and tensor.scale:
                min_scale = scale_range.get("min")
                max_scale = scale_range.get("max")
                for scale in tensor.scale:
                    if min_scale is not None and scale < min_scale:
                        violations.append(f"scale must be >= {min_scale} (got {scale})")
                    if max_scale is not None and scale > max_scale:
                        violations.append(f"scale must be <= {max_scale} (got {scale})")

            if offset_range and tensor.offset:
                min_offset = offset_range.get("min")
                max_offset = offset_range.get("max")
                for offset in tensor.offset:
                    if min_offset is not None and offset < min_offset:
                        violations.append(f"offset must be >= {min_offset} (got {offset})")
                    if max_offset is not None and offset > max_offset:
                        violations.append(f"offset must be <= {max_offset} (got {offset})")

            if violations:
                violation_msg = f"{description}: " + "; ".join(violations)
                return False, violation_msg

            return True, None

        return custom_rule

    def _create_parametric_rule(self, rule_config: Dict[str, Any]) -> Callable | None:
        """Create a parametric validation rule (e.g., large_quantization_range_warning).

        Args:
            rule_config: Rule configuration dictionary.

        Returns:
            Callable | None: Rule function or None if creation failed.
        """
        rule_name = rule_config.get("name")
        params = rule_config.get("parameters", {})

        if rule_name == "large_quantization_range_warning":
            threshold = params.get("threshold", 1000.0)

            def rule_wrapper(tensor_name: str, tensor: TensorEncoding) -> Tuple[bool, str | None]:
                return validation_rules.rule_large_quantization_range_warning(
                    tensor_name, tensor, threshold=threshold
                )

            return rule_wrapper

        if self._logger:
            self._logger.warning(f"Unknown parametric rule: {rule_name}")
        return None

    def _create_context_rule_from_config(self, rule_config: Dict[str, Any]) -> Callable | None:
        """Create a context-aware validation rule from configuration.

        Args:
            rule_config: Rule configuration dictionary.

        Returns:
            Callable | None: Rule function or None if creation failed.
        """
        rule_type = rule_config.get("type", "builtin")

        if rule_type == "builtin":
            rule_name = rule_config.get("name")
            if rule_name not in self._builtin_context_rules:
                if self._logger:
                    self._logger.warning(f"Built-in context rule not found: {rule_name}")
                return None
            return self._builtin_context_rules[rule_name]

        if self._logger:
            self._logger.warning(f"Custom context-aware rules not yet supported: {rule_type}")
        return None


def load_rules_from_json(
    config_path: Path | str, logger=None
) -> Tuple[List[Callable], List[Callable]]:
    """Load validation rules from JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file.
        logger: Optional logger instance.

    Returns:
        Tuple[List[Callable], List[Callable]]: (validation_rules, context_aware_rules)
    """
    loader = RuleConfigLoader(logger=logger)
    config = loader.load_config(config_path)
    return loader.create_rules_from_config(config)
