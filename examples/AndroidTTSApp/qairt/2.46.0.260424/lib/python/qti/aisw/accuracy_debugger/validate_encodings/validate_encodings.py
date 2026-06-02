# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
from pathlib import Path
from typing import Callable, List, Tuple

from qti.aisw.accuracy_debugger.encodings.encodings import ModelEncoding, TensorEncoding
from qti.aisw.accuracy_debugger.encodings.encodings_utils import TensorType
from qti.aisw.accuracy_debugger.utils.file_utils import dump_csv, dump_json
from qti.aisw.accuracy_debugger.validate_encodings import context_aware_rules, validation_rules
from qti.aisw.accuracy_debugger.graph_op.dlc_graph_utils import DLCConnectedGraph
from qti.aisw.accuracy_debugger.validate_encodings.rule_config_loader import load_rules_from_json
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class ValidateEncodings:
    """Validates AIMET encoding files against a set of quantization rules.

    By default all built-in rules defined in ``validation_rules.py`` and
    ``context_aware_rules.py`` are active.  When a JSON rule configuration
    file is supplied via ``load_rules_from_config()`` or the ``rule_config_path``
    argument of ``run()``, the built-in rules are **replaced entirely** by the
    rules defined in that file — only the rules explicitly listed in the JSON
    will run.  This lets customers triage with a precise, model-specific rule
    set without any implicit checks running in the background.

    See ``configs/example_rules.json`` for an annotated template that
    demonstrates every supported rule type.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initializes the ValidateEncodings class.

        Args:
            logger: Python Logger object. If None, a default logger will be created.
        """
        self._logger = logger or QAIRTLogger.register_area_logger(
            area=LogAreas.register_log_area("Validate Encodings"),
            level="INFO",
            formatter_val="simple",
            handler_list=["dev_console"],
        )
        self._validation_rules: List[Callable[[str, TensorEncoding], Tuple[bool, str | None]]] = []
        self._context_aware_rules: List[
            Callable[
                [str, TensorEncoding, ModelEncoding, DLCConnectedGraph | None], Tuple[bool, str | None]
            ]
        ] = []
        self._dlc_graph: DLCConnectedGraph | None = None
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register the full set of built-in validation rules.

        These rules run whenever no JSON rule configuration file is provided.
        When ``load_rules_from_config()`` is called the entire built-in set is
        replaced by the rules defined in the JSON file, so only the explicitly
        listed rules will run.
        """
        self._validation_rules = [
            validation_rules.rule_conv_weights_symmetric,
            validation_rules.rule_rmsnorm_weights_16bit,
            validation_rules.rule_rmsnorm_weights_asymmetric,
            validation_rules.rule_layernorm_weights_asymmetric,
            validation_rules.rule_batchnorm_weights_asymmetric,
            validation_rules.rule_layernorm_weights_16bit,
            lambda name, tensor: validation_rules.rule_large_quantization_range_warning(
                name, tensor, threshold=1000.0
            ),
        ]
        self._context_aware_rules = [
            context_aware_rules.rule_reshape_encoding_matches_predecessor,
            context_aware_rules.rule_transpose_encoding_matches_predecessor,
            context_aware_rules.rule_concat_inputs_same_range,
        ]

    def add_validation_rule(
        self, rule: Callable[[str, TensorEncoding], Tuple[bool, str | None]]
    ) -> None:
        """Add a custom validation rule.

        Args:
            rule: A callable that takes (tensor_name, tensor_encoding) and returns
                  (is_valid, rule_description). The rule should return (True, None)
                  if it doesn't apply to the given tensor.
        """
        self._validation_rules.append(rule)
        self._logger.info(f"Added custom validation rule: {rule.__name__}")

    def add_context_aware_rule(
        self,
        rule: Callable[
            [str, TensorEncoding, ModelEncoding, DLCConnectedGraph | None], Tuple[bool, str | None]
        ],
    ) -> None:
        """Add a custom context-aware validation rule.

        Args:
            rule: A callable that takes (tensor_name, tensor_encoding, model_encoding, dlc_graph)
                  and returns (is_valid, rule_description). The rule should return
                  (True, None) if it doesn't apply to the given tensor.
        """
        self._context_aware_rules.append(rule)
        self._logger.info(f"Added custom context-aware validation rule: {rule.__name__}")

    def clear_validation_rules(self) -> None:
        """Clear all validation rules."""
        self._validation_rules.clear()
        self._context_aware_rules.clear()
        self._logger.info("Cleared all validation rules")

    def reset_to_default_rules(self) -> None:
        """Reset to the full set of built-in validation rules.

        Discards any rules loaded from a JSON config file or added via
        ``add_validation_rule()`` / ``add_context_aware_rule()`` and
        restores the original built-in rule set.
        """
        self.clear_validation_rules()
        self._register_default_rules()
        self._logger.info("Reset to built-in validation rules")

    def load_rules_from_config(self, config_path: Path | str) -> None:
        """Load validation rules from JSON configuration file.

        Args:
            config_path: Path to the JSON configuration file.

        Raises:
            FileNotFoundError: If config file does not exist.
            json.JSONDecodeError: If config file is not valid JSON.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self._logger.info(f"Loading rules from configuration: {config_path}")

        # Load rules from JSON
        validation_rule_list, context_rule_list = load_rules_from_json(
            config_path, logger=self._logger
        )

        # Clear existing rules and set new ones
        self.clear_validation_rules()
        self._validation_rules = validation_rule_list
        self._context_aware_rules = context_rule_list

        self._logger.info(
            f"Loaded {len(self._validation_rules)} validation rules and "
            f"{len(self._context_aware_rules)} context-aware rules from configuration"
        )

    def validate(self, model_encoding: ModelEncoding) -> List[dict]:
        """Validate all tensors against registered rules.

        Args:
            model_encoding: ModelEncoding object to validate.

        Returns:
            List[dict]: List of violation dictionaries, each containing:
                - tensor_name: Name of the tensor
                - rule_description: Description of the violated rule
                - dtype: Data type of the tensor
                - bitwidth: Bitwidth of the quantization
                - is_symm: Whether quantization is symmetric
                - channels: Number of channels
        """
        violations = []

        # Validate parameter encodings.
        # get_type_encodings(ParamEncodings) handles both explicit param encodings
        # (V0/V1/DLC) and the V2 fallback (name-based filtering) transparently.
        param_encodings = model_encoding.get_type_encodings(tensor_type=TensorType.ParamEncodings)
        self._logger.info(f"Validating {len(param_encodings)} parameter tensors...")

        for tensor_name, tensor_encoding in param_encodings.items():
            # Apply regular rules
            for rule in self._validation_rules:
                is_valid, description = rule(tensor_name, tensor_encoding)
                if not is_valid:
                    violation = {
                        "tensor_name": tensor_name,
                        "rule_description": description,
                        "dtype": tensor_encoding.dtype.value if tensor_encoding.dtype else "N/A",
                        "bitwidth": (
                            tensor_encoding.bitwidth if tensor_encoding.bitwidth else "N/A"
                        ),
                        "is_symm": tensor_encoding.is_symm if tensor_encoding.is_symm else "N/A",
                        "channels": (
                            tensor_encoding.channels if tensor_encoding.channels else "N/A"
                        ),
                    }
                    violations.append(violation)
                    self._logger.warning(
                        f"Violation found - Tensor: {tensor_name}, Rule: {description}"
                    )

        # Validate activation encodings.
        # get_type_encodings(ActivationEncodings) handles both explicit activation
        # encodings (V0/V1/DLC) and the V2 fallback transparently.
        activation_encodings = model_encoding.get_type_encodings(
            tensor_type=TensorType.ActivationEncodings
        )
        self._logger.info(f"Validating {len(activation_encodings)} activation tensors...")

        for tensor_name, tensor_encoding in activation_encodings.items():
            # Apply regular rules
            for rule in self._validation_rules:
                is_valid, description = rule(tensor_name, tensor_encoding)
                if not is_valid:
                    violation = {
                        "tensor_name": tensor_name,
                        "rule_description": description,
                        "dtype": tensor_encoding.dtype.value if tensor_encoding.dtype else "N/A",
                        "bitwidth": (
                            tensor_encoding.bitwidth if tensor_encoding.bitwidth else "N/A"
                        ),
                        "is_symm": tensor_encoding.is_symm if tensor_encoding.is_symm else "N/A",
                        "channels": (
                            tensor_encoding.channels if tensor_encoding.channels else "N/A"
                        ),
                    }
                    violations.append(violation)
                    self._logger.warning(
                        f"Violation found - Tensor: {tensor_name}, Rule: {description}"
                    )

            # Apply context-aware rules only when a DLC connected graph is
            # available (scenarios 2 and 3).  In scenario 1 (JSON only) there
            # is no graph structure, so context-aware rules are skipped entirely
            # rather than falling back to unreliable name-based heuristics.
            if self._dlc_graph is not None:
                for rule in self._context_aware_rules:
                    is_valid, description = rule(
                        tensor_name, tensor_encoding, model_encoding, self._dlc_graph
                    )
                    if not is_valid:
                        violation = {
                            "tensor_name": tensor_name,
                            "rule_description": description,
                            "dtype": tensor_encoding.dtype.value if tensor_encoding.dtype else "N/A",
                            "bitwidth": (
                                tensor_encoding.bitwidth if tensor_encoding.bitwidth else "N/A"
                            ),
                            "is_symm": tensor_encoding.is_symm if tensor_encoding.is_symm else "N/A",
                            "channels": (
                                tensor_encoding.channels if tensor_encoding.channels else "N/A"
                            ),
                        }
                        violations.append(violation)
                        self._logger.warning(
                            f"Violation found - Tensor: {tensor_name}, Rule: {description}"
                        )

        return violations

    def run(
        self,
        encoding_path: Path | str | None = None,
        output_dir: Path | str = ".",
        dlc_file_path: Path | str | None = None,
        rule_config_path: Path | str | None = None,
    ) -> dict:
        """Run validation and save results.

        Three usage scenarios are supported:

        1. **JSON only** (``encoding_path`` is a .json/.encodings file,
           ``dlc_file_path`` is not set):
           Load encodings from the JSON file and run simple (per-tensor)
           validation rules only.  Context-aware rules are **skipped**
           because no graph structure is available.

        2. **JSON + DLC** (``encoding_path`` is a .json/.encodings file,
           ``dlc_file_path`` is also set):
           Load encodings from the JSON file, build the connected graph from
           the DLC, and run both simple and context-aware validation rules.

        3. **DLC only** (``encoding_path`` is a .dlc file, or
           ``encoding_path`` is None and ``dlc_file_path`` is set):
           Load encodings directly from the DLC file, build the connected
           graph from the same DLC, and run both simple and context-aware
           validation rules.

        Args:
            encoding_path: Path to a .json/.encodings or .dlc file, or None.
                When None, ``dlc_file_path`` must be provided (scenario 3).
            output_dir: Directory where validation_report.json and
                validation_report.csv will be written.
            dlc_file_path: Optional path to a quantized DLC file.  Used to
                build the connected graph for context-aware rules (scenarios
                2 and 3).  In scenario 3 it also serves as the encoding
                source when ``encoding_path`` is None.
            rule_config_path: Optional path to a JSON rule configuration file.
                When provided, the built-in rules are replaced entirely by the
                rules defined in that file — only the explicitly listed rules
                will run.  When omitted, all built-in rules are active.

        Returns:
            dict with keys:
                - violations: list of violation dicts
                - total_violations: int
                - json_report_path: str
                - csv_report_path: str

        Raises:
            ValueError: If neither ``encoding_path`` nor ``dlc_file_path``
                is provided, or if the combination of arguments is invalid.
            FileNotFoundError: If any supplied path does not exist.
        """
        output_dir = Path(output_dir)

        # ------------------------------------------------------------------
        # Resolve the three scenarios
        # ------------------------------------------------------------------
        dlc_file_path = Path(dlc_file_path) if dlc_file_path else None

        encoding_path = encoding_path or dlc_file_path
        if encoding_path:
            encoding_path = Path(encoding_path)
        else:
            raise ValueError(
                    "At least one of encoding_path or dlc_file_path must be provided."
                )

        # Scenario 3a: encoding_path is a DLC → use it as both encoding
        # source and graph source(if dlc_file_path is None).
        if not dlc_file_path and encoding_path.suffix.lower() == ".dlc":
            dlc_file_path = encoding_path

        # ------------------------------------------------------------------
        # Validate that all supplied paths exist
        # ------------------------------------------------------------------
        if encoding_path and not encoding_path.exists():
            raise FileNotFoundError(f"Encoding file not found: {encoding_path}")
        if dlc_file_path and not dlc_file_path.exists():
            raise FileNotFoundError(f"DLC file not found: {dlc_file_path}")

        # ------------------------------------------------------------------
        # Load custom rules if requested
        # ------------------------------------------------------------------
        if rule_config_path:
            self.load_rules_from_config(rule_config_path)

        # ------------------------------------------------------------------
        # Build connected graph (scenarios 2 and 3)
        # ------------------------------------------------------------------
        if dlc_file_path:
            self._logger.info(f"Building connected graph from DLC: {dlc_file_path}")
            try:
                self._dlc_graph = DLCConnectedGraph(dlc_file_path, logger=self._logger)
            except Exception as e:
                self._logger.warning(
                    f"Failed to build connected graph from DLC: {e}. "
                    "Context-aware rules will fall back to heuristics."
                )
                self._dlc_graph = None
        else:
            # Scenario 1: no DLC available — context-aware rules are skipped.
            self._logger.info(
                "No DLC file provided. Context-aware rules will be skipped."
            )
            self._dlc_graph = None

        # ------------------------------------------------------------------
        # Load encodings
        # ------------------------------------------------------------------
        output_dir.mkdir(parents=True, exist_ok=True)
        model_encoding = ModelEncoding()
        model_encoding.load(artifact=encoding_path)

        # ------------------------------------------------------------------
        # Run validation
        # ------------------------------------------------------------------
        violations = self.validate(model_encoding)

        # ------------------------------------------------------------------
        # Save results
        # ------------------------------------------------------------------
        json_report = output_dir / "validation_report.json"
        csv_report = output_dir / "validation_report.csv"

        self._logger.info(f"Saving validation results to {output_dir}...")
        dump_json(data=violations, json_path=json_report)

        csv_data: dict = {
            "tensor_name": [],
            "rule_description": [],
            "dtype": [],
            "bitwidth": [],
            "is_symm": [],
            "channels": [],
        }
        for violation in violations:
            for key in csv_data:
                csv_data[key].append(violation[key])
        dump_csv(data_frame=csv_data, csv_path=csv_report, index=False)

        if violations:
            self._logger.warning(
                f"Validation completed with {len(violations)} violation(s) found."
            )
        else:
            self._logger.info("All checked tensors satisfy the defined rules.")

        self._logger.info(f"Validation report saved to: {json_report}")
        self._logger.info(f"CSV report saved to: {csv_report}")

        return {
            "violations": violations,
            "total_violations": len(violations),
            "json_report_path": str(json_report),
            "csv_report_path": str(csv_report),
        }
