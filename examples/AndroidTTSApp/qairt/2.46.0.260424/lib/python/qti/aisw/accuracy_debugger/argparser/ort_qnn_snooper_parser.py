# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

"""ORT QNN Snooper Parser Module

This module provides argument parsing functionality for the ORT/QNN Execution Provider
Snooper tool. It handles command-line argument parsing, validation, and configuration
for debugging and analyzing ONNX QDQ models.

The parser supports various debugging algorithms, comparators, and layer-specific
configurations for model analysis.
"""

import argparse
from os import path

from qti.aisw.accuracy_debugger.argparser.parser import Parser
from qti.aisw.accuracy_debugger.utils.constants import (
    Algorithm,
)
from qti.aisw.tools.core.utilities.comparators.common import COMPARATORS
from qti.aisw.tools.core.utilities.comparators.factory import get_comparator


class OrtQnnSnooperParser(Parser):
    """Parser for ORT/QNN Execution Provider Snooper Tool.

    This class provides argument parsing and validation for the ORT/QNN-EP Snooper tool,
    which is used for debugging and analyzing ONNX models running on QNN-EP.
    It supports various debugging algorithms (ONESHOT, LAYERWISE) and multiple
    comparators for tensor comparison.

    Attributes:
        component (str): Component identifier for the parser, defaults to "snooping_ort_qnn"
        required (argparse.ArgumentGroup): Group for required arguments
        optional (argparse.ArgumentGroup): Group for optional arguments
    """

    def __init__(self, component="snooping_ort_qnn"):
        """Initialize the OrtQnnSnooperParser.

        Args:
            component (str, optional): Component identifier for the parser.
                Defaults to "snooping_ort_qnn".
        """
        super().__init__(component)

    def _initialize(self):
        """Initialize parser with ORT/QNN-EP snooper tool specific arguments.

        This method sets up all required and optional command-line arguments for the
        ORT/QNN snooper tool, including model paths, debugging algorithms, comparators,
        and layer-specific configurations.

        Required Arguments:
            --qdq_model: Path to the QDQ (Quantize-Dequantize) ONNX model
            --input_sample: Path to text file containing input sample data

        Optional Arguments:
            --reference_model: Path to the reference ONNX model
            --algorithm: Debugging algorithm (ONESHOT or LAYERWISE)
            --comparator: Tensor comparison metrics (MSE, STD, etc.)
            --working_directory: Path to working directory for intermediate files
            --output_directory: Path to output directory for results
            --golden_reference: Path to golden reference tensor files
            --set_intermediate_layers: Comma-separated layer names/types to debug
            --set_cpu_layers: Comma-separated layer names/types to run on CPU

        Raises:
            None: This method only configures the parser, validation happens later
        """
        self.required.add_argument(
            "--qdq_model", type=str, required=True, help="Path to QDQ ONNX model."
        )
        self.required.add_argument(
            "--input_sample",
            required=True,
            type=str,
            help="Path to text file containing input sample.",
        )
        self.optional.add_argument(
            "--reference_model",
            type=str,
            required=False,
            default=None,
            help="Path to reference ONNX model. If not provided, given qdq_model will be run on "
            "ORT-CPU to generate reference outputs. "
            "Note: This argument is allowed only with oneshot algorithm (for layerwise algorithm "
            "qdq_model itself will be used to generate ORT-CPU reference outputs).",
        )
        self.optional.add_argument(
            "--algorithm",
            type=Algorithm,
            required=False,
            choices=[Algorithm.ONESHOT, Algorithm.LAYERWISE],
            default=Algorithm.ONESHOT,
            help="Algorithm to use to debug the model.",
        )
        self.optional.add_argument(
            "--comparator",
            type=COMPARATORS,
            nargs="+",
            required=False,
            choices=[comp.value for comp in COMPARATORS],
            default=[COMPARATORS.MSE],
            help="Comparator to use to compare tensors. For multiple comparators, "
            "specify as follows: --comparator mse std",
        )
        self.optional.add_argument(
            "--working_directory",
            type=str,
            required=False,
            default=None,
            help="Path to working directory. If not specified, a directory with name "
            "working_directory will be created in the current directory.",
        )
        self.optional.add_argument(
            "--output_directory",
            type=str,
            required=False,
            default=None,
            help="Name of the output directory. If not specified, a directory with name "
            "<timestamp> will be created in the working directory.",
        )
        self.optional.add_argument(
            "--golden_reference",
            type=str,
            required=False,
            default=None,
            help="The path of directory where golden reference tensor files are saved.",
        )
        self.optional.add_argument(
            "--set_intermediate_layers",
            type=lambda s: [item.strip() for item in s.split(",")],
            default=None,
            required=False,
            help="Pass comma separated layer names/types which needs to be debugged."
            "Can specify layer names like conv1_output, fc_layer_output or layer types like Conv, Gemm, MatMul"
            "e.g., --set_intermediate_layers conv1_output,Gemm,MatMul,fc_layer_output",
        )
        self.optional.add_argument(
            "--set_cpu_layers",
            type=lambda s: [item.strip() for item in s.split(",")],
            default=None,
            required=False,
            help="Pass comma separated layer names/types which needs to be executed on CPU Execution-Provider."
            "This option is supported only with layerwise algorithm."
            "Can specify layer names like conv1_output, fc_layer_output or layer types like Conv, Gemm, MatMul"
            "e.g., --set_cpu_layers conv1_output,Gemm,MatMul,fc_layer_output",
        )
        self.optional.add_argument(
            "--dump_elementwise_stats",
            default=None,
            required=False,
            help="Dumps elementwise comparison statistics for the specified layer names/types "
            "along with their corresponding allowed step value. "
            "Can specify layer names like conv1_output, fc_layer_output or layer types like Conv, MatMul. "
            "Step value has to be an integer which represents allowed steps while comparing elementwise. "
            "e.g., --dump_elementwise_stats 'conv1_output=3;Gemm=1;MatMul=2;fc_layer_output=5'",
        )
        self.optional.add_argument(
            "--log_level",
            type=str.upper,
            required=False,
            default="INFO",
            choices=["ERROR", "WARN", "INFO", "DEBUG", "VERBOSE"],
            help="Enable verbose logging.",
        )

    def _verify_and_update_parsed_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validate and update parsed command-line arguments.

        This method performs validation checks on the parsed arguments and transforms
        them into the appropriate format for downstream processing. It validates file
        paths, converts comparator enums to comparator classes, and parses input samples.

        Args:
            args (argparse.Namespace): Parsed command-line arguments

        Returns:
            argparse.Namespace: Verified and updated arguments

        Raises:
            argparse.ArgumentTypeError: If required files don't exist or paths are invalid
            FileNotFoundError: If reference model or QDQ model files are not found
        """
        # Validate QDQ model path exists (required for parsing input_sample txt)
        if not path.exists(args.qdq_model):
            raise argparse.ArgumentTypeError(f"QDQ model file '{args.qdq_model}' does not exist.")

        # Validate input sample file exists
        if not path.exists(args.input_sample):
            raise argparse.ArgumentTypeError(
                f"Input sample file '{args.input_sample}' does not exist."
            )
        # Parse input sample file into structured format
        # This converts the text file into InputSample object(s) for processing
        args.input_sample = self._parse_input_sample(str(args.qdq_model), args.input_sample)

        # Convert comparator enums to comparator class instances
        # This allows the tool to use the actual comparator implementations
        args.comparator = [get_comparator(comp) for comp in args.comparator]

        # Parse dump_elementwise_stats argument value
        try:
            if args.dump_elementwise_stats:
                elementwise_dict = {}
                for arg in args.dump_elementwise_stats.split(";"):
                    layer, step = arg.split("=")
                    elementwise_dict[layer] = int(step)
                args.dump_elementwise_stats = elementwise_dict
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"--dump_elementwise_stats value is not in valid format, parsing error: {e}"
            )

        return args
