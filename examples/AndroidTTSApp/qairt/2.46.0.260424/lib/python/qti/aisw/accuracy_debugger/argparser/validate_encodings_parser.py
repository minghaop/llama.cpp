# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
import os

from qti.aisw.accuracy_debugger.argparser.parser import Parser


class ValidateEncodingParser(Parser):
    """This is an argparser for Validate Encoding Utility."""

    def __init__(self, component="validate_encoding"):
        super().__init__(component)

    def _initialize(self):
        """Create parser with Validate encoding utility specific arguments"""
        self.optional.add_argument(
            "--encoding_path",
            type=str,
            required=False,
            default=None,
            help="Path to an encoding file (.json / .encodings) or a quantized DLC file (.dlc). "
            "Optional. Three usage scenarios are supported:\n"
            "  1. JSON only: load encodings from the JSON file and run simple (per-tensor) "
            "validation rules only.\n"
            "  2. JSON + DLC: load encodings from the JSON file, build the connected graph "
            "from the DLC, and run both simple and context-aware validation rules.\n"
            "  3. DLC only: load encodings directly from the DLC file and run both simple "
            "and context-aware validation rules."
        )

        self.optional.add_argument(
            "--dlc_file_path",
            type=str,
            required=False,
            default=None,
            help="Path to quantized DLC file. Required for context-aware validation rules "
            "(reshape/transpose/concat operations)."
        )
        self.optional.add_argument(
            "--rule_config",
            type=str,
            required=False,
            default=None,
            help="Path to JSON rule configuration file. If provided, rules will be loaded "
            "from this file instead of using default rules. This allows users to extend "
            "or suppress validation rules as needed.",
        )
        self.optional.add_argument(
            "--working_directory",
            type=str,
            required=False,
            default=None,
            help="Path to working directory. Default: working_directory"
        )
        self.optional.add_argument(
            "--log_level",
            type=str,
            required=False,
            default="info",
            choices=["info", "debug", "warning", "error"],
            help="Log level. Default is info"
        )

    def _verify_and_update_parsed_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validates parsed arguments

        Args:
            args (argparse.Namespace): parsed arguments
        Returns:
            argparse.Namespace: Verified and updated arguments
        """
        if args.encoding_path and not os.path.isfile(args.encoding_path):
            raise ValueError(f"Invalid encoding path provided: {args.encoding_path}")

        if args.dlc_file_path and not os.path.isfile(args.dlc_file_path):
            raise ValueError(f"Invalid DLC file path provided: {args.dlc_file_path}")

        if args.rule_config and not os.path.isfile(args.rule_config):
            raise ValueError(f"Invalid rule configuration file path provided: {args.rule_config}")

        return args
