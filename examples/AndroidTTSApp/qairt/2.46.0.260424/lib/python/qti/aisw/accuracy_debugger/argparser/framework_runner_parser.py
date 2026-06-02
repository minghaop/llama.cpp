# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
from os import path

from qti.aisw.accuracy_debugger.argparser.parser import Parser


class FrameworkRunnerParser(Parser):
    """This is a parser for framework runner tool."""

    def __init__(self, component="framework_runner"):
        super().__init__(component)

    def _initialize(self):
        """Create parser with framework runner tool specific arguments"""
        super()._initialize()
        self.required.add_argument(
            "-m", "--input_model", type=str, required=True, help="path to the model file"
        )
        self.required.add_argument(
            "--input_sample",
            required=True,
            type=str,
            help="Path to text file containing input sample.",
        )
        self.optional.add_argument(
            "--working_directory",
            type=str,
            required=False,
            default=None,
            help="Path to working directory. If not specified a directory with name \
                working_directory will be created in the current directory.",
        )
        self.optional.add_argument(
            "--output_directory",
            type=str,
            required=False,
            default=None,
            help="Name of the output directory. If not specified a directory with name \
                <timestamp> will be created in the working directory.",
        )
        self.optional.add_argument(
            "-o",
            "--output_tensor",
            type=str,
            required=False,
            action="append",
            help="Name of the graph's specified output tensor(s).",
        )
        self.optional.add_argument(
            "--onnx_define_symbol",
            default=None,
            nargs=2,
            action="append",
            required=False,
            metavar=("SYMBOL", "VALUE"),
            help="Option to override specific input dimension symbols.",
        )
        self.optional.add_argument(
            "--log_level",
            type=str,
            required=False,
            default="info",
            choices=["info", "debug", "warning", "error"],
            help="Log level. Default is info",
        )

    def _verify_and_update_parsed_args(self, args: argparse.Namespace) -> argparse.Namespace:
        """Validates parsed arguments

        Args:
            args (argparse.Namespace): parsed arguments
        Returns:
            argparse.Namespace: Verified and updated arguments
        """
        args = super()._verify_and_update_parsed_args(args)
        if not path.exists(args.input_model):
            raise argparse.ArgumentTypeError(
                "Input model path, {}, doesn't exists.".format(args.input_model)
            )

        if not path.exists(args.input_sample):
            raise argparse.ArgumentTypeError(
                "Input sample file, {}, doesn't exists.".format(args.input_sample)
            )
        args.input_sample = self._parse_input_sample(
            str(args.input_model), args.input_sample, args.onnx_define_symbol
        )
        return args
