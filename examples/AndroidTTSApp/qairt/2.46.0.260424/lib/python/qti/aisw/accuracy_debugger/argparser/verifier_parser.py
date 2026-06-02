# ==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import argparse
from pathlib import Path

from qti.aisw.accuracy_debugger.argparser.parser import Parser
from qti.aisw.tools.core.utilities.comparators.common import COMPARATORS
from qti.aisw.tools.core.utilities.comparators.factory import get_comparator


class VerifierParser(Parser):
    """This is a parser for the Verifier utility."""

    def __init__(self, component="verification"):
        super().__init__(component)

    def _initialize(self):
        """Create parser with Verifier specific arguments"""
        self.required.add_argument(
            "--inference_tensor",
            type=str,
            required=True,
            default=None,
            help="Directory path of inference tensor files.",
        )

        self.required.add_argument(
            "--reference_tensor",
            type=str,
            required=True,
            default=None,
            help="Directory path of reference tensor files.",
        )

        self.optional.add_argument(
            "--comparators",
            type=COMPARATORS,
            nargs="+",
            required=False,
            choices=[comp.value for comp in COMPARATORS],
            default=[COMPARATORS.MSE],
            help="Comparator to use to compare tensors. For multiple comparators, "
            "specify as follows: --comparator mse std. Default comparator is mse",
        )

        self.optional.add_argument(
            "--reference_dtype",
            type=str,
            required=False,
            default="float32",
            help="Data type of reference tensor files.",
        )

        self.optional.add_argument(
            "--inference_dtype",
            type=str,
            required=False,
            default="float32",
            help="Data type of inference tensor files.",
        )

        self.optional.add_argument(
            "--dlc_file",
            type=str,
            required=False,
            default=None,
            help="Path to dlc file.",
        )

        self.optional.add_argument(
            "--graph_info",
            type=str,
            required=False,
            default=None,
            help="""Path to json file containing graph information like, tensor mapping, graph
            structure and layout information in the following format:
            {'tensor_mapping':{}, graph_structure:{}, layout_info:{}}""",
        )

        self.optional.add_argument(
            "--is_qnn_golden_reference",
            action="store_true",
            required=False,
            default=False,
            help="""Specifies that outputs passed with --reference_tensor are dumped by QNN.""",
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
        if args.dlc_file:
            args.dlc_file = Path(args.dlc_file)
        args.comparators = [get_comparator(comp) for comp in args.comparators]
        return args
