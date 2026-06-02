# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
import os

import numpy as np
from qti.aisw.accuracy_debugger.argparser.parser import Parser


class TensorVisualizerParser(Parser):
    """This is an argparser for Tensor Visualizer Utility."""

    def __init__(self, component="tensor_visualizer"):
        super().__init__(component)

    def _initialize(self):
        """Create parser with tensor visualizer utility specific arguments"""

        self.required.add_argument(
            "--target_tensors",
            type=str,
            required=True,
            help="Directory path to Target tensor files",
        )

        self.required.add_argument(
            "--golden_tensors",
            type=str,
            required=True,
            help="Directory path to Golden tensor files",
        )

        self.optional.add_argument(
            "-dt",
            "--data_type",
            type=str,
            required=False,
            default="float32",
            help="Data type to load the tensor file in. Default: float32",
        )

        self.optional.add_argument(
            "-wd",
            "--working_directory",
            type=str,
            required=False,
            default=None,
            help="Path to output directory. Default: tensor_visualizer_output_dir",
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
        try:
            np.dtype(args.data_type)
        except TypeError:
            raise ValueError("Invalid datatype passed in --datatype argument")

        for path in [args.target_tensors, args.golden_tensors]:
            if path and not os.path.isdir(path):
                raise ValueError(
                    "Invalid path passed in --target_tensors or --golden_tensors argument"
                )

        return args
