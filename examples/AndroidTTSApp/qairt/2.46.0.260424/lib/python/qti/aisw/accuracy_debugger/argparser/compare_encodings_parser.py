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


class CompareEncodingsParser(Parser):
    """This is an argparser for Compare Encodings Utility."""

    def __init__(self, component="compare_encodings"):
        super().__init__(component)

    def _initialize(self):
        """Create parser with Compare encodings utility specific arguments"""
        self.required.add_argument(
            "--encoding_path1",
            type=str,
            required=True,
            help="Path to an encoding file or quantized dlc file. Encoding file must end with either"
            ".json or .encodings. Dlc file must end with .dlc. Incase if dlc file provided, encodings"
            "are extracted from quantized dlc file.",
        )
        self.required.add_argument(
            "--encoding_path2",
            type=str,
            required=True,
            help="Path to an encoding file or quantized dlc file. Encoding file must end with either"
            ".json or .encodings. Dlc file must end with .dlc. Incase if dlc file provided, encodings"
            "are extracted from quantized dlc file.",
        )

        self.optional.add_argument(
            "--quantized_dlc1_path",
            type=str,
            required=False,
            help="Path to quantized dlc file related to encoding_path1 being passed."
            "If passed along side with framework model, "
            "it performs following operations on the encodings from encoding_path1:"
            "1.  Propagates convert_ops encodings to the its parent op considering the fact that"
            "parent op exists in the framework model"
            "2.  Resolves any activation name changes done. For e.g. matmul+add in framework"
            "model becomes fc in the dlc graph and the tensor name gets _fc suffix."
            "It also performs supergroup mapping which maps each QNN tensor to a set of tensors in"
            "framework graph which got fused to form the fused QNN op.",
            default=None,
        )
        self.optional.add_argument(
            "--quantized_dlc2_path",
            type=str,
            required=False,
            help="Path to quantized dlc file related to encoding_path2 being passed."
            "If passed along side with framework model, "
            "it performs following operations on the encodings from encoding_path2:"
            "1.  Propagates convert_ops encodings to the its parent op considering the fact that"
            "parent op exists in the framework model"
            "2.  Resolves any activation name changes done. For e.g. matmul+add in framework"
            "model becomes fc in the dlc graph and the tensor name gets _fc suffix."
            "It also performs supergroup mapping which maps each QNN tensor to a set of tensors in"
            "framework graph which got fused to form the fused QNN op.",
            default=None,
        )
        self.optional.add_argument(
            "--framework_model_path",
            type=str,
            required=False,
            help="path to the framework model. If passed"
            "along side with quantized dlc for any of the encoding_config, it performs following"
            "operations on the qairt encodings file:"
            "1.  Propagates convert_ops encodings to the its parent op considering the fact that"
            "parent op exists in the framework model"
            "2.  Resolves any activation name changes done. For e.g. matmul+add in framework"
            "model becomes fc in the dlc graph and the tensor name gets _fc suffix."
            "It also performs supergroup mapping.",
            default=None,
        )
        self.optional.add_argument(
            "--scale_threshold",
            default=0.001,
            type=float,
            required=False,
            help="threshold for scale comparision of two encodings. For e.g."
            "scale1=0.5, scale2=0.01. We compare scale1 and scale2 as:"
            "abs(scale1-scale2)<(min(scale1, scale2)*scale_threshold). This ensures that bound is"
            "maintained by the lowest scale value among the given two scales.",
        )
        self.optional.add_argument(
            "--working_directory",
            type=str,
            required=False,
            default=None,
            help="Path to working directory. Default: working_directory",
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
        for path in [
            args.encoding_path1,
            args.encoding_path2,
            args.quantized_dlc1_path,
            args.quantized_dlc2_path,
            args.framework_model_path,
        ]:
            if path and not os.path.isfile(path):
                raise ValueError(f"Invalid path provided: {path}")

        if args.scale_threshold <= 0:
            raise ValueError("--scale_threshold must be a number greater than 0")

        return args
