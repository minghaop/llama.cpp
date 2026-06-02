# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions

import argparse


class QairtCompareEncodingsCmdOptions(CmdOptions):
    def __init__(self, args, validate_args=True):
        super().__init__("compare_encodings", args, validate_args=validate_args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="Script to compare QNN encodings with AIMET encodings",
        )

        required = self.parser.add_argument_group("required arguments")

        required.add_argument(
            "--encoding1_file_path",
            type=str,
            required=True,
            help="Path to either QAIRT or AIMET encodings file",
        )
        required.add_argument(
            "--encoding2_file_path",
            type=str,
            required=True,
            help="Path to either QAIRT or AIMET encodings file",
        )

        optional = self.parser.add_argument_group("optional arguments")
        optional.add_argument(
            "--quantized_dlc1_path",
            type=str,
            required=False,
            help="Path to quantized dlc file related to encoding_file1 being passed."
            "If passed along side with framework model for any of the encoding_config, "
            "it performs following operations on the qairt encodings file:"
            "1.  Propagates convert_ops encodings to the its parent op considering the fact that"
            "parent op exists in the framework model"
            "2.  Resolves any activation name changes done. For e.g. matmul+add in framework"
            "model becomes fc in the dlc graph and the tensor name gets _fc suffix."
            "It also performs supergroup mapping.",
            default=None,
        )
        optional.add_argument(
            "--quantized_dlc2_path",
            type=str,
            required=False,
            help="Path to quantized dlc file related to encoding_file2 being passed."
            "If passed along side with framework model for any of the encoding_config, "
            "it performs following operations on the qairt encodings file:"
            "1.  Propagates convert_ops encodings to the its parent op considering the fact that"
            "parent op exists in the framework model"
            "2.  Resolves any activation name changes done. For e.g. matmul+add in framework"
            "model becomes fc in the dlc graph and the tensor name gets _fc suffix."
            "It also performs supergroup mapping.",
            default=None,
        )
        optional.add_argument(
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
        optional.add_argument(
            "--scale_threshold",
            default=0.001,
            type=float,
            required=False,
            help="threshold for scale comparision of two encodings. For e.g."
            "scale1=0.5, scale2=0.01. We compare scale1 and scale2 as:"
            "abs(scale1-scale2)<(min(scale1, scale2)*scale_threshold). This ensures that bound is"
            "maintained by the lowest scale value among the given two scales.",
        )
        optional.add_argument(
            "--working_dir",
            type=str,
            required=False,
            default="working_directory",
            help="Working directory for the {} to store temporary files. ".format(self.component)
            + "Creates a new directory if the specified working directory does not exist",
        )
        optional.add_argument(
            "--output_dirname",
            type=str,
            required=False,
            default="<curr_date_time>",
            help="output directory name for the {} to store temporary files under <working_dir>/{}. ".format(
                self.component, self.component
            )
            + "Creates a new directory if the specified working directory does not exist",
        )
        optional.add_argument(
            "-v", "--verbose", action="store_true", default=False, help="Verbose printing"
        )
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        return parsed_args

    def get_all_associated_parsers(self):
        return [self.parser]
