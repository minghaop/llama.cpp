# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse


class AccuracyDebuggerParser:
    """This is parser for accuracy debugger tool."""

    def __init__(self):
        self._parser = ""

    def _initilize(self):
        """Create parser with accuracy debugger tool components."""
        self._parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=True,
            allow_abbrev=False,
        )
        sub_parser = self._parser.add_subparsers(dest="component")
        framework_parser = sub_parser.add_parser(
            "framework_runner", help="Run framework runner", add_help=False
        )
        inference_parser = sub_parser.add_parser(
            "inference_engine", help="Run inference engine", add_help=False
        )
        verification_parser = sub_parser.add_parser(
            "verification", help="Run verification", add_help=False
        )
        compare_encodings_parser = sub_parser.add_parser(
            "compare_encodings", help="Run compare encodings", add_help=False
        )
        tensor_visualizer_parser = sub_parser.add_parser(
            "tensor_visualizer", help="Run tensor visualizer", add_help=False
        )
        snooping_parser = sub_parser.add_parser("snooping", help="Run snooping", add_help=False)
        snooping_ort_qnn_parser = sub_parser.add_parser(
            "snooping_ort_qnn", help="Run snooping_ort_qnn", add_help=False
        )
        validate_encoding_parser = sub_parser.add_parser(
            "validate_encoding", help="Run validate encoding", add_help=False
        )

    def parse(self, args):
        """Parse the arguments."""
        self._initilize()
        parsed_args, unknown_args = self._parser.parse_known_args(args)
        return parsed_args, unknown_args
