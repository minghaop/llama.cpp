# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
from abc import ABC, abstractmethod

from qti.aisw.accuracy_debugger.framework_runner.frameworks.onnx_framework import (
    CustomOnnxFramework,
)
from qti.aisw.accuracy_debugger.utils.helper import InputSample


class Parser(ABC):
    """Abstract class for all the parsers."""

    def __init__(self, component=None):
        self._parser = argparse.ArgumentParser(
            prog=f"qairt-accuracy-debugger {component}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            conflict_handler="resolve",
        )
        self.required = self._parser.add_argument_group("required arguments")
        self.optional = self._parser.add_argument_group("optional arguments")

    def parse(self, args: list) -> argparse.Namespace:
        """Parse the arguments.

        Args:
            args (list): Arguments from the user to be parsed
        Returns:
            argparse.Namespace: parsed arguments
        """
        self._initialize()
        parsed_args = self._parser.parse_args(args)

        return self._verify_and_update_parsed_args(parsed_args)

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def _verify_and_update_parsed_args(self, parsed_args: argparse.Namespace) -> argparse.Namespace:
        return parsed_args

    def _parse_input_sample(
        self, input_model: str, input_sample: str, onnx_symbols: list[tuple] = None
    ) -> list[InputSample]:
        """Returns InputSample object for the specified input sample file.

        Args:
            input_model (str): Model file path
            input_sample (str): Input sample file path
            onnx_symbols (list[tuple]): List of symbols and their values to override.

        Returns:
            list[InputSample]: List of input sample objects
        """
        input_sample_objs = []
        input_sample_line = ""

        with open(input_sample, "r") as file:
            for line in file:
                strip_line = line.strip()
                if strip_line:
                    input_sample_line = strip_line
                    break
        if not input_sample_line:
            raise argparse.ArgumentTypeError("Invalid input sample file supplied.")
        input_tensors = input_sample_line.split()

        # Get model input tensor details
        onnx_framework = CustomOnnxFramework(logger=None)
        model_inp_tensors = onnx_framework.get_input_tensor_details(
            model_path=input_model, onnx_symbols=onnx_symbols
        )

        # Check if the number of input tensors matches
        if len(input_tensors) != len(model_inp_tensors):
            raise argparse.ArgumentTypeError(
                f"Number of input tensors in input sample file: {len(input_tensors)} does not "
                f"match the number of input tensors in the model: {len(model_inp_tensors)}."
            )

        # Determine the format of the input tensors and create InputSample objects accordingly
        if ":=" in input_tensors[0]:
            # Format: name:=raw_file
            model_tensor_names = [tensor["name"] for tensor in model_inp_tensors]
            for tensor in input_tensors:
                name, raw_file_path = tensor.split(":=")
                if name not in model_tensor_names:
                    raise argparse.ArgumentTypeError(
                        f"Input tensor name: {name} not found in the model."
                    )

                model_inp_tensor = next((t for t in model_inp_tensors if t["name"] == name), None)
                input_sample_objs.append(
                    InputSample(
                        name=name,
                        raw_file=raw_file_path,
                        dimensions=model_inp_tensor["shape"],
                        data_type=model_inp_tensor["data_type"],
                    )
                )
        else:
            # Format: raw_file
            for i, raw_file_path in enumerate(input_tensors):
                input_sample_objs.append(
                    InputSample(
                        raw_file=raw_file_path,
                        name=model_inp_tensors[i]["name"],
                        dimensions=model_inp_tensors[i]["shape"],
                        data_type=model_inp_tensors[i]["data_type"],
                    )
                )
        return input_sample_objs
