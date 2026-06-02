# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
import os
from logging import Logger

from qti.aisw.accuracy_debugger.encodings.encodings import ModelEncoding
from qti.aisw.accuracy_debugger.encodings_converter.encoding_converter_utils import (
    get_resolved_names,
)
from qti.aisw.accuracy_debugger.encodings_converter.encodings_converter import EncodingsConverter
from qti.aisw.accuracy_debugger.graph_op.target_op import TargetOp
from qti.aisw.accuracy_debugger.utils.file_utils import dump_json
from qti.aisw.dlc_utils import modeltools  # type: ignore


class QairtEncodingsConverter(EncodingsConverter):
    """Class for converting encoding files dumped by the QAIRT Quantizer to the AIMET format."""

    def __init__(
        self,
        framework_model_path: str,
        quantized_dlc_path: str,
        working_dir: str,
        logger: Logger,
    ) -> None:
        """Initializes the QairtEncodingsConverter class and its base class EncodingsConverter.

        Args:
            framework_model_path: Path to the framework model.
            quantized_dlc_path: Path to the quantized DLC file.
            working_dir: Path to the working directory.
            logger: Logger object.
        """
        super().__init__(framework_model_path, working_dir, logger)
        self._dlc_path = quantized_dlc_path
        self._model_encoding = ModelEncoding()
        self._model_encoding.load(artifact=self._dlc_path, load_dlc=True)
        self._child_initialize()

    def _child_initialize(self):
        """Initializes the children class variables"""
        self._target_connected_graph = self._create_target_connected_graph()
        target_activations = []
        for op in self._target_connected_graph.values():
            target_activations.extend(op.outputs)
        target_activation_op_map = {}

        for _, op in self._target_connected_graph.items():
            for output_name in op.outputs:
                # Resolve target activation and op incase of name change
                # Skip for converted_QNN_DATATYPE activations
                if (
                    output_name not in self._framework_activations
                    and "converted_QNN_DATATYPE" not in output_name
                ):
                    resolved_name, modified_op = self._resolve_target_name_change(
                        output_name, op, target_activations
                    )
                else:
                    # output_name present in both framework and target graph
                    resolved_name, modified_op = output_name, op

                # encodings for output_name may not be present in the model_encoding if it is
                # one of (integer tensor, constant tensor) hence resolve only for those
                # output_name which has encodings present in model_encoding
                if output_name in self._model_encoding.activation_tensors:
                    resolved_tensor_encoding = self._model_encoding.get_tensor_encoding(
                        tensor_name=output_name
                    )
                    resolved_tensor_encoding = copy.deepcopy(resolved_tensor_encoding)
                    resolved_tensor_encoding.tensor_name = resolved_name
                    self._model_encoding.add(tensor_encoding=resolved_tensor_encoding)

                # Prepare tensor mapping for the target activations
                # framework_name: target_name
                if resolved_name in self._framework_activation_op_map:
                    self._resolved_target_activations[resolved_name] = output_name
                target_activation_op_map[resolved_name] = modified_op

        self._target_activation_op_map = target_activation_op_map

        tensor_mapping_path = os.path.join(self._working_dir, "tensor_mapping.json")
        dump_json(self._resolved_target_activations, tensor_mapping_path)

    def _modify_target_op(self, output_name: str, resolved_name: str, op: TargetOp) -> TargetOp:
        """Modifies the TargetOp if the activation has changed in the target DLC.

        Args:
            output_name: The output name of the operator.
            resolved_name: The resolved name for the operator output, also present in the framework.
            op: The object of the TargetOp class representing the current operator.

        Returns:
            TargetOp: The modified TargetOp object.
        """
        op_activations = op.outputs
        modified_op_activations = [
            resolved_name if activation == output_name else activation
            for activation in op_activations
        ]
        op.outputs = modified_op_activations
        for children_op in op.children_ops:
            children_op_inputs = children_op.inputs
            modified_op_inputs = [
                resolved_name if op_input == output_name else op_input
                for op_input in children_op_inputs
            ]
            children_op.inputs = modified_op_inputs

        return op

    def _resolve_target_name_change(
        self, output_name: str, op: TargetOp, target_activations: list
    ) -> tuple:
        """If the target activation name has been changed,
        resolve such names if possible and accordingly modify the target op object

        Args:
            output_name (str): The output name of the target op
            op (TargetOp): Object of current target op
            target_activations (list): list of activations which are present in target graph

        Returns:
            tuple: Resolved output name and modified TargetOp object
        """
        resolved_names = get_resolved_names(output_name)
        for resolved_name in resolved_names:
            if resolved_name in (set(self._framework_activations) - set(target_activations)):
                # resolved name present in framework graph but not in target graph
                # 419(in framework) -> 419_reshpe(target)
                # 419(in framework) -> 419.nchw(target)
                # and there is no 419 activation in target
                modified_op = self._modify_target_op(output_name, resolved_name, op)
                return resolved_name, modified_op

        # resolved name not present in framework graph
        # this is new logical node added by target
        # Matmul_0_pre_reshape(target)
        # do nothing, return output_name and op
        # or any of the resolved_names not in target graph
        # 419, 491.nchw, both in dlc, then do not resolve
        # the name for 419.nchw
        return output_name, op

    def _create_target_connected_graph(self) -> None:
        """Creates target connected graph from DLC graph with op_name as key and TargetOp object as
        value.
        """
        model_reader = modeltools.IrDlcReader()
        model_reader.open(self._dlc_path)
        ir_graph = model_reader.get_ir_graph()

        target_connected_graph = {}

        # Make target_op for inputs
        for idx, inp in enumerate(ir_graph.get_input_tensors_to_graph()):
            name = f"input_{idx}"
            target_op = TargetOp(name)
            target_op.op_type = "input"
            target_op.data_type = inp.data_type().name
            target_op.inputs = []
            target_op.outputs = [inp.name()]
            target_connected_graph[name] = target_op

        for op in ir_graph.get_ops():
            static_tensors = [
                op_input.name()
                for op_input in op.inputs()
                if "IrStaticTensor" in str(type(op_input))
            ]

            target_op = TargetOp(op.name)
            target_op.op_type = op.type
            target_op.data_type = op.outputs()[0].data_type().name
            target_op.inputs = [inp.name() for inp in op.inputs()]
            target_op.outputs = [output.name() for output in op.outputs()]
            target_op.static_tensors = static_tensors
            target_connected_graph[op.name] = target_op

        for _, node1 in target_connected_graph.items():
            for _, node2 in target_connected_graph.items():
                if any(output in node2.inputs for output in node1.outputs):
                    # node1 -> node2
                    node1.children_ops = [node2]
                    node2.parent_ops = [node1]

        return target_connected_graph
