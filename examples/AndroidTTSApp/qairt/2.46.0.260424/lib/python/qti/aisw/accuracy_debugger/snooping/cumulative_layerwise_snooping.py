# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import onnx  # noqa
import logging
from pathlib import Path

from qti.aisw.accuracy_debugger.snooping.subgraph_snooper import SubgraphSnooper
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.accuracy_debugger.utils.graph_utils import get_common_parent_activations, get_subgraph


class CumlativeLayerwiseSnooper(SubgraphSnooper):
    """Subclass for cumulative layerwise snooping algorithm."""

    def __init__(self, logger: logging.Logger):
        """Initializes the CumulativeLayerwiseSnooper.

        Args:
            logger (logging.Logger): A python logger instance
        """
        self._activation_status = {}
        super().__init__(name=Algorithm.CUMULATIVE, logger=logger, is_cumulative=True)
        self._model_inputs = None
        self._model_outputs = None

    def _get_subgraph_outputs(self, target_activation_op_map: dict) -> set:
        """Finds out the cumulative subgraph's output tensor

        Args:
            target_activation_op_map (dict): Target activations to Target op map.

        Returns:
            (set): set of all subgraph outputs
        """
        if self._model_outputs:
            return self._model_outputs

        subgraph_outputs = set()
        for activation, op in target_activation_op_map.items():
            if len(op.children_ops) == 0:
                subgraph_outputs.update([activation])

        self._model_outputs = subgraph_outputs

        return subgraph_outputs

    def _get_subgraph_info(
        self,
        target_activation: str,
        target_activation_op_map: dict,
        framework_activation_op_map: dict,
        supergroup_activations: set,
    ) -> tuple[set, set, set, set]:
        """Given target_activation contruct the subgraph for cumulative-layerwise snooping.
        For the given target graph:

        input_1 -> op_1 -> op_2 ->|
                                  | -> op_5
        input_2 -> op_3 -> op_4 ->|

        A sample subgraph for op_4 activation would look like:

        input_1 -> op_1 -> op_2 ->|
                                  | -> op_5
                        -> op_4 ->|

        Args:
            target_activation (str): activation name in the target graph.
            target_activation_op_map (dict): Target activations to Target op map.
            framework_activation_op_map (dict): Framework activation to framework op map.
            supergroup_activations (set): Activations of supergroups.

        Returns:
            tuple(set, set, set, set): tuple of following informations:
                1.set of all subgraph inputs
                2.set of all subgraph outputs
                3.set of all target subgraph activations(inputs to the subgraph are not included)
                4.set of all framework subgraph activations(inputs to the subgraph are not included)
        """
        target_op = target_activation_op_map[target_activation]

        # Get subgraph inputs
        subgraph_inputs = self._get_subgraph_inputs(
            target_activation,
            target_activation_op_map,
            framework_activation_op_map,
            supergroup_activations,
        )

        # Get model inputs
        if not self._model_inputs:
            self._model_inputs = set()
            for activation, op in target_activation_op_map.items():
                if op.op_type == "input":
                    self._model_inputs.update([activation])

        # Get subgraph outputs
        subgraph_outputs = self._get_subgraph_outputs(target_activation_op_map)

        target_subgraph_activations = set()
        framework_subgraph_activations = set()

        self._logger.debug("+" * 71)
        self._logger.debug(f"Subgraph Inputs: {subgraph_inputs}")
        self._logger.debug(f"Subgraph Outputs: {subgraph_outputs}")
        if subgraph_inputs:
            # set of all cumulative subgraph activations is defined as:
            # set of all activations in graph minus set of all activation in subgraph formed between
            # model_inputs and target_op
            self._logger.debug("Getting target subgraph")
            target_input2op_activations, _, _ = get_subgraph(
                self._model_inputs, target_op.outputs, target_activation_op_map
            )
            target_subgraph_activations = (
                target_activation_op_map.keys() - self._model_inputs
            ) - target_input2op_activations
            target_subgraph_activations.update(target_op.outputs)

            self._logger.debug("Getting framework subgraph")
            framework_input2op_activations, _, _ = get_subgraph(
                self._model_inputs, target_op.outputs, framework_activation_op_map
            )
            framework_subgraph_activations = (
                framework_activation_op_map.keys() - self._model_inputs
            ) - framework_input2op_activations
            framework_subgraph_activations.update(target_op.outputs)

        self._logger.debug("+" * 71)
        return (
            subgraph_inputs,
            subgraph_outputs,
            target_subgraph_activations,
            framework_subgraph_activations,
        )
