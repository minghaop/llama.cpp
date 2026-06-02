# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import onnx  # noqa
import logging

from qti.aisw.accuracy_debugger.snooping.subgraph_snooper import SubgraphSnooper
from qti.aisw.accuracy_debugger.utils.constants import Algorithm
from qti.aisw.accuracy_debugger.utils.graph_utils import get_subgraph


class LayerwiseSnooper(SubgraphSnooper):
    """Subclass for layerwise algorithm."""

    def __init__(self, logger: logging.Logger):
        """Initializes the LayerwiseSnooper.

        Args:
            logger (logging.Logger): A python logger instance
        """
        self._activation_status = {}
        super().__init__(name=Algorithm.LAYERWISE, logger=logger, is_cumulative=False)

    def _get_subgraph_outputs(
        self,
        target_activation: str,
        target_activation_op_map: dict,
    ) -> set:
        """Finds out the cumulative subgraph's output tensor

        Args:
            target_activation (str): activation name in the target graph.
            target_activation_op_map (dict): Target activations to Target op map.

        Returns:
            (set): set of all subgraph outputs
        """
        target_op = target_activation_op_map[target_activation]

        return target_op.outputs

    def _get_subgraph_info(
        self,
        target_activation: str,
        target_activation_op_map: dict,
        framework_activation_op_map: dict,
        supergroup_activations: set,
    ) -> tuple[set, set, set, set]:
        """Given target_activation contruct the subgraph for layerwise snooping.

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
        # Get subgraph inputs
        subgraph_inputs = self._get_subgraph_inputs(
            target_activation,
            target_activation_op_map,
            framework_activation_op_map,
            supergroup_activations,
        )

        # Get subgraph outputs
        subgraph_outputs = self._get_subgraph_outputs(target_activation, target_activation_op_map)

        target_subgraph_activations = set()
        framework_subgraph_activations = set()

        self._logger.debug("+" * 71)
        self._logger.debug(f"Subgraph Inputs: {subgraph_inputs}")
        self._logger.debug(f"Subgraph Outputs: {subgraph_outputs}")
        if subgraph_inputs:
            self._logger.debug("Getting target subgraph")
            target_subgraph_activations, _, _ = get_subgraph(
                subgraph_inputs, subgraph_outputs, target_activation_op_map
            )

            self._logger.debug("Getting framework subgraph")
            framework_subgraph_activations, _, _ = get_subgraph(
                subgraph_inputs, subgraph_outputs, framework_activation_op_map
            )
            if not framework_subgraph_activations:
                framework_subgraph_activations = [
                    "Due to converter optimizations, framework subgraph is not found"
                ]
        self._logger.debug("+" * 71)
        return (
            subgraph_inputs,
            subgraph_outputs,
            target_subgraph_activations,
            framework_subgraph_activations,
        )
